"""
Where a layer-split load of DeepSeek-V4.1 may change device, and the pool replica an explicit
placement may add.

A V4.1 compressed layer does not keep its own compressed KV. The kv source of each group owns
one paged pool (cache_plan.py) and every other layer of the group reads it; an index source
publishes a top-k selection that the layers above it up to the next index source reuse; and the
candidate source publishes candidate blocks for the index sources above it. The attention
kernels read all of these on the reading layer's own device.

A change of device between layers c - 1 and c (a cut at c) is FREE when nothing of the above is
shared across it: no layer at or past c reads a pool, a top-k selection or candidate blocks
produced below c. On DeepSeek-V4.1-Flash (kv groups 2-7, 8-13, 14-19 and 20-39, index sources
2, 8, 14, 20, 24, 28, 32 and 36, candidate source 20) the free cuts are layers 1, 2, 8, 14 and 20.

With a Cache attached, every change of device the autosplit makes must be a free cut, because
the cached path reads the Cache's pools on the reading layer's device. exllamav3's autosplit
(model_ls.py) fills the devices in layer order until each budget runs out and knows nothing about
kv groups, so DSV41LoadGuard checks each change of device it makes while the model loads: one
that is not a free cut fails at once, at the first layer past it, with a message that names the
free cuts and how to move the change with the split budgets (-gs / use_per_device /
reserve_per_device). Without a Cache any cut loads: the stateless path copies the pools,
selections and candidate blocks a layer on another device reads, whole, in every forward.

An explicit placement (EXL3_PLACEMENT / --placement / config.infer_params.placement,
model/placement.py) names the device of every layer, and DeepseekV41Model reads it when the model
object is built (DSV41Placement.from_generic): every change of device between its decoder layers
must be a free cut, except at most one, the split, which may fall where something is shared. The
first layer past the split then owns a replica of its kv group's pool on its own device
(cache/dsv41_replica.py), which the Cache allocates, so it must be known before any Cache is
built; every forward copies the entries the source added into it, and the top-k selections and
candidate blocks that straddle the split move once per forward (cache/dsv41.py, DeviceMemo).

Routes for a split at `split` (compute_routes), per compressed layer i with kv source
src = kv_source_for(i):

  (owner, replica_of, with_index_k)
    owner         the layer whose cache layer i reads ON ITS OWN DEVICE: src, unless
                  src < split <= i -- then the first layer >= split of that group (= split),
                  which owns a replica of src's pool
    replica_of    src on that replica owner, else None
    with_index_k  on the replica owner: an index source in [split, group end) borrows src's
                  index K, so the replica must carry it too

  topk_cross = {(index_source_for(i), i) : index_source_for(i) < split <= i}
  cand_cross = {(cand_src, i) : i an index source using candidates,
                                cand_src < split <= i}

These routes of one split stay exact with any number of other changes of device, as long as
those are free cuts: nothing is shared across a free cut, so no layer that reads across the split
reads across another change of device too.

The residual streams (hc_mult FP32 streams per token) cross every change of device once per
forward, into the first block past it. XDEV_BF16 (EXL3_DSV41_XDEV_BF16=1) narrows that copy to
BF16; the block widens it back to FP32 on its own device (modules/dsv41_block.py,
DSV41Block.prepare_for_device), and the pipelined prefill rounds on the first device at the end of
its first stage (pipeline.py).

This module imports only the standard library at module scope (torch is imported inside
DSV41LoadGuard.check), so tests and tools that must not import the compiled extension can load
it by path.
"""

from __future__ import annotations

import os

# Why DeepseekV41Model.load_gen and DSV41MoE.make_tp_allocation refuse tensor-parallel loading
TP_REFUSAL = ("DeepSeek-V4.1: tensor-parallel loading is not implemented; load it as a layer "
              "split (-gs / use_per_device), or with an explicit placement (EXL3_PLACEMENT / "
              "--placement)")

# Where the loading rules are documented, for the messages
DOC_REF = 'doc/env_vars.md, DeepSeek-V4.1, "Loading across GPUs"'
PLACEMENT_DOC_REF = 'doc/placement.md, "DeepSeek-V4.1"'

# What the refusals of a change of device inside a kv group add: the explicit placement's split
REPLICA_HINT = ("An explicit placement (EXL3_PLACEMENT / --placement) can instead change device "
                "inside one kv group; the layers past it then read a replica of its pool "
                f"({PLACEMENT_DOC_REF})")

# EXL3_DSV41_XDEV_BF16=1: the FP32 residual streams cross a change of device as BF16, half the
# bytes, at the cost of rounding them (doc/env_vars.md). Read once at import; the block and the
# pipelined prefill read this attribute on every call
XDEV_BF16 = os.environ.get("EXL3_DSV41_XDEV_BF16", "0") != "0"


# ---------------------------------------------------------------------------
# layer range strings

def format_layer_ranges(layers) -> str:
    """{12..22, 30} -> '12-22,30'; an empty set -> 'none'."""
    s = sorted(layers)
    if not s:
        return "none"
    parts, a, b = [], s[0], s[0]
    for i in s[1:]:
        if i == b + 1:
            b = i
            continue
        parts.append(f"{a}" if a == b else f"{a}-{b}")
        a = b = i
    parts.append(f"{a}" if a == b else f"{a}-{b}")
    return ",".join(parts)


def format_cuts(cuts) -> str:
    """[1, 2, 8, 14, 20] -> 'layer 1, 2, 8, 14 or 20'; [] -> 'no layer'."""
    cuts = [str(c) for c in cuts]
    if not cuts:
        return "no layer"
    if len(cuts) == 1:
        return f"layer {cuts[0]}"
    return f"layer {', '.join(cuts[:-1])} or {cuts[-1]}"


# ---------------------------------------------------------------------------
# topology

class Topology:
    """
    The V4.1 source topology read straight from config.json, for tools that
    must not import the package (and its compiled extension). Answers the
    same queries as DeepseekV41Config with the same formulas; the config test
    checks the two agree on every layer.
    """

    def __init__(self, num_hidden_layers, compress_ratios, kv_source_layer_ids,
                 index_source_layer_ids, candidate_source_layer_id = -1):
        self.num_hidden_layers = int(num_hidden_layers)
        self.compress_ratios = [int(r) for r in compress_ratios[:self.num_hidden_layers]]
        self.kv_source_layer_ids = tuple(int(i) for i in kv_source_layer_ids)
        self.index_source_layer_ids = tuple(int(i) for i in index_source_layer_ids)
        self.candidate_source_layer_id = int(candidate_source_layer_id)

    @classmethod
    def from_config_dict(cls, d: dict) -> "Topology":
        t = d.get("text_config", d)
        return cls(
            t["num_hidden_layers"], t["compress_ratios"], t["kv_source_layer_ids"],
            t["index_source_layer_ids"], t.get("candidate_source_layer_id", -1),
        )

    def is_kv_source(self, i: int) -> bool:
        return i in self.kv_source_layer_ids

    def is_index_source(self, i: int) -> bool:
        return i in self.index_source_layer_ids

    def kv_source_for(self, i: int) -> int | None:
        if self.compress_ratios[i] == 0:
            return None
        return max(s for s in self.kv_source_layer_ids if s <= i)

    def index_source_for(self, i: int) -> int | None:
        if self.compress_ratios[i] == 0:
            return None
        return max(s for s in self.index_source_layer_ids if s <= i)

    def index_owns_k(self, i: int) -> bool:
        return self.is_index_source(i) and self.is_kv_source(i)

    def uses_candidates(self, i: int) -> bool:
        return 0 <= self.candidate_source_layer_id < i

    def kv_groups(self) -> list[tuple[int, list[int]]]:
        groups: dict[int, list[int]] = {}
        for i in range(self.num_hidden_layers):
            s = self.kv_source_for(i)
            if s is not None:
                groups.setdefault(s, []).append(i)
        return sorted(groups.items())


# ---------------------------------------------------------------------------
# cuts

def crossings(topo, cut: int):
    """
    What a change of device between layers cut - 1 and cut would split, as three sets of
    (producer, reader) layer pairs with producer < cut <= reader:

      pools  a compressed layer at or past the cut and its kv source, which owns the pool it reads
      topk   a compressed layer at or past the cut and the index source whose top-k selection it
             reuses (an index source selects for itself)
      cand   an index source at or past the cut that uses candidates, and the candidate source

    ``topo`` is a Topology or a DeepseekV41Config (the same queries).
    """
    n = topo.num_hidden_layers
    if not 1 <= cut <= n - 1:
        raise ValueError(f"a cut must lie in 1..{n - 1}, got {cut}")
    cand_src = getattr(topo, "candidate_source_layer_id", -1)
    pools, topk, cand = set(), set(), set()
    for i in range(cut, n):
        src = topo.kv_source_for(i)
        if src is None:
            continue
        if src < cut:
            pools.add((src, i))
        isrc = topo.index_source_for(i)
        if isrc is not None and isrc < cut:
            topk.add((isrc, i))
        if topo.is_index_source(i) and topo.uses_candidates(i) and 0 <= cand_src < cut:
            cand.add((cand_src, i))
    return pools, topk, cand


def free_cuts(topo) -> list[int]:
    """The layers c in 1 .. n-1 at which a change of device splits nothing (see crossings)."""
    return [c for c in range(1, topo.num_hidden_layers) if not any(crossings(topo, c))]


def split_what(topo, cut: int, below = None) -> str:
    """
    What a change of device at layer `cut` separates, first match of: the pool of a kv group,
    a top-k selection, candidate blocks; `below` names the device of the producer.
    """
    n = topo.num_hidden_layers
    on = f", which is on {below}" if below is not None else ""
    pools, topk, cand = crossings(topo, cut)
    if pools:
        src = min(s for s, _ in pools)
        last = max(i for i in range(n) if topo.kv_source_for(i) == src)
        return f"layers {src}-{last} share the compressed-KV pool of layers.{src}{on}"
    if topk:
        src = min(s for s, _ in topk)
        readers = format_layer_ranges(i for s, i in topk if s == src)
        return f"layers {readers} reuse the top-k selection of layers.{src}{on}"
    src = min(s for s, _ in cand)
    readers = format_layer_ranges(i for s, i in cand if s == src)
    return f"layers {readers} mask their index scores to the candidate blocks of layers.{src}{on}"


def cut_message(topo, cut: int, below, above, free = None) -> str:
    """
    Why a change of device at layer `cut` (from device `below` to `above`) cannot take a Cache,
    and how to move it with the split budgets. `free` defaults to free_cuts(topo).
    """
    n = topo.num_hidden_layers
    free = free_cuts(topo) if free is None else list(free)
    what = split_what(topo, cut, below)
    lo = max((c for c in free if c < cut), default = None)
    hi = min((c for c in free if c > cut), default = None)
    moves = []
    if lo is not None:
        moves.append(f"less room on {below}, so that it holds layers up to {lo - 1} only")
    if hi is not None:
        moves.append(f"room for layers {cut}-{hi - 1} as well, so that the change falls at layer {hi}")
    else:
        moves.append(f"room for layers {cut}-{n - 1} as well, so that no change of device is left")
    return (
        f"DeepSeek-V4.1: the layer split put layers.{cut} on {above}, but {what}. With a Cache "
        f"attached, a change of device can only fall where no pool, top-k selection or candidate "
        f"list is shared across it: at {format_cuts(free)} on this model. Change the split budgets "
        f"(-gs / use_per_device / reserve_per_device): {', or '.join(moves)} ({DOC_REF}). "
        f"{REPLICA_HINT}")


# ---------------------------------------------------------------------------
# routes of a split

def _group_members(topo, src: int) -> list[int]:
    for s, members in topo.kv_groups():
        if s == src:
            return members
    raise ValueError(f"layer {src} is not a kv source")


def compute_routes(topo, split: int | None):
    """
    Pool routes plus the two small per-forward crossings for a split at
    ``split`` (None: one device, every layer reads its own source).

    Returns (routes, topk_cross, cand_cross); see the module docstring.
    """
    n = topo.num_hidden_layers
    routes: dict[int, tuple[int, int | None, bool]] = {}
    topk_cross: set[tuple[int, int]] = set()
    cand_cross: set[tuple[int, int]] = set()
    cand_src = getattr(topo, "candidate_source_layer_id", -1)

    for i in range(n):
        src = topo.kv_source_for(i)
        if src is None:
            continue
        if split is not None and src < split <= i:
            # the group is cut: its members at or past the split read a replica
            # owned by the first of them, which is the split layer itself
            members = _group_members(topo, src)
            owner = min(m for m in members if m >= split)
            if i == owner:
                wik = any(topo.is_index_source(j) and not topo.index_owns_k(j)
                          for j in members if j >= split)
                routes[i] = (owner, src, wik)
            else:
                routes[i] = (owner, None, False)
        else:
            routes[i] = (src, None, False)

        if split is not None:
            isrc = topo.index_source_for(i)
            if isrc is not None and isrc < split <= i:
                topk_cross.add((isrc, i))
            if (topo.is_index_source(i) and topo.uses_candidates(i)
                    and 0 <= cand_src < split <= i):
                cand_cross.add((cand_src, i))

    return routes, topk_cross, cand_cross


def default_routes(topo) -> dict[int, tuple[int, int | None, bool]]:
    """Every compressed layer reads its own kv source: no replicas."""
    return compute_routes(topo, None)[0]


# ---------------------------------------------------------------------------
# the explicit placement

def placement_cuts(generic, topo) -> list[int]:
    """
    The layers at which an explicit placement (model/placement.py) changes device between
    decoder layers. Refuses, as the loader does, a placement that names a layer the model does
    not have or leaves one unplaced.
    """
    n = topo.num_hidden_layers
    rules = generic.layer_map(range(n))
    return [i for i in range(1, n) if rules[i].device != rules[i - 1].device]


class DSV41Placement:
    """
    What an explicit placement means for V4.1's cross-device reads: its changes of device
    between decoder layers (cuts), the one of them that is not a free cut (split_at, None when
    every cut is free), and that split's pool routes and crossings (compute_routes). Built from
    the placement when the model object is built (from_generic); immutable.

    Refused with ValueError: more than one cut that is not free. At most one change of device
    may fall where a pool, a top-k selection or candidate blocks are shared across it, because
    one replica owner per kv group and one set of crossings are what the routes express.
    """

    def __init__(self, topo, cuts = (), spec: str | None = None):
        n = topo.num_hidden_layers
        self.num_layers = n
        self.cuts = tuple(sorted({int(c) for c in cuts}))
        for c in self.cuts:
            if not 1 <= c <= n - 1:
                raise ValueError(f"a change of device must lie at layer 1..{n - 1}, got {c}")
        self.spec = spec
        free = free_cuts(topo)
        inside = [c for c in self.cuts if c not in free]
        if len(inside) > 1:
            where = " and ".join(f"at layer {c} ({split_what(topo, c)})" for c in inside)
            raise ValueError(
                f"DeepSeek-V4.1: the explicit placement '{spec}' changes device in {len(inside)} "
                f"places where a pool, a top-k selection or candidate blocks are shared across the "
                f"change: {where}. At most one change of device may fall there (the layers past it "
                f"then read a replica of the pool); every other one must fall at "
                f"{format_cuts(free)} on this model ({PLACEMENT_DOC_REF})")
        self.split_at = inside[0] if inside else None
        self.routes, self.topk_cross, self.cand_cross = compute_routes(topo, self.split_at)

    @classmethod
    def from_generic(cls, generic, topo) -> "DSV41Placement | None":
        """The explicit placement `generic` (a parsed model/placement.py Placement) on this
        topology, or None without one."""
        if generic is None:
            return None
        return cls(topo, placement_cuts(generic, topo), spec = str(generic))

    # -- queries --

    @property
    def replicas(self) -> dict[int, int]:
        """{replica owner: source} for the cut kv group (empty without a split)."""
        return {i: r[1] for i, r in self.routes.items() if r[1] is not None}

    def needs(self) -> str:
        """The pool replica this placement needs, in words (the refusal of a changed placement
        at load uses it)."""
        if self.split_at is None:
            return "no pool replica"
        return f"a pool replica for a change of device at layer {self.split_at}"

    def crossings(self) -> str:
        """What moves between the devices besides the hidden state: the pool replicas ('+idxK'
        when one carries index keys) and the top-k selections and candidate blocks that cross the
        split, as the model's post-load line prints them."""
        def pairs(ps):
            by_src: dict[int, list[int]] = {}
            for a, b in sorted(ps):
                by_src.setdefault(a, []).append(b)
            return " ".join(f"{a}->{format_layer_ranges(bs)}" for a, bs in by_src.items()) or "none"
        rep = " ".join(f"{o}<-{s}{'+idxK' if self.routes[o][2] else ''}"
                       for o, s in sorted(self.replicas.items())) or "none"
        return (f"replicas {rep} | top-k cross {pairs(self.topk_cross)} | "
                f"candidates cross {pairs(self.cand_cross)}")

    def describe(self) -> str:
        cuts = format_layer_ranges(self.cuts)
        return f"changes of device {cuts} | {self.crossings()}"

    def __repr__(self):
        return f"DSV41Placement({self.spec!r}, cuts = {list(self.cuts)}, split_at = {self.split_at})"


# ---------------------------------------------------------------------------
# load-time enforcement

class DSV41LoadGuard:
    """
    Checks the autosplit's changes of device while DeepseekV41Model loads. Each DSV41MoE calls
    check() at the top of its load, i.e. once per block per attempted device, in layer order (the
    autosplit retries a block on the next device when its load raises OutOfMemoryError).

    A block offered another device than the block before it is a change of device at its layer.
    With a Cache attached, that layer neither a free cut nor the split of the explicit placement
    the model was built with (`split_at`, whose replica route the Cache has), check() raises
    RuntimeError at once, before anything of the block's MoE loads: the cached path reads the
    pools on the reading layer's own device, and failing here is minutes earlier and far clearer
    than a device mismatch inside the attention. Without a Cache every change is accepted.

    Only active between begin() and end() (DeepseekV41Model.load_gen), so module-level tests that
    load single blocks on chosen devices are unaffected.
    """

    def __init__(self, topo, cache_attached = None, split_at: int | None = None):
        self.topo = topo
        self.free = free_cuts(topo)
        self.split_at = split_at
        self._cache_attached = cache_attached or (lambda: False)
        self.active = False
        self.devices: dict[int, object] = {}

    def begin(self):
        self.active = True
        self.devices = {}

    def end(self):
        self.active = False

    def check(self, layer_idx: int, device) -> None:
        if not self.active or device is None:
            return
        import torch
        dev = torch.device(device)
        if dev.type != "cuda":
            return
        below = self.devices.get(layer_idx - 1)
        if below is not None and below != dev and layer_idx not in self.free \
                and layer_idx != self.split_at and self._cache_attached():
            raise RuntimeError(cut_message(self.topo, layer_idx, below, dev, self.free))
        self.devices[layer_idx] = dev
