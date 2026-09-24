"""
Where a layer-split load of DeepSeek-V4.1 may change device.

A V4.1 compressed layer does not keep its own compressed KV. The kv source of each group owns
one paged pool (cache_plan.py) and every other layer of the group reads it; an index source
publishes a top-k selection that the layers above it up to the next index source reuse; and the
candidate source publishes candidate blocks for the index sources above it. The attention
kernels read all of these on the reading layer's own device.

A change of device between layers c - 1 and c (a cut at c) is FREE when nothing of the above is
shared across it: no layer at or past c reads a pool, a top-k selection or candidate blocks
produced below c. On DeepSeek-V4.1-Flash (kv groups 2-7, 8-13, 14-19 and 20-39, index sources
2, 8, 14, 20, 24, 28, 32 and 36, candidate source 20) the free cuts are layers 1, 2, 8, 14 and 20.

With a Cache attached, every change of device must be a free cut, because the cached path reads
the Cache's pools on the reading layer's device. exllamav3's autosplit (model_ls.py) fills the
devices in layer order until each budget runs out and knows nothing about kv groups, so
DSV41LoadGuard checks each change of device it makes while the model loads: one that is not a
free cut fails at once, at the first layer past it, with a message that names the free cuts and
how to move the change with the split budgets (-gs / use_per_device / reserve_per_device).
Without a Cache any cut loads: the stateless path copies the pools, selections and candidate
blocks a layer on another device reads, whole, in every forward.

This module imports only the standard library at module scope (torch is imported inside
DSV41LoadGuard.check), so tests and tools that must not import the compiled extension can load
it by path.
"""

from __future__ import annotations

# Why DeepseekV41Model.load_gen and DSV41MoE.make_tp_allocation refuse tensor-parallel loading
TP_REFUSAL = ("DeepSeek-V4.1: tensor-parallel loading is not implemented; load it as a layer "
              "split (-gs / use_per_device)")

# Where the loading rules are documented, for the messages
DOC_REF = 'doc/env_vars.md, DeepSeek-V4.1, "Loading across GPUs"'


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


def cut_message(topo, cut: int, below, above, free = None) -> str:
    """
    Why a change of device at layer `cut` (from device `below` to `above`) cannot take a Cache,
    and how to move it with the split budgets. `free` defaults to free_cuts(topo).
    """
    n = topo.num_hidden_layers
    free = free_cuts(topo) if free is None else list(free)
    pools, topk, cand = crossings(topo, cut)
    if pools:
        src = min(s for s, _ in pools)
        last = max(i for i in range(n) if topo.kv_source_for(i) == src)
        what = (f"layers {src}-{last} share the compressed-KV pool of layers.{src}, which is on "
                f"{below}")
    elif topk:
        src = min(s for s, _ in topk)
        readers = format_layer_ranges(i for s, i in topk if s == src)
        what = f"layers {readers} reuse the top-k selection of layers.{src}, which is on {below}"
    else:
        src = min(s for s, _ in cand)
        readers = format_layer_ranges(i for s, i in cand if s == src)
        what = (f"layers {readers} mask their index scores to the candidate blocks of "
                f"layers.{src}, which is on {below}")
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
        f"(-gs / use_per_device / reserve_per_device): {', or '.join(moves)} ({DOC_REF})")


# ---------------------------------------------------------------------------
# load-time enforcement

class DSV41LoadGuard:
    """
    Checks the autosplit's changes of device while DeepseekV41Model loads. Each DSV41MoE calls
    check() at the top of its load, i.e. once per block per attempted device, in layer order (the
    autosplit retries a block on the next device when its load raises OutOfMemoryError).

    A block offered another device than the block before it is a change of device at its layer.
    With a Cache attached and that layer not a free cut, check() raises RuntimeError at once,
    before anything of the block's MoE loads: the cached path reads the pools on the reading
    layer's own device, and failing here is minutes earlier and far clearer than a device
    mismatch inside the attention. Without a Cache every change is accepted.

    Only active between begin() and end() (DeepseekV41Model.load_gen), so module-level tests that
    load single blocks on chosen devices are unaffected.
    """

    def __init__(self, topo, cache_attached = None):
        self.topo = topo
        self.free = free_cuts(topo)
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
                and self._cache_attached():
            raise RuntimeError(cut_message(self.topo, layer_idx, below, dev, self.free))
        self.devices[layer_idx] = dev
