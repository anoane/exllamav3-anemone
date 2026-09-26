"""
The RAM tier of one GPU's expert cache (doc/expert_tiers.md, "The RAM tier"), over the checkpoint's
own shards: a pinned slot arena registered with the disk engine (1 GiB chunks of whole slots),
experts read in place through the engine's extent API (O_DIRECT into 4 KiB-aligned slots), the
policy of model/expert_tier_policy.py run natively (exllamav3_ext/tier), and the disk slab for reads
that bypass the tier. This build copies with the CPU into a host buffer that stands in for the GPU's
pool and staging (the CPU replay of a routing trace, and the tests); the GPU runtime drives the same
host with the copy engines.

    index = expert_extents.build_extent_index(stc, keys)          # layer-major: key = lc * E + e
    tier = TierHost(index, layers = L, experts = E, policy = PolicyConfig(...), pool_slots = S,
                    ram_slots = R)
    tier.cold_fill(round_robin_order(L, E))
    tier.tick(); tier.call(lc, ids)                               # one decode call
    tier.prefetch(lc + 1); tier.layer(lc + 1, ids, half = 1)      # layer-mode prefill
"""

from __future__ import annotations
import dataclasses
import os

try:
    from .expert_tier_policy import PolicyConfig, DECODE, ROUTED, round_robin_order
except ImportError:                                               # loaded by path (tests)
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path
    _name = "_exl3_torch_free_expert_tier_policy"
    _m = _sys.modules.get(_name)
    if _m is None:
        _spec = _ilu.spec_from_file_location(_name, _Path(__file__).resolve().parent / "expert_tier_policy.py")
        _m = _ilu.module_from_spec(_spec)
        _sys.modules[_name] = _m
        _spec.loader.exec_module(_m)
    PolicyConfig, DECODE, ROUTED, round_robin_order = _m.PolicyConfig, _m.DECODE, _m.ROUTED, _m.round_robin_order


def _by_path(name: str, file: str):
    """A torch-free sibling module loaded by path (the tests load this one that way)"""
    import importlib.util as ilu
    import sys
    from pathlib import Path
    m = sys.modules.get(name)
    if m is None:
        spec = ilu.spec_from_file_location(name, Path(__file__).resolve().parent / file)
        m = ilu.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return m


def native_extents(index) -> tuple:
    """(files, extents, geometry) of an ExtentIndex in the native host's format"""
    files = index.files
    fidx = {f: i for i, f in enumerate(files)}
    ext = []
    for x in index.extents:
        pieces = [(fidx[p.path], p.offset, p.length) for p in x.pieces]
        trel = [(t[0], t[1]) for t in x.trellis]
        ext.append((x.name, pieces, trel))
    g = index.geometry
    geometry = {"proj_bytes": list(g.proj_bytes), "ram_slot_bytes": g.ram_slot_bytes}
    return files, ext, geometry


def _ext():
    from ..ext import exllamav3_ext
    return exllamav3_ext


class TierHost:
    """
    One GPU's cache layers (layers x experts keys, layer-major like the index) over one RAM tier.

      policy          PolicyConfig (the placement's words and the EXL3_MOE_TIER_* tuning)
      pool_slots      S, the VRAM pool (plus `pins` hot pins after it)
      ram_slots       R, the RAM tier
      slab_slots      the disk slab (reads that bypass the RAM tier; EXL3_MOE_TIER_DISK_SLAB)
      chunk_bytes     RAM tier chunk: one mapping, one registered io_uring buffer (at most 1 GiB)
      hugepage        MADV_HUGEPAGE on the chunks (EXL3_MOE_TIER_HUGEPAGE, default off)
      fill_inflight   cold fill reads in flight
      prefault_threads  threads that fault the RAM tier in before it is registered (0: registration does it)
      staging_slots   staging slots per half (default: experts per layer)
      deterministic   every read and copy finishes before a call returns (EXL3_MOE_TIER_DETERMINISTIC)
      ext             the extension module (default exllamav3_ext)
    """

    def __init__(self, index, layers: int, experts: int, policy: PolicyConfig, pool_slots: int, ram_slots: int,
                 pins: int = 0, slab_slots: int = 8, chunk_bytes: int = 1 << 30, hugepage: bool | None = None,
                 fill_inflight: int = 8, staging_slots: int = 0, deterministic: bool = True,
                 defer_returns: bool = False, prefault_threads: int = 8, ext = None, files_map = None,
                 io_direct: int = -1):
        if len(index) != layers * experts:
            raise ValueError(f"expert tier: {len(index)} extents for {layers} x {experts} experts")
        if hugepage is None:
            hugepage = os.environ.get("EXL3_MOE_TIER_HUGEPAGE", "0") == "1"
        self.layers, self.experts = layers, experts
        self.policy = policy
        self.geometry = index.geometry
        files, extents, geometry = native_extents(index)
        if files_map is not None:
            files = files_map(files)            # disk experts=<dir>: the checked copies
        self.files = files
        host = {"chunk_bytes": int(chunk_bytes), "hugepage": bool(hugepage), "slab_slots": int(slab_slots),
                "fill_inflight": int(fill_inflight), "staging_slots": int(staging_slots),
                "deterministic": bool(deterministic), "prefault_threads": int(prefault_threads),
                "io_direct": int(io_direct)}
        self.ext = ext or _ext()
        self.h = self.ext.TierHost(dataclasses.asdict(policy), layers, experts, pool_slots, ram_slots, pins, geometry,
                                   files, extents, host, defer_returns)

    def cold_fill(self, order = None):
        """Read the pool's and the RAM tier's first contents, in priority order (default: round robin
        across layers by expert index)"""
        self.h.cold_fill(list(order) if order is not None else round_robin_order(self.layers, self.experts))

    def call(self, lc: int, ids, mode: int = DECODE) -> list:
        """One decode (or routed prefill) call of cache layer lc; returns its record's entries"""
        return self.h.call(lc, [int(e) for e in ids], mode)

    def process(self, record):
        """The host side of a record decided elsewhere: (seq, lc, mode, [entry tuples])"""
        self.h.process(record)

    def layer(self, lc: int, ids, half: int = 0):
        self.h.layer(lc, [int(e) for e in ids], half)

    def stage(self, lc: int, half: int = 0) -> list:
        """Layer mode's first half (the GPU runtime's layer_plan): stage layer lc into `half`; the staged
        keys in staging order. The call's record follows through process()"""
        return list(self.h.stage(lc, half))

    def read_ahead(self, record):
        """prefetch=router: read the disk-only experts a prediction record lists into the slab"""
        self.h.read_ahead(record)

    def prefetch(self, lc: int, deadline_ns: int = 0):
        self.h.prefetch(lc, deadline_ns)

    def promote(self, lc: int):
        self.h.promote(lc)

    def tick(self, tokens: int = 1):
        self.h.tick(tokens)

    def complete(self):
        self.h.complete()

    def drain(self):
        self.h.drain()

    def poll(self):
        self.h.poll()

    def state(self) -> dict:
        return self.h.state()

    def counters(self) -> dict:
        return self.h.counters()

    def stats(self) -> dict:
        s = dict(self.h.stats())
        s.update({"policy_" + k: v for k, v in self.h.counters().items()})
        return s

    def arena(self) -> dict:
        return self.h.arena()

    def vram_slot(self, slot: int):
        return self.h.vram_slot(slot)

    def staging(self, half: int, i: int):
        return self.h.staging(half, i)

    def ram_trellis(self, r: int):
        return self.h.ram_trellis(r)

    def last_actions(self) -> list:
        return [tuple(a) for a in self.h.last_actions()]

    def error(self) -> str:
        return self.h.error()

    @classmethod
    def from_placement(cls, stc, modules: list, placement, device: str, tier_config, pool_slots: int,
                       ram_slots: int, slab_slots: int | None = None, seed: int = 0, ext = None, **kw) -> "TierHost":
        """
        The RAM tier of the experts=cache layers `modules` (BlockSparseMLP, one GPU, forward order)
        as the placement and the tuning ask: the policy from `device`'s cuda:<n> rule, the ram and disk
        rules and the TierConfig; the extents from the checkpoint headers; the pool and RAM tier
        sizes from the sizing (expert_tier_config.size_tiers). Hot pins (hot=) are placed by the GPU
        runtime and refused here.
        """
        try:
            from .expert_extents import build_for_modules
        except ImportError:
            import importlib.util as ilu
            import sys
            from pathlib import Path
            name = "_exl3_torch_free_expert_extents"
            m = sys.modules.get(name)
            if m is None:
                spec = ilu.spec_from_file_location(name, Path(__file__).resolve().parent / "expert_extents.py")
                m = ilu.module_from_spec(spec)
                sys.modules[name] = m
                spec.loader.exec_module(m)
            build_for_modules = m.build_for_modules
        if not modules:
            raise ValueError("expert tier: no experts=cache layers")
        experts = {m.num_experts for m in modules}
        if len(experts) != 1:
            raise ValueError(f"expert tier: the cache layers on {device} have different expert counts "
                             f"({sorted(experts)}); one count per GPU is supported")
        for m in modules:
            plan = m.placement_plan() if hasattr(m, "placement_plan") else None
            if plan is not None and len(plan) > 2 and plan[2]:
                raise ValueError(f"expert tier: {m.key} hot={plan[2]}: hot pins are placed by the GPU runtime")
        policy = PolicyConfig.from_placement(placement.device_store(device), placement.ram_store(),
                                             placement.disk_store(), tier_config, seed = seed)
        index, _ = build_for_modules(stc, modules)
        if slab_slots is None:
            slab_slots = 0 if policy.disk_off else (tier_config.disk_slab or 8)
        kw.setdefault("deterministic", True)
        kw.setdefault("hugepage", tier_config.hugepage)
        # the placement's disk rule: the shards (or a checked copy of them), the page-cache mode
        try:
            from .disk_source import expert_files, io_direct
        except ImportError:
            ds = _by_path("_exl3_torch_free_disk_source", "disk_source.py")
            expert_files, io_direct = ds.expert_files, ds.io_direct
        kw.setdefault("files_map", lambda files: expert_files(placement, files))
        kw.setdefault("io_direct", io_direct(placement))
        return cls(index, len(modules), experts.pop(), policy, pool_slots, ram_slots, slab_slots = slab_slots,
                   ext = ext, **kw)
