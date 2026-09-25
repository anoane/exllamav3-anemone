"""
The expert tier of a load: the VRAM expert cache of every GPU that holds experts=cache layers, over
its RAM tier, over the checkpoint on the SSD (doc/expert_tiers.md).

    stage 1   Model.load_gen (model.py): the placement and ram_budget.plan_component decide there is a
              tier; ExpertTierSet(config, placement, budget) goes on config.expert_tiers
    load      each experts=cache layer registers with its GPU's TierDevice before it loads
              (BlockSparseMLP_Tier, modules/block_sparse_mlp_tier.py): its routed experts get
              resident suh / svh and a trellis that is a view of one zeroed sentinel slot; the
              layer's per-expert trellis pointer tables are what the tier rewrites per call
    finalize  the end of the layer split (model_ls.py, before the memory fraction is lifted): each
              GPU's pool from the VRAM its placed modules left (expert_tier_config.size_tiers),
              the RAM tier, the extents of every cached expert in the checkpoint, the native
              runtime (exllamav3_ext TierGpu: lookup kernel, tier thread, copy engines), the cold
              fill, the tier thread
    forward   every MoE call of a cache layer: TierDevice.resolve() launches the lookup on the
              compute stream and makes the stream wait for the call's copies; the unchanged MoE
              kernels then read the rewritten tables

A tiered layer computes exactly what the same layer computes resident: the kernels, their arguments
and the trellis bytes are the same, only where the bytes live differs.
"""

from __future__ import annotations
import dataclasses
import os
import time

import torch

from .expert_tier_policy import PolicyConfig, DECODE, ROUTED, round_robin_order
from .expert_tier_config import TierConfig, size_tiers, check_geometries, ineligible
from .expert_extents import build_for_modules
from .expert_tier_host import native_extents
from ..util.host_budget import human, host_memory, GiB, MiB

POOL_CHUNK = 4 * GiB            # the pool in allocations of at most this many bytes (whole slots)
SLOT_ALIGN = 256


def _ext():
    from ..ext import exllamav3_ext
    return exllamav3_ext


def _numel(shape) -> int:
    n = 1
    for x in shape:
        n *= x
    return n


def measure_link(device: torch.device, nbytes: int = 64 * MiB, reps: int = 3) -> float:
    """Host-to-device GB/s of `device` from page-locked memory (the best of `reps` copies)"""
    src = torch.empty((nbytes,), dtype = torch.uint8, pin_memory = True)
    dst = torch.empty((nbytes,), dtype = torch.uint8, device = device)
    stream = torch.cuda.Stream(device = device)
    best = 0.0
    with torch.cuda.stream(stream):
        for _ in range(reps):
            t0 = time.perf_counter()
            dst.copy_(src, non_blocking = True)
            stream.synchronize()
            best = max(best, nbytes / max(1e-9, time.perf_counter() - t0) / 1e9)
    del src, dst
    return best


@dataclasses.dataclass
class TierLayer:
    module: object
    lc: int                         # index of the cache layer on its GPU (forward order)
    hot: int                        # experts pinned resident (hot=)
    tables: list | None = None      # the layer's int64 trellis pointer tables, projection order
    shapes: list | None = None      # the trellis shape of each projection


class TierDevice:
    """The VRAM expert cache of one GPU: its cache layers, pool, staging, and the native runtime"""

    def __init__(self, tier_set: "ExpertTierSet", device: torch.device):
        self.tiers = tier_set
        self.device = torch.device(device)
        self.name = f"cuda:{self.device.index}"
        self.layers: list[TierLayer] = []
        self.sentinel = None
        self.proj_shapes = None     # [(shape, nbytes)] per projection
        self.proj_off = None
        self.vs = 0                 # VRAM slot bytes (compact trellis of one expert)
        self.stride = 0
        self.pool = []              # [(tensor, first slot, slots)]
        self.staging = None
        self.nstage = 0
        self.handle = None
        self.slots = 0
        self.pins = 0
        self.seq = 0
        self.tokens = 0
        self.decode_rows = tier_set.cfg.decode_rows
        self.verify = tier_set.cfg.verify
        self.sizing = None

    # ------------------------------------------------------------------------------------ load

    def register(self, module, hot: int) -> TierLayer:
        layer = TierLayer(module, len(self.layers), hot)
        self.layers.append(layer)
        return layer

    def unregister(self, module):
        self.layers = [l for l in self.layers if l.module is not module]
        for i, l in enumerate(self.layers):
            l.lc = i

    def set_geometry(self, shapes: list):
        """The trellis shapes of a layer's projections (gate, up, down or up, down): the first layer
        sets them and allocates the zeroed sentinel slot, every other layer must match"""
        shapes = [tuple(x) for x in shapes]
        got = [(x, 2 * _numel(x)) for x in shapes]
        if self.proj_shapes is None:
            self.proj_shapes = got
            self.sentinel = torch.zeros((sum(n for _, n in got) + 255) // 256 * 256, dtype = torch.uint8,
                                        device = self.device)
        elif self.proj_shapes != got:
            a = tuple(x for x, _ in self.proj_shapes)
            raise ValueError(f"placement: the experts=cache layers on {self.name} have different expert shapes "
                             f"({a} and {tuple(shapes)}); one expert slot size per GPU is supported")

    def sentinel_view(self, proj: int) -> torch.Tensor:
        """The trellis stand-in of one projection: a view of the zeroed sentinel slot"""
        shape, nbytes = self.proj_shapes[proj]
        off = sum(n for _, n in self.proj_shapes[:proj])
        return self.sentinel[off : off + nbytes].view(torch.int16).view(shape)

    def set_tables(self, layer: TierLayer, tables: list, shapes: list):
        layer.tables = tables
        layer.shapes = shapes

    # ------------------------------------------------------------------------------------ finalize

    def build(self, slots: int, ram_slots: int, slab_slots: int, policy: PolicyConfig, cfg: TierConfig):
        """Allocate the pool (with the hot pins after it) and the staging, then the runtime; cold fill"""
        stc = self.tiers.config.stc
        modules = [l.module for l in self.layers]
        index, _ = build_for_modules(stc, modules)
        g = index.geometry
        self.vs = g.vram_slot_bytes
        self.proj_off = list(g.proj_off)
        if [n for _, n in self.proj_shapes] != list(g.proj_bytes):
            raise RuntimeError(f"expert tier: {self.name}: the checkpoint's trellis sizes {list(g.proj_bytes)} differ "
                               f"from the loaded shapes {[n for _, n in self.proj_shapes]}")
        self.stride = -(-self.vs // SLOT_ALIGN) * SLOT_ALIGN
        E = modules[0].num_experts
        L = len(modules)
        pin_keys = []
        for l in self.layers:
            pin_keys += [l.lc * E + e for e in range(l.hot)]
        self.pins = len(pin_keys)
        self.slots = slots
        total = slots + self.pins
        per = max(1, POOL_CHUNK // self.stride)
        addrs = []
        first = 0
        while first < total:
            n = min(per, total - first)
            t = torch.empty((n * self.stride,), dtype = torch.uint8, device = self.device)
            self.pool.append((t, first, n))
            addrs += [t.data_ptr() + i * self.stride for i in range(n)]
            first += n
        self.nstage = max(E - l.hot for l in self.layers)
        self.staging = torch.empty((self.nstage * self.stride,), dtype = torch.uint8, device = self.device)
        staging = [self.staging.data_ptr() + i * self.stride for i in range(self.nstage)]
        tables = []
        for l in self.layers:
            if l.tables is None:
                raise RuntimeError(f"expert tier: {l.module.key} registered but never finished loading")
            tables += [t.data_ptr() for t in l.tables]
        files, extents, geometry = native_extents(index)
        host = {"chunk_bytes": 1 << 30, "hugepage": cfg.hugepage, "slab_slots": max(slab_slots, 8),
                "fill_inflight": 8, "prefault_threads": 8, "compact_fill": cfg.compact}
        gpu = {"device": self.device.index, "pool": addrs, "staging": staging, "staging_per_half": self.nstage,
               "tables": tables, "proj_off": self.proj_off, "deterministic": cfg.deterministic,
               "spin_us": cfg.spin_us, "affinity": list(cfg.affinity or ()), "trace": 0}
        torch.cuda.synchronize(self.device)
        self.handle = _ext().TierGpu(dataclasses.asdict(policy), L, E, slots, ram_slots, self.pins, geometry, files,
                                     extents, host, gpu)
        order = round_robin_order(L, E, set(pin_keys))
        t0 = time.monotonic()
        self.handle.cold_fill(order, pin_keys)
        if policy.disk_off:
            self.handle.release_slab()
        s = self.handle.stats()
        print(f" -- expert cache {self.name}: cold fill of {s['cold_vram']} VRAM and {s['cold_ram']} RAM experts, "
              f"{human(s['cold_bytes'])} read in {time.monotonic() - t0:.1f} s; {human(s['pinned_bytes'])} of RAM "
              f"page-locked in {s['pin_ns'] / 1e9:.1f} s")
        self.handle.start()

    # ------------------------------------------------------------------------------------ forward

    def resolve(self, layer: TierLayer, selected: torch.Tensor, rows: int):
        """One MoE call: the lookup on the current stream, which then waits for the call's copies"""
        mode = DECODE if rows <= self.decode_rows else ROUTED
        if mode == DECODE and layer.lc == 0:
            self.tokens += 1
        self.seq = (self.seq + 1) & 0xFFFFFFFF or 1
        ids = selected if selected.is_contiguous() else selected.contiguous()
        self.handle.lookup(layer.lc, ids, mode, self.seq, self.tokens)
        if self.verify:
            torch.cuda.synchronize(self.device)
            self.handle.drain()
            self.handle.verify()

    def trellis_views(self, layer: TierLayer, e: int) -> list:
        """Expert e's trellis tensors where the last lookup put them (the per-expert dequant path;
        its caller has synchronized already)"""
        out = []
        for t, (shape, nbytes) in zip(layer.tables, self.proj_shapes):
            out.append(self.view_at(int(t[e].item()), shape, nbytes))
        return out

    def view_at(self, addr: int, shape: tuple, nbytes: int) -> torch.Tensor:
        for t in [p[0] for p in self.pool] + [self.staging, self.sentinel]:
            base = t.data_ptr()
            if base <= addr and addr + nbytes <= base + t.numel():
                off = addr - base
                return t[off : off + nbytes].view(torch.int16).view(shape)
        raise RuntimeError(f"expert tier: {self.name}: a trellis pointer outside the pool, staging and sentinel")

    # ------------------------------------------------------------------------------------ teardown

    def unload(self):
        if self.handle is not None:
            torch.cuda.synchronize(self.device)
            try:
                self.handle.drain()
            finally:
                self.handle.stop()
                self.handle = None
        self.pool = []
        self.staging = None
        self.seq = 0
        self.tokens = 0

    def stats(self) -> dict:
        if self.handle is None:
            return {}
        s = dict(self.handle.stats())
        s.update({"policy_" + k: v for k, v in self.handle.counters().items()})
        return s


class ExpertTierSet:
    """The expert tier of one component (config.expert_tiers): a TierDevice per cache GPU"""

    def __init__(self, config, placement, budget, tier_config: TierConfig | None = None):
        self.config = config
        self.placement = placement
        self.budget = budget
        self.cfg: TierConfig = tier_config or budget.tier_config
        self.devices: dict[str, TierDevice] = {}
        self.layer_of = {}              # id(module) -> (TierDevice, TierLayer)
        self.sizing = None
        self.final = False

    def register(self, module, device, hot: int):
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(f"placement: {module.key} is experts=cache, which needs a CUDA device")
        name = f"cuda:{dev.index}"
        td = self.devices.get(name)
        if td is None:
            td = self.devices[name] = TierDevice(self, dev)
        layer = td.register(module, hot)
        self.layer_of[id(module)] = (td, layer)
        return td, layer

    def unregister(self, module):
        x = self.layer_of.pop(id(module), None)
        if x is not None:
            x[0].unregister(module)

    def finalize(self, device_budget: dict, max_transient: dict, margin: int):
        """End of the layer split: size every GPU's pool from its room, then build and fill"""
        if self.final or not self.devices:
            return
        b = self.budget
        tl = b.tier_layers
        link, rooms = {}, {}
        for name, td in sorted(self.devices.items()):
            i = td.device.index
            torch.cuda.synchronize(td.device)
            link[name] = measure_link(td.device)
            alloc = torch.cuda.memory_allocated(i)
            free, _ = torch.cuda.mem_get_info(i)
            reserved = torch.cuda.memory_reserved(i)
            mt = max_transient.get(i, 0)
            room = free + reserved - alloc - mt - margin
            if i in device_budget:
                room = min(room, device_budget[i] - alloc - mt - margin)
            vs = sum(n for _, n in td.proj_shapes)
            stride = -(-vs // SLOT_ALIGN) * SLOT_ALIGN
            room -= sum(l.hot for l in td.layers) * stride
            rooms[name] = max(0, room)
        # the whole-layer prefill (two staging halves) is not in this build: one half is allocated
        cfg = dataclasses.replace(self.cfg, staging = "single")
        # read now, after the load allocated the stream / cpu arenas and the n-gram tables: held gives
        # them back, as the sizing counts them in what the component pins
        memory = host_memory()
        sizing = size_tiers(self.placement, tl, cfg, memory, vram_room = rooms, static_bytes = b.expert_bytes,
                            ngram_bytes = b.ngram_bytes, expert_cap = b.expert_cap, reserve = b.reserve,
                            held = b.expert_bytes + b.ngram_bytes, link_gbs = link)
        self.sizing = sizing
        print(sizing.text)
        # one RAM tier per GPU: the component's split by what each GPU's pool does not hold
        want = {d.device: max(0, d.cached - d.held) for d in sizing.devices}
        total = sum(want.values())
        left = sizing.tier_slots
        split = {}
        for n, (name, w) in enumerate(sorted(want.items())):
            share = left if n == len(want) - 1 else (sizing.tier_slots * w // total if total else 0)
            split[name] = share
            left -= share
        ram = self.placement.ram_store()
        disk = self.placement.disk_store()
        try:
            for d in sizing.devices:
                td = self.devices[d.device]
                policy = PolicyConfig.from_placement(self.placement.device_store(d.device), ram, disk, self.cfg)
                td.build(d.slots, split[d.device], sizing.slab_slots, policy, self.cfg)
        except Exception:
            self.unload()
            raise
        self.final = True

    def build_fixed(self, sizes: dict, slab_slots: int = 8, policy: dict | None = None):
        """Tests and tools: build every GPU's cache with given sizes, {device: (pool slots, RAM slots)},
        bypassing the sizing; `policy` overrides fields of the PolicyConfig the placement gives"""
        ram = self.placement.ram_store()
        disk = self.placement.disk_store()
        try:
            for name, td in sorted(self.devices.items()):
                slots, ram_slots = sizes[name]
                pc = PolicyConfig.from_placement(self.placement.device_store(name), ram, disk, self.cfg)
                if policy:
                    pc = dataclasses.replace(pc, **policy)
                td.build(slots, ram_slots, slab_slots, pc, self.cfg)
        except Exception:
            self.unload()
            raise
        self.final = True

    def unload(self):
        for td in self.devices.values():
            td.unload()
        self.final = False

    def stats(self) -> dict:
        return {name: td.stats() for name, td in self.devices.items()}
