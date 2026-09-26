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
    forward   every MoE call of a cache layer, right after routing, TierDevice.resolve():
                decode calls (rows <= EXL3_MOE_TIER_DECODE_ROWS) and routed calls (below
                EXL3_MOE_TIER_PREFILL_ROWS): the lookup kernel on the compute stream decides the
                call's slots and rewrites the tables, the fetch kernel copies what the call misses
                (the tier thread turns the call's record into copy commands);
                layer-mode calls (from EXL3_MOE_TIER_PREFILL_ROWS rows): the whole layer was staged
                one layer ahead on the copy engines (layer_plan, from the caller's thread, into one
                half of the staging area while the other half is computed); the compute stream waits
                for those copies and uploads the layer's tables the plan built; after the layer's
                compute, TierDevice.after_compute() plans the next cache layer
              the unchanged MoE kernels then read the tables. With prefetch=router, decode calls also
              predict a later cache layer's experts with that layer's router (disk read-ahead)

A tiered layer computes exactly what the same layer computes resident: the kernels, their arguments
and the trellis bytes are the same, only where the bytes live differs.
"""

from __future__ import annotations
import dataclasses
import os
import threading
import time

import torch

from .expert_tier_policy import PolicyConfig, DECODE, ROUTED, LAYER, round_robin_order
from .expert_tier_config import TierConfig, size_tiers, check_geometries, ineligible
from .expert_extents import build_for_modules
from .expert_tier_host import native_extents
from .disk_source import expert_files, io_direct
from . import moe_profile
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
    layer_idx: int = -1             # decoder layer
    tables: list | None = None      # the layer's int64 trellis pointer tables, projection order
    shapes: list | None = None      # the trellis shape of each projection
    prefetch_layer: int = 0         # prefetch=layer:D: read this layer's disk-only experts D layers ahead
    prefetch_router: int = 0        # prefetch=router:D: predict the experts of the cache layer D ahead
    predict_to: "TierLayer | None" = None


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
        self.prefill_rows = tier_set.cfg.prefill_rows
        self.halves = tier_set.cfg.staging_factor
        self.verify = tier_set.cfg.verify
        self.sizing = None
        self.mode = None            # mode of the last call (DECODE, ROUTED, LAYER)
        self.last_lc = -1
        self.planned = set()        # layer mode: cache layers planned in the current pass
        self.owner = None           # layer mode: the thread that drives the current pass

    # ------------------------------------------------------------------------------------ load

    def register(self, module, hot: int, layer_idx: int = -1) -> TierLayer:
        layer = TierLayer(module, len(self.layers), hot, layer_idx)
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
        """Allocate the pool (with the hot pins after it) and the staging, then the runtime; seed (the
        hot pins, the cold-fill order and the heat from the profiles or the heat file); cold fill"""
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
        seed = self.tiers.seed(self, E, policy.halflife)
        pin_keys = []
        for l in self.layers:
            pin_keys += [l.lc * E + e for e in seed.pins(l, E)]
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
        # the staging area: one half per layer in flight in layer mode (EXL3_MOE_TIER_STAGING=double
        # overlaps the next layer's copies with this layer's compute); decode and routed calls use
        # half 0 for their transients
        self.nstage = max(E - l.hot for l in self.layers)
        self.staging = torch.empty((self.halves * self.nstage * self.stride,), dtype = torch.uint8,
                                   device = self.device)
        staging = [self.staging.data_ptr() + i * self.stride for i in range(self.halves * self.nstage)]
        tables = []
        for l in self.layers:
            if l.tables is None:
                raise RuntimeError(f"expert tier: {l.module.key} registered but never finished loading")
            tables += [t.data_ptr() for t in l.tables]
        files, extents, geometry = native_extents(index)
        # disk experts=<dir>: every expert read from a checked copy of the shards (same names, same headers)
        files = expert_files(self.tiers.placement, files)
        host = {"chunk_bytes": 1 << 30, "hugepage": cfg.hugepage, "slab_slots": max(slab_slots, 8),
                "fill_inflight": 8, "prefault_threads": 8, "compact_fill": cfg.compact,
                "io_direct": io_direct(self.tiers.placement)}
        gpu = {"device": self.device.index, "pool": addrs, "staging": staging, "staging_per_half": self.nstage,
               "halves": self.halves, "tables": tables, "proj_off": self.proj_off,
               "deterministic": cfg.deterministic, "spin_us": cfg.spin_us, "affinity": list(cfg.affinity or ()),
               "trace": 0, "trace_path": self.tiers.trace_path(self),
               "prefetch_layer": [l.prefetch_layer for l in self.layers]}
        torch.cuda.synchronize(self.device)
        self.handle = _ext().TierGpu(dataclasses.asdict(policy), L, E, slots, ram_slots, self.pins, geometry, files,
                                     extents, host, gpu)
        order = seed.order(self, E, set(pin_keys))
        for k, v in seed.heat.items():
            self.handle.heat_set(k, v)
        t0 = time.monotonic()
        self.handle.cold_fill(order, pin_keys)
        if policy.disk_off:
            self.handle.release_slab()
        s = self.handle.stats()
        print(f" -- expert cache {self.name}: cold fill of {s['cold_vram']} VRAM and {s['cold_ram']} RAM experts, "
              f"{human(s['cold_bytes'])} read in {time.monotonic() - t0:.1f} s; {human(s['pinned_bytes'])} of RAM "
              f"page-locked in {s['pin_ns'] / 1e9:.1f} s; seeded from {seed.source}")
        self.handle.start()

    # ------------------------------------------------------------------------------------ forward

    def call_mode(self, rows: int) -> int:
        """DECODE up to EXL3_MOE_TIER_DECODE_ROWS rows, LAYER from EXL3_MOE_TIER_PREFILL_ROWS, ROUTED between"""
        if rows <= self.decode_rows:
            return DECODE
        return LAYER if rows >= self.prefill_rows else ROUTED

    def resolve(self, layer: TierLayer, selected: torch.Tensor, rows: int):
        """One MoE call, right after routing, on the current stream: the call's experts where the MoE
        kernels will read them, the layer's tables pointing at them, the stream waiting for their copies"""
        mode = self.call_mode(rows)
        if mode == DECODE and layer.lc == 0:
            self.tokens += 1
        self.seq = (self.seq + 1) & 0xFFFFFFFF or 1
        ids = selected if selected.is_contiguous() else selected.contiguous()
        h = self.handle
        if mode == LAYER:
            # a new pass: after other calls, or back at an earlier layer (the next chunk)
            me = threading.get_ident()
            if self.mode != LAYER or layer.lc <= self.last_lc:
                h.layer_begin()
                self.planned.clear()
                self.owner = me
            elif me != self.owner:
                self.mode = None
                raise RuntimeError(f"expert tier: {self.name}: cache layer {layer.lc} of a prefill pass was called "
                                   f"from another thread than the pass's earlier layers; the cache layers of one "
                                   f"GPU are driven by one thread at a time")
            if layer.lc not in self.planned:
                h.layer_plan(layer.lc, layer.lc % self.halves)
                self.planned.add(layer.lc)
            h.layer_run(layer.lc, ids, self.seq, self.tokens)
        else:
            h.lookup(layer.lc, ids, mode, self.seq, self.tokens)
        self.mode = mode
        self.last_lc = layer.lc
        if self.verify:
            torch.cuda.synchronize(self.device)
            self.handle.drain()
            self.handle.verify()

    def after_compute(self, layer: TierLayer):
        """The layer's MoE compute is enqueued. Layer mode: its staging half is free once that compute
        is done, and the next cache layer is planned into the other half now (its copies overlap this
        layer's compute and whatever runs in between)"""
        if self.mode != LAYER:
            return
        self.handle.layer_after(layer.lc)
        nxt = layer.lc + 1
        if nxt < len(self.layers) and nxt not in self.planned:
            self.handle.layer_plan(nxt, nxt % self.halves)
            self.planned.add(nxt)

    def predict(self, layer: TierLayer, z: torch.Tensor, rows: int):
        """prefetch=router: at a decode call of `layer`, the experts its target layer's router picks
        for this layer's router input are read from the disk ahead, when neither VRAM nor RAM holds them"""
        tgt = layer.predict_to
        if tgt is None or rows > self.decode_rows:
            return
        ids = tgt.module.tier_predict_ids(z)
        self.handle.predict(tgt.lc, ids, self.seq, self.tokens)

    def trellis_views(self, layer: TierLayer, e: int) -> list:
        """Expert e's trellis tensors where the last call put them (the per-expert dequant path; its
        caller has synchronized already). Layer mode: the addresses the plan chose (host-known)"""
        if self.mode == LAYER:
            addrs = self.handle.layer_addrs(layer.lc, e)
        else:
            addrs = torch.stack([t[e] for t in layer.tables]).tolist()
        return [self.view_at(int(a), shape, nbytes) for a, (shape, nbytes) in zip(addrs, self.proj_shapes)]

    def view_at(self, addr: int, shape: tuple, nbytes: int) -> torch.Tensor:
        # before the tier is built (the load's measuring forward) only the sentinel exists
        for t in [p[0] for p in self.pool] + [self.staging, self.sentinel]:
            if t is None:
                continue
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
        self.mode = None
        self.last_lc = -1
        self.planned.clear()

    def stats(self) -> dict:
        if self.handle is None:
            return {}
        s = dict(self.handle.stats())
        s.update({"policy_" + k: v for k, v in self.handle.counters().items()})
        return s

    def heat_rows(self, halflife: int) -> list:
        """[(layer key, [decayed routing count per expert])] of this GPU's cache layers, as the host has
        them (the heat file)"""
        if self.handle is None:
            return []
        st = self.handle.state()
        hv, he, epoch = st["heat_v"], st["heat_ep"], st["tokens"] // halflife
        out = []
        for l in self.layers:
            E = l.module.num_experts
            row = []
            for k in range(l.lc * E, (l.lc + 1) * E):
                d = epoch - he[k]
                row.append((hv[k] >> min(31, d) if d > 0 else hv[k]) / 65536.0)
            out.append((l.module.key, row))
        return out


class Seed:
    """How one GPU's cache starts: hot pins, cold-fill order and heat (profile=, else the heat file of
    EXL3_MOE_HEAT_FILE, else the first experts pinned, a round robin across layers and no heat)"""

    def __init__(self, source: str = "a round robin across layers (no profile or heat file)"):
        self.source = source
        self.shares = {}            # layer key -> per-expert share of its layer's selections
        self.heat = {}              # tier key -> 16.16 heat

    def pins(self, layer: TierLayer, E: int) -> list:
        s = self.shares.get(layer.module.key)
        if s is None or not layer.hot:
            return list(range(layer.hot))
        import numpy as np
        return sorted(int(e) for e in np.argsort(-s, kind = "stable")[:layer.hot])

    def order(self, td: TierDevice, E: int, pinned: set) -> list:
        if not self.shares:
            return round_robin_order(len(td.layers), E, pinned)
        return moe_profile.seed_order([(l.lc, l.module.key) for l in td.layers], self.shares, E, pinned)


class ExpertTierSet:
    """The expert tier of one component (config.expert_tiers): a TierDevice per cache GPU"""

    def __init__(self, config, placement, budget, tier_config: TierConfig | None = None, moe_keys: list | None = None):
        self.config = config
        self.placement = placement
        self.budget = budget
        self.cfg: TierConfig = tier_config or budget.tier_config
        self.devices: dict[str, TierDevice] = {}
        self.layer_of = {}              # id(module) -> (TierDevice, TierLayer)
        self.sizing = None
        self.final = False
        self.moe_keys = moe_keys        # every MoE layer's key in forward order (profiles without layer keys)
        self.heat_saved = 0.0           # time of the last heat file write

    def register(self, module, device, hot: int, layer_idx: int = -1):
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(f"placement: {module.key} is experts=cache, which needs a CUDA device")
        name = f"cuda:{dev.index}"
        td = self.devices.get(name)
        if td is None:
            td = self.devices[name] = TierDevice(self, dev)
        layer = td.register(module, hot, layer_idx)
        layer.prefetch_layer, layer.prefetch_router = self.prefetch_depths(layer_idx)
        self.layer_of[id(module)] = (td, layer)
        return td, layer

    def prefetch_depths(self, layer_idx: int) -> tuple:
        """(layer read-ahead depth, router look-ahead depth) of a cache layer from its prefetch= (auto:
        layer:1 when the disk holds experts, router off; the disk off: nothing to read ahead)"""
        if self.placement is None or layer_idx < 0:
            return 0, 0
        pf = self.placement.experts_for_layer(layer_idx).prefetch
        disk_off = self.placement.disk_store().experts == "off"
        if pf is None:
            return (0 if disk_off else 1), 0
        depth = dict(pf)
        return depth.get("layer", 0), depth.get("router", 0)

    def trace_path(self, td: "TierDevice") -> str:
        """EXL3_MOE_TIER_TRACE, with the GPU appended when several GPUs hold cache layers"""
        path = self.cfg.trace
        if not path:
            return ""
        return path if len(self.devices) <= 1 else f"{path}.cuda{td.device.index}"

    def link_predictions(self):
        """prefetch=router:D: each cache layer predicts the cache layer D after it on the same GPU"""
        for td in self.devices.values():
            for l in td.layers:
                t = l.lc + l.prefetch_router
                l.predict_to = td.layers[t] if l.prefetch_router and t < len(td.layers) else None

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
        cfg = self.cfg
        # read now, after the load allocated the stream / cpu arenas and the n-gram tables: held gives
        # back what of them is resident, as the sizing counts them in what the component pins (a row
        # cache's rows are faulted in as rows arrive: only its tags are resident yet)
        memory = host_memory()
        held = b.expert_bytes + b.ngram_resident_bytes()
        ram = self.placement.ram_store()
        req = ram.experts if ram.experts is not None else b.expert_cap
        if b.ngram_budget.is_auto and b.ngram_caches and req is not None and not req.is_auto:
            # ram ngram=auto sized its row caches before the tier's RAM was known: they get what is
            # left once the tier, the whole tables, the reserve and the page cache kept free are counted
            from .ram_budget import refit_row_caches, pagecache_auto
            whole = sum(t.nbytes for t in b.ngram_held)
            probe = size_tiers(self.placement, tl, cfg, memory, vram_room = rooms, static_bytes = b.expert_bytes,
                               ngram_bytes = whole, expert_cap = b.expert_cap, reserve = b.reserve, held = held,
                               link_gbs = link)
            pagecache = pagecache_auto(memory.mem_total) if ram.pagecache.is_auto else ram.pagecache.nbytes
            if probe.left_unpinned is not None and \
                    refit_row_caches(b, self.config, probe.left_unpinned - b.reserve - pagecache):
                print(f" -- n-gram row caches refitted to the expert tier: {human(b.ngram_bytes - whole)} "
                      f"({', '.join(f'{k} {human(v)}' for k, v in b.ngram_caches.items()) or 'none'})")
                memory = host_memory()
                held = b.expert_bytes + b.ngram_resident_bytes()
        sizing = size_tiers(self.placement, tl, cfg, memory, vram_room = rooms, static_bytes = b.expert_bytes,
                            ngram_bytes = b.ngram_bytes, expert_cap = b.expert_cap, reserve = b.reserve,
                            held = held, link_gbs = link)
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
            self.link_predictions()
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
            self.link_predictions()
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
        if self.final and self.cfg.heat_file:
            try:
                self.save_heat()
            except Exception as e:
                print(f" !! expert tier: the heat file {self.cfg.heat_file} was not written: {e}")
        for td in self.devices.values():
            td.unload()
        self.final = False

    def stats(self) -> dict:
        return {name: td.stats() for name, td in self.devices.items()}

    # ------------------------------------------------------------------------------------ seeding

    def seed(self, td: TierDevice, E: int, halflife: int) -> Seed:
        """The seed of one GPU's cache: the profiles its layers' rules name (profile=), else the heat
        file (EXL3_MOE_HEAT_FILE, when it exists and names these layers), else none"""
        specs = {}
        for l in td.layers:
            spec = self.placement.experts_for_layer(l.layer_idx).profile if self.placement and l.layer_idx >= 0 else None
            if spec is not None:
                specs.setdefault(spec, []).append(l)
        if specs:
            seed = Seed("profile " + ", ".join(sorted(specs)))
            fp = moe_profile.model_fingerprint(self.config, E)
            keys = self.moe_keys or [l.module.key for l in td.layers]
            model_dir = getattr(self.config, "directory", None)
            for spec, layers in specs.items():
                prof = moe_profile.Profile(spec, keys, E, model_dir, fp)
                for l in layers:
                    share = prof.shares(l.module.key)
                    if share is None:
                        continue
                    seed.shares[l.module.key] = share
                    top_k = l.module.num_experts_per_tok
                    for e in range(E):
                        seed.heat[l.lc * E + e] = moe_profile.seed_heat(float(share[e]), top_k, halflife)
            return seed
        path = self.cfg.heat_file
        if path and os.path.isfile(path):
            try:
                counts, keys, fp = moe_profile.load_counts(path)
            except Exception as e:
                print(f" !! expert tier: the heat file {path} is not readable ({e}); starting without it")
                return Seed()
            mine = moe_profile.model_fingerprint(self.config, E)
            bad = [k for k in moe_profile.MODEL_KEYS if fp and k in fp and k in mine and fp[k] != mine[k]]
            if bad:
                print(f" !! expert tier: the heat file {path} was written for another model ({', '.join(bad)} "
                      f"differ); starting without it")
                return Seed()
            rows = dict(zip(keys or [], counts))
            seed = Seed(f"the heat file {path}")
            for l in td.layers:
                row = rows.get(l.module.key)
                if row is None or len(row) != E:
                    continue
                tot = float(row.sum())
                seed.shares[l.module.key] = row / tot if tot > 0 else row
                for e in range(E):
                    seed.heat[l.lc * E + e] = max(0, min((1 << 32) - 1, int(round(float(row[e]) * 65536))))
            if seed.shares:
                return seed
        return Seed()

    def save_heat(self):
        """Write the heat of every cache layer to EXL3_MOE_HEAT_FILE (atomically; the .exl3moe format)"""
        import numpy as np
        rows = []
        for td in self.devices.values():
            if td.handle is not None:
                torch.cuda.synchronize(td.device)
                td.handle.drain()
                rows += td.heat_rows(self.cfg.heat_halflife)
        if not rows:
            return
        E = max(len(r) for _, r in rows)
        fp = moe_profile.model_fingerprint(self.config, E)
        moe_profile.write_safetensors(self.cfg.heat_file, {"counts": np.asarray([r for _, r in rows], dtype = np.float64)},
                                      {"layer_keys": [k for k, _ in rows], "fingerprint": fp,
                                       "format": "exl3-heat 1: decayed routing counts of the expert tier"})
        self.heat_saved = time.monotonic()

    def quiescent(self):
        """Between generations (the generator's queue drained): the heat file, at most once a minute"""
        if self.final and self.cfg.heat_file and time.monotonic() - self.heat_saved >= 60.0:
            try:
                self.save_heat()
            except Exception as e:
                print(f" !! expert tier: the heat file {self.cfg.heat_file} was not written: {e}")
