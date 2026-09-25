"""
Configuration and sizing of the expert tier: the experts=cache layers of a placement, whose routed
experts live in a VRAM cache per GPU, over a RAM tier, over the checkpoint on the SSD
(doc/expert_tiers.md). Torch-free.

- TierConfig: the tuning a load reads from the environment (EXL3_MOE_TIER_*, EXL3_MOE_HEAT_*),
  checked in full: a malformed value or an unknown name under those prefixes is an error, never
  a silent default. The placement's words (cache=, spare=, evict=, admit=, policy=, demote=,
  ...) say what the tier does; these say how.
- size_tiers: the pools one load allocates (VRAM slots per GPU, prefill staging, RAM tier slots,
  disk slab) from the placement, the model's expert geometry, the room each GPU has left and the
  host memory, with every load-time refusal of the tier and the summary the load prints. Stage 1
  of a load (before any module loads) calls it without VRAM rooms and gets the checks that do not
  need them; the tier's finalization (at the end of the layer split) calls it with them.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import difflib
import os
import re
import sys

try:
    from ..util.host_budget import Size, AUTO, GiB, MiB, human, HostLedger, HostMemory
except ImportError:
    # Loaded by file path (the torch-free tests) or under a stub package: the sibling by path too
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _name = "_exl3_torch_free_host_budget"
    _hb = sys.modules.get(_name)
    if _hb is None:
        _spec = _ilu.spec_from_file_location(_name, _Path(__file__).resolve().parents[1] / "util" / "host_budget.py")
        _hb = _ilu.module_from_spec(_spec)
        sys.modules[_name] = _hb
        _spec.loader.exec_module(_hb)
    Size, AUTO, GiB, MiB = _hb.Size, _hb.AUTO, _hb.GiB, _hb.MiB
    human, HostLedger, HostMemory = _hb.human, _hb.HostLedger, _hb.HostMemory

MAX_BSZN = 8        # rows of the largest decode-shaped MoE call (block_sparse_mlp.py, blocksparse_mlp.h)
PREFIXES = ("EXL3_MOE_TIER_", "EXL3_MOE_HEAT_")


# ---------------------------------------------------------------------------------------- settings

@dataclass(frozen = True)
class Setting:
    """One tuning variable: its name, the TierConfig field it sets, its default and its values"""
    var: str
    field: str
    default: object
    kind: str                       # bool, int, float, choice, path, slots, cpus
    lo: float | None = None
    hi: float | None = None
    choices: tuple = ()
    help: str = ""


SETTINGS = (
    Setting("EXL3_MOE_TIER_DEMOTE", "demote", True, "bool",
            help = "master switch of demotion: 0 drops every VRAM victim whatever ram demote= says"),
    Setting("EXL3_MOE_TIER_STAGING", "staging", "double", "choice", choices = ("double", "single"),
            help = "prefill staging: double = 2 x (E - hot) slots (copies overlap compute), single = E - hot"),
    Setting("EXL3_MOE_TIER_PREFILL_ROWS", "prefill_rows", 256, "int", 2, 1 << 20,
            help = "rows from which a call runs in layer mode (whole-layer double buffering)"),
    Setting("EXL3_MOE_TIER_DECODE_ROWS", "decode_rows", MAX_BSZN, "int", 1, MAX_BSZN,
            help = "rows up to which a call is a decode call (admits misses into the cache)"),
    Setting("EXL3_MOE_HEAT_HALFLIFE", "heat_halflife", 256, "int", 1, 1 << 30,
            help = "decode tokens per halving of an expert's heat"),
    Setting("EXL3_MOE_HEAT_PREFILL", "heat_prefill", 0.0625, "float", 0.0, 1.0,
            help = "heat one prefill assignment adds (a decode assignment adds 1)"),
    Setting("EXL3_MOE_HEAT_FILE", "heat_file", None, "path",
            help = "file the heat is saved to at unload and quiescent points, and read at load"),
    Setting("EXL3_MOE_TIER_ADMIT_P", "admit_p", None, "float", 0.0, 1.0,
            help = "pins the admission probability of admit=adaptive (tests, A/B)"),
    Setting("EXL3_MOE_TIER_ADAPT_EVERY", "adapt_every", 2048, "int", 1, 1 << 30,
            help = "accesses between two adaptations of the admission probability"),
    Setting("EXL3_MOE_TIER_ADMIT_PMIN", "admit_pmin", 0.05, "float", 0.0001, 1.0,
            help = "lower bound of the adaptive admission probability"),
    Setting("EXL3_MOE_TIER_SAMPLE", "sample", 16, "int", 1, 4096,
            help = "RAM tier entries sampled to find the coldest one"),
    Setting("EXL3_MOE_TIER_RAM_ADMIT", "ram_admit", "always", "choice", choices = ("always", "heat"),
            help = "RAM admission of an SSD fill: always (free, duplicate, then coldest) or heat (the coldest only when colder)"),
    Setting("EXL3_MOE_TIER_REFILL", "refill", True, "bool",
            help = "background reads that bring warm SSD-only experts into the RAM tier"),
    Setting("EXL3_MOE_TIER_REFILL_INFLIGHT", "refill_inflight", 2, "int", 1, 64,
            help = "refill reads in flight at most; further candidates are dropped, not queued"),
    Setting("EXL3_MOE_TIER_DISK_SLAB", "disk_slab", None, "slots", 1, 4096,
            help = "pinned slots for SSD reads that bypass the RAM tier; auto = 8 + the prefill read-ahead"),
    Setting("EXL3_MOE_TIER_HUGEPAGE", "hugepage", False, "bool",
            help = "transparent huge pages for the RAM tier and the disk slab (off: faulting 2 MiB pages in "
                   "can stall for seconds when the page cache is large)"),
    Setting("EXL3_MOE_TIER_COMPACT", "compact", True, "bool",
            help = "move the cold-filled RAM tier slots to the compact layout (one aligned copy per promotion: "
                   "49 against 40 GB/s from the odd offsets of the extent layout on the AI VM)"),
    Setting("EXL3_MOE_TIER_HEADROOM_MB", "headroom_mb", 1024, "int", 0, 1 << 20,
            help = "VRAM (MiB) left free on each GPU after its expert cache"),
    Setting("EXL3_MOE_TIER_MIN_LINK_GBS", "min_link_gbs", 2.0, "float", 0.0, 1e6,
            help = "host link (GB/s, measured at load) below which a GPU may not hold an expert cache; 0 allows any"),
    Setting("EXL3_MOE_TIER_SPIN_US", "spin_us", -1, "int", -1, 1 << 30,
            help = "tier thread spin: -1 while a forward pass is active, 0 never, N microseconds"),
    Setting("EXL3_MOE_TIER_AFFINITY", "affinity", None, "cpus",
            help = "CPUs the tier thread runs on (e.g. 16-23 or 4,6,8)"),
    Setting("EXL3_MOE_TIER_DETERMINISTIC", "deterministic", False, "bool",
            help = "the tier thread applies every call before the next one (replays, debugging)"),
    Setting("EXL3_MOE_TIER_VERIFY", "verify", False, "bool",
            help = "check the tier's invariants after every call (synchronizes; tests, debugging)"),
    Setting("EXL3_MOE_TIER_TRACE", "trace", None, "path",
            help = "file the tier writes every call record to"),
)
_BY_VAR = {s.var: s for s in SETTINGS}


def _parse_setting(s: Setting, raw: str):
    v = raw.strip()
    bad = f"{s.var}={raw!r}"
    if s.kind == "bool":
        if v not in ("0", "1"):
            raise ValueError(f"{bad} must be 0 or 1")
        return v == "1"
    if s.kind == "choice":
        if v.lower() not in s.choices:
            m = difflib.get_close_matches(v.lower(), s.choices, n = 1, cutoff = 0.6)
            raise ValueError(f"{bad} must be {' or '.join(s.choices)}" + (f" (did you mean {m[0]!r}?)" if m else ""))
        return v.lower()
    if s.kind == "path":
        if not v:
            raise ValueError(f"{bad} must be a path (unset it to disable)")
        return v
    if s.kind == "cpus":
        cpus = []
        for part in v.split(","):
            m = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", part.strip())
            if not m or (m.group(2) is not None and int(m.group(2)) < int(m.group(1))):
                raise ValueError(f"{bad} must be a CPU list such as 16-23 or 4,6,8")
            cpus.extend(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1))
        return tuple(sorted(set(cpus)))
    if s.kind == "slots":
        if v.lower() == "auto":
            return None
        if not re.fullmatch(r"[0-9]+", v) or not s.lo <= int(v) <= s.hi:
            raise ValueError(f"{bad} must be auto or a number of slots from {s.lo} to {s.hi}")
        return int(v)
    if s.kind == "int":
        if not re.fullmatch(r"-?[0-9]+", v) or not s.lo <= int(v) <= s.hi:
            raise ValueError(f"{bad} must be a whole number from {s.lo} to {s.hi}")
        return int(v)
    if s.kind == "float":
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v) or not s.lo <= float(v) <= s.hi:
            raise ValueError(f"{bad} must be a number from {s.lo:g} to {s.hi:g}")
        return float(v)
    raise AssertionError(s.kind)


@dataclass(frozen = True)
class TierConfig:
    """The expert tier's tuning, one field per Setting (defaults as SETTINGS lists them)"""
    demote: bool = True
    staging: str = "double"
    prefill_rows: int = 256
    decode_rows: int = MAX_BSZN
    heat_halflife: int = 256
    heat_prefill: float = 0.0625
    heat_file: str | None = None
    admit_p: float | None = None
    adapt_every: int = 2048
    admit_pmin: float = 0.05
    sample: int = 16
    ram_admit: str = "always"
    refill: bool = True
    refill_inflight: int = 2
    disk_slab: int | None = None
    hugepage: bool = False
    compact: bool = True
    headroom_mb: int = 1024
    min_link_gbs: float = 2.0
    spin_us: int = -1
    affinity: tuple | None = None
    deterministic: bool = False
    verify: bool = False
    trace: str | None = None

    @classmethod
    def from_env(cls, environ = None) -> "TierConfig":
        """Read and check every variable under EXL3_MOE_TIER_ / EXL3_MOE_HEAT_. Raises ValueError for
        a malformed value, an unknown name under those prefixes, or values that contradict"""
        environ = os.environ if environ is None else environ
        kw = {}
        for var in sorted(k for k in environ if k.startswith(PREFIXES)):
            s = _BY_VAR.get(var)
            if s is None:
                m = difflib.get_close_matches(var, list(_BY_VAR), n = 1, cutoff = 0.8)
                raise ValueError(f"{var} is not an expert tier setting" + (f" (did you mean {m[0]}?)" if m else "") +
                                 f"; the settings are listed in doc/expert_tiers.md")
            kw[s.field] = _parse_setting(s, str(environ[var]))
        cfg = cls(**kw)
        if cfg.prefill_rows <= cfg.decode_rows:
            raise ValueError(f"EXL3_MOE_TIER_PREFILL_ROWS={cfg.prefill_rows} must be above EXL3_MOE_TIER_DECODE_ROWS="
                             f"{cfg.decode_rows} (a call is decode, routed or layer mode by its rows)")
        return cfg

    @property
    def heat_prefill_q16(self) -> int:
        """EXL3_MOE_HEAT_PREFILL in the tier's 16.16 fixed point"""
        return int(round(self.heat_prefill * 65536))

    @property
    def staging_factor(self) -> int:
        return 2 if self.staging == "double" else 1

    def changed(self) -> list[tuple[str, object]]:
        """(variable, value) of every setting that differs from its default, for the load summary"""
        out = []
        for s in SETTINGS:
            v = getattr(self, s.field)
            if v != s.default:
                out.append((s.var, v))
        return out


# ---------------------------------------------------------------------------------------- sizing

@dataclass
class TierLayers:
    """The experts=cache layers of one component, as the loader knows them from the headers"""
    experts: dict                   # layer index -> routed experts E
    hot: dict                       # layer index -> hot count (resident pins, not cached)
    device: dict                    # layer index -> "cuda:<n>"
    top_k: int
    vram_slot: int                  # bytes of one expert in a VRAM slot (the three trellis tensors)
    ram_slot: int                   # bytes of one RAM tier slot (the expert's extent, 4 KiB-rounded, + 4 KiB)

    def cached(self, device: str | None = None) -> int:
        """Routed experts the tier serves (E - hot per layer), on one device or all"""
        return sum(self.experts[i] - self.hot.get(i, 0) for i in self.experts
                   if device is None or self.device[i] == device)

    def devices(self) -> list[str]:
        return sorted(set(self.device.values()), key = lambda d: int(d[5:]))


@dataclass
class DeviceTier:
    device: str
    cached: int                     # experts the cache serves on this GPU
    slots: int                      # S: pool slots
    spare: int
    held: int                       # experts the pool owns (S - spare, or everything when it fits)
    staging_slots: int
    vram_slot: int

    @property
    def pool_bytes(self) -> int:
        return self.slots * self.vram_slot

    @property
    def staging_bytes(self) -> int:
        return self.staging_slots * self.vram_slot


@dataclass
class TierSizing:
    devices: list = field(default_factory = list)       # DeviceTier
    cached: int = 0                 # N_c
    hot: int = 0
    static_bytes: int = 0           # arenas of stream / cpu / split layers
    tier_slots: int = 0             # R
    ram_bytes: int = 0              # static + R x ram_slot
    ngram_bytes: int = 0
    slab_slots: int = 0
    slab_bytes: int = 0
    disk_only: int = 0              # cached experts neither VRAM nor RAM holds
    left_unpinned: int | None = None
    final: bool = False             # sized with the GPUs' rooms (not the stage-1 checks alone)
    ram_sized: bool = False         # the RAM tier's size is known (an explicit size, or final)
    counted: bool = False           # the disk-only experts are known (every pool sized)
    text: str = ""


def pagecache_auto(mem_total: int | None) -> int:
    """ram pagecache=auto: max(8 GiB, MemTotal / 8)"""
    return max(8 * GiB, (mem_total or 0) // 8)


def size_tiers(placement, layers: TierLayers, cfg: TierConfig, memory: HostMemory, *,
               vram_room: dict | None = None, static_bytes: int = 0, ngram_bytes: int = 0,
               expert_cap: Size | None = None, reserve: int = 2 * GiB, already_pinned: int = 0,
               held: int = 0, link_gbs: dict | None = None, platform: str | None = None) -> TierSizing:
    """
    Size the expert tier of one component and check it, before anything of the tier is allocated.

      layers        the component's experts=cache layers (TierLayers)
      vram_room     per cache GPU, the bytes left for the tier after the placed modules, the Cache,
                    the generator's statics and the loader's margins; None at stage 1 (the checks
                    that need no VRAM figure, and explicit sizes, run; auto / all tier sizes wait)
      static_bytes  the arenas of the component's stream / cpu / split layers (ram_budget)
      ngram_bytes   the n-gram tables the component holds in RAM (ram_budget)
      expert_cap    --expert_ram / --draft_expert_ram
      held          the bytes of static_bytes and ngram_bytes the component holds already: at the end of
                    the load its stream / cpu arenas and n-gram tables are allocated, and `memory`, read
                    then, no longer counts them; they are given back to it here, so the ledger, which
                    counts them, does not take them twice (0 at the start of the load)
      link_gbs      per cache GPU, its measured host link (GB/s); None: not measured
      platform      sys.platform (tests pass another)

    Raises ValueError for a request that contradicts the placement or the flags, RuntimeError for
    what the hardware cannot hold. Returns the sizes and the load's summary text
    """
    ram = placement.ram_store()
    disk = placement.disk_store()
    out = TierSizing(static_bytes = static_bytes, ngram_bytes = ngram_bytes, final = vram_room is not None)
    if held > 0 and memory.available is not None:
        memory = replace(memory, available = memory.available + held,
                         mem_available = None if memory.mem_available is None else memory.mem_available + held,
                         cgroup_room = None if memory.cgroup_room is None else memory.cgroup_room + held)
    if not (platform or sys.platform).startswith("linux"):
        raise RuntimeError("placement: experts=cache needs the disk engine, which is Linux-only")
    if not cfg.demote and disk.experts == "off" and ram.policy != "inclusive":
        raise ValueError("EXL3_MOE_TIER_DEMOTE=0 would drop VRAM victims, but disk experts=off keeps no other "
                         "copy of them")
    for dev, gbs in sorted((link_gbs or {}).items()):
        if dev in layers.devices() and cfg.min_link_gbs > 0 and gbs < cfg.min_link_gbs:
            raise RuntimeError(f"placement: {dev} reaches host memory at {gbs:.2f} GB/s (measured at load), below "
                               f"the {cfg.min_link_gbs:.1f} GB/s an expert cache needs (EXL3_MOE_TIER_MIN_LINK_GBS); "
                               f"keep its layers experts=vram, or set EXL3_MOE_TIER_MIN_LINK_GBS=0 to allow it")
    out.cached = layers.cached()
    out.hot = sum(layers.hot.values())
    vs, rs = layers.vram_slot, layers.ram_slot

    # VRAM: one pool per GPU, after the prefill staging and the headroom
    for dev in layers.devices():
        d = placement.device_store(dev)
        n = layers.cached(dev)
        widest = max(layers.experts[i] - layers.hot.get(i, 0) for i in layers.experts if layers.device[i] == dev)
        staging = cfg.staging_factor * widest
        need = layers.top_k + d.spare
        if vram_room is None:
            slots = min(d.cache.nbytes // vs, n) if d.cache.is_bytes else None
        else:
            room = vram_room.get(dev, 0) - staging * vs - cfg.headroom_mb * MiB
            if d.cache.is_bytes and d.cache.nbytes > max(0, room):
                raise RuntimeError(f"placement: {dev} cache={d.cache} does not fit: {human(max(0, room))} is free on "
                                   f"{dev} after the placed modules, the cache and the margins (the prefill staging, "
                                   f"{human(staging * vs)} with EXL3_MOE_TIER_STAGING={cfg.staging}, and "
                                   f"EXL3_MOE_TIER_HEADROOM_MB={cfg.headroom_mb} are set aside first)")
            if d.cache.is_auto and room < need * vs:
                raise RuntimeError(f"placement: {dev} has {human(max(0, room))} left for the expert cache after the "
                                   f"placed modules, the margins and the prefill staging ({human(staging * vs)}, "
                                   f"EXL3_MOE_TIER_STAGING={cfg.staging}); an expert cache needs at least top-k + spare "
                                   f"= {need} slots ({human(need * vs)}); set EXL3_MOE_TIER_STAGING=single, raise -gs on "
                                   f"{dev}, or move layers")
            slots = min((d.cache.nbytes if d.cache.is_bytes else max(0, room)) // vs, n)
        if slots is not None and not d.cache.is_zero and slots < need and slots < n:
            raise RuntimeError(f"placement: {dev} cache={d.cache} holds {slots} expert slots; experts=cache needs at "
                               f"least top-k ({layers.top_k}) + spare ({d.spare}) = {need} slots ({human(need * vs)}), "
                               f"or cache=0")
        held = None if slots is None else (slots if slots >= n else max(0, slots - d.spare))
        out.devices.append(DeviceTier(dev, n, slots if slots is not None else 0, d.spare,
                                      held if held is not None else 0, staging, vs))
    # every pool known: with the GPUs' rooms, or explicit cache= sizes alone at stage 1
    sized = vram_room is not None or all(placement.device_store(d.device).cache.is_bytes for d in out.devices)
    vram_held = sum(d.held for d in out.devices)

    # RAM: the static arenas first, the rest is the tier
    exclusive = ram.policy != "inclusive"
    tier_all = (max(0, out.cached - vram_held) if exclusive else out.cached) if sized else None
    pagecache = pagecache_auto(memory.mem_total) if ram.pagecache.is_auto else ram.pagecache.nbytes
    avail = memory.available if memory.available is not None else 1 << 62
    auto_room = max(0, avail - already_pinned - reserve - pagecache)
    req = ram.experts if ram.experts is not None else (expert_cap if expert_cap is not None else AUTO)
    if ram.experts is not None and ram.experts.is_bytes and expert_cap is not None and ram.experts.nbytes > expert_cap.nbytes:
        raise ValueError(f"placement asks ram experts={ram.experts} ({human(ram.experts.nbytes)}) but --expert_ram "
                         f"caps this component at {expert_cap}")
    if req.is_bytes:
        want = req.nbytes
    elif tier_all is None:
        want = None                     # auto / all: sized with the GPUs' rooms
    elif req.is_all:
        want = static_bytes + tier_all * rs
        if expert_cap is not None and want > expert_cap.nbytes:
            raise ValueError(f"placement asks ram experts=all ({human(want)}) but --expert_ram caps this component "
                             f"at {expert_cap}")
    else:
        want = min(static_bytes + tier_all * rs, max(0, auto_room - ngram_bytes))
        if expert_cap is not None:
            want = min(want, expert_cap.nbytes)
    if want is not None and want < static_bytes:
        raise ValueError(f"placement: the stream / cpu layers keep {human(static_bytes)} of routed experts in RAM, more "
                         f"than ram experts={req}; raise it or move layers to experts=cache")
    if want is not None:
        cap_slots = (want - static_bytes) // rs
        out.tier_slots = min(cap_slots, tier_all) if tier_all is not None else min(cap_slots, out.cached)
        # a budget that leaves the tier less than one decode step (unless VRAM holds nearly everything)
        if out.cached and 0 < cap_slots < layers.top_k and (tier_all is None or tier_all > cap_slots):
            raise ValueError(f"placement: ram experts={req} leaves the RAM tier {cap_slots} slots, fewer than one "
                             f"decode step uses (top-k = {layers.top_k}); raise it, or use ram experts=0 (a VRAM cache "
                             f"straight over the disk)")
    out.ram_bytes = static_bytes + out.tier_slots * rs
    out.ram_sized = want is not None

    # inclusive keeps a RAM copy of everything the VRAM cache holds
    if not exclusive and sized and want is not None:
        partial = [d for d in out.devices if d.slots < d.cached]
        floor = sum(d.slots + d.spare for d in partial)
        if partial and out.tier_slots <= floor:
            raise ValueError(f"placement: policy=inclusive keeps a RAM copy of every expert the VRAM cache holds, so "
                             f"ram experts= must hold more than the cache's {sum(d.slots for d in partial)} + "
                             f"{sum(d.spare for d in partial)} slots ({human((floor + 1) * rs)}); it holds "
                             f"{out.tier_slots} ({human(out.tier_slots * rs)})")

    # the disk: what neither VRAM nor RAM holds
    if sized and want is not None:
        out.counted = True
        covered = (vram_held + out.tier_slots) if exclusive else out.tier_slots
        out.disk_only = max(0, out.cached - covered)
        if disk.experts == "off" and out.disk_only:
            raise RuntimeError(f"placement: disk experts=off needs every cached expert in VRAM or RAM, but "
                               f"{out.disk_only} of {out.cached} fit in neither ({vram_held} VRAM slots besides the "
                               f"spare ones, {out.tier_slots} RAM slots); raise ram experts= by "
                               f"{human(out.disk_only * rs)} or allow disk reads")
    if disk.experts != "off":
        out.slab_slots = cfg.disk_slab if cfg.disk_slab is not None else MAX_BSZN
        out.slab_bytes = out.slab_slots * rs

    # one host check of everything the component pins
    ledger = HostLedger(memory = memory, reserve = reserve)
    ledger.add("ram experts", out.ram_bytes)
    ledger.add("ngram", ngram_bytes)
    ledger.add("disk slab", out.slab_bytes)
    ledger.check("placement" if not already_pinned else
                 f"placement (after {human(already_pinned)} pinned by the draft component)")
    if memory.available is not None:
        out.left_unpinned = memory.available - already_pinned - ledger.total
    out.text = tier_summary(placement, out, cfg)
    return out


def check_geometries(shapes: dict):
    """One expert slot size per GPU: shapes = {device: {layer key: expert shape}} of its cache layers"""
    for dev, by_key in sorted(shapes.items()):
        distinct = sorted({tuple(v) for v in by_key.values()})
        if len(distinct) > 1:
            raise ValueError(f"placement: the experts=cache layers on {dev} have different expert shapes "
                             f"({distinct[0]} and {distinct[1]}); one expert slot size per GPU is supported")


def ineligible(key: str) -> str:
    """The refusal of a layer that cannot be an experts=cache layer"""
    return (f"placement: {key} cannot use experts=cache: it needs the fused MoE kernel (EXL3 mul1 codebook shared "
            f"by gate, up and down; silu, silu_ref or gelu gated, or relu2 gateless; no per-expert biases; unpadded "
            f"dims; at most 512 experts); use experts=vram, stream or cpu for this layer")


def tier_summary(placement, s: TierSizing, cfg: TierConfig) -> str:
    """The lines a load prints for its expert tier"""
    ram = placement.ram_store()
    disk = placement.disk_store()
    devs = []
    for d in s.devices:
        st = placement.device_store(d.device)
        pool = f"{d.slots} slots ({human(d.pool_bytes)}" if s.final or d.slots else "auto (sized at the end of the load"
        devs.append(f"{d.device} cache {pool}, {d.spare} spare, evict={st.evict} admit={st.admit}; "
                    f"staging {d.staging_slots} slots ({human(d.staging_bytes)}, {cfg.staging}))")
    pol = ram.policy if ram.policy == "inclusive" else f"{ram.policy} demote={ram.demote}"
    if not cfg.demote and ram.policy != "inclusive":
        pol += " (EXL3_MOE_TIER_DEMOTE=0: off)"
    tier = f"tier {s.tier_slots} slots" if s.ram_sized else f"tier sized at the end of the load (experts={ram.experts or 'auto'})"
    lines = [f" -- expert tiers: " + "; ".join(devs),
             f"    ram experts {human(s.ram_bytes)} pinned (static {human(s.static_bytes)} + {tier}), "
             f"policy {pol}, evict={ram.evict}",
             f"    ngram {human(s.ngram_bytes)}" + ("" if s.ngram_bytes else " (every row streams from disk)"),
             f"    disk experts={disk.experts} io={disk.io}: "
             + (f"{s.disk_only} experts on disk only" if s.counted else "experts on disk only counted at the end of the load")
             + (f", slab {s.slab_slots} slots ({human(s.slab_bytes)})" if s.slab_slots else "")
             + (f"; {human(s.left_unpinned)} of RAM left unpinned" if s.left_unpinned is not None else "")]
    changed = cfg.changed()
    if changed:
        lines.append("    tuning: " + " ".join(f"{k}={v if not isinstance(v, bool) else int(v)}" for k, v in changed))
    return "\n".join(lines)
