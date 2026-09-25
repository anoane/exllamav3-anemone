"""
The host-memory budgets of one model component at load time (doc/expert_tiers.md, "RAM budgets"):
the routed experts it keeps in system RAM and its n-gram tables, against

    caps      -er / --expert_ram, EXL3_EXPERT_RAM           routed experts of the main model
              -der / --draft_expert_ram                     those of the draft model / MTP head
              -ngr / --ngram_ram [SIZE], EXL3_NGRAM_RAM     n-gram / engram tables
    requests  the placement's ram rule: ram experts=<size|all|auto> ngram=<size|all|auto>

checked, item by item, against the host memory the process may still take (util/host_budget.py)
before anything is allocated. Model.load_gen calls plan_component() once per load; the CPU
worker's registration (charge_expert_ram) and the n-gram module's load (ngram_in_ram) consult its
result. Torch-free: the plan reads checkpoint headers, never tensor data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import prod
import os

try:
    from ..util.host_budget import (Size, AUTO, ALL, ZERO, GiB, parse_size, fmt_size, human, HostLedger,
                                    HostMemory, host_memory, reserve_bytes)
except ImportError:
    # Loaded by file path (the torch-free tests) or under a stub package: the sibling by path too
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path
    _name = "_exl3_torch_free_host_budget"
    _hb = _sys.modules.get(_name)
    if _hb is None:
        _spec = _ilu.spec_from_file_location(_name, _Path(__file__).resolve().parents[1] / "util" / "host_budget.py")
        _hb = _ilu.module_from_spec(_spec)
        _sys.modules[_name] = _hb
        _spec.loader.exec_module(_hb)
    Size, AUTO, ALL, ZERO, GiB = _hb.Size, _hb.AUTO, _hb.ALL, _hb.ZERO, _hb.GiB
    parse_size, fmt_size, human, HostLedger = _hb.parse_size, _hb.fmt_size, _hb.human, _hb.HostLedger
    HostMemory, host_memory, reserve_bytes = _hb.HostMemory, _hb.host_memory, _hb.reserve_bytes


# ---------------------------------------------------------------------------------------- flags

def as_expert_ram(value, what: str = "expert_ram") -> Size | None:
    """A cap on the RAM a component's routed experts may take (--expert_ram, --draft_expert_ram,
    EXL3_EXPERT_RAM, infer_params.expert_ram / draft_expert_ram): a size, or None / "" for no
    cap. Raises ValueError"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return parse_size(value, what, allow = ())


def as_ngram_ram(value, what: str = "ngram_ram") -> Size | None:
    """The n-gram / engram table budget (--ngram_ram [SIZE], EXL3_NGRAM_RAM,
    infer_params.ngram_ram): a size, 'all' (every table whole in RAM), True (the bare flag: all),
    False (0: every row streams from disk) or None / "" (unset: 0 unless the placement asks).
    Raises ValueError"""
    if value is True:
        return ALL
    if value is False:
        return ZERO
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return parse_size(value, what, allow = ("all",))


def ngram_ram_from_env(environ = None) -> Size | None:
    """EXL3_NGRAM_RAM, with the older EXL3_NGRAM_STREAM: 0 means every table whole in RAM (all),
    other values change nothing"""
    environ = os.environ if environ is None else environ
    size = as_ngram_ram(environ.get("EXL3_NGRAM_RAM"), "EXL3_NGRAM_RAM")
    if str(environ.get("EXL3_NGRAM_STREAM", "")).strip() == "0":
        if size is not None and not size.is_all:
            raise ValueError(f"EXL3_NGRAM_STREAM=0 (every n-gram table in RAM) contradicts "
                             f"EXL3_NGRAM_RAM={size}; unset EXL3_NGRAM_STREAM")
        return ALL
    return size


def component_cap(infer_params, component: str) -> tuple[Size | None, str]:
    """(cap, flag) on the RAM of a component's routed experts: the main model's text component
    takes --expert_ram, any other (MTP head, ...) --draft_expert_ram. A draft model loaded from its
    own directory is the text component of its own config, whose expert_ram model_init sets from
    --draft_expert_ram"""
    if component == "text":
        return as_expert_ram(getattr(infer_params, "expert_ram", None)), "--expert_ram"
    return as_expert_ram(getattr(infer_params, "draft_expert_ram", None)), "--draft_expert_ram"


# ---------------------------------------------------------------------------------------- bytes

def _a64(n: int) -> int:
    return (n + 63) & ~63


def arena_expert_bytes(proj_dims: dict, bias = False) -> int:
    """
    Bytes one routed expert takes in the CPU worker's arena (moe_cpu_host._HugeArena.rehome), from
    its projection dims {"g": (in, out, K) or None, "u": ..., "d": ...}: per projection the int16
    trellis, then fp16 suh (in), svh (out) and bias (out, when present: `bias` is one bool or one
    per projection, {"g": .., "u": .., "d": ..}), each 64-byte aligned. DeepSeek-V4.1 3.0 bpw:
    3 x 4,423,680 + 44,544 = 13,315,584 bytes
    """
    total = 0
    for p in ("g", "u", "d"):
        d = proj_dims.get(p)
        if not d:
            continue
        k, n, K = d
        has_bias = bias.get(p, False) if isinstance(bias, dict) else bias
        total += _a64((k // 16) * (n // 16) * 16 * K * 2) + _a64(2 * k) + _a64(2 * n) + (_a64(2 * n) if has_bias else 0)
    return total


def _meta(stc, key: str):
    """(shape, bytes) of one checkpoint tensor from the headers, None when absent"""
    s = stc.find_stc(key) if hasattr(stc, "find_stc") else stc
    fn = s.tensor_file_map.get(key)
    if fn is None:
        return None
    h = s.file_headers[fn][key]
    b, e = h["data_offsets"]
    return h["shape"], e - b


def linear_arena_bytes(stc, key: str) -> int:
    """arena_expert_bytes for one projection of one expert, read from the headers"""
    t = _meta(stc, key + ".trellis")
    if t is None:
        raise ValueError(f"RAM budget: {key}.trellis is not in the checkpoint (only EXL3 experts can be held in RAM)")
    total = _a64(t[1])
    for suffix, optional in ((".suh", False), (".svh", False), (".bias", True)):
        m = _meta(stc, key + suffix)
        if m is None:
            if optional:
                continue
            raise ValueError(f"RAM budget: {key}{suffix} is not in the checkpoint")
        total += _a64(2 * prod(m[0]))
    return total


# ---------------------------------------------------------------------------------------- modules

def walk(modules):
    """Every module of a tree, parents first, in load order"""
    seen, stack = set(), list(reversed(list(modules)))
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        yield m
        stack.extend(reversed(list(getattr(m, "modules", None) or [])))


def is_moe(m) -> bool:
    return hasattr(m, "load_cpu_offload") and hasattr(m, "num_experts") and hasattr(m, "ups")


def is_ngram(m) -> bool:
    """A table the n-gram budget governs (NGramEmbedding)"""
    return hasattr(m, "table_nbytes")


def is_engram(m) -> bool:
    """A DeepSeek-V4.1 engram layer: its tables are always read from disk"""
    return type(m).__name__ == "DSV41Engram"


_WORKER_ACTIVATIONS = ("silu", "silu_ref", "gelu", "swiglu_oai")


def header_eligible(stc, m, first: int = 0) -> bool:
    """
    Whether -mcl / -mcs would hold the routed experts [first, E) of an MoE layer in RAM: the checks
    of BlockSparseMLP_CPU._cpu_eligible and of load_cpu_offload / load_cpu_split's probes, from the
    headers (a CUDA device assumed): every expert local, an activation the worker implements,
    mul1 experts with K <= 8, and per-expert biases on all experts or none
    """
    if getattr(m, "num_local_experts", None) not in (None, m.num_experts):
        return False
    if not (m.activation_fn in _WORKER_ACTIVATIONS if m.gated else m.activation_fn == "relu2"):
        return False
    for l in ([m.gates[first]] if m.gated else []) + [m.ups[first], m.downs[first]]:
        t = _meta(stc, l.key + ".trellis")
        if _meta(stc, l.key + ".mul1") is None or t is None or t[0][-1] // 16 > 8:
            return False
    for ls in ([m.gates] if m.gated else []) + [m.ups, m.downs]:
        has = [_meta(stc, l.key + ".bias") is not None for l in ls[first:]]
        if any(has) and not all(has):
            return False
    return True


@dataclass
class RamLayer:
    key: str
    first: int          # the experts [first, first + count) are held in RAM
    count: int
    nbytes: int


def _held_bytes(stc, m, first: int, count: int) -> int:
    total = 0
    for e in range(first, first + count):
        for l in ([m.gates[e]] if m.gated else []) + [m.ups[e], m.downs[e]]:
            total += linear_arena_bytes(stc, l.key)
    return total


def ram_layers(model, infer_params, component: str, placement) -> tuple[str | None, list[RamLayer]]:
    """
    (label, layers) of the routed experts this load will keep in the CPU worker's arena: those the
    placement puts in RAM (experts=stream / cpu / split, or hot= on stream / cpu), else those of
    -mcl / -mcs for the main model's text component and of -dmcl for any other component, taken
    as the loader takes them (the first eligible layers in load order)
    """
    stc = model.config.stc
    moes = [m for m in walk(model.modules) if is_moe(m)]
    out = []
    if placement is not None and component == "text":
        for m in moes:
            plan = m.placement_plan()
            if plan is None or plan[0] not in ("ram", "split"):
                continue
            first = 0 if plan[0] == "ram" else m.num_experts - plan[2]
            if not 0 <= first < m.num_experts or not header_eligible(stc, m, first):
                continue        # the layer refuses to load, with the placement's own message
            out.append(RamLayer(m.key, first, m.num_experts - first, _held_bytes(stc, m, first, m.num_experts - first)))
        return "placement", out
    if component == "text":
        n, k = int(getattr(infer_params, "moe_cpu_offload", 0) or 0), int(getattr(infer_params, "moe_cpu_split", 0) or 0)
    else:
        n, k = int(getattr(infer_params, "draft_moe_cpu_offload", 0) or 0), 0
    if n > 0:
        for m in moes:
            if len(out) >= n:
                break
            if header_eligible(stc, m):
                out.append(RamLayer(m.key, 0, m.num_experts, _held_bytes(stc, m, 0, m.num_experts)))
        return (f"-mcl {n}" if component == "text" else f"-dmcl {n}"), out
    if k > 0:
        limit = int(os.environ.get("EXL3_MOE_CPU_SPLIT_LAYERS", 0) or 0)
        for m in moes:
            if limit and len(out) >= limit:
                break
            kk = min(k, m.num_experts - 1)
            if kk <= 0 or getattr(m, "routing_first", None) is not None:
                continue
            first = m.num_experts - kk
            if header_eligible(stc, m, first):
                out.append(RamLayer(m.key, first, kk, _held_bytes(stc, m, first, kk)))
        return f"-mcs {k}", out
    return None, out


# ---------------------------------------------------------------------------------------- n-gram

@dataclass
class NgramTable:
    key: str
    nbytes: int
    every_row: bool = False     # read by every gathered row (fills first)


def ngram_fill(tables: list[NgramTable], budget: int) -> list[NgramTable]:
    """The tables a budget of `budget` bytes holds whole: tables read by every row first, then the
    others, smallest first, each while it fits. A table that does not fit streams its rows from
    disk"""
    held, left = [], budget
    for t in sorted(tables, key = lambda t: (not t.every_row, t.nbytes, t.key)):
        if t.nbytes <= left:
            held.append(t)
            left -= t.nbytes
    return held


def pagecache_auto(mem_total: int | None) -> int:
    """ram pagecache=auto: max(8 GiB, MemTotal / 8)"""
    return max(8 * GiB, (mem_total or 0) // 8)


# ---------------------------------------------------------------------------------------- plan

@dataclass
class ComponentBudget:
    component: str
    label: str | None                   # what keeps experts in RAM: "placement", "-mcl 11", ...
    layers: list = field(default_factory = list)            # RamLayer
    expert_bytes: int = 0               # the arena of the RAM-held experts
    expert_cap: Size | None = None
    expert_request: Size | None = None
    ngram_budget: Size = ZERO           # resolved: the request, else the cap, else 0
    ngram_tables: list = field(default_factory = list)      # NgramTable
    ngram_held: list = field(default_factory = list)        # NgramTable held whole in RAM
    ngram_bytes: int = 0
    engram_keys: list = field(default_factory = list)       # DeepSeek-V4.1 engram layers (disk)
    memory: HostMemory | None = None
    reserve: int = 0

    def worth_reporting(self) -> bool:
        """Anything held in RAM, or a budget set: the load prints summary()"""
        return bool(self.layers or self.ngram_held or self.expert_cap is not None or
                    (self.ngram_tables or self.engram_keys) and self.ngram_budget != ZERO)

    def summary(self) -> str:
        parts = []
        if self.layers or self.expert_cap is not None:
            parts.append(f"routed experts {human(self.expert_bytes)}" +
                         (f" ({self.label}: {len(self.layers)} layers)" if self.layers else "") +
                         (f", cap {self.expert_cap}" if self.expert_cap is not None else ""))
        if self.ngram_tables:
            if self.ngram_held:
                parts.append(f"n-gram tables {human(self.ngram_bytes)} in RAM "
                             f"({len(self.ngram_held)} of {len(self.ngram_tables)} whole)")
            else:
                parts.append("n-gram tables 0 (every row streams from disk)")
        if self.engram_keys and self.ngram_budget != ZERO:
            parts.append(f"engram tables stream from disk (ngram={self.ngram_budget} holds no DeepSeek-V4.1 "
                         f"engram table)")
        mem = self.memory
        tail = f"; {human(mem.available)} available" if mem is not None and mem.available is not None else ""
        if tail and mem.cgroup_limits:
            tail += f" (memory cgroup {mem.cgroup_path})"
        return f" -- RAM budget ({self.component}): " + ", ".join(parts) + tail


def plan_component(model, placement, memory: HostMemory | None = None) -> ComponentBudget:
    """
    Stage 1 of a load (Model.load_gen, before any module loads): what this component will hold in
    host memory, its caps and requests, and one check of all of it against the host memory. Raises
    ValueError for a request or an arena above its cap, RuntimeError when the host cannot hold the
    total. Leaves infer_params.ngram_ram_plan ({table key: held in RAM}) for the n-gram modules
    """
    ip = model.config.infer_params
    component = getattr(model, "component", "text")
    ram = getattr(placement, "ram", None) if placement is not None and component == "text" else None
    cap, flag = component_cap(ip, component)
    b = ComponentBudget(component, None, expert_cap = cap, expert_request = ram.experts if ram is not None else None)
    b.label, b.layers = ram_layers(model, ip, component, placement)
    b.expert_bytes = sum(l.nbytes for l in b.layers)
    what = "placement" if placement is not None and component == "text" else (b.label or "--ngram_ram")

    # Routed experts: a request above the cap, or arenas above the request or the cap
    req = b.expert_request
    static = b.expert_bytes
    if req is not None and req.is_bytes:
        if cap is not None and req.nbytes > cap.nbytes:
            raise ValueError(f"placement asks ram experts={req} ({human(req.nbytes)}) but {flag} caps this "
                             f"component at {cap}")
        if static > req.nbytes:
            raise ValueError(f"placement: the stream / cpu layers keep {human(static)} of routed experts in RAM, "
                             f"more than ram experts={req}; raise it, or keep fewer layers' experts in RAM")
    elif req is not None and req.is_all and cap is not None and static > cap.nbytes:
        raise ValueError(f"placement asks ram experts=all ({human(static)}) but {flag} caps this component at {cap}")
    elif cap is not None and static > cap.nbytes:
        who = "placement: the stream / cpu layers" if what == "placement" else what
        raise ValueError(f"{who} keep{'s' if who == what else ''} {human(static)} of routed experts in RAM, "
                         f"more than {flag} {cap}; raise the cap or offload fewer layers")

    # N-gram tables: the request, capped by --ngram_ram; else --ngram_ram; else 0
    for m in walk(model.modules):
        if is_ngram(m):
            b.ngram_tables.append(NgramTable(m.key, int(m.table_nbytes())))
        elif is_engram(m):
            b.engram_keys.append(m.key)
    ng_cap = as_ngram_ram(getattr(ip, "ngram_ram", None)) if component == "text" else None
    ng_req = ram.ngram if ram is not None else None
    tables = sum(t.nbytes for t in b.ngram_tables)
    if ng_req is not None and ng_cap is not None and not ng_cap.is_all:
        asks = tables if ng_req.is_all else (ng_req.nbytes if ng_req.is_bytes else 0)
        if asks > ng_cap.nbytes:
            raise ValueError(f"placement asks ram ngram={ng_req} but --ngram_ram caps this component at {ng_cap}")
    b.ngram_budget = ng_req if ng_req is not None else (ng_cap if ng_cap is not None else ZERO)
    if b.engram_keys and getattr(placement, "disk", None) is not None and placement.disk.ngram == "off":
        raise ValueError(f"placement: 'disk ngram=off' would never read n-gram rows from disk, but the "
                         f"DeepSeek-V4.1 engram tables of {b.engram_keys[0]} are always read from disk; "
                         f"remove disk ngram=off")

    # One check of everything against the host
    b.memory = memory if memory is not None else host_memory()
    b.reserve = reserve_bytes()
    if b.ngram_budget.is_all:
        b.ngram_held = list(b.ngram_tables)
    elif b.ngram_budget.is_auto:
        pc = ram.pagecache if ram is not None else AUTO
        pagecache = pagecache_auto(b.memory.mem_total) if pc.is_auto else pc.nbytes
        room = max(0, (b.memory.available or 0) - b.reserve - pagecache - static)
        b.ngram_held = ngram_fill(b.ngram_tables, room)
    else:
        b.ngram_held = ngram_fill(b.ngram_tables, b.ngram_budget.nbytes)
    b.ngram_bytes = sum(t.nbytes for t in b.ngram_held)
    ledger = HostLedger(memory = b.memory, reserve = b.reserve)
    ledger.add("ram experts", static)
    ledger.add("ngram", b.ngram_bytes)
    ledger.check(what, "lower the budgets or use auto" if what == "placement" else
                 "offload fewer layers, lower --ngram_ram, or free host memory")
    # Kept per table key, so that a component loading later (an MTP head sharing the config)
    # leaves the main model's entries alone
    held = {t.key for t in b.ngram_held}
    plan = dict(getattr(ip, "ngram_ram_plan", None) or {})
    plan.update({t.key: t.key in held for t in b.ngram_tables})
    ip.ngram_ram_plan = plan
    return b


def ngram_in_ram(config, key: str, nbytes: int) -> bool:
    """Whether the n-gram table `key` (nbytes on disk) is held whole in RAM: the load's plan
    (plan_component), else the budget alone (0 / unset: no, all: yes, a size: when the table fits)"""
    ip = getattr(config, "infer_params", None)
    if ip is None:
        return False
    plan = getattr(ip, "ngram_ram_plan", None) or {}
    if key in plan:
        return plan[key]
    budget = as_ngram_ram(getattr(ip, "ngram_ram", None))
    if budget is None or budget.is_auto:
        return False
    return budget.is_all or nbytes <= budget.nbytes


def charge_expert_ram(host, key: str, proj_dims: dict, has_bias: bool, num_experts: int):
    """
    The CPU worker's guard, before a layer registers its RAM-held experts: the component's arena
    (every layer registered so far plus this one) must stay within its cap. Plan_component checks
    the same from the headers before anything loads; this catches whatever reaches the worker
    another way. A re-registration (autosplit rollback) is not charged again
    """
    if key in host.by_key:
        return
    cap, flag = component_cap(host.config.infer_params, getattr(host, "component", "text"))
    used = getattr(host, "expert_ram_used", 0) + arena_expert_bytes(proj_dims, has_bias) * num_experts
    if cap is not None and used > cap.nbytes:
        raise RuntimeError(f"CPU MoE: {key} would bring the routed experts held in RAM by this component to "
                           f"{human(used)}, more than {flag} {cap}; raise the cap or offload fewer layers")
    host.expert_ram_used = used
