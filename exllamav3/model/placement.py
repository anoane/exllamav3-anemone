"""
Explicit placement: the device every decoder layer loads on and, for MoE layers, where their
routed experts are stored and computed; with storage rules, how much of each memory level they
may take. For any model loaded in layer-split mode; see doc/placement.md for the full description
and doc/expert_tiers.md for the storage rules and budgets.

    EXL3_PLACEMENT="0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
    -placement / --placement "..."              (model_init; overrides the variable)
    config.infer_params.placement = "..."       (Python, before loading: a string, a dict or a Placement)

One rule per layer range, separated by ';' or newlines; '#' starts a comment; double quotes
protect a value with spaces, ';', '#' or '"':

    <layers>=cuda:<n> [experts=<mode>] [hot=<k>|<p>%] [cpu=<k>] [prefetch=...] [profile=...]
    cuda:<n> ... / ram ... / disk ...           storage rules (placement_storage.py)

  layers   decoder-layer indices from 0: '7', '0-11' or '0-3,8-11' (ASCII digits; whitespace
           around a comma is allowed, none inside a range). '*' takes every layer no other rule
           lists. 'embed' places the modules before the first layer, 'head' those after the last
           (final norm, output head); unlisted, they follow the first / last layer. Modules the
           loader always keeps in system RAM (the token embedding) stay there
  device   cuda:<n>, a logical index after CUDA_VISIBLE_DEVICES
  experts  where an MoE layer's routed experts live (default vram):
             vram    in the layer's GPU memory, computed there (no offload)
             cache   the GPU's expert cache over the RAM tier and the disk, computed on the
                     layer's GPU (expert tiers, doc/expert_tiers.md)
             stream  in system RAM, streamed to the layer's GPU and computed there; no CPU
                     expert arithmetic (as -mcl with -mcm stream_only)
             cpu     in system RAM, computed by the CPU worker, the hottest experts of a large
                     prefill chunk streamed to the GPU (as -mcl)
             split   the last cpu=<k> routed experts in system RAM on the CPU worker, the others
                     in the layer's GPU memory (as -mcs k, including its dynamic swapping)
             hybrid  older spelling of cpu hot=<k> (hot= required), printed as cpu hot=<k>
  hot      cache / stream / cpu: routed experts per layer kept resident in VRAM, a count or a
           share of the layer's experts ('10%'). stream / cpu with hot=<k> run as a split with
           E - k experts in RAM (as -mcs E-k, with -mcm stream_only for stream)
  cpu      split only, required: routed experts per layer held in system RAM, 1 .. E - 1
  prefetch cache / stream: read-ahead of experts ('off', 'auto', 'layer[:<d>]', 'router[:<d>]')
  profile  not vram: an expert profile that seeds the hot and cached experts

No placement is one value, None: the variable unset or empty (or holding only whitespace, ';' and
'#' comments), --placement "" (which clears a set variable), or the attribute None or "". Words
that might read as "no placement" (none, off, auto, default, 0, false) are refused, so a typo can
never pass as a layout or as its absence.

A value can also be a dict / JSON object (Placement.to_dict()) or '@<path>' to a file holding
either form (a file with nothing but whitespace, separators and comments, or a dict whose 'rules'
list is empty and that has no storage rules, is no placement too; the words above are refused
there as well). A placement replaces -mcl, -mcs, -mcm and EXL3_MOE_CPU_SPLIT_LAYERS and is
refused together with any of them at load. GPU budgets (-gs / use_per_device / reserve_per_device)
still cap memory but never move a layer: a layer that does not fit its device fails the load.
DeepSeek-V4.1 reads the placement when the model object is built and adds rules of its own
(doc/placement.md, "DeepSeek-V4.1"). Words this build cannot run yet are refused after every other
check (placement_storage.PENDING).

Torch-free, so it can be parsed and tested anywhere.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import re

try:
    from . import placement_storage as storage
except ImportError:
    # Loaded by file path (the torch-free tests): the sibling by path too
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path
    _name = "_exl3_torch_free_placement_storage"
    storage = _sys.modules.get(_name)
    if storage is None:
        _spec = _ilu.spec_from_file_location(_name, _Path(__file__).resolve().parent / "placement_storage.py")
        storage = _ilu.module_from_spec(_spec)
        _sys.modules[_name] = storage
        _spec.loader.exec_module(storage)

DeviceStore, RamStore, DiskStore = storage.DeviceStore, storage.RamStore, storage.DiskStore

EXPERT_MODES = ("vram", "cache", "stream", "cpu", "split", "hybrid")
# Per mode, spelled out so that a mode added to EXPERT_MODES without its entries fails loudly
# (KeyError) instead of taking another mode's meaning: the kind of host-held experts, for the
# one-kind-per-GPU rule (None: held in VRAM; gpu: computed on the GPU, cpu: by the CPU worker),
# and expert_plan's (storage, execution mode). hybrid is normalized to cpu at parse time
_HOST_KIND = {"vram": None, "cache": "gpu", "stream": "gpu", "cpu": "cpu", "split": "cpu", "hybrid": "cpu"}
_PLAN = {"vram": ("vram", None), "cache": ("cache", "gpu"), "stream": ("ram", "stream"),
         "cpu": ("ram", "hybrid"), "split": ("split", "hybrid"), "hybrid": ("ram", "hybrid")}
_ATTRS = storage.LAYER_ATTRS
# Whole values that might be meant as "no placement": refused (no placement is an empty value)
_NO_PLACEMENT_WORDS = ("none", "off", "auto", "default", "0", "false")
# ASCII digits only: \d would also take other scripts' digits
_DEVICE = storage.DEVICE
_RANGE = re.compile(r"^([0-9]+)(?:-([0-9]+))?$")
_SHARE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)%$")


@dataclass(frozen = True)
class Hot:
    """hot=: routed experts per layer kept resident in VRAM, a count or a share"""
    count: int | None = None
    percent: float | None = None

    def __str__(self):
        return f"{self.count}" if self.count is not None else f"{self.percent:g}%"

    def resolve(self, num_experts: int) -> int:
        """The count for a layer of num_experts routed experts (a share rounds half up)"""
        if self.count is not None:
            return self.count
        return math.floor(num_experts * self.percent / 100 + 0.5)


@dataclass(frozen = True)
class Experts:
    mode: str = "vram"
    cpu: int | None = None          # split only
    hot: Hot | None = None          # cache, stream, cpu
    prefetch: tuple | None = None   # None: auto; (): off; else ((method, depth), ...)
    profile: str | None = None

    def text(self) -> str:
        if self.mode == "vram":
            return ""
        s = f" experts={self.mode}"
        if self.hot is not None:
            s += f" hot={self.hot}"
        if self.mode == "split":
            s += f" cpu={self.cpu}"
        if self.prefetch is not None:
            s += " prefetch=" + ("off" if not self.prefetch else
                                 "+".join(m if d == 1 else f"{m}:{d}" for m, d in self.prefetch))
        if self.profile is not None:
            s += f" profile={storage.quote(self.profile)}"
        return s


@dataclass(frozen = True)
class Rule:
    target: str                    # "layers", "rest" ('*'), "embed" or "head"
    layers: tuple[int, ...]        # explicit indices for "layers", else ()
    device: str                    # "cuda:<n>"
    experts: Experts = field(default_factory = Experts)

    @property
    def cuda_index(self) -> int:
        return int(self.device[5:])

    def target_text(self) -> str:
        return format_ranges(self.layers) if self.target == "layers" else ("*" if self.target == "rest" else self.target)


def format_ranges(indices) -> str:
    idx = sorted(indices)
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ",".join(out)


def _rule_order(r: Rule):
    return {"embed": 0, "layers": 1, "rest": 2, "head": 3}[r.target], r.layers[:1]


@dataclass(frozen = True)
class Placement:
    rules: tuple[Rule, ...]
    # Storage rules (placement_storage.py): cuda:<n> rules in index order, the ram and disk rules;
    # a rule that only restates defaults is not kept, so equal placements compare equal
    devices: tuple = ()
    ram: RamStore | None = None
    disk: DiskStore | None = None

    def _rule(self, target):
        return next((r for r in self.rules if r.target == target), None)

    def rule_for_layer(self, layer_idx: int) -> Rule:
        for r in self.rules:
            if r.target == "layers" and layer_idx in r.layers:
                return r
        rest = self._rule("rest")
        if rest is None:
            raise ValueError(f"placement: layer {layer_idx} is not placed (add it to a rule or use '*')")
        return rest

    def device_for_layer(self, layer_idx: int) -> str:
        return self.rule_for_layer(layer_idx).device

    def experts_for_layer(self, layer_idx: int) -> Experts:
        return self.rule_for_layer(layer_idx).experts

    def embed_device(self) -> str | None:
        r = self._rule("embed")
        return r.device if r else None

    def head_device(self) -> str | None:
        r = self._rule("head")
        return r.device if r else None

    def cuda_indices(self) -> list[int]:
        return sorted({r.cuda_index for r in self.rules})

    def cache_devices(self) -> list[str]:
        """The devices that hold experts=cache layers, in index order"""
        return sorted({r.device for r in self.rules if r.experts.mode == "cache"}, key = lambda d: int(d[5:]))

    def device_store(self, device: str) -> DeviceStore:
        """The cuda:<n> rule of a device, its defaults when there is none"""
        return next((d for d in self.devices if d.device == device), DeviceStore(device))

    def ram_store(self) -> RamStore:
        return self.ram if self.ram is not None else RamStore()

    def disk_store(self) -> DiskStore:
        return self.disk if self.disk is not None else DiskStore()

    def layer_map(self, layer_indices) -> dict[int, Rule]:
        """The rule of every model layer. Refuses rules naming layers the model does not have,
        and layers no rule places"""
        known = set(layer_indices)
        if not known:
            raise ValueError("placement: this model has no decoder layers to place")
        listed = {i for r in self.rules if r.target == "layers" for i in r.layers}
        extra = sorted(listed - known)
        if extra:
            raise ValueError(f"placement names layers the model does not have: {format_ranges(extra)} "
                             f"(its layers are {format_ranges(known)})")
        return {i: self.rule_for_layer(i) for i in sorted(known)}

    def to_dict(self) -> dict:
        """The same placement as plain data (JSON-ready); parse() takes it back"""
        return storage.to_dict(self)

    def __str__(self):
        parts = [f"{r.target_text()}={r.device}{r.experts.text()}" for r in self.rules]
        parts += [d.text() for d in self.devices]
        parts += [x.text() for x in (self.ram, self.disk) if x is not None]
        return "; ".join(parts)


def _parse_layers(tok: str, where: str) -> tuple[int, ...]:
    out = []
    for part in tok.split(","):
        m = _RANGE.match(part.strip())
        if not m:
            raise ValueError(f"{where}: layers {tok!r} must look like 7, 0-11 or 0-3,8-11")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        if b < a:
            raise ValueError(f"{where}: range {part.strip()!r} runs backwards")
        out.extend(range(a, b + 1))
    if len(set(out)) != len(out):
        raise ValueError(f"{where}: a layer is listed twice within the rule")
    return tuple(sorted(out))


def _parse_hot(v: str, where: str) -> Hot | None:
    """hot=<k> or hot=<p>%; hot=0 is the default (None)"""
    if v.endswith("%"):
        m = _SHARE.match(v)
        p = float(m.group(1)) if m else -1.0
        if not 0 < p < 100:
            raise ValueError(f"{where}: hot={v}: a share must lie between 0% and 100%, exclusive")
        return Hot(percent = p)
    k = storage.uint(v, "hot", where)
    return Hot(count = k) if k else None


def _parse_prefetch(v: str, where: str):
    if v == "auto":
        return None
    if v == "off":
        return ()
    items = {}
    for part in v.split("+"):
        m = re.fullmatch(r"([a-z]+)(?::([0-9]+))?", part)
        if not m or m.group(1) not in storage.PREFETCH_METHODS:
            raise ValueError(f"{where}: prefetch={v} must be off, auto, or methods joined by '+' "
                             f"(layer[:<depth>], router[:<depth>])")
        depth = int(m.group(2) or 1)
        if depth < 1:
            raise ValueError(f"{where}: prefetch depth must be at least 1")
        if m.group(1) in items:
            raise ValueError(f"{where}: prefetch lists {m.group(1)} twice")
        items[m.group(1)] = depth
    return tuple((k, items[k]) for k in storage.PREFETCH_METHODS if k in items)


def _layer_rule(chunk: str, where: str) -> Rule:
    head, rest = chunk.split("=", 1)
    head = head.strip().lower()
    toks = storage.tokens(rest)
    if not toks:
        raise ValueError(f"{where}: missing device after '='")
    m = _DEVICE.match(toks[0].lower())
    if not m:
        raise ValueError(f"{where}: device {toks[0]!r} must be cuda:<n>")
    device = f"cuda:{int(m.group(1))}"
    attrs = storage.attrs(toks[1:], _ATTRS, where, "a layer rule", keep_case = ("profile",))
    mode = attrs.get("experts", "vram").lower()
    if mode not in EXPERT_MODES:
        raise ValueError(f"{where}: experts={mode!r} must be one of vram, cache, stream, cpu (or the older "
                         f"split cpu=<k> / hybrid hot=<k>){storage.suggest(mode, EXPERT_MODES)}")
    hot = _parse_hot(attrs["hot"], where) if "hot" in attrs else None
    cpu = None
    if "cpu" in attrs:
        if mode != "split":
            raise ValueError(f"{where}: cpu= only applies to experts=split (write experts=cpu hot=<k> to keep "
                             f"k routed experts per layer in VRAM)")
        cpu = storage.uint(attrs["cpu"], "cpu", where, 1)
    elif mode == "split":
        raise ValueError(f"{where}: experts=split needs cpu=<routed experts per layer in system RAM>")
    if mode == "hybrid":
        if hot is None:
            raise ValueError(f"{where}: experts=hybrid needs hot=<routed experts per layer kept in VRAM> "
                             f"(the same as experts=cpu hot=<k>)")
        mode = "cpu"
    if "hot" in attrs and mode in ("vram", "split"):
        raise ValueError(f"{where}: hot= applies to experts=cache, stream and cpu" +
                         (" (experts=vram keeps every routed expert in VRAM)" if mode == "vram" else
                          " (experts=split counts the other side: cpu=<k>)"))
    prefetch = _parse_prefetch(attrs["prefetch"].lower(), where) if "prefetch" in attrs else None
    if "prefetch" in attrs and _HOST_KIND[mode] != "gpu":
        raise ValueError(f"{where}: prefetch= applies to experts=cache and stream (experts={mode} "
                         f"{'fetches nothing' if mode == 'vram' else 'computes on the CPU worker'})")
    profile = attrs.get("profile")
    if profile is not None and mode == "vram":
        raise ValueError(f"{where}: profile= chooses hot or cached experts; experts=vram has none to choose")
    if head in ("embed", "head"):
        if attrs:
            raise ValueError(f"{where}: experts= applies to decoder layers, not to {head}")
        target, layers = head, ()
    elif head == "*":
        target, layers = "rest", ()
    else:
        target, layers = "layers", _parse_layers(head, where)
    return Rule(target, layers, device, Experts(mode, cpu, hot, prefetch, profile))


def parse(value) -> Placement | None:
    """
    Parse a placement: the rule text, a dict (Placement.to_dict()), a JSON object in a string,
    '@<path>' to a file holding the text or the JSON, a Placement (returned as is) or None. None,
    an empty string or one holding only separators and comments means no placement (None, its one
    representation), as does a file holding only those or a dict / JSON object with an empty
    'rules' list and nothing else. A whole value that reads as a word for "no placement"
    (_NO_PLACEMENT_WORDS) is refused, also inside a file. Raises ValueError on any error.
    """
    if value is None or isinstance(value, Placement):
        return value
    if isinstance(value, dict):
        return _parse_text(storage.dict_to_text(value), value)
    if not isinstance(value, str):
        raise ValueError(f"placement must be a string, got {type(value).__name__}")
    text = value
    if text.lstrip().startswith("@"):
        path = text.strip()[1:].strip()
        try:
            with open(path, encoding = "utf-8") as f:
                text = f.read()
        except OSError as e:
            raise ValueError(f"placement: cannot read {path!r} ({e.strerror})") from None
    if text.lstrip().startswith("{"):
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"placement: invalid JSON ({e.msg} at character {e.pos})") from None
        return _parse_text(storage.dict_to_text(d), value)
    return _parse_text(text, value)


def _parse_text(text: str, value) -> Placement | None:
    """
    Parse the rule text of a placement. value is what the caller passed (the text, a dict,
    '@<path>' or a JSON string); the refusal of a word for "no placement" names it
    """
    chunks = storage.split_rules(text)
    if not chunks:
        return None
    if len(chunks) == 1 and chunks[0].lower() in _NO_PLACEMENT_WORDS:
        raise ValueError(
            f"placement {value!r}: no placement is written as an empty value, not {chunks[0]!r}: "
            f"leave EXL3_PLACEMENT unset or empty, pass --placement \"\", or set "
            f"config.infer_params.placement = None, and the autosplit decides as usual")
    rules, seen, targets = [], {}, set()
    devices, ram, disk, store_rule_no, tier_keys = {}, None, None, {}, []
    for n, chunk in enumerate(chunks, 1):
        where = f"placement rule {n} {chunk!r}"
        m = re.match(r"\s*([^\s=]+)", chunk)
        first = m.group(1).lower() if m else ""
        if storage.storage_head(first):
            kind, store, keys = storage.parse_rule(first, chunk, where)
            key = store.device if kind == "device" else kind
            if key in store_rule_no:
                raise ValueError(f"{where}: {key} is set twice (rules {store_rule_no[key]} and {n}); "
                                 f"merge them into one rule")
            store_rule_no[key] = n
            if kind == "device":
                devices[store.device] = store
            elif kind == "ram":
                ram, tier_keys = store, keys
            else:
                disk = store
            continue
        if re.fullmatch(r"[a-z_.:]+", first) and first not in ("embed", "head"):
            raise ValueError(f"{where}: unknown rule {first!r}{storage.suggest(first, storage.RULE_HEADS)} "
                             f"(a rule starts with {storage.HEADS_HELP})")
        if "=" not in chunk:
            raise ValueError(f"{where}: expected <layers>=cuda:<n> [experts=...]")
        r = _layer_rule(chunk, where)
        if r.target != "layers":
            if r.target in targets:
                raise ValueError(f"{where}: {'*' if r.target == 'rest' else r.target} is placed twice")
            targets.add(r.target)
        for i in r.layers:
            if i in seen:
                raise ValueError(f"{where}: layer {i} is already placed by rule {seen[i]}")
            seen[i] = n
        rules.append(r)

    # One kind of host-held experts per GPU: the CPU worker keeps one streamed-prefill state per
    # device (bandwidth probe, streaming threshold), which GPU-computed and CPU-computed layers
    # use differently
    kinds = {}
    for r in rules:
        kind = _HOST_KIND[r.experts.mode]
        if kind is None:
            continue
        if kinds.setdefault(r.device, kind) != kind:
            raise ValueError(f"placement: {r.device} holds both GPU-computed (experts=stream / cache) and "
                             f"CPU-computed (experts=cpu / split) layers; one kind of host-held experts per GPU "
                             f"is supported")
    kept, ram, disk = storage.check(rules, devices, ram, disk, tier_keys)
    # Canonical rule order, so equal placements compare equal however they were written
    placement = Placement(tuple(sorted(rules, key = _rule_order)), kept, ram, disk)
    storage.refuse_pending(placement)
    return placement


def expert_plan(placement: Placement, layer_idx: int, num_experts: int):
    """
    (storage, mode, k) of one MoE layer:
      storage  "vram", "ram" (every routed expert in system RAM), "split" (k routed experts in
               system RAM, the others resident in VRAM) or "cache" (the GPU's expert cache over
               the RAM tier and the disk, k experts pinned resident)
      mode     execution mode of the RAM-held experts (moe_expert_policy.py): "stream" or
               "hybrid"; "gpu" for cache (computed on the layer's GPU from VRAM); None for vram
      k        split: routed experts in system RAM (cpu=<k>, or E - hot for stream / cpu with
               hot=); cache: the hot count; else 0
    """
    e = placement.experts_for_layer(layer_idx)
    storage_kind, mode = _PLAN[e.mode]
    if e.mode == "vram":
        return storage_kind, mode, 0
    if e.mode == "split":
        if not 0 < e.cpu < num_experts:
            raise ValueError(f"placement: layer {layer_idx} experts=split cpu={e.cpu} must leave between 1 and "
                             f"{num_experts - 1} of its {num_experts} routed experts on the GPU")
        return storage_kind, mode, e.cpu
    hot = e.hot.resolve(num_experts) if e.hot is not None else 0
    if hot >= num_experts:
        raise ValueError(f"placement: layer {layer_idx} experts={e.mode} hot={e.hot} keeps all {num_experts} "
                         f"routed experts in VRAM: use experts=vram")
    if e.mode == "cache":
        return storage_kind, mode, hot
    if hot:
        return "split", mode, num_experts - hot
    return storage_kind, mode, 0


def conflicts(infer_params, environ) -> list[str]:
    """
    The older placement settings that are set alongside a placement, by name
    """
    out = []
    if getattr(infer_params, "moe_cpu_offload", 0):
        out.append("-mcl / --moe_cpu_offload / EXL3_MOE_CPU_OFFLOAD")
    if getattr(infer_params, "moe_cpu_split", 0):
        out.append("-mcs / --moe_cpu_split / EXL3_MOE_CPU_SPLIT")
    if getattr(infer_params, "moe_cpu_mode", "compute") != "compute":
        out.append("-mcm / --moe_cpu_mode / EXL3_MOE_CPU_MODE")
    if int(environ.get("EXL3_MOE_CPU_SPLIT_LAYERS", 0) or 0):
        out.append("EXL3_MOE_CPU_SPLIT_LAYERS")
    return out
