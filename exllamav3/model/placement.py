"""
Explicit placement: the device every decoder layer loads on and, for MoE layers, where their
routed experts are stored and computed. For any model loaded in layer-split mode; see
doc/placement.md for the full description.

    EXL3_PLACEMENT="0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
    -placement / --placement "..."              (model_init; overrides the variable)
    config.infer_params.placement = "..."       (Python, before loading)

One rule per layer range, separated by ';' or newlines; '#' starts a comment:

    <layers>=cuda:<n> [experts=<mode>] [cpu=<k>]

  layers   decoder-layer indices from 0: '7', '0-11' or '0-3,8-11' (ASCII digits; whitespace
           around a comma is allowed, none inside a range). '*' takes every layer no other rule
           lists. 'embed' places the modules before the first layer, 'head' those after the last
           (final norm, output head); unlisted, they follow the first / last layer. Modules the
           loader always keeps in system RAM (the token embedding) stay there
  device   cuda:<n>, a logical index after CUDA_VISIBLE_DEVICES
  experts  where an MoE layer's routed experts live (default vram):
             vram    in the layer's GPU memory, computed there (no offload)
             stream  in system RAM, streamed to the layer's GPU and computed there; no CPU
                     expert arithmetic (as -mcl with -mcm stream_only)
             cpu     in system RAM, computed by the CPU worker, the hottest experts of a large
                     prefill chunk streamed to the GPU (as -mcl)
             split   the last cpu=<k> routed experts in system RAM on the CPU worker, the others
                     in the layer's GPU memory (as -mcs k, including its dynamic swapping)
  cpu      split only, required: routed experts per layer held in system RAM, 1 .. E - 1

No placement is one value, None: the variable unset or empty (or holding only whitespace, ';' and
'#' comments), --placement "" (which clears a set variable), or the attribute None or "". Words
that might read as "no placement" (none, off, auto, default, 0, false) are refused, so a typo can
never pass as a layout or as its absence.

A placement replaces -mcl, -mcs, -mcm and EXL3_MOE_CPU_SPLIT_LAYERS and is refused together with
any of them at load. GPU budgets (-gs / use_per_device / reserve_per_device) still cap memory but
never move a layer: a layer that does not fit its device fails the load. DeepSeek-V4.1 reads the
placement when the model object is built and adds rules of its own (doc/placement.md,
"DeepSeek-V4.1").

Torch-free, so it can be parsed and tested anywhere.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re

EXPERT_MODES = ("vram", "stream", "cpu", "split")
# Per mode, spelled out so that a mode added to EXPERT_MODES without its entries fails loudly
# (KeyError) instead of taking another mode's meaning: the kind of host-held experts, for the
# one-kind-per-GPU rule (None: held in VRAM), and expert_plan's (storage, execution mode)
_HOST_KIND = {"vram": None, "stream": "stream", "cpu": "cpu/split", "split": "cpu/split"}
_PLAN = {"vram": ("vram", None), "stream": ("ram", "stream"), "cpu": ("ram", "hybrid"),
         "split": ("split", "hybrid")}
_ATTRS = ("experts", "cpu")
# Whole values that might be meant as "no placement": refused (no placement is an empty value)
_NO_PLACEMENT_WORDS = ("none", "off", "auto", "default", "0", "false")
# ASCII digits only: \d would also take other scripts' digits
_DEVICE = re.compile(r"^cuda:([0-9]+)$")
_RANGE = re.compile(r"^([0-9]+)(?:-([0-9]+))?$")


@dataclass(frozen = True)
class Experts:
    mode: str = "vram"
    cpu: int | None = None

    def text(self) -> str:
        if self.mode == "vram":
            return ""
        return f" experts={self.mode}" + (f" cpu={self.cpu}" if self.mode == "split" else "")


@dataclass(frozen = True)
class Rule:
    target: str                    # "layers", "rest" ('*'), "embed" or "head"
    layers: tuple[int, ...]        # explicit indices for "layers", else ()
    device: str                    # "cuda:<n>"
    experts: Experts = field(default_factory = Experts)

    @property
    def cuda_index(self) -> int:
        return int(self.device[5:])


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

    def __str__(self):
        parts = []
        for r in self.rules:
            target = format_ranges(r.layers) if r.target == "layers" else ("*" if r.target == "rest" else r.target)
            parts.append(f"{target}={r.device}{r.experts.text()}")
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


def parse(value) -> Placement | None:
    """
    Parse a placement. None, an empty string or one holding only separators and comments means
    no placement (None, its one representation); a Placement is returned as is. A whole value
    that reads as a word for "no placement" (_NO_PLACEMENT_WORDS) is refused. Raises ValueError
    on any error.
    """
    if value is None or isinstance(value, Placement):
        return value
    if not isinstance(value, str):
        raise ValueError(f"placement must be a string, got {type(value).__name__}")
    chunks = [c.strip() for line in value.splitlines() for c in line.split("#", 1)[0].split(";")]
    chunks = [c for c in chunks if c]
    if not chunks:
        return None
    if len(chunks) == 1 and chunks[0].lower() in _NO_PLACEMENT_WORDS:
        raise ValueError(
            f"placement {value!r}: no placement is written as an empty value, not {chunks[0]!r}: "
            f"leave EXL3_PLACEMENT unset or empty, pass --placement \"\", or set "
            f"config.infer_params.placement = None, and the autosplit decides as usual")
    rules, seen, targets = [], {}, set()
    for n, chunk in enumerate(chunks, 1):
        where = f"placement rule {n} {chunk!r}"
        if "=" not in chunk:
            raise ValueError(f"{where}: expected <layers>=cuda:<n> [experts=...]")
        head, rest = chunk.split("=", 1)
        head = head.strip().lower()
        toks = rest.split()
        if not toks:
            raise ValueError(f"{where}: missing device after '='")
        m = _DEVICE.match(toks[0].lower())
        if not m:
            raise ValueError(f"{where}: device {toks[0]!r} must be cuda:<n>")
        device = f"cuda:{int(m.group(1))}"
        attrs = {}
        for t in toks[1:]:
            if "=" not in t:
                raise ValueError(f"{where}: {t!r} must be key=value ({', '.join(_ATTRS)})")
            k, v = (s.strip().lower() for s in t.split("=", 1))
            if k not in _ATTRS:
                raise ValueError(f"{where}: unknown attribute {k!r} (expected {', '.join(_ATTRS)})")
            if k in attrs:
                raise ValueError(f"{where}: {k}= given twice")
            attrs[k] = v
        mode = attrs.get("experts", "vram")
        if mode not in EXPERT_MODES:
            raise ValueError(f"{where}: experts={mode!r} must be one of {', '.join(EXPERT_MODES)}")
        cpu = None
        if "cpu" in attrs:
            if mode != "split":
                raise ValueError(f"{where}: cpu= only applies to experts=split")
            if not (attrs["cpu"].isascii() and attrs["cpu"].isdigit()) or int(attrs["cpu"]) < 1:
                raise ValueError(f"{where}: cpu= must be a positive integer")
            cpu = int(attrs["cpu"])
        elif mode == "split":
            raise ValueError(f"{where}: experts=split needs cpu=<routed experts per layer in system RAM>")
        if head in ("embed", "head"):
            if attrs:
                raise ValueError(f"{where}: experts= applies to decoder layers, not to {head}")
            target, layers = head, ()
        elif head == "*":
            target, layers = "rest", ()
        else:
            target, layers = "layers", _parse_layers(head, where)
        if target != "layers":
            if target in targets:
                raise ValueError(f"{where}: {'*' if target == 'rest' else target} is placed twice")
            targets.add(target)
        for i in layers:
            if i in seen:
                raise ValueError(f"{where}: layer {i} is already placed by rule {seen[i]}")
            seen[i] = n
        rules.append(Rule(target, layers, device, Experts(mode, cpu)))

    # One kind of host-held experts per GPU: the CPU worker keeps one streamed-prefill state per
    # device (bandwidth probe, streaming threshold), which GPU-only and CPU-computed layers use
    # differently
    kinds = {}
    for r in rules:
        kind = _HOST_KIND[r.experts.mode]
        if kind is None:
            continue
        if kinds.setdefault(r.device, kind) != kind:
            raise ValueError(f"placement: {r.device} holds both experts=stream and experts=cpu / split "
                             f"layers; one kind of host-held experts per GPU is supported")
    # Canonical rule order, so equal placements compare equal however they were written
    return Placement(tuple(sorted(rules, key = _rule_order)))


def expert_plan(placement: Placement, layer_idx: int, num_experts: int):
    """
    (storage, mode, split_k) of one MoE layer:
      storage  "vram", "ram" (every routed expert in system RAM) or "split"
      mode     execution mode of the RAM-held experts (moe_expert_policy.py): "stream" or
               "hybrid"; None for vram
      split_k  routed experts in system RAM for a split, else 0
    """
    e = placement.experts_for_layer(layer_idx)
    storage, mode = _PLAN[e.mode]
    if storage != "split":
        return storage, mode, 0
    if not 0 < e.cpu < num_experts:
        raise ValueError(f"placement: layer {layer_idx} experts=split cpu={e.cpu} must leave between 1 and "
                         f"{num_experts - 1} of its {num_experts} routed experts on the GPU")
    return storage, mode, e.cpu


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
