"""
The storage rules of an explicit placement (placement.py; doc/placement.md, doc/expert_tiers.md):
where a component's routed experts and n-gram tables may live besides the layers' GPUs, and how
much of each level they may take. Three rules, headed by the resource they govern:

    cuda:<n> [cache=auto|0|<size>] [spare=<k>] [evict=lru|lfu] [admit=adaptive|heat|always]
    ram      [experts=auto|all|<size>] [ngram=auto|all|<size>] [pagecache=auto|<size>]
             [policy=lazy-exclusive|exclusive|inclusive] [demote=swap|heat|all|off] [evict=lfu|lru]
    disk     [experts=model|off|<dir>] [ngram=model|off|<dir>] [io=auto|direct|buffered]

    0-11=cuda:0; 12-39=cuda:1 experts=cache; cuda:1 spare=12; ram experts=all; disk experts=off

Also here: the tokenizer shared with the layer rules (double quotes protect a value with spaces,
';', '#' or '"'), attribute parsing with did-you-mean hints, the checks that tie storage rules to
layer rules, and what each word needs to run (the checks at parse time; the expert tier's load
checks the rest). Sizes and their units come from util/host_budget.py. Torch-free.
"""

from __future__ import annotations
from dataclasses import dataclass
import difflib
import re

try:
    from ..util.host_budget import Size, AUTO, ALL, ZERO, parse_size, human, _UNIT, _AMBIGUOUS
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
    Size, AUTO, ALL, ZERO, parse_size, human = _hb.Size, _hb.AUTO, _hb.ALL, _hb.ZERO, _hb.parse_size, _hb.human
    _UNIT, _AMBIGUOUS = _hb._UNIT, _hb._AMBIGUOUS

LAYER_ATTRS = ("experts", "hot", "cpu", "prefetch", "profile")
DEVICE_ATTRS = ("cache", "spare", "evict", "admit")
RAM_ATTRS = ("experts", "ngram", "pagecache", "policy", "demote", "evict")
DISK_ATTRS = ("experts", "ngram", "io")
RULE_HEADS = ("ram", "disk", "embed", "head", "cuda:<n>")
HEADS_HELP = "layers (7, 0-11, 0-3,8-11), '*', 'embed', 'head', 'cuda:<n>', 'ram' or 'disk'"
# Value lists, the default first
VRAM_EVICT = ("lru", "lfu")
RAM_EVICT = ("lfu", "lru")
ADMIT = ("adaptive", "heat", "always")
POLICIES = ("exclusive", "lazy-exclusive", "inclusive")    # measured fastest on the test host (doc/expert_tiers.md)
DEMOTE = ("heat", "swap", "all", "off")
IO = ("auto", "direct", "buffered")
DEFAULT_SPARE = 8
PREFETCH_METHODS = ("layer", "router")
DEVICE = re.compile(r"^cuda:([0-9]+)$")      # ASCII digits only

# ---------------------------------------------------------------------------------------- tokens

def split_rules(text: str) -> list[str]:
    """Rules, split at ';' and newlines outside double quotes; '#' outside quotes starts a comment
    that runs to the end of the line"""
    out, cur, q, comment, i = [], [], False, False, 0
    while i < len(text):
        c = text[i]
        if comment:
            if c == "\n":
                comment = False
                out.append("".join(cur))
                cur = []
        elif q:
            cur.append(c)
            if c == "\\" and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif c == '"':
                q = False
        elif c == '"':
            q = True
            cur.append(c)
        elif c == "#":
            comment = True
        elif c in ";\n":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if q:
        raise ValueError(f"placement: unterminated quote in {text.strip()!r}")
    out.append("".join(cur))
    return [r.strip() for r in out if r.strip()]


def tokens(text: str) -> list[str]:
    """Whitespace-separated tokens; a double-quoted stretch stays inside its token"""
    out, cur, q, i = [], [], False, 0
    while i < len(text):
        c = text[i]
        if q:
            cur.append(c)
            if c == "\\" and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif c == '"':
                q = False
        elif c == '"':
            q = True
            cur.append(c)
        elif c.isspace():
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        out.append("".join(cur))
    return out


def unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return re.sub(r"\\(.)", r"\1", v[1:-1])
    return v


def quote(v: str) -> str:
    """A value as the canonical form writes it: quoted only when it needs to be"""
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' if re.search(r'[\s;#"]', v) or not v else v


def suggest(word: str, options) -> str:
    m = difflib.get_close_matches(word, list(options), n = 1, cutoff = 0.6)
    return f" (did you mean {m[0]!r}?)" if m else ""


_SCOPES = (("a layer rule", LAYER_ATTRS), ("a cuda:<n> rule", DEVICE_ATTRS),
           ("the ram rule", RAM_ATTRS), ("the disk rule", DISK_ATTRS))


def attrs(toks, allowed, where: str, scope: str, keep_case = ()) -> dict:
    """
    The key=value tokens of one rule, values unquoted and as written (the caller lower-cases
    keywords: choice(), sizes are case-insensitive), except that the words model / off of a
    keep_case key (paths, profile names, which keep their case) are lower-cased. Keys are
    case-insensitive. A key of another rule kind is named as such ("cache= belongs in a cuda:<n>
    rule"), a near miss gets a did-you-mean hint
    """
    out = {}
    for t in toks:
        if "=" not in t:
            hint = " (sizes are written without spaces: 48GiB)" \
                if t.lower() in _UNIT or t.lower() in _AMBIGUOUS else ""
            raise ValueError(f"{where}: {t!r} must be key=value ({', '.join(allowed)}){hint}")
        k, v = t.split("=", 1)
        k = k.strip().lower()
        if k not in allowed:
            elsewhere = [s for s, keys in _SCOPES if k in keys and s != scope]
            hint = f"; {k}= belongs in {' or '.join(elsewhere)}" if elsewhere else suggest(k, allowed)
            raise ValueError(f"{where}: unknown attribute {k!r} for {scope} (expected {', '.join(allowed)}){hint}")
        if k in out:
            raise ValueError(f"{where}: {k}= given twice")
        v = v.strip()
        if not v:
            raise ValueError(f"{where}: {k}= has no value")
        quoted = v.startswith('"')
        v = unquote(v)
        out[k] = v.lower() if not quoted and k in keep_case and v.lower() in ("model", "off") else v
    return out


def choice(v: str, choices, key: str, where: str) -> str:
    if v.lower() not in choices:
        raise ValueError(f"{where}: {key}={v} must be one of {', '.join(choices)}{suggest(v.lower(), choices)}")
    return v.lower()


def uint(v: str, key: str, where: str, minimum: int = 0) -> int:
    """A whole number of ASCII digits, at least `minimum`"""
    if not (v.isascii() and v.isdigit()) or int(v) < minimum:
        raise ValueError(f"{where}: {key}= must be {'a positive' if minimum else 'a non-negative'} integer")
    return int(v)


# ---------------------------------------------------------------------------------------- rules

@dataclass(frozen = True)
class DeviceStore:
    """cuda:<n>: the expert cache of one GPU (experts=cache layers on it)"""
    device: str
    cache: Size = AUTO              # auto, 0 (no cache: every miss streams through staging) or a size
    spare: int = DEFAULT_SPARE      # free slots kept ready for promotions
    evict: str = VRAM_EVICT[0]      # victim choice: lru (use stamp) or lfu (decayed heat)
    admit: str = ADMIT[0]           # decode misses admitted: adaptive, heat or always

    def text(self) -> str:
        s = self.device
        if self.cache != AUTO:
            s += f" cache={self.cache}"
        if self.spare != DEFAULT_SPARE:
            s += f" spare={self.spare}"
        if self.evict != VRAM_EVICT[0]:
            s += f" evict={self.evict}"
        if self.admit != ADMIT[0]:
            s += f" admit={self.admit}"
        return s


@dataclass(frozen = True)
class RamStore:
    """ram: the component's host-memory budgets and the RAM tier's policy"""
    experts: Size | None = None     # None: not requested here (the --expert_ram cap, else the default)
    ngram: Size | None = None       # None: not requested here (the --ngram_ram cap, else 0)
    pagecache: Size = AUTO          # what auto budgets leave unpinned; auto = max(8 GiB, MemTotal / 8)
    policy: str = POLICIES[0]
    demote: str = DEMOTE[0]
    evict: str = RAM_EVICT[0]

    def text(self) -> str:
        s = "ram"
        if self.experts is not None:
            s += f" experts={self.experts}"
        if self.ngram is not None:
            s += f" ngram={self.ngram}"
        if self.pagecache != AUTO:
            s += f" pagecache={self.pagecache}"
        if self.policy != POLICIES[0]:
            s += f" policy={self.policy}"
        if self.demote != DEMOTE[0] and self.policy != "inclusive":
            s += f" demote={self.demote}"
        if self.evict != RAM_EVICT[0]:
            s += f" evict={self.evict}"
        return s


@dataclass(frozen = True)
class DiskStore:
    """disk: where experts and n-gram rows are read from at runtime"""
    experts: str = "model"          # "model" (the checkpoint's shards in place), "off" or a directory
    ngram: str = "model"
    io: str = IO[0]

    def text(self) -> str:
        s = "disk"
        if self.experts != "model":
            s += f" experts={quote(self.experts)}"
        if self.ngram != "model":
            s += f" ngram={quote(self.ngram)}"
        if self.io != IO[0]:
            s += f" io={self.io}"
        return s


def storage_head(first: str) -> bool:
    """Whether a rule whose first token is `first` (lower case) is a storage rule"""
    return first in ("ram", "disk") or DEVICE.match(first) is not None


def parse_rule(first: str, chunk: str, where: str):
    """One storage rule: ("device" | "ram" | "disk", store, the RAM tier keys it sets)"""
    rest = chunk.strip()[len(first):]
    if rest.lstrip().startswith("="):
        example = {"ram": "ram experts=48GiB", "disk": "disk experts=model"}.get(first, f"{first} cache=75GiB")
        raise ValueError(f"{where}: {first} takes attributes, not '=<value>': write e.g. '{example}'")
    toks = tokens(rest)
    if first == "ram":
        a = attrs(toks, RAM_ATTRS, where, "the ram rule")
        if not a:
            raise ValueError(f"{where}: an empty ram rule; give experts=, ngram=, ...")
        kw = {}
        try:
            for k in ("experts", "ngram"):
                if k in a:
                    kw[k] = parse_size(a[k], k)
            if "pagecache" in a:
                kw["pagecache"] = parse_size(a["pagecache"], "pagecache", allow = ("auto",))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        kw["policy"] = choice(a.get("policy", POLICIES[0]), POLICIES, "policy", where)
        kw["demote"] = choice(a.get("demote", DEMOTE[0]), DEMOTE, "demote", where)
        kw["evict"] = choice(a.get("evict", RAM_EVICT[0]), RAM_EVICT, "evict", where)
        store = RamStore(**kw)
        if store.experts == AUTO and store.ngram == AUTO:
            raise ValueError(f"{where}: experts=auto and ngram=auto: at most one RAM budget can be auto")
        tier_keys = [k for k in ("policy", "demote", "evict") if k in a]
        if store.experts == ZERO and tier_keys:
            raise ValueError(f"{where}: {tier_keys[0]}= governs the RAM tier, which experts=0 disables")
        if store.policy == "inclusive" and "demote" in a:
            raise ValueError(f"{where}: demote= has no effect with policy=inclusive (every VRAM-cached expert "
                             f"keeps its RAM copy, so a victim is simply dropped); remove it")
        return "ram", store, tier_keys
    if first == "disk":
        a = attrs(toks, DISK_ATTRS, where, "the disk rule", keep_case = ("experts", "ngram"))
        if not a:
            raise ValueError(f"{where}: an empty disk rule; give experts=, ngram= or io=")
        io = choice(a.get("io", IO[0]), IO, "io", where)
        store = DiskStore(a.get("experts", "model"), a.get("ngram", "model"), io)
        if store.experts == "off" and store.ngram == "off" and "io" in a:
            raise ValueError(f"{where}: io= configures disk reads, which experts=off and ngram=off both disable")
        return "disk", store, []
    m = DEVICE.match(first)
    device = f"cuda:{int(m.group(1))}"
    a = attrs(toks, DEVICE_ATTRS, where, "a cuda:<n> rule")
    if not a:
        raise ValueError(f"{where}: an empty {device} rule; give cache=, spare=, evict= or admit=")
    if a.get("cache", "").lower() == "all":
        raise ValueError(f"{where}: cache=all would hold every expert of its layers: write experts=vram on them")
    try:
        cache = parse_size(a["cache"], "cache", allow = ("auto",)) if "cache" in a else AUTO
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    store = DeviceStore(device, cache,
                        uint(a["spare"], "spare", where) if "spare" in a else DEFAULT_SPARE,
                        choice(a.get("evict", VRAM_EVICT[0]), VRAM_EVICT, "evict", where),
                        choice(a.get("admit", ADMIT[0]), ADMIT, "admit", where))
    if cache == ZERO and any(k in a for k in ("spare", "evict", "admit")):
        raise ValueError(f"{where}: cache=0 has no slots for spare= / evict= / admit= to govern")
    return "device", store, []


def check(rules, devices: dict, ram: RamStore | None, disk: DiskStore | None, tier_keys = ()):
    """
    The checks that tie the storage rules to the layer rules (`rules`: placement.Rule, experts
    modes already normalized). Returns (devices, ram, disk) as the Placement keeps them: device
    rules in index order, and a rule that only restates defaults dropped, so that equal
    placements compare equal
    """
    if not rules:
        raise ValueError("placement: storage rules (cuda:<n>, ram, disk) need layer rules to go with; without a "
                         "placement, set RAM budgets with --expert_ram / --ngram_ram")
    cache_devices = {r.device for r in rules if r.experts.mode == "cache"}
    for d in devices.values():
        if d.device not in cache_devices:
            raise ValueError(f"placement: '{d.text()}' configures an expert cache, but no "
                             f"experts=cache layer is placed on {d.device}")
    ram_layers = [r for r in rules if r.experts.mode != "vram"]
    if ram is not None:
        if ram.experts is not None and not ram_layers:
            raise ValueError(f"placement: 'ram experts={ram.experts}' but no layer keeps routed experts in "
                             f"system RAM (every rule is experts=vram)")
        if tier_keys and not cache_devices:
            raise ValueError(f"placement: ram {tier_keys[0]}= governs the RAM tier of experts=cache layers, "
                             f"and there are none")
    r_eff = ram or RamStore()
    for d in devices.values():
        if d.spare == 0 and d.cache != ZERO and r_eff.policy != "inclusive" and r_eff.demote != "off":
            raise ValueError(f"placement: {d.device} spare=0 leaves no free slot for a demoted victim; spare=0 "
                             f"needs ram demote=off or policy=inclusive")
    if disk is not None:
        if disk.experts != "model" and not cache_devices:
            raise ValueError(f"placement: 'disk experts={disk.experts}' applies to experts=cache layers "
                             f"(stream and cpu layers keep every expert in RAM), and there are none")
        if disk.experts == "off":
            ahead = [r.experts.prefetch for r in rules if r.experts.prefetch]
            if ahead:
                words = "+".join(m if d == 1 else f"{m}:{d}" for m, d in ahead[0])
                raise ValueError(f"placement: prefetch={words} reads experts ahead from the disk, which 'disk "
                                 f"experts=off' never reads; remove it (or write prefetch=off)")
        if disk.experts == "off" and r_eff.demote == "off" and r_eff.policy != "inclusive":
            raise ValueError("placement: 'disk experts=off' keeps no copy of an expert outside VRAM and RAM, so "
                             "demote=off would lose VRAM victims; drop demote=off (with the disk off, every victim "
                             "moves back to the RAM slot its replacement left) or use policy=inclusive with "
                             "ram experts=all")
        if disk.ngram == "off" and (r_eff.ngram is None or r_eff.ngram != ALL):
            raise ValueError(f"placement: 'disk ngram=off' never reads n-gram rows from disk, which needs "
                             f"ngram=all, not ngram={r_eff.ngram if r_eff.ngram is not None else '0 (the default)'}")
    kept = tuple(devices[k] for k in sorted(devices, key = lambda s: int(s[5:])) if devices[k].text() != k)
    ram = ram if ram is not None and ram.text() != "ram" else None
    disk = disk if disk is not None and disk.text() != "disk" else None
    return kept, ram, disk


# ---------------------------------------------------------------------------------------- forms

def to_dict(placement) -> dict:
    """The placement as plain data (JSON-ready): {"rules": [{"layers", "device", attrs...}, ...],
    "devices": {"cuda:<n>": {attrs}}, "ram": {attrs}, "disk": {attrs}}; parse() takes it back"""
    d = {"rules": []}
    for r in placement.rules:
        e = {"layers": r.target_text(), "device": r.device}
        for tok in tokens(r.experts.text()):
            k, v = tok.split("=", 1)
            e[k] = unquote(v)
        d["rules"].append(e)
    for store in list(placement.devices) + [x for x in (placement.ram, placement.disk) if x is not None]:
        toks = tokens(store.text())
        body = {k: unquote(v) for k, v in (t.split("=", 1) for t in toks[1:])}
        if toks[0].startswith("cuda:"):
            d.setdefault("devices", {})[toks[0]] = body
        else:
            d[toks[0]] = body
    return d


def dict_to_text(d) -> str:
    """The rule text of a dict / JSON placement (to_dict's form)"""
    if not isinstance(d, dict) or not isinstance(d.get("rules"), list):
        raise ValueError("placement: a dict / JSON placement needs a 'rules' list")
    unknown = set(d) - {"rules", "devices", "ram", "disk"}
    if unknown:
        raise ValueError(f"placement: unknown keys {sorted(unknown)} in a dict / JSON placement "
                         f"(expected rules, devices, ram, disk)")
    lines = []
    for n, r in enumerate(d["rules"], 1):
        if not isinstance(r, dict) or "layers" not in r or "device" not in r:
            raise ValueError(f"placement: dict rule {n} needs 'layers' and 'device' ({r!r})")
        r = dict(r)
        head = f"{r.pop('layers')}={r.pop('device')}"
        lines.append(" ".join([head] + [f"{k}={quote(str(v))}" for k, v in r.items()]))
    devices = d.get("devices") or {}
    if not isinstance(devices, dict):
        raise ValueError("placement: 'devices' of a dict / JSON placement maps cuda:<n> to its attributes")
    for dev, body in devices.items():
        lines.append(" ".join([str(dev)] + [f"{k}={quote(str(v))}" for k, v in dict(body).items()]))
    for key in ("ram", "disk"):
        if d.get(key):
            lines.append(" ".join([key] + [f"{k}={quote(str(v))}" for k, v in dict(d[key]).items()]))
    return "\n".join(lines)
