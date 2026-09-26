"""
Expert profiles: per MoE layer, how often each routed expert is selected, measured ahead of time
on some workload and shipped as a file, so a load can start with the right experts resident
instead of learning them at runtime (doc/expert_tiers.md, "Profiles and the heat file").

A placement names one per layer rule (profile=<spec>). A spec is one name or path, or several with
weights: "code", "wiki:1,code:3", "/data/p/chat.exl3moe". Counts are normalized per layer before
the weighting, so a corpus with more tokens does not simply win. What a profile seeds:

  experts=cache            the hot pins (hot=: the layer's most selected experts, not its first),
                           the order of the cold fill (VRAM pool first, then the RAM tier: the most
                           selected experts of every layer, compared across layers by their share
                           of their layer's selections), and the initial heat (each expert at the
                           decayed count its share would reach in steady decode). It seeds; it never
                           freezes: the policies move experts by the live heat from the first call
  stream / cpu with hot=,  which experts stay resident in VRAM (the hottest E - k) and which go to
  split                    system RAM; the dynamic placement sweeps keep moving them by live traffic

Files, found by name in <model dir>/moe_profiles/, then each directory of EXL3_MOE_PROFILE_DIR
(os.pathsep-separated), then ~/.cache/exllamav3/moe_profiles/, trying the extensions in this order
(a path is taken as it is):

  .exl3moe, .safetensors   safetensors: "counts" [layers, experts] (any numeric dtype) and/or
                           "ranking" [layers, experts] (expert ids, most selected first); metadata
                           "layer_keys" (JSON list of the MoE modules' keys) and "fingerprint" (JSON)
  .npz                     counts_decode (preferred), counts or counts_prefill, [layers, experts] or
                           [prompts, layers, experts] (summed); layer keys from a <name>.meta.json
                           sidecar ({"layer_keys": [...], "fingerprint": {...}}) when present
  .json                    {"<layer key>": [count per expert], ...}

Rows are matched to layers by key when the file names them, else by MoE-layer ordinal. A profile
built for another model (architecture, layer count, experts per layer, hidden or expert width in
its fingerprint differing from the model's) is refused; a profile of another quantization of the
same model is used (routing differs a little between quantizations: a seed, not a placement).

The heat file (EXL3_MOE_HEAT_FILE) is written in the .exl3moe format: "counts" holds each expert's
decayed routing count, "layer_keys" the cache layers' keys; a later load reads it back as the seed
when the placement names no profile.

Torch-free (numpy only).
"""

from __future__ import annotations
import json
import os
import struct

EXTS = (".exl3moe", ".safetensors", ".npz", ".json")
MODEL_KEYS = ("architecture", "layers", "experts", "moe_intermediate_size", "hidden_size")
_ST_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2", "U8": "|u1",
              "U16": "<u2", "U32": "<u4", "U64": "<u8", "I8": "|i1"}


def _np():
    import numpy as np
    return np


def search_dirs(model_dir: str | None = None) -> list[str]:
    out = []
    if model_dir:
        out.append(os.path.join(model_dir, "moe_profiles"))
    env = os.environ.get("EXL3_MOE_PROFILE_DIR")
    if env:
        out += [d for d in env.split(os.pathsep) if d]
    out.append(os.path.join(os.path.expanduser("~"), ".cache", "exllamav3", "moe_profiles"))
    return out


def resolve_path(name: str, model_dir: str | None = None) -> str:
    """A path as given, else <name><ext> in the search directories (first match)"""
    if os.path.isfile(name):
        return name
    for d in search_dirs(model_dir):
        for ext in EXTS:
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return p
    raise ValueError(f"expert profile {name!r} not found: looked in {', '.join(search_dirs(model_dir))} for "
                     f"{name}{{{','.join(EXTS)}}}")


def parse_spec(spec: str) -> list[tuple[str, float]]:
    """'code:3,wiki:1' -> [('code', 3.0), ('wiki', 1.0)] (a weight defaults to 1; a path that exists
    keeps its colons)"""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        w = 1.0
        if ":" in part and not os.path.exists(part):
            head, _, tail = part.rpartition(":")
            try:
                w = float(tail)
                part = head
            except ValueError:
                pass
        if not (w >= 0.0) or w == float("inf"):
            raise ValueError(f"expert profile {spec!r}: the weight of {part!r} must be a finite number >= 0")
        out.append((part, w))
    if not out:
        raise ValueError(f"expert profile {spec!r} names no profile")
    if not any(w > 0 for _, w in out):
        raise ValueError(f"expert profile {spec!r}: every weight is 0")
    return out


def read_safetensors(path: str, names: tuple | None = None):
    """(tensors {name: ndarray}, metadata {str: str}) of a safetensors file, reading only `names`"""
    np = _np()
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        n = struct.unpack("<Q", head)[0]
        if n > 64 << 20:
            raise ValueError(f"{path}: a safetensors header of {n} bytes is implausible")
        hdr = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        out = {}
        for k, info in hdr.items():
            if k == "__metadata__" or (names is not None and k not in names):
                continue
            dt = _ST_DTYPES.get(info["dtype"])
            if dt is None:
                raise ValueError(f"{path}: tensor {k} has dtype {info['dtype']}, not a number type")
            a, b = info["data_offsets"]
            f.seek(base + a)
            buf = f.read(b - a)
            if len(buf) != b - a:
                raise ValueError(f"{path}: tensor {k} is cut short")
            out[k] = np.frombuffer(buf, dtype = np.dtype(dt)).reshape(info["shape"])
        return out, dict(hdr.get("__metadata__") or {})


def write_safetensors(path: str, tensors: dict, metadata: dict):
    """A safetensors file of float64 / int64 arrays, written to a temporary name and renamed"""
    np = _np()
    header, blobs, off = {}, [], 0
    for k, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {np.dtype("float64"): "F64", np.dtype("int64"): "I64"}[a.dtype]
        b = a.tobytes()
        header[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {k: v if isinstance(v, str) else json.dumps(v) for k, v in metadata.items()}
    hj = json.dumps(header).encode("utf-8")
    hj += b" " * (-len(hj) % 8)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_counts(path: str):
    """(counts [layers, experts] float64, layer keys or None, fingerprint or None) of one profile file"""
    np = _np()
    low = path.lower()
    if low.endswith((".exl3moe", ".safetensors")):
        t, md = read_safetensors(path, ("counts", "ranking"))
        keys = json.loads(md["layer_keys"]) if "layer_keys" in md else None
        fp = json.loads(md["fingerprint"]) if isinstance(md.get("fingerprint"), str) else None
        if "counts" in t:
            c = np.asarray(t["counts"], dtype = np.float64)
        elif "ranking" in t:
            rank = np.asarray(t["ranking"], dtype = np.int64)
            if rank.ndim != 2:
                raise ValueError(f"{path}: ranking must be [layers, experts]")
            # counts that reproduce the ranking under a descending stable sort
            c = np.empty(rank.shape, dtype = np.float64)
            np.put_along_axis(c, rank, np.broadcast_to(np.arange(rank.shape[1], 0, -1, dtype = np.float64),
                                                       rank.shape), axis = 1)
        else:
            raise ValueError(f"{path}: no 'counts' or 'ranking' tensor")
    elif low.endswith(".npz"):
        z = np.load(path, allow_pickle = False)
        name = next((n for n in ("counts_decode", "counts", "counts_prefill") if n in z), None)
        if name is None:
            raise ValueError(f"{path}: no counts_decode, counts or counts_prefill array")
        c = np.asarray(z[name], dtype = np.float64)
        if c.ndim == 3:
            c = c.sum(axis = 0)
        keys = fp = None
        side = os.path.splitext(path)[0] + ".meta.json"
        if os.path.isfile(side):
            m = json.load(open(side))
            keys = m.get("layer_keys") or None
            fp = m.get("fingerprint") if isinstance(m.get("fingerprint"), dict) else None
    elif low.endswith(".json"):
        d = json.load(open(path))
        if not isinstance(d, dict) or not d:
            raise ValueError(f"{path}: a JSON profile is an object of layer key -> counts")
        keys = list(d)
        c = np.asarray([d[k] for k in keys], dtype = np.float64)
        fp = None
    else:
        raise ValueError(f"{path}: not a profile (the extensions are {', '.join(EXTS)})")
    if c.ndim != 2 or c.shape[1] < 1:
        raise ValueError(f"{path}: counts must be [layers, experts], not {list(c.shape)}")
    if keys is not None and len(keys) != c.shape[0]:
        raise ValueError(f"{path}: {len(keys)} layer keys for {c.shape[0]} rows of counts")
    if not np.all(np.isfinite(c)) or (c < 0).any():
        raise ValueError(f"{path}: counts must be finite and not negative")
    return c, keys, fp


def model_fingerprint(config, num_experts: int | None = None) -> dict:
    """The model identity a profile is checked against (MODEL_KEYS)"""
    cd = getattr(config, "config_dict", None) or {}
    t = cd.get("text_config") if isinstance(cd.get("text_config"), dict) else cd
    arch = cd.get("architectures") or [None]
    fp = {"architecture": arch[0], "layers": t.get("num_hidden_layers"),
          "experts": num_experts if num_experts is not None else (t.get("n_routed_experts") or t.get("num_experts")),
          "moe_intermediate_size": t.get("moe_intermediate_size"), "hidden_size": t.get("hidden_size")}
    return {k: v for k, v in fp.items() if v is not None}


class Profile:
    """One placement's profile spec, merged: per layer, each expert's share of its layer's selections"""

    def __init__(self, spec: str, moe_keys: list[str], num_experts: int, model_dir: str | None = None,
                 fingerprint: dict | None = None):
        np = _np()
        self.spec = spec
        self.moe_keys = list(moe_keys)
        self.E = num_experts
        acc = {}
        self.paths = []
        for name, w in parse_spec(spec):
            path = resolve_path(name, model_dir)
            c, keys, fp = load_counts(path)
            if c.shape[1] != num_experts:
                raise ValueError(f"expert profile {path}: {c.shape[1]} experts per layer, the model's MoE layers have "
                                 f"{num_experts}; it was built for another model")
            if fp and fingerprint:
                bad = [f"{k}: profile {fp[k]!r}, model {fingerprint[k]!r}" for k in MODEL_KEYS
                       if k in fp and k in fingerprint and fp[k] != fingerprint[k]]
                if bad:
                    raise ValueError(f"expert profile {path} was built for another model ({'; '.join(bad)})")
            if keys is None:
                if c.shape[0] != len(self.moe_keys):
                    raise ValueError(f"expert profile {path} names no layers and has {c.shape[0]} rows, the model has "
                                     f"{len(self.moe_keys)} MoE layers")
                keys = self.moe_keys
            tot = c.sum(axis = 1, keepdims = True)
            share = c / np.where(tot > 0, tot, 1.0)
            for k, row in zip(keys, share):
                acc[k] = acc.get(k, 0.0) + w * row
            self.paths.append(path)
        self.share = {}
        for k, row in acc.items():
            s = row.sum()
            self.share[k] = row / s if s > 0 else row

    def shares(self, key: str):
        """Each expert's share of the layer's selections (sums to 1), None when the profile lacks the layer"""
        return self.share.get(key)

    def ranking(self, key: str) -> list[int] | None:
        """The layer's experts, most selected first (ties in expert order), None when it lacks the layer"""
        s = self.shares(key)
        if s is None:
            return None
        np = _np()
        return [int(e) for e in np.argsort(-s, kind = "stable")]


def seed_order(keys_by_layer: list[tuple[int, str]], shares: dict, E: int, pinned: set) -> list[int]:
    """The cold-fill order of the tier's keys (lc * E + e) from per-layer shares: every key by its
    expert's share of its layer, highest first; layers without shares fall back to the round robin
    (expert 0 of every layer, then expert 1, ...) after the profiled ones, by share 0"""
    order = []
    for lc, key in keys_by_layer:
        s = shares.get(key)
        for e in range(E):
            k = lc * E + e
            if k in pinned:
                continue
            order.append((-(float(s[e]) if s is not None else 0.0), e, lc, k))
    order.sort()
    return [k for _, _, _, k in order]


def seed_heat(share: float, top_k: int, halflife: int) -> int:
    """The 16.16 heat of an expert with this share of its layer's selections in steady decode:
    rate x halflife / ln 2 (top-k selections per token per layer)"""
    v = share * top_k * halflife / 0.6931471805599453
    return max(0, min((1 << 32) - 1, int(round(v * 65536))))
