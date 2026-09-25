"""
Where each routed expert of an MoE layer lies in the checkpoint, for the expert tier
(doc/expert_tiers.md, "The RAM tier"): one extent per expert, read in place from the shards by the
disk engine into a RAM tier slot, and the byte offsets of its trellis tensors inside that read, so
a promotion copies exactly the gate / up / down trellis into a VRAM slot. Torch-free: headers only.

An expert's extent covers every tensor of its projections (trellis, suh, svh, mul1, bias, ...)
from the first byte of the first to the last byte of the last. When those tensors share one file
and nothing else lies between them (every EXL3 checkpoint written by convert.py), the extent is
one piece: one O_DIRECT read, 13,315,596 bytes for a DeepSeek-V4.1 3.0 bpw expert. Otherwise
(projections split over files by an override collection, a foreign tensor inside the span) the
extent is one piece per projection, each covering that projection's trellis alone.

Slot geometry, for every expert of one set of layers (one GPU's cache layers share it):

  vram_slot_bytes   the three trellis tensors back to back (the compact layout of a VRAM slot):
                    3 x 4,423,680 = 13,271,040 bytes for V4.1
  proj_off          offset of each projection's trellis in a compact slot: (0, g, g + u)
  ram_slot_bytes    what one extent read needs in a 4 KiB-aligned slot: each piece rounded up
                    to 4 KiB, plus 4 KiB for a start that is not 4 KiB aligned (the disk engine
                    reads the aligned superset and places the payload at offset mod 4096):
                    13,320,192 bytes (3,252 pages) for V4.1, 80 slots per 1 GiB chunk
"""

from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass, field

PAGE = 4096


def _roundup(n: int, a: int) -> int:
    return -(-n // a) * a


@dataclass(frozen = True)
class Piece:
    """One contiguous read: [offset, offset + length) of a file (absolute file offsets)"""
    path: str
    offset: int
    length: int


@dataclass(frozen = True)
class ExpertExtent:
    """
    One routed expert: its read (one piece, or one per projection) and where each projection's
    trellis lies in it. trellis[p] = (piece index, offset from the piece's first byte, bytes), for
    p over the expert's projections in slot order (gate, up, down; or up, down without a gate).
    """
    name: str                               # the expert's first projection key, for messages
    pieces: tuple                           # Piece, ...
    trellis: tuple                          # (piece, offset in piece, bytes) per projection

    @property
    def contiguous(self) -> bool:
        return len(self.pieces) == 1

    @property
    def path(self) -> str:
        return self.pieces[0].path

    @property
    def offset(self) -> int:
        return self.pieces[0].offset

    @property
    def length(self) -> int:
        return sum(p.length for p in self.pieces)

    @property
    def trellis_bytes(self) -> tuple:
        return tuple(t[2] for t in self.trellis)

    def sub_offsets(self) -> tuple:
        """Offsets of the trellis tensors from the start of a one-piece extent (off_g, off_u, off_d)"""
        if not self.contiguous:
            raise ValueError(f"{self.name}: the extent is {len(self.pieces)} pieces, not one")
        return tuple(t[1] for t in self.trellis)

    def ram_bytes(self, page: int = PAGE) -> int:
        """Bytes this expert's read needs in a page-aligned slot, whatever its file offsets"""
        return sum(_roundup(p.length, page) + page for p in self.pieces)


@dataclass(frozen = True)
class SlotGeometry:
    vram_slot_bytes: int
    proj_off: tuple                         # compact offset of each projection's trellis
    proj_bytes: tuple                       # trellis bytes of each projection
    ram_slot_bytes: int

    @property
    def projections(self) -> int:
        return len(self.proj_bytes)

    def chunk_slots(self, chunk_bytes: int = 1 << 30) -> int:
        """RAM tier slots per chunk of chunk_bytes (80 per GiB for V4.1)"""
        return chunk_bytes // self.ram_slot_bytes


@dataclass
class ExtentIndex:
    """The extents of a list of experts (index order = the caller's order) and their geometry"""
    extents: list = field(default_factory = list)
    geometry: SlotGeometry | None = None

    @property
    def files(self) -> list:
        """Every file the extents read, in first-use order"""
        seen = {}
        for x in self.extents:
            for p in x.pieces:
                seen.setdefault(p.path, None)
        return list(seen)

    def __len__(self):
        return len(self.extents)


class _Files:
    """Per checkpoint file: every tensor's absolute byte range, sorted, to find what lies inside a span"""

    def __init__(self):
        self.ranges = {}                    # path -> (begins, ends, keys), sorted by begin

    def get(self, stc, path: str):
        r = self.ranges.get(path)
        if r is None:
            h = stc.file_headers[path]
            base = h["_header_offset"]
            items = sorted((base + v["data_offsets"][0], base + v["data_offsets"][1], k)
                           for k, v in h.items() if k not in ("__metadata__", "_header_offset"))
            r = ([b for b, _, _ in items], [e for _, e, _ in items], [k for _, _, k in items])
            self.ranges[path] = r
        return r

    def inside(self, stc, path: str, begin: int, end: int) -> list:
        """Keys of the tensors of `path` whose bytes intersect [begin, end), empty ones excluded"""
        b, e, k = self.get(stc, path)
        out = []
        i = max(0, bisect_left(b, begin) - 1)
        while i < len(b) and b[i] < end:
            if e[i] > begin and e[i] > b[i]:
                out.append(k[i])
            i += 1
        return out


def _locate(stc, key: str):
    """(path, absolute begin, absolute end) of one tensor, None when the checkpoint lacks it"""
    s = stc.find_stc(key) if hasattr(stc, "find_stc") else stc
    path = s.tensor_file_map.get(key)
    if path is None:
        return None
    h = s.file_headers[path]
    b, e = h[key]["data_offsets"]
    return s, path, h["_header_offset"] + b, h["_header_offset"] + e


def _own_tensors(stc, prefixes: tuple, by_prefix: dict) -> list:
    """(stc, path, begin, end, key) of every tensor under the given projection prefixes"""
    out = []
    for pre in prefixes:
        for key in by_prefix.get(pre, ()):
            loc = _locate(stc, key)
            if loc is not None:
                out.append(loc + (key,))
    return out


def _prefix_map(stc) -> dict:
    """projection prefix ("layers.3.ffn.experts.5.w1") -> the checkpoint keys under it"""
    maps = []
    if hasattr(stc, "stcs"):                # VariantSafetensorsCollection: overrides, then main
        maps = [s.tensor_file_map for _, _, s in stc.stcs] + [stc.main.tensor_file_map]
    else:
        maps = [stc.tensor_file_map]
    out = {}
    for m in maps:
        for key in m:
            pre, _, _ = key.rpartition(".")
            lst = out.setdefault(pre, [])
            if key not in lst:
                lst.append(key)
    return out


def build_extent_index(stc, experts: list, page: int = PAGE) -> ExtentIndex:
    """
    The extents of `experts`, each a tuple of projection keys in slot order ((gate, up, down), or
    (up, down) for gateless experts), from the checkpoint headers of `stc` (a SafetensorsCollection,
    a VariantSafetensorsCollection, or anything with the same file_headers / tensor_file_map /
    find_stc). Every expert must have the same number of projections and the same trellis sizes
    (one slot geometry). Raises ValueError naming the expert otherwise, or when a trellis is missing.
    """
    files = _Files()
    by_prefix = _prefix_map(stc)
    out = ExtentIndex()
    proj_bytes = None
    ram_slot = 0
    for keys in experts:
        keys = tuple(keys)
        name = keys[0]
        trel = []
        for k in keys:
            loc = _locate(stc, k + ".trellis")
            if loc is None:
                raise ValueError(f"expert extents: {k}.trellis is not in the checkpoint (only EXL3 experts can be "
                                 f"cached)")
            trel.append(loc)
        own = _own_tensors(stc, keys, by_prefix)
        paths = {path for _, path, _, _, _ in own}
        pieces = None
        if len(paths) == 1:
            path = next(iter(paths))
            begin = min(b for _, _, b, _, _ in own)
            end = max(e for _, _, _, e, _ in own)
            owned = {key for _, _, _, _, key in own}
            src = own[0][0]
            foreign = [k for k in files.inside(src, path, begin, end) if k not in owned]
            if not foreign:
                pieces = (Piece(path, begin, end - begin),)
                trellis = tuple((0, b - begin, e - b) for _, _, b, e in trel)
        if pieces is None:
            pieces = tuple(Piece(path, b, e - b) for _, path, b, e in trel)
            trellis = tuple((i, 0, e - b) for i, (_, _, b, e) in enumerate(trel))
        x = ExpertExtent(name, pieces, trellis)
        if proj_bytes is None:
            proj_bytes = x.trellis_bytes
        elif x.trellis_bytes != proj_bytes:
            raise ValueError(f"expert extents: {name} has trellis tensors of {x.trellis_bytes} bytes, other experts "
                             f"{proj_bytes}; one slot geometry is needed")
        ram_slot = max(ram_slot, x.ram_bytes(page))
        out.extents.append(x)
    if proj_bytes is not None:
        off, acc = [], 0
        for b in proj_bytes:
            off.append(acc)
            acc += b
        out.geometry = SlotGeometry(acc, tuple(off), tuple(proj_bytes), ram_slot)
    return out


def expert_keys(module) -> list:
    """The projection keys of every routed expert of an MoE module (BlockSparseMLP), in expert order"""
    gated = bool(getattr(module, "gates", None)) and getattr(module, "gated", True)
    out = []
    for e in range(module.num_experts):
        keys = ([module.gates[e].key] if gated else []) + [module.ups[e].key, module.downs[e].key]
        out.append(tuple(keys))
    return out


def build_for_modules(stc, modules: list, page: int = PAGE) -> tuple:
    """(ExtentIndex, [first index of each module]) for the routed experts of several MoE modules, in
    the given order (the tier's cache layers on one GPU, forward order: key = layer * E + expert)"""
    keys, starts = [], []
    for m in modules:
        starts.append(len(keys))
        keys.extend(expert_keys(m))
    return build_extent_index(stc, keys, page), starts
