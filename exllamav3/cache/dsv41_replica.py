"""
Cross-device replica of a DeepSeek-V4.1 compressed-KV pool, for the cached path.

Why it exists. Every compressed layer attends over its kv source's paged pool, and
dsa_attn reads that pool directly on the consumer's own device. The split of an explicit
placement that falls inside a kv group -- on DeepSeek-V4.1-Flash, "0-11=cuda:0; 12-39=cuda:1"
puts source 8 on the first device and its consumers 12 and 13 on the second -- therefore
leaves the consumers without their pool (architecture/dsv41/placement.py).
The stateless path rebuilds the pool every forward and simply mirrors it whole
(DSV41Attention._pools_on_my_device). The cached path cannot: at 1M tokens source 8's pool
is 640 MiB (512 MiB without the index keys its consumers do not read), and recopying that
for every decoded token would put half a GiB per token on the link between the devices.

What it is. CacheLayer_dsv41_replica is an ordinary paged CacheLayer_dsa, owned by the
first consumer on the foreign device and built with the source's page geometry. Pool
entries alias the token page table -- entry e of a sequence lives at row
bt[e // epp] * epp + e % epp, epp = PAGE_SIZE // m -- so a replica page holds exactly the
entries its source page holds. Registered in Cache like any other CacheLayer, it moves in
lockstep with the source through everything the generator does to pages: partial-page
prefix reuse (Cache.copy_page loops over every layer), defragmentation (cache_rotate over
Cache.get_all_tensors) and the CPU tier's per-page slabs.

How it stays current. sync_pool_replica copies only the entries a forward appended:
entries [e0, e1) of one sequence, with e0 = position // m and e1 = (position + tokens) // m
(modules/dsv41_cached.emission_range), exactly the range DeepSeek's reference writes
(compress_kv_cache[:, start_pos // ratio : start_pos // ratio + n_closed]). About 1 KiB per
entry, so 1 MiB per 2048-token rate-2 chunk and 1 KiB every second decode token.

Invariants the callers own:
  * every forward that writes source entries syncs the same range in the same forward;
  * rewinds need nothing: entries are position-indexed, a rewind only moves the position
    back, and the next forward rewrites (and so re-syncs) every entry from the new
    position // m on, exactly as it rewrites the source;
  * prefix-cache hits reuse replica pages that were synced when those tokens were first
    computed, and a partial-page hit copies the replica page with the source page.

The transfer. Rows are gathered on the source device into one packed buffer (one segment
per pool: pool_c, or pool_q + pool_s when quantized, then pool_r, then pool_idx when the
replica keeps index K), moved in one copy, and scattered with index_copy_ on the
destination's current stream. Without peer access the move is a D2H into a grow-only
pinned staging buffer per device pair followed by an H2D, ordered by events so that nothing
on either side ever waits on the host:

    src stream:  gather -> [wait: staging free] -> D2H -> record ready
    dst stream:  [wait: ready] -> H2D -> record staging free -> scatter

The staging buffer is capped at STAGE_MAX_BYTES; longer ranges are synced in pieces.
"""

from __future__ import annotations

import torch

from ..util.device_copy import needs_bounce, to_device
from .dsa import CacheLayer_dsa
from .dsv41 import DeviceMemo, check_pool_addressing


# Pinned staging per device pair is grow-only up to this size; a longer range is synced in
# pieces. Pinned host memory cannot be paged out, so the buffers stay small
STAGE_MAX_BYTES = 8 << 20
_STAGE_MIN_BYTES = 64 << 10

# Segment offsets inside the packed transfer buffer: keeps every typed view aligned
_SEG_ALIGN = 256

# Widest integer word dividing a pool row, so the gather and scatter move 4-byte words
_WORDS = ((4, torch.int32), (2, torch.int16), (1, torch.uint8))


class CacheLayer_dsv41_replica(CacheLayer_dsa):
    """
    Paged replica of one kv source's pool, allocated on the replica owner's device.

    Built for the owner (the first consumer of the group on a device other than the
    source's) from the owner's attention module, whose compress_rate, head_dim and
    rope_head_dim equal the source's -- consumers read their source's pool, so they share
    its geometry. The replica keeps index K (D_i = index_head_dim) only when with_index_k:
    with the split at 12 on DeepSeek-V4.1-Flash, consumers 12 and 13 do not score, so their replica
    of source 8 is 1,024 B per entry rather than the source's 1,280.

    Nothing here writes the pool on its own: sync_pool_replica fills it.
    """

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int = 0,
        v_bits: int | None = None,
        compand_a: float = 0.0,
        *,
        with_index_k: bool = False,
        **kwargs
    ):
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits, compand_a, **kwargs)
        # The source this replica mirrors: the owner's replica_of, set by its pool route. A
        # consumer's replica can only ever be of its own kv source, so that is the fallback
        replica_of = getattr(attention, "replica_of", None)
        if replica_of is None:
            replica_of = getattr(attention, "kv_source_layer", None)
        if replica_of is None:
            raise ValueError(f"{getattr(attention, 'key', attention)}: a pool replica needs a kv source")
        if replica_of == getattr(attention, "layer_idx", None):
            raise ValueError(f"layer {replica_of} is the kv source itself; it cannot own a replica of its pool")
        self.replica_of = replica_of
        self.with_index_k = bool(with_index_k)
        if self.with_index_k:
            D_i = getattr(attention, "index_head_dim", None)
            if not D_i:
                raise ValueError(f"with_index_k requires attention.index_head_dim, got {D_i}")
            self.D_i = D_i
        else:
            # CacheLayer_dsa keys this on V4's layer_type == "csa"; V4.1 decides per replica
            self.D_i = 0
        check_pool_addressing(self, f"replica of layer {replica_of} on layer "
                                    f"{getattr(attention, 'layer_idx', '?')}")

    def free(self):
        dev = self.device
        super().free()
        if dev is not None:
            dev = DeviceMemo._norm(dev)
            if dev.type == "cuda":
                release_staging(dev)

    def tp_export(self, plan):
        raise NotImplementedError("Tensor-parallel loading is not supported for V4.1 pool replicas")


# ---------------------------------------------------------------------------------------------
# Transfer

def _pairs(src: CacheLayer_dsa, dst: CacheLayer_dsa) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(source pool, replica pool) pairs to copy, after checking the two layers agree."""
    if src.device is None or dst.device is None:
        raise RuntimeError("sync_pool_replica: source and replica must both be allocated")
    geo = lambda l: (l.compress_rate, l.epp, l.D_c, l.D_r, l.k_bits)
    if geo(src) != geo(dst):
        raise ValueError(
            f"sync_pool_replica: geometry differs, source (m, epp, D_c, D_r, k_bits) = {geo(src)}, "
            f"replica {geo(dst)}")
    if dst.D_i and src.D_i != dst.D_i:
        raise ValueError(
            f"sync_pool_replica: replica keeps index K of width {dst.D_i}, source has {src.D_i}")
    pairs = [(src.pool_q, dst.pool_q), (src.pool_s, dst.pool_s)] if src.k_bits else \
            [(src.pool_c, dst.pool_c)]
    pairs.append((src.pool_r, dst.pool_r))
    if dst.D_i:
        pairs.append((src.pool_idx, dst.pool_idx))
    return pairs


def _row_view(t: torch.Tensor) -> torch.Tensor:
    """(rows, words) integer view of a page-major (num_pages, epp, D) pool."""
    flat = t.view(-1, t.shape[-1])
    rb = flat.shape[1] * flat.element_size()
    for w, dt in _WORDS:
        if rb % w == 0:
            return flat.view(dt)
    raise AssertionError("unreachable")


def _align(n: int, a: int) -> int:
    return -(-n // a) * a


def _seg(buf: torch.Tensor, off: int, nbytes: int, like: torch.Tensor) -> torch.Tensor:
    """Typed (n, words) view of bytes [off, off + nbytes) of a flat uint8 buffer."""
    return buf[off : off + nbytes].view(like.dtype).view(-1, like.shape[1])


def _bt_pages(bt_row: torch.Tensor, e1: int, epp: int, device: torch.device, what: str) -> torch.Tensor:
    """
    One sequence's block-table row as a flat page list on device, checked to cover entry e1 - 1.

    Only (P,) or (1, P) is accepted. A whole (B, P) batch table would flatten into one long
    "row" whose first P pages are sequence 0's, so a per-row caller that forgot to slice
    it would sync sequence 0's pages for every sequence and leave the others' replica
    entries stale, without any error.
    """
    if not (bt_row.dim() == 1 or (bt_row.dim() == 2 and bt_row.shape[0] == 1)):
        raise ValueError(
            f"sync_pool_replica: {what} must be one sequence's block-table row, (P,) or (1, P), "
            f"got shape {tuple(bt_row.shape)}; slice the batch's table per sequence")
    bt = bt_row.reshape(-1)
    if (e1 - 1) // epp >= bt.numel():
        raise IndexError(
            f"sync_pool_replica: entry {e1 - 1} is on page {(e1 - 1) // epp}, "
            f"{what} has {bt.numel()} pages")
    return to_device(bt, device)


def _entry_rows(bt: torch.Tensor, e0: int, e1: int, epp: int) -> torch.Tensor:
    """Flat pool rows of entries [e0, e1): bt[e // epp] * epp + e % epp, as int64 on bt's device."""
    e = torch.arange(e0, e1, dtype = torch.long, device = bt.device)
    return bt.index_select(0, e // epp).long() * epp + e % epp


class _Staging:
    """Grow-only pinned host buffer for one (source, replica) device pair."""
    __slots__ = ("host", "free_evt")

    def __init__(self):
        self.host = None
        self.free_evt = None    # recorded on the replica's stream after its H2D read the buffer


_staging: dict[tuple[int, int], _Staging] = {}
_p2p: dict[tuple[int, int], bool] = {}

# The D2H into the shared staging buffer waits until the previous H2D has read it, and the
# H2D waits until its D2H has landed. Tests turn each off to show it is load-bearing (the
# matching stress test must then fail)
_staging_wait = True
_ready_wait = True


def release_staging(device: torch.device | None = None):
    """
    Drop pinned staging buffers (all, or those whose pair involves device). Safe with copies
    in flight: torch's host allocator holds a freed pinned block until the events of the
    non-blocking copies that used it have completed.
    """
    for k in list(_staging):
        if device is None or device.index in k:
            del _staging[k]


def _transfer_mode(sdev: torch.device, ddev: torch.device) -> str:
    if sdev == ddev:
        return "local"
    if sdev.type == "cuda" and ddev.type == "cuda":
        key = (sdev.index, ddev.index)
        if key not in _p2p:
            _p2p[key] = torch.cuda.can_device_access_peer(sdev.index, ddev.index)
        # A reported peer link can still corrupt copies on some PCIe/IOMMU
        # topologies. Share the engine's probe and explicit bounce override.
        return "direct" if _p2p[key] and not needs_bounce(sdev, ddev) else "staged"
    return "direct"


def _staged(packed: torch.Tensor, sdev: torch.device, ddev: torch.device) -> torch.Tensor:
    """packed (flat uint8 on sdev) -> the same bytes on ddev, through pinned host memory,
    ordered with events on the two devices' current streams (no host synchronization)."""
    nbytes = packed.numel()
    key = (sdev.index, ddev.index)
    st = _staging.get(key)
    if st is None:
        st = _staging[key] = _Staging()
    if st.host is None or st.host.numel() < nbytes:
        size = max(_STAGE_MIN_BYTES, 1 << (nbytes - 1).bit_length())
        # A fresh buffer has no pending reader. The old one may still feed an H2D in flight;
        # the host allocator keeps it alive until that copy's event completes
        st.host = torch.empty(size, dtype = torch.uint8, pin_memory = True)
        st.free_evt = None
    host = st.host[:nbytes]
    s_src = torch.cuda.current_stream(sdev)
    s_dst = torch.cuda.current_stream(ddev)

    if st.free_evt is not None and _staging_wait:
        s_src.wait_event(st.free_evt)
    host.copy_(packed, non_blocking = True)             # on s_src
    ready = torch.cuda.Event()
    ready.record(s_src)

    if _ready_wait:
        s_dst.wait_event(ready)
    landed = torch.empty(nbytes, dtype = torch.uint8, device = ddev)
    landed.copy_(host, non_blocking = True)             # on s_dst
    st.free_evt = torch.cuda.Event()
    st.free_evt.record(s_dst)
    return landed


def sync_pool_replica(
    src: CacheLayer_dsa,
    dst: CacheLayer_dsa,
    bt_src_row: torch.Tensor,
    bt_dst_row: torch.Tensor,
    e0: int,
    e1: int,
    *,
    path: str | None = None,
) -> None:
    """
    Copy pool entries [e0, e1) of one sequence from a kv source's CacheLayer to its replica.

    bt_src_row / bt_dst_row: that sequence's block-table row, (1, P) or (P,) int, one copy
    per device (the page table is shared, so in production the values are the same; they
    may differ here). A (B, P) table with B > 1 is refused rather than flattened: batched
    callers slice it per sequence. Entry e is read from row bt_src[e // epp] * epp + e % epp
    of every source pool and written to the corresponding row of the replica: pool_c (or
    pool_q and pool_s when quantized), pool_r, and pool_idx when the replica keeps index K.
    Bitwise. e0 == e1 is a no-op. Both rows are checked before anything moves.

    Stream-ordered: reads on the source device's current stream, so a pending write of the
    same entries by the source layer completes first; writes on the replica device's
    current stream, so the consumers queued after this call see the entries. The host
    never synchronizes.

    path forces the transfer: "staged" (pinned host bounce), "direct" (one copy_; peer
    access when available) or "local" (same device); None chooses.
    """
    e0 = int(e0)
    e1 = int(e1)
    if e0 < 0 or e1 < e0:
        raise ValueError(f"sync_pool_replica: bad entry range [{e0}, {e1})")
    if e0 == e1:
        return

    pairs = _pairs(src, dst)
    views = [(_row_view(s), _row_view(d)) for s, d in pairs]
    for sv, dv in views:
        assert sv.dtype == dv.dtype and sv.shape[1] == dv.shape[1]
    row_bytes = [sv.shape[1] * sv.element_size() for sv, _ in views]
    entry_bytes = sum(row_bytes)
    step = max(1, (STAGE_MAX_BYTES - _SEG_ALIGN * len(views)) // entry_bytes)

    sdev = DeviceMemo._norm(src.device)
    ddev = DeviceMemo._norm(dst.device)
    mode = path or _transfer_mode(sdev, ddev)
    if mode == "local":
        assert sdev == ddev, "local sync needs both layers on one device"
    elif mode == "staged":
        assert sdev.type == "cuda" and ddev.type == "cuda" and sdev != ddev, \
            "staged sync is between two CUDA devices"
    else:
        assert mode == "direct", f"unknown sync path {mode!r}"

    # Both rows are checked before any piece moves, so a refusal leaves the replica untouched
    bt_s = _bt_pages(bt_src_row, e1, src.epp, sdev, "bt_src_row")
    bt_d = _bt_pages(bt_dst_row, e1, dst.epp, ddev, "bt_dst_row")

    for a in range(e0, e1, step):
        b = min(a + step, e1)
        n = b - a
        offs, off = [], 0
        for rb in row_bytes:
            offs.append(off)
            off = _align(off + n * rb, _SEG_ALIGN)
        nbytes = off

        rows_s = _entry_rows(bt_s, a, b, src.epp)
        packed = torch.empty(nbytes, dtype = torch.uint8, device = sdev)
        for (sv, _), o, rb in zip(views, offs, row_bytes):
            torch.index_select(sv, 0, rows_s, out = _seg(packed, o, n * rb, sv))

        if mode == "local":
            landed = packed
        elif mode == "staged":
            landed = _staged(packed, sdev, ddev)
        else:
            # Between CUDA devices torch orders this copy against both current streams;
            # with a CPU end it must block, or the scatter could read unfinished bytes
            landed = packed.to(ddev, non_blocking = sdev.type == "cuda" and ddev.type == "cuda")

        rows_d = _entry_rows(bt_d, a, b, dst.epp)
        for (_, dv), o, rb in zip(views, offs, row_bytes):
            dv.index_copy_(0, rows_d, _seg(landed, o, n * rb, dv))
