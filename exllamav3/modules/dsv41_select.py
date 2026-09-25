"""
DeepSeek-V4.1 indexer selection that stays bounded at 1M context: a tiled
top-k plus the two-level candidate stage.

V4's _indexer_topk scores every query row against the whole pool in one
(seq, next_pow2(ec)) fp16 matrix. At 1M on a rate-1 pool that is 4 GiB per
2048-row chunk (2 GiB on a rate-2 pool), and the autosplit measures at
past_len 0, so it never sees it coming. Here the rows go in slabs of
slab_rows and the pool in tiles of tile_entries: each tile is scored into one
fixed (slab_rows, tile_entries) fp16 buffer, reduced to its local top-k, and
merged into the running set with exllamav3's dsa_topk_tile /
dsa_topk_merge_tiles (the pattern of qsa_indexer._select_rows). The merge
keeps the single-pass kernel's total order -- score descending, index
ascending -- so the result is the one an untiled pass would give.

Selection semantics are DeepSeek's reference (inference/model.py,
Indexer.forward and select_candidate_blocks):

  score[r, j] = sum_h w[r, h] * relu(q[r, h] . k[j]) * scale
                for j < vis(r) = min((pos0 + r + 1) // m, ec), else -inf

  (the caller passes w = weights_proj(x) * 128^-0.5 * 32^-0.5 and scale 1,
  or raw weights and that scale), then the top `topk` visible entries,
  ascending, -1 padded.

The candidate stage (architecture/dsv41/candidates.py has the untiled
reference and the details):

  * want_cand (the candidate source, layer 20): block scores (max over 8
    visible entries, NaN-propagating, newest block pinned to +inf) are
    collected from the same tiles into a (slab_rows, ceil(T / 8)) fp16 buffer;
    the top n_blocks blocks become cand (seq, n_blocks) int32, ascending,
    -1 for blocks whose score is -inf or NaN. The source's own top-k is
    UNMASKED; layers 21-23 reuse it.
  * cand_in (index sources above the source: 24/28/32/36): every tile is
    masked to -inf outside the row's candidate blocks before its top-k.
    Until a row sees more than n_blocks * block = 16,384 entries this changes
    nothing, because every visible block is a candidate.

Backends: 'ext' is the Triton scorer (dsa_indexer_scores) plus the CUDA top-k
kernels; 'torch' runs the same slabs and tiles with plain torch ops on any
device (CPU tests, GPU cross-checks); 'auto' picks ext for CUDA tensors.
Ablation (tests only, modules/dsv41_ablation.py): EXL3_DSV41_ABLATE containing
'no_candidates' disables both halves of the candidate stage (cand is None,
cand_in is ignored).

Scores are fp16 in the ext backend (the scorer's store type). The torch
backend rounds to fp16 too by default (score_dtype) so the two select from
the same numbers; score_dtype float32/float64 keeps the torch backend's
accumulation precision, which the reference-math tests use.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..model.math_policy import STABLE_ARITHMETIC
from ..architecture.dsv41.candidates import (
    visible_counts,
    topk_total_order,
    sort_ids_ascending,
    publish_candidate_blocks,
    mask_scores_to_candidates,
)
from .dsv41_ablation import ablated

TOPK = 512
BLOCK = 8
N_BLOCKS = 2048
SLAB_ROWS = 256
TILE_ENTRIES = 65536

_NEG_INF = float("-inf")

# The scorer and the top-k kernels address the score tile and the block-max buffer as
# row * stride in int32
INT32_LIMIT = 2 ** 31


class SelectResult(NamedTuple):
    indices: torch.Tensor | None    # (seq, ceil32(topk)) int32 ascending, -1 padded; None = dense
    k_len: int                      # topk when indices is set, else 0
    cand: torch.Tensor | None       # (seq, n_blocks) int32 block ids ascending, -1 padded


def _ceil(a: int, b: int) -> int:
    return -(-a // b) * b


def _tile_quantum(epp: int, block: int) -> int:
    """Tiles must start on a pool page (paged addressing) and on a block boundary (candidates)."""
    return math.lcm(max(epp, 1), block)


def _effective_tile(rows_max: int, ec: int, slab_rows: int, tile_entries: int, quantum: int) -> int:
    """
    Tile width for a call. A call with fewer rows than a slab (decode, small chunks)
    spends the same score-tile budget, slab_rows * tile_entries entries, on wider tiles:
    potentially one pass over a 1M pool at decode instead of 16 tiles. Actual latency
    depends on device, pool layout and row count and must be measured separately.
    Widths stay powers of two, capped at the pool, because the tile's row stride is a
    constexpr of the scoring kernel: the compile count stays logarithmic.
    """
    tile = _ceil(max(tile_entries, 1), quantum)
    if 0 < rows_max < slab_rows:
        budget = (slab_rows * tile) // rows_max
        wide = min(1 << (max(ec, 1) - 1).bit_length(), 1 << (budget.bit_length() - 1))
        if wide > tile:
            tile = _ceil(wide, quantum)
    return tile


def _keys_for(idx_pool: torch.Tensor, bt_flat: torch.Tensor | None, epp: int,
              t0: int, t1: int) -> torch.Tensor:
    """Index keys of entries [t0, t1) as (t1 - t0, D) rows (torch backend)."""
    D = idx_pool.shape[-1]
    flat = idx_pool.reshape(-1, D)
    if bt_flat is None:
        return flat[t0 : t1]
    e = torch.arange(t0, t1, device = idx_pool.device, dtype = torch.long)
    phys = bt_flat.to(idx_pool.device).long()[e // epp] * epp + e % epp
    return flat[phys]


def _torch_scores(q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor, q_pos0: int, m: int,
                  bound: int, scale: float, score_dtype: torch.dtype, width: int | None = None
                  ) -> torch.Tensor:
    """
    The scorer's math in torch: acc = sum_h relu(q_h . k) * w_h in fp32 (fp64 when
    score_dtype is float64), head by head in the row-tiled kernel's order, times scale,
    rounded to score_dtype, -inf at columns >= min((q_pos0 + r + 1) // m, bound). Returned
    in the accumulation dtype, `width` columns (padding columns are -inf).

    The same function as the ext scorer for finite q and k: bitwise when every product and
    partial sum is exact (the tests' integer-valued inputs), otherwise equal up to the fp32
    accumulation order (tl.dot reduces over D in its own order, and at <= 4 rows the
    few-query kernel sums the heads as a tree). A NaN q . k propagates here, as through
    DeepSeek's relu_, while the Triton scorers' tl.maximum(logits, 0.0) returns the non-NaN
    operand and score that head 0; a NaN head weight propagates in both.
    """
    acc_dtype = torch.float64 if score_dtype == torch.float64 else torch.float32
    R, H, _ = q.shape
    W = keys.shape[0]
    width = W if width is None else width
    acc = torch.zeros((R, width), dtype = acc_dtype, device = q.device)
    if W:
        kt = keys.to(acc_dtype).t()
        qf = q.to(acc_dtype)
        wf = w.to(acc_dtype)
        a = acc[:, :W]
        for h in range(H):
            a += (qf[:, h] @ kt).clamp_min_(0.0) * wf[:, h : h + 1]
        if scale != 1.0:
            a *= scale
        if score_dtype == torch.float16:
            a.copy_(a.half())
    r = torch.arange(R, device = q.device, dtype = torch.long)
    vis = ((q_pos0 + r + 1) // m).clamp(min = 0, max = bound)
    col = torch.arange(width, device = q.device)
    acc.masked_fill_(col[None, :] >= vis[:, None], _NEG_INF)
    return acc


def _cand_flags(cand_rows: torch.Tensor, nb: int, device) -> torch.Tensor:
    """(rows, B) block ids -> (rows, nb + 1) bool keep flags; -1 and out-of-range ids land in slot nb."""
    c = cand_rows.to(device).long()
    c = torch.where((c < 0) | (c >= nb), torch.full_like(c, nb), c)
    flags = torch.zeros((c.shape[0], nb + 1), dtype = torch.bool, device = device)
    flags.scatter_(1, c, True)
    return flags


def _pin_newest(blk: torch.Tensor, vis: torch.Tensor, block: int):
    """
    Pin each row's newest visible block to +inf (rows with nothing visible get no pin).
    A gather/scatter, because nonzero() would synchronize the host at every
    candidate-source slab (layer 20 of every decode step).
    A row with nothing visible writes its block 0 back unchanged.
    """
    last = (vis - 1).div(block, rounding_mode = "floor").clamp_min(0).unsqueeze(1)
    cur = blk.gather(1, last)
    blk.scatter_(1, last, torch.where((vis > 0).unsqueeze(1), torch.full_like(cur, float("inf")), cur))


def _check_args(q_idx, wts, idx_pool, bt_row, epp, ec, cand_in, fn):
    """
    Shape and device checks for what the kernels cannot see: a wrong shape here does not
    fail, it selects from the wrong data. A one-row cand_in broadcasts over every query
    row; a page table shorter than ec makes the scorer read past it into whatever
    follows; a batch (B, npr) table flattens into one long row, so a sequence whose own
    row is too short reads the next sequence's pages; a paged pool whose pages are not
    epp entries long is addressed at the wrong rows; a pool on another device is
    dereferenced as if it were local. None of these checks reads device data.
    """
    seq = q_idx.shape[0]
    if q_idx.dim() != 3 or tuple(wts.shape) != tuple(q_idx.shape[:2]):
        raise ValueError(f"{fn}: q_idx must be (seq, H, D) and wts (seq, H); got "
                         f"{tuple(q_idx.shape)} and {tuple(wts.shape)}")
    if idx_pool.shape[-1] != q_idx.shape[-1]:
        raise ValueError(f"{fn}: pool head dim {idx_pool.shape[-1]} != query head dim {q_idx.shape[-1]}")
    if idx_pool.device != q_idx.device or wts.device != q_idx.device:
        raise ValueError(f"{fn}: q_idx ({q_idx.device}), wts ({wts.device}) and idx_pool "
                         f"({idx_pool.device}) must share a device; mirror the pool first")
    if cand_in is not None and (cand_in.dim() != 2 or cand_in.shape[0] != seq):
        raise ValueError(f"{fn}: cand_in must be (seq = {seq}, n_blocks); got {tuple(cand_in.shape)}")
    if bt_row is not None:
        if epp <= 0:
            raise ValueError(f"{fn}: paged pool needs epp")
        if not (bt_row.dim() == 1 or (bt_row.dim() == 2 and bt_row.shape[0] == 1)):
            raise ValueError(f"{fn}: bt_row must be one sequence's page table, (npr,) or (1, npr); "
                             f"got {tuple(bt_row.shape)}; slice the batch's table per sequence")
        if idx_pool.dim() == 3 and idx_pool.shape[1] != epp:
            raise ValueError(f"{fn}: paged pool {tuple(idx_pool.shape)} has {idx_pool.shape[1]} "
                             f"entries per page, epp = {epp}")
        if bt_row.numel() * epp < ec:
            raise ValueError(f"{fn}: page table covers {bt_row.numel()} pages * {epp} entries, "
                             f"fewer than ec = {ec}")
    elif idx_pool.numel() // max(idx_pool.shape[-1], 1) < ec:
        raise ValueError(f"{fn}: dense pool has fewer than ec = {ec} rows")


def _drop_unkept(ids: torch.Tensor, blk: torch.Tensor) -> torch.Tensor:
    """-1 for selected blocks whose score is -inf or NaN (DeepSeek: keep = value > -inf)."""
    v = blk.gather(1, ids.clamp_min(0).long())
    bad = (ids < 0) | ~(v > _NEG_INF)
    return torch.where(bad, torch.full_like(ids, -1), ids)


def select_topk(
    q_idx: torch.Tensor,
    wts: torch.Tensor,
    idx_pool: torch.Tensor,
    *,
    bt_row: torch.Tensor | None = None,
    epp: int = 0,
    pos0: int,
    m: int,
    ec: int,
    topk: int = TOPK,
    cand_in: torch.Tensor | None = None,
    want_cand: bool = False,
    block: int = BLOCK,
    n_blocks: int = N_BLOCKS,
    slab_rows: int = SLAB_ROWS,
    tile_entries: int = TILE_ENTRIES,
    backend: str = "auto",
    scale: float = 1.0,
    score_dtype: torch.dtype = torch.float16,
) -> SelectResult:
    """
    Top-k index selection for one sequence, tiled, with the V4.1 candidate stage.

    Device memory: at most transient_bytes() per call, freed on return. The autosplit does
    not reserve the part that grows with the context: its measuring forward runs at
    position 0, where a chunk sees only its own entries (and none are selected while
    ec <= topk). At long context leave up to transient_bytes(chunk, max_context,
    want_cand = True) of headroom on each device that holds an index source: 157 MiB for
    a 2048-row chunk at 1M entries, of which the same chunk at position 0 allocates 90 MiB.
    The unreserved difference, about 67 MiB, fits inside the autosplit's default margin
    (EXL3_AUTOSPLIT_MARGIN_MB, 256 MiB).

    :param q_idx:     (seq, H, D) fp16 roped indexer queries of rows pos0 .. pos0 + seq - 1
    :param wts:       (seq, H) head weights; scores are multiplied by `scale` on top
                      (default 1: the caller folded 128^-0.5 * 32^-0.5 in)
    :param idx_pool:  index keys, roped: paged (P, epp, D) / flat (P * epp, D) with bt_row,
                      or dense (>= ec, D) without
    :param bt_row:    (1, npr) or (npr,) int32 page table of this sequence (paged mode)
    :param epp:       pool entries per page (paged mode)
    :param pos0:      absolute position of query row 0
    :param m:         compress rate (entry j covers tokens [j*m, (j+1)*m))
    :param ec:        valid pool entries (row r sees min((pos0 + r + 1) // m, ec))
    :param cand_in:   (seq, B) int32 candidate blocks from the source layer, or None
    :param want_cand: publish this layer's candidate blocks (the source layer)
    :param tile_entries: pool entries per scored tile; rounded up to lcm(epp, block). A call
                      with fewer rows than slab_rows widens it (powers of two, up to the pool)
                      within the same slab_rows * tile_entries budget, so decode is one pass.
    :param backend:   'auto' | 'ext' | 'torch'
    :param score_dtype: torch backend only: precision scores are rounded to before selection
    :return: SelectResult(indices (seq, ceil32(topk)) int32 | None, k_len, cand | None).
             indices is None (dense: every entry selected) when ec <= topk; so is cand,
             since nothing is selected and the mask would be a no-op.
    """
    if want_cand and cand_in is not None:
        raise ValueError("select_topk: a layer either publishes candidates or consumes them, not both")
    if ablated("no_candidates"):
        cand_in, want_cand = None, False
    if ec <= topk:
        return SelectResult(None, 0, None)

    seq = q_idx.shape[0]
    device = q_idx.device
    if backend == "auto":
        backend = "ext" if q_idx.is_cuda else "torch"
    if backend not in ("ext", "torch"):
        raise ValueError(f"select_topk: unknown backend {backend!r}")
    if backend == "ext" and not q_idx.is_cuda:
        raise ValueError("select_topk: the ext backend needs CUDA tensors")
    _check_args(q_idx, wts, idx_pool, bt_row, epp, ec, cand_in, "select_topk")
    paged = bt_row is not None
    bt_flat = bt_row.reshape(-1) if paged else None

    quantum = _tile_quantum(epp if paged else 1, block)
    slab = max(1, slab_rows)
    kp = _ceil(topk, 32)
    rows_max = min(slab, seq)
    tile = _effective_tile(rows_max, ec, slab, tile_entries, quantum)
    # Refused before any backend runs: the defaults stay at 2^24 elements (256 x 65,536, or
    # a widened decode tile within the same budget), but a larger slab x tile budget would
    # wrap the kernels' int32 row offsets and score into another allocation
    s_stride = _ceil(tile, 128)
    nb_stride = _ceil(-(-ec // block), 8)
    if rows_max * s_stride >= INT32_LIMIT or (want_cand and rows_max * nb_stride >= INT32_LIMIT):
        raise ValueError(
            f"select_topk: a slab of {rows_max} rows x {s_stride} score columns (or x {nb_stride} "
            f"block maxima) reaches the kernels' int32 row addressing; lower slab_rows or "
            f"tile_entries")

    indices = torch.empty((seq, kp), dtype = torch.int32, device = device)
    cand = torch.full((seq, n_blocks), -1, dtype = torch.int32, device = device) if want_cand else None

    run = _ExtSlabs if backend == "ext" else _TorchSlabs
    runner = run(q_idx, wts, idx_pool, bt_flat, epp if paged else 0, pos0, m, ec, topk, kp,
                 block, n_blocks, tile, rows_max, want_cand, cand_in is not None, scale, score_dtype)
    for r0 in range(0, seq, slab):
        r1 = min(r0 + slab, seq)
        runner.slab(r0, r1, indices[r0 : r1], None if cand is None else cand[r0 : r1],
                    None if cand_in is None else cand_in[r0 : r1])
    return SelectResult(indices, topk, cand)


class _Slabs:
    """Shared slab/tile walk; subclasses supply scoring and top-k primitives."""

    def __init__(self, q, w, pool, bt_flat, epp, pos0, m, ec, topk, kp, block, n_blocks,
                 tile, rows_max, want_cand, has_cand_in, scale, score_dtype):
        self.q, self.w, self.pool, self.bt, self.epp = q, w, pool, bt_flat, epp
        self.pos0, self.m, self.ec, self.topk, self.kp = pos0, m, ec, topk, kp
        self.block, self.n_blocks, self.tile, self.rows_max = block, n_blocks, tile, rows_max
        self.want_cand, self.has_cand_in = want_cand, has_cand_in
        self.scale, self.score_dtype = scale, score_dtype
        self.device = q.device

    def slab(self, r0, r1, out, cand_out, cand_rows):
        rows = r1 - r0
        # Widest row of the slab is the last one; entries past it are -inf for every row here
        T_slab = min(self.ec, max(0, (self.pos0 + r1) // self.m))
        if T_slab <= 0:
            out.fill_(-1)
            return
        vis = visible_counts(rows, self.pos0 + r0, self.m, self.ec, self.device)
        nb = -(-T_slab // self.block)
        blk = self.new_block_buffer(rows, nb) if self.want_cand else None
        flags = _cand_flags(cand_rows, nb, self.device) if self.has_cand_in else None
        tiles = list(range(0, T_slab, self.tile))
        self.begin(rows, len(tiles))
        for n, t0 in enumerate(tiles):
            t1 = min(t0 + self.tile, T_slab)
            W8 = _ceil(t1 - t0, self.block)
            sc = self.score(r0, rows, t0, t1, W8)                       # (rows, W8), pad -inf
            b0, nbt = t0 // self.block, W8 // self.block
            sv = sc.view(rows, nbt, self.block)
            if blk is not None:
                # Candidate blocks from the UNMASKED scores (the source's own top-k is unmasked)
                blk[:, b0 : b0 + nbt] = sv.amax(dim = -1).to(blk.dtype)
            if flags is not None:
                sv.masked_fill_(~flags[:, b0 : b0 + nbt, None], _NEG_INF)
            self.select_tile(sc, t0, n, len(tiles), out)
        self.finish(out)
        if blk is not None:
            _pin_newest(blk, vis, self.block)
            self.select_blocks(blk, cand_out)


class _TorchSlabs(_Slabs):
    """Plain torch on any device. Running set kept as (values, ids), merged by stable sort."""

    def new_block_buffer(self, rows, nb):
        dt = torch.float64 if self.score_dtype == torch.float64 else torch.float32
        return torch.empty((rows, nb), dtype = dt, device = self.device)

    def begin(self, rows, n_tiles):
        self.run_v = self.run_i = None

    def score(self, r0, rows, t0, t1, W8):
        keys = _keys_for(self.pool, self.bt, self.epp, t0, t1)
        return _torch_scores(self.q[r0 : r0 + rows], self.w[r0 : r0 + rows], keys,
                             self.pos0 + r0 - t0 * self.m, self.m, t1 - t0, self.scale,
                             self.score_dtype, width = W8)

    def select_tile(self, sc, t0, n, n_tiles, out):
        i, v = topk_total_order(sc, self.topk)
        i = i + t0
        if self.run_v is not None:
            # Running set first: its ids are all lower, so the stable sort keeps index order
            v = torch.cat([self.run_v, v], dim = 1)
            i = torch.cat([self.run_i, i], dim = 1)
            j, v = topk_total_order(v, self.topk)
            i = i.gather(1, j)
        self.run_v, self.run_i = v, i

    def finish(self, out):
        ids = torch.where(self.run_v > _NEG_INF, self.run_i,
                          torch.full_like(self.run_i, -1))           # -inf: invisible or masked
        # NaN compares False above but is a real selection (torch.topk ranks it first)
        ids = torch.where(torch.isnan(self.run_v), self.run_i, ids)
        res = torch.full((out.shape[0], self.kp), -1, dtype = torch.long, device = self.device)
        res[:, : ids.shape[1]] = ids
        out.copy_(sort_ids_ascending(res).to(torch.int32))

    def select_blocks(self, blk, cand_out):
        i, _ = topk_total_order(blk, self.n_blocks)
        ids = _drop_unkept(i, blk)
        res = torch.full((cand_out.shape[0], self.n_blocks), -1, dtype = torch.long, device = self.device)
        res[:, : ids.shape[1]] = ids
        cand_out.copy_(sort_ids_ascending(res).to(torch.int32))


class _ExtSlabs(_Slabs):
    """Triton scorer into one fixed fp16 tile buffer, CUDA top-k kernels for selection."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from .attention_fn.dsa_triton import dsa_indexer_scores
        from ..ext import exllamav3_ext as ext
        self.ext = ext
        self.scores_fn = dsa_indexer_scores
        dev = self.device
        # Fixed row stride for every tile: it is a constexpr of the scoring kernel, so a
        # stride that followed the visible length would recompile at each new 128 entries
        self.s_stride = _ceil(self.tile, 128)
        self.backing = torch.empty((self.rows_max * self.s_stride,), dtype = torch.half, device = dev)
        # Two ping-pong merge workspaces: slot 0 = running set, slot 1 = the new tile
        self.ws = None
        self.q = self.q if self.q.dtype == torch.half else self.q.half()
        self.q = self.q.contiguous()
        self.w = self.w.contiguous()
        if self.bt is not None:
            self.bt = self.bt.to(device = dev, dtype = torch.int32).contiguous()
        self.pool_flat = self.pool.reshape(-1, self.pool.shape[-1])
        self.nb_stride = None

    def new_block_buffer(self, rows, nb):
        if self.nb_stride is None:
            T_max = self.ec
            self.nb_stride = _ceil(-(-T_max // self.block), 8)
            self.blk_backing = torch.empty((self.rows_max * self.nb_stride,), dtype = torch.half,
                                           device = self.device)
        return self.blk_backing[: rows * self.nb_stride].view(rows, self.nb_stride)[:, :nb]

    def begin(self, rows, n_tiles):
        self.rows = rows
        if n_tiles > 1:
            if self.ws is None:
                R, kp, dev = self.rows_max, self.kp, self.device
                self.ws = [(torch.empty((R, 2, kp), dtype = torch.int32, device = dev),
                            torch.empty((R, 2, kp), dtype = torch.half, device = dev),
                            torch.zeros((R, 2), dtype = torch.int32, device = dev)) for _ in range(2)]
            self.ws[0][2][:rows, 0].zero_()
            self.cur = 0

    def score(self, r0, rows, t0, t1, W8):
        sc = self.backing[: rows * self.s_stride].view(rows, self.s_stride)
        q = self.q[r0 : r0 + rows]
        w = self.w[r0 : r0 + rows]
        qp = self.pos0 + r0 - t0 * self.m
        # EXL3_STABLE_ARITHMETIC=1: score every row count with the query-tiled kernel, so a
        # decoded row selects exactly what the same row selects inside a prefill chunk (the
        # few-query kernel that decode would take reduces over the heads in another order)
        few_query = not STABLE_ARITHMETIC
        if self.bt is None:
            self.scores_fn(q, w, self.pool_flat[t0 : t1], qp, self.m, t1 - t0,
                           scores = sc, scale = self.scale, few_query = few_query)
        else:
            e = self.epp
            bt = self.bt[t0 // e : -(-t1 // e)]
            self.scores_fn(q, w, self.pool_flat, qp, self.m, t1 - t0,
                           scores = sc, block_table = bt, epp = e, scale = self.scale,
                           few_query = few_query)
        W = t1 - t0
        if W8 > W:
            sc[:, W : W8].fill_(_NEG_INF)
        return sc[:, :W8]

    def select_tile(self, sc, t0, n, n_tiles, out):
        ext, rows, k = self.ext, self.rows, self.topk
        W = sc.shape[1]
        if n_tiles == 1:
            ext.dsa_topk(sc, out, min(k, W), None, 0)
            return
        w_idx, w_scr, w_cnt = (t[:rows] for t in self.ws[self.cur])
        ext.dsa_topk_tile(sc, w_idx, w_scr, w_cnt, 1, min(k, W), t0)
        if n == n_tiles - 1:
            ext.dsa_topk_merge_tiles(w_idx, w_scr, w_cnt, out, None, None, k)
        else:
            n_idx, n_scr, n_cnt = (t[:rows] for t in self.ws[self.cur ^ 1])
            ext.dsa_topk_merge_tiles(w_idx, w_scr, w_cnt, n_idx[:, 0], n_scr[:, 0], n_cnt[:, 0], k)
            self.cur ^= 1

    def finish(self, out):
        # The kernels emit (score > v*) then (score == v*) runs; the reference sorts
        out.copy_(sort_ids_ascending(out))

    def select_blocks(self, blk, cand_out):
        k = min(self.n_blocks, blk.shape[1])
        self.ext.dsa_topk(blk, cand_out, k, None, 0)
        cand_out.copy_(sort_ids_ascending(_drop_unkept(cand_out, blk)))


def transient_bytes(seq: int, ec: int, slab_rows: int = SLAB_ROWS, tile_entries: int = TILE_ENTRIES,
                    want_cand: bool = False, *, cand_in: bool = False, topk: int = TOPK,
                    block: int = BLOCK, n_blocks: int = N_BLOCKS, epp: int = 256) -> int:
    """
    Upper bound on the device memory select_topk (ext backend) allocates in one call,
    outputs included. Every term is an allocation in the code above; the largest ones
    scale with slab_rows * tile_entries (the score tile) and slab_rows * ec / block (the
    candidate block buffer), never with seq * ec. The autosplit reserves only what a chunk
    at position 0 allocates (see select_topk); leave the rest of this bound as headroom
    (157 MiB for a 2048-row chunk at 1M entries with want_cand, 90 MiB at position 0).
    """
    if ec <= topk:
        return 0
    R = min(max(slab_rows, 1), seq)
    kp = _ceil(topk, 32)
    tile = _effective_tile(R, ec, max(slab_rows, 1), tile_entries, _tile_quantum(epp, block))
    n_tiles = -(-ec // tile)
    nb = -(-ec // block)
    s_stride = _ceil(tile, 128)
    b = 0
    b += seq * kp * 4                                   # indices (output)
    b += R * s_stride * 2                               # fp16 score tile
    if n_tiles > 1:
        b += 2 * (R * 2 * kp * (4 + 2) + R * 2 * 4)     # merge workspaces
    b += R * kp * (4 + 4 + 8)                           # ascending sort: key, values, int64 order
    b += R * (-(-tile // block)) * 2                    # per-tile block view temporaries
    split_ws = 16 * 32 * 4096 * 6 + 16 * 32 * 4         # dsa_topk's per-device split workspace (one-time)
    b += split_ws
    if want_cand:
        nb_stride = _ceil(nb, 8)
        b += seq * n_blocks * 4                         # cand (output)
        b += R * nb_stride * 2                          # fp16 block scores
        b += R * (-(-tile // block)) * 2                # amax temporary
        b += R * n_blocks * (8 + 2 + 1 + 4)             # gather ids/values, keep mask, where
        b += R * n_blocks * (4 + 4 + 8)                 # ascending sort
        b += R * 8 * 3                                  # pin
    if cand_in:
        b += R * (nb + 1)                               # keep flags
        b += R * n_blocks * 8 * 3                       # int64 ids, clamp, where
        b += R * (-(-tile // block))                    # ~flags per tile
    return b


def select_topk_reference(
    q_idx: torch.Tensor,
    wts: torch.Tensor,
    idx_pool: torch.Tensor,
    *,
    bt_row: torch.Tensor | None = None,
    epp: int = 0,
    pos0: int,
    m: int,
    ec: int,
    topk: int = TOPK,
    cand_in: torch.Tensor | None = None,
    want_cand: bool = False,
    block: int = BLOCK,
    n_blocks: int = N_BLOCKS,
    scale: float = 1.0,
    score_dtype: torch.dtype = torch.float16,
) -> SelectResult:
    """
    Untiled torch selection, for tests: one (seq, ec) score matrix, the reference
    candidate functions from architecture/dsv41/candidates.py, one total-order top-k.
    Same arguments and result as select_topk.
    """
    if want_cand and cand_in is not None:
        raise ValueError("select_topk_reference: publish or consume candidates, not both")
    if ablated("no_candidates"):
        cand_in, want_cand = None, False
    if ec <= topk:
        return SelectResult(None, 0, None)
    seq = q_idx.shape[0]
    _check_args(q_idx, wts, idx_pool, bt_row, epp, ec, cand_in, "select_topk_reference")
    bt_flat = bt_row.reshape(-1) if bt_row is not None else None
    keys = _keys_for(idx_pool, bt_flat, epp, 0, ec)
    scores = _torch_scores(q_idx, wts, keys, pos0, m, ec, scale, score_dtype)
    vis = visible_counts(seq, pos0, m, ec, q_idx.device)
    cand = publish_candidate_blocks(scores, vis, block, n_blocks) if want_cand else None
    if cand_in is not None:
        scores = mask_scores_to_candidates(scores, cand_in, block)
    i, v = topk_total_order(scores, topk)
    ids = torch.where((v > _NEG_INF) | torch.isnan(v), i, torch.full_like(i, -1))
    kp = _ceil(topk, 32)
    res = torch.full((seq, kp), -1, dtype = torch.long, device = q_idx.device)
    res[:, : ids.shape[1]] = ids
    return SelectResult(sort_ids_ascending(res).to(torch.int32), topk, cand)
