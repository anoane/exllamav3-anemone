"""
DeepSeek-V4.1 cached (generation) path: the pieces that do not need a module.

DSV41Attention._forward_cached (modules/dsv41.py) is written in terms of these so that the
position arithmetic, the compressor carry and the sliding-ring bookkeeping can be checked on
the CPU against brute force and against DeepSeek's reference (tests/test_dsv41_cached_.py,
tests/test_dsv41_compressor_.py). Nothing here touches the extension at import time.

Geometry, for one kv source at compression rate m (1 or 2):

  * Entry e covers tokens [e*m, (e+1)*m) and closes on its last token -- the (p + 1) % m == 0
    rule. A chunk [pos0, pos0 + seq) therefore emits exactly the entries
    [pos0 // m, (pos0 + seq) // m): at rate 2 a chunk that starts on an odd position first
    closes the group opened by the previous chunk's last row.
  * Entry e rotates at the FIRST token of its group, e * m (DeepSeek: freqs_cis[j * ratio] in
    prefill, freqs_cis[start_pos + 1 - ratio] in decode).
  * Entry e lives at pool row bt[e // epp] * epp + e % epp, epp = PAGE_SIZE // m: token page i
    holds the entries its own tokens produce, so pools share the token page table.

Selection (modules/dsv41_select.py), the per-forward device memo (cache/dsv41.py, DeviceMemo)
and the cross-device pool replica of an explicit placement's split (cache/dsv41_replica.py) are
separate modules.
"""

from __future__ import annotations

import torch

from ..constants import PAGE_SIZE
from ..util.device_copy import to_device


# ---------------------------------------------------------------------------------------------
# Entry geometry

def emission_range(pos0: int, seq: int, m: int) -> tuple[int, int]:
    """Entries a chunk [pos0, pos0 + seq) closes at rate m: [pos0 // m, (pos0 + seq) // m)."""
    return pos0 // m, (pos0 + seq) // m


def entry_rows(bt_row: torch.Tensor, e: torch.Tensor, epp: int) -> torch.Tensor:
    """Flat pool rows of entries e through one job's block-table row (1, P) or (P,)."""
    bt = to_device(bt_row.reshape(-1), e.device).long()
    return bt[e // epp] * epp + e % epp


def rope_positions(e: torch.Tensor, m: int) -> torch.Tensor:
    """Rotation position of entry e: the first token of its group."""
    return e * m


def rope_gptj_torch(x: torch.Tensor, inv_freq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    GPT-J (interleaved-pair) rotation of the last dim of x at per-row positions, returned in
    x's dtype. x (n, ..., rd), positions (n,). CPU/GPU reference for ext.rope and DeepSeek's
    apply_rotary_emb (pairs (2k, 2k+1) as complex numbers). The angle position * inv_freq is
    formed in fp32, as both of those form it (a 60K position has an fp32 angle ulp of ~4e-3);
    the rotation itself runs in float64.
    """
    n = x.shape[0]
    xd = x.double()
    ang = (positions.float().reshape(n, *([1] * (x.dim() - 2)), 1) *
           inv_freq.float().to(x.device)).double()
    c, s = torch.cos(ang), torch.sin(ang)
    x0, x1 = xd[..., 0::2], xd[..., 1::2]
    out = torch.empty_like(xd)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out.to(x.dtype)


# ---------------------------------------------------------------------------------------------
# Compressor carry

class CompressCarry:
    """
    Rate-m pooling across chunk boundaries with a position-indexed carry ring.

    carry is ONE slot's (R, 2 * hd) fp32 ring of raw [kv | score] rows, row = abs position % R
    (DSV41LayerState.comp_carry[slot], R = PAGE_SIZE + m). A chunk starting mid-group
    (pos0 % m != 0) prepends the pending rows it finds there; every row of the chunk is then
    written back at its own position. Because the ring is indexed by position, a rewind needs
    no bookkeeping: re-advancing from any position whose predecessor row is still in the ring
    reads the same pending row it read the first time.

    The pooling itself is architecture/dsv41/compressor.py's group_pool + rms_norm, which are
    local to each group and row. CPU regressions compare chunked and one-shot latents
    bitwise (tests/test_dsv41_cached_.py, against compress_reference). GPU reduction order
    can depend on row count; the GPU tests use explicit numerical tolerances rather
    than assuming bitwise equality for all schedules or input ranges, except under
    EXL3_STABLE_ARITHMETIC=1, where rms_norm reduces each row alone and chunked latents
    are bitwise equal (tests/test_dsv41_stable_norm_gpu_.py). With per_row (EXL3_EXACT_ROWS)
    every closed entry is pooled and normalized by a call of its own, the call a one-row step
    that closes it makes.
    """

    @staticmethod
    def step(carry, kv, score, pos0: int, m: int, norm_weight = None, eps: float = 1e-6,
             per_row: bool = False):
        """
        carry   (R, 2 * hd) fp32 ring of this slot, or None at rate 1
        kv      (seq, hd) raw compressor kv rows of this chunk
        score   (seq, hd) raw gate rows (None at rate 1)
        per_row pool and normalize one closed entry per call (the carry ring is read and
                written as without it)
        returns (latents (n, hd) fp32 pre-RoPE, first_entry)
                n = (pos0 + seq) // m - pos0 // m, first_entry = pos0 // m
        """
        from ..architecture.dsv41.compressor import group_pool, rms_norm
        seq, hd = kv.shape
        first = pos0 // m
        if m == 1:
            # a softmax over one row is 1: every token closes its own group, no state
            if per_row and seq > 1:
                return torch.cat([rms_norm(kv[i:i + 1], norm_weight, eps) for i in range(seq)], dim = 0), first
            return rms_norm(kv, norm_weight, eps), first
        assert carry is not None and score is not None and score.shape == kv.shape
        R = carry.shape[0]
        assert carry.shape[1] == 2 * hd and R >= m, f"carry ring {tuple(carry.shape)} for hd {hd}"
        dev = carry.device
        kvf, scf = kv.float(), score.float()
        pend = pos0 % m
        if pend:
            # read BEFORE this chunk's rows are written: a chunk of R or more rows wraps
            src = torch.arange(pos0 - pend, pos0, device = dev) % R
            c = carry.index_select(0, src)
            kvf = torch.cat([c[:, :hd], kvf], dim = 0)
            scf = torch.cat([c[:, hd:], scf], dim = 0)
        j0 = max(0, seq - R)
        dst = torch.arange(pos0 + j0, pos0 + seq, device = dev) % R
        carry.index_copy_(0, dst, torch.cat([kv[j0:].float(), score[j0:].float()], dim = 1))
        closed = kvf.shape[0] // m * m
        if closed == 0:
            return kvf.new_zeros((0, hd)), first
        if per_row and closed > m:
            # a one-row step that closes a group pools its (1, m, hd) rows, the pending ones from
            # the carry ring: the same FP32 values as the rows of this call
            return torch.cat([
                rms_norm(group_pool(kvf[g:g + m].view(1, m, hd), scf[g:g + m].view(1, m, hd)), norm_weight, eps)
                for g in range(0, closed, m)], dim = 0), first
        lat = group_pool(kvf[:closed].view(-1, m, hd), scf[:closed].view(-1, m, hd))
        return rms_norm(lat, norm_weight, eps), first


# ---------------------------------------------------------------------------------------------
# Sliding-window ring

def ring_update(ring: torch.Tensor, kv: torch.Tensor, pos0: int, window_beg: int, w: int) -> int | None:
    """
    Keep the trailing window resident after a chunk; a verbatim copy of V4's post-attention
    ring logic (modules/dsv4.py, DSV4Attention._forward_cached_one), kept here so V4.1 does
    not depend on a V4 method body and so it can be simulated on the CPU.

    ring (ring_rows, D): row i holds absolute position window_beg + i. kv (seq, D): this
    chunk's rows. Three regimes: in-place append while the chunk fits; a page-granular shift
    for small appends near the end; a window rebase for chunks that overflow the ring. Returns
    the window_beg shift to record in the job state (rs.wshift, applied by post_advance), or
    None when window_beg stays put. Must run AFTER attention: the kernel reads the pre-update
    ring through ring_beg.
    """
    ring_rows = ring.shape[0]
    seq = kv.shape[0]
    offset = pos0 - window_beg
    pos_end = pos0 + seq
    if offset + seq <= ring_rows:
        ring[offset: offset + seq].copy_(kv)
        return None
    if seq < PAGE_SIZE:
        need = offset + seq - ring_rows
        shift = -(-need // PAGE_SIZE) * PAGE_SIZE
        ring[:ring_rows - shift].copy_(ring[shift:].clone())
        ring[offset - shift: offset - shift + seq].copy_(kv)
        return shift
    new_beg = max(pos_end - (w - 1), 0) // PAGE_SIZE * PAGE_SIZE
    n_keep = pos_end - new_beg
    n_from_kv = min(n_keep, seq)
    n_from_ring = n_keep - n_from_kv          # rows below pos0 still needed
    assert 0 < n_keep <= ring_rows
    if n_from_ring > 0:
        src0 = pos0 - n_from_ring - window_beg
        ring[:n_from_ring].copy_(ring[src0: src0 + n_from_ring].clone())
    ring[n_from_ring: n_keep].copy_(kv[seq - n_from_kv:])
    return new_beg - window_beg
