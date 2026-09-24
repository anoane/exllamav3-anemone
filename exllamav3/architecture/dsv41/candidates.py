# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
Two-level candidate filtering: the untiled reference.

Follows DeepSeek's reference inference/model.py (Indexer.forward and
select_candidate_blocks) and vLLM's kernels
(vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:
_block_scores_kernel, select_candidate_blocks, apply_candidate_mask;
Apache-2.0, see NOTICE). The two agree; where a detail is only implicit in one
of them, the comment says which one fixes it.

V4 has one selection stage: each indexer scores all compressed positions and
keeps index_topk. V4.1 adds a coarse stage in front of it. The indexer on
candidate_source_layer publishes, per query row, the candidate_topk_blocks
best blocks of candidate_block_size consecutive compressed positions.
Indexers on layers strictly above it set their scores to -inf outside those
blocks before their own top-k. The source layer's own top-k is unmasked.

For DeepSeek-V4.1-Flash: layer 20 publishes 2048 blocks of 8 entries (16,384
candidate positions of its rate-1 pool) and the index sources 24/28/32/36
narrow to index_topk = 512 inside them. Until a row can see more than 2048
blocks (16,384 entries), every visible block is published and the mask
changes nothing.

Block scoring, as both references do it:
  * block b covers positions [b * block, (b + 1) * block)
  * score = max over the block's VISIBLE positions (invisible ones are -inf),
    NaN-propagating (torch amax; vLLM's PropagateNan.ALL)
  * the block holding the row's newest visible position,
    (vis_end - 1) // block, is pinned to +inf: it is only partly filled and
    would otherwise lose to older full blocks. No pin when vis_end == 0.
  * top n_blocks by score (torch.topk semantics: NaN ranks above +inf); a
    selected block whose score is -inf (not reachable) or NaN is dropped to
    -1. DeepSeek: keep = top.values > -inf; vLLM: where(value > -inf, idx, -1).
    A NaN block therefore still occupies one of the n_blocks slots.

Ties: both references use torch.topk, whose order among equal scores is
unspecified. Here ties are broken by ascending index -- the total order of
exllamav3's dsa_topk kernels (score descending, index ascending) -- so the
reference, the tiled torch path and the kernel path select identical sets for
every score except signed zeros and NaNs. The kernels rank by the fp16 bit
pattern (dsa_topk.cu, topk_key_u): +0 above -0, and a NaN with the sign bit
set below -inf. The torch path (topk_total_order) treats +0 and -0 as equal,
breaking the tie by index, and ranks every NaN first. The two differ only when
the top-k cut falls among such values.

Output: [T, n_blocks] int32 block ids, ascending, -1 padded at the end.
"""

from __future__ import annotations

import torch

INT_MAX = 2 ** 31 - 1


def visible_counts(rows: int, pos0: int, m: int, ec: int, device = None) -> torch.Tensor:
    """
    Compressed entries each query row can see: min((pos0 + r + 1) // m, ec).

    DeepSeek's prefill (start_pos 0) is arange(1, seqlen + 1) // ratio, its
    decode end_pos // ratio; a chunk starting at pos0 is the same rows of the
    one-shot prefill. An entry becomes visible once the query has passed the
    last token of its group.
    """
    r = torch.arange(rows, device = device, dtype = torch.long)
    return ((pos0 + r + 1) // m).clamp(min = 0, max = ec)


def topk_total_order(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top k per row under (score descending, index ascending), NaN ranked above
    +inf as torch.topk does. Returns (idx int64 [R, k'], val [R, k']) with
    k' = min(k, width) in selection order; the caller decides what -inf / NaN
    selections mean. A stable descending sort is the whole trick: equal scores
    keep their index order.
    """
    k = min(k, scores.shape[-1])
    val, idx = torch.sort(scores, dim = -1, descending = True, stable = True)
    return idx[..., :k], val[..., :k]


def sort_ids_ascending(ids: torch.Tensor) -> torch.Tensor:
    """Ascending ids with every -1 moved to the end."""
    key = torch.where(ids < 0, torch.full_like(ids, INT_MAX), ids)
    key = key.sort(dim = -1).values
    return torch.where(key == INT_MAX, torch.full_like(key, -1), key)


def block_scores(scores: torch.Tensor, vis_end: torch.Tensor, block: int) -> torch.Tensor:
    """
    [R, P] per-position scores -> [R, ceil(P / block)] block scores: NaN-
    propagating max over each block's visible positions, the newest visible
    block pinned to +inf. vLLM's _block_scores_kernel (start 0), DeepSeek's
    F.pad(-inf) + amax + masked_fill(arange == last, inf).
    """
    R, P = scores.shape
    vis_end = vis_end.to(scores.device).long().view(R)
    col = torch.arange(P, device = scores.device)
    s = scores.masked_fill(col[None, :] >= vis_end[:, None], float("-inf"))
    nb = -(-P // block)
    pad = nb * block - P
    if pad:
        s = torch.cat([s, s.new_full((R, pad), float("-inf"))], dim = 1)
    b = s.view(R, nb, block).amax(dim = -1)
    last = (vis_end - 1).div(block, rounding_mode = "floor")
    pin = (vis_end > 0) & (last < nb)
    rows = torch.nonzero(pin).view(-1)
    b[rows, last[rows]] = float("inf")
    return b


def publish_candidate_blocks(scores: torch.Tensor, vis_end: torch.Tensor, block: int,
                             n_blocks: int) -> torch.Tensor:
    """
    Coarse stage, run on the candidate source layer.

    :param scores:   [R, P] per-position scores of this row's query (any float
                     dtype); positions >= vis_end are ignored
    :param vis_end:  [R] visible entry count per row
    :return:         [R, n_blocks] int32 block ids, ascending, -1 padded
    """
    R, P = scores.shape
    out = torch.full((R, n_blocks), -1, dtype = torch.int32, device = scores.device)
    if P == 0 or R == 0:
        return out
    b = block_scores(scores, vis_end, block)
    idx, val = topk_total_order(b, n_blocks)
    keep = val > float("-inf")                        # False for -inf and for NaN
    ids = torch.where(keep, idx, torch.full_like(idx, -1))
    out[:, :ids.shape[1]] = sort_ids_ascending(ids).to(torch.int32)
    return out


def candidate_keep(blocks: torch.Tensor, block: int, num_positions: int) -> torch.Tensor:
    """[R, B] block ids (-1 ignored) -> [R, num_positions] bool, True inside a candidate block."""
    R = blocks.shape[0]
    nb = -(-num_positions // block)
    b = blocks.long()
    b = torch.where((b < 0) | (b >= nb), torch.full_like(b, nb), b)
    flags = torch.zeros((R, nb + 1), dtype = torch.bool, device = blocks.device)
    flags.scatter_(1, b, True)
    return flags[:, :nb].repeat_interleave(block, dim = 1)[:, :num_positions]


def mask_scores_to_candidates(scores: torch.Tensor, blocks: torch.Tensor,
                              block: int) -> torch.Tensor:
    """
    Fine stage, run on every indexer above the candidate source.

    Positions outside the published blocks become -inf, so the layer's own
    top-k can only choose inside them. Causal masking is the scorer's job and
    already done (vLLM's apply_candidate_mask re-applies it; a no-op here).

    :param scores: [R, P] float
    :param blocks: [R, B] int32 block ids, -1 ignored
    :return:       [R, P] masked copy
    """
    keep = candidate_keep(blocks.to(scores.device), block, scores.shape[1])
    return scores.masked_fill(~keep, float("-inf"))
