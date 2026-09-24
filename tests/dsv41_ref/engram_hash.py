# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
Engram n-gram hashing, reference implementation.

Derived from vLLM vllm/models/deepseek_v4_1/common/engram.py (Apache-2.0),
itself a port of DeepSeek's reference inference/engram.py. See NOTICE.

For every token position the model hashes the n-grams ending at that position
(sizes 2 .. max_ngram_size), each split over n_heads heads, into per-(size,
head) prime-sized bucket ranges. The accumulator is XOR, not a sum:

    rolling = 0
    for shift in 0 .. max_ngram_size-1:              # shift = lookback distance
        value    = compressed_id(token[p - shift])   # pad if out of range/dead
                                                     # (pad = token_map[pad_token_id])
        rolling ^= value * multipliers[layer][shift]
        if shift > 0:                                # an (shift+1)-gram is complete
            col    = (shift - 1) * n_heads + head
            index  = rolling % primes[layer][col] + offsets[layer][col]

``multipliers`` are odd and bounded so ``value * multiplier`` cannot overflow
int64; the primes are disjoint across (size, head) so bucket ranges never
collide. Both come from EngramLayout, which is verified against the
checkpoint's declared engram_num_embeddings.

This is the semantics, written for clarity and testability rather than speed.
The engine hashes with exllamav3/architecture/dsv41/engram_torch.py.
"""

from __future__ import annotations

import numpy as np


def engram_hash_reference(
    token_ids: np.ndarray,
    layout,
    token_map: np.ndarray,
    *,
    dead_mask: np.ndarray | None = None,
    dead_id: int = -1,
) -> np.ndarray:
    """
    Bucket index per (position, layer, hash column).

    :param token_ids: [T] int64 token ids for one contiguous sequence
    :param layout:    EngramLayout
    :param token_map: [vocab] int64, token id -> compressed vocab id
    :param dead_mask: [T] bool, positions whose token must not contribute
    :return:          [T, n_layers, n_hash_cols] int64 bucket indices
    """
    T = int(token_ids.shape[0])
    n_layers = len(layout.layer_ids)
    n_heads = layout.n_heads
    max_ngram = layout.max_ngram_size
    # the pad fills blocked slots as a COMPRESSED id, like every other value
    # hashed here (DeepSeek: pad_id = token_map[engram_pad_id])
    pad_id = int(token_map[layout.pad_token_id])

    src = token_map[token_ids].astype(np.int64)          # [T] compressed ids
    if dead_mask is not None:
        src = np.where(dead_mask, dead_id, src)

    out = np.zeros((T, n_layers, layout.n_hash_cols), dtype = np.int64)
    pos = np.arange(T)

    for li in range(n_layers):
        rolling = np.zeros(T, dtype = np.int64)
        blocked = np.zeros(T, dtype = bool)
        flat_primes = np.array(
            [p for per_ngram in layout.primes[li] for p in per_ngram], dtype = np.int64
        )
        flat_offsets = layout.offsets[li].astype(np.int64)
        for shift in range(max_ngram):
            look = pos - shift
            oob = look < 0
            value = np.where(oob, pad_id, src[np.clip(look, 0, T - 1)])
            blocked |= oob | (value == dead_id)
            value = np.where(blocked, pad_id, value)
            rolling = rolling ^ (value * np.int64(layout.hash_multipliers[li][shift]))
            if shift > 0:
                for head in range(n_heads):
                    col = (shift - 1) * n_heads + head
                    out[:, li, col] = rolling % flat_primes[col] + flat_offsets[col]
    return out
