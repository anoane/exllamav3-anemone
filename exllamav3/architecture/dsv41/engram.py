# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
Engram bucket layout and hash multipliers for DeepSeek-V4.1.

Derived from vLLM vllm/models/deepseek_v4_1/common/engram.py (Apache-2.0),
itself a port of DeepSeek's reference inference/engram.py. See NOTICE.

Engram is an n-gram hash lookup gated into the hyper-connection stream. It
exists only on the backbone layers listed in ``engram_layer_ids``.

Everything in this module is deterministic and CPU-only: the bucket layout
and the hash multipliers are derived from the config alone. They must match
the reference bit-for-bit -- the primes are drawn in a fixed order and never
reused, so any divergence silently sends every lookup into the wrong bucket.
"""

from __future__ import annotations

import numpy as np

_INT64_MAX = np.iinfo(np.int64).max


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """Next prime strictly above ``start`` that has not been handed out yet."""
    n = start + 1
    while True:
        if n not in seen and is_prime(n):
            return n
        n += 1


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> np.ndarray:
    """
    One multiplier per (layer, lookback), from a per-layer RNG so layers hash
    differently. Kept odd and bounded so token_id * multiplier cannot overflow
    int64. Seed is 10007 * layer_id; do not change it.
    """
    bound = max(1, (_INT64_MAX // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


class EngramLayout:
    """
    Bucket layout of the n-gram hash tables.

    A position is hashed as ``max_ngram_size - 1`` n-grams (2-gram .. max),
    each split over ``n_heads`` heads. Every (n-gram size, head) pair owns its
    own prime-sized bucket range in the layer's table; primes are drawn in
    order and never reused, keeping the ranges disjoint.
    """

    def __init__(self, config) -> None:
        self.layer_ids = tuple(config.engram_layer_ids)
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.max_ngram_size = int(config.engram_max_ngram_size)
        self.n_heads = int(config.engram_n_heads)
        self.head_dim = int(config.engram_head_dim)
        self.compressed_vocab_size = int(config.engram_compressed_vocab_size)
        self.pad_token_id = int(config.engram_pad_token_id)
        assert len(self.layer_ids) == len(self.num_embeddings), \
            "engram_layer_ids and engram_num_embeddings must be the same length"

        primes = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], int(config.engram_vocab_size) - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes = tuple(primes)

        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        self.offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])

        self.hash_multipliers = compute_hash_multipliers(
            self.layer_ids, self.max_ngram_size, self.compressed_vocab_size
        )

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        if not getattr(config, "engram_layer_ids", None):
            return None
        return cls(config)
