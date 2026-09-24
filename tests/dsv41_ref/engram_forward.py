# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
Engram forward: hash -> gather -> wkv -> gate.

Derived from vLLM vllm/models/deepseek_v4_1/common/engram.py (Apache-2.0).
See NOTICE.

Stages, with the shapes DeepSeek-V4.1-Flash actually uses:

  1. hash      token ids            -> [T, n_hash_cols]      24 buckets/token
  2. gather    table rows by index  -> [T, n_hash_cols, 256] fp8 + e8m0 scales
  3. flatten                        -> [T, 6144]             24 * 256
  4. wkv       6144 -> 25600        -> [T, 5, 5120]          (hc_mult + 1) blocks
  5. gate      per stream           -> [T, hc_mult]          sigmoid(signed sqrt)
  6. inject    out = hidden + gate * value

The wkv output is (hc_mult + 1) * hidden_size: the LEADING hc_mult blocks
are the gate keys, one per stream, and the TRAILING block is the value every
stream receives (DeepSeek's Engram.forward: kv.split([hc_mult * dim, dim])).
Every position injects, including the first max_ngram_size - 1: they hash
with padded lookback. Only dead tokens (image spans) have their gate shut.
tests/test_dsv41_engram_ref_.py checks this reference against DeepSeek's own
Engram.forward.
"""

from __future__ import annotations

import numpy as np

from .engram_gate import engram_gate_reference
from .engram_hash import engram_hash_reference


class EngramForward:
    """Reference engram stage for one layer. Semantics, not speed."""

    def __init__(self, layout, layer_index: int, table, wkv, q_weight, k_weight,
                 hc_mult: int, hidden_size: int, eps: float = 1e-20):
        self.layout = layout
        self.li = layer_index
        self.table = table
        self.wkv = wkv                     # callable: [T, 6144] -> [T, out]
        self.q_weight = q_weight           # [hc_mult, hidden]
        self.k_weight = k_weight           # [hc_mult, hidden]
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.eps = eps                     # config rms_norm_eps (1e-20 for V4.1-Flash)

    def expected_gather_width(self) -> int:
        return self.layout.n_hash_cols * self.layout.head_dim

    def __call__(self, token_ids, token_map, hidden, *, dead_mask = None):
        """
        :param token_ids: [T] int64, one sequence from position 0
        :param token_map: [vocab] int64 compressed ids
        :param hidden:    [T, hc_mult, hidden_size] float32 -- the streams
        :param dead_mask: [T] bool, True for tokens that take no part (image spans)
        :return: (out [T, hc_mult, hidden_size], gate [T, hc_mult])
        """
        T = int(token_ids.shape[0])

        # 1-2: hash and gather this layer's buckets
        idx = engram_hash_reference(token_ids, self.layout, token_map,
                                    dead_mask = dead_mask)[:, self.li, :]   # [T, cols]
        rows = self.table.gather(idx.reshape(-1))                          # [T*cols, hd]
        gathered = rows.reshape(T, -1)                                     # [T, 6144]
        if gathered.shape[1] != self.expected_gather_width():
            raise ValueError(
                f"gathered width {gathered.shape[1]} != n_hash_cols*head_dim "
                f"{self.expected_gather_width()}")

        # 3-4: project to hc_mult keys (leading) + one shared value (trailing)
        proj = self.wkv(gathered)                                          # [T, out]
        if proj.shape[1] != (self.hc_mult + 1) * self.hidden_size:
            raise ValueError(f"wkv output {proj.shape[1]} != (hc_mult + 1) * hidden_size "
                             f"{(self.hc_mult + 1) * self.hidden_size}")
        blocks = proj.reshape(T, self.hc_mult + 1, self.hidden_size)
        key = blocks[:, :self.hc_mult, :]                                  # [T, hc, D]
        value = blocks[:, self.hc_mult, :]                                 # [T, D]

        # 5: gate; only dead tokens are shut
        mask = None if dead_mask is None else ~np.asarray(dead_mask, dtype = bool)
        gate = engram_gate_reference(hidden, key, self.q_weight, self.k_weight,
                                     eps = self.eps, token_mask = mask)

        # 6: every stream receives the same value, scaled by its own gate
        out = hidden.astype(np.float32) + gate[:, :, None] * value[:, None, :]
        return out, gate
