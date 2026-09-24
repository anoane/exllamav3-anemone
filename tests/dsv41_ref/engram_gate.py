# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
Engram gating into the hyper-connection streams.

Derived from vLLM vllm/models/deepseek_v4_1/common/engram.py
(_fused_engram_post_wkv_kernel, Apache-2.0). See NOTICE.

After the n-gram rows are gathered and projected through engram.wkv, the
result is gated into each of the hc_mult hyper-connection residual streams.
The gate is a signed square-root of an RMS-normalised four-way product
between the hidden state, the engram key, and a per-stream (q, k) pair:

    hidden_rms = rsqrt(sum(hidden^2)/D + eps)
    key_rms    = rsqrt(sum(key^2)/D + eps)
    dot        = sum(hidden * q * k * key) * hidden_rms * key_rms * rsqrt(D)
    gate       = sigmoid(copysign(sqrt(max(|dot|, clamp)), dot))

The sqrt is what keeps the gate responsive: without it the normalised dot
sits close to zero and sigmoid saturates flat. The clamp (1e-6) keeps the
sqrt's gradient finite at dot = 0, where the sign of zero decides the side, as in
DeepSeek's torch.copysign; eps is the model's rms_norm_eps (1e-20 for
V4.1-Flash, which must stay fp32). Tokens masked out -- dead tokens only, i.e.
image spans -- get gate = 0. Positions at the start of the sequence are NOT
masked: they hash with padded lookback and inject like any other.

q_weight and k_weight are [hc_mult, hidden_size] -- one (q, k) pair per
residual stream, not per n-gram size. For DeepSeek-V4.1-Flash that is
[4, 5120].
"""

from __future__ import annotations

import numpy as np


def engram_gate_reference(
    hidden: np.ndarray,
    key: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    token_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    :param hidden:   [T, hc_mult, D] float32 -- per-stream hidden states
    :param key:      [T, hc_mult, D] float32 -- engram wkv output per stream
    :param q_weight: [hc_mult, D] float32
    :param k_weight: [hc_mult, D] float32
    :param token_mask: [T] bool -- False zeroes the gate for that token
    :return:         [T, hc_mult] float32 gates in (0, 1)
    """
    hidden = hidden.astype(np.float32)
    key = key.astype(np.float32)
    D = hidden.shape[-1]

    hidden_rms = 1.0 / np.sqrt((hidden * hidden).sum(-1) / D + eps)   # [T, hc]
    key_rms = 1.0 / np.sqrt((key * key).sum(-1) / D + eps)            # [T, hc]

    dot = (hidden * q_weight[None] * k_weight[None] * key).sum(-1)    # [T, hc]
    dot = dot * hidden_rms * key_rms / np.sqrt(float(D))

    mag = np.sqrt(np.maximum(np.abs(dot), clamp_value))
    gate_input = np.copysign(mag, dot)          # -0.0 goes negative, as torch.copysign
    gate = 1.0 / (1.0 + np.exp(-gate_input))

    if token_mask is not None:
        gate = np.where(token_mask[:, None], gate, 0.0)
    return gate.astype(np.float32)
