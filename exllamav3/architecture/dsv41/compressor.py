# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
DeepSeek-V4.1 KV compressor math.

Derived in part from vLLM vllm/models/deepseek_v4_1/compressor.py (Apache-2.0);
see NOTICE.

V4.1 replaces V4's overlapping sliding-window compressor with plain grouped
pooling. For group g of ``compress_ratio`` consecutive tokens, per channel c:

    latent[g, c] = rmsnorm( sum_t softmax_t(score[t, c]) * kv[t, c] )

Two rates occur in DeepSeek-V4.1-Flash:

  ratio 0   sliding window only; no compressor on the layer at all.
  ratio 1   every token is its own group. A softmax over one element is 1, so
            the gate cancels identically and the checkpoint carries no
            ``wgate``; no cross-chunk state is needed either (layers 20-39).
  ratio 2   token pairs pooled with a learned per-channel gate (layers 2-19).

Three differences from V4 that this file exists to pin down:

  * No overlap. A V4 window attends its own m rows plus the previous window's
    first m, so V4 must carry a Ca slice between chunks. V4.1 groups are
    disjoint and carry only the open group's raw rows.
  * No position bias. V4 adds a learned within-window ``ape`` to the gate
    before the softmax. V4.1 has no such tensor -- ``compressor_expected_keys``
    lists norm/wkv/wgate and nothing else, which the checkpoint confirms.
  * The latent returned here is pre-RoPE. Rotation happens at cache insertion,
    where the group's cache position is known. That is the reference's own
    split (``forward`` -> latent, ``insert_cache`` -> rope + quant + store) and
    it keeps the compressed-position convention out of this file.

A group closes at the token whose absolute position p satisfies
``p % ratio == ratio - 1``; only those rows produce a latent. The reference
kernel writes a full-length latent buffer and lets the store skip non-boundary
rows; here the closed rows are returned densely instead, since the caller
already knows the first group index.
"""

from __future__ import annotations

import math

import torch

from ...model.math_policy import STABLE_ARITHMETIC

_MIN_EPS = 2.0 ** -126
_MAX_EPS = float.fromhex("0x1.fffffep+127")


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float, per_row: bool = False) -> torch.Tensor:
    """
    RMSNorm in fp32 with a scaled sum of squares.

    Scaling by max(abs(x), sqrt(eps)) preserves the mathematical normalization
    while avoiding x**2 overflow for finite FP32 projections. This is not a
    bitwise reproduction of the reference's direct sum of squares; that formula
    can silently normalize a large finite vector to zero.

    The mean is a torch reduction. On the CPU a row's result does not depend on how many
    rows share the call; on a GPU the reduction strategy follows the row count, so the same
    latent normalised in chunks of different sizes agrees to about 1e-6 (3e-7 relative
    measured), not bitwise. Under EXL3_STABLE_ARITHMETIC=1 a CUDA input goes to
    modules/dsv41_compress.py instead, the same formula with one fixed reduction per row, so
    the cached and the stateless compression paths normalise a latent identically whatever
    the row count of the call. The CPU path is the same in both cases.

    per_row (EXL3_EXACT_ROWS): every row of a (rows, d) input gets the bits of its one-row
    call. The mean is then one reduction per row, on the freshly allocated (1, d) square a
    one-row call reduces. Everything else serves all rows at once: the maximum of absolute
    values has one bit pattern under any reduction order, and the rest is elementwise.
    """
    if not math.isfinite(eps) or not _MIN_EPS <= eps <= _MAX_EPS:
        raise ValueError("compressor RMSNorm requires epsilon in the positive normal FP32 range")
    if per_row and x.dim() != 2:
        raise ValueError(f"compressor RMSNorm: per_row takes (rows, d) latents, got {tuple(x.shape)}")
    x = x.float()
    if STABLE_ARITHMETIC and x.is_cuda:
        # imported here so this module, which the CPU reference tests load, never needs Triton
        from ...modules.dsv41_compress import rms_norm_rows
        return rms_norm_rows(x, weight, eps)
    scale = x.abs().amax(dim = -1, keepdim = True).clamp_min(math.sqrt(eps))
    scaled = x / scale
    eps_scaled = math.sqrt(eps) / scale
    if per_row and x.shape[0] > 1:
        mean = torch.cat([scaled[i:i + 1].square().mean(dim = -1, keepdim = True) for i in range(x.shape[0])],
                         dim = 0)
    else:
        mean = scaled.square().mean(dim = -1, keepdim = True)
    y = scaled * torch.rsqrt(mean + eps_scaled.square())
    return y if weight is None else y * weight.float()


def group_pool(kv: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """
    Gated pool over the group dimension: ``sum_t softmax_t(score) * kv``.

    kv, score   [n_groups, compress_ratio, head_dim]
    returns     [n_groups, head_dim]

    The softmax is written out rather than handed to ``torch.softmax`` because
    torch picks its kernel by shape when the softmax dimension is not last, so
    the same group pooled in a chunk of 16 and a chunk of 48 can differ by an
    ulp. Max/exp/normalize over a dimension of 1 or 2 is elementwise, and with
    it the chunked path reproduces a single-shot pass bitwise on the CPU --
    which is what makes ``tests/test_dsv41_compressor_.py`` a gate rather than
    a tolerance check. On a GPU the pooling stays elementwise, but the latent's
    RMSNorm (rms_norm) reduces by row count, so chunked and one-shot latents
    agree to about 1e-6 there (bitwise under EXL3_STABLE_ARITHMETIC=1, see
    rms_norm).
    """
    mx = score.amax(dim = 1, keepdim = True)
    e = (score - mx).exp()
    denominator = e.sum(dim = 1, keepdim = True)
    p = e / denominator
    if kv.shape[1] == 2:
        a, b = kv.unbind(1)
        # Do not first round a tiny softmax numerator to a subnormal or zero:
        # multiplying it by a large finite projection can produce a normal,
        # meaningful contribution. Split the exponential across two products.
        # Keep the usual expression unchanged outside this extreme range.
        half_e = ((score - mx) * .5).exp()
        products = torch.where(e < _MIN_EPS,
                               (kv * half_e) * (half_e / denominator), kv * p)
        # Rounded softmax weights need not add to exactly one: a weighted sum
        # can overflow even when a == b == FLT_MAX. Its exact value lies in
        # the operands' convex interval, so restore that invariant after the
        # products. Unlike endpoint interpolation, this retains a small term
        # when the other weight rounds to one. NaNs still propagate.
        pooled = products.sum(dim = 1)
        return torch.minimum(torch.maximum(pooled, torch.minimum(a, b)), torch.maximum(a, b))
    return (kv * p).sum(dim = 1)


class CompressorState:
    """
    Open-group carry for one sequence.

    A group spans ``compress_ratio`` tokens, so at a chunk boundary at most
    ``ratio - 1`` of them are pending. At ratio 2 that is a single row, which
    is why the reference state cache is exactly ``2 * head_dim`` fp32
    (kv_state ++ score_state) and needs no online-softmax accumulator: the
    pending rows stay raw and the softmax is taken once, when the group
    closes.

    ``position`` is the absolute token position of the first pending row, kept
    so a caller can assert chunk contiguity.
    """

    def __init__(self, head_dim: int, compress_ratio: int, device = None):
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.device = device
        self.kv = None
        self.score = None
        self.position = None

    def reset(self):
        self.kv = None
        self.score = None
        self.position = None

    @property
    def pending(self) -> int:
        return 0 if self.kv is None else self.kv.shape[0]

    def push(self, kv: torch.Tensor, score: torch.Tensor, position: int):
        """Keep the trailing partial group. Rows are stored raw, in fp32."""
        assert kv.shape[0] < self.compress_ratio, \
            "a full group must be pooled, not carried"
        if kv.shape[0] == 0:
            self.reset()
            return
        self.kv = kv.float().contiguous()
        self.score = score.float().contiguous() if score is not None else None
        self.position = position

    def take(self):
        """Pop the carried rows, leaving the state empty."""
        kv, score, pos = self.kv, self.score, self.position
        self.reset()
        return kv, score, pos


def compress_chunk(
    kv: torch.Tensor,
    score: torch.Tensor | None,
    position: int,
    compress_ratio: int,
    norm_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    state: CompressorState | None = None,
):
    """
    Pool one chunk into compressed latents.

    kv, score   [seq, head_dim], score None at ratio 1
    position    absolute position of kv[0]
    returns     (latent [n_closed, head_dim] fp32, first_group)

    ``first_group`` is the compressed-cache index of latent[0], i.e.
    ``(position - pending) // compress_ratio`` once the carried rows are
    prepended. With a state, rows left over at the end of the chunk are
    carried; without one they are dropped, which reproduces the HF
    ``cache = None`` semantics V4's stateless path already follows.
    """
    assert compress_ratio in (1, 2), \
        f"V4.1 compressor supports rate 1 and 2, got {compress_ratio}"
    seq, hd = kv.shape

    if compress_ratio == 1:
        # softmax over a single element is 1: the gate cancels and every token
        # closes its own group, so no state is ever carried
        assert score is None, "ratio 1 carries no gate"
        return rms_norm(kv, norm_weight, eps), position

    assert score is not None and score.shape == kv.shape

    kv = kv.float()
    score = score.float()
    start = position
    if state is not None and state.pending:
        c_kv, c_score, c_pos = state.take()
        assert c_pos + c_kv.shape[0] == position, \
            f"non-contiguous chunk: carried rows end at {c_pos + c_kv.shape[0]}, " \
            f"chunk starts at {position}"
        kv = torch.cat([c_kv.to(kv.device), kv], dim = 0)
        score = torch.cat([c_score.to(score.device), score], dim = 0)
        start = c_pos

    assert start % compress_ratio == 0, \
        f"chunk starts mid-group at {start} (rate {compress_ratio}) with no state"

    n = kv.shape[0]
    closed = (n // compress_ratio) * compress_ratio
    rest = n - closed

    if state is not None:
        state.push(kv[closed:], score[closed:], start + closed)

    if closed == 0:
        return kv.new_zeros((0, hd)), start // compress_ratio

    g_kv = kv[:closed].view(-1, compress_ratio, hd)
    g_score = score[:closed].view(-1, compress_ratio, hd)
    return rms_norm(group_pool(g_kv, g_score), norm_weight, eps), start // compress_ratio


def compress_reference(
    kv: torch.Tensor,
    score: torch.Tensor | None,
    compress_ratio: int,
    norm_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    """
    Single-shot pooling of a whole sequence from position 0.

    The independent implementation the chunked path is checked against; a
    chunked run with a state must reproduce it exactly.
    """
    return compress_chunk(kv, score, 0, compress_ratio, norm_weight, eps, None)[0]


def indexer_k_from_latent(
    latent: torch.Tensor,
    wk: torch.Tensor,
    k_norm_weight: torch.Tensor | None,
    eps: float = 1e-6,
):
    """
    Index keys from the compressor latent: ``k_norm(wk(latent))``.

    This is the V4.1 structural change the indexer is built around. V4 runs a
    second full compressor over hidden states to build index keys
    (``{key}.indexer.compressor.*``, head_dim -> index_head_dim through its own
    wkv/wgate); V4.1 reuses the kv source's already-pooled latent and spends
    one small GEMM, head_dim -> index_head_dim, on it. That is why an index
    source which is not also a kv source owns neither ``wk`` nor ``k_norm``:
    it has no latent of its own to project, and reads the kv source's keys.

    latent  [n_groups, head_dim] -- group-boundary rows only
    wk      [index_head_dim, head_dim]
    returns [n_groups, index_head_dim] fp32, pre-RoPE
    """
    k = torch.nn.functional.linear(latent.float(), wk.float())
    return rms_norm(k, k_norm_weight, eps)
