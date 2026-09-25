"""
Row-local DeepSeek-V4.1 engram gate, for EXL3_STABLE_ARITHMETIC=1.

The engram gate of every (token, residual stream) is a sigmoid of a scaled dot product between
the stream and its n-gram key, both RMS-normalized (DSV41Engram.forward). Written with torch
reductions, the norms and the dot product reduce in a strategy torch picks from the tensor shape,
so a token's gate changes in the last bit with the number of tokens in the call. Here one Triton
program owns one (token, stream) vector, so the reduction layout depends only on the width.

Normalizing and dotting are scaled before the reductions (each vector by its largest magnitude),
so no intermediate sum of squares or products overflows for finite inputs; vectors far outside the
ordinary activation range (largest magnitude below 1e-15 or above 1e15) finish the scalar
normalization in FP64, while every reduction stays FP32. Ordinary activations never take that
branch. The result follows DeepSeek's formula, sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6))), with
an exact zero dot treated as positive, as a torch sum of zeros is.

The kernel's device assertions are compiled only with EXL3_DSA_DEBUG_BOUNDS=1, the debug switch of
the DSA kernels (attention_fn/dsa_triton.py), read once at import (DEBUG_ASSERTS). Then a nonfinite
input, or a gate outside [0, 1], stops the kernel with a CUDA device-side assertion, which leaves
the process's CUDA context unusable (restart the process), and Triton's debug mode also compiles
int32-overflow checks into the kernel's address arithmetic (the host-side int32 guard of
stable_engram_gate makes those redundant) and grows the kernel. By default the kernel is compiled
without any of that, like every other kernel of the default build: it does not check its
operands, as the torch formula does not either.
"""

import math
import os

import torch
import triton
import triton.language as tl

_MIN_EPS = 2.0 ** -126
_MAX_EPS = float.fromhex("0x1.fffffep+127")

# Smallest positive normal FP32 value: the floor of the per-vector scales
FLT_MIN = tl.constexpr(1.1754943508222875e-38)

# EXL3_DSA_DEBUG_BOUNDS=1: compile the kernel in Triton's debug mode, with its device assertions
# (same variable and test as attention_fn/dsa_triton.py). Read once, at import
DEBUG_ASSERTS = os.environ.get("EXL3_DSA_DEBUG_BOUNDS", "0") != "0"


@triton.jit(debug = DEBUG_ASSERTS)
def _gate_kernel(h, key, qk, out, H: tl.constexpr, D: tl.constexpr, KEY_STRIDE: tl.constexpr,
                 BLOCK: tl.constexpr, SQRT_EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    active = col < D
    a = tl.load(h + row * D + col, active, other = 0)
    b = tl.load(key + (row // H) * KEY_STRIDE + (row % H) * D + col, active, other = 0)
    w = tl.load(qk + (row % H) * D + col, active, other = 0)
    finite = (tl.abs(a) < float("inf")) & (tl.abs(b) < float("inf")) & (tl.abs(w) < float("inf"))
    tl.device_assert(tl.sum((active & ~finite).to(tl.int32), 0) == 0,
                     "V4.1 engram gate: nonfinite input")

    # Scale each vector by its largest magnitude before the reductions. Do not use sqrt(eps) as a
    # floor of the dot operands' scales: a dominating epsilon could underflow their product and
    # lose its sign, and the gate's signed magnitude floor would turn that into a finite error
    sa = tl.maximum(tl.max(tl.abs(a), 0), FLT_MIN)
    sb = tl.maximum(tl.max(tl.abs(b), 0), FLT_MIN)
    sw = tl.maximum(tl.max(tl.abs(w), 0), FLT_MIN)
    an, bn, wn = a / sa, b / sb, w / sw
    da, db = tl.maximum(sa, SQRT_EPS), tl.maximum(sb, SQRT_EPS)
    ra, rb, ea, eb = sa / da, sb / db, SQRT_EPS / da, SQRT_EPS / db
    ia = ra * tl.rsqrt((tl.sum(an * an, 0) / D * ra) * ra + ea * ea)
    ib = rb * tl.rsqrt((tl.sum(bn * bn, 0) / D * rb) * rb + eb * eb)
    inner = tl.sum((an * wn) * bn, 0)
    magnitude = ((tl.abs(inner) * ia) * ib) * (D ** -0.5) * sw

    # Extreme ranges need wider scalar rescaling: a tiny normalization factor followed by a huge
    # learned weight can be a meaningful finite result although the intermediate factor flushes to
    # zero in FP32. The feature reductions stay FP32; FP32's whole finite range fits in these FP64
    # squares and products, and a truly huge final magnitude may saturate the gate
    if ((sa < 1.e-15) | (sb < 1.e-15) | (sw < 1.e-15)
            | (sa > 1.e15) | (sb > 1.e15) | (sw > 1.e15) | (SQRT_EPS > 1.e15)):
        ad, bd, wd = sa.to(tl.float64), sb.to(tl.float64), sw.to(tl.float64)
        ed = tl.full((), SQRT_EPS, tl.float64)
        na = tl.sqrt(ad * ad * (tl.sum(an * an, 0) / D).to(tl.float64) + ed * ed)
        nb = tl.sqrt(bd * bd * (tl.sum(bn * bn, 0) / D).to(tl.float64) + ed * ed)
        magnitude = (((tl.abs(inner).to(tl.float64) * wd) * ad) * bd / (na * nb)
                     * (D ** -0.5)).to(tl.float32)

    # The sign comes from the dot of the max-scaled vectors, before the positive normalization
    # factors (which could underflow) are applied.
    # An exact zero sum counts as positive (a torch sum starts from +0.0). A huge magnitude
    # saturates the gate instead of overflowing the normalization into NaN
    root = tl.sqrt(tl.maximum(magnitude, 1.e-6))
    signed = tl.where(inner < 0., -root, root)
    value = 1. / (1. + tl.exp(-signed))
    tl.device_assert((value >= 0.) & (value <= 1.), "V4.1 engram gate: invalid result")
    tl.store(out + row, value)


def stable_engram_gate(h: torch.Tensor, key: torch.Tensor, qk: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Engram gates, one Triton program per (token, stream).

    h       (batch, tokens, streams, dim) FP32 CUDA residual streams
    key     (batch, tokens, streams, dim) FP32 n-gram keys; a view is fine (the model passes a
            slice of the projection output), copied only if its last two dims are not packed
    qk      (streams, dim) FP32 folded query/key weights
    eps     RMSNorm epsilon, a positive normal FP32 value
    returns FP32 gates (batch, tokens, streams), without masking dead tokens

    Invalid arguments raise ValueError before anything runs; nonfinite values stop the kernel with
    a device-side assertion (see the module docstring).
    """
    if not math.isfinite(eps) or not _MIN_EPS <= eps <= _MAX_EPS:
        raise ValueError("engram gate requires epsilon in the positive normal FP32 range")
    if h.ndim != 4 or key.shape != h.shape or not h.is_cuda:
        raise ValueError("engram gate requires matching CUDA [batch, tokens, streams, dim] inputs")
    B, L, H, D = h.shape
    if not 0 < H <= 64 or not 0 < D <= 16384 or qk.shape != (H, D):
        raise ValueError("engram gate: unsupported stream or feature dimensions")
    for tensor in (h, key, qk):
        if tensor.dtype != torch.float32 or tensor.device != h.device:
            raise ValueError("engram gate requires FP32 tensors on one device")
    # the kernel addresses h as row * dim + column and the keys as token * stride in int32
    if h.numel() >= 1 << 31:
        raise ValueError("engram gate: addressing exceeds the int32 limit")
    h, qk = h.contiguous(), qk.contiguous()
    k = key.reshape(-1, H, D)
    if k.stride(-1) != 1 or k.stride(-2) != D:
        k = k.contiguous()
    if B * L * k.stride(0) >= 1 << 31:
        raise ValueError("engram gate: addressing exceeds the int32 limit")
    out = torch.empty((B, L, H), dtype = torch.float32, device = h.device)
    if B * L:
        with torch.cuda.device(h.device):
            _gate_kernel[(B * L * H,)](
                h, k, qk, out, H, D, k.stride(0), triton.next_power_of_2(D), math.sqrt(eps),
                num_warps = 8 if D >= 4096 else 4, enable_fp_fusion = False,
            )
    return out
