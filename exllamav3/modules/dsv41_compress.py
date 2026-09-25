"""
Triton kernels for DeepSeek-V4.1's KV compressor.

rms_norm_rows: row-local RMSNorm of the compressor latents, for EXL3_STABLE_ARITHMETIC=1.
architecture/dsv41/compressor.py normalizes every pooled latent with torch reductions. On a GPU,
torch chooses the reduction strategy from the tensor shape, so the same latent normalized in calls
of different row counts (prefill chunk sizes, decode) can differ in the last bit, and the cached
and the stateless compression paths close different numbers of groups per call. Under the stable
arithmetic profile compressor.rms_norm calls rms_norm_rows instead: one Triton program per row,
whose reduction layout depends only on the row width. The formula is compressor.rms_norm's
overflow-safe scaled one, not a bitwise reproduction of the torch path.

fused_compress: the rate-2 pair pooling and that RMSNorm in one kernel, over FP32 projections,
with FP32 position-indexed carry rings for a pair left open at a chunk boundary, for
EXL3_DSV41_FUSED_COMPRESS=1 (the cached path of the rate-2 kv sources; see cache/dsv41.py). One
program per group: the pair's column softmax, the weighted sum (clamped back between the pair's
values, which rounded weights could leave), then the same scaled RMSNorm. The projections
themselves still come from the engine's quantized Linear.

The kernels' device assertions are compiled only with EXL3_DSA_DEBUG_BOUNDS=1, the debug switch
of the DSA kernels (attention_fn/dsa_triton.py), read once at import (DEBUG_ASSERTS). Then a
nonfinite projection, gate, carry row, weight or normalized value stops them with a CUDA
device-side assertion instead of reaching the cache. That leaves the process's CUDA context
unusable (every later CUDA call fails) and the process has to be restarted. Triton's debug mode
also compiles int32-overflow checks into the kernels' address arithmetic (redundant with the
host-side int32 guards) and grows the kernels. By default the kernels are compiled without any of
that, like every other kernel of the default build: they do not check their operands, as the
torch paths do not either.
"""

import math
import os

import torch
import triton
import triton.language as tl

_MIN_EPS = 2.0 ** -126
_MAX_EPS = float.fromhex("0x1.fffffep+127")
# smallest normal FP32 value
FLT_MIN = tl.constexpr(1.1754943508222875e-38)

# EXL3_DSA_DEBUG_BOUNDS=1: compile the kernels of this module in Triton's debug mode, with their
# device assertions (same variable and test as attention_fn/dsa_triton.py). Read once, at import
DEBUG_ASSERTS = os.environ.get("EXL3_DSA_DEBUG_BOUNDS", "0") != "0"


@triton.jit(debug = DEBUG_ASSERTS)
def _rms_norm_rows_kernel(x, weight, out, D: tl.constexpr, BLOCK: tl.constexpr, SQRT_EPS: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < D
    a = tl.load(x + row * D + column, mask = active, other = 0).to(tl.float32)
    tl.device_assert(tl.sum((active & ~(tl.abs(a) < float("inf"))).to(tl.int32), 0) == 0,
                     "V4.1 compressor: nonfinite projection")

    # Scale the norm reduction to avoid squaring large finite projections. The sqrt(eps) lower
    # bound also keeps tiny or all-zero rows away from underflow
    maximum = tl.maximum(tl.max(tl.where(active, tl.abs(a), 0), 0), SQRT_EPS)
    scaled = a / maximum
    eps_scaled = SQRT_EPS / maximum
    inv = tl.rsqrt(tl.sum(tl.where(active, scaled * scaled, 0), 0) / D + eps_scaled * eps_scaled)
    norm_weight = tl.load(weight + column, mask = active, other = 0).to(tl.float32)
    result = scaled * inv * norm_weight
    tl.device_assert(tl.sum((active & ~(tl.abs(result) < float("inf"))).to(tl.int32), 0) == 0,
                     "V4.1 compressor: nonfinite normalized latent")
    tl.store(out + row * D + column, result, mask = active)


def rms_norm_rows(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """
    compressor.rms_norm's scaled RMSNorm over the last dimension of a CUDA tensor, one program per
    row, so a row's result does not depend on the other rows of the call.

    x       (..., dim) CUDA tensor, any float dtype and layout, dim <= 4096
    weight  (dim,) FP16, BF16 or FP32 on x's device, or None
    eps     positive normal FP32 value
    returns FP32 tensor of x's shape

    Invalid arguments raise ValueError before anything runs; nonfinite values stop the kernel with
    a device-side assertion (see the module docstring).
    """
    if not math.isfinite(eps) or not _MIN_EPS <= eps <= _MAX_EPS:
        raise ValueError("compressor RMSNorm requires epsilon in the positive normal FP32 range")
    if not x.is_cuda:
        raise ValueError("rms_norm_rows: x must be a CUDA tensor")
    shape = tuple(x.shape)
    dim = shape[-1] if shape else 0
    if not 0 < dim <= 4096:
        raise ValueError(f"rms_norm_rows: row width {dim} outside 1..4096")
    # the kernel addresses rows as row * dim + column in int32
    if math.prod(shape) >= 1 << 31:
        raise ValueError("rms_norm_rows: addressing exceeds the int32 limit")
    if weight is None:
        weight = torch.ones(dim, device = x.device, dtype = torch.float32)
    elif (tuple(weight.shape) != (dim,) or weight.device != x.device or
          weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError("rms_norm_rows: the weight must be (dim,) FP16/BF16/FP32 on x's device")
    rows = x.float().reshape(-1, dim).contiguous()
    out = torch.empty_like(rows)
    if rows.shape[0]:
        with torch.cuda.device(x.device):
            _rms_norm_rows_kernel[(rows.shape[0],)](
                rows, weight.contiguous(), out, dim, triton.next_power_of_2(dim), math.sqrt(eps),
                enable_fp_fusion = False,
            )
    return out.view(shape)


@triton.jit(do_not_specialize = ["pos0", "rows"], debug = DEBUG_ASSERTS)
def _pool_pairs_norm_kernel(kv, gate, carry_kv, carry_gate, weight, out, pos0, rows,
                            carry_rows: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                            SQRT_EPS: tl.constexpr):
    # One program per group of two positions: group g covers absolute positions p, p + 1 with
    # p = (pos0 // 2 + g) * 2. A pair's first row lies before this call (row < 0) only when pos0
    # is odd; it then comes from the carry ring, which the previous call left in place
    group = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    active = column < D
    absolute = (pos0 // 2 + group) * 2
    row = absolute - pos0
    old = absolute % carry_rows
    a = tl.load(kv + row * D + column, mask = active & (row >= 0) & (row < rows), other = 0).to(tl.float32)
    a += tl.load(carry_kv + old * D + column, mask = active & (row < 0), other = 0).to(tl.float32)
    b = tl.load(kv + (row + 1) * D + column, mask = active & (row + 1 < rows), other = 0).to(tl.float32)
    ga = tl.load(gate + row * D + column, mask = active & (row >= 0) & (row < rows), other = 0).to(tl.float32)
    ga += tl.load(carry_gate + old * D + column, mask = active & (row < 0), other = 0).to(tl.float32)
    gb = tl.load(gate + (row + 1) * D + column, mask = active & (row + 1 < rows), other = 0).to(tl.float32)
    finite = (tl.abs(a) < float("inf")) & (tl.abs(b) < float("inf"))
    finite &= (tl.abs(ga) < float("inf")) & (tl.abs(gb) < float("inf"))
    tl.device_assert(tl.sum((active & ~finite).to(tl.int32), 0) == 0,
                     "V4.1 compressor: nonfinite projection or carry")

    # Column softmax over the pair, then the weighted sum
    maximum = tl.maximum(ga, gb)
    pa, pb = tl.exp(ga - maximum), tl.exp(gb - maximum)
    denominator = pa + pb
    ha, hb = tl.exp((ga - maximum) * .5), tl.exp((gb - maximum) * .5)
    # A subnormal exp(delta) can lose precision or flush to zero before a large projection makes
    # the product normal again: split just those factors in two square roots, and keep the
    # ordinary-range products otherwise
    term_a = tl.where(pa < FLT_MIN, (a * ha) * (ha / denominator), a * (pa / denominator))
    term_b = tl.where(pb < FLT_MIN, (b * hb) * (hb / denominator), b * (pb / denominator))
    # Separately rounded weights need not sum to one, and even two FLT_MAX values could then
    # overflow; the exact weighted mean lies between a and b
    pooled = term_a + term_b
    a = tl.minimum(tl.maximum(pooled, tl.minimum(a, b)), tl.maximum(a, b))

    # Scaled RMSNorm, as in _rms_norm_rows_kernel
    maximum = tl.maximum(tl.max(tl.where(active, tl.abs(a), 0), 0), SQRT_EPS)
    scaled = a / maximum
    eps_scaled = SQRT_EPS / maximum
    inv = tl.rsqrt(tl.sum(tl.where(active, scaled * scaled, 0), 0) / D + eps_scaled * eps_scaled)
    norm_weight = tl.load(weight + column, mask = active, other = 0).to(tl.float32)
    result = scaled * inv * norm_weight
    tl.device_assert(tl.sum((active & ~(tl.abs(result) < float("inf"))).to(tl.int32), 0) == 0,
                     "V4.1 compressor: nonfinite normalized latent")
    # Only groups closed by this call are stored; the last program of an odd tail only checks
    # the pending row, which becomes carry
    complete = row + 2 <= rows
    tl.store(out + group * D + column, result, mask = active & complete)


def fused_compress(
    kv: torch.Tensor,
    gate: torch.Tensor,
    pos0: int,
    norm_weight: torch.Tensor,
    eps: float,
    carry_kv: torch.Tensor,
    carry_gate: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """
    Rate-2 compression of one chunk: pool every group this chunk closes and normalize it, then
    record the chunk's rows in the carry rings, all on the current stream.

    kv, gate               (rows, dim) contiguous FP32 CUDA projections of positions
                           pos0 .. pos0 + rows - 1, dim <= 4096
    pos0                   absolute position of the chunk's first row
    norm_weight            (dim,) contiguous FP16, BF16 or FP32 on kv's device
    eps                    positive normal FP32 value
    carry_kv, carry_gate   (ring_rows, dim) contiguous FP32 on kv's device, ring_rows >= 2:
                           row p % ring_rows holds position p's raw projections
    returns                (latent (closed, dim) FP32, index of the first closed group,
                           pos0 // 2), as CompressCarry.step does

    Invalid arguments raise ValueError before anything runs; nonfinite values stop the kernel
    with a device-side assertion (see the module docstring).
    """
    if pos0 < 0 or not math.isfinite(eps) or not _MIN_EPS <= eps <= _MAX_EPS:
        raise ValueError("fused_compress requires a nonnegative position and a positive normal FP32 epsilon")
    for name, t in (("kv", kv), ("gate", gate), ("carry_kv", carry_kv), ("carry_gate", carry_gate)):
        if t.ndim != 2 or not t.is_cuda or t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError(f"fused_compress: {name} must be a contiguous 2-D FP32 CUDA tensor")
    rows, dim = kv.shape
    if gate.shape != kv.shape or carry_gate.shape != carry_kv.shape or carry_kv.shape[1] != dim:
        raise ValueError("fused_compress: projection, gate and carry shapes do not match")
    ring_rows = carry_kv.shape[0]
    if ring_rows < 2:
        raise ValueError("fused_compress: the carry rings need at least two rows")
    if not 0 < dim <= 4096:
        raise ValueError(f"fused_compress: row width {dim} outside 1..4096")
    if tuple(norm_weight.shape) != (dim,) or not norm_weight.is_contiguous() or \
            norm_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("fused_compress: the norm weight must be a contiguous (dim,) FP16/BF16/FP32 tensor")
    if any(t.device != kv.device for t in (gate, carry_kv, carry_gate, norm_weight)):
        raise ValueError("fused_compress: every tensor must be on the projection's device")
    # The kernel addresses rows as row * dim + column and positions as pos0 + row in int32
    limit = 1 << 31
    if rows * dim >= limit or ring_rows * dim >= limit or pos0 + rows >= limit:
        raise ValueError("fused_compress: addressing exceeds the int32 limit")

    first = pos0 // 2
    closed = (pos0 + rows) // 2 - first
    out = torch.empty((closed, dim), device = kv.device, dtype = torch.float32)
    if rows:
        with torch.cuda.device(kv.device):
            # One program per group the chunk touches, including a trailing open one, so that
            # its pending row is checked for nonfinite values before it becomes carry
            _pool_pairs_norm_kernel[(triton.cdiv(pos0 % 2 + rows, 2),)](
                kv, gate, carry_kv, carry_gate, norm_weight, out, pos0, rows, ring_rows, dim,
                triton.next_power_of_2(dim), math.sqrt(eps),
                enable_fp_fusion = False,
            )
        # The kernel's reads of the old ring finish before these writes on this stream,
        # including an odd-position chunk whose first group takes its row from the ring
        start = max(0, rows - ring_rows)
        indices = torch.arange(pos0 + start, pos0 + rows, device = kv.device) % ring_rows
        carry_kv.index_copy_(0, indices, kv[start:])
        carry_gate.index_copy_(0, indices, gate[start:])
    return out, first
