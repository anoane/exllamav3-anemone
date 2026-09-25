"""
Row-local RMSNorm of DeepSeek-V4.1 compressor latents, for EXL3_STABLE_ARITHMETIC=1.

architecture/dsv41/compressor.py normalizes every pooled latent with torch reductions. On a GPU,
torch chooses the reduction strategy from the tensor shape, so the same latent normalized in calls
of different row counts (prefill chunk sizes, decode) can differ in the last bit, and the cached
and the stateless compression paths close different numbers of groups per call. Under the stable
arithmetic profile compressor.rms_norm calls rms_norm_rows instead: one Triton program per row,
whose reduction layout depends only on the row width. The formula is compressor.rms_norm's
overflow-safe scaled one, not a bitwise reproduction of the torch path.

The kernel's device assertions are compiled only with EXL3_DSA_DEBUG_BOUNDS=1, the debug switch of
the DSA kernels (attention_fn/dsa_triton.py), read once at import (DEBUG_ASSERTS). Then a nonfinite
projection, weight or normalized value stops the kernel with a CUDA device-side assertion instead
of reaching the cache. That leaves the process's CUDA context unusable (every later CUDA call
fails) and the process has to be restarted. Triton's debug mode also compiles int32-overflow
checks into the kernel's address arithmetic (redundant with the host-side int32 guard) and grows
the kernel. By default the kernel is compiled without any of that, like every other kernel of the
default build: it does not check its operands, as the torch path does not either.
"""

import math
import os

import torch
import triton
import triton.language as tl

_MIN_EPS = 2.0 ** -126
_MAX_EPS = float.fromhex("0x1.fffffep+127")

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
