"""
GPU: ext.silu_ref_mul, DeepSeek's reference swiglu (exllamav3_ext/silu_ref.cuh), on every visible
GPU. Needs the compiled extension; skipped when there is no GPU.

  * FP16 and FP32 inputs against an FP64 oracle of min(g, limit) and clamp(u, -limit, limit), then
    silu(g) * u: the FP16 result is within one FP16 step of it (the kernel computes in FP32 with
    the fast exponential and rounds once);
  * a limit of 0 clamps nothing;
  * the difference from the plain silu_mul, which clamps the ACTIVATED gate: a gate above the limit
    gives silu(limit) * u here and limit * u there;
  * in place (z is y) gives the same result as a separate output;
  * a negative or nonfinite limit is refused, and so are the layouts the pair-wise kernels cannot
    take: an odd length, a strided or misaligned view, unequal lengths, a partially overlapping
    output.

    python tests/test_silu_ref_gpu_.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import skip


def oracle(g, u, limit):
    g, u = g.double(), u.double()
    if limit > 0:
        g = g.clamp(max = limit)
        u = u.clamp(-limit, limit)
    return g * torch.sigmoid(g) * u


def within_one_fp16_step(z, reference):
    # |z - reference| no larger than the FP16 spacing at the reference's magnitude: correct
    # rounding is within half of it, and the FP32 arithmetic adds far less than the other half
    spacing = torch.exp2(torch.floor(torch.log2(reference.abs().clamp(min = 2.0 ** -14))) - 10)
    return bool(((z.double() - reference).abs() <= spacing).all())


def check(device, ext):
    g0 = torch.Generator(device = device).manual_seed(1)
    n = 4096 * 8
    for dtype in (torch.half, torch.float):
        for limit in (10.0, 0.0, 7.0):
            g = (torch.randn(n, generator = g0, device = device) * 8).to(dtype)
            u = (torch.randn(n, generator = g0, device = device) * 8).to(dtype)
            z = torch.empty(n, dtype = torch.half, device = device)
            ext.silu_ref_mul(g, u, z, limit)
            assert within_one_fp16_step(z, oracle(g, u, limit)), (dtype, limit)
            if dtype == torch.half:
                y = u.clone()
                ext.silu_ref_mul(g, y, y, limit)
                assert torch.equal(y, z), "in place differs"
    # a gate above the limit: the reference clamps the raw gate (silu(1) = 0.731), silu_mul the
    # activated one (min(silu(2), 1) = 1)
    g = torch.full((2,), 2.0, dtype = torch.half, device = device)
    u = torch.ones(2, dtype = torch.half, device = device)
    ref, plain = torch.empty_like(u), torch.empty_like(u)
    ext.silu_ref_mul(g, u, ref, 1.0)
    ext.silu_mul(g, u, plain, 1.0)
    assert plain[0].item() == 1.0 and abs(ref[0].item() - 0.7311) < 1e-3, (ref, plain)
    for limit in (-1.0, float("nan"), float("inf")):
        try:
            ext.silu_ref_mul(g, u, ref, limit)
        except RuntimeError:
            continue
        raise AssertionError(f"limit {limit} was accepted")
    # The kernels read and write element pairs: layouts they cannot take are refused, never computed
    # with a dropped last element or misread
    h = lambda n: torch.randn(n, generator = g0, device = device).half()
    empty = lambda n: torch.empty(n, dtype = torch.half, device = device)
    shared = empty(10)
    for case, args in (
        ("odd length", (h(7), h(7), empty(7))),
        ("strided input", (h(16)[::2], h(8), empty(8))),
        ("offset FP16 view", (h(9)[1:], h(8), empty(8))),
        ("offset FP32 view", (h(9).float()[1:], h(8).float(), empty(8))),
        ("offset output", (h(8), h(8), empty(9)[1:])),
        ("unequal lengths", (h(8), h(8), empty(6))),
        ("partially overlapping output", (shared[:8], h(8), shared[2:])),
    ):
        try:
            ext.silu_ref_mul(*args, 10.0)
        except RuntimeError:
            continue
        raise AssertionError(f"{case} was accepted")
    print(f"  OK  {device}: silu_ref_mul within one FP16 step of the FP64 oracle (FP16 and FP32 inputs, "
          f"limits 10, 7, 0), in place, raw-gate clamp, refusals of limits and layouts")


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        skip("silu_ref_mul on the GPU", "no CUDA device")
        return 0
    from exllamav3.ext import exllamav3_ext as ext
    for d in range(torch.cuda.device_count()):
        check(torch.device("cuda", d), ext)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
