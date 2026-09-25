"""GPU: the row-local compressor RMSNorm of EXL3_STABLE_ARITHMETIC=1 (modules/dsv41_compress.py)
against an independent FP64 oracle, on every visible GPU:

  * a row's result does not depend on the other rows of the call: rows normalized in calls of
    1 to 4096 rows equal the same rows of one call bitwise;
  * FP64 oracle over zero, tiny, ordinary and large finite inputs, both weight kinds, batched and
    strided layouts, and the extreme finite values (FLT_MAX rows) the scaled formula exists for;
  * invalid arguments are refused before a launch;
  * nonfinite inputs stop the kernel with a device-side assertion when EXL3_DSA_DEBUG_BOUNDS=1
    compiles the assertions in, and pass unchecked in a default build (no assertion compiled).
    Those negative controls run in child processes, one per case and setting, because a failed
    device assertion leaves the CUDA context unusable.

The kernel loads by path, without the compiled extension. Run on GPUs no other process needs:

    python tests/test_dsv41_compress_gpu_.py [--device N]

Skipped when no GPU is visible."""

import argparse
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file, skip


def kernel():
    return load_package_file("exllamav3/modules/dsv41_compress.py", "_dsv41_compress").rms_norm_rows


def oracle(x, weight, eps):
    xd = x.double()
    y = xd * torch.rsqrt(xd.square().mean(-1, keepdim = True) + eps)
    return y if weight is None else y * weight.double()


def row_invariance_and_oracle(device):
    rms_norm_rows = kernel()
    g = torch.Generator(device = device).manual_seed(1234)
    worst, calls = 0.0, 0
    for dim in (128, 512):
        n = 4370
        weight = 1 + torch.randn(dim, generator = g, device = device) * 0.1
        for scale in (0.0, 1e-30, 1e-12, 1.0, 70000.0, 1e30):
            x = torch.randn(n, dim, generator = g, device = device) * scale
            whole = rms_norm_rows(x, weight, 1e-20)
            pos, parts = 0, []
            for rows in (1, 2, 7, 257, 4096, 7):
                end = min(pos + rows, n)
                parts.append(rms_norm_rows(x[pos:end], weight, 1e-20))
                pos, calls = end, calls + 1
            assert pos == n
            assert torch.equal(torch.cat(parts), whole), f"dim {dim} scale {scale}: chunked != one call"
            reference = oracle(x, weight, 1e-20)
            atol = 1e-25 if 0 < scale < 1e-20 else 3e-5
            torch.testing.assert_close(whole.double(), reference, rtol = 3e-5, atol = atol)
            assert torch.isfinite(whole).all()
            worst = max(worst, (whole.double() - reference).abs().max().item())
        # batched and strided inputs, no weight and an FP16 weight
        x = (torch.randn(2, 9, dim * 2, generator = g, device = device) * 3)[..., ::2]
        for w in (None, weight.half()):
            y = rms_norm_rows(x, w, 1e-20)
            assert y.dtype == torch.float32 and y.shape == x.shape
            assert torch.equal(y, rms_norm_rows(x.contiguous().view(-1, dim), w, 1e-20).view(x.shape))
            torch.testing.assert_close(y.double(), oracle(x, None if w is None else w.float(), 1e-20),
                                       rtol = 3e-5, atol = 3e-5)
            calls += 2
        assert rms_norm_rows(torch.empty((0, dim), device = device), None, 1e-20).shape == (0, dim)
    print(f"  OK  {device}: {calls} calls, chunked rows bitwise equal to one call, "
          f"FP64 oracle max_abs {worst:.3g}")


def extreme_finite(device):
    rms_norm_rows = kernel()
    limit = torch.finfo(torch.float32).max
    rows = torch.tensor([[limit, limit], [-limit, limit], [limit, 1e-30], [1e-30, -limit],
                         [1e-38, 1e-38], [0.0, -0.0]], dtype = torch.float32, device = device)
    x = rows.repeat(1, 64)                                    # (6, 128)
    x[:, 1::2] *= 0.5
    for eps in (2.0 ** -126, 1e-20, limit):
        y = rms_norm_rows(x, None, eps)
        assert torch.isfinite(y).all(), "a finite row overflowed"
        torch.testing.assert_close(y.double(), oracle(x, None, eps), rtol = 3e-5, atol = 3e-5)
    print(f"  OK  {device}: extreme finite rows (FLT_MAX, tiny, signed zero) finite and within the oracle")


def refusals(device):
    rms_norm_rows = kernel()
    x = torch.ones(1, 128, device = device)
    weight = torch.ones(128, device = device)

    class Oversized:                                          # never allocated: only its shape is read
        is_cuda = True
        shape = (1 << 24, 128)

    cases = (
        (Oversized(), weight, 1e-6),
        (x.cpu(), weight.cpu(), 1e-6),
        (torch.ones(1, 8192, device = device), None, 1e-6),
        (x, weight.double(), 1e-6),
        (x, weight[:64], 1e-6),
        (x, weight, float("nan")),
        (x, weight, 1e-50),
        (x, weight, 1e100),
    )
    for i, (source, norm, eps) in enumerate(cases):
        try:
            rms_norm_rows(source, norm, eps)
        except ValueError:
            continue
        raise AssertionError(f"refusal case {i} was accepted")
    print(f"  OK  {device}: {len(cases)} invalid argument sets refused before a launch")


def negative(device, case):
    module = load_package_file("exllamav3/modules/dsv41_compress.py", "_dsv41_compress")
    rms_norm_rows = module.rms_norm_rows
    x = torch.ones(3, 128, device = device)
    weight = torch.ones(128, device = device)
    if case == "inf":
        x[-1, 0] = float("inf")
    elif case == "nan":
        x[0, 5] = float("nan")
    elif case == "weight":
        weight[0] = float("nan")
    try:
        rms_norm_rows(x, weight, 1e-20)
        torch.cuda.synchronize(device)
    except RuntimeError as e:
        if "assert" not in str(e).lower() or not module.DEBUG_ASSERTS:
            raise
        print(f"  OK  {device}: nonfinite {case} stopped by a device assertion (EXL3_DSA_DEBUG_BOUNDS=1)",
              flush = True)
        return
    if module.DEBUG_ASSERTS:
        raise AssertionError(f"nonfinite {case} was accepted with EXL3_DSA_DEBUG_BOUNDS=1")
    print(f"  OK  {device}: nonfinite {case} not checked in a default build (no assertion compiled)", flush = True)


def negative_controls(d):
    """Every negative case in a child process, with the assertions compiled in and without them"""
    for case in ("inf", "nan", "weight"):
        for debug in ("1", None):
            env = {k: v for k, v in os.environ.items() if k != "EXL3_DSA_DEBUG_BOUNDS"}
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            if debug is not None:
                env["EXL3_DSA_DEBUG_BOUNDS"] = debug
            r = subprocess.run([sys.executable, __file__, "--device", str(d), "--negative-case", case],
                               capture_output = True, text = True, env = env)
            if r.returncode:
                raise AssertionError(f"negative control {case} (EXL3_DSA_DEBUG_BOUNDS={debug}) on cuda:{d} "
                                     f"failed:\n{r.stdout}\n{r.stderr}")
            print(r.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type = int, default = None, help = "one GPU (default: every visible GPU)")
    parser.add_argument("--negative-case", choices = ("inf", "nan", "weight"))
    args = parser.parse_args()
    if args.negative_case:
        negative(torch.device("cuda", args.device or 0), args.negative_case)
        return 0
    devices = range(torch.cuda.device_count()) if args.device is None else (args.device,)
    for d in devices:
        device = torch.device("cuda", d)
        refusals(device)
        row_invariance_and_oracle(device)
        extreme_finite(device)
        negative_controls(d)
    print("OK")
    return 0


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        skip("V4.1 row-local compressor norm on the GPU", "no CUDA device")
        sys.exit(0)
    sys.exit(main())
