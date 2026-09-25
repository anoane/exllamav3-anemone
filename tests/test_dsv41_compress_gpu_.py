"""GPU: the compressor kernels of modules/dsv41_compress.py against independent FP64 oracles, on
every visible GPU.

rms_norm_rows, the row-local RMSNorm of EXL3_STABLE_ARITHMETIC=1:

  * a row's result does not depend on the other rows of the call: rows normalized in calls of
    1 to 4096 rows equal the same rows of one call bitwise;
  * FP64 oracle over zero, tiny, ordinary and large finite inputs, both weight kinds, batched and
    strided layouts, and the extreme finite values (FLT_MAX rows) the scaled formula exists for;
  * invalid arguments are refused before a launch;
  * nonfinite inputs stop the kernel with a device-side assertion when EXL3_DSA_DEBUG_BOUNDS=1
    compiles the assertions in, and pass unchecked in a default build (no assertion compiled).
    Those negative controls run in child processes, one per case and setting, because a failed
    device assertion leaves the CUDA context unusable.

fused_compress, the rate-2 pair pooling of EXL3_DSV41_FUSED_COMPRESS=1, the same way: FP64 oracle
with pair gates from ordinary to 140000 apart, chunked calls from odd positions bitwise equal to
one call, a rewind by the largest the ring allows replayed from the ring, the rings' contents,
extreme finite pairs, refusals, and nonfinite rows, gates, carry rows and weights stopped by the
device assertion with EXL3_DSA_DEBUG_BOUNDS=1 and passing unchecked without it.

The kernels load by path, without the compiled extension. Run on GPUs no other process needs:

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


def fused_kernel():
    return load_package_file("exllamav3/modules/dsv41_compress.py", "_dsv41_compress").fused_compress


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


def pair_oracle(kv, gate, weight, eps):
    x = (kv.double().view(-1, 2, kv.shape[-1]) * gate.double().view(-1, 2, kv.shape[-1]).softmax(1)).sum(1)
    return x * torch.rsqrt(x.square().mean(-1, keepdim = True) + eps) * weight.double()


def fused_chunked_and_oracle(device):
    fused = fused_kernel()
    g = torch.Generator(device = device).manual_seed(1234)
    worst, calls = 0.0, 0
    for dim in (128, 512):
        n = 4370
        weight = 1 + torch.randn(dim, generator = g, device = device) * 0.1
        for scale in (0.0, 1e-30, 1e-12, 1.0, 70000.0, 1e30):
            kv = torch.randn(n, dim, generator = g, device = device) * scale
            gate = torch.randn(n, dim, generator = g, device = device) * 1000
            gate[::2, :4] = torch.tensor([1000.0, 70000.0, -70000.0, 0.0], device = device)
            gate[1::2, :4] = torch.tensor([1000.25, 70000.25, 70000.0, 0.0], device = device)
            rings = lambda: (torch.zeros(258, dim, device = device), torch.zeros(258, dim, device = device))
            whole, first = fused(kv, gate, 0, weight, 1e-20, *rings())
            assert first == 0 and whole.shape == (n // 2, dim)
            reference = pair_oracle(kv, gate, weight, 1e-20)
            atol = 1e-25 if 0 < scale < 1e-20 else 3e-5
            torch.testing.assert_close(whole.double(), reference, rtol = 3e-5, atol = atol)
            assert torch.isfinite(whole).all()
            worst = max(worst, (whole.double() - reference).abs().max().item())
            ck, cg = rings()
            pos, parts = 0, []
            for rows in (1, 2, 7, 257, 4096, 7):
                end = min(pos + rows, n)
                got, first = fused(kv[pos:end].contiguous(), gate[pos:end].contiguous(), pos, weight, 1e-20, ck, cg)
                assert first == pos // 2
                parts.append(got)
                pos, calls = end, calls + 1
            assert pos == n
            assert torch.equal(torch.cat(parts), whole), f"dim {dim} scale {scale}: chunked != one call"
            rows_ = torch.arange(n - 258, n, device = device) % 258
            assert torch.equal(ck[rows_], kv[-258:]) and torch.equal(cg[rows_], gate[-258:])
            # the largest rewind the ring allows (buf_rows - 1), onto an odd position
            rewind = n - 257
            replay, first = fused(kv[rewind:].contiguous(), gate[rewind:].contiguous(), rewind, weight, 1e-20, ck, cg)
            assert torch.equal(replay, whole[first:]), f"dim {dim} scale {scale}: rewind replay"
            calls += 2
        assert fused(kv[:0], gate[:0], 5, weight, 1e-20, *rings())[0].shape == (0, dim)
    print(f"  OK  {device}: fused pooling, {calls} calls, chunked and rewound calls bitwise equal to one "
          f"call, FP64 oracle max_abs {worst:.3g}")


def fused_extreme_finite(device):
    fused = fused_kernel()
    limit = torch.finfo(torch.float32).max
    pairs = [(limit, limit), (-limit, -limit), (limit, -limit), (-limit, limit),
             (1e-30, limit), (limit, 1e-30), (-1e-30, -limit), (-limit, -1e-30)]
    kv = torch.tensor(pairs, dtype = torch.float32, device = device).reshape(-1, 1).expand(-1, 128).contiguous()
    kv[:, 1::2] *= 0.5
    weight = torch.ones(128, device = device)
    calls = 0
    for ga, gb in ((0., -17.328678), (0., 0.), (-50., 50.), (50., -50.), (limit, -limit)):
        gate = torch.empty_like(kv)
        gate[::2], gate[1::2] = ga, gb
        for eps in (2.0 ** -126, 1e-20, limit):
            ck, cg = torch.zeros(258, 128, device = device), torch.zeros(258, 128, device = device)
            outputs, pos = [], 0
            for end in (1, 6, len(kv)):
                outputs.append(fused(kv[pos:end].contiguous(), gate[pos:end].contiguous(), pos, weight, eps, ck, cg)[0])
                pos, calls = end, calls + 1
            got = torch.cat(outputs)
            assert torch.isfinite(got).all(), "a finite convex pair overflowed"
            torch.testing.assert_close(got.double(), pair_oracle(kv, gate, weight, eps), rtol = 3e-5, atol = 3e-5)
    print(f"  OK  {device}: fused pooling, extreme finite pairs finite and within the oracle ({calls} odd-chunk calls)")


def fused_refusals(device):
    fused = fused_kernel()
    kv = torch.ones(3, 128, device = device)
    ring = torch.zeros(258, 128, device = device)
    weight = torch.ones(128, device = device)

    class Oversized:                                          # never allocated: only its shape is read
        ndim, is_cuda, dtype = 2, True, torch.float32
        shape = (1 << 24, 128)
        device = kv.device

        def is_contiguous(self):
            return True

    ok = (kv, kv.clone(), 0, weight, 1e-20, ring, ring.clone())
    cases = (
        dict(pos0 = -1), dict(eps = float("nan")), dict(eps = 1e-50), dict(eps = 1e100),
        dict(kv = kv.cpu()), dict(gate = kv.half()), dict(carry_kv = ring[:, :64]),
        dict(carry_gate = ring.t().contiguous().t()), dict(kv = torch.ones(3, 2, 128, device = device)),
        dict(gate = torch.ones(4, 128, device = device)), dict(carry_kv = ring[:1], carry_gate = ring[:1]),
        dict(kv = torch.ones(3, 8192, device = device), gate = torch.ones(3, 8192, device = device),
             carry_kv = torch.zeros(258, 8192, device = device), carry_gate = torch.zeros(258, 8192, device = device)),
        dict(norm_weight = weight.double()), dict(norm_weight = weight[:64]), dict(norm_weight = weight.cpu()),
        dict(pos0 = (1 << 31) - 2),
        dict(kv = Oversized(), gate = Oversized()),
    )
    names = ("kv", "gate", "pos0", "norm_weight", "eps", "carry_kv", "carry_gate")
    for i, case in enumerate(cases):
        args = dict(zip(names, ok), **case)
        try:
            fused(**args)
        except ValueError:
            continue
        raise AssertionError(f"fused refusal case {i} ({sorted(case)}) was accepted")
    print(f"  OK  {device}: fused pooling, {len(cases)} invalid argument sets refused before a launch")


def negative(device, case):
    module = load_package_file("exllamav3/modules/dsv41_compress.py", "_dsv41_compress")
    if case.startswith("fused_"):
        what = f"fused pooling, nonfinite {case[6:]}"
        kv = torch.ones(3, 128, device = device)
        gate = torch.zeros_like(kv)
        ck, cg = torch.ones(258, 128, device = device), torch.zeros(258, 128, device = device)
        weight = torch.ones(128, device = device)
        pos = 0
        if case == "fused_pending_kv":                        # the odd row that becomes carry
            kv[-1, 0] = float("inf")
        elif case == "fused_gate":
            gate[0, 0] = float("nan")
        elif case == "fused_carry":                           # read by a chunk that starts mid-pair
            pos = 1
            ck[0, 0] = float("nan")
        elif case == "fused_weight":
            weight[0] = float("nan")
        run = lambda: module.fused_compress(kv, gate, pos, weight, 1e-20, ck, cg)
    else:
        what = f"nonfinite {case}"
        x = torch.ones(3, 128, device = device)
        weight = torch.ones(128, device = device)
        if case == "inf":
            x[-1, 0] = float("inf")
        elif case == "nan":
            x[0, 5] = float("nan")
        elif case == "weight":
            weight[0] = float("nan")
        run = lambda: module.rms_norm_rows(x, weight, 1e-20)
    try:
        run()
        torch.cuda.synchronize(device)
    except RuntimeError as e:
        if "assert" not in str(e).lower() or not module.DEBUG_ASSERTS:
            raise
        print(f"  OK  {device}: {what} stopped by a device assertion (EXL3_DSA_DEBUG_BOUNDS=1)", flush = True)
        return
    if module.DEBUG_ASSERTS:
        raise AssertionError(f"{what} was accepted with EXL3_DSA_DEBUG_BOUNDS=1")
    print(f"  OK  {device}: {what} not checked in a default build (no assertion compiled)", flush = True)


def negative_controls(d):
    """Every negative case in a child process, with the assertions compiled in and without them"""
    for case in NEGATIVE_CASES:
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


NEGATIVE_CASES = ("inf", "nan", "weight", "fused_pending_kv", "fused_gate", "fused_carry", "fused_weight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type = int, default = None, help = "one GPU (default: every visible GPU)")
    parser.add_argument("--negative-case", choices = NEGATIVE_CASES)
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
        fused_refusals(device)
        fused_chunked_and_oracle(device)
        fused_extreme_finite(device)
        negative_controls(d)
    print("OK")
    return 0


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        skip("V4.1 compressor kernels on the GPU", "no CUDA device")
        sys.exit(0)
    sys.exit(main())
