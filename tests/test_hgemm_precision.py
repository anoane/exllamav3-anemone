"""EXL3_HGEMM_FP32_REDUCTION: fp32 split-K reductions for fp16-output cuBLAS GEMMs in hgemm.cu.

The variable is read once per process, so every case runs in a child process. The switch tests
need the compiled extension but no GPU; the precision tests need a CUDA device.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "requires CUDA")


def _run(value, code):
    env = dict(os.environ)
    env.pop("EXL3_HGEMM_FP32_REDUCTION", None)
    if value is not None:
        env["EXL3_HGEMM_FP32_REDUCTION"] = value
    env["EXL3_HGEMM_F16ACC"] = "0"  # cuBLAS for hgemm_batched regardless of the device
    prelude = "import hashlib, json, torch\nfrom exllamav3.ext import exllamav3_ext as ext\n"
    out = subprocess.run([sys.executable, "-c", prelude + code], env = env, capture_output = True, text = True, cwd = ROOT)
    assert out.returncode == 0, (value, out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(torch.version.hip is not None, reason = "CUDA only")
@pytest.mark.parametrize("value,want", [(None, False), ("0", False), ("1", True)])
def test_switch(value, want):
    assert _run(value, "print(json.dumps(ext.hgemm_fp32_reduction_status()))") == want


# Mismatches against the fp64 product rounded once to fp16, on a split-K-prone shape (few rows,
# large K). fp32 outputs are returned as a hash of all their bits so on/off can be compared exactly
PRECISION = """
torch.manual_seed(0)
out = {}
for m in (8, 256, 2048):
    a = torch.randn(m, 5120, device = "cuda", dtype = torch.half)
    b = (torch.randn(5120, 1280, device = "cuda") / 64).half()
    ref = (a.double() @ b.double()).half()
    c = torch.empty(m, 1280, device = "cuda", dtype = torch.half)
    ext.hgemm(a, b, c)
    cb = torch.empty(2, m, 1280, device = "cuda", dtype = torch.half)
    ext.hgemm_batched(torch.stack((a, a)), torch.stack((b, b)), cb)
    c32 = torch.empty(m, 1280, device = "cuda", dtype = torch.float)
    ext.hgemm(a, b, c32)
    out[m] = {
        "hgemm": int((c != ref).sum()),
        "batched": int((cb != ref).sum()),
        "finite": bool(torch.isfinite(c).all() and torch.isfinite(cb).all()),
        "fp32_bits": hashlib.sha256(c32.cpu().numpy().tobytes()).hexdigest(),
    }
    torch.cuda.synchronize()
print(json.dumps(out))
"""


@needs_cuda
def test_fp32_reduction_precision():
    off = _run(None, PRECISION)
    on = _run("1", PRECISION)
    print({m: (off[m]["hgemm"], on[m]["hgemm"], off[m]["batched"], on[m]["batched"]) for m in off})
    for m in off:
        assert on[m]["finite"] and off[m]["finite"]
        # The switch may make cuBLAS pick another kernel with a different fp32 summation order, so
        # it must not be worse beyond that noise (0.1% of the compared outputs)
        slack = int(m) * 1280 // 1000
        assert on[m]["hgemm"] <= off[m]["hgemm"] + slack, (m, off[m]["hgemm"], on[m]["hgemm"])
        assert on[m]["batched"] <= off[m]["batched"] + 2 * slack, (m, off[m]["batched"], on[m]["batched"])
        # fp32 outputs never see the flag
        assert on[m]["fp32_bits"] == off[m]["fp32_bits"], m


# Partial sums of +-32 * 4096 = 131072 exceed the fp16 range while the exact result is 0. With fp32
# reductions the result must be exactly 0 for every split point cuBLAS may choose. Without them
# the outcome depends on the device and cuBLAS version, so it is only printed
CANCEL = """
out = {}
for m in (1, 8, 256):
    k = 8192
    a = torch.ones(m, k, device = "cuda", dtype = torch.half)
    b = torch.full((k, 128), 32.0, device = "cuda", dtype = torch.half)
    b[k // 2:] = -32.0
    c = torch.empty(m, 128, device = "cuda", dtype = torch.half)
    ext.hgemm(a, b, c)
    cb = torch.empty(2, m, 128, device = "cuda", dtype = torch.half)
    ext.hgemm_batched(torch.stack((a, a)), torch.stack((b, b)), cb)
    out[m] = {
        "zero": bool((c == 0).all() and (cb == 0).all()),
        "finite": bool(torch.isfinite(c).all() and torch.isfinite(cb).all()),
        "max_abs": max(float(c.float().abs().max()), float(cb.float().abs().max())),
    }
print(json.dumps(out))
"""


@needs_cuda
def test_fp32_reduction_cancelling_partials():
    off = _run(None, CANCEL)
    on = _run("1", CANCEL)
    print("off", off, "on", on)
    assert all(r["zero"] for r in on.values()), on
