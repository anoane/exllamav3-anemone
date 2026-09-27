"""
CPU MoE experts with activation "gelu" (Gemma 4) compute the tanh approximation, the same function
as the GPU kernels (act_gelu, _gelu) and the model definition (gelu_pytorch_tanh), on both paths
for RAM-resident experts: the native worker (cpu/moe_mul1.cpp, -mcl/-mcs) and the host's streamed
per-expert fallback (MoeCpuHost._act).

The native check reads the activation back through a whole layer: with x = 0, gate bias c and up
bias 1, the down projection's input is the constant vector gelu(c), and every later step (Hadamard,
per-row int8 quantization, GEMM) is exactly linear in it, so out(c) = gelu(c) / gelu(20) * out(20)
to float rounding. The exact-erf form is 11% off the tanh form at c = -3. Far negative gates must
give 0, not inf/NaN, from the worker's vectorized -Ofast epilogue.
"""
import os, sys, platform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_host import MoeCpuHost

GELU = 1   # MoeCpuLayer.activation

# Non-x86 builds stub the native worker out (EXL3_AARCH64_STUB)
native = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64"),
    reason = "CPU MoE worker is x86 only"
)


@pytest.fixture
def cpu_runtime():
    # The native pool pins its caller; restore affinity and torch threads for later tests
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    threads = torch.get_num_threads()
    try:
        yield
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)
        torch.set_num_threads(threads)


def _native_layer_runner():
    g = torch.Generator().manual_seed(1234)
    hid, inter, K = 512, 640, 4

    def trellis(k, n):
        return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.int16, generator = g)

    def signs(n, scale):
        s = torch.randint(0, 2, (n,), generator = g).float() * 2 - 1
        return (s * scale * (1.0 + 0.1 * torch.randn(n, generator = g))).half().contiguous()

    gt, ut, dt = [trellis(hid, inter)], [trellis(hid, inter)], [trellis(inter, hid)]
    gs, us, ds = [signs(hid, 0.015)], [signs(hid, 0.015)], [signs(inter, 0.015)]
    gv, uv, dv = [signs(inter, 1.0)], [signs(inter, 1.0)], [signs(hid, 1.0)]

    def run(c, rows, lim = 0.0):
        # c: constant gate bias, or a gate bias vector
        bg = [torch.full((inter,), c, dtype = torch.half) if isinstance(c, float) else c.half()]
        bu = [torch.ones(inter, dtype = torch.half)]
        h = ext.exl3_moe_cpu_make_layer(gt, gs, gv, ut, us, uv, dt, ds, dv, bg, bu, [], GELU, lim, 0)
        try:
            x = torch.zeros(rows, hid, dtype = torch.half)
            sel = torch.zeros(rows, 1, dtype = torch.int32)
            w = torch.ones(rows, 1, dtype = torch.half)
            out = torch.zeros(rows, hid, dtype = torch.float)
            ext.exl3_moe_cpu_forward(h, x, sel, w, out, 2)
        finally:
            ext.exl3_moe_cpu_free_layer(h)
        return out.double()

    return run, inter


def _readback(out, ref, scale):
    # Activation value from the output, given the output at a known activation value
    return ((out * ref).sum() / (ref * ref).sum() * scale).item()


@native
def test_native_gelu_is_tanh(cpu_runtime):
    run, _ = _native_layer_runner()
    for rows in (1, 2):   # single-token decode finish and the batched path
        ref = run(20.0, rows)
        assert ref.abs().max() > 0
        for c in (-3.0, -2.0, -1.5, -0.5, 1.0):
            out = run(c, rows)
            got = _readback(out, ref, 20.0)
            want = F.gelu(torch.tensor(c, dtype = torch.float64), approximate = "tanh").item()
            erf = F.gelu(torch.tensor(c, dtype = torch.float64)).item()
            assert got == pytest.approx(want, rel = 1e-4), \
                f"rows {rows}, gelu({c}): worker {got:.7e}, tanh {want:.7e}, erf {erf:.7e}"


@native
def test_native_gelu_negative_tail(cpu_runtime):
    run, inter = _native_layer_runner()
    for rows in (1, 2):
        # Uniform far negative gates activate to (almost exactly) 0, with and without act_limit
        for lim in (0.0, 5.0):
            ref = run(20.0, rows, lim)
            assert ref.abs().max() > 0
            for c in (-12.0, -20.0, -100.0):
                got = _readback(run(c, rows, lim), ref, min(20.0, lim or 20.0))
                assert abs(got) < 1e-6, f"rows {rows}, act_limit {lim}, gelu({c}): worker {got}"
        # One far negative gate among ordinary ones must not poison the token
        g = torch.Generator().manual_seed(5)
        bias = torch.randn(inter, generator = g) * 2
        bias[7] = -12.0
        out = run(bias, rows)
        assert torch.isfinite(out).all(), f"rows {rows}: {(~torch.isfinite(out)).sum()} non-finite"


def test_host_gelu_is_tanh():
    g = torch.Generator().manual_seed(0)
    gate = (torch.randn(64, 256, generator = g) * 3).half()
    up = torch.randn(64, 256, generator = g).half()
    for lim in (0.0, 1.5):
        a = MoeCpuHost._act(None, {"activation": GELU, "act_limit": lim}, gate, up)
        av, uf = F.gelu(gate.float(), approximate = "tanh"), up.float()
        if lim:
            av, uf = av.clamp(max = lim), uf.clamp(-lim, lim)
        assert torch.equal(a, (av * uf).half()), f"act_limit {lim}"


if __name__ == "__main__":
    test_native_gelu_is_tanh(None)
    test_native_gelu_negative_tail(None)
    test_host_gelu_is_tanh()
    print("PASS")
