"""
EXL3_DSV41_FUSED_COMPRESS: the fused FP32 pooling of DeepSeek-V4.1's rate-2 kv sources, on the CPU.

  * the layer state allocates the fused kernel's two FP32 rings instead of the torch carry ring
    when the switch is on, and nothing else changes (DSV41LayerState.__init__, compiled from the
    source);
  * the cached compression takes the kernel exactly when the state holds those rings
    (DSV41Attention._compress_store, compiled from the source and run against stand-ins);
  * the kernel itself (modules/dsv41_compress.py, _pool_pairs_norm_kernel) on Triton's CPU
    interpreter, in a child process with TRITON_INTERPRET=1: against an FP64 oracle, chunked
    calls bitwise equal to one call, a rewind replayed from the ring, the default torch pooling
    (CompressCarry.step) within the oracle's tolerance, and extreme finite pairs. Skipped where Triton is not
    installed. The GPU test (test_dsv41_compress_gpu_.py) runs the wrapper and the device
    assertions on real devices.

    python -m pytest tests/test_dsv41_fused_compress_.py
"""

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file

spec = importlib.util.spec_from_file_location("stable_helpers", Path(__file__).with_name("test_stable_arithmetic_.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def layer_state(fused, m, source = True):
    ns = dict(torch = torch, PAGE_SIZE = 256, FUSED_COMPRESS = fused)
    init = helpers.method("exllamav3/cache/dsv41.py", "DSV41LayerState", "__init__", ns)
    module = types.SimpleNamespace(head_dim = 512, rope_head_dim = 64, layer_type = "compressed",
                                   sliding_window = 128, compress_ratio = m, is_kv_source = source)
    state = types.SimpleNamespace()
    init(state, module, 2, 4096, 0)
    return state


def test_layer_state_rings_follow_the_switch():
    default, fused = layer_state(False, 2), layer_state(True, 2)
    assert default.comp_carry.shape == (2, 258, 1024) and default.comp_carry.dtype == torch.float
    assert default.comp_buf_kv is None and default.comp_buf_gate is None
    assert fused.comp_carry is None
    for ring in (fused.comp_buf_kv, fused.comp_buf_gate):
        assert ring.shape == (2, 258, 512) and ring.dtype == torch.float
    assert default.buf_rows == fused.buf_rows == 258
    size = lambda s: sum(t.numel() * t.element_size() for t in (s.comp_carry, s.comp_buf_kv, s.comp_buf_gate)
                         if t is not None)
    assert size(default) == size(fused)
    # rate-1 sources and the layers that only read a pool carry nothing either way
    for fused_ in (False, True):
        for state in (layer_state(fused_, 1), layer_state(fused_, 2, source = False)):
            assert state.comp_carry is None and state.comp_buf_kv is None and state.buf_rows == 0


def test_cached_compression_follows_the_rings():
    calls = []
    D, seq, pos0, slot = 8, 5, 3, 1

    def fused_compress(kv, gate, pos0_, weight, eps, ck, cg):
        calls.append(("fused", kv.dtype, gate.dtype, tuple(kv.shape), pos0_, ck.data_ptr(), cg.data_ptr()))
        return torch.zeros(((pos0_ + kv.shape[0]) // 2 - pos0_ // 2, D)), pos0_ // 2

    def step(carry, kv, score, pos0_, m, weight, eps, per_row):
        # per_row is EXL3_EXACT_ROWS' per-entry pooling: off here, and never asked for without the switch
        assert per_row is False
        calls.append(("torch", None if carry is None else carry.data_ptr(), m))
        return torch.zeros(((pos0_ + kv.shape[0]) // m - pos0_ // m, D)), pos0_ // m

    ns = dict(torch = torch, fused_compress = fused_compress, CompressCarry = types.SimpleNamespace(step = step),
              emission_range = lambda p, s, m: (p // m, (p + s) // m), EXACT_ROWS = False)
    store = helpers.method("exllamav3/modules/dsv41.py", "DSV41Attention", "_compress_store", ns)
    projection = lambda: types.SimpleNamespace(forward = lambda x, p: torch.ones((1, seq, D), dtype = torch.float))
    stored = []
    attn = types.SimpleNamespace(
        compressor = types.SimpleNamespace(wkv = projection(), wgate = projection(), rms_norm_eps = 1e-20,
                                           norm = types.SimpleNamespace(weight = torch.ones(D))),
        compress_ratio = 2, _store_entries = lambda latent, e0, *a: stored.append((latent.shape[0], e0)))
    x = torch.zeros((1, seq, 16), dtype = torch.half)

    rings = torch.zeros((2, 2, 258, D))
    fused_state = types.SimpleNamespace(comp_buf_kv = rings[0], comp_buf_gate = rings[1], comp_carry = None)
    store(attn, x, {}, fused_state, slot, None, None, pos0)
    assert calls == [("fused", torch.float, torch.float, (seq, D), pos0,
                      rings[0][slot].data_ptr(), rings[1][slot].data_ptr())]
    calls.clear()
    carry = torch.zeros((2, 258, 2 * D))
    torch_state = types.SimpleNamespace(comp_buf_kv = None, comp_buf_gate = None, comp_carry = carry)
    store(attn, x, {}, torch_state, slot, None, None, pos0)
    assert calls == [("torch", carry[slot].data_ptr(), 2)]
    # rate 1 never pools: the torch path without a ring, whatever the state holds
    calls.clear()
    attn.compress_ratio = 1
    store(attn, x, {}, fused_state, slot, None, None, pos0)
    assert calls == [("torch", None, 1)]
    assert stored == [(3, 1), (3, 1), (seq, pos0)]


def test_kernel_on_the_triton_interpreter():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    r = subprocess.run([sys.executable, __file__, "--interpret"], capture_output = True, text = True, env = env,
                       timeout = 1800)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr


# ---- Triton interpreter (child process) ----

def pair_oracle(kv, gate, weight, eps):
    x = (kv.double().view(-1, 2, kv.shape[-1]) * gate.double().view(-1, 2, kv.shape[-1]).softmax(1)).sum(1)
    return x * torch.rsqrt(x.square().mean(-1, keepdim = True) + eps) * weight.double()


def interpret():
    import triton
    kernels = load_package_file("exllamav3/modules/dsv41_compress.py", "_dsv41_compress_interpreted")
    assert type(kernels._pool_pairs_norm_kernel).__name__ == "InterpretedFunction"

    def fused(kv, gate, pos0, weight, eps, ck, cg):
        # fused_compress's launch and ring update, on CPU tensors (the wrapper itself refuses
        # them; the GPU test runs it)
        rows, dim = kv.shape
        first = pos0 // 2
        out = torch.empty(((pos0 + rows) // 2 - first, dim), dtype = torch.float32)
        if rows:
            kernels._pool_pairs_norm_kernel[(triton.cdiv(pos0 % 2 + rows, 2),)](
                kv, gate, ck, cg, weight, out, pos0, rows, ck.shape[0], dim, triton.next_power_of_2(dim),
                math.sqrt(eps), enable_fp_fusion = False)
            start = max(0, rows - ck.shape[0])
            indices = torch.arange(pos0 + start, pos0 + rows) % ck.shape[0]
            ck.index_copy_(0, indices, kv[start:])
            cg.index_copy_(0, indices, gate[start:])
        return out, first

    pkg = "_fused_compress_pkg"
    for name in (pkg, f"{pkg}.architecture", f"{pkg}.architecture.dsv41", f"{pkg}.model", f"{pkg}.modules",
                 f"{pkg}.util"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    constants = types.ModuleType(f"{pkg}.constants")
    constants.PAGE_SIZE = 256
    sys.modules[constants.__name__] = constants
    load_package_file("exllamav3/util/device_copy.py", f"{pkg}.util.device_copy", f"{pkg}.util")
    load_package_file("exllamav3/model/math_policy.py", f"{pkg}.model.math_policy", f"{pkg}.model")
    load_package_file("exllamav3/architecture/dsv41/compressor.py", f"{pkg}.architecture.dsv41.compressor",
                      f"{pkg}.architecture.dsv41")
    cached = load_package_file("exllamav3/modules/dsv41_cached.py", f"{pkg}.modules.dsv41_cached", f"{pkg}.modules")

    g = torch.Generator().manual_seed(1234)
    worst = 0.0
    for dim in (128, 512):
        n = 517
        weight = 1 + torch.randn(dim, generator = g) * 0.1
        for scale in (0.0, 1e-30, 1e-12, 1.0, 70000.0, 1e30):
            kv = torch.randn(n, dim, generator = g) * scale
            gate = torch.randn(n, dim, generator = g) * 1000
            gate[::2, :4] = torch.tensor([1000.0, 70000.0, -70000.0, 0.0])
            gate[1::2, :4] = torch.tensor([1000.25, 70000.25, 70000.0, 0.0])
            ring = lambda: (torch.zeros(258, dim), torch.zeros(258, dim))
            whole, first = fused(kv, gate, 0, weight, 1e-20, *ring())
            assert first == 0 and whole.shape == (n // 2, dim)
            reference = pair_oracle(kv[:n // 2 * 2], gate[:n // 2 * 2], weight, 1e-20)
            atol = 1e-25 if 0 < scale < 1e-20 else 3e-5
            torch.testing.assert_close(whole.double(), reference, rtol = 3e-5, atol = atol)
            assert torch.isfinite(whole).all()
            worst = max(worst, (whole.double() - reference).abs().max().item())

            # chunked, odd boundaries included: bitwise equal to one call
            ck, cg = ring()
            carry = torch.zeros(258, 2 * dim)
            pos, parts, parts_torch = 0, [], []
            for rows in (1, 2, 7, 257, 243, 7):
                end = min(pos + rows, n)
                got, first = fused(kv[pos:end].contiguous(), gate[pos:end].contiguous(), pos, weight, 1e-20, ck, cg)
                assert first == pos // 2
                parts.append(got)
                lat, _ = cached.CompressCarry.step(carry, kv[pos:end], gate[pos:end], pos, 2, weight, 1e-20)
                parts_torch.append(lat)
                pos = end
            assert pos == n
            assert torch.equal(torch.cat(parts), whole), f"dim {dim} scale {scale}: chunked != one call"
            # the default torch pooling computes the same function, with the same safeguards, to the
            # oracle's tolerance (the exponentials differ in the last bits, which cancellation in an
            # opposite-sign pair magnifies on single near-zero results)
            torch.testing.assert_close(whole, torch.cat(parts_torch), rtol = 3e-5, atol = atol)
            # the rings hold the last 258 raw rows at their positions
            rows_ = torch.arange(n - 258, n) % 258
            assert torch.equal(ck[rows_], kv[-258:]) and torch.equal(cg[rows_], gate[-258:])
            # a rewind to an odd position replays from the ring
            rewind = n - 257
            replay, first = fused(kv[rewind:].contiguous(), gate[rewind:].contiguous(), rewind, weight, 1e-20, ck, cg)
            assert torch.equal(replay, whole[first:]), f"dim {dim} scale {scale}: rewind replay"
    print(f"  OK  fused pooling on the interpreter: FP64 oracle max_abs {worst:.3g}, chunked and rewound "
          f"calls bitwise equal to one call, torch pooling within the oracle's tolerance")

    limit = torch.finfo(torch.float32).max
    pairs = [(limit, limit), (-limit, -limit), (limit, -limit), (-limit, limit),
             (1e-30, limit), (limit, 1e-30), (-1e-30, -limit), (-limit, -1e-30)]
    kv = torch.tensor(pairs, dtype = torch.float32).reshape(-1, 1).expand(-1, 128).contiguous()
    kv[:, 1::2] *= 0.5
    weight = torch.ones(128)
    for ga, gb in ((0., -17.328678), (0., 0.), (-50., 50.), (50., -50.), (limit, -limit)):
        gate = torch.empty_like(kv)
        gate[::2], gate[1::2] = ga, gb
        for eps in (2.0 ** -126, 1e-20, limit):
            ck, cg = torch.zeros(258, 128), torch.zeros(258, 128)
            outputs, pos = [], 0
            for end in (1, 6, len(kv)):
                outputs.append(fused(kv[pos:end].contiguous(), gate[pos:end].contiguous(), pos, weight, eps, ck, cg)[0])
                pos = end
            got = torch.cat(outputs)
            assert torch.isfinite(got).all(), "a finite convex pair overflowed"
            torch.testing.assert_close(got.double(), pair_oracle(kv, gate, weight, eps), rtol = 3e-5, atol = 3e-5)
    print("  OK  fused pooling on the interpreter: extreme finite pairs finite and within the oracle")


if __name__ == "__main__":
    if "--interpret" in sys.argv:
        interpret()
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-q"]))
