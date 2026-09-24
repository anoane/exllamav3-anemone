"""CPU-only projection-boundary and all-code E8M0 regressions; no extension import.

    python tests/test_dsv41_precision_.py
"""

import ast
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv41_ref import engram_tables


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compressor_class():
    # Execute the actual constructor/forward without importing CUDA module plumbing.
    package = "precision_stub"
    for suffix in ("", ".modules", ".architecture", ".architecture.dsv41"):
        module = types.ModuleType(package + suffix)
        module.__path__ = []
        sys.modules[module.__name__] = module
    math = load_file(package + ".architecture.dsv41.compressor",
                     "exllamav3/architecture/dsv41/compressor.py")

    class Linear:
        def __init__(self, config, key, hidden, width, out_dtype=None, **kw):
            self.out_dtype = out_dtype
            scale = 500.0 if key.endswith("wgate") else 200.0
            self.weight = torch.full((width, hidden), scale)
        def forward(self, x, params):
            return torch.nn.functional.linear(x.float(), self.weight).to(self.out_dtype)

    class RMSNorm:
        def __init__(self, config, key, eps):
            self.weight = torch.ones(4)

    path = ROOT / "exllamav3/modules/dsv41.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "DSV41Compressor")
    ns = dict(torch=torch, Linear=Linear, RMSNorm=RMSNorm, __package__=package + ".modules")
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    return ns["DSV41Compressor"], math


def check_projection_range():
    cls, math = compressor_class()
    x = torch.tensor([[200, 200], [201, 200]], dtype=torch.half)
    for ratio in (1, 2):
        comp = cls(None, "compressor", hidden_size=2, head_dim=4, compress_ratio=ratio)
        result, _ = comp.forward(x, {}, 0)
        assert comp.wkv.out_dtype == torch.float and result.dtype == torch.float
        assert torch.isfinite(result).all() and torch.allclose(result, torch.ones_like(result))
        if comp.wgate is not None:
            assert comp.wgate.out_dtype == torch.float
            raw = comp.wkv.forward(x, {})
            score = comp.wgate.forward(x, {})
            broken = math.compress_chunk(raw.half().float(), score.half().float(), 0, ratio)[0]
            assert not torch.isfinite(broken).all(), "overflow negative control did not trigger"
    scores = torch.tensor([[[1000.0], [1000.25]]])
    values = torch.tensor([[[1.0], [-1.0]]])
    assert abs(float(math.group_pool(values, scores)) - float(math.group_pool(values, scores.half().float()))) > 0.1


def check_e8m0():
    codes = np.arange(256, dtype=np.uint8)
    actual = engram_tables.e8m0_to_f32(codes)
    expected = np.exp2(codes[:255].astype(np.int32) - 127).astype(np.float32)
    assert np.array_equal(actual[:255], expected) and np.isnan(actual[255])
    # Load the actual production decoder with only its hash-layout import stubbed.
    module = types.ModuleType("precision_engram_layout")
    module.compute_hash_multipliers = None
    sys.modules["precision_stub.architecture.dsv41.engram"] = module
    prod = load_file("precision_stub.architecture.dsv41.engram_torch",
                     "exllamav3/architecture/dsv41/engram_torch.py")
    scales = torch.tensor([[0], [116], [123], [255]], dtype=torch.uint8)
    # 0x38 is +1 in E4M3; NaN scale must stay NaN, never become infinity.
    weights = torch.full((4, 32), 0x38, dtype=torch.uint8)
    out = prod.dequant_rows(weights, scales)
    assert torch.equal(out[0], torch.zeros(32, dtype=torch.half))
    assert out[1, 0] == 2.0 ** -11 and out[2, 0] == 2.0 ** -4
    assert torch.isnan(out[3]).all()


def check_norm_range():
    _, math = compressor_class()
    torch.manual_seed(123)
    for scale in (0.0, 1e-30, 1e-12, 1.0, 70000.0, 1e30):
        x = torch.randn(7, 512) * scale
        weight = 1 + torch.randn(512) * .1
        expected = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-20) * weight.double()
        actual = math.rms_norm(x, weight, 1e-20)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, rtol=1e-6,
                                   atol=1e-25 if 0 < scale < 1e-20 else 1e-6)
    for eps in (0.0, -1.0, 1e-50, 1e100, float("nan"), float("inf")):
        try:
            math.rms_norm(x, weight, eps)
        except ValueError:
            continue
        raise AssertionError("invalid normalization epsilon accepted")


def check_extreme_convex_pool():
    _, math = compressor_class()
    limit = torch.finfo(torch.float32).max
    pairs = [(limit, limit), (-limit, -limit), (limit, -limit), (-limit, limit),
             (1e-30, limit), (limit, 1e-30), (-1e-30, -limit), (-limit, -1e-30)]
    kv = torch.tensor(pairs, dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 32).contiguous()
    kv[:, :, 1::2] *= .5
    for ga, gb in ((0., -17.328678), (0., 0.), (-50., 50.), (50., -50.), (limit, -limit)):
        gate = torch.zeros_like(kv)
        gate[:, 0] = ga
        gate[:, 1] = gb
        expected = (kv.double() * gate.double().softmax(1)).sum(1)
        actual = math.group_pool(kv, gate)
        assert torch.isfinite(actual).all(), "a finite convex pair overflowed"
        torch.testing.assert_close(actual.double(), expected, rtol=3e-7, atol=1e-30)
        for eps in (2.0 ** -126, 1e-20, limit):
            normalized = math.rms_norm(actual, None, eps)
            ref = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + eps)
            assert torch.isfinite(normalized).all()
            torch.testing.assert_close(normalized.double(), ref, rtol=1e-6, atol=1e-7)


# pytest entry points: the same checks

def test_projection_range():
    check_projection_range()


def test_e8m0():
    check_e8m0()


def test_norm_range():
    check_norm_range()


def test_extreme_convex_pool():
    check_extreme_convex_pool()


if __name__ == "__main__":
    check_projection_range()
    check_e8m0()
    check_norm_range()
    check_extreme_convex_pool()
    print("OK compressor projection range/rounding and all E8M0 encodings")
