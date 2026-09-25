"""
GPU: the row-local DeepSeek-V4.1 engram gate of EXL3_STABLE_ARITHMETIC=1
(modules/dsv41_engram_math.py) against an FP64 evaluation of DeepSeek's gate formula, on every
visible GPU:

  * a token's gate does not depend on the other tokens of the call: gates computed in chunks of
    1 to 33 tokens equal one call bitwise, also for strided (sliced) inputs and weights;
  * within 3e-6 of the FP64 oracle for widths 17, 128 and 5120, and for extreme finite inputs
    (zero, tiny, huge, FLT_MAX weights, both ends of the epsilon range), where the result must
    stay finite;
  * an exact zero dot counts as positive whatever the signs of the zero operands, as in the
    reference's torch sum.

The kernel loads by path, without the compiled extension:

    python -m pytest tests/test_dsv41_stable_engram_gpu_.py
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file


def oracle(h, key, qk, eps):
    a, b, w = h.cpu().double(), key.cpu().double(), qk.cpu().double()
    rstd = torch.rsqrt(a.square().mean(-1) + eps) * torch.rsqrt(b.square().mean(-1) + eps)
    dot = (a * w * b).sum(-1) * rstd * h.shape[-1] ** -0.5
    return torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class StableEngramGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gate = staticmethod(load_package_file("exllamav3/modules/dsv41_engram_math.py",
                                                  "_dsv41_engram_math").stable_engram_gate)

    def test_chunk_invariance_and_fp64_oracle(self):
        rng = torch.Generator().manual_seed(791)
        for dev in range(torch.cuda.device_count()):
            for dim in (17, 128, 5120):
                h = torch.randn((2, 65, 4, dim), generator = rng).to(dev)
                kv = torch.randn((2, 65, 5, dim), generator = rng).to(dev)
                key = kv[:, :, :4]                               # a view, as the model passes it
                qk = torch.randn((4, dim), generator = rng).to(dev)
                whole = self.gate(h, key, qk, 1e-20)
                torch.testing.assert_close(whole.cpu().double(), oracle(h, key, qk, 1e-20), rtol = 3e-6, atol = 3e-6)
                for chunk in (1, 7, 9, 17, 33):
                    parts = [self.gate(h[:, p : p + chunk], key[:, p : p + chunk], qk, 1e-20)
                             for p in range(0, 65, chunk)]
                    with self.subTest(device = dev, dim = dim, chunk = chunk):
                        self.assertTrue(torch.equal(torch.cat(parts, dim = 1), whole))
                # feature and learned-weight strides may be views too
                hs = torch.empty((*h.shape[:-1], dim * 2), device = dev)[..., ::2]
                ks = torch.empty((*h.shape[:-1], dim * 2), device = dev)[..., ::2]
                ws = torch.empty((4, dim * 2), device = dev)[:, ::2]
                hs.copy_(h)
                ks.copy_(key)
                ws.copy_(qk)
                with self.subTest(device = dev, dim = dim, layout = "strided"):
                    self.assertTrue(torch.equal(self.gate(hs, ks, ws, 1e-20), whole))

    def test_extreme_finite_inputs_and_empty(self):
        big = torch.finfo(torch.float32).max
        for dev in range(torch.cuda.device_count()):
            for hs, ks, ws in ((0., 0., 1.), (1e-30, 1e-30, 1.), (1e30, 1e30, 1.), (1e30, 1e-30, 1.),
                               (1., 1., 1e30), (1., 1., big), (1e-20, 1e30, big)):
                h = torch.full((1, 9, 4, 128), hs, device = dev)
                k = torch.full_like(h, ks)
                w = torch.full((4, 128), ws, device = dev)
                w[1::2].neg_()
                for eps in (2.0 ** -126, 1e-20, float.fromhex("0x1.fffffep+127")):
                    with self.subTest(device = dev, hs = hs, ks = ks, ws = ws, eps = eps):
                        out = self.gate(h, k, w, eps)
                        self.assertTrue(bool(torch.isfinite(out).all()))
                        torch.testing.assert_close(out.cpu().double(), oracle(h, k, w, eps), rtol = 3e-6, atol = 3e-6)
            empty = torch.empty((1, 0, 4, 128), device = dev)
            self.assertEqual(self.gate(empty, empty, torch.ones((4, 128), device = dev), 1e-20).shape, (1, 0, 4))

    def test_exact_zero_dot_is_positive(self):
        positive = torch.sigmoid(torch.tensor(1e-6, dtype = torch.float64).sqrt())
        for dev in range(torch.cuda.device_count()):
            for hs, ks, ws in ((-0.0, 0.0, 1.0), (0.0, -0.0, 1.0), (-0.0, -0.0, -1.0), (0.0, 0.0, -1.0)):
                h = torch.full((1, 3, 4, 128), hs, device = dev)
                k = torch.full_like(h, ks)
                w = torch.full((4, 128), ws, device = dev)
                with self.subTest(device = dev, hs = hs, ks = ks, ws = ws):
                    out = self.gate(h, k, w, 1e-20)
                    torch.testing.assert_close(out.cpu().double(), oracle(h, k, w, 1e-20), rtol = 0, atol = 1e-7)
                    torch.testing.assert_close(out.cpu().double(), torch.full_like(out.cpu().double(), positive.item()),
                                               rtol = 0, atol = 1e-7)

    def test_refusals(self):
        dev = 0
        h = torch.ones((1, 2, 4, 128), device = dev)
        qk = torch.ones((4, 128), device = dev)
        for args in ((h.cpu(), h.cpu(), qk.cpu(), 1e-20), (h, h[:, :1], qk, 1e-20), (h, h, qk[:2], 1e-20),
                     (h.double(), h.double(), qk.double(), 1e-20), (h, h, qk, float("nan")), (h, h, qk, 1e-50),
                     (h[0], h[0], qk, 1e-20)):
            with self.subTest(case = [tuple(a.shape) if isinstance(a, torch.Tensor) else a for a in args]):
                with self.assertRaises(ValueError):
                    self.gate(*args)


if __name__ == "__main__":
    unittest.main()
