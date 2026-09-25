"""
GPU, EXL3_STABLE_ARITHMETIC=1: DeepSeek-V4.1 compressor latents do not depend on the chunking.

The production paths are used as they run in the model: the stateless and stateful
compress_chunk (architecture/dsv41/compressor.py) and the cached path's carry ring
(CompressCarry.step, modules/dsv41_cached.py), whose RMSNorm goes to the row-local kernel under
the profile. On every visible GPU, at rates 1 and 2 and widths 128 and 512, latents computed in
chunks of 1 to 145 rows must equal one call over all rows bitwise, and compressor.rms_norm must
match an FP64 oracle on batched, strided, empty, zero, tiny, ordinary and large finite inputs.

    EXL3_STABLE_ARITHMETIC=1 python -m pytest tests/test_dsv41_stable_norm_gpu_.py

Needs CUDA and the compiled extension; skipped without the profile.
"""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class StableCompressorNorm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from exllamav3.model.math_policy import STABLE_ARITHMETIC
        if not STABLE_ARITHMETIC:
            raise unittest.SkipTest("needs EXL3_STABLE_ARITHMETIC=1")
        from exllamav3.architecture.dsv41 import compressor
        from exllamav3.modules.dsv41_cached import CompressCarry
        cls.cp, cls.carry = compressor, CompressCarry

    def test_chunk_and_carry_invariance(self):
        rng = torch.Generator().manual_seed(987)
        for device in range(torch.cuda.device_count()):
            for dim in (128, 512):
                kv = torch.randn((258, dim), generator = rng).to(device) * 5
                gate = torch.randn((258, dim), generator = rng).to(device) * 3
                weight = (1 + torch.randn(dim, generator = rng) * 0.1).half().to(device)
                for ratio in (1, 2):
                    scores = gate if ratio == 2 else None
                    baseline, _ = self.cp.compress_chunk(kv, scores, 0, ratio, weight, 1e-20)
                    for chunk in (1, 7, 9, 17, 33, 128, 145):
                        state = self.cp.CompressorState(dim, ratio)
                        ring = torch.zeros((258, 2 * dim), device = device)
                        stateful, cached = [], []
                        for start in range(0, len(kv), chunk):
                            x = kv[start : start + chunk]
                            s = None if scores is None else scores[start : start + chunk]
                            a, ia = self.cp.compress_chunk(x, s, start, ratio, weight, 1e-20, state)
                            b, ib = self.carry.step(ring, x, s, start, ratio, weight, 1e-20)
                            self.assertEqual(ia, ib)
                            stateful.append(a)
                            cached.append(b)
                        with self.subTest(device = device, dim = dim, ratio = ratio, chunk = chunk):
                            self.assertTrue(torch.equal(torch.cat(stateful), baseline))
                            self.assertTrue(torch.equal(torch.cat(cached), baseline))

    def test_scaled_norm_fp64_oracle_and_layouts(self):
        rng = torch.Generator().manual_seed(388)
        for device in range(torch.cuda.device_count()):
            for dim in (128, 512):
                for scale in (0.0, 1e-30, 1.0, 70000.0, 1e30):
                    # a strided input with a leading batch dimension
                    x = (torch.randn((2, 9, dim * 2), generator = rng) * scale).to(device)[..., ::2]
                    for weight in (None, torch.linspace(0.7, 1.3, dim, device = device).half()):
                        y = self.cp.rms_norm(x, weight, 1e-20)
                        xd = x.cpu().double()
                        ref = xd * torch.rsqrt(xd.square().mean(-1, keepdim = True) + 1e-20)
                        if weight is not None:
                            ref *= weight.cpu().double()
                        with self.subTest(device = device, dim = dim, scale = scale, weight = weight is not None):
                            self.assertEqual(y.dtype, torch.float32)
                            self.assertEqual(y.shape, x.shape)
                            torch.testing.assert_close(y.cpu().double(), ref, rtol = 2e-6,
                                                       atol = 1e-25 if 0 < scale < 1e-20 else 2e-6)
                empty = self.cp.rms_norm(torch.empty((0, dim), device = device), None, 1e-20)
                self.assertEqual(empty.shape, (0, dim))


if __name__ == "__main__":
    unittest.main()
