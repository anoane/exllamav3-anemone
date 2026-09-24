"""cuBLAS GEMM precision contract, output layout and CUDA-graph regressions.

`hgemm` always runs cuBLAS. `hgemm_batched` and `hgemm_recon` run cuBLAS unless the FP16-partial
kernel (EXL3_HGEMM_F16ACC) is active on the device, which has its own error contract and is
covered by test_hgemm_f16acc.py; those two entry points are checked here only on devices where
cuBLAS is selected, and test_cublas_coverage_of_batched_and_recon reports the other devices as a
skip. Run with EXL3_HGEMM_F16ACC=0 to cover them on every device, or with EXL3_HGEMM_FIXED_ROWS=128
(or EXL3_STABLE_ARITHMETIC=1, which implies it) to check the same contract through the fixed
128-row tiles (which always use cuBLAS). The tests do not promise bitwise equality between
different matrix shapes or GPU types.

EXL3_HGEMM_FP32_REDUCTION (the policy's switch) is checked in child processes that import the
compiled extension and need no GPU: read once, default 1, only 0 and 1 accepted. A child that
cannot import the extension fails that test, naming the cause.
"""
import os
import subprocess
import sys
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GemmPrecision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from exllamav3.ext import exllamav3_ext
        cls.ext = exllamav3_ext
        cls.devices = list(range(torch.cuda.device_count()))
        cls.cublas_devices = [d for d in cls.devices if cls.ext.hgemm_f16acc_status(d) == 0]
        forced = []
        if os.environ.get("EXL3_HGEMM_F16ACC") == "0":
            forced.append("EXL3_HGEMM_F16ACC=0")
        # the native policy value, so that EXL3_STABLE_ARITHMETIC=1 alone counts too
        if cls.ext.hgemm_fixed_rows() == 128:
            forced.append("fixed-row GEMMs (EXL3_HGEMM_FIXED_ROWS=128 or EXL3_STABLE_ARITHMETIC=1)")
        for what in forced:
            if cls.cublas_devices != cls.devices:
                raise AssertionError(f"{what} did not select cuBLAS on every device")

    def entry_points(self, device):
        fns = [self.ext.hgemm]
        if device in self.cublas_devices:
            fns.append(self.ext.hgemm_recon)
        return fns

    def test_cublas_coverage_of_batched_and_recon(self):
        # Reports, instead of dropping silently, the devices on which the tests below check only
        # hgemm: there the FP16-partial kernel runs hgemm_recon and hgemm_batched. A test of its
        # own, so the report never cuts the other tests short on any runner
        excluded = [d for d in self.devices if d not in self.cublas_devices]
        if excluded:
            self.skipTest(f"hgemm_recon / hgemm_batched not checked on devices {excluded}: "
                          f"EXL3_HGEMM_F16ACC is active there; run with EXL3_HGEMM_F16ACC=0")

    def test_exact_products_strided_outputs_and_batches(self):
        # Small dyadic operands keep products and sums exactly representable in FP32, which
        # permits a zero-tolerance comparison with an independent CPU FP64 reference
        rng = torch.Generator().manual_seed(441)
        for device in self.devices:
            for rows in (1, 7, 127, 128, 129, 513):
                a_cpu = (torch.randint(-8, 9, (2, rows, 256), generator = rng) / 16).half()
                w_cpu = (torch.randint(-8, 9, (2, 256, 128), generator = rng) / 16).half()
                ref = a_cpu.double() @ w_cpu.double()
                a, w = a_cpu.to(device), w_cpu.to(device)
                for dtype in (torch.half, torch.float):
                    with self.subTest(device = device, rows = rows, dtype = dtype):
                        if device in self.cublas_devices:
                            c = torch.empty((2, rows, 128), dtype = dtype, device = device)
                            self.ext.hgemm_batched(a, w, c)
                            torch.testing.assert_close(c.cpu(), ref.to(dtype), rtol = 0, atol = 0)
                        for i in range(2):
                            # Output is a strided view; the padding columns must stay untouched
                            backing = torch.full((rows, 160), -123, dtype = dtype, device = device)
                            view = backing[:, :128]
                            for fn in self.entry_points(device):
                                fn(a[i], w[i], view)
                                torch.testing.assert_close(view.cpu(), ref[i].to(dtype), rtol = 0, atol = 0)
                                self.assertTrue((backing[:, 128:] == -123).all().item())

    def test_large_cancelling_partials(self):
        # Every output is exactly zero, although same-sign partial sums over K exceed the FP16
        # range. An FP16 intermediate reduction (split-K through the output type) would overflow
        # to inf and give NaN. This only exercises the contract where cuBLAS picks a split-K
        # algorithm for the shape on the GPU at hand; elsewhere it passes either way
        for device in self.devices:
            a = torch.ones((2, 512, 4096), dtype = torch.half, device = device)
            w = torch.full((2, 4096, 128), 512, dtype = torch.half, device = device)
            w[:, 2048:].neg_()
            for dtype in (torch.half, torch.float):
                with self.subTest(device = device, dtype = dtype):
                    c = torch.empty((2, 512, 128), dtype = dtype, device = device)
                    if device in self.cublas_devices:
                        self.ext.hgemm_batched(a, w, c)
                        self.assertTrue(torch.isfinite(c).all().item())
                        self.assertEqual(torch.count_nonzero(c).item(), 0)
                    for fn in self.entry_points(device):
                        c[0].fill_(1)
                        fn(a[0], w[0], c[0])
                        self.assertTrue(torch.isfinite(c[0]).all().item())
                        self.assertEqual(torch.count_nonzero(c[0]).item(), 0)

    def test_graph_replay_and_current_stream(self):
        # Correctness of repeated calls on a side stream and of CUDA graph replay (the math mode
        # is set on PyTorch's borrowed cuBLAS handle for each call). The operands are small
        # integers whose partial sums are exact in FP16, so this cannot see the reduction policy,
        # nor a missing restore of the handle's math mode (PyTorch sets the mode again whenever
        # it hands out the handle); test_large_cancelling_partials covers the policy
        for device in self.devices:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device = device)
                a = torch.ones((129, 256), dtype = torch.half, device = device)
                w = torch.ones((256, 128), dtype = torch.half, device = device)
                c = torch.empty((129, 128), dtype = torch.half, device = device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.ext.hgemm(a, w, c)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream = stream):
                    self.ext.hgemm(a, w, c)
                for value in (1, 2, 3):
                    with torch.cuda.stream(stream):
                        a.fill_(value)
                        graph.replay()
                    stream.synchronize()
                    self.assertTrue((c == value * 256).all().item())


class ReductionSwitch(unittest.TestCase):
    """EXL3_HGEMM_FP32_REDUCTION, through exllamav3_ext.hgemm_fp32_reduction() in a fresh process"""

    CHILD = "\n".join([
        "import os, torch",
        "from exllamav3.ext import exllamav3_ext as ext",
        "first = ext.hgemm_fp32_reduction()",
        "os.environ['EXL3_HGEMM_FP32_REDUCTION'] = '0' if first else '1'",
        "print(first, ext.hgemm_fp32_reduction())",
    ])

    def child(self, value):
        env = {k: v for k, v in os.environ.items() if k != "EXL3_HGEMM_FP32_REDUCTION"}
        if value is not None:
            env["EXL3_HGEMM_FP32_REDUCTION"] = value
        return subprocess.run([sys.executable, "-c", self.CHILD], env = env, capture_output = True,
                              text = True, timeout = 600)

    def test_values_read_once(self):
        # the second call must return the first value although the variable changed in between
        for value, expected in ((None, "1 1"), ("1", "1 1"), ("0", "0 0")):
            with self.subTest(value = value):
                r = self.child(value)
                self.assertEqual(r.returncode, 0,
                                 f"test_hgemm_precision_.py::ReductionSwitch: the child process with "
                                 f"EXL3_HGEMM_FP32_REDUCTION={value!r} failed (an extension without "
                                 f"hgemm_fp32_reduction, or one that does not import, fails here):\n{r.stderr}")
                self.assertEqual(r.stdout.strip(), expected)

    def test_other_values_refused(self):
        for value in ("", "2", "true", "01", " 1"):
            with self.subTest(value = value):
                r = self.child(value)
                self.assertNotEqual(r.returncode, 0, r.stdout)
                self.assertIn("EXL3_HGEMM_FP32_REDUCTION must be 0 or 1", r.stderr)


if __name__ == "__main__":
    unittest.main()
