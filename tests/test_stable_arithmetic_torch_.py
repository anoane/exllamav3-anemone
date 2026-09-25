"""
EXL3_STABLE_ARITHMETIC=1: production linear, MLP, DeepSeek-V4.1 mHC mixing and grouped
o-projection dispatch executed on CPU tensors, with exact stand-ins for the native kernels.
Checks which path every row count takes, not kernel arithmetic. No CUDA or compiled extension
needed:

    python -m pytest tests/test_stable_arithmetic_torch_.py
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

spec = importlib.util.spec_from_file_location("stable_helpers", Path(__file__).with_name("test_stable_arithmetic_.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

REJECT = Mock(side_effect = AssertionError("small-row shortcut used under stable arithmetic"))


class StableDispatchTests(unittest.TestCase):

    def test_quantized_linear_original_basis_at_every_row_count(self):
        ext = NS(reconstruct_had_slice = Mock(side_effect = lambda w, *a: w.fill_(0.125)),
                 had_r_128 = Mock(side_effect = AssertionError("rotated basis used")),
                 hgemm_recon = Mock(side_effect = lambda x, w, y: y.copy_(x.float() @ w.float())))
        ns = dict(torch = torch, STABLE_ARITHMETIC = True, no_fused_reconstruct = False,
                  MAX_RECONSTRUCT_SLICE_N = 32768, ext = ext)
        forward = helpers.method("exllamav3/modules/quant/exl3.py", "LinearEXL3", "reconstruct_hgemm", ns)
        layer = NS(in_features = 128, out_features = 128, default_out_dtype = torch.half, _fused_reconstruct = None,
                   trellis = torch.empty(0), suh = None, svh = None, K = 3, mcg = False, mul1 = False, bias = None)
        rows_list = (1, 7, 8, 9, 128, 144, 145, 257, 1023, 1024)
        for rows in rows_list:
            y = forward(layer, torch.ones((1, rows, 128), dtype = torch.half), None)
            self.assertEqual(y.shape, (1, rows, 128))
            self.assertTrue(bool((y == 16).all()))
        self.assertEqual(ext.reconstruct_had_slice.call_count, len(rows_list))
        ext.had_r_128.assert_not_called()

    def test_gated_mlp_takes_the_linears_at_every_row_count(self):
        ns = dict(torch = torch, STABLE_ARITHMETIC = True, MAX_BSZN = 8, to2 = lambda x, *a: x)
        forward = helpers.method("exllamav3/modules/mlp.py", "GatedMLP", "forward", ns)
        layer = NS(num_slices = 1, bc = NS(run_bszN = REJECT), multi_gu = [object()], out_dtype = torch.half,
                   gates = [NS(forward = lambda x, p: x * 2)], ups = [NS(forward = lambda x, p: x * 3)],
                   downs = [NS(forward = lambda x, p: x.clone())], interm_dtype = torch.half, interm_div = 1.0,
                   act_limit = 0,
                   activation_fn_call = lambda g, u, a, lim: a.copy_(g * u), tp_reduce = False)
        for rows in (1, 7, 8, 9, 17, 32, 33, 128):
            y = forward(layer, torch.ones((1, rows, 16), dtype = torch.half), {})
            self.assertTrue(bool((y == 6).all()))
        REJECT.assert_not_called()

    def test_mlp_decode_takes_the_linears(self):
        ns = dict(torch = torch, STABLE_ARITHMETIC = True, to2 = lambda x, *a: x)
        forward = helpers.method("exllamav3/modules/mlp.py", "MLP", "forward", ns)
        layer = NS(num_slices = 1, bc = NS(run_bsz1 = REJECT), out_dtype = torch.half, interm_dtype = torch.half,
                   ups = [NS(forward = lambda x, p: x * 3)], downs = [NS(forward = lambda x, p: x * 2)],
                   activation_fn_call = lambda u: u.clone(), tp_reduce = False)
        y = forward(layer, torch.ones((1, 1, 16), dtype = torch.half), {})
        self.assertTrue(bool((y == 6).all()))
        REJECT.assert_not_called()
        # Default dispatch: single-token decode takes the native block
        ns["STABLE_ARITHMETIC"] = False
        forward = helpers.method("exllamav3/modules/mlp.py", "MLP", "forward", ns)
        layer.bc = NS(run_bsz1 = Mock(side_effect = lambda x, d: d.fill_(6)))
        y = forward(layer, torch.ones((1, 1, 16), dtype = torch.half), {})
        layer.bc.run_bsz1.assert_called_once()
        self.assertTrue(bool((y == 6).all()))


class DeepseekV41DispatchTests(unittest.TestCase):
    """DeepSeek-V4.1 hyper-connection mixing and the grouped attention output projection"""

    @staticmethod
    def mix_fused(stable, chunks):
        calls = []

        def hc_mix(*args):
            partials, post, comb = args[7:10]
            calls.append(partials.shape[1])
            for chunk in range(partials.shape[1]):
                partials[:, chunk].fill_(0.01 * (chunk + 1))
            post.fill_(2)
            comb.fill_(3)

        ext = NS(hc_mix_num_chunks = Mock(return_value = chunks), hc_mix = hc_mix)
        ns = dict(torch = torch, STABLE_ARITHMETIC = stable, ext = ext,
                  g_tensor_cache = NS(get_bucketed = lambda dev, n, dtype, tag: torch.empty(n, dtype = dtype, device = dev)))
        forward = helpers.method("exllamav3/modules/dsv41_block.py", "DSV41HyperConnection", "mix_fused", ns)
        layer = NS(fn = torch.zeros(24, 512), base = torch.zeros(24), scale = torch.ones(3),
                   rms_eps = 1e-6, hc_eps = 1e-6, sinkhorn_iters = 20)
        return lambda rows: forward(layer, torch.ones((1, rows, 4, 128))), ext, calls

    def test_mhc_one_partition_at_every_row_count(self):
        forward, ext, calls = self.mix_fused(True, 5)
        reference = None
        for rows in (1, 7, 17, 32, 33, 128, 145, 256, 600):
            first = [v[:, 0] for v in forward(rows)]
            if reference is None:
                reference = first
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference, first)), rows)
        ext.hc_mix_num_chunks.assert_not_called()
        self.assertEqual(set(calls), {1})
        p = torch.tensor(0.01)
        self.assertTrue(torch.allclose(reference[2], torch.sigmoid(p * torch.rsqrt(p / 512 + 1e-6)) + 1e-6))

    def test_mhc_default_partition_follows_the_row_count(self):
        forward, ext, calls = self.mix_fused(False, 3)
        post, comb, pre = forward(7)
        ext.hc_mix_num_chunks.assert_called_once_with(7, 512)
        self.assertEqual(calls, [3])
        # the pre-mix is derived from the sum of the kernel's three partial dots
        p = torch.tensor(0.01) + 0.02 + 0.03
        self.assertTrue(torch.allclose(pre, torch.sigmoid(p * torch.rsqrt(p / 512 + 1e-6)) + 1e-6))

    @staticmethod
    def project_o_grouped(stable):
        def mgemm(a, trellis, c, *args):
            c.copy_(a * 2)

        ext = NS(exl3_mgemm = Mock(side_effect = mgemm))
        ns = dict(torch = torch, STABLE_ARITHMETIC = stable, ext = ext,
                  g_tensor_cache = NS(get = lambda dev, shape, dtype, tag: torch.empty(shape, dtype = dtype)))
        forward = helpers.method("exllamav3/modules/dsv4.py", "DSV4Attention", "_project_o_grouped", ns)
        grouped = NS(out_features = 4, ptrs_trellis = None, ptrs_suh = None, ptrs_svh = None, K = 3,
                     mcg = False, mul1 = False)
        per_group = Mock(side_effect = lambda x, p: x * 2)
        layer = NS(o_groups = 2, woa_multi_ready = True, device = None, wo_a_multi = grouped,
                   woa_indices = None, wob_multi = None, out_dtype = torch.half,
                   wo_a = [NS(forward = per_group) for _ in range(2)],
                   wo_b = NS(forward = lambda x, p, out_dtype: x))
        return lambda rows: forward(layer, torch.ones((2, 1, rows, 4)), {}, None), ext, per_group

    def test_grouped_o_projection_per_group_at_every_row_count(self):
        forward, ext, per_group = self.project_o_grouped(True)
        for rows in (1, 7, 17, 32, 33, 128):
            output = forward(rows)
            self.assertEqual(output.shape, (1, rows, 8))
            self.assertTrue(bool((output == 2).all()))
        ext.exl3_mgemm.assert_not_called()
        self.assertEqual(per_group.call_count, 12)

    def test_grouped_o_projection_default_uses_the_grouped_gemm_up_to_32_rows(self):
        forward, ext, per_group = self.project_o_grouped(False)
        for rows in (1, 7, 32, 33, 128):
            output = forward(rows)
            self.assertEqual(output.shape, (1, rows, 8))
            self.assertTrue(bool((output == 2).all()))
        self.assertEqual(ext.exl3_mgemm.call_count, 3)
        self.assertEqual(per_group.call_count, 4)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
