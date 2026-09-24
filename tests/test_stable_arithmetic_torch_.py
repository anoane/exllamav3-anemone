"""
EXL3_STABLE_ARITHMETIC=1: production linear and MLP dispatch executed on CPU tensors, with exact
stand-ins for the native kernels. Checks which path every row count takes, not kernel arithmetic.
No CUDA or compiled extension needed:

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


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
