"""
Deterministic int8 router projection (ext.routing_gemm_det, routing_gemm.cu + det_gemm.cuh): must
match an fp32 reference to half-output tolerance over row counts, expert counts (including
non-multiples of the tile) and K values, be bit-reproducible, agree bit for bit across every
visible GPU (the point of the int8 scheme: fp16 tensor cores do not across architectures), and
feed ext.routing_std so that multi-row routing selects the same experts as the reference on rows
that are not near-ties. Also checks the FMA-only transcendentals the routing activations use.

Under EXL3_STABLE_ARITHMETIC=1 (set for the whole process) two more cases run: a row's logits do
not depend on how many rows share the call, and the full router takes the same projection for a
single row; by default the single-row case checks that one row still takes the FMA GEMV. A source
check (no GPU) pins that both row-count dependences stay gated on the profile.
"""
import sys, os, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

DEVICE = torch.device("cuda:0")
STABLE = os.environ.get("EXL3_STABLE_ARITHMETIC") == "1"


def quant_gate(gate):
    gate_t = gate.T.contiguous()
    E, K = gate_t.shape
    g8 = torch.empty((2, E, K), dtype = torch.int8, device = gate.device)
    sb = torch.empty((E,), dtype = torch.float, device = gate.device)
    ext.det_quant_weight(gate_t, g8, sb)
    return gate_t, g8, sb


class TestRowCountPolicySource(unittest.TestCase):
    """
    The router's row-count dependences stay upstream's by default, checked on the native source (no
    GPU): rg_slices counts the row tiles unless EXL3_STABLE_ARITHMETIC, and a single row takes the
    GEMV only outside the profile. No default-environment GPU test compares two row counts, so a
    slice change made unconditional would otherwise pass them all
    """

    def test_row_count_dependences_are_gated(self):
        native = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "exllamav3_ext")
        with open(os.path.join(native, "routing_gemm.cu")) as f:
            slices = f.read().split("static int rg_slices", 1)[1].split("\n}", 1)[0]
        self.assertIn("CEIL_DIVIDE(E, RG_BN) * (stable_arithmetic() ? 1 : CEIL_DIVIDE(R, RG_BM))", slices)
        with open(os.path.join(native, "routing.cu")) as f:
            dispatch = f.read()
        self.assertIn("if ((!bsz1 || stable_arithmetic()) && gate_i8.has_value()", dispatch)
        self.assertIn("else if (bsz1 && !stable_arithmetic() && gate_t.has_value()", dispatch)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRoutingGemmDet(unittest.TestCase):

    def test_policy_matches_environment(self):
        self.assertEqual(ext.stable_arithmetic(), STABLE)

    @unittest.skipUnless(STABLE, "needs EXL3_STABLE_ARITHMETIC=1")
    def test_same_rows_across_row_counts(self):
        # Same-shape repeats and cross-device checks do not detect a reduction that changes with
        # the row count. Hold inputs and weights fixed while crossing the static-workspace, row-tile
        # and default split-K boundaries
        rng = torch.Generator().manual_seed(841)
        for K, E in ((144, 40), (1040, 200), (2560, 128), (5120, 384)):
            x = torch.randn((4096, K), generator = rng).half()
            gate = (torch.randn((K, E), generator = rng) / K ** 0.5).half()
            reference = None
            for device in range(torch.cuda.device_count()):
                dev = torch.device("cuda", device)
                xd = x.to(dev)
                _, g8, sb = quant_gate(gate.to(dev))
                for rows in (4096, 2048, 257, 129, 65, 64, 17, 2, 1):
                    out = torch.empty((rows, E), dtype = torch.half, device = dev)
                    ext.routing_gemm_det(xd[:rows], g8, sb, out)
                    prefix = out[:min(rows, 37)].cpu()
                    if reference is None:
                        reference = prefix
                    self.assertTrue(torch.equal(prefix, reference[:len(prefix)]), (K, E, device, rows))

    def test_one_row_dispatch(self):
        # Full router (routing_ds3_nogroup) on one row. Default: the FMA GEMV over the fp16 weights,
        # the same with and without the int8 gate. EXL3_STABLE_ARITHMETIC: the int8 projection of
        # multi-row calls, so each row decoded alone equals the same row in a 257-row call, logits,
        # selection and weights
        rng = torch.Generator().manual_seed(482)
        x = torch.randn((257, 5120), generator = rng).half()
        gate = (torch.randn((5120, 384), generator = rng) / 5120 ** 0.5).half()

        def route(xr, gd, gt, bias, g8, sb):
            rows = xr.shape[0]
            logits = torch.empty((rows, 384), dtype = torch.half, device = xr.device)
            sel = torch.empty((rows, 6), dtype = torch.long, device = xr.device)
            weights = torch.empty((rows, 6), dtype = torch.half, device = xr.device)
            ext.routing_ds3_nogroup(xr, gd, logits, bias, sel, weights, 1.5, gt, 1, g8, sb)
            return logits, sel, weights

        for device in range(torch.cuda.device_count()):
            dev = torch.device("cuda", device)
            xd, gd = x.to(dev), gate.to(dev)
            gt, g8, sb = quant_gate(gd)
            bias = torch.zeros((384,), dtype = torch.half, device = dev)
            if not STABLE:
                for row in range(8):
                    with_i8 = route(xd[row:row + 1], gd, gt, bias, g8, sb)
                    without = route(xd[row:row + 1], gd, gt, bias, None, None)
                    self.assertTrue(torch.equal(with_i8[0], without[0]), (device, row))
                continue
            ref = torch.empty((257, 384), dtype = torch.half, device = dev)
            ext.routing_gemm_det(xd, g8, sb, ref)
            full = route(xd, gd, gt, bias, g8, sb)
            self.assertTrue(torch.equal(full[0], ref), device)
            for rows in (1, 2, 17):
                logits, _, _ = route(xd[:rows], gd, gt, bias, g8, sb)
                self.assertTrue(torch.equal(logits, ref[:rows]), (device, rows))
            # Every row of the batch as its own decode call, not just row zero: rare midpoint
            # roundings may occur anywhere
            for row in range(32):
                one = route(xd[row:row + 1], gd, gt, bias, g8, sb)
                for got, expected in zip(one, full):
                    self.assertTrue(torch.equal(got, expected[row:row + 1]), (device, row))

    def test_matches_reference(self):
        torch.manual_seed(0)
        for K in (2560, 2048, 2880, 1040):
            for E in (512, 128, 64, 200, 40):
                gate = (torch.randn(K, E, device = DEVICE) * (1.0 / K ** 0.5)).half()
                gate_t, g8, sb = quant_gate(gate)
                # the weight quantization itself: 14-bit fixed point per row, error well under half's
                deq = (g8[0].float() * 128 + g8[1].float()) * sb[:, None]
                self.assertLess((deq - gate_t.float()).abs().max().item(), 1e-4 * gate_t.float().abs().max().item())
                for R in (2, 3, 16, 17, 64, 65, 300, 2048):
                    x = torch.randn(R, K, device = DEVICE).half()
                    ref = x.float() @ gate.float()
                    out = torch.empty(R, E, dtype = torch.half, device = DEVICE)
                    ext.routing_gemm_det(x, g8, sb, out)
                    err = (out.float() - ref).abs().max().item()
                    scale = ref.abs().max().item()
                    self.assertLess(err, 2e-3 * scale + 1e-3, (K, E, R, err, scale))
                    out2 = torch.empty_like(out)
                    ext.routing_gemm_det(x, g8, sb, out2)
                    self.assertTrue(torch.equal(out, out2), (K, E, R))

    def test_cross_device_identity(self):
        n = torch.cuda.device_count()
        if n < 2:
            self.skipTest("needs two or more GPUs")
        torch.manual_seed(3)
        K, E = 2560, 512
        gate = (torch.randn(K, E) * (1.0 / K ** 0.5)).half()
        for R in (2, 64, 65, 300, 1000):
            x = torch.randn(R, K).half()
            outs = []
            for d in range(n):
                dev = torch.device(f"cuda:{d}")
                _, g8, sb = quant_gate(gate.to(dev))
                out = torch.empty(R, E, dtype = torch.half, device = dev)
                ext.routing_gemm_det(x.to(dev), g8, sb, out)
                outs.append(out.cpu())
            for d in range(1, n):
                self.assertTrue(torch.equal(outs[0], outs[d]), (R, torch.cuda.get_device_name(d)))

    def test_routing_std_multirow_selection(self):
        torch.manual_seed(2)
        K, E, topk = 2560, 128, 8
        gate = (torch.randn(K, E, device = DEVICE) * (1.0 / K ** 0.5)).half()
        gate_t, g8, sb = quant_gate(gate)
        x = torch.randn(300, K, device = DEVICE).half()
        logits = torch.empty(300, E, dtype = torch.half, device = DEVICE)
        sel = torch.empty(300, topk, dtype = torch.long, device = DEVICE)
        w = torch.empty(300, topk, dtype = torch.half, device = DEVICE)
        ext.routing_std(x, gate, logits, sel, w, None, gate_t, None, g8, sb)
        ref = x.float() @ gate.float()
        ref_top = torch.topk(ref, topk, dim = -1).indices
        agree = (torch.sort(sel, dim = 1).values == torch.sort(ref_top, dim = 1).values).all(dim = 1)
        tv = torch.topk(ref, topk + 1, dim = -1).values
        clear = (tv[:, topk - 1] - tv[:, topk]) > 1e-2
        self.assertTrue(agree[clear].all().item())
        self.assertGreater(clear.float().mean().item(), 0.5)

    def test_det_transcendentals(self):
        x = torch.cat((torch.linspace(-80, 80, 100001), torch.randn(100000) * 5)).to(DEVICE)
        y = torch.empty(3 * x.numel(), dtype = torch.float, device = DEVICE)
        ext.det_math_test(x, y)
        e, l, sp = y.view(3, -1)
        rel = lambda a, b: ((a - b).abs() / b.abs().clamp_min(1e-30)).max().item()
        self.assertLess(rel(e.double(), torch.exp(x.double())), 4e-7)
        self.assertLess(rel(l.double(), torch.log(x.double().abs() + 1e-30)), 4e-7)
        self.assertLess(rel(sp.double(), torch.nn.functional.softplus(x.double())), 4e-7)


if __name__ == "__main__":
    unittest.main()
