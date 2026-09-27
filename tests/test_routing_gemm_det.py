"""
Deterministic int8 router projection (ext.routing_gemm_det, routing_gemm.cu + det_gemm.cuh): must
match an fp32 reference to half-output tolerance over row counts, expert counts (including
non-multiples of the tile) and K values, be bit-reproducible, agree bit for bit across every
visible GPU (the point of the int8 scheme: fp16 tensor cores do not across architectures), and
feed ext.routing_std so that multi-row routing selects the same experts as the reference on rows
that are not near-ties. Also checks the FMA-only transcendentals the routing activations use, and
EXL3_ROUTING_ROW_INVARIANT: off, single rows keep the GEMV; on, a row's logits are bitwise the same
in calls of any row count, one included.
"""
import sys, os, subprocess, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

DEVICE = torch.device("cuda:0")


def quant_gate(gate):
    gate_t = gate.T.contiguous()
    E, K = gate_t.shape
    g8 = torch.empty((2, E, K), dtype = torch.int8, device = gate.device)
    sb = torch.empty((E,), dtype = torch.float, device = gate.device)
    ext.det_quant_weight(gate_t, g8, sb)
    return gate_t, g8, sb


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRoutingGemmDet(unittest.TestCase):

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

    def test_row_invariant_default_off(self):
        if os.environ.get("EXL3_ROUTING_ROW_INVARIANT", "0") != "0":
            self.skipTest("EXL3_ROUTING_ROW_INVARIANT is set")
        # by default a single row takes the fp16 GEMV whether or not the int8 gate is given
        torch.manual_seed(4)
        K, E, topk = 2560, 128, 8
        gate = (torch.randn(K, E, device = DEVICE) * (1.0 / K ** 0.5)).half()
        gate_t, g8, sb = quant_gate(gate)
        x = torch.randn(1, K, device = DEVICE).half()
        outs = []
        for q in ((g8, sb), (None, None)):
            logits = torch.empty(1, E, dtype = torch.half, device = DEVICE)
            sel = torch.empty(1, topk, dtype = torch.long, device = DEVICE)
            w = torch.empty(1, topk, dtype = torch.half, device = DEVICE)
            ext.routing_std(x, gate, logits, sel, w, None, gate_t, None, *q)
            outs.append(logits)
        self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_row_invariant(self):
        if torch.cuda.get_device_capability(DEVICE)[0] < 8:
            self.skipTest("the int8 projection needs sm_80+")
        # the switch is read once per process: run this test in a child with it set
        if os.environ.get("EXL3_ROUTING_ROW_INVARIANT", "0") == "0":
            env = dict(os.environ, EXL3_ROUTING_ROW_INVARIANT = "1")
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "TestRoutingGemmDet.test_row_invariant"],
                               env = env, capture_output = True, text = True, timeout = 1800)
            self.assertEqual(r.returncode, 0, r.stderr[-4000:])
            return
        torch.manual_seed(6)
        rows = (1, 2, 63, 64, 65, 128, 129, 1024, 1025, 4096)
        for K, E in ((2560, 512), (2048, 128), (4096, 256), (1040, 200)):
            gate = (torch.randn(K, E, device = DEVICE) * (1.0 / K ** 0.5)).half()
            gate_t, g8, sb = quant_gate(gate)
            x = torch.randn(rows[-1], K, device = DEVICE).half()
            full = torch.empty(rows[-1], E, dtype = torch.half, device = DEVICE)
            ext.routing_gemm_det(x, g8, sb, full)
            for R in rows:
                out = torch.empty(R, E, dtype = torch.half, device = DEVICE)
                ext.routing_gemm_det(x[:R].contiguous(), g8, sb, out)
                self.assertTrue(torch.equal(out, full[:R]), (K, E, R))
            # the full router: one row alone (decode) equals the same row of a multi-row call in
            # logits, selection and weights
            def route(xr):
                n = xr.shape[0]
                logits = torch.empty(n, E, dtype = torch.half, device = DEVICE)
                sel = torch.empty(n, 8, dtype = torch.long, device = DEVICE)
                w = torch.empty(n, 8, dtype = torch.half, device = DEVICE)
                ext.routing_std(xr, gate, logits, sel, w, None, gate_t, None, g8, sb)
                return logits, sel, w
            many = route(x[:257].contiguous())
            for i in range(0, 257, 8):
                one = route(x[i:i + 1].contiguous())
                for a, b in zip(one, many):
                    self.assertTrue(torch.equal(a, b[i:i + 1]), (K, E, i))

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
