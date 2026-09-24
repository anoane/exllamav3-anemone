"""Execute the production streamed dispatcher on CPU tensors with mocked CUDA.

The native projection/gather boundaries use synthetic arithmetic: exact values for the
layout, padding, tier-dispatch and sentinel checks, and random ones for the check that slot
mode fixes the order of the accumulation. Not real kernels. Run directly with Torch installed;
no CUDA initialization or model is required.
"""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slot_tests", Path(__file__).with_name("test_moe_stream_slots_.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class StreamDispatcherTests(unittest.TestCase):
    def test_native_dimension_overflow_fails_before_allocation(self):
        cls = next(n for n in ast.parse((ROOT / "exllamav3/model/moe_cpu_host.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "MoeCpuHost")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_submit_prefill_streamed")
        fn.body = [n for n in fn.body if not isinstance(n, ast.ImportFrom)]
        ns = dict(TUNING = NS(stream_deterministic = True))  # No Torch/worker is available.
        exec(compile(ast.Module(body = [fn], type_ignores = []), "<production-dimension-admission>", "exec"), ns)
        for rows, width, topk in ((2**31, 16, 1), (1, 2**31, 1), (2**26 + 1, 16, 32)):
            with self.subTest(rows = rows, width = width, topk = topk), self.assertRaisesRegex(ValueError, "int32"):
                ns[fn.name](None, 0, NS(shape = (rows, width)), NS(shape = (rows, topk)), None,
                            dict(num_experts = 4), None, None, None, None, None, None)

    def test_workspace_uses_logical_not_packed_width(self):
        cls = next(n for n in ast.parse((ROOT / "exllamav3/model/moe_cpu_host.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "MoeCpuHost")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prefill_worst_case_parts")

        class Imports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None

        fn = Imports().visit(fn)
        tuning = NS(stream_deterministic = True)
        ns = dict(torch = torch, TUNING = tuning, PAD_MAX = 1.1, slot_rows_bound = helpers.slots.slot_rows_bound)
        exec(compile(ast.Module(body = [fn], type_ignores = []), "<production-workspace>", "exec"), ns)
        recon = NS(cap = 4, worst_case_bytes = Mock(side_effect = lambda a, slot_mode: 100000 if slot_mode else 200000))
        spec = dict(hi = 32, ho = 32, topk = 6, num_experts = 4, expert_bytes = 256,
                    proj_dims = {"g": (32, 16, 1), "u": (32, 16, 1), "d": (16, 32, 1)})
        # 7 rows must take the streamed branch: keep the fake floor below them
        host = NS(specs = [spec], aux = {0: {}}, cap_rows = 64, stream_min_rows = 4, wslot_size = 1024,
                  _device_buffers = lambda *a: {},
                  _stream_fused_t = lambda *a: 0, _stream_recon_layer = lambda *a: recon)
        with patch.object(torch.cuda, "device", lambda _: nullcontext()):
            for width, expected in ((32, 100000), (16, 200000), (None, 200000)):
                fixed, variable = ns[fn.name](host, 0, 7, "cpu", 42, hidden_width = width)
                self.assertEqual(variable, expected)
                self.assertGreaterEqual(fixed, helpers.slots.slot_rows_bound(42, 1.1, 4) * 32 * 4)

    def run_case(self, rows, topk, fused_t, recon_enabled, padded, deterministic, batch_size,
                 mixed = False, weight = 0.375, inexact = False):
        cls = next(n for n in ast.parse((ROOT / "exllamav3/model/moe_cpu_host.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "MoeCpuHost")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_submit_prefill_streamed")
        fn.body = [n for n in fn.body if not isinstance(n, ast.ImportFrom)]
        ext = NS()
        ns = dict(torch = torch, np = helpers.np, ext = ext, slot_layout = helpers.slots.slot_layout,
                  plan_groups = helpers.production_groups(),
                  TUNING = NS(stream_deterministic = deterministic, stream_debug = False, mtile = True))
        exec(compile(ast.Module(body = [fn], type_ignores = []), "<production-stream-dispatch>", "exec"), ns)
        E, h, nd = 4, 16, 32 if padded else 16
        gen = torch.Generator().manual_seed(1234)
        if inexact:
            # Random activations and routing weights: the sums are not exact, so their order shows.
            # The per-expert scales are exact in FP16 and shared by all three tiers
            y = torch.randn((rows, h), generator = gen).half()
            factors = [1.375, 0.8125, 1.6875, 0.5625]
        else:
            y = torch.arange(rows*h).reshape(rows, h).remainder(7).half()
            factors = [e+1 for e in range(E)]
        selected = (torch.arange(rows*topk).reshape(rows, topk).remainder(E+1)-1).long()
        if mixed:
            selected.fill_(-1)
            selected[:, 0] = 0          # 129: per-expert fallback
            selected[:10, 1] = 1        # 10: fused
            selected[:80, 2] = 2        # 80/79: reconstruct with one padding row
            selected[:79, 3] = 3
        if inexact:
            # Shuffle each row's routes: the routing order then differs from the expert order
            perm = torch.argsort(torch.rand((rows, topk), generator = gen), dim = 1)
            selected = selected.gather(1, perm)
        weights = torch.rand((rows, topk), generator = gen) * 0.9 + 0.05 if inexact else \
            torch.full((rows, topk), weight, dtype = torch.float32)
        flat, shifted = selected.flatten(), selected.flatten()+1
        counts1 = torch.bincount(shifted, minlength = E+1).tolist()
        neg, counts = counts1[0], counts1[1:]
        active = [e for e, c in enumerate(counts) if c]
        dims = {p: (16, nd if p == "d" else 16, 1) for p in ("g", "u", "d")}
        aux = {suffix+"_"+p: [torch.full((16,), factors[e] if p == "d" else 1, dtype = torch.half)
                              for e in range(E)] for suffix in ("suh", "svh") for p in dims}
        factor = {t.data_ptr(): factors[i] for i, t in enumerate(aux["svh_d"])}
        stream = NS(wait_event = Mock())
        st = dict(copy_stream = stream, wslot_used = [False]*2, wconsumed_ev = [NS(record = Mock()) for _ in range(2)],
                  wready_ev = [NS(record = Mock()) for _ in range(2)], swz = False,
                  vram_slots = [torch.empty(batch_size*128, dtype = torch.half) for _ in range(2)],
                  w_scratch = None)
        host = NS(aux = {0: aux}, wslot_size = batch_size*256,
                  batch_experts = batch_size, gpu_base_ptr = 0, pinned = True,
                  layer_blocks = {0: [(0, e*256) for e in range(E)]},
                  arena_views = [torch.zeros(E*128, dtype = torch.half)], next_wslot = 0, num_wslots = 2, wseq = 0,
                  _stream_fused_t = lambda *a: fused_t if not padded else 0,
                  _stream_fused_bufs = lambda *a: [None]*4)
        spec = dict(num_experts = E, proj_dims = dims, proj_bytes = (32, 32, 64), expert_bytes = 256,
                    hi = h, ho = nd, activation = 0, act_limit = 0)

        seen = dict(scratch = [], gather_out_zero = [])

        def fused(*a):
            x, out, ec, tok, w = a[:5]
            scratch, bases, lo, hi = a[30:34]
            seen["scratch"].append(scratch is not None)
            off = 0
            for e, c in enumerate(ec[:-1].tolist()):
                if lo <= c <= hi:
                    val = x[tok[off:off+c]].float() * factor[a[21][e].item()] * w[off:off+c, None].float()
                    if scratch is None:
                        out.index_add_(0, tok[off:off+c], val)
                    else:
                        scratch[bases[e]:bases[e]+c].copy_(val)
                off += c

        def reconstructed(x, out, tok, w, ids, starts, counts, ptrs, out_slab):
            cmax = max(counts)
            if out_slab is not None:
                self.assertEqual(out_slab.shape, (len(ids)*cmax, nd))
                out_slab.fill_(float("nan"))  # Padding must never be read by gather.
            for b, (e, start, c) in enumerate(zip(ids, starts, counts)):
                ri = tok[start:start+c]
                val = x[ri].float()*factors[e]
                if out_slab is None:
                    out.index_add_(0, ri, val * w[start:start+c, None].float())
                else:
                    out_slab[b*cmax:b*cmax+c, :h].copy_(val)

        def gather(out, scratch, ids, inv, starts, bases, kinds, ws):
            seen["gather_out_zero"].append(bool((out == 0).all()))
            ids, inv = ids.view(rows, -1), inv.view(rows, -1)
            for row in range(rows):
                acc = torch.zeros(h)
                for e, pos in zip(ids[row].tolist(), inv[row].tolist()):
                    if e >= 0 and kinds[e]:
                        value = scratch[bases[e]+pos-starts[e]]
                        acc += value * (ws[pos].float() if kinds[e] == 2 else 1)
                out[row] += acc

        ext.exl3_moe, ext.exl3_moe_gather = fused, gather
        recon = NS(max_rows = 100, cap = 4, run_group = reconstructed) if recon_enabled else None
        host._stream_recon_layer = lambda *a: recon
        host._dq_linear = lambda x, trellis, dims, suh, svh, bias, scratch: \
            torch.nn.functional.pad(x*suh[0], (0, dims[1]-x.shape[1]))
        host._act = lambda spec, g, u: u
        with patch.object(torch.cuda, "stream", lambda _: nullcontext()), \
             patch.object(torch.cuda, "current_stream", lambda: stream):
            out = ns[fn.name](host, 0, y, selected, weights, spec, active, st, counts, flat, shifted, neg)
        if deterministic:
            # Slot mode: every fused launch got the scratch, and nothing reached `out` before the
            # gather (one per 32 routes; slot mode switched off would fail here, whatever the values)
            self.assertTrue(all(seen["scratch"]))
            self.assertEqual(len(seen["gather_out_zero"]), math_ceil(topk, 32))
            self.assertTrue(seen["gather_out_zero"][0])
        else:
            self.assertFalse(any(seen["scratch"]))
            self.assertEqual(seen["gather_out_zero"], [])
        expected = torch.zeros(rows, h)
        if inexact:
            # The routing (k) order, with each tier's own arithmetic, as the gather sums
            ft = fused_t if not padded else 0
            w16 = weights.half().float()
            for row in range(rows):
                acc = torch.zeros(h)
                for k in range(topk):
                    e = int(selected[row, k])
                    if e < 0:
                        continue
                    if (ft and counts[e] <= ft) or (recon_enabled and counts[e] <= 100):
                        acc += (y[row].float() * factors[e]) * w16[row, k]
                    else:
                        acc += (y[row] * aux["suh_d"][e][0]).float() * weights[row, k]
                expected[row] += acc
            if deterministic:
                torch.testing.assert_close(out, expected, rtol = 0, atol = 0)
            else:
                torch.testing.assert_close(out, expected, rtol = 1e-5, atol = 1e-5)
        else:
            for k in range(topk):
                e = selected[:, k]
                expected += y.float() * (e+1).float().unsqueeze(1) * weights[:, k, None]
            torch.testing.assert_close(out, expected, rtol = 0, atol = 0)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(host.wseq, math_ceil(len(active), batch_size))
        return out

    def test_dispatch_matrix(self):
        for rows, topk in ((1, 6), (33, 6), (257, 6), (7, 40)):
            for recon, padded in ((True, False), (False, False), (True, True), (False, True)):
                for det in (False, True):
                    for batch in (1, 2, 4):
                        with self.subTest(rows = rows, topk = topk, recon = recon, padded = padded,
                                          det = det, batch = batch):
                            self.run_case(rows, topk, 64, recon, padded, det, batch)

    def test_all_three_tiers_in_one_call(self):
        for deterministic in (False, True):
            for batch in (1, 2, 4):
                with self.subTest(deterministic = deterministic, batch = batch):
                    self.run_case(129, 6, 64, True, False, deterministic, batch, mixed = True)

    def test_per_expert_path_preserves_fp32_weights(self):
        self.run_case(7, 2, 0, False, False, True, 2, weight = 0.375051)

    def test_accumulation_order_with_inexact_values(self):
        # Random values over all three tiers, up to four routes per row: slot mode must give the
        # routing-order sum bitwise, for every staging-batch size (the same slot values reach the
        # gather however the experts are batched). The atomic path adds per expert, in batch
        # order, which differs in the last bits: this is what the check can see
        outs = []
        for batch in (1, 2, 4):
            with self.subTest(batch = batch):
                outs.append(self.run_case(129, 6, 64, True, False, True, batch, mixed = True, inexact = True))
        self.assertTrue(all(torch.equal(outs[0], o) for o in outs[1:]))
        atomic = self.run_case(129, 6, 64, True, False, False, 2, mixed = True, inexact = True)
        self.assertFalse(torch.equal(atomic, outs[0]))


def math_ceil(n, d):
    return (n+d-1)//d


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
