"""
Row stripes of the fused MoE kernel (exl3_moe with tile_rows = True, used by fused-only prefill,
EXL3_MOE_FUSED_PREFILL).

CPU (no extension): the binding's argument list (every earlier argument stays in place for
positional callers); row stripes as a compile-time option of the kernel (the kernel's arguments
unchanged, a row-striped twin of every instance, the launcher's twin tables, further stripes as a
jump back to the phases); and the resident layer's worst-case prefill estimate under fused-only
prefill. The stripes themselves are checked against the real kernel on the GPU.

GPU: experts with up to 700 rows run through 64/128/256-row temp buffers in stripes and must
write, bit for bit, the same per-assignment slots as an unstriped launch whose buffers hold every
row. Both launches use the same concurrency and active-expert counts, so the same group geometry;
only the rows per GEMM call differ, which the kernel's per-row arithmetic does not depend on.
Random EXL3 trellises (mul1 codebook, 3 bits), no checkpoint needed:

    python -m pytest tests/test_moe_stripe_layout_.py
"""
import ast
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
QUANT = ROOT / "exllamav3" / "exllamav3_ext" / "quant"
GETTER = re.compile(r"fp_exl3_moe_kernel (exl3_moe_kernel_\w+)\(\) \{ return exl3_moe_kernel<([^>]*)>; \}")
TABLES = ("exl3_moe_kernel_instances", "exl3_moe_kernel_instances_m32", "exl3_moe_kernel_instances_m64")
DEFAULT = re.compile(r"exl3_moe_kernel_k\d_n(128|256)_cb[12](_m32|_m64)?")    # a default instance's getter


def instances():
    """Every fused MoE kernel instance: getter name -> template arguments"""
    defined = {}
    for path in (QUANT / "comp_units").glob("exl3_moe_inst_*.cu"):
        for name, args in GETTER.findall(path.read_text()):
            defined[name] = [a.strip() for a in args.split(",")]
    return defined


def table_entries(table):
    launcher = (QUANT / "exl3_moe.cu").read_text()
    body = launcher.split(f"fp_exl3_moe_kernel {table}[] =", 1)[1].split("};", 1)[0]
    return re.findall(r"(exl3_moe_kernel_\w+)\(\)", body)


class MoeStripeLayoutTests(unittest.TestCase):

    def test_binding_keeps_legacy_positional_arguments(self):
        native = Path(__file__).resolve().parents[1] / "exllamav3" / "exllamav3_ext"
        header = (native / "quant" / "exl3_moe.cuh").read_text()
        args = header.split("void exl3_moe\n(\n", 1)[1].split("\n);", 1)[0]
        names = [part.strip().split("=")[0].strip().split()[-1] for part in args.split(",")]
        binding = (native / "bindings.cpp").read_text().split('m.def("exl3_moe",', 1)[1].split(");", 1)[0]
        self.assertEqual(re.findall(r'py::arg\("([^"]+)"\)', binding), names)
        self.assertEqual(len(names), 36)
        self.assertEqual(names[-2:], ["m_tile", "tile_rows"])
        self.assertIn('py::arg("tile_rows") = false', binding)


class MoeStripeInstanceTests(unittest.TestCase):
    """Row stripes are a template parameter of exl3_moe_kernel (tile_rows), so the default instances
    compile without them: each default instance has a row-striped twin (_st) that the launcher
    picks for tile_rows, and the kernel's launch arguments are unchanged"""

    def test_every_instance_has_a_striped_twin(self):
        defined = instances()
        plain = {n: a for n, a in defined.items() if DEFAULT.fullmatch(n)}
        self.assertEqual(len(plain), 54)
        for name, args in plain.items():
            self.assertEqual(defined.get(name + "_st"),
                             (args + ["MOE_TILESIZE_M"] if len(args) == 3 else args) + ["true"], name)
        for table in TABLES:
            self.assertEqual([e + "_st" for e in table_entries(table)], table_entries(table + "_st"), table)

    def test_stripes_are_a_template_parameter(self):
        common = (QUANT / "exl3_moe_common.cuh").read_text()
        self.assertNotIn("tile_rows", common.split("#define EXL3_MOE_KERNEL_ARGS", 1)[1])
        kernel = (QUANT / "exl3_moe_kernel.cuh").read_text()
        template = kernel.split("__global__", 1)[0].rsplit("template<", 1)[1]
        self.assertIn("bool tile_rows = false", template)
        launcher = (QUANT / "exl3_moe.cu").read_text()
        self.assertIn("tile_rows ? moe_family_striped : moe_family_default", launcher)

    def test_further_stripes_jump_back_to_the_phases(self):
        # A striped instance runs the phases for an expert's first stripe and jumps back to them for
        # the others, after a group barrier; there is no loop around the phases, whose code in the
        # default instances is then exactly the single pass
        kernel = (QUANT / "exl3_moe_kernel.cuh").read_text()
        lines = kernel.split("\n")
        label = lines.index("    next_stripe:")
        self.assertTrue(lines[label + 1].strip().startswith("// Gather + input hadamard"), lines[label + 1])
        self.assertEqual(kernel.count("next_stripe"), 3)                  # the label, the comment, the jump
        self.assertEqual(kernel.count("goto next_stripe;"), 1)
        after = kernel.split("\n        had_d_out();\n", 1)[1].split("        // Draw the next ticket", 1)[0]
        self.assertIn("        if constexpr (tile_rows)\n", after)
        jump = after.split("goto next_stripe;", 1)[0]
        self.assertIn("if (start + token_count < end)", jump)
        self.assertIn("group_barrier(group_idx, group_size, barrier_counters_sense);", jump)
        self.assertIn("token_count = MIN(end - start, max_tokens_per_expert);", jump)
        self.assertIn("fused_base[expert_idx] + row_offset", kernel)


class ResidentFusedOnlyPlanningTests(unittest.TestCase):
    """BlockSparseMLP.prefill_worst_case_parts: under fused-only prefill no reconstruct or
    per-expert working set exists, so the autosplit reserves only the whole-call buffers"""

    def estimate(self, fused_prefill):
        cls = next(n for n in ast.parse((ROOT / "exllamav3/modules/block_sparse_mlp.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "BlockSparseMLP")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prefill_worst_case_parts")
        ns = dict(torch = torch, FUSED_DET = True, PAD_MAX = 1.1, FUSED_PREFILL = fused_prefill)
        exec(compile(ast.Module(body = [fn], type_ignores = []), "<production-worst-case>", "exec"), ns)
        recon = NS(worst_case_bytes = Mock(return_value = 10 ** 9))
        layer = NS(cpu_offload = False, multi_up = object(), expert_size = 64, intermediate_size_padded = 128,
                   interm_dtype = torch.half, device = "cpu", _batch_recon_layer = Mock(return_value = recon))
        return ns["prefill_worst_case_parts"](layer, 512, 3072), layer._batch_recon_layer

    def test_variable_part_only_with_tiers(self):
        (fixed, variable), recon = self.estimate(False)
        self.assertEqual(variable, 10 ** 9)
        (fixed_f, variable_f), recon_f = self.estimate(True)
        self.assertEqual((fixed_f, variable_f), (fixed, 0))
        recon_f.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MoeStripeKernelTests(unittest.TestCase):

    H, I, K, E, T = 256, 256, 3, 4, 700
    INT_MAX = (1 << 31) - 1

    @classmethod
    def setUpClass(cls):
        from exllamav3.ext import exllamav3_ext
        cls.ext = exllamav3_ext

    def experts(self, device, gen):
        def trellis(k, n):
            return torch.randint(0, 65536, (k // 16, n // 16, 16 * self.K), dtype = torch.int32,
                                 generator = gen).to(torch.int16).to(device)
        def scale(n):
            s = (torch.rand((n,), generator = gen) * 0.2 + 0.9) * torch.where(torch.rand((n,), generator = gen) < 0.5, -1.0, 1.0)
            return s.half().to(device)
        tabs = []
        keep = []
        for k, n in ((self.H, self.I), (self.H, self.I), (self.I, self.H)):   # gate, up, down
            t = [trellis(k, n) for _ in range(self.E)]
            su = [scale(k) for _ in range(self.E)]
            sv = [scale(n) for _ in range(self.E)]
            keep += t + su + sv
            tabs.append(tuple(torch.tensor([x.data_ptr() for x in xs], dtype = torch.long, device = device)
                              for xs in (t, su, sv)))
        return tabs, keep

    def run_kernel(self, device, x, tabs, routing, rows, tile_rows, act = (0, 0.0)):
        ext = self.ext
        expert_count, token_sorted, weight_sorted, fused_base, launches = routing
        C = ext.exl3_moe_max_concurrency(torch.device(device).index)
        temps = [torch.empty((C, rows, d), dtype = torch.half, device = device) for d in (self.H, self.H, self.I, self.I)]
        out = torch.zeros((self.T, self.H), dtype = torch.float, device = device)
        scratch = torch.full((token_sorted.numel(), self.H), float("nan"), dtype = torch.float, device = device)
        (gt, gs, gv), (ut, us, uv), (dt, ds, dv) = tabs
        for num_active, lo, hi, m_tile in launches:
            ext.exl3_moe(
                x, out, expert_count, token_sorted, weight_sorted, *temps,
                act[0], float(self.K), float(self.K), float(self.K),
                gt, gs, gv, ut, us, uv, dt, ds, dv,
                False, True, False, True, False, True,
                act[1], num_active, scratch, fused_base, lo, hi, m_tile,
                tile_rows = tile_rows,
            )
        torch.cuda.synchronize(device)
        self.assertEqual(torch.count_nonzero(out).item(), 0)   # slot mode leaves the output alone
        return scratch

    def test_stripes_equal_unstriped_launch(self):
        for device in range(torch.cuda.device_count()):
            gen = torch.Generator().manual_seed(1734)
            tabs, keep = self.experts(device, gen)
            x = (torch.randn((self.T, self.H), generator = gen) * 0.05).half().to(device)
            # Expert 0 takes every token, 1/2/3 split the second pick: 700 / 45 / 30 / 625 rows
            sel = torch.zeros((self.T, 2), dtype = torch.long)
            sel[:45, 1] = 1
            sel[45:75, 1] = 2
            sel[75:, 1] = 3
            weights = (torch.rand((self.T, 2), generator = gen) * 0.9 + 0.1).half()
            flat = sel.flatten().to(device)
            order = flat.argsort(stable = True)
            token_sorted = torch.arange(self.T, device = device).repeat_interleave(2)[order]
            weight_sorted = weights.flatten().to(device)[order]
            expert_count = torch.bincount(flat, minlength = self.E + 1)
            fused_base = torch.cumsum(expert_count, 0) - expert_count
            counts = expert_count[:self.E].tolist()
            self.assertEqual(counts, [700, 45, 30, 625])
            # One launch per row tile over its expert range, as the module does
            launches = [(3, 33, self.INT_MAX, 64), (1, 17, 32, 32)]
            routing = (expert_count, token_sorted, weight_sorted, fused_base, launches)
            reference = self.run_kernel(device, x, tabs, routing, 704, False)
            self.assertTrue(bool(torch.isfinite(reference).all()))
            self.assertGreater(torch.count_nonzero(reference).item(), 0)
            for rows in (64, 128, 256):
                with self.subTest(device = device, rows = rows):
                    striped = self.run_kernel(device, x, tabs, routing, rows, True)
                    self.assertTrue(torch.equal(striped, reference))
            # the same for DeepSeek's reference swiglu (activation 5, limit 10), which runs the fused
            # kernel's silu_ref instances
            ref_act = (5, 10.0)
            reference5 = self.run_kernel(device, x, tabs, routing, 704, False, ref_act)
            self.assertTrue(bool(torch.isfinite(reference5).all()))
            self.assertFalse(torch.equal(reference5, reference))
            with self.subTest(device = device, act = "silu_ref"):
                self.assertTrue(torch.equal(self.run_kernel(device, x, tabs, routing, 128, True, ref_act), reference5))
            # Without stripes, experts above the buffers' rows are skipped (left to the caller): by
            # the default instances, and by the reference-swiglu instances, which run in stripes and
            # get the launch's row limit capped at the buffers' rows instead
            starts = fused_base.tolist()
            for act, full in (((0, 0.0), reference), (ref_act, reference5)):
                skipped = self.run_kernel(device, x, tabs, routing, 256, False, act)
                for e, c in enumerate(counts):
                    block = skipped[starts[e] : starts[e] + c]
                    with self.subTest(device = device, act = act[0], expert = e):
                        if c > 256:
                            self.assertTrue(bool(torch.isnan(block).all()))
                        else:
                            self.assertTrue(torch.equal(block, full[starts[e] : starts[e] + c]))
            del keep


if __name__ == "__main__":
    unittest.main()
