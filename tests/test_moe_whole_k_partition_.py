"""
Whole-K partition of the fused MoE kernel's GEMMs (EXL3_STABLE_ARITHMETIC).

CPU: the partition formula, read from exl3_gemm_kernel_inner, gives every block whole output
columns (complete K ranges) and covers every column exactly once for any grid width; the default
split of the flattened (N, K) tiles divides columns of a real projection shape; every fused MoE
kernel instance has a whole-K twin (row-striped as well), and the launcher's twin tables mirror
the default ones. The build option EXLLAMA_NO_WHOLE_K_MOE (setup.py and the JIT build in ext.py)
leaves out exactly the whole-K units and tells the native code, which then compiles without them;
an MoE layer that would need them under EXL3_STABLE_ARITHMETIC is refused at load.

GPU (a process started with EXL3_STABLE_ARITHMETIC=1, which selects the whole-K kernel instances):
the same experts computed with different active-expert hints, which change the width of the expert
groups and with it the default K split, give bitwise-identical outputs, through the 64-, 32- and
16-row instances (N = 256 and N = 128) of the mul1 codebook and the 16-row instances of mcg. The
hints give groups of 8 blocks and of up to 32; on GPUs with 128 or more SMs hints 1-4 all reach 32,
so there the comparison is 8 against 32 blocks. A launch without row stripes skips the experts
above the temp buffers' rows, as the default instances do. Random EXL3 trellises (3 bits), no
checkpoint needed:

    EXL3_STABLE_ARITHMETIC=1 python -m pytest tests/test_moe_whole_k_partition_.py
"""
import importlib.util
import os
from pathlib import Path
import re
import runpy
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
QUANT = ROOT / "exllamav3" / "exllamav3_ext" / "quant"
SOURCE = QUANT / "exl3_gemm_inner.cuh"
STABLE = os.environ.get("EXL3_STABLE_ARITHMETIC") == "1"


def production_partition():
    """The whole-K and default (slice_beg, slice_end) formulas of the kernel, as integer functions"""
    text = SOURCE.read_text()
    block = text.split("if constexpr (whole_k)", 1)[1].split("int slice_len", 1)[0]
    whole, default = block.split("\n    else\n", 1)
    def formula(part, name):
        expr = re.search(name + r" = (.*);", part).group(1).replace("/", "//").replace("blockIdx.x", "b")
        return eval("lambda tiles_k, tiles_n, b, num_slices: " + expr)
    return ((formula(whole, "slice_beg"), formula(whole, "slice_end")),
            (formula(default, "slice_beg"), formula(default, "slice_end")))


class WholeKPartitionTests(unittest.TestCase):

    def test_exact_cover_without_partial_columns(self):
        (beg, end), _ = production_partition()
        for nk in (1, 2, 3, 16, 144, 320, 512):
            for nn in range(1, 129):
                for blocks in range(1, 65):
                    intervals = [(beg(nk, nn, b, blocks), end(nk, nn, b, blocks)) for b in range(blocks)]
                    self.assertEqual(intervals[0][0], 0)
                    self.assertEqual(intervals[-1][1], nk * nn)
                    previous = 0
                    for lo, hi in intervals:
                        self.assertEqual(lo, previous)
                        self.assertLessEqual(lo, hi)
                        self.assertEqual(lo % nk, 0)
                        self.assertEqual(hi % nk, 0)
                        previous = hi
                    lengths = [(hi - lo) // nk for lo, hi in intervals]
                    self.assertLessEqual(max(lengths) - min(lengths), 1)

    def test_default_partition_splits_real_projection_columns(self):
        # The fused MoE up projection of a 5120-wide model into 2304 intermediate columns:
        # MOE_TILESIZE_K = 32 gives tiles_k = 160, N = 128 tiles give tiles_n = 18. On an 8-block
        # group the default partition starts slices inside columns; whole-K never does
        (wk_beg, _), (beg, _) = production_partition()
        nk, nn, blocks = 5120 // 32, 2304 // 128, 8
        self.assertTrue(any(beg(nk, nn, b, blocks) % nk for b in range(blocks)))
        self.assertTrue(all(wk_beg(nk, nn, b, blocks) % nk == 0 for b in range(blocks)))

    def test_every_instance_has_a_whole_k_twin(self):
        # tile_rows and whole_k set: the whole-K instances run in row stripes, and every one of them
        # lives in a *_wk.cu unit, which EXLLAMA_NO_WHOLE_K_MOE leaves out
        getter = re.compile(r"fp_exl3_moe_kernel (exl3_moe_kernel_\w+)\(\) \{ return exl3_moe_kernel<([^>]*)>; \}")
        defined, units = {}, {}
        for path in (QUANT / "comp_units").glob("exl3_moe_inst_*.cu"):
            for name, args in getter.findall(path.read_text()):
                defined[name] = [a.strip() for a in args.split(",")]
                units[name] = path.name
        plain = {n: a for n, a in defined.items() if re.fullmatch(r"exl3_moe_kernel_k\d_n(128|256)_cb[12](_m32|_m64)?", n)}
        self.assertEqual(len(plain), 54)
        for name, args in plain.items():
            twin = defined.get(name + "_wk")
            self.assertIsNotNone(twin, name)
            self.assertEqual(twin, (args + ["MOE_TILESIZE_M"] if len(args) == 3 else args) + ["true", "true"], name)
        for name, unit in units.items():
            self.assertEqual(name.endswith("_wk"), unit.endswith("_wk.cu"), name)
        launcher = (QUANT / "exl3_moe.cu").read_text()
        for table in ("exl3_moe_kernel_instances", "exl3_moe_kernel_instances_m32", "exl3_moe_kernel_instances_m64"):
            entries = lambda t: re.findall(r"(exl3_moe_kernel_\w+)\(\)", launcher.split(f"fp_exl3_moe_kernel {t}[] =", 1)[1].split("};", 1)[0])
            self.assertEqual([e + "_wk" for e in entries(table)], entries(table + "_wk"), table)

    def test_native_code_compiles_without_the_whole_k_units(self):
        # Every reference to a whole-K instance or table outside its own units sits inside
        # #ifndef EXLLAMA_NO_WHOLE_K_MOE, and the launcher refuses the missing family instead of
        # launching a null kernel
        for path in (QUANT / "exl3_moe.cu", QUANT / "comp_units" / "exl3_moe_instances.cuh"):
            depth = 0
            for line in path.read_text().split("\n"):
                if line.startswith("#ifndef EXLLAMA_NO_WHOLE_K_MOE"):
                    depth += 1
                elif line.startswith("#endif") or line.startswith("#else"):
                    depth = 0
                elif re.search(r"\bexl3_moe_kernel_\w+_wk\b", line) and "fp_exl3_moe_kernel exl3_moe_kernel_k##K##" not in line:
                    self.assertEqual(depth, 1, (path.name, line))
        launcher = (QUANT / "exl3_moe.cu").read_text()
        self.assertIn("static const MoeInstanceFamily moe_family_whole_k = { nullptr, nullptr, nullptr };", launcher)
        self.assertIn("TORCH_CHECK(family.rows16, ", launcher)
        self.assertIn("EXLLAMA_NO_WHOLE_K_MOE", launcher.split("TORCH_CHECK(family.rows16, ", 1)[1].split(");", 1)[0])
        self.assertIn('m.def("exl3_moe_whole_k_built"',
                      (ROOT / "exllamav3" / "exllamav3_ext" / "bindings.cpp").read_text())


def built_sources(runner, skip):
    """The sources and nvcc flags a build gets, EXLLAMA_NO_WHOLE_K_MOE set or not"""
    env = {k: v for k, v in os.environ.items() if k != "EXLLAMA_NO_WHOLE_K_MOE"}
    if skip:
        env["EXLLAMA_NO_WHOLE_K_MOE"] = ""       # presence is what counts, as for EXLLAMA_NOCOMPILE
    with mock.patch.dict(os.environ, env, clear = True):
        return runner()


class BuildOptionTests(unittest.TestCase):
    """EXLLAMA_NO_WHOLE_K_MOE, in the setup.py build and in the JIT build (ext.py): without it every
    source is compiled; with it (any value) exactly the whole-K units (*_wk.cu) are left out and
    nvcc gets -DEXLLAMA_NO_WHOLE_K_MOE"""

    def check(self, runner):
        full, full_flags = built_sources(runner, False)
        skip, skip_flags = built_sources(runner, True)
        names = lambda srcs: sorted(os.path.basename(s) for s in srcs)
        dropped = sorted(set(names(full)) - set(names(skip)))
        whole_k_units = sorted(p.name for p in (QUANT / "comp_units").glob("exl3_moe_inst_*_wk.cu"))
        self.assertGreaterEqual(len(whole_k_units), 38)
        self.assertEqual(dropped, whole_k_units)
        self.assertFalse(set(names(skip)) - set(names(full)))
        self.assertNotIn("-DEXLLAMA_NO_WHOLE_K_MOE", full_flags)
        self.assertIn("-DEXLLAMA_NO_WHOLE_K_MOE", skip_flags)
        self.assertEqual([f for f in skip_flags if f != "-DEXLLAMA_NO_WHOLE_K_MOE"], full_flags)

    def test_setup_py(self):
        def run():
            captured = {}
            def extension(name, sources, extra_compile_args, **kw):
                captured.update(sources = sources, flags = extra_compile_args["nvcc"])
            stub = types.SimpleNamespace(setup = lambda **kw: None)
            from torch.utils import cpp_extension
            env = {k: v for k, v in os.environ.items() if k != "EXLLAMA_NOCOMPILE"}
            with mock.patch.dict(sys.modules, {"setuptools": stub}), \
                 mock.patch.object(cpp_extension, "CUDAExtension", extension), \
                 mock.patch.dict(os.environ, env, clear = True):
                cwd = os.getcwd()
                os.chdir(ROOT)
                try:
                    runpy.run_path(str(ROOT / "setup.py"))
                finally:
                    os.chdir(cwd)
            return captured["sources"], captured["flags"]
        self.check(run)

    def test_jit_build(self):
        def run():
            captured = {}
            def load(name, sources, extra_cuda_cflags, **kw):
                captured.update(sources = sources, flags = list(extra_cuda_cflags))
                return types.SimpleNamespace()
            from torch.utils import cpp_extension
            # ext.py as a module of a stand-in package, so its relative import does not load the
            # real exllamav3 (and with it the extension)
            package = types.ModuleType("extpkg_under_test")
            package.__path__ = []
            util = types.ModuleType("extpkg_under_test.util")
            util.__path__ = []
            arch_list = types.ModuleType("extpkg_under_test.util.arch_list")
            arch_list.maybe_set_arch_list_env = lambda: None
            spec = importlib.util.spec_from_file_location("extpkg_under_test.ext", ROOT / "exllamav3" / "ext.py")
            module = importlib.util.module_from_spec(spec)
            real_find_spec = importlib.util.find_spec
            no_prebuilt = lambda name, *a: None if name == "exllamav3_ext" else real_find_spec(name, *a)
            with mock.patch.dict(sys.modules, {"extpkg_under_test": package, "extpkg_under_test.util": util,
                                               "extpkg_under_test.util.arch_list": arch_list}), \
                 mock.patch.object(cpp_extension, "load", load), \
                 mock.patch.object(importlib.util, "find_spec", no_prebuilt):
                spec.loader.exec_module(module)
            return captured["sources"], captured["flags"]
        self.check(run)


class LoadRefusalTests(unittest.TestCase):
    """math_policy.require_whole_k_moe, which the resident and the streamed MoE load paths call under
    EXL3_STABLE_ARITHMETIC"""

    def policy(self):
        spec = importlib.util.spec_from_file_location("math_policy_under_test", ROOT / "exllamav3" / "model" / "math_policy.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"EXL3_STABLE_ARITHMETIC": "0"}):
            spec.loader.exec_module(module)
        return module

    def test_refusal(self):
        require = self.policy().require_whole_k_moe
        require("model.layers.3.mlp", True, stable = True)
        require("model.layers.3.mlp", False, stable = False)
        with self.assertRaises(ValueError) as cm:
            require("model.layers.3.mlp", False, stable = True)
        message = str(cm.exception)
        for part in ("model.layers.3.mlp", "EXL3_STABLE_ARITHMETIC=1", "whole-K", "EXLLAMA_NO_WHOLE_K_MOE", "rebuild"):
            self.assertIn(part, message)

    def test_both_load_paths_check(self):
        bsm = (ROOT / "exllamav3" / "modules" / "block_sparse_mlp.py").read_text()
        load_local = bsm.split("    def load_local(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("require_whole_k_moe(self.key, ext.exl3_moe_whole_k_built())", load_local)
        host = (ROOT / "exllamav3" / "model" / "moe_cpu_host.py").read_text()
        register = host.split("    def register_layer(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("require_whole_k_moe(key, ext.exl3_moe_whole_k_built())", register)


@unittest.skipUnless(torch.cuda.is_available() and STABLE, "needs CUDA and EXL3_STABLE_ARITHMETIC=1")
class WholeKGeometryTests(unittest.TestCase):

    H, K, T = 256, 3, 700
    INT_MAX = (1 << 31) - 1
    # Expert 0 takes every row's first route; experts 1-5 split the second routes into 45, 30, 12, 80
    # and 533 rows. With 256-row temp buffers the 64-row launch runs 700 and 533 rows in stripes,
    # 45 rows, and 80 rows as a 64-row tile plus a 16-row tile; the 32-row launch runs 30 rows and
    # the 16-row launch (the path of decode and of every mcg model under the profile) 12 rows
    SECOND = (45, 30, 12, 80)
    E = 6
    MUL1_LAUNCHES = ((33, INT_MAX, 64), (17, 32, 32), (1, 16, 16))
    MCG_LAUNCHES = ((1, INT_MAX, 16),)           # the mcg codebook has 16-row tiles only

    @classmethod
    def setUpClass(cls):
        from exllamav3.ext import exllamav3_ext
        cls.ext = exllamav3_ext
        assert cls.ext.stable_arithmetic()

    def run_hints(self, device, intermediate, mcg, tile_rows = True, hints = (-1, 1, 2, 3, 4)):
        """Per active-count hint, the slot outputs of one set of random experts"""
        ext = self.ext
        gen = torch.Generator().manual_seed(2297)

        def trellis(k, n):
            return torch.randint(0, 65536, (k // 16, n // 16, 16 * self.K), dtype = torch.int32,
                                 generator = gen).to(torch.int16).to(device)

        def scale(n):
            s = (torch.rand((n,), generator = gen) * 0.2 + 0.9) * torch.where(torch.rand((n,), generator = gen) < 0.5, -1.0, 1.0)
            return s.half().to(device)

        keep, tabs = [], []
        for k, n in ((self.H, intermediate), (self.H, intermediate), (intermediate, self.H)):
            t, su, sv = [trellis(k, n) for _ in range(self.E)], [scale(k) for _ in range(self.E)], [scale(n) for _ in range(self.E)]
            keep += t + su + sv
            tabs += [torch.tensor([x.data_ptr() for x in xs], dtype = torch.long, device = device) for xs in (t, su, sv)]
        x = (torch.randn((self.T, self.H), generator = gen) * 0.05).half().to(device)
        sel = torch.zeros((self.T, 2), dtype = torch.long)
        start = 0
        for e, count in enumerate(self.SECOND + (self.T - sum(self.SECOND),), 1):
            sel[start : start + count, 1] = e
            start += count
        weights = (torch.rand((self.T, 2), generator = gen) * 0.9 + 0.1).half()
        flat = sel.flatten().to(device)
        order = flat.argsort(stable = True)
        token_sorted = torch.arange(self.T, device = device).repeat_interleave(2)[order]
        weight_sorted = weights.flatten().to(device)[order]
        expert_count = torch.bincount(flat, minlength = self.E + 1)
        fused_base = torch.cumsum(expert_count, 0) - expert_count
        C = ext.exl3_moe_max_concurrency(device)
        temps = [torch.empty((C, 256, d), dtype = torch.half, device = device)
                 for d in (self.H, self.H, intermediate, intermediate)]
        codebook = (True, False) * 3 if mcg else (False, True) * 3
        results = []
        # -1: unknown count, C groups of the default width (8 blocks); 1..4: fewer, wider groups (up
        # to 32 blocks, the cap every one of them reaches on GPUs with 128 or more SMs, where the
        # comparison is then 8 against 32 blocks)
        for hint in hints:
            out = torch.zeros((self.T, self.H), dtype = torch.float, device = device)
            scratch = torch.full((token_sorted.numel(), self.H), float("nan"), dtype = torch.float, device = device)
            for lo, hi, m_tile in (self.MCG_LAUNCHES if mcg else self.MUL1_LAUNCHES):
                ext.exl3_moe(
                    x, out, expert_count, token_sorted, weight_sorted, *temps,
                    0, float(self.K), float(self.K), float(self.K), *tabs,
                    *codebook,
                    0.0, hint, scratch, fused_base, lo, hi, m_tile,
                    tile_rows = tile_rows,
                )
            torch.cuda.synchronize(device)
            if tile_rows:
                self.assertTrue(bool(torch.isfinite(scratch).all()))
            results.append(scratch)
        del keep
        return results

    def test_group_width_does_not_change_the_result(self):
        # mul1 with a 512-wide intermediate: the N = 256 16-row instance, m32 and m64 (with a
        # 16-row tile inside); mul1 at 384: the N = 128 16-row instance; mcg: its 16-row instances,
        # N = 256 at 512 and N = 128 at 384
        for device in range(torch.cuda.device_count()):
            for intermediate, mcg in ((512, False), (384, False), (512, True), (384, True)):
                results = self.run_hints(device, intermediate, mcg)
                for hint, r in zip((1, 2, 3, 4), results[1:]):
                    with self.subTest(device = device, intermediate = intermediate, mcg = mcg, hint = hint):
                        self.assertTrue(torch.equal(r, results[0]))

    def test_unstriped_launch_skips_the_oversized_experts(self):
        # The whole-K instances always run in row stripes; a launch without tile_rows gets its row
        # limit capped at the temp buffers' 256 rows, so, as with an unstriped instance, the 700- and
        # 533-row experts are skipped (their slots keep the NaN fill) and the others are computed
        counts = (self.T,) + self.SECOND + (self.T - sum(self.SECOND),)
        starts = [sum(counts[:e]) for e in range(len(counts))]
        for device in range(torch.cuda.device_count()):
            for mcg in (False, True):
                striped = self.run_hints(device, 512, mcg, hints = (-1,))[0]
                plain = self.run_hints(device, 512, mcg, tile_rows = False, hints = (-1,))[0]
                for e, (start, count) in enumerate(zip(starts, counts)):
                    with self.subTest(device = device, mcg = mcg, expert = e):
                        block = plain[start : start + count]
                        if count > 256:
                            self.assertTrue(bool(torch.isnan(block).all()))
                        else:
                            self.assertTrue(torch.equal(block, striped[start : start + count]))


if __name__ == "__main__":
    unittest.main()
