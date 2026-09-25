"""
EXL3_STABLE_ARITHMETIC=1: the Python dispatch it changes, checked on the production source without
the compiled extension (methods and conditions are compiled from the engine files and run against
stand-ins), and the whole package in a fresh process with only that variable set (needs the
prebuilt extension, no GPU; reports a skip otherwise).

    python -m pytest tests/test_stable_arithmetic_.py

The variable parser itself is covered by test_math_policy_.py, the executed CPU-tensor paths by
test_stable_arithmetic_torch_.py.
"""
import ast
import importlib.machinery
import importlib.util
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def function(path, cls, name):
    """The AST of one method of a production class, or of a module-level function (cls None)"""
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body if cls is None else \
        next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def method(path, cls, name, ns):
    """Compile one method of a production class into `ns` (without its module's imports)"""
    fn = function(path, cls, name)
    fn.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body = future + [fn], type_ignores = []), str(path), "exec"), ns)
    return ns[name]


def condition(path, cls, name, marker):
    """The test of the one `if`/`elif` in a production method whose source contains `marker`"""
    fn = function(path, cls, name)
    tests = [n.test for n in ast.walk(fn) if isinstance(n, ast.If) and marker in ast.unparse(n.test)]
    assert len(tests) == 1, (name, marker, len(tests))
    return compile(ast.Expression(tests[0]), str(path), "eval")


def assignment(path, cls, name, target):
    fn = function(path, cls, name)
    value = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == target for t in n.targets))
    return compile(ast.Expression(value), str(path), "eval")


def module_assignment(path, target):
    """The value of the module-level assignment `target = ...` of a production file"""
    tree = ast.parse((ROOT / path).read_text())
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == target for t in n.targets))


def jit_debug_arguments(path):
    """{name: source of the debug= argument (None without one)} for every @triton.jit kernel"""
    out = {}
    for n in ast.parse((ROOT / path).read_text()).body:
        if not isinstance(n, ast.FunctionDef):
            continue
        for d in n.decorator_list:
            call = d if isinstance(d, ast.Call) else None
            if ast.unparse(call.func if call else d) == "triton.jit":
                out[n.name] = next((ast.unparse(k.value) for k in call.keywords if k.arg == "debug"), None) \
                    if call else None
    return out


class QuantizedLinearTests(unittest.TestCase):
    PATH = "exllamav3/modules/quant/exl3.py"

    def test_no_row_threshold(self):
        ns = dict(AUTO_RECONSTRUCT_THRESHOLD = 144, torch = NS(float = "float"))
        forward = method(self.PATH, "LinearEXL3", "forward", ns)
        layer = NS(key = "test", config = NS(infer_params = NS(no_reconstruct = False)),
                   default_out_dtype = "half", out_features = 128,
                   bc = NS(run_alloc = Mock(return_value = "direct")), reconstruct_hgemm = Mock(return_value = "folded"))
        for stable in (False, True):
            ns["STABLE_ARITHMETIC"] = stable
            for rows in (1, 7, 8, 9, 17, 128, 144, 145, 257, 1023, 1024, 4096):
                x = NS(shape = (1, rows, 128), numel = lambda: rows * 128, is_contiguous = lambda: True)
                with self.subTest(stable = stable, rows = rows):
                    self.assertEqual(forward(layer, x, {}), "folded" if stable or rows > 144 else "direct")

    def test_original_basis_at_every_row_count(self):
        code = assignment(self.PATH, "LinearEXL3", "reconstruct_hgemm", "use_fused")
        for rows in (1, 7, 128, 257, 1023, 1024, 4096):
            for stable in (False, True):
                ns = dict(self = NS(_fused_reconstruct = True), rows = rows, STABLE_ARITHMETIC = stable)
                self.assertEqual(eval(code, ns), stable or rows >= 1024)

    def test_refusals_at_load(self):
        ns = dict(torch = torch, ext = NS(BC_LinearEXL3 = Mock()), g_tensor_cache = NS(get = Mock()),
                  frac_k = lambda K: None)
        init = method(self.PATH, "LinearEXL3", "__init__", ns)

        def build(stable, no_reconstruct = False, k = 256, n = 256):
            ns["STABLE_ARITHMETIC"] = stable
            layer = type("Layer", (), {})()
            init(layer, NS(infer_params = NS(no_reconstruct = no_reconstruct)), k, n,
                 suh = torch.ones(k, dtype = torch.half), svh = torch.ones(n, dtype = torch.half),
                 trellis = torch.zeros((k // 16, n // 16, 48), dtype = torch.int16), key = "test")
            return layer

        build(False, True)
        build(False, False, 272, 256)
        build(True)
        with self.assertRaisesRegex(ValueError, "no_reconstruct"):
            build(True, True)
        with self.assertRaisesRegex(ValueError, "128-aligned"):
            build(True, False, 272, 256)


class MoeDispatchTests(unittest.TestCase):
    PATH = "exllamav3/modules/block_sparse_mlp.py"

    def test_every_row_count_takes_the_fused_path(self):
        eligible = assignment(self.PATH, "BlockSparseMLP", "forward", "bszn_eligible")
        fused_path = condition(self.PATH, "BlockSparseMLP", "forward", "self.f_threshold")
        layer = NS(bc = object(), f_threshold = 4, is_quantized = True, support_quant_paths = True,
                   config = NS(infer_params = NS(no_reconstruct = False)))
        for stable in (False, True):
            for bsz in (1, 2, 3, 4, 8, 9, 32, 33, 4096):
                ns = dict(self = layer, bsz = bsz, MAX_BSZN = 8, STABLE_ARITHMETIC = stable)
                ns["bszn_eligible"] = eval(eligible, ns)
                with self.subTest(stable = stable, bsz = bsz):
                    self.assertEqual(ns["bszn_eligible"], not stable and bsz <= 8)
                    self.assertEqual(bool(eval(fused_path, ns)), stable or bsz > 8)

    def test_shared_gate_projection_at_every_row_count(self):
        code = condition(self.PATH, "BlockSparseMLP", "forward", "bsz > 32")
        for stable in (False, True):
            for bsz in (1, 8, 32, 33):
                self.assertEqual(bool(eval(code, dict(STABLE_ARITHMETIC = stable, bsz = bsz))), stable or bsz > 32)


class CpuHostTests(unittest.TestCase):

    def register(self, stable, mode, whole_k_built = True):
        spec = importlib.util.spec_from_file_location("math_policy_under_test", ROOT / "exllamav3/model/math_policy.py")
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        ns = dict(torch = torch, STABLE_ARITHMETIC = stable, validate_streaming = Mock(),
                  require_whole_k_moe = lambda key, built: policy.require_whole_k_moe(key, built, stable = stable),
                  ext = NS(exl3_moe_whole_k_built = lambda: whole_k_built))
        register_layer = method("exllamav3/model/moe_cpu_host.py", "MoeCpuHost", "register_layer", ns)
        host = NS(by_key = {}, specs = [], started = False, _spawn = Mock(), live_layers = 0, aux = {},
                  conn = NS(send = Mock()), wslot_size = 1 << 30, num_wslots = 2)
        dims = {p: (64, 64, 3) for p in ("g", "u", "d")}
        idx = register_layer(host, "model.layers.3.mlp", ["g"], ["u"], ["d"], 0, 0.0, 64, 64, 2,
                             proj_dims = dims, aux = {}, device = "cuda:0", mode = mode)
        return idx, host

    def test_worker_computed_experts_refused_before_the_worker_starts(self):
        for mode in ("hybrid", "stream"):
            self.assertEqual(self.register(False, mode)[0], 0)
        self.assertEqual(self.register(True, "stream")[0], 0)
        with self.assertRaisesRegex(RuntimeError, "EXL3_STABLE_ARITHMETIC=1") as cm:
            self.register(True, "hybrid")
        self.assertIn("stream_only", str(cm.exception))
        # next to a placement -mcm is refused, so the message also names the way out for -dmcl layers
        self.assertIn("keep those layers' experts in VRAM", str(cm.exception))

    def test_streamed_experts_refused_by_a_build_without_whole_k_kernels(self):
        # EXLLAMA_NO_WHOLE_K_MOE leaves out the whole-K fused kernels the profile runs streamed experts with
        self.assertEqual(self.register(False, "stream", whole_k_built = False)[0], 0)
        with self.assertRaisesRegex(ValueError, "EXLLAMA_NO_WHOLE_K_MOE") as cm:
            self.register(True, "stream", whole_k_built = False)
        self.assertIn("model.layers.3.mlp", str(cm.exception))


class DeepseekV41AttentionTests(unittest.TestCase):

    def test_one_softmax_partition_in_every_attention_call(self):
        tree = ast.parse((ROOT / "exllamav3/modules/dsv41.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "dsa_attn"]
        self.assertTrue(calls)
        for call in calls:
            keyword = next((k for k in call.keywords if k.arg == "n_splits"), None)
            self.assertIsNotNone(keyword, f"dsa_attn call at line {call.lineno} without n_splits")
            code = compile(ast.Expression(keyword.value), "exllamav3/modules/dsv41.py", "eval")
            self.assertEqual(eval(code, dict(STABLE_ARITHMETIC = False)), 0)
            self.assertEqual(eval(code, dict(STABLE_ARITHMETIC = True)), 1)

    def test_quantized_cache_is_refused(self):
        # A packed pool is read with row-count-dependent arithmetic (staged to FP16 from 64 rows)
        n = NS(compressed = False, contract = "deepseek")
        for stable in (False, True):
            register = method("exllamav3/architecture/deepseek_v41.py", "DeepseekV41Config",
                              "register_packed_pool", dict(STABLE_ARITHMETIC = stable))
            config = NS(_dsv41_numerics = n, _dsv41_packed_pools = set())
            if stable:
                with self.assertRaisesRegex(ValueError, "quantized Cache"):
                    register(config, "pool")
                self.assertEqual(config._dsv41_packed_pools, set())
            else:
                register(config, "pool")
                self.assertEqual(config._dsv41_packed_pools, {"pool"})

    def test_index_scores_take_the_tiled_kernel_at_every_row_count(self):
        for stable in (False, True):
            ns = dict(torch = torch, STABLE_ARITHMETIC = stable, _NEG_INF = float("-inf"))
            score = method("exllamav3/modules/dsv41_select.py", "_ExtSlabs", "score", ns)
            slabs = NS(backing = torch.zeros(4 * 256), s_stride = 256, q = torch.zeros(4, 2, 8),
                       w = torch.zeros(4, 2), pos0 = 300, m = 1, pool_flat = torch.zeros(320, 8),
                       scale = 1.0, scores_fn = Mock())
            for bt, epp in ((None, 0), (torch.arange(5, dtype = torch.int32), 64)):
                slabs.bt, slabs.epp = bt, epp
                for rows in (1, 4):
                    score(slabs, 0, rows, 0, 300, 304)
            self.assertEqual(slabs.scores_fn.call_count, 4)
            for call in slabs.scores_fn.call_args_list:
                self.assertIs(call.kwargs["few_query"], not stable)
        code = condition("exllamav3/modules/attention_fn/dsa_triton.py", None, "dsa_indexer_scores",
                         "few_query")
        for rows in range(1, 9):
            for few_query in (True, False):
                self.assertEqual(bool(eval(code, dict(R = rows, few_query = few_query))),
                                 rows <= 4 and few_query)


class DeepseekV41CompressorTests(unittest.TestCase):
    PATH = "exllamav3/architecture/dsv41/compressor.py"

    def test_row_local_norm_for_cuda_inputs_only(self):
        code = condition(self.PATH, None, "rms_norm", "STABLE_ARITHMETIC")
        for stable in (False, True):
            for cuda in (False, True):
                ns = dict(STABLE_ARITHMETIC = stable, x = NS(is_cuda = cuda))
                self.assertEqual(bool(eval(code, ns)), stable and cuda)

    def test_cpu_reference_unchanged_under_the_profile(self):
        ns = dict(math = math, torch = torch, _MIN_EPS = 2.0 ** -126, _MAX_EPS = float.fromhex("0x1.fffffep+127"))
        g = torch.Generator().manual_seed(3)
        x = torch.randn(37, 512, generator = g) * torch.logspace(-30, 30, 37).unsqueeze(1)
        w = 1 + torch.randn(512, generator = g) * 0.1
        results = []
        for stable in (False, True):
            ns["STABLE_ARITHMETIC"] = stable
            rms_norm = method(self.PATH, None, "rms_norm", dict(ns))
            results.append(rms_norm(x, w, 1e-20))
        self.assertTrue(torch.equal(*results))


class DeepseekV41DebugAssertTests(unittest.TestCase):
    """
    The V4.1 Triton kernels compile their device assertions (Triton's debug mode) only with
    EXL3_DSA_DEBUG_BOUNDS, read at import with the same test as the DSA kernels; a default build
    compiles them without
    """
    FILES = ("exllamav3/modules/dsv41_compress.py",)

    def test_every_kernel_follows_the_variable(self):
        upstream = ast.unparse(module_assignment("exllamav3/modules/attention_fn/dsa_triton.py",
                                                 "dsa_debug_bounds"))
        for path in self.FILES:
            with self.subTest(path = path):
                kernels = jit_debug_arguments(path)
                self.assertTrue(kernels)
                self.assertEqual(set(kernels.values()), {"DEBUG_ASSERTS"}, kernels)
                value = module_assignment(path, "DEBUG_ASSERTS")
                self.assertEqual(ast.unparse(value), upstream)
                code = compile(ast.Expression(value), path, "eval")
                for environ, expected in (({}, False), ({"EXL3_DSA_DEBUG_BOUNDS": "0"}, False),
                                          ({"EXL3_DSA_DEBUG_BOUNDS": "1"}, True)):
                    self.assertIs(eval(code, dict(os = NS(environ = environ))), expected)


def _extension_missing():
    """
    Why the package test cannot run here, or None. It needs the prebuilt extension (importing
    exllamav3 without one would build it) with the stable-arithmetic functions; no GPU is used
    """
    spec = importlib.util.find_spec("exllamav3_ext")
    if not spec or not spec.origin or not any(spec.origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
        return "needs the prebuilt extension (exllamav3_ext); importing exllamav3 would build it"
    try:
        import exllamav3_ext
    except ImportError as e:
        return f"the prebuilt extension does not import ({e})"
    if not hasattr(exllamav3_ext, "stable_arithmetic"):
        return "the prebuilt extension predates EXL3_STABLE_ARITHMETIC (no stable_arithmetic); rebuild it"
    return None


@unittest.skipIf(_extension_missing(), _extension_missing())
class PackageTests(unittest.TestCase):

    def test_only_the_variable_set_in_a_fresh_process(self):
        code = "\n".join([
            "import exllamav3",
            "from exllamav3.ext import exllamav3_ext as ext",
            "from exllamav3.modules import mlp, block_sparse_mlp",
            "from exllamav3.modules.quant import exl3, fp16",
            "from exllamav3.model import moe_cpu_host",
            "from exllamav3.modules import dsv4, dsv41, dsv41_block, dsv41_select",
            "from exllamav3.architecture.dsv41 import compressor",
            "print(mlp.STABLE_ARITHMETIC, block_sparse_mlp.STABLE_ARITHMETIC, block_sparse_mlp.FUSED_PREFILL,",
            "      exl3.STABLE_ARITHMETIC, fp16.HGEMM_FIXED_ROWS, moe_cpu_host.STABLE_ARITHMETIC,",
            "      moe_cpu_host.TUNING.fused_prefill, ext.stable_arithmetic(), ext.hgemm_fixed_rows(),",
            "      dsv4.STABLE_ARITHMETIC, dsv41.STABLE_ARITHMETIC, dsv41_block.STABLE_ARITHMETIC,",
            "      dsv41_select.STABLE_ARITHMETIC, compressor.STABLE_ARITHMETIC)",
        ])
        names = ("EXL3_STABLE_ARITHMETIC", "EXL3_HGEMM_FIXED_ROWS", "EXL3_MOE_FUSED_PREFILL",
                 "EXL3_MOE_FUSED_DET", "EXL3_NO_FUSED_RECONSTRUCT")
        environ = {k: v for k, v in os.environ.items() if k not in names}
        environ["EXL3_STABLE_ARITHMETIC"] = "1"
        environ["CUDA_VISIBLE_DEVICES"] = ""            # the policy needs no GPU
        r = subprocess.run([sys.executable, "-c", code], env = environ, capture_output = True, text = True,
                           cwd = ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["True"] * 4 + ["128"] + ["True"] * 3 + ["128"] + ["True"] * 5)


if __name__ == "__main__":
    unittest.main()
