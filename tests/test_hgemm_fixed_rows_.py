"""
EXL3_HGEMM_FIXED_ROWS=128: fixed 128-row cuBLAS geometry for the native GEMMs, the FP16 linear
dispatch that follows it, and the eager fallback of the native graph wrapper.

The CPU classes (a source check of every native graph capture, the LinearFP16 dispatch) run
anywhere, without the compiled extension. The GPU classes need a process started with the
variable set (or EXL3_STABLE_ARITHMETIC=1, which implies it), because it is read once (at import
in Python, on first use natively):

    EXL3_HGEMM_FIXED_ROWS=128 python -m pytest tests/test_hgemm_fixed_rows_.py
    EXL3_HGEMM_FIXED_ROWS=128 EXL3_TEST_MODEL=/path/to/exl3/model python -m pytest tests/test_hgemm_fixed_rows_.py

EXL3_TEST_MODEL (any EXL3 model with the native attention and MLP blocks) enables the decode
check; without it that test reports a skip. Row-count independence is checked bitwise. Nothing
here claims equality with the default dispatch or between GPU types.
"""
import ast
import gc
import importlib.util
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "exllamav3" / "exllamav3_ext"
# The policy as the engine reads it (math_policy.py is torch-free and loads by path):
# EXL3_HGEMM_FIXED_ROWS=128, or implied by EXL3_STABLE_ARITHMETIC=1
_spec = importlib.util.spec_from_file_location("_math_policy", ROOT / "exllamav3/model/math_policy.py")
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
FIXED_ROWS = _policy.hgemm_fixed_rows() == 128
NEEDS = "needs CUDA and fixed-row GEMMs (EXL3_HGEMM_FIXED_ROWS=128 or EXL3_STABLE_ARITHMETIC=1)"


def function_bodies(text):
    """(name line, body) of every function definition in a native source file, relying on this
    code base's layout: a definition's braces stand alone at column 0, and its name line is the
    last column-0 line before them other than the parentheses of a multi-line parameter list"""
    lines = text.split("\n")
    out, start = [], None
    for i, line in enumerate(lines):
        if line == "{" and start is None:
            start = i
        elif line == "}" and start is not None:
            j = start - 1
            while j > 0 and (not lines[j] or lines[j][0] in " \t()"):
                j -= 1
            out.append((lines[j], "\n".join(lines[start : i + 1])))
            start = None
    return out


def strip_comments(code):
    """C/C++ source without its // and /* */ comments (string literals here hold neither)"""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", code, flags = re.S))


def method(path, cls, name, ns):
    """Compile one method of a production class into `ns` (without its module's imports)"""
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body = future + [fn], type_ignores = []), str(path), "exec"), ns)
    return ns[name]


class NativeGraphFallbackTests(unittest.TestCase):
    """
    Under EXL3_HGEMM_FIXED_ROWS the native Graph wrapper is disabled (its capture owns no
    allocator pool for the GEMM scratch) and Graph::capture_begin refuses to start. Every native
    block that captures must check `disabled` first and run eagerly; one that did not would raise
    on its second call under the policy
    """

    def test_every_capture_checks_disabled_first(self):
        sites = []
        for path in sorted(NATIVE.rglob("*")):
            if path.suffix not in (".cpp", ".cu", ".cuh", ".h") or path.name == "graph.cu":
                continue
            for signature, body in function_bodies(path.read_text()):
                # comments stripped: one that mentions `disabled` must not satisfy the check
                code = strip_comments(body)
                pos = code.find("capture_begin()")
                if pos < 0:
                    continue
                sites.append(f"{path.name}: {signature.strip()}")
                with self.subTest(site = sites[-1]):
                    # the branch that guards the capture tests the graph's disabled flag first
                    self.assertRegex(code[:pos], r"if \([^)]*(->|\.)disabled\s*\|\|",
                                     "native graph capture without an eager fallback")
        # attention, MLA, MLP (2), block-sparse MLP, GDN (2), DSV4 attention (2)
        self.assertGreaterEqual(len(sites), 9, sites)

    def test_wrapper_is_disabled_by_the_policy_and_refuses_capture(self):
        bodies = dict(function_bodies((NATIVE / "graph.cu").read_text()))
        ctor = bodies["Graph::Graph()"]
        self.assertIn("disabled = hgemm_fixed_rows() != 0;", ctor)
        begin = bodies["cudaStream_t Graph::capture_begin()"]
        check = begin.find("TORCH_CHECK(!disabled")
        self.assertGreaterEqual(check, 0)
        self.assertLess(check, begin.find("cudaStreamCreateWithFlags"))
        self.assertLess(check, begin.find("cudaStreamBeginCapture"))


class LinearFP16DispatchTests(unittest.TestCase):
    """Production LinearFP16.forward, run on CPU tensors with a stand-in for the native GEMM"""

    def run_forward(self, fixed_rows, rows, dtype, out_dtype = None):
        ext = NS(hgemm = Mock(side_effect = lambda x, w, y: y.copy_(x.float() @ w.float())))
        ns = dict(torch = torch, ext = ext, HGEMM_FIXED_ROWS = fixed_rows)
        forward = method("exllamav3/modules/quant/fp16.py", "LinearFP16", "forward", ns)
        layer = NS(out_features = 8, out_dtype = None, _pinned_store = None, bias = None,
                   weight = torch.full((16, 8), 0.25, dtype = dtype))
        y = forward(layer, torch.ones((1, rows, 16), dtype = dtype), {}, out_dtype)
        self.assertEqual(y.shape, (1, rows, 8))
        self.assertTrue(bool((y == 4).all()))
        return ext.hgemm.call_count

    def test_fp16_operands_take_the_native_gemm_at_every_row_count(self):
        for rows in (1, 7, 128, 129, 300):
            self.assertEqual(self.run_forward(128, rows, torch.half), 1, rows)

    def test_other_operand_types_keep_torch_matmul(self):
        for dtype in (torch.float, torch.bfloat16):
            self.assertEqual(self.run_forward(128, 7, dtype, dtype), 0, dtype)

    def test_default_dispatch_unchanged(self):
        self.assertEqual(self.run_forward(0, 7, torch.half), 0)
        # A wider output always took the native GEMM
        for fixed_rows in (0, 128):
            self.assertEqual(self.run_forward(fixed_rows, 7, torch.half, torch.float), 1)


@unittest.skipUnless(torch.cuda.is_available() and FIXED_ROWS, NEEDS)
class FixedRowGemmTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from exllamav3.ext import exllamav3_ext
        cls.ext = exllamav3_ext

    def test_native_and_python_policies_agree(self):
        from exllamav3.model.math_policy import HGEMM_FIXED_ROWS
        self.assertEqual(self.ext.hgemm_fixed_rows(), 128)
        self.assertEqual(HGEMM_FIXED_ROWS, 128)
        for device in range(torch.cuda.device_count()):
            self.assertEqual(self.ext.hgemm_f16acc_status(device), 0)

    def test_shape_and_batch_invariance(self):
        ext = self.ext
        rng = torch.Generator().manual_seed(296)
        for k, n in ((96, 128), (1040, 200), (5120, 384)):
            x = torch.randn((4096, k), generator = rng).half()
            w = (torch.randn((k, n), generator = rng) / k ** 0.5).half()
            for device in range(torch.cuda.device_count()):
                xd, wd = x.to(device), w.to(device)
                for dtype in (torch.half, torch.float):
                    reference = torch.empty((4096, n), dtype = dtype, device = device)
                    ext.hgemm(xd, wd, reference)
                    for rows in (1, 2, 17, 127, 128, 129, 257, 2048, 4096):
                        with self.subTest(device = device, k = k, n = n, rows = rows, dtype = dtype):
                            # A strided output with spare rows (a reusable capacity buffer): the
                            # geometry must not change, and nothing outside the product is written
                            storage = torch.full((rows + 7, n + 16), 42, dtype = dtype, device = device)
                            out = storage[:, :n]
                            ext.hgemm_recon(xd[:rows], wd, out)
                            self.assertTrue(torch.equal(out[:rows], reference[:rows]))
                            self.assertTrue(bool((storage[:, n:] == 42).all()))
                            self.assertTrue(bool((storage[rows:] == 42).all()))
                    a = xd[:129].repeat(2, 1, 1)
                    b = wd.repeat(2, 1, 1)
                    c = torch.empty((2, 129, n), dtype = dtype, device = device)
                    ext.hgemm_batched(a, b, c)
                    self.assertTrue(torch.equal(c[0], reference[:129]))
                    self.assertTrue(torch.equal(c[1], reference[:129]))

    def test_nondefault_stream_and_torch_graph(self):
        ext = self.ext
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    a = torch.ones((2, 129, 96), dtype = torch.half, device = device)
                    b = torch.full((2, 96, 200), 0.25, dtype = torch.half, device = device)
                    c = torch.empty((2, 129, 200), dtype = torch.float, device = device)
                    for _ in range(3):
                        ext.hgemm_batched(a, b, c)
                stream.synchronize()
                # torch.cuda.graph owns the pool the per-call scratch comes from
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream = stream):
                    ext.hgemm_batched(a, b, c)
                for multiplier in (1.0, 2.0, 0.5):
                    a.fill_(multiplier)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    self.assertTrue(bool((c == 24 * multiplier).all()))


@unittest.skipUnless(torch.cuda.is_available() and FIXED_ROWS, NEEDS)
class EagerNativeBlockTests(unittest.TestCase):
    """Decode through the native blocks with their graphs disabled: runs, and matches the
    teacher-forced no-cache forward to kernel precision (the two paths use different kernels)"""

    def test_decode_matches_forward(self):
        model_dir = os.environ.get("EXL3_TEST_MODEL")
        if not model_dir:
            self.skipTest("set EXL3_TEST_MODEL to an EXL3 model directory")
        from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
        config = Config.from_directory(model_dir)
        model = Model.from_config(config)
        cache = Cache(model, max_num_tokens = 4096)
        model.load()
        tokenizer = Tokenizer.from_config(config)
        try:
            with torch.inference_mode():
                generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = 1)
                ids = tokenizer.encode("The quick brown fox jumps over the lazy dog because", add_bos = True)
                job = Job(input_ids = ids, max_new_tokens = 12, sampler = ArgmaxSampler(), return_logits = True)
                generator.enqueue(job)
                tokens, logits = [], []
                while generator.num_remaining_jobs():
                    for r in generator.iterate():
                        if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                            tokens += r["token_ids"].view(-1).tolist()
                            logits.append(r["logits"].view(-1, r["logits"].shape[-1])[0].clone())
                n = min(len(tokens), len(logits))
                self.assertGreaterEqual(n, 4, "too few decode steps to check")
                full = torch.cat((ids, torch.tensor([tokens[:n - 1]])), dim = 1)
                ref = model.forward(full, params = {"last_tokens_only": n})[0].float()
                for step in range(n):
                    a = logits[step].float().to(ref.device).view(-1)
                    b = ref[step].view(-1)
                    m = torch.isfinite(a) & torch.isfinite(b)
                    rfn = ((a[m] - b[m]).norm() / b[m].norm()).item()
                    self.assertLess(rfn, 0.05, f"decode step {step}")
        finally:
            model.unload()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
