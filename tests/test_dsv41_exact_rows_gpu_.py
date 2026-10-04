"""
GPU, EXL3_EXACT_ROWS=1: a DeepSeek-V4.1 operation that follows the row count of its call gives a
row of a flagged call of K rows, K = 2 to 8, the bits a one-row call gives that row.

The model is loaded as the environment places it (EXL3_PLACEMENT, CUDA_VISIBLE_DEVICES), so every
GPU holds its own layers, and on each GPU the first layer that has the operation is tested. The
K-row call runs with params {"exact_rows": True}, the flag DeepseekV41Model.prepare_inputs sets,
on one tensor of K rows; the K one-row calls run with a plain params dict on standalone clones of
those rows, so the comparison also covers what the mode rests on: that a one-row call gives a row
the same bits wherever the row lies in memory. All comparisons are torch.equal.

Covered: every Linear of the trunk (attention wq_a, wq_b, wkv, wo_b; compressor wkv, wgate;
indexer wq_b, weights_proj, wk; engram wkv; the output head), the grouped output projection
(DSV41Attention._project_o_rows), the MoE layer (router, expert cache lookup, routed and shared
experts), the hyper-connection mix and both collapses, the engram gate (DSV41Engram._gate_rows),
and the compressor's pooling and norm at rates 1 and 2. Two controls check, on every GPU, that
the comparison discriminates: an unflagged call of several rows must differ from the one-row
calls (with the int8 activation path on: with EXL3_INT8_GEMV=0 it need not), and the rows of an
EXL3 linear launched with the kernel shape of a one-row call and another block count must differ
from the one-row calls (with the int8 path off, which the test sets in-process). The converse,
test_forced_shape_rows_equal_one_row_calls, runs with every extension: K rows launched with a
forced kernel shape and block count equal one-row calls launched with the same, for every kernel
shape a linear's widths take, with FP16 and FP32 output.

Not covered: the index selection and the attention kernel, which need real pools, and the order
of these operations in a forward. tools/dsv41_rowprobe.py under EXL3_EXACT_ROWS=1 is the evidence
for those (idx.scores, idx.topk, attn.dsa_attn and the forward's logits in its tables).

A flagged call takes the extension's row-exact entry points where it has them (exact_rows_caps:
the rows of an EXL3 linear, of the grouped output projection and of the router in one native call,
the MoE experts of all rows in one launch pair, the hyper-connection sums), else the Python loops
of one call per row. The tests above cover whichever the extension gives. The test_native_* tests
need the entry points and are skipped, with a printed note, for an extension built before them:
each operation with the entry points against the Python row loop on the same input (the consumers'
ROWS_NATIVE set to 0), with the row loops made to raise where the entry point must serve; the
linears on the activations of a real forward too; the expert launch pair and the router against
one-row calls on crafted and natural routing; and what each entry point refuses.

With EXL3_INT8_GEMV=0, an extension that reports bit 32 (EXACT_ROWS_CAP_ONE_LAUNCH) makes ONE
launch for the rows of an EXL3 linear and of the grouped projection, under the launch record of a
one-row call, where that call is the cooperative FP16 kernel and the GPU is one of the types the
launch was verified on (ONE_LAUNCH_SM). The rows have the same bits either way, so the
test_one_launch_* tests read the extension's count of such calls (exact_rows_one_launches): with
the int8 path off (set in-process: the extension reads the variable on every launch) a flagged
call must be one launch and equal the one-row calls of that setting, on random rows and on the
activations of a real forward, and a first flagged call that finds no launch record of the
one-row call must launch per row; with the int8 path on, no call may be one launch, and the rows
equal that setting's one-row calls. Run the whole file under both settings, each with one
autotune file:

    EXL3_EXACT_ROWS=1 DSV41_MODEL_DIR=<checkpoint> python -m pytest tests/test_dsv41_exact_rows_gpu_.py
    EXL3_EXACT_ROWS=1 EXL3_INT8_GEMV=0 DSV41_MODEL_DIR=<checkpoint> python -m pytest tests/test_dsv41_exact_rows_gpu_.py

Needs CUDA, the compiled extension and the checkpoint; skipped without the switch. Loads the whole
model: run it alone on the host.
"""
import importlib
import os
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

ROWS = range(2, 9)
FLAGGED = {"exact_rows": True}
# The row-exact entry points of the extension (math_policy.EXACT_ROWS_CAP_*), and the modules that
# hold what the extension reports (ROWS_NATIVE)
CAP_LINEAR, CAP_MGEMM, CAP_ROUTER, CAP_MOE, CAP_HC = 1, 2, 4, 8, 16
# Not an entry point: those of CAP_LINEAR and CAP_MGEMM make one launch for the rows where they can
CAP_ONE_LAUNCH = 32
# The compute capabilities on which the extension makes that launch (exl3_gemm_one_row_route_ok)
ONE_LAUNCH_SM = ((8, 0), (8, 9), (12, 0))
# What the extension takes when EXL3_INT8_GEMV is not set (exl3_gemv_int8.cu): the plain int8 mode
INT8_GEMV_DEFAULT = "2"
NATIVE_CONSUMERS = ("linear", "dsv41", "dsv41_moe", "block_sparse_mlp", "block_sparse_mlp_routing", "dsv41_block")
# name, the Linear of a block or None, input dims, out_dtype the engine passes
LINEAR_OPS = (
    ("attn.wq_a", lambda b: b.attn.q_a, 3, None),
    ("attn.wq_b", lambda b: b.attn.q_b, 3, None),
    ("attn.wkv", lambda b: b.attn.wkv, 3, None),
    ("attn.wo_b", lambda b: b.attn.wo_b, 3, None),
    ("comp.wkv", lambda b: b.attn.compressor.wkv if b.attn.compressor else None, 3, None),
    ("comp.wgate", lambda b: b.attn.compressor.wgate if b.attn.compressor else None, 3, None),
    ("idx.wq_b", lambda b: b.attn.indexer.wq_b if b.attn.indexer else None, 3, None),
    ("idx.weights_proj", lambda b: b.attn.indexer.weights_proj if b.attn.indexer else None, 3, None),
    ("idx.wk", lambda b: b.attn.indexer.wk if b.attn.indexer else None, 2, None),
    ("engram.wkv", lambda b: b.engram.wkv if b.engram is not None else None, 3, torch.float),
)


@contextmanager
def rows_native(caps: int):
    """The consumers' ROWS_NATIVE set to caps for the block: what the extension reports, or 0 for
    the Python row loops alone."""
    mods = [importlib.import_module(f"exllamav3.modules.{name}") for name in NATIVE_CONSUMERS]
    saved = [m.ROWS_NATIVE for m in mods]
    for m in mods:
        m.ROWS_NATIVE = caps
    try:
        yield
    finally:
        for m, value in zip(mods, saved):
            m.ROWS_NATIVE = value


def _refuse(*args, **kwargs):
    raise AssertionError("the Python row loop ran where the row-exact entry point must serve")


@contextmanager
def int8_gemv(value: str):
    """EXL3_INT8_GEMV replaced for the launches inside: the extension reads the variable on every
    launch (exl3_gemv_int8.cu, exl3_gemv_int8_mode). Only ever replaced, never added or removed:
    native threads call getenv at run time, and a replaced value leaves the environment array as
    it is (setUpClass sets the variable before the model, and its threads, exist)."""
    old = os.environ["EXL3_INT8_GEMV"]
    os.environ["EXL3_INT8_GEMV"] = value
    try:
        yield
    finally:
        os.environ["EXL3_INT8_GEMV"] = old


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ExactRows(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from exllamav3.model.math_policy import EXACT_ROWS
        if not EXACT_ROWS:
            raise unittest.SkipTest("needs EXL3_EXACT_ROWS=1")
        path = os.environ.get("DSV41_MODEL_DIR")
        if not path:
            raise unittest.SkipTest("needs DSV41_MODEL_DIR")
        # Before the model and its threads exist (int8_gemv): the default, which changes nothing
        os.environ.setdefault("EXL3_INT8_GEMV", INT8_GEMV_DEFAULT)
        # The setting of this process, and a setting with the int8 path on for the tests that need one
        cls.int8 = os.environ["EXL3_INT8_GEMV"]
        cls.int8_off = int(cls.int8) == 0
        cls.int8_on = INT8_GEMV_DEFAULT if cls.int8_off else cls.int8
        from exllamav3 import Cache, Config, Model
        config = Config.from_directory(path)
        cls.model = model = Model.from_config(config)
        # as serving loads it: with a Cache attached, which the expert cache and the pool routes need
        cls.cache = Cache(model, max_num_tokens = 65536, max_batch_size = 1)
        model.load(progressbar = False, max_chunk_size = 2048, max_output_size = 32)
        n = config.num_hidden_layers
        cls.blocks = model.modules[model.first_block_idx : model.first_block_idx + n]
        cls.tail = {m.key: m for m in model.modules[model.first_block_idx + n:]}
        cls.devices = []
        for b in cls.blocks:
            if torch.device(b.device) not in cls.devices:
                cls.devices.append(torch.device(b.device))
        cls.rng = torch.Generator().manual_seed(20261004)
        from exllamav3.ext import exllamav3_ext as ext
        cls.ext = ext
        cls.caps = int(ext.exact_rows_caps()) if hasattr(ext, "exact_rows_caps") else 0
        cls.real_inputs = None
        cls.real_woa = {}
        print(f" -- exact rows: {n} layers on {', '.join(str(d) for d in cls.devices)}; row-exact entry points of "
              f"the extension: {cls.caps or 'none (the Python row loops serve every flagged call)'}; "
              f"EXL3_INT8_GEMV={cls.int8}", flush = True)

    @classmethod
    def tearDownClass(cls):
        model = getattr(cls, "model", None)
        if model is not None:
            model.unload()

    # -- helpers --

    def randn(self, shape, device, dtype = torch.half, scale = 1.0):
        return (torch.randn(shape, generator = self.rng) * scale).to(dtype).to(device)

    def first(self, device, have):
        """The first block on `device` for which have(block) is true, or None."""
        return next((b for b in self.blocks if torch.device(b.device) == device and have(b)), None)

    def same_rows(self, what, many, ones):
        """many: the flagged call's result, rows along dim 0; ones: the one-row results."""
        self.assertEqual(many.shape[0], len(ones), what)
        for j, one in enumerate(ones):
            self.assertEqual(many[j].dtype, one.dtype, f"{what} row {j}")
            self.assertTrue(torch.equal(many[j].reshape(-1), one.reshape(-1)), f"{what}: row {j} differs")

    def linear_rows(self, what, lin, x, out_dtype = None):
        """lin on x (the rows along the dim before the last) flagged, against one call per cloned row."""
        K = x.numel() // x.shape[-1]
        many = lin.forward(x, dict(FLAGGED), out_dtype).clone()
        ones = []
        for j in range(K):
            row = x.reshape(K, -1)[j : j + 1].clone().view(*([1] * (x.dim() - 1)), x.shape[-1])
            ones.append(lin.forward(row, {}, out_dtype).clone())
        self.same_rows(what, many.reshape(K, -1), ones)

    # -- the tests --

    @torch.inference_mode()
    def test_linears(self):
        ops = LINEAR_OPS
        tested = set()
        for device in self.devices:
            for name, get, dims, out_dtype in ops:
                b = self.first(device, lambda blk: get(blk) is not None)
                if b is None:
                    continue
                lin = get(b)
                tested.add(name)
                for K in ROWS:
                    with self.subTest(op = name, layer = b.layer_idx, device = str(device), K = K):
                        shape = (1, K, lin.in_features) if dims == 3 else (K, lin.in_features)
                        self.linear_rows(f"{name} L{b.layer_idx}", lin, self.randn(shape, device), out_dtype)
        self.assertEqual(tested, {name for name, _, _, _ in ops}, "an operation exists on no device")
        head = self.tail["head"]
        for K in ROWS:
            with self.subTest(op = "head", device = str(head.device), K = K):
                self.linear_rows("head", head, self.randn((1, K, head.in_features), head.device))

    @torch.inference_mode()
    def test_control_unflagged_calls_differ(self):
        # Without the flag a call of several rows takes other kernels than one-row calls (measured
        # with tools/dsv41_rowprobe.py on DeepSeek-V4.1-Flash: attn.wq_b from 2 rows on layers 0
        # to 10, attn.wq_a and attn.wq_b from 3 rows on every layer). If none differs on a GPU, the
        # comparisons of this file show nothing there. With the int8 activation path off
        # (EXL3_INT8_GEMV=0) one-row and multi-row calls take the same cooperative kernel, and
        # where the launch autotuner holds one record for every row bucket they are equal:
        # test_control_forced_grid_differs is the control of that setting
        if self.int8_off:
            note = "skipped test_control_unflagged_calls_differ: EXL3_INT8_GEMV=0, where unflagged calls can " \
                   "equal one-row calls; test_control_forced_grid_differs is the control there"
            print(f" -- exact rows: {note}", flush = True)
            self.skipTest(note)
        for device in self.devices:
            b = self.first(device, lambda blk: True)
            differing = []
            for name, lin in (("attn.wq_a", b.attn.q_a), ("attn.wq_b", b.attn.q_b)):
                for K in ROWS:
                    x = self.randn((1, K, lin.in_features), device)
                    many = lin.forward(x, {}).clone()
                    ones = [lin.forward(x[:, j:j + 1].clone(), {}).clone() for j in range(K)]
                    if any(not torch.equal(many[0, j].reshape(-1), ones[j].reshape(-1)) for j in range(K)):
                        differing.append(f"{name} K={K}")
            print(f" -- exact rows: control on {device} (L{b.layer_idx}): unflagged calls differ from one-row "
                  f"calls for {', '.join(differing) or 'NOTHING'}", flush = True)
            with self.subTest(device = str(device)):
                self.assertTrue(differing, f"unflagged calls of 2 to 8 rows equal the one-row calls on {device}: "
                                           f"this test does not discriminate there")

    def one_row_tag(self, inner, device, fp32: bool) -> int:
        """The kernel ext.exl3_gemm launches for one row of this EXL3 linear under the current
        EXL3_INT8_GEMV: 0 the int8 GEMV, 90 the FP16 GEMV, 1 to 4 the cooperative kernel's shape."""
        A = self.randn((1, inner.in_features), device)
        C = torch.empty((1, inner.out_features), dtype = torch.float if fp32 else torch.half, device = device)
        return int(self.ext.exl3_gemm(A, inner.trellis, C, inner.suh, torch.empty_like(A), inner.svh,
                                      -1, bool(inner.mcg), bool(inner.mul1), 0))

    @torch.inference_mode()
    def test_control_forced_grid_differs(self):
        # With the int8 path off, the rows of a call and one-row calls take the same cooperative
        # kernel, and what separates them is the launch: the block count decides where a column's K
        # reduction is cut and handed from block to block through the FP16 output. Here the K rows
        # are launched with the kernel shape of the one-row call and 2, then 4 blocks: on every GPU
        # at least one such launch must differ from the one-row calls, or equal rows in this file
        # show nothing about the launch. Raw kernel calls on both sides (ext.exl3_gemm)
        for device in self.devices:
            b = self.first(device, lambda blk: type(getattr(blk.attn.q_a, "inner", None)).__name__ == "LinearEXL3")
            with self.subTest(device = str(device)):
                self.assertIsNotNone(b, f"no layer on {device} has an EXL3 attn.wq_a")
                inner = b.attn.q_a.inner
                fp32 = inner.default_out_dtype == torch.float
                gemm = lambda A, C, shape, blocks: int(self.ext.exl3_gemm(
                    A, inner.trellis, C, inner.suh, torch.empty_like(A), inner.svh, shape, bool(inner.mcg),
                    bool(inner.mul1), blocks))
                out = lambda rows: torch.empty((rows, inner.out_features), dtype = torch.float if fp32 else torch.half,
                                               device = device)
                differing, tags = [], set()
                with int8_gemv("0"):
                    for K in ROWS:
                        x = self.randn((K, inner.in_features), device)
                        ones = []
                        for j in range(K):
                            C1 = out(1)
                            tags.add(gemm(x[j:j + 1].clone(), C1, -1, 0))
                            ones.append(C1.reshape(-1).clone())
                        self.assertEqual(len(tags), 1, f"one-row calls took the kernels {sorted(tags)}")
                        tag = next(iter(tags))
                        self.assertIn(tag, range(1, 5), "a one-row call with the int8 path off is not the cooperative kernel")
                        for blocks in (2, 4):
                            C = out(K)
                            self.assertEqual(gemm(x, C, tag, blocks), tag)
                            if any(not torch.equal(C[j], ones[j]) for j in range(K)):
                                differing.append(f"K={K} with {blocks} blocks")
                print(f" -- exact rows: forced-grid control on {device} (L{b.layer_idx} attn.wq_a, kernel shape "
                      f"{sorted(tags)}): rows differ from one-row calls for {', '.join(differing) or 'NOTHING'}",
                      flush = True)
                self.assertTrue(differing, f"K rows under the one-row kernel shape with 2 and 4 blocks equal the "
                                           f"one-row calls on {device}: the comparison does not see the launch")

    @torch.inference_mode()
    def test_forced_shape_rows_equal_one_row_calls(self):
        # What the single launch rests on, for every kernel shape and not only the one the launch
        # autotuner picked on this host: with the kernel shape and the block count fixed, the
        # cooperative kernel gives a row of a K-row call the bits of a one-row call. Raw kernel
        # calls on both sides (ext.exl3_gemm with a forced shape and block count, the int8 path
        # off), FP16 and FP32 output, on two linears whose widths take every shape between them
        ext = self.ext
        if not hasattr(ext, "exl3_gemm_shape_compat"):
            note = "skipped test_forced_shape_rows_equal_one_row_calls: the extension has no exl3_gemm_shape_compat"
            print(f" -- exact rows: {note}", flush = True)
            self.skipTest(note)
        for device in self.devices:
            is_exl3 = lambda lin: type(getattr(lin, "inner", None)).__name__ == "LinearEXL3"
            b = self.first(device, lambda blk: is_exl3(blk.attn.q_a) and is_exl3(blk.attn.q_b))
            with self.subTest(device = str(device)):
                self.assertIsNotNone(b, f"no layer on {device} has EXL3 attn.wq_a and attn.wq_b")
            if b is None:
                continue
            covered = set()
            for name, lin in (("attn.wq_a", b.attn.q_a), ("attn.wq_b", b.attn.q_b)):
                inner = lin.inner
                bits, half_k = int(inner.K), float(inner.K) != int(inner.K)
                with torch.cuda.device(device):     # the shape filter asks the current device
                    shapes = [shape for shape in range(1, ext.exl3_gemm_num_kernel_shapes() + 1)
                              if ext.exl3_gemm_shape_compat(shape, 1, inner.in_features, inner.out_features, bits, half_k)]
                gemm = lambda A, C, shape, blocks: int(ext.exl3_gemm(
                    A, inner.trellis, C, inner.suh, torch.empty_like(A), inner.svh, shape, bool(inner.mcg),
                    bool(inner.mul1), blocks))
                for shape in shapes:
                    for dtype in (torch.half, torch.float):
                        for blocks in (7, 32):
                            for K in ROWS:
                                with self.subTest(op = name, device = str(device), shape = shape, dtype = str(dtype),
                                                  blocks = blocks, K = K), int8_gemv("0"):
                                    x = self.randn((K, inner.in_features), device)
                                    ones = []
                                    for j in range(K):
                                        C1 = torch.empty((1, inner.out_features), dtype = dtype, device = device)
                                        self.assertEqual(gemm(x[j:j + 1].clone(), C1, shape, blocks), shape)
                                        ones.append(C1.reshape(-1).clone())
                                    C = torch.empty((K, inner.out_features), dtype = dtype, device = device)
                                    self.assertEqual(gemm(x, C, shape, blocks), shape)
                                    self.same_rows(f"{name} L{b.layer_idx} @ {device}, shape {shape}, {blocks} blocks, "
                                                   f"{dtype}", C, ones)
                                    covered.add(shape)
            print(f" -- exact rows: forced shapes on {device} (L{b.layer_idx} attn.wq_a, attn.wq_b; 7 and 32 blocks; FP16 "
                  f"and FP32 output): K rows against one-row calls for kernel shapes {sorted(covered) or 'NONE'}", flush = True)
            with self.subTest(device = str(device)):
                self.assertTrue(covered, f"no kernel shape was tested on {device}")

    @torch.inference_mode()
    def test_grouped_output_projection(self):
        # DSV41Attention._project_o_rows, which a flagged _forward_cached_row calls on dsa_attn's
        # group-major (groups, rows, width) output
        for device in self.devices:
            at = self.first(device, lambda blk: True).attn
            G, width = at.o_groups, at.num_q_heads // at.o_groups * at.head_dim
            for K in ROWS:
                with self.subTest(layer = at.layer_idx, device = str(device), K = K):
                    out = self.randn((G, K, width), device)
                    many = at._project_o_rows(out, dict(FLAGGED), None)
                    self.assertEqual(tuple(many.shape[:2]), (1, K))
                    ones = [at._project_o_grouped(out[:, j:j + 1].clone().unsqueeze(1), {}, None).clone()
                            for j in range(K)]
                    self.same_rows(f"wo_a/wo_b L{at.layer_idx}", many[0], ones)

    @torch.inference_mode()
    def test_moe(self):
        from exllamav3.modules.dsv41_moe import DSV41MoE
        cached = False
        for device in self.devices:
            # a layer whose experts are in the expert cache where the device has one, else a resident one
            in_cache = lambda blk: getattr(blk.mlp, "tier", None) is not None
            b = self.first(device, in_cache) or self.first(device, lambda blk: True)
            mlp = b.mlp
            self.assertIs(type(mlp), DSV41MoE)
            cached |= in_cache(b)
            for K in ROWS:
                with self.subTest(layer = b.layer_idx, device = str(device), K = K,
                                  experts = "cache" if in_cache(b) else "resident"):
                    x = self.randn((1, K, mlp.hidden_size), device)
                    many = mlp.forward(x, dict(FLAGGED)).clone()
                    self.assertEqual(tuple(many.shape[:2]), (1, K))
                    # a one-row result is a view of a buffer the next call overwrites
                    ones = [mlp.forward(x[:, j:j + 1].clone(), {}).clone() for j in range(K)]
                    self.same_rows(f"moe L{b.layer_idx}", many[0], ones)
        print(f" -- exact rows: MoE tested {'with' if cached else 'WITHOUT'} an expert-cache layer", flush = True)

    @torch.inference_mode()
    def test_hyper_connection_mix(self):
        names = ("post", "comb", "collapsed", "pre")
        H, D = self.model.config.hc_mult, self.model.config.hidden_size
        for device in self.devices:
            b = self.first(device, lambda blk: True)
            for site, hc in (("hc_attn", b.attn_hc), ("hc_ffn", b.mlp_hc)):
                for K in ROWS:
                    for carried, own_pre in ((True, False), (False, False), (True, True)):
                        with self.subTest(site = site, layer = b.layer_idx, device = str(device), K = K,
                                          carried = carried, own_pre = own_pre):
                            streams = self.randn((1, K, H, D), device, torch.float)
                            pre = torch.rand((1, K, H), generator = self.rng).to(device) if carried else None
                            # post and comb are views of workspaces the next call overwrites
                            many = [t.clone() for t in hc.mix_delayed(streams, dict(FLAGGED), pre, own_pre)]
                            ones = [[t.clone() for t in hc.mix_delayed(
                                streams[:, j:j + 1].clone(), {}, None if pre is None else pre[:, j:j + 1].clone(),
                                own_pre)] for j in range(K)]
                            for i, name in enumerate(names):
                                self.same_rows(f"{site}.mix:{name} L{b.layer_idx}", many[i][0],
                                               [one[i] for one in ones])
            head = self.tail["hc_head"]
            for K in ROWS:
                with self.subTest(site = "head collapse", device = str(device), K = K):
                    streams = self.randn((1, K, H, D), device, torch.float)
                    pre = torch.rand((1, K, H), generator = self.rng).to(device)
                    many = head.forward(streams, dict(FLAGGED, dsv41_hc_pre = pre))
                    ones = [head.forward(streams[:, j:j + 1].clone(), {"dsv41_hc_pre": pre[:, j:j + 1].clone()})
                            for j in range(K)]
                    self.same_rows("head collapse", many[0], ones)

    @torch.inference_mode()
    def test_mix_refuses_the_torch_body(self):
        # a flagged call the fused kernel cannot take must not fall back to torch's batched mix
        b = self.blocks[0]
        streams = self.randn((1, 3, self.model.config.hc_mult, self.model.config.hidden_size), "cpu", torch.float)
        with self.assertRaisesRegex(RuntimeError, "EXL3_EXACT_ROWS"):
            b.attn_hc.mix_delayed(streams, dict(FLAGGED), None)

    @torch.inference_mode()
    def test_engram_gate(self):
        # DSV41Engram._gate_rows, which a flagged forward calls on the streams and on the key, a
        # view into the projection's (1, L, (H + 1) * D) output
        engrams = [b.engram for b in self.blocks if b.engram is not None]
        self.assertTrue(engrams, "the model has no engram layer")
        for eng in engrams:
            device = eng.qk.device
            H, D = eng.qk.shape[-2], eng.qk.shape[-1]
            eps = eng.config.rms_norm_eps
            for K in ROWS:
                with self.subTest(engram = eng.key, device = str(device), K = K):
                    h = self.randn((1, K, H, D), device, torch.float)
                    kv = self.randn((1, K, (H + 1) * D), device, torch.float)
                    key = kv[..., :H * D].view(1, K, H, D)
                    many = eng._gate_rows(h, key, eps)
                    self.assertEqual(tuple(many.shape), (1, K, H))
                    ones = []
                    for l in range(K):
                        kv1 = kv[:, l:l + 1].clone()
                        ones.append(eng._gate(h[:, l:l + 1].clone(), kv1[..., :H * D].view(1, 1, H, D), eps))
                    self.same_rows(f"{eng.key} gate", many[0], ones)

    @torch.inference_mode()
    def test_compressor(self):
        from exllamav3.constants import PAGE_SIZE
        from exllamav3.modules.dsv41_cached import CompressCarry
        hd, eps = self.model.config.head_dim, self.model.config.rms_norm_eps
        for device in self.devices:
            weight = (1 + torch.randn(hd, generator = self.rng) * 0.1).half().to(device)
            for K in ROWS:
                for pos0 in (1200, 1201):
                    kv = self.randn((K, hd), device, torch.float, 5.0)
                    gate = self.randn((K, hd), device, torch.float, 3.0)
                    with self.subTest(rate = 1, device = str(device), K = K, pos0 = pos0):
                        many, first = CompressCarry.step(None, kv, None, pos0, 1, weight, eps, True)
                        ones = [CompressCarry.step(None, kv[j:j + 1].clone(), None, pos0 + j, 1, weight, eps)
                                for j in range(K)]
                        self.assertEqual((first, many.shape[0]), (pos0, K))
                        self.assertEqual([f for _, f in ones], list(range(pos0, pos0 + K)))
                        self.same_rows("compressor rate 1", many, [lat for lat, _ in ones])
                    with self.subTest(rate = 2, device = str(device), K = K, pos0 = pos0):
                        # the ring already holds the row before pos0: an odd pos0 closes its group
                        ring = self.randn((PAGE_SIZE + 2, 2 * hd), device, torch.float)
                        ring_many, ring_ones = ring.clone(), ring.clone()
                        many, first = CompressCarry.step(ring_many, kv, gate, pos0, 2, weight, eps, True)
                        ones = [CompressCarry.step(ring_ones, kv[j:j + 1].clone(), gate[j:j + 1].clone(),
                                                   pos0 + j, 2, weight, eps)[0] for j in range(K)]
                        ones = [row for lat in ones for row in lat]
                        self.assertEqual((first, many.shape[0]), (pos0 // 2, (pos0 + K) // 2 - pos0 // 2))
                        self.same_rows("compressor rate 2", many, ones)
                        self.assertTrue(torch.equal(ring_many, ring_ones), "the carry rings differ")

    # -- the row-exact entry points of the extension (test_native_*) --

    def need(self, bits: int):
        """Skip, with a note, unless the extension reports these row-exact entry points."""
        if not self.caps or self.caps & bits != bits:
            note = f"skipped {self.id().rsplit('.', 1)[-1]}: the extension reports the row-exact entry points " \
                   f"{self.caps or 'none'}, the test needs {bits or 'some'}"
            print(f" -- exact rows: {note}", flush = True)
            self.skipTest(note)

    @contextmanager
    def counted(self, name: str):
        """ext.<name> wrapped for the block: a list that grows by one per call."""
        calls, orig = [], getattr(self.ext, name)

        def wrapper(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)
        with mock.patch.object(self.ext, name, wrapper):
            yield calls

    def same(self, what, native, loop):
        self.assertEqual(native.dtype, loop.dtype, what)
        self.assertEqual(tuple(native.shape), tuple(loop.shape), what)
        self.assertTrue(torch.equal(native, loop), f"{what}: the entry point's rows differ from the Python row loop's")

    def linear_both(self, what, lin, x, out_dtype) -> bool:
        """lin on x flagged, with the extension's entry points and with the Python row loop alone.
        Returns whether the entry point served: then the row loop must not have run."""
        m_linear = importlib.import_module("exllamav3.modules.linear")
        with rows_native(self.caps):
            served = lin.rows_native(x, FLAGGED)
            if served:
                with mock.patch.object(m_linear, "forward_rows", _refuse):
                    native = lin.forward(x, dict(FLAGGED), out_dtype).clone()
            else:
                native = lin.forward(x, dict(FLAGGED), out_dtype).clone()
        with rows_native(0):
            loop = lin.forward(x, dict(FLAGGED), out_dtype).clone()
        self.same(what, native, loop)
        return served

    @torch.inference_mode()
    def test_native_equals_python_loop_linears(self):
        self.need(CAP_LINEAR)
        served = {}
        for device in self.devices:
            for name, get, dims, out_dtype in LINEAR_OPS:
                b = self.first(device, lambda blk: get(blk) is not None)
                if b is None:
                    continue
                lin = get(b)
                for K in ROWS:
                    with self.subTest(op = name, layer = b.layer_idx, device = str(device), K = K):
                        shape = (1, K, lin.in_features) if dims == 3 else (K, lin.in_features)
                        took = self.linear_both(f"{name} L{b.layer_idx}", lin, self.randn(shape, device), out_dtype)
                        served.setdefault(str(device), {}).setdefault(name, set()).add(took)
        head = self.tail["head"]
        for K in ROWS:
            with self.subTest(op = "head", device = str(head.device), K = K):
                took = self.linear_both("head", head, self.randn((1, K, head.in_features), head.device), None)
                served.setdefault(str(torch.device(head.device)), {}).setdefault("head", set()).add(took)
        for device, ops in served.items():
            entry = sorted(name for name, took in ops.items() if took == {True})
            loops = sorted(name for name, took in ops.items() if took != {True})
            print(f" -- exact rows: linears on {device}: one native call for the rows of {', '.join(entry) or 'NONE'}; "
                  f"the Python row loop for {', '.join(loops) or 'none'}", flush = True)
            with self.subTest(device = device):
                self.assertTrue(entry, f"no linear on {device} went through run_alloc_rows")

    @classmethod
    def captured_linear_inputs(cls) -> dict:
        """
        {(operation, device): (the Linear, out_dtype, x)}: what the first layer of each device that
        has the operation, and the head, were called with in one cached forward of 8 rows that
        follows a prefill of 1200 random tokens, driven as the generator drives a verify forward
        (the params of tools/dsv41_rowprobe.py, Driver). 1200 is past every change of the attention
        plan, so the forward is one the mode flags.
        """
        if cls.real_inputs is not None:
            return cls.real_inputs
        from exllamav3.constants import PAGE_SIZE
        model, cache = cls.model, cls.cache
        P, K = 1200, 8
        ids = torch.randint(0, model.config.vocab_size, (1, P + K), generator = cls.rng)
        state = cache.get_new_state()
        pages = cache.max_num_tokens // PAGE_SIZE // cache.num_slots
        block_table = torch.arange(state.slot * pages, (state.slot + 1) * pages, dtype = torch.int32)[None, :]
        targets = {}
        for device in cls.devices:
            for name, get, _, _ in LINEAR_OPS:
                b = next((b for b in cls.blocks if torch.device(b.device) == device and get(b) is not None), None)
                if b is not None:
                    targets[(name, str(device))] = get(b)
        head = cls.tail["head"]
        targets[("head", str(torch.device(head.device)))] = head
        captured, hooked = {}, []
        woa, hooked_attn = {}, []

        def hook_woa(key, at):
            orig = at._project_o_rows

            def project(out, params, out_dtype):
                # dsa_attn's group-major (groups, rows, width) output of the forward's rows
                if key not in woa and 2 <= out.shape[1] <= 8:
                    woa[key] = (at, out.clone())
                return orig(out, params, out_dtype)
            at._project_o_rows = project
            hooked_attn.append(at)

        def hook(key, lin):
            orig = lin.forward

            def forward(x, params, out_dtype = None):
                # the call of the forward's rows, with the out_dtype the engine passes; the nested
                # one-row calls of the Python row loop are not captured
                if key not in captured and 2 <= x.numel() // x.shape[-1] <= 8:
                    captured[key] = (lin, out_dtype, x.clone())
                return orig(x, params, out_dtype)
            lin.forward = forward
            hooked.append(lin)
        try:
            for key, lin in targets.items():
                hook(key, lin)
            for device in cls.devices:
                hook_woa(str(device), next(b for b in cls.blocks if torch.device(b.device) == device).attn)
            model.prefill(ids[:, :P], {
                "attn_mode": "flash_attn", "block_table": block_table, "cache": cache,
                "cache_seqlens": torch.tensor([0], dtype = torch.int32),
                "recurrent_states": [state], "indexed_embeddings": []})
            captured.clear()        # the prefill's own calls of 2 to 8 entries, if any
            woa.clear()
            model.forward(ids[:, P:], {
                "attn_mode": "flash_attn", "block_table": block_table, "cache": cache,
                "cache_seqlens": torch.tensor([state.position], dtype = torch.int32),
                "recurrent_states": [state], "indexed_embeddings": [], "positions": None,
                "recurrent_history": True, "pinned_staging": True})
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
        finally:
            for lin in hooked:
                del lin.forward
            for at in hooked_attn:
                del at._project_o_rows
            cache.release_state(state)
        cls.real_inputs = captured
        cls.real_woa = woa
        return captured

    @torch.inference_mode()
    def test_native_equals_python_loop_linears_real_activations(self):
        self.need(CAP_LINEAR)
        captured = self.captured_linear_inputs()
        self.assertTrue(captured, "the forward reached no Linear with 2 to 8 rows")
        print(f" -- exact rows: real activations of {len(captured)} linears: "
              f"{', '.join(sorted(f'{name} @ {device}' for name, device in captured))}", flush = True)
        loops = []
        for (name, device), (lin, out_dtype, x) in sorted(captured.items(), key = lambda kv: kv[0]):
            with self.subTest(op = name, device = device, rows = x.numel() // x.shape[-1]):
                if not self.linear_both(f"{name} @ {device}, real activations", lin, x, out_dtype):
                    loops.append(f"{name} @ {device} (contiguous: {x.is_contiguous()})")
                # and against the one-row calls themselves, as test_linears does on random rows
                self.linear_rows(f"{name} @ {device}, real activations", lin, x, out_dtype)
        print(f" -- exact rows: of those, the Python row loop serves {', '.join(loops) or 'none'}", flush = True)

    @torch.inference_mode()
    def test_native_equals_python_loop_grouped_output_projection(self):
        self.need(CAP_MGEMM)
        for device in self.devices:
            at = self.first(device, lambda blk: True).attn
            G, width = at.o_groups, at.num_q_heads // at.o_groups * at.head_dim
            if not at.woa_multi_ready and at.device is not None:
                at._build_woa_multi()
            self.assertIsNotNone(at.wo_a_multi, f"L{at.layer_idx}: the grouped GEMM does not serve wo_a")
            for K in ROWS:
                with self.subTest(layer = at.layer_idx, device = str(device), K = K):
                    out = self.randn((G, K, width), device)
                    with rows_native(self.caps), self.counted("exl3_mgemm_rows") as calls, \
                            mock.patch.object(at, "_project_o_grouped", _refuse):
                        native = at._project_o_rows(out, dict(FLAGGED), None).clone()
                    self.assertEqual(len(calls), 1, "exl3_mgemm_rows calls")
                    with rows_native(0):
                        loop = at._project_o_rows(out, dict(FLAGGED), None).clone()
                    self.same(f"wo_a/wo_b L{at.layer_idx}", native, loop)

    # -- one launch for the rows (test_one_launch_*) --

    def launches(self, call):
        """call(), and how many calls of the row-exact entry points inside it were ONE launch for
        their rows (exact_rows_one_launches)."""
        before = int(self.ext.exact_rows_one_launches())
        result = call()
        return result, int(self.ext.exact_rows_one_launches()) - before

    def one_launch_device(self, device) -> bool:
        """Whether the extension makes the single launch on this GPU (ONE_LAUNCH_SM)."""
        return tuple(torch.cuda.get_device_capability(torch.device(device))) in ONE_LAUNCH_SM

    def one_launch_rows(self, what, lin, x, out_dtype, expect: int):
        """lin on x flagged, under the current EXL3_INT8_GEMV, against one call per cloned row of
        that setting; the flagged call must be `expect` one-launch calls (1: one launch for the
        rows, 0: a launch per row). The one-row calls run first, so the launch record of the
        one-row call is in the process when the flagged call looks for it."""
        K = x.numel() // x.shape[-1]
        ones = []
        for j in range(K):
            row = x.reshape(K, -1)[j : j + 1].clone().view(*([1] * (x.dim() - 1)), x.shape[-1])
            ones.append(lin.forward(row, {}, out_dtype).clone())
        flagged = lambda: lin.forward(x, dict(FLAGGED), out_dtype).clone()
        many, grew = self.launches(flagged)
        if expect and not grew:
            # tolerated once: a first flagged call that found no record makes the one-row launches,
            # which leave the record; the next call must find it
            print(f" -- exact rows: {what}: the first flagged call was not one launch (no launch record of the "
                  f"one-row call in the process?), repeated", flush = True)
            self.same_rows(f"{what} (launch per row)", many.reshape(K, -1), ones)
            many, grew = self.launches(flagged)
        self.assertEqual(grew, expect, f"{what}: one-launch calls of the flagged call")
        self.same_rows(what, many.reshape(K, -1), ones)

    @torch.inference_mode()
    def test_one_launch_linears(self):
        self.need(CAP_LINEAR | CAP_ONE_LAUNCH)
        captured = self.captured_linear_inputs()
        targets = []
        for device in self.devices:
            for name, get, dims, out_dtype in LINEAR_OPS:
                b = self.first(device, lambda blk: get(blk) is not None)
                if b is not None:
                    targets.append((name, str(device), get(b), dims, out_dtype))
        head = self.tail["head"]
        targets.append(("head", str(torch.device(head.device)), head, 3, None))
        one, per_row, python, real, biased = {}, {}, {}, set(), set()
        for name, device, lin, dims, out_dtype in targets:
            for K in ROWS:
                shape = (1, K, lin.in_features) if dims == 3 else (K, lin.in_features)
                inputs = [("random rows", self.randn(shape, device), out_dtype)]
                if (name, device) in captured:
                    # the first K rows the forward passed (idx.wk gets pool entries, not the forward's rows)
                    _, real_dtype, xr = captured[(name, device)]
                    if K <= xr.numel() // xr.shape[-1]:
                        rows = xr.reshape(-1, xr.shape[-1])[:K].clone()
                        inputs.append(("real activations", rows.view(*xr.shape[:-2], K, xr.shape[-1]), real_dtype))
                        real.add(f"{name} @ {device}")
                for kind, x, dt in inputs:
                    what = f"{name} @ {device}, {kind}, K={K}"
                    with self.subTest(op = name, device = device, K = K, input = kind):
                        with rows_native(self.caps):
                            if not lin.rows_native(x, FLAGGED):
                                python.setdefault(device, set()).add(name)    # the Python row loop serves it
                                continue
                            inner = lin.inner
                            fp32 = (dt or inner.default_out_dtype) == torch.float
                            with int8_gemv("0"):
                                tag = self.one_row_tag(inner, x.device, fp32)
                                self.assertNotEqual(tag, 0, f"{what}: the int8 GEMV ran with EXL3_INT8_GEMV=0")
                                # The FP16 GEMV (90) has another instance for one row, a width that
                                # is not a multiple of 128 would put a transform block across two
                                # rows, and a GPU outside ONE_LAUNCH_SM has no bitwise run of the
                                # launch: a launch per row for each
                                aligned = inner.in_features % 128 == 0 and inner.out_features % 128 == 0
                                expect = 1 if 1 <= tag <= 4 and aligned and self.one_launch_device(device) else 0
                                self.one_launch_rows(f"{what}, int8 off", lin, x, dt, expect)
                            (one if expect else per_row).setdefault(device, set()).add(name)
                            if inner.bias is not None:
                                biased.add(f"{name} @ {device}")
                            # with the int8 path on, no flagged call is one launch, whatever the codebook
                            with int8_gemv(self.int8_on):
                                self.one_launch_rows(f"{what}, int8 on", lin, x, dt, 0)
        for device in sorted({d for _, d, _, _, _ in targets}):
            names = lambda group: ", ".join(sorted(group.get(device, ()))) or "none"
            print(f" -- exact rows: linears on {device} with EXL3_INT8_GEMV=0: one launch for the rows of "
                  f"{names(one)}; a launch per row for {names(per_row)}; not through run_alloc_rows: "
                  f"{names(python)}; with the int8 path on: a launch per row for all", flush = True)
            with self.subTest(device = device):
                if self.one_launch_device(device):
                    self.assertTrue(one.get(device), f"no linear on {device} was one launch for its rows")
                else:
                    print(f" -- exact rows: {device} has compute capability "
                          f"{torch.cuda.get_device_capability(torch.device(device))}: the extension makes no single "
                          f"launch there", flush = True)
        # run_alloc_rows adds the bias row by row after the single launch: which tested linears have one
        print(f" -- exact rows: tested linears with a bias: "
              f"{', '.join(sorted(biased)) or 'none (the bias add after the single launch did not run)'}", flush = True)
        print(f" -- exact rows: one-launch linears also on the real activations of {', '.join(sorted(real)) or 'NONE'}",
              flush = True)
        self.assertTrue(real, "no linear was tested on real activations")

    @torch.inference_mode()
    def test_one_launch_first_call_without_record(self):
        # The single launch never makes the launch record it runs under: a flagged call that finds
        # none in the process launches per row (which loads or tunes the record), and the next one
        # is one launch. The bucket must be one no other call of this process touches: the FP32
        # output of attn.wq_a, which the engine reads as FP16. Sorts ahead of the other
        # test_one_launch_* tests; one-row calls of that bucket made earlier would fail it
        self.need(CAP_LINEAR | CAP_ONE_LAUNCH)
        K = 4
        for device in self.devices:
            b = self.first(device, lambda blk: type(getattr(blk.attn.q_a, "inner", None)).__name__ == "LinearEXL3")
            with self.subTest(device = str(device)):
                self.assertIsNotNone(b, f"no layer on {device} has an EXL3 attn.wq_a")
                lin = b.attn.q_a
                x = self.randn((1, K, lin.in_features), device)
                flagged = lambda: lin.forward(x, dict(FLAGGED), torch.float).clone()
                with rows_native(self.caps), int8_gemv("0"):
                    self.assertTrue(lin.rows_native(x, FLAGGED), "attn.wq_a does not go through run_alloc_rows")
                    first, grew = self.launches(flagged)
                    self.assertEqual(grew, 0, f"attn.wq_a FP32 @ {device}: the first flagged call was one launch: a "
                                              f"launch record of the one-row call was already in the process")
                    ones = [lin.forward(x[:, j:j + 1].clone(), {}, torch.float).clone() for j in range(K)]
                    self.same_rows(f"attn.wq_a FP32 @ {device}, first flagged call", first.reshape(K, -1), ones)
                    tag = self.one_row_tag(lin.inner, device, True)
                    aligned = lin.inner.in_features % 128 == 0 and lin.inner.out_features % 128 == 0
                    expect = 1 if 1 <= tag <= 4 and aligned and self.one_launch_device(device) else 0
                    second, grew = self.launches(flagged)
                    self.assertEqual(grew, expect, f"attn.wq_a FP32 @ {device}: one-launch calls of the second flagged call")
                    self.same_rows(f"attn.wq_a FP32 @ {device}, second flagged call", second.reshape(K, -1), ones)
                print(f" -- exact rows: attn.wq_a FP32 on {device} (L{b.layer_idx}): first flagged call a launch per row "
                      f"(no launch record yet), second {'one launch' if expect else 'a launch per row'}", flush = True)

    @torch.inference_mode()
    def test_one_launch_grouped_output_projection(self):
        # exl3_mgemm_rows against the one-row grouped call, as _project_o_grouped makes it, under
        # both settings of EXL3_INT8_GEMV: one launch for the rows with the int8 path off, a launch
        # per row with it on (the grouped call itself never takes the int8 path; the variable
        # decides so that the default setting launches as it did before the single launch)
        self.need(CAP_MGEMM | CAP_ONE_LAUNCH)
        self.captured_linear_inputs()
        ext = self.ext
        for device in self.devices:
            at = self.first(device, lambda blk: True).attn
            G, width = at.o_groups, at.num_q_heads // at.o_groups * at.head_dim
            if not at.woa_multi_ready and at.device is not None:
                at._build_woa_multi()
            mu = at.wo_a_multi
            self.assertIsNotNone(mu, f"L{at.layer_idx}: the grouped GEMM does not serve wo_a")
            self.assertEqual(width, mu.in_features)
            n = mu.out_features
            ah = torch.empty((G, 1, width), dtype = torch.half, device = device)
            real = self.real_woa.get(str(device))
            self.assertIsNotNone(real, f"the forward's grouped projection was not captured on {device}")
            self.assertIs(real[0], at)
            for setting in ("0", self.int8_on):
                for K in ROWS:
                    inputs = [("random rows", self.randn((G, K, width), device))]
                    if K <= real[1].shape[1]:
                        inputs.append(("real activations", real[1][:, :K].contiguous()))
                    for kind, out in inputs:
                        what = f"wo_a L{at.layer_idx} @ {device}, {kind}, K={K}, EXL3_INT8_GEMV={setting}"
                        with self.subTest(layer = at.layer_idx, device = str(device), K = K, input = kind,
                                          int8 = setting), int8_gemv(setting):
                            ones = []
                            for j in range(K):
                                C1 = torch.empty((G, 1, n), dtype = torch.half, device = device)
                                tag = ext.exl3_mgemm(
                                    out[:, j:j + 1].contiguous(), mu.ptrs_trellis, C1, mu.ptrs_suh, ah, mu.ptrs_svh,
                                    at.woa_indices, None, mu.K, -1, mu.mcg, mu.mul1, -1, -1, 0, 1, None, None)
                                self.assertIn(int(tag), range(1, 5), what)
                                ones.append(C1.reshape(-1).clone())
                            A = out.transpose(0, 1).contiguous()
                            C = torch.empty((K, G, n), dtype = torch.half, device = device)
                            rows = lambda: ext.exl3_mgemm_rows(A, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh,
                                                               at.woa_indices, mu.K, mu.mcg, mu.mul1)
                            expect = 1 if setting == "0" and width % 128 == 0 and n % 128 == 0 \
                                and self.one_launch_device(device) else 0
                            _, grew = self.launches(rows)
                            if expect and not grew:
                                print(f" -- exact rows: {what}: the first call was not one launch, repeated", flush = True)
                                self.same_rows(f"{what} (launch per row)", C.reshape(K, -1).clone(), ones)
                                _, grew = self.launches(rows)
                            self.assertEqual(grew, expect, f"{what}: one-launch calls of exl3_mgemm_rows")
                            self.same_rows(what, C.reshape(K, -1), ones)

    def moe_layers(self, device):
        """A layer whose experts are in the expert cache and a resident one, where the device has them."""
        in_cache = lambda blk: getattr(blk.mlp, "tier", None) is not None
        found = (self.first(device, in_cache), self.first(device, lambda blk: not in_cache(blk)))
        return [(b, "cache" if in_cache(b) else "resident") for b in found if b is not None]

    @torch.inference_mode()
    def test_native_equals_python_loop_moe(self):
        self.need(CAP_ROUTER | CAP_MOE)
        m_moe = importlib.import_module("exllamav3.modules.dsv41_moe")
        for device in self.devices:
            # every GPU must show the entry points on a layer of its own, at every K: a layer they
            # do not serve compares the Python row loop with itself
            one_call = []
            for b, experts in self.moe_layers(device):
                mlp = b.mlp
                served = []
                for K in ROWS:
                    with self.subTest(layer = b.layer_idx, device = str(device), K = K, experts = experts):
                        x = self.randn((1, K, mlp.hidden_size), device)
                        with rows_native(self.caps):
                            one = bool(mlp.rows_native(x, FLAGGED))
                            served.append(one)
                            if one:
                                with self.counted("routing_ds3_nogroup_rows") as calls, \
                                        mock.patch.object(m_moe, "forward_rows", _refuse):
                                    native = mlp.forward(x, dict(FLAGGED)).clone()
                                self.assertEqual(len(calls), 1, "routing_ds3_nogroup_rows calls")
                            else:
                                native = mlp.forward(x, dict(FLAGGED)).clone()
                        with rows_native(0):
                            loop = mlp.forward(x, dict(FLAGGED)).clone()
                        self.same(f"moe L{b.layer_idx}", native, loop)
                every_k = len(served) == len(ROWS) and all(served)
                print(f" -- exact rows: MoE L{b.layer_idx} on {device} ({experts} experts): bc.sh_coop "
                      f"{getattr(mlp.bc, 'sh_coop', None)}, one forward for the rows at K = "
                      f"{', '.join(str(K) for K, one in zip(ROWS, served) if one) or 'NO K'}", flush = True)
                one_call.append(every_k)
            with self.subTest(device = str(device)):
                self.assertTrue(any(one_call), f"no tested MoE layer on {device} is served by the row-exact entry "
                                               f"points at every K")

    def moe_launch(self, mlp, y, sel, w, rows_entry: bool):
        """The fused decode launch pair of an MoE layer on a given routing, as BlockSparseMLP.forward
        makes it (the expert cache lookup of the call first, where the layer has one). Returns a
        copy of the routed sum, (rows, hidden)."""
        cached = getattr(mlp, "tier", None) is not None
        if cached:
            mlp.tier_resolve(sel, y.shape[0], {})
        if rows_entry:
            mlp.bc.run_bszN_rows(y, sel, w)
        else:
            mlp.bc.run_bszN(y, sel, w)
        out = mlp.experts_cfg.out_bszn[:y.shape[0]].clone()
        if cached:
            mlp.tier_after_compute({})
        return out

    @torch.inference_mode()
    def test_native_moe_selections(self):
        # The gate of the single launch pair: run_bszN_rows on K rows against K one-row run_bszN
        # calls on the same routing. From 6 rows on the stock launch of the sm_120 card switches
        # the tile, and rows that picked one expert are grouped at every K: neither may show here.
        # Every GPU must contribute a layer: a pass says nothing about a card that was skipped
        self.need(CAP_MOE)
        tested = []
        for device in self.devices:
            cc = self.ext.g_get_cc(device.index if device.index is not None else torch.cuda.current_device())
            on_device = len(tested)
            for b, experts in self.moe_layers(device):
                mlp = b.mlp
                topk, cfg = mlp.num_experts_per_tok, mlp.routing_cfg
                if mlp.bc is None or not mlp.bc.rows_exact_ok(topk):
                    print(f" -- exact rows: MoE L{b.layer_idx} on {device}: run_bszN_rows does not serve the layer "
                          f"(bc.sh_coop {getattr(mlp.bc, 'sh_coop', None)})", flush = True)
                    continue
                tested.append(f"L{b.layer_idx} on {device} (cc {cc}, {experts} experts, bc.sh_coop {mlp.bc.sh_coop})")
                for K in ROWS:
                    y = self.randn((K, mlp.hidden_size), device)
                    # any valid routing serves: the unflagged router's, on these rows
                    sel, w = (t.clone() for t in mlp.routing_fn(K, cfg, y, {}))
                    one_zero, empty_row = w.clone(), w.clone()
                    one_zero[0, 0] = 0
                    empty_row[K - 1] = 0
                    cases = (
                        ("the router's selection", sel, w),
                        ("every row the same experts", sel[:1].expand(K, topk).contiguous(), w),
                        ("a zero weight", sel, one_zero),
                        ("a row of zero weights", sel, empty_row),
                    )
                    for case, sel_c, w_c in cases:
                        with self.subTest(layer = b.layer_idx, device = str(device), K = K, routing = case):
                            many = self.moe_launch(mlp, y, sel_c, w_c, True)
                            ones = [self.moe_launch(mlp, y[j:j + 1].clone(), sel_c[j:j + 1].clone(),
                                                    w_c[j:j + 1].clone(), False) for j in range(K)]
                            self.same_rows(f"moe experts L{b.layer_idx}, {case}", many, ones)
            with self.subTest(device = str(device)):
                self.assertGreater(len(tested), on_device, f"run_bszN_rows serves no tested MoE layer on {device}")
        print(f" -- exact rows: run_bszN_rows against one-row run_bszN, K = 2..8, on {'; '.join(tested) or 'NOTHING'}",
              flush = True)

    @torch.inference_mode()
    def test_native_router(self):
        self.need(CAP_ROUTER)
        from exllamav3.modules.block_sparse_mlp_routing import _gate_t, _esb_h, ROUTING_ACT_SQRTSP
        for device in self.devices:
            b = self.first(device, lambda blk: True)
            cfg = b.mlp.routing_cfg
            _gate_t(cfg)
            E, topk = cfg.num_experts, cfg.num_experts_per_tok
            route = lambda fn, z, logits, sel, w: fn(
                z, cfg.gate_tensor, logits, _esb_h(cfg), sel, w, cfg.routed_scaling_factor, cfg.gate_tensor_t,
                ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb)
            for K in ROWS:
                with self.subTest(layer = b.layer_idx, device = str(device), K = K):
                    z = self.randn((K, b.mlp.hidden_size), device)
                    logits = torch.empty((K, E), dtype = torch.half, device = device)
                    sel = torch.empty((K, topk), dtype = torch.long, device = device)
                    w = torch.empty((K, topk), dtype = torch.half, device = device)
                    route(self.ext.routing_ds3_nogroup_rows, z, logits, sel, w)
                    for j in range(K):
                        # into the bsz1 buffers, as a decode step routes
                        route(self.ext.routing_ds3_nogroup, z[j:j + 1].clone(), cfg.router_logits_bsz1,
                              cfg.selected_experts_bsz1, cfg.routing_weights_bsz1)
                        for name, many, one in (("logits", logits, cfg.router_logits_bsz1),
                                                ("experts", sel, cfg.selected_experts_bsz1),
                                                ("weights", w, cfg.routing_weights_bsz1)):
                            self.assertTrue(torch.equal(many[j], one[0]), f"router L{b.layer_idx} {name}: row {j} differs")

    @torch.inference_mode()
    def test_native_equals_python_loop_hyper_connection(self):
        self.need(CAP_HC)
        names = ("post", "comb", "collapsed", "pre")
        H, D = self.model.config.hc_mult, self.model.config.hidden_size
        for device in self.devices:
            b = self.first(device, lambda blk: True)
            for site, hc in (("hc_attn", b.attn_hc), ("hc_ffn", b.mlp_hc)):
                for K in ROWS:
                    for carried, own_pre in ((True, False), (False, False), (True, True)):
                        with self.subTest(site = site, layer = b.layer_idx, device = str(device), K = K,
                                          carried = carried, own_pre = own_pre):
                            streams = self.randn((1, K, H, D), device, torch.float)
                            pre = torch.rand((1, K, H), generator = self.rng).to(device) if carried else None
                            with rows_native(self.caps), self.counted("hc_partials_rows") as partials, \
                                    self.counted("hc_collapse_rows") as collapses:
                                native = [t.clone() for t in hc.mix_delayed(streams, dict(FLAGGED), pre, own_pre)]
                            # the model's first sublayer collapses by selecting stream 0
                            self.assertEqual((len(partials), len(collapses)), (1, 1 if carried or own_pre else 0),
                                             "hc_partials_rows, hc_collapse_rows calls")
                            with rows_native(0):
                                loop = [t.clone() for t in hc.mix_delayed(streams, dict(FLAGGED), pre, own_pre)]
                            for i, name in enumerate(names):
                                self.same(f"{site}.mix:{name} L{b.layer_idx}", native[i], loop[i])
            head = self.tail["hc_head"]
            for K in ROWS:
                with self.subTest(site = "head collapse", device = str(device), K = K):
                    streams = self.randn((1, K, H, D), device, torch.float)
                    pre = torch.rand((1, K, H), generator = self.rng).to(device)
                    with rows_native(self.caps), self.counted("hc_collapse_rows") as collapses:
                        native = head.forward(streams, dict(FLAGGED, dsv41_hc_pre = pre)).clone()
                    self.assertEqual(len(collapses), 1, "hc_collapse_rows calls")
                    with rows_native(0):
                        loop = head.forward(streams, dict(FLAGGED, dsv41_hc_pre = pre)).clone()
                    self.same("head collapse", native, loop)

    def refused(self, what, call):
        with self.subTest(refusal = what), self.assertRaises(RuntimeError):
            call()

    @torch.inference_mode()
    def test_native_refusals(self):
        # Each entry point takes 2 to 8 rows of contiguous operands with matching row counts, and
        # refuses everything else before it launches anything
        self.need(0)
        ext, device = self.ext, self.devices[0]
        b = self.first(device, lambda blk: True)
        half = lambda *shape: self.randn(shape, device)
        if self.caps & CAP_LINEAR:
            lin = b.attn.q_a
            n, bc = lin.in_features, lin.inner.bc
            run = lambda x: bc.run_alloc_rows(x, lin.out_features, False)
            self.refused("run_alloc_rows: 1 row", lambda: run(half(1, 1, n)))
            self.refused("run_alloc_rows: 9 rows", lambda: run(half(1, 9, n)))
            self.refused("run_alloc_rows: 1 dimension", lambda: run(half(n)))
            self.refused("run_alloc_rows: non-contiguous", lambda: run(half(1, 4, 2 * n)[..., ::2]))
            self.refused("run_alloc_rows: CPU", lambda: run(half(1, 4, n).cpu()))
            self.refused("run_alloc_rows: FP32", lambda: run(half(1, 4, n).float()))
            self.refused("run_alloc_rows: width", lambda: run(half(1, 4, n + 8)))
        if self.caps & CAP_MGEMM:
            at = b.attn
            if not at.woa_multi_ready and at.device is not None:
                at._build_woa_multi()
            mu, G = at.wo_a_multi, at.o_groups
            k, n = mu.in_features, mu.out_features
            ah = torch.empty((G, 1, k), dtype = torch.half, device = device)
            run = lambda A, C: ext.exl3_mgemm_rows(A, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh, at.woa_indices,
                                                   mu.K, mu.mcg, mu.mul1)
            self.refused("exl3_mgemm_rows: 1 row", lambda: run(half(1, G, k), half(1, G, n)))
            self.refused("exl3_mgemm_rows: 9 rows", lambda: run(half(9, G, k), half(9, G, n)))
            self.refused("exl3_mgemm_rows: A and C rows", lambda: run(half(3, G, k), half(2, G, n)))
            self.refused("exl3_mgemm_rows: A and C groups", lambda: run(half(3, G, k), half(3, G + 1, n)))
            self.refused("exl3_mgemm_rows: non-contiguous", lambda: run(half(3, G, 2 * k)[..., ::2], half(3, G, n)))
            self.refused("exl3_mgemm_rows: 2 dimensions", lambda: run(half(3, G * k), half(3, G, n)))
            tables = lambda B, indices: ext.exl3_mgemm_rows(half(3, G, k), B, half(3, G, n), mu.ptrs_suh, ah, mu.ptrs_svh,
                                                            indices, mu.K, mu.mcg, mu.mul1)
            self.refused("exl3_mgemm_rows: int32 indices", lambda: tables(mu.ptrs_trellis, at.woa_indices.int()))
            self.refused("exl3_mgemm_rows: no indices", lambda: tables(mu.ptrs_trellis, at.woa_indices[:0]))
            self.refused("exl3_mgemm_rows: CPU indices", lambda: tables(mu.ptrs_trellis, at.woa_indices.cpu()))
            self.refused("exl3_mgemm_rows: CPU pointer table", lambda: tables(mu.ptrs_trellis.cpu(), at.woa_indices))
            self.refused("exl3_mgemm_rows: short pointer table", lambda: tables(mu.ptrs_trellis[:G - 1], at.woa_indices))
        mlp = b.mlp
        cfg = mlp.routing_cfg
        E, topk, hidden = cfg.num_experts, cfg.num_experts_per_tok, mlp.hidden_size
        sel = lambda rows: torch.zeros((rows, topk), dtype = torch.long, device = device)
        if self.caps & CAP_ROUTER:
            from exllamav3.modules.block_sparse_mlp_routing import _gate_t, _esb_h, ROUTING_ACT_SQRTSP
            _gate_t(cfg)
            run = lambda z, rows: ext.routing_ds3_nogroup_rows(
                z, cfg.gate_tensor, half(rows, E), _esb_h(cfg), sel(rows), half(rows, topk), cfg.routed_scaling_factor,
                cfg.gate_tensor_t, ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb)
            self.refused("routing_ds3_nogroup_rows: 1 row", lambda: run(half(1, hidden), 1))
            self.refused("routing_ds3_nogroup_rows: 9 rows", lambda: run(half(9, hidden), 9))
            self.refused("routing_ds3_nogroup_rows: hidden and scores rows", lambda: run(half(3, hidden), 4))
            self.refused("routing_ds3_nogroup_rows: non-contiguous", lambda: run(half(3, 2 * hidden)[..., ::2], 3))
            topk_out = lambda sel_t, w_t: ext.routing_ds3_nogroup_rows(
                half(3, hidden), cfg.gate_tensor, half(3, E), _esb_h(cfg), sel_t, w_t, cfg.routed_scaling_factor,
                cfg.gate_tensor_t, ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb)
            self.refused("routing_ds3_nogroup_rows: int32 topk_indices", lambda: topk_out(sel(3).int(), half(3, topk)))
            self.refused("routing_ds3_nogroup_rows: CPU topk_weights", lambda: topk_out(sel(3), half(3, topk).cpu()))
            self.refused("routing_ds3_nogroup_rows: topk shapes", lambda: topk_out(sel(3), half(3, topk + 1)))
        if self.caps & CAP_MOE and mlp.bc is not None:
            run = lambda y, rows: mlp.bc.run_bszN_rows(y, sel(rows), half(rows, topk))
            self.refused("run_bszN_rows: 1 row", lambda: run(half(1, hidden), 1))
            self.refused("run_bszN_rows: 9 rows", lambda: run(half(9, hidden), 9))
            self.refused("run_bszN_rows: y and selection rows", lambda: run(half(3, hidden), 4))
            self.refused("run_bszN_rows: non-contiguous", lambda: run(half(3, 2 * hidden)[..., ::2], 3))
            self.refused("run_bszN_rows: weights shape", lambda: mlp.bc.run_bszN_rows(half(3, hidden), sel(3), half(3, topk + 1)))
            # more picks than any scratch holds (256 slots): refused ahead of the shared expert's launch pair
            wide = torch.zeros((3, 128), dtype = torch.long, device = device)
            self.refused("run_bszN_rows: picks exceed the scratch", lambda: mlp.bc.run_bszN_rows(half(3, hidden), wide, half(3, 128)))
        if self.caps & CAP_HC:
            f32 = lambda *shape: self.randn(shape, device, torch.float)
            self.refused("hc_collapse_rows: 1 row", lambda: ext.hc_collapse_rows(f32(1, 1, 4), f32(1, 1, 4, 64)))
            self.refused("hc_collapse_rows: 9 rows", lambda: ext.hc_collapse_rows(f32(1, 9, 4), f32(1, 9, 4, 64)))
            self.refused("hc_collapse_rows: 2 sequences", lambda: ext.hc_collapse_rows(f32(2, 3, 4), f32(2, 3, 4, 64)))
            self.refused("hc_collapse_rows: pre and streams rows", lambda: ext.hc_collapse_rows(f32(1, 3, 4), f32(1, 4, 4, 64)))
            self.refused("hc_collapse_rows: FP16", lambda: ext.hc_collapse_rows(half(1, 3, 4), half(1, 3, 4, 64)))
            self.refused("hc_partials_rows: 1 row", lambda: ext.hc_partials_rows(f32(1, 5, 25)))
            self.refused("hc_partials_rows: 9 rows", lambda: ext.hc_partials_rows(f32(9, 5, 25)))
            self.refused("hc_partials_rows: 2 dimensions", lambda: ext.hc_partials_rows(f32(3, 25)))
            self.refused("hc_partials_rows: no chunks", lambda: ext.hc_partials_rows(f32(3, 0, 25)))
            self.refused("hc_collapse_rows: empty", lambda: ext.hc_collapse_rows(f32(1, 3, 4), f32(1, 3, 4, 0)))

    @torch.inference_mode()
    def test_router_refuses_flagged_rows_without_the_entry_point(self):
        # A flagged call of several rows must never reach the multi-row projection: without the
        # row-exact router the MoE layer routes one row per call, and the router refuses the rest.
        # Runs with every extension
        from exllamav3.modules.block_sparse_mlp_routing import routing_sqrtsp
        mlp = self.blocks[0].mlp
        z = self.randn((3, mlp.hidden_size), mlp.device)
        with rows_native(0), self.assertRaisesRegex(RuntimeError, "EXL3_EXACT_ROWS"):
            routing_sqrtsp(3, mlp.routing_cfg, z, dict(FLAGGED))


if __name__ == "__main__":
    unittest.main()
