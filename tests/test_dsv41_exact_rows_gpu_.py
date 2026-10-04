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

The index selection is covered on the pools of real forwards (prefixes of 1200 and 19000 tokens:
the second is past the 16,384 entries from which the candidate mask cuts) and on crafted pools:
select_topk with one_row on K rows against K one-row calls, indices, candidate blocks and the
scores themselves, on every GPU (the captured operands are copied to a GPU that holds no index
source), with the calls that must refuse the shared pass: among them every call on a GPU that is
not of a type the pass is verified on (ROWS_PASS_SM; shown with the compute capability replaced,
and expected of a real GPU of another type). Not covered: the attention kernel and
the order of these operations in a forward. tools/dsv41_rowprobe.py under EXL3_EXACT_ROWS=1 is
the evidence for those (attn.dsa_attn and the forward's logits in its tables).

Where a flagged call takes a batched form, a test also shows that the form ran, since both forms
give the same bits: the scorer and top-k calls of a selection, hgemm_rows calls, the first-use
record of the hyper-connection sums (dsv41_block.ROWS_SUMMED, with the per-row forms made to
raise), and the extension's counts of grouped MoE launch pairs and of router projections that
were one launch (exact_rows_served).

A flagged call takes the extension's row-exact entry points where it has them (exact_rows_caps:
the rows of an EXL3 linear, of an FP16 linear through the native GEMM, of the grouped output
projection and of the router in one native call, the MoE experts of all rows in one launch pair,
with the rows grouped by expert and the router's projection as one launch where the extension
reports those forms), else the Python loops of one call per row. The index selection, the
hyper-connection sums, the compressor and the engram gate take their batched forms with every
extension (math_policy.exact_rows_forms; the consumers' ROWS_FORMS); each is also compared in
place with its form of one call per row, entry or token, by clearing its bit (rows_forms). The
tests above cover whichever the extension gives. The test_native_* tests
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
# hgemm_rows, and the two forms the callers ask of run_bszN_rows and routing_ds3_nogroup_rows by an
# argument: the rows grouped by expert, the router's projection as one launch
CAP_HGEMM, CAP_MOE_GROUPED, CAP_ROUTER_ONE_LAUNCH = 64, 128, 256
# The compute capabilities on which the extension makes that launch (exl3_gemm_one_row_route_ok),
# groups the MoE rows, and on which select_topk shares a pass (dsv41_select.ROWS_PASS_SM)
ONE_LAUNCH_SM = ((8, 0), (8, 9), (12, 0))
# The batched forms that are Python alone (math_policy.EXACT_ROWS_FORM_*), and the modules that
# hold them (ROWS_FORMS)
FORM_SELECT, FORM_SUMS, FORM_COMPRESS, FORM_GATE = 1, 2, 4, 8
FORM_CONSUMERS = ("dsv41", "dsv41_block", "dsv41_cached", "dsv41_engram")
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


@contextmanager
def rows_forms(mask: int):
    """The consumers' ROWS_FORMS set to mask for the block: a cleared bit gives that operation its
    form of one call per row, entry or token."""
    mods = [importlib.import_module(f"exllamav3.modules.{name}") for name in FORM_CONSUMERS]
    saved = [m.ROWS_FORMS for m in mods]
    for m in mods:
        m.ROWS_FORMS = mask
    try:
        yield
    finally:
        for m, value in zip(mods, saved):
            m.ROWS_FORMS = value


def _refuse(*args, **kwargs):
    raise AssertionError("the Python row loop ran where the row-exact entry point must serve")


class _RefusedKernel:
    """In place of a Triton kernel that must not serve a call: launching or compiling it raises."""

    def __getitem__(self, grid):
        raise AssertionError("a scoring kernel ran that this call must not take")

    def run(self, *args, **kwargs):
        raise AssertionError("a scoring kernel was compiled that this call must not take")


def _bits(a, b) -> bool:
    """Bitwise equality of two FP32 or FP16 tensors: signed zeros and NaN payloads count."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    as_int = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return torch.equal(a.contiguous().view(as_int), b.contiguous().view(as_int))


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
        # what the captured forward gave the operations that are not Linears (captured_linear_inputs)
        cls.real_hc, cls.real_head, cls.real_moe, cls.real_comp, cls.real_gate = {}, {}, {}, [], {}
        cls.real_select, cls.real_select_far = [], None
        from exllamav3.model.math_policy import EXACT_ROWS_FORMS_ALL
        cls.forms = EXACT_ROWS_FORMS_ALL
        for name in FORM_CONSUMERS:
            held = importlib.import_module(f"exllamav3.modules.{name}").ROWS_FORMS
            assert held == cls.forms == FORM_SELECT | FORM_SUMS | FORM_COMPRESS | FORM_GATE, \
                f"{name}.ROWS_FORMS is {held}: the batched forms must all be on under the switch"
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
            # random rows, and the rows the layer got in a real flagged forward where it was captured
            self.captured_linear_inputs()
            real = self.real_moe.get((str(device), b.layer_idx))
            for K in ROWS:
                inputs = [("random rows", self.randn((1, K, mlp.hidden_size), device))]
                if real is not None:
                    inputs.append(("real activations", real[:, :K].clone()))
                for kind, x in inputs:
                    with self.subTest(layer = b.layer_idx, device = str(device), K = K, input = kind,
                                      experts = "cache" if in_cache(b) else "resident"):
                        many = mlp.forward(x, dict(FLAGGED)).clone()
                        self.assertEqual(tuple(many.shape[:2]), (1, K))
                        # a one-row result is a view of a buffer the next call overwrites
                        ones = [mlp.forward(x[:, j:j + 1].clone(), {}).clone() for j in range(K)]
                        self.same_rows(f"moe L{b.layer_idx}, {kind}", many[0], ones)
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
                            ones = [[t.clone() for t in hc.mix_delayed(
                                streams[:, j:j + 1].clone(), {}, None if pre is None else pre[:, j:j + 1].clone(),
                                own_pre)] for j in range(K)]
                            # Twice: the first flagged call with an operand shape on a device returns
                            # the per-row sums and records whether torch's sums over the rows equal
                            # them (dsv41_block._sum_rows); the second takes what was recorded
                            for call in ("first", "second"):
                                # post and comb are views of workspaces the next call overwrites
                                many = [t.clone() for t in hc.mix_delayed(streams, dict(FLAGGED), pre, own_pre)]
                                for i, name in enumerate(names):
                                    self.same_rows(f"{site}.mix:{name} L{b.layer_idx}, {call} flagged call", many[i][0],
                                                   [one[i] for one in ones])
            head = self.tail["hc_head"]
            for K in ROWS:
                with self.subTest(site = "head collapse", device = str(device), K = K):
                    streams = self.randn((1, K, H, D), device, torch.float)
                    pre = torch.rand((1, K, H), generator = self.rng).to(device)
                    ones = [head.forward(streams[:, j:j + 1].clone(), {"dsv41_hc_pre": pre[:, j:j + 1].clone()})
                            for j in range(K)]
                    for call in ("first", "second"):
                        many = head.forward(streams, dict(FLAGGED, dsv41_hc_pre = pre))
                        self.same_rows(f"head collapse, {call} flagged call", many[0], ones)

    def hc_kernel(self, hc, st):
        """ext.hc_mix on st (R, H, D) with a workspace of its own and the column partition of a
        one-row call, as a flagged mix_fused sizes it: (partials (R, chunks, M + 1), post, comb)."""
        R, H, D = st.shape
        M1 = 2 * H + H * H + 1
        chunks = self.ext.hc_mix_num_chunks(1, H * D)
        f32 = lambda *shape: torch.empty(shape, dtype = torch.float, device = st.device)
        partials, post, comb = f32(R, chunks, M1), f32(R, H), f32(R, H, H)
        unused = torch.empty((R, D), dtype = torch.half, device = st.device)
        self.ext.hc_mix(st, hc.fn, hc.base, hc.scale, hc.rms_eps, hc.hc_eps, hc.sinkhorn_iters, partials, post, comb, unused)
        return partials, post, comb

    @torch.inference_mode()
    def test_hyper_connection_sums_over_rows(self):
        # What the sums over the rows rest on, on every GPU, on random rows and on the streams and
        # carried pre-mixes of a real forward (the first and the last layer of each GPU):
        #  - the fused kernel with the column partition of one row gives every row the partials,
        #    post and comb of its one-row call;
        #  - torch's sum of the partials over the chunks, and of the stream products over the
        #    streams, run on all rows, give every row the bits of the sum over its own operand, and
        #    of the extension's per-row entry points;
        #  - through the module: the first flagged call records both sums as row-exact
        #    (ROWS_SUMMED), and the next one computes them with the per-row forms made to raise;
        #    without the form's bit (ROWS_FORMS) the per-row forms serve and nothing is recorded.
        # Control: the same sums reduced along the fastest dimension, another layout of torch's
        # reduction, must differ on the real data, or bitwise equality here shows nothing
        m_block = importlib.import_module("exllamav3.modules.dsv41_block")
        names = ("post", "comb", "collapsed", "pre")
        H, D = self.model.config.hc_mult, self.model.config.hidden_size
        self.captured_linear_inputs()
        self.assertTrue(self.real_hc and self.real_head, "the captured forward reached no hyper-connection site")
        for device in self.devices:
            sites = [(f"{site} L{layer}", hc, streams, pre) for (dev, layer, site), (hc, streams, pre)
                     in sorted(self.real_hc.items(), key = lambda kv: kv[0]) if dev == str(device)]
            self.assertTrue(sites, f"no hyper-connection site was captured on {device}")
            controls = {"partials": False, "collapse": False}
            for site, hc, real_streams, real_pre in sites:
                for K in ROWS:
                    inputs = (("random rows", self.randn((1, K, H, D), device, torch.float),
                               torch.rand((1, K, H), generator = self.rng).to(device)),
                              ("real activations", real_streams[:, :K].contiguous(),
                               None if real_pre is None else real_pre[:, :K].contiguous()))
                    for kind, streams, pre in inputs:
                        what = f"{site} on {device}, {kind}, K={K}"
                        with self.subTest(site = site, device = str(device), K = K, input = kind):
                            # the kernel, and the chunk sum
                            partials, post, comb = self.hc_kernel(hc, streams[0])
                            sums = partials.sum(dim = 1)
                            for r in range(K):
                                p1, post1, comb1 = self.hc_kernel(hc, streams[0, r:r + 1].clone())
                                for name, many, one in (("partials", partials, p1), ("post", post, post1), ("comb", comb, comb1)):
                                    self.assertTrue(_bits(many[r], one[0]), f"{what}: {name} of row {r} differ")
                                self.assertTrue(_bits(sums[r], p1.sum(dim = 1)[0]), f"{what}: chunk sum of row {r} differs")
                            if self.caps & CAP_HC:
                                self.assertTrue(_bits(sums, self.ext.hc_partials_rows(partials)), f"{what}: hc_partials_rows")
                            if kind == "real activations":
                                controls["partials"] |= not _bits(sums, partials.transpose(1, 2).contiguous().sum(dim = 2))
                            # the stream collapse
                            if pre is not None:
                                product = pre.unsqueeze(-1) * streams
                                self.assertTrue(product.is_contiguous(), what)
                                collapsed = product.sum(dim = 2)
                                for r in range(K):
                                    one = (pre[:, r:r + 1].clone().unsqueeze(-1) * streams[:, r:r + 1].clone()).sum(dim = 2)
                                    self.assertTrue(_bits(collapsed[0, r], one[0, 0]), f"{what}: collapse of row {r} differs")
                                if self.caps & CAP_HC:
                                    self.assertTrue(_bits(collapsed, self.ext.hc_collapse_rows(pre, streams)),
                                                    f"{what}: hc_collapse_rows")
                                if kind == "real activations":
                                    controls["collapse"] |= not _bits(collapsed, product.transpose(2, 3).contiguous().sum(dim = 3))
                            # through the module
                            for carried, own_pre in ((pre is not None, False), (False, False), (pre is not None, True)):
                                carry = pre if carried else None
                                ones = [[t.clone() for t in hc.mix_delayed(
                                    streams[:, j:j + 1].clone(), {}, None if carry is None else carry[:, j:j + 1].clone(),
                                    own_pre)] for j in range(K)]
                                with mock.patch.dict(m_block.ROWS_SUMMED, clear = True):
                                    first = [t.clone() for t in hc.mix_delayed(streams, dict(FLAGGED), carry, own_pre)]
                                    recorded = dict(m_block.ROWS_SUMMED)
                                    self.assertEqual(len(recorded), 2 if carried or own_pre else 1, f"{what}: sums recorded")
                                    self.assertTrue(all(recorded.values()),
                                                    f"{what}: torch's sum over the rows differs from the per-row sums: {recorded}")
                                    with mock.patch.object(m_block, "_rows_collapse", _refuse), \
                                            mock.patch.object(m_block, "_rows_partials", _refuse):
                                        second = [t.clone() for t in hc.mix_delayed(streams, dict(FLAGGED), carry, own_pre)]
                                with rows_forms(self.forms & ~FORM_SUMS), mock.patch.dict(m_block.ROWS_SUMMED, clear = True):
                                    per_row = [t.clone() for t in hc.mix_delayed(streams, dict(FLAGGED), carry, own_pre)]
                                    self.assertFalse(m_block.ROWS_SUMMED, f"{what}: sums over the rows without their bit")
                                for call, many in (("first", first), ("second", second), ("per-row", per_row)):
                                    for i, name in enumerate(names):
                                        for j in range(K):
                                            self.assertTrue(_bits(many[i][0, j], ones[j][i][0, 0]),
                                                            f"{what}, {call} flagged call (carried {carried}, own_pre "
                                                            f"{own_pre}): {name} of row {j} differs")
            with self.subTest(device = str(device)):
                self.assertTrue(all(controls.values()),
                                f"on {device} the sums reduced along the fastest dimension equal the sums over the middle "
                                f"one ({controls}): the comparison does not see torch's reduction layout")
            # the head collapse, on the GPU of the head and (copied) on every other one
            head = self.tail["hc_head"]
            real_x, real_pre = next(iter(self.real_head.values()))
            for K in ROWS:
                for kind, x, pre in (("random rows", self.randn((1, K, H, D), device, torch.float),
                                      torch.rand((1, K, H), generator = self.rng).to(device)),
                                     ("real activations", real_x[:, :K].to(device).contiguous(),
                                      real_pre[:, :K].to(device).contiguous())):
                    with self.subTest(site = "head collapse", device = str(device), K = K, input = kind):
                        ones = [head.forward(x[:, j:j + 1].clone(), {"dsv41_hc_pre": pre[:, j:j + 1].clone()})
                                for j in range(K)]
                        with mock.patch.dict(m_block.ROWS_SUMMED, clear = True):
                            first = head.forward(x, dict(FLAGGED, dsv41_hc_pre = pre))
                            self.assertEqual(list(m_block.ROWS_SUMMED.values()), [True], "head collapse: sums recorded")
                            with mock.patch.object(m_block, "_rows_collapse", _refuse):
                                second = head.forward(x, dict(FLAGGED, dsv41_hc_pre = pre))
                        with rows_forms(self.forms & ~FORM_SUMS), mock.patch.dict(m_block.ROWS_SUMMED, clear = True):
                            per_row = head.forward(x, dict(FLAGGED, dsv41_hc_pre = pre))
                            self.assertFalse(m_block.ROWS_SUMMED, "head collapse: sums over the rows without their bit")
                        for call, many in (("first", first), ("second", second), ("per-row", per_row)):
                            for j in range(K):
                                self.assertTrue(_bits(many[0, j], ones[j][0, 0]),
                                                f"head collapse on {device}, {kind}, {call} flagged call: row {j} differs")

    def test_sum_rows_check(self):
        # dsv41_block._sum_rows: the first call with a key computes both forms and returns the
        # per-row one; equal forms make later calls batched, forms that differ in one bit (an ulp,
        # the sign of a zero) keep the per-row form for good and never call the batched one again
        import contextlib
        import io
        m_block = importlib.import_module("exllamav3.modules.dsv41_block")
        ones = torch.tensor([[1.0, -0.0, 3.0]])
        ulp = ones.clone()
        ulp[0, 2] = torch.nextafter(ulp[0, 2], torch.tensor(4.0))
        cases = (("equal forms", ones.clone(), True), ("one ulp", ulp, False),
                 ("the sign of a zero", torch.tensor([[1.0, 0.0, 3.0]]), False),
                 ("another shape", ones.clone().view(3), False), ("another dtype", ones.double(), False))
        for name, many, same in cases:
            with self.subTest(case = name), mock.patch.dict(m_block.ROWS_SUMMED, clear = True):
                batched, rows, key = mock.Mock(return_value = many), mock.Mock(return_value = ones), ("cpu", (1, 3))
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    first = m_block._sum_rows(key, batched, rows)
                self.assertIs(first, ones, "the first call must return the per-row result")
                self.assertEqual((batched.call_count, rows.call_count), (1, 1))
                self.assertIs(m_block.ROWS_SUMMED[key], same)
                self.assertEqual("EXL3_EXACT_ROWS" in printed.getvalue(), not same, "the printed line")
                later = m_block._sum_rows(key, batched, rows)
                self.assertIs(later, many if same else ones)
                self.assertEqual((batched.call_count, rows.call_count), (2, 1) if same else (1, 2))

    @torch.inference_mode()
    def test_native_hgemm_rows(self):
        # hgemm_rows against one hgemm call per cloned row, each with an output of its own, FP32 and
        # FP16 output, on every GPU: the indexer's head-weight projection (an FP16 weight through
        # cuBLAS), copied to a GPU that holds none, on random rows and on the rows of a real
        # forward. And through the Linear: one hgemm_rows call, no Python row loop
        self.need(CAP_HGEMM)
        ext = self.ext
        m_linear = importlib.import_module("exllamav3.modules.linear")
        captured = self.captured_linear_inputs()
        any_block = next(b for b in self.blocks if b.attn.indexer is not None)
        real = next((x for (name, _), (_, _, x) in sorted(captured.items(), key = lambda kv: kv[0])
                     if name == "idx.weights_proj"), None)
        self.assertIsNotNone(real, "the captured forward did not reach idx.weights_proj")
        for device in self.devices:
            own = self.first(device, lambda blk: blk.attn.indexer is not None)
            lin = (own or any_block).attn.indexer.weights_proj
            self.assertEqual(type(lin.inner).__name__, "LinearFP16", "idx.weights_proj is not an FP16 linear")
            weight = lin.inner.weight.to(device).contiguous()
            k, n = weight.shape
            for K in ROWS:
                for kind, x in (("random rows", self.randn((K, k), device)),
                                ("real activations", real.reshape(-1, k)[:K].to(device).contiguous())):
                    for dtype in (torch.float, torch.half):
                        with self.subTest(device = str(device), K = K, input = kind, dtype = str(dtype)):
                            many = ext.hgemm_rows(x, weight, dtype == torch.float)
                            self.assertEqual((tuple(many.shape), many.dtype), ((K, n), dtype))
                            ones = []
                            for j in range(K):
                                c = torch.empty((1, n), dtype = dtype, device = device)
                                ext.hgemm(x[j:j + 1].clone(), weight, c)
                                ones.append(c)
                            self.same_rows(f"hgemm_rows on {device}, {kind}, {dtype}", many, ones)
                if own is not None:
                    with self.subTest(device = str(device), K = K, through = "Linear"):
                        x = self.randn((1, K, lin.in_features), device)
                        with rows_native(self.caps), self.counted("hgemm_rows") as calls, \
                                mock.patch.object(m_linear, "forward_rows", _refuse):
                            self.assertTrue(lin.rows_native(x, FLAGGED), "idx.weights_proj does not go through hgemm_rows")
                            lin.forward(x, dict(FLAGGED))
                        self.assertEqual(len(calls), 1, "hgemm_rows calls of a flagged idx.weights_proj")
                        with rows_native(self.caps & ~CAP_HGEMM), self.counted("hgemm_rows") as calls:
                            self.assertFalse(lin.rows_native(x, FLAGGED))
                            lin.forward(x, dict(FLAGGED))
                        self.assertEqual(len(calls), 0, "hgemm_rows calls without its capability bit")

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
        # on random rows and on the streams and keys of a real forward, on the engram's own GPU and,
        # with its qk copied there, on every other one. A flagged gate is ONE _gate call; without
        # the form's bit (ROWS_FORMS) it is one _gate call per token, with the same bits
        engrams = [b.engram for b in self.blocks if b.engram is not None]
        self.assertTrue(engrams, "the model has no engram layer")
        self.captured_linear_inputs()
        self.assertEqual(set(self.real_gate), {eng.key for eng in engrams}, "engram gates of the captured forward")
        for eng in engrams:
            H, D = eng.qk.shape[-2], eng.qk.shape[-1]
            eps = eng.config.rms_norm_eps
            real_h, real_key = self.real_gate[eng.key]
            for device in self.devices:
                for K in ROWS:
                    kv = self.randn((1, K, (H + 1) * D), device, torch.float)
                    inputs = (("random rows", self.randn((1, K, H, D), device, torch.float), kv, kv[..., :H * D].view(1, K, H, D)),
                              ("real activations", real_h[:, :K].to(device), None, real_key[:, :K].to(device)))
                    for kind, h, kv, key in inputs:
                        with self.subTest(engram = eng.key, device = str(device), K = K, input = kind), \
                                mock.patch.object(eng, "qk", eng.qk.to(device)):
                            with mock.patch.object(eng, "_gate", wraps = eng._gate) as gate:
                                many = eng._gate_rows(h, key, eps)
                            self.assertEqual(gate.call_count, 1, "_gate calls of a flagged gate")
                            self.assertEqual(tuple(many.shape), (1, K, H))
                            with rows_forms(self.forms & ~FORM_GATE), \
                                    mock.patch.object(eng, "_gate", wraps = eng._gate) as gate:
                                loop = eng._gate_rows(h, key, eps)
                            self.assertEqual(gate.call_count, K, "_gate calls of a flagged gate without its bit")
                            self.assertTrue(_bits(many, loop), f"{eng.key} gate on {device}, {kind}: the two forms differ")
                            ones = []
                            for l in range(K):
                                if kv is None:
                                    key1 = key[:, l:l + 1].clone()
                                else:
                                    # a one-token step's key: a view into its own projection output
                                    key1 = kv[:, l:l + 1].clone()[..., :H * D].view(1, 1, H, D)
                                ones.append(eng._gate(h[:, l:l + 1].clone(), key1, eps))
                            self.same_rows(f"{eng.key} gate on {device}, {kind}", many[0], ones)

    @torch.inference_mode()
    def test_compressor(self):
        from exllamav3.constants import PAGE_SIZE
        from exllamav3.modules.dsv41_cached import CompressCarry
        m_comp = importlib.import_module("exllamav3.architecture.dsv41.compressor")
        hd, eps = self.model.config.head_dim, self.model.config.rms_norm_eps

        @contextmanager
        def one_call_each(pooled: bool):
            # a flagged step pools all groups in ONE group_pool call and normalizes in ONE rms_norm call
            with mock.patch.object(m_comp, "group_pool", wraps = m_comp.group_pool) as pool, \
                    mock.patch.object(m_comp, "rms_norm", wraps = m_comp.rms_norm) as norm:
                yield
            self.assertEqual((pool.call_count, norm.call_count), (1 if pooled else 0, 1), "group_pool, rms_norm calls")
        for device in self.devices:
            weight = (1 + torch.randn(hd, generator = self.rng) * 0.1).half().to(device)
            for K in ROWS:
                for pos0 in (1200, 1201):
                    kv = self.randn((K, hd), device, torch.float, 5.0)
                    gate = self.randn((K, hd), device, torch.float, 3.0)
                    with self.subTest(rate = 1, device = str(device), K = K, pos0 = pos0):
                        with one_call_each(False):
                            many, first = CompressCarry.step(None, kv, None, pos0, 1, weight, eps, True)
                        ones = [CompressCarry.step(None, kv[j:j + 1].clone(), None, pos0 + j, 1, weight, eps)
                                for j in range(K)]
                        self.assertEqual((first, many.shape[0]), (pos0, K))
                        self.assertEqual([f for _, f in ones], list(range(pos0, pos0 + K)))
                        self.same_rows("compressor rate 1", many, [lat for lat, _ in ones])
                        with rows_forms(self.forms & ~FORM_COMPRESS):
                            loop, _ = CompressCarry.step(None, kv, None, pos0, 1, weight, eps, True)
                        self.assertTrue(_bits(many, loop), "compressor rate 1: the two forms differ")
                    with self.subTest(rate = 2, device = str(device), K = K, pos0 = pos0):
                        # the ring already holds the row before pos0: an odd pos0 closes its group
                        ring = self.randn((PAGE_SIZE + 2, 2 * hd), device, torch.float)
                        ring_many, ring_ones = ring.clone(), ring.clone()
                        with one_call_each((pos0 + K) // 2 > pos0 // 2):
                            many, first = CompressCarry.step(ring_many, kv, gate, pos0, 2, weight, eps, True)
                        ones = [CompressCarry.step(ring_ones, kv[j:j + 1].clone(), gate[j:j + 1].clone(),
                                                   pos0 + j, 2, weight, eps)[0] for j in range(K)]
                        ones = [row for lat in ones for row in lat]
                        self.assertEqual((first, many.shape[0]), (pos0 // 2, (pos0 + K) // 2 - pos0 // 2))
                        self.same_rows("compressor rate 2", many, ones)
                        self.assertTrue(torch.equal(ring_many, ring_ones), "the carry rings differ")
                        ring_loop = ring.clone()
                        with rows_forms(self.forms & ~FORM_COMPRESS):
                            loop, _ = CompressCarry.step(ring_loop, kv, gate, pos0, 2, weight, eps, True)
                        self.assertTrue(_bits(many, loop), "compressor rate 2: the two forms differ")
                        self.assertTrue(torch.equal(ring_many, ring_loop), "the carry rings of the two forms differ")
            # A rate above 2 (no layer of this model has one): the pooling sums more than two terms,
            # in an order torch may take from the operand, so every group is pooled and normalized
            # by a call of its own, as a one-row step that closes it does
            for rate in (3, 4):
                K, pos0 = max(ROWS), 1200
                groups = (pos0 + K) // rate - pos0 // rate
                with self.subTest(rate = rate, device = str(device)):
                    self.assertGreater(groups, 1, "the case must close several groups")
                    kv = self.randn((K, hd), device, torch.float, 5.0)
                    gate = self.randn((K, hd), device, torch.float, 3.0)
                    ring = self.randn((PAGE_SIZE + rate, 2 * hd), device, torch.float)
                    ring_many, ring_ones = ring.clone(), ring.clone()
                    with mock.patch.object(m_comp, "group_pool", wraps = m_comp.group_pool) as pool, \
                            mock.patch.object(m_comp, "rms_norm", wraps = m_comp.rms_norm) as norm:
                        many, first = CompressCarry.step(ring_many, kv, gate, pos0, rate, weight, eps, True)
                    self.assertEqual((pool.call_count, norm.call_count), (groups, groups),
                                     f"group_pool, rms_norm calls at rate {rate}")
                    ones = [CompressCarry.step(ring_ones, kv[j:j + 1].clone(), gate[j:j + 1].clone(),
                                               pos0 + j, rate, weight, eps)[0] for j in range(K)]
                    ones = [row for lat in ones for row in lat]
                    self.assertEqual((first, many.shape[0]), (pos0 // rate, groups))
                    self.same_rows(f"compressor rate {rate}", many, ones)
                    self.assertTrue(torch.equal(ring_many, ring_ones), "the carry rings differ")
        # the kv and gate rows, the carry ring and the norm weight of the real forward's kv sources,
        # on every GPU (copied to the ones that hold no such source)
        self.captured_linear_inputs()
        rates = sorted({c["m"] for c in self.real_comp})
        self.assertTrue(self.real_comp, "the captured forward compressed nothing")
        print(f" -- exact rows: compressor on the real rows of {len(self.real_comp)} kv sources, rates {rates}", flush = True)
        for device in self.devices:
            for n, c in enumerate(self.real_comp):
                move = lambda t: None if t is None else t.to(device)
                kv_r, gate_r, weight_r, m, pos0 = move(c["kv"]), move(c["score"]), move(c["weight"]), c["m"], c["pos0"]
                for K in ROWS:
                    with self.subTest(rate = m, device = str(device), K = K, source = n, input = "real activations"):
                        ring_many, ring_ones = move(c["carry"]), move(c["carry"])
                        if ring_many is not None:
                            ring_many, ring_ones = ring_many.clone(), ring_ones.clone()
                        row = lambda t, j: None if t is None else t[j:j + 1].clone()
                        many, first = CompressCarry.step(ring_many, kv_r[:K], None if gate_r is None else gate_r[:K],
                                                         pos0, m, weight_r, c["eps"], True)
                        ones = [CompressCarry.step(ring_ones, row(kv_r, j), row(gate_r, j), pos0 + j, m, weight_r,
                                                   c["eps"])[0] for j in range(K)]
                        ones = [lat_row for lat in ones for lat_row in lat]
                        self.assertEqual((first, many.shape[0]), (pos0 // m, (pos0 + K) // m - pos0 // m))
                        self.same_rows(f"compressor rate {m} on {device}, real rows", many, ones)
                        if ring_many is not None:
                            self.assertTrue(torch.equal(ring_many, ring_ones), "the carry rings differ")
                        ring_loop = None if c["carry"] is None else move(c["carry"]).clone()
                        with rows_forms(self.forms & ~FORM_COMPRESS):
                            loop, _ = CompressCarry.step(ring_loop, kv_r[:K], None if gate_r is None else gate_r[:K],
                                                         pos0, m, weight_r, c["eps"], True)
                        self.assertTrue(_bits(many, loop), f"compressor rate {m} on {device}, real rows: the two forms differ")
        # per_row is defined on (rows, d) latents: any other shape is refused, never reduced whole
        with self.assertRaisesRegex(ValueError, "per_row"):
            m_comp.rms_norm(self.randn((1, 3, hd), self.devices[0], torch.float), None, eps, True)

    # -- the index selection --

    @contextmanager
    def scored(self):
        """dsa_indexer_scores wrapped for the block: per call, (its one_row argument, a copy of the
        (rows, T) scores it returned)."""
        m_triton = importlib.import_module("exllamav3.modules.attention_fn.dsa_triton")
        calls, orig = [], m_triton.dsa_indexer_scores

        def wrapper(*args, **kwargs):
            out = orig(*args, **kwargs)
            calls.append((bool(kwargs.get("one_row")), out.clone()))
            return out
        with mock.patch.object(m_triton, "dsa_indexer_scores", wrapper):
            yield calls

    def one_row_selections(self, q, w, pool, K, *, pos0, m, cand_in = None, **kw):
        """K one-row select_topk calls on cloned rows, the call a decode step makes at each row's
        position: [(SelectResult, the (1, ec) scores)]."""
        from exllamav3.modules.dsv41_select import select_topk
        ones = []
        for j in range(K):
            with self.scored() as scores:
                one = select_topk(q[j:j + 1].clone(), w[j:j + 1].clone(), pool, pos0 = pos0 + j, m = m,
                                  ec = (pos0 + j + 1) // m,
                                  cand_in = None if cand_in is None else cand_in[j:j + 1].clone(), **kw)
            self.assertEqual([(flag, s.shape[0]) for flag, s in scores], [(False, 1)], "scorer calls of a one-row call")
            ones.append((one, scores[0][1]))
        return ones

    def select_rows(self, what, q, w, pool, K, *, pos0, m, cand_in = None, served = True, **kw) -> bool:
        """
        select_topk with one_row on the first K rows against K one-row calls: the indices, the
        candidate blocks and the scores below each row's own entry count, bit for bit; the scores
        from there on -inf. The shared pass must be ONE scorer call, pinned, that never reaches
        the query-tiled kernel, and one dsa_topk call (two where candidates are published).
        served False, and every call on a GPU that is not of a type the pass is verified on
        (pass_device): the call must refuse the pass (None) before it launches anything. Returns
        whether it served.
        """
        m_triton = importlib.import_module("exllamav3.modules.attention_fn.dsa_triton")
        from exllamav3.modules.dsv41_select import select_topk
        want_cand = bool(kw.get("want_cand"))
        served = served and self.pass_device(q.device)
        with self.scored() as scores, self.counted("dsa_topk") as topks, \
                mock.patch.object(m_triton, "_dsa_indexer_kernel", _RefusedKernel()):
            many = select_topk(q[:K], w[:K], pool, pos0 = pos0, m = m, ec = (pos0 + K) // m,
                               cand_in = None if cand_in is None else cand_in[:K], one_row = True, **kw)
        if many is None:
            self.assertFalse(served, f"{what}: the rows did not share a pass")
            self.assertEqual((len(scores), len(topks)), (0, 0), f"{what}: a refused pass launched")
            return False
        self.assertTrue(served, f"{what}: the rows shared a pass that their one-row calls cannot share")
        self.assertEqual([(flag, sc.shape[0]) for flag, sc in scores], [(True, K)], f"{what}: scorer calls")
        self.assertEqual(len(topks), 2 if want_cand else 1, f"{what}: dsa_topk calls")
        many_scores = scores[0][1]
        ones = self.one_row_selections(q, w, pool, K, pos0 = pos0, m = m, cand_in = cand_in, **kw)
        self.assertEqual(many.k_len, ones[0][0].k_len, what)
        self.assertEqual(tuple(many.indices.shape[:1]), (K,), what)
        self.assertEqual(many.cand is not None, want_cand, what)
        for j, (one, one_scores) in enumerate(ones):
            ec_j = (pos0 + j + 1) // m
            self.assertEqual(tuple(one_scores.shape), (1, ec_j), f"{what}: scores of the one-row call {j}")
            self.assertTrue(_bits(many_scores[j, :ec_j], one_scores[0]), f"{what}: scores of row {j} differ")
            self.assertTrue(bool((many_scores[j, ec_j:] == float("-inf")).all()),
                            f"{what}: row {j} has scores past its own entry count")
            self.assertTrue(torch.equal(many.indices[j], one.indices[0]), f"{what}: indices of row {j} differ")
            self.assertEqual(one.cand is not None, want_cand, what)
            if want_cand:
                self.assertTrue(torch.equal(many.cand[j], one.cand[0]), f"{what}: candidate blocks of row {j} differ")
        return True

    def pass_device(self, device) -> bool:
        """Whether select_topk shares a pass between rows on this GPU (dsv41_select.ROWS_PASS_SM)."""
        return tuple(torch.cuda.get_device_capability(torch.device(device))) in ONE_LAUNCH_SM

    def selection_on(self, c, device):
        """A captured selection's operands on `device` (copied when they live elsewhere)."""
        move = lambda t: None if t is None else t.to(device)
        return move(c["q"]), move(c["w"]), move(c["pool"]), dict(c["args"], bt_row = move(c["bt_row"])), move(c["cand_in"])

    @staticmethod
    def selection_name(c):
        a = c["args"]
        kind = "candidate source" if a["want_cand"] else "candidate consumer" if c["cand_in"] is not None else "source"
        return f"rate-{a['m']} {kind} at position {a['pos0']} (pool on {c['q'].device})"

    @torch.inference_mode()
    def test_select_rows_equal_one_row_calls(self):
        # The selections of two real forwards, on every GPU: on the GPU that made them, and copied
        # to every other one (in the serving placement the sm_89 card holds no index source)
        captured = self.captured_selections()
        self.assertTrue(captured, "the forwards made no selection of 2 to 8 rows")
        names = sorted({self.selection_name(c) for c in captured})
        print(f" -- exact rows: {len(captured)} captured selections: {'; '.join(names)}", flush = True)
        self.assertTrue(any(c["args"]["want_cand"] for c in captured), "no candidate source was captured")
        # the candidate mask cuts only past n_blocks * block entries
        cutting = [c for c in captured if c["cand_in"] is not None
                   and (c["args"]["pos0"] + 1) // c["args"]["m"] > c["args"]["n_blocks"] * c["args"]["block"]]
        self.assertTrue(cutting, "no captured selection has a candidate mask that cuts")
        # What the shared pass rests on is seen only where rows differ in their entry counts: the
        # earlier row's key tile then holds real keys where its one-row call loads zeros. Every
        # rate must have such calls on every GPU that shares a pass
        for device in self.devices:
            unequal = {}
            for c in captured:
                q, w, pool, kw, cand_in = self.selection_on(c, device)
                pos0, m = kw.pop("pos0"), kw.pop("m")
                for K in ROWS:
                    with self.subTest(selection = self.selection_name(c), device = str(device), K = K):
                        took = self.select_rows(f"{self.selection_name(c)} on {device}, K={K}", q, w, pool, K,
                                                pos0 = pos0, m = m, cand_in = cand_in, **kw)
                        unequal[m] = unequal.get(m, 0) + int(took and (pos0 + 1) // m != (pos0 + K) // m)
            print(f" -- exact rows: selections on {device}: one pass for rows of different entry counts in "
                  f"{', '.join(f'{n} calls at rate {m}' for m, n in sorted(unequal.items()))}", flush = True)
            if self.pass_device(device):
                with self.subTest(device = str(device)):
                    self.assertTrue(unequal and all(unequal.values()),
                                    f"on {device} no shared pass had rows of different entry counts at some rate "
                                    f"({unequal}): the key columns past a row's own count were never real keys")

    @torch.inference_mode()
    def test_select_pass_needs_a_verified_gpu_type(self):
        # The shared pass rests on a property of the tensor-core instruction that is compared on
        # the GPU types of ROWS_PASS_SM alone (the list of exact_rows_device_ok). With any other
        # compute capability select_topk must refuse the pass before it launches anything: shown
        # on every GPU with the capability replaced. The same selections serve as they are in
        # test_select_rows_equal_one_row_calls
        m_select = importlib.import_module("exllamav3.modules.dsv41_select")
        self.assertEqual(tuple(m_select.ROWS_PASS_SM), ONE_LAUNCH_SM, "the GPU types of the shared pass")
        self.assertFalse(m_select._rows_pass_device(torch.device("cpu")))
        captured = self.captured_selections()
        self.assertTrue(captured, "the forwards made no selection of 2 to 8 rows")
        for device in self.devices:
            self.assertEqual(m_select._rows_pass_device(device), self.pass_device(device), f"the type of {device}")
            for c in captured[:2] + captured[-2:]:
                q, w, pool, kw, cand_in = self.selection_on(c, device)
                pos0, m = kw.pop("pos0"), kw.pop("m")
                for sm in ((7, 5), (8, 6), (9, 0), (10, 0), (12, 1)):
                    for K in (2, 8):
                        with self.subTest(selection = self.selection_name(c), device = str(device), sm = sm, K = K), \
                                mock.patch.dict(m_select._ROWS_PASS_DEVICES, clear = True), \
                                mock.patch.object(torch.cuda, "get_device_capability", return_value = sm):
                            took = self.select_rows(f"{self.selection_name(c)} on {device} as sm {sm}, K={K}", q, w,
                                                    pool, K, pos0 = pos0, m = m, cand_in = cand_in, **kw)
                            self.assertFalse(took, f"a shared pass on compute capability {sm}")
            self.assertEqual(m_select._rows_pass_device(device), self.pass_device(device), f"the type of {device}")

    @torch.inference_mode()
    def test_select_control_unpinned_differs(self):
        # The same 8 rows without one_row take the query-tiled kernel, which adds the heads in
        # another order: on every GPU some score of some captured selection must differ from the
        # one-row calls, or equal scores in the test above show nothing there
        m_triton = importlib.import_module("exllamav3.modules.attention_fn.dsa_triton")
        from exllamav3.modules.dsv41_select import select_topk
        captured, K = self.captured_selections(), 8
        for device in self.devices:
            differing = []
            for c in captured:
                q, w, pool, kw, cand_in = self.selection_on(c, device)
                pos0, m = kw.pop("pos0"), kw.pop("m")
                with self.scored() as scores, mock.patch.object(m_triton, "_dsa_indexer_fewq_kernel", _RefusedKernel()):
                    select_topk(q[:K], w[:K], pool, pos0 = pos0, m = m, ec = (pos0 + K) // m, cand_in = cand_in, **kw)
                self.assertEqual([(flag, sc.shape[0]) for flag, sc in scores], [(False, K)])
                ones = self.one_row_selections(q, w, pool, K, pos0 = pos0, m = m, cand_in = cand_in, **kw)
                if any(not _bits(scores[0][1][j, :(pos0 + j + 1) // m], one_scores[0])
                       for j, (_, one_scores) in enumerate(ones)):
                    differing.append(self.selection_name(c))
            print(f" -- exact rows: selection control on {device}: the unpinned scores of 8 rows differ from the "
                  f"one-row calls for {len(differing)} of {len(captured)} captured selections", flush = True)
            with self.subTest(device = str(device)):
                self.assertTrue(differing, f"unpinned scores of 8 rows equal the one-row calls on {device}: the "
                                           f"score comparison does not discriminate there")

    @torch.inference_mode()
    def test_select_crafted_pools(self):
        # Pools and positions that put the rows of a call on both sides of something the selection
        # decides by size: the split of dsa_topk (from 32,768 columns) and its span count, the
        # block count against the 2048 published blocks, candidate lists with ids a row's own call
        # drops, ties at the cut, both parities at rate 2. And the calls that must refuse the pass:
        # rows whose one-row calls take two tile widths, and a tile outside the score budget
        source = next(b.attn for b in self.blocks if b.attn.indexer is not None)
        H, D = source.index_n_heads, source.index_head_dim
        topk, block, n_blocks = source.index_topk, source.candidate_block_size, source.candidate_topk_blocks
        self.assertEqual((topk, block, n_blocks), (512, 8, 2048), "the cases below are laid out for these sizes")
        kmax = max(ROWS)

        def lists(pos0, kind):
            nb_last = -(-(pos0 + kmax) // block)
            rows = []
            for _ in range(kmax):
                if kind == "few":
                    ids = torch.randperm(nb_last - 64, generator = self.rng)[:40]
                else:
                    # the last block of the call (outside the first rows' own block count) and an id no pool has
                    ids = torch.cat([torch.randperm(nb_last - 1, generator = self.rng)[:1800],
                                     torch.tensor([nb_last - 1, 100000])])
                row = torch.full((n_blocks,), -1, dtype = torch.int32)
                row[:ids.numel()] = ids.sort().values.to(torch.int32)
                rows.append(row)
            return torch.stack(rows)

        # name, m, pos0 (None: 5000 - K), publish candidates, candidate lists, integer operands,
        # geometry, the layouts it runs on, whether K rows share a pass
        every, paged = ("dense", 128, 256), (256,)
        cases = (
            ("the top-k split starts inside the call", 1, 32755, False, None, False, {}, every, lambda K, p: True),
            ("the split's span count changes", 1, 36855, False, None, False, {}, (256,), lambda K, p: True),
            ("the block count reaches 2048", 1, 16370, True, None, False, {}, ("dense", 256), lambda K, p: True),
            ("the block count passes 2048", 1, 16379, True, None, False, {}, ("dense", 256), lambda K, p: True),
            ("candidate lists", 1, 20006, False, "lists", False, {}, ("dense", 256), lambda K, p: True),
            ("few candidate blocks", 1, 20006, False, "few", False, {}, (256,), lambda K, p: True),
            ("ties at the cut", 1, 3000, False, None, True, {}, ("dense", 256), lambda K, p: True),
            ("ties among the published blocks", 1, 3000, True, None, True, {}, (256,), lambda K, p: True),
            ("rate 2, even position", 2, 3000, False, None, False, {}, every, lambda K, p: True),
            ("rate 2, odd position", 2, 3001, False, None, False, {}, every, lambda K, p: True),
            ("the rows straddle a tile width", 1, 65533, False, None, False, {}, paged, lambda K, p: p + K <= 65536),
            ("past the tile width", 1, 65540, False, None, False, {}, paged, lambda K, p: True),
            ("a tile outside the score budget", 1, None, False, None, False, dict(slab_rows = 16, tile_entries = 1024),
             paged, lambda K, p: K == 2),
        )
        entries = 65540 + kmax
        for device in self.devices:
            pools = {}
            for layout in every:
                for integer in (False, True):
                    n = 4096 if integer else (entries if layout != 128 else 40000)
                    rows = n if layout == "dense" else -(-n // layout) * layout
                    keys = torch.randint(-2, 3, (rows, D), generator = self.rng).half().to(device) if integer \
                        else self.randn((rows, D), device)
                    if layout == "dense":
                        pools[(layout, integer)] = (keys, None, 0)
                    else:
                        table = torch.randperm(rows // layout, generator = self.rng).to(torch.int32)[None, :].to(device)
                        pools[(layout, integer)] = (keys.view(-1, layout, D), table, layout)
            served, refused = set(), set()
            for name, m, pos0_case, want_cand, cand_kind, integer, geometry, layouts, shares in cases:
                for layout in layouts:
                    pool, bt_row, epp = pools[(layout, integer)]
                    for K in ROWS:
                        pos0 = 5000 - K if pos0_case is None else pos0_case
                        if integer:
                            q = torch.randint(-2, 3, (kmax, H, D), generator = self.rng).half().to(device)
                            w = torch.full((kmax, H), 2.0 ** -6, device = device)
                        else:
                            q = self.randn((kmax, H, D), device)
                            w = (torch.rand((kmax, H), generator = self.rng) * 2.0 ** -6).to(device)
                        cand_in = None if cand_kind is None else lists(pos0, cand_kind).to(device)
                        what = f"{name}, {layout} pool, rate {m}, position {pos0} on {device}, K={K}"
                        with self.subTest(case = name, layout = str(layout), device = str(device), K = K):
                            took = self.select_rows(what, q, w, pool, K, pos0 = pos0, m = m, bt_row = bt_row, epp = epp,
                                                    topk = topk, cand_in = cand_in, want_cand = want_cand, block = block,
                                                    n_blocks = n_blocks, served = shares(K, pos0), **geometry)
                            (served if took else refused).add(name)
            print(f" -- exact rows: crafted selections on {device}: one pass for {len(served)} cases, refused "
                  f"(the row loop serves) in: {', '.join(sorted(refused)) or 'NONE'}", flush = True)
            with self.subTest(device = str(device)):
                if self.pass_device(device):
                    self.assertEqual(refused, {"the rows straddle a tile width", "a tile outside the score budget"})
                else:
                    self.assertFalse(served, f"{device} is not of a type the shared pass is verified on")

    @torch.inference_mode()
    def test_index_select_falls_back(self):
        # DSV41Attention._index_select inside a real flagged forward, three times per index source:
        # with select_topk made to refuse the shared pass (the row loop: 1 + rows calls), without
        # the form's bit (ROWS_FORMS: the row loop, rows calls, the pass never asked for) and as
        # it is (one call on a GPU of a verified type). All must select the same
        m_dsv41 = importlib.import_module("exllamav3.modules.dsv41")
        seen, hooked = [], []

        def hook(at):
            orig = at._index_select

            def index_select(x, params, q_res, idx_pool, bt_row, epp, pos0, ec, b):
                if not (params.get("exact_rows") and x.shape[1] > 1):
                    return orig(x, params, q_res, idx_pool, bt_row, epp, pos0, ec, b)
                real, calls = m_dsv41.select_topk, []

                def counting(*args, **kw):
                    calls.append(bool(kw.get("one_row")))
                    return real(*args, **kw)

                def refusing(*args, **kw):
                    calls.append(bool(kw.get("one_row")))
                    return None if kw.get("one_row") else real(*args, **kw)
                with mock.patch.object(m_dsv41, "select_topk", refusing):
                    loop = orig(x, params, q_res, idx_pool, bt_row, epp, pos0, ec, b)
                loop_calls = list(calls)
                calls.clear()
                with mock.patch.object(m_dsv41, "ROWS_FORMS", self.forms & ~FORM_SELECT), \
                        mock.patch.object(m_dsv41, "select_topk", counting):
                    masked = orig(x, params, q_res, idx_pool, bt_row, epp, pos0, ec, b)
                masked_calls = list(calls)
                calls.clear()
                with mock.patch.object(m_dsv41, "select_topk", counting):
                    one = orig(x, params, q_res, idx_pool, bt_row, epp, pos0, ec, b)
                same = all(
                    other.k_len == one.k_len and torch.equal(other.indices, one.indices)
                    and (other.cand is None) == (one.cand is None)
                    and (one.cand is None or torch.equal(other.cand, one.cand)) for other in (loop, masked))
                seen.append((at.layer_idx, str(at.device), x.shape[1], loop_calls, masked_calls, list(calls), same))
                return one
            at._index_select = index_select
            hooked.append(at)
        try:
            for b in self.blocks:
                if b.attn.is_index_source:
                    hook(b.attn)
            self.verify_forward(1200, 8)
        finally:
            for at in hooked:
                del at._index_select
        self.assertEqual(len(seen), len(hooked), "not every index source selected in the flagged forward")
        for layer, device, rows, loop_calls, masked_calls, calls, same in seen:
            with self.subTest(layer = layer, device = device):
                self.assertEqual(loop_calls, [True] + [False] * rows, "select_topk calls of the row loop")
                self.assertEqual(masked_calls, [False] * rows, "select_topk calls without the form's bit")
                self.assertEqual(calls, [True] if self.pass_device(device) else [True] + [False] * rows,
                                 "select_topk calls of the shared pass")
                self.assertTrue(same, f"L{layer}: the shared pass and the row loop select differently")
        print(f" -- exact rows: _index_select on layers {[layer for layer, *_ in seen]}: one select_topk call for "
              f"{seen[0][2]} rows on a GPU of a verified type, and the row loop when that call refuses or is not "
              f"asked for", flush = True)

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
        m_cached = importlib.import_module("exllamav3.modules.dsv41_cached")
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
        hcs, heads, moes, comps, gates, selections, unhook = {}, {}, {}, [], {}, [], []
        rows_of = lambda t, dim: 2 <= t.shape[dim] <= 8

        def instance_hook(obj, name, wrapper):
            orig = getattr(obj, name)
            setattr(obj, name, lambda *args, **kwargs: wrapper(orig, *args, **kwargs))
            unhook.append((obj, name))

        def hook_hc(key, hc):
            def mix(orig, streams, params, carried_pre, own_pre = False):
                if key not in hcs and rows_of(streams, 1):
                    hcs[key] = (hc, streams.clone(), None if carried_pre is None else carried_pre.clone())
                return orig(streams, params, carried_pre, own_pre)
            instance_hook(hc, "mix_delayed", mix)

        def head_collapse(orig, x, params, out_dtype = None):
            if not heads and rows_of(x, 1):
                heads[str(x.device)] = (x.clone(), params["dsv41_hc_pre"].to(x.device).clone())
            return orig(x, params, out_dtype)

        def hook_moe(key, mlp):
            def forward(orig, x, params, out_dtype = None):
                if key not in moes and params.get("exact_rows") and rows_of(x, -2):
                    moes[key] = x.clone()
                return orig(x, params, out_dtype)
            instance_hook(mlp, "forward", forward)

        def hook_gate(eng):
            def gate_rows(orig, h, key, eps):
                if eng.key not in gates and rows_of(h, 1):
                    gates[eng.key] = (h.clone(), key.clone())
                return orig(h, key, eps)
            instance_hook(eng, "_gate_rows", gate_rows)

        step_orig = m_cached.CompressCarry.step

        def step(carry, kv, score, pos0, m, norm_weight = None, eps = 1e-6, per_row = False):
            if per_row and rows_of(kv, 0):
                comps.append(dict(carry = None if carry is None else carry.clone(), kv = kv.clone(),
                                  score = None if score is None else score.clone(), pos0 = pos0, m = m,
                                  weight = norm_weight.clone(), eps = eps))
            return step_orig(carry, kv, score, pos0, m, norm_weight, eps, per_row)

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
                on_device = [b for b in cls.blocks if torch.device(b.device) == device]
                hook_woa(str(device), on_device[0].attn)
                # the hyper-connection sites of the first and of the last block of the device
                for b in {on_device[0].layer_idx: on_device[0], on_device[-1].layer_idx: on_device[-1]}.values():
                    hook_hc((str(device), b.layer_idx, "hc_attn"), b.attn_hc)
                    hook_hc((str(device), b.layer_idx, "hc_ffn"), b.mlp_hc)
                in_cache = lambda blk: getattr(blk.mlp, "tier", None) is not None
                for b in (next((b for b in on_device if in_cache(b)), None),
                          next((b for b in on_device if not in_cache(b)), None)):
                    if b is not None:
                        hook_moe((str(device), b.layer_idx), b.mlp)
            instance_hook(cls.tail["hc_head"], "forward", head_collapse)
            for b in cls.blocks:
                if b.engram is not None:
                    hook_gate(b.engram)
            clear = lambda: [c.clear() for c in (captured, woa, hcs, heads, moes, comps, gates, selections)]
            with mock.patch.object(m_cached.CompressCarry, "step", staticmethod(step)), cls.recorded_selections(selections):
                # the prefill's own calls of 2 to 8 rows or entries, if any, are dropped
                cls.verify_forward(1200, 8, after_prefill = clear)
        finally:
            for lin in hooked:
                del lin.forward
            for at in hooked_attn:
                del at._project_o_rows
            for obj, name in unhook:
                delattr(obj, name)
        cls.real_inputs = captured
        cls.real_woa = woa
        cls.real_hc, cls.real_head, cls.real_moe, cls.real_comp, cls.real_gate = hcs, heads, moes, comps, gates
        cls.real_select = selections
        return captured

    @classmethod
    def verify_forward(cls, P: int, K: int = 8, after_prefill = None):
        """A prefill of P random tokens, in chunks as a job prefills, then one cached forward of K
        rows, driven as the generator drives a verify forward (the params of
        tools/dsv41_rowprobe.py, Driver). The caller's hooks see both; after_prefill runs between."""
        from exllamav3.constants import PAGE_SIZE
        model, cache = cls.model, cls.cache
        ids = torch.randint(0, model.config.vocab_size, (1, P + K), generator = cls.rng)
        state = cache.get_new_state()
        pages = cache.max_num_tokens // PAGE_SIZE // cache.num_slots
        block_table = torch.arange(state.slot * pages, (state.slot + 1) * pages, dtype = torch.int32)[None, :]
        try:
            for a in range(0, P, 2048):
                model.prefill(ids[:, a : min(a + 2048, P)], {
                    "attn_mode": "flash_attn", "block_table": block_table, "cache": cache,
                    "cache_seqlens": torch.tensor([a], dtype = torch.int32),
                    "recurrent_states": [state], "indexed_embeddings": []})
            assert state.position == P, f"the prefill left the state at {state.position}, expected {P}"
            if after_prefill is not None:
                after_prefill()
            model.forward(ids[:, P:], {
                "attn_mode": "flash_attn", "block_table": block_table, "cache": cache,
                "cache_seqlens": torch.tensor([state.position], dtype = torch.int32),
                "recurrent_states": [state], "indexed_embeddings": [], "positions": None,
                "recurrent_history": True, "pinned_staging": True})
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
        finally:
            cache.release_state(state)

    @classmethod
    @contextmanager
    def recorded_selections(cls, into: list):
        """The index selections of a flagged forward, recorded for the block: the operands of every
        select_topk call DSV41Attention._index_select makes with one_row on 2 to 8 rows, copied
        (the pool too: the Cache's pages are reused after the forward)."""
        m_dsv41 = importlib.import_module("exllamav3.modules.dsv41")
        orig = m_dsv41.select_topk

        def select(q_idx, wts, idx_pool, **kw):
            if kw.get("one_row") and 2 <= q_idx.shape[0] <= 8:
                copy = lambda t: None if t is None else t.clone()
                into.append(dict(
                    q = q_idx.clone(), w = wts.clone(), pool = idx_pool.clone(), bt_row = copy(kw.get("bt_row")),
                    cand_in = copy(kw.get("cand_in")),
                    args = {k: kw[k] for k in ("epp", "pos0", "m", "topk", "want_cand", "block", "n_blocks")}))
            return orig(q_idx, wts, idx_pool, **kw)
        with mock.patch.object(m_dsv41, "select_topk", select):
            yield into

    @classmethod
    def captured_selections(cls) -> list:
        """The selections of two flagged forwards of 8 rows: after 1200 tokens (every index source
        selects, every visible block is a candidate) and after 19000 (the rate-1 pools hold more
        than 16,384 entries: the candidate mask of the sources above the candidate source cuts)."""
        cls.captured_linear_inputs()
        if cls.real_select_far is None:
            far = []
            with cls.recorded_selections(far):
                cls.verify_forward(19000, 8, after_prefill = far.clear)
            cls.real_select_far = far
        return cls.real_select + cls.real_select_far

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
                            if not lin.rows_native(x, FLAGGED) or type(lin.inner).__name__ != "LinearEXL3":
                                # the Python row loop serves it, or hgemm_rows (an FP16 weight)
                                python.setdefault(device, set()).add(name)
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
        # The MoE layer through its forward, as the engine calls it: with the extension's entry
        # points against the Python row loop. Where the extension reports the grouped MoE launch
        # and the router's single launch, the engine itself must ask for them (the extension's
        # counts around the forward), and with those two bits masked it must not: the launch per
        # slot and the projection per row, with the same bits
        self.need(CAP_ROUTER | CAP_MOE)
        m_moe = importlib.import_module("exllamav3.modules.dsv41_moe")
        batched = self.caps & (CAP_MOE_GROUPED | CAP_ROUTER_ONE_LAUNCH)
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
                                    native, grew = self.served_forms(lambda: mlp.forward(x, dict(FLAGGED)).clone())
                                self.assertEqual(len(calls), 1, "routing_ds3_nogroup_rows calls")
                                if self.caps & CAP_MOE_GROUPED:
                                    self.assertEqual(grew[CAP_MOE_GROUPED],
                                                     (2 if mlp.bc.sh_coop else 1) if mlp.bc.rows_grouped() else 0,
                                                     "grouped launch pairs of a flagged MoE forward")
                                if self.caps & CAP_ROUTER_ONE_LAUNCH:
                                    self.assertEqual(grew[CAP_ROUTER_ONE_LAUNCH], 1,
                                                     "one-launch projections of a flagged MoE forward")
                            else:
                                native = mlp.forward(x, dict(FLAGGED)).clone()
                        if one and batched:
                            with rows_native(self.caps & ~batched), mock.patch.object(m_moe, "forward_rows", _refuse):
                                plain, grew = self.served_forms(lambda: mlp.forward(x, dict(FLAGGED)).clone())
                            self.assertFalse(any(grew.values()), f"batched forms served without their bits: {grew}")
                            self.same(f"moe L{b.layer_idx}, without the grouped launch and the single projection",
                                      native, plain)
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

    def served(self, cap: int, call):
        """call(), and how many launches of the batched form of capability bit `cap` it made
        (exact_rows_served)."""
        before = int(self.ext.exact_rows_served(cap))
        result = call()
        return result, int(self.ext.exact_rows_served(cap)) - before

    def served_forms(self, call):
        """call(), and the launches it made of each batched form the extension counts:
        {capability bit: launches}, for the bits the extension reports (exact_rows_served)."""
        bits = [bit for bit in (CAP_MOE_GROUPED, CAP_ROUTER_ONE_LAUNCH) if self.caps & bit]
        before = {bit: int(self.ext.exact_rows_served(bit)) for bit in bits}
        result = call()
        return result, {bit: int(self.ext.exact_rows_served(bit)) - before[bit] for bit in bits}

    def moe_launch(self, mlp, y, sel, w, rows_entry: bool, grouped: bool = False):
        """The fused decode launch pair of an MoE layer on a given routing, as BlockSparseMLP.forward
        makes it (the expert cache lookup of the call first, where the layer has one). rows_entry:
        run_bszN_rows, with its grouped argument when asked (an extension that reports
        CAP_MOE_GROUPED). Returns a copy of the routed sum, (rows, hidden)."""
        cached = getattr(mlp, "tier", None) is not None
        if cached:
            mlp.tier_resolve(sel, y.shape[0], {})
        if rows_entry and grouped:
            mlp.bc.run_bszN_rows(y, sel, w, True)
        elif rows_entry:
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
        # the tile, which may not show here. Both forms of the entry point are compared: a launch
        # per slot, and (an extension that reports CAP_MOE_GROUPED) the rows that picked one expert
        # as one run, which the extension makes on the GPU types of bc.rows_grouped and counts:
        # the shared expert's launch pair and the routed one (exact_rows_served). On random rows
        # and on the rows of a real forward.
        # Every GPU must contribute a layer: a pass says nothing about a card that was skipped
        self.need(CAP_MOE)
        self.captured_linear_inputs()
        forms = (False, True) if self.caps & CAP_MOE_GROUPED else (False,)
        tested, grouped_on = [], []
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
                groups = bool(self.caps & CAP_MOE_GROUPED) and bool(mlp.bc.rows_grouped())
                if groups:
                    grouped_on.append(f"L{b.layer_idx} on {device}")
                real = self.real_moe.get((str(device), b.layer_idx))
                for K in ROWS:
                    inputs = [("random rows", self.randn((K, mlp.hidden_size), device))]
                    if real is not None:
                        inputs.append(("real activations", real.reshape(-1, mlp.hidden_size)[:K].clone()))
                    for kind, y in inputs:
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
                            ones = [self.moe_launch(mlp, y[j:j + 1].clone(), sel_c[j:j + 1].clone(),
                                                    w_c[j:j + 1].clone(), False) for j in range(K)]
                            for grouped in forms:
                                with self.subTest(layer = b.layer_idx, device = str(device), K = K, routing = case,
                                                  input = kind, grouped = grouped):
                                    what = f"moe experts L{b.layer_idx}, {case}, {kind}, grouped {grouped}"
                                    if self.caps & CAP_MOE_GROUPED:
                                        many, grew = self.served(
                                            CAP_MOE_GROUPED, lambda: self.moe_launch(mlp, y, sel_c, w_c, True, grouped))
                                        expect = (2 if mlp.bc.sh_coop else 1) if grouped and groups else 0
                                        self.assertEqual(grew, expect, f"{what}: grouped launch pairs")
                                    else:
                                        many = self.moe_launch(mlp, y, sel_c, w_c, True)
                                    self.same_rows(what, many, ones)
            with self.subTest(device = str(device)):
                self.assertGreater(len(tested), on_device, f"run_bszN_rows serves no tested MoE layer on {device}")
        print(f" -- exact rows: run_bszN_rows against one-row run_bszN, K = 2..8, on {'; '.join(tested) or 'NOTHING'}",
              flush = True)
        if self.caps & CAP_MOE_GROUPED:
            print(f" -- exact rows: with the rows grouped by expert on {'; '.join(grouped_on) or 'NO LAYER'}", flush = True)
            for device in self.devices:
                if tuple(torch.cuda.get_device_capability(device)) in ONE_LAUNCH_SM:
                    with self.subTest(device = str(device)):
                        self.assertTrue(any(name.endswith(f"on {device}") for name in grouped_on),
                                        f"no tested MoE layer on {device} groups the rows")

    @torch.inference_mode()
    def test_moe_grouped_form(self):
        # What the grouped launch is: the launch of an unflagged call with the tile of a one-row
        # call. Below Blackwell the tile does not follow the slot count, so run_bszN_rows with the
        # rows grouped is run_bszN itself: the two must be equal at every K. On Blackwell they are
        # the same launch below 32 slots; from there the unflagged call may take the wide tile
        # (reported: where it differs from the one-row calls, the comparison of
        # test_native_moe_selections sees a changed tile), and the grouped launch must not
        self.need(CAP_MOE | CAP_MOE_GROUPED)
        self.captured_linear_inputs()
        for device in self.devices:
            blackwell = torch.cuda.get_device_capability(device)[0] >= 12
            for b, experts in self.moe_layers(device):
                mlp = b.mlp
                topk, cfg = mlp.num_experts_per_tok, mlp.routing_cfg
                if mlp.bc is None or not mlp.bc.rows_exact_ok(topk) or not mlp.bc.rows_grouped():
                    print(f" -- exact rows: MoE L{b.layer_idx} on {device}: run_bszN_rows does not group the rows",
                          flush = True)
                    continue
                real = self.real_moe.get((str(device), b.layer_idx))
                wide_differs = []
                for K in ROWS:
                    inputs = [("random rows", self.randn((K, mlp.hidden_size), device))]
                    if real is not None:
                        inputs.append(("real activations", real.reshape(-1, mlp.hidden_size)[:K].clone()))
                    for kind, y in inputs:
                        with self.subTest(layer = b.layer_idx, device = str(device), K = K, input = kind):
                            sel, w = (t.clone() for t in mlp.routing_fn(K, cfg, y, {}))
                            rows = self.moe_launch(mlp, y, sel, w, True, True)
                            stock = self.moe_launch(mlp, y, sel, w, False)
                            if not blackwell or K * topk < 32:
                                self.assertTrue(torch.equal(rows, stock),
                                                f"moe experts L{b.layer_idx}, {kind}, K={K}: the grouped rows differ "
                                                f"from the unflagged call, which is the same launch here")
                            else:
                                ones = [self.moe_launch(mlp, y[j:j + 1].clone(), sel[j:j + 1].clone(),
                                                        w[j:j + 1].clone(), False) for j in range(K)]
                                self.same_rows(f"moe experts L{b.layer_idx}, {kind}, {K * topk} slots", rows, ones)
                                if any(not torch.equal(stock[j], ones[j].reshape(-1)) for j in range(K)):
                                    wide_differs.append(K)
                if blackwell:
                    print(f" -- exact rows: MoE L{b.layer_idx} on {device} ({experts} experts): from 32 slots the "
                          f"unflagged call differs from the one-row calls at K = {sorted(set(wide_differs)) or 'NO K'}; "
                          f"the grouped rows equal them at every K", flush = True)

    @torch.inference_mode()
    def test_native_router(self):
        # routing_ds3_nogroup_rows against one-row routing_ds3_nogroup calls, in both forms of the
        # projection: a launch per row, and (an extension that reports CAP_ROUTER_ONE_LAUNCH) one
        # launch of the FMA GEMV for the rows, which the extension counts (exact_rows_served).
        # Without the transposed gate a one-row call is not the GEMV: the entry point must then
        # launch per row whatever it is asked. On random rows and on the rows of a real forward
        self.need(CAP_ROUTER)
        from exllamav3.modules.block_sparse_mlp_routing import _gate_t, _esb_h, ROUTING_ACT_SQRTSP
        self.captured_linear_inputs()
        one_launch = bool(self.caps & CAP_ROUTER_ONE_LAUNCH)
        forms = (False, True) if one_launch else (False,)
        for device in self.devices:
            b = self.first(device, lambda blk: True)
            cfg = b.mlp.routing_cfg
            _gate_t(cfg)
            self.assertIsNotNone(cfg.gate_tensor_t, f"L{b.layer_idx}: the router has no transposed gate")
            E, topk = cfg.num_experts, cfg.num_experts_per_tok
            route = lambda fn, z, logits, sel, w, gate_t, *form: fn(
                z, cfg.gate_tensor, logits, _esb_h(cfg), sel, w, cfg.routed_scaling_factor, gate_t,
                ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb, *form)
            real = next((x for (dev, _), x in sorted(self.real_moe.items()) if dev == str(device)), None)
            for K in ROWS:
                inputs = [("random rows", self.randn((K, b.mlp.hidden_size), device))]
                if real is not None:
                    inputs.append(("real activations", real.reshape(-1, b.mlp.hidden_size)[:K].clone()))
                for kind, z in inputs:
                    for gate_t in (cfg.gate_tensor_t, None):
                        ones = []
                        for j in range(K):
                            # into the bsz1 buffers, as a decode step routes
                            route(self.ext.routing_ds3_nogroup, z[j:j + 1].clone(), cfg.router_logits_bsz1,
                                  cfg.selected_experts_bsz1, cfg.routing_weights_bsz1, gate_t)
                            ones.append((cfg.router_logits_bsz1.clone(), cfg.selected_experts_bsz1.clone(),
                                         cfg.routing_weights_bsz1.clone()))
                        for form in forms:
                            with self.subTest(layer = b.layer_idx, device = str(device), K = K, input = kind,
                                              gate_t = gate_t is not None, one_launch = form):
                                logits = torch.empty((K, E), dtype = torch.half, device = device)
                                sel = torch.empty((K, topk), dtype = torch.long, device = device)
                                w = torch.empty((K, topk), dtype = torch.half, device = device)
                                call = lambda: route(self.ext.routing_ds3_nogroup_rows, z, logits, sel, w, gate_t,
                                                     *((True,) if form else ()))
                                if one_launch:
                                    _, grew = self.served(CAP_ROUTER_ONE_LAUNCH, call)
                                    self.assertEqual(grew, 1 if form and gate_t is not None else 0,
                                                     f"router L{b.layer_idx}: projections that were one launch")
                                else:
                                    call()
                                for j in range(K):
                                    for name, many, one in zip(("logits", "experts", "weights"), (logits, sel, w), ones[j]):
                                        self.assertTrue(torch.equal(many[j], one[0]),
                                                        f"router L{b.layer_idx} {name}, {kind}: row {j} differs")

    @torch.inference_mode()
    def test_native_equals_python_loop_hyper_connection(self):
        # The per-row forms of the two torch sums: the extension's entry points against the Python
        # loops. On this model a flagged call takes the sums over all rows (exact_rows_sum), so
        # that gate is shut here: the per-row forms stay the reference and the fallback
        self.need(CAP_HC)
        m_block = importlib.import_module("exllamav3.modules.dsv41_block")
        with mock.patch.object(m_block, "exact_rows_sum", lambda terms, width: False):
            self.hyper_connection_per_row_forms()

    def hyper_connection_per_row_forms(self):
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
        if self.caps & CAP_HGEMM:
            run = lambda a, w = None: ext.hgemm_rows(a, half(64, 32) if w is None else w, True)
            self.refused("hgemm_rows: 1 row", lambda: run(half(1, 64)))
            self.refused("hgemm_rows: 9 rows", lambda: run(half(9, 64)))
            self.refused("hgemm_rows: 3 dimensions", lambda: run(half(1, 3, 64)))
            self.refused("hgemm_rows: non-contiguous", lambda: run(half(3, 128)[:, ::2]))
            self.refused("hgemm_rows: FP32 rows", lambda: run(half(3, 64).float()))
            self.refused("hgemm_rows: FP32 weight", lambda: run(half(3, 64), half(64, 32).float()))
            self.refused("hgemm_rows: width", lambda: run(half(3, 48)))
            self.refused("hgemm_rows: CPU rows", lambda: run(half(3, 64).cpu(), half(64, 32).cpu()))
            self.refused("hgemm_rows: CPU weight", lambda: run(half(3, 64), half(64, 32).cpu()))
            self.refused("hgemm_rows: non-contiguous weight", lambda: run(half(3, 64), half(64, 64)[:, ::2]))
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
