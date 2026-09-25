"""
CPU-only tests of the execution mode of RAM-held MoE experts (EXL3_MOE_CPU_MODE, -mcm /
--moe_cpu_mode, config.infer_params.moe_cpu_mode): parsing, configuration, admission at
registration, and dispatch; and of the MoE load hooks under an explicit placement
(EXL3_PLACEMENT). No Torch, CUDA or compiled extension is needed.

    python tests/test_moe_expert_policy_.py

moe_expert_policy.py and placement.py are loaded by path. Everything else runs the real engine source: the
methods under test are extracted from moe_cpu_host.py, block_sparse_mlp_cpu.py, config.py and
model_init.py with ast and executed against fake devices and tensors, with the worker process
and the GPU kernels mocked. This checks dispatch and admission, not CUDA arithmetic, event
ordering, performance or model quality.
"""
import __future__
import argparse
import ast
from contextlib import nullcontext, redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = "exllamav3/model/moe_cpu_host.py"
MIXIN = "exllamav3/modules/block_sparse_mlp_cpu.py"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


policy = load("_moe_policy_subject", "exllamav3/model/moe_expert_policy.py")
placement = load("_placement_subject", "exllamav3/model/placement.py")
ram_budget = load("_ram_budget_subject", "exllamav3/model/ram_budget.py")


class Device:
    def __init__(self, value):
        if isinstance(value, Device):
            self.type, self.index = value.type, value.index
        else:
            parts = value.split(":")
            self.type, self.index = parts[0], int(parts[1]) if len(parts) > 1 else None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


class Tensor:
    def __init__(self, data, device = "cuda:1"):
        self.data = np.asarray(data)
        self.device = Device(device)

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), self.device)

    def __add__(self, value):
        return Tensor(self.data + value, self.device)

    def scatter_add_(self, dim, indices, values):
        assert dim == 0
        np.add.at(self.data, indices.data, values.data)
        return self

    def tolist(self):
        return self.data.tolist()


TORCH = NS(
    device = Device, long = np.int64, float = np.float32,
    zeros = lambda size, dtype, device: Tensor(np.zeros(size, dtype = dtype), device),
    zeros_like = lambda x, dtype: Tensor(np.zeros_like(x.data, dtype = dtype), x.device),
    ones_like = lambda x: Tensor(np.ones_like(x.data), x.device),
    cuda = NS(current_device = lambda: 1, device = lambda _: nullcontext(),
              Stream = lambda **_: object(), Event = lambda **_: object()),
)


def source_node(path, name):
    return next(n for n in ast.parse((ROOT / path).read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def module_assignment(path, name):
    """The value of a module-level assignment in the engine source"""
    node = next(n for n in ast.parse((ROOT / path).read_text()).body
                if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))
    return eval(compile(ast.Expression(node.value), path, "eval"), {"re": re})


def execute(nodes, extra = None):
    ns = dict(vars(policy), torch = TORCH, os = os,
              parse_placement = placement.parse, expert_plan = placement.expert_plan,
              as_expert_ram = ram_budget.as_expert_ram, as_ngram_ram = ram_budget.as_ngram_ram,
              ngram_ram_from_env = ram_budget.ngram_ram_from_env,
              _LAYER_KEY = module_assignment(MIXIN, "_LAYER_KEY"),
              TUNING = NS(stream_debug = False, stream_fused_t = 8, stream_deterministic = False,
                          fused_prefill = False),
              STABLE_ARITHMETIC = False,
              _split_fused = False, _split_prof = False)
    ns.update(extra or {})
    tree = ast.fix_missing_locations(ast.Module(body = nodes, type_ignores = []))
    exec(compile(tree, "<engine-source>", "exec", flags = __future__.annotations.compiler_flag), ns)
    return ns


def source_class(path, name, methods = None):
    node = source_node(path, name)
    node.bases, node.decorator_list = [], []
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        for method in node.body:
            method.decorator_list = []
    return execute([node])[name]


InferParams = source_class("exllamav3/model/config.py", "InferParams")
Host = source_class(HOST, "MoeCpuHost", {
    "gpu_only", "_stream_only", "_require_cpu_compute", "register_layer",
    "submit", "submit_issue", "submit_issue_fused", "_issue_compute", "submit_prefill",
    "_ensure_stream_state", "prefill_worst_case_parts",
})
Mixin = source_class(MIXIN, "BlockSparseMLP_CPU", {
    "placement_plan", "cpu_expert_mode", "_cpu_eligible", "cpu_maybe_offload_load", "cpu_maybe_split_load",
    "cpu_split_submit", "can_defer_load",
})


def make_host():
    h = Host()
    h.specs, h.aux, h.by_key, h.sstate = [], {}, {}, {}
    h.live_layers, h.wslot_size, h.num_wslots = 0, 4096, 2
    h.started, h.stream_min_rows, h.cap_rows = False, 128, 256
    h._spawn, h.conn = Mock(), NS(send = Mock())
    return h


def register(h, key = "layer", **kwargs):
    opts = dict(proj_dims = {"g": (16, 16, 2), "u": (16, 16, 2), "d": (16, 16, 2)},
                aux = {"gpu_tensors": True}, device = "cuda:1", mode = "stream")
    opts.update(kwargs)
    return h.register_layer(key, ["g"] * 4, ["u"] * 4, ["d"] * 4, 0, 0.0, 16, 16, 2, **opts)


class ParserTests(unittest.TestCase):

    def test_cpu_modes(self):
        for value in policy.CPU_MODES:
            self.assertEqual(policy.parse_cpu_mode(value), value)
        for value in (None, True, [], "", "stream", "hybrid", "COMPUTE", "offload_only", "storage_only",
                      "stream-only", " compute", "stream_only "):
            with self.subTest(value = value), self.assertRaisesRegex(ValueError, "compute, stream_only"):
                policy.parse_cpu_mode(value)

    def test_execution_mode(self):
        self.assertEqual(policy.execution_mode("compute"), "hybrid")
        self.assertEqual(policy.execution_mode("stream_only"), "stream")
        # every accepted value has an execution mode of its own, and only those
        self.assertEqual(set(policy._EXECUTION), set(policy.CPU_MODES))
        with self.assertRaises(ValueError):
            policy.execution_mode("stream")

    def test_streamed_experts(self):
        # hybrid: the upstream count threshold, unchanged (a threshold of 0 streams every expert)
        self.assertEqual(policy.streamed_experts([0, 1, 8, 2], 8, False), [2])
        self.assertEqual(policy.streamed_experts([0, 1, 8, 2], 0, False), [0, 1, 2, 3])
        # stream: every selected expert whatever the threshold, never an unselected one
        self.assertEqual(policy.streamed_experts([0, 1, 8, 2], 999, True), [1, 2, 3])
        self.assertEqual(policy.streamed_experts([0, 0], 1, True), [])

    def test_validate_streaming(self):
        good = dict(proj_dims = {"u": 1}, expert_bytes = 192)
        policy.validate_streaming(good, True, 192, 1)
        for spec_, aux, slot, slots in ((dict(good, proj_dims = None), True, 4096, 2),
                                        (good, False, 4096, 2),
                                        (dict(good, expert_bytes = None), True, 4096, 2),
                                        (dict(good, expert_bytes = 0), True, 4096, 2),
                                        (good, True, 191, 2),
                                        (good, True, 4096, 0)):
            with self.subTest(spec = spec_, aux = aux, slot = slot, slots = slots), self.assertRaises(RuntimeError):
                policy.validate_streaming(spec_, aux, slot, slots)


class ConfigurationTests(unittest.TestCase):

    def test_environment(self):
        with patch.dict(os.environ, {}, clear = True):
            self.assertEqual(InferParams().moe_cpu_mode, "compute")
        with patch.dict(os.environ, {"EXL3_MOE_CPU_MODE": "stream_only"}, clear = True):
            self.assertEqual(InferParams().moe_cpu_mode, "stream_only")
        # a bad value fails when the Config is created, not at the first forward
        for value in ("typo", "offload_only", "Stream_only"):
            with patch.dict(os.environ, {"EXL3_MOE_CPU_MODE": value}, clear = True), \
                    self.assertRaisesRegex(ValueError, "moe_cpu_mode"):
                InferParams()

    def test_cli(self):
        add = execute([source_node("exllamav3/model_init.py", "add_args")])["add_args"]
        parser = argparse.ArgumentParser()
        add(parser)
        self.assertIsNone(parser.parse_args(["-m", "x"]).moe_cpu_mode)
        for flag in ("-mcm", "--moe_cpu_mode"):
            for value in policy.CPU_MODES:
                self.assertEqual(parser.parse_args(["-m", "x", flag, value]).moe_cpu_mode, value)
        for value in ("typo", "offload_only", "hybrid"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["-m", "x", "-mcm", value])

    def test_cli_overrides_environment_and_reaches_the_draft(self):
        # Run the real start of model_init.init, up to the tensor overrides
        node = source_node("exllamav3/model_init.py", "init")
        end = next(i for i, n in enumerate(node.body)
                   if isinstance(n, ast.If) and ast.unparse(n.test) == "args.override")
        node.body = node.body[:end] + ast.parse("return config, draft_config").body
        factory = NS(from_directory = lambda *a, **k: NS(infer_params = InferParams()))
        init = execute([node], {"Config": factory, "Path": Path})["init"]
        for cli, expected in ((None, "compute"), ("stream_only", "stream_only")):
            with patch.dict(os.environ, {"EXL3_MOE_CPU_MODE": "compute"}, clear = True):
                args = argparse.Namespace(model_dir = "main", draft_model_dir = "draft", layer_map = None,
                                          moe_cpu_mode = cli)
                main, draft = init(args)
            self.assertEqual(main.infer_params.moe_cpu_mode, expected)
            self.assertEqual(draft.infer_params.moe_cpu_mode, expected)


class AdmissionTests(unittest.TestCase):

    def test_registration_records_device_and_mode(self):
        h = make_host()
        self.assertEqual(register(h), 0)
        self.assertEqual(register(h, key = "other", mode = "stream", device = "cuda"), 1)
        self.assertEqual([(s["device_index"], s["mode"]) for s in h.specs], [(1, "stream"), (1, "stream")])
        self.assertEqual(h.specs[0]["expert_bytes"], 192)
        self.assertTrue(h.gpu_only(0))
        h._spawn.assert_called()

    def test_hybrid_registration_is_unchanged(self):
        h = make_host()
        register(h, mode = "hybrid", device = None, proj_dims = None, aux = None)
        self.assertEqual((h.specs[0]["device_index"], h.specs[0]["mode"]), (None, "hybrid"))
        self.assertFalse(h.gpu_only(0))
        register(h, key = "b", mode = "hybrid")
        self.assertEqual(h.specs[1]["device_index"], 1)

    def test_stream_admission_fails_before_the_worker_starts(self):
        for opts in ({"aux": None}, {"proj_dims": None}, {"device": None}, {"device": "cpu"}):
            h = make_host()
            with self.subTest(opts = opts), self.assertRaises(RuntimeError):
                register(h, **opts)
            h._spawn.assert_not_called()
        for field, value in (("wslot_size", 191), ("num_wslots", 0)):
            h = make_host()
            setattr(h, field, value)
            with self.subTest(field = field), self.assertRaisesRegex(RuntimeError, "EXL3_MOE_CPU_WSLOT"):
                register(h)
            h._spawn.assert_not_called()

    def test_unknown_mode(self):
        h = make_host()
        for mode in ("resident", "compute", None):
            with self.subTest(mode = mode), self.assertRaises(ValueError):
                register(h, mode = mode)
        h._spawn.assert_not_called()

    def test_one_mode_per_device(self):
        h = make_host()
        register(h, key = "a", mode = "stream", device = "cuda:1")
        register(h, key = "b", mode = "hybrid", device = "cuda:0")
        with self.assertRaisesRegex(RuntimeError, "cuda:1"):
            register(h, key = "c", mode = "hybrid", device = "cuda:1")
        # an autosplit retry may move a layer, but not into a device of the other mode
        with self.assertRaisesRegex(RuntimeError, "cuda:0"):
            register(h, key = "a", mode = "stream", device = "cuda:0")
        self.assertEqual(register(h, key = "a", mode = "stream", device = "cuda:1"), 0)

    def test_retry_rebinds_device_and_aux(self):
        h = make_host()
        register(h)
        new_aux = {"gpu_tensors_on_0": True}
        self.assertEqual(register(h, device = "cuda:0", aux = new_aux), 0)
        self.assertIs(h.aux[0], new_aux)
        self.assertEqual(h.specs[0]["device_index"], 0)
        self.assertTrue(h._stream_only(0, Device("cuda:0")))
        with self.assertRaisesRegex(RuntimeError, "registered on cuda:0"):
            h._stream_only(0, Device("cuda:1"))
        with self.assertRaisesRegex(RuntimeError, "layout"):
            register(h, device = "cuda:0", proj_dims = {"u": (16, 32, 2), "d": (32, 16, 2)})
        h._spawn.assert_called_once()

    def test_cpu_job_entry_points_refuse_stream_layers_first(self):
        h = make_host()
        register(h)
        y = NS(device = Device("cuda:1"))      # no shape or data: the guard must come first
        for method, args in (("submit", (0, y, None, None)),
                             ("submit_issue", (0, y, None, None)),
                             ("submit_issue_fused", (0, y, None, None, None, None, 0)),
                             ("_issue_compute", (0, y, None, None, {}, None, 16))):
            with self.subTest(method = method), self.assertRaisesRegex(RuntimeError, "GPU-only"):
                getattr(h, method)(*args)


class DispatchTests(unittest.TestCase):

    def host(self, mode = "stream"):
        h = make_host()
        register(h, mode = mode)
        h._ensure_stream_state = Mock(return_value = {"stream_t": 999})
        h._submit_prefill_streamed = Mock(return_value = "gpu")
        # the real submit stays in place, so an unintended CPU fallback hits its guard
        return h

    def test_one_row_streams_every_selected_expert(self):
        h = self.host()
        self.assertEqual(h.submit_prefill(0, Tensor([[1.0] * 16]), Tensor([[0, 3]]), None), "gpu")
        self.assertEqual(h._submit_prefill_streamed.call_args.args[5], [0, 3])
        self.assertEqual(h._ensure_stream_state.call_args.args[1:], (True,))

    def test_many_rows_stream_cold_experts_too(self):
        h = self.host()
        h.submit_prefill(0, Tensor(np.ones((256, 16))), Tensor([[i % 4, -1] for i in range(256)]), None)
        self.assertEqual(h._submit_prefill_streamed.call_args.args[5], [0, 1, 2, 3])

    def test_no_selected_ram_expert_adds_zero_without_a_job(self):
        h = self.host()
        out = h.submit_prefill(0, Tensor([[1.0] * 16]), Tensor([[-1, -1]]), None)
        np.testing.assert_array_equal(out.data, np.zeros((1, 16), dtype = np.float32))
        self.assertEqual(out.data.dtype, np.float32)
        h._submit_prefill_streamed.assert_not_called()

    def test_wrong_device_is_refused(self):
        h = self.host()
        with self.assertRaisesRegex(RuntimeError, "registered on cuda:1"):
            h.submit_prefill(0, Tensor([[1.0] * 16], "cuda:0"), Tensor([[0, 1]], "cuda:0"), None)
        h._submit_prefill_streamed.assert_not_called()

    def test_hybrid_small_calls_and_cold_experts_stay_on_the_cpu(self):
        for rows in (1, 256):
            h = self.host(mode = "hybrid")
            h.submit = Mock(return_value = "cpu")
            self.assertEqual(h.submit_prefill(0, Tensor(np.ones((rows, 16))), Tensor([[0, 1]] * rows), None), "cpu")
            h.submit.assert_called_once()
            h._submit_prefill_streamed.assert_not_called()

    def test_split_decode_streams_only_in_stream_mode(self):
        for mode in ("stream", "hybrid"):
            m = Mixin()
            m.cpu_host = make_host()
            register(m.cpu_host, mode = mode)
            m.cpu_host.submit_prefill = Mock(return_value = "gpu")
            m.cpu_host.submit_issue = Mock(return_value = "pending")
            m._split_map, m.cpu_layer_idx = None, 0
            m._split_translate = Mock(return_value = "translated")
            result = m.cpu_split_submit(NS(device = "cuda:1"), 1, "raw", "weights")
            if mode == "stream":
                self.assertEqual(result, ("gpu", None))
                m.cpu_host.submit_issue.assert_not_called()
            else:
                self.assertEqual(result, (None, "pending"))
                m.cpu_host.submit_prefill.assert_not_called()

    def test_stream_state_skips_the_bandwidth_probe(self):
        h = make_host()
        h._device_buffers = Mock(return_value = {k: None for k in (
            "vram_slots", "native_slots", "swz", "w_scratch", "fused_bufs", "recon")})
        st = h._ensure_stream_state("cuda:1", True)
        self.assertEqual((st["stream_t"], st["bw"]), (1, None))
        self.assertIs(h._ensure_stream_state("cuda:1", True), st)
        h._device_buffers.assert_called_once()

    def test_small_calls_budget_the_streamed_workspace(self):
        totals = []
        for mode in ("hybrid", "stream"):
            h = make_host()
            register(h, mode = mode)
            h._device_buffers = Mock(return_value = {})
            h._stream_fused_t = Mock(return_value = 0)
            h._stream_recon_layer = Mock(return_value = None)
            totals.append(sum(h.prefill_worst_case_parts(0, 1, Device("cuda:1"), 2)))
        self.assertGreater(totals[1], totals[0])

    def test_every_cpu_job_writer_is_guarded(self):
        # Structural complement of the guard tests: every method that publishes a compute job
        # into the worker's ring checks the layer's mode first, and the streamed path builds a
        # CPU tail only for hybrid layers, after refusing a stream-mode call that would need one
        node = source_node(HOST, "MoeCpuHost")
        writers = []
        for method in (n for n in node.body if isinstance(n, ast.FunctionDef)):
            text = ast.unparse(method)
            if "job[5] = 0" in text or "job[5] = 2" in text:
                writers.append(method.name)
                body = [s for s in method.body if not isinstance(s, ast.Expr) or
                        not isinstance(s.value, ast.Constant)]
                self.assertEqual(ast.unparse(body[0]), "self._require_cpu_compute(layer_idx)", method.name)
        self.assertEqual(set(writers), {"submit_issue_fused", "_issue_compute"})
        fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "_submit_prefill_streamed")
        text = ast.unparse(fn)
        # The stream-mode check precedes any work, and the only CPU job it could issue (the CPU
        # tail) is built on the hybrid branch alone
        self.assertLess(text.index("if stream_mode and sum("), text.index("torch.zeros("))
        tail = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "not stream_mode")
        self.assertIn("self._issue_compute(", ast.unparse(tail))
        self.assertEqual(text.count("self._issue_compute("), 1)


class MixinTests(unittest.TestCase):

    def test_cpu_expert_mode(self):
        m = Mixin()
        for ip, expected in ((NS(), "hybrid"), (NS(moe_cpu_mode = "compute"), "hybrid"),
                             (NS(moe_cpu_mode = "stream_only"), "stream")):
            m.config = NS(infer_params = ip)
            self.assertEqual(m.cpu_expert_mode(), expected)
        m.config = NS(infer_params = NS(moe_cpu_mode = "typo"))
        with self.assertRaises(ValueError):
            m.cpu_expert_mode()

    def test_load_hooks_register_their_mode_and_device(self):
        # load_cpu_offload / load_cpu_split resolve the mode before any side effect (a bad
        # Python value fails before anything loads) and hand it to register_layer with the
        # layer's device
        for name in ("load_cpu_offload", "load_cpu_split"):
            fn = source_node(MIXIN, "BlockSparseMLP_CPU")
            fn = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == name)
            body = [s for s in fn.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
            self.assertEqual(ast.unparse(body[0]), "mode = self.cpu_expert_mode()", name)
            call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == "register_layer")
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            self.assertEqual((kw.get("mode"), kw.get("device")), ("mode", "self.device"), name)


class _Base:
    def can_defer_load(self):
        return True


class Layer(Mixin, _Base):
    """A block-sparse MoE layer as the load hooks see it; the worker registration is mocked"""

    def __init__(self, key = "model.layers.12.mlp", layer_idx = None, activation = "silu", gated = True,
                 num_experts = 64, **ip):
        ip = dict(dict(moe_cpu_offload = 0, moe_cpu_split = 0, draft_moe_cpu_offload = 0,
                       moe_cpu_offload_assigned = {}, moe_cpu_component = "text", moe_cpu_mode = "compute",
                       placement = None), **ip)
        self.config = NS(infer_params = NS(**ip))
        self.key, self.layer_idx, self.activation_fn, self.gated = key, layer_idx, activation, gated
        self.num_experts = self.num_local_experts = num_experts
        self.routing_first, self.cpu_split_first = None, None
        self.load_cpu_offload = Mock(return_value = True)
        self.load_cpu_split = Mock(return_value = True)


class PlacementHookTests(unittest.TestCase):

    def test_placement_plan(self):
        pl = placement.parse("0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-30=cuda:2 experts=split cpu=16; "
                             "*=cuda:2 experts=cpu")
        self.assertIsNone(Layer().placement_plan())
        self.assertEqual(Layer(placement = pl).placement_plan(), ("ram", "stream", 0))
        self.assertEqual(Layer(placement = str(pl)).placement_plan(), ("ram", "stream", 0))   # a string is parsed
        # layer_idx when the module carries it, else the layer in its key
        self.assertEqual(Layer("x", layer_idx = 25, placement = pl).placement_plan(), ("split", "hybrid", 16))
        self.assertEqual(Layer("model.language_model.layers.3.mlp", placement = pl).placement_plan(),
                         ("vram", None, 0))
        self.assertEqual(Layer("model.layers.35.block_sparse_moe", placement = pl).placement_plan(),
                         ("ram", "hybrid", 0))
        with self.assertRaisesRegex(ValueError, "which decoder layer"):
            Layer("mtp.layer.0.mlp", placement = pl).placement_plan()
        # only the text component is placed (an MTP head or vision tower sharing the config is not)
        self.assertIsNone(Layer(placement = pl, moe_cpu_component = "mtp").placement_plan())
        # the placement decides the execution mode, not moe_cpu_mode
        self.assertEqual(Layer(placement = pl).cpu_expert_mode(), "stream")
        self.assertEqual(Layer("model.layers.35.mlp", placement = pl).cpu_expert_mode(), "hybrid")

    def test_whole_layer_claims(self):
        pl = placement.parse("0-11=cuda:0; 12-22=cuda:1 experts=cpu; *=cuda:1 experts=split cpu=16")
        claimed = []
        for i in range(40):
            m = Layer(f"model.layers.{i}.mlp", placement = pl, moe_cpu_offload_assigned = {})
            if m.cpu_maybe_offload_load("cuda:1"):
                claimed.append(i)
                m.load_cpu_offload.assert_called_once_with("cuda:1")
                self.assertEqual(m.config.infer_params.moe_cpu_offload_assigned, {"text": 1})
            else:
                m.load_cpu_offload.assert_not_called()
        self.assertEqual(claimed, list(range(12, 23)))

    def test_split_claims(self):
        pl = placement.parse("0-11=cuda:0; *=cuda:1 experts=split cpu=16")
        m = Layer("model.layers.20.mlp", placement = pl)
        self.assertFalse(m.cpu_maybe_offload_load("cuda:1"))
        m.cpu_maybe_split_load("cuda:1")
        m.load_cpu_split.assert_called_once_with("cuda:1", 16)
        self.assertEqual(m.config.infer_params.moe_cpu_split_assigned, 1)
        m = Layer("model.layers.3.mlp", placement = pl)
        m.cpu_maybe_split_load("cuda:0")
        m.load_cpu_split.assert_not_called()
        with self.assertRaisesRegex(ValueError, "must leave between 1 and 15"):
            Layer("model.layers.20.mlp", placement = pl, num_experts = 16).cpu_maybe_split_load("cuda:1")

    def test_a_placed_layer_that_cannot_offload_is_an_error(self):
        # ineligible layers fall back to VRAM silently with -mcl / -mcs, never under a placement
        ram = placement.parse("*=cuda:0 experts=cpu")
        split = placement.parse("*=cuda:0 experts=split cpu=8")
        for kwargs, device in ((dict(activation = "relu2"), "cuda:0"),          # gated relu2: unsupported
                               (dict(), "cpu"),
                               (dict(), None)):
            with self.subTest(kwargs = kwargs, device = device):
                with self.assertRaisesRegex(RuntimeError, "use experts=vram"):
                    Layer(placement = ram, **kwargs).cpu_maybe_offload_load(device)
                with self.assertRaisesRegex(RuntimeError, "use experts=vram"):
                    Layer(placement = split, **kwargs).cpu_maybe_split_load(device)
        m = Layer(placement = ram)
        m.load_cpu_offload.return_value = False                          # e.g. not mul1
        with self.assertRaisesRegex(RuntimeError, "cannot hold its routed experts"):
            m.cpu_maybe_offload_load("cuda:0")
        m = Layer(placement = split)
        m.num_local_experts = 32                                          # a tensor-parallel shard
        with self.assertRaisesRegex(RuntimeError, "cannot hold 8"):
            m.cpu_maybe_split_load("cuda:0")

    def test_older_settings_unchanged_without_a_placement(self):
        claimed = [i for i in range(4) if Layer(f"model.layers.{i}.mlp", moe_cpu_offload = 2,
                                                moe_cpu_offload_assigned = {"text": min(i, 2)})
                   .cpu_maybe_offload_load("cuda:0")]
        self.assertEqual(claimed, [0, 1])
        # ineligible: no claim, no error
        self.assertFalse(Layer(activation = "relu2", moe_cpu_offload = 2).cpu_maybe_offload_load("cuda:0"))
        m = Layer(moe_cpu_split = 16)
        m.cpu_maybe_split_load("cuda:0")
        m.load_cpu_split.assert_called_once_with("cuda:0", 16)
        m = Layer(moe_cpu_split = 16)
        m.routing_first = 0
        m.cpu_maybe_split_load("cuda:0")
        m.load_cpu_split.assert_not_called()

    def test_one_eligibility_helper(self):
        # the eligible-activation test exists once, in _cpu_eligible, which both claim paths
        # call (the worker's activation ids in load_cpu_offload / load_cpu_split are a mapping)
        cls = source_node(MIXIN, "BlockSparseMLP_CPU")
        methods = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        holders = [name for name, text in methods.items() if "self.activation_fn in (" in text]
        self.assertEqual(holders, ["_cpu_eligible"])
        for name in ("cpu_maybe_offload_load", "cpu_maybe_split_load"):
            self.assertEqual(methods[name].count("self._cpu_eligible(device)"), 2, name)

    def test_placed_split_keeps_router_loads_undeferred(self):
        split = placement.parse("*=cuda:0 experts=split cpu=8")
        with patch.dict(os.environ, {"EXL3_MOE_CPU_SPLIT_STATS": "stats.json"}):
            self.assertFalse(Layer(placement = split).can_defer_load())
            self.assertFalse(Layer(moe_cpu_split = 8).can_defer_load())
            self.assertTrue(Layer(placement = placement.parse("*=cuda:0 experts=cpu")).can_defer_load())
        self.assertTrue(Layer(placement = split).can_defer_load())


if __name__ == "__main__":
    unittest.main(verbosity = 2)
