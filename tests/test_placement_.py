"""
CPU-only tests of the explicit placement (EXL3_PLACEMENT, -placement / --placement,
config.infer_params.placement; exllamav3/model/placement.py): grammar, canonical form, no
placement (its one representation and the words refused in its place), refusals, layer coverage, per-layer expert plans, conflicts with the older settings, the
loader's module targets, the real autosplit loop under a placement, Model._resolve_placement's
load-time checks, and the controls.

    python tests/test_placement_.py

model/placement.py is loaded by path. The loader, config and model_init cases (LoaderTests)
import exllamav3 (the built extension, no GPU); where the package does not import, each of them
fails, naming the test and the cause. Nothing in this file is skipped.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_placement_subject", ROOT / "exllamav3/model/placement.py")
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)

try:
    import exllamav3
    _import_error = None
except Exception as e:
    # kept for LoaderTests, which fail with it: a broken import must not pass as "nothing to test"
    _import_error = e


class GrammarTests(unittest.TestCase):

    def test_canonical_form(self):
        s = P.parse("12-22 = CUDA:1 experts=stream ;0-11=cuda:0\n23-39=cuda:1 # tail\nhead=cuda:1")
        self.assertEqual(str(s), "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1; head=cuda:1")
        self.assertEqual(P.parse(str(s)), s)
        self.assertEqual(P.parse("3,1-2=cuda:0 # a; b = c\n4=cuda:0"), P.parse("1-3=cuda:0; 4=cuda:0"))
        # whitespace around a comma of a layer list is ignored (spaces or tabs), none inside a range
        for spaced in ("0-3, 8-11", "0-3 ,8-11", "0-3\t,\t8-11", " 0-3 , 8-11 "):
            self.assertEqual(P.parse(f"{spaced}=cuda:0; *=cuda:1"), P.parse("0-3,8-11=cuda:0; *=cuda:1"), spaced)
        with self.assertRaisesRegex(ValueError, "must look like"):
            P.parse("0 - 3=cuda:0")
        h = P.parse("*=cuda:0; 30-39=cuda:1 experts=split cpu=64; embed=cuda:1")
        self.assertEqual(str(h), "embed=cuda:1; 30-39=cuda:1 experts=split cpu=64; *=cuda:0")
        self.assertIs(P.parse(h), h)
        self.assertEqual(h.cuda_indices(), [0, 1])
        self.assertEqual((h.embed_device(), h.head_device()), ("cuda:1", None))

    def test_no_placement(self):
        # one representation, None, for every spelling of "no placement"
        for value in (None, "", "  ", "# nothing\n ;", ";;", " ; # later"):
            self.assertIsNone(P.parse(value), value)
        # words that might read as "no placement" are refused, never taken as it (or as a layout)
        for word in ("none", "off", "auto", "default", "0", "false", "NONE", " None ", "none;", "off # x",
                     "Auto\n"):
            with self.subTest(word = word):
                with self.assertRaises(ValueError) as cm:
                    P.parse(word)
                e = str(cm.exception)
                self.assertIn(f"placement {word!r}: no placement is written as an empty value, not "
                              f"{word.split('#')[0].strip(' ;').strip()!r}", e)
                self.assertIn("leave EXL3_PLACEMENT unset or empty, pass --placement \"\", or set "
                              "config.infer_params.placement = None", e)
        # ... but they are only refused as the whole value
        with self.assertRaisesRegex(ValueError, "must look like"):
            P.parse("none=cuda:0")

    def test_refusals(self):
        for bad, why in (
            ("0-3=cuda:0; 3-5=cuda:1", "already placed"),
            ("0-3=gpu:0", "must be cuda:<n>"),
            ("0-3=cpu", "must be cuda:<n>"),
            ("*=cpu", "must be cuda:<n>"),
            ("embed=cpu", "must be cuda:<n>"),
            ("0-3=cuda:x", "must be cuda:<n>"),
            ("0-3=cuda:0 experts=split", "needs cpu="),
            ("0-3=cuda:0 cpu=4", "only applies to experts=split"),
            ("0-3=cuda:0 experts=split cpu=0", "positive integer"),
            ("0-3=cuda:0 experts=split cpu=-1", "positive integer"),
            # digits are ASCII: other scripts' digits (and superscripts) are refused, not read
            ("\u0663=cuda:0", "must look like"),
            ("0-3=cuda:\u0660", "must be cuda:<n>"),
            ("0-3=cuda:0 experts=split cpu=\u0667", "positive integer"),
            ("0-3=cuda:0 experts=split cpu=\u00b2", "positive integer"),
            ("0-3=cuda:0 experts=cpu node=0", "unknown attribute"),
            ("0-3=cuda:0 hot=4", "unknown attribute"),
            ("0-3=cuda:0 experts=hybrid", "must be one of"),
            ("0-3=cuda:0 experts=stream experts=cpu", "given twice"),
            ("embed=cuda:0; embed=cuda:1", "placed twice"),
            ("*=cuda:0; *=cuda:1", "placed twice"),
            ("head=cuda:0 experts=cpu", "applies to decoder layers"),
            ("5-2=cuda:0", "backwards"),
            ("1,1=cuda:0", "listed twice"),
            ("a-b=cuda:0", "must look like"),
            ("0-3", "expected <layers>=cuda:<n>"),
            ("0-3=", "missing device"),
            ("0-3=cuda:0 experts", "key=value"),
            ("0-3=cuda:0 experts=stream; 4-7=cuda:0 experts=cpu", "one kind of host-held experts per GPU"),
            ("0-3=cuda:0 experts=split cpu=8; *=cuda:0 experts=stream", "one kind of host-held experts per GPU"),
        ):
            with self.subTest(bad = bad), self.assertRaisesRegex(ValueError, why):
                P.parse(bad)
        with self.assertRaisesRegex(ValueError, "must be a string"):
            P.parse(12)
        # different kinds on different GPUs, or stream next to vram, are fine
        P.parse("0-3=cuda:0 experts=stream; 4-7=cuda:1 experts=cpu; 8-9=cuda:1 experts=split cpu=2; *=cuda:0")

    def test_layer_coverage(self):
        h = P.parse("*=cuda:0; 30-39=cuda:1")
        layers = h.layer_map(range(40))
        self.assertEqual((layers[29].device, layers[30].device), ("cuda:0", "cuda:1"))
        with self.assertRaisesRegex(ValueError, "layer 10 is not placed"):
            P.parse("0-9=cuda:0").layer_map(range(12))
        with self.assertRaisesRegex(ValueError, "does not have: 40"):
            P.parse("0-40=cuda:0").layer_map(range(40))
        with self.assertRaisesRegex(ValueError, "no decoder layers"):
            P.parse("*=cuda:0").layer_map([])

    def test_every_mode_is_mapped(self):
        # A mode added to EXPERT_MODES without its entries must fail, not borrow another's meaning
        self.assertEqual(set(P._HOST_KIND), set(P.EXPERT_MODES))
        self.assertEqual(set(P._PLAN), set(P.EXPERT_MODES))

    def test_expert_plans(self):
        x = P.parse("0-9=cuda:0; 10-19=cuda:1 experts=stream; 20-29=cuda:2 experts=cpu; "
                    "30-39=cuda:2 experts=split cpu=48")
        self.assertEqual([P.expert_plan(x, i, 64) for i in (0, 10, 20, 30)],
                         [("vram", None, 0), ("ram", "stream", 0), ("ram", "hybrid", 0), ("split", "hybrid", 48)])
        for k in (64, 65):
            with self.assertRaisesRegex(ValueError, "must leave between 1 and 63"):
                P.expert_plan(P.parse(f"*=cuda:0 experts=split cpu={k}"), 3, 64)
        self.assertEqual(P.expert_plan(P.parse("*=cuda:0 experts=split cpu=63"), 3, 64), ("split", "hybrid", 63))

    def test_conflicts(self):
        clean = NS(moe_cpu_offload = 0, moe_cpu_split = 0, moe_cpu_mode = "compute")
        self.assertEqual(P.conflicts(clean, {}), [])
        self.assertEqual(P.conflicts(NS(), {"EXL3_MOE_CPU_SPLIT_LAYERS": "0"}), [])
        old = NS(moe_cpu_offload = 11, moe_cpu_split = 64, moe_cpu_mode = "stream_only")
        got = P.conflicts(old, {"EXL3_MOE_CPU_SPLIT_LAYERS": "4"})
        self.assertEqual(len(got), 4, got)
        for name in ("-mcl", "-mcs", "-mcm", "EXL3_MOE_CPU_SPLIT_LAYERS"):
            self.assertTrue(any(name in g for g in got), name)


class LoaderTests(unittest.TestCase):

    def setUp(self):
        if _import_error is not None:
            raise AssertionError(
                f"test_placement_.py::{self.id().split('.', 1)[-1]}: importing exllamav3 failed "
                f"({type(_import_error).__name__}: {_import_error}), so the loader, config and "
                f"model_init placement code cannot be tested. This test needs the package with its "
                f"compiled extension: build it, or put a prebuilt exllamav3_ext on PYTHONPATH"
            ) from _import_error

    def test_module_targets(self):
        from exllamav3.model.model_ls import placement_targets
        M = lambda li: NS(layer_idx = li)
        mods = [M(None), M(0), M(1), M(None), M(2), M(3), M(None), M(None)]
        two = P.parse("0-1=cuda:1; 2-3=cuda:0")
        # embed/head follow the first/last layer, a module between layers the one before it
        self.assertEqual(placement_targets(mods, two),
                         ["cuda:1", "cuda:1", "cuda:1", "cuda:1", "cuda:0", "cuda:0", "cuda:0", "cuda:0"])
        three = P.parse("0-1=cuda:1; 2=cuda:0; 3=cuda:1; embed=cuda:2; head=cuda:2")
        self.assertEqual(placement_targets(mods, three),
                         ["cuda:2", "cuda:1", "cuda:1", "cuda:1", "cuda:0", "cuda:1", "cuda:2", "cuda:2"])
        # PLE models (e.g. Qwen3.8-Flash-Next) put a per-layer embedding module with layer_idx
        # -(i + 1) right before layer i: it goes where its layer goes, and is not a layer itself
        ple = [M(None), M(None), M(-1), M(0), M(1), M(-3), M(2), M(3), M(None), M(None)]
        self.assertEqual(placement_targets(ple, P.parse("0-1=cuda:0; 2-3=cuda:1")),
                         ["cuda:0"] * 5 + ["cuda:1"] * 5)
        self.assertEqual(placement_targets(ple, P.parse("embed=cuda:1; 0-1=cuda:0; *=cuda:1")),
                         ["cuda:1", "cuda:1", "cuda:0", "cuda:0", "cuda:0"] + ["cuda:1"] * 5)
        self.assertEqual(placement_targets(ple, P.parse("0=cuda:1; *=cuda:0; head=cuda:2")),
                         ["cuda:1"] * 4 + ["cuda:0"] * 4 + ["cuda:2"] * 2)

    def test_autosplit_loop_follows_the_placement(self):
        # The real Model_LSMixin._load_autosplit over stand-in modules: CUDA is stubbed out of it
        # and loads are bookkeeping against a simulated budget per device
        import torch
        import exllamav3.model.model_ls as model_ls
        from exllamav3.model.model_ls import Model_LSMixin

        class Torch:
            def __getattr__(self, k):
                return getattr(torch, k)
            def zeros(self, *shape, dtype = None, device = None, **kw):
                return torch.zeros(*shape, dtype = dtype)
            def device(self, *a, **kw):
                return torch.device("cuda", a[0]) if len(a) == 1 and isinstance(a[0], int) else torch.device(*a, **kw)

        class Module:
            def __init__(self, sim, key, layer_idx = None, size = 1, caps = None):
                self.sim, self.key, self.layer_idx, self.size = sim, key, layer_idx, size
                self.caps, self.device, self.modules = caps or {}, None, []
            def __iter__(self):
                yield self
            def can_defer_load(self):
                return False
            def load(self, device, **kw):
                if device.type == "cuda":
                    if self.sim.used.get(device.index, 0) + self.size > self.sim.budget[device.index]:
                        raise torch.cuda.OutOfMemoryError(f"sim: cuda:{device.index} is full at {self.key}")
                    self.sim.used[device.index] = self.sim.used.get(device.index, 0) + self.size
                self.device = device
            def unload(self):
                if self.device is not None and self.device.type == "cuda":
                    self.sim.used[self.device.index] -= self.size
                self.device = None

        class LS(Model_LSMixin):
            def __init__(self):
                self.caps = {}
            def get_layer_instances(self, layer_idx):
                return [(layer_idx, 0)]
            def __iter__(self):
                return iter([])

        def run(spec, budget):
            sim = NS(used = {}, budget = budget, fractions = [])
            mods = [Module(sim, "embed", caps = {"prefer_cpu": True})] + \
                   [Module(sim, f"layers.{i}", i) for i in range(6)] + [Module(sim, "norm"), Module(sim, "head")]
            cfg = NS(stc = NS(begin_deferred_load = lambda **kw: None, end_deferred_load = lambda: None,
                              abort_deferred_load = lambda: None, close = lambda: None),
                     infer_params = NS(vision_pinned = False))
            with patch.object(model_ls, "torch", Torch()), \
                    patch.object(model_ls, "set_memory_fraction_use", lambda use, i: sim.fractions.append(i) or use), \
                    patch.object(model_ls, "unset_memory_fraction", lambda devs: None), \
                    patch.object(model_ls, "free_mem", lambda: None):
                for _ in LS()._load_autosplit(False, None, [100, 100], [0, 1], 2048, 32, 1, None, False, cfg,
                                              mods, False, 1, {}, True, placement = P.parse(spec)):
                    pass
            return [str(m.device) for m in mods], sim.fractions

        # returning to a device: each budget is set once, every module where its rule says
        devs, fractions = run("0-1=cuda:0; 2-3=cuda:1; 4-5=cuda:0; head=cuda:1", {0: 10, 1: 10})
        self.assertEqual(devs, ["cpu", "cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:0", "cuda:0",
                                "cuda:1", "cuda:1"])
        self.assertEqual(fractions, [0, 1])
        # the autosplit would move layer 2 to the next device; a placement stops with the reason
        with self.assertRaisesRegex(RuntimeError, r"placement: layers\.2 does not fit on cuda:0 .*"
                                                  r"\(sim: cuda:0 is full at layers\.2\)"):
            run("*=cuda:0", {0: 2, 1: 10})

    def test_resolve_placement(self):
        import torch
        from exllamav3.model.model import Model
        ip = NS(placement = "0-1=cuda:0; 2-3=cuda:1 experts=cpu", moe_cpu_offload = 0, moe_cpu_split = 0,
                moe_cpu_mode = "compute")
        model = NS(config = NS(infer_params = ip), modules = [NS(layer_idx = i) for i in (None, 0, 1, 2, 3, None)])
        resolve = lambda *a: Model._resolve_placement(model, *a)
        with patch.object(torch.cuda, "device_count", lambda: 2), patch.dict(os.environ, {}, clear = True):
            pl = resolve(None, False)
            self.assertEqual(str(pl), "0-1=cuda:0; 2-3=cuda:1 experts=cpu")
            self.assertIs(ip.placement, pl)                          # left parsed
            with self.assertRaisesRegex(ValueError, "layer-split load"):
                resolve("cuda:0", False)
            with self.assertRaisesRegex(ValueError, "layer-split load"):
                resolve(None, True)
            ip.moe_cpu_offload = 4
            with self.assertRaisesRegex(ValueError, "cannot be combined with: -mcl"):
                resolve(None, False)
            ip.moe_cpu_offload = 0
            ip.placement = "0-4=cuda:0"
            with self.assertRaisesRegex(ValueError, "does not have: 4"):
                resolve(None, False)
            ip.placement = "*=cuda:2"
            with self.assertRaisesRegex(ValueError, "only 2 CUDA device"):
                resolve(None, False)
            ip.placement = None
            self.assertIsNone(resolve(None, False))
            # a mistyped mode is reported as invalid, not as a setting the placement replaces
            ip.placement, ip.moe_cpu_mode = "*=cuda:0", "stream-only"
            with self.assertRaisesRegex(ValueError, "moe_cpu_mode must be one of"):
                resolve(None, False)
            ip.moe_cpu_mode = "compute"
        # the attribute set to an empty value is no placement too, and a word for it is refused
        with patch.object(torch.cuda, "device_count", lambda: 2), patch.dict(os.environ, {}, clear = True):
            ip.placement = " ; "
            self.assertIsNone(resolve(None, False))
            ip.placement = "off"
            with self.assertRaisesRegex(ValueError, "no placement is written as an empty value"):
                resolve(None, False)
        # PLE modules (negative layer_idx) are not layers to place
        model.modules = [NS(layer_idx = i) for i in (None, -1, 0, 1, -3, 2, 3, None)]
        with patch.object(torch.cuda, "device_count", lambda: 2), patch.dict(os.environ, {}, clear = True):
            ip.placement = "0-1=cuda:0; 2-3=cuda:1"
            self.assertEqual(str(resolve(None, False)), "0-1=cuda:0; 2-3=cuda:1")

    def test_controls(self):
        from exllamav3.model.config import InferParams
        with patch.dict(os.environ, {}, clear = True):
            self.assertIsNone(InferParams().placement)
        with patch.dict(os.environ, {"EXL3_PLACEMENT": "*=cuda:0 # all"}, clear = True):
            self.assertEqual(str(InferParams().placement), "*=cuda:0")
        with patch.dict(os.environ, {"EXL3_PLACEMENT": "0-3=cuda:x"}, clear = True), \
                self.assertRaisesRegex(ValueError, "must be cuda:<n>"):
            InferParams()
        # the variable empty is no placement; a word for it is refused when the Config is created
        with patch.dict(os.environ, {"EXL3_PLACEMENT": ""}, clear = True):
            self.assertIsNone(InferParams().placement)
        with patch.dict(os.environ, {"EXL3_PLACEMENT": "none"}, clear = True), \
                self.assertRaisesRegex(ValueError, "no placement is written as an empty value"):
            InferParams()
        from exllamav3 import model_init
        parser = argparse.ArgumentParser()
        model_init.add_args(parser)
        self.assertIsNone(parser.parse_args(["-m", "x"]).placement)
        for flag in ("-placement", "--placement"):
            self.assertEqual(parser.parse_args(["-m", "x", flag, "*=cuda:0"]).placement, "*=cuda:0")
        self.assertEqual(parser.parse_args(["-m", "x", "--placement", ""]).placement, "")

    def test_model_init(self):
        # The real start of model_init.init, up to the tensor overrides, with a stand-in Config
        import ast
        from exllamav3.model.config import InferParams
        tree = ast.parse((ROOT / "exllamav3/model_init.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init")
        end = next(i for i, n in enumerate(node.body)
                   if isinstance(n, ast.If) and ast.unparse(n.test) == "args.override")
        node.body = node.body[:end] + ast.parse("return config, draft_config").body
        ns = dict(Config = NS(from_directory = lambda *a, **k: NS(infer_params = InferParams())),
                  Path = Path, parse_placement = P.parse)
        exec(compile(ast.fix_missing_locations(ast.Module(body = [node], type_ignores = [])),
                     "<model_init.init>", "exec"), ns)
        init = ns["init"]
        base = dict(model_dir = "main", layer_map = None, tensor_parallel = False,
                    placement = "0-1=cuda:0; *=cuda:1 experts=stream")
        with patch.dict(os.environ, {"EXL3_PLACEMENT": "*=cuda:0"}, clear = True):
            # --placement overrides the variable; a draft from its own directory gets no placement
            main, draft = init(argparse.Namespace(draft_model_dir = "draft", **base))
            self.assertEqual(str(main.infer_params.placement), "0-1=cuda:0; *=cuda:1 experts=stream")
            self.assertIsNone(draft.infer_params.placement)
            # without --placement the variable stays; --placement "" clears it, also next to -tp
            main, _ = init(argparse.Namespace(draft_model_dir = "draft", **dict(base, placement = None)))
            self.assertEqual(str(main.infer_params.placement), "*=cuda:0")
            for tp in (False, True):
                main, _ = init(argparse.Namespace(draft_model_dir = None, **dict(base, placement = "",
                                                                                   tensor_parallel = tp)))
                self.assertIsNone(main.infer_params.placement)
            with self.assertRaisesRegex(ValueError, "no placement is written as an empty value"):
                init(argparse.Namespace(draft_model_dir = None, **dict(base, placement = "off")))
            # an MTP head shares the model's config, placement included
            main, draft = init(argparse.Namespace(draft_model_dir = None, mtp = True, **base))
            self.assertIs(draft, main)
            self.assertEqual(str(main.infer_params.placement), "0-1=cuda:0; *=cuda:1 experts=stream")
            # tensor-parallel loading cannot take a placement
            with self.assertRaisesRegex(AssertionError, "layer-split"):
                init(argparse.Namespace(draft_model_dir = None, **dict(base, tensor_parallel = True)))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
