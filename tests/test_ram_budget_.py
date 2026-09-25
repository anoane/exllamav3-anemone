"""
CPU-only tests of the RAM budgets (exllamav3/model/ram_budget.py, doc/expert_tiers.md): the caps
-er / --expert_ram, -der / --draft_expert_ram, -ngr / --ngram_ram [SIZE] and their variables, the
stage-1 plan a load makes before anything is allocated (RAM-held experts from the checkpoint
headers, the n-gram tables a budget holds, one host-memory check), the CPU worker's cumulative
guard, and the flags' way through model_init and InferParams.

    python tests/test_ram_budget_.py

ram_budget.py is torch-free and loaded by path; the plan runs against stand-in modules and a
stand-in checkpoint (headers only, DeepSeek-V4.1-Flash 3.0 bpw shapes). The flag, config and
module cases import exllamav3 (the built extension, no GPU) and fail where it does not import.
"""
import argparse
import ast
from contextlib import redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


RB = load("_ram_budget_subject", "exllamav3/model/ram_budget.py")
GiB, MiB = 1 << 30, 1 << 20
V41_EXPERT = 13_315_584         # 3 x 4,423,680 trellis + fp16 suh / svh, 64-byte aligned


class Checkpoint:
    """Headers only: tensor_file_map and file_headers, as SafetensorsCollection keeps them"""

    def __init__(self):
        self.tensor_file_map, self.file_headers = {}, {"f": {}}
        self.end = 0

    def add(self, key, shape, esize):
        n = esize
        for d in shape:
            n *= d
        self.tensor_file_map[key] = "f"
        self.file_headers["f"][key] = {"shape": list(shape), "data_offsets": [self.end, self.end + n]}
        self.end += n

    def has_tensor(self, key):
        return key in self.tensor_file_map

    def get_tensor_size(self, key):
        b, e = self.file_headers["f"][key]["data_offsets"]
        return e - b

    def add_linear(self, key, k, n, K = 3, mul1 = True, bias = False):
        self.add(key + ".trellis", (k // 16, n // 16, 16 * K), 2)
        self.add(key + ".suh", (k,), 4)
        self.add(key + ".svh", (n,), 4)
        if mul1:
            self.add(key + ".mul1", (), 4)
        if bias:
            self.add(key + ".bias", (n,), 4)


class MoE:
    """A stand-in BlockSparseMLP: what ram_budget reads of one"""

    def __init__(self, stc, key, layer_idx, num_experts = 384, hidden = 5120, inter = 2304, plan = None,
                 activation = "silu", **linear_kw):
        self.key, self.layer_idx, self.num_experts = key, layer_idx, num_experts
        self.gated, self.activation_fn, self.num_local_experts, self.routing_first = True, activation, None, None
        self.gates, self.ups, self.downs, self.modules = [], [], [], []
        self._plan = plan
        for e in range(num_experts):
            for lst, name, k, n in ((self.gates, "w1", hidden, inter), (self.ups, "w3", hidden, inter),
                                    (self.downs, "w2", inter, hidden)):
                lk = f"{key}.experts.{e}.{name}"
                stc.add_linear(lk, k, n, **linear_kw)
                lst.append(NS(key = lk))

    def load_cpu_offload(self, *a, **k):
        raise AssertionError("never called by the plan")

    def placement_plan(self):
        return self._plan


class NGram:
    def __init__(self, key, nbytes):
        self.key, self._n, self.modules = key, nbytes, []

    def table_nbytes(self):
        return self._n


class DSV41Engram:
    def __init__(self, key):
        self.key, self.modules = key, []


def ip(**kw):
    base = dict(expert_ram = None, draft_expert_ram = None, ngram_ram = None, moe_cpu_offload = 0,
                moe_cpu_split = 0, draft_moe_cpu_offload = 0)
    base.update(kw)
    return NS(**base)


def model(modules, stc, component = "text", **ip_kw):
    blocks = [NS(key = f"block.{i}", modules = [m]) for i, m in enumerate(modules)]
    return NS(modules = blocks, component = component, config = NS(stc = stc, infer_params = ip(**ip_kw)))


def memory(avail_gib, swap = 0, total_gib = 160):
    return RB.HostMemory(int(avail_gib * GiB), int(avail_gib * GiB), total_gib * GiB, swap)


def v41(layers, **kw):
    stc = Checkpoint()
    return stc, [MoE(stc, f"model.layers.{i}.mlp", i, **kw) for i in layers]


class BytesTests(unittest.TestCase):

    def test_arena_expert_bytes(self):
        dims = {"g": (5120, 2304, 3), "u": (5120, 2304, 3), "d": (2304, 5120, 3)}
        self.assertEqual(RB.arena_expert_bytes(dims), V41_EXPERT)
        self.assertEqual(RB.arena_expert_bytes(dims, True), V41_EXPERT + 2 * 4608 + 10240)
        self.assertEqual(RB.arena_expert_bytes(dims, {"g": False, "u": False, "d": True}), V41_EXPERT + 10240)
        self.assertEqual(RB.arena_expert_bytes(dict(dims, g = None)), V41_EXPERT - 4_423_680 - 10240 - 4608)
        # 64-byte alignment of the small tensors
        self.assertEqual(RB.arena_expert_bytes({"u": (16, 16, 2), "d": (16, 16, 2)}), 2 * (64 + 64 + 64))
        self.assertEqual(RB.arena_expert_bytes({"u": (48, 16, 2), "d": (16, 48, 2)}), 2 * (192 + 128 + 64))

    def test_headers_agree(self):
        stc, (m,) = v41([0], num_experts = 2)
        self.assertEqual(sum(RB.linear_arena_bytes(stc, l.key) for l in (m.gates[0], m.ups[0], m.downs[0])), V41_EXPERT)
        with self.assertRaisesRegex(ValueError, "x.trellis is not in the checkpoint"):
            RB.linear_arena_bytes(stc, "x")

    def test_eligibility(self):
        stc, (m,) = v41([0], num_experts = 4)
        self.assertTrue(RB.header_eligible(stc, m))
        stc2, (k8,) = v41([0], num_experts = 2, K = 9)
        self.assertFalse(RB.header_eligible(stc2, k8))
        stc3, (nomul,) = v41([0], num_experts = 2, mul1 = False)
        self.assertFalse(RB.header_eligible(stc3, nomul))
        stc4, (act,) = v41([0], num_experts = 2, activation = "tanh")
        self.assertFalse(RB.header_eligible(stc4, act))
        # biases on some experts only: the worker refuses the layer
        stc5, (mixed,) = v41([0], num_experts = 2)
        stc5.add(mixed.ups[1].key + ".bias", (2304,), 2)
        self.assertFalse(RB.header_eligible(stc5, mixed))


class FlagParsingTests(unittest.TestCase):

    def test_expert_ram(self):
        self.assertIsNone(RB.as_expert_ram(None))
        self.assertIsNone(RB.as_expert_ram(" "))
        self.assertEqual(str(RB.as_expert_ram("48GiB")), "48GiB")
        self.assertEqual(str(RB.as_expert_ram("48")), "48GiB")
        self.assertEqual(str(RB.as_expert_ram(96)), "96GiB")
        with self.assertRaisesRegex(ValueError, r"^expert_ram=48G: '48G' is ambiguous"):
            RB.as_expert_ram("48G")
        with self.assertRaisesRegex(ValueError, r"^expert_ram=all: all is not allowed here \(use a size\)$"):
            RB.as_expert_ram("all")

    def test_ngram_ram(self):
        self.assertIsNone(RB.as_ngram_ram(None))
        self.assertEqual(RB.as_ngram_ram(True), RB.ALL)               # the bare flag, as before
        self.assertEqual(RB.as_ngram_ram(False), RB.ZERO)
        self.assertEqual(str(RB.as_ngram_ram("12GiB")), "12GiB")
        self.assertEqual(RB.as_ngram_ram("all"), RB.ALL)
        with self.assertRaisesRegex(ValueError, r"^ngram_ram=auto: auto is not allowed here \(use all or a size\)$"):
            RB.as_ngram_ram("auto")
        with self.assertRaisesRegex(ValueError, r"^ngram_ram=prompt.txt: not a size"):
            RB.as_ngram_ram("prompt.txt")

    def test_environment(self):
        self.assertIsNone(RB.ngram_ram_from_env({}))
        self.assertIsNone(RB.ngram_ram_from_env({"EXL3_NGRAM_STREAM": "1"}))
        self.assertEqual(RB.ngram_ram_from_env({"EXL3_NGRAM_STREAM": "0"}), RB.ALL)
        self.assertEqual(str(RB.ngram_ram_from_env({"EXL3_NGRAM_RAM": "6GiB"})), "6GiB")
        self.assertEqual(RB.ngram_ram_from_env({"EXL3_NGRAM_RAM": "all", "EXL3_NGRAM_STREAM": "0"}), RB.ALL)
        with self.assertRaisesRegex(ValueError, r"^EXL3_NGRAM_STREAM=0 \(every n-gram table in RAM\) contradicts "
                                                r"EXL3_NGRAM_RAM=6GiB; unset EXL3_NGRAM_STREAM$"):
            RB.ngram_ram_from_env({"EXL3_NGRAM_RAM": "6GiB", "EXL3_NGRAM_STREAM": "0"})
        with self.assertRaisesRegex(ValueError, r"^EXL3_NGRAM_RAM=6G: '6G' is ambiguous"):
            RB.ngram_ram_from_env({"EXL3_NGRAM_RAM": "6G"})

    def test_component_cap(self):
        p = ip(expert_ram = "40GiB", draft_expert_ram = "2GiB")
        self.assertEqual([(str(c), f) for c, f in (RB.component_cap(p, "text"), RB.component_cap(p, "mtp"))],
                         [("40GiB", "--expert_ram"), ("2GiB", "--draft_expert_ram")])
        self.assertEqual(RB.component_cap(ip(), "text"), (None, "--expert_ram"))


class PlanTests(unittest.TestCase):
    """plan_component: stage 1 of a load, with DeepSeek-V4.1-Flash 3.0 bpw shapes"""

    def plan(self, mdl, placement = None, avail = 150, reserve_mb = 2048):
        with patch.dict(os.environ, {"EXL3_HOST_MEM_RESERVE_MB": str(reserve_mb)}):
            return RB.plan_component(mdl, placement, memory = memory(avail))

    def test_mcl_counts_the_first_eligible_layers(self):
        stc, moes = v41(range(4), num_experts = 8)
        stc2 = Checkpoint()
        bad = MoE(stc2, "model.layers.9.mlp", 9, num_experts = 8, K = 9)       # K > 8: stays in VRAM
        stc.tensor_file_map.update(stc2.tensor_file_map)
        stc.file_headers["f"].update(stc2.file_headers["f"])
        b = self.plan(model([bad] + moes, stc, moe_cpu_offload = 2))
        self.assertEqual(b.label, "-mcl 2")
        self.assertEqual([l.key for l in b.layers], ["model.layers.0.mlp", "model.layers.1.mlp"])
        self.assertEqual(b.expert_bytes, 2 * 8 * V41_EXPERT)

    def test_mcl_above_the_cap(self):
        stc, moes = v41(range(11))
        with self.assertRaises(ValueError) as cm:
            self.plan(model(moes, stc, moe_cpu_offload = 11, expert_ram = RB.as_expert_ram("40GiB")))
        self.assertEqual(str(cm.exception), "-mcl 11 keeps 52.4 GiB of routed experts in RAM, more than "
                                            "--expert_ram 40GiB; raise the cap or offload fewer layers")
        b = self.plan(model(moes, stc, moe_cpu_offload = 11, expert_ram = RB.as_expert_ram("53GiB")))
        self.assertEqual(b.expert_bytes, 11 * 384 * V41_EXPERT)
        self.assertEqual(b.summary(), " -- RAM budget (text): routed experts 52.4 GiB (-mcl 11: 11 layers), "
                                      "cap 53GiB; 150.0 GiB available")

    def test_mcs_and_split_layers(self):
        stc, moes = v41(range(3), num_experts = 16)
        b = self.plan(model(moes, stc, moe_cpu_split = 12))
        self.assertEqual((b.label, [(l.first, l.count) for l in b.layers]), ("-mcs 12", [(4, 12)] * 3))
        # -mcs larger than a layer is capped to E - 1, as the loader does
        b = self.plan(model(moes, stc, moe_cpu_split = 99))
        self.assertEqual([(l.first, l.count) for l in b.layers], [(1, 15)] * 3)
        with patch.dict(os.environ, {"EXL3_MOE_CPU_SPLIT_LAYERS": "2"}):
            b = self.plan(model(moes, stc, moe_cpu_split = 12))
        self.assertEqual(len(b.layers), 2)
        self.assertEqual(b.expert_bytes, 2 * 12 * V41_EXPERT)

    def test_draft_component(self):
        stc, moes = v41([0], num_experts = 8)
        mdl = model(moes, stc, component = "mtp", draft_moe_cpu_offload = 1, moe_cpu_offload = 5,
                    draft_expert_ram = RB.as_expert_ram("64MiB"))
        with self.assertRaises(ValueError) as cm:
            self.plan(mdl)
        self.assertEqual(str(cm.exception), "-dmcl 1 keeps 102 MiB of routed experts in RAM, more than "
                                            "--draft_expert_ram 64MiB; raise the cap or offload fewer layers")
        # --expert_ram is the main model's: it does not cap the MTP head
        mdl.config.infer_params.draft_expert_ram = None
        mdl.config.infer_params.expert_ram = RB.as_expert_ram("1MiB")
        self.assertEqual(self.plan(mdl).expert_bytes, 8 * V41_EXPERT)

    def test_placement_layers(self):
        stc, moes = v41(range(4), num_experts = 8)
        moes[1]._plan = ("ram", "hybrid", 0)
        moes[2]._plan = ("split", "stream", 3)
        moes[0]._plan = moes[3]._plan = ("vram", None, 0)
        placement = NS(ram = None, disk = None)
        b = self.plan(model(moes, stc, moe_cpu_offload = 4), placement)     # -mcl is refused next to a placement
        self.assertEqual((b.label, [(l.key, l.first, l.count) for l in b.layers]),
                         ("placement", [("model.layers.1.mlp", 0, 8), ("model.layers.2.mlp", 5, 3)]))
        with self.assertRaises(ValueError) as cm:
            self.plan(model(moes, stc, expert_ram = RB.as_expert_ram("100MiB")), placement)
        self.assertEqual(str(cm.exception), "placement: the stream / cpu layers keep 140 MiB of routed experts in RAM, "
                                            "more than --expert_ram 100MiB; raise the cap or offload fewer layers")

    def test_host_check(self):
        stc, moes = v41(range(11))
        with self.assertRaises(RuntimeError) as cm:
            self.plan(model(moes, stc, moe_cpu_offload = 11), avail = 50)
        self.assertEqual(str(cm.exception),
                         "-mcl 11: ram experts=52.4 GiB needs 52.4 GiB of host memory, but 50.0 GiB is available and "
                         "2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: pinned and anonymous memory "
                         "cannot be reclaimed); offload fewer layers, lower --ngram_ram, or free host memory")
        self.plan(model(moes, stc, moe_cpu_offload = 11), avail = 50, reserve_mb = 0)    # 0 disables it
        self.plan(model(moes, stc, moe_cpu_offload = 11), avail = 54.4)

    def test_ngram_budgets(self):
        stc = Checkpoint()
        tables = [NGram("a.ngram", 10 * GiB), NGram("b.ngram", 3 * GiB), NGram("c.ngram", 5 * GiB)]
        cases = ((None, []), (RB.ZERO, []), (RB.ALL, ["a.ngram", "b.ngram", "c.ngram"]),
                 (RB.as_ngram_ram("9GiB"), ["b.ngram", "c.ngram"]), (RB.as_ngram_ram("12GiB"), ["b.ngram", "c.ngram"]),
                 (RB.as_ngram_ram("4GiB"), ["b.ngram"]), (RB.as_ngram_ram("18GiB"), ["a.ngram", "b.ngram", "c.ngram"]))
        for budget, held in cases:
            with self.subTest(budget = str(budget)):
                mdl = model(tables, stc, ngram_ram = budget)
                mdl.config.infer_params.ngram_ram_plan = {"other.ngram": True}     # another component's
                b = self.plan(mdl)
                self.assertEqual(sorted(t.key for t in b.ngram_held), held)
                self.assertEqual(mdl.config.infer_params.ngram_ram_plan,
                                 {"other.ngram": True, **{t.key: t.key in held for t in tables}})
                self.assertEqual(b.ngram_bytes, sum(t._n for t in tables if t.key in held))
        with self.assertRaises(RuntimeError) as cm:
            self.plan(model(tables, stc, ngram_ram = RB.ALL), avail = 16)
        self.assertEqual(str(cm.exception),
                         "--ngram_ram: ngram=18.0 GiB needs 18.0 GiB of host memory, but 16.0 GiB is available and 2.0 "
                         "GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: pinned and anonymous memory cannot be "
                         "reclaimed); offload fewer layers, lower --ngram_ram, or free host memory")
        # other components hold no n-gram tables of the main model
        b = self.plan(model(tables, stc, component = "mtp", ngram_ram = RB.ALL))
        self.assertEqual(b.ngram_held, [])

    def test_engram_note(self):
        stc = Checkpoint()
        b = self.plan(model([DSV41Engram("model.layers.1.engram")], stc, ngram_ram = RB.as_ngram_ram("6GiB")))
        self.assertTrue(b.worth_reporting())
        self.assertIn("engram tables stream from disk (ngram=6GiB holds no DeepSeek-V4.1 engram table)", b.summary())
        self.assertFalse(self.plan(model([DSV41Engram("x")], stc)).worth_reporting())

    def test_nothing_to_report(self):
        stc, moes = v41([0], num_experts = 2)
        b = self.plan(model(moes, stc))
        self.assertEqual((b.layers, b.expert_bytes, b.ngram_tables), ([], 0, []))
        self.assertFalse(b.worth_reporting())


class NgramDecisionTests(unittest.TestCase):

    def test_plan_then_budget(self):
        cfg = NS(infer_params = NS(ngram_ram = RB.as_ngram_ram("4GiB"), ngram_ram_plan = {"a": True, "b": False}))
        self.assertTrue(RB.ngram_in_ram(cfg, "a", 99 * GiB))       # the load's plan decides
        self.assertFalse(RB.ngram_in_ram(cfg, "b", 1))
        self.assertTrue(RB.ngram_in_ram(cfg, "c", 4 * GiB))        # not planned: the budget alone
        self.assertFalse(RB.ngram_in_ram(cfg, "c", 4 * GiB + 1))
        for budget, expected in ((None, False), ("0", False), ("all", True), (True, True), (False, False)):
            cfg = NS(infer_params = NS(ngram_ram = budget))
            self.assertEqual(RB.ngram_in_ram(cfg, "x", 1), expected, budget)
        self.assertFalse(RB.ngram_in_ram(NS(), "x", 1))            # no infer_params: stream


class GuardTests(unittest.TestCase):
    """charge_expert_ram: the worker-side cap, per component, before each registration"""

    DIMS = {"g": (5120, 2304, 3), "u": (5120, 2304, 3), "d": (2304, 5120, 3)}

    def host(self, component = "text", **kw):
        return NS(by_key = {}, component = component, expert_ram_used = 0, config = NS(infer_params = ip(**kw)))

    def test_cumulative(self):
        h = self.host(expert_ram = RB.as_expert_ram("10GiB"))
        RB.charge_expert_ram(h, "layers.0", self.DIMS, False, 384)          # 4.76 GiB
        RB.charge_expert_ram(h, "layers.1", self.DIMS, False, 384)          # 9.52 GiB
        self.assertEqual(h.expert_ram_used, 2 * 384 * V41_EXPERT)
        h.by_key["layers.1"] = 1
        RB.charge_expert_ram(h, "layers.1", self.DIMS, False, 384)          # rollback retry: not again
        with self.assertRaises(RuntimeError) as cm:
            RB.charge_expert_ram(h, "layers.2", self.DIMS, False, 384)
        self.assertEqual(str(cm.exception), "CPU MoE: layers.2 would bring the routed experts held in RAM by this "
                                            "component to 14.3 GiB, more than --expert_ram 10GiB; raise the cap or "
                                            "offload fewer layers")
        self.assertEqual(h.expert_ram_used, 2 * 384 * V41_EXPERT)           # unchanged by the refusal

    def test_components(self):
        h = self.host("mtp", expert_ram = RB.as_expert_ram("1MiB"), draft_expert_ram = RB.as_expert_ram("20MiB"))
        RB.charge_expert_ram(h, "mtp.0", self.DIMS, False, 1)
        with self.assertRaisesRegex(RuntimeError, r"more than --draft_expert_ram 20MiB"):
            RB.charge_expert_ram(h, "mtp.1", self.DIMS, False, 1)
        RB.charge_expert_ram(self.host(), "x", self.DIMS, False, 10 ** 6)    # no cap

    def test_called_before_registration(self):
        # both registration paths charge the budget before the worker is told about the layer
        tree = ast.parse((ROOT / "exllamav3/modules/block_sparse_mlp_cpu.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BlockSparseMLP_CPU")
        for name, count in (("load_cpu_offload", "self.num_experts"), ("load_cpu_split", "split_k")):
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))]
            names = [getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in calls]
            charge = calls[names.index("charge_expert_ram")]
            register = calls[names.index("register_layer")]
            self.assertLess(charge.lineno, register.lineno, name)
            self.assertEqual(ast.unparse(charge.args[-1]), count, name)


class EngineTests(unittest.TestCase):
    """The flags through model_init and InferParams, and the n-gram module's table size (the
    package, no GPU)"""

    def parser(self, draft = False):
        from exllamav3 import model_init
        p = argparse.ArgumentParser()
        model_init.add_args(p, add_draft_model_args = draft)
        return p

    def refused(self, parser, argv):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            parser.parse_args(argv)
        return err.getvalue()

    def test_cli(self):
        p = self.parser(draft = True)
        a = p.parse_args(["-m", "x"])
        self.assertEqual((a.expert_ram, a.ngram_ram, a.draft_expert_ram), (None, None, None))
        a = p.parse_args(["-m", "x", "-er", "48GiB", "-ngr", "-der", "2GiB"])
        self.assertEqual((a.expert_ram, a.ngram_ram, a.draft_expert_ram), ("48GiB", "all", "2GiB"))
        self.assertEqual(p.parse_args(["-m", "x", "--ngram_ram", "12GiB"]).ngram_ram, "12GiB")
        self.assertEqual(p.parse_args(["-m", "x", "--expert_ram", "96"]).expert_ram, "96")
        self.assertIn("expert_ram=48G: '48G' is ambiguous: write 48GiB (2^30 bytes) or 48GB (10^9 bytes)",
                      self.refused(p, ["-m", "x", "-er", "48G"]))
        # a bare -ngr before something that is not a size: the size error, never a silent swallow
        self.assertIn("ngram_ram=prompt.txt: not a size", self.refused(p, ["-m", "x", "-ngr", "prompt.txt"]))
        self.assertIn("draft_expert_ram=all: all is not allowed here", self.refused(p, ["-m", "x", "-der", "all"]))
        self.assertNotIn("draft_expert_ram", vars(self.parser().parse_args(["-m", "x"])))

    def test_infer_params(self):
        from exllamav3.model.config import InferParams
        with patch.dict(os.environ, {}, clear = True):
            p = InferParams()
            self.assertEqual((p.expert_ram, p.draft_expert_ram, p.ngram_ram, p.ngram_stream_from_disk),
                             (None, None, None, True))
            p.expert_ram = "48GiB"
            p.ngram_ram = "6GiB"
            self.assertEqual((str(p.expert_ram), str(p.ngram_ram), p.ngram_stream_from_disk), ("48GiB", "6GiB", True))
            p.ngram_stream_from_disk = False                       # the older switch: every table in RAM
            self.assertTrue(p.ngram_ram.is_all)
            p.ngram_stream_from_disk = True
            self.assertIsNone(p.ngram_ram)
            with self.assertRaisesRegex(ValueError, "expert_ram=48G: '48G' is ambiguous"):
                p.expert_ram = "48G"
        with patch.dict(os.environ, {"EXL3_EXPERT_RAM": "40GiB", "EXL3_NGRAM_STREAM": "0"}, clear = True):
            p = InferParams()
            self.assertEqual((str(p.expert_ram), p.ngram_ram.is_all, p.ngram_stream_from_disk), ("40GiB", True, False))
        with patch.dict(os.environ, {"EXL3_EXPERT_RAM": "40G"}, clear = True), \
                self.assertRaisesRegex(ValueError, "EXL3_EXPERT_RAM=40G: '40G' is ambiguous"):
            InferParams()

    def init(self):
        # The real start of model_init.init, up to the tensor overrides, with a stand-in Config
        from exllamav3.model.config import InferParams
        from exllamav3.model.placement import parse
        tree = ast.parse((ROOT / "exllamav3/model_init.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init")
        end = next(i for i, n in enumerate(node.body)
                   if isinstance(n, ast.If) and ast.unparse(n.test) == "args.override")
        node.body = node.body[:end] + ast.parse("return config, draft_config").body
        ns = dict(Config = NS(from_directory = lambda *a, **k: NS(infer_params = InferParams())),
                  Path = Path, parse_placement = parse)
        exec(compile(ast.fix_missing_locations(ast.Module(body = [node], type_ignores = [])),
                     "<model_init.init>", "exec"), ns)
        return ns["init"]

    def test_init(self):
        init = self.init()
        base = dict(model_dir = "main", layer_map = None, tensor_parallel = False, draft_model_dir = None,
                    expert_ram = None, ngram_ram = None, draft_expert_ram = None)
        with patch.dict(os.environ, {"EXL3_EXPERT_RAM": "10GiB", "EXL3_NGRAM_RAM": "6GiB"}, clear = True):
            main, _ = init(argparse.Namespace(**base))
            self.assertEqual((str(main.infer_params.expert_ram), str(main.infer_params.ngram_ram)), ("10GiB", "6GiB"))
            # the CLI overrides the variables
            main, _ = init(argparse.Namespace(**dict(base, expert_ram = "20GiB", ngram_ram = "all")))
            self.assertEqual((str(main.infer_params.expert_ram), str(main.infer_params.ngram_ram)), ("20GiB", "all"))
            # an older caller's ngram_ram = False (the store_true default) changes nothing
            main, _ = init(argparse.Namespace(**dict(base, ngram_ram = False)))
            self.assertEqual(str(main.infer_params.ngram_ram), "6GiB")
            # a draft model from its own directory: -der is its cap, EXL3_EXPERT_RAM is not
            main, draft = init(argparse.Namespace(**dict(base, draft_model_dir = "draft", draft_expert_ram = "2GiB")))
            self.assertEqual((str(main.infer_params.expert_ram), str(draft.infer_params.expert_ram)), ("10GiB", "2GiB"))
            main, draft = init(argparse.Namespace(**dict(base, draft_model_dir = "draft")))
            self.assertIsNone(draft.infer_params.expert_ram)
            # an MTP head shares the config: -der is its draft_expert_ram
            main, draft = init(argparse.Namespace(**dict(base, mtp = True, draft_expert_ram = "3GiB")))
            self.assertIs(draft, main)
            self.assertEqual((str(main.infer_params.expert_ram), str(main.infer_params.draft_expert_ram)),
                             ("10GiB", "3GiB"))
            with self.assertRaisesRegex(AssertionError, "--draft_expert_ram requires a draft model"):
                init(argparse.Namespace(**dict(base, draft_expert_ram = "3GiB")))

    def test_ngram_table_nbytes(self):
        from exllamav3.modules.ngram_embedding import NGramEmbedding
        stc = Checkpoint()
        for i in range(3):
            stc.add(f"p.ngram.shard_{i}.trellis", (1000, 41), 2)
        m = object.__new__(NGramEmbedding)
        m.key, m.config = "p.ngram", NS(stc = stc)
        self.assertEqual(m.table_nbytes(), 3 * 1000 * 41 * 2)
        stc2 = Checkpoint()
        stc2.add("q.ngram.weight", (500, 160), 2)
        m.key, m.config = "q.ngram", NS(stc = stc2)
        self.assertEqual(m.table_nbytes(), 500 * 160 * 2)
        self.assertTrue(RB.is_ngram(m))

    def test_load_gen_plans_before_loading(self):
        # Model.load_gen makes the plan right after the placement, before any module loads
        tree = ast.parse((ROOT / "exllamav3/model/model.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_gen")
        text = ast.unparse(fn)
        self.assertLess(text.index("self._resolve_placement("), text.index("plan_component(self, placement)"))
        self.assertLess(text.index("plan_component(self, placement)"), text.index("self._load_single("))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
