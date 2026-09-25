"""
CPU tests for tools/dsv41_fit.py on synthetic sizes, no checkpoint: the estimate rejects
geometry and budgets that could manufacture free memory, the output/head phases, and the layout
the tool builds from the loader's settings (explicit placement, -mcl, -mcs) with the loader's own
refusals, and the engine's rule for a DeepSeek-V4.1 placement's changes of device. Stdlib and the tool only: nothing imports torch or the extension.

    python -m pytest tests/test_dsv41_fit_inputs_.py
    python tests/test_dsv41_fit_inputs_.py
"""
import importlib.util
import os
import sys
from types import SimpleNamespace
import unittest

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dsv41_fit", os.path.join(root, "tools", "dsv41_fit.py"))
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)

DEVICES = [("first", 64.0), ("second", 96.0)]


class FitRefusals(unittest.TestCase):
    def test_invalid_geometry_and_budgets_are_refused(self):
        base = dict(max_seq_len = 1048576, chunk = 4096, devices = DEVICES)
        for change in (
            {"max_seq_len": -256}, {"max_seq_len": 0}, {"chunk": -1}, {"chunk": 0},
            {"slots": -1}, {"slots": 0}, {"slots": 0.5}, {"cache_bits": 1}, {"cache_bits": 9},
            {"cache_bits": 2.0}, {"cache_bits": False}, {"margin_gib": -1},
            {"context_gib": [float("nan")]}, {"load_overhead_gib": [1]},
            {"load_overhead_gib": [0, -1, 0]}, {"devices": [("test", 0)]}, {"devices": None},
            {"devices": []}, {"host_ram_gib": float("inf")},
        ):
            with self.subTest(change = change), self.assertRaises(ValueError):
                fit.compute(None, None, **dict(base, **change))


def topology(n = 2, ratios = None, kv = (), ix = ()):
    return fit.P.Topology(n, ratios or [0] * n, list(kv), list(ix))


def synthetic(*, head_format = "exl3", vocab = 129280, hidden = 5120, n = 2, experts = 0):
    topo = topology(n)
    routed = [{e: 100 for e in range(experts)} for _ in range(n)]
    aux = [{e: 10 for e in range(experts)} for _ in range(n)]
    return SimpleNamespace(
        text = {"hidden_size": hidden, "vocab_size": vocab, "hc_mult": 4}, topo = topo,
        head_format = head_format, rest = [1000 * (i + 1) for i in range(n)], routed = routed, aux = aux,
        engram_dev = [0] * n, head = 500, embed = 0,
        routed_total = lambda i: sum(routed[i].values()), aux_total = lambda i: sum(aux[i].values()),
        num_experts = lambda: [len(r) for r in routed],
        model_dir = "synthetic", num_files = 1, num_tensors = 3, total = 3500)


def estimate(*, head_format = "exl3", vocab = 129280, hidden = 5120, split = True, **kw):
    options = dict(max_seq_len = 256, chunk = 4096, context_gib = [0, 0], devices = DEVICES,
                   load_overhead_gib = (0, 0, 0), margin_gib = 0,
                   transient_mib_per_token = 0, index_tile_gib = 0)
    options.update(kw)
    sizes = synthetic(head_format = head_format, vocab = vocab, hidden = hidden)
    layout = fit.build_layout(sizes.topo, sizes.num_experts(), placement = "0=cuda:0; 1=cuda:1" if split else None)
    return fit.compute(sizes, layout, **options)


class FitPhases(unittest.TestCase):
    def test_defaults_match_load_and_pad_head_width(self):
        r = estimate(vocab = 129281)
        self.assertEqual(r["max_output_size"], 32)
        self.assertEqual(r["max_output_factor"], 1)
        self.assertEqual(r["output"]["padded_vocab"], 129408)
        self.assertEqual(r["output"]["bytes"], 32 * 129408 * 2)
        self.assertEqual(r["output"]["head_reconstruct_bytes"], 0)

    def test_output_budget_is_capped_like_load(self):
        r = estimate(chunk = 1)
        self.assertEqual(r["effective_output_rows"], 1)
        self.assertTrue(any("capped" in message for message in r["warnings"]))

    def test_packed_long_head_reconstruction_and_dense_head(self):
        r = estimate(max_output_size = 2048)
        self.assertEqual(r["output"]["bytes"], 529530880)
        self.assertEqual(r["output"]["head_reconstruct_bytes"], 335544320)
        d = estimate(max_output_size = 2048, head_format = "dense")
        self.assertEqual(d["output"]["head_reconstruct_bytes"], 0)
        self.assertEqual(r["devices"][1]["phases"]["head"] - d["devices"][1]["phases"]["head"],
                         335544320 + 2048 * 5120 * 2)

    def test_reconstruction_transform_allowance_covers_dispatch_boundary(self):
        for rows in (144, 145, 1023, 1024):
            with self.subTest(rows = rows):
                r = estimate(max_output_size = rows)
                o = r["output"]
                self.assertEqual(o["head_transform_input_allowance"], rows * 5120 * 2 if rows > 144 else 0)
                self.assertEqual(r["devices"][1]["phases"]["head"], o["head_input_bytes"] + o["bytes"]
                                 + o["head_reconstruct_bytes"] + o["head_transform_input_allowance"])

    def test_unaligned_hidden_keeps_original_and_padded_inputs(self):
        r = estimate(hidden = 5121, max_output_size = 2048)
        o = r["output"]
        self.assertEqual(o["head_input_bytes"], 4096 * 5121 * 2)
        self.assertEqual(o["head_input_padding_bytes"], 2048 * 5248 * 2)
        self.assertEqual(o["head_transform_input_allowance"], 2048 * 5248 * 2)
        self.assertEqual(r["devices"][1]["phases"]["head"], sum(o[k] for k in (
            "bytes", "head_input_bytes", "head_input_padding_bytes", "head_reconstruct_bytes",
            "head_transform_input_allowance")))

    def test_only_head_device_has_output_phase(self):
        a = estimate(max_output_size = 32)
        b = estimate(max_output_size = 4096, max_output_factor = 3)
        self.assertEqual(a["devices"][0], b["devices"][0])
        self.assertGreater(b["devices"][1]["need_gib"], a["devices"][1]["need_gib"])
        self.assertEqual(estimate(split = False)["output"]["device"], 0)

    def test_output_dtype_and_retention_scaling(self):
        half = estimate(max_output_size = 2048)
        bf = estimate(max_output_size = 2048, output_dtype = "bfloat16")
        full = estimate(max_output_size = 2048, output_dtype = "float32", max_output_factor = 3)
        self.assertEqual(bf["output"]["bytes"], half["output"]["bytes"])
        self.assertEqual(full["output"]["bytes"], 2 * half["output"]["bytes"])
        self.assertEqual(full["output"]["retained_bytes"], 6 * half["output"]["bytes"])

    def test_phases_take_maximum_not_sum(self):
        r = estimate(max_output_size = 2048, max_output_factor = 3,
                     transient_mib_per_token = 0.30)
        d = r["devices"][1]
        self.assertAlmostEqual(d["need_gib"], (d["total"] + max(d["phases"].values())) / fit.GiB)
        self.assertEqual(d["phases"]["output_consumer"], r["output"]["retained_bytes"])
        self.assertLess(d["reserve"], sum(d["phases"].values()))

    def test_collector_is_opt_in_and_separate_from_head(self):
        plain = estimate(max_output_size = 2048)
        r = estimate(max_output_size = 2048, validation_logit_rows = 256)
        self.assertEqual(plain["output"]["validation_fp32_matrix_allowance"], 0)
        self.assertEqual(r["output"]["validation_fp32_matrix_allowance"], 3 * 256 * 129280 * 4)
        self.assertEqual(r["devices"][1]["phases"]["head"], plain["devices"][1]["phases"]["head"])
        self.assertEqual(r["devices"][1]["phases"]["output_consumer"],
                         r["output"]["retained_bytes"] + 3 * 256 * 129280 * 4)
        self.assertEqual(estimate(chunk = 16, validation_logit_rows = 256)["output"]["effective_validation_rows"], 16)

    def test_measured_resident_replaces_header_and_historical_overhead(self):
        measured = [2**30, 2**31]
        r = estimate(measured_resident_bytes = measured, load_overhead_gib = (3, 4, 5))
        again = estimate(measured_resident_bytes = measured, load_overhead_gib = (7, 8, 9))
        for i, d in enumerate(r["devices"]):
            self.assertEqual(d["resident_baseline"], measured[i])
            self.assertEqual(d["resident_estimate"], measured[i])
            self.assertAlmostEqual(d["need_gib"], (measured[i] + max(d["phases"].values())) / fit.GiB)
            self.assertEqual(d["need_gib"], again["devices"][i]["need_gib"])
            self.assertEqual(d["total"], estimate()["devices"][i]["total"])
        self.assertTrue(any("not verified" in message for message in r["warnings"]))

    def test_unmeasured_resident_keeps_historical_overhead_once(self):
        r = estimate(load_overhead_gib = (0.29, 0.41, 0.68))
        for d in r["devices"]:
            self.assertAlmostEqual(d["resident_estimate"], d["total"] + 0.29 * fit.GiB)
            self.assertAlmostEqual(d["need_gib"], d["total"] / fit.GiB + 0.29 + max(d["phases"].values()) / fit.GiB)

    def test_malformed_new_inputs(self):
        changes = (
            {"max_output_size": 0}, {"max_output_size": -1}, {"max_output_size": True},
            {"max_output_size": 1.5}, {"max_output_factor": 0}, {"max_output_factor": False},
            {"max_output_factor": 1.5}, {"max_output_factor": 2**63},
            {"output_dtype": "int8"}, {"output_dtype": None},
            {"validation_logit_rows": -1}, {"validation_logit_rows": True},
            {"validation_logit_rows": 1.5}, {"measured_resident_bytes": []},
            {"measured_resident_bytes": [1]}, {"measured_resident_bytes": [1, 2, 3]},
            {"measured_resident_bytes": [1, 0]}, {"measured_resident_bytes": [1, -1]},
            {"measured_resident_bytes": [1, True]}, {"measured_resident_bytes": [1, 1.5]},
            {"measured_resident_bytes": [1, float("nan")]},
            {"measured_resident_bytes": [1, float("inf")]},
            {"measured_resident_bytes": [1, 2**63]}, {"measured_resident_bytes": "1,2"},
            {"vocab": 0}, {"vocab": True}, {"hidden": 0}, {"hidden": True},
            {"hidden": 5120.5}, {"devices": DEVICES[:1]},
        )
        for change in changes:
            with self.subTest(change = change), self.assertRaises(ValueError):
                estimate(**change)

    def test_known_output_bytes_can_exceed_capacity(self):
        r = estimate(max_output_size = 4096, max_output_factor = 4,
                     devices = [("first", 1), ("last", 1)])
        self.assertFalse(r["ok"])
        self.assertFalse(r["devices"][1]["fits"])

    def test_report_is_advisory_in_human_and_json_output(self):
        r = estimate(head_format = "unknown")
        text = fit.report(synthetic(), r)
        self.assertTrue(r["estimate_only"])
        self.assertFalse(r["conservative_bound"])
        self.assertIn("ESTIMATE ONLY", text)
        self.assertIn("estimated fit", text)
        self.assertIn("cuda:1: second", text)
        self.assertTrue(any("Head storage format" in message for message in r["warnings"]))

    def test_measured_snapshot_cannot_be_reused_across_search_placements(self):
        with self.assertRaisesRegex(ValueError, "not search"):
            fit.search(synthetic(), measured_resident_bytes = [2**30, 2**31])


class FitLayouts(unittest.TestCase):
    """The layout the loader builds from each setting, and its refusals."""

    def sizes(self, n = 6, experts = 8):
        return synthetic(n = n, experts = experts)

    def test_no_setting_is_one_device_with_every_expert_resident(self):
        s = self.sizes()
        lay = fit.build_layout(s.topo, s.num_experts())
        self.assertEqual(lay.device, [0] * 6)
        self.assertIsNone(lay.split_at)
        self.assertEqual(lay.cuts, ())
        self.assertEqual({e[0] for e in lay.experts}, {"vram"})
        # the name of the layout without settings is not a word the placement grammar refuses
        self.assertEqual(lay.format(), "cuda:0 only")

    def test_mcl_takes_the_first_layers_in_load_order(self):
        s = self.sizes()
        lay = fit.build_layout(s.topo, s.num_experts(), mcl = 2)
        self.assertEqual([e[0] for e in lay.experts], ["ram", "ram", "vram", "vram", "vram", "vram"])
        self.assertEqual(lay.device, [0] * 6)
        self.assertEqual(lay.format(), "-mcl 2")
        r = fit.compute(s, lay, max_seq_len = 256, chunk = 16, devices = DEVICES)
        self.assertEqual(r["arena_layers"], [0, 1])
        self.assertEqual(r["arena"], 2 * 8 * 100)
        self.assertEqual(r["devices"][0]["cpu"], [0, 1])
        # the GPU keeps the rest of the layer plus the routed experts' suh/svh, the other layers
        # whole, and the head
        self.assertEqual(r["devices"][0]["weights"],
                         (1000 + 80) + (2000 + 80) + sum((i + 1) * 1000 + 8 * 100 for i in range(2, 6)) + s.head)

    def test_mcs_splits_every_layer_capped_at_one_gpu_expert(self):
        s = self.sizes()
        lay = fit.build_layout(s.topo, s.num_experts(), mcs = 100)
        self.assertEqual(lay.experts, [("split", 7, "cpu")] * 6)
        r = fit.compute(s, lay, max_seq_len = 256, chunk = 16, devices = DEVICES)
        self.assertEqual(r["arena_layers"], list(range(6)))
        self.assertEqual(r["arena"], 6 * 7 * 100)
        lay = fit.build_layout(s.topo, s.num_experts(), mcs = 3, mcs_layers = 2)
        self.assertEqual([e[:2] for e in lay.experts], [("split", 3), ("split", 3)] + [("vram", 0)] * 4)
        self.assertIn("first 2 layers", lay.format())

    def test_layers_without_routed_experts_are_not_offloaded(self):
        s = synthetic(n = 3, experts = 0)
        for kw in ({"mcl": 3}, {"mcs": 4}):
            lay = fit.build_layout(s.topo, s.num_experts(), **kw)
            self.assertEqual({e[0] for e in lay.experts}, {"vram"})

    def test_refusals_mirror_the_loader(self):
        s = self.sizes()
        ne = s.num_experts()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            fit.build_layout(s.topo, ne, mcl = 1, mcs = 1)
        for kw in ({"mcl": -1}, {"mcs": True}, {"mcs_layers": 1.5}):
            with self.subTest(kw = kw), self.assertRaises(ValueError):
                fit.build_layout(s.topo, ne, **kw)
        with self.assertRaisesRegex(ValueError, "devices lists 1"):
            fit.compute(s, fit.build_layout(s.topo, ne, placement = "0-1=cuda:0; 2-5=cuda:1"),
                        max_seq_len = 256, chunk = 16, devices = DEVICES[:1])

    @unittest.skipIf(fit.G is None, "no explicit placement in this tree")
    def test_explicit_placement(self):
        s = self.sizes()
        ne = s.num_experts()
        # one kind of host-held experts per GPU, as the grammar requires
        lay = fit.build_layout(s.topo, ne, placement = "0-1=cuda:0 experts=stream; 2-3=cuda:1 experts=cpu; "
                                                       "4=cuda:1 experts=split cpu=3; 5=cuda:1")
        self.assertEqual(lay.device, [0, 0, 1, 1, 1, 1])
        # every layer of this synthetic model is sliding: every change of device is a free cut
        self.assertEqual((lay.cuts, lay.split_at), ((2,), None))
        self.assertEqual(lay.experts, [("ram", 0, "stream")] * 2 + [("ram", 0, "cpu")] * 2 +
                         [("split", 3, "cpu"), ("vram", 0, None)])
        self.assertEqual(lay.head_device, 1)
        r = fit.compute(s, lay, max_seq_len = 256, chunk = 16, devices = DEVICES)
        self.assertEqual(r["arena_layers"], [0, 1, 2, 3, 4])
        self.assertEqual(r["arena"], 4 * 800 + 300)
        self.assertEqual(r["devices"][0]["cpu"], [0, 1])
        self.assertEqual(r["devices"][1]["cpu"], [2, 3])
        self.assertEqual(r["devices"][1]["split"], [4])
        self.assertEqual(r["devices"][1]["resident"], [5])
        # the head may be placed on another device than the last layer
        lay = fit.build_layout(s.topo, ne, placement = "0-2=cuda:0; 3-5=cuda:1; head=cuda:0")
        self.assertEqual(lay.head_device, 0)
        r = fit.compute(s, lay, max_seq_len = 256, chunk = 16, devices = DEVICES)
        self.assertEqual(r["output"]["device"], 0)
        self.assertIn("head", r["devices"][0]["phases"])
        # a reversed device order is taken as written: capacities are by CUDA index
        lay = fit.build_layout(s.topo, ne, placement = "0-2=cuda:1; *=cuda:0")
        self.assertEqual(lay.device, [1, 1, 1, 0, 0, 0])
        self.assertEqual(lay.cuts, (3,))
        r = fit.compute(s, lay, max_seq_len = 256, chunk = 16, devices = DEVICES)
        self.assertEqual([d["index"] for d in r["devices"]], [0, 1])
        self.assertEqual(r["devices"][1]["layers"], [0, 1, 2])

    @unittest.skipIf(fit.G is None, "no explicit placement in this tree")
    def test_explicit_placement_refusals(self):
        s = self.sizes()
        ne = s.num_experts()
        both = "0-2=cuda:0; *=cuda:1"
        for kw, what in (({"mcl": 1}, "EXL3_MOE_CPU_OFFLOAD"), ({"mcs": 2}, "EXL3_MOE_CPU_SPLIT"),
                         ({"mcs_layers": 2}, "EXL3_MOE_CPU_SPLIT_LAYERS"),
                         ({"moe_cpu_mode": "stream_only"}, "EXL3_MOE_CPU_MODE")):
            with self.subTest(kw = kw), self.assertRaisesRegex(ValueError, what):
                fit.build_layout(s.topo, ne, placement = both, **kw)
        # the default mode is no conflict, and the mode alone never refuses the other settings
        fit.build_layout(s.topo, ne, placement = both, moe_cpu_mode = "compute")
        fit.build_layout(s.topo, ne, mcs = 2, moe_cpu_mode = "stream_only")
        # no placement is an empty value; the words the engine refuses are refused here too
        self.assertIsNone(fit.build_layout(s.topo, ne, placement = " ; # none").split_at)
        with self.assertRaisesRegex(ValueError, "no placement is written as an empty value"):
            fit.build_layout(s.topo, ne, placement = "none")
        # the engine's DeepSeek-V4.1 rule on a topology with a kv group (layers 2-5, source 2):
        # changes of device at free cuts (1, 2) are fine, one inside the group is the split with its
        # replica, two inside are refused with the engine's message
        s.topo = topology(6, [0, 0, 2, 2, 2, 2], kv = (2,), ix = (2,))
        lay = fit.build_layout(s.topo, ne, placement = "0=cuda:0; 1=cuda:1; 2-5=cuda:0")
        self.assertEqual((lay.cuts, lay.split_at, lay.replicas), ((1, 2), None, {}))
        lay = fit.build_layout(s.topo, ne, placement = "0-2=cuda:0; 3-5=cuda:1")
        self.assertEqual((lay.cuts, lay.split_at, lay.replicas), ((3,), 3, {3: 2}))
        with self.assertRaisesRegex(ValueError, "changes device in 2 places"):
            fit.build_layout(s.topo, ne, placement = "0-2=cuda:0; 3=cuda:1; 4-5=cuda:0")
        with self.assertRaisesRegex(ValueError, "must leave between 1 and 7"):
            fit.build_layout(s.topo, ne, placement = "*=cuda:0 experts=split cpu=8")
        with self.assertRaisesRegex(ValueError, "not placed"):
            fit.build_layout(s.topo, ne, placement = "0-2=cuda:0")

    @unittest.skipIf(fit.G is None, "no explicit placement in this tree")
    def test_search_rows_are_loadable_placements(self):
        s = self.sizes()
        ne = s.num_experts()
        for split, k in ((1, 0), (2, 3), (3, 3), (5, 1)):
            lay = fit.split_layout(s.topo, ne, split, k)
            again = fit.build_layout(s.topo, ne, placement = lay.format())
            self.assertEqual((again.device, again.experts), (lay.device, lay.experts), lay.format())
            self.assertEqual(again.format(), lay.format())
        rows = fit.search(s, max_seq_len = 256, chunk = 16, devices = DEVICES)
        self.assertTrue(rows)
        for r in rows:
            again = fit.build_layout(s.topo, ne, placement = r["placement"])
            self.assertEqual((list(again.cuts), again.split_at), (r["cuts"], r["split"]))

    def test_tool_stays_off_torch(self):
        self.assertNotIn("torch", vars(fit))
        with open(fit.__file__) as f:
            self.assertNotIn("import torch", f.read())


if __name__ == "__main__":
    unittest.main()
