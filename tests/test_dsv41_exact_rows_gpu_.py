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
and the compressor's pooling and norm at rates 1 and 2. A control checks, on every GPU, that
the comparison discriminates: an unflagged call of several rows must differ from the one-row
calls.

Not covered: the index selection and the attention kernel, which need real pools, and the order
of these operations in a forward. tools/dsv41_rowprobe.py under EXL3_EXACT_ROWS=1 is the evidence
for those (idx.scores, idx.topk, attn.dsa_attn and the forward's logits in its tables).

    EXL3_EXACT_ROWS=1 DSV41_MODEL_DIR=<checkpoint> python -m pytest tests/test_dsv41_exact_rows_gpu_.py

Needs CUDA, the compiled extension and the checkpoint; skipped without the switch. Loads the whole
model: run it alone on the host.
"""
import os
import unittest

import torch

ROWS = range(2, 9)
FLAGGED = {"exact_rows": True}


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
        print(f" -- exact rows: {n} layers on {', '.join(str(d) for d in cls.devices)}", flush = True)

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
        ops = (
            # name, the Linear of a block or None, input dims, out_dtype the engine passes
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
        # comparisons of this file show nothing there (an environment that turns the int8
        # activation path off, say)
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


if __name__ == "__main__":
    unittest.main()
