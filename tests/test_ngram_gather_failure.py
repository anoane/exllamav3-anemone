"""
NGramEmbedding: a failed staging or upload (a short read from the table file, an OOM on the
uploads) must fail only the call that needs those rows. The staging set it held has to be
released, or once every set is stranded each later forward and prefetch of the module fails with
an IndexError until a reload; and a failed prefetch that no forward takes must not re-raise its
error in a later, unrelated forward, prefetch or unload().

No GPU or model is required: a small unquantized table is written to a temp dir and run on the
CPU in both disk and RAM modes, with failures injected into the row gather and the uploads.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import NGramEmbedding
from exllamav3.modules import ngram_embedding as ne

KEY = "model.layers.0.ple.ngram_embedding"
PARENT = "model.layers.0.ple"
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
NUM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
EOS = 0
VOCAB = 5000
MAX_POSITIONS = 1024


def history(seq):
    # (1, ctx + seq) ids with an eos boundary
    h = torch.randint(1, VOCAB, (1, NGRAM_SIZE - 1 + seq))
    h[0, seq // 3] = EOS
    return h


class NGramGatherFailureTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        sizes = torch.tensor([ne._find_nth_prime_after(1000 + 50 * i, 1) for i in range(NUM_HEADS)])
        offsets = torch.cat([torch.zeros(1, dtype = torch.long), sizes.cumsum(0)[:-1]])
        save_file({
            f"{KEY}.weight": torch.randn(int(sizes.sum()), ne.ROW_DIM).half(),
            f"{PARENT}.ngram_heads_offsets": offsets.int(),
            f"{PARENT}.ngram_heads_vocab_sizes": sizes.int(),
            f"{PARENT}.layer_multipliers": torch.tensor([1, 1000003, 998244353]),
        }, os.path.join(cls.tmp.name, "model.safetensors"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.prefetch_enabled = ne.PREFETCH_ENABLED
        ne.PREFETCH_ENABLED = True
        self.fail = 0
        self.stcs = []

    def tearDown(self):
        ne.PREFETCH_ENABLED = self.prefetch_enabled
        for stc in self.stcs:
            stc.close()

    def make_module(self, stream):
        stc = SafetensorsCollection(self.tmp.name)
        self.stcs.append(stc)
        mod = NGramEmbedding(config = SimpleNamespace(stc = stc), key = KEY, ngram_size = NGRAM_SIZE,
                             heads_per_ngram = HEADS_PER_NGRAM, ple_embed_dim = NUM_HEADS * ne.ROW_DIM,
                             eos_token_id = EOS, stream_from_disk = stream)
        mod.load(torch.device("cpu"))
        # _PinSet.grow pins its buffers, which needs a CUDA device; sets already large enough
        # are kept as they are, so plain ones let the staging path run on the CPU
        n = MAX_POSITIONS * NUM_HEADS
        for _ in range(ne.MAX_PIN_SETS):
            pin = ne._PinSet()
            pin.uids = torch.empty(n, dtype = torch.int64)
            pin.inverse = torch.empty(n, dtype = torch.int64)
            pin.heads = torch.empty(n, dtype = torch.int32)
            pin.packed = torch.empty((n, ne.ROW_DIM), dtype = mod._row_dtype)
            mod._pins.append(pin)

        gather_rows = mod._gather_rows
        def flaky_gather_rows(uids, out):
            if self.fail > 0:
                self.fail -= 1
                raise RuntimeError("ngram_gather_cpu: short read")
            gather_rows(uids, out)
        mod._gather_rows = flaky_gather_rows
        return mod

    def check(self, mod, h):
        self.assertTrue(torch.equal(mod.forward(h, {}), mod.forward_reference(h, {})))

    def assert_released(self, mod):
        self.assertFalse(mod._pending)
        self.assertFalse(any(p.held for p in mod._pins))

    @torch.inference_mode()
    def test_failed_inline_gather(self):
        for stream in (True, False):
            with self.subTest(stream = stream):
                mod = self.make_module(stream)
                decode = [history(1) for _ in range(ne.MAX_PIN_SETS + 1)]
                self.fail = ne.MAX_PIN_SETS
                for h in decode[:ne.MAX_PIN_SETS]:
                    with self.assertRaisesRegex(RuntimeError, "short read"):
                        mod.forward(h, {})
                    self.assert_released(mod)
                for h in decode + [history(300)]:
                    self.check(mod, h)
                x = history(400)
                mod.prefetch(x)
                self.check(mod, x)
                self.assertEqual(mod.prefetch_stats["hit"], 1)
                mod.unload()

    @torch.inference_mode()
    def test_failed_prefetch_taken(self):
        for stream in (True, False):
            with self.subTest(stream = stream):
                mod = self.make_module(stream)
                for _ in range(ne.MAX_PIN_SETS):
                    x = history(300)
                    self.fail = 1
                    mod.prefetch(x)
                    self.assertEqual(len(mod._pending), 1)
                    with self.assertRaisesRegex(RuntimeError, "short read"):
                        mod.forward(x, {})
                    self.assert_released(mod)
                self.assertEqual(mod.prefetch_stats["hit"], ne.MAX_PIN_SETS)
                self.check(mod, history(1))
                x = history(500)
                mod.prefetch(x)
                self.check(mod, x)
                self.assert_released(mod)
                if stream:
                    # a failing open in prefetch() must not strand the set either
                    def fail_open():
                        raise OSError("open failed")
                    mod.handles[0]._ensure_open = fail_open
                    with self.assertRaisesRegex(OSError, "open failed"):
                        mod.prefetch(history(300))
                    del mod.handles[0]._ensure_open
                    self.assert_released(mod)
                    self.check(mod, history(300))
                # nor a failing submit
                mod._executor.shutdown()
                with self.assertRaisesRegex(RuntimeError, "after shutdown"):
                    mod.prefetch(history(300))
                self.assert_released(mod)
                mod._executor = None
                mod.unload()

    @torch.inference_mode()
    def test_failed_upload(self):
        for stream in (True, False):
            with self.subTest(stream = stream):
                mod = self.make_module(stream)
                x = history(300)
                mod.prefetch(x)
                mod._pending[0]["future"].result()
                # staging succeeds, the uploads after it fail
                mod.device = "no_such_device"
                for h in (x, history(1)):
                    with self.assertRaises(RuntimeError):
                        mod.forward(h, {})
                    self.assert_released(mod)
                mod.device = torch.device("cpu")
                self.assertEqual(mod.prefetch_stats["hit"], 1)
                self.check(mod, history(1))
                self.check(mod, x)
                mod.unload()

    @torch.inference_mode()
    def test_failed_prefetch_not_taken(self):
        for stream in (True, False):
            with self.subTest(stream = stream):
                mod = self.make_module(stream)
                a, b, c = history(300), history(400), history(500)
                self.fail = 1
                mod.prefetch(a)
                mod._pending[0]["future"].exception()
                mod.prefetch(b)
                # every set is held by a queued prefetch: retires the failed one for a, whose
                # error belongs to no caller
                mod.prefetch(c)
                self.assertEqual(mod.prefetch_stats["retired"], 1)
                self.check(mod, c)
                self.check(mod, b)
                self.assertEqual(mod.prefetch_stats["hit"], 2)
                self.assert_released(mod)
                # a failed prefetch left behind at unload
                self.fail = 1
                mod.prefetch(a)
                mod._pending[0]["future"].exception()
                mod.unload()
                self.assertIsNone(mod.mode)
                self.assertIsNone(mod.tables)
                self.assertIsNone(mod.handles)
                self.assertIsNone(mod._executor)
                self.assertFalse(mod._pending or mod._pins)


if __name__ == "__main__":
    unittest.main()
