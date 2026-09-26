"""
CPU-only tests of how a GPU's expert cache drives its calls (exllamav3/model/expert_tier.py,
TierDevice; doc/expert_tiers.md, "Prefill: layer mode"), against a stand-in for the native runtime
that records what it is asked: which mode each row count runs in, when a layer-mode pass begins
(after other calls, or back at an earlier layer), that every cache layer is planned once per pass
(at its call, or right after the previous layer's compute) into the staging half its index picks,
that a pass driven from two threads is refused, not interleaved, and that the trellis views of the
load's measuring forward (before the pool and the staging exist) are the sentinel's.

    python -m pytest -q tests/test_expert_tier_device_.py
"""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exllamav3.model.expert_tier import TierDevice, TierLayer
from exllamav3.model.expert_tier_policy import DECODE, ROUTED, LAYER


class Runtime:
    """What the native TierGpu is asked, in order"""

    def __init__(self):
        self.calls = []

    def layer_begin(self):
        self.calls.append(("begin",))

    def layer_plan(self, lc, half):
        self.calls.append(("plan", lc, half))

    def layer_run(self, lc, ids, seq, tokens):
        self.calls.append(("run", lc))

    def layer_after(self, lc):
        self.calls.append(("after", lc))

    def lookup(self, lc, ids, mode, seq, tokens):
        self.calls.append(("lookup", lc, mode))


def device(layers = 4, halves = 2):
    cfg = NS(decode_rows = 8, prefill_rows = 256, staging_factor = halves, verify = False)
    td = TierDevice(NS(cfg = cfg), torch.device("cuda", 0))
    for i in range(layers):
        td.register(object(), 0, 10 + i)
    td.handle = Runtime()
    return td


IDS = torch.zeros((1, 6), dtype = torch.long)


def forward(td, rows):
    """One pass over the cache layers, as the MoE module drives them: resolve, then after the compute"""
    for layer in td.layers:
        td.resolve(layer, IDS, rows)
        td.after_compute(layer)


class Modes(unittest.TestCase):

    def test_row_counts(self):
        td = device()
        self.assertEqual([td.call_mode(r) for r in (1, 8, 9, 255, 256, 4096)],
                         [DECODE, DECODE, ROUTED, ROUTED, LAYER, LAYER])

    def test_layer_passes(self):
        td = device(layers = 3)
        forward(td, 4096)
        forward(td, 4096)                   # the next chunk: back at layer 0, a new pass
        want = [("begin",), ("plan", 0, 0), ("run", 0), ("after", 0), ("plan", 1, 1), ("run", 1), ("after", 1),
                ("plan", 2, 0), ("run", 2), ("after", 2)]
        self.assertEqual(td.handle.calls, want + want)
        # decode calls in between: lookups, never a layer plan; the next prefill begins a pass
        td.handle.calls.clear()
        forward(td, 1)
        forward(td, 100)
        forward(td, 300)
        self.assertEqual(td.handle.calls[:6], [("lookup", 0, DECODE), ("lookup", 1, DECODE), ("lookup", 2, DECODE),
                                               ("lookup", 0, ROUTED), ("lookup", 1, ROUTED), ("lookup", 2, ROUTED)])
        self.assertEqual(td.handle.calls[6:8], [("begin",), ("plan", 0, 0)])

    def test_single_staging(self):
        td = device(layers = 3, halves = 1)
        forward(td, 4096)
        self.assertEqual([c for c in td.handle.calls if c[0] == "plan"], [("plan", 0, 0), ("plan", 1, 0), ("plan", 2, 0)])

    def test_a_pass_from_two_threads_is_refused(self):
        td = device(layers = 4)
        td.resolve(td.layers[0], IDS, 4096)
        td.after_compute(td.layers[0])
        errors = []

        def other():
            try:
                td.resolve(td.layers[1], IDS, 4096)
            except RuntimeError as e:
                errors.append(str(e))
        t = threading.Thread(target = other)
        t.start()
        t.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("called from another thread than the pass's earlier layers", errors[0])
        # the pass is abandoned: the next layer-mode call, from any thread, begins a new one
        td.handle.calls.clear()
        td.resolve(td.layers[1], IDS, 4096)
        self.assertEqual(td.handle.calls, [("begin",), ("plan", 1, 1), ("run", 1)])

    def test_views_before_the_tier_is_built(self):
        """The load's measuring forward runs before the pool and the staging exist: a per-expert dequant
        call then reads the trellis views at the sentinel, which is all there is"""
        td = device(layers = 1)
        td.pool, td.staging = [], None
        td.sentinel = torch.zeros((4096,), dtype = torch.uint8)
        v = td.view_at(td.sentinel.data_ptr() + 64, (4, 8), 64)
        self.assertEqual((v.shape, v.dtype, v.data_ptr()), (torch.Size([4, 8]), torch.int16, td.sentinel.data_ptr() + 64))
        with self.assertRaisesRegex(RuntimeError, "a trellis pointer outside the pool, staging and sentinel"):
            td.view_at(td.sentinel.data_ptr() + 4090, (4, 8), 64)


if __name__ == "__main__":
    unittest.main(verbosity = 2)
