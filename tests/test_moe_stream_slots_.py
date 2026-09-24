"""CPU-only regression tests for streamed deterministic contribution layouts.

This tests production layout and grouping code, not CUDA arithmetic. Hardware
tests must additionally exercise the native slot writer, gather, and DMA reuse.
"""
import ast
import importlib.util
import math
from pathlib import Path
import random
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stream_slots", ROOT / "exllamav3/model/moe_stream_slots.py")
slots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slots)


def production_groups(padding = 1.1, max_rows = 16384):
    tree = ast.parse((ROOT / "exllamav3/modules/moe_batch_recon.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plan_groups")
    ns = dict(PAD_MAX = padding, RECON_ROWS = max_rows, RECON_BATCH = 16)
    exec(compile(ast.Module(body = [node], type_ignores = []), "<production-plan_groups>", "exec"), ns)
    return ns["plan_groups"]


def batches(counts, batch_size, fused, recon_max, cap, grouping):
    active = [e for e, c in enumerate(counts) if c]
    result = []
    for i in range(0, len(active), batch_size):
        batch = active[i:i+batch_size]
        recon = [e for e in batch if fused < counts[e] <= recon_max]
        singles = {e for e in batch if counts[e] > max(fused, recon_max)}
        result.append((batch, grouping(recon, counts.__getitem__, cap), singles))
    return result


class StreamSlotsTests(unittest.TestCase):
    def test_mixed_tiers_and_inactive_expert(self):
        counts = [1, 17, 33, 300, 299, 1100, 0]
        plan = [([0, 1, 2, 3, 4, 5], [[3, 4]], {5})]
        base, kind, rows = slots.slot_layout(counts, plan)
        self.assertEqual(kind, [1, 1, 1, 2, 2, 1, 0])
        self.assertEqual(base[4] - base[3], 300)
        self.assertEqual(rows, sum(counts)+1)

    def test_empty(self):
        self.assertEqual(slots.slot_layout([0, 0], []), ([0, 0], [0, 0], 0))
        self.assertEqual(slots.slot_rows_bound(0, 1.1, 16), 0)

    def test_padding_bound_for_random_actual_group_plans(self):
        rng = random.Random(4204)
        for padding in (0, 0.9, 1, 1.1, 2, 100, math.inf, math.nan):
            grouping = production_groups(padding)
            for _ in range(100):
                counts = [rng.randrange(0, 2000) for _ in range(32)]
                cap, batch_size = rng.randrange(1, 17), rng.randrange(1, 25)
                plan = batches(counts, batch_size, 256, 1024, cap, grouping)
                base, kind, n = slots.slot_layout(counts, plan)
                self.assertLessEqual(n, slots.slot_rows_bound(sum(counts), padding, cap))
                used = set()
                for e, c in enumerate(counts):
                    self.assertEqual(kind[e] == 0, c == 0)
                    these = set(range(base[e], base[e]+c))
                    self.assertFalse(used.intersection(these))
                    self.assertTrue(all(0 <= row < n for row in these))
                    used.update(these)

    def test_gather_assignment_identity_and_batch_partition(self):
        rng = np.random.default_rng(1201)
        # Includes duplicate routes, -1 sentinels, unsorted routes, and wide top-k.
        for rows, topk in ((1, 1), (7, 6), (257, 9), (9, 40)):
            ids = rng.integers(-1, 13, (rows, topk))
            weights = rng.random((rows, topk)).astype(np.float16)
            values = rng.normal(size = (rows, topk, 5)).astype(np.float32)
            flat = ids.reshape(-1)
            order = np.argsort(flat)
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            counts = np.bincount(flat+1, minlength = 14)[1:].tolist()
            offs = np.cumsum([int((flat == -1).sum())]+counts)
            reference = None
            for batch_size in (1, 2, 7, 24):
                plan = batches(counts, batch_size, 16, 1000, 16, production_groups())
                base, kind, n = slots.slot_layout(counts, plan)
                scratch = np.full((n, 5), np.nan, np.float32)
                for a, e in enumerate(flat):
                    if e < 0:
                        continue
                    value = values.reshape(-1, 5)[a]
                    if kind[e] == 1:
                        value = value * np.float32(weights.reshape(-1)[a])
                    scratch[base[e]+inv[a]-offs[e]] = value
                result = np.zeros((rows, 5), np.float32)
                for r in range(rows):
                    for k in range(topk):
                        e = ids[r, k]
                        if e < 0:
                            continue
                        a = r*topk+k
                        value = scratch[base[e]+inv[a]-offs[e]]
                        if kind[e] == 2:
                            value = value * np.float32(weights[r, k])
                        result[r] += value
                self.assertTrue(np.isfinite(result).all())
                if reference is None:
                    reference = result
                else:
                    np.testing.assert_array_equal(result, reference)


if __name__ == "__main__":
    unittest.main()
