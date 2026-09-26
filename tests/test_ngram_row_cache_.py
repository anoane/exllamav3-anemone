"""
CPU-only tests of the n-gram / engram row cache and whole tables (exllamav3/modules/ngram_row_cache.py;
doc/expert_tiers.md, "-ngr / --ngram_ram"): a gather through the cache always returns the table's
bytes (hits, misses, rows that collide in one slot, several threads at once), a gather's misses land
in the staging rows and the cache only once their read completed, and a table read whole is the
file's bytes. Then the PLE n-gram module's gather with a row cache against the same gather without
(the extension's ngram_gather_cpu on a scratch file).

    python -m pytest -q tests/test_ngram_row_cache_.py
"""
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exllamav3.modules.ngram_row_cache import RowCache, Landing, CachedTicket, read_whole

ROWS = 200_000


def table(row_bytes, seed):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, size = (ROWS, row_bytes), dtype = np.uint8))


def gather(cache, tables, uids):
    """A gather through the cache: hits from it, misses from the tables (standing in for the disk)"""
    n = uids.numel()
    outs = [torch.zeros((n + 5, t.shape[1]), dtype = torch.uint8) for t in tables]
    mpos, mslots = cache.lookup(uids, outs)
    miss = uids.index_select(0, mpos)
    scratch = [t.index_select(0, miss) for t in tables]
    Landing(cache, miss, mslots, mpos, scratch, outs).land()
    return [o[:n] for o in outs], mpos.numel()


class Cache(unittest.TestCase):

    def test_hits_misses_and_bytes(self):
        tables = [table(256, 1), table(8, 2)]
        cache = RowCache("t", [256, 8], 64 << 20)
        self.assertEqual(cache.slots, (64 << 20) // (256 + 8 + 8))
        rng = np.random.default_rng(3)
        hot = np.unique(rng.integers(0, ROWS, size = 5000))
        for it in range(20):
            ids = np.unique(np.concatenate([hot[rng.integers(0, len(hot), size = 400)],
                                            rng.integers(0, ROWS, size = 200)]))
            uids = torch.from_numpy(ids.astype(np.int64))
            outs, misses = gather(cache, tables, uids)
            for o, t in zip(outs, tables):
                self.assertTrue(torch.equal(o, t.index_select(0, uids)), it)
        s = cache.stats()
        self.assertGreater(s["hits"], 0)
        self.assertEqual(s["bytes"], cache.slots * (256 + 8 + 8))
        # the same rows again: every one a hit but those that lost their slot to another of the set
        _, misses = gather(cache, tables, uids)
        self.assertEqual(misses, uids.numel() - cache.slot_of(uids).unique().numel())

    def test_collisions_never_serve_another_row(self):
        tables = [table(16, 4)]
        cache = RowCache("t", [16], 1024 * 24)          # 1024 slots for 200,000 rows
        self.assertEqual(cache.slots, 1024)
        rng = random.Random(5)
        for it in range(200):
            ids = sorted(set(rng.randrange(ROWS) for _ in range(rng.randrange(1, 3000))))
            uids = torch.tensor(ids, dtype = torch.long)
            outs, _ = gather(cache, tables, uids)
            self.assertTrue(torch.equal(outs[0], tables[0].index_select(0, uids)), it)
        # an insert of rows that collide keeps one row per slot, tag and bytes together
        uids = torch.arange(0, 50_000, dtype = torch.long)
        slots = cache.slot_of(uids)
        cache.insert(uids, slots, [tables[0].index_select(0, uids)])
        tags = cache.tags.clone()
        held = tags[tags >= 0]
        self.assertTrue(torch.equal(cache.rows[0].index_select(0, cache.slot_of(held)), tables[0].index_select(0, held)))

    def test_threads(self):
        tables = [table(64, 6)]
        cache = RowCache("t", [64], 16 << 20)
        errors = []

        def work(seed):
            rng = np.random.default_rng(seed)
            for _ in range(60):
                uids = torch.from_numpy(np.unique(rng.integers(0, 20_000, size = 800)).astype(np.int64))
                outs, _ = gather(cache, tables, uids)
                if not torch.equal(outs[0], tables[0].index_select(0, uids)):
                    errors.append(seed)
        th = [threading.Thread(target = work, args = (s,)) for s in range(6)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        self.assertEqual(errors, [])

    def test_too_small(self):
        with self.assertRaisesRegex(ValueError, "holds 3 rows; too small to help"):
            RowCache("t", [256], 3 * 264)

    def test_resize(self):
        """The end of a load shrinks a cache (ram ngram=auto yields to the expert tier): what it held is
        dropped and gathers stay exact; below 1024 rows it holds nothing and every lookup misses; the
        misses of a lookup made before a resize never land in the resized cache"""
        tables = [table(256, 11), table(8, 12)]
        per = 256 + 8 + 8
        cache = RowCache("t", [256, 8], 8 << 20)
        uids = torch.arange(0, 20_000, 7, dtype = torch.long)
        gather(cache, tables, uids)
        _, misses = gather(cache, tables, uids)
        self.assertLess(misses, uids.numel())
        stale = torch.arange(150_000, 153_000, dtype = torch.long)
        outs = [torch.zeros((stale.numel(), t.shape[1]), dtype = torch.uint8) for t in tables]
        mpos, mslots = cache.lookup(stale, outs)
        self.assertEqual(mpos.numel(), stale.numel())
        cache.resize(1024 * per)
        self.assertEqual((cache.slots, cache.nbytes), (1024, 1024 * per))
        self.assertTrue(bool((cache.tags == -1).all()))
        cache.insert(stale, mslots, [t.index_select(0, stale) for t in tables])     # old geometry: dropped
        self.assertTrue(bool((cache.tags == -1).all()))
        for _ in range(3):
            outs, _ = gather(cache, tables, uids)
            for o, t in zip(outs, tables):
                self.assertTrue(torch.equal(o, t.index_select(0, uids)))
        cache.resize(1000 * per)
        self.assertEqual((cache.slots, cache.nbytes, cache.stats()["bytes"]), (0, 0, 0))
        outs, misses = gather(cache, tables, uids)
        self.assertEqual(misses, uids.numel())
        for o, t in zip(outs, tables):
            self.assertTrue(torch.equal(o, t.index_select(0, uids)))

    def test_landing_on_wait(self):
        """A prefetch's misses land when the consumer waits for its ticket, not before, and only once;
        a released ticket never lands"""
        tables = [table(32, 7)]
        cache = RowCache("t", [32], 1 << 20)
        uids = torch.tensor([3, 17, 40_000], dtype = torch.long)
        out = torch.zeros((3, 32), dtype = torch.uint8)
        mpos, mslots = cache.lookup(uids, [out])
        scratch = [tables[0].index_select(0, uids)]

        class Ticket:
            def __init__(self):
                self.calls = []

            def promote(self, cls):
                self.calls.append(("promote", cls))

            def wait(self):
                self.calls.append(("wait",))

            def release(self):
                self.calls.append(("release",))

        tk = Ticket()
        ct = CachedTicket(tk, Landing(cache, uids, mslots, mpos, scratch, [out]))
        self.assertEqual(int(out.sum()), 0)
        ct.promote(0)
        ct.wait()
        ct.wait()
        ct.release()
        self.assertEqual(tk.calls, [("promote", 0), ("wait",), ("wait",), ("release",)])
        self.assertTrue(torch.equal(out, scratch[0]))
        out2 = torch.zeros_like(out)
        mpos2, _ = cache.lookup(uids, [out2])
        self.assertEqual(mpos2.numel(), 0)
        self.assertTrue(torch.equal(out2, scratch[0]))
        # released without a wait: nothing lands
        cache2 = RowCache("t", [32], 1 << 20)
        mpos, mslots = cache2.lookup(uids, [out2])
        CachedTicket(Ticket(), Landing(cache2, uids, mslots, mpos, scratch, [out2])).release()
        self.assertEqual(cache2.lookup(uids, [out2])[0].numel(), 3)


class Whole(unittest.TestCase):

    def test_read_whole(self):
        from types import SimpleNamespace as NS
        data = np.random.default_rng(8).integers(0, 256, size = 3 * (1 << 20) + 777, dtype = np.uint8)
        with tempfile.NamedTemporaryFile(delete = False) as f:
            f.write(data.tobytes())
        try:
            h = NS(filename = f.name, abs_offset = 777, num_rows = (3 << 20) // 64, row_bytes = 64, key = "t")
            got = read_whole(h, threads = 4, chunk = 1 << 18)
            self.assertEqual(got.shape, (h.num_rows, 64))
            self.assertTrue(np.array_equal(got.numpy().reshape(-1), data[777 : 777 + (3 << 20)]))
            h2 = NS(filename = f.name, abs_offset = 777, num_rows = (4 << 20) // 64, row_bytes = 64, key = "t")
            with self.assertRaisesRegex(OSError, "ended at byte"):
                read_whole(h2)
        finally:
            os.unlink(f.name)


class NGramModule(unittest.TestCase):
    """NGramEmbedding._gather_rows with a row cache returns what the disk gather returns"""

    def test_gather_rows_cached(self):
        from exllamav3.modules.ngram_embedding import NGramEmbedding
        from exllamav3.loader.safetensors import DiskTensorHandle
        rows, words = 50_000, 41
        data = np.random.default_rng(9).integers(-30000, 30000, size = (rows, words), dtype = np.int16)
        with tempfile.NamedTemporaryFile(delete = False) as f:
            f.write(b"\0" * 100)
            f.write(data.tobytes())
        try:
            def module(cache_bytes):
                m = object.__new__(NGramEmbedding)
                m.key = "p.ngram"
                m.tables = None
                m.handles = [DiskTensorHandle("p.ngram.trellis", f.name, 100, [rows, words], torch.int16)]
                m.rows_per_shard = rows
                m.io_direct = -1
                m.row_cache = RowCache("p.ngram", [words * 2], cache_bytes) if cache_bytes else None
                return m
            plain, cached = module(0), module(8 << 20)
            rng = np.random.default_rng(10)
            for it in range(30):
                uids = torch.from_numpy(np.unique(rng.integers(0, 3000 if it % 2 else rows, size = 900)).astype(np.int64))
                a = torch.zeros((uids.numel(), words), dtype = torch.int16)
                b = torch.ones((uids.numel(), words), dtype = torch.int16)
                plain._gather_rows(uids, a)
                cached._gather_rows(uids, b)
                self.assertTrue(torch.equal(a, torch.from_numpy(data[uids.numpy()])), it)
                self.assertTrue(torch.equal(a, b), it)
            s = cached.row_cache.stats()
            self.assertGreater(s["hits"], 0)
            for m in (plain, cached):
                m.handles[0].close()
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main(verbosity = 2)
