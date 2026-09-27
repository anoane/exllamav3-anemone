"""
DSV4State.rollback_capacity() must only report rewinds the recurrent rings can serve in place, without
loading a model or allocating GPU memory. The ring bookkeeping of DSAttention._forward_cached_one (SWA ring
append/shift/rebase, compressor ring store) is replayed on position tags, and every forward checks that the
rows it reads are the ones for its positions.
"""

import os
import random
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.cache.dsa import DSV4State
from exllamav3.constants import PAGE_SIZE

WINDOW = 128


class _LayerState:
    window = WINDOW

    def get_checkpoint_size(self):
        return 0

    def clear(self, slot):
        pass

    def stash(self, slot, position):
        return None

    def unstash(self, slot, stashed, position):
        pass


class _Rings:
    """
    Position tags of one layer's SWA ring and compressor sub-window ring, updated like
    DSAttention._forward_cached_one and dsv4_compress: the SWA ring is linear from window_beg (append,
    page shift for small appends near the ring end, window rebase for large chunks), the compressor
    ring is indexed by position % (PAGE_SIZE + m) and stores at most its last PAGE_SIZE + m rows.
    """

    def __init__(self, m):
        self.m = m
        self.ring_rows = -(-(WINDOW + 2 * PAGE_SIZE) // PAGE_SIZE) * PAGE_SIZE
        self.buf_rows = PAGE_SIZE + m
        self.ring = [None] * self.ring_rows
        self.buf = [None] * self.buf_rows

    def forward(self, rs, seq):
        w = WINDOW
        pos0 = rs.position

        # Window rows read from the ring: all w - 1 positions below pos0 (or all history)
        n_prev = min(w - 1, pos0 - rs.window_beg, pos0)
        assert n_prev == min(w - 1, pos0), \
            f"pos {pos0}: window truncated to {n_prev} rows (window_beg {rs.window_beg})"
        for a in range(pos0 - n_prev, pos0):
            assert self.ring[a - rs.window_beg] == a, f"pos {pos0}: SWA ring row for {a} overwritten"

        # Open compressor window read from the ring
        for a in range(pos0 - pos0 % self.m, pos0):
            assert self.buf[a % self.buf_rows] == a, f"pos {pos0}: compressor ring row for {a} overwritten"

        offset = pos0 - rs.window_beg
        pos_end = pos0 + seq
        chunk = list(range(pos0, pos_end))
        if offset + seq <= self.ring_rows:
            self.ring[offset : offset + seq] = chunk
        elif seq < PAGE_SIZE:
            need = offset + seq - self.ring_rows
            shift = -(-need // PAGE_SIZE) * PAGE_SIZE
            self.ring[:self.ring_rows - shift] = self.ring[shift:]
            rs.wshift = shift
            self.ring[offset - shift : offset - shift + seq] = chunk
        else:
            new_beg = max(pos_end - (w - 1), 0) // PAGE_SIZE * PAGE_SIZE
            n_keep = pos_end - new_beg
            n_from_kv = min(n_keep, seq)
            n_from_ring = n_keep - n_from_kv
            if n_from_ring > 0:
                src0 = pos0 - n_from_ring - rs.window_beg
                self.ring[:n_from_ring] = self.ring[src0 : src0 + n_from_ring]
            self.ring[n_from_ring : n_keep] = chunk[seq - n_from_kv:]
            rs.wshift = new_beg - rs.window_beg

        for a in chunk[max(seq - self.buf_rows, 0):]:
            self.buf[a % self.buf_rows] = a


class DSV4RollbackCapacityTest(unittest.TestCase):

    def make_state(self):
        layers = {0: _LayerState()}
        cache = SimpleNamespace(
            get_all_recurrent_layers = lambda: layers,
            model = SimpleNamespace(loaded_tp = False),
        )
        return DSV4State(cache, 0, 0)

    def forward(self, state, rings, seq):
        # Every layer computes the same window shift; advance once, like advance_recurrent_states
        for r in rings:
            r.forward(state, seq)
        state.position += seq
        state.post_advance()

    def test_rewind_after_window_rebase(self):
        # A 2175-token chunk from 0 rebases the window to 2048 and keeps only w - 1 rows
        state = self.make_state()
        rings = [_Rings(4)]
        self.forward(state, rings, 2175)
        self.assertEqual(state.window_beg, 2048)
        self.assertEqual(state.rollback_capacity(), 0)
        # Tokens processed after the rebase can be rewound, and the window stays whole
        for _ in range(10):
            self.forward(state, rings, 1)
        self.assertEqual(state.rollback_capacity(), 10)
        state.rewind(10)
        self.forward(state, rings, 1)

    def test_verify_rebase_then_reject(self):
        # A verify window of 301 tokens near the ring end takes the rebase branch, which keeps only
        # 265 rows; 250 rejected drafts would leave the next query 15 of its 127 window rows
        state = self.make_state()
        rings = [_Rings(4)]
        self.forward(state, rings, 1300)
        for _ in range(200):
            self.forward(state, rings, 1)
        self.forward(state, rings, 301)
        self.assertEqual(state.window_beg, 1536)
        self.assertEqual(state.rollback_capacity(), 265 - (WINDOW - 1))
        cap = state.rollback_capacity()
        state.rewind(cap)
        self.forward(state, rings, 1)

    def test_rewind_below_rejected_drafts(self):
        # Rejected drafts ran past the current position and overwrote compressor ring slots modulo
        # PAGE_SIZE + m: the next rewind is bounded from the furthest position written
        state = self.make_state()
        rings = [_Rings(4), _Rings(128)]
        self.forward(state, rings, 99)
        for _ in range(600):
            self.forward(state, rings, 1)
        self.assertEqual(state.window_beg, 0)
        self.assertLessEqual(state.rollback_capacity(), PAGE_SIZE)
        self.forward(state, rings, 201)
        state.rewind(200)
        self.assertEqual(state.rollback_capacity(), PAGE_SIZE - 200)
        # high_water survives stash/unstash and is cleared by reset
        restored = DSV4State(state.cache, 1, state.position, clear = False, stashed = state.stash())
        self.assertEqual(restored.rollback_capacity(), PAGE_SIZE - 200)
        restored.reset()
        restored.position += 10
        restored.post_advance()
        self.assertEqual(restored.rollback_capacity(), 10)
        state.rewind(PAGE_SIZE - 200)
        self.forward(state, rings, 1)

    def test_guaranteed_rollback_after_prefill(self):
        # Rewinds within PAGE_SIZE of tokens processed since the last chunk stay in place (the banned-string
        # contract), and draft verification with fewer than PAGE_SIZE - 1 drafts never exceeds capacity
        for prompt in (1, 99, 255, 256, 700, 768, 769, 1000, 2175, 4096):
            state = self.make_state()
            rings = [_Rings(4)]
            self.forward(state, rings, prompt)
            for t in range(1, 2 * PAGE_SIZE):
                self.forward(state, rings, 1)
                self.assertGreaterEqual(state.rollback_capacity(), min(t, PAGE_SIZE), f"prompt {prompt}, t {t}")
            for d in (1, 4, 15, 64, 254):
                self.forward(state, rings, d + 1)
                self.assertGreaterEqual(state.rollback_capacity(), d, f"prompt {prompt}, drafts {d}")
                state.rewind(d)

    def test_random_rewinds_within_capacity(self):
        rnd = random.Random(1)
        for trial in range(300):
            state = self.make_state()
            rings = [_Rings(4), _Rings(128)]
            self.forward(state, rings, rnd.choice([1, 17, 255, 256, 700, 769, 1500, 2175, 3000]))
            for _ in range(40):
                op = rnd.random()
                if op < 0.3:
                    seq = rnd.choice([1, 1, 1, 5, 16, 100, 255, 256, 300, 800, 2000])
                    self.forward(state, rings, seq)
                else:
                    d = rnd.choice([1, 4, 15, 100, 254, 255, 300, 500])
                    self.forward(state, rings, d + 1)
                    k = rnd.randint(0, state.rollback_capacity())
                    state.rewind(k)
                    self.forward(state, rings, 1)


if __name__ == "__main__":
    unittest.main()
