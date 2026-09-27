"""DeepSeek-V4 checkpoint size accounting on CPU, without loading a model or allocating GPU memory."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.cache.dsa import DSV4LayerState, DSV4State
from exllamav3.cache.recurrent import RecurrentCache

POSITIONS = (0, 1, 127, 128, 255, 256, 767, 768, 769, 4096)


def make_module(layer_type, compress_rate = None):
    lin = lambda n: SimpleNamespace(wkv = SimpleNamespace(out_features_unpadded = n))
    return SimpleNamespace(
        layer_type = layer_type,
        head_dim = 64,
        rope_head_dim = 16,
        index_head_dim = 32,
        sliding_window = 128,
        compress_rate = compress_rate,
        compressor = lin(2 * 64),
        indexer = lin(2 * 32),
    )


def make_layer_state(layer_type, compress_rate = None):
    ls = DSV4LayerState(make_module(layer_type, compress_rate), 1, 0, 0)
    ls.alloc(torch.device("cpu"))
    return ls


def stashed_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


class DSV4CheckpointSizeTest(unittest.TestCase):

    layer_types = (("sliding", None), ("csa", 4), ("hca", 128))

    def test_declared_size_bounds_stash(self):
        # RecurrentCache evicts on the declared size, so it must cover every byte stash() copies
        for layer_type, m in self.layer_types:
            ls = make_layer_state(layer_type, m)
            for position in POSITIONS:
                with self.subTest(layer_type = layer_type, position = position):
                    stashed = stashed_bytes(ls.stash(0, position))
                    self.assertLessEqual(stashed, ls.get_checkpoint_size())
                    # Exact once the ring is full, so the bound is not padded
                    if position >= ls.ring_rows:
                        self.assertEqual(stashed, ls.get_checkpoint_size())

    def test_recurrent_cache_stays_within_budget(self):
        layers = {f"l{i}": make_layer_state(t, m) for i, (t, m) in enumerate(self.layer_types)}
        cache = SimpleNamespace(
            model = SimpleNamespace(loaded_tp = False),
            get_all_recurrent_layers = lambda: layers,
        )
        state = DSV4State(cache, 0, 0)
        max_size = 5 * state.checkpoint_size
        rc = RecurrentCache(cache.model, max_size)
        for i in range(12):
            state.position = 4096 + 256 * i
            rc.put(i, state)
            held = sum(stashed_bytes(v[k]) for v in rc.values() for k in layers)
            self.assertLessEqual(held, max_size)


if __name__ == "__main__":
    unittest.main()
