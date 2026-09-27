"""
RecurrentCache.put() with checkpoints that do not fit in the whole cache. Evicting could never make room for such a
checkpoint, so put() used to drain every stored checkpoint and then raise KeyError('dictionary is empty') from
popitem(), out of Generator.iterate() at the first recurrent checkpoint (e.g. with recurrent_cache_size = 0). It must
instead skip the checkpoint, without copying it off the device and without touching the other entries.

No GPU or model is required: the cache only needs a state's checkpoint_size and stash().
"""

import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.cache.recurrent import RecurrentCache


class FakeState:
    def __init__(self, checkpoint_size):
        self.checkpoint_size = checkpoint_size
        self.num_stashes = 0

    def stash(self):
        self.num_stashes += 1
        return {"checkpoint_size": self.checkpoint_size}


def fill(max_size, sizes):
    cache = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size)
    for key, size in sizes:
        cache.put(key, FakeState(size))
    return cache


def test_size_zero_stores_nothing():
    cache = fill(0, [])
    state = FakeState(1024)
    cache.put("a", state)
    assert list(cache.keys()) == []
    assert cache.metrics["stash_too_large"] == 1
    assert state.num_stashes == 0, "an unstorable checkpoint should not be copied off the device"


def test_oversize_keeps_existing_entries():
    cache = fill(100, [("a", 40), ("b", 40)])
    state = FakeState(101)
    cache.put("big", state)
    assert list(cache.keys()) == ["a", "b"]
    assert cache.current_size == 80
    assert cache.metrics["stash_evictions"] == 0
    assert cache.metrics["stash_too_large"] == 1
    assert state.num_stashes == 0


def test_warns_once_unless_disabled():
    out = io.StringIO()
    with redirect_stdout(out):
        cache = fill(100, [("a", 101), ("b", 200), ("c", 40)])
    assert list(cache.keys()) == ["c"]
    assert cache.metrics["stash_too_large"] == 2
    assert out.getvalue().count("Warning") == 1, "a cache smaller than one checkpoint should warn once"
    out = io.StringIO()
    with redirect_stdout(out):
        cache = fill(0, [("a", 101), ("b", 200)])
    assert cache.metrics["stash_too_large"] == 2
    assert out.getvalue() == "", "size 0 disables checkpoints by design and should not warn"


def test_lru_eviction_unchanged():
    cache = fill(100, [("a", 40), ("b", 40), ("c", 40)])
    assert list(cache.keys()) == ["b", "c"]
    assert cache.current_size == 80
    assert cache.metrics["stash_evictions"] == 1


def test_exactly_full_is_stored():
    cache = fill(100, [("a", 40), ("b", 100)])
    assert list(cache.keys()) == ["b"]
    assert cache.current_size == 100
    assert cache.metrics["stash_evictions"] == 1


def test_existing_key_moves_to_end():
    cache = fill(100, [("a", 40), ("b", 40)])
    state = FakeState(40)
    cache.put("a", state)
    assert list(cache.keys()) == ["b", "a"]
    assert state.num_stashes == 0


if __name__ == "__main__":
    test_size_zero_stores_nothing()
    test_oversize_keeps_existing_entries()
    test_warns_once_unless_disabled()
    test_lru_eviction_unchanged()
    test_exactly_full_is_stored()
    test_existing_key_moves_to_end()
    print("OK")
