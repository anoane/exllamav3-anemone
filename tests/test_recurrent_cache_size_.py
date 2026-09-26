"""RecurrentCache.put with a checkpoint larger than the whole cache.

A Generator built with recurrent_cache_size = 0 (tabby: sysmem_recurrent_cache: 0), or with a size below one
checkpoint, used to raise KeyError('dictionary is empty') at the first checkpoint: put() evicted every entry and
then popped from the empty cache. The checkpoint is now dropped and counted, and the cache keeps its other entries.

Run: python -m pytest -q tests/test_recurrent_cache_size_.py  or  python tests/test_recurrent_cache_size_.py
"""
from types import SimpleNamespace
from exllamav3.cache.recurrent import RecurrentCache


class _State:
    def __init__(self, size):
        self.size = size

    def stash(self):
        return {"checkpoint_size": self.size}


def _cache(max_size):
    return RecurrentCache(SimpleNamespace(loaded_tp = False), max_size)


def test_size_zero_drops_every_checkpoint():
    cache = _cache(0)
    cache.put("a", _State(1024))
    cache.put("b", _State(1))
    assert len(cache) == 0
    assert cache.current_size == 0
    assert cache.metrics["stash_too_large"] == 2


def test_checkpoint_larger_than_cache_keeps_the_others():
    cache = _cache(100)
    cache.put("a", _State(40))
    cache.put("b", _State(40))
    cache.put("big", _State(101))
    assert list(cache.keys()) == ["a", "b"]
    assert cache.current_size == 80
    assert cache.metrics["stash_too_large"] == 1
    assert cache.metrics["stash_evictions"] == 0


def test_eviction_still_makes_room():
    cache = _cache(100)
    cache.put("a", _State(40))
    cache.put("b", _State(40))
    cache.put("c", _State(40))
    assert list(cache.keys()) == ["b", "c"]
    assert cache.current_size == 80
    assert cache.metrics["stash_evictions"] == 1
    assert cache.metrics["stash_too_large"] == 0


def test_exactly_full_fits():
    cache = _cache(100)
    cache.put("a", _State(100))
    assert list(cache.keys()) == ["a"]
    assert cache.metrics["stash_too_large"] == 0


if __name__ == "__main__":
    test_size_zero_drops_every_checkpoint()
    test_checkpoint_larger_than_cache_keeps_the_others()
    test_eviction_still_makes_room()
    test_exactly_full_fits()
    print("  -- recurrent cache size: OK")
