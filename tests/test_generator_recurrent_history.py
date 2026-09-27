"""Draft length vs. recurrent cache history validation, without loading a model or allocating GPU memory."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.generator.generator import Generator
from exllamav3.modules.gated_delta_net import GDNState
from exllamav3.modules.short_conv import ShortConvState
from exllamav3.modules.sliding_attn import SWAState
from exllamav3.cache.dsa import DSV4State

_torch_empty = torch.empty


def _unpinned_empty(*args, pin_memory = False, **kwargs):
    # Draft buffers are pinned, which needs a CUDA device. Pinning is irrelevant here
    return _torch_empty(*args, **kwargs)


class RecurrentHistoryTest(unittest.TestCase):

    def make_generator(self, state_cls = GDNState, recurrent = True, max_history = 0, draft_model = None, **kwargs):
        model = SimpleNamespace(config = SimpleNamespace(vocab_size = 256), caps = {"recurrent_states": recurrent})
        cache = SimpleNamespace(
            num_slots = 4,
            max_num_tokens = 4096,
            max_history = max_history,
            recurrent_state_cls = state_cls if recurrent else None,
            reset_states = Mock(),
        )
        draft_cache = SimpleNamespace(max_num_tokens = 4096) if draft_model is not None else None
        with patch("exllamav3.generator.generator.PageTable", return_value = SimpleNamespace(max_pages = 16)), \
             patch("exllamav3.generator.generator.ThreadPoolExecutor"), \
             patch("torch.empty", _unpinned_empty):
            return Generator(model, cache, tokenizer = None, draft_model = draft_model, draft_cache = draft_cache,
                             **kwargs)

    def draft_model(self, **caps):
        return SimpleNamespace(caps = {"recurrent_states": False, **caps})

    def test_rejects_short_history(self):
        cases = [
            # (state_cls, max_history, generator kwargs, expected draft length)
            (GDNState, 0, dict(ngram_match_min = 2), 4),
            (GDNState, 4, dict(ngram_match_min = 2, num_draft_tokens = 15), 15),
            (ShortConvState, 0, dict(ngram_match_min = 2), 4),
            (ShortConvState, 3, dict(ngram_match_min = 2, num_draft_tokens = 4), 4),
            (GDNState, 0, dict(draft_model = self.draft_model()), 4),
        ]
        for state_cls, max_history, kwargs, n in cases:
            with self.subTest(state_cls = state_cls, max_history = max_history, kwargs = kwargs):
                with self.assertRaisesRegex(
                    AssertionError,
                    rf"{n} draft tokens on a recurrent model requires Cache\(max_history >= {n}\), "
                    rf"got max_history = {max_history}"
                ):
                    self.make_generator(state_cls = state_cls, max_history = max_history, **kwargs)

    def test_accepts_sufficient_history(self):
        cases = [
            (GDNState, 4, dict(ngram_match_min = 2), 4),
            (GDNState, 15, dict(ngram_match_min = 2, num_draft_tokens = 15), 15),
            (GDNState, 32, dict(ngram_match_min = 2, num_draft_tokens = 15), 15),
            (ShortConvState, 4, dict(ngram_match_min = 2), 4),
            (GDNState, 4, dict(draft_model = self.draft_model()), 4),
        ]
        for state_cls, max_history, kwargs, n in cases:
            with self.subTest(state_cls = state_cls, max_history = max_history, kwargs = kwargs):
                gen = self.make_generator(state_cls = state_cls, max_history = max_history, **kwargs)
                self.assertEqual(gen.num_draft_tokens, n)

    def test_unchanged_without_drafting(self):
        # No drafting needs no history, recurrent or not
        for state_cls in (GDNState, ShortConvState, SWAState, None):
            with self.subTest(state_cls = state_cls):
                gen = self.make_generator(state_cls = state_cls, max_history = 0)
                self.assertEqual(gen.num_draft_tokens, 0)

    def test_unchanged_for_in_place_rollback(self):
        # States that rewind in place (sliding window, DSV4) keep working without history
        for state_cls in (SWAState, DSV4State):
            self.assertTrue(getattr(state_cls, "guaranteed_rollback", 0))
            for kwargs in (dict(ngram_match_min = 2, num_draft_tokens = 15), dict(draft_model = self.draft_model())):
                with self.subTest(state_cls = state_cls, kwargs = kwargs):
                    gen = self.make_generator(state_cls = state_cls, max_history = 0, **kwargs)
                    self.assertGreater(gen.num_draft_tokens, 0)

    def test_unchanged_for_non_recurrent_model(self):
        gen = self.make_generator(recurrent = False, max_history = 0, ngram_match_min = 2, num_draft_tokens = 15)
        self.assertEqual(gen.num_draft_tokens, 15)


if __name__ == "__main__":
    unittest.main()
