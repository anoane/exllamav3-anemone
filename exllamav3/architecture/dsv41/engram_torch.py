# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Modified for bounded chunk history; see NOTICE and LICENSE-DEEPSEEK.

"""
Engram hashing for DeepSeek-V4.1, in torch -- the production path.

Derived from DeepSeek's reference inference/engram.py (NgramHashState,
build_compressed_token_map); see NOTICE. Differences are structural only:

  * the reference keeps a full-sequence cache of compressed ids and indexes
    back into it; here each forward is handed a LOOKBACK WINDOW -- the last
    max_ngram_size - 1 compressed ids before the chunk -- so the same code
    serves one-shot prefill, chunked prefill and token-by-token decode.
  * "before the start of the sequence" is an explicit UNK in that window,
    blocking exactly like the reference's (positions < shift) test.

Semantics that are easy to get wrong, all from the reference:
  * hashes run over COMPRESSED ids (" The", "the" and "THE" collapse);
  * blocking is STICKY: once a lookback hits the sequence start or a dead
    token, that shift and every larger one take pad_id;
  * positions 0 .. max_ngram_size-2 are NOT skipped -- they hash with padded
    lookback and engram injects there too.
"""

from __future__ import annotations

import numpy as np
import torch

from .engram import compute_hash_multipliers

DEAD = -1   # takes no part in an n-gram (image spans); blocks lookback
UNK = -2    # before the start of the sequence; blocks lookback


def build_compressed_token_map(tokenizer_json: str) -> tuple[torch.Tensor, int]:
    """
    DeepSeek's build_compressed_token_map on the raw Rust tokenizer.

    The reference takes a transformers tokenizer only to reach .backend_tokenizer
    and len(); both come straight from `tokenizers` here. Returns (int64 lookup
    over the full vocab, compressed vocab size) -- the size matters beyond bounds,
    since every hash multiplier derives from it.
    """
    from tokenizers import Regex, Tokenizer, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = Tokenizer.from_file(tokenizer_json)
    n = backend.get_vocab_size(with_added_tokens = True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * n
    for token_id in range(n):
        text = backend.decode([token_id], skip_special_tokens = False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return torch.tensor(lookup, dtype = torch.long), len(key_to_new)


class EngramHasher:
    """Hash ids for every engram layer, from compressed ids and a lookback window."""

    def __init__(self, layout, token_map: torch.Tensor, compressed_vocab_size: int, pad_token_id: int):
        self.layout = layout
        self.max_ngram = layout.max_ngram_size
        self.ctx = self.max_ngram - 1
        self.token_map = token_map.long().cpu()
        self.pad_id = int(self.token_map[pad_token_id])
        # [n_layers, max_ngram], odd, bounded against int64 overflow (reference)
        self.multipliers = torch.as_tensor(np.asarray(compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, compressed_vocab_size)), dtype = torch.long)
        # [n_layers, max_ngram-1, n_heads] bucket moduli, and their running offsets
        self.primes = torch.tensor(layout.primes, dtype = torch.long)
        flat = self.primes.flatten(1)                                        # [n_layers, 24]
        self.offsets = torch.cumsum(torch.cat(
            [torch.zeros_like(flat[:, :1]), flat[:, :-1]], dim = 1), dim = 1)

    def compress(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> compressed ids; ids outside the text vocab are dead."""
        return to_compressed(ids, self.token_map)

    def hash(self, compressed: torch.Tensor, lookback: torch.Tensor) -> torch.Tensor:
        """
        compressed  [B, L] compressed ids of this chunk (DEAD allowed)
        lookback    [B, ctx] the ctx compressed ids before the chunk, oldest first;
                    UNK where the sequence had not started
        returns     [B, L, n_layers, 24] int64 row ids into each layer's table
        """
        return self.hash_hist(torch.cat([lookback.long(), compressed.long()], dim = 1))

    def hash_hist(self, hist: torch.Tensor, table_index: int | None = None) -> torch.Tensor:
        """
        hist         [B, ctx + L] the chunk's compressed ids behind its ctx-id lookback
        table_index  None: every engram layer, [B, L, n_layers, 24]; else only that
                     layer's [B, L, 24] (the same numbers, without hashing the other table)
        """
        B, L = hist.shape[0], hist.shape[1] - self.ctx
        hist = hist.long()
        mult, primes, offsets = self.multipliers, self.primes, self.offsets
        if table_index is not None:
            sel = slice(table_index, table_index + 1)
            mult, primes, offsets = mult[sel], primes[sel], offsets[sel]
        blocked = torch.zeros((B, L), dtype = torch.bool)
        tokens = []
        for shift in range(self.max_ngram):
            src = hist[:, self.ctx - shift: self.ctx - shift + L]
            blocked = blocked | (src == UNK) | (src == DEAD)
            tokens.append(torch.where(blocked, self.pad_id, src))
        tokens = torch.stack(tokens, dim = -1)                                # [B, L, max_ngram]
        products = tokens.unsqueeze(2) * mult                                 # [B, L, n_layers, max_ngram]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])           # [B, L, n_layers, n_heads]
        out = torch.cat(hashes, dim = -1) + offsets
        return out if table_index is None else out[:, :, 0]


def to_compressed(ids: torch.Tensor, token_map: torch.Tensor) -> torch.Tensor:
    """
    Token ids -> compressed ids (host int64). Ids outside the text vocab -- negative, or
    exllamav3's multimodal alias ids past it -- are DEAD: they block the lookback of every
    n-gram that would span them.
    """
    ids = ids.long().cpu()
    n = token_map.numel()
    ok = (ids >= 0) & (ids < n)
    return torch.where(ok, token_map[ids.clamp(0, n - 1)], DEAD)


def engram_hash_chunk(hist: torch.Tensor, table_index: int, hasher: EngramHasher) -> torch.Tensor:
    """[B, ctx + L] compressed history -> [B, L, 24] row ids into table `table_index`."""
    return hasher.hash_hist(hist, table_index)


def dequant_rows(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Gathered table rows -> fp16 [U, head_dim]: fp8_e4m3 values times 2^(e8m0 - 127) per 32.

    E8M0 scales are decoded exactly, including code 0 (2^-127) and 255 (NaN). An fp8
    value is k * 2^-9 with at most 4 significant bits, so value * 2^(e-127) is exact in fp16
    whenever it stays a multiple of 2^-24 (e >= 112) and at most 65504 (448 * 2^(e-127),
    e <= 134). A full scan of both tables' scales finds codes 116..123 only. Inside that
    window this is the reference's bf16 row bit for bit, and the exl3 wkv takes it with no
    further rounding (tests/test_dsv41_engram_ref_.py checks it on real rows).
    """
    U = w.shape[0]
    v = w.view(torch.float8_e4m3fn).float().view(U, -1, 32)
    codes = s.to(torch.int32)
    bits = (codes << 23) | (((codes == 0) | (codes == 255)).to(torch.int32) << 22)
    scale = bits.view(torch.float32)
    return (v * scale.unsqueeze(-1)).view(U, -1).half()
