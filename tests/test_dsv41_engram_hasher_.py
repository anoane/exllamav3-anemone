"""EngramHasher vs DeepSeek's reference NgramHashState (inference/engram.py).

The oracle is DeepSeek's own file, imported as-is from DSV41_DEEPSEEK_REF. It wants a
transformers tokenizer only for .backend_tokenizer and len(), which a two-line shim
supplies. Checks: the prime layout, one-shot hashes on real-vocab ids with dead tokens
mixed in, and a chunked split whose second chunk sees only a 3-id lookback window --
against the reference's own two-call (start_pos) run. CPU only; imports exllamav3, so it
needs the built extension.

    python tests/test_dsv41_engram_hasher_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""
import json, os, sys, types
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip
from dsv41_ref import deepseek

def main(checkpoint_dir):
    ref = deepseek.load_engram()
    sys.path.insert(0, _H)
    from exllamav3.architecture.dsv41 import engram as port_engram
    from exllamav3.architecture.dsv41.engram_torch import (
        EngramHasher, build_compressed_token_map, UNK, DEAD)

    t = json.load(open(os.path.join(checkpoint_dir, "config.json")))
    t = t.get("text_config", t)
    args = types.SimpleNamespace(
        engram_layer_ids = t["engram_layer_ids"], engram_max_ngram_size = t["engram_max_ngram_size"],
        engram_n_heads = t["engram_n_heads"], engram_head_dim = t["engram_head_dim"],
        engram_vocab_size = t["engram_vocab_size"], engram_num_embeddings = t["engram_num_embeddings"],
        engram_compressed_vocab_size = t["engram_compressed_vocab_size"],
        engram_pad_id = t["engram_pad_token_id"], max_batch_size = 2, max_seq_len = 4096)

    # reference layout (sympy primes) vs the port's layout
    ref_layout = ref.EngramLayout.from_args(args)
    port_layout = port_engram.EngramLayout(types.SimpleNamespace(**{k: t[k] for k in t if k.startswith("engram_")}))
    assert tuple(tuple(tuple(p) for p in l) for l in port_layout.primes) == ref_layout.primes, "prime layout differs"

    # reference hash state through a shim tokenizer
    from tokenizers import Tokenizer
    backend = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    class Shim:
        def __init__(self, b): self.backend_tokenizer = b
        def __len__(self): return self.backend_tokenizer.get_vocab_size(with_added_tokens = True)
    shim = Shim(backend)
    ref_state = ref.NgramHashState(args, ref_layout, shim)

    token_map, size = build_compressed_token_map(os.path.join(checkpoint_dir, "tokenizer.json"))
    assert size == t["engram_compressed_vocab_size"], size
    assert torch.equal(token_map, ref_state.token_map.long()), "compressed token map differs"
    hasher = EngramHasher(port_layout, token_map, size, t["engram_pad_token_id"])

    torch.manual_seed(0)
    B, L = 2, 300
    ids = torch.randint(0, 129000, (B, L))
    mask = torch.ones(B, L, dtype = torch.bool); mask[1, 100:110] = False    # a dead span

    # one shot
    want = ref_state(ids, 0, mask)
    comp = hasher.compress(ids); comp = torch.where(mask, comp, DEAD)
    got = hasher.hash(comp, torch.full((B, hasher.ctx), UNK))
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.equal(got, want.long()), f"one-shot hashes differ at {int((got != want).sum())} entries"

    # chunked: the second chunk sees only a ctx-id lookback window
    cut = 173
    want1 = ref_state(ids[:, :cut], 0, mask[:, :cut])
    want2 = ref_state(ids[:, cut:], cut, mask[:, cut:])
    got1 = hasher.hash(comp[:, :cut], torch.full((B, hasher.ctx), UNK))
    got2 = hasher.hash(comp[:, cut:], comp[:, cut - hasher.ctx: cut])
    assert torch.equal(got1, want1.long()) and torch.equal(got2, want2.long()), "chunked hashes differ"
    assert torch.equal(torch.cat([got1, got2], 1), got), "chunked != one shot"

    # positions 0..ctx-1 must be hashed (padded), not skipped
    assert (got[:, :hasher.ctx] >= 0).all()
    print(f"  OK  engram hasher == DeepSeek NgramHashState: {tuple(got.shape)} bitwise, one-shot and "
          f"chunked at {cut} (3-id lookback), dead span honoured, token map {size} ids")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("engram hasher", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
    elif deepseek.ref_dir() is None:
        skip("engram hasher", "DSV41_DEEPSEEK_REF not set")
    else:
        main(d)
