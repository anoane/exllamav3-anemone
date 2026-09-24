"""CPU-only check of the reference engram n-gram hash (tests/dsv41_ref/engram_hash.py) against a
real checkpoint's bucket layout (architecture/dsv41/engram.py): every index inside its bucket
range and the table, deterministic, different per layer. No exllamav3 import.

    python tests/test_dsv41_engram_hash_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file, model_dir, skip
from dsv41_ref import engram_hash as eh

def main(path):
    eg = load_package_file("exllamav3/architecture/dsv41/engram.py", "_dsv41_engram")
    t = json.load(open(os.path.join(path, "config.json")))
    t = t.get("text_config", t)
    cfg = type("C", (), {})()
    for k in ("engram_layer_ids","engram_num_embeddings","engram_max_ngram_size",
              "engram_n_heads","engram_head_dim","engram_vocab_size",
              "engram_compressed_vocab_size","engram_pad_token_id"):
        setattr(cfg, k, t[k])
    L = eg.EngramLayout(cfg)
    rng = np.random.default_rng(0)
    token_map = rng.integers(0, L.compressed_vocab_size, size = 200000, dtype = np.int64)
    ids = rng.integers(0, 200000, size = 256, dtype = np.int64)
    H = eh.engram_hash_reference(ids, L, token_map)
    assert H.shape == (256, len(L.layer_ids), L.n_hash_cols)
    for li in range(len(L.layer_ids)):
        primes = np.array([p for per in L.primes[li] for p in per], dtype = np.int64)
        offs = L.offsets[li].astype(np.int64)
        for col in range(L.n_hash_cols):
            v = H[:, li, col]
            assert (v >= offs[col]).all() and (v < offs[col] + primes[col]).all(), \
                f"layer {li} col {col}: index outside its bucket range"
        assert H[:, li, :].max() < L.num_embeddings[li], "index past table end"
    assert (H == eh.engram_hash_reference(ids, L, token_map)).all(), "not deterministic"
    assert not (H[:, 0, :] == H[:, 1, :]).all(), "layers must hash differently"
    print(f"  OK  engram hash: {H.shape}, layers {L.layer_ids}, "
          f"{L.n_hash_cols} buckets/layer, all ranges respected")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("engram hash", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
