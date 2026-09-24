"""GPU checks of V4.1 attention details that CPU tests cannot see (they mock rope).

1. Pool rope positions. Entry i of a chunk rotates at (first_group + i) * m --
   the first token of its group -- recomputed here with an independent
   plain-torch GPT-J rotation. Also asserts the old, wrong positions
   (first_group * m + i) would FAIL, so the check cannot pass vacuously.
2. Consumer top-k. Past index_topk entries a consumer must attend over the
   top-k selection its nearest index source published, not the whole pool.

    PYTHONPATH=$PWD python3 tests/test_dsv41_attn_gpu_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip
if __name__ == "__main__" and not torch.cuda.is_available():
    skip("V4.1 attention on the GPU", "no CUDA device")
    sys.exit(0)
from exllamav3 import Config, Model
import exllamav3.modules.dsv41 as dsv41

def gptj_rope(x, pos, inv_freq):
    # x (n, rd) fp32; interleaved pairs (2k, 2k+1) rotated by pos * inv_freq[k]
    ang = pos.float().unsqueeze(1) * inv_freq.float().unsqueeze(0)          # (n, rd/2)
    c, s_ = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[:, 0::2], x[:, 1::2]
    out = torch.empty_like(x)
    out[:, 0::2] = x0 * c - x1 * s_
    out[:, 1::2] = x0 * s_ + x1 * c
    return out

def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()

def main(path):
    dev = torch.device("cuda:0")
    cfg = Config.from_directory(path)
    model = Model.from_config(cfg)
    B = model.modules[model.first_block_idx: model.first_block_idx + cfg.num_hidden_layers]
    torch.manual_seed(0)
    seq = 700
    x = (torch.randn(1, seq, cfg.hidden_size, device=dev) * 0.5).half()

    # ---- 1. rope positions, rate 2 (layer 2) and rate 1 (layer 20)
    for L in (2, 20):
        a = B[L].attn; a.load(dev)
        params = {"attn_mode": "flash_attn_nc", "position": 0, "dsv41_pools": {}}
        pools = a.publish_pools(x, params, 0)
        latent, first = a.compressor.forward(x[0], params, 0)
        n, m, rd, hd = pools.entries, a.compress_ratio, a.rope_head_dim, a.head_dim
        raw = latent[:, hd - rd:].half().float()
        good = gptj_rope(raw, (first + torch.arange(n, device=dev)) * m, a.inv_freq_compress)
        bad = gptj_rope(raw, first * m + torch.arange(n, device=dev), a.inv_freq_compress)
        got = pools.pool_r[:n].float()
        e_good, e_bad = rel(got, good), rel(got, bad)
        print(f"  L{L} rate {m}: pool_r vs (first+i)*m rel {e_good:.2e} | vs old first*m+i rel {e_bad:.2e}")
        assert e_good < 2e-3, f"L{L}: pool rope does not match (first+i)*m"
        if m > 1:
            assert e_bad > 1e-1, f"L{L}: the check cannot tell the old positions apart"
        a.unload(); torch.cuda.empty_cache()

    # ---- 2. consumer 21 must receive source 20's top-k (700 > index_topk 512)
    seen = {}
    orig = dsv41.dsa_attn
    def spy(*args, **kw):
        seen.setdefault("indices", []).append(kw.get("indices"))
        return orig(*args, **kw)
    dsv41.dsa_attn = spy
    try:
        params = {"attn_mode": "flash_attn_nc", "position": 0, "dsv41_pools": {}}
        for L in (20, 21):
            a = B[L].attn; a.load(dev); a.forward(x, params); a.unload()
    finally:
        dsv41.dsa_attn = orig
    src_idx, con_idx = seen["indices"]
    assert src_idx is not None, "index source 20 did not select top-k at 700 tokens"
    assert con_idx is not None, "consumer 21 attended densely instead of using source 20's top-k"
    assert torch.equal(src_idx.cpu(), con_idx.cpu()), "consumer 21 did not reuse source 20's selection"
    print(f"  consumer 21 reused index source 20's top-k: indices {tuple(con_idx.shape)}")
    print("  OK  V4.1 pool rope positions and consumer top-k")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 attention on the GPU", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
