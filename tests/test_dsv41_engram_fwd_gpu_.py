# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Transcribes parts of DeepSeek's reference inference code as a test oracle; see NOTICE.

"""DSV41Engram.forward vs DeepSeek's reference Engram.forward, on real weights.

The oracle transcribes DeepSeek's inference/model.py Engram.forward and its
ParallelEngramEmbedding lookup. Rows reach it by a DIFFERENT path than the module
uses -- the loader's Python DiskTensorHandle.read_rows instead of the C++
ngram_gather_cpu pool -- so the gather is checked independently too. The exl3
wkv is shared (quantization is common-mode). Needs a GPU.

    PYTHONPATH=$PWD python3 tests/test_dsv41_engram_fwd_gpu_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip

def main(checkpoint_dir):
    from exllamav3 import Config, Model
    from exllamav3.architecture.dsv41.engram_torch import UNK
    dev = torch.device("cuda:0")
    cfg = Config.from_directory(checkpoint_dir)
    model = Model.from_config(cfg)
    torch.manual_seed(0)
    B, L = 1, 257
    ids = torch.randint(0, 129000, (B, L))
    H, D = cfg.hc_mult, cfg.hidden_size
    x = torch.randn(B, L, H, D, device=dev) * 0.7
    for layer, eg in model.engram_modules.items():
        eg.load(dev)
        params = {"input_ids": ids, "position": 0}
        got = eg.forward(x.clone(), params)

        # ---- oracle ----
        hid = eg.hasher.hash(eg.hasher.compress(ids), torch.full((B, eg.hasher.ctx), UNK))[:, :, eg.table_index]
        w = eg.w_handle.read_rows(hid.reshape(-1)).view(torch.uint8)              # python path
        sc = eg.s_handle.read_rows(hid.reshape(-1)).view(torch.uint8)
        vals = w.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32)).to(dev)
        scl = sc.to(dev).float() if sc.dtype != torch.uint8 else torch.pow(2.0, sc.to(dev).float() - 127)
        rows = (vals * scl.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)          # [B*L*24, 256]
        kv = eg.wkv.forward(rows.view(B, L, -1).half(), params, out_dtype=torch.float)
        key, value = kv.split([H * D, D], dim=-1)
        key = key.float().unflatten(-1, (H, D))
        weight = eg.q_weight.to(torch.bfloat16).float() * eg.k_weight.to(torch.bfloat16).float()
        h, eps = x.float(), cfg.rms_norm_eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * D ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        want = h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)

        e = ((got - want).norm() / want.norm()).item()
        inj = ((want - x).norm() / x.norm()).item()
        print(f"  layer {layer}: module vs reference rel-L2 {e:.2e}   (engram moves the streams by rel {inj:.3f}; "
              f"gate mean {gate.mean():.3f})")
        assert e < 1e-4, f"layer {layer}: engram forward differs from the reference"
        assert inj > 1e-3, "engram injected nothing -- the check would not catch a skipped engram"
        eg.unload()
    print("  OK  engram forward == DeepSeek reference Engram.forward on both engram layers")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not torch.cuda.is_available():
        skip("engram forward vs DeepSeek on the GPU", "no CUDA device")
    elif not d:
        skip("engram forward vs DeepSeek on the GPU", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
    else:
        main(d)
