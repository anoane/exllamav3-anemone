# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Transcribes parts of DeepSeek's reference inference code as a test oracle; see NOTICE.

"""The V4.1 attention front (DSV41Attention._project_qkv) against DeepSeek's, on real weights.

    PYTHONPATH=$PWD python3 tests/test_dsv41_qkv_front_gpu_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

DeepSeek's V4.1 reference (inference/model.py, Attention.forward):

    qr = q_norm(wq_a(x))                 # weighted RMSNorm over the q_lora latent
    q  = wq_b(qr).unflatten(heads)       # NO per-head norm
    kv = kv_norm(wkv(x))                 # weighted RMSNorm over head_dim
    rope(q[..., -rd:]); rope(kv[..., -rd:])

transcribed here around the layer's own projections (the exl3 Linears, so quantization is
common-mode): both norms in float64 with the layer's weights and eps, the rotation in float64
by the fp32 angle t * inv_freq with the layer's table (sliding layer 0: the main table; 2 and
20: the compressed one). Gate: q_res, q and kv within rel-L2 1e-3 of the transcription, at
start positions 0 and 70,000.

V4's front (DSV4Attention._project_qkv) also applies an UNWEIGHTED per-head RMSNorm to q, which
fixes every head's query length. It must differ from the transcription by more than 0.1, or the
check could not see that bug.

The comparisons run under the setting 'precise' (config.dsv41_numerics), pinned whatever
EXL3_DSV41_NUMERICS is set. Under 'deepseek:window' and 'vllm:window' the front must hand
round_window_ its rotated kv (the 'precise' kv, to within the projections' run-to-run noise) and
return exactly the rounded result, with q unchanged: the window rounding is wired after the
rotation, and to nothing else.
"""
import os, sys
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip
if __name__ == "__main__" and not torch.cuda.is_available():
    skip("V4.1 attention front on the GPU", "no CUDA device")
    sys.exit(0)

from exllamav3 import Config, Model
from exllamav3.modules.dsv4 import DSV4Attention
from exllamav3.modules import dsv41_rounding as rounding
from exllamav3.architecture.dsv41 import numerics


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def rms64(x, w, eps):
    x = x.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim = True) + eps) * w.double()


def rope64(x, inv, pos):
    """x (seq, heads, rd) rotated in float64 by the fp32 angles pos * inv (GPT-J pairs)."""
    ang = (pos.float().unsqueeze(-1) * inv.float().unsqueeze(0)).double()[:, None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    x = x.double()
    out = torch.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return out


def main(path):
    dev = torch.device("cuda:0")
    cfg = Config.from_directory(path)
    cfg.dsv41_numerics = "precise"
    model = Model.from_config(cfg)
    B = model.modules[model.first_block_idx: model.first_block_idx + cfg.num_hidden_layers]
    torch.manual_seed(0)
    T = 300
    x = (torch.randn(1, T, cfg.hidden_size, device = dev) * 0.5).half()
    eps = cfg.rms_norm_eps
    worst, v4_min = 0.0, float("inf")
    for L in (0, 2, 20):
        a = B[L].attn
        a.load(dev)
        try:
            hd, rd, H = a.head_dim, a.rope_head_dim, a.num_q_heads
            inv = a._rope_type()
            with torch.inference_mode():
                for p0 in (0, 70_000):
                    q_res, q, kv = a._project_qkv(x, {}, p0)
                    pos = torch.arange(p0, p0 + T, device = dev)
                    # the transcription, around the layer's own projections
                    qr_ref = rms64(a.q_a.forward(x, {}), a.q_norm.weight, eps)
                    q_ref = a.q_b.forward(qr_ref.half(), {}).view(T, H, hd).double()
                    kv_ref = rms64(a.wkv.forward(x, {}), a.kv_norm.weight, eps).view(T, 1, hd)
                    q_ref[..., -rd:] = rope64(q_ref[..., -rd:], inv, pos)
                    kv_ref[..., -rd:] = rope64(kv_ref[..., -rd:], inv, pos)
                    e = (rel(q_res[0], qr_ref[0]), rel(q[0], q_ref), rel(kv[0], kv_ref[:, 0]))
                    print(f"  L{L} position {p0:>6}: q_res {e[0]:.2e}  q {e[1]:.2e}  kv {e[2]:.2e}")
                    assert max(e) < 1e-3, f"layer {L} at {p0}: front differs from DeepSeek's by {max(e):.2e}"
                    worst = max(worst, *e)
                    # the window rounding: round_window_ of the rotated kv, returned as is
                    for setting in ("deepseek:window", "vllm:window"):
                        seen = []
                        real = rounding.round_window_
                        def spy(policy, t, rope_dim):
                            seen.append(t.clone())
                            return real(policy, t, rope_dim)
                        cfg.dsv41_numerics = setting
                        rounding.round_window_ = spy
                        try:
                            q_res_w, q_w, kv_w = a._project_qkv(x, {}, p0)
                        finally:
                            rounding.round_window_ = real
                            cfg.dsv41_numerics = "precise"
                        assert len(seen) == 1, f"layer {L}: round_window_ called {len(seen)} times"
                        want = real(numerics.parse(setting), seen[0].clone(), rd)
                        assert torch.equal(kv_w, want) and not torch.equal(kv_w, seen[0]), \
                            f"layer {L} at {p0}: {setting} kv is not round_window_ of the rotated kv"
                        e_in = rel(seen[0], kv)
                        assert e_in < 1e-6, f"layer {L} at {p0}: {setting} rounded another kv ({e_in:.2e})"
                        assert rel(q_w, q) < 1e-6 and rel(q_res_w, q_res) < 1e-6, \
                            f"layer {L} at {p0}: {setting} changed the query"
                # V4's front at position 0 (its per-head q norm) must be told apart
                _, q4, _ = DSV4Attention._project_qkv(a, x, {}, 0)
                pos = torch.arange(0, T, device = dev)
                qr_ref = rms64(a.q_a.forward(x, {}), a.q_norm.weight, eps)
                q_ref = a.q_b.forward(qr_ref.half(), {}).view(T, H, hd).double()
                q_ref[..., -rd:] = rope64(q_ref[..., -rd:], inv, pos)
                e4 = rel(q4[0], q_ref)
                print(f"  L{L} V4's front (per-head q norm): q {e4:.2e}")
                assert e4 > 0.1, f"layer {L}: V4's front is not told apart ({e4:.2e})"
                v4_min = min(v4_min, e4)
        finally:
            a.unload()
            torch.cuda.empty_cache()
    print(f"  OK  V4.1 attention front == DeepSeek's transcription on layers 0, 2, 20 (worst {worst:.2e}); "
          f"V4's per-head q norm differs by >= {v4_min:.2f}; deepseek:window and vllm:window round "
          f"exactly the front's kv")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 attention front on the GPU", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
