# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""V4.1 delayed mHC vs the vLLM torch reference, on REAL checkpoint tensors.

The oracle below is vLLM's mhc_pre_delayed_torch / mhc_post_torch
(vllm/model_executor/kernels/mhc/torch.py, Apache-2.0; see NOTICE), copied
verbatim apart from formatting. The chain test runs two layers' four sublayers
with the sublayer bodies replaced by fixed random maps, so it checks the
pre-mix HAND-OFF (attention <- previous FFN, FFN <- this attention, head <-
last FFN) -- the part a single-sublayer comparison cannot see.

On a GPU mix_delayed runs V4's fused ext.hc_mix for post and comb and re-derives
the pre-mix from the kernel's partial dots; the last check holds it to the torch
body (mix_torch) at 1 to 4096 rows: post, comb and pre within 1e-5 rel-L2, the
collapse bitwise.
Needs a GPU (the unweighted RMSNorm goes through the extension).

    PYTHONPATH=$PWD python3 tests/test_dsv41_mhc_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip
if __name__ == "__main__" and not torch.cuda.is_available():
    skip("V4.1 delayed mHC", "no CUDA device")
    sys.exit(0)
from exllamav3 import Config, Model

def ref_pre_delayed(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None):
    hc_mult = residual.shape[1]
    x = (residual.flatten(1) if x is None else x).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    post = torch.sigmoid(mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]) * hc_post_mult_value
    comb = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult) * hc_scale[2]
    comb = comb + hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = residual[:, 0] if pre_mix is None else \
        (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)
    return post.unsqueeze(-1), comb, layer_input, pre

def ref_post(x, residual, post_layer_mix, comb_res_mix):
    mixed = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    return (mixed + post_layer_mix.float() * x.unsqueeze(-2).float()).to(residual.dtype)

def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

def main(path):
    dev = torch.device("cuda:0")
    cfg = Config.from_directory(path)
    model = Model.from_config(cfg)
    B = model.modules[model.first_block_idx: model.first_block_idx + cfg.num_hidden_layers]
    H, D = cfg.hc_mult, cfg.hidden_size
    kw = dict(rms_eps=cfg.rms_norm_eps, hc_pre_eps=cfg.hc_eps, hc_sinkhorn_eps=cfg.hc_eps,
              hc_post_mult_value=2.0, sinkhorn_repeat=cfg.hc_sinkhorn_iters)
    layers = (0, 1)
    sites = []
    for L in layers:
        for hc in (B[L].attn_hc, B[L].mlp_hc):
            hc.load(dev); sites.append(hc)
    assert type(sites[0]).__name__ == "DSV41HyperConnection", type(sites[0]).__name__

    torch.manual_seed(0)
    T = 257
    emb = torch.randn(T, D, device=dev) * 0.5
    maps = [torch.randn(D, D, device=dev) * (0.3 / D ** 0.5) for _ in sites]   # stand-in sublayers
    body = lambda i, x: torch.tanh(x.float() @ maps[i])

    # --- reference chain (vLLM layout: residual (T, H, D)) ---
    residual = emb.unsqueeze(1).expand(-1, H, -1).contiguous()
    pre = None; ref_inputs = []
    for i, hc in enumerate(sites):
        post, comb, x, new_pre = ref_pre_delayed(residual, hc.fn, hc.scale, hc.base, pre_mix=pre, **kw)
        ref_inputs.append(x.float())
        residual = ref_post(body(i, x), residual, post, comb)
        pre = new_pre
    ref_final = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)

    # --- ours (exllamav3 layout: streams (b, s, H, D)) ---
    params = {}
    streams = emb.unsqueeze(0).unsqueeze(2).expand(-1, -1, H, -1).contiguous().float()
    carried = None; worst_in = 0.0
    for i, hc in enumerate(sites):
        post, comb, y, new_pre = hc.mix_delayed(streams, params, carried)
        worst_in = max(worst_in, rel(y[0], ref_inputs[i]))
        streams = hc.apply_(streams, body(i, y[0]).unsqueeze(0), post, comb, params)
        carried = new_pre
    ours_final = (carried.unsqueeze(-1) * streams).sum(dim=2)[0]

    # the carried pre must MATTER: collapsing the FFN with its own (V4-style)
    # pre-mix must give a measurably different input than the delayed one
    post, comb, y_v41, own = sites[1].mix_delayed(streams, params, carried)
    y_v4 = (own.unsqueeze(-1) * streams).sum(dim=2)
    delay_effect = rel(y_v4, y_v41)

    e_final = rel(ours_final, ref_final)
    print(f"  sublayer inputs vs reference: worst rel-L2 {worst_in:.2e} over {len(sites)} sites")
    print(f"  final head collapse vs reference: rel-L2 {e_final:.2e}")
    print(f"  delayed vs immediate pre-mix differ by rel-L2 {delay_effect:.2e} (must be >> tolerance)")
    assert worst_in < 1e-4, f"sublayer input mismatch {worst_in:.2e}"
    assert e_final < 1e-4, f"final collapse mismatch {e_final:.2e}"
    assert delay_effect > 1e-3, "delay has no effect -- the test would not catch V4 semantics"

    # fused (ext.hc_mix + re-derived pre) vs the torch body, across the row counts where the
    # kernel's chunking and the workspaces (bucketed at <= 32 rows) change
    worst = {"post": 0.0, "comb": 0.0, "pre": 0.0}
    for R in (1, 2, 32, 33, 511, 512, 4096):
        st = (torch.randn(1, R, H, D, device=dev) * 0.5).float().contiguous()
        carried = torch.rand(1, R, H, device=dev) + 0.1
        for hc in sites[:2]:
            post_f, comb_f, y_f, pre_f = hc.mix_delayed(st, params, carried)
            post_t, comb_t, pre_t = hc.mix_torch(st, params)
            y_t = (carried.unsqueeze(-1) * st).sum(dim=2)
            for k, (a, b) in {"post": (post_f, post_t), "comb": (comb_f, comb_t), "pre": (pre_f, pre_t)}.items():
                worst[k] = max(worst[k], rel(a, b))
            assert torch.equal(y_f, y_t), f"R={R}: fused collapse differs from the torch expression"
            _, _, y_own, _ = hc.mix_delayed(st, params, None, True)
            assert torch.equal(y_own, (pre_f.unsqueeze(-1) * st).sum(dim=2)), f"R={R}: own-pre collapse"
    print(f"  fused vs torch mix at R in 1..4096: post {worst['post']:.2e}, comb {worst['comb']:.2e}, "
          f"pre {worst['pre']:.2e} rel-L2; collapse bitwise")
    assert max(worst.values()) <= 1e-5, worst
    print(f"  OK  V4.1 delayed mHC matches the vLLM reference on real hc tensors (layers {layers})")

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 delayed mHC", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
