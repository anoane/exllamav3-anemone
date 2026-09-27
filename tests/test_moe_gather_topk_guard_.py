"""
exl3_moe_gather (the deterministic slot reduction, EXL3_MOE_FUSED_DET=1 by default) resolves each
token's slots into a fixed MOE_GATHER_MAX_TOPK-entry shared list and refuses wider selections. The
measurement passes (conversion/measure_model.py) forward quantized MoE blocks with
activate_all_experts, which selects every expert per token. With small calibration sizes the
experts land in the fused / batched reconstruct tiers and the layer used to fail with "top-k too
large"; such selections must take the accumulating path instead, while normal top-k routing keeps
the gather.

CPU only: the extension calls are replaced by recorders (the gather mirrors the kernel's host-side
check, limit read from exl3_moe.cu) and _run_batch_recon by a recorder of the scratch it receives,
everything else (routing, tier plan, slot tables, batched reconstruct row caps) is the upstream
Python.
"""
import os, re, sys
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import exllamav3.modules.block_sparse_mlp as bsm
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.block_sparse_mlp_routing import routing_std
from exllamav3.modules.moe_batch_recon import BatchReconLayer

H, I, E, K = 256, 128, 64, 8
TOKENS = 300

_cu = os.path.join(os.path.dirname(bsm.__file__), "..", "exllamav3_ext", "quant", "exl3_moe.cu")
with open(_cu, encoding = "utf-8") as f:
    KERNEL_MAX_TOPK = int(re.search(r"#define MOE_GATHER_MAX_TOPK (\d+)", f.read()).group(1))


class ExtRecorder:
    def __init__(self):
        self.calls = []

    def exl3_moe(self, *args):
        self.calls.append(("exl3_moe", args[-5]))  # scratch

    def exl3_moe_gather(self, output_state, scratch, flat_expert, *args):
        topk = flat_expert.numel() // output_state.shape[0]
        self.calls.append(("gather", topk))
        if topk > KERNEL_MAX_TOPK:
            raise RuntimeError("exl3_moe_gather: top-k too large")


def make_layer():
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    mq = NS(K = 4, ptrs_trellis = None, ptrs_suh = None, ptrs_svh = None, mcg = False, mul1 = False)
    ones = lambda n: [torch.ones(n, dtype = torch.half)] * E
    recon = BatchReconLayer(
        (H, I, 4), (H, I, 4), (I, H, 4), (False, False), (False, False), (False, False), 0, 0.0,
        torch.device("cpu"), {"g": (ones(H), ones(I)), "u": (ones(H), ones(I)), "d": (ones(I), ones(H))},
        folded = False)
    m.__dict__.update(
        alt_residual_channel = False, hidden_size = H, expert_size = H, intermediate_size = I,
        intermediate_size_padded = I, bc = NS(), router_pre_norm = None, routing_gate = object(),
        routing_cfg = NS(gate_tensor = (torch.randn(H, E, generator = torch.Generator().manual_seed(0)) * 0.05).half(),
                         num_experts = E, num_experts_per_tok = K, per_expert_scale = None),
        routed_pre_norm = None, latent_in = None, latent_out = None, routing_device = None,
        cpu_split_first = None, cpu_offload = False, num_local_experts = E, num_experts = E,
        num_experts_per_tok = K, f_threshold = 4, is_quantized = True,
        config = NS(infer_params = NS(no_reconstruct = False)), support_quant_paths = True,
        routing_first = 0, fused_mode_buffers = bsm.FusedBuffers(None, None, None, None),
        fused_rows = bsm.FUSED_ROWS_WIDE, mtile_ok = True, gated = True, activation_fn_idx = 0,
        act_limit = 0.0, multi_gate = mq, multi_up = mq, multi_down = mq, interm_dtype = torch.half,
        batch_recon = recon, shared_experts = None, routed_post_norm = None, tp_reduce = False,
        device = torch.device("cpu"),
    )

    def routing_fn(bsz, cfg, y, params):
        if params.get("activate_all_experts"):
            return routing_std(bsz, cfg, y, params)
        w, sel = torch.softmax((y @ cfg.gate_tensor).float(), -1).topk(K, -1)
        return sel, (w / w.sum(-1, keepdim = True)).half()
    m.routing_fn = routing_fn

    recon_scratch = []
    def run_batch_recon(recon, y, fhs_ext, token_sorted, weight_sorted, counts, groups, scratch = None, tables = None):
        recon_scratch.append(scratch)
        return {e for g in groups for e in g}
    m._run_batch_recon = run_batch_recon
    return m, recon_scratch


def run(monkeypatch, params, tokens = TOKENS, tier = None):
    rec = ExtRecorder()
    monkeypatch.setattr(bsm, "ext", rec)
    monkeypatch.setattr(bsm, "FUSED_DET", True)
    m, recon_scratch = make_layer()
    if tier == "fused":
        assert tokens <= m.fused_rows
    elif tier == "batched":
        assert m.fused_rows < tokens <= m.batch_recon.max_rows
    x = (torch.randn(tokens, H, generator = torch.Generator().manual_seed(1)) * 0.1).half()
    out = m.forward(x, params)
    assert out.shape == (tokens, H)
    return rec.calls, recon_scratch


# All experts selected: every expert gets one row per token, so the token count picks the tier
@pytest.mark.parametrize("tokens, tier", [(200, "fused"), (300, "batched")])
def test_all_experts_selection_accumulates(monkeypatch, tokens, tier):
    # measure_model.py's params for the quantized block forwards
    params = {"activate_all_experts": True, "attn_mode": "flash_attn_nc"}
    calls, recon_scratch = run(monkeypatch, params, tokens, tier)
    assert E > KERNEL_MAX_TOPK
    assert not any(c[0] == "gather" for c in calls)
    fused_calls = [c for c in calls if c[0] == "exl3_moe"]
    assert all(c[1] is None for c in fused_calls)
    if tier == "fused":
        assert fused_calls and not recon_scratch
    else:
        assert recon_scratch and all(s is None for s in recon_scratch)


def test_topk_selection_keeps_gather(monkeypatch):
    calls, recon_scratch = run(monkeypatch, {"attn_mode": "flash_attn_nc"})
    assert ("gather", K) in calls
    assert all(c[1] is not None for c in calls if c[0] == "exl3_moe")


def test_limit_matches_kernel():
    assert bsm.MOE_GATHER_MAX_TOPK == KERNEL_MAX_TOPK


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
