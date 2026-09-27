"""
BlockSparseMLP must export the selection (e_score_correction) bias in its checkpoint precision
when the bias key is the routing gate's own ".bias" (DeepSeek-V4: key_routing_gate = "gate",
key_e_score_bias = "gate.bias"). Regression: the routing gate Linear also loaded that tensor as
an fp16 logits bias, and since the converter collects tensors parent-first
(for m in module: q_tensors.update(m.get_tensors())), the gate's fp16 copy overwrote the fp32
one, so every converted DeepSeek-V4 model stored the selection bias in fp16.

CPU-only: loads a tiny synthetic checkpoint on the CPU, no kernels run.
"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from safetensors.torch import save_file
from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.model.config import InferParams
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.block_sparse_mlp_routing import _esb_h

H, I, E, K = 64, 32, 16, 2
KEY = "layers.3.ffn"


def _save_checkpoint(directory, extra):
    g = torch.Generator().manual_seed(0)
    rnd = lambda *shape: (0.02 * torch.randn(*shape, generator = g)).bfloat16()
    t = {f"{KEY}.gate.weight": rnd(E, H)}
    for e in range(E):
        t[f"{KEY}.experts.{e}.w1.weight"] = rnd(I, H)
        t[f"{KEY}.experts.{e}.w3.weight"] = rnd(I, H)
        t[f"{KEY}.experts.{e}.w2.weight"] = rnd(H, I)
    t.update(extra)
    os.makedirs(directory, exist_ok = True)
    save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(directory, "model.safetensors"))


def _load(directory, key_e_score_bias, router_type):
    config = SimpleNamespace(
        stc = SafetensorsCollection(str(directory), load_method = "python"),
        infer_params = InferParams(),
    )
    mlp = BlockSparseMLP(
        config = config,
        key = KEY,
        hidden_size = H,
        intermediate_size = I,
        num_experts = E,
        num_experts_per_tok = K,
        key_up = "experts.{expert_idx}.w3",
        key_gate = "experts.{expert_idx}.w1",
        key_down = "experts.{expert_idx}.w2",
        key_routing_gate = "gate",
        key_e_score_bias = key_e_score_bias,
        router_type = router_type,
        activation_fn = "silu",
        qmap = "block.mlp",
        interm_dtype = torch.half,
        out_dtype = torch.float,
    )
    mlp.load(torch.device("cpu"))
    return mlp


def _collect(mlp):
    # Same collection order as the converter (convert_model.py)
    t = {}
    for m in mlp:
        t.update(m.get_tensors())
    return t


def _bias(dtype):
    # DeepSeek-V4-Flash magnitudes: ~20 with a small inter-expert spread, where fp16 steps are 1/64
    g = torch.Generator().manual_seed(1)
    return (20.0 + 0.4 * torch.randn(E, generator = g)).to(dtype)


def test_selection_bias_exported_in_checkpoint_precision(tmp_path):
    bias = _bias(torch.float)
    _save_checkpoint(tmp_path / "src", {f"{KEY}.gate.bias": bias})
    mlp = _load(tmp_path / "src", "gate.bias", "sqrtsp")
    t = _collect(mlp)
    assert t[f"{KEY}.gate.bias"].dtype == torch.float
    assert torch.equal(t[f"{KEY}.gate.bias"], bias)
    assert mlp.routing_gate.inner.bias is None

    # Reload the export: fp32 is kept and the kernels get the centered fp16 copy
    _save_checkpoint(tmp_path / "out", {f"{KEY}.gate.bias": t[f"{KEY}.gate.bias"]})
    mlp = _load(tmp_path / "out", "gate.bias", "sqrtsp")
    assert mlp.e_score_correction_bias.dtype == torch.float
    esb_h = _esb_h(SimpleNamespace(e_score_bias_h = None, e_score_correction_bias = mlp.e_score_correction_bias))
    assert torch.equal(esb_h, (bias - bias.mean()).half())


def test_fp16_selection_bias_still_loads(tmp_path):
    # Models converted before the fix store the selection bias in fp16
    bias = _bias(torch.half)
    _save_checkpoint(tmp_path, {f"{KEY}.gate.bias": bias})
    mlp = _load(tmp_path, "gate.bias", "sqrtsp")
    assert mlp.e_score_correction_bias.dtype == torch.half
    assert torch.equal(mlp.e_score_correction_bias, bias)
    t = _collect(mlp)
    assert torch.equal(t[f"{KEY}.gate.bias"], bias)


@pytest.mark.parametrize("key_e_score_bias, router_type", [
    ("gate.e_score_correction_bias", "ds3"),    # distinct selection-bias key
    (None, "std_bias"),                         # gate.bias is a real logits bias (gpt-oss)
])
def test_distinct_gate_bias_kept(tmp_path, key_e_score_bias, router_type):
    gate_bias = _bias(torch.half) - 20.0
    extra = {f"{KEY}.gate.bias": gate_bias}
    if key_e_score_bias:
        extra[f"{KEY}.{key_e_score_bias}"] = _bias(torch.float)
    _save_checkpoint(tmp_path, extra)
    mlp = _load(tmp_path, key_e_score_bias, router_type)
    assert torch.equal(mlp.routing_gate.inner.bias, gate_bias)
    t = _collect(mlp)
    assert torch.equal(t[f"{KEY}.gate.bias"], gate_bias)
    if key_e_score_bias:
        assert t[f"{KEY}.{key_e_score_bias}"].dtype == torch.float


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
