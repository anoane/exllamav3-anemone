"""
With a static split placement (-mcs / --moe_cpu_split, EXL3_MOE_CPU_SWAP=0 and
EXL3_MOE_CPU_SPLIT_STATS), BlockSparseMLP_CPU.cpu_post_load must permute every per-expert routing
tensor into the same hot-to-cold order as the router columns, including DeepSeek-V4's vision
selection bias (gate.bias_vl). Regression: e_score_bias_vl stayed in checkpoint order, so image
rows added the bias of expert j to the score of expert perm[j] and picked the wrong experts.
CPU only, no weights.
"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

E, H, TAIL = 64, 128, 16


def test_split_perm_permutes_vision_bias():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(H, E, generator = g)             # gate_tensor (hidden, E): logits = y @ W
    b_tx = torch.randn(E, generator = g) * 0.5       # gate.bias
    b_vl = torch.randn(E, generator = g) * 0.5       # gate.bias_vl
    counts = torch.randint(0, 10000, (E,), generator = g).tolist()
    perm = sorted(range(E), key = lambda e: -counts[e])  # load_cpu_split's static ordering
    assert perm != list(range(E))

    m = BlockSparseMLP.__new__(BlockSparseMLP)
    m.cpu_split_first = E - TAIL
    m._split_perm = perm
    m._split_dynamic = False
    m.routing_gate = SimpleNamespace(inner = SimpleNamespace(weight = w.clone(), bias = None))
    m.e_score_correction_bias = b_tx.clone()
    m.e_score_bias_vl = b_vl.clone()
    m.per_expert_scale = None
    m.cpu_post_load()

    p = torch.tensor(perm)
    assert torch.equal(m.routing_gate.inner.weight, w[:, p])
    assert torch.equal(m.e_score_correction_bias, b_tx[p])
    assert torch.equal(m.e_score_bias_vl, b_vl[p])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
