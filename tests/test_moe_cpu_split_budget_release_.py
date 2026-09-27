"""
BlockSparseMLP_CPU.cpu_unload must give back the layer's slot in the split-layer budget
(EXL3_MOE_CPU_SPLIT_LAYERS, counted in infer_params.moe_cpu_split_assigned), like it does for the
whole-layer offload budget. Regression: the slot was claimed on load and never released, so an
autosplit OOM retry of a split layer, or a Model.unload() + Model.load() of the same config, split
fewer layers than requested and silently loaded their experts to VRAM.
"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

NUM_EXPERTS = 8
SPLIT_K = 4
SPLIT_LAYERS = 2


def _stub(ip):
    # Real class without weights. Only the weight-touching registration (load_cpu_split) is
    # replaced; it changes the same module state that cpu_unload restores
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    m.config = SimpleNamespace(infer_params = ip)
    m.num_experts = NUM_EXPERTS
    m.num_local_experts = NUM_EXPERTS
    m.routing_first = None
    m.routing_last = None
    m.gated = True
    m.activation_fn = "silu"
    m.gates, m.ups, m.downs, m.modules = [], [], [], []
    m._cpu_init_state()

    def load_cpu_split(device, split_k, **kwargs):
        m._split_saved = (m.gates, m.ups, m.downs, m.modules,
                          m.num_local_experts, m.routing_first, m.routing_last)
        m.num_local_experts = NUM_EXPERTS - split_k
        m.routing_first = 0
        m.routing_last = NUM_EXPERTS - split_k
        m.cpu_split_first = NUM_EXPERTS - split_k
        return True

    m.load_cpu_split = load_cpu_split
    return m


def _layers(n = 4):
    ip = SimpleNamespace(moe_cpu_split = SPLIT_K, moe_cpu_offload = 0, draft_moe_cpu_offload = 0)
    return ip, [_stub(ip) for _ in range(n)]


def _load(m):
    m.cpu_maybe_split_load("cuda:0")
    return m.cpu_split_first is not None


@pytest.fixture(autouse = True)
def _split_layers_env(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_SPLIT_LAYERS", str(SPLIT_LAYERS))


def test_autosplit_retry_reclaims_slot():
    # Autosplit: layer 1 is split on one device, OOMs later in its load, is unloaded and
    # retried on the next device
    ip, layers = _layers()
    split = [_load(layers[0])]
    assert _load(layers[1])
    layers[1].cpu_unload()
    split += [_load(m) for m in layers[1:]]
    assert split == [True, True, False, False]
    assert ip.moe_cpu_split_assigned == SPLIT_LAYERS


def test_reload_same_config_splits_same_layers():
    ip, layers = _layers()
    first = [_load(m) for m in layers]
    layers[0].cpu_unload()
    layers[0].cpu_unload()  # repeated unload must not release a slot twice
    layers[3].cpu_unload()  # unsplit layer holds no slot
    assert ip.moe_cpu_split_assigned == SPLIT_LAYERS - 1
    for m in layers:
        m.cpu_unload()
    assert ip.moe_cpu_split_assigned == 0
    second = [_load(m) for m in layers]
    assert first == second == [True, True, False, False]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
