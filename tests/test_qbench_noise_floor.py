"""
qbench noise floor with an exllamav3 reference: Exl3Backend.run streams modules one at a time and loads
prefer_cpu modules (the embedding, always module 0) on CPU, so their output is a CPU tensor. The noise draw
must use a generator on the activation's device; a single generator on the compute device made the very
first draw fail ("Expected a 'cpu' device type for generator but found 'cuda'").

No model is needed (the native extension must be importable, as for any exllamav3 test): the backend is
created via __new__ around a stub module chain. Generator creation is recorded, so every noise draw is checked
against the activation's device even on a CPU-only host, where the stubs all run on CPU.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))

import torch

import qbench.engines
from qbench.engines import Exl3Backend

# qbench.measure.BF16_ROUNDING_EPS, not imported since qbench.measure depends on the datasets package
EPS = 2 ** -9
DIM = 64


class _StubModule:
    def __init__(self, key, weight, prefer_cpu = False):
        self.key = key
        self.caps = {"prefer_cpu": True} if prefer_cpu else {}
        self.modules = []
        self.weight = weight
        self.device = None

    def load(self, device):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    def unload(self):
        self.device = None

    def prepare_for_device(self, x, params):
        return x.to(self.device)

    def forward(self, x, params):
        w = self.weight.to(self.device)
        return w[x] if not x.is_floating_point() else x @ w


def _backend(device):
    g = torch.Generator().manual_seed(0)
    modules = [
        _StubModule("model.embed_tokens", torch.randn(32, DIM, generator = g), prefer_cpu = True),
        _StubModule("model.layers.0", torch.randn(DIM, DIM, generator = g) / DIM ** 0.5),
        _StubModule("model.norm", torch.eye(DIM)),
        _StubModule("lm_head", torch.randn(DIM, 32, generator = g)),
    ]
    stc = SimpleNamespace(begin_deferred_load = lambda: None, end_deferred_load = lambda: None, close = lambda: None)
    backend = Exl3Backend.__new__(Exl3Backend)
    backend.device = device
    backend.config = SimpleNamespace(stc = stc)
    backend.model = SimpleNamespace(modules = modules, prepare_inputs = lambda row, params: row)
    backend.info = None
    return backend


def _run(noise_eps):
    backend = _backend(torch.device("cuda", 0))
    ids = torch.randint(0, 32, (3, 16), generator = torch.Generator().manual_seed(1))
    rows = {}
    def callback(r, logits):
        rows[r] = logits.cpu()
    backend.run(ids, callback, noise_eps = noise_eps)
    return torch.cat([rows[r] for r in range(ids.shape[0])])


@torch.inference_mode()
def test_noise_floor_cpu_embedding(monkeypatch):
    # Record the device each generator is requested for (backed by a CPU generator when CUDA is unavailable),
    # and check it against the activation's device at every noise draw
    requested = {}
    real_generator = torch.Generator
    def generator(device = "cpu"):
        device = torch.device(device)
        g = real_generator(device = device if device.type != "cuda" or torch.cuda.is_available() else "cpu")
        requested[id(g)] = device
        return g
    real_noise = qbench.engines.apply_mult_noise
    draws = []
    def apply_mult_noise(x, eps, gen):
        assert requested[id(gen)] == x.device, f"generator on {requested[id(gen)]}, activation on {x.device}"
        draws.append(x.device)
        return real_noise(x, eps, gen)
    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(qbench.engines, "apply_mult_noise", apply_mult_noise)

    clean = _run(None)
    assert not draws
    noisy = _run(EPS)
    # Embedding and layer outputs of each row are perturbed, the embedding's on CPU
    assert len(draws) == 3 * 2 and draws[0].type == "cpu"
    assert noisy.shape == clean.shape == (3, 16, 32)
    assert torch.isfinite(noisy).all()
    rel = ((noisy - clean).norm() / clean.norm()).item()
    assert 0.0 < rel < 20 * EPS
    # Seeded: the floor is reproducible
    assert torch.equal(noisy, _run(EPS))
