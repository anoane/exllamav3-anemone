"""The engram hook in DSV41Block, and the engram's registration in the model graph.

This test FAILS if the engram is not invoked. It builds DSV41Block for layers 0, 1
and 14 with torch-only stand-ins for attention, MoE and the norms (the hc mixer's
unweighted RMSNorm is swapped for a torch one, since the extension's needs a GPU;
the mixer's own mix_delayed / apply_ math runs in torch) and a spy engram, then
checks, per forward:
  - the spy runs exactly once on layers 1 and 14 and never on layer 0;
  - it runs BEFORE attn_hc.mix_delayed, on the (b, s, 4, 5120) fp32 streams;
  - what it returns is what the attention sublayer is built from: perturbing its
    output changes the attention input.
The same check is then run with the engram ablated (EXL3_DSV41_ABLATE=engram) and
must fail -- proof that the check can see a skipped engram. v4mix must change the
attention input of a layer that has a carried pre-mix.

Then the real DeepseekV41Model graph (from the checkpoint's config, NOT loaded):
  - the engram is a registered submodule of exactly blocks 1 and 14;
  - both engrams are in model.get_recurrent_layers() and model._get_prefetch_layers,
    and a Cache would give them DSV41EngramState under keys distinct from every
    other recurrent layer;
  - DSV41Engram.forward runs (no NotImplementedError) on CPU with a small stand-in
    wkv, real table rows gathered from disk, and moves the streams.
CPU only. The graph part needs the checkpoint and is skipped without it:

    PYTHONPATH=$PWD python3 tests/test_dsv41_block_hook_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""
import os, sys
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import ablate, model_dir, skip

from exllamav3.modules import Module

H, D = 4, 5120
LOG = []


class Stub(Module):
    """Attention / MoE stand-in: a fixed random map of the collapsed input."""
    def __init__(self, key, seed):
        super().__init__(None, key, None)
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(D, D, generator = g) * (0.3 / D ** 0.5)
        self.inputs = []
    def optimizer_targets(self): return []
    def forward(self, x, params, out_dtype = None):
        LOG.append((self.key, None))
        self.inputs.append(x.detach().clone())
        return torch.tanh(x.float() @ self.w)


class TorchNorm(Module):
    """Unweighted RMS norm in torch (the extension's needs a GPU)."""
    def __init__(self, key, eps = 1e-6):
        super().__init__(None, key, None)
        self.eps = eps
    def optimizer_targets(self): return []
    def forward(self, x, params, out_dtype = None):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim = True) + self.eps)
        return y.to(out_dtype or x.dtype)


class SpyEngram(Module):
    def __init__(self, key):
        super().__init__(None, key, None)
        self.delta = None
    def optimizer_targets(self): return []
    def forward(self, x, params, out_dtype = None):
        LOG.append((self.key, (tuple(x.shape), x.dtype)))
        return x.clone() if self.delta is None else x + self.delta
    def commit_only(self, params):
        LOG.append((self.key + ":commit_only", None))


def make_block(idx, engram):
    from exllamav3.modules.dsv41_block import DSV41Block, DSV41HyperConnection

    def hc(tag, seed):
        m = DSV41HyperConnection(None, f"layers.{idx}.hc_{tag}", hc_mult = H, hidden_size = D,
                                 sinkhorn_iters = 20, hc_eps = 1e-6, rms_norm_eps = 1e-6)
        m.modules.remove(m.norm)
        m.norm = TorchNorm(f"layers.{idx}.hc_{tag}.norm")
        m.register_submodule(m.norm)
        g = torch.Generator().manual_seed(seed)
        m.fn = torch.randn((2 + H) * H, H * D, generator = g) * 0.02
        m.base = torch.randn((2 + H) * H, generator = g) * 0.5
        m.scale = torch.rand(3, generator = g) + 0.5
        orig = m.mix_delayed
        def mix_delayed(streams, params, carried, *a, _o = orig, _k = m.key):
            LOG.append((_k, None))
            return _o(streams, params, carried, *a)
        m.mix_delayed = mix_delayed
        # the fused ext.hc_apply is CUDA-only; its torch form (HyperConnection.apply_'s fallback)
        m.apply_ = lambda x, y, post, comb, params: \
            post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x)
        return m

    return DSV41Block(
        config = None, key = f"layers.{idx}", layer_idx = idx,
        attn_norm = TorchNorm(f"layers.{idx}.attn_norm"), attn = Stub(f"layers.{idx}.attn", 100 + idx),
        attn_hc = hc("attn", 200 + idx),
        mlp_norm = TorchNorm(f"layers.{idx}.ffn_norm"), mlp = Stub(f"layers.{idx}.ffn", 300 + idx),
        mlp_hc = hc("ffn", 400 + idx),
        engram = engram,
    )


def check_hook(blocks, spies, x0):
    """One model-order pass through the blocks; asserts the hook contract."""
    params = {"dsv41_hc_pre": None}
    x = x0.clone()
    for idx, b in blocks.items():
        LOG.clear()
        x = b.forward(x, params)
        calls = [k for k, _ in LOG]
        eng = [i for i, (k, _) in enumerate(LOG) if k == f"layers.{idx}.engram"]
        if idx in spies:
            assert len(eng) == 1, f"layer {idx}: engram called {len(eng)} times ({calls})"
            mix = calls.index(f"layers.{idx}.hc_attn")
            assert eng[0] < mix, f"layer {idx}: engram ran after attn_hc.mix_delayed ({calls})"
            assert LOG[eng[0]][1] == ((1, 5, H, D), torch.float32), LOG[eng[0]][1]
        else:
            assert not eng and not any("engram" in k for k in calls), f"layer {idx}: {calls}"
    return x


def part_block():
    torch.manual_seed(0)
    spies = {1: SpyEngram("layers.1.engram"), 14: SpyEngram("layers.14.engram")}
    blocks = {i: make_block(i, spies.get(i)) for i in (0, 1, 14)}
    for i, s in spies.items():
        assert s in blocks[i].modules, f"layer {i}: engram is not a registered submodule"
    x0 = torch.randn(1, 5, H, D) * 0.5

    with ablate(""):
        check_hook(blocks, spies, x0)
    print("  OK  engram called exactly once on layers 1 and 14, never on 0, before "
          "attn_hc.mix_delayed, with (1, 5, 4, 5120) fp32 streams")

    # the attention input must be built from the engram's output
    carried = torch.rand(1, 5, H) + 0.1
    def attn_input(delta):
        spies[1].delta = delta
        blocks[1].attn.inputs.clear()
        blocks[1].forward(x0.clone(), {"dsv41_hc_pre": carried})
        spies[1].delta = None
        return blocks[1].attn.inputs[0].float()
    a0, a0b = attn_input(None), attn_input(None)
    a1 = attn_input(torch.randn(1, 5, H, D) * 0.5)
    assert torch.equal(a0, a0b), "stand-in forward is not deterministic"
    d = ((a1 - a0).norm() / a0.norm()).item()
    assert d > 1e-2, f"perturbing the engram output moved the attention input by only {d:.2e}"
    print(f"  OK  perturbing the engram output moves the attention input by rel-L2 {d:.3f}")

    # negative control: with the engram ablated the same check must fail
    with ablate("engram"):
        try:
            check_hook(blocks, spies, x0)
            failed = False
        except AssertionError as e:
            failed = True
            why = str(e)
    commits = [k for k, _ in LOG if k.endswith(":commit_only")]
    assert failed, "the hook check passed with the engram ablated -- it cannot see a skipped engram"
    print(f"  OK  negative control: with EXL3_DSV41_ABLATE=engram the check fails ({why[:60]}...)")
    assert commits == ["layers.1.engram:commit_only"], commits
    print("  OK  an ablated engram still commits its n-gram context (commit_only)")

    # v4mix collapses with the site's own pre-mix: a different attention input
    blocks[1].attn.inputs.clear()
    pre = torch.rand(1, 5, H) + 0.1
    blocks[1].forward(x0.clone(), {"dsv41_hc_pre": pre})
    with ablate("v4mix"):
        blocks[1].forward(x0.clone(), {"dsv41_hc_pre": pre})
    a, b = blocks[1].attn.inputs[0].float(), blocks[1].attn.inputs[1].float()
    d = ((a - b).norm() / a.norm()).item()
    assert d > 1e-3, f"v4mix changed the attention input by only {d:.2e}"
    print(f"  OK  v4mix ablation changes the attention input by rel-L2 {d:.3f}")


class FakeWkv(Module):
    """Small stand-in for the exl3 wkv: rank-16 fp32 map 6144 -> 25600 on the CPU."""
    def __init__(self):
        super().__init__(None, "fake_wkv", None)
        g = torch.Generator().manual_seed(1)
        self.a = torch.randn(6144, 16, generator = g) / 6144 ** 0.5
        self.b = torch.randn(16, 5 * D, generator = g)
        self.inputs = []
    def optimizer_targets(self): return []
    def forward(self, x, params, out_dtype = None):
        self.inputs.append(x)
        return ((x.float() @ self.a) @ self.b).to(out_dtype or torch.half)


def part_model(checkpoint_dir):
    from exllamav3 import Config, Model
    from exllamav3.modules.dsv41_engram import DSV41Engram
    from exllamav3.architecture.dsv41.engram_state import DSV41EngramState

    cfg = Config.from_directory(checkpoint_dir)
    model = Model.from_config(cfg)
    blocks = model.modules[model.first_block_idx: model.first_block_idx + cfg.num_hidden_layers]
    owners = {}
    for i, b in enumerate(blocks):
        for m in b:
            if isinstance(m, DSV41Engram):
                owners.setdefault(m, []).append(i)
                assert m in b.modules, f"layer {i}: engram found but not a direct submodule"
    assert sorted(i for v in owners.values() for i in v) == list(cfg.engram_layer_ids) == [1, 14], owners
    assert all(len(v) == 1 for v in owners.values()), owners
    top = [m for m in model.modules if isinstance(m, DSV41Engram)]
    assert not top, "an engram sits in the top-level module list"
    egs = list(owners)
    print(f"  OK  engram registered under exactly blocks {sorted(v[0] for v in owners.values())}")

    rl = model.get_recurrent_layers()
    pf = model._get_prefetch_layers
    for eg in egs:
        assert eg in rl and eg in pf, eg.key
        assert eg.layer_state_cls is DSV41EngramState
    keys = [(m.layer_idx, 0) for m in rl]
    assert len(set(keys)) == len(keys)
    states = {(m.layer_idx, 0): m.layer_state_cls(m, 2, 0, 0) for m in egs}    # as Cache() does
    assert all(isinstance(s, DSV41EngramState) for s in states.values())
    print(f"  OK  both engrams in get_recurrent_layers() ({len(rl)} layers) and _get_prefetch_layers; "
          f"Cache would key them {sorted(states)}")

    eg = model.engram_modules[1]
    eg.open_table()
    eg.load_gate(torch.device("cpu"))
    eg.wkv = FakeWkv()
    ids = torch.tensor([[0, 3, 671, 6102, 14, 344, 17]])
    x = torch.randn(1, 7, H, D) * 0.5
    try:
        y = eg.forward(x.clone(), {"input_ids": ids})
    except NotImplementedError as e:
        raise AssertionError(f"DSV41Engram.forward still raises NotImplementedError: {e}")
    assert y.shape == x.shape and y.dtype == torch.float32 and torch.isfinite(y).all()
    d = ((y - x).norm() / x.norm()).item()
    emb = eg.wkv.inputs[0]
    assert emb.shape == (1, 7, 6144) and emb.dtype == torch.half and (emb != 0).any()
    assert d > 1e-3, f"engram forward moved the streams by only {d:.2e}"
    print(f"  OK  DSV41Engram.forward on CPU (stand-in wkv, real table rows): streams moved by rel-L2 {d:.3f}")

    # no token ids: only the autosplit's measuring pass may run (it hashes stand-ins and
    # reads nothing); a real forward must refuse rather than inject a wrong n-gram term
    try:
        eg.forward(x.clone(), {})
        raise AssertionError("DSV41Engram.forward ran without params['input_ids']")
    except KeyError as e:
        assert "input_ids" in str(e), e
    ym = eg.forward(x.clone(), {"autosplit_measure": True})
    assert ym.shape == x.shape and torch.isfinite(ym).all()
    print("  OK  a forward without input_ids raises; the autosplit measuring pass still runs")


def main(checkpoint_dir):
    part_block()
    if checkpoint_dir is None:
        skip("engram registration in the model graph", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        return
    part_model(checkpoint_dir)
    print("  OK  engram hook and registration")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else model_dir())
