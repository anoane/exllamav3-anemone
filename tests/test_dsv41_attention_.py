"""CPU-only check of V4.1 attention: who builds what, who reads whose pool, and the pool math.

Loads exllamav3/modules/dsv41.py against stand-ins for the pieces that need a compiled
extension (the V4 base class, the dsa kernels, the cache classes), so the
construction rules, the source resolution and the cached-path bookkeeping run without a GPU
or a checkpoint load. The stub base records what it was driven with, which is itself part of
what is being checked: V4.1 has no layer kinds, so the base class must be given the kind that
yields the right rope table and caps.

The rope stand-in is a real GPT-J rotation honouring ext.rope's contract: position /
position_ids with an inv_freq vector, or a per-position angle table (the mode every V4.1
rotation uses). A no-op stand-in is how pool entries rotated first_group * m + j apart instead
of (first_group + j) * m could go unnoticed; the pool checks below fail on that, and on the
index-K / rope aliasing and padded head-weight bugs. Selection runs the real
modules/dsv41_select.py (torch backend on the CPU).

    python tests/test_dsv41_attention_.py [checkpoint-dir]

The layer topology comes from the checkpoint's config.json when a directory is given (or
DSV41_MODEL_DIR is set), and is DeepSeek-V4.1-Flash's otherwise.
"""
import json, os, sys, types, zlib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import ablate, load_package_file, model_dir

_P = "dsv41stub"


def _seeded(key, *shape):
    g = torch.Generator().manual_seed(zlib.crc32(key.encode()))
    return torch.randn(*shape, generator = g)


class _StubLinear:
    """Linear stand-in: a lazily created random (in, out) weight, exllamav3's output-padding
    rule (pad_to 128 unless trimmed) and out_dtype handling."""
    def __init__(self, config, key, in_features = None, out_features = None, *, out_dtype = None,
                 trim_padded_out = False, pad_to = 128, **kw):
        self.key = key
        self.in_features = in_features
        self.out_features_unpadded = out_features
        self.out_features = -(-out_features // pad_to) * pad_to if out_features else out_features
        self.out_dtype = out_dtype
        self.trim_padded_out = trim_padded_out
        self._w = None
    def weight(self):
        if self._w is None:
            w = torch.zeros(self.in_features, self.out_features)
            w[:, :self.out_features_unpadded] = _seeded(self.key, self.in_features,
                                                        self.out_features_unpadded) * self.in_features ** -0.5
            self._w = w
        return self._w
    def forward(self, x, params, out_dtype = None, **kw):
        y = (x.float() @ self.weight()).to(out_dtype or self.out_dtype or torch.half)
        if self.trim_padded_out:
            y = y[..., :self.out_features_unpadded].contiguous()
        return y


class _StubNorm:
    """RMSNorm stand-in, unweighted (weight None), so every caller -- including the compressor,
    which hands norm.weight to compress_chunk -- sees the same function."""
    def __init__(self, config, key, eps = 1e-6, *a, **kw):
        self.key, self.eps = key, eps
        self.weight = None
    def forward(self, x, params, out_dtype = None, **kw):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + self.eps)
        return y.to(out_dtype or x.dtype)


class _StubBase:
    """Stands in for DSV4Attention: records its kwargs, registers nothing real."""
    def __init__(self, config, key, layer_idx, layer_type, **kwargs):
        self.config, self.key, self.layer_idx = config, key, layer_idx
        self.layer_type = layer_type
        self.base_kwargs = kwargs
        self.head_dim = kwargs["head_dim"]
        self.num_q_heads = kwargs.get("num_q_heads", 64)
        self.rope_head_dim = 64
        self.index_n_heads = kwargs.get("index_n_heads")
        self.index_head_dim = kwargs.get("index_head_dim")
        self.index_topk = kwargs.get("index_topk")
        self.compress_rate = kwargs.get("compress_rate")
        self.sliding_window = 128
        self.idx_wq_b = self.idx_weights = None
        self.modules = []
        self.submodules = []
        self.caps = {"recurrent_cache": True}
        if layer_type in ("csa", "hca"):
            self.caps["kv_cache"] = True
        self.device = None
    def register_submodule(self, m):
        self.submodules.append(m)
        self.modules.append(m)
    def load(self, device, **kw):
        self.device = device
    def unload(self):
        self.device = None


def _torch_rope(x, inv_freq, position = 0, position_ids = None):
    """ext.rope's contract on a (bsz, seq, heads, rd) tensor or trailing-slice view, in place:
    token t of row b rotates at p = position_ids[b, t] when given, else at position + t. With an
    inv_freq vector the angle is p * inv_freq; with a (bsz, n, rd/2) table (table mode) it is
    the table's row p of batch row b."""
    bsz, seq = x.shape[0], x.shape[1]
    pos = position_ids.long() if position_ids is not None else \
        (position + torch.arange(seq)).unsqueeze(0).expand(bsz, seq)
    if inv_freq.dim() > 1:
        tab = inv_freq.reshape(bsz, -1, inv_freq.shape[-1])
        ang = torch.stack([tab[b][pos[b]] for b in range(bsz)]).double().unsqueeze(-2)  # (b, s, 1, rd/2)
    else:
        ang = pos.double().unsqueeze(-1).unsqueeze(-1) * inv_freq.double()          # (b, s, 1, rd/2)
    c, s = torch.cos(ang), torch.sin(ang)
    xd = x.double()
    x0, x1 = xd[..., 0::2].clone(), xd[..., 1::2].clone()
    out = torch.empty_like(xd)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    x.copy_(out.to(x.dtype))


class _CacheLayerQuant:
    pass


def _pkg(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _DeviceMemo:
    @staticmethod
    def get(params, key, t, device):
        return t if t is None or device is None else t.to(device)


def _install_stubs():
    _pkg(_P)
    _pkg(f"{_P}.modules")
    _pkg(f"{_P}.modules.attention_fn")
    _pkg(f"{_P}.util")
    _pkg(f"{_P}.architecture")
    _pkg(f"{_P}.architecture.dsv41")
    _pkg(f"{_P}.cache", CacheLayer_quant = _CacheLayerQuant)
    _pkg(f"{_P}.cache.dsv41", CacheLayer_dsv41 = type("CacheLayer_dsv41", (), {}),
         DSV41LayerState = type("DSV41LayerState", (), {}), DeviceMemo = _DeviceMemo)
    _pkg(f"{_P}.modules.linear", Linear = _StubLinear)
    _pkg(f"{_P}.modules.rmsnorm", RMSNorm = _StubNorm)
    _pkg(f"{_P}.modules.dsv4", DSV4Attention = _StubBase, _ext_rope = _torch_rope)
    _pkg(f"{_P}.modules.attention_fn.dsa_triton", dsa_attn = lambda *a, **kw: None)
    _pkg(f"{_P}.util.tensor", get_for_device = None)
    _pkg(f"{_P}.util.rope", RopeStyle = types.SimpleNamespace(GPTJ = 1))
    _pkg(f"{_P}.ext", exllamav3_ext = None)
    _pkg(f"{_P}.constants", PAGE_SIZE = 256)
    load_package_file("exllamav3/util/device_copy.py", f"{_P}.util.device_copy")
    # the pooling math, the selection and the cached-path helpers are the real files
    load_package_file("exllamav3/architecture/dsv41/compressor.py", f"{_P}.architecture.dsv41.compressor")
    load_package_file("exllamav3/architecture/dsv41/candidates.py", f"{_P}.architecture.dsv41.candidates")
    load_package_file("exllamav3/modules/dsv41_ablation.py", f"{_P}.modules.dsv41_ablation", f"{_P}.modules")
    load_package_file("exllamav3/modules/dsv41_select.py", f"{_P}.modules.dsv41_select", f"{_P}.modules")
    load_package_file("exllamav3/modules/dsv41_cached.py", f"{_P}.modules.dsv41_cached", f"{_P}.modules")


# DeepSeek-V4.1-Flash, used when no checkpoint is given
_FALLBACK = {
    "num_hidden_layers": 40,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
}


def main(path = None):
    _install_stubs()
    dsv41 = load_package_file("exllamav3/modules/dsv41.py", f"{_P}.modules.dsv41", f"{_P}.modules")
    dc = sys.modules[f"{_P}.modules.dsv41_cached"]

    t = _FALLBACK
    if path:
        c = json.load(open(os.path.join(path, "config.json")))
        c = c.get("text_config", c)
        t = {k: c[k] for k in _FALLBACK if k in c}
        for k, v in _FALLBACK.items():
            t.setdefault(k, v)

    ratios = t["compress_ratios"]
    kvs, ixs = set(t["kv_source_layer_ids"]), set(t["index_source_layer_ids"])
    cand = t["candidate_source_layer_id"]
    assert len(ratios) >= t["num_hidden_layers"], "compress_ratios shorter than the model"
    config = types.SimpleNamespace(candidate_source_layer_id = cand, candidate_block_size = 8,
                                   candidate_topk_blocks = 2048)

    def kv_src(i):
        return max((s for s in kvs if s <= i), default = None) if ratios[i] else None

    def ix_src(i):
        return max(s for s in ixs if s <= i) if ratios[i] else None

    layers, built = {}, {}
    for i in range(t["num_hidden_layers"]):
        a = dsv41.DSV41Attention(
            config, f"layers.{i}.attn", i,
            compress_ratio = ratios[i],
            is_kv_source = i in kvs,
            is_index_source = i in ixs,
            index_owns_k = (i in ixs and i in kvs),
            kv_source_layer = kv_src(i),
            index_source_layer = ix_src(i),
            uses_candidates = 0 <= cand < i,
            hidden_size = 5120, q_lora_rank = 1280, head_dim = 512,
            index_n_heads = 32, index_head_dim = 128, index_topk = 512,
            rms_norm_eps = 1e-20,
        )
        layers[i] = a
        built[i] = set(a.expected_keys())

    # the base class must be driven with the kind that gives the right rope and caps,
    # and never with V4's own compressors
    for i, a in layers.items():
        assert a.layer_type == ("sliding" if ratios[i] == 0 else "hca"), \
            f"layer {i}: base kind {a.layer_type} for rate {ratios[i]}"
        assert a.base_kwargs["tp_defer_compressors"] is True, \
            f"layer {i}: V4's compressors were not suppressed"
        assert a.base_kwargs["compress_rate"] == (ratios[i] or None), \
            f"layer {i}: base compress_rate {a.base_kwargs['compress_rate']}"

    # only sources carry weights, and wgate/wk only where the checkpoint has them
    for i, keys in built.items():
        k = f"layers.{i}.attn"
        assert (f"{k}.compressor.wkv" in keys) == (i in kvs), f"layer {i}: compressor presence"
        assert (f"{k}.compressor.wgate" in keys) == (i in kvs and ratios[i] == 2), \
            f"layer {i}: wgate exists only on rate-2 kv sources"
        assert (f"{k}.indexer.wq_b" in keys) == (i in ixs), f"layer {i}: indexer presence"
        assert (f"{k}.indexer.wk" in keys) == (i in ixs and i in kvs), \
            f"layer {i}: index K is owned only by index sources that are kv sources"
        if i in ixs:
            assert layers[i].idx_wq_b is not None and layers[i].idx_weights is not None, \
                f"layer {i}: inherited scoring was not aliased to the V4.1 indexer"
            # the scoring kernel reads (seq, n_heads) contiguous head weights: a padded
            # 128-wide projection handed row r the weights of row r / 4, or zeros
            w = layers[i].idx_weights
            assert w.trim_padded_out and w.out_features != w.out_features_unpadded, \
                f"layer {i}: weights_proj must trim its padded output"

    # cached-path ownership: one pool per kv source, a ring on every layer
    for i, a in layers.items():
        assert bool(a.caps.get("kv_cache")) == (i in kvs), f"layer {i}: kv_cache cap"
        assert a.layer_state_cls.__name__ == "DSV41LayerState", f"layer {i}: layer state class"
        assert a.kv_source_layer == kv_src(i), f"layer {i}: the pool it reads"
        assert a.is_candidate_source == (i == cand), f"layer {i}: candidate source flag"
    lt, kw = layers[2].cache_layer_type(object, {})
    assert lt.__name__ == "CacheLayer_dsv41" and kw == {}
    lt, kw = layers[2].cache_layer_type(_CacheLayerQuant, {"k_bits": 5, "v_bits": 5})
    assert kw == {"k_bits": 5, "v_bits": 5}, kw
    # a consumer owns no pool: asking it for a cache layer class is a bug, not a pool
    try:
        layers[13].cache_layer_type(object, {})
    except AssertionError:
        pass
    else:
        raise AssertionError("a consumer was given a pool of its own")

    # stateless source resolution: every consumer reads the source below it, and a pool
    # published by the wrong source is refused
    params = {}
    for i, a in layers.items():
        if a.is_kv_source:
            params.setdefault("dsv41_pools", {})[i] = dsv41.SourcePools(
                torch.zeros(4, 448, dtype = torch.half),
                torch.zeros(4, 64, dtype = torch.half),
                torch.zeros(4, 128, dtype = torch.half), 4, ratios[i], 0)
    groups = {}
    for i, a in layers.items():
        src = a.resolve_pools(params)
        if ratios[i] == 0:
            assert src is None, f"layer {i}: a sliding layer resolved a pool"
            continue
        assert a.kv_source_layer == kv_src(i), f"layer {i}: wrong source"
        groups.setdefault(a.kv_source_layer, []).append(i)
        if a.is_index_source:
            assert a.resolve_index_pool(params).shape[-1] == 128, f"layer {i}: index pool"

    # a source that did not publish must be refused, not silently attended over nothing
    victim = layers[max(kvs)]
    holed = {"dsv41_pools": {k: v for k, v in params["dsv41_pools"].items()
                             if k != victim.kv_source_layer}}
    for i in groups[victim.kv_source_layer]:
        try:
            layers[i].resolve_pools(holed)
        except RuntimeError as e:
            assert "has not published" in str(e), str(e)
        else:
            raise AssertionError(f"layer {i}: missing source went unreported")

    # rate-mismatched pool is refused too
    bad = {"dsv41_pools": dict(params["dsv41_pools"])}
    bad["dsv41_pools"][2] = dsv41.SourcePools(None, None, None, 0, 1, 0)
    try:
        layers[3].resolve_pools(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("rate-2 layer accepted a rate-1 pool")

    # the stateless forward refuses what it cannot compute: rows past the int32 addressing of
    # ext.rope / dsa_attn (65,536 at 64 x 512), any start past position 0 (it keeps no
    # history, so later rows would attend entries of the wrong tokens), and a batch (it
    # publishes one pool per source, so rows past the first would attend row 0's), on every
    # layer, sliding ones included, not only once a kv source is reached
    fake = types.SimpleNamespace(shape = (1, 65536, 5120), device = torch.device("cpu"))
    short = lambda b: types.SimpleNamespace(shape = (b, 8, 5120), device = torch.device("cpu"))
    for L, x_, p_, why in ((3, fake, {}, "int32"), (21, fake, {}, "int32"),
                           (3, short(1), {"position": 100}, "position 0"),
                           (0, short(2), {}, "one sequence"), (2, short(2), {}, "one sequence"),
                           (21, short(2), {}, "one sequence")):
        try:
            layers[L]._forward_nc(x_, p_, None)
        except ValueError as e:
            assert why in str(e), str(e)
        else:
            raise AssertionError(f"layer {L}: _forward_nc accepted {x_.shape} at {p_}")
    assert dsv41.check_row_limit("x", 65535, 64, 512) is None

    # rope convention, verified against the reference kernel's cos/sin index: the entry that
    # holds token p rotates at the first token of its group, and a group closes on the token
    # with (p + 1) % m == 0
    for m in (1, 2):
        for p in range(8):
            assert int(dc.rope_positions(torch.tensor(p // m), m)) == (p // m) * m
            e0, e1 = dc.emission_range(p, 1, m)
            assert (e1 > e0) == ((p + 1) % m == 0) and e0 == p // m

    # ---- pool math with a real rotation: stateless publish_pools and cached _store_entries
    torch.manual_seed(0)
    inv = 1.0 / (160000.0 ** (torch.arange(0, 64, 2).float() / 64))
    for L in (2, 20):
        a = layers[L]
        a.inv_freq_compress = inv
        m, hd, rd = ratios[L], 512, 64
        x = (torch.randn(1, 37, 5120) * 0.5).half()
        pools = a.publish_pools(x, {}, 0)
        latent, first = a.compressor.forward(x[0], {}, 0)
        n = pools.entries
        pos_ok = (first + torch.arange(n)) * m
        pos_old = first * m + torch.arange(n)
        want_r = dc.rope_gptj_torch(latent[:, hd - rd:].half().float(), inv, pos_ok)
        old_r = dc.rope_gptj_torch(latent[:, hd - rd:].half().float(), inv, pos_old)
        e_ok = ((pools.pool_r[:n].float() - want_r).norm() / want_r.norm()).item()
        e_old = ((pools.pool_r[:n].float() - old_r).norm() / old_r.norm()).item()
        assert e_ok < 2e-3, f"L{L}: pool_r vs (first + i) * m rel {e_ok:.2e}"
        if m == 2:
            assert e_old > 0.1, f"L{L}: the check cannot tell the old positions apart ({e_old:.2e})"
        k = a.indexer.produce_k(latent, {}).half().float()
        want_k = k.clone()
        want_k[:, -rd:] = dc.rope_gptj_torch(k[:, -rd:], inv, pos_ok)
        e_k = ((pools.pool_idx[:n].float() - want_k).norm() / want_k.norm()).item()
        assert e_k < 2e-3, f"L{L}: pool_idx vs rope(k_norm(wk(latent))) rel {e_k:.2e}"

        # cached store through a random block table, for a 1-entry step (decode) and a
        # 5-entry one: same values at entry rows, and the caller's latent left untouched
        epp = 256 // m
        P = 8
        kl = types.SimpleNamespace(
            pool_c = torch.zeros(P, epp, 448, dtype = torch.half), pool_r = torch.zeros(P, epp, 64, dtype = torch.half),
            pool_idx = torch.zeros(P, epp, 128, dtype = torch.half), epp = epp, D_c = 448, D_r = 64, D_i = 128,
            quant = False)
        bt = torch.randperm(P).int().unsqueeze(0)
        for e0, cnt in ((epp - 1, 1), (3 * epp - 2, 5)):
            lat16 = latent[:cnt].half().clone()          # an FP16 latent: rope must not touch it
            before = lat16.clone()
            a._store_entries(lat16, e0, kl, bt, e0 * m, cnt * m, {})
            assert torch.equal(lat16, before), f"L{L}: _store_entries rotated the caller's latent (n={cnt})"
            e = torch.arange(e0, e0 + cnt)
            rows = dc.entry_rows(bt, e, epp)
            pos = e * m
            got_r = kl.pool_r.view(-1, 64)[rows].float()
            ref_r = dc.rope_gptj_torch(before[:, hd - rd:].float(), inv, pos)
            assert ((got_r - ref_r).norm() / ref_r.norm()).item() < 2e-3, f"L{L}: stored pool_r"
            assert torch.equal(kl.pool_c.view(-1, 448)[rows], before[:, :448]), f"L{L}: stored pool_c"
            kk = a.indexer.produce_k(before, {}).half().float()
            kk[:, -rd:] = dc.rope_gptj_torch(kk[:, -rd:], inv, pos)
            got_k = kl.pool_idx.view(-1, 128)[rows].float()
            assert ((got_k - kk).norm() / kk.norm()).item() < 2e-3, \
                f"L{L}: stored index K is not rope(k_norm(wk(PRE-RoPE latent))) (n={cnt})"

    # ---- selection routing: the candidate source publishes, users mask, a missing publish
    # raises, and the head weights reach select at (seq, 32) scaled by 128^-0.5 * 32^-0.5
    calls = []
    def fake_select(q, w, pool, **kw):
        calls.append((q.shape, w.clone(), kw))
        return sel.SelectResult(torch.zeros(q.shape[0], 512, dtype = torch.int32), 512,
                                torch.zeros(q.shape[0], 2048, dtype = torch.int32) if kw["want_cand"] else None)
    sel = sys.modules[f"{_P}.modules.dsv41_select"]
    real = dsv41.select_topk
    dsv41.select_topk = fake_select
    try:
        for L in (20, 24):
            layers[L].inv_freq_compress = inv
        x = (torch.randn(1, 5, 5120) * 0.5).half()
        q_res = torch.randn(1, 5, 1280).half()
        pool = torch.zeros(700, 128, dtype = torch.half)
        p = {}
        try:
            layers[24]._indexer_topk(x, p, q_res, pool, 700, 695)
        except RuntimeError as e:
            assert "candidate source" in str(e), str(e)
        else:
            raise AssertionError("a candidate user ran without its source's blocks")
        layers[20]._indexer_topk(x, p, q_res, pool, 700, 695)
        assert calls[-1][2]["want_cand"] and calls[-1][2]["cand_in"] is None
        assert p["dsv41_candidates"][0] is not None
        w_expect = layers[20].idx_weights.forward(x, {})[0].float() * (128 ** -0.5 * 32 ** -0.5)
        assert calls[-1][1].shape == (5, 32) and torch.equal(calls[-1][1], w_expect), "head weights"
        layers[24]._indexer_topk(x, p, q_res, pool, 700, 695)
        assert not calls[-1][2]["want_cand"] and calls[-1][2]["cand_in"] is p["dsv41_candidates"][0]
        with ablate("no_candidates", sys.modules[f"{_P}.modules.dsv41_ablation"]):
            layers[24]._indexer_topk(x, {}, q_res, pool, 700, 695)
            assert calls[-1][2]["cand_in"] is None
    finally:
        dsv41.select_topk = real

    gs = ", ".join(f"{{{v[0]}-{v[-1]}}}" for v in groups.values())
    print(f"  OK  attention topology: {len(layers)} layers, kv groups {gs}; "
          f"{len(kvs)} compressors, {len(ixs)} indexers, "
          f"{sum(1 for i in ixs if i in kvs)} own K; a missing source refused; int32 rows, position > 0 "
          f"and a stateless batch refused; pool ownership and "
          f"cache classes; pool rope at (first + i) * m (old positions fail); cached store keeps "
          f"the latent and derives K pre-RoPE; candidates routed; head weights unpadded")


def test_attention():
    """pytest entry point: the same checks, with the topology of DSV41_MODEL_DIR if set."""
    main(model_dir())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else model_dir())
