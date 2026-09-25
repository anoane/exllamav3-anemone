"""CPU checks of the DeepSeek-V4.1 cached (generation) path. No GPU: CUDA_VISIBLE_DEVICES is
cleared before torch loads, and the run fails if a CUDA context appears anyway.

    PYTHONPATH=$PWD python3 tests/test_dsv41_cached_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

1. Cache for the real V4.1 graph (built lazily from the checkpoint config, nothing loaded):
   exactly the four kv-source pools, their byte sizes at 1M tokens, 40 V4.1 layer states,
   the rate-2 carry rings; then the pool route of an explicit placement's split at 12 adds
   exactly one replica pool.
2. emission_range / rope_positions / entry_rows vs brute force of the (p + 1) % m == 0 rule
   through random block tables; emission_range, the range the cached forward stores, against
   the range DeepSeek's reference Compressor writes (its model.py) and the range exllamav3's
   chunked compress_chunk emits.
3. CompressCarry vs architecture/dsv41/compressor.compress_reference: random chunkings with
   odd boundaries, decode steps and rewinds up to the reported rollback limit, bitwise; a
   rewind past the limit must break it.
4. Selection: the per-forward registry on the real graph (every compressed layer reads
   index_source_for(i); a missing publish raises; dense_consumers yields None), and an
   untiled torch.topk selection (this file's oracle of the selection contract) vs DeepSeek's
   own Indexer.forward, candidates included. The production selector breaks exact ties toward
   the lower index where torch.topk leaves the order unspecified; tests/test_dsv41_select_.py
   checks it against DeepSeek's select_candidate_blocks with that tie rule.
5. Rope positions of pool entries: (g + j) * m, checked against DeepSeek's
   apply_rotary_emb/precompute_freqs_cis with exllamav3's yarn table; the old g * m + j must
   fail, and the ablation token must restore it.
6. ring_update vs a naive full history over random chunk sequences with shifts, rebases and
   rewinds bounded by DSV41State.rollback_capacity; V4's looser bound must fail.
7. DSV41LayerState stash/unstash round trip, and the check that a layer's pool is on its
   device (the error names the free cuts and the explicit placement's split).
8. Checkpoint bytes on the real graph: a DSV41State checkpoint declares exactly the bytes it
   stashes (the generator budgets host RAM by that figure), the SWA ring keeps only its
   position - window_beg history rows, and a trimmed checkpoint restores into another slot.
9. The int32 pool limit: a source pool (and a replica) at the largest addressable
   max_num_tokens builds, one page more is refused, for fp16 pools and for quantized (-cq) pools
   at their packed widths.
10. DeviceMemo (the per-forward moves to another device): a tensor moves once
   per (key, device) and forward, a republished one is moved again, and the move is the
   engine's to_device.

Oracles from DeepSeek's reference inference code are imported as-is from DSV41_DEEPSEEK_REF
(dsv41_ref.deepseek.load_model: its GPU-kernel, vision and image modules stubbed); without it
the sections that need them are skipped.

The functions that take (cfg, model) run under pytest too, on the graph built from
DSV41_MODEL_DIR, and skip when it is unset.
"""
import os, sys, math, types, random

if __name__ == "__main__":
    # run as the CPU suite: no GPU at all
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pytest
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _H not in sys.path:
    sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import ablate, model_dir, skip
from dsv41_ref import deepseek


def load_deepseek_ref(test):
    """DeepSeek's model.py, or None after reporting `test` as skipped when it is not configured."""
    if deepseek.ref_dir() is None:
        skip(test, "DSV41_DEEPSEEK_REF not set")
        return None
    return deepseek.load_model()


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------------------------

def build_model(path):
    from exllamav3 import Config, Model
    from exllamav3.cache.dsv41 import DSV41State
    cfg = Config.from_directory(path)
    model = Model.from_config(cfg)
    assert model.recurrent_state_cls is DSV41State, model.recurrent_state_cls
    return cfg, model


@pytest.fixture(scope = "module")
def built():
    d = model_dir()
    if d is None:
        pytest.skip("DSV41_MODEL_DIR is not set")
    return build_model(d)


@pytest.fixture
def cfg(built):
    return built[0]


@pytest.fixture
def model(built):
    return built[1]


def attn_layers(cfg, model):
    blocks = model.modules[model.first_block_idx: model.first_block_idx + cfg.num_hidden_layers]
    return [b.attn for b in blocks]


def test_cache(cfg, model):
    from exllamav3.cache.cache import Cache
    from exllamav3.cache.dsv41 import CacheLayer_dsv41, DSV41LayerState, DSV41State
    from exllamav3.cache.quant import CacheLayer_quant
    A = attn_layers(cfg, model)
    T = 1_048_576
    cache = Cache(model, max_num_tokens = T, max_batch_size = 1)
    keys = set(cache.layers)
    assert keys == {(2, 0), (8, 0), (14, 0), (20, 0)}, f"pool keys {sorted(keys)}"
    for k, l in cache.layers.items():
        assert type(l) is CacheLayer_dsv41, type(l)
        assert l.D_i == 128 and l.D_c == 448 and l.D_r == 64, (k, l.D_c, l.D_r, l.D_i)
        assert l.epp == 256 // cfg.compress_ratios[k[0]], (k, l.epp)
    sizes = {k: l.storage_size() for k, l in cache.layers.items()}
    assert sizes[(2, 0)] == sizes[(8, 0)] == sizes[(14, 0)] == 671_088_640, sizes
    assert sizes[(20, 0)] == 1_342_177_280, sizes
    total = sum(sizes.values())
    assert total == 3 * 671_088_640 + 1_342_177_280

    rl = cache.recurrent_layers
    states = [v for v in rl.values() if isinstance(v, DSV41LayerState)]
    assert len(states) == 40, f"{len(states)} DSV41LayerState"
    for (i, _), s in rl.items():
        if not isinstance(s, DSV41LayerState):
            continue
        assert tuple(s.ring.shape) == (1, 768, 512) and s.ring.dtype == torch.half
        want_carry = cfg.is_kv_source(i) and cfg.compress_ratios[i] == 2
        assert (s.comp_carry is not None) == want_carry, f"layer {i}: carry ring presence"
        if want_carry:
            assert tuple(s.comp_carry.shape) == (1, 258, 1024) and s.comp_carry.dtype == torch.float
            assert s.rollback_limit() == 257
        assert s.comp_buf_kv is None and s.comp_ovl is None and s.idx_buf_kv is None
    st = cache.get_new_state()
    assert isinstance(st, DSV41State) and st.position == 0
    st.free()

    # quantized Cache request: k_bits reaches the source pools
    lt, kw = A[2].cache_layer_type(CacheLayer_quant, {"k_bits": 6, "v_bits": 6})
    assert lt is CacheLayer_dsv41 and kw == {"k_bits": 6, "v_bits": 6}, (lt, kw)
    check_numerics_vs_packed_pools(cfg, model)

    # a consumer owns no pool: it is never asked for a cache layer class
    assert not any(A[i].caps.get("kv_cache") for i in range(cfg.num_hidden_layers)
                   if not cfg.is_kv_source(i))
    # a route without a replica must not claim a pool it does not have
    try:
        A[13].set_pool_route(13)
    except AssertionError:
        pass
    else:
        raise AssertionError("consumer 13 routed to itself without a replica")
    cache.detach_from_model()
    model.__dict__.pop("_get_cache_layers", None)

    # replica route (an explicit placement's split at 12): layer 12 owns a replica of source 8;
    # 13 reads it
    import exllamav3.cache.dsv41_replica as rep_mod
    A[12].set_pool_route(12, replica_of = 8)
    A[13].set_pool_route(12)
    try:
        cache2 = Cache(model, max_num_tokens = T, max_batch_size = 1)
        extra = set(cache2.layers) - keys
        assert extra == {(12, 0)}, f"extra pools {sorted(extra)}"
        r = cache2.layers[(12, 0)]
        assert type(r) is rep_mod.CacheLayer_dsv41_replica, type(r)
        assert r.storage_size() == 536_870_912, r.storage_size()
        assert A[13].pool_owner_layer == 12 and not A[13].caps["kv_cache"]
        assert A[12].caps["kv_cache"] and A[12].replica_of == 8
        cache2.detach_from_model()
    finally:
        model.__dict__.pop("_get_cache_layers", None)
        A[12].set_pool_route(8)
        A[13].set_pool_route(8)
    print(f"  OK  cache: 4 source pools {total:,} B at 1M (3 x 671,088,640 + 1,342,177,280), "
          f"40 DSV41LayerState, carry rings on 2/8/14; replica (12,0) {r.storage_size():,} B")


def check_numerics_vs_packed_pools(cfg, model):
    """
    The attention numerics round compressed entries in the FP16 pool, so the compressed part is
    refused with a quantized (packed) pool: when such a Cache is built under it, and when the
    setting switches to it while such a Cache exists. Both are configuration-time errors, so the
    rounders' overflow policy (a block kept unrounded by default, a refusal under
    EXL3_DSV41_NUMERICS_STRICT) changes neither: checked under both, with the package's own
    rounding module, the one the attention layers call.
    """
    from exllamav3.modules import dsv41 as attention, dsv41_rounding as rounding
    assert attention.rounding is rounding, "the attention does not call the package's rounders"
    previous = rounding.set_strict(False)
    try:
        for strict in (False, True):
            rounding.set_strict(strict)
            _packed_pool_refusals(cfg, model)
    finally:
        rounding.set_strict(previous)
    print("  OK  numerics vs a quantized Cache: compressed rounding refused when the Cache is built and "
          "while one exists, under the default overflow policy and the strict switch; index/window "
          "rounding and FP16 pools unaffected")


def _packed_pool_refusals(cfg, model):
    import gc
    from exllamav3.cache.cache import Cache
    from exllamav3.cache.quant import CacheLayer_quant
    from exllamav3.model.math_policy import STABLE_ARITHMETIC
    saved = cfg.dsv41_numerics
    attached = set(model.cache_weakrefs)
    quant = lambda: Cache(model, max_num_tokens = 65536, max_batch_size = 1,
                          layer_type = CacheLayer_quant, k_bits = 6, v_bits = 6)
    if STABLE_ARITHMETIC:
        # EXL3_STABLE_ARITHMETIC=1 refuses every quantized V4.1 Cache when it is built, whatever
        # the numerics, so the checks below need the default environment; check the refusal
        try:
            quant()
        except ValueError as e:
            assert "EXL3_STABLE_ARITHMETIC=1" in str(e), e
        else:
            raise AssertionError("built a quantized V4.1 Cache under EXL3_STABLE_ARITHMETIC=1")
        assert set(model.cache_weakrefs) == attached, "a refused Cache stayed attached"
        print("  OK  numerics vs a quantized Cache: under EXL3_STABLE_ARITHMETIC=1 the quantized Cache "
              "itself is refused (the numerics checks run in the default environment)")
        return
    try:
        cfg.dsv41_numerics = "deepseek:index"
        qc = quant()
        assert all(l.quant for l in qc.layers.values()) and len(qc.layers) == 4
        try:
            cfg.dsv41_numerics = "deepseek"
        except ValueError as e:
            assert "quantized pool" in str(e), e
        else:
            raise AssertionError("switched to compressed rounding with a packed pool attached")
        assert str(cfg.dsv41_numerics) == "deepseek:index"
        cfg.dsv41_numerics = "vllm:index,window"                 # no compressed part: fine
        qc.detach_from_model()
        del qc
        gc.collect()
        cfg.dsv41_numerics = "deepseek"                          # the packed Cache is gone
        try:
            quant()
        except ValueError as e:
            assert "quantized Cache" in str(e), e
        else:
            raise AssertionError("built a packed pool under compressed rounding")
        assert set(model.cache_weakrefs) == attached, "a refused Cache stayed attached"
        Cache(model, max_num_tokens = 65536, max_batch_size = 1).detach_from_model()   # FP16: fine
    finally:
        model.__dict__.pop("_get_cache_layers", None)
        cfg.dsv41_numerics = saved


def test_geometry():
    from exllamav3.modules.dsv41_cached import emission_range, rope_positions, entry_rows
    rng = random.Random(1)
    PAGE = 256
    n = 0
    for _ in range(10_000):
        m = rng.choice((1, 2))
        pos0 = rng.randrange(0, 60_000)
        seq = rng.choice((1, 1, 2, 3, rng.randrange(1, 5000)))
        e0, e1 = emission_range(pos0, seq, m)
        closing = [p for p in range(pos0, pos0 + seq) if (p + 1) % m == 0]
        assert list(range(e0, e1)) == [p // m for p in closing], (pos0, seq, m)
        if e1 > e0:
            e = torch.arange(e0, e1)
            # rope at the group's FIRST token
            assert torch.equal(rope_positions(e, m), torch.tensor([p + 1 - m for p in closing]))
            # pool row via the TOKEN page table: the group's tokens sit on one page
            pages = (pos0 + seq) // PAGE + 2
            bt = torch.randperm(4 * pages)[:pages].int().unsqueeze(0)
            epp = PAGE // m
            first_tok = e * m
            brute = bt[0, first_tok // PAGE].long() * epp + (first_tok % PAGE) // m
            assert torch.equal(entry_rows(bt, e, epp), brute), (pos0, seq, m)
            n += 1
    print(f"  OK  geometry: emission range, rope positions and pool rows == brute force "
          f"over 10,000 (pos0, seq) pairs ({n} emitting), m in {{1, 2}}")


def v41_capacity(position, window_beg, high_water, window = 128, limit = 257):
    """DSV41State.rollback_capacity on a stand-in state whose Cache holds one layer state with
    the given window and rollback_limit (the rate-2 carry ring: 257)."""
    from exllamav3.cache.dsv41 import DSV41State

    class _L:
        def __init__(self): self.window = window
        def rollback_limit(self): return limit

    st = types.SimpleNamespace(position = position, window_beg = window_beg, high_water = high_water,
                               cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {(0, 0): _L()}))
    st._window = lambda: DSV41State._window(st)
    return DSV41State.rollback_capacity(st)


def test_carry():
    from exllamav3.modules.dsv41_cached import CompressCarry
    from exllamav3.architecture.dsv41.compressor import compress_reference
    from exllamav3.cache.dsv41 import DSV41LayerState
    torch.manual_seed(0)
    hd, eps, m = 64, 1e-20, 2
    R = 256 + m                      # DSV41LayerState.buf_rows
    total = 1400
    kv = torch.randn(total, hd)
    sc = torch.randn(total, hd)
    w = torch.randn(hd) * 0.1 + 1.0
    ref = compress_reference(kv, sc, m, w, eps)

    def run(schedule, ring_rows = R):
        carry = torch.zeros(ring_rows, 2 * hd)
        got = torch.full((total // m, hd), float("nan"))
        pos = 0
        for kind, n in schedule:
            if kind == "rewind":
                pos -= n
                continue
            lat, first = CompressCarry.step(carry, kv[pos:pos + n], sc[pos:pos + n], pos, m, w, eps)
            assert first == pos // m and lat.shape[0] == (pos + n) // m - pos // m
            got[first: first + lat.shape[0]] = lat
            pos += n
        return got, pos

    rng = random.Random(7)
    n_rw = n_dec = 0
    for trial in range(60):
        sched, pos, hw = [], 0, 0
        while pos < total:
            r = rng.random()
            cap = v41_capacity(pos, 0, hw)
            if r < 0.2 and cap > 0:
                back = rng.randint(1, cap) if rng.random() < 0.5 else cap
                sched.append(("rewind", back)); pos -= back; n_rw += 1
            elif r < 0.6:
                sched.append(("fwd", 1)); pos += 1; n_dec += 1           # decode step
            else:
                n = rng.choice((rng.randrange(1, 40) * 2 + 1, rng.randrange(1, 700), 300, 511))
                n = min(n, total - pos)
                sched.append(("fwd", n)); pos += n
            hw = max(hw, pos)
        got, end = run(sched)
        assert end == total
        assert torch.equal(got, ref), \
            f"trial {trial}: chunked+rewound carry != one shot, max err {(got - ref).abs().max():.3e}"

    # negative controls. (a) Two rewinds with a short replay between them: from the current
    # position the second one looks legal (250 <= 257), but it lands below the ring's reach,
    # which only the high-water rule sees -- and doing it anyway corrupts the reopened group
    sched = [("fwd", 1001), ("rewind", 257), ("fwd", 1), ("rewind", 250), ("fwd", 655)]
    assert v41_capacity(745, 0, 1001) == 1, v41_capacity(745, 0, 1001)
    got, _ = run(sched)
    assert not torch.equal(got, ref), "carry test is blind to a rewind below the high-water reach"
    # (b) the ring really is what carries the pending row: an 8-row ring fails a 400 rewind
    got, _ = run([("fwd", 601), ("fwd", 400), ("rewind", 400), ("fwd", 399)], 8)
    assert not torch.equal(got, ref), "carry test is blind: a rewind past the ring did not matter"
    print(f"  OK  compressor carry: 60 random schedules ({n_dec} decode steps, {n_rw} rewinds within "
          f"DSV41State.rollback_capacity) == compress_reference bitwise; a second rewind below "
          f"the high-water reach and a too-short ring both corrupt it")


def test_select_registry(cfg, model):
    from exllamav3.modules.dsv41_select import SelectResult
    A = attn_layers(cfg, model)
    fake_kl = types.SimpleNamespace(pool_idx = torch.zeros(1), epp = 256)
    markers = {}
    for a in A:
        a.device = torch.device("cpu")
    for a in A:
        if a.is_index_source:
            t = torch.full((1, 32), a.layer_idx, dtype = torch.int32)
            markers[a.layer_idx] = t
            a._index_select = (lambda t: (lambda *args, **kw: SelectResult(t, 17, None)))(t)
    try:
        def run(params, ec):
            got = {}
            for a in A:
                if not a.compress_ratio:
                    continue
                got[a.layer_idx] = a._select_cached(None, params, None, fake_kl, None, ec, 0, 0)
            return got

        # ec <= index_topk: dense pool everywhere, nothing published
        p = {}
        got = run(p, 512)
        assert all(v == (None, 0) for v in got.values()) and not p.get("dsv41_topk")
        # top-k regime: every compressed layer uses index_source_for(i)
        p = {}
        got = run(p, 513)
        for i, (idx, k) in got.items():
            src = cfg.index_source_for(i)
            assert idx is markers[src] and k == 17, f"layer {i} read {idx[0, 0].item()}, expected {src}"
        assert set(p["dsv41_topk"]) == {(i, 0) for i in markers}
        # a consumer whose source did not publish must raise
        try:
            A[3]._select_cached(None, {"dsv41_topk": {}}, None, fake_kl, None, 600, 0, 0)
        except RuntimeError as e:
            assert "has not published" in str(e)
        else:
            raise AssertionError("missing publish went unreported")
        # dense_consumers: consumers get None, sources still select
        with ablate("dense_consumers"):
            got = run({}, 600)
        for i, (idx, _) in got.items():
            assert (idx is None) == (i not in markers), f"layer {i} under dense_consumers"
    finally:
        for a in A:
            a.__dict__.pop("_index_select", None)
            a.device = None
    n_c = sum(1 for a in A if a.compress_ratio)
    print(f"  OK  selection registry: {n_c} compressed layers read index_source_for(i); "
          f"missing publish raises; dense_consumers -> None on {n_c - len(markers)} consumers")


def untiled_select(q_idx, wts, idx_pool, *, pos0, m, ec, topk, cand_in, want_cand, block, n_blocks):
    """
    The selection contract in one untiled pass with torch.topk, as DeepSeek writes it: score_j =
    sum_h w_h * relu(q_h . k_j) over visible entries j < min((pos0 + r + 1) // m, ec); the
    candidate source's blocks (NaN-propagating max over each block's visible entries, the
    newest block pinned to +inf, top n_blocks, -inf blocks dropped); the users' mask to them;
    top-k ascending, -1 padded. Returns (indices, cand).
    """
    seq = q_idx.shape[0]
    e = torch.arange(ec)
    k = idx_pool.reshape(-1, q_idx.shape[-1])[:ec]
    sc = torch.einsum("shd,td->sht", q_idx.float(), k.float()).relu_()
    sc = (sc * wts.float().unsqueeze(-1)).sum(dim = 1)
    vis = torch.clamp((pos0 + torch.arange(seq) + 1) // m, max = ec)
    sc = sc.masked_fill(e.unsqueeze(0) >= vis.unsqueeze(1), -math.inf)
    cand = None
    if want_cand:
        bs = torch.nn.functional.pad(sc, (0, -ec % block), value = -math.inf)
        bs = bs.view(seq, -1, block).amax(dim = -1)
        nb = bs.shape[1]
        last = (vis - 1).div(block, rounding_mode = "floor")
        bs = bs.masked_fill(torch.arange(nb).unsqueeze(0) == last.unsqueeze(1), math.inf)
        kb = min(n_blocks, nb)
        top = bs.topk(kb, dim = -1)
        cand = torch.full((seq, n_blocks), -1, dtype = torch.int32)
        cand[:, :kb] = torch.where(top.values > -math.inf, top.indices, torch.full_like(top.indices, -1)).int()
    if cand_in is not None:
        nb = -(-ec // block)
        flags = torch.zeros((seq, nb + 1), dtype = torch.bool)
        c = cand_in.long()
        flags.scatter_(1, torch.where(c >= 0, c, torch.full_like(c, nb)), True)
        keep = flags[:, :nb].repeat_interleave(block, dim = 1)[:, :ec]
        sc = sc.masked_fill(~keep, -math.inf)
    kk = min(topk, ec)
    v, i = sc.topk(kk, dim = -1)
    i = torch.where(v > -math.inf, i, torch.full_like(i, ec)).sort(dim = -1).values
    idx = torch.full((seq, -(-kk // 32) * 32), -1, dtype = torch.int32)
    idx[:, :kk] = torch.where(i < ec, i, torch.full_like(i, -1)).int()
    return types.SimpleNamespace(indices = idx, cand = cand)


def test_select_vs_deepseek():
    """The untiled selection contract (untiled_select) vs DeepSeek's Indexer.forward: candidate
    source, candidate user, prefill and decode. Small shapes so the candidate stage bites:
    8-entry blocks, 6 candidate blocks, top-16."""
    ds = load_deepseek_ref("selection vs DeepSeek Indexer.forward")
    if ds is None:
        return
    torch.manual_seed(3)
    args = ds.ModelArgs(
        max_batch_size = 1, max_seq_len = 512, dim = 64, q_lora_rank = 32, head_dim = 32,
        rope_head_dim = 16, n_layers = 3, n_mtp_layers = 0, compress_ratios = (0, 1, 1),
        kv_source_layers = (1,), index_source_layers = (1, 2), index_n_heads = 4,
        index_head_dim = 32, index_topk = 16, candidate_source_layer = 1,
        candidate_topk_blocks = 6, candidate_block_size = 8, norm_eps = 1e-20)
    src, usr = ds.Indexer(args, 1).float(), ds.Indexer(args, 2).float()
    for mod in (src, usr):
        for p in mod.parameters():
            p.data.normal_(0, 0.2)
    freqs = ds.precompute_freqs_cis(16, 512, 0, 10000.0, 40, 32, 1)
    src.freqs_cis = usr.freqs_cis = freqs
    H, D = args.index_n_heads, args.index_head_dim

    def mine(ix, x, qr, pos0, ec, cand_in, want):
        q = torch.nn.functional.linear(qr, ix.wq_b.weight).unflatten(-1, (H, D))
        ds.apply_rotary_emb(q[..., -16:], freqs[pos0: pos0 + x.shape[1]])
        w = torch.nn.functional.linear(x, ix.weights_proj.weight) * (D ** -0.5 * H ** -0.5)
        return untiled_select(q[0], w[0], src.k_cache[0, :ec], pos0 = pos0, m = 1, ec = ec, topk = 16,
                              cand_in = cand_in, want_cand = want, block = 8, n_blocks = 6)

    def as_sets(idx):
        return [set(v for v in row.tolist() if v >= 0) for row in idx]

    T0 = 150
    x = torch.randn(1, 160, 64)
    qr = torch.randn(1, 160, 32)
    lat = torch.randn(1, 160, 32)
    checked = 0
    for pos0, n in ((0, T0),) + tuple((T0 + i, 1) for i in range(6)):
        xs, qs, ls = x[:, pos0: pos0 + n], qr[:, pos0: pos0 + n], lat[:, pos0: pos0 + n]
        ref_s = src(xs, qs, ls, pos0, 0)
        cand_mask = ds.shared_attn.candidates
        ref_u = usr(xs, qs * 0.7, None, pos0, 0)
        ec = pos0 + n
        r_s = mine(src, xs, qs, pos0, ec, None, True)
        r_u = mine(usr, xs, qs * 0.7, pos0, ec, r_s.cand, False)
        # candidate blocks as a position mask, compared with the reference's bool mask
        flags = torch.zeros(n, -(-ec // 8) + 1, dtype = torch.bool)
        c = r_s.cand.long()
        flags.scatter_(1, torch.where(c >= 0, c, torch.full_like(c, flags.shape[1] - 1)), True)
        my_mask = flags[:, :-1].repeat_interleave(8, dim = 1)[:, :ec]
        assert torch.equal(my_mask, cand_mask[0].reshape(n, -1)[:, :ec]), f"candidates differ at pos0 {pos0}"
        assert as_sets(r_s.indices) == as_sets(ref_s[0]), f"source top-k differs at pos0 {pos0}"
        assert as_sets(r_u.indices) == as_sets(ref_u[0]), f"masked user top-k differs at pos0 {pos0}"
        checked += n
    # the candidate stage must matter here, or the comparison proves nothing
    r_plain = mine(usr, x[:, :T0], qr[:, :T0] * 0.7, 0, T0, None, False)
    r_mask = mine(usr, x[:, :T0], qr[:, :T0] * 0.7, 0, T0, mine(src, x[:, :T0], qr[:, :T0], 0, T0, None, True).cand, False)
    differ = sum(a != b for a, b in zip(as_sets(r_plain.indices), as_sets(r_mask.indices)))
    assert differ > 20, f"candidate masking changed only {differ} rows"
    print(f"  OK  untiled selection == DeepSeek Indexer.forward on {checked} query rows "
          f"(prefill {T0} + 6 decode steps): source top-k, candidate blocks, masked user top-k "
          f"({differ}/{T0} rows change when the mask is dropped)")


def test_rope_positions(cfg, model):
    from exllamav3.modules.dsv41_cached import rope_positions, rope_gptj_torch
    from exllamav3.util.rope import yarn_inv_freq
    # the method both paths use, and its ablation
    a2 = attn_layers(cfg, model)[2]
    assert a2._entry_rope_positions(37, 5, "cpu").tolist() == [[74, 76, 78, 80, 82]]
    with ablate("rope_consecutive"):
        assert a2._entry_rope_positions(37, 5, "cpu").tolist() == [[74, 75, 76, 77, 78]]
    print("  OK  entry rope positions (g + j) * m; rope_consecutive gives g * m + j")
    ds = load_deepseek_ref("entry rope vs DeepSeek apply_rotary_emb")
    if ds is None:
        return
    rd = cfg.qk_rope_head_dim
    inv = yarn_inv_freq(rd, cfg.compress_rope_theta, "cpu", rope_scaling = cfg.rope_scaling)
    rs = cfg.rope_scaling
    freqs = ds.precompute_freqs_cis(rd, 70_000, rs["original_max_position_embeddings"],
                                    cfg.compress_rope_theta, rs["factor"], rs["beta_fast"], rs["beta_slow"])
    ds_inv = torch.angle(freqs[1]).float()
    e_inv = rel(inv, ds_inv)
    assert e_inv < 1e-6, f"yarn table vs DeepSeek precompute_freqs_cis rel {e_inv:.2e}"
    torch.manual_seed(5)
    for g, n, m in ((0, 50, 2), (37, 64, 2), (1000, 300, 2), (5, 40, 1), (31000, 200, 2)):
        j = torch.arange(n)
        pos = rope_positions(g + j, m)
        assert torch.equal(pos, (g + j) * m)
        x = torch.randn(n, rd)
        mine = rope_gptj_torch(x, inv, pos)
        per_row = torch.stack([rope_gptj_torch(x[i: i + 1], inv, pos[i: i + 1])[0] for i in range(n)])
        assert torch.equal(mine, per_row), "vectorised rotation != per-row rotation"
        ref = x.clone().unsqueeze(0)
        ds.apply_rotary_emb(ref, freqs[pos])
        e = rel(mine, ref[0])
        assert e < 1e-5, f"g={g} m={m}: rotation vs DeepSeek apply_rotary_emb rel {e:.2e}"
        old = rope_gptj_torch(x, inv, g * m + j)
        e_old = rel(old, ref[0])
        if m == 2:
            assert e_old > 0.1, f"g={g}: the old positions g*m+j are not told apart (rel {e_old:.2e})"
    print(f"  OK  entry rope: yarn table == DeepSeek precompute_freqs_cis (rel {e_inv:.1e}); "
          f"(g + j) * m rotation == DeepSeek apply_rotary_emb(freqs_cis[(g + j) * m]) to < 1e-5; "
          f"old g * m + j fails (rel > 0.1)")


def test_ring():
    from exllamav3.modules.dsv41_cached import ring_update
    w, rows, D = 128, 768, 4

    def capacity(rule, position, window_beg, hw):
        if rule == "v4":
            return position - window_beg
        return v41_capacity(position, window_beg, hw, window = w)

    def simulate(rule, seed, steps = 400):
        rng = random.Random(seed)
        ring = torch.zeros(rows, D)
        hist = []
        pos, beg, hw = 0, 0, 0
        shifts = rebases = rewinds = 0
        for _ in range(steps):
            r = rng.random()
            if r < 0.12 and pos > 0:
                cap = capacity(rule, pos, beg, hw)
                if cap > 0:
                    back = rng.randint(1, cap) if rng.random() < 0.5 else cap
                    pos -= back; del hist[pos:]; rewinds += 1
                continue
            seq = rng.choice((1, 1, 1, 2, 5, rng.randrange(1, 255), rng.randrange(256, 1500)))
            kv = torch.randn(seq, D)
            # what the kernel sees: ring rows at abs - beg for abs in [win_floor, pos0)
            n_prev = min(w - 1, pos - beg, pos)
            if n_prev != min(w - 1, pos):
                return False, (shifts, rebases, rewinds)
            for a in range(pos - n_prev, pos):
                if not torch.equal(ring[a - beg], hist[a]):
                    return False, (shifts, rebases, rewinds)
            hist += list(kv)
            sh = ring_update(ring, kv, pos, beg, w)
            if sh is not None:
                if seq < 256: shifts += 1
                else: rebases += 1
                beg += sh
            pos += seq
            hw = max(hw, pos)
        return True, (shifts, rebases, rewinds)

    tot = [0, 0, 0]
    for seed in range(40):
        ok, c = simulate("v41", seed)
        assert ok, f"seed {seed}: window rows lost under DSV41State.rollback_capacity"
        tot = [a + b for a, b in zip(tot, c)]
    assert all(t > 20 for t in tot), f"simulation did not exercise every branch: {tot}"
    fails = sum(1 for seed in range(40) if not simulate("v4", seed)[0])
    assert fails > 0, "V4's position - window_beg bound never lost a window row: the check is blind"
    print(f"  OK  ring_update: 40 random runs ({tot[0]} shifts, {tot[1]} rebases, {tot[2]} rewinds) "
          f"keep every visible window row; V4's looser rewind bound loses rows in {fails}/40")


def test_state_and_guard(cfg, model):
    from exllamav3.cache.dsv41 import DSV41LayerState
    A = attn_layers(cfg, model)
    s = DSV41LayerState(A[2], 2, 0, 0)
    s.alloc(torch.device("cpu"))
    s.ring.normal_(); s.comp_carry.normal_()
    # .cpu() of a CPU tensor is the tensor itself: copy, as a device stash would
    blob = [t.clone() for t in s.stash(1, 500)]
    ring1, carry1 = s.ring[1].clone(), s.comp_carry[1].clone()
    s.clear(1)
    assert s.ring[1].abs().sum() == 0 and s.comp_carry[1].abs().sum() == 0
    s.unstash(1, blob, 500)
    assert torch.equal(s.ring[1, :500], ring1[:500]) and torch.equal(s.comp_carry[1], carry1)
    # upper bound of a stash: the whole 768-row ring (V4's figure counted 128 rows of it)
    n_ck = s.get_checkpoint_size()
    assert n_ck == 768 * 512 * 2 + 258 * 1024 * 4, n_ck
    s.free()

    a = A[3]
    kl = types.SimpleNamespace(device = torch.device("meta"))
    cache = types.SimpleNamespace(layers = {(2, 0): kl})
    a.device = torch.device("cpu")
    try:
        a._pool_layer(types.SimpleNamespace(cache = cache), 0)
    except RuntimeError as e:
        msg = str(e)
        assert "the compressed-KV pool it reads (layers.2) is on meta, but this layer is on cpu" in msg, msg
        assert "every change of device at layer 1, 2, 8, 14 or 20" in msg and "Loading across GPUs" in msg, msg
        assert "An explicit placement (EXL3_PLACEMENT / --placement) can instead change device" in msg, msg
    else:
        raise AssertionError("a pool on another device went unreported")
    finally:
        a.device = None
    print(f"  OK  layer state: stash/clear/unstash round trip incl. the carry ring "
          f"(checkpoint {n_ck:,} B); a pool on another device is refused, naming the free cuts "
          f"and the explicit placement's split")


def test_checkpoint_bytes(cfg, model):
    """The generator's checkpoint cache (cache/recurrent.py RecurrentCache.put) evicts against
    stashed["checkpoint_size"] to hold its host-RAM budget (4 GiB by default). V4's per-layer
    estimate counts one 128-row window of the 768-row SWA ring while its stash keeps up to 768
    rows, so a V4.1 checkpoint was 4.1x its declared size. Every figure here is walked from
    the stashed tensors themselves."""
    from exllamav3.cache.cache import Cache
    from exllamav3.cache.dsv41 import DSV41State, DSV41LayerState, _nbytes
    cache = Cache(model, max_num_tokens = 65536, max_batch_size = 2)
    try:
        layers = cache.get_all_recurrent_layers()
        for l in layers.values():
            l.alloc(torch.device("cpu"))
            if isinstance(l, DSV41LayerState):
                for t in l._tensors():
                    t.normal_()
        # the Cache also holds the engram lookback rings (DSV41EngramState, layers 1 and 14):
        # DSV41State charges them their full declared size, since their walk can be shorter
        ring_states = {k for k, l in layers.items() if isinstance(l, DSV41LayerState)}
        other = {k: l for k, l in layers.items() if k not in ring_states}
        n_ring = len(ring_states)
        carry = 3 * 258 * 1024 * 4
        other_bound = sum(l.get_checkpoint_size() for l in other.values())
        v4_estimate = n_ring * 128 * 512 * 2 + carry
        bound = sum(l.get_checkpoint_size() for l in layers.values())
        assert bound == n_ring * 768 * 512 * 2 + carry + other_bound, bound
        rows_seen = []
        for pos, wb in ((0, 0), (1, 0), (500, 0), (768, 0), (700, 256), (4096, 3840), (4300, 3840), (4991, 4608)):
            st = DSV41State(cache, 0, pos, test_state = True)
            st.window_beg = wb
            s = st.stash()
            walked = sum(_nbytes(s[k]) for k in ring_states) \
                + sum(max(_nbytes(s[k]), other[k].get_checkpoint_size()) for k in other)
            assert s["checkpoint_size"] == walked, (pos, wb, s["checkpoint_size"], walked)
            assert walked <= bound, (pos, wb, walked, bound)
            rows = {s[k][0].shape[0] for k in ring_states}
            assert rows == {min(768, pos - wb)}, (pos, wb, rows)
            rows_seen.append((pos, wb, walked))
        # the pre-fix accounting: what V4 declares vs what V4's ring stash held at a full ring
        v4_actual = n_ring * 768 * 512 * 2 + carry
        assert v4_actual > 4 * v4_estimate

        # round trip of a trimmed checkpoint into another slot: history rows and every carry
        # ring restored, rows past the history zero (the forward writes them before reading)
        pos, wb = 4300, 3840
        st = DSV41State(cache, 0, pos, clear = False, test_state = True)
        st.window_beg = wb
        for k in ring_states:
            for t in layers[k]._tensors():
                t.normal_()
        # engram rings: random compressed ids for the last R positions of slot 0
        g = torch.Generator().manual_seed(0)
        for l in other.values():
            l.clear(0); l.clear(1)
            l.write(0, pos - l.R, torch.randint(0, 99092, (l.R,), generator = g))
        ref = {k: [t[0].clone() for t in layers[k]._tensors()] for k in ring_states}
        s = st.stash()
        # .cpu() of a CPU tensor is the tensor itself: copy, as a device stash would
        s = {k: ([t.clone() for t in v] if k in ring_states else v.clone() if k in other else v)
             for k, v in s.items()}
        st1 = DSV41State(cache, 1, pos, clear = False, stashed = s)
        assert st1.window_beg == wb and st1.high_water == pos, (st1.window_beg, st1.high_water)
        n = pos - wb
        for k in ring_states:
            ts = layers[k]._tensors()
            assert torch.equal(ts[0][1, :n], ref[k][0][:n]) and ts[0][1, n:].abs().sum() == 0, k
            for t, r in zip(ts[1:], ref[k][1:]):
                assert torch.equal(t[1], r), k
        # engram rings: the restored slot serves the same lookback at the checkpoint AND below
        # it -- the job state restores window_beg, which allows an in-place rewind under pos
        for k, l in other.items():
            for q in (pos, pos - 100, pos - l.R + 3):
                assert torch.equal(l.lookback(1, q), l.lookback(0, q)), (k, q)
    finally:
        cache.detach_from_model()
    print(f"  OK  checkpoint bytes: declared == stashed on the real graph at {len(rows_seen)} (position, "
          f"window_beg) points, e.g. {rows_seen[-3][2]:,} B at (4096, 3840) where V4's accounting declared "
          f"{v4_estimate:,} B for a {v4_actual:,} B stash; ring keeps position - window_beg rows; "
          f"trimmed checkpoint restores into slot 1")


def test_pool_limit(cfg, model):
    """dsa_attn and the pool scatters address a pool row as row * width in int32: a pool whose
    entries times its widest row (448 fp16 values) reach 2^31 is refused at construction."""
    from exllamav3.cache.dsv41 import CacheLayer_dsv41
    from exllamav3.cache.dsv41_replica import CacheLayer_dsv41_replica
    A = attn_layers(cfg, model)
    rows = (2 ** 31 - 1) // 448
    lim1 = (rows // 256) * 256                  # rate 1: 256 entries per token page
    lim2 = (rows // 128) * 256                  # rate 2: 128
    assert lim1 == 4_793_344 and lim2 == 9_586_944, (lim1, lim2)
    made = 0
    for make, lim in ((lambda n: CacheLayer_dsv41(cfg, A[20], 0, n), lim1),
                      (lambda n: CacheLayer_dsv41(cfg, A[2], 0, n), lim2),
                      (lambda n: CacheLayer_dsv41_replica(cfg, A[12], 0, n), lim2)):
        make(lim)
        try:
            make(lim + 256)
        except ValueError as e:
            assert "int32" in str(e) and f"{lim:,}" in str(e), e
            made += 1
        else:
            raise AssertionError("a pool past the int32 row addressing was accepted")
    print(f"  OK  pool limit: max_num_tokens {lim1:,} (rate 1) / {lim2:,} (rate 2) build, one page "
          f"more is refused naming the limit ({made} layer kinds: source pools and a replica)")


def test_quantized_pool_limit():
    """
    check_pool_addressing on packed pools: pool_q is G * k_bits int32 words wide and pool_s G
    wide (G = 448 / 32 = 14), so at k_bits 8 the widest row of a rate-1 kv source with index K
    is its 128-wide index K, and that of a rate-2 source without index K the 112-word pool_q.
    Needs no model: the attention is a namespace with V4.1-Flash's dimensions.
    """
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.cache.dsv41 import CacheLayer_dsv41
    def attn(m, i, idx):
        return types.SimpleNamespace(
            key = f"model.layers.{i}.attn", layer_idx = i, kv_source_layer = i, compress_rate = m,
            head_dim = 512, rope_head_dim = 64, index_head_dim = 128, layer_type = "hca",
            is_kv_source = True, owns_index_k = idx)
    for what, m, i, idx, width, epp in (("rate-1 source with index K", 1, 20, True, 128, 256),
                                        ("rate-2 source without index K", 2, 8, False, 112, 128)):
        build = lambda t: CacheLayer_dsv41(None, attn(m, i, idx), 0, t, k_bits = 8)
        lim = ((2 ** 31 - 1) // width) // epp * PAGE_SIZE
        layer = build(lim)
        assert layer.capacity * width < 2 ** 31, (what, layer.capacity)
        try:
            build(lim + PAGE_SIZE)
        except ValueError as e:
            assert "int32" in str(e) and f"{lim:,}" in str(e), e
        else:
            raise AssertionError(f"{what}: a quantized pool past the int32 limit was accepted")
        print(f"  OK  {what}, k_bits 8: {lim:,} tokens build ({width}-wide rows), one page more is "
              f"refused")


def test_entry_range_vs_deepseek():
    """emission_range, the entries a cached forward stores, is the range DeepSeek's reference
    Compressor writes (model.py; its GPU kernels, vision and image modules are stubbed, and the
    Compressor calls none of them)."""
    from exllamav3.modules.dsv41_cached import emission_range
    ref = load_deepseek_ref("emission_range vs DeepSeek's Compressor")
    if ref is None:
        return
    g = torch.Generator().manual_seed(2)
    checked = 0
    for ratio in (1, 2):
        for L in (1, 2, 3, 4, 7, 8, 33):
            args = ref.ModelArgs(dim = 16, head_dim = 8, rope_head_dim = 4, compress_ratios = (ratio,),
                                 max_batch_size = 1, max_seq_len = 256)
            comp = ref.Compressor(args, 0)
            with torch.no_grad():
                for p in comp.parameters():
                    p.copy_(torch.randn(tuple(p.shape), generator = g).to(p.dtype))
            dt = comp.wkv.weight.dtype
            x = torch.randn(1, L + 21, 16, generator = g).to(dt)
            # the only call pattern the reference supports: one prefill at 0, then 1-token steps
            for p, n in [(0, L)] + [(L + i, 1) for i in range(21)]:
                with torch.no_grad():
                    lat = comp(x[:, p : p + n], p)
                # Attention._compress_kv: compress_kv_cache[:, start_pos // ratio : start_pos // ratio + latent.size(1)]
                want = (p // ratio, p // ratio + lat.size(1)) if lat is not None else None
                e0, e1 = emission_range(p, n, ratio)
                if want is None:
                    assert e0 == e1, (ratio, L, p, n, e0, e1)
                else:
                    assert (e0, e1) == want, (ratio, L, p, n, (e0, e1), want)
                checked += 1
    print(f"  OK  emission_range == the entries DeepSeek's Compressor writes ({checked} forwards)")


def test_entry_range_vs_compress_chunk():
    """emission_range against exllamav3's chunked compressor, over arbitrary chunk splits."""
    from exllamav3.architecture.dsv41.compressor import compress_chunk, CompressorState
    from exllamav3.modules.dsv41_cached import emission_range
    g = torch.Generator().manual_seed(2)
    checked = 0
    rng = random.Random(2)
    hd = 8
    for m in (1, 2):
        for _ in range(40):
            state = CompressorState(hd, m)
            pos = 0
            for _ in range(rng.randint(1, 12)):
                n = rng.choice([1, 1, 2, 3, rng.randint(1, 300)])
                kv = torch.randn(n, hd, generator = g)
                score = torch.randn(n, hd, generator = g) if m == 2 else None
                lat, fg = compress_chunk(kv, score, pos, m, None, 1e-6, state)
                assert (fg, fg + lat.shape[0]) == emission_range(pos, n, m), (m, pos, n, fg, lat.shape)
                pos += n
                checked += 1
    print(f"  OK  emission_range == the entries compress_chunk emits ({checked} forwards)")


def test_device_memo():
    """The stateless path's per-forward moves: once per (key, device) and forward."""
    from exllamav3.cache.dsv41 import DeviceMemo
    orig = DeviceMemo.__dict__["_move"]
    calls = []
    def counting(t, device):
        calls.append(device)
        return orig.__func__(t, device)
    DeviceMemo._move = staticmethod(counting)
    try:
        params = {}
        topk = torch.randint(0, 1 << 20, (2048, 512), dtype = torch.int32)
        other = torch.device("meta")    # stands in for the reader's device
        a = DeviceMemo.get(params, ("topk", 8), topk, other)
        b = DeviceMemo.get(params, ("topk", 8), topk, other)
        assert a is b and a.device == other and a.shape == topk.shape and a.dtype == topk.dtype
        assert len(calls) == 1, calls
        assert set(params["dsv41_dev_memo"]) == {(("topk", 8), other)}
        # already on the device: handed back as is, no move, nothing memoised
        assert DeviceMemo.get(params, ("topk", 8), topk, "cpu") is topk and len(calls) == 1
        # another key moves its own tensor once
        cand = torch.zeros((2048, 2048), dtype = torch.int32)
        c1 = DeviceMemo.get(params, ("cand", 16), cand, other)
        assert DeviceMemo.get(params, ("cand", 16), cand, other) is c1 and len(calls) == 2
        # a producer that republishes under the same key is moved again, never served stale
        topk2 = topk.clone()
        d = DeviceMemo.get(params, ("topk", 8), topk2, other)
        assert d is not a and len(calls) == 3
        assert DeviceMemo.get(params, ("topk", 8), topk2, other) is d and len(calls) == 3
        assert DeviceMemo.get(params, ("topk", 8), None, other) is None
        # a new forward (fresh params) starts a fresh memo
        assert DeviceMemo.get({}, ("topk", 8), topk2, other) is not d and len(calls) == 4
    finally:
        DeviceMemo._move = orig
    # the move itself is the engine's device copy (EXLLAMA_NO_P2P_COPY's verdict included),
    # stream-ordered between two GPUs
    from unittest.mock import patch
    import exllamav3.cache.dsv41 as C
    marker = object()
    source = types.SimpleNamespace(device = torch.device("cuda:0"))
    with patch.object(C, "to_device", return_value = marker) as move:
        assert DeviceMemo._move(source, torch.device("cuda:1")) is marker
        move.assert_called_once_with(source, torch.device("cuda:1"), non_blocking = True)
    print("  OK  device memo: one move per (key, device) and forward, a republished tensor moved "
          "again, moves through to_device")


def main(path):
    cfg, model = build_model(path)
    test_cache(cfg, model)
    test_geometry()
    test_carry()
    test_select_registry(cfg, model)
    test_select_vs_deepseek()
    test_rope_positions(cfg, model)
    test_ring()
    test_state_and_guard(cfg, model)
    test_checkpoint_bytes(cfg, model)
    test_pool_limit(cfg, model)
    test_quantized_pool_limit()
    test_entry_range_vs_deepseek()
    test_entry_range_vs_compress_chunk()
    test_device_memo()
    assert not torch.cuda.is_initialized(), "a CUDA context was created"
    print("  OK  no CUDA context created")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 cached path", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
