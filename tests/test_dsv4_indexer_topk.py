"""
DeepSeek-V4's CSA indexer scores every query row against the visible key pool and keeps the top
index_topk entries per row, in one (rows, entries) fp16 matrix that grows with the context. The
load-time measuring forward (one chunk at context 0) never reaches the top-k regime, so the
autosplit measure must reserve the largest matrix the cache can reach. Regression: it reserved
nothing, and a device packed by autosplit could OoM partway through a long prefill.
EXL3_DSV4_INDEXER_SCORE_MB (opt-in) caps the matrix by running wide selections as row slabs,
which must not change the selection.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
import pytest
import torch

from exllamav3.modules import dsv4
from exllamav3.modules.dsv4 import DSV4Attention

H_I, D_I, M = 64, 128, 4        # DeepSeek-V4 indexer heads and head dim, CSA compress rate
EPP = 256 // M                  # pool entries per cache page


def stride(ec):
    return max(128, 1 << (ec - 1).bit_length())


def run_indexer_topk(topk, wts, *args, **kwargs):
    stub = SimpleNamespace(
        index_n_heads = H_I, index_head_dim = D_I, rope_head_dim = 64, inv_freq_compress = None,
        compress_rate = M, index_topk = topk,
        idx_weights = SimpleNamespace(forward = lambda x, params: wts),
    )
    stub._indexer_topk_slabs = lambda *a: DSV4Attention._indexer_topk_slabs(stub, *a)
    return DSV4Attention._indexer_topk(stub, *args, **kwargs)


def spy_indexer_topk(monkeypatch, seq, ctx, paged, cap_mb):
    """Run _indexer_topk on meta tensors (no kernels) for a seq-row chunk ending at context ctx,
    return (rows, q_pos0, score backing or None, allocated score matrix) of every scoring call."""
    calls = []

    def scores_spy(q_idx, weights, k_idx, q_pos0, compress_rate, bound_max, scores = None,
                   block_table = None, epp = 0):
        T = bound_max if block_table is not None else k_idx.shape[0]
        R = q_idx.shape[0]
        backing = None
        if scores is None:
            # The launcher's own allocation: T rounded up to a power of two
            scores = torch.empty((R, stride(T)), dtype = torch.half, device = "meta")
        else:
            # The launcher's backing contract
            T_pad = -(-max(T, 1) // 128) * 128
            assert scores.shape[0] >= R and scores.shape[1] >= T_pad and scores.stride() == (scores.shape[1], 1)
            backing = scores._base
        calls.append((R, q_pos0, backing, scores if backing is None else backing))
        return scores[:R, :T]

    noop = lambda *args: None
    monkeypatch.setattr(dsv4, "dsv4_indexer_score_mb", cap_mb)
    monkeypatch.setattr(dsv4, "dsa_indexer_scores", scores_spy)
    monkeypatch.setattr(dsv4, "_ext_rope", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsv4, "ext", SimpleNamespace(dsa_topk = noop))

    ec = ctx // M
    pages = -(-ec // EPP)
    pool = torch.empty((pages * EPP if paged else ec, D_I), dtype = torch.half, device = "meta")
    bt = torch.arange(pages, dtype = torch.int32).unsqueeze(0) if paged else None
    x = torch.empty((1, seq, 16), dtype = torch.half, device = "meta")
    q = torch.empty((seq, H_I * D_I), dtype = torch.half, device = "meta")
    wts = torch.empty((1, seq, H_I), dtype = torch.half, device = "meta")
    run_indexer_topk(512, wts, x, {}, None, pool, ec, ctx - seq, q_idx_pre = q,
                     block_table = bt, epp = EPP if paged else 0)
    return calls


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("seq, ctx, cap_mb", [
    (2048, 131072, 0), (2048, 262144 + 4, 0), (2048, 1048576, 0),  # default: never slabbed
    (8192, 1048576, 0), (65281, 262144 + 65281, 0),
    (2048, 131072, 128), (2048, 131072 + 256, 256),                # within the cap
    (512, 1048576, 1), (1, 1048576, 1), (4, 1048576, 1),           # <= 512 rows, decode
])
def test_dsv4_indexer_single_pass(monkeypatch, paged, seq, ctx, cap_mb):
    """Off, or within the cap: one scoring call over all rows, sized by the launcher as upstream."""
    calls = spy_indexer_topk(monkeypatch, seq, ctx, paged, cap_mb)
    assert len(calls) == 1
    rows, q_pos0, backing, matrix = calls[0]
    assert (rows, q_pos0, backing) == (seq, ctx - seq, None)
    assert matrix.shape == (seq, stride(ctx // M))


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("seq, ctx, cap_mb", [
    (2048, 131072 + 256, 128), (2048, 262144 + 4, 128), (2048, 1048576, 256),
    (2048, 1048576, 512), (8192, 1048576, 256), (1537, 1048576, 256), (1040, 1048576, 1),
    (2048, 1048576, 1), (65281, 65281, 64), (65281, 262144 + 65281, 128),
    (1040, 131100 + 1040, 1),
])
def test_dsv4_indexer_slabs_bounded(monkeypatch, paged, seq, ctx, cap_mb):
    """Past the cap: the fewest slabs of whole 64-row tiles (at least 512 rows) within it,
    covering every row in order at its own position, all sharing one score backing at the
    launcher's own stride, none small enough (<= 16 rows) for the few-query scoring kernel or
    the split top-k."""
    calls = spy_indexer_topk(monkeypatch, seq, ctx, paged, cap_mb)
    s = stride(ctx // M)
    rows = max(512, (cap_mb << 20) // (2 * s) // 64 * 64)
    backing = calls[0][3]
    assert backing.shape == (rows, s)
    assert backing.numel() * 2 <= max(cap_mb << 20, 512 * s * 2)
    assert len(calls) == -(-seq // rows) > 1
    r = 0
    for i, (n, q_pos0, b, matrix) in enumerate(calls):
        assert b is backing and matrix is backing
        assert q_pos0 == ctx - seq + r and 16 < n <= rows
        if i < len(calls) - 2:
            assert n == rows
        if i < len(calls) - 1:
            assert n % 64 == 0
        r += n
    assert r == seq


@pytest.mark.parametrize("chunk, ctx, cap_mb, score_shape", [
    (2048, 1048576, 0, (2048, 262144)),     # 1 GiB: a default chunk at 1M
    (2048, 262144 + 4, 0, (2048, 131072)),  # 512 MiB just past 256K
    (2048, 131072, 0, (2048, 32768)),       # 128 MiB at 128K
    (8192, 1048576, 0, (8192, 262144)),
    (2048, 1048576, 256, (512, 262144)),    # 256 MiB cap at 1M: 512-row slabs
    (2048, 1048576, 128, (512, 262144)),    # below the 512-row floor
    (2048, 131072, 64, (1024, 32768)),
    (2048, 131072, 512, (2048, 32768)),     # a cap above the whole chunk's matrix
    (1100, 1048576, 99, (512, 262144)),
    (8192, 131072, 99, (6336, 8192)),       # a narrower pool reaches more of the cap
    (2048, 2048, 0, None),                  # never reaches the top-k regime
])
def test_dsv4_indexer_reservation(monkeypatch, chunk, ctx, cap_mb, score_shape):
    """The autosplit measure allocates the largest score matrix any chunk up to this one can
    reach at any context the cache holds, with the cap off or on."""
    monkeypatch.delenv("EXL3_AUTOSPLIT_WORSTCASE", raising = False)
    monkeypatch.setattr(dsv4, "dsv4_indexer_score_mb", cap_mb)
    sizes = []
    def empty_spy(shape, dtype = None, device = None):
        sizes.append((tuple(shape), dtype))
        return torch.empty(shape, dtype = dtype, device = "meta")
    monkeypatch.setattr(dsv4, "torch", SimpleNamespace(empty = empty_spy, half = torch.half, int32 = torch.int32))
    stub = SimpleNamespace(
        indexer = object(), device = torch.device("meta"), layer_idx = 3, index_topk = 512,
        num_q_heads = 64, head_dim = 512, index_n_heads = H_I, index_head_dim = D_I,
        q_a = SimpleNamespace(out_features_unpadded = 1024))
    cache = SimpleNamespace(layers = {(3, 0): SimpleNamespace(capacity = ctx // M)})
    DSV4Attention.autosplit_extra_measure(stub, {"cache": cache, "batch_shape": (1, chunk)})
    if score_shape is None:
        assert sizes == []
        return
    assert sizes[-1] == (score_shape, torch.half)
    assert len(sizes) == 7 and all(s[0][0] == chunk for s in sizes[:6])
    # Every selection the cache can run (any chunk up to this one, any pool width in the top-k
    # regime) allocates at most the reservation, and one allocates exactly it
    reserved = score_shape[0] * score_shape[1]
    reach = 0
    for ec in sorted({513, *(1 << b for b in range(10, 40) if (1 << b) <= ctx // M), ctx // M}):
        s = stride(ec)
        for seq in {1, 512, 513, 1040, chunk - 1, chunk}:
            rows = dsv4._idx_score_slab_rows(ec)
            n = min(seq, rows) if rows and seq > 512 else seq
            assert n * s <= reserved
            reach = max(reach, n * s)
    assert reach == reserved


def test_dsv4_indexer_reservation_off(monkeypatch):
    """EXL3_AUTOSPLIT_WORSTCASE=0 keeps the plain measured forward."""
    monkeypatch.setenv("EXL3_AUTOSPLIT_WORSTCASE", "0")
    monkeypatch.setattr(dsv4, "torch", SimpleNamespace(empty = lambda *a, **k: pytest.fail("allocated")))
    DSV4Attention.autosplit_extra_measure(SimpleNamespace(), {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires CUDA")
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("seq, pos0, topk", [
    (1100, 3000, 96),         # 512 + 512 + 76
    (1040, 3000, 96),         # 512 + 448 + 80: a 16-row remainder takes a tile
    (1537, 900, 300),         # rows seeing exactly k entries
    (600, 131100, 512),       # past 32768 entries: 512 + 88
    (1040, 131100, 512),      # past 32768 entries: 512 + 448 + 80 keeps the tail off split top-k
])
def test_dsv4_indexer_slabs_match_single_pass(monkeypatch, paged, seq, pos0, topk):
    """Slabbed selection must return exactly the single-pass indices, order included."""
    from exllamav3.modules.attention_fn.dsa_triton import dsa_indexer_scores
    from exllamav3.ext import exllamav3_ext as ext

    device = "cuda:0"
    ec = (pos0 + seq) // M
    monkeypatch.setattr(dsv4, "dsv4_indexer_score_mb", 1)
    monkeypatch.setattr(dsv4, "_ext_rope", lambda *args, **kwargs: None)
    torch.manual_seed(seq + pos0)

    # Keys drawn from a few prototypes: exact score ties across slabs exercise the
    # index-ascending tie order
    protos = torch.randn((6, D_I), dtype = torch.half, device = device)
    keys = protos[torch.randint(0, 6, (ec,), device = device)]
    bt = None
    if paged:
        pages = -(-ec // EPP)
        perm = torch.randperm(pages, device = device)
        pool = torch.zeros((pages, EPP, D_I), dtype = torch.half, device = device)
        j = torch.arange(ec, device = device)
        pool[perm[j // EPP], j % EPP] = keys
        pool = pool.view(-1, D_I)
        bt = perm.int().unsqueeze(0)
    else:
        pool = keys
    q = torch.randn((seq, H_I * D_I), dtype = torch.half, device = device)
    wts = torch.randn((1, seq, H_I), dtype = torch.half, device = device)
    x = torch.empty((1, seq, 16), dtype = torch.half, device = device)
    epp = EPP if paged else 0

    calls = []
    real_scores = dsv4.dsa_indexer_scores
    def scores_spy(q_idx, *args, **kwargs):
        calls.append(q_idx.shape[0])
        return real_scores(q_idx, *args, **kwargs)
    monkeypatch.setattr(dsv4, "dsa_indexer_scores", scores_spy)
    got, k = run_indexer_topk(topk, wts, x, {}, None, pool, ec, pos0, q_idx_pre = q,
                              block_table = bt, epp = epp)
    assert len(calls) > 1 and min(calls) > 16

    scores = dsa_indexer_scores(q.view(seq, H_I, D_I), wts[0], pool, pos0, M, ec,
                                block_table = bt, epp = epp)
    ref = torch.empty_like(got)
    ext.dsa_topk(scores, ref, k, None, 0)
    assert k == min(topk, ec)
    assert torch.equal(got, ref)
    # The slab stride is the launcher's own
    assert scores.stride(0) == dsv4._idx_score_stride(ec)
