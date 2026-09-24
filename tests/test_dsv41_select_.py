# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""Tiled indexer top-k and the V4.1 candidate stage vs DeepSeek's reference math. CPU only (no
GPU; it imports exllamav3, so it needs the built extension).

Oracles:
  * DeepSeek's own select_candidate_blocks, executed from inference/model.py
    (DSV41_DEEPSEEK_REF) -- importing the file as-is needs tilelang (and sympy,
    through engram.py), so the one function is lifted out with ast and run
    as-is, which needs neither;
  * a transcription of the Indexer.forward tail (einsum -> relu * weights ->
    causal -inf -> candidates -> topk -> sort -> -1), whose key lines are
    asserted to be present verbatim in that file;
  * a float64 transcription of vLLM's _block_scores_kernel,
    select_candidate_blocks and apply_candidate_mask
    (vllm/model_executor/kernels/attention/dsa/candidate_blocks.py).

Where the references use torch.topk, ties are unspecified; with continuous
float64 inputs and 32 heads there are none (a score is exactly 0 only when all
32 relus are), so those comparisons are exact. The tie rule of
the port (score descending, index ascending) is checked on integer-valued
inputs, where every accumulation order gives the same bits and ties are
everywhere.

Without DSV41_DEEPSEEK_REF the DeepSeek-oracle sections are skipped; the vLLM
transcriptions and the tiled-vs-untiled checks always run.

    python tests/test_dsv41_select_.py      or      python -m pytest tests/test_dsv41_select_.py
    (test_select: every check without reference code; test_select_vs_deepseek: the
    DeepSeek-oracle sections)"""
import ast, os, sys, time
import torch
import torch.nn.functional as F

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
from exllamav3.modules import dsv41_select as S
from exllamav3.modules.dsv41_select import select_topk, select_topk_reference, transient_bytes
from exllamav3.architecture.dsv41 import candidates as C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import ablate, skip
from dsv41_ref import deepseek

NEG_INF = float("-inf")
MiB = 1 << 20

# Indexer.forward lines the transcription below mirrors (checked verbatim)
DS_LINES = [
    'self.uses_candidates = 0 <= args.candidate_source_layer < layer_id',
    'index_score = torch.einsum("bshd,btd->bsht", q, index_k)',
    'index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)',
    'compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)',
    'index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)',
    'compress_lens = end_pos // ratio',
    'index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size',
    'index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)',
    'topk = min(self.index_topk, end_pos // ratio)',
    'idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values',
    'return torch.where(idxs < compress_lens, idxs + offset, -1).int()',
]


def load_deepseek():
    if deepseek.ref_dir() is None:
        return None
    path = os.path.join(deepseek.ref_dir(), "model.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{deepseek.ENV}={deepseek.ref_dir()} has no model.py")
    src = open(path).read()
    for line in DS_LINES:
        assert line in src, f"DeepSeek reference changed; transcription line not found: {line}"
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_candidate_blocks")
    ns = {"torch": torch, "F": F}
    exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), ns)
    return ns["select_candidate_blocks"]


def ds_indexer(q, index_k, weights, start_pos, ratio, index_topk, scb, *, source = False,
               cand_mask = None, topk_blocks = 2048, block_size = 8):
    """DeepSeek Indexer.forward from the einsum on (q, index_k, weights already computed).
    q [b, s, h, d], index_k [b, end_pos // ratio, d], weights [b, s, h] pre-scaled."""
    bsz, seqlen = q.shape[:2]
    end_pos = start_pos + seqlen
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf)
    else:
        compress_lens = end_pos // ratio
    candidates = None
    if source:
        candidates = scb(index_score, compress_lens, topk_blocks, block_size)
    elif cand_mask is not None:
        index_score = index_score.masked_fill(~cand_mask, -torch.inf)
    topk = min(index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs, -1).int(), candidates


# ---- vLLM candidate_blocks.py, transcribed to float64 torch (start = 0, row_repeat = 1) ----

def vllm_block_scores(logits, ends, block):
    rows, width = logits.shape
    nblocks = -(-width // block)
    blocks = torch.arange(nblocks)
    offsets = torch.arange(block)
    cols = blocks[:, None] * block + offsets[None, :]                     # [nblocks, block]
    out = torch.empty((rows, nblocks), dtype = torch.float64)
    for r in range(rows):
        end = int(ends[r])
        valid = (cols < end) & (cols < width)
        values = torch.where(valid, logits[r].double()[cols.clamp(max = width - 1)],
                             torch.tensor(NEG_INF, dtype = torch.float64))
        nan = torch.isnan(values).any(dim = 1)                             # PropagateNan.ALL
        mx = torch.where(torch.isnan(values), NEG_INF, values).amax(dim = 1)
        reduced = torch.where(nan, torch.tensor(float("nan"), dtype = torch.float64), mx)
        reduced = torch.where((end > 0) & (blocks == (end - 1) // block),
                              torch.tensor(float("inf"), dtype = torch.float64), reduced)
        out[r] = reduced
    return out


def vllm_select_candidate_blocks(logits, ends, topk_blocks, block):
    rows, width = logits.shape
    out = torch.full((rows, topk_blocks), -1, dtype = torch.int32)
    if not width:
        return out
    scores = vllm_block_scores(logits, ends, block)
    top = scores.topk(min(topk_blocks, scores.shape[1]), dim = -1)
    k = top.values.shape[1]
    out[:, :k] = torch.where(top.values > NEG_INF, top.indices, -1).int()   # _store_candidates_kernel
    return out


def vllm_apply_candidate_mask(logits, ends, cands, block):
    rows, width = logits.shape
    nblocks = -(-width // block)
    out = logits.clone().double()
    cols = torch.arange(width)
    for r in range(rows):
        flags = torch.zeros(nblocks + 1, dtype = torch.bool)
        b = cands[r].long()
        b = torch.where(b * block >= width, nblocks, b)
        flags[b[b >= 0]] = True
        valid = cols < int(ends[r])
        keep = flags[(cols // block).clamp(max = nblocks)] & valid
        keep = keep | ((cols == width - 1) & flags[nblocks])
        out[r] = torch.where(keep, out[r], NEG_INF)
    return out


# ---- helpers ----

def ids_to_set(row):
    return set(int(x) for x in row.tolist() if x >= 0)


def ints(seq, T, H, D, gen):
    """Integer-valued inputs: every product and partial sum is exact in fp32, so any
    accumulation order gives the same bits, and ties are everywhere."""
    q = torch.randint(-3, 4, (seq, H, D), generator = gen).half()
    k = torch.randint(-3, 4, (T, D), generator = gen).half()
    w = torch.randint(-4, 5, (seq, H), generator = gen).float() / 64
    return q, w, k


def conts(seq, T, H, D, gen):
    q = torch.randn((seq, H, D), generator = gen, dtype = torch.float64)
    k = torch.randn((T, D), generator = gen, dtype = torch.float64)
    w = torch.randn((seq, H), generator = gen, dtype = torch.float64) * (D ** -0.5 * H ** -0.5)
    return q, w, k


def boundary_ties(scores, k):
    """Rows (with > k visible) whose k-th score is shared by entries on both sides of the cut."""
    v = scores.sort(dim = 1, descending = True).values
    fin = (v > NEG_INF).sum(1)
    kth = v[:, k - 1:k]
    return int(((fin > k) & (v[:, k:k + 1] == kth).squeeze(1)).sum())


def check(name, ok, detail = ""):
    assert ok, f"FAIL {name} {detail}"
    print(f"  OK  {name}" + (f"  ({detail})" if detail else ""))


def check_vs_deepseek(scb, gen):
    """DeepSeek's oracles: the Indexer.forward transcription and its select_candidate_blocks, at
    prefill and at decode past 16,384 entries, where the candidate stage bites."""
    # ------------------------------------------------------------------ DeepSeek, prefill
    # 32 heads as in the model: an exact-zero score (every head's relu 0) is then a 2^-32
    # event, so the continuous inputs have no ties for torch.topk to break arbitrarily
    H, D = 32, 32
    for seqlen, ratio in [(700, 1), (1400, 2)]:
        ec = seqlen // ratio
        q, w, k = conts(seqlen, ec, H, D, gen)
        want_i, want_c = ds_indexer(q[None], k[None], w[None], 0, ratio, 512, scb, source = True)
        got = select_topk(q, w, k, pos0 = 0, m = ratio, ec = ec, want_cand = True,
                          slab_rows = 97, tile_entries = 128, score_dtype = torch.float64)
        keep = C.candidate_keep(got.cand, 8, ec)
        check(f"DeepSeek prefill seqlen {seqlen} ratio {ratio}, source: indices and candidate mask",
              torch.equal(got.indices, want_i[0]) and torch.equal(keep, want_c[0]),
              f"{seqlen} rows x 512, rows with < 512 visible -1 padded")
        qu, wu, _ = conts(seqlen, 0, H, D, gen)
        want_u, _ = ds_indexer(qu[None], k[None], wu[None], 0, ratio, 512, scb, cand_mask = want_c)
        got_u = select_topk(qu, wu, k, pos0 = 0, m = ratio, ec = ec, cand_in = got.cand,
                            slab_rows = 97, tile_entries = 128, score_dtype = torch.float64)
        check(f"DeepSeek prefill seqlen {seqlen} ratio {ratio}, user: indices",
              torch.equal(got_u.indices, want_u[0]))

    # ------------------------------------------------------------------ DeepSeek, decode past 16384 (the stage bites)
    H, D = 32, 128
    T = 20000
    _, _, k = conts(1, T, H, D, gen)
    pos0, n = 16370, 30                          # crosses 16384 (2048 blocks) at row 14
    tail = list(range(19990, 20000))
    rows = list(range(pos0, pos0 + n)) + tail
    qs, ws, _ = conts(len(rows), 0, H, D, gen)   # queries of the source (layer 20) at those rows
    qu, wu, _ = conts(len(rows), 0, H, D, gen)   # and of a user (layer 24)
    src = select_topk(qs[:n], ws[:n], k, pos0 = pos0, m = 1, ec = pos0 + n,
                      want_cand = True, slab_rows = 11, tile_entries = 4096, score_dtype = torch.float64)
    src_t = select_topk(qs[n:], ws[n:], k, pos0 = 19990, m = 1, ec = T,
                        want_cand = True, slab_rows = 4, tile_entries = 8192, score_dtype = torch.float64)
    usr = select_topk(qu[:n], wu[:n], k, pos0 = pos0, m = 1, ec = pos0 + n,
                      cand_in = src.cand, slab_rows = 11, tile_entries = 4096, score_dtype = torch.float64)
    usr_t = select_topk(qu[n:], wu[n:], k, pos0 = 19990, m = 1, ec = T,
                        cand_in = src_t.cand, slab_rows = 4, tile_entries = 8192, score_dtype = torch.float64)
    unm_t = select_topk(qu[n:], wu[n:], k, pos0 = 19990, m = 1, ec = T,
                        slab_rows = 4, tile_entries = 8192, score_dtype = torch.float64)
    s_idx = torch.cat([src.indices, src_t.indices]); s_cand = torch.cat([src.cand, src_t.cand])
    u_idx = torch.cat([usr.indices, usr_t.indices])
    n_bad_s = n_bad_c = n_bad_u = 0
    for j, p in enumerate(rows):
        wi, wc = ds_indexer(qs[j][None, None], k[None, :p + 1], ws[j][None, None], p, 1, 512, scb,
                            source = True)
        n_bad_s += int(not torch.equal(s_idx[j], wi[0, 0]))
        n_bad_c += int(not torch.equal(C.candidate_keep(s_cand[j:j + 1], 8, p + 1)[0], wc[0, 0]))
        ui, _ = ds_indexer(qu[j][None, None], k[None, :p + 1], wu[j][None, None], p, 1, 512, scb,
                           cand_mask = wc)
        n_bad_u += int(not torch.equal(u_idx[j], ui[0, 0]))
    check("DeepSeek decode rows 16370-16399 + 19990-19999, source top-512", n_bad_s == 0,
          f"{len(rows)} rows, {n_bad_s} differ")
    check("DeepSeek decode rows, candidate blocks == select_candidate_blocks mask", n_bad_c == 0,
          f"{n_bad_c} differ; kept blocks per row {int((s_cand[0] >= 0).sum())}..{int((s_cand[-1] >= 0).sum())}")
    check("DeepSeek decode rows, masked user top-512", n_bad_u == 0, f"{n_bad_u} differ")
    moved = (usr_t.indices != unm_t.indices).any(1).float().mean().item()
    check("candidate mask bites past 16384 (user selection differs from unmasked)", moved > 0.5,
          f"{moved:.0%} of rows at 19990-19999 changed")
    # rate 2 decode rows (no candidates on rate-2 layers; checks the causal rule at odd/even ends)
    q2, w2, k2 = conts(40, 5000, 32, 64, gen)
    p0 = 9960
    got = select_topk(q2, w2, k2, pos0 = p0, m = 2, ec = (p0 + 40) // 2, slab_rows = 16,
                      tile_entries = 1024, score_dtype = torch.float64)
    bad = 0
    for r in range(40):
        p = p0 + r
        wi, _ = ds_indexer(q2[r][None, None], k2[None, :(p + 1) // 2], w2[r][None, None], p, 2, 512, scb)
        bad += int(not torch.equal(got.indices[r], wi[0, 0]))
    check("DeepSeek decode rows at ratio 2 (positions 9960-9999)", bad == 0, f"{bad} of 40 differ")


def check_candidate_stage(gen, scb = None):
    """The candidate writer and mask against float64 transcriptions of vLLM's kernels, and
    against DeepSeek's own select_candidate_blocks when `scb` is given."""
    # ------------------------------------------------------------------ 3/7. candidate writer and mask vs vLLM (float64)
    P = 20000
    vis = torch.tensor([0, 1, 7, 8, 9, 100, 16383, 16384, 16385, 16392, 16393, 20000, 20000, 20000, 20000, 20000])
    R = len(vis)
    sc = torch.randn((R, P), generator = gen, dtype = torch.float64)
    sc[12, 800:1600] = NEG_INF                    # 100 visible -inf blocks
    sc[5, 16:24] = NEG_INF                        # a -inf block in a short row (fewer than 2048 blocks)
    sc[13] = float("nan")                         # NaN row: every visible block NaN
    sc[14, 3:5] = float("nan"); sc[14, 900] = float("nan")     # a few NaN blocks
    sc[15] = torch.randint(0, 40, (P,), generator = gen).double()   # ties everywhere
    sc = sc.masked_fill(torch.arange(P)[None] >= vis[:, None], NEG_INF)
    b_ref = C.block_scores(sc, vis, 8)
    b_vll = vllm_block_scores(sc, vis, 8)
    check("block scores == vLLM _block_scores_kernel (NaN-propagating max, newest pinned)",
          torch.equal(torch.isnan(b_ref), torch.isnan(b_vll))
          and torch.equal(b_ref.nan_to_num(nan = 7.5), b_vll.nan_to_num(nan = 7.5)))
    c_ref = C.publish_candidate_blocks(sc, vis, 8, 2048)
    c_vll = vllm_select_candidate_blocks(sc, vis, 2048, 8)
    c_ds = scb(sc[:15], vis[:15].unsqueeze(-1), 2048, 8) if scb is not None else None
    for r in range(R):
        a, b = ids_to_set(c_ref[r]), ids_to_set(c_vll[r])
        if r == 15:
            # ties: same size, same multiset of block scores, and among tied blocks the lowest ids
            va = sorted(b_ref[r, list(a)].tolist()); vb = sorted(b_vll[r, list(b)].tolist())
            assert len(a) == len(b) == 2048 and va == vb, "tie row: different score multiset"
            thr = min(v for v in va if v != float("inf"))
            tied = (b_ref[r] == thr).nonzero().view(-1).tolist()
            took = sorted(x for x in a if b_ref[r, x] == thr)
            assert took == tied[:len(took)] and len(took) < len(tied), "tie row: not the lowest ids"
        else:
            assert a == b, f"row {r}: {len(a)} vs {len(b)} blocks"
        if c_ds is not None and r < 15:
            assert torch.equal(C.candidate_keep(c_ref[r:r + 1], 8, P)[0], c_ds[r]), f"row {r} vs DeepSeek"
    n13, n14, n5 = len(ids_to_set(c_ref[13])), len(ids_to_set(c_ref[14])), len(ids_to_set(c_ref[5]))
    check("publish_candidate_blocks == vLLM select_candidate_blocks" +
          (" == DeepSeek select_candidate_blocks" if scb is not None else ""), True,
          f"rows with 0..2500 visible blocks; a -inf block dropped (13-block row keeps {n5}); "
          f"all-NaN row keeps {n13}; row with 2 NaN blocks keeps {n14}; tie row index-ascending")
    m_ref = C.mask_scores_to_candidates(sc, c_ref, 8)
    m_vll = vllm_apply_candidate_mask(sc, vis, c_ref, 8)
    check("mask_scores_to_candidates == vLLM apply_candidate_mask",
          torch.equal(torch.isnan(m_ref), torch.isnan(m_vll)) and
          torch.equal(m_ref.nan_to_num(nan = 7.5), m_vll.nan_to_num(nan = 7.5)))
    # the same scores driven through the TILED select path: H = 2, D = 1, q = (1, -1), w = (1, -1)
    # makes score_j = relu(k_j) - relu(-k_j) = k_j exactly, incl. +-inf and NaN
    qq = torch.tensor([[[1.0], [-1.0]]], dtype = torch.float64)
    ww = torch.tensor([[1.0, -1.0]], dtype = torch.float64)
    for r in range(6, R):
        v = int(vis[r])
        got = select_topk(qq, ww, sc[r, :v].unsqueeze(1), pos0 = v - 1, m = 1, ec = v,
                          want_cand = True, slab_rows = 1, tile_entries = 3000, score_dtype = torch.float64)
        a, b = ids_to_set(got.cand[0]), ids_to_set(c_vll[r])
        if r == 15:
            assert got.cand[0].tolist() == c_ref[r].tolist(), "tiled tie row differs from reference"
        else:
            assert a == b, f"tiled writer row {r}: {len(a)} vs {len(b)}"
    check("tiled select_topk candidate writer == vLLM on the same rows (-inf, NaN, ties)", True)


def check_port(gen, scb = None):
    """Every check that needs no reference code (the candidate stage also runs against
    DeepSeek's select_candidate_blocks when `scb` is given)."""
    # ------------------------------------------------------------------ 1. tiled == untiled
    H, D = 8, 32
    n_cfg = 0
    ties_seen = 0
    for T, seq, m, pos0, slab, tile in [
        (600, 40, 1, 560, 7, 64),
        (600, 600, 1, 0, 256, 200),               # non-multiple tile, one-shot prefill
        (5000, 50, 2, 9950, 16, 1000),
        (5000, 300, 2, 4700, 33, 4096),
        (70000, 64, 1, 69936, 64, 65536),         # two tiles, the second short
        (70000, 48, 1, 69952, 9, 7000),           # tile not a power of two, 10 tiles
        (70000, 40, 2, 139960, 13, 12345),        # rounds up to 12352
    ]:
        ec = min(T, (pos0 + seq) // m)
        q, w, k = ints(seq, T, H, D, gen)
        sc = S._torch_scores(q, w, k[:ec], pos0, m, ec, 1.0, torch.float16)
        ties_seen += boundary_ties(sc, 512)
        for mode in ("plain", "source"):
            ref = select_topk_reference(q, w, k, pos0 = pos0, m = m, ec = ec, want_cand = mode == "source")
            got = select_topk(q, w, k, pos0 = pos0, m = m, ec = ec, want_cand = mode == "source",
                              slab_rows = slab, tile_entries = tile)
            assert torch.equal(got.indices, ref.indices), (T, seq, m, pos0, slab, tile, mode)
            if mode == "source":
                assert torch.equal(got.cand, ref.cand), (T, seq, m, pos0, slab, tile, "cand")
                qu, wu, _ = ints(seq, T, H, D, gen)
                ru = select_topk_reference(qu, wu, k, pos0 = pos0, m = m, ec = ec, cand_in = ref.cand)
                gu = select_topk(qu, wu, k, pos0 = pos0, m = m, ec = ec, cand_in = got.cand,
                                 slab_rows = slab, tile_entries = tile)
                assert torch.equal(gu.indices, ru.indices), (T, seq, m, pos0, slab, tile, "user")
            n_cfg += 1
    check("tiled == untiled on integer inputs (T 600/5000/70000, m 1/2, odd slabs and tiles)",
          ties_seen > 0, f"{n_cfg} runs incl. candidates and users, bitwise; {ties_seen} rows with ties "
                          f"straddling the top-512 cut, all resolved index-ascending")
    q, w, k = conts(30, 30000, H, D, gen)
    ref = select_topk_reference(q, w, k, pos0 = 29970, m = 1, ec = 30000, want_cand = True, score_dtype = torch.float64)
    got = select_topk(q, w, k, pos0 = 29970, m = 1, ec = 30000, want_cand = True, slab_rows = 7,
                      tile_entries = 3000, score_dtype = torch.float64)
    check("tiled == untiled on continuous float64 inputs (T 30000, candidates)",
          torch.equal(got.indices, ref.indices) and torch.equal(got.cand, ref.cand))
    # the tie rule itself
    s = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0, NEG_INF, 3.0]])
    i, v = C.topk_total_order(s, 3)
    check("tie rule: score descending, index ascending", i.tolist() == [[1, 2, 4]])

    # ------------------------------------------------------------------ 2. fill boundaries
    q, w, k = ints(1026, 1026, 4, 16, gen)
    r = select_topk(q[:512], w[:512], k, pos0 = 0, m = 1, ec = 512)
    r2 = select_topk(q[:513], w[:513], k, pos0 = 0, m = 1, ec = 513)
    ok = r.indices is None and r.k_len == 0 and r2.indices is not None and r2.k_len == 512
    ok = ok and r2.indices.shape == (513, 512)
    ok = ok and torch.equal(r2.indices[511, :512], torch.arange(512, dtype = torch.int32))
    ok = ok and (r2.indices[100, 101:] == -1).all().item() and (r2.indices[512] >= 0).all().item()
    check("fill boundary m=1: ec 512 -> None, 513 -> indices", ok,
          "row 511 selects all 512, row 100 has 101 then -1, row 512 has 512 of 513")
    r = select_topk(q[:1025], w[:1025], k, pos0 = 0, m = 2, ec = 1025 // 2)
    r2 = select_topk(q[:1026], w[:1026], k, pos0 = 0, m = 2, ec = 1026 // 2)
    ok = r.indices is None and r2.indices is not None and r2.indices.shape == (1026, 512)
    ok = ok and (r2.indices[0] == -1).all().item() and r2.indices[1, 0].item() == 0 and (r2.indices[1, 1:] == -1).all().item()
    check("fill boundary m=2: seq_len 1025 (ec 512) -> None, 1026 (ec 513) -> indices", ok,
          "row 0 sees nothing, row 1 sees entry 0")

    check_candidate_stage(gen, scb)

    # ------------------------------------------------------------------ 4. the mask is a no-op up to 16384
    P = 16384
    vis = torch.tensor([513, 1000, 8191, 16383, 16384, 16384])
    sc = torch.randn((len(vis), P), generator = gen, dtype = torch.float64)
    sc = sc.masked_fill(torch.arange(P)[None] >= vis[:, None], NEG_INF)
    cb = C.publish_candidate_blocks(sc, vis, 8, 2048)
    check("mask is a no-op for T <= 16384 (reference)",
          torch.equal(C.mask_scores_to_candidates(sc, cb, 8), sc),
          f"every visible block kept: {(cb >= 0).sum(1).tolist()}")
    H, D = 8, 32
    q, w, k = ints(64, 16392, H, D, gen)
    qu, wu, _ = ints(64, 16392, H, D, gen)
    src = select_topk(q, w, k, pos0 = 16384 - 64, m = 1, ec = 16384, want_cand = True, tile_entries = 4096)
    a = select_topk(qu, wu, k, pos0 = 16384 - 64, m = 1, ec = 16384, cand_in = src.cand, tile_entries = 4096)
    b = select_topk(qu, wu, k, pos0 = 16384 - 64, m = 1, ec = 16384, tile_entries = 4096)
    check("mask is a no-op for T <= 16384 (tiled select, users with and without cand_in)",
          torch.equal(a.indices, b.indices))
    src = select_topk(q[:8], w[:8], k, pos0 = 16392 - 8, m = 1, ec = 16392, want_cand = True, tile_entries = 4096)
    check("16385+ entries: 2049 visible blocks, one dropped", ((src.cand >= 0).sum(1) == 2048).all().item()
          and (src.cand[-1] == 2048).any().item(), "newest block 2048 pinned in")

    # ------------------------------------------------------------------ 5. paged == dense
    for m, epp, T in [(1, 256, 9000), (2, 128, 3000)]:
        q, w, k = ints(40, T, H, D, gen)
        pages = -(-T // epp)
        phys = torch.randperm(pages + 7, generator = gen)[:pages].int()
        pool = torch.zeros((pages + 7, epp, D), dtype = torch.half)
        flat = pool.view(-1, D)
        e = torch.arange(T)
        flat[phys[e // epp].long() * epp + e % epp] = k
        pos0 = T * m - 40
        dense = select_topk(q, w, k, pos0 = pos0, m = m, ec = T, want_cand = True, tile_entries = 1000)
        paged = select_topk(q, w, pool, bt_row = phys[None], epp = epp, pos0 = pos0, m = m, ec = T,
                            want_cand = True, tile_entries = 1000)
        ref = select_topk_reference(q, w, pool, bt_row = phys[None], epp = epp, pos0 = pos0, m = m, ec = T,
                                    want_cand = True)
        assert torch.equal(dense.indices, paged.indices) and torch.equal(dense.cand, paged.cand)
        assert torch.equal(paged.indices, ref.indices) and torch.equal(paged.cand, ref.cand)
    check("paged pool through random block tables == dense (m=1 epp 256, m=2 epp 128)", True,
          "tiles rounded to the page size")

    # few rows widen the tile within the same budget (powers of two, capped at the pool)
    q, w, k = ints(3, 70000, H, D, gen)
    a = select_topk(q, w, k, pos0 = 69997, m = 1, ec = 70000, want_cand = True, tile_entries = 4096)
    b = select_topk(q, w, k, pos0 = 69997, m = 1, ec = 70000, want_cand = True, tile_entries = 4096, slab_rows = 3)
    widths = (S._effective_tile(3, 70000, 256, 4096, 8), S._effective_tile(3, 70000, 3, 4096, 8),
              S._effective_tile(256, 1 << 20, 256, 65536, 256), S._effective_tile(1, 1 << 20, 256, 65536, 256),
              S._effective_tile(100, 1 << 20, 256, 65536, 256))
    check("few rows: wider tiles, same result", torch.equal(a.indices, b.indices) and torch.equal(a.cand, b.cand)
          and widths == (131072, 4096, 65536, 1 << 20, 131072),
          f"3 rows: tile 4096 -> {widths[0]}; 1M pool: 256 rows {widths[2]}, 1 row {widths[3]}, 100 rows {widths[4]}")

    # ------------------------------------------------------------------ 6. transient bytes at 1M
    tb_s = transient_bytes(4096, 1 << 20, want_cand = True)
    tb_u = transient_bytes(4096, 1 << 20, cand_in = True)
    tb_p = transient_bytes(4096, 1 << 20)
    untiled = 4096 * (1 << 20) * 2
    check("transient_bytes(4096, 1M) <= 512 MiB", max(tb_s, tb_u, tb_p) <= 512 * MiB,
          f"source {tb_s / MiB:.1f} MiB, user {tb_u / MiB:.1f} MiB, plain {tb_p / MiB:.1f} MiB; "
          f"untiled V4 score matrix {untiled / MiB:.0f} MiB")

    # ------------------------------------------------------------------ ablation and argument checks
    q, w, k = ints(16, 20000, H, D, gen)
    src = select_topk(q, w, k, pos0 = 19984, m = 1, ec = 20000, want_cand = True, tile_entries = 8192)
    qu, wu, _ = ints(16, 20000, H, D, gen)
    masked = select_topk(qu, wu, k, pos0 = 19984, m = 1, ec = 20000, cand_in = src.cand, tile_entries = 8192)
    plain = select_topk(qu, wu, k, pos0 = 19984, m = 1, ec = 20000, tile_entries = 8192)
    with ablate("engram, no_candidates"):
        abl = select_topk(qu, wu, k, pos0 = 19984, m = 1, ec = 20000, cand_in = src.cand, tile_entries = 8192)
        abl_src = select_topk(q, w, k, pos0 = 19984, m = 1, ec = 20000, want_cand = True, tile_entries = 8192)
    check("ablation no_candidates: cand None, cand_in ignored",
          abl_src.cand is None and torch.equal(abl.indices, plain.indices)
          and not torch.equal(masked.indices, plain.indices))
    try:
        select_topk(q, w, k, pos0 = 19984, m = 1, ec = 20000, want_cand = True, cand_in = src.cand)
        raised = False
    except ValueError:
        raised = True
    check("want_cand together with cand_in is rejected", raised)
    # arguments the kernels cannot validate: each of these would otherwise run and select from
    # the wrong data (a one-row cand_in broadcasts over all 16 rows and changes 15 of them; a
    # page table one page short makes the ext scorer read the page ids stored after it; a
    # two-row batch table whose rows are each too short, but together long enough, is read
    # as one row, so selection reaches the second sequence's pages; a paged pool whose pages
    # hold 256 entries addressed with epp = 512 reads the wrong rows)
    pages = -(-20000 // 256)
    pool = torch.zeros((pages, 256, D), dtype = torch.half)
    pool.view(-1, D)[:20000] = k
    bt = torch.arange(pages, dtype = torch.int32)[None]
    bt_batch = torch.arange(2 * (pages // 2 + 1), dtype = torch.int32).view(2, -1)
    base = dict(pos0 = 19984, m = 1, ec = 20000, tile_entries = 8192)
    bad_calls = {
        "one-row cand_in for 16 query rows": lambda: select_topk(qu, wu, k, cand_in = src.cand[:1], **base),
        "wts rows != q rows": lambda: select_topk(q, w[:8], k, **base),
        "q with a leading batch dim": lambda: select_topk(q[None], w[None], k, **base),
        "pool head dim != q head dim": lambda: select_topk(q, w, k[:, : D // 2], **base),
        "page table shorter than ec": lambda: select_topk(q, w, pool, bt_row = bt[:, :-1], epp = 256, **base),
        "batch (2, P) page table": lambda: select_topk(q, w, pool, bt_row = bt_batch, epp = 256, **base),
        "paged pool pages != epp": lambda: select_topk(q, w, pool, bt_row = bt, epp = 512, **base),
        "dense pool shorter than ec": lambda: select_topk(q, w, k[:19999], **base),
        "reference, one-row cand_in": lambda: select_topk_reference(qu, wu, k, pos0 = 19984, m = 1, ec = 20000,
                                                                   cand_in = src.cand[:1]),
        "reference, batch page table": lambda: select_topk_reference(q, w, pool, bt_row = bt_batch, epp = 256,
                                                                    pos0 = 19984, m = 1, ec = 20000),
        # 16 rows x a 2^27-entry tile: the int32 offsets of the score tile would wrap
        "slab x tile budget past int32": lambda: select_topk(q, w, k, pos0 = 19984, m = 1, ec = 20000,
                                                             tile_entries = 1 << 27),
    }
    missed = []
    for name, call in bad_calls.items():
        try:
            call()
            missed.append(name)
        except ValueError:
            pass
    ok_paged = torch.equal(select_topk(q, w, pool, bt_row = bt, epp = 256, **base).indices,
                           select_topk(q, w, k, **base).indices)
    check("argument checks: misshapen q/wts/cand_in, short or batch page table, page size != epp, "
          "short pool, head-dim mismatch, an int32-wrapping budget rejected",
          not missed and ok_paged, f"{len(bad_calls)} bad calls raise ValueError; missed: {missed}")


def main():
    t_start = time.time()
    gen = torch.Generator().manual_seed(0)
    with ablate(""):
        scb = load_deepseek()
        if scb is None:
            skip("DeepSeek-oracle sections", "DSV41_DEEPSEEK_REF not set")
        else:
            check_vs_deepseek(scb, gen)
        check_port(gen, scb)
    print(f"  all select tests passed in {time.time() - t_start:.1f} s")


def test_select():
    """pytest entry point: the checks that need no reference code."""
    with ablate(""):
        check_port(torch.Generator().manual_seed(0))


def test_select_vs_deepseek():
    """pytest entry point: the DeepSeek-oracle sections, skipped without DSV41_DEEPSEEK_REF."""
    scb = load_deepseek()
    if scb is None:
        import pytest
        pytest.skip("DeepSeek-oracle sections: DSV41_DEEPSEEK_REF not set")
    gen = torch.Generator().manual_seed(0)
    with ablate(""):
        check_vs_deepseek(scb, gen)
        check_candidate_stage(gen, scb)


if __name__ == "__main__":
    main()
