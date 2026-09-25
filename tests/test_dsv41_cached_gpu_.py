"""DeepSeek-V4.1 cached attention vs the stateless path, on real single-layer weights, each GPU.

    PYTHONPATH=$PWD python3 tests/test_dsv41_cached_gpu_.py [checkpoint-dir] [--quick] [--numerics S]

(checkpoint-dir defaults to $DSV41_MODEL_DIR; the DeepSeek-oracle part of the candidate check,
the comparison with DeepSeek's select_candidate_blocks, needs DSV41_DEEPSEEK_REF and is skipped
without it; the rest of that check runs either way.) --numerics sets config.dsv41_numerics for
the run: 'precise' by default, the setting the gates below were measured with; the model's
default, deepseek:index, rounds the index query and keys identically on both paths, and the
selection checks read the rounded values.

Loads single attention layers 2 (rate-2 kv + index source), 3 (its consumer), 8 (rate-2
source), 12 (consumer of 8), 20 (rate-1 kv + index + candidate source) and 24 (index source
borrowing 20's K; a candidate user) on one card, and repeats on every visible card. With two
or more cards it then repeats the same checks with layer 12 on the other card, reading a replica
of 8's pool (the split of an explicit placement at 12), with each of the first two cards as home.
Never the full model; a few GiB per card. Each layer gets the same synthetic hidden
states. For T = 700 and 3000 tokens the cached path runs chunked prefill at chunk sizes 2048,
1000 and 7 with token-by-token decode spliced in -- across the rate-1 top-k threshold (512 ->
513 entries) at 700, across the rate-2 one (1025 -> 1026 tokens) at 3000, and from an odd
prefill length so the first decode step closes a carried rate-2 group -- and every layer's
per-row output is compared with the stateless one-shot pass over the same T tokens.

TWO PROJECTION MODES, because the comparison is only as sharp as its noise floor:

  exact    every Linear of the tested layers is replaced by a float64 matmul of its own
           dequantized weight (get_weight_tensor): a row's projection no longer depends on
           how many rows share the call. What remains is the attention path itself. Gate:
           rel-L2 <= 1e-3 per layer and schedule. Measured floor 3e-4, all of it dsa_attn's
           few-query split kernel (n_splits = 1 brings it to 7e-5); chunk 100 is bitwise.

  exl3     the production kernels. exllamav3 computes a projection with the exl3 GEMM
           kernel at <= 144 rows (AUTO_RECONSTRUCT_THRESHOLD) and with reconstruct + hgemm
           above, so the rows a chunk-7 or decode step computes differ from the one-shot
           rows before attention sees them (q_a: 7e-3 at 1 row, measured below). Gates:
           rows computed in steps of >= 145 rows: rel-L2 <= 3e-3;
           rows computed in steps of <= 144 rows: rel-L2 <= max(3e-3, 3 x the projection
           difference measured on those very rows).

Top-k near-ties: index scores are stored fp16 and dsa_topk breaks ties by index, so a score
that moves by an ulp can swap one entry at the 512th place, and with these random inputs one
swapped entry moves that row's output by 3e-2..1e-1. Rows whose selection differs from the
stateless one are therefore checked, not compared: the cached selection must be a top-k of the
cached path's OWN inputs (its query, head weights and stored index K, recomputed in fp32) up to
entries within 2e-3 of the k-th score, measured against the row's term scale
mean_j sum_h |w_h| relu(q_h . k_j); the stateless selection is checked the same way against its
inputs. What is left to explain a differing row is then the input difference itself (0 in exact
mode for prefill rows; the exl3 projection difference printed per schedule). Such rows are
counted and left out of the rel-L2.

Also negative controls that must FAIL the exact gate: a carry ring cleared before every
step, and the rope_consecutive and dense_consumers ablations.

And the generation-state paths (state_checks), exact mode, each against a fresh run of the
final token stream at the same step sizes (gate 1e-6: same kernels, so any difference is a
state bug): in-place rewinds replayed with different rows, including rewinds by the full
reported capacity; a checkpoint restored into another slot with the original slot and the
pools past the checkpoint zeroed; a permuted block table; bsz 2 with rows at different
positions; and the candidate stage with effect: cached against stateless, layer 24's
selection inside its candidates and a masked top-k of the recorded inputs, and (with
DSV41_DEEPSEEK_REF) the candidate blocks against DeepSeek's own select_candidate_blocks.
"""
import os, sys, time, math, argparse
import torch

_T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _T)
sys.path.insert(0, os.path.dirname(_T))
from dsv41_ref import ablate, model_dir, skip
from dsv41_ref import deepseek

LAYERS = (2, 3, 8, 12, 20, 24)
SMALL_M = 144                     # exllamav3.modules.quant.exl3.AUTO_RECONSTRUCT_THRESHOLD


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------------------------
# exact projection mode (test only)

def densify(attn):
    from exllamav3.modules.linear import Linear
    for lin in attn.modules:
        if not isinstance(lin, Linear):
            continue
        W = lin.inner.get_weight_tensor().float()           # (in, out), dequantized
        b = lin.inner.get_bias_tensor()

        def fwd(x, params, out_dtype = None, lin = lin, W = W, b = b):
            shp = x.shape
            x2 = x.reshape(-1, shp[-1]).double()
            y = torch.empty((x2.shape[0], W.shape[1]), dtype = torch.float64, device = x.device)
            for c0 in range(0, W.shape[1], 4096):
                y[:, c0:c0 + 4096] = x2 @ W[:, c0:c0 + 4096].double()
            if b is not None:
                y += b.double()
            y = y.to(out_dtype or lin.out_dtype or torch.half).view(*shp[:-1], -1)
            if lin.trim_padded_out and lin.out_features != lin.out_features_unpadded:
                y = y[..., :lin.out_features_unpadded].contiguous()
            return y
        lin.forward = fwd
    attn.wo_a_multi = None          # the grouped o_proj's exl3 mgemm shortcut (seq <= 32)
    attn.woa_multi_ready = True


def undensify(attn):
    from exllamav3.modules.linear import Linear
    for lin in attn.modules:
        if isinstance(lin, Linear):
            lin.__dict__.pop("forward", None)
    attn.woa_multi_ready = False


# ---------------------------------------------------------------------------------------------

def schedule(T, chunk):
    segs = [("pre", 505), ("dec", 195)] if T == 700 else \
           [("pre", 1001), ("dec", 40), ("pre", 1942), ("dec", 17)]
    assert sum(n for _, n in segs) == T
    steps = []
    for kind, n in segs:
        steps += [1] * n if kind == "dec" else [min(chunk, n - i) for i in range(0, n, chunk)]
    return steps


class Rig:
    """The six layers loaded on one card under one Cache; with replica, layer 12 on the other
    card, routed to a replica of 8's pool (the split of an explicit placement at 12)."""

    def __init__(self, cfg, model, A, devs, home, batch = 1, replica = False):
        from exllamav3.cache.cache import Cache
        self.cfg, self.model, self.A = cfg, model, A
        self.dev = {L: devs[home] for L in LAYERS}
        self.replica = replica
        if replica:
            self.dev[12] = devs[1 - home]
            A[12].set_pool_route(12, replica_of = 8)
        model.__dict__.pop("_get_cache_layers", None)
        self.cache = Cache(model, max_num_tokens = 4096 * batch, max_batch_size = batch)
        for L in LAYERS:
            A[L].load(self.dev[L])
        self.mode = "exl3"

    def close(self):
        for L in LAYERS:
            undensify(self.A[L])
            self.A[L].unload()
        self.cache.detach_from_model()
        self.model.__dict__.pop("_get_cache_layers", None)
        if self.replica:
            self.A[12].set_pool_route(8)
        torch.cuda.empty_cache()

    def set_mode(self, mode):
        for L in LAYERS:
            (densify if mode == "exact" else undensify)(self.A[L])
        self.mode = mode

    def inputs(self, T, seed):
        torch.manual_seed(seed)
        x = (torch.randn(1, T, self.cfg.hidden_size) * 0.5).half()
        return {d: x.to(d) for d in set(self.dev.values())}

    def src(self, L):
        return self.cfg.index_source_for(L)

    def stateless(self, xs, record = None):
        """Outputs (rows, hidden) fp32 on CPU, the selection each layer attended over, and --
        into `record` -- each index source's selection inputs (q_idx, weights, index K)."""
        import exllamav3.modules.dsv41 as D
        real = D.select_topk
        cur = [None]
        wrapped = []
        if record is not None:
            inner = real
            def sel_wrapped(*a, **kw):
                record[cur[0]] = (a[0].float().cpu(), a[1].float().cpu(),
                                  a[2].reshape(-1, a[2].shape[-1]).float().cpu())
                return inner(*a, **kw)
            D.select_topk = sel_wrapped
            for L in LAYERS:
                a = self.A[L]
                if a.is_index_source:
                    orig = a._index_select
                    def isel(*args, L = L, orig = orig):
                        cur[0] = L
                        return orig(*args)
                    a._index_select = isel
                    wrapped.append(a)
        try:
            p = {"attn_mode": "flash_attn_nc", "position": 0, "dsv41_pools": {}, "dsv41_topk": {},
                 "dsv41_candidates": {}}
            out = {}
            with torch.inference_mode():
                for L in LAYERS:
                    out[L] = self.A[L].forward(xs[self.dev[L]], p)[0].float().cpu()
        finally:
            D.select_topk = real
            for a in wrapped:
                a.__dict__.pop("_index_select", None)
        sel = {L: [(0, p["dsv41_topk"].get((self.src(L), 0), (None, 0))[0])] for L in LAYERS}
        return out, sel

    def cached(self, xs, steps, hook = None):
        """Outputs, the selection every layer attended over at each step, and each index
        source's selection inputs as the cached path saw them: its per-row query and head
        weights, and (read after the run: entries never change once written) its index K."""
        import exllamav3.modules.dsv41 as D
        st = self.cache.get_new_state()
        outs = {L: [] for L in LAYERS}
        sels = {L: [] for L in LAYERS}
        qin = {L: [] for L in LAYERS if self.A[L].is_index_source}
        cur = [None]
        real = D.select_topk
        inner = real
        def sel_wrapped(*a, **kw):
            qin[cur[0]].append((kw["pos0"], a[0].float().cpu(), a[1].float().cpu()))
            return inner(*a, **kw)
        D.select_topk = sel_wrapped
        wrapped = []
        for L in LAYERS:
            a = self.A[L]
            orig = a._select_cached
            def rec(*args, L = L, orig = orig):
                r = orig(*args)
                sels[L].append((args[6], r[0]))      # (pos0, indices or None)
                return r
            a._select_cached = rec
            if a.is_index_source:
                orig_i = a._index_select
                def isel(*args, L = L, orig_i = orig_i):
                    cur[0] = L
                    return orig_i(*args)
                a._index_select = isel
            wrapped.append(a)
        try:
            pos = 0
            with torch.inference_mode():
                for n in steps:
                    if hook is not None:
                        hook(self, st, pos)
                    p = {"attn_mode": "flash_attn", "recurrent_states": [st]}
                    for L in LAYERS:
                        outs[L].append(self.A[L].forward(xs[self.dev[L]][:, pos:pos + n], p)[0].float().cpu())
                    st.position += n
                    st.post_advance()
                    pos += n
        finally:
            D.select_topk = real
            for a in wrapped:
                a.__dict__.pop("_select_cached", None)
                a.__dict__.pop("_index_select", None)
            st.free()
        T = pos
        crec = {}
        for L, parts in qin.items():
            if not parts:
                continue
            q = torch.zeros((T,) + tuple(parts[0][1].shape[1:]))
            w = torch.zeros((T, parts[0][2].shape[1]))
            for p0, qq, ww in parts:
                q[p0:p0 + qq.shape[0]] = qq
                w[p0:p0 + ww.shape[0]] = ww
            kl = self.cache.layers[(self.A[L].pool_owner_layer, 0)]
            K = kl.pool_idx.reshape(-1, kl.pool_idx.shape[-1]).float().cpu()
            crec[L] = (q, w, K)
        return {L: torch.cat(v, dim = 0) for L, v in outs.items()}, sels, crec

    def pools(self, L):
        kl = self.cache.layers[(L, 0)]
        return {n: getattr(kl, n).clone() for n in ("pool_c", "pool_r", "pool_idx") if getattr(kl, n) is not None}


def selection_masks(sel, T, m, device = "cpu"):
    """(T, T // m) bool: the entries each row attended over. sel: [(pos0, indices|None)] in
    row order (one entry per step, the stateless pass is one step at pos0 0)."""
    ec = T // m
    out = torch.zeros((T, ec), dtype = torch.bool)
    starts = [p for p, _ in sel] + [T]
    for i, (p0, idx) in enumerate(sel):
        p1 = starts[i + 1]
        r = torch.arange(p0, p1)
        vis = ((r + 1) // m).clamp(max = ec)
        if idx is None:
            out[p0:p1] = torch.arange(ec).unsqueeze(0) < vis.unsqueeze(1)
        else:
            ii = idx.cpu().long()[: p1 - p0]
            ii = torch.where(ii >= 0, ii, torch.full_like(ii, ec))
            buf = torch.zeros((p1 - p0, ec + 1), dtype = torch.bool)
            buf.scatter_(1, ii, True)
            out[p0:p1] = buf[:, :ec]
    return out


def tie_check(rows, sel, q, w, K, m, k, tol):
    """For each row, `sel` (T, entries) must be a top-k of the fp32 scores recomputed from that
    row's own inputs (q, w, K) up to near-ties: exactly min(k, visible) entries, and every entry
    where it differs from torch.topk within tol of the k-th score. Distances are relative to
    the row's term scale, mean_j sum_h |w_h| relu(q_h . k_j): head weights carry either sign, so
    a score can sit near zero while its terms -- and their fp16 rounding -- do not.
    Returns ([(row, distance)] that fail, worst distance)."""
    worst = 0.0
    bad = []
    Kf = K.float()
    for r in rows:
        vis = min((r + 1) // m, Kf.shape[0])
        t = torch.relu(q[r].float() @ Kf[:vis].T)                       # (H, vis)
        s = (t * w[r].float().unsqueeze(1)).sum(0)
        scale = max((t * w[r].float().abs().unsqueeze(1)).sum(0).mean().item(), 1e-12)
        kk = min(k, vis)
        top = s.topk(kk)
        thr = top.values[-1].item()
        ref = torch.zeros(vis, dtype = torch.bool)
        ref[top.indices] = True
        mine = sel[r, :vis]
        if int(mine.sum()) != kk or sel[r, vis:].any():
            bad.append((r, float("inf")))
            continue
        diff = (mine ^ ref).nonzero().flatten()
        d = ((s[diff] - thr).abs().max().item() / scale) if len(diff) else 0.0
        worst = max(worst, d)
        if d > tol:
            bad.append((r, d))
    return bad, worst


def compare(rig, ref, sel_ref, got, sel_got, crec, steps, T, gate_big, gate_small, tag, proj_noise = None):
    """Per layer: rows whose selection differs from the stateless one are checked, not
    compared -- the cached selection must be a top-k of its OWN inputs up to near-ties
    (tie_check, 2e-3; the stateless one is checked the same way by check_selection), which
    leaves input noise as the only cause. The rel-L2 runs over the remaining rows, split by
    the size of the step that computed them."""
    cfg = rig.cfg
    step_n = torch.zeros(T, dtype = torch.long)
    pos = 0
    for n in steps:
        step_n[pos:pos + n] = n
        pos += n
    small = step_n <= SMALL_M
    ok = True
    lines = []
    for L in LAYERS:
        m = cfg.compress_ratios[L]
        S = rig.src(L)
        ma = selection_masks(sel_ref[L], T, m)
        mb = selection_masks(sel_got[L], T, m)
        diff_rows = (ma != mb).any(1).nonzero().flatten().tolist()
        bad, worst = [], 0.0
        if diff_rows:
            q, w, K = crec[S]
            bad, worst = tie_check(diff_rows, mb, q, w, K[: T // m], m, cfg.index_topk, 2e-3)
        keep = torch.ones(T, dtype = torch.bool)
        keep[diff_rows] = False
        big_rows = keep & ~small
        sm_rows = keep & small
        e_big = rel(got[L][big_rows], ref[L][big_rows]) if big_rows.any() else 0.0
        e_small = rel(got[L][sm_rows], ref[L][sm_rows]) if sm_rows.any() else 0.0
        g_small = gate_small if proj_noise is None else max(gate_small, 3 * proj_noise[L])
        passed = not bad and e_big <= gate_big and e_small <= g_small
        ok &= passed
        lines.append(f"    L{L:2d} (m{m}, dev {str(rig.dev[L])[-1]}) rows>144 {e_big:.1e}  rows<=144 {e_small:.1e}"
                     f" (gate {g_small:.0e})  selection differs on {len(diff_rows)}/{T} rows"
                     f"{f', all valid top-k of the cached inputs (worst tie {worst:.1e})' if diff_rows and not bad else ''}"
                     f"{f'  INVALID cached selection on {len(bad)} rows, e.g. row {bad[0][0]} at {bad[0][1]:.1e}' if bad else ''}"
                     f"  {'ok' if passed else 'FAIL'}")
    print(f"  {'PASS' if ok else 'FAIL'} {tag}")
    for l in lines:
        print(l)
    return ok


def projection_noise(rig, xs, steps, T):
    """exl3 mode, per layer: the rel difference of q_res and kv between the step-wise and the
    one-shot projections, taken separately for each size of step <= 144 rows (a decode step
    and a 7-row chunk run different exl3 kernels) and maxed."""
    out = {}
    with torch.inference_mode():
        for L in LAYERS:
            a = rig.A[L]
            x = xs[rig.dev[L]]
            qr1, _, kv1 = a._project_qkv(x, {}, 0)
            qr1, kv1 = qr1[0].cpu(), kv1[0].cpu()
            parts_q, parts_k, sizes = [], [], []
            pos = 0
            for n in steps:
                qr, _, kv = a._project_qkv(x[:, pos:pos + n], {}, pos)
                parts_q.append(qr[0].cpu()); parts_k.append(kv[0].cpu()); sizes += [n] * n
                pos += n
            qr2, kv2 = torch.cat(parts_q), torch.cat(parts_k)
            sizes = torch.tensor(sizes)
            worst = 0.0
            for n in sorted(set(sizes.tolist())):
                if n > SMALL_M:
                    continue
                mk = sizes == n
                worst = max(worst, rel(qr2[mk], qr1[mk]), rel(kv2[mk], kv1[mk]))
            out[L] = worst
    return out


def check_selection(rig, rec, sel_ref, T, mode):
    """Stateless selection of each index source vs an fp32 torch top-k from its own inputs
    (every differing row must be a near-tie)."""
    cfg = rig.cfg
    ok = True
    for L in LAYERS:
        m = cfg.compress_ratios[L]
        ec = T // m
        if not cfg.is_index_source(L) or ec <= cfg.index_topk:
            continue
        q, w, K = rec[L]
        K = K[:ec]
        mine = selection_masks(sel_ref[L], T, m)
        tops = torch.zeros((T, ec), dtype = torch.bool)
        for r0 in range(0, T, 256):
            r1 = min(r0 + 256, T)
            s = torch.einsum("rhd,td->rht", q[r0:r1], K).relu_()
            s = (s * w[r0:r1].unsqueeze(-1)).sum(1)
            vis = ((torch.arange(r0, r1) + 1) // m).clamp(max = ec)
            s = s.masked_fill(torch.arange(ec)[None] >= vis[:, None], -math.inf)
            v, ii = s.topk(min(cfg.index_topk, ec), dim = -1)
            ii = torch.where(v > -math.inf, ii, torch.full_like(ii, ec))
            buf = torch.zeros((r1 - r0, ec + 1), dtype = torch.bool)
            buf.scatter_(1, ii, True)
            tops[r0:r1] = buf[:, :ec]
        drows = (mine != tops).any(1).nonzero().flatten().tolist()
        bad, worst = tie_check(drows, mine, q, w, K, m, cfg.index_topk, 2e-3)
        good = not bad
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} [{mode}] T={T} stateless selection of L{L} vs fp32 torch top-k: "
              f"{len(drows)}/{T} rows differ, " +
              (f"all near-ties (worst {worst:.1e})" if good else f"{len(bad)} NOT near-ties, e.g. {bad[:3]}"))
    return ok


# ---------------------------------------------------------------------------------------------
# generation-state paths: exact projections, bitwise against a fresh run of the same steps

def drive(rig, ops, st = None, bt = None):
    """bsz 1 forward sequence. ops: ("fwd", x (1, n, H) cpu half) | ("rewind", k or "cap") |
    ("stash", name). Returns (state, {L: {position: output row}}, stashes, rewind log); a
    replayed position keeps its last output."""
    st = st if st is not None else rig.cache.get_new_state()
    out = {L: {} for L in LAYERS}
    stashes, log = {}, []
    with torch.inference_mode():
        for op in ops:
            if op[0] == "fwd":
                x, pos = op[1], st.position
                p = {"attn_mode": "flash_attn", "recurrent_states": [st]}
                if bt is not None:
                    p["block_table"] = bt
                for L in LAYERS:
                    y = rig.A[L].forward(x.to(rig.dev[L]), p)[0].float().cpu()
                    for i in range(x.shape[1]):
                        out[L][pos + i] = y[i]
                st.position += x.shape[1]
                st.post_advance()
            elif op[0] == "rewind":
                k = st.rollback_capacity() if op[1] == "cap" else op[1]
                log.append((st.position, k))
                st.rewind(k)
            else:
                stashes[op[1]] = st.stash()
    return st, out, stashes, log


def dec(x, p0, p1):
    return [("fwd", x[:, i:i + 1]) for i in range(p0, p1)]


def same(a, b, positions, tag):
    """Same kernels at the same step sizes: any difference is a state bug. Gate 1e-6."""
    e = {L: rel(torch.stack([a[L][p] for p in positions]), torch.stack([b[L][p] for p in positions]))
         for L in LAYERS}
    good = all(v <= 1e-6 for v in e.values())
    print(f"  {'PASS' if good else 'FAIL'} [exact] {tag}: " + " ".join(f"L{L} {v:.1e}" for L, v in e.items()))
    return good


def state_checks(rig, cfg):
    """
    What the schedules above do not reach, each against a fresh run of the final token stream
    with the same step sizes:
      rewinds   in place, replayed with DIFFERENT rows: to an odd position (the carry's pending
                row), a second rewind after a short replay (bounded from the high-water
                position), and rewinds by the full reported capacity (ring- and carry-limited)
      restore   a checkpoint taken at an odd position (window-trimmed ring, pending carry row)
                restored into another slot that shares a permuted block table, with the
                original slot's states and every pool entry past the checkpoint zeroed
      tables    a permuted generator-style block table vs the slot identity
      rows      bsz 2 with rows at different positions, one crossing the top-k threshold
      cands     the candidate stage with effect (candidate_topk_blocks 72 < 138 blocks)
                vs DeepSeek's select_candidate_blocks and masked top-k on the recorded inputs
    """
    H = cfg.hidden_size
    ok = True
    torch.manual_seed(5)
    X = (torch.randn(1, 1400, H) * 0.5).half()
    ALT = (torch.randn(1, 600, H) * 0.5).half()

    ops = [("fwd", X[:, :901])] + dec(X, 901, 961) + [("rewind", 10)] + dec(ALT, 0, 5) + [("rewind", 3)]
    ops += [("fwd", ALT[:, 5:12])] + dec(ALT, 12, 152)
    F = torch.cat([X[:, :951], ALT[:, 0:2], ALT[:, 5:152]], dim = 1)
    st, a, _, log = drive(rig, ops)
    st.free()
    st, b, _, _ = drive(rig, [("fwd", F[:, :901])] + dec(F, 901, 953) + [("fwd", F[:, 953:960])] + dec(F, 960, 1100))
    st.free()
    ok &= same(a, b, range(1100), f"rewinds {log} (odd landing; second rewind after a 5-row replay)")

    for pre, dec_to, first, end in ((700, 780, None, 900), (1001, 1300, 100, 1320)):
        ops = [("fwd", X[:, :pre])] + dec(X, pre, dec_to)
        if first:
            ops += [("rewind", first)] + dec(ALT, 0, 20)
        ops += [("rewind", "cap")]
        st, a, _, log = drive(rig, ops)
        p = st.position
        # the stream below p is X: the capacity rewind lands below any replayed rows
        assert p <= (dec_to - first if first else dec_to) and 100 + end - p <= ALT.shape[1], (p, log)
        st, a2, _, _ = drive(rig, dec(ALT, 100, 100 + end - p), st = st)
        st.free()
        for L in LAYERS:
            a[L].update(a2[L])
        F = torch.cat([X[:, :p], ALT[:, 100:100 + end - p]], dim = 1)
        q = min(p, pre)
        st, b, _, _ = drive(rig, [("fwd", F[:, :q])] + dec(F, q, end))
        st.free()
        ok &= same(a, b, range(end), f"rewind by the full capacity {log} -> {p}, replayed with different rows")

    # restore + permuted table. The identity run goes first: its slot's identity pages overlap
    # the permuted table's pages
    pages = torch.tensor([[17, 3, 29, 8, 21]], dtype = torch.int32)
    st, ident, _, _ = drive(rig, [("fwd", X[:, :901])] + dec(X, 901, 1100))
    st.free()
    # only the loaded layers' states are materialized; a checkpoint walks every layer state
    full = rig.cache.recurrent_layers
    rig.cache.recurrent_layers = {k: v for k, v in full.items() if k[0] in LAYERS}
    try:
        st0, perm, stashes, _ = drive(rig, [("fwd", X[:, :901])] + dec(X, 901, 951) + [("stash", "c")] +
                                     dec(X, 951, 1100), bt = pages)
        ok &= same(perm, ident, range(1100), f"block table {pages.tolist()[0]} vs slot identity")
        from exllamav3.modules.dsv41_cached import entry_rows
        for rsl in rig.cache.recurrent_layers.values():
            for t in rsl._tensors():
                t[st0.slot].zero_()

        def zero_pools_past(position):
            for kl in rig.cache.layers.values():
                e = torch.arange(position // kl.compress_rate, kl.epp * pages.shape[1], device = kl.device)
                rows = entry_rows(pages.to(kl.device), e, kl.epp)
                for n in ("pool_c", "pool_r", "pool_idx"):
                    t = getattr(kl, n)
                    if t is not None:
                        t.view(-1, t.shape[-1]).index_fill_(0, rows, 0)

        zero_pools_past(951)
        st1 = rig.cache.new_from_stashed(stashes["c"], 951)
        assert st1.slot != st0.slot
        st1, res, _, _ = drive(rig, dec(X, 951, 1100), st = st1, bt = pages)
        ok &= same(res, perm, range(951, 1100), f"checkpoint at 951 ({stashes['c']['checkpoint_size']:,} B) "
                                               f"restored into slot {st1.slot}, slot {st0.slot} and pools past it zeroed")
        st1.free()
        # negative control: the same checkpoint with its carry rings zeroed must not reproduce it
        bad = {k: ([v[0]] + [torch.zeros_like(t) for t in v[1:]] if isinstance(k, tuple) and len(v) > 1 else v)
               for k, v in stashes["c"].items()}
        zero_pools_past(951)
        st1 = rig.cache.new_from_stashed(bad, 951)
        st1, res, _, _ = drive(rig, dec(X, 951, 1100), st = st1, bt = pages)
        errs = {L: rel(torch.stack([res[L][i] for i in range(951, 1100)]),
                       torch.stack([perm[L][i] for i in range(951, 1100)])) for L in LAYERS}
        caught = all(errs[L] > 1e-3 for L in (2, 3, 8, 12))
        ok &= caught
        print(f"  {'PASS' if caught else 'FAIL'} [exact] negative control, checkpoint with its carry rings zeroed: " +
              " ".join(f"L{L} {v:.1e}{'*' if L in (2, 3, 8, 12) else ''}" for L, v in errs.items()) +
              "  (* must exceed 1e-3)")
        st0.free(); st1.free()
    finally:
        rig.cache.recurrent_layers = full

    # bsz 2, rows at different positions, generator-style table
    XA, XB = X[:, :1200], ALT[:, :]
    bt2 = torch.tensor([[11, 2, 30, 7, 19], [5, 26, 14, 0, 9]], dtype = torch.int32)
    st, refA, _, _ = drive(rig, [("fwd", XA[:, :901])] + dec(XA, 901, 1151), bt = bt2[0:1])
    st.free()
    st, refB, _, _ = drive(rig, [("fwd", XB[:, :301])] + dec(XB, 301, 551), bt = bt2[1:2])
    st.free()
    s0, s1 = rig.cache.get_new_state(), rig.cache.get_new_state()
    got = [{L: {} for L in LAYERS}, {L: {} for L in LAYERS}]
    with torch.inference_mode():
        for st, x, n, b in ((s0, XA, 901, 0), (s1, XB, 301, 1)):
            p = {"attn_mode": "flash_attn", "recurrent_states": [st], "block_table": bt2[b:b + 1]}
            for L in LAYERS:
                y = rig.A[L].forward(x[:, :n].to(rig.dev[L]), p)[0].float().cpu()
                for i in range(n):
                    got[b][L][i] = y[i]
            st.position += n
            st.post_advance()
        for _ in range(250):
            pa, pb = s0.position, s1.position
            xb = torch.cat([XA[:, pa:pa + 1], XB[:, pb:pb + 1]], dim = 0)
            p = {"attn_mode": "flash_attn", "recurrent_states": [s0, s1], "block_table": bt2}
            for L in LAYERS:
                y = rig.A[L].forward(xb.to(rig.dev[L]), p).float().cpu()
                got[0][L][pa], got[1][L][pb] = y[0, 0], y[1, 0]
            for st in (s0, s1):
                st.position += 1
                st.post_advance()
    s0.free(); s1.free()
    ok &= same(got[0], refA, range(1151), "bsz 2 row 0 (decode 901..1150, top-k regime) vs alone")
    ok &= same(got[1], refB, range(551), "bsz 2 row 1 (decode 301..550, crosses 512 entries at rate 1) vs alone")

    ok &= candidates_active(rig, cfg, X[:, :1100])
    return ok


def candidates_active(rig, cfg, X, n_blocks = 72, block = 8):
    import exllamav3.modules.dsv41 as D
    from exllamav3.modules.dsv41_cached import entry_rows
    # only the candidate comparison needs DeepSeek's code; the rest runs without it
    ds = deepseek.load_model() if deepseek.ref_dir() is not None else None
    if ds is None:
        skip("candidate blocks vs DeepSeek select_candidate_blocks", "DSV41_DEEPSEEK_REF not set")
    T = X.shape[1]
    srcs = [L for L in LAYERS if rig.A[L].is_index_source]
    saved = {L: rig.A[L].candidate_topk_blocks for L in (20, 24)}
    rec, cur = {}, [None]
    real = D.select_topk
    inner = real
    def sel(*a, **kw):
        r = inner(*a, **kw)
        rec.setdefault(cur[0], []).append(dict(
            pos0 = kw["pos0"], ec = kw["ec"], q = a[0].float().cpu(), w = a[1].float().cpu(),
            cand_in = None if kw["cand_in"] is None else kw["cand_in"].cpu(),
            idx = r.indices.cpu(), cand = None if r.cand is None else r.cand.cpu()))
        return r
    for L in srcs:
        orig = rig.A[L]._index_select
        def isel(*args, L = L, orig = orig):
            cur[0] = L
            return orig(*args)
        rig.A[L]._index_select = isel
    for L in (20, 24):
        rig.A[L].candidate_topk_blocks = n_blocks
    D.select_topk = sel
    try:
        p = {"attn_mode": "flash_attn_nc", "position": 0, "dsv41_pools": {}, "dsv41_topk": {}, "dsv41_candidates": {}}
        with torch.inference_mode():
            ref = {L: rig.A[L].forward(X.to(rig.dev[L]), p)[0].float().cpu() for L in LAYERS}
        K_nc = rig.A[20].resolve_index_pool(p)[:T].float().cpu()
        runs = [("stateless", {L: rec.pop(L) for L in (20, 24)}, K_nc)]
        rec.clear()
        st, got, _, _ = drive(rig, [("fwd", X[:, :701])] + dec(X, 701, 760) + [("fwd", X[:, 760:T])])
        kl = rig.cache.layers[(20, 0)]
        btr = kl.slot_bt(rig.cache.num_slots)[st.slot]
        K_c = kl.pool_idx.reshape(-1, kl.D_i)[entry_rows(btr, torch.arange(T, device = kl.device), kl.epp)].float().cpu()
        st.free()
        runs.append(("cached", {L: rec.pop(L) for L in (20, 24)}, K_c))
    finally:
        D.select_topk = real
        for L in srcs:
            rig.A[L].__dict__.pop("_index_select", None)
        for L in (20, 24):
            rig.A[L].candidate_topk_blocks = saved[L]

    ok = True
    for tag, recs, K in runs:
        n_rows = n_cdiff = n_cbad = n_out = n_sbad = 0
        for r20, r24 in zip(recs[20], recs[24]):
            pos0, ec, seq = r20["pos0"], r20["ec"], r20["q"].shape[0]
            assert r24["pos0"] == pos0 and torch.equal(r24["cand_in"], r20["cand"]), \
                f"{tag}: layer 24 did not receive layer 20's candidates at {pos0}"
            vis = torch.clamp(torch.arange(pos0 + 1, pos0 + seq + 1), max = ec)
            def scores(r):
                s = (torch.einsum("shd,td->sht", r["q"], K[:ec]).relu_() * r["w"].unsqueeze(-1)).sum(1)
                return s.masked_fill(torch.arange(ec)[None] >= vis[:, None], -math.inf)
            s20, s24 = scores(r20), scores(r24)
            nb = -(-ec // block)
            c = r20["cand"].long()
            flags = torch.zeros((seq, nb + 1), dtype = torch.bool)
            flags.scatter_(1, torch.where(c >= 0, c, torch.full_like(c, nb)), True)
            have = flags[:, :nb].repeat_interleave(block, dim = 1)[:, :ec]
            if ds is not None:
                want = ds.select_candidate_blocks(s20, vis.unsqueeze(-1), n_blocks, block)
                bs = torch.nn.functional.pad(s20, (0, -ec % block), value = -math.inf).view(seq, -1, block).amax(-1)
                bs[torch.arange(seq), (vis - 1) // block] = math.inf
                for r in (want != have).any(1).nonzero().flatten().tolist():
                    thr = bs[r].topk(min(n_blocks, nb)).values[-1]
                    blocks = (want[r, ::block] != have[r, ::block]).nonzero().flatten()
                    n_cdiff += 1
                    n_cbad += int(((bs[r, blocks] - thr).abs().max() / s20[r][s20[r] > -math.inf].abs().mean()).item() > 2e-3)
            m24 = s24.masked_fill(~have, -math.inf)
            k = min(cfg.index_topk, ec)
            ii = r24["idx"].long()[:, :k]
            mine = torch.zeros((seq, ec + 1), dtype = torch.bool)
            mine.scatter_(1, torch.where(ii >= 0, ii, torch.full_like(ii, ec)), True)
            mine = mine[:, :ec]
            n_out += int((mine & ~have).any(1).sum())
            v, ti = m24.topk(k, dim = -1)
            top = torch.zeros((seq, ec + 1), dtype = torch.bool)
            top.scatter_(1, torch.where(v > -math.inf, ti, torch.full_like(ti, ec)), True)
            top = top[:, :ec]
            for r in (mine != top).any(1).nonzero().flatten().tolist():
                d = (mine[r] ^ top[r]).nonzero().flatten()
                n_sbad += int(((m24[r, d] - v[r, -1]).abs().max() / s24[r][s24[r] > -math.inf].abs().mean()).item() > 2e-3)
            n_rows += seq
        good = n_cbad == 0 and n_out == 0 and n_sbad == 0 and n_rows == T
        ok &= good
        vs_ds = (f"vs DeepSeek select_candidate_blocks on {n_rows} rows: {n_cdiff} candidate rows differ "
                 f"({n_cbad} beyond near-ties), " if ds is not None else
                 f"on {n_rows} rows (DeepSeek comparison skipped): ")
        print(f"  {'PASS' if good else 'FAIL'} [exact] candidate stage with effect ({tag}, {n_blocks} of up to "
              f"{-(-T // block)} blocks) {vs_ds}layer 24 outside its candidates on {n_out} rows, "
              f"not a masked top-k on {n_sbad}")
    e = {L: rel(torch.stack([got[L][i] for i in range(T)]), ref[L]) for L in LAYERS}
    good = all(v <= 1e-3 for v in e.values())
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} [exact] candidate stage with effect, cached vs stateless (gate 1e-3): " +
          " ".join(f"L{L} {v:.1e}" for L, v in e.items()))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs = "?", default = model_dir())
    ap.add_argument("--quick", action = "store_true", help = "T=700 only, home cuda:0 only")
    ap.add_argument("--numerics", default = "precise", help = "config.dsv41_numerics for the run")
    args = ap.parse_args()
    if not args.path:
        skip("V4.1 cached vs stateless on GPUs", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        return
    if torch.cuda.device_count() < 1:
        skip("V4.1 cached vs stateless on GPUs", "no CUDA device")
        return

    from exllamav3 import Config, Model
    from exllamav3.cache.dsv41 import DSV41State

    devs = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    print("  devices: " + " | ".join(f"{d} {torch.cuda.get_device_name(d)}" for d in devs))
    cfg = Config.from_directory(args.path)
    cfg.dsv41_numerics = args.numerics
    print(f"  numerics: {cfg.dsv41_numerics}")
    model = Model.from_config(cfg)
    assert model.recurrent_state_cls is DSV41State, model.recurrent_state_cls
    A = [b.attn for b in model.modules[model.first_block_idx: model.first_block_idx + cfg.num_hidden_layers]]

    Ts = (700,) if args.quick else (700, 3000)
    homes = (0,) if args.quick else tuple(range(len(devs)))
    runs = [(home, False) for home in homes]
    if len(devs) >= 2:
        runs += [(home, True) for home in ((0,) if args.quick else (0, 1))]
    ok = True
    t_start = time.time()
    for home, replica in runs:
        rig = Rig(cfg, model, A, devs, home, replica = replica)
        try:
            if replica:
                print(f"== home cuda:{home}, layer 12 + replica of 8 on cuda:{1 - home} (an explicit "
                      f"placement's split at 12)")
            else:
                print(f"== cuda:{home}: layers {', '.join(map(str, LAYERS))}")
            for T in Ts:
                xs = rig.inputs(T, 1000 + T)
                for mode in ("exact", "exl3"):
                    rig.set_mode(mode)
                    rec = {}
                    ref, sel_ref = rig.stateless(xs, rec)
                    ok &= check_selection(rig, rec, sel_ref, T, mode)
                    pnoise = None
                    for chunk in (2048, 1000, 7):
                        steps = schedule(T, chunk)
                        if mode == "exl3":
                            pnoise = projection_noise(rig, xs, steps, T)
                        t0 = time.time()
                        got, sel_got, crec = rig.cached(xs, steps)
                        tag = (f"[{mode}] T={T} chunk {chunk} ({len(steps)} steps, "
                               f"{sum(1 for n in steps if n == 1)} decode) {time.time() - t0:.1f}s")
                        if mode == "exact":
                            ok &= compare(rig, ref, sel_ref, got, sel_got, crec, steps, T, 1e-3, 1e-3, tag)
                        else:
                            ok &= compare(rig, ref, sel_ref, got, sel_got, crec, steps, T, 3e-3, 3e-3, tag,
                                          proj_noise = pnoise)
                            print(f"      exl3 projection difference, worst step size <= 144: " +
                                  " ".join(f"L{L} {pnoise[L]:.1e}" for L in LAYERS))

                    if mode == "exact" and home == 0 and not replica:
                        # negative controls: each must FAIL the exact gate
                        steps = schedule(T, 7)
                        def clear_carry(rig_, st, pos):
                            for L in (2, 8):
                                rsl = rig_.cache.get_recurrent_layer((L, 0))
                                if rsl.comp_carry is not None:
                                    rsl.comp_carry[st.slot].zero_()
                        controls = [("carry cleared before every step", steps, clear_carry, None, (2, 3, 8, 12))]
                        controls.append(("EXL3_DSV41_ABLATE=rope_consecutive", schedule(T, 1000), None,
                                         "rope_consecutive", (2, 3, 8, 12)))
                        if T == 3000:
                            controls.append(("EXL3_DSV41_ABLATE=dense_consumers", schedule(T, 1000), None,
                                             "dense_consumers", (3, 12)))
                        for name, st_, hook, abl, must in controls:
                            with ablate(abl or ""):
                                got, sel_got, _ = rig.cached(xs, st_, hook)
                            errs = {L: rel(got[L], ref[L]) for L in LAYERS}
                            sel_changed = {L: bool((selection_masks(sel_ref[L], T, cfg.compress_ratios[L]) !=
                                                    selection_masks(sel_got[L], T, cfg.compress_ratios[L])).any())
                                           for L in must}
                            caught = all(errs[L] > 1e-3 or sel_changed[L] for L in must)
                            ok &= caught
                            print(f"  {'PASS' if caught else 'FAIL'} [exact] T={T} negative control, {name}: " +
                                  " ".join(f"L{L} {errs[L]:.1e}{'*' if L in must else ''}" for L in LAYERS) +
                                  "  (* must exceed 1e-3 or change the selection)")

        finally:
            rig.close()

    # bsz 2: the per-row loop (own slot, own block-table row, own ring, registry keyed by row)
    # against each sequence's own stateless pass, exact projections
    T = 700
    rig = Rig(cfg, model, A, devs, 0, batch = 2)
    try:
        rig.set_mode("exact")
        xs0, xs1 = rig.inputs(T, 11), rig.inputs(T, 12)
        refs = [rig.stateless(xs0)[0], rig.stateless(xs1)[0]]
        xb = {d: torch.cat([xs0[d], xs1[d]], dim = 0) for d in xs0}
        sts = [rig.cache.get_new_state(), rig.cache.get_new_state()]
        outs = {L: [] for L in LAYERS}
        pos = 0
        with torch.inference_mode():
            for n in schedule(T, 1000):
                p = {"attn_mode": "flash_attn", "recurrent_states": sts}
                for L in LAYERS:
                    outs[L].append(rig.A[L].forward(xb[rig.dev[L]][:, pos:pos + n], p).float().cpu())
                for st in sts:
                    st.position += n
                    st.post_advance()
                pos += n
        for st in sts:
            st.free()
        worst = {L: max(rel(torch.cat(outs[L], 1)[b], refs[b][L]) for b in (0, 1)) for L in LAYERS}
        good = all(v <= 1e-3 for v in worst.values())
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} [exact] T={T} bsz 2 (slots {sts[0].slot}, {sts[1].slot}), "
              f"chunk 1000 + decode, each row vs its own stateless pass: " +
              " ".join(f"L{L} {v:.1e}" for L, v in worst.items()))
    finally:
        rig.close()

    # generation-state paths (rewinds, checkpoint restore, block tables, mixed positions, the
    # candidate stage with effect), exact projections, bitwise against fresh runs
    rig = Rig(cfg, model, A, devs, 0, batch = 2)
    try:
        rig.set_mode("exact")
        ok &= state_checks(rig, cfg)
    finally:
        rig.close()

    print(f"{'ALL PASS' if ok else 'FAILURES'}  ({time.time() - t_start:.0f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
