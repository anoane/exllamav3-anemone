"""select_topk on GPU: the ext backend (Triton scorer + CUDA top-k kernels) against the torch
backend, vLLM's own candidate kernels, and its memory bound. Needs one GPU with 4 GiB free.

1. Integer-valued q/k/w: every product and partial sum is exact in fp32, so the Triton
   scorer and torch produce the same fp16 scores bit for bit and ties are everywhere.
   At T = 65,536 and 262,144 (rate 1, rows far past the 16,384-entry point where the
   candidate stage bites, plus a chunk crossing it): source indices + candidates and
   masked user indices must be IDENTICAL on every row, ties included (both backends
   break ties by ascending index). The paged pool through a random block table must
   give the same result as the dense one. The mask must actually move the user's
   selection, keep it inside the candidate blocks, and every row's newest block must
   be a candidate.
   Decode shapes (1-4 rows: the scorer's few-query kernel, the split top-k) at
   T = 262,144, tiled and single-pass, and a prefill from position 0 whose first
   512 rows see fewer than 512 entries, likewise identical.
2. Rate 2, paged, 128 entries per page, T = 262,144: the rate-2 pool shape, at the
   end of the pool and from position 0 (row 0 sees nothing).
3. Continuous inputs: the two backends accumulate in different orders, so fp16
   rounding can differ in the last bit. Every mismatch must sit at the top-k cut:
   the swapped entries' float64 scores within 2 fp16 ulps of the row's k-th score.
4. Rows whose weights hold a NaN (every visible score NaN): identical in both backends;
   with > 2048 NaN blocks the pinned block is outranked and no candidate survives,
   as in DeepSeek's and vLLM's torch.topk. NaN enters only through the weights: a NaN
   q . k would score 0 in the Triton scorer (tl.maximum) and NaN in the torch backend.
5. vLLM's Triton select_candidate_blocks / apply_candidate_mask, run on the same fp32
   logits: same block-score multiset per row, and the user top-512 over vLLM-masked
   logits == ours. The file comes from DSV41_VLLM_CANDIDATE_BLOCKS (a path to vLLM's
   model_executor/kernels/attention/dsa/candidate_blocks.py) or, failing that, from the
   docker image DSV41_VLLM_IMAGE; with neither this section is skipped, and a variable
   that is set but does not give the file raises.
6. Peak memory for seq 4096 at T = 262,144 and 1,048,576, and for seq 16 and 1 at
   1,048,576 (widened tiles), source and user:
   max_memory_allocated - base <= transient_bytes() + 64 MiB.
7. No host synchronization: source, user and plain calls at decode (1 row) and prefill
   (512 rows) shapes run under torch.cuda.set_sync_debug_mode("error"), and with ~50-100 ms
   of GPU work queued ahead a source call returns long before that work finishes. The
   newest-block pin must not use nonzero(), which blocks the host on every
   candidate-source slab, i.e. at layer 20 of every decode step. A page table shorter
   than ec is refused (the scorer would read the page ids stored after it).

    python tests/test_dsv41_select_gpu_.py      (DSV41_SELECT_DEVICE=cuda:N, default cuda:0)
    or python -m pytest tests/test_dsv41_select_gpu_.py"""
import atexit, importlib.util, os, subprocess, sys, tempfile, time
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
from exllamav3.modules import dsv41_select as S
from exllamav3.modules.dsv41_select import select_topk, transient_bytes
from exllamav3.architecture.dsv41 import candidates as C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import ablate, skip

MiB = 1 << 20
NEG_INF = float("-inf")
H, D = 32, 128
VLLM_IMAGE = os.environ.get("DSV41_VLLM_IMAGE")
VLLM_FILE = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/attention/dsa/candidate_blocks.py"


def check(name, ok, detail = ""):
    assert ok, f"FAIL {name} {detail}"
    print(f"  OK  {name}" + (f"  ({detail})" if detail else ""))


def ints(seq, T, dev, gen):
    q = torch.randint(-3, 4, (seq, H, D), generator = gen, device = dev).half()
    k = torch.randint(-3, 4, (T, D), generator = gen, device = dev).half()
    w = torch.randint(-4, 5, (seq, H), generator = gen, device = dev).float() / 64
    return q, w, k


def conts(seq, T, dev, gen):
    q = torch.randn((seq, H, D), generator = gen, device = dev).half()
    k = torch.randn((T, D), generator = gen, device = dev).half()
    w = torch.randn((seq, H), generator = gen, device = dev) * (D ** -0.5 * H ** -0.5)
    return q, w, k


def paged(k, epp, gen):
    T = k.shape[0]
    pages = -(-T // epp)
    phys = torch.randperm(pages + 5, generator = gen, device = k.device)[:pages].int()
    pool = torch.zeros((pages + 5, epp, k.shape[1]), dtype = k.dtype, device = k.device)
    e = torch.arange(T, device = k.device)
    pool.view(-1, k.shape[1])[phys[e // epp].long() * epp + e % epp] = k
    return pool, phys[None]


def timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t


def load_vllm():
    """
    vLLM's candidate_blocks.py as a module, or None when neither DSV41_VLLM_CANDIDATE_BLOCKS nor
    DSV41_VLLM_IMAGE is set. A variable that is set but does not give the file raises, rather
    than turning the section into a skip.
    """
    path = os.environ.get("DSV41_VLLM_CANDIDATE_BLOCKS")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DSV41_VLLM_CANDIDATE_BLOCKS={path!r} is not a file")
        src, origin = open(path).read(), path
    elif VLLM_IMAGE:
        r = subprocess.run(["docker", "run", "--rm", "--entrypoint", "cat", VLLM_IMAGE, VLLM_FILE],
                           capture_output = True, text = True, timeout = 180)
        if r.returncode != 0:
            raise RuntimeError(f"DSV41_VLLM_IMAGE={VLLM_IMAGE!r}: reading {VLLM_FILE} exited with "
                               f"{r.returncode}: {r.stderr.strip()[-500:]}")
        src, origin = r.stdout, f"{VLLM_IMAGE}:{VLLM_FILE}"
    else:
        return None
    if "def select_candidate_blocks" not in src:
        raise RuntimeError(f"{origin} has no select_candidate_blocks")
    src = src.replace("from vllm.triton_utils import tl, triton", "import triton\nimport triton.language as tl")
    fd, fn = tempfile.mkstemp(suffix = "_vllm_candidate_blocks.py")
    # triton.jit reads the source back from the file while it compiles, so it stays until exit
    atexit.register(lambda: os.path.exists(fn) and os.remove(fn))
    with os.fdopen(fd, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("vllm_candidate_blocks", fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # the checks compare the real function: clear an ablation the runner set, for this run only
    with ablate(""):
        run()


def test_select_gpu():
    """pytest entry point: the same checks, skipped without a GPU."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("select GPU: no CUDA device")
    main()


def run():
    dev = torch.device(os.environ.get("DSV41_SELECT_DEVICE", "cuda:0"))
    torch.cuda.set_device(dev)
    free, total = torch.cuda.mem_get_info(dev)
    assert free > 4 << 30, f"{dev}: only {free / 2**30:.1f} GiB free; a model may be loaded"
    gen = torch.Generator(device = dev).manual_seed(0)
    print(f"  device {dev} ({torch.cuda.get_device_name(dev)}), {free / 2**30:.1f} GiB free")

    # ------------------------------------------------------------------ 1. integer inputs, rate 1
    for T, seq, pos0 in [(65536, 512, 65536 - 512), (65536, 512, 16384 - 256), (262144, 512, 262144 - 512)]:
        ec = pos0 + seq
        q, w, k = ints(seq, T, dev, gen)
        qu, wu, _ = ints(seq, T, dev, gen)
        kw = dict(pos0 = pos0, m = 1, ec = ec)
        se, te = timed(lambda: select_topk(q, w, k, want_cand = True, backend = "ext", **kw))
        st, tt = timed(lambda: select_topk(q, w, k, want_cand = True, backend = "torch", **kw))
        ue = select_topk(qu, wu, k, cand_in = se.cand, backend = "ext", **kw)
        ut = select_topk(qu, wu, k, cand_in = st.cand, backend = "torch", **kw)
        plain = select_topk(qu, wu, k, backend = "ext", **kw)
        sc = S._torch_scores(q[:64], w[:64], k[:ec], pos0, 1, ec, 1.0, torch.float16)
        v = sc.sort(dim = 1, descending = True).values
        ties = int((v[:, 511] == v[:, 512]).sum())
        check(f"T {T} rows {pos0}..{ec - 1}: ext == torch, source indices + candidates, user indices",
              torch.equal(se.indices, st.indices) and torch.equal(se.cand, st.cand)
              and torch.equal(ue.indices, ut.indices),
              f"all {seq} rows bitwise; {ties}/64 sampled rows tie across the top-512 cut; "
              f"ext {te * 1e3:.0f} ms, torch {tt * 1e3:.0f} ms")
        pool, bt = paged(k, 256, gen)
        sp = select_topk(q, w, pool, bt_row = bt, epp = 256, want_cand = True, backend = "ext", **kw)
        up = select_topk(qu, wu, pool, bt_row = bt, epp = 256, cand_in = sp.cand, backend = "ext", **kw)
        check(f"T {T}: paged (random block table, epp 256) == dense, ext",
              torch.equal(sp.indices, se.indices) and torch.equal(sp.cand, se.cand)
              and torch.equal(up.indices, ue.indices))
        vis = C.visible_counts(seq, pos0, 1, ec, dev)
        keep = C.candidate_keep(se.cand, 8, ec)
        ids = ue.indices.long()
        inside = torch.where(ids >= 0, keep.gather(1, ids.clamp_min(0)), torch.ones_like(ids, dtype = torch.bool))
        newest = ((vis - 1) // 8)[:, None]
        pinned = (se.cand.long() == newest).any(1)
        nblk = (se.cand >= 0).sum(1)
        bites = vis > 16384
        moved = (ue.indices != plain.indices).any(1)
        check(f"T {T}: user picks inside candidate blocks, newest block always a candidate",
              inside.all().item() and pinned.all().item()
              and (nblk == torch.clamp((vis + 7) // 8, max = 2048)).all().item(),
              f"{int(bites.sum())} rows past 16384 entries: mask moved {moved[bites].float().mean():.0%} of them; "
              f"{int((~bites).sum())} rows at or below: moved {int(moved[~bites].sum())}")
        assert not moved[~bites].any() and moved[bites].float().mean() > 0.5
        del q, w, k, qu, wu, pool

    # decode shapes and a prefill from position 0
    T = 262144
    q, w, k = ints(8, T, dev, gen)
    qu, wu, _ = ints(8, T, dev, gen)
    n_ok = 0
    # slab_rows = seq keeps the given tile; the default slab widens it to one pass
    for seq, pos0, tile, slab in [(1, T - 1, 65536, 1), (1, T - 1, 65536, 256), (3, T - 3, 65536, 3),
                                  (4, 20000, 65536, 256), (2, 16383, 8192, 2)]:
        kw = dict(pos0 = pos0, m = 1, ec = pos0 + seq, tile_entries = tile, slab_rows = slab)
        se = select_topk(q[:seq], w[:seq], k, want_cand = True, backend = "ext", **kw)
        st = select_topk(q[:seq], w[:seq], k, want_cand = True, backend = "torch", **kw)
        ue = select_topk(qu[:seq], wu[:seq], k, cand_in = se.cand, backend = "ext", **kw)
        ut = select_topk(qu[:seq], wu[:seq], k, cand_in = se.cand, backend = "torch", **kw)
        assert torch.equal(se.indices, st.indices) and torch.equal(se.cand, st.cand) \
            and torch.equal(ue.indices, ut.indices), (seq, pos0, tile)
        n_ok += 1
    check("decode shapes (1-4 rows, tiled and single-pass, up to T 262144): ext == torch", n_ok == 5)
    q, w, k = ints(1100, 4096, dev, gen)
    kw = dict(pos0 = 0, m = 1, ec = 1100)
    a = select_topk(q, w, k, want_cand = True, backend = "ext", **kw)
    b = select_topk(q, w, k, want_cand = True, backend = "torch", **kw)
    ok = torch.equal(a.indices, b.indices) and torch.equal(a.cand, b.cand)
    ok = ok and torch.equal(a.indices[300, :301], torch.arange(301, device = dev, dtype = torch.int32)) \
        and (a.indices[300, 301:] == -1).all().item() and (a.indices[1099] >= 0).all().item()
    check("prefill from position 0 (1100 rows, first 512 see < 512 entries): ext == torch", ok,
          "row 300 = entries 0..300 then -1")
    del q, w, k, qu, wu

    # ------------------------------------------------------------------ 2. rate 2, paged epp 128
    T = 262144
    q, w, k = ints(1100, T, dev, gen)
    pool, bt = paged(k, 128, gen)
    for seq, pos0 in [(384, 2 * T - 384), (1100, 0)]:
        kw = dict(bt_row = bt, epp = 128, pos0 = pos0, m = 2, ec = min(T, (pos0 + seq) // 2))
        a = select_topk(q[:seq], w[:seq], pool, backend = "ext", **kw)
        b = select_topk(q[:seq], w[:seq], pool, backend = "torch", **kw)
        assert torch.equal(a.indices, b.indices), (seq, pos0)
        if pos0 == 0:
            assert (a.indices[0] == -1).all().item() and a.indices[1, 0].item() == 0
    check("rate 2, paged epp 128, T 262144: ext == torch (pool end, and from position 0)", True)
    del q, w, k, pool

    # ------------------------------------------------------------------ 3. continuous inputs
    T, seq = 65536, 512
    pos0 = T - seq
    q, w, k = conts(seq, T, dev, gen)
    kw = dict(pos0 = pos0, m = 1, ec = T)
    se = select_topk(q, w, k, want_cand = True, backend = "ext", **kw)
    st = select_topk(q, w, k, want_cand = True, backend = "torch", **kw)
    ue = select_topk(q.flip(0), w.flip(0), k, cand_in = se.cand, backend = "ext", **kw)
    ut = select_topk(q.flip(0), w.flip(0), k, cand_in = se.cand, backend = "torch", **kw)
    worst = 0.0
    n_rows = 0
    for (x, y, qq, ww, cand) in [(se.indices, st.indices, q, w, None), (ue.indices, ut.indices, q.flip(0), w.flip(0), se.cand)]:
        rows = (x != y).any(1).nonzero().view(-1).tolist()
        n_rows += len(rows)
        for r in rows:
            s64 = S._torch_scores(qq[r:r + 1].double(), ww[r:r + 1].double(), k.double(), pos0 + r, 1, T,
                                  1.0, torch.float64)[0]
            if cand is not None:
                s64 = C.mask_scores_to_candidates(s64[None], cand[r:r + 1], 8)[0]
            kth = s64.sort(descending = True).values[511]
            diff = set(x[r].tolist()) ^ set(y[r].tolist())
            rel = max(abs(s64[j].item() - kth.item()) / abs(kth.item()) for j in diff)
            worst = max(worst, rel)
    check("continuous inputs: every ext/torch index mismatch sits at the top-512 cut", worst <= 2 * 2 ** -10,
          f"{n_rows} of {2 * seq} rows differ, worst swapped entry {worst:.1e} (rel) from the k-th score")
    # candidate blocks: the 2048th block score sits far down the row, where one-ulp differences
    # in single entries can reorder blocks; every swapped block must sit at the 2048-block cut
    c_rows = (se.cand != st.cand).any(1).nonzero().view(-1).tolist()
    worst_c = 0.0
    vis = C.visible_counts(seq, pos0, 1, T, dev)
    for r in c_rows:
        s64 = S._torch_scores(q[r:r + 1].double(), w[r:r + 1].double(), k.double(), pos0 + r, 1, T,
                              1.0, torch.float64)
        b64 = C.block_scores(s64, vis[r:r + 1], 8)[0]
        kept = b64.gather(0, se.cand[r].long().clamp_min(0))
        cut = kept[torch.isfinite(kept)].min()
        diff = set(se.cand[r].tolist()) ^ set(st.cand[r].tolist())
        worst_c = max(worst_c, max(abs(b64[j].item() - cut.item()) / abs(cut.item()) for j in diff))
    check("continuous inputs: every ext/torch candidate mismatch sits at the 2048-block cut",
          worst_c <= 2 * 2 ** -10, f"{len(c_rows)} of {seq} rows differ, worst swapped block {worst_c:.1e} (rel) "
                                   f"from the cut")
    del q, w, k

    # ------------------------------------------------------------------ 4. NaN rows
    T, seq = 65536, 256
    q, w, k = ints(seq, T, dev, gen)
    w[5] = float("nan")
    w[77, 3] = float("nan")
    kw = dict(pos0 = T - seq, m = 1, ec = T)
    a = select_topk(q, w, k, want_cand = True, backend = "ext", **kw)
    b = select_topk(q, w, k, want_cand = True, backend = "torch", **kw)
    check("NaN weight rows: ext == torch; no candidate survives 8000+ NaN blocks",
          torch.equal(a.indices, b.indices) and torch.equal(a.cand, b.cand)
          and (a.cand[[5, 77]] == -1).all().item() and (a.cand[6] >= 0).sum().item() == 2048,
          f"NaN rows select entries {a.indices[5, :3].tolist()}.. (NaN ranks first, index-ascending)")
    del q, w, k

    # ------------------------------------------------------------------ 5. vLLM's kernels
    V = load_vllm()
    if V is None:
        skip("vLLM candidate kernels", "neither DSV41_VLLM_CANDIDATE_BLOCKS nor DSV41_VLLM_IMAGE gives candidate_blocks.py")
    else:
        T, seq = 65536, 256
        pos0 = T - seq
        q, w, k = ints(seq, T, dev, gen)
        qu, wu, _ = ints(seq, T, dev, gen)
        kw = dict(pos0 = pos0, m = 1, ec = T)
        se = select_topk(q, w, k, want_cand = True, backend = "ext", **kw)
        ue = select_topk(qu, wu, k, cand_in = se.cand, backend = "ext", **kw)
        vis = C.visible_counts(seq, pos0, 1, T, dev).int()
        logits = S._torch_scores(q, w, k, pos0, 1, T, 1.0, torch.float16).float()
        out = torch.empty((seq, 2048), dtype = torch.int32, device = dev)
        V.select_candidate_blocks(logits, None, vis, 2048, 8, out)
        bs = C.block_scores(logits, vis, 8)
        a = torch.sort(bs.gather(1, se.cand.long().clamp_min(0)).masked_fill(se.cand < 0, NEG_INF), dim = 1).values
        b = torch.sort(bs.gather(1, out.long().clamp_min(0)).masked_fill(out < 0, NEG_INF), dim = 1).values
        same_set = sum(set(se.cand[r].tolist()) == set(out[r].tolist()) for r in range(seq))
        check("candidate blocks vs vLLM select_candidate_blocks (Triton): same block-score multiset per row",
              torch.equal(a, b), f"identical id sets on {same_set}/{seq} rows" +
              ("" if same_set == seq else "; the rest differ only among tied block scores "
                                          "(vLLM keeps torch.topk's order, ours is index-ascending)"))
        lu = S._torch_scores(qu, wu, k, pos0, 1, T, 1.0, torch.float16).float()
        V.apply_candidate_mask(lu, None, vis, se.cand, 8)
        i, v = C.topk_total_order(lu, 512)
        ids = C.sort_ids_ascending(torch.where(v > NEG_INF, i, torch.full_like(i, -1))).int()
        check("user top-512 over vLLM apply_candidate_mask logits == ext user indices", torch.equal(ids, ue.indices))
        del q, w, k, qu, wu, logits, lu

    # ------------------------------------------------------------------ 7. no host sync
    T = 262144
    k = torch.randn((T, D), generator = gen, device = dev).half()
    spin = int(1e8)                              # torch.cuda._sleep cycles: ~50-100 ms of queued GPU work
    for seq in (1, 512):
        q = torch.randn((seq, H, D), generator = gen, device = dev).half()
        w = torch.randn((seq, H), generator = gen, device = dev) * (D ** -0.5 * H ** -0.5)
        kw = dict(pos0 = T - seq, m = 1, ec = T)
        src = select_topk(q, w, k, want_cand = True, **kw)                 # compile outside the checks
        calls = {"source": dict(want_cand = True), "user": dict(cand_in = src.cand), "plain": {}}
        for extra in calls.values():
            select_topk(q, w, k, **extra, **kw)
        synced = []
        for name, extra in calls.items():
            torch.cuda.synchronize(dev)
            torch.cuda.set_sync_debug_mode("error")
            try:
                r = select_topk(q, w, k, **extra, **kw)
            except RuntimeError:
                synced.append(name)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        host_ms = []
        for _ in range(3):
            torch.cuda.synchronize(dev)
            torch.cuda._sleep(spin)
            t0 = time.perf_counter()
            select_topk(q, w, k, want_cand = True, **kw)
            host_ms.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.synchronize(dev)
        _, gpu_s = timed(lambda: torch.cuda._sleep(spin))
        check(f"no host sync, seq {seq}, T {T}: source / user / plain under sync debug mode",
              not synced and max(host_ms) < 0.5 * gpu_s * 1e3,
              f"synchronizing: {synced or 'none'}; source call returns in {max(host_ms):.2f} ms with "
              f"{gpu_s * 1e3:.0f} ms of GPU work queued ahead")
        del q, w, src
    pages = -(-T // 256)
    pool = k.view(pages, 256, D)
    bt = torch.arange(pages, dtype = torch.int32, device = dev)[None]
    try:
        select_topk(k[:1, None].expand(1, H, D).contiguous(), torch.ones((1, H), device = dev), pool,
                    bt_row = bt[:, :-1], epp = 256, pos0 = T - 1, m = 1, ec = T)
        refused = False
    except ValueError:
        refused = True
    check("page table one page short of ec is refused (ext)", refused)
    del k, pool

    # ------------------------------------------------------------------ 6. peak memory
    for seq, T in ((4096, 262144), (4096, 1 << 20), (16, 1 << 20), (1, 1 << 20)):
        q, w, k = conts(seq, T, dev, gen)
        cand = None
        for mode in ("source", "user"):
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            (r, dt) = timed(lambda: select_topk(q, w, k, pos0 = T - seq, m = 1, ec = T, backend = "ext",
                                                want_cand = mode == "source", cand_in = cand))
            peak = torch.cuda.max_memory_allocated(dev) - base
            bound = transient_bytes(seq, T, want_cand = mode == "source", cand_in = mode == "user")
            untiled = seq * (1 << (T - 1).bit_length()) * 2
            check(f"peak memory, seq {seq}, T {T}, {mode}: <= transient_bytes + 64 MiB",
                  peak <= bound + 64 * MiB,
                  f"peak {peak / MiB:.1f} MiB, bound {bound / MiB:.1f} MiB, untiled score matrix "
                  f"{untiled / MiB:.0f} MiB; {dt:.2f} s")
            if mode == "source":
                cand = r.cand
            del r
        del q, w, k, cand
    print("  all GPU select tests passed")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        skip("select GPU", "no CUDA device")
    else:
        main()
