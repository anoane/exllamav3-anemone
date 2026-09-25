"""GPU checks for the V4.1 cross-device pool replica (exllamav3/cache/dsv41_replica.py), which the
split of an explicit placement inside a kv group needs (architecture/dsv41/placement.py).

One process at a time:

    PYTHONPATH=$PWD python3 tests/test_dsv41_replica_gpu_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

Needs two CUDA devices and well under 1 GiB on each, and the checkpoint's config (the layer
geometry is built from it). Every check runs in both directions, so the device order only
changes which card is printed first.

  1. Real geometry, bit-exact. Source 8 (with index K, as V4.1's kv sources own it) on
     one card, layer 12's replica on the other, both built from the config graph. Syncs
     of 1 KiB (one entry), 1 MiB (1,024 entries) and a page-straddling range: replica
     rows equal source rows bitwise, every other row unchanged. Both transfer paths:
     "staged" (pinned host bounce, what auto picks without peer access) and "direct".
     Then the page-straddling range again with a 64 KiB staging cap, so one call moves it
     in 80 pieces through the same staging buffer with no host synchronization between.
  2. Timing, per sync and per MiB, both paths, reported, not gated: it depends on the link
     between the cards. A bulk copy of the whole test pool calibrates the link.
  3. Stream ordering. 32 syncs of distinct ranges issued back to back without any host
     synchronization, the source rewritten on its own stream before each, and one side's
     stream held back by a sleep kernel: the replica's, so every H2D is still pending when
     the next D2H wants the staging buffer, and the source's, so every D2H is still
     pending when its H2D is issued. Must be bit-exact on both paths (direct relies on
     torch's own cross-device barrier). With the staging wait off under the replica stall,
     and with the ready wait off under the source stall, the run must FAIL, which shows
     each event is load-bearing and that this check can see it.
  4. copy_page on a replica page (whole, odd partial, one token), and through
     Cache.copy_page's loop with the source and the replica on different cards.
  5. The CPU suite's generator-shaped schedule (chunked prefill, decode, speculative
     rewinds, prefix sharing, partial-page reuse, defragmentation) with the source on one
     card and the replica on the other: after every step the replica equals a whole-pool
     copy of the source. The high-water ablation must fail here too.
  6. DeviceMemo moving a 2048 x 512 int32 top-k (4 MiB) across the cards: once,
     bit-exact, timed.
"""
import os, sys, time, types
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_H, os.path.join(_H, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dsv41_ref import model_dir, skip
if __name__ == "__main__" and torch.cuda.device_count() < 2:
    skip("V4.1 pool replica on GPUs", "needs two CUDA devices")
    sys.exit(0)

import test_dsv41_replica_ as T
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.cache import Cache
from exllamav3.cache.dsa import CacheLayer_dsa
from exllamav3.cache import dsv41_replica as R
from exllamav3.cache.dsv41 import DeviceMemo
from exllamav3.cache.dsv41_replica import CacheLayer_dsv41_replica, sync_pool_replica

NUM_PAGES = 256         # 65,536 tokens: 32,768 rate-2 entries, 40 MiB source + 32 MiB replica


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def real_layers(path):
    """Attention modules 8 and 12 from the real config graph (construction only)."""
    from exllamav3 import Config, Model
    config = Config.from_directory(path)
    model = Model.from_config(config)
    n = config.num_hidden_layers
    blocks = model.modules[model.first_block_idx : model.first_block_idx + n]
    return config, blocks[8].attn, blocks[12].attn


def build(config, a8, a12, sd, dd, g):
    tokens = NUM_PAGES * PAGE_SIZE
    src = CacheLayer_dsa(config, a8, 0, tokens)
    # a V4.1 kv source's pool (CacheLayer_dsv41) has index K, as source 8 owns it; the
    # replica for consumers 12/13 does not keep it
    src.D_i = a8.index_head_dim
    dst = CacheLayer_dsv41_replica(config, a12, 0, tokens)
    src.alloc(torch.device("cuda", sd))
    dst.alloc(torch.device("cuda", dd))
    T.fill_(src, g)
    T.fill_(dst, g)
    assert (src.epp, src.D_c, src.D_r, src.D_i) == (128, 448, 64, 128)
    assert (dst.epp, dst.D_c, dst.D_r, dst.D_i) == (128, 448, 64, 0)
    return src, dst


def bts(src, dst, pages):
    return (torch.tensor([pages], dtype = torch.int32, device = src.device),
            torch.tensor([pages], dtype = torch.int32, device = dst.device))


def check_exact(src, dst, pages, e0, e1, path):
    ss, sd = T.snapshot(src), T.snapshot(dst)
    bt_s, bt_d = bts(src, dst, pages)
    sync_pool_replica(src, dst, bt_s, bt_d, e0, e1, path = path)
    sync_all()
    T._check_sync(src, dst, pages, pages, e0, e1, ss, sd)


def time_sync(src, dst, pages, e0, n, path, reps = 21):
    """(host time of the call, call until both cards are idle), medians over reps: the link
    may be shared with other traffic, so means are noisy."""
    bt_s, bt_d = bts(src, dst, pages)
    for _ in range(3):
        sync_pool_replica(src, dst, bt_s, bt_d, e0, e0 + n, path = path)
    call, done = [], []
    for _ in range(reps):
        sync_all()
        t0 = time.perf_counter()
        sync_pool_replica(src, dst, bt_s, bt_d, e0, e0 + n, path = path)
        call.append(time.perf_counter() - t0)
        sync_all()
        done.append(time.perf_counter() - t0)
    med = lambda v: sorted(v)[len(v) // 2]
    return med(call), med(done)


def ordering_stress(src, dst, pages, stall = "dst", ablate = None, n = 32, width = 64,
                    sleep_cycles = 400_000_000, path = "staged"):
    """
    Returns the number of (range, pool) pairs whose replica rows are wrong.

    stall picks the stream a sleep kernel holds back. "dst": every H2D is still pending when
    the next D2H wants the staging buffer, which the staging-free wait covers. "src": every
    D2H is still pending when its H2D is issued, which the ready wait covers. ablate turns
    one of the two waits off ("staging" or "ready").
    """
    assert stall in ("dst", "src") and ablate in (None, "staging", "ready")
    R._staging_wait = ablate != "staging"
    R._ready_wait = ablate != "ready"
    try:
        epp = src.epp
        names = T.synced_names(src, dst)
        bt_s, bt_d = bts(src, dst, pages)
        g = torch.Generator(device = src.device)
        g.manual_seed(5 + (stall == "src") + 2 * (ablate is not None))
        # everything the loop needs is made up front: a host-to-device tensor build would
        # synchronize the source stream and close the race window this test exists to open
        plan = []
        for k in range(n):
            e0, e1 = k * width, (k + 1) * width
            rows = torch.tensor(T.rows_of(pages, e0, e1, epp), dtype = torch.long, device = src.device)
            news = [torch.randn((width, getattr(src, nm).shape[-1]), generator = g,
                                device = src.device).half() for nm in names]
            plan.append((e0, e1, rows, news))
        sync_all()
        with torch.cuda.device(dst.device if stall == "dst" else src.device):
            torch.cuda._sleep(sleep_cycles)      # hold one side's stream back
        expected = []
        for k, (e0, e1, rows, news) in enumerate(plan):
            for name, new in zip(names, news):
                t = getattr(src, name)
                t.view(-1, t.shape[-1]).index_copy_(0, rows, new)   # the source writes on its stream
                expected.append((k, name, e0, e1, new))
            sync_pool_replica(src, dst, bt_s, bt_d, e0, e1, path = path)
        sync_all()
        bad = 0
        for k, name, e0, e1, new in expected:
            t = getattr(dst, name)
            D = t.shape[-1]
            rows = torch.tensor(T.rows_of(pages, e0, e1, epp), dtype = torch.long, device = dst.device)
            if not torch.equal(T.as_bytes(t.view(-1, D)[rows]), T.as_bytes(new)):
                bad += 1
        return bad
    finally:
        R._staging_wait = True
        R._ready_wait = True


def copy_page_checks(src, dst):
    m, epp = dst.compress_rate, dst.epp
    for A, B, t in ((3, 7, 256), (5, 9, 77), (11, 13, 1)):
        snap = T.snapshot(dst)
        dst.copy_page(dst, A, B, t)
        sync_all()
        ne = min(t // m, epp)
        for name, s in snap.items():
            want = s.clone()
            want[B, :ne] = s[A, :ne]
            assert torch.equal(T.as_bytes(getattr(dst, name)), T.as_bytes(want)), (name, A, B, t)
    # Cache.copy_page walks every CacheLayer of the cache: the replica page moves with the source's
    fake = types.SimpleNamespace(layers = {(8, 0): src, (12, 0): dst}, num_layers = 2,
                                 model = types.SimpleNamespace(loaded_tp = False))
    A, B, t = 20, 30, 201
    ss, sd = T.snapshot(src), T.snapshot(dst)
    Cache.copy_page(fake, fake, A, B, t)
    sync_all()
    ne = t // m
    for layer, snap in ((src, ss), (dst, sd)):
        for name, s in snap.items():
            want = s.clone()
            want[B, :ne] = s[A, :ne]
            assert torch.equal(T.as_bytes(getattr(layer, name)), T.as_bytes(want)), (name, t)


def main(path):
    assert torch.cuda.device_count() >= 2, "needs two CUDA devices"
    for i in range(2):
        print(f"cuda:{i}  {torch.cuda.get_device_name(i)}  sm_{''.join(map(str, torch.cuda.get_device_capability(i)))}")
    print(f"peer access 0->1 {torch.cuda.can_device_access_peer(0, 1)}, "
          f"1->0 {torch.cuda.can_device_access_peer(1, 0)}; "
          f"auto path: {R._transfer_mode(torch.device('cuda', 0), torch.device('cuda', 1))}")
    config, a8, a12 = real_layers(path)
    g = torch.Generator().manual_seed(11)
    import random
    rng = random.Random(11)

    for sd, dd in ((0, 1), (1, 0)):
        print(f"\n== source 8 on cuda:{sd}, replica (layer 12) on cuda:{dd}")
        src, dst = build(config, a8, a12, sd, dd, g)
        epp = src.epp
        pages = rng.sample(range(NUM_PAGES), 64)
        entry_bytes = (dst.D_c + dst.D_r) * 2
        assert entry_bytes == 1024

        # 1. bit-exact
        for path in ("staged", "direct"):
            for e0, e1, what in ((5 * epp - 1, 5 * epp, "1 KiB"),
                                 (3 * epp - 17, 3 * epp - 17 + 1024, "1 MiB"),
                                 (epp - 3, 40 * epp + 9, "straddling")):
                check_exact(src, dst, pages, e0, e1, path)
                print(f"  exact {path:6s} {what:10s} [{e0}, {e1}) = {(e1 - e0) * entry_bytes:,} B: ok")

        # 2. timing
        tb = src.pool_c.numel() * 2 + src.pool_r.numel() * 2
        sync_all()
        t0 = time.perf_counter()
        _bulk = [src.pool_c.to(dst.device), src.pool_r.to(dst.device)]
        sync_all()
        bulk = time.perf_counter() - t0
        del _bulk
        print(f"  bulk copy of the test pool ({tb / 2**20:.0f} MiB): {bulk * 1e3:.1f} ms = "
              f"{bulk * 1e3 / (tb / 2**20):.2f} ms/MiB")
        for path in ("staged", "direct"):
            for n in (1, 1024, 2048):
                call, done = time_sync(src, dst, pages, 3 * epp - 17, n, path)
                mib = n * entry_bytes / 2**20
                print(f"  time {path:6s} {n:5d} entries ({n * entry_bytes:>9,} B): host call {call * 1e3:6.3f} ms, "
                      f"until both idle {done * 1e3:6.3f} ms ({done * 1e3 / mib:7.2f} ms/MiB)")

        # 1b. one call moved in many pieces through the same staging buffer, no host sync between
        cap = R.STAGE_MAX_BYTES
        R.STAGE_MAX_BYTES = 64 << 10
        try:
            for path in ("staged", "direct"):
                check_exact(src, dst, pages, epp - 3, 40 * epp + 9, path)
        finally:
            R.STAGE_MAX_BYTES = cap
        step = ((64 << 10) - R._SEG_ALIGN * 2) // entry_bytes      # two pools: pool_c, pool_r
        pieces = -(-(39 * epp + 12) // step)
        print(f"  exact staged/direct [{epp - 3}, {40 * epp + 9}) in {pieces} pieces of <= {step} "
              f"entries per call: ok")

        # 3. ordering, with each side's stream stalled in turn
        for stall in ("dst", "src"):
            for path in ("staged", "direct"):
                bad = ordering_stress(src, dst, pages, stall = stall, path = path)
                assert bad == 0, f"ordering ({path}, {stall} stalled): {bad} (range, pool) pairs wrong"
        bad_staging = ordering_stress(src, dst, pages, stall = "dst", ablate = "staging")
        assert bad_staging > 0, "staging-wait ablation did not fail: the stress test is blind to that race"
        bad_ready = ordering_stress(src, dst, pages, stall = "src", ablate = "ready")
        assert bad_ready > 0, "ready-wait ablation did not fail: the stress test is blind to that race"
        print(f"  ordering: 32 back-to-back syncs exact (staged, direct) with the replica's stream "
              f"stalled and with the source's stalled; staging wait off {bad_staging}/{32 * 2}, "
              f"ready wait off {bad_ready}/{32 * 2} (range, pool) pairs wrong, as they must be")

        # 4. copy_page
        copy_page_checks(src, dst)
        print("  copy_page on replica pages, and Cache.copy_page across both cards: ok")

        # 6. device memo
        topk = torch.randint(0, 1 << 20, (2048, 512), dtype = torch.int32, device = src.device)
        params = {}
        sync_all()
        t0 = time.perf_counter()
        a = DeviceMemo.get(params, ("topk", 8), topk, dst.device)
        sync_all()
        dt = time.perf_counter() - t0
        t1 = time.perf_counter()
        b = DeviceMemo.get(params, ("topk", 8), topk, dst.device)
        dt2 = time.perf_counter() - t1
        assert a is b and a.device == dst.device and torch.equal(a.cpu(), topk.cpu())
        print(f"  DeviceMemo 2048x512 int32 (4 MiB): first get {dt * 1e3:.1f} ms, repeat {dt2 * 1e6:.0f} us, exact")

        del src, dst, a, b, topk
        R.release_staging()
        torch.cuda.empty_cache()

    # 5. schedules across the cards
    for sd, dd in ((0, 1), (1, 0)):
        t0 = time.perf_counter()
        res = T.run_schedules(src_dev = f"cuda:{sd}", dst_dev = f"cuda:{dd}", seeds = (0,), steps = 80,
                              configs = [(2, 0, True, False), (1, 0, True, True), (2, 8, True, False)])
        tot = {k: sum(st[k] for _, st in res) for k in res[0][1]}
        print(f"\n== schedules cuda:{sd} -> cuda:{dd}: {len(res)} ok in {time.perf_counter() - t0:.1f} s  {tot}")
        failed = T.run_high_water_ablation(src_dev = f"cuda:{sd}", dst_dev = f"cuda:{dd}", seeds = (0,), steps = 80)
        print(f"  high-water ablation fails as it must on {len(failed)}/{len(failed)} runs: {failed[0][2]}")

    peak = [torch.cuda.max_memory_allocated(i) / 2**20 for i in range(2)]
    print(f"\npeak allocated: cuda:0 {peak[0]:.0f} MiB, cuda:1 {peak[1]:.0f} MiB")
    print("ALL OK")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 pool replica on GPUs", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
    else:
        main(d)
