"""CPU checks for the V4.1 cross-device pool replica (exllamav3/cache/dsv41_replica.py), which the
split of an explicit placement inside a kv group needs (architecture/dsv41/placement.py).

Run from the repo root, or with PYTHONPATH set to it:

    PYTHONPATH=$PWD python3 tests/test_dsv41_replica_.py

Check 1 builds the graph from DSV41_MODEL_DIR and is skipped when it is unset.

No GPU. The "two devices" are two CacheLayers on CPU tensors: the local transfer path
still gathers into, and scatters from, the same packed segment layout the CUDA paths move.

  1. Geometry. The replica built for layer 12 from the real config graph matches source
     8's CacheLayer (epp 128, D 448/64/0) and holds 536,870,912 B at 1,048,576 tokens; the
     int32 limit: a replica whose entries times its widest row reach 2^31 is refused at
     construction, the largest addressable max_num_tokens builds.
  2. The same limit for a quantized (-cq) replica, at its packed widths: at k_bits 8 a rate-2
     replica without index K builds at its computed limit and refuses one page more. Needs no
     model.
  3. Row copies. Bitwise equal, untouched rows unchanged, e0 == e1 a no-op, over random
     block tables and page-straddling ranges; m in {1, 2}, with and without index K,
     fp16 and quantized (pool_q / pool_s); whole and in pieces.
  4. Incremental == full copy. A generator-shaped schedule -- chunked prefill, decode,
     speculative steps with rewinds up to guaranteed_rollback, prefix sharing,
     partial-page reuse through Cache.copy_page, defragmentation over
     Cache.get_all_tensors, freed and reused pages -- where the source writes only the
     groups that close (the reference's rule, token by token, independent of
     emission_range) and the replica is synced with emission_range. After every step
     the replica must equal a whole-pool copy of the source. The naive scheme that
     syncs only entries past the furthest one ever synced must FAIL the same schedule,
     or the schedule would be blind to rewinds.
  5. The transfer policy: reported peer access never overrides the engine's bounce verdict.

The range the cached forward syncs (emission_range, against DeepSeek's Compressor and the
chunked compressor), the source pools' int32 limits and the per-forward device memo are checked
in test_dsv41_cached_.py.
"""
import os, random, sys, types
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _H not in sys.path:
    sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip

from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.cache import Cache
from exllamav3.cache.dsa import CacheLayer_dsa, DSV4State
from exllamav3.cache import dsv41_replica as R
from exllamav3.cache.dsv41_replica import CacheLayer_dsv41_replica, sync_pool_replica
from exllamav3.modules.dsv41_cached import emission_range

POOLS = ("pool_c", "pool_q", "pool_s", "pool_r", "pool_idx")


# ---------------------------------------------------------------------------------------------
# helpers, shared with tests/test_dsv41_replica_gpu_.py

def attn_ns(m, layer_idx, kv_source_layer, *, csa = False, head_dim = 512, rope_head_dim = 64,
            index_head_dim = 128):
    """The attention attributes a CacheLayer_dsa reads, with V4.1-Flash's dimensions."""
    return types.SimpleNamespace(
        key = f"model.layers.{layer_idx}.attn", layer_idx = layer_idx,
        kv_source_layer = kv_source_layer, compress_rate = m, head_dim = head_dim,
        rope_head_dim = rope_head_dim, index_head_dim = index_head_dim,
        layer_type = "csa" if csa else "hca",
    )


def make_pair(m, k_bits, src_idx, dst_idx, num_pages, src_dev = "cpu", dst_dev = "cpu"):
    """Source 8 (with index K when src_idx, as V4.1's kv sources own it) and the
    replica owned by consumer 12, allocated on their devices."""
    tokens = num_pages * PAGE_SIZE
    src = CacheLayer_dsa(None, attn_ns(m, 8, 8, csa = src_idx), 0, tokens, k_bits = k_bits)
    dst = CacheLayer_dsv41_replica(None, attn_ns(m, 12, 8), 0, tokens, k_bits = k_bits,
                                   with_index_k = dst_idx)
    src.alloc(torch.device(src_dev))
    dst.alloc(torch.device(dst_dev))
    return src, dst


def rand_like(t, g):
    """Random content for a pool tensor, generated on CPU."""
    if t.dtype == torch.half:
        return torch.randn(tuple(t.shape), generator = g).half()
    return torch.randint(-2**31, 2**31 - 1, tuple(t.shape), generator = g,
                         dtype = torch.int64).to(t.dtype)


def fill_(layer, g):
    for n in POOLS:
        t = getattr(layer, n)
        if t is not None:
            t.copy_(rand_like(t, g))


def synced_names(src, dst):
    """Pools the replica mirrors, written out independently of the module's own list."""
    names = ["pool_q", "pool_s"] if src.k_bits else ["pool_c"]
    names.append("pool_r")
    if dst.D_i:
        names.append("pool_idx")
    return names


def as_bytes(t):
    return t.detach().cpu().contiguous().view(-1).view(torch.uint8)


def snapshot(layer):
    return {n: getattr(layer, n).detach().cpu().clone() for n in POOLS if getattr(layer, n) is not None}


def rows_of(pages, e0, e1, epp):
    return [pages[e // epp] * epp + e % epp for e in range(e0, e1)]


# ---------------------------------------------------------------------------------------------
# 1. geometry from the real config graph

def test_real_geometry():
    from exllamav3 import Config, Model
    path = model_dir()
    if path is None:
        skip("test_real_geometry", "DSV41_MODEL_DIR not set")
        return
    config = Config.from_directory(path)
    model = Model.from_config(config)
    n = config.num_hidden_layers
    blocks = model.modules[model.first_block_idx : model.first_block_idx + n]
    a8, a12, a20, a28 = (blocks[i].attn for i in (8, 12, 20, 28))
    assert a8.is_kv_source and a8.compress_ratio == 2
    assert not a12.is_kv_source and a12.compress_ratio == 2 and a12.kv_source_layer == 8
    assert not a28.is_kv_source and a28.compress_ratio == 1 and a28.kv_source_layer == 20

    N = 1 << 20
    src = CacheLayer_dsa(config, a8, 0, N)
    rep = CacheLayer_dsv41_replica(config, a12, 0, N)
    geo = lambda l: (l.compress_rate, l.epp, l.D_c, l.D_r, l.num_pages, l.capacity)
    assert geo(rep) == geo(src) == (2, 128, 448, 64, 4096, 524288), (geo(rep), geo(src))
    assert rep.D_i == 0 and rep.replica_of == 8
    assert rep.storage_size() == 536_870_912, rep.storage_size()
    print(f"  layer 12 <- 8: m {rep.compress_rate}, epp {rep.epp}, D {rep.D_c}/{rep.D_r}/{rep.D_i}, "
          f"{rep.storage_size():,} B at {N:,} tokens")

    # the owner's replica_of attribute (its pool route), when set, is what the replica records
    a12.replica_of = 8
    try:
        assert CacheLayer_dsv41_replica(config, a12, 0, N).replica_of == 8
    finally:
        del a12.replica_of

    # a replica that keeps index K: with the split at 28, layer 28 reading source 20 (rate 1)
    rep28 = CacheLayer_dsv41_replica(config, a28, 0, N, with_index_k = True)
    src20 = CacheLayer_dsa(config, a20, 0, N)
    assert (rep28.compress_rate, rep28.epp, rep28.D_c, rep28.D_r, rep28.D_i) == (1, 256, 448, 64, 128)
    assert (rep28.epp, rep28.D_c, rep28.D_r) == (src20.epp, src20.D_c, src20.D_r)
    assert rep28.storage_size() == 1_048_576 * 1280 == 1_342_177_280, rep28.storage_size()
    print(f"  layer 28 <- 20 with index K: m 1, epp 256, D 448/64/128, {rep28.storage_size():,} B")

    # quantized nope part: G*bits int32 words + G fp16 scales per entry, rope stays fp16
    rep8 = CacheLayer_dsv41_replica(config, a12, 0, N, k_bits = 8)
    assert rep8.storage_size() == 524288 * (14 * 8 * 4 + (14 + 64) * 2), rep8.storage_size()

    try:
        rep.tp_export(None)
        raise AssertionError("tp_export must raise")
    except NotImplementedError:
        pass
    try:
        CacheLayer_dsv41_replica(config, a8, 0, N)
        raise AssertionError("a source cannot own a replica of itself")
    except ValueError:
        pass

    # the kernels address a pool row as row * width in int32: 448-wide rows reach 2^31 past
    # 4,793,490 entries, i.e. 9,586,944 tokens at rate 2 and 4,793,344 at rate 1
    for a, kw, lim in ((a12, {}, 9_586_944), (a28, {"with_index_k": True}, 4_793_344)):
        CacheLayer_dsv41_replica(config, a, 0, lim, **kw)
        try:
            CacheLayer_dsv41_replica(config, a, 0, lim + PAGE_SIZE, **kw)
            raise AssertionError(f"layer {a.layer_idx}: a replica past the int32 limit was accepted")
        except ValueError as e:
            assert "int32" in str(e) and f"{lim:,}" in str(e), e
    print("  int32 limit: replicas at 9,586,944 (rate 2) / 4,793,344 (rate 1) tokens build, one page more is refused")
    print("test_real_geometry: ok")


def test_quantized_pool_limit():
    """
    check_pool_addressing on a packed replica: pool_q is G * k_bits int32 words wide and pool_s G
    wide (G = 448 / 32 = 14), so at k_bits 8 the widest row of a rate-2 replica without index K
    is the 112-word pool_q.
    """
    width, epp = 112, 128
    build = lambda t: CacheLayer_dsv41_replica(None, attn_ns(2, 12, 8), 0, t, k_bits = 8)
    lim = ((2 ** 31 - 1) // width) // epp * PAGE_SIZE
    layer = build(lim)
    assert layer.capacity * width < 2 ** 31, layer.capacity
    try:
        build(lim + PAGE_SIZE)
        raise AssertionError("a quantized replica past the int32 limit was accepted")
    except ValueError as e:
        assert "int32" in str(e) and f"{lim:,}" in str(e), e
    print(f"  rate-2 replica, k_bits 8: {lim:,} tokens build ({width}-wide rows), one page more is refused")
    print("test_quantized_pool_limit: ok")


# ---------------------------------------------------------------------------------------------
# 3. row copies

def _check_sync(src, dst, pages_s, pages_d, e0, e1, snap_s, snap_d):
    epp = src.epp
    rs = torch.tensor(rows_of(pages_s, e0, e1, epp), dtype = torch.long)
    rd = torch.tensor(rows_of(pages_d, e0, e1, epp), dtype = torch.long)
    for n in synced_names(src, dst):
        want = snap_d[n].clone()
        D = want.shape[-1]
        if e1 > e0:
            want.view(-1, D)[rd] = snap_s[n].view(-1, D)[rs]
        got = getattr(dst, n)
        assert torch.equal(as_bytes(got), as_bytes(want)), \
            f"{n}: replica differs from the expected rows ({e0}, {e1})"
    for n, t in snap_s.items():
        assert torch.equal(as_bytes(getattr(src, n)), as_bytes(t)), f"source {n} changed"
    if dst.pool_idx is None:
        assert "pool_idx" not in snap_d


def test_sync_rows():
    g = torch.Generator().manual_seed(3)
    rng = random.Random(3)
    cases = 0
    stage_max = R.STAGE_MAX_BYTES
    try:
        for m in (1, 2):
            epp = PAGE_SIZE // m
            for k_bits in (0, 4, 8):
                for src_idx, dst_idx in ((False, False), (True, False), (True, True)):
                    for same_bt in (True, False):
                        num_pages, P = 24, 10
                        src, dst = make_pair(m, k_bits, src_idx, dst_idx, num_pages)
                        fill_(src, g)
                        fill_(dst, g)
                        pages_s = rng.sample(range(num_pages), P)
                        pages_d = list(pages_s) if same_bt else rng.sample(range(num_pages), P)
                        bt_s = torch.tensor([pages_s], dtype = torch.int32)
                        bt_d = torch.tensor([pages_d], dtype = torch.int32)
                        ranges = [
                            (3 * epp - 5, 5 * epp + 7),     # straddles three page boundaries
                            (epp - 1, epp),                 # last entry of a page
                            (2 * epp, 2 * epp + 1),         # first entry of a page
                            (0, P * epp),                   # the whole table
                            (7, 7),                         # empty
                        ]
                        for _ in range(4):
                            k = rng.randint(1, P - 1)
                            ranges.append((k * epp - rng.randint(1, epp - 1),
                                           min(P * epp, k * epp + rng.randint(1, 2 * epp))))
                        for i, (e0, e1) in enumerate(ranges):
                            # pieces: a tiny staging cap splits the range into 1-5 entry pieces
                            R.STAGE_MAX_BYTES = 8192 if i % 2 else stage_max
                            ss, sd = snapshot(src), snapshot(dst)
                            # also accept a 1-D block-table row
                            b_s = bt_s if i % 3 else bt_s.view(-1)
                            sync_pool_replica(src, dst, b_s, bt_d, e0, e1)
                            _check_sync(src, dst, pages_s, pages_d, e0, e1, ss, sd)
                            cases += 1
    finally:
        R.STAGE_MAX_BYTES = stage_max

    # e0 == e1 does nothing at all, not even look at the block table
    src, dst = make_pair(2, 0, True, False, 4)
    fill_(dst, g)
    sd = snapshot(dst)
    sync_pool_replica(src, dst, torch.zeros((1, 0), dtype = torch.int32),
                      torch.zeros((1, 0), dtype = torch.int32), 300, 300)
    for n, t in sd.items():
        assert torch.equal(as_bytes(getattr(dst, n)), as_bytes(t))

    # refusals
    bt = torch.arange(4, dtype = torch.int32).view(1, -1)
    def raises(exc, fn):
        try:
            fn()
        except exc:
            return
        raise AssertionError(f"expected {exc.__name__}")
    s_noidx, d_idx = make_pair(2, 0, False, True, 4)
    raises(ValueError, lambda: sync_pool_replica(s_noidx, d_idx, bt, bt, 0, 3))     # index K it lacks
    s1, _ = make_pair(1, 0, True, False, 4)
    _, d2 = make_pair(2, 0, True, False, 4)
    raises(ValueError, lambda: sync_pool_replica(s1, d2, bt, bt, 0, 3))             # rate
    sq, _ = make_pair(2, 8, True, False, 4)
    _, d16 = make_pair(2, 0, True, False, 4)
    raises(ValueError, lambda: sync_pool_replica(sq, d16, bt, bt, 0, 3))            # k_bits
    raises(ValueError, lambda: sync_pool_replica(src, dst, bt, bt, 5, 4))           # e1 < e0
    raises(IndexError, lambda: sync_pool_replica(src, dst, bt[:, :1], bt, 0, 200))  # beyond the row
    unalloc = CacheLayer_dsv41_replica(None, attn_ns(2, 12, 8), 0, 4 * PAGE_SIZE)
    raises(RuntimeError, lambda: sync_pool_replica(src, unalloc, bt, bt, 0, 3))

    # A whole (B, P) batch table is refused on either side. Flattened, it used to sync
    # sequence 0's pages whatever row the caller meant, leaving row 1's replica entries
    # stale without an error
    bt2 = torch.tensor([[0, 1], [2, 3]], dtype = torch.int32)
    fill_(src, g)
    sd = snapshot(dst)
    raises(ValueError, lambda: sync_pool_replica(src, dst, bt2, bt2, 0, 10))
    raises(ValueError, lambda: sync_pool_replica(src, dst, bt, bt2, 0, 10))
    raises(ValueError, lambda: sync_pool_replica(src, dst, bt2, bt, 0, 10))
    raises(ValueError, lambda: sync_pool_replica(src, dst, bt.view(1, 1, -1), bt, 0, 10))
    # A replica row too short for the range is refused before any piece is scattered
    try:
        R.STAGE_MAX_BYTES = 8192
        raises(IndexError, lambda: sync_pool_replica(src, dst, bt, bt[:, :2], 0, 300))
    finally:
        R.STAGE_MAX_BYTES = stage_max
    for n, t in sd.items():
        assert torch.equal(as_bytes(getattr(dst, n)), as_bytes(t)), f"refused sync changed {n}"
    # the sliced rows of the same table sync exactly those rows
    ss, sd = snapshot(src), snapshot(dst)
    sync_pool_replica(src, dst, bt2[1:2], bt2[1], 0, 200)
    _check_sync(src, dst, [2, 3], [2, 3], 0, 200, ss, sd)
    print(f"test_sync_rows: ok ({cases} ranges)")


# ---------------------------------------------------------------------------------------------
# 4. incremental == full copy under a generator-shaped schedule

class Schedule:
    """
    Drives a source layer and its replica the way the generator drives a kv source: a paged
    allocator, jobs with block tables and positions, forwards that write the closing groups
    into the source and sync the same range, speculative rewinds, prefix sharing with
    partial-page copy_page, defragmentation, freed pages reused.

    policy "range" syncs emission_range(position, tokens) -- the design. policy
    "high_water" is the naive incremental scheme that copies only entries past the
    furthest one it has synced for the job; it misses every entry a rewind rewrites.
    """

    def __init__(self, src, dst, seed, policy = "range", path = None, max_job_pages = 10):
        self.src, self.dst = src, dst
        self.m, self.epp, self.num_pages = src.compress_rate, src.epp, src.num_pages
        self.rng = random.Random(seed)
        self.g = torch.Generator().manual_seed(seed)
        self.free = list(range(self.num_pages))
        self.rng.shuffle(self.free)
        self.refs = {}      # page -> number of jobs whose block table holds it
        self.jobs = {}
        self.next_id = 0
        self.policy = policy
        self.path = path
        self.max_job_pages = max_job_pages
        self.cache = types.SimpleNamespace(
            layers = {(8, 0): src, (12, 0): dst}, num_layers = 2,
            model = types.SimpleNamespace(loaded_tp = False))
        self.src_names = [n for n in POOLS if getattr(src, n) is not None]
        self.names = synced_names(src, dst)
        self.stats = dict(forwards = 0, entries = 0, rewinds = 0, rewound_tokens = 0,
                          forks = 0, partial = 0, defrags = 0, frees = 0, checks = 0, starved = 0)

    # -- jobs

    def _alloc(self):
        p = self.free.pop()
        self.refs[p] = 1
        return p

    def new_job(self, pages = None, pos = 0):
        j = self.next_id
        self.next_id += 1
        self.jobs[j] = dict(pages = list(pages or []), pos = pos, floor = pos, hw = pos // self.m)
        return j

    def forward(self, j, n):
        job, m, epp = self.jobs[j], self.m, self.epp
        p = job["pos"]
        n = min(n, self.max_job_pages * PAGE_SIZE - p)
        if n <= 0:
            return
        need = -(-(p + n) // PAGE_SIZE)
        if need - len(job["pages"]) > len(self.free):
            self.stats["starved"] += 1
            return
        while len(job["pages"]) < need:
            job["pages"].append(self._alloc())
        # the source writes the groups that close in this forward, and only those: a group
        # closes on the token q with q % m == m - 1 and is stored at entry q // m
        closed = [q // m for q in range(p, p + n) if q % m == m - 1]
        e0, e1 = emission_range(p, n, m)
        assert closed == list(range(e0, e1)), (p, n, m)
        if closed:
            rows = torch.tensor([job["pages"][e // epp] * epp + e % epp for e in closed],
                                dtype = torch.long, device = self.src.device)
            for name in self.src_names:
                t = getattr(self.src, name)
                D = t.shape[-1]
                new = rand_like(torch.empty((len(closed), D), dtype = t.dtype), self.g)
                t.view(-1, D).index_copy_(0, rows, new.to(t.device))
        bt_s = torch.tensor([job["pages"]], dtype = torch.int32, device = self.src.device)
        bt_d = torch.tensor([job["pages"]], dtype = torch.int32, device = self.dst.device)
        if self.policy == "range":
            sync_pool_replica(self.src, self.dst, bt_s, bt_d, e0, e1, path = self.path)
        else:
            a = max(e0, job["hw"])
            if a < e1:
                sync_pool_replica(self.src, self.dst, bt_s, bt_d, a, e1, path = self.path)
            job["hw"] = max(job["hw"], e1)
        job["pos"] = p + n
        self.stats["forwards"] += 1
        self.stats["entries"] += e1 - e0

    def rewind(self, j, r):
        # DSV4State.rewind: position only; bounded by guaranteed_rollback, and never into
        # pages another job shares
        job = self.jobs[j]
        r = min(r, DSV4State.guaranteed_rollback, job["pos"] - job["floor"])
        if r > 0:
            job["pos"] -= r
            self.stats["rewinds"] += 1
            self.stats["rewound_tokens"] += r

    def speculate(self, j):
        # a draft of k tokens verified in one forward, some rejected, then the next step
        k = self.rng.randint(1, 8)
        self.forward(j, k + 1)
        self.rewind(j, self.rng.randint(0, k))

    def fork(self, a):
        # a new job reusing A's first k full pages (prefix cache) and, sometimes, the
        # first t tokens of A's next page through Cache.copy_page, as job.py does
        A = self.jobs[a]
        full = A["pos"] // PAGE_SIZE
        if full == 0 or not self.free:
            return
        k = self.rng.randint(1, full)
        A["floor"] = max(A["floor"], k * PAGE_SIZE)     # shared pages are immutable
        pages, pos = A["pages"][:k], k * PAGE_SIZE
        for p in pages:
            self.refs[p] += 1
        rest = A["pos"] - pos
        if rest > 0 and k < len(A["pages"]) and self.rng.random() < 0.7:
            t = self.rng.randint(1, min(PAGE_SIZE - 1, rest))
            new = self._alloc()
            Cache.copy_page(self.cache, self.cache, A["pages"][k], new, t)
            pages = pages + [new]
            pos += t
            self.stats["partial"] += 1
        self.new_job(pages, pos)
        self.stats["forks"] += 1

    def defrag(self):
        # ext.cache_rotate moves every tensor of Cache.get_all_tensors in lockstep; a page
        # permutation over the same list is the CPU equivalent
        perm = torch.randperm(self.num_pages, generator = self.g)
        for t in Cache.get_all_tensors(self.cache):
            moved = torch.empty_like(t)
            moved[perm.to(t.device)] = t
            t.copy_(moved)
        pl = perm.tolist()
        for job in self.jobs.values():
            job["pages"] = [pl[p] for p in job["pages"]]
        self.free = [pl[p] for p in self.free]
        self.refs = {pl[p]: c for p, c in self.refs.items()}
        self.stats["defrags"] += 1

    def free_job(self, j):
        # pages go back to the allocator with their stale entries in both pools, and are
        # handed to the next job that needs one
        for p in self.jobs.pop(j)["pages"]:
            self.refs[p] -= 1
            if self.refs[p] == 0:
                del self.refs[p]
                self.free.append(p)
        self.stats["frees"] += 1

    # -- the property

    def check(self):
        for n in self.names:
            s, d = getattr(self.src, n), getattr(self.dst, n)
            if not torch.equal(as_bytes(s), as_bytes(d)):
                D = s.shape[-1]
                bad = (as_bytes(s).view(-1, D * s.element_size()) !=
                       as_bytes(d).view(-1, D * s.element_size())).any(dim = 1).nonzero().flatten()
                raise AssertionError(f"{n}: replica != whole-pool copy of the source at "
                                     f"{bad.numel()} rows, first {bad[:4].tolist()}")
        self.stats["checks"] += 1

    def run(self, steps):
        self.new_job()
        for _ in range(steps):
            live = list(self.jobs)
            j = self.rng.choice(live)
            op = self.rng.random()
            if op < 0.25:
                self.forward(j, self.rng.choice([2, 3, 7, 64, 255, 256, 257, 300, 700,
                                                 self.rng.randint(1, 1100)]))
            elif op < 0.45:
                for _ in range(self.rng.randint(1, 9)):
                    self.forward(j, 1)
            elif op < 0.70:
                self.speculate(j)
            elif op < 0.78:
                self.rewind(j, self.rng.randint(1, PAGE_SIZE))
            elif op < 0.86:
                self.fork(j)
            elif op < 0.91:
                self.defrag()
            elif op < 0.96:
                if len(live) > 1:
                    self.free_job(j)
            else:
                self.new_job()
            self.check()
        return self.stats


def run_schedules(src_dev = "cpu", dst_dev = "cpu", seeds = (0, 1), steps = 160, path = None,
                  configs = None, num_pages = 48):
    configs = configs or [(m, kb, si, di) for m in (1, 2) for kb in (0, 8)
                          for si, di in ((True, False), (True, True))]
    out = []
    for m, kb, si, di in configs:
        for seed in seeds:
            src, dst = make_pair(m, kb, si, di, num_pages, src_dev, dst_dev)
            st = Schedule(src, dst, seed, path = path).run(steps)
            assert st["rewinds"] and st["forks"] and st["defrags"], st
            out.append(((m, kb, si, di, seed), st))
    return out


def run_high_water_ablation(src_dev = "cpu", dst_dev = "cpu", seeds = (0, 1, 2), steps = 160, path = None):
    """The naive scheme must fail on every seed; returns the step count at which it did."""
    failed = []
    for m in (1, 2):
        for seed in seeds:
            src, dst = make_pair(m, 0, True, False, 48, src_dev, dst_dev)
            try:
                Schedule(src, dst, seed, policy = "high_water", path = path).run(steps)
            except AssertionError as e:
                failed.append((m, seed, str(e)[:90]))
                continue
            raise AssertionError(f"high-water sync passed the schedule (m {m}, seed {seed}): "
                                 f"the schedule does not exercise rewinds")
    return failed


def test_incremental_equals_full_copy():
    res = run_schedules()
    tot = {k: sum(st[k] for _, st in res) for k in res[0][1]}
    print(f"test_incremental_equals_full_copy: ok ({len(res)} schedules; {tot})")
    failed = run_high_water_ablation()
    for m, seed, msg in failed:
        print(f"  high-water ablation m={m} seed={seed} FAILS as it must: {msg}")


def test_copy_policy():
    """Reported peer access must not override the engine's corruption verdict."""
    from unittest.mock import patch
    a, b = torch.device("cuda:0"), torch.device("cuda:1")
    with patch.dict(R._p2p, {}, clear = True), \
            patch.object(torch.cuda, "can_device_access_peer", return_value = True), \
            patch.object(R, "needs_bounce", return_value = True) as bounce:
        assert R._transfer_mode(a, b) == "staged"
        bounce.assert_called_once_with(a, b)
    with patch.dict(R._p2p, {}, clear = True), \
            patch.object(torch.cuda, "can_device_access_peer", return_value = True), \
            patch.object(R, "needs_bounce", return_value = False):
        assert R._transfer_mode(a, b) == "direct"
    with patch.dict(R._p2p, {}, clear = True), \
            patch.object(torch.cuda, "can_device_access_peer", return_value = False), \
            patch.object(R, "needs_bounce") as bounce:
        assert R._transfer_mode(a, b) == "staged"
        bounce.assert_not_called()
    print("test_copy_policy: ok (peer capability and bounce verdict)")


if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    test_real_geometry()
    test_quantized_pool_limit()
    test_sync_rows()
    test_incremental_equals_full_copy()
    test_copy_policy()
    print("ALL OK")
