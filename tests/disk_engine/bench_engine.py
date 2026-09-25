"""
The disk engine through Python, called exactly as the model calls it. For each call it splits
the time into the Python share (interpreter, pybind dispatch, argument conversion, return), the
native submission, the I/O wait and the release, using stamps the native side takes on the same
clock as time.monotonic_ns (ext.disk_profile). It also times the original ngram_gather_cpu path
(EXL3_DISK_BACKEND=original) on the same rows, which is what auto's n-gram route is decided
against. tests/disk_engine/bench_suite.py runs it next to the native driver and merges both.

    python tests/disk_engine/bench_engine.py --scratch DIR [--scratch-mb 1024] [--json OUT]
    python tests/disk_engine/bench_engine.py --layer PATH:OFF:ROW_BYTES:ROWS+PATH:OFF:8:ROWS \\
        --extent-list LIST --json OUT          (model files: the page cache is left as found)

Variants (--variants): pread, odirect, io_uring (the engine with that backend), original (two
ngram_gather_cpu calls per layer through the original pool; engram cases only). A variant may
carry knobs: "io_uring:direct=all".

Cases (--case, repeatable; KIND:key=value,...):
  engram    u=48 cache=cold iters=300   one engram layer as DSV41Engram._stage calls it: u unique
                                        ids, both tables in one disk_gather_rows call, argument
                                        lists built once, class and hold positional (original:
                                        the two ngram_gather_cpu calls)
  keywords  u=48 ...                    the same with cls= / hold= keywords and lists built per call
  prefetch  u=8192 overlap_ms=20 ...    the prefill look-ahead: a class-2 ticket, after the overlap
                                        promoted to class 0, waited for and released; `stall` is
                                        the forward's wait after the overlap
  extents   k=4 iters=5                 k expert extents in one disk_read_extents call
  null      iters=1000                  an empty call: the fixed cost

cache = cold | warm | F: before every call exactly the pages the call reads are dropped
(posix_fadvise DONTNEED), read, or made resident in fraction F, and checked with cachestat(2);
never a global drop_caches. --preserve-cache (default for files this run did not create) puts
every call's pages back as they were afterwards. Uses the full extension, or the test build with
EXL3_DISK_TEST_MINI=1 (tests/disk_engine/mini_ext.py).
"""

import argparse
import ctypes
import json
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.sysconf("SC_PAGESIZE")


def load_ext():
    if os.environ.get("EXL3_DISK_TEST_MINI") == "1":
        sys.path.insert(0, HERE)
        from mini_ext import load_ext as mini
        return mini()
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
    from exllamav3.ext import exllamav3_ext
    return exllamav3_ext


# ---- page-cache state of exact ranges (cachestat(2), Linux 6.5+) -------------------------------------

class _CsRange(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint64), ("len", ctypes.c_uint64)]


class _CsStat(ctypes.Structure):
    _fields_ = [("nr_cache", ctypes.c_uint64), ("nr_dirty", ctypes.c_uint64),
                ("nr_writeback", ctypes.c_uint64), ("nr_evicted", ctypes.c_uint64),
                ("nr_recently_evicted", ctypes.c_uint64)]


_libc = ctypes.CDLL(None, use_errno = True)
_libc.syscall.restype = ctypes.c_long
_SYS_CACHESTAT = 451 if platform.machine() in ("x86_64", "aarch64") else None


def resident(fd, p0, n):
    """Resident pages of [p0, p0 + n), or None when cachestat is unavailable."""
    if _SYS_CACHESTAT is None or n <= 0:
        return None if n > 0 else 0
    r = _CsRange(p0 * PAGE, n * PAGE)
    s = _CsStat()
    rc = _libc.syscall(ctypes.c_long(_SYS_CACHESTAT), ctypes.c_uint(fd), ctypes.byref(r),
                       ctypes.byref(s), ctypes.c_uint(0))
    if rc != 0:
        e = ctypes.get_errno()
        if e in (38, 1, 95):            # ENOSYS, EPERM, EOPNOTSUPP
            return None
        raise OSError(e, os.strerror(e))
    return int(s.nr_cache)


def merge(ranges):
    """[(fd, p0, n)] sorted and merged (overlapping or adjacent pages of one descriptor)."""
    out = []
    for fd, p0, n in sorted(ranges):
        if out and out[-1][0] == fd and p0 <= out[-1][1] + out[-1][2]:
            f, q0, m = out[-1]
            out[-1] = (f, q0, max(m, p0 + n - q0))
        else:
            out.append((fd, p0, n))
    return out


def runs(fd, p0, n, out):
    c = resident(fd, p0, n)
    if c is None:
        return False
    if c == 0:
        return True
    if c >= n:
        out.append((fd, p0, n))
        return True
    h = n // 2
    return runs(fd, p0, h, out) and runs(fd, p0 + h, n - h, out)


def drop(ranges):
    for fd, p0, n in ranges:
        os.posix_fadvise(fd, p0 * PAGE, n * PAGE, os.POSIX_FADV_DONTNEED)


def warm(ranges):
    got = 0
    for fd, p0, n in ranges:
        off, end = p0 * PAGE, (p0 + n) * PAGE
        while off < end:
            b = os.pread(fd, min(1 << 20, end - off), off)
            if not b:
                break
            off += len(b)
            got += len(b)
    return got


def minus(all_, inn):
    out, j = [], 0
    for fd, p0, n in all_:
        cur, end = p0, p0 + n
        while j < len(inn) and (inn[j][0], inn[j][1] + inn[j][2]) <= (fd, cur):
            j += 1
        k = j
        while k < len(inn) and inn[k][0] == fd and inn[k][1] < end:
            if inn[k][1] > cur:
                out.append((fd, cur, inn[k][1] - cur))
            cur = max(cur, inn[k][1] + inn[k][2])
            k += 1
        if cur < end:
            out.append((fd, cur, end - cur))
    return out


class Cache:
    """Per-call page-cache preparation and restore, with a tally for the report."""

    def __init__(self, mode, preserve, rng):
        self.mode, self.preserve, self.rng = mode, preserve, rng
        self.frac = 0.0 if mode == "cold" else 1.0 if mode == "warm" else float(mode)
        assert 0.0 <= self.frac <= 1.0, f"cache {mode}"
        self.calls = self.pages = self.res = self.want = self.restored = self.prep_bytes = 0
        self.known = True

    def prepare(self, ranges):
        ranges = merge(ranges)
        before = []
        if self.preserve:
            for fd, p0, n in ranges:
                if not runs(fd, p0, n, before):
                    raise RuntimeError("--preserve-cache needs cachestat(2) (Linux 6.5+)")
        drop(ranges)
        n = sum(r[2] for r in ranges)
        want = 0
        if self.frac >= 1.0:
            self.prep_bytes += warm(ranges)
            want = n
        elif self.frac > 0.0:
            want = int(round(self.frac * n))
            pages = [(fd, p0 + i) for fd, p0, m in ranges for i in range(m)]
            pick = self.rng.choice(len(pages), size = want, replace = False) if want else []
            self.prep_bytes += warm(merge([(pages[i][0], pages[i][1], 1) for i in pick]))
        res = 0
        for fd, p0, m in ranges:
            c = resident(fd, p0, m)
            if c is None:
                self.known = False
                break
            res += c
        self.calls += 1
        self.pages += n
        self.want += want
        self.res += res
        return ranges, before

    def restore(self, token):
        if not self.preserve:
            return
        ranges, before = token
        drop(minus(ranges, before))
        back = [r for r in before if (resident(*r) or 0) < r[2]]
        self.restored += sum(r[2] for r in back)
        self.prep_bytes += warm(back)

    def summary(self):
        return {"mode": self.mode, "method": "cachestat" if _SYS_CACHESTAT else "none",
                "pages_per_call": self.pages / self.calls if self.calls else None,
                "target_resident_fraction": self.want / self.pages if self.pages else None,
                "resident_fraction": (self.res / self.pages) if self.known and self.pages else None,
                "off_target_pages": abs(self.res - self.want) if self.known else None,
                "restored_pages": self.restored, "prep_read_bytes": self.prep_bytes}


# ---- statistics ------------------------------------------------------------------------------------

def pct(v):
    """Nearest-rank percentiles, in microseconds."""
    a = np.sort(np.asarray(v, dtype = np.float64)) / 1e3
    if not len(a):
        return {"n": 0}
    q = lambda p: float(a[max(1, min(len(a), int(np.ceil(p * len(a))))) - 1])
    return {"n": int(len(a)), "min": float(a[0]), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99),
            "p99.9": q(0.999), "max": float(a[-1]), "mean": float(a.mean())}


def aligned_u8(nbytes, align = 4096):
    t = torch.empty(nbytes + align, dtype = torch.uint8)
    pad = (-t.data_ptr()) % align
    return t[pad : pad + nbytes]


# ---- data sources ----------------------------------------------------------------------------------

def parse_table(s):
    path, off, rb, rows = s.rsplit(":", 3)
    return path, int(off), int(rb), int(rows)


class Files:
    def __init__(self):
        self.fds = {}

    def fd(self, path):
        if path not in self.fds:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)     # no readahead around our pages
            self.fds[path] = fd
        return self.fds[path]

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}


class Layer:
    def __init__(self, files, spec):
        self.tables = [parse_table(t) for t in spec.split("+")]
        self.fds = [files.fd(t[0]) for t in self.tables]
        self.offs = [t[1] for t in self.tables]
        self.rbs = [t[2] for t in self.tables]
        self.rows = min(t[3] for t in self.tables)

    def pages(self, ids):
        out = []
        for fd, off, rb in zip(self.fds, self.offs, self.rbs):
            a = off + ids * rb
            p0 = a // PAGE
            p1 = (a + rb + PAGE - 1) // PAGE
            out += [(fd, int(x), int(y - x)) for x, y in zip(p0, p1)]
        return out


# ---- the benchmark ----------------------------------------------------------------------------------

class Bench:
    def __init__(self, ext, layers, extents, preserve, seed):
        self.ext, self.layers, self.extents, self.preserve = ext, layers, extents, preserve
        self.rng = np.random.default_rng(seed)
        self.li = 0

    def next_layer(self):
        L = self.layers[self.li % len(self.layers)]
        self.li += 1
        return L

    def ids(self, L, u):
        assert u <= L.rows, f"u={u} exceeds the table's {L.rows} rows"
        a = np.unique(self.rng.integers(0, L.rows, size = u))
        while len(a) < u:
            a = np.unique(np.concatenate([a, self.rng.integers(0, L.rows, size = u - len(a))]))
        return a.astype(np.int64)

    def timed(self, label, variant, params, cache, iters, one, units, unit, native):
        """one(i) -> (cache token, call) where call() runs the timed call."""
        ext = self.ext
        wall, py, pre, post, sub, io, rel = [], [], [], [], [], [], []
        ext.disk_profile(native)
        try:
            for i in range(iters):
                token, call = one(i)
                t0 = time.monotonic_ns()
                call()
                t1 = time.monotonic_ns()
                wall.append(t1 - t0)
                if native:
                    e, s, d, x = ext.disk_profile_last()
                    assert t0 <= e <= s <= d <= x <= t1, "profile stamps out of order"
                    pre.append(e - t0)              # Python, dispatch, argument conversion
                    post.append(t1 - x)             # return to Python, GIL taken back
                    py.append((e - t0) + (t1 - x))
                    sub.append(s - e)
                    io.append(d - s)
                    rel.append(x - d)
                cache.restore(token)
        finally:
            ext.disk_profile(False)
        r = {"variant": variant, "case": label, "kind": label.split(":")[0], "params": params,
             "cache": cache.summary(), "calls": len(wall), "wall_us": pct(wall),
             "throughput": units * len(wall) / (sum(wall) / 1e9) if wall else None,
             "throughput_unit": unit}
        if py:
            share = np.asarray(py, dtype = np.float64) / np.asarray(wall, dtype = np.float64)
            r.update({"python_us": pct(py), "python_in_us": pct(pre), "python_out_us": pct(post),
                      "submit_us": pct(sub), "io_us": pct(io), "release_us": pct(rel),
                      "python_share": float(np.median(share)),
                      "python_share_of_mean": float(np.sum(py) / np.sum(wall))})
        return r

    def engram(self, variant, spec, u, cache, iters, keywords = False):
        ext = self.ext
        # one staging set per layer, as the module keeps its pin sets
        sets = {}
        for L in self.layers:
            outs = [torch.empty((u, rb), dtype = torch.uint8) for rb in L.rbs]
            sets[id(L)] = (outs, [L.fds, L.offs, L.rbs, outs])
        hold = u < 256

        def one(i):
            L = self.next_layer()
            ids = self.ids(L, u)
            uids = torch.from_numpy(ids)
            outs, argv = sets[id(L)]
            token = cache.prepare(L.pages(ids))
            if variant == "original":
                def call():
                    # what DSV41Engram._stage does without the engine: one call per table
                    for k in range(len(outs)):
                        ext.ngram_gather_cpu(L.fds[k], L.offs[k], L.rbs[k], uids, 0, outs[k])
            elif keywords:
                def call():
                    ext.disk_gather_rows(uids, 0, list(L.fds), list(L.offs), list(L.rbs),
                                         list(outs), cls = 0, hold = hold)
            else:
                fds, offs, rbs, o = argv
                def call():
                    # what DSV41Engram._stage does: lists built once, class and hold positional
                    ext.disk_gather_rows(uids, 0, fds, offs, rbs, o, 0, hold)
            return token, call

        return self.timed(spec, variant, {"u": u, "tables": len(self.layers[0].tables)}, cache,
                          iters, one, u, "rows/s", native = variant != "original")

    def prefetch(self, variant, spec, u, overlap_ms, cache, iters):
        ext = self.ext
        wall, stall = [], []
        for _ in range(iters):
            L = self.next_layer()
            ids = self.ids(L, u)
            uids = torch.from_numpy(ids)
            outs = [torch.empty((u, rb), dtype = torch.uint8) for rb in L.rbs]
            token = cache.prepare(L.pages(ids))
            t0 = time.monotonic_ns()
            t = ext.disk_gather_rows(uids, 0, L.fds, L.offs, L.rbs, outs, 2, False, False)
            if overlap_ms:
                time.sleep(overlap_ms / 1e3)
            t1 = time.monotonic_ns()
            t.promote(0)
            t.wait()
            t.release()
            t2 = time.monotonic_ns()
            wall.append(t2 - t0)
            stall.append(t2 - t1)
            cache.restore(token)
        return {"variant": variant, "case": spec, "kind": "prefetch",
                "params": {"u": u, "overlap_ms": overlap_ms}, "cache": cache.summary(),
                "calls": iters, "wall_us": pct(wall), "stall_us": pct(stall),
                "throughput": u * iters / (sum(wall) / 1e9), "throughput_unit": "rows/s"}

    def extents_case(self, variant, spec, k, cache, iters):
        ext = self.ext
        L = max(x[2] for x in self.extents)
        slot = L + 2 * 65536 + 4096 - (L % 4096)
        dst = aligned_u8(slot * k)
        dsto = torch.arange(k, dtype = torch.long) * slot
        pick = lambda: [self.extents[int(j)] for j in self.rng.integers(0, len(self.extents), size = k)]

        def one(i):
            ex = pick()
            fds = torch.tensor([x[0] for x in ex], dtype = torch.long)
            offs = torch.tensor([x[1] for x in ex], dtype = torch.long)
            lens = torch.tensor([x[2] for x in ex], dtype = torch.long)
            token = cache.prepare([(x[0], x[1] // PAGE, (x[1] + x[2] + PAGE - 1) // PAGE - x[1] // PAGE)
                                   for x in ex])
            def call():
                ext.disk_read_extents(fds, offs, lens, dst, dsto, slot, None, 1, False, True)
            return token, call

        r = self.timed(spec, variant, {"k": k, "extent_bytes": L}, cache, iters, one,
                       L * k / 1e9, "GB/s", native = True)
        return r

    def null(self, variant, spec, iters):
        ext = self.ext
        L = self.layers[0]
        u = torch.empty(0, dtype = torch.long)
        outs = [torch.empty((0, rb), dtype = torch.uint8) for rb in L.rbs]
        cache = Cache("warm", False, self.rng)

        def one(i):
            return (None, None), lambda: ext.disk_gather_rows(u, 0, L.fds, L.offs, L.rbs, outs, 0, False)

        cache.restore = lambda token: None
        return self.timed(spec, variant, {}, cache, iters, one, 1, "calls/s", native = True)


def parse_case(s):
    kind, _, rest = s.partition(":")
    kv = dict(x.split("=", 1) for x in rest.split(",") if x)
    return kind, kv


def fmt(r):
    def g(k, f = "p50"):
        v = r.get(k, {}).get(f)
        return float("nan") if v is None else v
    s = (f"  {r['variant']:<14} {r['case']:<34} wall p50 {g('wall_us'):9.1f} p99 "
         f"{g('wall_us', 'p99'):9.1f} us")
    if "python_us" in r:
        s += (f" | python {g('python_us'):5.1f} us (in {g('python_in_us'):4.1f} out "
              f"{g('python_out_us'):4.1f}), share {100 * r['python_share']:5.2f} %")
    if "stall_us" in r:
        s += f" | stall p50 {g('stall_us'):9.1f} us"
    return s + f" | {r['throughput']:.4g} {r['throughput_unit']}"


DEFAULT_CASES = ["null:iters=2000", "engram:u=48,cache=cold,iters=300", "engram:u=48,cache=warm,iters=300",
                 "keywords:u=48,cache=warm,iters=300", "engram:u=96,cache=cold,iters=200",
                 "engram:u=8192,cache=cold,iters=5", "engram:u=8192,cache=warm,iters=5",
                 "prefetch:u=8192,overlap_ms=20,cache=cold,iters=5", "extents:k=1,iters=5",
                 "extents:k=4,iters=5"]


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", help = "create a scratch file of --scratch-mb MiB here and use it")
    ap.add_argument("--scratch-mb", type = int, default = 1024)
    ap.add_argument("--layer", action = "append", default = [],
                    help = "PATH:OFFSET:ROW_BYTES:ROWS[+...]: the tables one engram call reads")
    ap.add_argument("--extent-list", help = "lines 'PATH OFFSET LENGTH'")
    ap.add_argument("--extent-region", action = "append", default = [],
                    help = "PATH:OFFSET:LENGTH: random extents of --extent-bytes inside it")
    ap.add_argument("--extent-bytes", type = int, default = 13315596)
    ap.add_argument("--variants", default = "pread,odirect,io_uring,original")
    ap.add_argument("--case", action = "append", default = [])
    ap.add_argument("--preserve-cache", type = int, choices = (0, 1), default = None)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--json")
    a = ap.parse_args()
    assert bool(a.scratch) != bool(a.layer), "either --scratch or --layer"
    ext = load_ext()
    files = Files()
    owned = None
    try:
        if a.scratch:
            os.makedirs(a.scratch, exist_ok = True)
            owned = os.path.join(a.scratch, "bench_engine.bin")
            size = a.scratch_mb << 20
            with open(owned, "wb") as f:
                for _ in range(a.scratch_mb // 4):
                    f.write(os.urandom(4 << 20))
                f.flush()
                os.fsync(f.fileno())
            fd = files.fd(owned)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            s_off = (size * 9 // 10) // 4096 * 4096 + 2712
            rows = min((s_off - 664) // 256, (size - s_off) // 8)
            a.layer = [f"{owned}:664:256:{rows}+{owned}:{s_off}:8:{rows}"]
            a.extent_region = [f"{owned}:0:{size}"]
        preserve = bool(a.preserve_cache if a.preserve_cache is not None else not owned)
        layers = [Layer(files, s) for s in a.layer]
        extents = []
        if a.extent_list:
            with open(a.extent_list) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        path, off, n = line.rsplit(" ", 2)
                        extents.append((files.fd(path), int(off), int(n)))
        rng = np.random.default_rng(a.seed + 1)
        for reg in a.extent_region:
            path, off, n = reg.rsplit(":", 2)
            off, n = int(off), int(n)
            for x in rng.integers(off, off + n - a.extent_bytes, size = 4096):
                extents.append((files.fd(path), int(x), a.extent_bytes))
        cases = a.case or DEFAULT_CASES
        results = []
        b = Bench(ext, layers, extents, preserve, a.seed)
        for v in a.variants.split(","):
            name, _, knobs = v.partition(":")
            cfg = {"backend": name}
            for kv in filter(None, knobs.split(";")):
                k, _, val = kv.partition("=")
                cfg[k] = val
            info = ext.disk_engine_configure(cfg)
            print(f"-- {v}: {info['describe']} (n-gram route {info['ngram_route']})", flush = True)
            # untimed: the engine opens the files on first use, and an idle device may wake slowly
            warm_cache = Cache("cold", preserve, b.rng)
            for L in layers:
                b.engram(name, "warmup", 1, warm_cache, 1)
            for spec in cases:
                kind, kv = parse_case(spec)
                iters = int(kv.get("iters", 100))
                cache = Cache(kv.get("cache", "cold"), preserve, b.rng)
                if name == "original" and kind not in ("engram",):
                    continue
                if kind in ("engram", "keywords"):
                    r = b.engram(name, spec, int(kv.get("u", 48)), cache, iters, keywords = kind == "keywords")
                elif kind == "prefetch":
                    r = b.prefetch(name, spec, int(kv.get("u", 8192)), float(kv.get("overlap_ms", 20)),
                                   cache, iters)
                elif kind == "extents":
                    if not extents:
                        continue
                    r = b.extents_case(name, spec, int(kv.get("k", 4)), cache, iters)
                elif kind == "null":
                    r = b.null(name, spec, iters)
                else:
                    raise SystemExit(f"unknown case kind {kind}")
                r["variant_spec"] = v
                r["describe"] = info["describe"]
                r["ngram_route"] = info["ngram_route"]
                print(fmt(r), flush = True)
                results.append(r)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(results, f, indent = 1)
    finally:
        try:
            ext.disk_engine_shutdown()
        finally:
            files.close()
            if owned:
                os.unlink(owned)


if __name__ == "__main__":
    main()
