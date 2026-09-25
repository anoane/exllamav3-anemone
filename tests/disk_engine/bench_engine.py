"""
The disk engine through Python: per backend and access pattern, the time of each call split into
the Python share (interpreter, pybind dispatch, argument conversion, return), the native
submission, the I/O wait and the release, plus latency percentiles and throughput. The native
side stamps its own entry and exit (ext.disk_profile), on the same clock as time.monotonic_ns.

    python tests/disk_engine/bench_engine.py --scratch DIR [--scratch-mb 1024] [--iters 300]
    python tests/disk_engine/bench_engine.py --file SHARD --no-cold      (maintenance window)

Patterns (one Python call each, as the model makes them):
  engram U     one DeepSeek-V4.1 engram layer exactly as DSV41Engram._stage calls it: U unique
               rows of a 256-byte table and of an 8-byte table in one disk_gather_rows call, the
               argument lists built once, class and hold positional (U = 48 decode, 96, 8192)
  keywords U   the same with cls= / hold= keywords and the lists built per call (pybind's
               keyword path; what the model did before)
  sync-cls2 U  a synchronous class-2 gather (how the look-ahead used to stage)
  prefetch U   the prefill look-ahead: a class-2 ticket (wait=False), after --overlap-ms of
               simulated compute promoted to class 0, waited for and released; stall is the
               time the forward waits after the overlap
  original U   the same rows through the original path: two ngram_gather_cpu calls
  extents k    k expert extents of 13,315,596 bytes in one disk_read_extents call
  null         an empty gather: the fixed cost of a call

--cold drops the scratch file's pages before each call (only a file this run created). Uses the
full extension, or the test build with EXL3_DISK_TEST_MINI=1 (tests/disk_engine/mini_ext.py).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_ext():
    if os.environ.get("EXL3_DISK_TEST_MINI") == "1":
        sys.path.insert(0, HERE)
        from mini_ext import load_ext as mini
        return mini()
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
    from exllamav3.ext import exllamav3_ext
    return exllamav3_ext


def pct(v):
    a = np.sort(np.asarray(v, dtype = np.float64)) / 1e3
    if not len(a):
        return {}
    q = lambda p: float(a[min(len(a) - 1, int(p * len(a)))])
    return {"p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "p99.9": q(0.999), "mean": float(a.mean())}


def aligned_u8(nbytes, align = 4096):
    t = torch.empty(nbytes + align, dtype = torch.uint8)
    pad = (-t.data_ptr()) % align
    return t[pad : pad + nbytes]


class Bench:
    def __init__(self, ext, path, size, cold, iters, extent_gb):
        self.ext, self.path, self.size, self.cold, self.iters = ext, path, size, cold, iters
        self.extent_gb = extent_gb
        self.fd = os.open(path, os.O_RDONLY)
        self.w_base = 664
        self.s_base = min(1 << 30, size // 2) + 2712
        self.rows = min((self.s_base - self.w_base) // 256, (size - self.s_base) // 8)
        self.rng = np.random.default_rng(0)

    def drop(self):
        if self.cold:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def warm(self):
        """Read the row tables once, so warm cases start from the page cache (the extent cases
        of the previous backend drop the pages they read through it)."""
        if self.cold:
            return
        end = min(self.size, self.s_base + self.rows * 8)
        for off in range(0, end, 1 << 20):
            os.pread(self.fd, min(1 << 20, end - off), off)

    def run(self, label, prep, call, units, iters):
        ext = self.ext
        wall, py, pre, post, sub, io, rel = [], [], [], [], [], [], []
        native = not label.startswith("original")     # the original gather has no stamps
        ext.disk_profile(native)
        for i in range(iters):
            args = prep()
            self.drop()
            t0 = time.monotonic_ns()
            call(*args)
            t1 = time.monotonic_ns()
            wall.append(t1 - t0)
            if native:
                e, s, d, x = ext.disk_profile_last()
                pre.append(e - t0)                  # Python, dispatch, argument conversion
                post.append(t1 - x)                 # return to Python, GIL taken back
                py.append((e - t0) + (t1 - x))
                sub.append(s - e)
                io.append(d - s)
                rel.append(x - d)
        ext.disk_profile(False)
        r = {"case": label, "calls": iters, "wall_us": pct(wall),
             "throughput": units * iters / (sum(wall) / 1e9)}
        if py:
            r.update({"python_us": pct(py), "python_in_us": pct(pre), "python_out_us": pct(post),
                      "submit_us": pct(sub), "io_us": pct(io), "release_us": pct(rel),
                      "python_share": float(np.median(np.asarray(py) / np.asarray(wall)))})
        return r

    def engram(self, U, mode = "engine"):
        ext, fd = self.ext, self.fd
        w = torch.empty((U, 256), dtype = torch.uint8)
        s = torch.empty((U, 8), dtype = torch.uint8)
        fds, bases, rbs, outs = [fd, fd], [self.w_base, self.s_base], [256, 8], [w, s]
        hold = U < 256

        def prep():
            u = np.unique(self.rng.integers(0, self.rows, size = U)).astype(np.int64)
            return (torch.from_numpy(u),)

        if mode == "engine":
            def call(u):
                # what DSV41Engram._stage does: the whole staging set, lists built once per
                # descriptor pair, class and hold positional
                ext.disk_gather_rows(u, 0, fds, bases, rbs, outs, 0, hold)
            label = f"engram U={U}"
        elif mode == "sync2":
            def call(u):
                # a synchronous class-2 gather (the prefill look-ahead before it was promoted)
                ext.disk_gather_rows(u, 0, fds, bases, rbs, outs, 2, False)
            label = f"sync-cls2 U={U}"
        elif mode == "keywords":
            def call(u):
                ext.disk_gather_rows(u, 0, [fd, fd], [self.w_base, self.s_base], [256, 8], [w, s],
                                     cls = 0, hold = hold)
            label = f"keywords U={U}"
        elif mode == "sliced":
            def call(u):
                n = u.numel()
                # the same with the outputs sliced to U rows first (two tensor slices in Python)
                ext.disk_gather_rows(u, 0, [fd, fd], [self.w_base, self.s_base], [256, 8],
                                     [w[:n], s[:n]])
            label = f"sliced U={U}"
        else:
            # as DSV41Engram._fds() prepares the original path: no readahead around the rows
            # (on a small scratch file, readahead would fetch neighbouring rows the real table
            # never has)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)

            def call(u):
                n = u.numel()
                ext.ngram_gather_cpu(fd, self.w_base, 256, u, 0, w[:n])
                ext.ngram_gather_cpu(fd, self.s_base, 8, u, 0, s[:n])
            label = f"original U={U}"
        iters = self.iters if U < 4096 else max(5, self.iters // 20)
        return self.run(label, prep, call, U, iters)

    def prefetch(self, U, overlap_ms):
        """The prefill look-ahead as DSV41Engram stages it: class 2, not waited for; after the
        overlap the forward promotes it to class 0, waits, releases."""
        ext, fd = self.ext, self.fd
        w = torch.empty((U, 256), dtype = torch.uint8)
        s = torch.empty((U, 8), dtype = torch.uint8)
        fds, bases, rbs, outs = [fd, fd], [self.w_base, self.s_base], [256, 8], [w, s]
        iters = max(5, self.iters // 20)
        wall, stall = [], []
        for _ in range(iters):
            u = torch.from_numpy(np.unique(self.rng.integers(0, self.rows, size = U)).astype(np.int64))
            self.drop()
            t0 = time.monotonic_ns()
            t = ext.disk_gather_rows(u, 0, fds, bases, rbs, outs, 2, False, False)
            if overlap_ms:
                time.sleep(overlap_ms / 1e3)
            t1 = time.monotonic_ns()
            t.promote(0)
            t.wait()
            t.release()
            t2 = time.monotonic_ns()
            wall.append(t2 - t0)
            stall.append(t2 - t1)
        return {"case": f"prefetch U={U} +{overlap_ms:g}ms", "calls": iters, "wall_us": pct(wall),
                "stall_us": pct(stall), "throughput": U * iters / (sum(wall) / 1e9)}

    def extents(self, k, L = 13315596):
        ext, fd = self.ext, self.fd
        g = ext.disk_extent_geometry(fd, 1, L)
        slot = g[1] + 4096
        dst = aligned_u8(slot * k)
        fds = torch.full((k,), fd, dtype = torch.long)
        dsto = torch.arange(k, dtype = torch.long) * slot
        lens = torch.full((k,), L, dtype = torch.long)

        def prep():
            return (torch.from_numpy(self.rng.integers(0, self.size - L, size = k).astype(np.int64)),)

        def call(offs):
            ext.disk_read_extents(fds, offs, lens, dst, dsto, slot, cls = 1, wait = True)

        iters = max(3, min(self.iters, int(self.extent_gb * 1e9 // (L * k))))
        r = self.run(f"extents k={k}", prep, call, L * k / 1e9, iters)
        r["throughput_unit"] = "GB/s"
        return r

    def null(self):
        ext, fd = self.ext, self.fd
        u = torch.empty(0, dtype = torch.long)
        w = torch.empty((0, 256), dtype = torch.uint8)
        s = torch.empty((0, 8), dtype = torch.uint8)
        return self.run("null", lambda: (), lambda: ext.disk_gather_rows(
            u, 0, [fd, fd], [self.w_base, self.s_base], [256, 8], [w, s]), 1, self.iters)


def fmt(r):
    def g(k, f = "p50"):
        return r.get(k, {}).get(f, float("nan"))
    unit = r.get("throughput_unit", "/s")
    if "stall_us" in r:
        return (f"  {r['case']:<22} wall p50 {g('wall_us'):9.1f} p90 {g('wall_us', 'p90'):9.1f} "
                f"p99 {g('wall_us', 'p99'):9.1f} us | stall after the overlap p50 "
                f"{g('stall_us'):9.1f} p90 {g('stall_us', 'p90'):9.1f} p99 "
                f"{g('stall_us', 'p99'):9.1f} us | {r['throughput']:.4g} {unit}")
    return (f"  {r['case']:<16} wall p50 {g('wall_us'):9.1f} p90 {g('wall_us', 'p90'):9.1f} "
            f"p99 {g('wall_us', 'p99'):9.1f} p99.9 {g('wall_us', 'p99.9'):9.1f} us | "
            f"python {g('python_us'):5.1f} (in {g('python_in_us'):4.1f} out "
            f"{g('python_out_us'):4.1f}) submit {g('submit_us'):7.1f} io {g('io_us'):9.1f} "
            f"release {g('release_us'):5.1f} us | python share "
            f"{100 * r.get('python_share', float('nan')):5.2f} % | "
            f"{r['throughput']:.4g} {unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--scratch")
    ap.add_argument("--scratch-mb", type = int, default = 1024)
    ap.add_argument("--iters", type = int, default = 300)
    ap.add_argument("--cold", action = "store_true")
    ap.add_argument("--backends", default = "pread,odirect,io_uring,io_uring:all")
    ap.add_argument("--extent-gb", type = float, default = 0.4, help = "bytes read per extent case")
    ap.add_argument("--overlap-ms", type = float, default = 20.0,
                    help = "simulated compute between a prefetch and its forward")
    ap.add_argument("--only", default = "",
                    help = "comma-separated case prefixes to run (e.g. 'engram U=8192,original')")
    ap.add_argument("--json")
    a = ap.parse_args()
    assert bool(a.file) != bool(a.scratch), "one of --file or --scratch"
    ext = load_ext()
    owned = False
    if a.scratch:
        os.makedirs(a.scratch, exist_ok = True)
        path = os.path.join(a.scratch, "bench_engine.bin")
        rng = np.random.default_rng(1)
        with open(path, "wb") as f:
            for _ in range(a.scratch_mb):
                f.write(rng.integers(0, 256, size = 1 << 20, dtype = np.uint8).tobytes())
        os.sync()
        owned = True
    else:
        path = a.file
    assert not a.cold or owned, "--cold only for a --scratch file this run created"
    size = os.path.getsize(path)
    print(f"file {path} ({size / 2**30:.2f} GiB){', cold' if a.cold else ', warm after first touch'}")
    results = []
    try:
        for spec in a.backends.split(","):
            name, _, direct = spec.partition(":")
            cfg = {"backend": name}
            if direct:
                cfg["direct"] = direct
            info = ext.disk_engine_configure(cfg)
            print(f"-- {spec}: {info['describe']}")
            b = Bench(ext, path, size, a.cold, a.iters, a.extent_gb)
            b.warm()
            only = [x.strip() for x in a.only.split(",") if x.strip()]
            want = lambda label: not only or any(label.startswith(x) for x in only)
            cases = [("null", b.null)]
            cases += [(f"engram U={U}", lambda U = U: b.engram(U)) for U in (48, 96, 8192, 32768)]
            cases += [("keywords U=48", lambda: b.engram(48, mode = "keywords")),
                      ("sliced U=48", lambda: b.engram(48, mode = "sliced"))]
            for U in (8192, 32768):
                cases += [(f"sync-cls2 U={U}", lambda U = U: b.engram(U, mode = "sync2")),
                          (f"prefetch U={U} +0ms", lambda U = U: b.prefetch(U, 0)),
                          (f"prefetch U={U} +{a.overlap_ms:g}ms", lambda U = U: b.prefetch(U, a.overlap_ms))]
            cases += [(f"extents k={k}", lambda k = k: b.extents(k)) for k in (1, 2, 4, 8)]
            rs = [f() for label, f in cases if want(label)]
            originals = [U for U in (48, 96, 8192, 32768) if want(f"original U={U}")]
            if name == "pread" and not direct and originals:
                ext.disk_engine_configure({"backend": "auto"})
                b.warm()
                for U in originals:
                    rs.append(b.engram(U, mode = "original"))
            for r in rs:
                r["backend"] = spec
                print(fmt(r))
            results += rs
            os.close(b.fd)
    finally:
        ext.disk_engine_shutdown()
        if owned:
            os.unlink(path)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent = 1)


if __name__ == "__main__":
    main()
