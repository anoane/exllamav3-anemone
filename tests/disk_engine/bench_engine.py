"""
The disk engine through Python: per backend and access pattern, the time of each call split into
the Python share (interpreter, pybind dispatch, argument conversion, return), the native
submission, the I/O wait and the release, plus latency percentiles and throughput. The native
side stamps its own entry and exit (ext.disk_profile), on the same clock as time.monotonic_ns.

    python tests/disk_engine/bench_engine.py --scratch DIR [--scratch-mb 1024] [--iters 300]
    python tests/disk_engine/bench_engine.py --file SHARD --no-cold      (maintenance window)

Patterns (one Python call each, as the model makes them):
  engram U     one DeepSeek-V4.1 engram layer: U unique rows of a 256-byte table and of an 8-byte
               table in one disk_gather_rows call (U = 48 decode, 96, 8192 prefill-sized)
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

        def prep():
            u = np.unique(self.rng.integers(0, self.rows, size = U)).astype(np.int64)
            return (torch.from_numpy(u),)

        if mode == "engine":
            def call(u):
                # what DSV41Engram does: the whole staging set, argument lists built once per set
                ext.disk_gather_rows(u, 0, fds, bases, rbs, outs)
            label = f"engram U={U}"
        elif mode == "sliced":
            def call(u):
                n = u.numel()
                # the same with the outputs sliced to U rows first (two tensor slices in Python)
                ext.disk_gather_rows(u, 0, [fd, fd], [self.w_base, self.s_base], [256, 8],
                                     [w[:n], s[:n]])
            label = f"sliced U={U}"
        else:
            def call(u):
                n = u.numel()
                ext.ngram_gather_cpu(fd, self.w_base, 256, u, 0, w[:n])
                ext.ngram_gather_cpu(fd, self.s_base, 8, u, 0, s[:n])
            label = f"original U={U}"
        iters = self.iters if U < 4096 else max(5, self.iters // 20)
        return self.run(label, prep, call, U, iters)

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
            rs = [b.null()]
            for U in (48, 96, 8192):
                rs.append(b.engram(U))
            rs.append(b.engram(48, mode = "sliced"))
            for k in (1, 2, 4, 8):
                rs.append(b.extents(k))
            if name == "pread" and not direct:
                ext.disk_engine_configure({"backend": "auto"})
                b.warm()
                for U in (48, 96, 8192):
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
