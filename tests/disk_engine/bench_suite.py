#!/usr/bin/env python3
"""
Disk engine benchmark suite: runs the native driver (disk_engine_bench) over a grid of backends
and knobs, runs the Python-path benchmark (bench_engine.py) for the Python share and the
original-gather comparison, and turns the results into a report, a comparison between two
machines, and the decision that sets what EXL3_DISK_BACKEND=auto means.

Python only orchestrates: every timed call runs in the native driver (or, for the Python
share, in bench_engine.py calling the extension exactly as the model does). Standard library
only, so it runs on any machine that can build the driver.

    bench_suite.py env     [--file F]... [--env-label TEXT] [--driver PATH]
    bench_suite.py plan    --preset smoke|window [--scratch DIR | --model DIR | --layer ...] [--scale F]
    bench_suite.py run     --preset smoke|window --out DIR (--scratch DIR [--scratch-mb N] | --model DIR
                           | --layer SPEC... --extent-list F) [--budget-gb G] [--scale F]
                           [--repeats R] [--python off|mini|full] [--only BLOCKS] [--env-label TEXT]
    bench_suite.py report  DIR [--md OUT]
    bench_suite.py compare DIR_A DIR_B [--md OUT]         (e.g. this VM against bare metal)
    bench_suite.py decide  DIR [--decode-cache 0.75] [--tolerance 0.05] [--max-foreign 1.0]
                           [--apply SRC_ROOT] [--force]

Blocks of a preset, run in this order until the read budget is spent:
  decision  every backend at its default knobs, the decode row cases, --repeats times with the
            backend order rotated (this is what `decide` ranks)
  python    bench_engine.py: the same decode rows (and cold prefill rows) through Python, per
            engine backend and the original gather, --repeats times with the order rotated (the
            n-gram route is decided from these)
  baseline  every backend at its default knobs, once: prefill rows, extents 1-8 in flight, and
            decode rows next to expert and prefill bulk reads
  python-extra  once: an empty call, warm rows with and without keyword arguments, prefill
            look-ahead, extents (the Python share of every pattern)
  keepalive decode rows with 20 ms and 200 ms of idle before each call, and the first-read
            latency after 1-500 ms of idle, with EXL3_DISK_KEEPALIVE_MS 0 and 5, --repeats times
            (`decide` sets the keep-alive's auto from these)
  sweeps    one knob at a time around each backend's defaults (threads, qd, sqpoll, register,
            align, direct, uring_async, spin_us, chunk, engram_qd, windows)

Results: DIR/env.json, DIR/native.jsonl (one object per case and run), DIR/python.jsonl,
DIR/run.log; `report` writes DIR/report.md, `decide` DIR/decision.json and DIR/decision.md.
The rule `decide` applies is documented in doc/disk_engine.md, "How auto is decided".
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import resource
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
AUTO_H = os.path.join("exllamav3", "exllamav3_ext", "disk", "disk_auto.h")
BACKENDS = ["pread", "odirect", "io_uring"]


def log(msg, fh = None):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush = True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


# ---- environment -----------------------------------------------------------------------------------

def _read(path, default = None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def block_device(path):
    """Facts about the block device under a file: name, transport, queue settings."""
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    d = {"st_dev": f"{major}:{minor}"}
    link = f"/sys/dev/block/{major}:{minor}"
    if not os.path.exists(link):
        d["note"] = "no block device (tmpfs, network or overlay file system?)"
        return d
    real = os.path.realpath(link)
    disk = os.path.dirname(real) if os.path.exists(os.path.join(real, "partition")) else real
    name = os.path.basename(disk)
    d["partition"] = os.path.basename(real) if disk != real else None
    d["device"] = name
    slaves = sorted(os.listdir(os.path.join(disk, "slaves"))) if os.path.isdir(os.path.join(disk, "slaves")) else []
    if slaves:
        d["dm_name"] = _read(os.path.join(disk, "dm", "name"))
        d["slaves"] = slaves
    under = disk if not slaves else os.path.realpath(f"/sys/class/block/{slaves[0]}")
    if slaves and os.path.exists(os.path.join(under, "partition")):
        under = os.path.dirname(under)
    q = lambda k: _read(os.path.join(disk, "queue", k))
    d.update({k: q(k) for k in ("scheduler", "nr_requests", "max_sectors_kb", "max_hw_sectors_kb",
                                "logical_block_size", "physical_block_size", "io_poll", "rotational",
                                "read_ahead_kb", "write_cache", "dma_alignment")})
    mq = os.path.join(disk, "mq")
    d["nr_hw_queues"] = len(os.listdir(mq)) if os.path.isdir(mq) else None
    d["model"] = _read(os.path.join(under, "device", "model"))
    d["vendor"] = _read(os.path.join(under, "device", "vendor"))
    d["queue_depth"] = _read(os.path.join(under, "device", "queue_depth"))
    rp = os.path.realpath(under)
    uname = os.path.basename(under)
    d["transport"] = ("nvme" if uname.startswith("nvme") else "virtio-blk" if uname.startswith("vd")
                      else "virtio-scsi" if "/virtio" in rp else "scsi/ata" if uname.startswith("sd")
                      else "other")
    return d


def fs_type(path):
    best, typ, opts = "", None, None
    path = os.path.realpath(path)
    for line in (_read("/proc/self/mountinfo", "") or "").splitlines():
        parts = line.split()
        if " - " not in line:
            continue
        mnt = parts[4]
        post = line.split(" - ", 1)[1].split()
        if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
            best, typ, opts = mnt, post[0], parts[5]
    return {"mount": best, "type": typ, "options": opts}


def virt():
    out = {}
    cpuinfo = _read("/proc/cpuinfo", "") or ""
    m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    out["cpu_model"] = m.group(1) if m else platform.processor()
    out["hypervisor_flag"] = bool(re.search(r"^flags\s*:.*\bhypervisor\b", cpuinfo, re.M))
    out["dmi"] = {k: _read(f"/sys/class/dmi/id/{k}") for k in ("sys_vendor", "product_name", "board_vendor")}
    if shutil.which("systemd-detect-virt"):
        r = subprocess.run(["systemd-detect-virt"], capture_output = True, text = True)
        out["systemd_detect_virt"] = r.stdout.strip() or r.stderr.strip()
    return out


def env_record(files, label, notes, driver):
    mem = {}
    for line in (_read("/proc/meminfo", "") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "SwapTotal", "Cached"):
            mem[k] = v.strip()
    e = {"label": label, "notes": notes, "recorded": datetime.datetime.now().isoformat(timespec = "seconds"),
         "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:8],
         "kernel": platform.release(), "machine": platform.machine(), "cpus_online": os.cpu_count(),
         "cpus_usable": len(os.sched_getaffinity(0)), "memory": mem, "virt": virt(),
         "io_uring_disabled": _read("/proc/sys/kernel/io_uring_disabled"),
         "loadavg": _read("/proc/loadavg"), "source_digest": source_digest(),
         "files": []}
    for f in files:
        try:
            bd = block_device(f)
        except Exception as ex:             # sysfs layouts vary; the run does not depend on it
            bd = {"error": str(ex)}
        e["files"].append({"path": f, "size": os.path.getsize(f), "fs": fs_type(f), "block": bd})
    if driver and os.path.exists(driver):
        cmd = [driver, "--probe"] + sum((["--file", f] for f in files), [])
        r = subprocess.run(cmd, capture_output = True, text = True)
        e["probe"] = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr.strip()}
    return e


def source_digest():
    h = hashlib.sha256()
    d = os.path.join(ROOT, "exllamav3", "exllamav3_ext", "disk")
    names = sorted(os.listdir(d)) if os.path.isdir(d) else []
    for n in names:
        with open(os.path.join(d, n), "rb") as f:
            h.update(n.encode() + b"\0" + f.read())
    for n in ("disk_engine_bench.cpp", "bench_engine.py", "bench_suite.py"):
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(n.encode() + b"\0" + f.read())
    return h.hexdigest()[:12]


def env_summary(e):
    """One line per fact, for the report's environment table."""
    rows = [("Storage path", e.get("label") or "(not given: pass --env-label)")]
    if e.get("notes"):
        rows.append(("Notes", e["notes"]))
    v = e.get("virt", {})
    dmi = v.get("dmi", {})
    virt_s = v.get("systemd_detect_virt") or ("hypervisor flag set" if v.get("hypervisor_flag") else "bare metal (no hypervisor flag)")
    rows.append(("Machine", f"{v.get('cpu_model')}, {e.get('cpus_online')} CPUs online "
                            f"({e.get('cpus_usable')} usable), {e.get('memory', {}).get('MemTotal', '?')} RAM; "
                            f"virtualization: {virt_s}; {dmi.get('sys_vendor') or ''} {dmi.get('product_name') or ''}".rstrip()))
    rows.append(("Kernel", f"{e.get('kernel')} ({e.get('machine')}), io_uring_disabled={e.get('io_uring_disabled')}"))
    p = e.get("probe", {})
    for f in e.get("files", []):
        b, fs = f.get("block", {}), f.get("fs", {})
        pf = next((x for x in p.get("files", []) if x.get("path") == f["path"]), {})
        dev = (f"{b.get('device')} \"{(b.get('vendor') or '').strip()} {(b.get('model') or '').strip()}\" "
               f"({b.get('transport')}), {b.get('nr_hw_queues')} hw queues, scheduler "
               f"{(b.get('scheduler') or '').strip()}, nr_requests {b.get('nr_requests')}, queue_depth "
               f"{b.get('queue_depth')}, max_sectors_kb {b.get('max_sectors_kb')}, logical block "
               f"{b.get('logical_block_size')} B, io_poll {b.get('io_poll')}, rotational {b.get('rotational')}")
        if b.get("slaves"):
            dev = f"{b.get('dm_name')} over {','.join(b['slaves'])}; " + dev
        rows.append((f"File {os.path.basename(f['path'])}",
                     f"{f['size'] / 2**30:.1f} GiB on {fs.get('type')} ({fs.get('mount')}); O_DIRECT "
                     f"{pf.get('o_direct', '?')}, DIO align mem {pf.get('dio_mem_align', '?')} / offset "
                     f"{pf.get('dio_offset_align', '?')}; residency via {pf.get('residency_method', '?')}; "
                     f"device {dev}"))
    rows.append(("io_uring", f"{p.get('io_uring', '?')}"))
    for x in p.get("engines", []):
        rows.append((f"Engine {x.get('backend')}", x.get("describe") or x.get("error")))
    rows.append(("Auto (disk_auto.h)", f"backend {p.get('auto_backend', '?')}, n-gram route {p.get('auto_ngram_route', '?')}"))
    rows.append(("Source digest", e.get("source_digest")))
    return rows


# ---- data sources ----------------------------------------------------------------------------------

def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 512 << 20:
            raise ValueError(f"{path}: implausible safetensors header of {n} bytes")
        h = json.loads(f.read(n))
    return 8 + n, h


def model_sources(model_dir, max_extent_files = 0):
    """Engram layers (row table + scale table per layer) and expert extents (every routed expert's
    contiguous span) of a DeepSeek-V4.1 EXL3 checkpoint, from the safetensors headers only."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        wmap = json.load(f)["weight_map"]
    headers = {}

    def hdr(fn):
        if fn not in headers:
            headers[fn] = st_header(os.path.join(model_dir, fn))
        return headers[fn]

    layers = []
    for k in sorted(k for k in wmap if k.endswith(".engram.embed.weight")):
        pre = k[: -len(".weight")]
        tabs = []
        for t, rb in ((pre + ".weight", None), (pre + ".scale", None)):
            if t not in wmap:
                break
            fn = wmap[t]
            base, h = hdr(fn)
            m = h[t]
            shape = m["shape"]
            off0, off1 = m["data_offsets"]
            row_bytes = (off1 - off0) // shape[0]
            tabs.append((os.path.join(model_dir, fn), base + off0, row_bytes, shape[0]))
        if len(tabs) == 2:
            layers.append(tabs)
    groups = {}
    pat = re.compile(r"^(.*\.experts\.\d+)\.")
    for k, fn in wmap.items():
        m = pat.match(k)
        if m:
            groups.setdefault((fn, m.group(1)), []).append(k)
    extents, bad = [], 0
    files = sorted({fn for fn, _ in groups})
    if max_extent_files:
        files = files[:max_extent_files]
    for fn in files:
        base, h = hdr(fn)
        for (f2, g), keys in groups.items():
            if f2 != fn:
                continue
            spans = sorted(h[k]["data_offsets"] for k in keys)
            lo, hi = spans[0][0], spans[-1][1]
            if sum(b - a for a, b in spans) != hi - lo:
                bad += 1
                continue
            extents.append((os.path.join(model_dir, fn), base + lo, hi - lo))
    return layers, extents, bad


def layer_arg(tabs):
    return "+".join(f"{p}:{o}:{rb}:{n}" for p, o, rb, n in tabs)


# ---- presets ---------------------------------------------------------------------------------------

def _it(n, scale, lo = 1):
    return max(lo, int(round(n * scale)))


def preset(name, scale, h):
    """Blocks: list of dicts with name, kind (native|python), repeats, points [(backend, knobs,
    cases)]. `h` is the warm fraction of the decode-hit case."""
    S = lambda n, lo = 1: _it(n, scale, lo)
    if name == "smoke":
        dec = [f"rows:u=48,cache=cold,iters={S(60)}", f"rows:u=48,cache={h},iters={S(60)}",
               f"rows:u=48,cache=warm,iters={S(60)}", f"rows:u=96,cache=cold,iters={S(30)}"]
        base = ["null:iters=500", f"rows:u=8192,cache=cold,iters={S(2)}", f"rows:u=8192,cache=warm,iters={S(2)}",
                f"extents:k=1,cache=cold,iters={S(4)}", f"extents:k=4,cache=cold,iters={S(2)}",
                f"mixed:bulk=none,iters={S(15)}", f"mixed:bulk=extents,bulk_cls=1,iters={S(15)}",
                f"mixed:bulk=extents,bulk_cls=2,iters={S(15)}", f"mixed:bulk=rows,bulk_cls=2,iters={S(15)}"]
        py = [f"engram:u=48,cache=cold,iters={S(40)}", f"engram:u=48,cache={h},iters={S(40)}",
              f"engram:u=8192,cache=cold,iters={S(2)}"]
        py_extra = ["null:iters=500", f"engram:u=48,cache=warm,iters={S(40)}", f"keywords:u=48,cache=warm,iters={S(40)}",
                    f"prefetch:u=8192,overlap_ms=20,cache=cold,iters={S(2)}", f"extents:k=4,cache=cold,iters={S(2)}"]
        ka = [f"rows:u=48,cache={h},gap_us=20000,iters={S(20)}", f"rows:u=48,cache={h},gap_us=200000,iters={S(4)}",
              f"wake:gaps=1/10/50/200,iters={S(2)}"]
        rows_s = [f"rows:u=48,cache=cold,iters={S(40)}"]
        ext_s = [f"extents:k=4,cache=cold,iters={S(2)}"]
        sweeps = [
            ("threads", ["pread"], "threads", ["8"], rows_s + ext_s),
            ("qd", ["io_uring"], "qd", ["32"], rows_s + ext_s),
            ("sqpoll", ["io_uring"], "sqpoll", ["1"], rows_s + ext_s),
            ("register", ["io_uring"], "register", ["none"], rows_s + ext_s),
            ("align", ["odirect"], "align", ["4096"], rows_s + ext_s),
            ("direct", ["io_uring"], "direct", ["all"], rows_s + ext_s),
            ("uring_async", ["io_uring"], "uring_async", ["64"], [f"rows:u=8192,cache=warm,iters={S(2)}"]),
            ("spin_us", ["io_uring"], "spin_us", ["0"], rows_s),
            ("chunk", ["io_uring"], "chunk", ["512K"], ext_s),
            ("window_expert", ["io_uring"], "window_expert", ["8M"], [f"mixed:bulk=extents,bulk_cls=1,iters={S(15)}"]),
            ("window_prefetch", ["io_uring"], "window_prefetch", ["32M/0"], [f"mixed:bulk=extents,bulk_cls=2,iters={S(15)}"]),
        ]
        reps = 2
    elif name == "window":
        dec = [f"rows:u=48,cache=cold,iters={S(1000)}", f"rows:u=48,cache={h},iters={S(1000)}",
               f"rows:u=48,cache=warm,iters={S(1000)}", f"rows:u=96,cache=cold,iters={S(500)}"]
        base = ["null:iters=5000", f"rows:u=96,cache=warm,iters={S(500)}",
                f"rows:u=8192,cache=cold,iters={S(10, 3)}", f"rows:u=8192,cache=warm,iters={S(10, 3)}",
                f"extents:k=1,cache=cold,iters={S(40, 3)}", f"extents:k=2,cache=cold,iters={S(20, 3)}",
                f"extents:k=4,cache=cold,iters={S(10, 3)}", f"extents:k=8,cache=cold,iters={S(6, 3)}",
                f"mixed:bulk=none,iters={S(300)}", f"mixed:bulk=extents,bulk_cls=1,iters={S(100)}",
                f"mixed:bulk=extents,bulk_cls=2,iters={S(100)}", f"mixed:bulk=rows,bulk_cls=2,iters={S(100)}"]
        py = [f"engram:u=48,cache=cold,iters={S(300)}", f"engram:u=48,cache={h},iters={S(300)}",
              f"engram:u=8192,cache=cold,iters={S(8, 3)}"]
        py_extra = ["null:iters=5000", f"engram:u=48,cache=warm,iters={S(300)}", f"keywords:u=48,cache=warm,iters={S(300)}",
                    f"engram:u=96,cache=cold,iters={S(200)}", f"engram:u=8192,cache=warm,iters={S(8, 3)}",
                    f"prefetch:u=8192,overlap_ms=20,cache=cold,iters={S(8, 3)}", f"extents:k=4,cache=cold,iters={S(5, 3)}"]
        ka = [f"rows:u=48,cache={h},gap_us=20000,iters={S(200)}", f"rows:u=48,cache={h},gap_us=200000,iters={S(20)}",
              f"wake:gaps=1/5/10/20/50/100/200/500,iters={S(6, 2)}"]
        rows_s = [f"rows:u=48,cache=cold,iters={S(500)}"]
        pref_s = [f"rows:u=8192,cache=cold,iters={S(5, 2)}"]
        ext_s = [f"extents:k=4,cache=cold,iters={S(6, 2)}"]
        warm_s = [f"rows:u=48,cache=warm,iters={S(500)}"]
        sweeps = [
            ("threads", ["pread", "odirect"], "threads", ["8", "48", "96"], rows_s + pref_s + ext_s),
            ("qd", ["io_uring"], "qd", ["32", "64", "256"], rows_s + pref_s + ext_s),
            ("sqpoll", ["io_uring"], "sqpoll", ["1"], rows_s + warm_s + pref_s + ext_s),
            ("register", ["io_uring"], "register", ["none", "files", "buffers"], rows_s + ext_s),
            ("align", ["odirect", "io_uring"], "align", ["4096"], rows_s + ext_s),
            ("direct", ["io_uring"], "direct", ["all", "none"], rows_s + warm_s + ext_s),
            ("uring_async", ["io_uring"], "uring_async", ["64"], [f"rows:u=8192,cache=warm,iters={S(10, 3)}"] + warm_s),
            ("spin_us", ["io_uring", "pread"], "spin_us", ["0", "200"], rows_s + warm_s),
            ("chunk", ["io_uring", "odirect"], "chunk", ["512K", "4M"], ext_s + [f"extents:k=1,cache=cold,iters={S(20, 3)}"]),
            ("engram_qd", ["io_uring"], "engram_qd", ["32", "64"], pref_s + [f"rows:u=96,cache=cold,iters={S(300)}"]),
            ("window_expert", ["io_uring"], "window_expert", ["8M", "64M"],
             [f"mixed:bulk=extents,bulk_cls=1,iters={S(100)}"]),
            ("window_prefetch", ["io_uring"], "window_prefetch", ["32M/0"],
             [f"mixed:bulk=extents,bulk_cls=2,iters={S(100)}"]),
        ]
        reps = 3
    else:
        raise SystemExit(f"unknown preset {name} (smoke, window)")
    blocks = [{"name": "decision", "kind": "native", "repeats": reps,
               "points": [(b, {}, dec) for b in BACKENDS], "sweep": "defaults"},
              {"name": "python", "kind": "python", "repeats": reps,
               "variants": BACKENDS + ["original"], "cases": py},
              # decode with idle gaps between calls, without and with the keep-alive (its auto)
              {"name": "keepalive", "kind": "native", "repeats": reps, "sweep": "keepalive_ms",
               "points": [(b, {"keepalive_ms": v}, ka) for b in BACKENDS for v in ("0", "5")]},
              {"name": "baseline", "kind": "native", "repeats": 1,
               "points": [(b, {}, base) for b in BACKENDS], "sweep": "defaults"}]
    blocks.append({"name": "python-extra", "kind": "python", "repeats": 1,
                   "variants": BACKENDS + ["original"], "cases": py_extra})
    for sname, bes, knob, values, cases in sweeps:
        blocks.append({"name": f"sweep:{sname}", "kind": "native", "repeats": 1, "sweep": sname,
                       "points": [(b, {knob: v}, cases) for b in bes for v in values]})
    return blocks


def parse_case(spec):
    kind, _, rest = spec.partition(":")
    kv = dict(x.split("=", 1) for x in rest.split(",") if x)
    return kind, kv


def case_key(spec):
    """The case without its iteration count: what two runs must share to be compared."""
    kind, kv = parse_case(spec)
    kv.pop("iters", None)
    return kind + (":" + ",".join(f"{k}={kv[k]}" for k in sorted(kv)) if kv else "")


def estimate_bytes(spec, tables = 2, extent_bytes = 13315596, page = 4096):
    """Upper estimate of what a case reads from the device, preparation included."""
    kind, kv = parse_case(spec)
    it = int(kv.get("iters", 1))
    if kind in ("rows", "engram", "keywords"):
        return it * int(kv.get("u", 48)) * tables * page
    if kind == "prefetch":
        return it * int(kv.get("u", 8192)) * tables * page
    if kind == "extents":
        return it * int(kv.get("k", 4)) * extent_bytes
    if kind == "mixed":
        secs = it * int(kv.get("period_us", 2000)) / 1e6 + 0.02
        dec = it * int(kv.get("u", 48)) * tables * page
        bulk = kv.get("bulk", "extents")
        if bulk == "extents":
            return dec + int(secs * (12e9 if kv.get("bulk_cls", "1") == "1" else 8e9))
        if bulk == "rows":
            return dec + int(secs / 0.02 + 1) * int(kv.get("bulk_u", 8192)) * tables * page
        return dec
    return 0


# ---- running -------------------------------------------------------------------------------------

def children_read_bytes():
    # Linux reports ru_inblock of waited-for children as their read_bytes / 512
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_inblock * 512


class Runner:
    def __init__(self, a, out, logf):
        self.a, self.out, self.logf = a, out, logf
        self.budget = a.budget_gb * 1e9 if a.budget_gb > 0 else float("inf")
        self.used = 0.0
        self.native = open(os.path.join(out, "native.jsonl"), "a")
        self.python = open(os.path.join(out, "python.jsonl"), "a")

    def remaining(self):
        return self.budget - self.used

    def run_cmd(self, cmd, env = None, timeout = None):
        r0 = children_read_bytes()
        t0 = time.monotonic()
        log("$ " + " ".join(shlex.quote(c) for c in cmd), self.logf)
        p = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = timeout)
        read = children_read_bytes() - r0
        self.used += read
        if self.logf:
            self.logf.write(p.stdout)
            self.logf.write(p.stderr)
            self.logf.flush()
        for line in p.stdout.splitlines():
            if not line.startswith("--"):
                print("   " + line, flush = True)
        if p.returncode != 0:
            log(f"exit {p.returncode}: {p.stderr.strip()[-2000:]}", self.logf)
        log(f"   {time.monotonic() - t0:.1f} s, read {read / 1e9:.2f} GB, budget used "
            f"{self.used / 1e9:.2f} of {self.budget / 1e9:.1f} GB", self.logf)
        return p

    def native_point(self, block, sweep, backend, knobs, cases, rep, order, src):
        est = sum(estimate_bytes(c) for c in cases)
        if est > self.remaining():
            # run what fits: the driver stops at --max-read-gb, later cases are marked skipped
            log(f"   {block} {backend} {knobs}: estimated {est / 1e9:.2f} GB, {self.remaining() / 1e9:.2f} GB left",
                self.logf)
        if self.remaining() <= 0:
            return False
        tmp = os.path.join(self.out, ".point.jsonl")
        if os.path.exists(tmp):
            os.unlink(tmp)
        left = self.remaining()
        cmd = [self.a.driver, "--out", tmp, "--label", f"{block}/{backend}/{rep}",
               "--seed", str(1000 * rep + order + 1), "--set", f"backend={backend}",
               # 0 = no limit; otherwise what is left of the run's budget (at least 1 MB)
               "--max-read-gb", "0" if left == float("inf") else f"{max(left, 1e6) / 1e9:.6f}",
               "--preserve-cache", "1" if src["preserve"] else "0",
               "--idle-sample-ms", str(self.a.idle_sample_ms)]
        for k, v in knobs.items():
            cmd += ["--set", f"{k}={v}"]
        cmd += src["args"]
        for c in cases:
            cmd += ["--case", c]
        env = dict(os.environ)
        for k in list(env):
            if k.startswith("EXL3_DISK_"):
                del env[k]                      # the knobs of this point only
        env["CUDA_VISIBLE_DEVICES"] = ""
        p = self.run_cmd(self.a.nice + cmd, env = env)
        n = 0
        if os.path.exists(tmp):
            with open(tmp) as f:
                for line in f:
                    r = json.loads(line)
                    r.update({"block": block, "sweep": sweep, "backend": backend, "knobs": knobs,
                              "repeat": rep, "order": order, "case_key": case_key(r["case"]),
                              "exit": p.returncode})
                    self.native.write(json.dumps(r) + "\n")
                    n += 1
            os.unlink(tmp)
            self.native.flush()
        if p.returncode != 0 and n == 0:
            self.native.write(json.dumps({"block": block, "sweep": sweep, "backend": backend,
                                          "knobs": knobs, "repeat": rep, "failed": p.stderr.strip()[-2000:]}) + "\n")
            self.native.flush()
        return True

    def python_run(self, block, variants, cases, rep, src):
        est = sum(estimate_bytes(c) for c in cases) * len(variants)
        if est > self.remaining():
            log(f"   python: estimated {est / 1e9:.2f} GB but {self.remaining() / 1e9:.2f} GB left: skipped", self.logf)
            return False
        tmp = os.path.join(self.out, ".python.json")
        if os.path.exists(tmp):
            os.unlink(tmp)
        cmd = [self.a.python_exe, os.path.join(HERE, "bench_engine.py"), "--json", tmp,
               "--variants", ",".join(variants), "--seed", str(rep),
               "--preserve-cache", "1" if src["preserve"] else "0"] + src["py_args"]
        for c in cases:
            cmd += ["--case", c]
        env = dict(os.environ)
        for k in list(env):
            if k.startswith("EXL3_DISK_") and k not in ("EXL3_DISK_MINI_BUILD_DIR",):
                del env[k]
        env["CUDA_VISIBLE_DEVICES"] = ""
        if self.a.python == "mini":
            env["EXL3_DISK_TEST_MINI"] = "1"
        p = self.run_cmd(self.a.nice + cmd, env = env, timeout = 7200)
        if os.path.exists(tmp):
            with open(tmp) as f:
                for r in json.load(f):
                    r.update({"block": block, "repeat": rep, "case_key": case_key(r["case"]),
                              "order": variants.index(r["variant"])})
                    self.python.write(json.dumps(r) + "\n")
            os.unlink(tmp)
            self.python.flush()
        elif p.returncode != 0:
            self.python.write(json.dumps({"block": block, "repeat": rep, "failed": p.stderr.strip()[-2000:]}) + "\n")
            self.python.flush()
        return True


def sources(a):
    """Driver and bench_engine.py arguments for the data source; creates the scratch file."""
    if a.scratch:
        os.makedirs(a.scratch, exist_ok = True)
        path = os.path.join(a.scratch, "bench_suite_scratch.bin")
        size = a.scratch_mb << 20
        assert a.scratch_mb <= 2048, "scratch files are limited to 2 GiB here (--scratch-mb)"
        st = os.statvfs(a.scratch)
        if st.f_bavail * st.f_frsize < size + (512 << 20):
            raise SystemExit(f"{a.scratch}: not enough free space for a {a.scratch_mb} MiB scratch file")
        with open(path, "wb") as f:
            for _ in range(a.scratch_mb // 4):
                f.write(os.urandom(4 << 20))
            f.flush()
            os.fsync(f.fileno())
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        s_off = (size * 9 // 10) // 4096 * 4096 + 2712
        rows = min((s_off - 664) // 256, (size - s_off) // 8)
        layers = [[(path, 664, 256, rows), (path, s_off, 8, rows)]]
        args = ["--layer", layer_arg(layers[0]), "--extent-region", f"{path}:0:{size}"]
        return {"files": [path], "owned": path, "preserve": False, "args": args,
                "py_args": args, "layers": layers, "extents": None}
    if a.model:
        layers, extents, bad = model_sources(a.model)
        if not layers:
            raise SystemExit(f"{a.model}: no engram tables found (pass --layer)")
        lst = os.path.join(a.out, "extents.txt") if getattr(a, "out", None) else None
        if lst:
            with open(lst, "w") as f:
                f.write(f"# {len(extents)} expert extents of {a.model} ({bad} non-contiguous skipped)\n")
                for p, o, n in extents:
                    f.write(f"{p} {o} {n}\n")
        args = sum((["--layer", layer_arg(L)] for L in layers), [])
        if lst:
            args += ["--extent-list", lst]
        files = sorted({t[0] for L in layers for t in L} | {e[0] for e in extents[:1]})
        return {"files": files, "owned": None, "preserve": True, "args": args, "py_args": args,
                "layers": layers, "extents": len(extents)}
    if a.layer:
        args = sum((["--layer", s] for s in a.layer), [])
        if a.extent_list:
            args += ["--extent-list", a.extent_list]
        files = sorted({t.rsplit(":", 3)[0] for s in a.layer for t in s.split("+")})
        return {"files": files, "owned": None, "preserve": True, "args": args, "py_args": args,
                "layers": None, "extents": None}
    raise SystemExit("one of --scratch, --model or --layer")


def cmd_run(a):
    os.makedirs(a.out, exist_ok = True)
    if os.path.exists(os.path.join(a.out, "native.jsonl")) and not a.append:
        raise SystemExit(f"{a.out} already holds results (--append to add to them)")
    logf = open(os.path.join(a.out, "run.log"), "a")
    a.nice = ["nice", "-n", str(a.nice_level)] if a.nice_level else []
    src = sources(a)
    try:
        e = env_record(src["files"], a.env_label, a.env_notes, a.driver)
        e.update({"preset": a.preset, "scale": a.scale, "budget_gb": a.budget_gb,
                  "decode_cache": a.decode_cache, "source": {k: src[k] for k in ("owned", "preserve", "extents")}})
        with open(os.path.join(a.out, "env.json"), "w") as f:
            json.dump(e, f, indent = 1)
        for k, v in env_summary(e):
            log(f"{k}: {v}", logf)
        blocks = preset(a.preset, a.scale, a.decode_cache)
        if a.repeats:
            for b in blocks:
                if b["name"] in ("decision", "python"):
                    b["repeats"] = a.repeats
        only = [x for x in a.only.split(",") if x] if a.only else []
        r = Runner(a, a.out, logf)
        for b in blocks:
            if only and not any(b["name"] == o or b["name"].startswith(o + ":") or
                                (o == "sweeps" and b["name"].startswith("sweep:")) for o in only):
                continue
            if b["kind"] == "python" and a.python == "off":
                continue
            log(f"== {b['name']} ({b['repeats']} run(s))", logf)
            for rep in range(b["repeats"]):
                if b["kind"] == "python":
                    k = rep % len(b["variants"])
                    variants = b["variants"][k:] + b["variants"][:k]        # rotated order
                    r.python_run(b["name"], variants, b["cases"], rep, src)
                    continue
                pts = b["points"]
                k = rep % len(pts)
                for order, (backend, knobs, cases) in enumerate(pts[k:] + pts[:k]):
                    if r.remaining() <= 0:
                        break
                    r.native_point(b["name"], b["sweep"], backend, knobs, cases, rep, order, src)
            if r.remaining() <= 0:
                log(f"read budget of {a.budget_gb} GB spent: remaining blocks skipped", logf)
                break
        log(f"done: {r.used / 1e9:.2f} GB read by the benchmarks", logf)
    finally:
        if src.get("owned") and os.path.exists(src["owned"]):
            os.unlink(src["owned"])
            log(f"removed {src['owned']}", logf)
    write_report(a.out)


def cmd_plan(a):
    blocks = preset(a.preset, a.scale, a.decode_cache)
    total = 0
    for b in blocks:
        if b["kind"] == "python":
            est = sum(estimate_bytes(c) for c in b["cases"]) * len(b["variants"]) * b["repeats"]
            print(f"{b['name']:<24} python x{b['repeats']}: {len(b['cases'])} cases x {len(b['variants'])} variants, ~{est / 1e9:.1f} GB")
        else:
            est = sum(estimate_bytes(c) for _, _, cs in b["points"] for c in cs) * b["repeats"]
            print(f"{b['name']:<24} native x{b['repeats']}: {len(b['points'])} points, ~{est / 1e9:.1f} GB")
            if a.verbose:
                for be, kn, cs in b["points"]:
                    print(f"    {be} {kn}: " + " ".join(cs))
        total += est
    print(f"total ~{total / 1e9:.1f} GB read (upper estimate; --budget-gb caps it)")


# ---- loading and aggregating results ---------------------------------------------------------------

def load(d):
    nat, py = [], []
    p = os.path.join(d, "native.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            nat = [json.loads(x) for x in f if x.strip()]
    p = os.path.join(d, "python.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            py = [json.loads(x) for x in f if x.strip()]
    env = {}
    p = os.path.join(d, "env.json")
    if os.path.exists(p):
        with open(p) as f:
            env = json.load(f)
    return env, nat, py


def g(r, *path, default = None):
    for k in path:
        if not isinstance(r, dict) or k not in r:
            return default
        r = r[k]
    return r


def knobs_s(k):
    return ",".join(f"{a}={b}" for a, b in sorted((k or {}).items())) or "defaults"


def valid(r, max_foreign = None):
    """Why a native or python record cannot be used, or None."""
    if r.get("failed"):
        return "run failed"
    if r.get("skipped"):
        return r["skipped"]
    if r.get("errors"):
        return f"read error: {r.get('error')}"
    if r.get("verify_failures"):
        return f"{r['verify_failures']} verify failures"
    if r.get("truncated"):
        return "truncated by the read budget"
    c = r.get("cache") or {}
    if c.get("off_target_pages") is not None and c.get("pages_per_call"):
        off = c["off_target_pages"] / max(1.0, c["pages_per_call"] * max(1, r.get("calls", 1)))
        if off > 0.02:
            return f"cache state off target ({off:.1%} of pages)"
    if max_foreign is not None:
        fb = g(r, "host", "foreign_cores_before")
        if fb is not None and fb > max_foreign:
            return f"host busy ({fb:.1f} other cores)"
    return None


def agg(rs, *path):
    v = [g(r, *path) for r in rs]
    v = [x for x in v if isinstance(x, (int, float))]
    if not v:
        return None, None, None
    return statistics.median(v), min(v), max(v)


def group(nat, block_pred):
    out = {}
    for r in nat:
        if "case" not in r or not block_pred(r.get("block", "")):
            continue
        k = (r.get("block"), r.get("backend"), knobs_s(r.get("knobs")), r["case_key"])
        out.setdefault(k, []).append(r)
    return out


def fmt(v, digits = 3):
    if v is None:
        return "-"
    if isinstance(v, float) and v != v:
        return "-"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.{digits - 1}f}M"
    if a >= 1e4:
        return f"{v / 1e3:.{max(0, digits - 2)}f}k"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits - 1}f}"


def fmt_spread(t):
    m, lo, hi = t
    if m is None:
        return "-"
    if lo is None or hi is None or lo == hi:
        return fmt(m)
    return f"{fmt(m)} ({fmt(lo)}-{fmt(hi)})"


def op_lat(rs, cls, q):
    vals = []
    for r in rs:
        for c in g(r, "engine", "classes", default = []) or []:
            if c.get("class") == cls and g(c, "op_service_us", q) is not None:
                vals.append(c["op_service_us"][q])
    return statistics.median(vals) if vals else None


def read_amp(rs):
    v = [r["device_read_bytes"] / r["payload_bytes"] for r in rs
         if isinstance(r.get("device_read_bytes"), (int, float)) and r.get("payload_bytes")]
    return statistics.median(v) if v else None


# ---- report ----------------------------------------------------------------------------------------

def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def cache_s(rs):
    c = rs[0].get("cache") or {}
    got = agg(rs, "cache", "resident_fraction")[0]
    mode = c.get("mode", "?")
    if got is None:
        return mode
    return f"{mode} ({got:.0%} resident)"


def py_share_for(py, backend, key):
    rs = [r for r in py if r.get("variant") == backend and r.get("case_key") == key and not valid(r)]
    if not rs:
        return None, None
    return agg(rs, "python_us", "p50")[0], agg(rs, "python_share")[0]


def python_key(native_key):
    kind, kv = parse_case(native_key)
    if kind != "rows":
        return None
    kv.pop("cls", None)
    kv.pop("hold", None)
    return "engram:" + ",".join(f"{k}={kv[k]}" for k in sorted(kv))


def report_md(d):
    env, nat, py = load(d)
    L = [f"# Disk engine benchmark: {os.path.basename(os.path.abspath(d))}", ""]
    L.append(f"Preset `{env.get('preset')}`, scale {env.get('scale')}, read budget {env.get('budget_gb')} GB, "
             f"recorded {env.get('recorded')}. Latencies in microseconds (ms where marked); for repeated "
             f"cases the median of the runs, with the range in parentheses. p99.9 needs ~1000 calls to mean "
             f"anything; `n` is the calls per run.")
    L += ["", "## Environment", "", md_table(["", "This run"], env_summary(env))]
    wake = [g(r, "host", "first_read_after_idle_us") for r in nat]
    wake = sorted(x for x in wake if isinstance(x, (int, float)))
    if wake:
        L += ["", f"First read after {g(nat[0], 'host', 'idle_ms')} ms of idle (one cold row or small extent "
              f"per file, before each case, untimed in the case): median {fmt(statistics.median(wake))} us, "
              f"max {fmt(wake[-1])} us over {len(wake)} cases. A device or host that sleeps when idle "
              f"shows it here, not in the cases."]
    dec = os.path.join(d, "decision.md")
    if os.path.exists(dec):
        with open(dec) as f:
            L += ["", f.read().strip()]

    def rows_table(title, pred, note = None):
        gr = group(nat, pred)
        items = [(k, v) for k, v in gr.items() if parse_case(k[3])[0] == "rows"]
        if not items:
            return
        L.extend(["", f"## {title}", ""])
        if note:
            L.extend([note, ""])
        head = ["Backend", "Knobs", "Case", "Cache", "n", "p50", "p90", "p99", "p99.9", "Rows/s", "IOPS",
                "Op p50 / p99", "Read amp.", "CPU cores", "CPU us/op", "Py us", "Py %", "Other cores", "Valid"]
        rows = []
        for (blk, be, kn, key), rs in sorted(items, key = lambda x: (x[0][3], x[0][1], x[0][2])):
            bad = [valid(r) for r in rs if valid(r)]
            pk = python_key(key)
            pus, psh = py_share_for(py, be, pk) if kn == "defaults" and pk else (None, None)
            rows.append([be, kn, key.split(":", 1)[1].replace("cache=", "").replace(",", " "), cache_s(rs),
                         rs[0].get("calls"), fmt_spread(agg(rs, "lat_us", "p50")), fmt(agg(rs, "lat_us", "p90")[0]),
                         fmt_spread(agg(rs, "lat_us", "p99")), fmt(agg(rs, "lat_us", "p99.9")[0]),
                         fmt(agg(rs, "rows_per_s")[0]), fmt(agg(rs, "iops")[0]),
                         f"{fmt(op_lat(rs, 0, 'p50'))} / {fmt(op_lat(rs, 0, 'p99'))}", fmt(read_amp(rs)),
                         fmt(agg(rs, "cpu", "cores")[0]), fmt(agg(rs, "cpu", "us_per_op")[0]),
                         fmt(pus), f"{psh * 100:.1f}" if psh is not None else "-",
                         fmt(agg(rs, "host", "foreign_cores_before")[0]), "yes" if not bad else "; ".join(sorted(set(bad)))])
        L.append(md_table(head, rows))

    rows_table("Row reads: decode (decision block)", lambda b: b == "decision",
               "One call = u unique ids of one engram layer, both tables (256-byte rows and 8-byte scales) "
               "in one engine call, class 0 with hold, as the DeepSeek-V4.1 engram reads at decode. Cache: "
               "the state of exactly the pages the call reads (resident share measured). IOPS = device "
               "reads per second of call time. Op = one device read's service time (engine histogram). "
               "Read amp. = bytes read from the device / payload bytes. Py = the Python side of the same "
               "call through the extension (bench_engine.py).")
    rows_table("Row reads: other sizes (baseline block)", lambda b: b == "baseline")

    gr = group(nat, lambda b: b == "baseline")
    ex = [(k, v) for k, v in gr.items() if parse_case(k[3])[0] == "extents"]
    if ex:
        L += ["", "## Expert extents (baseline block)", "",
              "One call = k extents of one routed expert each (13,315,596 bytes; real extents from the "
              "checkpoint, or random unaligned offsets on a scratch file), all in flight at once, class 1. "
              "Latency in ms.", ""]
        rows = []
        for (blk, be, kn, key), rs in sorted(ex, key = lambda x: (int(parse_case(x[0][3])[1].get("k", 0)), x[0][1])):
            rows.append([be, kn, parse_case(key)[1].get("k"), rs[0].get("calls"),
                         fmt((agg(rs, "lat_us", "p50")[0] or 0) / 1e3), fmt((agg(rs, "lat_us", "p99")[0] or 0) / 1e3),
                         fmt(agg(rs, "gb_per_s")[0]), fmt(agg(rs, "iops")[0]),
                         f"{fmt(op_lat(rs, 1, 'p50'))} / {fmt(op_lat(rs, 1, 'p99'))}", fmt(read_amp(rs)),
                         fmt(agg(rs, "cpu", "us_per_gb")[0]), "yes" if g(rs[0], "params", "registered") else "no",
                         fmt(agg(rs, "host", "foreign_cores_before")[0]),
                         "yes" if not any(valid(r) for r in rs) else "; ".join(sorted({valid(r) for r in rs if valid(r)}))])
        L.append(md_table(["Backend", "Knobs", "k", "n", "p50 ms", "p99 ms", "GB/s", "IOPS", "Op p50 / p99 us",
                           "Read amp.", "CPU us/GB", "Registered", "Other cores", "Valid"], rows))

    mx = [(k, v) for k, v in gr.items() if parse_case(k[3])[0] == "mixed"]
    if mx:
        L += ["", "## Decode rows next to bulk reads (baseline block)", "",
              "A 48-row decode batch (class 0, hold, armed 0.5 ms ahead as at sampling) every 2 ms, cold, "
              "while one thread streams expert extents 4 deep (class 1 demand: never held, 32 MiB window; "
              "class 2 prefetch: held while a decode batch is out) or makes 8192-row class-2 gathers "
              "(prefill staging). `vs alone` = decode p50 against the same backend without bulk.", ""]
        alone = {}
        for (blk, be, kn, key), rs in mx:
            if parse_case(key)[1].get("bulk") == "none":
                alone[(be, kn)] = agg(rs, "lat_us", "p50")[0]
        rows = []
        for (blk, be, kn, key), rs in sorted(mx, key = lambda x: (x[0][1], x[0][3])):
            kv = parse_case(key)[1]
            bulk = kv.get("bulk")
            desc = "none" if bulk == "none" else f"{bulk} cls {kv.get('bulk_cls')}"
            p50 = agg(rs, "lat_us", "p50")[0]
            a0 = alone.get((be, kn))
            held = sum(c.get("held", 0) for r in rs for c in (g(r, "engine", "classes", default = []) or []) if c.get("class", 0) >= 2)
            rows.append([be, kn, desc, rs[0].get("calls"), fmt(p50), fmt(agg(rs, "lat_us", "p90")[0]),
                         fmt(agg(rs, "lat_us", "p99")[0]), fmt(agg(rs, "lat_us", "p99.9")[0]),
                         f"{p50 / a0:.2f}x" if p50 and a0 and bulk != "none" else "-",
                         fmt(agg(rs, "bulk_gb_per_s")[0]) if bulk == "extents" else
                         (fmt(agg(rs, "bulk_rows_per_s")[0]) + " rows/s" if bulk == "rows" else "-"),
                         held, fmt(agg(rs, "cpu", "cores")[0]), fmt(agg(rs, "host", "foreign_cores_before")[0]),
                         "yes" if not any(valid(r) for r in rs) else "; ".join(sorted({valid(r) for r in rs if valid(r)}))])
        L.append(md_table(["Backend", "Knobs", "Bulk", "n", "Decode p50", "p90", "p99", "p99.9", "vs alone",
                           "Bulk rate", "Held admissions", "CPU cores", "Other cores", "Valid"], rows))

    kg = group(nat, lambda b: b == "keepalive")
    gap = [(k, v) for k, v in kg.items() if parse_case(k[3])[0] == "rows"]
    if gap:
        L += ["", "## Idle gaps and the keep-alive (keepalive block)", "",
              "Decode rows (48 ids, both tables) after the device sat idle for `gap_us` before each call, "
              "as between tokens (20 ms) or before the first token of a request (200 ms), with "
              "`EXL3_DISK_KEEPALIVE_MS` off and at 5 ms. Keep-alive reads/s = the small O_DIRECT reads it "
              "issued per second of case time.", ""]
        rows = []
        for (blk, be, kn, key), rs in sorted(gap, key = lambda x: (x[0][3], x[0][1], x[0][2])):
            kr = [g(r, "engine", "keepalive_reads") for r in rs]
            wall = [g(r, "host", "case_wall_s") for r in rs]
            rate = (sum(x for x in kr if x) / sum(w for w in wall if w)) if any(wall) else None
            kv = parse_case(key)[1]
            rows.append([be, kn, f"{int(kv.get('gap_us', 0)) / 1000:g} ms", kv.get("cache"), rs[0].get("calls"),
                         fmt_spread(agg(rs, "lat_us", "p50")), fmt(agg(rs, "lat_us", "p90")[0]),
                         fmt(agg(rs, "lat_us", "p99")[0]), fmt(rate), fmt(agg(rs, "cpu", "cores")[0]),
                         "yes" if not any(valid(r) for r in rs) else "; ".join(sorted({valid(r) for r in rs if valid(r)}))])
        L.append(md_table(["Backend", "Knobs", "Idle before each call", "Cache", "n", "p50", "p90", "p99",
                           "Keep-alive reads/s", "CPU cores", "Valid"], rows))
    wk = [(k, v) for k, v in list(kg.items()) + list(gr.items()) if parse_case(k[3])[0] == "wake"]
    if wk:
        gaps = sorted({c["gap_ms"] for _, rs in wk for r in rs for c in (r.get("curve") or [])})
        L += ["", "First-read latency (one cold row, us, p50 over the runs) against the idle time before it:", ""]
        rows = []
        for (blk, be, kn, key), rs in sorted(wk, key = lambda x: (x[0][1], x[0][2])):
            cells = []
            for gp in gaps:
                v = [c["p50"] for r in rs for c in (r.get("curve") or []) if c["gap_ms"] == gp and c.get("p50") is not None]
                cells.append(fmt(statistics.median(v)) if v else "-")
            rows.append([blk, be, kn] + cells)
        L.append(md_table(["Block", "Backend", "Knobs"] + [f"{gp} ms" for gp in gaps], rows))

    null = [(k, v) for k, v in gr.items() if parse_case(k[3])[0] == "null"]
    if null:
        L += ["", "Empty native call (fixed cost, no I/O): " + ", ".join(
            f"{k[1]} p50 {fmt(agg(v, 'lat_us', 'p50')[0])} us" for k, v in sorted(null))]

    if py:
        L += ["", "## Through Python (bench_engine.py)", "",
              "Each call as the model makes it. Python = interpreter, pybind dispatch and argument "
              "conversion going in, return and GIL coming back (native stamps on the same clock); share = "
              "median of Python / wall per call. `original` = the two ngram_gather_cpu calls through the "
              "original pool (no stamps). Stall (prefetch) = the forward's wait after 20 ms of overlap.", ""]
        grp = {}
        for r in py:
            if "case" in r:
                grp.setdefault((r["variant"], r["case_key"]), []).append(r)
        rows = []
        for (v, key), rs in sorted(grp.items(), key = lambda x: (x[0][1], x[0][0])):
            rows.append([v, key, cache_s(rs) if (rs[0].get("cache") or {}).get("pages_per_call") else "-", rs[0].get("calls"),
                         fmt_spread(agg(rs, "wall_us", "p50")), fmt(agg(rs, "wall_us", "p99")[0]),
                         fmt(agg(rs, "python_us", "p50")[0]), fmt(agg(rs, "python_in_us", "p50")[0]),
                         fmt(agg(rs, "python_out_us", "p50")[0]),
                         f"{agg(rs, 'python_share')[0] * 100:.2f}" if agg(rs, "python_share")[0] is not None else "-",
                         fmt(agg(rs, "io_us", "p50")[0]), fmt(agg(rs, "stall_us", "p50")[0]),
                         "yes" if not any(valid(r) for r in rs) else "; ".join(sorted({valid(r) for r in rs if valid(r)}))])
        L.append(md_table(["Variant", "Case", "Cache", "n", "Wall p50", "Wall p99", "Python p50", "in", "out",
                           "Share %", "I/O p50", "Stall p50", "Valid"], rows))

    sw = group(nat, lambda b: b.startswith("sweep:"))
    if sw:
        base = group(nat, lambda b: b in ("decision", "baseline"))
        bmap = {}
        for (blk, be, kn, key), rs in base.items():
            bmap.setdefault((be, key), []).extend(rs)
        L += ["", "## Knob sweeps (one knob at a time, against the backend's defaults)", "",
              "Ratio = sweep point / defaults on this host (p50 latency: below 1 is better; throughput: "
              "above 1 is better). Single runs: differences under ~10 % are noise here.", ""]
        rows = []
        for (blk, be, kn, key), rs in sorted(sw.items(), key = lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
            kind = parse_case(key)[0]
            p50 = agg(rs, "lat_us", "p50")[0]
            thr_k = "gb_per_s" if kind == "extents" else "bulk_gb_per_s" if kind == "mixed" else "rows_per_s"
            thr = agg(rs, thr_k)[0]
            b = bmap.get((be, key))
            bp = agg(b, "lat_us", "p50")[0] if b else None
            bt = agg(b, thr_k)[0] if b else None
            rows.append([blk.split(":", 1)[1], be, kn, key, fmt(p50), fmt(agg(rs, "lat_us", "p99")[0]),
                         fmt(thr), f"{p50 / bp:.2f}" if p50 and bp else "-", f"{thr / bt:.2f}" if thr and bt else "-",
                         fmt(agg(rs, "cpu", "cores")[0]), "yes" if not any(valid(r) for r in rs) else
                         "; ".join(sorted({valid(r) for r in rs if valid(r)}))])
        L.append(md_table(["Sweep", "Backend", "Knob", "Case", "p50", "p99", "Throughput", "p50 ratio",
                           "Throughput ratio", "CPU cores", "Valid"], rows))
    return "\n".join(L) + "\n"


def write_report(d, out = None):
    s = report_md(d)
    out = out or os.path.join(d, "report.md")
    with open(out, "w") as f:
        f.write(s)
    print(f"report: {out}")


# ---- compare two machines --------------------------------------------------------------------------

def compare_md(da, db):
    ea, na, pa = load(da)
    eb, nb, pb = load(db)
    A, B = os.path.basename(os.path.abspath(da)), os.path.basename(os.path.abspath(db))
    L = [f"# Disk engine: {A} against {B}", "",
         "Same cases, same knobs; medians over runs. Ratio = B / A (latency: below 1 means B is faster; "
         "throughput: above 1 means B is faster).", "", "## Environments", ""]
    sa, sb = dict(env_summary(ea)), dict(env_summary(eb))
    keys = list(dict.fromkeys(list(sa) + list(sb)))
    L.append(md_table(["", A, B], [[k, sa.get(k, "-"), sb.get(k, "-")] for k in keys]))
    ga = group(na, lambda b: True)
    gb = group(nb, lambda b: True)
    rows = []
    for k in sorted(set(ga) & set(gb), key = lambda k: (k[0], k[3], k[1], k[2])):
        ra, rb = ga[k], gb[k]
        kind = parse_case(k[3])[0]
        thr_k = {"rows": "rows_per_s", "extents": "gb_per_s", "mixed": "bulk_gb_per_s"}.get(kind)
        a50, b50 = agg(ra, "lat_us", "p50")[0], agg(rb, "lat_us", "p50")[0]
        a99, b99 = agg(ra, "lat_us", "p99")[0], agg(rb, "lat_us", "p99")[0]
        at, bt = (agg(ra, thr_k)[0], agg(rb, thr_k)[0]) if thr_k else (None, None)
        ai, bi = agg(ra, "iops")[0], agg(rb, "iops")[0]
        rows.append([k[0], k[1], k[2], k[3], fmt(a50), fmt(b50), f"{b50 / a50:.2f}" if a50 and b50 else "-",
                     fmt(a99), fmt(b99), f"{b99 / a99:.2f}" if a99 and b99 else "-",
                     fmt(at), fmt(bt), f"{bt / at:.2f}" if at and bt else "-", fmt(ai), fmt(bi),
                     "" if not any(valid(r) for r in ra + rb) else "invalid runs included"])
    L += ["", "## Native cases", "",
          md_table(["Block", "Backend", "Knobs", "Case", f"p50 {A}", f"p50 {B}", "ratio", f"p99 {A}", f"p99 {B}",
                    "ratio", f"thr. {A}", f"thr. {B}", "ratio", f"IOPS {A}", f"IOPS {B}", "Note"], rows)]
    gpa, gpb = {}, {}
    for src, dst in ((pa, gpa), (pb, gpb)):
        for r in src:
            if "case" in r:
                dst.setdefault((r["variant"], r["case_key"]), []).append(r)
    prow = []
    for k in sorted(set(gpa) & set(gpb)):
        a50, b50 = agg(gpa[k], "wall_us", "p50")[0], agg(gpb[k], "wall_us", "p50")[0]
        prow.append([k[0], k[1], fmt(a50), fmt(b50), f"{b50 / a50:.2f}" if a50 and b50 else "-",
                     fmt(agg(gpa[k], "python_us", "p50")[0]), fmt(agg(gpb[k], "python_us", "p50")[0])])
    if prow:
        L += ["", "## Through Python", "",
              md_table(["Variant", "Case", f"wall p50 {A}", f"wall p50 {B}", "ratio", f"Python us {A}", f"Python us {B}"], prow)]
    return "\n".join(L) + "\n"


# ---- the auto decision ---------------------------------------------------------------------------

def read_auto(src_root):
    p = os.path.join(src_root, AUTO_H)
    with open(p) as f:
        s = f.read()
    mb = re.findall(r"^constexpr Backend kAutoBackend = Backend::(\w+);$", s, re.M)
    mr = re.findall(r"^constexpr bool kAutoNgramEngine = (true|false);$", s, re.M)
    mk = re.findall(r"^constexpr int kAutoKeepaliveMs = (\d+);$", s, re.M)
    assert len(mb) == 1 and len(mr) == 1 and len(mk) == 1, \
        f"{p}: expected exactly one kAutoBackend, kAutoNgramEngine and kAutoKeepaliveMs"
    name = {"Pread": "pread", "ODirect": "odirect", "IoUring": "io_uring"}[mb[0]]
    return name, mr[0] == "true", int(mk[0]), s


def decide(d, a):
    env, nat, py = load(d)
    h = a.decode_cache
    key_dec = f"rows:cache={h},u=48"
    key_p99 = key_dec
    notes, invalid = [], []
    cur_backend, cur_route, cur_ka = "pread", False, 0
    if a.apply or a.src:
        cur_backend, cur_route, cur_ka, _ = read_auto(a.apply or a.src)

    dec = [r for r in nat if r.get("block") == "decision" and "case" in r]
    if not dec:
        raise SystemExit(f"{d}: no decision block in native.jsonl")
    per = {}
    for r in dec:
        why = valid(r, a.max_foreign)
        if why:
            invalid.append(f"{r['backend']} {r['case_key']} run {r.get('repeat')}: {why}")
        per.setdefault((r["backend"], r["case_key"]), []).append(r)
    cands = []
    for b in BACKENDS:
        rs = per.get((b, key_dec), [])
        if not rs:
            notes.append(f"{b}: no `{key_dec}` runs")
            continue
        if any(g(r, "config", "fallbacks") for r in rs):
            notes.append(f"{b}: excluded, it fell back on this host ({g(rs[0], 'config', 'fallbacks')})")
            continue
        cands.append(b)
    if invalid and not a.force:
        return {"ok": False, "why": "invalid decision runs (see list); rerun idle, or --force", "invalid": invalid,
                "notes": notes}

    def metric(b, key, *path, block = "decision", reduce = statistics.median):
        rs = [r for r in nat if r.get("block") == block and r.get("backend") == b and r.get("case_key") == key
              and not r.get("knobs") and "case" in r]
        v = [g(r, *path) for r in rs]
        v = [x for x in v if isinstance(x, (int, float))]
        return (reduce(v), v) if v else (None, [])

    table = {}
    for b in cands:
        m, runs = metric(b, key_dec, "lat_us", "p50")
        spread = (max(runs) - min(runs)) / m if m and len(runs) > 1 else 0.0
        table[b] = {"decode_p50": m, "runs": runs, "spread": spread,
                    "decode_p99": metric(b, key_p99, "lat_us", "p99")[0],
                    "cold_p50": metric(b, "rows:cache=cold,u=48", "lat_us", "p50")[0],
                    "warm_p50": metric(b, "rows:cache=warm,u=48", "lat_us", "p50")[0],
                    "mixed_p50": metric(b, "mixed:bulk=extents,bulk_cls=1", "lat_us", "p50", block = "baseline")[0],
                    "extents_gbps": metric(b, "extents:cache=cold,k=4", "gb_per_s", block = "baseline")[0],
                    "cpu_us_per_op": metric(b, key_dec, "cpu", "us_per_op")[0]}
    steps = []
    alive = [b for b in cands if table[b]["decode_p50"] is not None]
    if not alive:
        return {"ok": False, "why": "no candidate has decode measurements", "notes": notes, "invalid": invalid}

    def narrow(label, key, lower = True):
        nonlocal alive
        vals = {b: table[b][key] for b in alive if table[b][key] is not None}
        if len(alive) <= 1 or not vals:
            return
        best = min(vals.values()) if lower else max(vals.values())
        tol = max(a.tolerance, max(table[b]["spread"] for b in alive))
        keep = [b for b in alive if b in vals and
                (vals[b] <= best * (1 + tol) if lower else vals[b] >= best * (1 - tol))]
        steps.append({"step": label, "tolerance": tol, "values": vals, "kept": keep})
        alive = keep

    narrow(f"decode p50 (u=48, cache {h})", "decode_p50")
    narrow("decode p99", "decode_p99")
    narrow("decode p50 next to expert reads (class 1)", "mixed_p50")
    narrow("extent throughput (k=4)", "extents_gbps", lower = False)
    narrow("CPU us per device read at decode", "cpu_us_per_op")
    if len(alive) > 1:
        winner = cur_backend if cur_backend in alive else alive[0]
        steps.append({"step": "still tied: keep the current auto backend", "kept": [winner]})
    else:
        winner = alive[0]
    robust = {c: min((b for b in cands if table[b][f"{c}_p50"] is not None), key = lambda b: table[b][f"{c}_p50"],
                     default = None) for c in ("cold", "warm")}

    # the n-gram route: the winner through Python against the original gather
    route, route_why = cur_route, "no Python runs: route unchanged"
    pyv = [r for r in py if "case" in r and not valid(r, a.max_foreign)]

    def pym(variant, key):
        v = [g(r, "wall_us", "p50") for r in pyv if r["variant"] == variant and r["case_key"] == key]
        v = [x for x in v if isinstance(x, (int, float))]
        return statistics.median(v) if v else None

    pk, pk8 = f"engram:cache={h},u=48", "engram:cache=cold,u=8192"
    we, wo = pym(winner, pk), pym("original", pk)
    pe, po = pym(winner, pk8), pym("original", pk8)
    if we and wo:
        faster = we <= wo * (1 - a.tolerance)
        prefill_ok = pe is None or po is None or pe <= po * (1 + a.tolerance)
        route = faster and prefill_ok
        route_why = (f"{winner} through Python {we:.0f} us vs original {wo:.0f} us at decode (u=48, cache {h})"
                     + (f"; 8192 cold rows {pe / 1e3:.1f} ms vs {po / 1e3:.1f} ms" if pe and po else "")
                     + (": engine" if route else ": original" + ("" if faster else " (not faster by the tolerance)")
                        + ("" if prefill_ok else " (prefill would regress)")))
    # the keep-alive: decode rows after 20 ms of idle, the winner with and without it
    ka, ka_why, ka_table = cur_ka, "no keep-alive runs: unchanged", {}
    kkey = f"rows:cache={h},gap_us=20000,u=48"
    kr = [r for r in nat if r.get("block") == "keepalive" and r.get("backend") == winner and "case" in r]
    bad_k = [valid(r, a.max_foreign) for r in kr if valid(r, a.max_foreign)]
    for v in sorted({(r.get("knobs") or {}).get("keepalive_ms") for r in kr}, key = lambda x: int(x)):
        rs = [r for r in kr if (r.get("knobs") or {}).get("keepalive_ms") == v]
        m = [g(r, "lat_us", "p50") for r in rs if r["case_key"] == kkey]
        m200 = [g(r, "lat_us", "p50") for r in rs if r["case_key"] == kkey.replace("20000", "200000")]
        ka_table[v] = {"gap20_p50": statistics.median(m) if m else None,
                       "gap200_p50": statistics.median(m200) if m200 else None}
    if bad_k and not a.force:
        ka_why = "invalid keep-alive runs: unchanged"
    elif ka_table.get("0", {}).get("gap20_p50") and ka_table.get("5", {}).get("gap20_p50"):
        off, on = ka_table["0"]["gap20_p50"], ka_table["5"]["gap20_p50"]
        ka = 5 if on <= off * (1 - a.tolerance) else 0
        ka_why = (f"{winner}, decode rows after 20 ms idle: {off:.0f} us without, {on:.0f} us with a 5 ms "
                  f"keep-alive" + (f" (after 200 ms idle: {ka_table['0']['gap200_p50']:.0f} vs "
                                   f"{ka_table['5']['gap200_p50']:.0f} us)"
                                   if ka_table["0"]["gap200_p50"] and ka_table["5"]["gap200_p50"] else "")
                  + (": on" if ka else ": off (not faster by the tolerance)"))
    return {"ok": True, "backend": winner, "ngram_engine": route, "route_why": route_why,
            "keepalive_ms": ka, "keepalive_why": ka_why, "keepalive_table": ka_table,
            "advice": window_advice(nat, a),
            "current": {"backend": cur_backend, "ngram_engine": cur_route, "keepalive_ms": cur_ka},
            "decode_cache": h,
            "tolerance": a.tolerance, "candidates": table, "steps": steps, "robust": robust,
            "notes": notes, "invalid": invalid, "forced": bool(invalid and a.force),
            "env_label": env.get("label"), "results": os.path.abspath(d),
            "decided": datetime.datetime.now().isoformat(timespec = "seconds")}


def window_advice(nat, a):
    """Not applied: what the class-1 window sweep says about EXL3_DISK_WINDOW_EXPERT, per backend
    swept. A window is worth considering when the bulk rate next to decode stays within the
    tolerance of the default's and decode latency next to that bulk improves by the tolerance."""
    key = "mixed:bulk=extents,bulk_cls=1"
    out = []
    ok = lambda r: "case" in r and r.get("case_key") == key and not valid(r, a.max_foreign)
    for be in BACKENDS:
        base = [r for r in nat if r.get("block") == "baseline" and r.get("backend") == be and not r.get("knobs") and ok(r)]
        sw = [r for r in nat if r.get("block") == "sweep:window_expert" and r.get("backend") == be and ok(r)]
        if not base or not sw:
            continue
        d0, t0 = agg(base, "lat_us", "p50")[0], agg(base, "bulk_gb_per_s")[0]
        if not d0 or not t0:
            continue
        pts = []
        for v in sorted({r["knobs"].get("window_expert") for r in sw}):
            rs = [r for r in sw if r["knobs"].get("window_expert") == v]
            d1, t1 = agg(rs, "lat_us", "p50")[0], agg(rs, "bulk_gb_per_s")[0]
            if d1 and t1:
                pts.append((v, d1, t1))
        good = [p for p in pts if p[2] >= t0 * (1 - a.tolerance) and p[1] <= d0 * (1 - a.tolerance)]
        line = (f"{be}: decode rows next to a class-1 extent stream, p50 {d0:.0f} us at the default window "
                f"({t0:.1f} GB/s bulk); " + "; ".join(f"{v}: {d1:.0f} us ({t1:.1f} GB/s)" for v, d1, t1 in pts))
        if good:
            v, d1, t1 = min(good, key = lambda p: p[1])
            line += f". Consider EXL3_DISK_WINDOW_EXPERT={v}"
        out.append(line)
    return out


def cache_desc(h):
    if h == "cold":
        return "no pages resident (cold)"
    if h == "warm":
        return "every page resident (warm)"
    return f"{float(h):.0%} of its pages resident"


def decision_md(r):
    L = ["## Decision: what `auto` resolves to", ""]
    if not r["ok"]:
        L += [f"**No decision**: {r['why']}.", ""] + [f"- {x}" for x in r.get("invalid", []) + r.get("notes", [])]
        return "\n".join(L)
    L.append(f"**`auto` = `{r['backend']}`, n-gram route `{'engine' if r['ngram_engine'] else 'original'}`, "
             f"keep-alive {r['keepalive_ms']} ms** (was `{r['current']['backend']}`, "
             f"`{'engine' if r['current']['ngram_engine'] else 'original'}`, {r['current']['keepalive_ms']} ms).")
    L += ["", f"Decode-critical metric: p50 of a 48-row two-table call with {cache_desc(r['decode_cache'])} "
          f"(median of the decision runs); ties within {r['tolerance']:.0%} or the run-to-run spread go to the "
          f"next step.", ""]
    rows = []
    for b, t in r["candidates"].items():
        rows.append([b, fmt(t["decode_p50"]), " / ".join(fmt(x) for x in t["runs"]), f"{t['spread']:.1%}",
                     fmt(t["decode_p99"]), fmt(t["cold_p50"]), fmt(t["warm_p50"]), fmt(t["mixed_p50"]),
                     fmt(t["extents_gbps"]), fmt(t["cpu_us_per_op"])])
    L.append(md_table(["Backend", "Decode p50", "Runs", "Spread", "Decode p99", "Cold p50", "Warm p50",
                       "Next to expert reads p50", "Extents k=4 GB/s", "CPU us/op"], rows))
    L += [""] + [f"{i + 1}. {s['step']}: kept {', '.join(s['kept'])}" +
                 (f" (tolerance {s['tolerance']:.1%})" if "tolerance" in s else "") for i, s in enumerate(r["steps"])]
    L += ["", f"Also fastest cold: {r['robust'].get('cold')}, warm: {r['robust'].get('warm')}.",
          f"N-gram route: {r['route_why']}.", f"Keep-alive: {r['keepalive_why']}."]
    for x in r.get("advice", []):
        L.append(f"Advice (not applied to `disk_auto.h`): {x}.")
    if r.get("forced"):
        L.append("**Forced despite invalid runs** (listed below).")
    L += [f"- {x}" for x in r.get("invalid", []) + r.get("notes", [])]
    return "\n".join(L)


AUTO_TEMPLATE = """#pragma once

// What EXL3_DISK_BACKEND=auto (or unset) resolves to.
//
// This file is written whole by `python tests/disk_engine/bench_suite.py decide --apply`
// from a benchmark run on the test host (doc/disk_engine.md, "How auto is decided"). Change it
// by hand only together with the evidence below.
//
//   kAutoBackend      the engine backend for the extension's disk_* entry points
//   kAutoNgramEngine  whether ngram_gather_cpu (PLE n-gram tables, the DeepSeek-V4.1 engram)
//                     uses the engine (true) or keeps its original thread pool (false)
//   kAutoKeepaliveMs  EXL3_DISK_KEEPALIVE_MS=auto (unset): the keep-alive read period, 0 off
//
{evidence}

#include "disk_engine.h"

namespace exl3_disk
{{

constexpr Backend kAutoBackend = Backend::{enum};
constexpr bool kAutoNgramEngine = {route};
constexpr int kAutoKeepaliveMs = {ka};

}}  // namespace exl3_disk
"""


def evidence_lines(r):
    ev = [f"Evidence: decided {r['decided'][:10]} from {os.path.basename(r['results'])}",
          f"({r.get('env_label') or 'environment not labelled'}).",
          f"Decode rows (u=48, two tables, {cache_desc(r['decode_cache'])}), p50 us:"]
    ev.append(", ".join(f"{b} {t['decode_p50']:.0f}" for b, t in r["candidates"].items()
                        if t["decode_p50"] is not None) + ".")
    ev += [s["step"] + ": " + ", ".join(s["kept"]) + "." for s in r["steps"][1:]]
    ev.append("N-gram route: " + r["route_why"] + ".")
    ev.append("Keep-alive: " + r["keepalive_why"] + ".")
    out, line = [], "//"
    for w in " ".join(ev).split():
        if len(line) + 1 + len(w) > 96:
            out.append(line)
            line = "//"
        line += " " + w
    out.append(line)
    return "\n".join(out)


def apply_auto(r, src_root):
    enum = {"pread": "Pread", "odirect": "ODirect", "io_uring": "IoUring"}[r["backend"]]
    new = AUTO_TEMPLATE.format(evidence = evidence_lines(r), enum = enum,
                               route = "true" if r["ngram_engine"] else "false", ka = int(r["keepalive_ms"]))
    p = os.path.join(src_root, AUTO_H)
    _, _, _, old = read_auto(src_root)
    with open(p, "w") as f:
        f.write(new)
    b2, r2, k2, _ = read_auto(src_root)
    assert b2 == r["backend"] and r2 == r["ngram_engine"] and k2 == r["keepalive_ms"], \
        "disk_auto.h did not round-trip"
    import difflib
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/" + AUTO_H, "b/" + AUTO_H))


def cmd_decide(a):
    r = decide(a.dir, a)
    with open(os.path.join(a.dir, "decision.json"), "w") as f:
        json.dump(r, f, indent = 1)
    md = decision_md(r)
    with open(os.path.join(a.dir, "decision.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    if a.apply:
        if not r["ok"]:
            raise SystemExit("not applied: no valid decision")
        diff = apply_auto(r, a.apply)
        patch = os.path.join(a.dir, "disk_auto.patch")
        with open(patch, "w") as f:
            f.write(diff)
        print(diff or f"{AUTO_H}: unchanged")
        print(f"patch: {patch}")
    write_report(a.dir)
    return 0 if r["ok"] else 1


# ---- main ------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)

    def src_args(p):
        p.add_argument("--scratch", help = "directory for a scratch file (created, deleted afterwards)")
        p.add_argument("--scratch-mb", type = int, default = 1024)
        p.add_argument("--model", help = "DeepSeek-V4.1 EXL3 checkpoint: engram tables and expert extents")
        p.add_argument("--layer", action = "append", default = [], help = "PATH:OFF:ROW_BYTES:ROWS[+...]")
        p.add_argument("--extent-list", help = "lines 'PATH OFFSET LENGTH'")

    def plan_args(p):
        p.add_argument("--preset", default = "smoke", choices = ("smoke", "window"))
        p.add_argument("--scale", type = float, default = 1.0, help = "multiply every iteration count")
        p.add_argument("--decode-cache", default = "0.75",
                       help = "warm fraction of the decode-critical case (the steady-state estimate)")

    p = sub.add_parser("env")
    p.add_argument("--file", action = "append", default = [])
    p.add_argument("--env-label", default = "")
    p.add_argument("--env-notes", default = "")
    p.add_argument("--driver", default = "")
    p = sub.add_parser("plan")
    plan_args(p)
    p.add_argument("-v", "--verbose", action = "store_true")
    p = sub.add_parser("run")
    plan_args(p)
    src_args(p)
    p.add_argument("--out", required = True)
    p.add_argument("--append", action = "store_true")
    p.add_argument("--driver", required = True, help = "disk_engine_bench (tests/disk_engine/build.sh)")
    p.add_argument("--budget-gb", type = float, default = 8.0, help = "bytes read from the device, all runs")
    p.add_argument("--repeats", type = int, default = 0, help = "decision and python runs (preset default)")
    p.add_argument("--python", default = "mini", choices = ("off", "mini", "full"))
    p.add_argument("--python-exe", dest = "python_exe", default = sys.executable)
    p.add_argument("--only", default = "", help = "blocks: decision,python,python-extra,keepalive,baseline,sweeps or sweep:NAME")
    p.add_argument("--env-label", default = "")
    p.add_argument("--env-notes", default = "")
    p.add_argument("--idle-sample-ms", type = int, default = 200)
    p.add_argument("--nice", dest = "nice_level", type = int, default = 10)
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.add_argument("--md")
    p = sub.add_parser("compare")
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--md")
    p = sub.add_parser("decide")
    p.add_argument("dir")
    p.add_argument("--decode-cache", default = "0.75")
    p.add_argument("--tolerance", type = float, default = 0.05)
    p.add_argument("--max-foreign", type = float, default = 1.0,
                   help = "other cores busy before a decision case beyond which it is invalid")
    p.add_argument("--apply", help = "source root whose disk_auto.h to rewrite")
    p.add_argument("--src", help = "source root to read the current auto choice from (without writing)")
    p.add_argument("--force", action = "store_true", help = "decide despite invalid runs")
    a = ap.parse_args()

    if a.cmd == "env":
        e = env_record(a.file, a.env_label, a.env_notes, a.driver)
        print(json.dumps(e, indent = 1))
        print(md_table(["", "This machine"], env_summary(e)))
    elif a.cmd == "plan":
        cmd_plan(a)
    elif a.cmd == "run":
        a.driver = os.path.abspath(a.driver)
        cmd_run(a)
    elif a.cmd == "report":
        write_report(a.dir, a.md)
    elif a.cmd == "compare":
        s = compare_md(a.dir_a, a.dir_b)
        if a.md:
            with open(a.md, "w") as f:
                f.write(s)
        print(s)
    elif a.cmd == "decide":
        sys.exit(cmd_decide(a))


if __name__ == "__main__":
    main()
