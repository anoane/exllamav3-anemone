"""
Disk I/O engine through the extension (exllamav3_ext/disk, doc/disk_engine.md): every backend
byte-exact against Python preads, ngram_gather_cpu on the engine against its original pool,
two-table engram gathers, extents with the slot geometry, asynchronous tickets and completion
flags, errors, threads, and the knob routing. CPU only.

The full extension is used by default. EXL3_DISK_TEST_MINI=1 builds a small extension with the
same sources and bindings instead (tests/disk_engine/mini_ext.py). EXL3_DISK_TEST_DIR puts the
scratch files on a chosen file system (O_DIRECT needs a real one; tmpfs may refuse it).

    EXL3_DISK_TEST_MINI=1 EXL3_DISK_TEST_DIR=/path/on/ext4 python -m pytest tests/test_disk_engine_.py
"""

import gc
import os
import subprocess
import sys
import threading
import time
import weakref

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BACKENDS = [
    {"backend": "pread"},
    {"backend": "odirect"},
    {"backend": "io_uring"},
    {"backend": "io_uring", "direct": "all"},
    {"backend": "io_uring", "direct": "none", "register": "none"},
    {"backend": "pread", "direct": "extents", "threads": 2},
    {"backend": "io_uring", "fault_no_uring": 1},       # io_uring refused: the pool takes over
]


def load_ext():
    if os.environ.get("EXL3_DISK_TEST_MINI") == "1":
        sys.path.insert(0, os.path.join(HERE, "disk_engine"))
        from mini_ext import load_ext as mini
        return mini()
    from exllamav3.ext import exllamav3_ext
    return exllamav3_ext


@pytest.fixture(scope = "module")
def ext():
    e = load_ext()
    yield e
    e.disk_engine_shutdown()


@pytest.fixture(scope = "module")
def data(tmp_path_factory):
    d = os.environ.get("EXL3_DISK_TEST_DIR") or str(tmp_path_factory.mktemp("disk_engine"))
    os.makedirs(d, exist_ok = True)
    path = os.path.join(d, "disk_engine_test.bin")
    size = 40 * (1 << 20) + 4321
    rng = np.random.default_rng(1234)
    buf = rng.integers(0, 256, size = size, dtype = np.uint8)
    buf.tofile(path)
    fd = os.open(path, os.O_RDONLY)
    yield path, fd, buf
    os.close(fd)
    os.unlink(path)


def ref_rows(buf, base, row_bytes, uids, uid_base = 0):
    out = np.empty((len(uids), row_bytes), dtype = np.uint8)
    for i, u in enumerate(uids):
        o = base + (int(u) - uid_base) * row_bytes
        out[i] = buf[o : o + row_bytes]
    return out


def aligned_u8(nbytes, align = 4096):
    """A uint8 tensor view that starts on an `align` boundary (slots must be 4 KiB aligned)."""
    t = torch.empty(nbytes + align, dtype = torch.uint8)
    pad = (-t.data_ptr()) % align
    return t[pad : pad + nbytes]


@pytest.mark.parametrize("cfg", BACKENDS, ids = lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_ngram_gather_engine_equals_original(ext, data, cfg):
    path, fd, buf = data
    rng = np.random.default_rng(7)
    for row_bytes in (8, 256, 264, 320):
        base = int(rng.integers(0, 4096))
        rows = (len(buf) - base) // row_bytes
        for n in (1, 24, 48, 700):
            uids = np.unique(rng.integers(0, rows, size = n)).astype(np.int64)
            if n == 700:
                uids = np.unique(np.concatenate([uids, [rows - 1], np.arange(10, 20)]))
            u = torch.from_numpy(uids)
            want = ref_rows(buf, base, row_bytes, uids)

            ext.disk_engine_configure({"backend": "auto"})
            assert ext.disk_ngram_route() == "original"
            a = torch.zeros((len(uids), row_bytes), dtype = torch.uint8)
            ext.ngram_gather_cpu(fd, base, row_bytes, u, 0, a)

            info = ext.disk_engine_configure(cfg)
            assert info["ngram_route"] == "engine" and info["backend_named"]
            b = torch.zeros((len(uids), row_bytes), dtype = torch.uint8)
            ext.ngram_gather_cpu(fd, base, row_bytes, u, 0, b)

            assert np.array_equal(a.numpy(), want), f"original gather differs ({row_bytes}, {n})"
            assert torch.equal(a, b), f"engine differs from the original ({cfg}, {row_bytes}, {n})"
    # int16 rows (the PLE trellis layout) and a shard-local uid_base
    uids = torch.tensor([5, 6, 7, 100, 3000], dtype = torch.long) + 1000
    out = torch.zeros((5, 80), dtype = torch.int16)
    ext.ngram_gather_cpu(fd, 64, 160, uids, 1000, out)
    want = ref_rows(buf, 64, 160, uids.numpy(), 1000)
    assert np.array_equal(out.numpy().view(np.uint8).reshape(5, 160), want)


@pytest.mark.parametrize("cfg", BACKENDS, ids = lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_gather_rows_two_tables(ext, data, cfg):
    """One call per engram layer: the 256-byte weight rows and the 8-byte scale rows."""
    path, fd, buf = data
    ext.disk_engine_configure(cfg)
    rng = np.random.default_rng(11)
    w_base, s_base = 664, 30 * (1 << 20) + 2712
    rows = min((s_base - w_base) // 256, (len(buf) - s_base) // 8)
    for n in (24, 48, 96, 8192):
        uids = np.unique(rng.integers(0, rows, size = n)).astype(np.int64)
        u = torch.from_numpy(uids)
        w = torch.zeros((len(uids), 256), dtype = torch.uint8)
        s = torch.zeros((len(uids), 8), dtype = torch.uint8)
        assert ext.disk_gather_rows(u, 0, [fd, fd], [w_base, s_base], [256, 8], [w, s]) is None
        assert np.array_equal(w.numpy(), ref_rows(buf, w_base, 256, uids))
        assert np.array_equal(s.numpy(), ref_rows(buf, s_base, 8, uids))

        # asynchronous, with a completion flag; the ticket keeps the outputs alive
        flag = torch.zeros(4, dtype = torch.int32)
        w2 = torch.zeros_like(w)
        t = ext.disk_gather_rows(u, 0, [fd, fd], [w_base, s_base], [256, 8], [w2, s],
                                 cls = 2, wait = False, flag = flag, flag_index = 2,
                                 flag_value = 12345)
        t.wait(timeout = 60)
        assert t.done() and t.error() == 0
        assert int(flag[2]) == 12345 and int(flag[0]) == 0
        times = t.times()
        assert times["t_done"] >= times["t_first_issue"] >= times["t_submit"] > 0
        t.release()
        assert torch.equal(w, w2)


@pytest.mark.parametrize("cfg", BACKENDS, ids = lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_read_extents(ext, data, cfg):
    path, fd, buf = data
    ext.disk_engine_configure(cfg)
    rng = np.random.default_rng(5)
    L = [1, 4095, 13315596, 1310721, 3000001, len(buf) - 7]
    offs = [int(rng.integers(0, len(buf) - l + 1)) for l in L]
    offs[0] = len(buf) - 1                     # the last byte of the file
    offs[-1] = 7
    geo = [ext.disk_extent_geometry(fd, o, l) for o, l in zip(offs, L)]
    slot = max(g[1] for g in geo)
    dst = aligned_u8(slot * len(L))
    dst.fill_(0xA5)
    pay = torch.full((len(L),), -1, dtype = torch.long)
    t = ext.disk_read_extents(
        torch.full((len(L),), fd, dtype = torch.long), torch.tensor(offs), torch.tensor(L), dst,
        torch.arange(len(L), dtype = torch.long) * slot, slot, payload_offsets = pay)
    t.wait(timeout = 60)
    t.release()
    for i, (o, l) in enumerate(zip(offs, L)):
        p = int(pay[i])
        assert p == geo[i][0] == o % 4096
        got = dst[i * slot + p : i * slot + p + l].numpy()
        assert np.array_equal(got, buf[o : o + l]), f"extent {i} ({o}, {l}) differs"
    # synchronous form
    dst2 = aligned_u8(slot)
    assert ext.disk_read_extents(torch.tensor([fd]), torch.tensor([offs[2]]), torch.tensor([L[2]]),
                                 dst2, torch.tensor([0]), slot, wait = True) is None
    p = geo[2][0]
    assert np.array_equal(dst2[p : p + L[2]].numpy(), buf[offs[2] : offs[2] + L[2]])


def test_errors(ext, data):
    path, fd, buf = data
    ext.disk_engine_configure({"backend": "io_uring"})
    u = torch.tensor([0, 1, 2], dtype = torch.long)
    out = torch.zeros((3, 256), dtype = torch.uint8)
    closed = os.open(path, os.O_RDONLY)
    os.close(closed)
    with pytest.raises(RuntimeError, match = "fstat"):
        ext.disk_gather_rows(u, 0, [closed], [0], [256], [out])
    with pytest.raises(RuntimeError, match = "beyond the file"):
        ext.disk_gather_rows(u + len(buf) // 256, 0, [fd], [0], [256], [out])
    with pytest.raises(RuntimeError, match = "below uid_base"):
        ext.disk_gather_rows(u, 1, [fd], [0], [256], [out])
    with pytest.raises(RuntimeError, match = "int64"):
        ext.disk_gather_rows(u.int(), 0, [fd], [0], [256], [out])
    with pytest.raises(RuntimeError, match = "one entry per table"):
        ext.disk_gather_rows(u, 0, [fd, fd], [0], [256], [out])
    with pytest.raises(RuntimeError, match = "output holds"):
        ext.disk_gather_rows(u, 0, [fd], [0], [256], [out[:2]])
    with pytest.raises(RuntimeError, match = "class"):
        ext.disk_gather_rows(u, 0, [fd], [0], [256], [out], cls = 7)
    with pytest.raises(RuntimeError, match = "flag"):
        ext.disk_gather_rows(u, 0, [fd], [0], [256], [out], flag = torch.zeros(2, dtype = torch.float))
    dst = aligned_u8(1 << 20)
    with pytest.raises(RuntimeError, match = "aligned"):
        ext.disk_read_extents(torch.tensor([fd]), torch.tensor([0]), torch.tensor([100]), dst,
                              torch.tensor([8]), 8192, wait = True)
    with pytest.raises(RuntimeError, match = "outside dst"):
        ext.disk_read_extents(torch.tensor([fd]), torch.tensor([0]), torch.tensor([100]), dst,
                              torch.tensor([1 << 20]), 8192, wait = True)
    with pytest.raises(RuntimeError, match = "ngram_gather_cpu"):
        ext.ngram_gather_cpu(fd, len(buf), 256, u, 0, out)
    with pytest.raises(RuntimeError, match = "EXL3_DISK_QD"):
        ext.disk_engine_configure({"backend": "io_uring", "qd": "0"})
    with pytest.raises(RuntimeError, match = "unknown knob"):
        ext.disk_engine_configure({"no_such_knob": 1})


def test_ticket_lifetime(ext, data):
    """Dropping every reference while reads are in flight: the ticket waits them out."""
    path, fd, buf = data
    ext.disk_engine_configure({"backend": "io_uring", "direct": "all", "hold_arm_ms": 10000,
                               "refill_age_ms": 0})
    slot = 14 << 20
    for _ in range(4):
        dst = aligned_u8(slot)
        t = ext.disk_read_extents(torch.tensor([fd]), torch.tensor([4097]), torch.tensor([13 << 20]),
                                  dst, torch.tensor([0]), slot, cls = 1)
        del dst
        del t                           # releases: waits for the reads in flight
    # a ticket that is not admitted (class 3 under an armed hold, lo window 0): times out, is
    # promoted, completes; another is cancelled
    ext.disk_arm_hold()
    dst = aligned_u8(slot)
    t = ext.disk_read_extents(torch.tensor([fd]), torch.tensor([0]), torch.tensor([1 << 20]),
                              dst, torch.tensor([0]), slot, cls = 3)
    with pytest.raises(TimeoutError):
        t.wait(timeout = 0.05)
    assert not t.done()
    t.promote(1)
    t.wait(timeout = 60)
    t2 = ext.disk_read_extents(torch.tensor([fd]), torch.tensor([0]), torch.tensor([1 << 20]),
                               aligned_u8(slot), torch.tensor([0]), slot, cls = 3)
    t2.cancel()
    with pytest.raises(RuntimeError, match = "cancelled"):
        t2.wait(timeout = 60)
    t2.release()
    with pytest.raises(RuntimeError, match = "released"):
        t2.done()


@pytest.mark.parametrize("cfg", [{"backend": "pread"}, {"backend": "io_uring"}],
                         ids = ["pread", "io_uring"])
def test_threads(ext, data, cfg):
    """Python threads gather at once (the GIL is released inside); every result is exact."""
    path, fd, buf = data
    ext.disk_engine_configure(cfg)
    errors = []

    def worker(k):
        try:
            rng = np.random.default_rng(100 + k)
            for i in range(30):
                uids = np.unique(rng.integers(0, len(buf) // 264 - 1, size = 64)).astype(np.int64)
                u = torch.from_numpy(uids)
                out = torch.zeros((len(uids), 264), dtype = torch.uint8)
                if i % 2:
                    ext.ngram_gather_cpu(fd, 0, 264, u, 0, out)
                else:
                    ext.disk_gather_rows(u, 0, [fd], [0], [264], [out], cls = k % 4)
                if not np.array_equal(out.numpy(), ref_rows(buf, 0, 264, uids)):
                    errors.append((k, i))
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append((k, repr(e)))

    ts = [threading.Thread(target = worker, args = (k,)) for k in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert not errors, errors


def test_stats_trace_profile(ext, data):
    path, fd, buf = data
    ext.disk_engine_configure({"backend": "io_uring", "trace": 4096})
    ext.disk_stats(reset = True)
    u = torch.arange(0, 2000, 7, dtype = torch.long)
    out = torch.zeros((len(u), 256), dtype = torch.uint8)
    ext.disk_profile(True)
    ext.disk_gather_rows(u, 0, [fd], [0], [256], [out])
    t_enter, t_sub, t_done, t_exit = ext.disk_profile_last()
    ext.disk_profile(False)
    assert 0 < t_enter <= t_sub <= t_done <= t_exit
    s = ext.disk_stats()
    c0 = s["classes"][0]
    assert c0["tickets"] == 1 and c0["ops"] == len(u) and c0["op_errors"] == 0
    assert c0["bytes"] == len(u) * 256 and sum(c0["hist_ticket"]) == 1
    assert len(s["hist_lower_ns"]) == len(c0["hist_queue"]) and s["stray_cqes"] == 0
    tr = ext.disk_trace(clear = True)
    assert tr.shape == (len(u), 8) and bool((tr[:, 2] == 256).all())
    info = ext.disk_engine_info()
    assert info["backend"] == "io_uring" and "io_uring" in info["describe"]
    assert ext.disk_set_thread_class(2) == 0 and ext.disk_set_thread_class(0) == 2


def test_registered_buffer(ext, data):
    path, fd, buf = data
    ext.disk_engine_configure({"backend": "io_uring"})
    arena = aligned_u8(32 << 20)
    idx = ext.disk_register_buffer(arena)
    if idx < 0:
        pytest.skip("registered buffers unavailable")
    t = ext.disk_read_extents(torch.tensor([fd]), torch.tensor([123]), torch.tensor([9 << 20]),
                              arena, torch.tensor([8 << 20]), 16 << 20)
    t.wait(timeout = 60)
    t.release()
    assert np.array_equal(arena[(8 << 20) + 123 : (8 << 20) + 123 + (9 << 20)].numpy(),
                          buf[123 : 123 + (9 << 20)])
    ext.disk_unregister_buffer(idx)
    with pytest.raises(RuntimeError):
        ext.disk_unregister_buffer(idx)


def _run(code, env):
    e = dict(os.environ)
    for k in list(e):
        if k.startswith("EXL3_DISK_") and k not in ("EXL3_DISK_TEST_MINI", "EXL3_DISK_TEST_DIR",
                                                    "EXL3_DISK_MINI_BUILD_DIR"):
            del e[k]
    e.update(env)
    pre = "import sys; sys.path.insert(0, %r); from test_disk_engine_ import load_ext; ext = load_ext()\n" % HERE
    return subprocess.run([sys.executable, "-c", pre + code], env = e, capture_output = True,
                          text = True, timeout = 600)


def test_knob_routing(data):
    """ngram_gather_cpu follows EXL3_DISK_BACKEND, read once per process."""
    path, fd, buf = data
    code = ("import os, torch\n"
            f"fd = os.open({path!r}, os.O_RDONLY)\n"
            "u = torch.tensor([1, 2, 9], dtype = torch.long)\n"
            "o = torch.zeros((3, 256), dtype = torch.uint8)\n"
            "ext.ngram_gather_cpu(fd, 0, 256, u, 0, o)\n"
            "print(ext.disk_ngram_route(), int(o.sum()))\n")
    ref = int(ref_rows(buf, 0, 256, [1, 2, 9]).astype(np.int64).sum())
    r = _run(code, {})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["original", str(ref)]
    r = _run(code, {"EXL3_DISK_BACKEND": "auto"})
    assert r.stdout.split() == ["original", str(ref)], r.stderr
    r = _run(code, {"EXL3_DISK_BACKEND": "io_uring", "EXL3_DISK_QD": "16"})
    assert r.stdout.split() == ["engine", str(ref)], r.stderr
    r = _run(code, {"EXL3_DISK_BACKEND": "uring"})
    assert r.returncode != 0 and "EXL3_DISK_BACKEND=uring" in r.stderr
    r = _run(code, {"EXL3_DISK_BACKEND": "pread", "EXL3_DISK_WINDOW_PREFETCH": "1M/2M"})
    assert r.returncode != 0 and "EXL3_DISK_WINDOW_PREFETCH" in r.stderr


@pytest.mark.parametrize("cfg", [{"backend": "pread", "fault_delay_us": 2000},
                                 {"backend": "io_uring", "direct": "all"}],
                         ids = ["pread", "io_uring"])
def test_ticket_wait_release_threads(ext, data, cfg):
    """DiskTicket.wait() on one Python thread while another releases (or drops) the ticket: the
    wait returns or raises "released", nothing is used after it was freed, nothing hangs."""
    path, fd, buf = data
    ext.disk_engine_configure(cfg)
    slot = 9 << 20
    errors = []
    for i in range(24):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        dst = aligned_u8(slot)
        t = ext.disk_read_extents(torch.tensor([fd]), torch.tensor([4096 * (i + 1) + 5]),
                                  torch.tensor([8 << 20]), dst, torch.tensor([0]), slot, cls = 1)
        go = threading.Event()

        def waiter():
            go.wait()
            try:
                t.wait(timeout = 60)
            except RuntimeError as e:
                if "released" not in str(e) and "cancelled" not in str(e):
                    errors.append(repr(e))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        def releaser():
            go.wait()
            t.release()

        ws = [threading.Thread(target = waiter) for _ in range(1 + i % 3)]
        rs = [threading.Thread(target = releaser) for _ in range(1 + (i % 2))]
        for x in ws + rs: x.start()
        go.set()
        for x in ws + rs: x.join(timeout = 120)
        assert not any(x.is_alive() for x in ws + rs), "a wait or release hung"
        with pytest.raises(RuntimeError, match = "released"):
            t.done()
        del t, dst
    assert not errors, errors
    s = ext.disk_stats()
    assert s["tickets_live"] == 0 and s["inflight_now"] == 0


def test_sync_calls_use_the_synchronous_path(ext, data):
    """wait=True runs the engine's synchronous path: with io_uring a class-0 caller submits its
    own reads before the call's submission returns, and reaps their completions itself (no
    reaper hand-off).

    Both checks hold on a loaded host. Submission is checked by order, not by time: a call
    counts when its first read was issued (trace) before its submission returned
    (disk_profile), which only the caller can do; the asynchronous hand-off (wait=False, then
    wait()) is the control, where the reaper issues the reads after the submission returned.
    Reaping is checked with the caller and the engine's threads on two different CPUs: on a
    shared CPU the reaper, which the ring wakes as the caller's reads complete, can preempt the
    caller under load and drain the completions first (it took 22 % of them on a busy host),
    which says nothing about the path the call took."""
    path, fd, buf = data
    own_cpus = os.sched_getaffinity(0)
    cpus = sorted(own_cpus)
    pin = len(cpus) >= 2
    cfg = {"backend": "io_uring", "trace": 8192}
    if pin:
        cfg["affinity"] = str(cpus[-1])             # the engine's threads
    ext.disk_engine_configure(cfg)
    try:
        if pin:
            os.sched_setaffinity(0, {cpus[0]})     # this thread only
        w_base, s_base = 664, 30 * (1 << 20) + 2712
        rng = np.random.default_rng(3)
        uids = np.unique(rng.integers(0, 3000, size = 48)).astype(np.int64)
        u = torch.from_numpy(uids)
        w = torch.zeros((len(uids), 256), dtype = torch.uint8)
        s = torch.zeros((len(uids), 8), dtype = torch.uint8)
        fds, offs, rbs, outs = [fd, fd], [w_base, s_base], [256, 8], [w, s]
        ext.disk_gather_rows(u, 0, fds, offs, rbs, outs)       # both tables in the page cache

        def calls(wait):
            """50 class-0 calls; returns how many issued their first read before the submission
            returned, and the engine's stats over the 50"""
            ext.disk_trace(clear = True)
            ext.disk_stats(reset = True)
            t_sub = []
            ext.disk_profile(True)
            try:
                for _ in range(50):
                    t = ext.disk_gather_rows(u, 0, fds, offs, rbs, outs, 0, False, wait)
                    t_sub.append(ext.disk_profile_last()[1])
                    if t is not None:
                        t.wait(timeout = 60)
                        t.release()
            finally:
                ext.disk_profile(False)
            st = ext.disk_stats()
            tr = ext.disk_trace(clear = True).numpy()
            ids = np.unique(tr[:, 0])                  # ascending: one ticket per call, in order
            assert len(ids) == 50, len(ids)
            early = sum(int(tr[tr[:, 0] == k, 6].min() <= ts) for k, ts in zip(ids, t_sub))
            return early, st

        early, st = calls(True)
        assert np.array_equal(w.numpy(), ref_rows(buf, w_base, 256, uids))
        assert np.array_equal(s.numpy(), ref_rows(buf, s_base, 8, uids))
        assert early >= 38, f"only {early} of 50 synchronous calls issued their reads themselves"
        if not pin:
            pytest.skip("one CPU: the reaper shares the caller's CPU (submission checked only)")
        assert st["inline_reaped"] > 0
        assert st["reaper_reaped"] <= st["inline_reaped"] // 4, st
        early, _ = calls(False)
        assert early <= 25, f"control: {early} of 50 handed-off calls had a read issued before " \
                            "the submission returned"
    finally:
        os.sched_setaffinity(0, own_cpus)
        ext.disk_engine_shutdown()


def test_registered_buffer_follows_its_engine(ext, data):
    """A registered tensor is dropped once its engine is gone (configure / shutdown), not kept
    until some unrelated unregister call."""
    path, fd, buf = data
    ext.disk_engine_configure({"backend": "io_uring"})
    arena = aligned_u8(16 << 20)
    idx = ext.disk_register_buffer(arena)
    if idx < 0:
        pytest.skip("registered buffers unavailable")
    ref = weakref.ref(arena)
    ext.disk_engine_configure({"backend": "io_uring"})     # the old engine is gone
    del arena
    gc.collect()
    assert ref() is None, "the registered tensor outlived its engine"
    with pytest.raises(RuntimeError, match = "no registered buffer"):
        ext.disk_unregister_buffer(idx)
    arena = aligned_u8(16 << 20)
    idx = ext.disk_register_buffer(arena)
    ref = weakref.ref(arena)
    ext.disk_engine_shutdown()
    del arena
    gc.collect()
    assert ref() is None, "the registered tensor outlived shutdown"


def test_forget(ext, data, tmp_path_factory):
    """disk_engine_forget closes the engine's descriptors of a file; an unlinked file is closed
    once idle; a later read reopens."""
    path, fd, buf = data
    ext.disk_engine_configure({"backend": "io_uring"})
    d = os.environ.get("EXL3_DISK_TEST_DIR") or str(tmp_path_factory.mktemp("disk_forget"))
    p2 = os.path.join(d, "disk_engine_forget.bin")
    np.random.default_rng(9).integers(0, 256, size = 1 << 20, dtype = np.uint8).tofile(p2)
    real = os.path.realpath(p2)

    def engine_fds():
        n = 0
        for x in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{x}").startswith(real):
                    n += 1
            except OSError:
                pass
        return n

    f2 = os.open(p2, os.O_RDONLY)
    u = torch.tensor([1, 2, 3], dtype = torch.long)
    out = torch.zeros((3, 256), dtype = torch.uint8)
    ext.disk_gather_rows(u, 0, [f2], [0], [256], [out])
    assert engine_fds() == 2                        # ours and the engine's
    assert ext.disk_engine_forget(p2) == 1
    assert engine_fds() == 1
    assert ext.disk_engine_forget(p2) == 0
    ext.disk_gather_rows(u, 0, [f2], [0], [256], [out])     # reopened
    os.close(f2)
    os.unlink(p2)
    for _ in range(200):                            # the engine notices the unlink on its own
        ext.disk_gather_rows(u, 0, [fd], [0], [256], [out])
    assert engine_fds() == 0, "an unlinked file is still held open by the engine"
    assert ext.disk_engine_forget(os.path.join(d, "no_such_file")) == 0


def test_hold_class_and_zero_window(ext, data):
    path, fd, buf = data
    ext.disk_engine_configure({"backend": "io_uring"})
    u = torch.tensor([1], dtype = torch.long)
    out = torch.zeros((1, 256), dtype = torch.uint8)
    with pytest.raises(RuntimeError, match = "hold"):
        ext.disk_gather_rows(u, 0, [fd], [0], [256], [out], cls = 2, hold = True)
    ext.disk_gather_rows(u, 0, [fd], [0], [256], [out], cls = 1, hold = True)
    for k in ("window_expert", "window_prefetch", "window_refill"):
        with pytest.raises(RuntimeError, match = k.upper()):
            ext.disk_engine_configure({"backend": "io_uring", k: "0"})
