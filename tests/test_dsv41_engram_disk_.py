"""
DeepSeek-V4.1 engram staging through the disk engine (modules/dsv41_engram.py), CPU only, no
checkpoint: the hash is replaced by a deterministic stand-in, the tables are a scratch file.

  - inline staging (class 0, hold at decode size) and the prefetch worker's asynchronous class-2
    staging, promoted to class 0 and waited for by the consumer, give the same bytes as the
    original two ngram_gather_cpu calls, for every backend;
  - a retired prefetch releases its disk ticket before its staging set is reused;
  - the route comes from the extension (disk_ngram_route), not from the environment: where the
    extension says "original" (Windows), EXL3_DISK_BACKEND=odirect in the environment must not
    send the gather to the engine;
  - a placement's disk io= (the module's io_direct): buffered and O_DIRECT row reads, same bytes;
  - the n-gram budget (--ngram_ram): a table held whole in RAM and a row cache of the streamed ones
    give the same bytes, inline and through the prefetch worker (the misses land in the cache when
    the consumer waits), on the engine and on the original route.

The full extension is used by default; EXL3_DISK_TEST_MINI=1 uses the small test build of the
same sources and bindings (tests/disk_engine/mini_ext.py) in place of exllamav3_ext.

    EXL3_DISK_TEST_MINI=1 EXL3_DISK_TEST_DIR=/path/on/ext4 python -m pytest tests/test_dsv41_engram_disk_.py
"""

import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _import_module():
    if os.environ.get("EXL3_DISK_TEST_MINI") == "1":
        sys.path.insert(0, os.path.join(HERE, "disk_engine"))
        from mini_ext import load_ext
        mini = load_ext()

        def missing(name):
            def f(*a, **k):
                raise RuntimeError(f"{name} is not in the disk engine test extension")
            return f

        class Stub(types.ModuleType):
            def __getattr__(self, k):
                if k.startswith("__"):
                    raise AttributeError(k)
                return getattr(mini, k) if hasattr(mini, k) else missing(k)

        fake = types.ModuleType("exllamav3.ext")
        fake.exllamav3_ext = Stub("exllamav3_ext")
        sys.modules["exllamav3.ext"] = fake
    import exllamav3.modules.dsv41_engram as dm
    return dm


dm = _import_module()
ext = dm.ext
HEAD_DIM, N_COLS, CTX = 256, 24, 3


class _Handle:
    """What the module uses of a DiskTensorHandle."""
    def __init__(self, filename, abs_offset, row_bytes):
        self.filename, self.abs_offset, self.row_bytes = filename, abs_offset, row_bytes
        self.fd = None

    def _ensure_open(self):
        if self.fd is None:
            self.fd = os.open(self.filename, os.O_RDONLY)
        return self.fd

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@pytest.fixture(scope = "module")
def table(tmp_path_factory):
    d = os.environ.get("EXL3_DISK_TEST_DIR") or str(tmp_path_factory.mktemp("engram_disk"))
    os.makedirs(d, exist_ok = True)
    path = os.path.join(d, "engram_disk_test.bin")
    rows = 60000
    w_off, s_off = 1000, 1000 + rows * HEAD_DIM + 777
    size = s_off + rows * (HEAD_DIM // 32)
    buf = np.random.default_rng(4).integers(0, 256, size = size, dtype = np.uint8)
    buf.tofile(path)
    yield path, rows, w_off, s_off, buf
    os.unlink(path)
    ext.disk_engine_shutdown()


def _hash_stub(rows):
    # deterministic ids in [0, rows), shape [B, L, 24], with duplicates and runs
    def f(hist, table_index, hasher):
        B, T = hist.shape
        L = T - CTX
        base = hist[:, CTX:, None].long() * 7919 + torch.arange(N_COLS)[None, None, :] * 131
        return (base % rows).view(B, L, N_COLS)
    return f


def _module(table, monkeypatch):
    path, rows, w_off, s_off, buf = table
    monkeypatch.setattr(dm, "engram_hash_chunk", _hash_stub(rows))
    eg = dm.DSV41Engram.__new__(dm.DSV41Engram)
    eg.layout = types.SimpleNamespace(head_dim = HEAD_DIM, n_hash_cols = N_COLS)
    eg.table_index = 0
    eg.ctx = CTX
    eg.device = None
    eg.engram_ctx = types.SimpleNamespace(hasher = None, executor = ThreadPoolExecutor(1))
    eg.w_handle = _Handle(path, w_off, HEAD_DIM)
    eg.s_handle = _Handle(path, s_off, HEAD_DIM // 32)
    eg._pins, eg._pending, eg._disk_argv = [], [], None
    eg.io_direct = -1
    eg.ram_tables, eg.streamed, eg.row_cache = {}, [0, 1], None
    eg.prefetch_stats = {"hit": 0, "miss": 0, "retired": 0}
    return eg


def _expect(table, hist):
    path, rows, w_off, s_off, buf = table
    idx = _hash_stub(rows)(hist, 0, None)
    u, inv = np.unique(idx.reshape(-1).numpy(), return_inverse = True)
    w = np.stack([buf[w_off + r * HEAD_DIM : w_off + (r + 1) * HEAD_DIM] for r in u])
    s = np.stack([buf[s_off + r * 8 : s_off + (r + 1) * 8] for r in u])
    return u, inv.reshape(-1), w, s


def _check(pin, U, want):
    u, inv, w, s = want
    assert U == len(u)
    assert np.array_equal(pin.w[:U].numpy(), w)
    assert np.array_equal(pin.s[:U].numpy(), s)
    assert np.array_equal(pin.inv[:len(inv)].numpy(), inv)


CONFIGS = [{"backend": "original"}, {"backend": "auto"}, {"backend": "pread"},
           {"backend": "odirect"}, {"backend": "io_uring"}, {"backend": "io_uring", "direct": "all"}]


@pytest.mark.parametrize("cfg", CONFIGS, ids = lambda c: ",".join(f"{k}={v}" for k, v in c.items()))
def test_stage_inline_and_prefetch(table, monkeypatch, cfg):
    info = ext.disk_engine_configure(cfg)
    # auto follows the recorded auto choice (disk/disk_auto.h), original never uses the engine
    engine = {"original": False, "auto": info["auto_ngram_route"] == "engine"}.get(cfg["backend"], True)
    assert dm._disk_engine() == engine and info["ngram_route"] == ("engine" if engine else "original")
    eg = _module(table, monkeypatch)
    rng = np.random.default_rng(1)
    for B, L in ((1, 1), (1, 2), (2, 300), (1, 2048)):
        hist = torch.from_numpy(rng.integers(0, 1 << 20, size = (B, CTX + L))).long()
        want = _expect(table, hist)
        n = B * L * N_COLS

        # inline, as forward() stages a miss (decode-sized: hold)
        pin = eg._acquire_pin(n)
        U, t = eg._stage(hist, pin, False, None, 0, B * L < dm.PREFETCH_MIN_TOKENS)
        assert t is None
        _check(pin, U, want)
        pin.held = False

        # the prefetch worker: class 2, not waited for; the consumer promotes and waits
        pin = eg._acquire_pin(n)
        pin.w.zero_()
        fds = eg._fds()
        f = eg.engram_ctx.executor.submit(eg._stage, hist, pin, False, fds, 2, False, False)
        U, t = f.result()
        assert (t is not None) == engine
        eg._finish(t)
        _check(pin, U, want)
        if t is not None:
            with pytest.raises(RuntimeError, match = "released"):
                t.done()
        pin.held = False
    st = ext.disk_stats() if engine else None
    if engine:
        assert st["tickets_live"] == 0 and st["inflight_now"] == 0


@pytest.mark.parametrize("cfg", [{"backend": "pread"}, {"backend": "io_uring"}], ids = ["pread", "io_uring"])
def test_retire_releases_the_ticket(table, monkeypatch, cfg):
    """A prefetch nobody takes: _retire releases its ticket (queued reads cancelled, those in
    flight waited for) before the staging set is free again."""
    ext.disk_engine_configure(dict(cfg, hold_arm_ms = 10000))
    eg = _module(table, monkeypatch)
    rng = np.random.default_rng(2)
    hist = torch.from_numpy(rng.integers(0, 1 << 20, size = (1, CTX + 1024))).long()
    ext.disk_arm_hold()                 # class 2 stays queued: the retire must cancel it
    pin = eg._acquire_pin(1024 * N_COLS)
    fds = eg._fds()
    entry = {"hist": hist, "pin": pin,
             "future": eg.engram_ctx.executor.submit(eg._stage, hist, pin, False, fds, 2, False, False)}
    eg._pending.append(entry)
    entry["future"].result()
    eg._drain_prefetch()
    assert not pin.held and not eg._pending and eg.prefetch_stats["retired"] == 1
    st = ext.disk_stats()
    assert st["tickets_live"] == 0 and st["inflight_now"] == 0 and st["queued_ops_now"] == 0
    assert st["classes"][2]["ops_cancelled"] > 0


def test_route_comes_from_the_extension(table, monkeypatch):
    """Windows: the extension routes to the original gather whatever EXL3_DISK_BACKEND says, and
    the module must follow it (the engine entry points do not exist there)."""
    eg = _module(table, monkeypatch)
    monkeypatch.setenv("EXL3_DISK_BACKEND", "odirect")
    monkeypatch.setattr(dm, "ext", types.SimpleNamespace(
        disk_ngram_route = lambda: "original",
        ngram_gather_cpu = ext.ngram_gather_cpu,
        disk_gather_rows = lambda *a, **k: pytest.fail("disk_gather_rows called on the original route")))
    rng = np.random.default_rng(3)
    hist = torch.from_numpy(rng.integers(0, 1 << 20, size = (1, CTX + 5))).long()
    pin = eg._acquire_pin(5 * N_COLS)
    U, t = eg._stage(hist, pin, False, None, 0, True)
    assert t is None
    _check(pin, U, _expect(table, hist))


@pytest.mark.parametrize("cfg", [{"backend": "pread"}, {"backend": "io_uring"}], ids = ["pread", "io_uring"])
def test_page_cache_modes(table, monkeypatch, cfg):
    """disk io=buffered / direct: the engram's row gathers in the placement's page-cache mode, the same
    bytes either way, inline and through the prefetch worker"""
    ext.disk_engine_configure(cfg)
    eg = _module(table, monkeypatch)
    rng = np.random.default_rng(5)
    for io in (0, 1, -1):
        eg.io_direct = io
        hist = torch.from_numpy(rng.integers(0, 1 << 20, size = (1, CTX + 300))).long()
        want = _expect(table, hist)
        pin = eg._acquire_pin(300 * N_COLS)
        U, t = eg._stage(hist, pin, False, None, 0, False)
        assert t is None
        _check(pin, U, want)
        pin.held = False
        pin = eg._acquire_pin(300 * N_COLS)
        pin.w.zero_()
        U, t = eg.engram_ctx.executor.submit(eg._stage, hist, pin, False, eg._fds(), 2, False, False).result()
        eg._finish(t)
        _check(pin, U, want)
        pin.held = False


def _whole(table, which):
    path, rows, w_off, s_off, buf = table
    if which == 0:
        return torch.from_numpy(buf[w_off : w_off + rows * HEAD_DIM].reshape(rows, HEAD_DIM).copy())
    return torch.from_numpy(buf[s_off : s_off + rows * 8].reshape(rows, 8).copy())


@pytest.mark.parametrize("cfg", [{"backend": "original"}, {"backend": "pread"}, {"backend": "io_uring"}],
                         ids = ["original", "pread", "io_uring"])
@pytest.mark.parametrize("held", [(), (1,), (0, 1)], ids = ["none-whole", "scales-whole", "all-whole"])
def test_whole_tables_and_row_cache(table, monkeypatch, cfg, held):
    """--ngram_ram: the tables held whole serve their rows by index; the streamed ones go through a row
    cache. Repeated gathers hit the cache; every gather, inline or prefetched (the misses land in the
    staging rows and the cache when the consumer waits), has the table's bytes"""
    from exllamav3.modules.ngram_row_cache import RowCache
    ext.disk_engine_configure(cfg)
    eg = _module(table, monkeypatch)
    eg.ram_tables = {t: _whole(table, t) for t in held}
    eg.streamed = [t for t in (0, 1) if t not in held]
    rbs = (HEAD_DIM, HEAD_DIM // 32)
    eg.row_cache = RowCache("engram", [rbs[t] for t in eg.streamed], 16 << 20) if eg.streamed else None
    if not eg.streamed:
        monkeypatch.setattr(dm, "ext", types.SimpleNamespace(
            disk_ngram_route = ext.disk_ngram_route,
            ngram_gather_cpu = lambda *a, **k: pytest.fail("a whole table was read from disk"),
            disk_gather_rows = lambda *a, **k: pytest.fail("a whole table was read from disk")))
    rng = np.random.default_rng(6)
    hists = [torch.from_numpy(rng.integers(0, 1 << 20, size = (1, CTX + L))).long() for L in (1, 300, 2048)]
    for rep in range(2):
        for hist in hists:
            L = hist.shape[1] - CTX
            want = _expect(table, hist)
            pin = eg._acquire_pin(L * N_COLS)
            pin.w.zero_()
            U, t = eg._stage(hist, pin, False, None, 0, L < dm.PREFETCH_MIN_TOKENS)
            eg._finish(t)
            _check(pin, U, want)
            pin.held = False
            pin = eg._acquire_pin(L * N_COLS)
            pin.w.zero_()
            pin.s.zero_()
            U, t = eg.engram_ctx.executor.submit(eg._stage, hist, pin, False, eg._fds(), 2, False, False).result()
            eg._finish(t)
            _check(pin, U, want)
            pin.held = False
    if eg.row_cache is not None:
        st = eg.row_cache.stats()
        assert st["hits"] > 0 and st["misses"] > 0, st
