"""
The VRAM expert cache end to end on a GPU (exllamav3_ext/tier/tier_gpu.cpp): the lookup kernel, the
tier thread's policy, and every byte the copy engines move.

A synthetic checkpoint (tests/test_expert_tier_ram_.py's: unaligned experts, one split across an
override file, one with a foreign tensor in its span) is read in place into the RAM tier and the
VRAM pool; then call by call, in deterministic mode, over every policy x demote x admit, with the
disk off, hot pins, spare=0 in place and deferred (production) returns:
  - the records equal the policy core's replay of the same calls (same cold fill);
  - every VALID or PINNED pool slot, every slot a transient was staged into and every RAM tier slot
    holds exactly its expert's checkpoint trellis bytes (read back from the GPU and the arena);
  - the device directory equals the mirror (verify()).
Production mode (the tier thread, returns when a demotion lands, all-hit calls not waiting for the
host) runs the same traces with byte checks after a drain; a pageable device-to-host copy on the
calling thread right after every call (the synchronous copy that holds the driver's locks while it
waits for the stream) must not stall the tier. With DSV41_MODEL_DIR, two real DeepSeek-V4.1 layers give the
promotion cost per expert (RAM extent layout: three copies from unaligned sources; compact layout
after a demotion: one), the lookup-to-ready latency of a decode call with 1 and 6 RAM misses, and
the promotion rate while demotions run.

    python -m pytest -q -s tests/expert_tier/tier_runtime_gpu_.py      (a GPU window on the AI VM)
"""
import dataclasses
import importlib.util
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


P = load("_exl3_torch_free_expert_tier_policy", ROOT / "exllamav3/model/expert_tier_policy.py")
X = load("_expert_extents_for_ram", ROOT / "exllamav3/model/expert_extents.py")
H = load("_expert_tier_host_subject", ROOT / "exllamav3/model/expert_tier_host.py")
RT = load("_tier_ram_test_helpers", ROOT / "tests/test_expert_tier_ram_.py")
DEV = int(os.environ.get("EXL3_TIER_TEST_DEVICE", "0"))
EXT = None


def ext():
    global EXT
    if EXT is None:
        if not torch.cuda.is_available():
            raise AssertionError("tier_runtime_gpu_: needs a CUDA GPU")
        from exllamav3.ext import exllamav3_ext
        if not hasattr(exllamav3_ext, "TierGpu"):
            raise AssertionError("tier_runtime_gpu_: this exllamav3_ext has no TierGpu (rebuild it)")
        EXT = exllamav3_ext
    return EXT


def cfg(**kw):
    base = {"admit": "always", "evict": "lru", "policy": "lazy-exclusive", "demote": "swap", "halflife": 64}
    base.update(kw)
    return P.PolicyConfig(**base)


class Gpu:
    """One GPU's cache over an ExtentIndex: pool + pins, one staging half, fake layer tables"""

    def __init__(self, index, L, E, policy, S, R, pins = 0, deterministic = True, slab = 6, chunk_bytes = None,
                 spin_us = -1, compact = False):
        g = index.geometry
        self.L, self.E, self.S, self.R, self.pins = L, E, S, R, pins
        self.dev = torch.device("cuda", DEV)
        self.vs = g.vram_slot_bytes
        stride = -(-self.vs // 256) * 256
        self.pool = torch.zeros(((S + pins) * stride,), dtype = torch.uint8, device = self.dev)
        self.staging = torch.zeros((E * stride,), dtype = torch.uint8, device = self.dev)
        self.stride = stride
        P_ = len(g.proj_bytes)
        self.tables = [torch.zeros(E, dtype = torch.long, device = self.dev) for _ in range(L * P_)]
        files, extents, geometry = H.native_extents(index)
        host = {"slab_slots": slab, "fill_inflight": 4, "chunk_bytes": chunk_bytes or (1 << 30),
                "prefault_threads": 2, "compact_fill": compact}
        gpu = {"device": DEV, "pool": [self.pool.data_ptr() + i * stride for i in range(S + pins)],
               "staging": [self.staging.data_ptr() + i * stride for i in range(E)], "staging_per_half": E,
               "tables": [t.data_ptr() for t in self.tables], "proj_off": list(g.proj_off),
               "deterministic": deterministic, "trace": 1 << 30, "spin_us": spin_us}
        self.h = ext().TierGpu(dataclasses.asdict(policy), L, E, S, R, pins, geometry, files, extents, host, gpu)
        self.seq = 0
        self.tokens = 0

    def slot(self, s):
        return bytes(self.pool[s * self.stride : s * self.stride + self.vs].cpu().numpy())

    def stage(self, i):
        return bytes(self.staging[i * self.stride : i * self.stride + self.vs].cpu().numpy())

    def call(self, lc, ids, mode = 0, step = True):
        self.seq += 1
        self.h.lookup(lc, torch.tensor(ids, dtype = torch.long, device = self.dev), mode, self.seq, self.tokens)
        if step:
            assert self.h.step(True) == 1
            recs = self.h.records()
            assert len(recs) == 1
            return recs[0]


def trace_ops(rng, L, E, n, topk = 3):
    w = [1.0 / (i + 1) ** 0.9 for i in range(E)]
    perm = [rng.sample(range(E), E) for _ in range(L)]
    ops = []
    for i in range(n):
        lc = i % L
        if lc == 0:
            ops.append(("tick", 1))
        if rng.random() < 0.05:
            ops.append(("call", lc, [rng.randrange(E) for _ in range(rng.randrange(9, 40))], P.ROUTED))
        else:
            rows = rng.choice((1, 1, 2, 4))
            ids = []
            for _ in range(rows):
                row = set()
                while len(row) < topk:
                    row.add(perm[lc][rng.choices(range(E), w)[0]])
                ids += list(row)
            ops.append(("call", lc, ids, P.DECODE))
    return ops


class Runtime(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix = "tier_gpu_", dir = os.environ.get("EXL3_DISK_TEST_DIR"))
        cls.ck = RT.Ckpt(cls.tmp.name)
        ext().disk_engine_configure({"backend": "io_uring"})

    @classmethod
    def tearDownClass(cls):
        ext().disk_engine_shutdown()
        cls.tmp.cleanup()

    def bytes_ok(self, g, where):
        ck = self.ck
        st = g.h.state()
        for s, (state, key) in enumerate(zip(st["vram_state"], st["vram_key"])):
            if state in (P.VALID, P.PINNED, P.RETIRING) and key >= 0:
                self.assertEqual(g.slot(s), ck.expected[key], f"{where}: pool slot {s} key {key} state {state}")
        for r, key in enumerate(st["ram_key"]):
            if key >= 0:
                self.assertEqual(bytes(g.h.ram_trellis(r).numpy()), ck.expected[key], f"{where}: RAM slot {r} key {key}")

    def run_det(self, policy, S, R, ops, pins = (), what = "", check_every = 1, compact = False):
        """Deterministic: records vs the core's replay, bytes after every call"""
        ck = self.ck
        g = Gpu(ck.index, ck.L, ck.E, policy, S, R, pins = len(pins), compact = compact)
        order = P.round_robin_order(ck.L, ck.E, set(pins))
        g.h.cold_fill(order, list(pins))
        script = [("place", k, S + i, 0) for i, k in enumerate(pins)] + [("cold_fill", order)] + list(ops)
        ref = ext().tier_policy_replay(dataclasses.asdict(policy), ck.L, ck.E, S, R, len(pins), False, script)
        self.assertEqual(ref["error"], "", what)
        refs = [(a, b, c, [tuple(x) for x in ents]) for a, b, c, ents in ref["records"]]
        self.bytes_ok(g, f"{what} cold fill")
        n = 0
        for i, op in enumerate(ops):
            if op[0] == "tick":
                g.tokens += op[1]
                continue
            rec = g.call(op[1], op[2], op[3] if len(op) > 3 else 0)
            rec = (rec[0], rec[1], rec[2], [tuple(e) for e in rec[3]])
            self.assertEqual(rec, refs[n], f"{what}: call {n}")
            for e in rec[3]:
                if e[1] == P.TRANSIENT:
                    self.assertEqual(g.stage(e[2]), ck.expected[e[0]], f"{what}: call {n} transient {e[0]}")
                else:
                    self.assertEqual(g.slot(e[2]), ck.expected[e[0]], f"{what}: call {n} slot {e[2]}")
            if n % check_every == 0:
                self.bytes_ok(g, f"{what}: call {n}")
            n += 1
        g.h.drain()
        g.h.verify()
        self.bytes_ok(g, f"{what}: end")
        self.assertEqual(dict(g.h.counters()), dict(ref["counters"]), what)
        return g

    def test_micro_and_combinations(self):
        calls = lambda lst: [("call", 0, ids, P.DECODE) for ids in lst]
        self.run_det(cfg(spare = 2, policy = "exclusive", demote = "off"), 5, 0,
                     calls([[0, 1], [2, 3], [0, 2], [1, 3], [4, 5]]), what = "S1")
        self.run_det(cfg(spare = 2, policy = "inclusive", demote = "off"), 4, 8, calls([[0, 1], [2, 0], [3, 4], [1, 2]]),
                     what = "S5")
        rng = random.Random(7)
        ops = trace_ops(rng, self.ck.L, self.ck.E, 240)
        for pol in ("exclusive", "lazy-exclusive", "inclusive"):
            for dem in ("swap", "heat", "all", "off"):
                if pol == "inclusive" and dem != "off":
                    continue
                for adm in ("always", "heat", "adaptive"):
                    S, R = 10, 14 if pol != "inclusive" else 20
                    self.run_det(cfg(spare = 2, policy = pol, demote = dem, admit = adm, adapt_every = 37), S, R, ops,
                                 what = f"{pol}/{dem}/{adm}", check_every = 7)

    def test_disk_off_pins_inplace(self):
        ck = self.ck
        rng = random.Random(11)
        ops = trace_ops(rng, ck.L, ck.E, 300)
        N = ck.L * ck.E
        # disk experts=off: VRAM + RAM hold every expert, every victim goes back to RAM
        S, spare = 12, 3
        self.run_det(cfg(spare = spare, disk_off = True), S, N - (S - spare), ops, what = "disk off", check_every = 5)
        # the same with the RAM tier compacted after the cold fill (one aligned copy per promotion)
        g = self.run_det(cfg(spare = spare, disk_off = True), S, N - (S - spare), ops, what = "disk off, compact",
                         check_every = 5, compact = True)
        self.assertEqual(g.h.stats()["compacted"], N - (S - spare))
        # hot pins: two per GPU, never victims
        self.run_det(cfg(spare = 2), 8, 12, ops, pins = (3, 17), what = "pins", check_every = 5)
        # spare=0: victims replaced in place (demote=off)
        self.run_det(cfg(spare = 0, demote = "off"), 8, 12, ops, what = "spare=0", check_every = 5)

    def run_prod(self, policy, S, R, ops, what, compact = False):
        """Production: the thread, deferred returns, all-hit calls not waiting for the host"""
        ck = self.ck
        g = Gpu(ck.index, ck.L, ck.E, policy, S, R, deterministic = False, compact = compact)
        order = P.round_robin_order(ck.L, ck.E)
        g.h.cold_fill(order, [])
        g.h.start()
        for op in ops:
            if op[0] == "tick":
                g.tokens += op[1]
                continue
            g.call(op[1], op[2], op[3], step = False)
        torch.cuda.synchronize(g.dev)
        g.h.drain()
        g.h.verify()
        self.bytes_ok(g, what)
        s = g.h.stats()
        self.assertEqual((s["overflow"], s["device_error"]), (0, 0), what)
        self.assertEqual(g.h.state()["pending"], [], what)
        g.h.stop()
        return g, s

    def test_production(self):
        ck = self.ck
        rng = random.Random(5)
        ops = trace_ops(rng, ck.L, ck.E, 3000)
        for pol, dem, compact in (("lazy-exclusive", "swap", False), ("exclusive", "all", True),
                                  ("lazy-exclusive", "heat", False)):
            _, s = self.run_prod(cfg(spare = 3, policy = pol, demote = dem, admit = "adaptive"), 10, 14, ops,
                                 f"production {pol}/{dem}", compact)
            print(f"\n  production {pol}/{dem} compact={int(compact)}: {s['records']} records, "
                  f"{s['host_records']} with host work, {s['returns']} returns, {s['plan_commands']} commands, "
                  f"max lag {s['max_lag']}", end = "")
        # a stricter check of the data path in production: every call's staged / cached bytes, read on
        # the compute stream right after the lookup (the stream waited for ready)
        g = Gpu(ck.index, ck.L, ck.E, cfg(spare = 3, policy = "lazy-exclusive", demote = "swap"), 10, 14,
                deterministic = False)
        g.h.cold_fill(P.round_robin_order(ck.L, ck.E), [])
        g.h.start()
        base_p, base_s = g.pool.data_ptr(), g.staging.data_ptr()
        checked = 0
        for op in ops[:1500]:
            if op[0] == "tick":
                g.tokens += op[1]
                continue
            g.call(op[1], op[2], op[3], step = False)
            # a synchronous pageable copy on this thread while the call may still wait for the host
            probe = g.tables[op[1] * 3][:385].tolist()
            self.assertEqual(len(probe), min(385, ck.E))
            for x in set(op[2]):
                a = int(g.tables[op[1] * 3][x].item())
                if base_p <= a < base_p + g.pool.numel():
                    got = bytes(g.pool[a - base_p : a - base_p + g.vs].cpu().numpy())
                else:
                    got = bytes(g.staging[a - base_s : a - base_s + g.vs].cpu().numpy())
                self.assertEqual(got, ck.expected[op[1] * ck.E + x], f"production data path: layer {op[1]} expert {x}")
                checked += 1
        g.h.drain()
        g.h.verify()
        g.h.stop()
        print(f"\n  production data path: {checked} expert reads checked", end = "")

    @unittest.skipUnless(os.environ.get("DSV41_MODEL_DIR"), "DSV41_MODEL_DIR is not set")
    def test_v41_promotion_cost(self):
        d = os.environ["DSV41_MODEL_DIR"]
        paths = sorted(str(p) for p in Path(d).glob("*.safetensors"))
        layers = (12, 25)
        keys = [tuple(f"layers.{l}.ffn.experts.{e}.{p}" for p in ("w1", "w3", "w2")) for l in layers for e in range(384)]
        index = X.build_extent_index(RT.FakeSTC(paths), keys)
        L, E, S, R = 2, 384, 32, 400
        for compact, demote in ((False, "swap"), (True, "swap"), (True, "off")):
            self.v41_run(index, L, E, S, R, compact, demote)

    def v41_run(self, index, L, E, S, R, compact, demote):
        pol = cfg(spare = 8, policy = "exclusive", demote = demote, admit = "always", halflife = 1 << 20)
        g = Gpu(index, L, E, pol, S, R, deterministic = False, slab = 8, compact = compact)
        t0 = time.monotonic()
        g.h.cold_fill(P.round_robin_order(L, E), [])
        s = g.h.stats()
        print(f"\n  V4.1 cold fill (compact={int(compact)}): {s['cold_bytes'] / 1e9:.2f} GB in "
              f"{time.monotonic() - t0:.2f} s (compaction {s['compact_ns'] / 1e9:.2f} s), pinned "
              f"{s['pinned_bytes'] / 2**30:.2f} GiB in {s['pin_ns'] / 1e9:.2f} s", end = "")
        g.h.start()
        st = g.h.state()
        in_ram = [k for k in st["ram_key"] if k >= 0]
        ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        for nmiss in (1, 6):
            ts = []
            for rep in range(60):
                st = g.h.state()
                ram = [k for k in st["ram_key"] if k >= 0 and k // E == 0]
                miss = ram[(rep * 7) % max(1, len(ram) - nmiss):][:nmiss]
                ids = [k % E for k in miss]
                torch.cuda.synchronize(g.dev)
                ev0.record()
                g.call(0, ids, 0, step = False)
                ev1.record()
                ev1.synchronize()
                ts.append(ev0.elapsed_time(ev1) * 1000)
                g.h.drain()
            ts.sort()
            per = ts[len(ts) // 2] / nmiss
            print(f"\n  V4.1 decode call with {nmiss} RAM miss(es), compact={int(compact)} demote={demote}: median "
                  f"{ts[len(ts) // 2]:.0f} us ({per:.0f} us per promoted expert, {g.vs / per / 1e3:.1f} GB/s), "
                  f"p90 {ts[int(len(ts) * 0.9)]:.0f} us", end = "")
        s = g.h.stats()
        print(f"\n  V4.1 copies: H2D {s['h2d_copies']} ({s['h2d_bytes'] / 1e9:.2f} GB), D2H {s['d2h_copies']} "
              f"({s['d2h_bytes'] / 1e9:.2f} GB)", end = "")
        g.h.drain()
        g.h.verify()
        g.h.stop()
        del g
        torch.cuda.empty_cache()

    def test_copy_rates(self):
        """The copy engines as the tier uses them: 4,423,680-byte H2D copies (one V4.1 projection) from
        page-locked memory at an aligned and an odd source offset (the extent layout's trellis lies at
        odd offsets), and the H2D rate kept while D2H copies run beside it at 1:1 and 1:20"""
        n, reps = 4423680, 96
        src = torch.empty((reps * (n + 4096) + 64,), dtype = torch.uint8, pin_memory = True)
        back = torch.empty((reps * n,), dtype = torch.uint8, pin_memory = True)
        dst = torch.empty((reps * n,), dtype = torch.uint8, device = DEV)
        dsrc = torch.empty((reps * n,), dtype = torch.uint8, device = DEV)
        s_h2d = torch.cuda.Stream(device = DEV, priority = -1)
        s_d2h = torch.cuda.Stream(device = DEV, priority = 0)

        def h2d(off, d2h_every = 0):
            torch.cuda.synchronize(DEV)
            t0 = time.perf_counter()
            with torch.cuda.stream(s_h2d):
                for i in range(reps):
                    a = i * (n + 4096) + off
                    dst[i * n : (i + 1) * n].copy_(src[a : a + n], non_blocking = True)
            if d2h_every:
                with torch.cuda.stream(s_d2h):
                    for i in range(0, reps, d2h_every):
                        back[i * n : (i + 1) * n].copy_(dsrc[i * n : (i + 1) * n], non_blocking = True)
            s_h2d.synchronize()
            t = time.perf_counter() - t0
            torch.cuda.synchronize(DEV)
            return reps * n / t / 1e9

        for _ in range(2):
            h2d(0)
        a0 = max(h2d(0) for _ in range(3))
        a11 = max(h2d(11) for _ in range(3))
        a4k = max(h2d(4096 - 7) for _ in range(3))
        d1 = max(h2d(0, 1) for _ in range(3))
        d20 = max(h2d(0, 20) for _ in range(3))
        print(f"\n  H2D 4.4 MB copies: aligned {a0:.1f} GB/s, source offset 11 {a11:.1f} GB/s ({a11 / a0:.2f}x), "
              f"offset 4089 {a4k:.1f} GB/s; with D2H 1:1 {d1:.1f} GB/s ({d1 / a0:.2f}x), 1:20 {d20:.1f} GB/s "
              f"({d20 / a0:.2f}x)", end = "")


if __name__ == "__main__":
    unittest.main()
