"""
The VRAM expert cache's lookup kernel (exllamav3_ext/tier/tier_kernels.cu) against the policy core
(tier_policy.cpp, itself equal to exllamav3/model/expert_tier_policy.py): the same calls give the
same records, the same mirror state and a device directory equal to the mirror, exactly.

The runtime runs policy-only here (no extents: nothing is copied; slot addresses are made up and
never dereferenced), deterministic, driven call by call from this thread (step()): the lookup
kernel decides on the GPU, the core applies its record (TierCore::process), the next lookup sees
the returned slots. Covered: the micro-scenarios of doc/expert_tiers_impl_plan.md 3.7 that the
VRAM side decides (S1 LRU and protection, S2 admit=heat, S6 adaptive with a pinned p, S9
starvation, and S3 / S4 / S5 / S8 with their RAM side run by the core), hot pins, spare=0 in place,
routed calls, heat decay across ticks, and seeded random replays up to the V4.1 shape (28 layers x
384 experts, 6,000 slots) at batch sizes 1 to 8 with every admit x evict, the adaptive probability
live and pinned. After every call of the micro-scenarios, and every 97th call of the replays, the
layer's pointer tables hold every selected expert's slot (or staging slot) plus the projection
offsets. Timing of one lookup at the V4.1 shape (thread running, production mode) is printed.

Needs a CUDA GPU and the full extension (exllamav3_ext with exllamav3_ext/tier); on the AI VM,
through /root/rnd/gpu_window.sh:

    python -m pytest -q -s tests/expert_tier/tier_kernel_gpu_.py
"""
import dataclasses
import importlib.util
import os
from pathlib import Path
import random
import sys
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


P = load("_expert_tier_policy_kernel_ref", ROOT / "exllamav3/model/expert_tier_policy.py")
EXT = None
DEV = int(os.environ.get("EXL3_TIER_TEST_DEVICE", "0"))


def ext():
    global EXT
    if EXT is None:
        if not torch.cuda.is_available():
            raise AssertionError("tier_kernel_gpu_: needs a CUDA GPU")
        try:
            from exllamav3.ext import exllamav3_ext
        except Exception as e:
            raise AssertionError(f"tier_kernel_gpu_: the full extension is needed: {e}")
        if not hasattr(exllamav3_ext, "TierGpu"):
            raise AssertionError("tier_kernel_gpu_: this exllamav3_ext has no TierGpu (rebuild it)")
        EXT = exllamav3_ext
    return EXT


POOL_BASE = 0x7F00_0000_0000
STAGE_BASE = 0x7E00_0000_0000
SLOT = 1 << 24


class Rig:
    """A policy-only runtime on the GPU, fed a replay script (the operations of
    expert_tier_policy.replay: call, tick, place, ram_add, heat, seq, cold_fill)"""

    def __init__(self, cfg, L, E, S, R, pins = 0, projections = 3, nstage = None, deterministic = True,
                 spin_us = -1, check_every = 1):
        self.cfg, self.L, self.E, self.S, self.R, self.pins = cfg, L, E, S, R, pins
        self.P = projections
        self.nstage = nstage or E
        self.dev = torch.device("cuda", DEV)
        self.tables = [torch.full((E,), -1, dtype = torch.long, device = self.dev) for _ in range(L * self.P)]
        self.proj_off = [0, 4096, 8192][:self.P]
        gpu = {"device": DEV, "pool": [POOL_BASE + i * SLOT for i in range(S + pins)],
               "staging": [STAGE_BASE + i * SLOT for i in range(self.nstage)], "staging_per_half": self.nstage,
               "tables": [t.data_ptr() for t in self.tables], "proj_off": self.proj_off,
               "deterministic": deterministic, "trace": 1 << 30, "spin_us": spin_us}
        self.h = ext().TierGpu(dataclasses.asdict(cfg), L, E, S, R, pins, None, [], [], {}, gpu)
        self.seq = 0
        self.tokens = 0
        self.dirty = True
        self.calls = 0
        self.check_every = check_every

    def ids_tensor(self, ids):
        return torch.tensor(ids, dtype = torch.long, device = self.dev)

    def op(self, op, tc = None):
        name = op[0]
        if name == "tick":
            self.tokens += op[1]
        elif name == "place":
            self.h.place(op[1], op[2], op[3])
            self.dirty = True
        elif name == "ram_add":
            self.h.ram_add(op[1], bool(op[2]), op[3], op[4])
            self.dirty = True
        elif name == "heat":
            self.h.heat_set(op[1], op[2])
            self.dirty = True
        elif name == "seq":
            self.seq = op[1]
            self.h.set_seq(op[1])
        elif name == "cold_fill":
            self.h.cold_fill(list(op[1]), [])
            self.dirty = False
        elif name == "check":
            pass
        elif name == "call":
            if self.dirty:
                self.h.upload()
                self.dirty = False
            lc, ids = op[1], list(op[2])
            mode = op[3] if len(op) > 3 else P.DECODE
            self.seq += 1
            t = self.ids_tensor(ids)
            self.h.lookup(lc, t, mode, self.seq, self.tokens)
            n = self.h.step(True)
            assert n == 1, n
            recs = self.h.records()
            assert len(recs) == 1
            self.last = recs[0]
            self.calls += 1
            if tc is not None and self.calls % self.check_every == 0:
                self.check_tables(tc, lc, self.last)
        else:
            raise ValueError(name)

    def run(self, script, tc = None):
        out = []
        for op in script:
            self.op(op, tc)
            if op[0] == "call":
                out.append(self.normalize(self.last))
        return out

    @staticmethod
    def normalize(rec):
        seq, lc, mode, ents = rec
        return (seq, lc, mode, [tuple(e) for e in ents])

    def check_tables(self, tc, lc, rec):
        seq, _, mode, ents = rec
        for e in ents:
            key, kind, dst = e[0], e[1], e[2]
            x = key - lc * self.E
            base = (STAGE_BASE + dst * SLOT) if kind == P.TRANSIENT else (POOL_BASE + dst * SLOT)
            for q in range(self.P):
                got = int(self.tables[lc * self.P + q][x].item())
                tc.assertEqual(got, base + self.proj_off[q], f"call {seq}: layer {lc} expert {x} projection {q}")


def reference(cfg, L, E, S, R, pins, script):
    """The native policy core's replay (equal to the Python reference, tests/test_expert_tier_native_.py)"""
    r = ext().tier_policy_replay(dataclasses.asdict(cfg), L, E, S, R, pins, False, script)
    if r["error"]:
        raise AssertionError(f"reference replay failed: {r['error']}")
    recs = [(a, b, c, [tuple(x) for x in ents]) for a, b, c, ents in r["records"]]
    return recs, r["state"], r["counters"]


def tokens_at_last_call(script):
    t, at = 0, 0
    for op in script:
        if op[0] == "tick":
            t += op[1]
        elif op[0] == "call":
            at = t
    return at


def cfg(**kw):
    base = {"admit": "always", "evict": "lru", "policy": "exclusive", "demote": "off", "halflife": 1 << 30}
    base.update(kw)
    return P.PolicyConfig(**base)


class Kernel(unittest.TestCase):

    def same(self, c, L, E, S, R, script, pins = 0, what = "", check_every = 1, projections = 3, nstage = None):
        ref, ref_state, ref_counters = reference(c, L, E, S, R, pins, script)
        rig = Rig(c, L, E, S, R, pins, projections = projections, nstage = nstage, check_every = check_every)
        got = rig.run(script, self)
        self.assertEqual(len(got), len(ref), what)
        for i, (g, r) in enumerate(zip(got, ref)):
            self.assertEqual(g, r, f"{what}: call {i}")
        st = rig.h.state()
        # the mirror's token count is the last record's (a tick after the last call reaches it with
        # the next call)
        ref_state = dict(ref_state, tokens = tokens_at_last_call(script))
        for k in ref_state:
            self.assertEqual(st[k], ref_state[k], f"{what}: state {k}")
        self.assertEqual(dict(rig.h.counters()), dict(ref_counters), what)
        rig.h.verify()
        s = rig.h.stats()
        self.assertEqual((s["overflow"], s["device_error"]), (0, 0), what)
        return rig, got

    def test_micro_scenarios(self):
        calls = lambda lst: [("call", 0, ids) for ids in lst]
        # S1: LRU, protection, spare, ties
        _, got = self.same(cfg(spare = 2), 1, 8, 5, 0, calls([[0, 1], [2, 3], [0, 2], [1, 3], [4, 5]]), what = "S1")
        self.assertEqual([[(e[0], e[1], e[5]) for e in r[3]] for r in got],
                         [[(0, 1, -1), (1, 1, -1)], [(2, 1, -1), (3, 1, 0)], [(0, 1, 1), (2, 0, -1)],
                          [(1, 1, 0), (3, 0, -1)], [(4, 1, 2), (5, 1, 1)]])
        # S2: admit=heat
        self.same(cfg(spare = 2, admit = "heat"), 1, 8, 4, 0, calls([[0, 1], [0, 2], [3, 0], [2, 4], [2, 5], [2, 4]]),
                  what = "S2")
        # S3: one promotion from RAM under every policy x demote
        for pol in ("exclusive", "lazy-exclusive"):
            for dem in ("off", "swap", "heat", "all"):
                for h in (5, 1):
                    seed = [("place", 0, 0, 1), ("place", 1, 1, 2), ("seq", 2), ("ram_add", 2, 0, -1, 0),
                            ("ram_add", 3, 0, -1, 0)] + [("heat", k, u * P.Q16) for k, u in ((0, h), (1, 1), (2, 4), (3, 2))]
                    self.same(cfg(spare = 1, policy = pol, demote = dem), 1, 8, 3, 2, seed + [("call", 0, [2])],
                              what = f"S3 {pol} {dem} {h}")
        # S4: disk experts=off
        seed = [("place", 0, 0, 1), ("place", 1, 1, 1), ("seq", 1)] + [("ram_add", e, 0, -1, 0) for e in (2, 3, 4, 5)]
        self.same(cfg(spare = 2, policy = "lazy-exclusive", demote = "swap", disk_off = True), 1, 6, 4, 4,
                  seed + calls([[2, 3], [4, 0], [5, 1], [2, 4], [0, 1], [3, 5]]), what = "S4")
        # S5: inclusive
        self.same(cfg(spare = 2, policy = "inclusive"), 1, 8, 4, 5, calls([[0, 1], [2, 0], [3, 4], [1, 2]]), what = "S5")
        # S6: adaptive with a pinned probability
        for p in (0.0, 1.0):
            self.same(cfg(spare = 2, admit = "adaptive", p = p), 1, 8, 4, 0, calls([[0, 1], [2, 3], [4, 0]]),
                      what = f"S6 {p}")
        # S7: heat across ticks and a routed call
        s7 = [("call", 0, [0, 1]), ("tick", 1), ("call", 0, [0]), ("tick", 1), ("call", 0, [1]), ("tick", 3),
              ("call", 0, [1, 1, 2], P.ROUTED), ("tick", 35), ("call", 0, [3])]
        self.same(cfg(spare = 1, halflife = 2), 1, 8, 4, 0, s7, what = "S7")
        # S8: evict=lfu, demote=heat
        self.same(cfg(spare = 2, policy = "lazy-exclusive", demote = "heat", evict = "lfu"), 1, 8, 4, 1,
                  calls([[0, 1], [0, 2], [0, 3], [3, 4]]), what = "S8")
        # S9: starvation (the victim is still RETIRING: deferred returns are the production mode; here
        # with spare = 1 and the slot taken)
        self.same(cfg(spare = 1), 1, 8, 3, 0, calls([[0, 1], [2, 3]]), what = "S9")

    def test_pins_inplace_routed_gateless(self):
        # hot pins (slots after the pool) are hits and never victims
        seed = [("place", 3, 4, 0), ("place", 11, 5, 0)]
        script = seed + [("call", lc, ids) for lc, ids in ((0, [3, 1, 2]), (1, [3, 0]), (0, [4, 5, 6]), (0, [3, 7]),
                                                             (1, [0, 1, 2, 3, 4]))]
        self.same(cfg(spare = 1), 2, 8, 4, 0, script, pins = 2, what = "pins")
        # spare=0: victims replaced in place
        self.same(cfg(spare = 0), 1, 8, 3, 0, [("call", 0, ids) for ids in ([0, 1, 2], [3], [4, 0], [5, 6, 7], [0])],
                  what = "spare=0")
        # routed calls: hits keep their slots, misses stage in order, nothing is stamped
        script = [("call", 0, [0, 1]), ("call", 0, [5, 0, 6, 5, 7, 1], P.ROUTED), ("call", 0, [2, 3]),
                  ("call", 0, list(range(8)) * 3, P.ROUTED)]
        self.same(cfg(spare = 1), 1, 8, 5, 0, script, what = "routed")
        # two projections (gateless experts)
        self.same(cfg(spare = 1), 1, 8, 3, 0, [("call", 0, [0, 1]), ("call", 0, [2, 3])], projections = 2,
                  what = "gateless")

    def test_random_replays(self):
        rnd = random.Random(20260925)
        shapes = [(1, 16, 5), (2, 32, 9), (4, 64, 40), (3, 128, 200), (2, 384, 700), (28, 384, 6000)]
        combos = [(a, e) for a in ("always", "heat", "adaptive") for e in ("lru", "lfu")]
        n_calls = int(os.environ.get("EXL3_TIER_KERNEL_CALLS", 20000))
        for si, (L, E, S) in enumerate(shapes):
            for ci, (admit, evict) in enumerate(combos):
                seed = si * 100 + ci
                r = random.Random(seed)
                spare = r.choice([1, 2, 4, 8]) if S > 8 else 1
                spare = min(spare, S - 1)
                pins = r.choice([0, 0, 2, 4]) if S > 16 else 0
                topk = r.choice([2, 4, 6, 8])
                p = r.choice([None, 0.3]) if admit == "adaptive" else None
                c = cfg(spare = spare, admit = admit, evict = evict, policy = "lazy-exclusive", demote = "swap",
                        halflife = r.choice([8, 64, 256]), seed = seed, adapt_every = 97, p = p)
                R = min(L * E // 2, 64)
                script = replay_script(r, L, E, S, n_calls if S < 6000 else 2 * n_calls, topk, pins,
                                       drift = seed % 2 == 0)
                t0 = time.monotonic()
                self.same(c, L, E, S, R, script, pins = pins, what = f"shape {L}x{E} S={S} {admit}/{evict} p={p}",
                          check_every = 97)
                print(f"\n  replay {L}x{E} S={S} pins={pins} spare={spare} top-{topk} {admit}/{evict}: "
                      f"{sum(1 for o in script if o[0] == 'call')} calls in {time.monotonic() - t0:.1f} s", end = "")

    def test_lookup_timing(self):
        """Per-call cost at the V4.1 shape (28 x 384, 6,000 slots): production mode, the tier thread
        running, all-hit calls (the kernel releases itself) and calls with 1, 6 and 48 misses"""
        L, E, S = 28, 384, 6000
        c = cfg(spare = 8, policy = "lazy-exclusive", demote = "off", admit = "always")
        rig = Rig(c, L, E, S, 0, deterministic = False)
        order = P.round_robin_order(L, E)
        rig.h.cold_fill(order, [])
        rig.h.start()
        try:
            held = {k for k in order[:S - 8]}
            dev = rig.dev
            hot = [k for k in order[:S - 8] if k // E == 5]
            cold = [k for k in order if k not in held and k // E == 5]
            def run(kind, n_miss, reps = 400):
                ts = []
                ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                for i in range(reps):
                    rows = 8 if n_miss == 48 else 1
                    k = 6 * rows
                    miss = [cold[(i * 48 + j) % len(cold)] % E for j in range(n_miss)]
                    hit = [hot[j % len(hot)] % E for j in range(k - n_miss)]
                    ids = torch.tensor(miss + hit, dtype = torch.long, device = dev)
                    ev0.record()
                    rig.seq += 1
                    rig.h.lookup(5, ids, 0, rig.seq, i)
                    ev1.record()
                    ev1.synchronize()
                    ts.append(ev0.elapsed_time(ev1) * 1000)
                ts.sort()
                print(f"\n  lookup {kind}: median {ts[len(ts) // 2]:.1f} us, p99 {ts[int(len(ts) * 0.99)]:.1f} us", end = "")
            run("all hits", 0)
            run("1 miss", 1)
            run("6 misses", 6)
            run("48 misses (bsz 8)", 48)
            rig.h.drain()
            rig.h.verify()
        finally:
            rig.h.stop()


def replay_script(r, L, E, S, n, topk, pins, drift):
    """Calls over L layers in forward order, batch sizes 1..8 (some routed ones of 9..40 rows), Zipf
    routing that drifts, a tick per token; `pins` hot pins placed first (slots after the pool)"""
    script = [("place", k, S + i, 0) for i, k in enumerate(r.sample(range(L * E), pins))]
    weights = [1.0 / (i + 1) ** r.choice([0.6, 1.0]) for i in range(E)]
    perm = list(range(E))
    calls = 0
    while calls < n:
        if drift and calls % 500 == 0:
            r.shuffle(perm)
        routed = r.random() < 0.05
        rows = r.randint(9, 40) if routed else r.choice([1, 1, 1, 2, 4, 8])
        for lc in range(L):
            ids = []
            for _ in range(rows):
                row = set()
                while len(row) < topk:
                    row.add(perm[r.choices(range(E), weights)[0]])
                ids += list(row)
            script.append(("call", lc, ids, P.ROUTED if routed else P.DECODE))
            calls += 1
            if calls >= n:
                break
        if not routed:
            script.append(("tick", 1))
    return script


if __name__ == "__main__":
    unittest.main()
