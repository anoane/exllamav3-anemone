"""
The native expert tier policy core (exllamav3_ext/tier/tier_policy.cpp) against the Python
reference (exllamav3/model/expert_tier_policy.py): the same script of calls gives the same records,
the same actions, the same counters and the same final state, exactly (heat, draws and the
adaptive probability included), on the micro-scenarios of doc/expert_tiers_impl_plan.md 3.7 and on
200 random configurations (EXL3_TIER_RANDOM_CALLS operations each, default 1,000).

Needs the extension: EXL3_TIER_TEST_MINI=1 builds the small one of tests/expert_tier (C++ only;
EXL3_TIER_MINI_BUILD_DIR sets its build directory), otherwise exllamav3_ext is used.

    EXL3_TIER_TEST_MINI=1 python -m pytest tests/test_expert_tier_native_.py
"""
import dataclasses
import importlib.util
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "expert_tier"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


P = load("_expert_tier_policy_native_ref", ROOT / "exllamav3/model/expert_tier_policy.py")
TP = load("_expert_tier_policy_tests", ROOT / "tests/test_expert_tier_policy_.py")
EXT = None


def ext():
    global EXT
    if EXT is None:
        try:
            from mini_ext import load_any
            EXT = load_any()
        except Exception as e:
            raise AssertionError(f"the tier's native code is needed (EXL3_TIER_TEST_MINI=1 builds a small extension): {e}")
    return EXT


def python_run(cfg, L, E, S, R, pins, defer, script):
    core = P.TierCore(cfg, L, E, S, R, pins, defer)
    records, actions, error = [], [], ""
    try:
        idx = 0
        for i, op in enumerate(script):
            out = P.replay(core, [op])
            for rec, acts in out:
                if rec is not None:
                    records.append((rec.seq, rec.lc, rec.mode, [x.astuple() for x in rec.entries]))
                actions.append((i, [tuple(a) for a in acts]))
            idx += 1
    except (P.TierError, AssertionError) as e:
        error = str(e) or "error"
    v, m = core.vram, core.ram
    state = {
        "vram_state": list(v.state), "vram_key": list(v.key), "vram_stamp": list(v.stamp), "slot_of": list(v.slot_of),
        "vram_free": list(v.free), "occupied": v.occupied, "access": v.access,
        "ram_key": list(m.key), "ram_dup": [int(x) for x in m.dup], "ram_compact": [int(x) for x in m.compact],
        "ram_stamp": list(m.stamp), "ram_of": list(m.of), "ram_free": list(m.free), "own": list(m.lists[False]),
        "dups": list(m.lists[True]), "draws": m.draws, "heat_v": list(core.heat.v), "heat_ep": list(core.heat.ep),
        "tokens": core.heat.tokens, "p": core.adapt.p, "seq": core.seq, "pending": list(core.pending),
    }
    return {"records": records, "actions": actions, "counters": dict(core.c), "state": state, "error": error}


def native_run(cfg, L, E, S, R, pins, defer, script):
    d = dataclasses.asdict(cfg)
    r = ext().tier_policy_replay(d, L, E, S, R, pins, defer, script)
    r["records"] = [(a, b, c, [tuple(x) for x in ents]) for a, b, c, ents in r["records"]]
    r["actions"] = [(i, [tuple(x) for x in acts]) for i, acts in r["actions"]]
    return r


class Parity(unittest.TestCase):

    def same(self, cfg, L, E, S, R, script, pins = 0, defer = False, what = ""):
        py = python_run(cfg, L, E, S, R, pins, defer, script)
        nat = native_run(cfg, L, E, S, R, pins, defer, script)
        self.assertEqual(bool(py["error"]), bool(nat["error"]), f"{what}: python {py['error']!r} native {nat['error']!r}")
        self.assertEqual(py["records"], nat["records"], what)
        self.assertEqual(py["actions"], nat["actions"], what)
        self.assertEqual(py["counters"], nat["counters"], what)
        for k in py["state"]:
            self.assertEqual(py["state"][k], nat["state"][k], f"{what}: state {k}")
        return py

    def cfg(self, **kw):
        return P.PolicyConfig(**{"admit": "always", "evict": "lru", "policy": "exclusive", "demote": "off",
                                 "halflife": 1 << 30, **kw})

    def test_micro_scenarios(self):
        calls = lambda lst: [("call", 0, ids) for ids in lst]
        self.same(self.cfg(spare = 2), 1, 8, 5, 0, calls([[0, 1], [2, 3], [0, 2], [1, 3], [4, 5]]), what = "S1")
        self.same(self.cfg(spare = 2, admit = "heat"), 1, 8, 4, 0,
                  calls([[0, 1], [0, 2], [3, 0], [2, 4], [2, 5], [2, 4]]), what = "S2")
        for pol in ("exclusive", "lazy-exclusive"):
            for dem in ("off", "swap", "heat", "all"):
                for h in (5, 1):
                    seed = [("place", 0, 0, 1), ("place", 1, 1, 2), ("seq", 2), ("ram_add", 2, 0, -1, 0),
                            ("ram_add", 3, 0, -1, 0)] + [("heat", k, u * P.Q16) for k, u in ((0, h), (1, 1), (2, 4), (3, 2))]
                    self.same(self.cfg(spare = 1, policy = pol, demote = dem), 1, 8, 3, 2,
                              seed + [("call", 0, [2]), ("check", 1)], what = f"S3 {pol} {dem} {h}")
        seed = [("place", 0, 0, 1), ("place", 1, 1, 1), ("seq", 1)] + [("ram_add", e, 0, -1, 0) for e in (2, 3, 4, 5)]
        self.same(self.cfg(spare = 2, policy = "lazy-exclusive", demote = "swap", disk_off = True), 1, 6, 4, 4,
                  seed + calls([[2, 3], [4, 0], [5, 1], [2, 4], [0, 1], [3, 5]]), what = "S4")
        self.same(self.cfg(spare = 2, policy = "inclusive"), 1, 8, 4, 5, calls([[0, 1], [2, 0], [3, 4], [1, 2]]), what = "S5")
        for p in (0.0, 1.0):
            self.same(self.cfg(spare = 2, admit = "adaptive", p = p), 1, 8, 4, 0, calls([[0, 1], [2, 3], [4, 0]]),
                      what = f"S6 {p}")
        s7 = [("call", 0, [0, 1]), ("tick", 1), ("call", 0, [0]), ("tick", 4), ("call", 0, [1, 1, 2], P.ROUTED),
              ("tick", 35)]
        py = self.same(self.cfg(spare = 1, halflife = 2), 1, 8, 4, 0, s7, what = "S7")
        self.assertEqual(py["state"]["heat_v"][:3], [131072, 24576, 4096])     # stored, decayed on read
        self.assertEqual(py["state"]["heat_ep"][:3], [0, 2, 2])
        self.same(self.cfg(spare = 2, evict = "lfu", policy = "lazy-exclusive", demote = "heat"), 1, 8, 4, 1,
                  calls([[0, 1], [0, 2], [0, 3], [3, 4]]), what = "S8")
        self.same(self.cfg(spare = 1), 1, 8, 3, 0, calls([[0, 1], [2, 3]]), what = "S9")

    def test_semantics_scenarios(self):
        self.same(self.cfg(spare = 0, policy = "lazy-exclusive"), 1, 8, 2, 2,
                  [("call", 0, x) for x in ([0, 1], [2], [3, 0])], what = "spare=0")
        self.same(P.PolicyConfig(spare = 2, admit = "always", policy = "exclusive", demote = "all", halflife = 1 << 30),
                  1, 8, 5, 4, [("call", 0, [0, 1, 2]), ("call", 0, [3]), ("check", 0), ("call", 0, [0, 4]),
                               ("complete",), ("check", 1)], defer = True, what = "rescue")
        self.same(P.PolicyConfig(spare = 1, admit = "always", policy = "lazy-exclusive", halflife = 1 << 30,
                                 refill_inflight = 2), 2, 8, 4, 4,
                  [("cold_fill", P.round_robin_order(2, 8)), ("layer", 0, [4, 5, 6, 4, 2]), ("check", 1)],
                  what = "layer mode")
        # a failing invariant (a duplicate under exclusive, seeded) is an error on both sides
        py = self.same(self.cfg(spare = 1), 1, 8, 3, 2, [("place", 0, 0, 0), ("ram_add", 0, 1, -1, 0), ("check", 1)],
                       what = "error")
        self.assertTrue(py["error"])

    def test_random(self):
        configs = int(os.environ.get("EXL3_TIER_RANDOM_CONFIGS", "200"))
        calls = int(os.environ.get("EXL3_TIER_RANDOM_CALLS", "1000"))
        rng = random.Random(4242)
        for i in range(configs):
            cfg, L, E, S, R = TP.random_config(rng)
            defer = rng.random() < 0.25
            script = [("cold_fill", P.round_robin_order(L, E))]
            for op in TP.random_script(rng, L, E, calls):
                script.append(op)
                if defer and rng.random() < 0.3:
                    script.append(("complete",))
                if rng.random() < 0.05:
                    script.append(("check", int(not defer)))
            script += [("complete",), ("check", 1)]
            with self.subTest(i = i, cfg = cfg, L = L, E = E, S = S, R = R, defer = defer):
                py = self.same(cfg, L, E, S, R, script, defer = defer, what = f"random {i}")
                self.assertFalse(py["error"])

    def test_config_refused(self):
        with self.assertRaisesRegex(Exception, "policy=lazy is not a known value"):
            ext().tier_policy_replay({"policy": "lazy"}, 1, 8, 4, 0, 0, False, [])
        with self.assertRaisesRegex(Exception, "spare >= 0"):
            ext().tier_policy_replay({"spare": -1}, 1, 8, 4, 0, 0, False, [])


if __name__ == "__main__":
    unittest.main()
