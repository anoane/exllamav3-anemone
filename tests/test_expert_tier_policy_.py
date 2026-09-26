"""
The expert tier's reference policy (exllamav3/model/expert_tier_policy.py), torch-free:

  1. the micro-scenarios S1-S9 of doc/expert_tiers_impl_plan.md 3.7: every record, counter and
     final VRAM / RAM state exactly;
  2. invariants after every call on random configurations (every policy x demote x admit x evict x
     RAM admission, disk on and off where the RAM tier allows, spare 0-8, deterministic and
     production-like deferred returns);
  3. agreement with the policy simulator the first defaults were chosen on (tests/expert_tier/
     sim_presets_ref.py) on scaled-down cases of its grid: VRAM misses, SSD reads and demotions per
     token (EXL3_TIER_SIM_FULL=1 runs the 24 full-size cases, minutes);
  4. the adaptive admission rule on scripted cache lives, exactly;
  5. layer-mode prefill, refills, the host mirror (VramDirectory.apply) and cold fill.

    python -m pytest tests/test_expert_tier_policy_.py
    EXL3_TIER_RANDOM_CALLS=5000 python -m pytest tests/test_expert_tier_policy_.py -k invariants
"""
import importlib.util
import itertools
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


P = load("_expert_tier_policy_subject", ROOT / "exllamav3/model/expert_tier_policy.py")
FP = P.Q16


def core(E = 8, S = 4, spare = 2, R = 0, L = 1, **kw):
    cfg = P.PolicyConfig(**{"spare": spare, "admit": "always", "evict": "lru", "policy": "exclusive",
                            "demote": "off", "halflife": 1 << 30, **kw})
    return P.TierCore(cfg, L, E, S, R)


def run(c, calls):
    return [c.call(0, ids)[0].short() for ids in calls]


def vram_valid(c):
    return {s: (c.vram.key[s], c.vram.stamp[s]) for s in range(c.vram.S) if c.vram.state[s] == P.VALID}


def free_slots(c):
    return [s for s in range(c.vram.S) if c.vram.state[s] == P.FREE]


def counters(c, *names):
    return {n: c.c[n] for n in names}


class MicroScenarios(unittest.TestCase):
    """doc/expert_tiers_impl_plan.md 3.7; E = 8, one cache layer, deterministic mode"""

    def test_s1_lru_protection_spare_ties(self):
        c = core(S = 5, spare = 2)
        recs = run(c, [[0, 1], [2, 3], [0, 2], [1, 3], [4, 5]])
        self.assertEqual(recs, [
            [("admit", 0), ("admit", 1)],
            [("admit", 2), ("admit", 3, 0)],          # 0 retired: stamp tie with 1, lower slot
            [("admit", 0, 1), ("hit", 2)],
            [("admit", 1, 0), ("hit", 3)],            # 3 protected by phase 1
            [("admit", 4, 2), ("admit", 5, 1)],
        ])
        self.assertEqual(vram_valid(c), {0: (4, 5), 3: (3, 4), 4: (5, 5)})
        self.assertEqual(free_slots(c), [1, 2])
        self.assertEqual(counters(c, "hits", "admits", "retires", "drops", "ssd", "d2h"),
                         {"hits": 2, "admits": 8, "retires": 5, "drops": 5, "ssd": 8, "d2h": 0})
        c.check()

    def test_s2_admit_heat(self):
        c = core(S = 4, spare = 2, admit = "heat")
        recs = run(c, [[0, 1], [0, 2], [3, 0], [2, 4], [2, 5], [2, 4]])
        self.assertEqual(recs, [
            [("admit", 0), ("admit", 1)],
            [("hit", 0), ("admit", 2, 1)],            # heat 1 >= 1
            [("admit", 3, 2), ("hit", 0)],
            [("transient", 2), ("transient", 4)],     # the candidate is expert 0 (heat 3)
            [("admit", 2, 0), ("admit", 5, 3)],       # heat 3 >= 3
            [("hit", 2), ("admit", 4, 5)],
        ])
        self.assertEqual(counters(c, "hits", "admits", "transients", "retires", "ssd"),
                         {"hits": 3, "admits": 7, "transients": 2, "retires": 5, "ssd": 9})
        self.assertEqual({s: k for s, (k, _) in vram_valid(c).items()}, {0: 4, 2: 2})
        c.check()

    def s3(self, policy, demote, h):
        c = core(S = 3, spare = 1, R = 2, policy = policy, demote = demote)
        c.vram.place(0, 0, 1)
        c.vram.place(1, 1, 2)
        c.seq = 2
        c.ram.add(2, False, 0)
        c.ram.add(3, False, 0)
        for k, u in ((0, h), (1, 1), (2, 4), (3, 2)):
            c.heat.set(k, u * FP)
        rec, acts = c.call(0, [2])
        self.assertEqual(rec.short(), [("admit", 2, 0)])
        self.assertEqual(counters(c, "admits", "retires", "h2d_ram", "ssd"),
                         {"admits": 1, "retires": 1, "h2d_ram": 1, "ssd": 0})
        c.check()
        return c, acts

    def test_s3_policy_x_demote(self):
        # (policy, demote, victim heat) -> (RAM, D2H, duplicates reclaimed)
        own, dup = "own", "dup"
        want = {
            ("exclusive", "off", 5): ({3: (own, 1)}, 0, 0),
            ("exclusive", "off", 1): ({3: (own, 1)}, 0, 0),
            ("exclusive", "swap", 5): ({0: (own, 0), 3: (own, 1)}, 1, 0),
            ("exclusive", "swap", 1): ({3: (own, 1)}, 0, 0),
            ("exclusive", "heat", 5): ({0: (own, 0), 3: (own, 1)}, 1, 0),
            ("exclusive", "heat", 1): ({0: (own, 0), 3: (own, 1)}, 1, 0),
            ("exclusive", "all", 5): ({0: (own, 0), 3: (own, 1)}, 1, 0),
            ("exclusive", "all", 1): ({0: (own, 0), 3: (own, 1)}, 1, 0),
            ("lazy-exclusive", "off", 5): ({2: (dup, 0), 3: (own, 1)}, 0, 0),
            ("lazy-exclusive", "off", 1): ({2: (dup, 0), 3: (own, 1)}, 0, 0),
            ("lazy-exclusive", "swap", 5): ({0: (own, 0), 3: (own, 1)}, 1, 0),      # over 2's duplicate
            ("lazy-exclusive", "swap", 1): ({2: (dup, 0), 3: (own, 1)}, 0, 0),
            ("lazy-exclusive", "heat", 5): ({0: (own, 0), 3: (own, 1)}, 1, 1),
            ("lazy-exclusive", "heat", 1): ({0: (own, 0), 3: (own, 1)}, 1, 1),
            ("lazy-exclusive", "all", 5): ({0: (own, 0), 3: (own, 1)}, 1, 1),
            ("lazy-exclusive", "all", 1): ({0: (own, 0), 3: (own, 1)}, 1, 1),
        }
        for (pol, dem, h), (ram, d2h, reclaim) in want.items():
            with self.subTest(policy = pol, demote = dem, h = h):
                c, acts = self.s3(pol, dem, h)
                self.assertEqual(c.ram_view(), ram)
                self.assertEqual(counters(c, "d2h", "drops", "dup_reclaim"),
                                 {"d2h": d2h, "drops": 1 - d2h, "dup_reclaim": reclaim})
                self.assertEqual(c.ram.compact[0], d2h == 1)      # a demoted slot is in compact layout
                if dem == "swap" and d2h:
                    self.assertIn(("d2h", 0, 0, 0), acts)          # into the slot 2 came from

    def test_s3_master_switch(self):
        """EXL3_MOE_TIER_DEMOTE=0 turns every demote= into off"""
        for pol, dem in itertools.product(("exclusive", "lazy-exclusive"), ("swap", "heat", "all")):
            c = core(S = 3, spare = 1, R = 2, policy = pol, demote = dem, demote_on = False)
            c.vram.place(0, 0, 1)
            c.vram.place(1, 1, 2)
            c.seq = 2
            c.ram.add(2, False, 0)
            c.ram.add(3, False, 0)
            for k, u in ((0, 5), (1, 1), (2, 4), (3, 2)):
                c.heat.set(k, u * FP)
            c.call(0, [2])
            self.assertEqual(counters(c, "d2h", "drops"), {"d2h": 0, "drops": 1})

    def test_s4_disk_off(self):
        c = core(E = 6, S = 4, spare = 2, R = 4, policy = "lazy-exclusive", demote = "swap", disk_off = True)
        c.vram.place(0, 0, 1)
        c.vram.place(1, 1, 1)
        c.seq = 1
        for e in (2, 3, 4, 5):
            c.ram.add(e, False, 0)
        recs = []
        for ids in [[2, 3], [4, 0], [5, 1], [2, 4], [0, 1], [3, 5]]:
            recs.append(c.call(0, ids)[0].short())
            c.check()
            c.disk_off_complete()
            for k in range(6):      # each expert is VALID in VRAM or owned by RAM
                self.assertTrue(c.vram.slot_of[k] >= 0 or c.ram.owns(k), k)
        self.assertEqual([[r[2] for r in x] for x in recs], [[0, 1], [2, 3], [4, 0], [5, 1], [2, 4], [0, 1]])
        self.assertEqual(counters(c, "admits", "retires", "d2h", "h2d_ram", "ssd"),
                         {"admits": 12, "retires": 12, "d2h": 12, "h2d_ram": 12, "ssd": 0})
        self.assertEqual({s: k for s, (k, _) in vram_valid(c).items()}, {0: 3, 1: 5})
        self.assertEqual(c.ram_view(), {0: ("own", 0), 1: ("own", 2), 2: ("own", 1), 4: ("own", 3)})

    def test_s5_inclusive(self):
        c = core(S = 4, spare = 2, R = 5, policy = "inclusive", demote = "off")
        recs = []
        for ids in [[0, 1], [2, 0], [3, 4], [1, 2]]:
            recs.append(c.call(0, ids)[0].short())
            c.check()           # every VRAM expert has a RAM copy after every call
        self.assertEqual(recs, [[("admit", 0), ("admit", 1)], [("admit", 2, 1), ("hit", 0)],
                                [("admit", 3, 0), ("admit", 4, 2)], [("admit", 1, 3), ("admit", 2, 4)]])
        self.assertEqual(counters(c, "hits", "admits", "retires", "drops", "h2d_ram", "ssd"),
                         {"hits": 1, "admits": 7, "retires": 5, "drops": 5, "h2d_ram": 2, "ssd": 5})
        self.assertEqual(c.ram_view(), {0: ("own", 0), 1: ("dup", 1), 2: ("dup", 2), 3: ("own", 3), 4: ("own", 4)})

    def test_s6_adaptive_pinned(self):
        c = core(S = 4, spare = 2, admit = "adaptive", p = 0.0)
        self.assertEqual(run(c, [[0, 1], [2, 3], [4, 0]]),
                         [[("admit", 0), ("admit", 1)], [("transient", 2), ("transient", 3)],
                          [("transient", 4), ("hit", 0)]])
        self.assertEqual(counters(c, "admits", "transients", "ssd"), {"admits": 2, "transients": 3, "ssd": 5})
        c = core(S = 4, spare = 2, admit = "adaptive", p = 1.0)
        self.assertEqual(run(c, [[0, 1], [2, 3], [4, 0]]),
                         [[("admit", 0), ("admit", 1)], [("admit", 2, 0), ("admit", 3, 1)],
                          [("admit", 4, 2), ("admit", 0, 3)]])
        self.assertEqual(counters(c, "admits", "retires", "ssd"), {"admits": 6, "retires": 4, "ssd": 6})

    def test_s7_heat(self):
        c = core(S = 4, spare = 1, halflife = 2)
        seen = []
        c.heat.tokens = 0
        c.call(0, [0, 1])
        seen.append((c.heat.read(0), c.heat.read(1)))
        c.heat.tokens = 1
        c.call(0, [0])
        seen.append((c.heat.read(0), c.heat.read(1)))
        c.heat.tokens = 2
        seen.append((c.heat.read(0), c.heat.read(1)))
        c.heat.tokens = 5
        c.call(0, [1, 1, 2], P.ROUTED)
        seen.append((c.heat.read(0), c.heat.read(1), c.heat.read(2)))
        c.heat.tokens = 40
        seen.append((c.heat.read(0), c.heat.read(1), c.heat.read(2)))
        self.assertEqual(seen, [(65536, 65536), (131072, 65536), (65536, 32768), (32768, 24576, 4096), (0, 0, 0)])
        # saturation at 2^32 - 1
        c.heat.set(5, (1 << 32) - 10)
        c.heat.add(5, 100)
        self.assertEqual(c.heat.read(5), (1 << 32) - 1)

    def test_s8_lfu_demote_heat(self):
        c = core(S = 4, spare = 2, R = 1, evict = "lfu", policy = "lazy-exclusive", demote = "heat")
        recs = run(c, [[0, 1], [0, 2], [0, 3], [3, 4]])
        self.assertEqual(recs, [[("admit", 0), ("admit", 1)], [("hit", 0), ("admit", 2, 1)],
                                [("hit", 0), ("admit", 3, 2)], [("hit", 3), ("admit", 4, 0)]])
        self.assertEqual(counters(c, "hits", "admits", "retires", "drops", "d2h", "ssd", "ram_evict", "dup_reclaim"),
                         {"hits": 3, "admits": 5, "retires": 3, "drops": 1, "d2h": 2, "ssd": 5, "ram_evict": 1,
                          "dup_reclaim": 3})
        self.assertEqual(c.ram_view(), {0: ("own", 0)})
        c.check()

    def test_s9_starvation(self):
        c = core(S = 3, spare = 1)
        self.assertEqual(run(c, [[0, 1], [2, 3]]), [[("admit", 0), ("admit", 1)], [("admit", 2, 0), ("transient", 3)]])
        self.assertEqual(counters(c, "admits", "transients", "starved"), {"admits": 3, "transients": 1, "starved": 1})


class Semantics(unittest.TestCase):

    def test_spare_zero_in_place(self):
        """spare=0: once the pool is full an admission takes its victim's slot"""
        c = core(S = 2, spare = 0, R = 2, policy = "lazy-exclusive", demote = "off")
        recs = run(c, [[0, 1], [2], [3, 0]])
        self.assertEqual(recs, [[("admit", 0), ("admit", 1)], [("admit", 2, 0)], [("admit", 3, 1), ("admit", 0, 2)]])
        self.assertEqual(c.c["inplace"], 3)
        self.assertEqual({s: k for s, (k, _) in vram_valid(c).items()}, {0: 0, 1: 3})
        self.assertEqual(c.ram_view(), {0: ("dup", 0), 3: ("dup", 1)})
        self.assertFalse(any(a[0] == "return" for _, acts in c.log for a in acts))
        c.check()

    def test_decode_stamps_only(self):
        """Routed (prefill) calls neither stamp nor admit: the decode working set survives a prompt"""
        c = core(S = 4, spare = 2)
        c.call(0, [0, 1])
        before = dict(vram_valid(c))
        rec, _ = c.call(0, [0, 1, 2, 3, 4], P.ROUTED)
        self.assertEqual(rec.short(), [("hit", 0), ("hit", 1), ("transient", 2), ("transient", 3), ("transient", 4)])
        self.assertEqual([x.dst for x in rec.entries], [0, 1, 0, 1, 2])       # staging slots of half 0
        self.assertEqual(vram_valid(c), before)
        self.assertEqual(c.heat.read(2), 4096)

    def test_counts_and_dedupe(self):
        c = core(S = 8, spare = 2)
        rec, _ = c.call(0, [3, 1, 3, 3, 1, 7])
        self.assertEqual([(x.key, x.cnt) for x in rec.entries], [(3, 3), (1, 2), (7, 1)])
        self.assertEqual(c.heat.read(3), 3 * FP)
        self.assertEqual(c.vram.access, 3)

    def test_ram_admission_and_refused(self):
        """A decode transient read from the SSD enters RAM: free, duplicate, coldest (ram_admit=heat: when warmer)"""
        for ram_admit, want in (("always", {1: ("own", 0)}), ("heat", {0: ("own", 0)})):
            c = core(S = 3, spare = 1, R = 1, admit = "adaptive", p = 0.0, policy = "exclusive", ram_admit = ram_admit)
            c.vram.place(5, 0, 0)
            c.vram.place(6, 1, 0)
            c.ram.add(0, False, 0)
            c.heat.set(0, 3 * FP)
            c.call(0, [1])          # a transient (p = 0); heat 1 < 3
            self.assertEqual(c.ram_view(), want, ram_admit)
            self.assertEqual(c.c["ram_admit_refused"], int(ram_admit == "heat"))

    def test_rescue_and_deferred_returns(self):
        """Production: a demoted victim's slot stays RETIRING until its copy lands; a miss on it is
        served from that slot (rescue), a starved admission becomes a transient"""
        cfg = P.PolicyConfig(spare = 2, admit = "always", evict = "lru", policy = "exclusive", demote = "all",
                             halflife = 1 << 30)
        c = P.TierCore(cfg, 1, 8, 5, 4, defer_returns = True)
        c.call(0, [0, 1, 2])
        rec, acts = c.call(0, [3])                        # victim 0 demoted, slot 0 RETIRING
        self.assertEqual(rec.short(), [("admit", 3, 0)])
        self.assertIn(("d2h", 0, 0, 0), acts)
        self.assertEqual(c.vram.state[0], P.RETIRING)
        c.check(deterministic = False)
        rec, acts = c.call(0, [0, 4])                     # 0: served from its retiring slot
        self.assertEqual(rec.short(), [("admit", 0, 1), ("transient", 4)])    # 4: no free slot left
        self.assertEqual(acts[0], ("d2d", 0, 0, P.ADMIT, 4))
        self.assertIn(("ram_free", 0, 0), acts)           # exclusive: its RAM copy is released
        self.assertEqual(counters(c, "rescued", "starved", "h2d_ram"), {"rescued": 1, "starved": 1, "h2d_ram": 0})
        c.check(deterministic = False)
        self.assertEqual(c.complete(), [("return", 0), ("return", 1)])
        c.check()

    def test_layer_mode_and_refill(self):
        """Layer mode stages every non-VRAM expert; warm SSD-only experts are refilled into RAM, at
        most refill_inflight per call"""
        cfg = P.PolicyConfig(spare = 1, admit = "always", policy = "lazy-exclusive", demote = "swap",
                             halflife = 1 << 30, refill_inflight = 2)
        c = P.TierCore(cfg, 2, 8, 4, 4)
        c.cold_fill(P.round_robin_order(2, 8))           # VRAM: 0:0 1:0 0:1 (keys 0, 8, 1); RAM: 9 2 10 3
        self.assertEqual({s: k for s, (k, _) in vram_valid(c).items()}, {0: 0, 1: 8, 2: 1})
        self.assertEqual(c.ram_view(), {9: ("own", 0), 2: ("own", 1), 10: ("own", 2), 3: ("own", 3)})
        acts = c.layer_call(0, [4, 5, 6, 4, 2])
        stage = [a for a in acts if a[0].startswith("stage")]
        self.assertEqual(stage, [("stage_ram", 2, 1, 0), ("stage_ram", 3, 3, 1), ("stage_ssd", 4, 2),
                                 ("stage_ssd", 5, 3), ("stage_ssd", 6, 4), ("stage_ssd", 7, 5)])
        # 4 (heat 2/16), 5 and 6 (1/16) were used and read from the SSD; RAM is full of owners at
        # heat 0: the two warmest candidates in order win, the third is dropped
        # RAM is full of owners; 2 was just used (heat 1/16), 9, 10 and 3 are cold (0): the coldest
        # (then lowest key) makes room for 4, then for 5; 6 is dropped (two refills per call)
        self.assertEqual([a for a in acts if a[0] in ("refill", "ram_evict")],
                         [("ram_evict", 3, 3), ("refill", 4, 3), ("ram_evict", 9, 0), ("refill", 5, 0)])
        self.assertEqual(counters(c, "refills", "refill_dropped", "prefill_h2d", "prefill_ssd"),
                         {"refills": 2, "refill_dropped": 1, "prefill_h2d": 2, "prefill_ssd": 4})
        self.assertEqual(c.vram_view()[0][1:], (0, 0))    # no stamps in layer mode
        c.check()
        c2 = P.TierCore(P.PolicyConfig(spare = 1, halflife = 1 << 30, refill = False), 2, 8, 4, 4)
        c2.cold_fill(P.round_robin_order(2, 8))
        c2.layer_call(0, [4, 5, 6])
        self.assertEqual(c2.c["refills"], 0)

    def test_layer_mode_halves(self):
        """layer_plan (the staging, before the layer's routing is known) then layer_record (heat, refills,
        with the call's experts) equal layer_call: the same actions, counters and state, on random
        traces over every policy"""
        rng = random.Random(17)
        for pol in P.POLICIES:
            cfg = P.PolicyConfig(spare = 2, admit = "always", policy = pol, demote = "off" if pol == "inclusive" else "heat",
                                 halflife = 64, refill_inflight = 2)
            a, b = P.TierCore(cfg, 3, 16, 8, 20), P.TierCore(cfg, 3, 16, 8, 20)
            a.cold_fill(P.round_robin_order(3, 16))
            b.cold_fill(P.round_robin_order(3, 16))
            for i in range(300):
                lc = rng.randrange(3)
                ids = [rng.randrange(16) for _ in range(rng.randrange(1, 40))]
                if rng.random() < 0.5:
                    self.assertEqual([x.astuple() for x in a.call(lc, ids)[0].entries],
                                     [x.astuple() for x in b.call(lc, ids)[0].entries])
                    continue
                if i % 7 == 0:
                    a.tick()
                    b.tick()
                want = a.layer_call(lc, ids)
                got = b.layer_plan(lc)
                b.seq += 1
                rec = P.Record(b.seq, lc, P.LAYER)
                for e in dict.fromkeys(ids):
                    rec.entries.append(P.Entry(lc * 16 + e, P.TRANSIENT, -1, ids.count(e), 0))
                got += b.layer_record(rec)
                self.assertEqual(got, want, (pol, i))
                self.assertEqual((a.vram.key, a.ram.key, a.heat.v, a.seq), (b.vram.key, b.ram.key, b.heat.v, b.seq))
            self.assertEqual(a.c, b.c, pol)
        with self.assertRaisesRegex(P.TierError, "key 40 outside layer 0"):
            a.layer_record(P.Record(1, 0, P.LAYER, [P.Entry(40, P.TRANSIENT, -1, 1, 0)]))

    def test_mirror_apply(self):
        """A mirror that applies the records reaches the same directory as the one that decided them"""
        rng = random.Random(3)
        for spare in (0, 2):
            cfg = P.PolicyConfig(spare = spare, admit = "adaptive", p = 0.6, evict = "lru", halflife = 64)
            c = P.TierCore(cfg, 3, 16, 12, 10)
            m = P.VramDirectory(c.K, 12, spare)
            for i in range(400):
                if i % 7 == 0:
                    c.tick()
                mode = P.ROUTED if i % 50 == 49 else P.DECODE
                rec, _ = c.call(rng.randrange(3), [rng.randrange(16) for _ in range(rng.choice((6, 12)))], mode)
                m.apply(rec)
                for x in c.log[-1][1]:
                    if x[0] == "return":
                        m.return_slot(x[1])
                self.assertEqual((m.state, m.key, m.stamp, m.slot_of, m.free, m.occupied, m.access),
                                 (c.vram.state, c.vram.key, c.vram.stamp, c.vram.slot_of, c.vram.free,
                                  c.vram.occupied, c.vram.access))

    def test_cold_fill_inclusive_and_order(self):
        self.assertEqual(P.round_robin_order(2, 3), [0, 3, 1, 4, 2, 5])
        self.assertEqual(P.round_robin_order(2, 3, pinned = {0, 4}), [3, 1, 2, 5])
        c = P.TierCore(P.PolicyConfig(spare = 1, policy = "inclusive", demote = "off"), 2, 3, 3, 5)
        vk, rk = c.cold_fill(P.round_robin_order(2, 3))
        self.assertEqual((vk, rk), ([0, 3], [0, 3, 1, 4, 2]))
        self.assertEqual(c.ram_view(), {0: ("dup", 0), 3: ("dup", 1), 1: ("own", 2), 4: ("own", 3), 2: ("own", 4)})
        c = P.TierCore(P.PolicyConfig(spare = 1), 2, 3, 6, 0)
        vk, rk = c.cold_fill(P.round_robin_order(2, 3))      # everything fits: no spare needed
        self.assertEqual((len(vk), rk), (6, []))

    def test_from_placement(self):
        """The placement's words and the tuning map onto the policy's settings"""
        T = load("_etc_for_policy", ROOT / "exllamav3/model/expert_tier_config.py")
        PL = load("_placement_for_policy", ROOT / "exllamav3/model/placement.py")
        pl = PL.parse("*=cuda:0 experts=cache; cuda:0 spare=12 evict=lfu admit=heat; "
                      "ram experts=64GiB policy=exclusive demote=heat evict=lru; disk experts=off")
        tc = T.TierConfig.from_env({"EXL3_MOE_TIER_ADMIT_P": "0.25", "EXL3_MOE_HEAT_PREFILL": "0.125",
                                    "EXL3_MOE_TIER_RAM_ADMIT": "heat", "EXL3_MOE_TIER_REFILL": "0"})
        cfg = P.PolicyConfig.from_placement(pl.device_store("cuda:0"), pl.ram_store(), pl.disk_store(), tc, seed = 7)
        self.assertEqual(cfg, P.PolicyConfig(policy = "exclusive", demote = "heat", admit = "heat", evict = "lfu",
                                             ram_evict = "lru", ram_admit = "heat", disk_off = True, spare = 12,
                                             seed = 7, prefill_inc = 8192, p = 0.25, refill = False))
        d = P.PolicyConfig.from_placement(pl.device_store("cuda:1"), PL.storage.RamStore(), PL.storage.DiskStore(),
                                          T.TierConfig())
        self.assertEqual(d, P.PolicyConfig())       # the defaults agree

    def test_config_refusals(self):
        with self.assertRaisesRegex(ValueError, "policy='lazy'"):
            P.PolicyConfig(policy = "lazy")
        with self.assertRaisesRegex(ValueError, "spare >= 0"):
            P.PolicyConfig(spare = -1)


class Adaptive(unittest.TestCase):

    def test_scripted_lives(self):
        a = P.AdaptiveP(4, 12, 0.05)
        self.assertEqual(a.p, 0.25)
        got = [a.update(lv, lr) for lv, lr in ((1, 3), (1, 1), (3, 1), (9, 1), (1, 99), (1, 99), (1, 99))]
        self.assertEqual(got, [0.1875, 0.1875, 0.234375, 0.25, 0.1275, 0.065025, 0.05])
        self.assertEqual(0.065025 * (1.0 + (1 / 100 - 0.5)), 0.03316275)    # clamped to pmin above
        b = P.AdaptiveP(4, 12, 0.05, pinned = 0.7)
        self.assertEqual((b.update(1, 99), b.q32), (0.7, int(0.7 * 4294967296.0)))
        self.assertEqual((P.p_q32(0.0), P.p_q32(1.0)), (0, 1 << 32))

    def test_adapts_on_schedule(self):
        cfg = P.PolicyConfig(spare = 1, admit = "adaptive", adapt_every = 10, halflife = 1 << 30)
        c = P.TierCore(cfg, 1, 16, 4, 4)
        c.cold_fill(list(range(16)))
        rng = random.Random(0)
        for _ in range(30):
            c.call(0, [rng.randrange(16) for _ in range(2)])
        self.assertEqual(c.c["adapts"], c.vram.access // 10)
        self.assertTrue(cfg.pmin <= c.adapt.p <= c.adapt.pmax)


# ---------------------------------------------------------------------------------------- random

def random_config(rng):
    policy = rng.choice(P.POLICIES)
    demote = rng.choice(P.DEMOTES)
    E, L = rng.choice((16, 32, 64)), rng.choice((1, 2, 3))
    S = rng.randrange(4, 65)
    spare = rng.randrange(0, 9) if demote == "off" or policy == "inclusive" else rng.randrange(1, 9)
    spare = min(spare, S - 1)
    R = rng.randrange(0, 129)
    disk_off = False
    K = E * L
    if policy == "inclusive":
        R = max(R, S + spare + 1)
    elif rng.random() < 0.3 and spare > 0 and R >= K - (S - spare):
        disk_off = True                     # every key fits VRAM (C) + RAM
    cfg = P.PolicyConfig(policy = policy, demote = demote, admit = rng.choice(P.ADMITS),
                         evict = rng.choice(P.EVICTS), ram_evict = rng.choice(P.EVICTS),
                         ram_admit = rng.choice(P.RAM_ADMITS), demote_on = disk_off or rng.random() < 0.8,
                         disk_off = disk_off, spare = spare, sample = rng.choice((1, 4, 16)),
                         seed = rng.randrange(1 << 64), halflife = rng.choice((8, 64, 256)),
                         adapt_every = rng.choice((64, 2048)), p = rng.choice((None, None, 0.3)),
                         refill = rng.random() < 0.8, refill_inflight = rng.choice((1, 2, 4)))
    return cfg, L, E, S, R


def random_script(rng, L, E, n):
    ops, zipf = [], [1.0 / (i + 1) ** rng.choice((0.6, 1.0)) for i in range(E)]
    perm = [rng.sample(range(E), E) for _ in range(L)]
    for i in range(n):
        lc = i % L
        if lc == 0:
            ops.append(("tick", 1))
        r = rng.random()
        if r < 0.9:
            rows = rng.choice((1, 1, 1, 2, 4, 8))
            ids = [perm[lc][e] for e in rng.choices(range(E), zipf, k = rows * min(6, E))]
            ops.append(("call", lc, ids, P.DECODE))
        elif r < 0.97:
            ops.append(("call", lc, [rng.randrange(E) for _ in range(rng.randrange(8, 64))], P.ROUTED))
        else:
            ops.append(("layer", lc, [rng.randrange(E) for _ in range(rng.randrange(8, 200))]))
    return ops


class Invariants(unittest.TestCase):

    def test_invariants_random(self):
        configs = int(os.environ.get("EXL3_TIER_RANDOM_CONFIGS", "200"))
        calls = int(os.environ.get("EXL3_TIER_RANDOM_CALLS", "1000"))
        rng = random.Random(20260925)
        for i in range(configs):
            cfg, L, E, S, R = random_config(rng)
            defer = rng.random() < 0.25
            with self.subTest(i = i, cfg = cfg, L = L, E = E, S = S, R = R, defer = defer):
                c = P.TierCore(cfg, L, E, S, R, defer_returns = defer)
                c.cold_fill(P.round_robin_order(L, E))
                if cfg.disk_off:
                    c.disk_off_complete()
                for op in random_script(rng, L, E, calls):
                    P.replay(c, [op])
                    if defer and rng.random() < 0.3:
                        c.complete()
                    c.check(deterministic = not defer)
                    if cfg.disk_off:
                        c.disk_off_complete()
                if cfg.disk_off:
                    self.assertEqual(c.c["ssd"] + c.c["prefill_ssd"], 0)
                c.complete()
                c.check()
                k = c.c
                self.assertEqual(k["hits"] + k["admits"] + k["transients"], c.vram.access)
                self.assertLessEqual(k["retires"], k["admits"])
                self.assertEqual(k["d2h"] + k["drops"], k["retires"])     # every victim moved or dropped


# ---------------------------------------------------------------------------------------- simulator

SIM = None


def sim():
    global SIM
    if SIM is None:
        SIM = load("_sim_presets_ref", ROOT / "tests/expert_tier/sim_presets_ref.py")
    return SIM


# DESIGN.md 2.3's presets (policy, demote as the simulator names it, admit); swapheat = demote=swap
PRESETS = [("lazy-exclusive", "heat", "heat"), ("lazy-exclusive", "swapheat", "adaptive"),
           ("lazy-exclusive", "all", "adaptive"), ("lazy-exclusive", "off", "adaptive"),
           ("exclusive", "all", "always"), ("lazy-exclusive", "off", "always"), ("inclusive", "off", "always")]


def core_on_trace(policy, demote, admit, trace, warm, vs, rs):
    cfg = P.PolicyConfig(policy = policy, demote = "swap" if demote == "swapheat" else demote, admit = admit,
                         evict = "lru", ram_evict = "lfu", spare = 0, halflife = 256, refill = False)
    tokens, layers, _ = trace.shape
    c = P.TierCore(cfg, layers, sim().E, vs, rs)
    for t in range(tokens):
        if t == warm:
            c.c = {k: 0 for k in P.COUNTERS}
        c.tick()
        for l in range(layers):
            c.call(l, [int(e) for e in trace[t, l]])
    n = tokens - warm
    return {"h2d": (c.c["h2d_ram"] + c.c["ssd"]) / n, "ssd": c.c["ssd"] / n, "d2h": c.c["d2h"] / n}


class Simulator(unittest.TestCase):
    """
    The reference core against the simulator the first defaults were chosen on (same traces). The core is
    per call with protection, fixed-point lazily decayed heat, its own draws and tie-breaks; the
    simulator per access with float heat. Per token they agree within 3 % or 0.2 per token scaled to
    the layers (0.2 at 28 layers). admit=adaptive presets get 5 % or twice that margin, and their SSD
    reads may only be lower: the two draw different random streams, and the core measures the RAM
    tier's cache life by the age of its least recently inserted or read entry, as PROMOTE defines it,
    where the simulator takes the last access of that entry's expert anywhere (a duplicate of an
    expert the pool keeps using looks young, so its p climbs above the floor now and then and it
    admits more). On the full grid (EXL3_TIER_SIM_FULL=1: 24 cases x 7 presets, 37 minutes on the AI
    VM) that makes the core read fewer experts from the SSD than the simulator in one case (C/52 GiB,
    Zipf 0.6, lazy-exclusive/off/adaptive: 1.78 vs 2.75 per token, 1.90 vs 4.74 with drift); every
    other comparison agrees within the margins.
    """

    def cases(self):
        S = sim()
        if os.environ.get("EXL3_TIER_SIM_FULL") == "1":
            for (name, layers, vg, rams), s, dr in itertools.product(S.GRID, S.ZIPF, S.DRIFTS):
                for rg in rams:
                    yield name, layers, int(vg * S.GIB // S.VRAM_SLOT), int(rg * S.GIB // S.RAM_SLOT), s, dr, S.TOKENS, S.WARM
            return
        # scaled down: 4 layers, the pools scaled by 4 / layers, 3,000 tokens (1,000 warm-up)
        for name, layers, vg, rg, s, dr in (("C", 28, 81, 32, 0.6, (0, 0.0)), ("C", 28, 81, 52, 1.0, (1500, 0.25)),
                                            ("D", 40, 75, 64, 0.6, (1500, 0.25)), ("D", 40, 75, 96, 1.0, (0, 0.0))):
            f = 4 / layers
            yield (name, 4, int(vg * S.GIB // S.VRAM_SLOT * f), int(rg * S.GIB // S.RAM_SLOT * f), s, dr, 3000, 1000)

    def test_agreement(self):
        S = sim()
        worst = []
        for name, layers, vs, rs, s, (de, df), tokens, warm in self.cases():
            trace = S.gen_trace(layers, tokens, s, de, df, S.SEED)
            for pol, dem, adm in PRESETS:
                if pol == "inclusive" and rs <= vs:
                    continue                # the tier refuses inclusive with RAM not above the pool (R7)
                ref = S.simulate_preset(pol, dem, adm, trace, warm, vs, rs, S.SEED)
                got = core_on_trace(pol, dem, adm, trace, warm, vs, rs)
                margin = 0.2 * layers / 28
                rel = 0.03
                if adm == "adaptive":
                    rel, margin = 0.05, 2 * margin
                for m in ("h2d", "ssd", "d2h"):
                    tol = max(rel * ref[m], margin)
                    worst.append((abs(got[m] - ref[m]) / max(ref[m], 1e-9), name, vs, rs, s, de, pol, dem, adm, m,
                                  got[m], ref[m]))
                    diff = got[m] - ref[m]
                    if adm == "adaptive" and m == "ssd":
                        diff = max(0.0, diff)           # fewer SSD reads than the simulator is allowed
                    with self.subTest(case = (name, vs, rs, s, de), preset = (pol, dem, adm), metric = m):
                        self.assertLessEqual(abs(diff), tol, (got, ref))
        worst.sort(reverse = True)
        print("\nlargest relative differences (core vs simulator):")
        for w in worst[:8]:
            print("  %.4f %s vs=%d rs=%d s=%.1f drift=%d %s/%s/%s %s: %.3f vs %.3f" % w)


if __name__ == "__main__":
    unittest.main()
