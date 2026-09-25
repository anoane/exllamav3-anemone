"""
CPU-only tests of the expert tier's configuration and sizing (exllamav3/model/expert_tier_config.py,
doc/expert_tiers.md): every tuning variable, its values and its refusals; and the pools a load
sizes for DeepSeek-V4.1-Flash 3.0 bpw (40 layers x 384 experts, top-6; 13,271,040-byte VRAM
slots, 13,320,192-byte RAM slots) with every load-time refusal of the tier, for the worked
examples of doc/expert_tiers.md and the three test layouts:

    (a) PRO 6000 + RAM for every expert outside VRAM, no CMP:   *=cuda:0 experts=cache; ram experts=all; disk experts=off
    (b) PRO 6000 + a 96 GiB RAM tier + the SSD, no CMP:         *=cuda:0 experts=cache; ram experts=96GiB
    (c) CMP (resident) + PRO (cache) + RAM, no expert SSD:      0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off

    python tests/test_expert_tier_config_.py

expert_tier_config.py and placement.py are torch-free and loaded by path; experts=cache, which
this build refuses at parse time (placement_storage.PENDING), is parsed with that gate lifted.
"""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load("_expert_tier_config_subject", "exllamav3/model/expert_tier_config.py")
P = load("_placement_tier_config_subject", "exllamav3/model/placement.py")
GiB, MiB = 1 << 30, 1 << 20
VS, RS = 13_271_040, 13_320_192
MEM_TOTAL = 140_802_264 * 1024          # MemTotal of the 134.3 GiB VM DESIGN.md sized against
STAGING = 2 * 384 * VS                  # double staging of one 384-expert layer: 9.49 GiB
HEADROOM = 1024 * MiB


def parse(text):
    with patch.dict(P.storage.PENDING, {}, clear = True):
        return P.parse(text)


def layers(which = range(40), hot = 0, device = lambda i: "cuda:0"):
    return T.TierLayers({i: 384 for i in which}, {i: hot for i in which}, {i: device(i) for i in which}, 6, VS, RS)


def memory(avail_gib, total = MEM_TOTAL, cgroup_gib = None):
    avail = int(avail_gib * GiB)
    if cgroup_gib is None:
        return T.HostMemory(avail, avail, total, 0)
    room = int(cgroup_gib * GiB)
    return T.HostMemory(min(avail, room), avail, total, 0, room, "/system.slice/run-t.scope")


def room(pool_gib, staging = STAGING):
    """A GPU room that leaves pool_gib for the pool after the staging and the headroom"""
    return int(pool_gib * GiB) + staging + HEADROOM


def size(text, lay, mem, rooms = None, cfg = None, **kw):
    kw.setdefault("platform", "linux")          # the sizing, wherever the test runs
    return T.size_tiers(parse(text), lay, cfg or T.TierConfig(), mem, vram_room = rooms, **kw)


def refused(test, exc, text, lay, mem, rooms = None, **kw):
    with test.assertRaises(exc) as cm:
        size(text, lay, mem, rooms, **kw)
    return str(cm.exception)


class ConfigTests(unittest.TestCase):

    def test_defaults(self):
        c = T.TierConfig.from_env({})
        self.assertEqual(c, T.TierConfig())
        self.assertEqual((c.demote, c.staging, c.prefill_rows, c.decode_rows, c.heat_halflife, c.heat_prefill_q16),
                         (True, "double", 256, 8, 256, 4096))
        self.assertEqual((c.admit_p, c.adapt_every, c.admit_pmin, c.sample, c.ram_admit, c.refill, c.refill_inflight),
                         (None, 2048, 0.05, 16, "always", True, 2))
        self.assertEqual((c.disk_slab, c.headroom_mb, c.min_link_gbs, c.spin_us, c.affinity, c.deterministic,
                          c.verify, c.trace, c.heat_file), (None, 1024, 2.0, -1, None, False, False, None, None))
        self.assertEqual(c.changed(), [])
        # SETTINGS and the dataclass agree on every default
        for s in T.SETTINGS:
            self.assertEqual(getattr(T.TierConfig(), s.field), s.default, s.var)
        self.assertEqual(len({s.field for s in T.SETTINGS}), len(T.SETTINGS))

    def test_values(self):
        env = {"EXL3_MOE_TIER_DEMOTE": "0", "EXL3_MOE_TIER_STAGING": "Single", "EXL3_MOE_TIER_PREFILL_ROWS": "512",
               "EXL3_MOE_TIER_DECODE_ROWS": "4", "EXL3_MOE_HEAT_HALFLIFE": "128", "EXL3_MOE_HEAT_PREFILL": "0.125",
               "EXL3_MOE_HEAT_FILE": "/var/lib/exl3/v41.heat", "EXL3_MOE_TIER_ADMIT_P": "0.5",
               "EXL3_MOE_TIER_ADAPT_EVERY": "4096", "EXL3_MOE_TIER_ADMIT_PMIN": "0.01", "EXL3_MOE_TIER_SAMPLE": "32",
               "EXL3_MOE_TIER_RAM_ADMIT": "heat", "EXL3_MOE_TIER_REFILL": "0", "EXL3_MOE_TIER_REFILL_INFLIGHT": "4",
               "EXL3_MOE_TIER_DISK_SLAB": "64", "EXL3_MOE_TIER_HUGEPAGE": "1", "EXL3_MOE_TIER_HEADROOM_MB": "512",
               "EXL3_MOE_TIER_MIN_LINK_GBS": "0",
               "EXL3_MOE_TIER_SPIN_US": "0", "EXL3_MOE_TIER_AFFINITY": "16-19,23, 21", "EXL3_MOE_TIER_DETERMINISTIC": "1",
               "EXL3_MOE_TIER_VERIFY": "1", "EXL3_MOE_TIER_TRACE": "/tmp/t.bin", "UNRELATED": "x"}
        c = T.TierConfig.from_env(env)
        self.assertEqual((c.demote, c.staging, c.staging_factor, c.prefill_rows, c.decode_rows, c.heat_halflife),
                         (False, "single", 1, 512, 4, 128))
        self.assertEqual((c.heat_prefill, c.heat_prefill_q16, c.heat_file, c.admit_p, c.adapt_every, c.admit_pmin),
                         (0.125, 8192, "/var/lib/exl3/v41.heat", 0.5, 4096, 0.01))
        self.assertEqual((c.sample, c.ram_admit, c.refill, c.refill_inflight, c.disk_slab, c.hugepage, c.headroom_mb),
                         (32, "heat", False, 4, 64, True, 512))
        self.assertEqual((c.min_link_gbs, c.spin_us, c.affinity, c.deterministic, c.verify, c.trace),
                         (0.0, 0, (16, 17, 18, 19, 21, 23), True, True, "/tmp/t.bin"))
        self.assertEqual(len(c.changed()), len(T.SETTINGS))
        self.assertIsNone(T.TierConfig.from_env({"EXL3_MOE_TIER_DISK_SLAB": "auto"}).disk_slab)

    def test_refusals(self):
        cases = {
            "EXL3_MOE_TIER_DEMOTE=yes": "EXL3_MOE_TIER_DEMOTE='yes' must be 0 or 1",
            "EXL3_MOE_TIER_STAGING=triple": "EXL3_MOE_TIER_STAGING='triple' must be double or single",
            "EXL3_MOE_TIER_STAGING=singel": "EXL3_MOE_TIER_STAGING='singel' must be double or single (did you mean 'single'?)",
            "EXL3_MOE_TIER_PREFILL_ROWS=1": "EXL3_MOE_TIER_PREFILL_ROWS='1' must be a whole number from 2 to 1048576",
            "EXL3_MOE_TIER_DECODE_ROWS=9": "EXL3_MOE_TIER_DECODE_ROWS='9' must be a whole number from 1 to 8",
            "EXL3_MOE_TIER_DECODE_ROWS=4.0": "EXL3_MOE_TIER_DECODE_ROWS='4.0' must be a whole number from 1 to 8",
            "EXL3_MOE_HEAT_PREFILL=2": "EXL3_MOE_HEAT_PREFILL='2' must be a number from 0 to 1",
            "EXL3_MOE_HEAT_PREFILL=1/16": "EXL3_MOE_HEAT_PREFILL='1/16' must be a number from 0 to 1",
            "EXL3_MOE_TIER_ADMIT_P=1.5": "EXL3_MOE_TIER_ADMIT_P='1.5' must be a number from 0 to 1",
            "EXL3_MOE_TIER_ADMIT_PMIN=0": "EXL3_MOE_TIER_ADMIT_PMIN='0' must be a number from 0.0001 to 1",
            "EXL3_MOE_TIER_AFFINITY=23-16": "EXL3_MOE_TIER_AFFINITY='23-16' must be a CPU list such as 16-23 or 4,6,8",
            "EXL3_MOE_TIER_DISK_SLAB=0": "EXL3_MOE_TIER_DISK_SLAB='0' must be auto or a number of slots from 1 to 4096",
            "EXL3_MOE_TIER_SPIN_US=-2": "EXL3_MOE_TIER_SPIN_US='-2' must be a whole number from -1 to 1073741824",
            "EXL3_MOE_TIER_MIN_LINK_GBS=-1": "EXL3_MOE_TIER_MIN_LINK_GBS='-1' must be a number from 0 to 1e+06",
            "EXL3_MOE_HEAT_FILE=": "EXL3_MOE_HEAT_FILE='' must be a path (unset it to disable)",
            "EXL3_MOE_TIER_STAGIN=double": "EXL3_MOE_TIER_STAGIN is not an expert tier setting (did you mean "
                                           "EXL3_MOE_TIER_STAGING?); the settings are listed in doc/expert_tiers.md",
            "EXL3_MOE_HEAT_HALF_LIFE=5": "EXL3_MOE_HEAT_HALF_LIFE is not an expert tier setting (did you mean "
                                         "EXL3_MOE_HEAT_HALFLIFE?); the settings are listed in doc/expert_tiers.md",
            "EXL3_MOE_TIER_BOGUS_KNOB=1": "EXL3_MOE_TIER_BOGUS_KNOB is not an expert tier setting; the settings are "
                                          "listed in doc/expert_tiers.md",
        }
        for assignment, msg in cases.items():
            k, v = assignment.split("=", 1)
            with self.subTest(assignment), self.assertRaises(ValueError) as cm:
                T.TierConfig.from_env({k: v})
            self.assertEqual(str(cm.exception), msg)
        with self.assertRaises(ValueError) as cm:
            T.TierConfig.from_env({"EXL3_MOE_TIER_PREFILL_ROWS": "8"})
        self.assertEqual(str(cm.exception), "EXL3_MOE_TIER_PREFILL_ROWS=8 must be above EXL3_MOE_TIER_DECODE_ROWS=8 "
                                            "(a call is decode, routed or layer mode by its rows)")


class SizingTests(unittest.TestCase):
    """The worked examples and load-time refusals of doc/expert_tiers.md (DESIGN.md 3.8 / 3.10)"""

    def test_t5_cmp_plus_pro_no_disk(self):
        text = "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off"
        lay = layers(range(12, 40), device = lambda i: "cuda:1")
        s = size(text, lay, memory(126), {"cuda:1": room(81)})
        d = s.devices[0]
        self.assertEqual((d.device, d.cached, d.slots, d.held, d.staging_slots), ("cuda:1", 10752, 6553, 6545, 768))
        self.assertEqual((s.cached, s.tier_slots, s.disk_only, s.slab_slots), (10752, 4207, 0, 0))
        self.assertEqual(s.ram_bytes, 4207 * RS)                                      # 52.2 GiB
        self.assertEqual(s.text.splitlines()[1], "    ram experts 52.2 GiB pinned (static 0 MiB + tier 4207 slots), "
                                                 "policy lazy-exclusive demote=swap, evict=lfu")
        self.assertTrue(s.final)

    def test_t6_pro_alone(self):
        s = size("*=cuda:0 experts=cache; cuda:0 cache=75GiB; ram experts=96GiB ngram=6GiB", layers(), memory(126),
                 {"cuda:0": room(80)}, ngram_bytes = 6 * GiB)
        self.assertEqual((s.devices[0].slots, s.devices[0].held, s.tier_slots), (6068, 6060, 7738))
        self.assertEqual(s.disk_only, 15360 - 6060 - 7738)                            # 1562 only on the SSD
        self.assertEqual((s.slab_slots, s.slab_bytes), (8, 8 * RS))
        self.assertEqual(s.left_unpinned, 126 * GiB - 7738 * RS - 6 * GiB - 8 * RS)
        self.assertIn("disk experts=model io=auto: 1562 experts on disk only, slab 8 slots (102 MiB); 23.9 GiB of "
                      "RAM left unpinned", s.text)

    def test_t7_hot_and_pagecache(self):
        lay = layers(hot = 16)
        s = size("*=cuda:0 experts=cache hot=16; ram experts=auto pagecache=24GiB", lay, memory(60),
                 {"cuda:0": room(66, staging = 2 * 368 * VS)})
        self.assertEqual((lay.cached(), s.hot, s.devices[0].staging_slots), (14720, 640, 736))
        self.assertEqual((s.devices[0].slots, s.tier_slots, s.disk_only), (5339, 2740, 6649))

    def test_default_auto(self):
        # no ram rule, no flag: auto, leaving max(8 GiB, MemTotal / 8) of page cache
        s = size("*=cuda:0 experts=cache", layers(), memory(126), {"cuda:0": room(80)})
        self.assertEqual((s.devices[0].slots, s.tier_slots, s.disk_only), (6472, 8642, 254))

    def test_static_arenas_first(self):
        static = 10 * 384 * 13_315_584
        s = size("0-9=cuda:0 experts=cpu; 10-39=cuda:1 experts=cache; cuda:1 cache=40GiB; ram experts=96GiB",
                 layers(range(10, 40), device = lambda i: "cuda:1"), memory(126), {"cuda:1": room(41)},
                 static_bytes = static)
        self.assertEqual(s.tier_slots, (96 * GiB - static) // RS)
        self.assertEqual(s.ram_bytes, static + s.tier_slots * RS)
        msg = refused(self, ValueError, "0-9=cuda:0 experts=cpu; 10-39=cuda:1 experts=cache; ram experts=40GiB",
                      layers(range(10, 40), device = lambda i: "cuda:1"), memory(126), {"cuda:1": room(41)},
                      static_bytes = static)
        self.assertEqual(msg, "placement: the stream / cpu layers keep 47.6 GiB of routed experts in RAM, more than ram "
                              "experts=40GiB; raise it or move layers to experts=cache")

    def test_flags_are_caps(self):
        cap = T.Size("bytes", 48 * GiB)
        s = size("*=cuda:0 experts=cache; cuda:0 cache=75GiB", layers(), memory(126), {"cuda:0": room(80)}, expert_cap = cap)
        self.assertEqual(s.tier_slots, 48 * GiB // RS)
        s = size("*=cuda:0 experts=cache; ram experts=auto", layers(), memory(126), {"cuda:0": room(80)}, expert_cap = cap)
        self.assertEqual(s.tier_slots, 48 * GiB // RS)
        self.assertEqual(refused(self, ValueError, "*=cuda:0 experts=cache; ram experts=64GiB", layers(), memory(126),
                                 {"cuda:0": room(80)}, expert_cap = cap),
                         "placement asks ram experts=64GiB (64.0 GiB) but --expert_ram caps this component at 48GiB")
        self.assertIn("placement asks ram experts=all (", refused(self, ValueError, "*=cuda:0 experts=cache; ram experts=all",
                                                                  layers(), memory(126), {"cuda:0": room(80)}, expert_cap = cap))

    def test_draft_pinned_first(self):
        s = size("*=cuda:0 experts=cache", layers(), memory(126), {"cuda:0": room(80)}, already_pinned = 5 * GiB)
        self.assertEqual(s.tier_slots, (126 * GiB - 5 * GiB - 2 * GiB - MEM_TOTAL // 8) // RS)

    def test_staging_single(self):
        cfg = T.TierConfig(staging = "single")
        s = size("*=cuda:0 experts=cache", layers(), memory(126), {"cuda:0": room(80)}, cfg = cfg)
        self.assertEqual(s.devices[0].staging_slots, 384)
        self.assertEqual(s.devices[0].slots, (80 * GiB + 384 * VS) // VS)             # the other half is pool

    def test_everything_fits_in_vram(self):
        # a pool that holds every cached expert needs no spare slots and no RAM tier
        s = size("*=cuda:0 experts=cache; ram experts=all; disk experts=off", layers(range(4)), memory(126),
                 {"cuda:0": room(80)})
        self.assertEqual((s.devices[0].slots, s.devices[0].held, s.tier_slots, s.disk_only), (1536, 1536, 0, 0))

    def test_inclusive(self):
        s = size("*=cuda:0 experts=cache; ram experts=all policy=inclusive", layers(), memory(250), {"cuda:0": room(80)})
        self.assertEqual(s.tier_slots, 15360)
        msg = refused(self, ValueError, "*=cuda:0 experts=cache; ram experts=64GiB policy=inclusive", layers(),
                      memory(250), {"cuda:0": room(80)})
        self.assertEqual(msg, f"placement: policy=inclusive keeps a RAM copy of every expert the VRAM cache holds, so ram "
                              f"experts= must hold more than the cache's 6472 + 8 slots ({T.human(6481 * RS)}); it "
                              f"holds 5159 (64.0 GiB)")

    def test_load_refusals(self):
        # (DESIGN L1) disk experts=off, tiers too small
        self.assertEqual(refused(self, RuntimeError, "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=32GiB; "
                                 "disk experts=off", layers(range(12, 40), device = lambda i: "cuda:1"), memory(126),
                                 {"cuda:1": room(81)}),
                         "placement: disk experts=off needs every cached expert in VRAM or RAM, but 1628 of 10752 fit in "
                         "neither (6545 VRAM slots besides the spare ones, 2579 RAM slots); raise ram experts= by 20.2 GiB "
                         "or allow disk reads")
        # (L2) budgets above free RAM; the disk slab is pinned too
        self.assertEqual(refused(self, RuntimeError, "*=cuda:0 experts=cache; cuda:0 cache=75GiB; ram experts=96GiB "
                                 "ngram=6GiB", layers(), memory(53), {"cuda:0": room(80)}, ngram_bytes = 6 * GiB),
                         "placement: ram experts=96.0 GiB + ngram=6.0 GiB + disk slab=102 MiB need 102.1 GiB of host "
                         "memory, but 53.0 GiB is available and 2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: "
                         "pinned and anonymous memory cannot be reclaimed); lower the budgets or use auto")
        # (L3) cache below its minimum
        self.assertEqual(refused(self, RuntimeError, "*=cuda:0 experts=cache; cuda:0 cache=100MiB; ram experts=8GiB",
                                 layers(), memory(126), {"cuda:0": room(80)}),
                         "placement: cuda:0 cache=100MiB holds 7 expert slots; experts=cache needs at least top-k (6) + "
                         "spare (8) = 14 slots (177 MiB), or cache=0")
        # (L4) cache above the device's room
        self.assertEqual(refused(self, RuntimeError, "*=cuda:0 experts=cache; cuda:0 cache=90GiB; ram experts=8GiB",
                                 layers(), memory(126), {"cuda:0": room(80)}),
                         "placement: cuda:0 cache=90GiB does not fit: 80.0 GiB is free on cuda:0 after the placed "
                         "modules, the cache and the margins (the prefill staging, 9.5 GiB with "
                         "EXL3_MOE_TIER_STAGING=double, and EXL3_MOE_TIER_HEADROOM_MB=1024 are set aside first)")
        # (L8) a RAM tier smaller than one decode step
        self.assertEqual(refused(self, ValueError, "*=cuda:0 experts=cache; ram experts=50MiB", layers(), memory(126),
                                 {"cuda:0": room(80)}),
                         "placement: ram experts=50MiB leaves the RAM tier 3 slots, fewer than one decode step uses "
                         "(top-k = 6); raise it, or use ram experts=0 (a VRAM cache straight over the disk)")
        # staging leaves no room for the smallest cache
        self.assertEqual(refused(self, RuntimeError, "*=cuda:0 experts=cache", layers(), memory(126),
                                 {"cuda:0": STAGING + HEADROOM + 100 * MiB}),
                         "placement: cuda:0 has 100 MiB left for the expert cache after the placed modules, the margins "
                         "and the prefill staging (9.5 GiB, EXL3_MOE_TIER_STAGING=double); an expert cache needs at "
                         "least top-k + spare = 14 slots (177 MiB); set EXL3_MOE_TIER_STAGING=single, raise -gs on "
                         "cuda:0, or move layers")
        # the master switch cannot drop victims the disk does not hold
        self.assertEqual(refused(self, ValueError, "*=cuda:0 experts=cache; ram experts=all; disk experts=off", layers(),
                                 memory(126), cfg = T.TierConfig(demote = False)),
                         "EXL3_MOE_TIER_DEMOTE=0 would drop VRAM victims, but disk experts=off keeps no other copy of them")
        size("*=cuda:0 experts=cache; ram experts=all policy=inclusive; disk experts=off", layers(), memory(250),
             cfg = T.TierConfig(demote = False))
        # a slow host link
        self.assertEqual(refused(self, RuntimeError, "*=cuda:0 experts=cache", layers(), memory(126),
                                 link_gbs = {"cuda:0": 0.38}),
                         "placement: cuda:0 reaches host memory at 0.38 GB/s (measured at load), below the 2.0 GB/s an "
                         "expert cache needs (EXL3_MOE_TIER_MIN_LINK_GBS); keep its layers experts=vram, or set "
                         "EXL3_MOE_TIER_MIN_LINK_GBS=0 to allow it")
        size("*=cuda:0 experts=cache", layers(), memory(126), link_gbs = {"cuda:0": 0.38},
             cfg = T.TierConfig(min_link_gbs = 0.0))
        # the CMP's link does not matter when it holds no cache layer (layout c)
        size("0-11=cuda:0; 12-39=cuda:1 experts=cache", layers(range(12, 40), device = lambda i: "cuda:1"), memory(126),
             link_gbs = {"cuda:0": 0.38, "cuda:1": 26.9})
        self.assertEqual(refused(self, RuntimeError, "*=cuda:0 experts=cache", layers(), memory(126), platform = "win32"),
                         "placement: experts=cache needs the disk engine, which is Linux-only")

    def test_geometry_and_eligibility(self):
        T.check_geometries({"cuda:1": {"a": (5120, 2304, 3), "b": (5120, 2304, 3)}})
        with self.assertRaises(ValueError) as cm:
            T.check_geometries({"cuda:1": {"a": (5120, 2304, 3), "b": (5120, 2304, 4)}})
        self.assertEqual(str(cm.exception), "placement: the experts=cache layers on cuda:1 have different expert shapes "
                                            "((5120, 2304, 3) and (5120, 2304, 4)); one expert slot size per GPU is "
                                            "supported")
        self.assertTrue(T.ineligible("model.layers.3.mlp").startswith(
            "placement: model.layers.3.mlp cannot use experts=cache: it needs the fused MoE kernel"))


class StageOneTests(unittest.TestCase):
    """size_tiers before any module loads: explicit sizes only, auto / all wait for the GPUs' rooms"""

    def test_explicit_sizes_checked_early(self):
        s = size("*=cuda:0 experts=cache; ram experts=96GiB", layers(), memory(126))
        self.assertFalse(s.final)
        self.assertEqual((s.tier_slots, s.ram_bytes, s.devices[0].slots), (7738, 7738 * RS, 0))
        self.assertIn("cuda:0 cache auto (sized at the end of the load, 8 spare", s.text)
        with self.assertRaisesRegex(RuntimeError, r"^placement: ram experts=96.0 GiB \+ disk slab=102 MiB need"):
            size("*=cuda:0 experts=cache; ram experts=96GiB", layers(), memory(53))
        with self.assertRaisesRegex(ValueError, "leaves the RAM tier 3 slots"):
            size("*=cuda:0 experts=cache; ram experts=50MiB", layers(), memory(126))
        with self.assertRaisesRegex(RuntimeError, "cuda:0 cache=100MiB holds 7 expert slots"):
            size("*=cuda:0 experts=cache; cuda:0 cache=100MiB", layers(), memory(126))
        # auto / all: nothing to size yet, nothing refused
        s = size("*=cuda:0 experts=cache; ram experts=all; disk experts=off", layers(), memory(10))
        self.assertEqual((s.tier_slots, s.disk_only, s.ram_sized, s.counted), (0, 0, False, False))
        self.assertIn("(static 0 MiB + tier sized at the end of the load (experts=all))", s.text)
        self.assertIn("disk experts=off io=auto: experts on disk only counted at the end of the load", s.text)
        # explicit cache and RAM sizes are enough to find experts that fit nowhere
        with self.assertRaisesRegex(RuntimeError, "disk experts=off needs every cached expert in VRAM or RAM"):
            size("*=cuda:0 experts=cache; cuda:0 cache=40GiB; ram experts=40GiB; disk experts=off", layers(), memory(126))


class TestLayoutTests(unittest.TestCase):
    """The three test layouts at the plan's estimates (pools with double staging: (a) and (b) about
    65.5 GiB on the PRO alone, (c) about 71.5 GiB on the PRO next to the CMP)"""

    A = "*=cuda:0 experts=cache; ram experts=all; disk experts=off"
    B = "*=cuda:0 experts=cache; ram experts=96GiB"
    C = "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off"

    def test_a_needs_the_whole_vm(self):
        s = size(self.A, layers(), memory(150, total = 160 * GiB), {"cuda:0": room(65.5)})
        self.assertEqual((s.devices[0].slots, s.tier_slots, s.disk_only), (5299, 15360 - 5291, 0))
        self.assertEqual(T.human(s.ram_bytes), "124.9 GiB")
        # under the daily 128 GiB cgroup it cannot fit: the refusal says so, with the cgroup
        msg = refused(self, RuntimeError, self.A, layers(), memory(150, total = 160 * GiB, cgroup_gib = 126),
                      {"cuda:0": room(65.5)})
        self.assertEqual(msg, "placement: ram experts=124.9 GiB needs 124.9 GiB of host memory, but 126.0 GiB is "
                              "available (the memory cgroup /system.slice/run-t.scope allows 126.0 GiB more; MemAvailable "
                              "is 150.0 GiB) and 2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: pinned and "
                              "anonymous memory cannot be reclaimed); lower the budgets or use auto")

    def test_b_ssd_tier(self):
        s = size(self.B, layers(), memory(126, total = 160 * GiB, cgroup_gib = 126), {"cuda:0": room(65.5)})
        self.assertEqual((s.tier_slots, s.disk_only), (7738, 15360 - 5291 - 7738))
        self.assertEqual(T.human(s.disk_only * RS), "28.9 GiB")

    def test_c_daily_recipe(self):
        lay = layers(range(12, 40), device = lambda i: "cuda:1")
        s = size(self.C, lay, memory(126, total = 160 * GiB, cgroup_gib = 126), {"cuda:1": room(71.5)},
                 link_gbs = {"cuda:0": 0.38, "cuda:1": 26.9})
        self.assertEqual((s.devices[0].slots, s.tier_slots, s.disk_only, s.slab_slots), (5784, 10752 - 5776, 0, 0))
        self.assertEqual(T.human(s.ram_bytes), "61.7 GiB")


if __name__ == "__main__":
    unittest.main(verbosity = 2)
