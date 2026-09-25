"""
CPU-only tests of the host-memory budget helpers (exllamav3/util/host_budget.py): sizes with units,
the host memory available to the process (MemAvailable and the memory cgroup), the ledger and its
refusal, and check_host_memory built on them.

    python tests/test_host_budget_.py

host_budget.py is torch-free and loaded by path; the check_host_memory case imports
exllamav3.util.memory (Torch, no GPU).
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


HB = load("_host_budget_subject", "exllamav3/util/host_budget.py")
GiB, MiB = 1 << 30, 1 << 20


def fake_host(root, mem_available_kib, mem_total_kib = 167_772_160, swap_kib = 0, cgroup = None):
    """A /proc and /sys/fs/cgroup under `root`. cgroup: {relative path: {file: text}}, the first
    key being the process's own cgroup"""
    proc = os.path.join(root, "proc")
    os.makedirs(os.path.join(proc, "self"), exist_ok = True)
    with open(os.path.join(proc, "meminfo"), "w") as f:
        f.write(f"MemTotal:       {mem_total_kib} kB\nMemFree:        1000 kB\n"
                f"MemAvailable:   {mem_available_kib} kB\nSwapTotal:      {swap_kib} kB\n")
    cg_root = os.path.join(root, "cgroup")
    own = next(iter(cgroup)) if cgroup else ""
    with open(os.path.join(proc, "self", "cgroup"), "w") as f:
        f.write(f"0::/{own}\n")
    for rel, files in (cgroup or {}).items():
        d = os.path.join(cg_root, rel)
        os.makedirs(d, exist_ok = True)
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text + "\n")
    os.makedirs(cg_root, exist_ok = True)
    return proc, cg_root


class SizeTests(unittest.TestCase):

    def test_units(self):
        cases = {"48GiB": 51539607552, "48GB": 48_000_000_000, "48": 51539607552, "48gi": 48 * GiB,
                 "0": 0, "1.5GiB": 3 * GiB // 2, "512MiB": 512 * MiB, "2TiB": 2 << 40, "4096B": 4096,
                 "64KiB": 65536, "64kb": 64000, "0.5": GiB // 2, "3TB": 3 * 10 ** 12, "1Ti": 1 << 40}
        for text, n in cases.items():
            with self.subTest(text):
                s = HB.parse_size(text, "x")
                self.assertEqual((s.kind, s.nbytes), ("bytes", n))
        self.assertEqual(str(HB.parse_size("1.5", "x")), "1536MiB")
        self.assertEqual(str(HB.parse_size("512mi", "x")), "512MiB")
        self.assertEqual(str(HB.parse_size("48GB", "x")), "48GB")
        self.assertEqual(str(HB.parse_size("49152MiB", "x")), "48GiB")
        self.assertEqual(str(HB.parse_size("0", "x")), "0")
        # decimals are exact (no float rounding): 0.1 GB is 100,000,000 bytes
        self.assertEqual(HB.parse_size("0.1GB", "x").nbytes, 100_000_000)

    def test_words(self):
        self.assertEqual(HB.parse_size("auto", "x"), HB.AUTO)
        self.assertEqual(HB.parse_size(" ALL ", "x"), HB.ALL)
        self.assertTrue(HB.parse_size("all", "x").is_all)
        self.assertTrue(HB.parse_size("0", "x").is_zero)
        with self.assertRaisesRegex(ValueError, r"^pagecache=all: all is not allowed here \(use auto or a size\)$"):
            HB.parse_size("all", "pagecache", allow = ("auto",))
        with self.assertRaisesRegex(ValueError, r"^cap=auto: auto is not allowed here \(use a size\)$"):
            HB.parse_size("auto", "cap", allow = ())

    def test_refusals(self):
        self.assertEqual(self._msg("48G", "experts"),
                         "experts=48G: '48G' is ambiguous: write 48GiB (2^30 bytes) or 48GB (10^9 bytes)")
        self.assertEqual(self._msg("512m", "ngram"),
                         "ngram=512m: '512m' is ambiguous: write 512MiB (2^20 bytes) or 512MB (10^6 bytes)")
        self.assertEqual(self._msg("1T", "x"), "x=1T: '1T' is ambiguous: write 1TiB (2^40 bytes) or 1TB (10^12 bytes)")
        not_a_size = ("not a size (use 0, auto, all, or a number with B, KiB, MiB, GiB, TiB, KB, MB, GB or TB "
                      "(a bare number is GiB))")
        for text in ("48GIGS", "٣GiB", "-1GiB", "1e3", "48 GiB", "", "GiB", "1.GiB", ".5GiB", "0x10"):
            with self.subTest(text):
                self.assertEqual(self._msg(text, "experts"), f"experts={text.strip()}: {not_a_size}")

    def _msg(self, text, what):
        with self.assertRaises(ValueError) as cm:
            HB.parse_size(text, what)
        return str(cm.exception)

    def test_equality_and_printing(self):
        a = HB.parse_size("48GiB", "x")
        self.assertEqual(a, HB.Size("bytes", 48 * GiB))
        self.assertEqual(hash(a), hash(HB.Size("bytes", 48 * GiB)))
        self.assertNotEqual(a, HB.ALL)
        self.assertEqual(HB.fmt_size(48 * 10 ** 9), "48GB")
        self.assertEqual(HB.fmt_size(3 * GiB // 2), "1536MiB")
        self.assertEqual(HB.fmt_size(13_271_040), "12960KiB")
        self.assertEqual(HB.fmt_size(13_315_585), "13315585B")
        self.assertEqual(HB.human(52 * GiB + GiB // 2), "52.5 GiB")
        self.assertEqual(HB.human(177 * MiB), "177 MiB")
        # a Size from another copy of the module (loaded by path vs through the package) compares
        other = load("_host_budget_copy", "exllamav3/util/host_budget.py")
        self.assertEqual(other.parse_size("48GiB", "x"), a)
        self.assertEqual(other.ALL, HB.ALL)


class MemoryTests(unittest.TestCase):

    def test_meminfo(self):
        with tempfile.TemporaryDirectory() as d:
            proc, cg = fake_host(d, 53 * GiB // 1024, swap_kib = 0)
            m = HB.host_memory(proc, cg)
            self.assertEqual((m.available, m.mem_available, m.swap_total, m.cgroup_room), (53 * GiB, 53 * GiB, 0, None))
            self.assertFalse(m.cgroup_limits)

    def test_cgroup_limits(self):
        with tempfile.TemporaryDirectory() as d:
            # a scope under a slice: the scope has the 128 GiB limit, 20 GiB of it in use
            proc, cg = fake_host(d, 150 * GiB // 1024, cgroup = {
                "system.slice/run-x.scope": {"memory.max": "137438953472", "memory.current": "21474836480"},
                "system.slice": {"memory.max": "max", "memory.current": "99999999999"},
            })
            m = HB.host_memory(proc, cg)
            self.assertEqual(m.available, 108 * GiB)
            self.assertEqual((m.cgroup_room, m.cgroup_path, m.mem_available), (108 * GiB, "/system.slice/run-x.scope", 150 * GiB))
            self.assertTrue(m.cgroup_limits)
            ledger = HB.HostLedger(memory = m, reserve = 2 * GiB)
            ledger.add("ram experts", 110 * GiB)
            with self.assertRaises(RuntimeError) as cm:
                ledger.check("placement")
            self.assertEqual(str(cm.exception),
                             "placement: ram experts=110.0 GiB needs 110.0 GiB of host memory, but 108.0 GiB is "
                             "available (the memory cgroup /system.slice/run-x.scope allows 108.0 GiB more; "
                             "MemAvailable is 150.0 GiB) and 2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) "
                             "(no swap: pinned and anonymous memory cannot be reclaimed); lower the budgets or use auto")

    def test_cgroup_page_cache_is_reclaimable(self):
        # the loader reads whole checkpoints through the page cache; the cgroup charges it but
        # drops it before refusing a charge, so the file LRU counts as room
        with tempfile.TemporaryDirectory() as d:
            proc, cg = fake_host(d, 150 * GiB // 1024, cgroup = {"a": {
                "memory.max": str(128 * GiB), "memory.current": str(120 * GiB),
                "memory.stat": f"anon {10 * GiB}\nfile {110 * GiB}\nshmem {4 * GiB}\n"
                               f"active_file {50 * GiB}\ninactive_file {56 * GiB}\n"}})
            self.assertEqual(HB.host_memory(proc, cg).available, (128 - 120 + 106) * GiB)

    def test_cgroup_high_and_unlimited(self):
        with tempfile.TemporaryDirectory() as d:
            proc, cg = fake_host(d, 150 * GiB // 1024, cgroup = {"a/b": {
                "memory.max": "max", "memory.high": str(64 * GiB), "memory.current": str(4 * GiB)}})
            self.assertEqual(HB.host_memory(proc, cg).available, 60 * GiB)
        with tempfile.TemporaryDirectory() as d:
            proc, cg = fake_host(d, 150 * GiB // 1024, cgroup = {"a": {"memory.max": "max", "memory.current": "1"}})
            m = HB.host_memory(proc, cg)
            self.assertEqual((m.available, m.cgroup_room), (150 * GiB, None))
        # a cgroup whose limit is above MemAvailable does not limit
        with tempfile.TemporaryDirectory() as d:
            proc, cg = fake_host(d, 50 * GiB // 1024, cgroup = {"a": {"memory.max": str(128 * GiB), "memory.current": "0"}})
            m = HB.host_memory(proc, cg)
            self.assertEqual(m.available, 50 * GiB)
            self.assertFalse(m.cgroup_limits)

    def test_unknown_platform(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {"psutil": None}):
            m = HB.host_memory(os.path.join(d, "none"), os.path.join(d, "none"))
            self.assertIsNone(m.available)
            ledger = HB.HostLedger(memory = m, reserve = GiB)
            ledger.add("ram experts", 1 << 50)
            ledger.check("placement")          # unknown: nothing to check against


class LedgerTests(unittest.TestCase):

    def test_design_message(self):
        m = HB.HostMemory(53 * GiB, 53 * GiB, 160 * GiB, 0)
        ledger = HB.HostLedger(memory = m, reserve = 2 * GiB)
        ledger.add("ram experts", 96 * GiB)
        ledger.add("ngram", 6 * GiB)
        ledger.add("nothing", 0)                # zero items are not listed
        self.assertEqual(ledger.total, 102 * GiB)
        with self.assertRaises(RuntimeError) as cm:
            ledger.check("placement")
        self.assertEqual(str(cm.exception),
                         "placement: ram experts=96.0 GiB + ngram=6.0 GiB need 102.0 GiB of host memory, but 53.0 GiB "
                         "is available and 2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: pinned and "
                         "anonymous memory cannot be reclaimed); lower the budgets or use auto")

    def test_fits_and_reserve(self):
        m = HB.HostMemory(104 * GiB, 104 * GiB, 160 * GiB, 8 * GiB)
        ledger = HB.HostLedger(memory = m, reserve = 2 * GiB)
        ledger.add("ram experts", 96 * GiB)
        ledger.add("ngram", 6 * GiB)
        ledger.check("placement")              # 102 + 2 = 104: fits exactly
        ledger.add("staging", 1)
        with self.assertRaisesRegex(RuntimeError, r"need 102\.0 GiB .* kept free \(EXL3_HOST_MEM_RESERVE_MB\); lower"):
            ledger.check("placement")          # swap present: no swap note
        # reserve 0 disables the check
        HB.HostLedger(memory = m, reserve = 0, items = [HB.LedgerItem("ram experts", 1 << 50)]).check("x")
        with patch.dict(os.environ, {"EXL3_HOST_MEM_RESERVE_MB": "0"}):
            self.assertEqual(HB.reserve_bytes(), 0)
            HB.HostLedger(memory = m, items = [HB.LedgerItem("ram experts", 1 << 50)]).check("x")
        with patch.dict(os.environ, {"EXL3_HOST_MEM_RESERVE_MB": "-5"}):
            self.assertEqual(HB.reserve_bytes(), 0)
        with patch.dict(os.environ, {}, clear = True):
            self.assertEqual(HB.reserve_bytes(), 2048 * MiB)
        with patch.dict(os.environ, {"EXL3_HOST_MEM_RESERVE_MB": "2GiB"}), \
                self.assertRaisesRegex(ValueError, "EXL3_HOST_MEM_RESERVE_MB='2GiB' must be a whole number of MiB"):
            HB.reserve_bytes()


class CheckHostMemoryTests(unittest.TestCase):
    """exllamav3.util.memory.check_host_memory: the per-allocation guard, now cgroup-aware, with its
    message unchanged"""

    def test_delegates(self):
        from exllamav3.util import memory, host_budget
        fake = host_budget.HostMemory(10 * GiB, 20 * GiB, 160 * GiB, 0, 10 * GiB, "/x")
        with patch.object(host_budget, "host_memory", lambda *a, **k: fake), \
                patch.dict(os.environ, {"EXL3_HOST_MEM_RESERVE_MB": "1024"}):
            memory.check_host_memory(9 * GiB, "thing")
            with self.assertRaises(RuntimeError) as cm:
                memory.check_host_memory(9 * GiB + 1, "thing")
            self.assertEqual(str(cm.exception),
                             "thing needs 9216 MiB of host memory, but only 10240 MiB is available and "
                             "EXL3_HOST_MEM_RESERVE_MB = 1024 MiB is kept free. Lower the amount of data held in "
                             "host memory (fewer offloaded experts, no --ngram_ram, ...) or set "
                             "EXL3_HOST_MEM_RESERVE_MB=0 to skip this check.")
        with patch.object(host_budget, "host_memory", lambda *a, **k: fake), \
                patch.dict(os.environ, {"EXL3_HOST_MEM_RESERVE_MB": "0"}):
            memory.check_host_memory(1 << 50, "thing")


if __name__ == "__main__":
    unittest.main(verbosity = 2)
