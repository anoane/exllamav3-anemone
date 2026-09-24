"""
CPU-only checks of the opt-in arithmetic policy parser (exllamav3/model/math_policy.py): accepted
values, refusals, and the constants read when the module is imported. The module is torch-free and
is loaded by path, so neither torch nor the compiled extension is needed.

    python -m pytest tests/test_math_policy_.py
"""
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "exllamav3" / "model" / "math_policy.py"
POLICY_VARS = ("EXL3_STABLE_ARITHMETIC", "EXL3_HGEMM_FIXED_ROWS", "EXL3_MOE_FUSED_PREFILL", "EXL3_MOE_FUSED_DET",
               "EXL3_NO_FUSED_RECONSTRUCT")


def load(environ):
    """Import a fresh copy of the module under `environ` (only the policy variables set)"""
    with patch.dict(os.environ, environ, clear = False):
        for name in POLICY_VARS:
            if name not in environ:
                os.environ.pop(name, None)
        spec = importlib.util.spec_from_file_location("math_policy_under_test", PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FixedRowsTests(unittest.TestCase):

    def test_accepted_values(self):
        policy = load({})
        for environ, expected in (({}, 0), ({"EXL3_HGEMM_FIXED_ROWS": "0"}, 0), ({"EXL3_HGEMM_FIXED_ROWS": "128"}, 128)):
            with self.subTest(environ = environ):
                self.assertEqual(policy.hgemm_fixed_rows(environ), expected)
                self.assertEqual(load(environ).HGEMM_FIXED_ROWS, expected)

    def test_other_values_refused(self):
        # The native parser (hgemm.cu) accepts exactly the same two spellings
        policy = load({})
        for value in ("", "1", "64", "256", "0128", " 128", "128 ", "true"):
            with self.subTest(value = value), self.assertRaisesRegex(ValueError, "EXL3_HGEMM_FIXED_ROWS"):
                policy.hgemm_fixed_rows({"EXL3_HGEMM_FIXED_ROWS": value})

    def test_invalid_value_fails_the_import(self):
        # Read when the package is imported: a bad value stops the import, before any model loads
        code = f"import importlib.util as u; s = u.spec_from_file_location('p', {str(PATH)!r}); s.loader.exec_module(u.module_from_spec(s))"
        environ = dict(os.environ, EXL3_HGEMM_FIXED_ROWS = "64")
        r = subprocess.run([sys.executable, "-c", code], env = environ, capture_output = True, text = True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ValueError: EXL3_HGEMM_FIXED_ROWS must be 0 or 128", r.stderr)


class FusedPrefillTests(unittest.TestCase):

    def test_accepted_values(self):
        policy = load({})
        for environ, expected in (({}, False), ({"EXL3_MOE_FUSED_PREFILL": "0"}, False),
                                  ({"EXL3_MOE_FUSED_PREFILL": "1"}, True),
                                  ({"EXL3_MOE_FUSED_PREFILL": "1", "EXL3_MOE_FUSED_DET": "1"}, True),
                                  ({"EXL3_MOE_FUSED_PREFILL": "0", "EXL3_MOE_FUSED_DET": "0"}, False)):
            with self.subTest(environ = environ):
                self.assertEqual(policy.fused_prefill_enabled(environ), expected)
                self.assertEqual(load(environ).FUSED_PREFILL, expected)
        self.assertEqual(policy.FUSED_COUNT_LIMIT, 2 ** 31 - 1)

    def test_other_values_refused(self):
        policy = load({})
        for value in ("", "2", "-1", "true", "yes", "128", " 1"):
            with self.subTest(value = value), self.assertRaisesRegex(ValueError, "EXL3_MOE_FUSED_PREFILL"):
                policy.fused_prefill_enabled({"EXL3_MOE_FUSED_PREFILL": value})

    def test_atomic_accumulation_refused(self):
        # The setting exists for reproducible arithmetic, which atomic accumulation would undo
        policy = load({})
        with self.assertRaisesRegex(ValueError, "EXL3_MOE_FUSED_DET=0"):
            policy.fused_prefill_enabled({"EXL3_MOE_FUSED_PREFILL": "1", "EXL3_MOE_FUSED_DET": "0"})


class StableArithmeticTests(unittest.TestCase):

    def test_off_by_default(self):
        for environ in ({}, {"EXL3_STABLE_ARITHMETIC": "0"}):
            module = load(environ)
            self.assertEqual((module.STABLE_ARITHMETIC, module.HGEMM_FIXED_ROWS, module.FUSED_PREFILL), (False, 0, False))

    def test_one_variable_implies_the_others(self):
        # Unset, or set to the implied value: both accepted (drivers that set all three still work)
        for extra in ({}, {"EXL3_HGEMM_FIXED_ROWS": "128"}, {"EXL3_MOE_FUSED_PREFILL": "1"},
                      {"EXL3_HGEMM_FIXED_ROWS": "128", "EXL3_MOE_FUSED_PREFILL": "1", "EXL3_MOE_FUSED_DET": "1"},
                      {"EXL3_NO_FUSED_RECONSTRUCT": "0"}):
            with self.subTest(extra = extra):
                module = load({"EXL3_STABLE_ARITHMETIC": "1", **extra})
                self.assertEqual((module.STABLE_ARITHMETIC, module.HGEMM_FIXED_ROWS, module.FUSED_PREFILL), (True, 128, True))

    def test_other_values_refused(self):
        policy = load({})
        for value in ("", "2", "true", "yes", " 1"):
            with self.subTest(value = value), self.assertRaisesRegex(ValueError, "EXL3_STABLE_ARITHMETIC must be 0 or 1"):
                policy.stable_arithmetic_enabled({"EXL3_STABLE_ARITHMETIC": value})

    def test_contradictions_refused_by_every_reader(self):
        policy = load({})
        for extra, message in (({"EXL3_HGEMM_FIXED_ROWS": "0"}, "implies EXL3_HGEMM_FIXED_ROWS=128"),
                               ({"EXL3_MOE_FUSED_PREFILL": "0"}, "implies EXL3_MOE_FUSED_PREFILL=1"),
                               ({"EXL3_MOE_FUSED_DET": "0"}, "EXL3_MOE_FUSED_DET=0"),
                               ({"EXL3_NO_FUSED_RECONSTRUCT": "1"}, "EXL3_NO_FUSED_RECONSTRUCT"),
                               ({"EXL3_NO_FUSED_RECONSTRUCT": ""}, "EXL3_NO_FUSED_RECONSTRUCT")):
            environ = {"EXL3_STABLE_ARITHMETIC": "1", **extra}
            for reader in (policy.stable_arithmetic_enabled, policy.hgemm_fixed_rows, policy.fused_prefill_enabled):
                with self.subTest(extra = extra, reader = reader.__name__), self.assertRaisesRegex(ValueError, message):
                    reader(environ)

    def test_only_the_variable_in_a_fresh_process(self):
        # The import-time constants when EXL3_STABLE_ARITHMETIC=1 is the only policy variable set
        code = ("import importlib.util as u; s = u.spec_from_file_location('p', %r); m = u.module_from_spec(s); "
                "s.loader.exec_module(m); print(m.STABLE_ARITHMETIC, m.HGEMM_FIXED_ROWS, m.FUSED_PREFILL)" % str(PATH))
        environ = {k: v for k, v in os.environ.items() if k not in POLICY_VARS}
        environ["EXL3_STABLE_ARITHMETIC"] = "1"
        r = subprocess.run([sys.executable, "-c", code], env = environ, capture_output = True, text = True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["True", "128", "True"])


if __name__ == "__main__":
    unittest.main()
