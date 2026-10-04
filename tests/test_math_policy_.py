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
               "EXL3_NO_FUSED_RECONSTRUCT", "EXL3_EXACT_ROWS", "EXL3_DSV41_FUSED_COMPRESS")


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


class ExactRowsTests(unittest.TestCase):

    def test_accepted_values(self):
        policy = load({})
        for environ, expected in (({}, False), ({"EXL3_EXACT_ROWS": "0"}, False), ({"EXL3_EXACT_ROWS": "1"}, True),
                                  ({"EXL3_EXACT_ROWS": "1", "EXL3_STABLE_ARITHMETIC": "0"}, True),
                                  ({"EXL3_EXACT_ROWS": "1", "EXL3_DSV41_FUSED_COMPRESS": "0"}, True),
                                  ({"EXL3_EXACT_ROWS": "0", "EXL3_STABLE_ARITHMETIC": "1"}, False),
                                  ({"EXL3_EXACT_ROWS": "0", "EXL3_DSV41_FUSED_COMPRESS": "1"}, False)):
            with self.subTest(environ = environ):
                self.assertIs(policy.exact_rows_enabled(environ), expected)
                self.assertIs(load(environ).EXACT_ROWS, expected)
        self.assertEqual(policy.EXACT_ROWS_MAX, 8)

    def test_other_values_refused(self):
        policy = load({})
        for value in ("", "2", "-1", "true", "yes", "8", " 1", "1 "):
            with self.subTest(value = value), self.assertRaisesRegex(ValueError, "EXL3_EXACT_ROWS must be 0 or 1"):
                policy.exact_rows_enabled({"EXL3_EXACT_ROWS": value})

    def test_refused_combinations(self):
        # Each names both variables. The stable profile replaces the one-row arithmetic this mode keeps;
        # the fused compressor is on for any value but 0 (cache/dsv41.py, FUSED_COMPRESS)
        policy = load({})
        for extra, other in (({"EXL3_STABLE_ARITHMETIC": "1"}, "EXL3_STABLE_ARITHMETIC"),
                             ({"EXL3_DSV41_FUSED_COMPRESS": "1"}, "EXL3_DSV41_FUSED_COMPRESS"),
                             ({"EXL3_DSV41_FUSED_COMPRESS": ""}, "EXL3_DSV41_FUSED_COMPRESS"),
                             ({"EXL3_DSV41_FUSED_COMPRESS": "yes"}, "EXL3_DSV41_FUSED_COMPRESS")):
            environ = {"EXL3_EXACT_ROWS": "1", **extra}
            with self.subTest(extra = extra):
                with self.assertRaises(ValueError) as caught:
                    policy.exact_rows_enabled(environ)
                self.assertIn("EXL3_EXACT_ROWS", str(caught.exception))
                self.assertIn(other, str(caught.exception))
                with self.assertRaises(ValueError):
                    load(environ)
        # an invalid value of the other profile is that profile's error
        with self.assertRaisesRegex(ValueError, "EXL3_STABLE_ARITHMETIC must be 0 or 1"):
            policy.exact_rows_enabled({"EXL3_EXACT_ROWS": "1", "EXL3_STABLE_ARITHMETIC": "2"})

    def test_row_range(self):
        policy = load({})
        for rows, inside in ((0, False), (1, False), (2, True), (8, True), (9, False)):
            with self.subTest(rows = rows):
                self.assertIs(policy.exact_rows(rows, True), inside)
                self.assertIs(policy.exact_rows(rows, False), False)
                # the import-time constant is the default
                self.assertIs(policy.exact_rows(rows), False)
                self.assertIs(load({"EXL3_EXACT_ROWS": "1"}).exact_rows(rows), inside)

    def test_invalid_value_fails_the_import(self):
        code = f"import importlib.util as u; s = u.spec_from_file_location('p', {str(PATH)!r}); s.loader.exec_module(u.module_from_spec(s))"
        for extra, message in (({"EXL3_EXACT_ROWS": "2"}, "ValueError: EXL3_EXACT_ROWS must be 0 or 1"),
                               ({"EXL3_EXACT_ROWS": "1", "EXL3_STABLE_ARITHMETIC": "1"}, "EXL3_STABLE_ARITHMETIC=1")):
            environ = {k: v for k, v in os.environ.items() if k not in POLICY_VARS}
            environ.update(extra)
            with self.subTest(extra = extra):
                r = subprocess.run([sys.executable, "-c", code], env = environ, capture_output = True, text = True)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(message, r.stderr)


if __name__ == "__main__":
    unittest.main()
