"""
CPU-only tests of expert profiles and the heat file (exllamav3/model/moe_profile.py;
doc/expert_tiers.md, "Profiles and the heat file"): every file format (.exl3moe / .safetensors with
counts or a ranking, .npz with and without a sidecar, .json), the search directories, weighted specs
normalized per layer, layer matching by key and by ordinal, the model identity check, the seeds the
expert tier takes from a profile (hot pins, cold-fill order, heat), and the heat file's round trip.

    python tests/test_moe_profile_.py

moe_profile.py is torch-free (numpy) and loaded by path.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_moe_profile_subject", ROOT / "exllamav3/model/moe_profile.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)

KEYS = [f"model.layers.{i}.mlp" for i in range(3)]
E = 8
FP = {"architecture": "TestMoE", "layers": 3, "experts": E, "moe_intermediate_size": 64, "hidden_size": 128}


def counts(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1000, size = (3, E)).astype(np.float64)


class Files(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix = "exl3-profile-")
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.d, name)

    def test_formats_agree(self):
        c = counts(1)
        M.write_safetensors(self.path("a.exl3moe"), {"counts": c}, {"layer_keys": KEYS, "fingerprint": FP})
        np.savez(self.path("b.npz"), counts_decode = np.stack([c, 0 * c]), counts_prefill = 5 * c)
        with open(self.path("b.meta.json"), "w") as f:
            json.dump({"layer_keys": KEYS, "fingerprint": FP}, f)
        with open(self.path("c.json"), "w") as f:
            json.dump({k: list(r) for k, r in zip(KEYS, c)}, f)
        rank = np.argsort(-c, axis = 1, kind = "stable")
        M.write_safetensors(self.path("d.safetensors"), {"ranking": rank.astype(np.int64)}, {"layer_keys": KEYS})
        for name in ("a.exl3moe", "b.npz", "c.json", "d.safetensors"):
            with self.subTest(name):
                got, keys, fp = M.load_counts(self.path(name))
                self.assertEqual(keys, KEYS)
                self.assertEqual(np.argsort(-got, axis = 1, kind = "stable").tolist(), rank.tolist())
                if name in ("a.exl3moe", "b.npz"):
                    self.assertEqual(fp, FP)
        # an .npz without a sidecar is positional; decode counts are preferred, prompts summed
        np.savez(self.path("e.npz"), counts = c)
        got, keys, _ = M.load_counts(self.path("e.npz"))
        self.assertEqual((keys, got.tolist()), (None, c.tolist()))

    def test_refusals(self):
        np.savez(self.path("x.npz"), other = np.zeros((3, E)))
        with self.assertRaisesRegex(ValueError, "no counts_decode, counts or counts_prefill array"):
            M.load_counts(self.path("x.npz"))
        M.write_safetensors(self.path("y.exl3moe"), {"counts": -counts(2)}, {})
        with self.assertRaisesRegex(ValueError, "counts must be finite and not negative"):
            M.load_counts(self.path("y.exl3moe"))
        M.write_safetensors(self.path("z.exl3moe"), {"counts": counts(2)}, {"layer_keys": KEYS[:2]})
        with self.assertRaisesRegex(ValueError, "2 layer keys for 3 rows of counts"):
            M.load_counts(self.path("z.exl3moe"))
        with open(self.path("w.txt"), "w") as f:
            f.write("x")
        with self.assertRaisesRegex(ValueError, "not a profile"):
            M.load_counts(self.path("w.txt"))
        for spec, msg in (("", "names no profile"), ("a:-1", "must be a finite number >= 0"),
                          ("a:0,b:0", "every weight is 0")):
            with self.assertRaisesRegex(ValueError, msg):
                M.parse_spec(spec)
        self.assertEqual(M.parse_spec("code:3, wiki ,chat:0.5"), [("code", 3.0), ("wiki", 1.0), ("chat", 0.5)])

    def test_search_dirs(self):
        model = self.path("model")
        os.makedirs(os.path.join(model, "moe_profiles"))
        other = self.path("other")
        os.makedirs(other)
        M.write_safetensors(os.path.join(other, "code.npz.exl3moe"), {"counts": counts(3)}, {})
        M.write_safetensors(os.path.join(model, "moe_profiles", "code.exl3moe"), {"counts": counts(4)}, {})
        M.write_safetensors(os.path.join(other, "code.exl3moe"), {"counts": counts(5)}, {})
        M.write_safetensors(os.path.join(other, "chat.json.exl3moe"), {"counts": counts(5)}, {})
        with patch.dict(os.environ, {"EXL3_MOE_PROFILE_DIR": other}):
            # the model's own directory first, then EXL3_MOE_PROFILE_DIR
            self.assertEqual(M.resolve_path("code", model), os.path.join(model, "moe_profiles", "code.exl3moe"))
            self.assertEqual(M.resolve_path("code", None), os.path.join(other, "code.exl3moe"))
            with self.assertRaisesRegex(ValueError, "expert profile 'nope' not found: looked in"):
                M.resolve_path("nope", model)
        p = os.path.join(other, "code.exl3moe")
        self.assertEqual(M.resolve_path(p, None), p)

    def test_profile_merge_and_seeds(self):
        a, b = counts(6), counts(7)
        M.write_safetensors(self.path("a.exl3moe"), {"counts": a}, {"layer_keys": KEYS, "fingerprint": FP})
        np.savez(self.path("b.npz"), counts_decode = b)          # positional: matched by MoE ordinal
        prof = M.Profile(f"{self.path('a.exl3moe')}:3,{self.path('b.npz')}", KEYS, E, None, FP)
        for i, k in enumerate(KEYS):
            want = 3 * a[i] / a[i].sum() + b[i] / b[i].sum()
            want = want / want.sum()
            np.testing.assert_allclose(prof.shares(k), want, rtol = 1e-12)
            self.assertEqual(prof.ranking(k), [int(e) for e in np.argsort(-want, kind = "stable")])
        self.assertIsNone(prof.shares("model.layers.9.mlp"))
        # another model: refused; another quantization of the same model: used
        with self.assertRaisesRegex(ValueError, r"was built for another model \(hidden_size: profile 128, model 256\)"):
            M.Profile(self.path("a.exl3moe"), KEYS, E, None, dict(FP, hidden_size = 256))
        M.Profile(self.path("a.exl3moe"), KEYS, E, None, dict(FP, bits = 3.0))
        with self.assertRaisesRegex(ValueError, "8 experts per layer, the model's MoE layers have 16"):
            M.Profile(self.path("a.exl3moe"), KEYS, 16, None, FP)
        np.savez(self.path("c.npz"), counts_decode = np.ones((2, E)))
        with self.assertRaisesRegex(ValueError, "names no layers and has 2 rows, the model has 3 MoE layers"):
            M.Profile(self.path("c.npz"), KEYS, E, None, FP)
        # cold-fill order: by share across layers, the pins left out; a layer without shares last, round robin
        shares = {KEYS[0]: np.array([0.5, 0.3, 0.2]), KEYS[1]: np.array([0.1, 0.1, 0.8])}
        order = M.seed_order([(0, KEYS[0]), (1, KEYS[1]), (2, KEYS[2])], shares, 3, pinned = {5})
        self.assertEqual(order, [0, 1, 2, 3, 4, 6, 7, 8])
        # steady-decode heat: share x top-k x halflife / ln 2 in 16.16
        self.assertEqual(M.seed_heat(0.25, 6, 256), int(round(0.25 * 6 * 256 / np.log(2) * 65536)))
        self.assertEqual(M.seed_heat(10.0, 8, 1 << 20), (1 << 32) - 1)

    def test_heat_file_round_trip(self):
        rows = np.array([[0.0, 1.5, 2.25], [3.0, 0.0, 0.5]])
        path = self.path("heat.exl3moe")
        M.write_safetensors(path, {"counts": rows}, {"layer_keys": KEYS[:2], "fingerprint": FP,
                                                     "format": "exl3-heat 1"})
        got, keys, fp = M.load_counts(path)
        self.assertEqual((got.tolist(), keys, fp), (rows.tolist(), KEYS[:2], FP))
        self.assertEqual([f for f in os.listdir(self.d) if ".tmp." in f], [])       # written atomically


if __name__ == "__main__":
    unittest.main(verbosity = 2)
