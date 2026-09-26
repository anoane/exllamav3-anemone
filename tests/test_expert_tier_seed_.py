"""
CPU-only tests of how a GPU's expert cache is seeded at load (exllamav3/model/expert_tier.py,
ExpertTierSet.seed and Seed; doc/expert_tiers.md, "Profiles and the heat file"): with no profile
and no heat file, the first hot= experts are pinned and the cold fill runs round robin; a profile=
of the layer rules pins each layer's most selected experts, orders the cold fill by share and
seeds the steady-decode heat; without a profile, the heat file (EXL3_MOE_HEAT_FILE) does the same
from the saved heat, which it gives back exactly; a heat file of another model, one that cannot be
read, or one that names other layers seeds nothing; a profile wins over the heat file.

    python -m pytest -q tests/test_expert_tier_seed_.py
"""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exllamav3.model import moe_profile
from exllamav3.model.expert_tier import ExpertTierSet, TierDevice, Seed
from exllamav3.model.expert_tier_config import TierConfig
from exllamav3.model.expert_tier_policy import round_robin_order
from exllamav3.model.placement import parse

E, TOP_K, HALFLIFE = 8, 2, 256
LAYERS = [3, 4, 5]
KEYS = [f"model.layers.{i}.mlp" for i in LAYERS]
CD = {"architectures": ["TestMoE"], "num_hidden_layers": 6, "n_routed_experts": E, "moe_intermediate_size": 64,
      "hidden_size": 128}


def tier_set(placement_text, heat_file = None, hot = 0):
    placement = parse(placement_text) if placement_text else None
    if placement is not None:
        placement.layer_map(range(6))
    ts = ExpertTierSet(NS(config_dict = CD, directory = None), placement, None,
                       tier_config = TierConfig(heat_file = heat_file), moe_keys = KEYS)
    td = TierDevice(ts, torch.device("cuda", 0))
    for i, k in zip(LAYERS, KEYS):
        td.register(NS(key = k, num_experts = E, num_experts_per_tok = TOP_K), hot, i)
    return ts, td


def counts(seed):
    return np.random.default_rng(seed).permutation(np.arange(1, E + 1) * 10.0)


class Seeds(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix = "exl3-seed-")
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_seed(self):
        ts, td = tier_set("*=cuda:0 experts=cache hot=2", os.path.join(self.d, "missing.exl3moe"), hot = 2)
        s = ts.seed(td, E, HALFLIFE)
        self.assertEqual((s.shares, s.heat), ({}, {}))
        self.assertIn("round robin", s.source)
        self.assertEqual([s.pins(l, E) for l in td.layers], [[0, 1]] * 3)
        self.assertEqual(s.order(td, E, set()), round_robin_order(3, E, set()))

    def test_profile(self):
        rows = np.stack([counts(i) for i in range(3)])
        path = os.path.join(self.d, "code.exl3moe")
        fp = moe_profile.model_fingerprint(NS(config_dict = CD), E)
        moe_profile.write_safetensors(path, {"counts": rows}, {"layer_keys": KEYS, "fingerprint": fp})
        ts, td = tier_set(f'*=cuda:0 experts=cache hot=2 profile="{path}"', hot = 2)
        s = ts.seed(td, E, HALFLIFE)
        self.assertEqual(s.source, f"profile {path}")
        for lc, l in enumerate(td.layers):
            share = rows[lc] / rows[lc].sum()
            self.assertEqual(s.pins(l, E), sorted(int(e) for e in np.argsort(-share, kind = "stable")[:2]))
            for e in range(E):
                self.assertEqual(s.heat[lc * E + e], moe_profile.seed_heat(float(share[e]), TOP_K, HALFLIFE))
        order = s.order(td, E, set())
        self.assertEqual(sorted(order), list(range(3 * E)))
        firsts = [k for k in order[:3]]
        self.assertEqual(sorted(firsts), sorted(lc * E + int(np.argmax(rows[lc])) for lc in range(3)))
        # a profile of another model is refused
        other = dict(fp, hidden_size = 256)
        moe_profile.write_safetensors(path, {"counts": rows}, {"layer_keys": KEYS, "fingerprint": other})
        ts, td = tier_set(f'*=cuda:0 experts=cache profile="{path}"')
        with self.assertRaisesRegex(ValueError, "was built for another model"):
            ts.seed(td, E, HALFLIFE)

    def test_heat_file(self):
        heat = os.path.join(self.d, "heat.exl3moe")
        rows = np.stack([counts(10 + i) for i in range(3)]) / 7.0
        fp = moe_profile.model_fingerprint(NS(config_dict = CD), E)
        moe_profile.write_safetensors(heat, {"counts": rows}, {"layer_keys": KEYS, "fingerprint": fp})
        ts, td = tier_set("*=cuda:0 experts=cache hot=1", heat, hot = 1)
        s = ts.seed(td, E, HALFLIFE)
        self.assertEqual(s.source, f"the heat file {heat}")
        for lc, l in enumerate(td.layers):
            self.assertEqual(s.pins(l, E), [int(np.argmax(rows[lc]))])
            for e in range(E):
                self.assertEqual(s.heat[lc * E + e], int(round(rows[lc][e] * 65536)))
        # a profile wins over the heat file
        path = os.path.join(self.d, "p.json")
        with open(path, "w") as f:
            f.write("{" + ", ".join(f'"{k}": {counts(20).tolist()}' for k in KEYS) + "}")
        ts, td = tier_set(f'*=cuda:0 experts=cache profile="{path}"', heat)
        self.assertTrue(ts.seed(td, E, HALFLIFE).source.startswith("profile "))
        # another model's heat file, an unreadable one, one naming other layers: no seed
        other = dict(fp, layers = 61)
        moe_profile.write_safetensors(heat, {"counts": rows}, {"layer_keys": KEYS, "fingerprint": other})
        ts, td = tier_set("*=cuda:0 experts=cache", heat)
        self.assertEqual(ts.seed(td, E, HALFLIFE).shares, {})
        with open(heat, "wb") as f:
            f.write(b"not a safetensors file")
        self.assertEqual(ts.seed(td, E, HALFLIFE).shares, {})
        moe_profile.write_safetensors(heat, {"counts": rows}, {"layer_keys": ["o.0", "o.1", "o.2"], "fingerprint": fp})
        self.assertEqual(ts.seed(td, E, HALFLIFE).shares, {})


if __name__ == "__main__":
    unittest.main(verbosity = 2)
