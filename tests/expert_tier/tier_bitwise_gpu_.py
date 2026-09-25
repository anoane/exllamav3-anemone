"""
experts=cache computes exactly what the same layer computes resident: real DeepSeek-V4.1 3.0 bpw MoE
layers (DSV41_MODEL_DIR; layer 1 hash-routed, 12, 25, 39), loaded once resident and once through the
VRAM expert cache, give bit-identical outputs (torch.equal on the fp32 outputs viewed as int32) for
the same inputs:
  - rows 1, 2, 5, 8 (decode: the fused decode kernels, admission), 9, 33, 200 (routed: transients in
    staging), 1024 and a skewed 2048 (one row repeated, one token for the hash layer: six experts with
    2,048 rows each, above the batched reconstruct tier's row cap, take the per-expert dequant path with
    the cache's trellis views);
  - inputs randn scaled to RMS 1, and one row repeated with 1e-3 noise (heavy experts);
  - pools of 16 slots (almost every call misses; spare 8) and 512, over a RAM tier holding every other
    expert (disk experts=off) or a small one over the SSD, under every policy x demote, admit=always /
    heat / adaptive, evict=lru / lfu, deterministic and production (tier thread) modes;
  - the arithmetic mode of the process (run once more with EXL3_STABLE_ARITHMETIC=1).
Then a churn run: thousands of decode calls with random batch sizes over the four layers on a 16-slot
pool in production mode, a 200-row routed call every 500 calls, every output compared with the resident
reference, the directory verified at the end. (run_single_expert_dq_views itself:
tests/expert_tier/dq_views_gpu_.py.)

    DSV41_MODEL_DIR=... python -m pytest -q -s tests/expert_tier/tier_bitwise_gpu_.py   (a GPU window)
"""
import gc
import os
import random
import sys
import time
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = os.environ.get("DSV41_MODEL_DIR")
DEV = torch.device("cuda", int(os.environ.get("EXL3_TIER_TEST_DEVICE", "0")))
LAYERS = [int(x) for x in os.environ.get("EXL3_TIER_TEST_LAYERS", "1,12,25,39").split(",")]
ROWS = [1, 2, 5, 8, 9, 33, 200, 1024]


def need():
    if not MODEL:
        raise AssertionError("tier_bitwise_gpu_: DSV41_MODEL_DIR is not set")
    if not torch.cuda.is_available():
        raise AssertionError("tier_bitwise_gpu_: needs a CUDA GPU")


class Rig:
    """A V4.1 model skeleton; its MoE modules of LAYERS, loaded resident or through the cache"""

    def __init__(self, placement_text):
        need()
        from exllamav3 import Config, Model
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
        from exllamav3.model.placement import parse
        self.config = Config.from_directory(MODEL)
        self.model = Model.from_config(self.config)
        mods = {}
        stack = list(self.model.modules)
        while stack:
            m = stack.pop()
            if isinstance(m, BlockSparseMLP) and getattr(m, "layer_idx", None) in LAYERS:
                mods[m.layer_idx] = m
            stack.extend(getattr(m, "modules", []) or [])
        self.mods = [mods[i] for i in LAYERS]
        ip = self.config.infer_params
        self.placement = parse(placement_text) if placement_text else None
        if self.placement is not None:
            self.placement.layer_map(m.layer_idx for m in self.model.modules
                                     if m.layer_idx is not None and m.layer_idx >= 0)
        ip.placement = self.placement
        ip.moe_cpu_component = "text"
        self.E = self.mods[0].num_experts
        self.H = self.mods[0].hidden_size

    def inputs(self, seed = 1234):
        """Per layer: [(rows, x, input_ids)], randn at RMS 1 and a skewed batch"""
        g = torch.Generator(device = "cpu").manual_seed(seed)
        cases = []
        for rows in ROWS:
            x = torch.randn((rows, self.H), generator = g)
            x = x / x.pow(2).mean(-1, keepdim = True).sqrt()
            cases.append((rows, x.half(), torch.randint(0, 120000, (1, rows), generator = g)))
        base = torch.randn((1, self.H), generator = g)
        x = base.repeat(2048, 1) + 1e-3 * torch.randn((2048, self.H), generator = g)
        x = x / x.pow(2).mean(-1, keepdim = True).sqrt()
        tok = torch.randint(0, 120000, (1,), generator = g).repeat(2048).view(1, 2048)
        cases.append((2048, x.half(), tok))
        return cases

    def run(self, m, rows, x, ids):
        params = {"input_ids": ids}
        y = m.forward(x.to(DEV).view(1, rows, self.H), params)
        return y.float().clone()


def resident_reference(rig, cases):
    """Every layer loaded resident (no placement), one after the other"""
    ip = rig.config.infer_params
    ip.placement = None
    out = {}
    try:
        for m in rig.mods:
            m.load(DEV)
            out[m.layer_idx] = [rig.run(m, r, x, i) for r, x, i in cases]
            m.unload()
    finally:
        ip.placement = rig.placement
    torch.cuda.synchronize()
    return out


def tiered(rig, text, sizes, policy = None, cfg_kw = None):
    from exllamav3.model.expert_tier import ExpertTierSet
    from exllamav3.model.expert_tier_config import TierConfig
    cfg = TierConfig(**(cfg_kw or {}))
    tiers = ExpertTierSet(rig.config, rig.placement, None, tier_config = cfg)
    rig.config.expert_tiers = tiers
    for m in rig.mods:
        m.load(DEV)
    tiers.build_fixed({f"cuda:{DEV.index}": sizes}, policy = policy)
    return tiers


def drop(rig):
    for m in rig.mods:
        m.unload()
    t = rig.config.expert_tiers
    if t is not None:
        t.unload()
    rig.config.expert_tiers = None
    gc.collect()
    torch.cuda.empty_cache()


class Bitwise(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        need()
        cls.rig = Rig("*=cuda:0 experts=cache; ram experts=all; disk experts=off")
        cls.cases = cls.rig.inputs()
        t0 = time.monotonic()
        cls.ref = resident_reference(cls.rig, cls.cases)
        print(f"\n  resident reference: {len(cls.rig.mods)} layers x {len(cls.cases)} inputs in "
              f"{time.monotonic() - t0:.1f} s", end = "")

    def compare(self, what):
        rig = self.rig
        for m in rig.mods:
            for (rows, x, ids), want in zip(self.cases, self.ref[m.layer_idx]):
                got = rig.run(m, rows, x, ids)
                self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)),
                                f"{what}: layer {m.layer_idx} rows {rows}: max diff "
                                f"{(got - want).abs().max().item():.3e}")

    def test_configurations(self):
        rig = self.rig
        n = len(LAYERS) * rig.E
        runs = [
            ("pool 16, lazy-exclusive/swap, adaptive, production", (16, n - 8), None, {}),
            ("pool 16, deterministic", (16, n - 8), None, {"deterministic": True}),
            ("pool 512, lazy-exclusive/swap", (512, n - 504), None, {}),
            ("pool 16, exclusive/all, admit=always, evict=lfu", (16, n - 8),
             {"policy": "exclusive", "demote": "all", "admit": "always", "evict": "lfu"}, {}),
            ("pool 16, exclusive/heat, admit=heat", (16, n - 8),
             {"policy": "exclusive", "demote": "heat", "admit": "heat"}, {}),
            ("pool 16, inclusive", (16, n), {"policy": "inclusive", "demote": "off"}, {}),
            ("pool 24 spare 0 in place, demote off (disk)", (24, 100),
             {"spare": 0, "demote": "off", "disk_off": False}, {}),
            ("pool 16 over the SSD, RAM 64 slots", (16, 64), {"disk_off": False}, {}),
        ]
        only = os.environ.get("EXL3_TIER_TEST_RUNS")
        for i, (what, sizes, policy, cfg_kw) in enumerate(runs):
            if only and str(i) not in only.split(","):
                continue
            t0 = time.monotonic()
            tiers = tiered(rig, None, sizes, policy, cfg_kw)
            try:
                self.compare(what)
                td = next(iter(tiers.devices.values()))
                torch.cuda.synchronize()
                td.handle.drain()
                td.handle.verify()
                s = td.stats()
                print(f"\n  {what}: identical; {s['policy_hits']} hits {s['policy_admits']} admits "
                      f"{s['policy_transients']} transients {s['policy_d2h']} D2H {s['policy_ssd']} SSD "
                      f"({time.monotonic() - t0:.1f} s)", end = "")
            finally:
                drop(rig)

    def test_churn(self):
        rig = self.rig
        n = len(LAYERS) * rig.E
        tiered(rig, None, (16, n - 8), None, {})
        try:
            rnd = random.Random(99)
            dec = [i for i, c in enumerate(self.cases) if c[0] <= 8]
            pre = [i for i, c in enumerate(self.cases) if c[0] == 200]
            calls = int(os.environ.get("EXL3_TIER_CHURN_CALLS", 3000))
            for k in range(calls):
                ci = rnd.choice(pre) if k % 500 == 499 else rnd.choice(dec)
                li = rnd.randrange(len(rig.mods))
                m = rig.mods[li]
                rows, x, ids = self.cases[ci]
                got = rig.run(m, rows, x, ids)
                want = self.ref[m.layer_idx][ci]
                self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)),
                                f"churn call {k}: layer {m.layer_idx} rows {rows}")
            td = next(iter(rig.config.expert_tiers.devices.values()))
            torch.cuda.synchronize()
            td.handle.drain()
            td.handle.verify()
            s = td.stats()
            self.assertEqual(s["overflow"], 0)
            self.assertLessEqual(s["policy_d2h"], s["policy_retires"])
            print(f"\n  churn: {calls} calls identical; {s['records']} records, {s['host_records']} with host work, "
                  f"{s['policy_admits']} admits, {s['policy_d2h']} D2H, max lag {s['max_lag']}", end = "")
        finally:
            drop(rig)


if __name__ == "__main__":
    unittest.main()
