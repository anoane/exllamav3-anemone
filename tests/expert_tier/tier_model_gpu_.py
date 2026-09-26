"""
The whole model with its experts in different places computes the same logits, bit for bit, under
the stable arithmetic profile (EXL3_STABLE_ARITHMETIC=1: results that do not depend on how many rows
a kernel sees, so a layer's experts may be resident, streamed from RAM or served by the expert
cache): DeepSeek-V4.1-Flash (DSV41_MODEL_DIR), CMP + PRO as served (CUDA_VISIBLE_DEVICES=1,0), four
placements that differ only in where the experts of layers 12-26 live:

    vram      0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1                       (the anchor layout)
    cache4    0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-26=cuda:1 experts=cache; 27-39=cuda:1; ram experts=all; disk experts=off
    stream15  0-11=cuda:0; 12-26=cuda:1 experts=stream; 27-39=cuda:1
    cache15   0-11=cuda:0; 12-26=cuda:1 experts=cache; 27-39=cuda:1; ram experts=all; disk experts=off
    cache15s  cache15 with the RAM tier small (ram experts=16GiB): the cache over the SSD

vram vs cache4 compares resident and cached experts (layers 23-26), stream15 vs cache15 / cache15s
streamed and cached ones (12-26), vram vs stream15 resident and streamed ones (23-26). Each placement
loads in its own process, scores TOKENS canonical tokens teacher-forced in 4,096-row chunks (every
prefill call of a cache layer runs in layer mode), then GEN greedy decode steps (decode calls), and
saves the logits' log-probabilities of the actual tokens, the top-1 ids and the decoded ids. Every
array must be equal across the placements.

    DSV41_MODEL_DIR=... python tests/expert_tier/tier_model_gpu_.py       (a GPU window; ~10 minutes)
    python tests/expert_tier/tier_model_gpu_.py --one <name> <out.npz>   (one placement, the child)
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("DSV41_MODEL_DIR")
IDS = os.environ.get("EXL3_TIER_TEST_IDS")        # .npy of token ids; default: this repository's docs, tokenized
TOKENS = int(os.environ.get("EXL3_TIER_MODEL_TOKENS", "8192"))
GEN = int(os.environ.get("EXL3_TIER_MODEL_GEN", "32"))
STABLE = {"EXL3_STABLE_ARITHMETIC": "1", "EXL3_HGEMM_FIXED_ROWS": "128", "EXL3_MOE_FUSED_PREFILL": "1",
          "EXL3_DSV41_NUMERICS": "deepseek:index", "EXL3_MOE_PINNED_ARENA": "1", "EXL3_MOE_CPU_WSLOTS": "2",
          "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1,0"}
BASE = "0-11=cuda:0; "
PLACEMENTS = {
    "vram": BASE + "12-22=cuda:1 experts=stream; 23-39=cuda:1",
    "cache4": BASE + "12-22=cuda:1 experts=stream; 23-26=cuda:1 experts=cache; 27-39=cuda:1; ram experts=all; "
                     "disk experts=off",
    "stream15": BASE + "12-26=cuda:1 experts=stream; 27-39=cuda:1",
    "cache15": BASE + "12-26=cuda:1 experts=cache; 27-39=cuda:1; ram experts=all; disk experts=off",
    "cache15s": BASE + "12-26=cuda:1 experts=cache; 27-39=cuda:1; ram experts=16GiB",
}
PAIRS = [("vram", "cache4"), ("stream15", "cache15"), ("stream15", "cache15s"), ("vram", "stream15")]


def one(name: str, out: str):
    """Child: load one placement, score, decode, save"""
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    from exllamav3 import Config, Model, Cache
    config = Config.from_directory(MODEL)
    if IDS:
        ids = np.load(IDS)[:TOKENS + 1].astype(np.int64)
    else:
        from exllamav3 import Tokenizer
        text = "\n\n".join(p.read_text() for p in sorted((ROOT / "doc").glob("*.md")))
        ids = Tokenizer.from_config(config).encode(text)[0].numpy().astype(np.int64)
        while len(ids) < TOKENS + 1:
            ids = np.concatenate([ids, ids])
        ids = ids[:TOKENS + 1]
    config.infer_params.placement = PLACEMENTS[name]
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = -(-(TOKENS + GEN + 4096) // 4096) * 4096, max_batch_size = 1,
                  max_history = 0)
    model.load(max_chunk_size = 4096, max_output_size = 4096, max_batch_size = 1, progressbar = False,
               reserve_per_device = [96 / 1024])
    x = torch.as_tensor(ids[:TOKENS], dtype = torch.long)[None]
    actual, top1 = [], []
    with torch.inference_mode():
        state = cache.get_new_state()
        for s0 in range(0, TOKENS, 4096):
            y = model.forward(x[:, s0:s0 + 4096], {"attn_mode": "flash_attn", "recurrent_states": [state]})[0]
            for r0 in range(0, y.shape[0], 512):
                lp = torch.log_softmax(y[r0:r0 + 512].float(), dim = -1)
                nxt = torch.as_tensor(ids[s0 + r0 + 1:s0 + r0 + 1 + lp.shape[0]], device = lp.device)
                actual.append(lp.gather(1, nxt[:, None])[:, 0].cpu())
                top1.append(lp.argmax(-1).cpu())
            last = y[-1].clone()
            del y
        tok = int(last.argmax())
        gen = []
        for _ in range(GEN):
            gen.append(tok)
            y = model.forward(torch.tensor([[tok]], dtype = torch.long), {"attn_mode": "flash_attn",
                                                                          "recurrent_states": [state]})[0]
            tok = int(y[-1].argmax())
            actual.append(torch.log_softmax(y.float(), dim = -1)[-1:, tok].cpu())
        cache.release_state(state)
    stats = {}
    tiers = getattr(config, "expert_tiers", None)
    if tiers is not None:
        for s in tiers.stats().values():
            for k in ("policy_hits", "policy_admits", "policy_transients", "layer_plans", "ssd_reads"):
                stats[k] = stats.get(k, 0) + int(s.get(k, 0))
    np.savez(out, actual = torch.cat(actual).numpy(), top1 = torch.cat(top1).numpy(), gen = np.array(gen),
             stats = np.array([stats.get(k, 0) for k in ("policy_hits", "policy_admits", "policy_transients",
                                                         "layer_plans", "ssd_reads")]))
    model.unload()


class WholeModel(unittest.TestCase):

    def test_same_logits(self):
        if not MODEL:
            raise AssertionError("tier_model_gpu_: DSV41_MODEL_DIR is not set")
        if IDS and not os.path.isfile(IDS):
            raise AssertionError(f"tier_model_gpu_: no token ids at {IDS} (EXL3_TIER_TEST_IDS)")
        import numpy as np
        only = os.environ.get("EXL3_TIER_MODEL_ONLY")
        names = [n for n in PLACEMENTS if not only or n in only.split(",")]
        res = {}
        with tempfile.TemporaryDirectory(prefix = "tier_model_") as d:
            for name in names:
                out = os.path.join(d, f"{name}.npz")
                env = dict(os.environ, **STABLE)
                t0 = time.monotonic()
                r = subprocess.run([sys.executable, __file__, "--one", name, out], env = env, capture_output = True,
                                   text = True)
                self.assertEqual(r.returncode, 0, f"{name}: the child failed\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
                res[name] = dict(np.load(out))
                s = res[name]["stats"]
                print(f"\n  {name}: {time.monotonic() - t0:.0f} s; tier hits {s[0]} admits {s[1]} transients {s[2]} "
                      f"layer plans {s[3]} SSD reads {s[4]}", end = "")
        for a, b in PAIRS:
            if a not in res or b not in res:
                continue
            for k in ("actual", "top1", "gen"):
                x, y = res[a][k], res[b][k]
                same = x.shape == y.shape and np.array_equal(x.view(np.int32) if x.dtype == np.float32 else x,
                                                             y.view(np.int32) if y.dtype == np.float32 else y)
                diff = "" if same else f" (first difference at {int(np.argmax(x != y)) if x.shape == y.shape else 'shape'})"
                self.assertTrue(same, f"{a} vs {b}: {k} differ{diff}")
            print(f"\n  {a} vs {b}: identical ({res[a]['actual'].shape[0]} positions, {GEN} generated)", end = "")
        self.assertGreater(res.get("cache15", {"stats": [0, 0, 0, 1]})["stats"][3], 0)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
