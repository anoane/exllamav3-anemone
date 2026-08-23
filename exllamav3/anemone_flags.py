"""
ANEMONE feature flags — CLI surface.

Every ANEMONE extension is opt-in and defaults OFF, so an unflagged run behaves exactly
like upstream. Flags are translated into the environment variables the implementation
reads, and, for the few gates that are captured in module-level constants at import time,
the corresponding module globals are refreshed as well (setting the env alone would be too
late once the module is imported).

Quality note, measured on 364,544 tokens of held-out text (DeepSeek-V4-Flash, exl3
2.74 bpw per-expert hybrid), with each domain's calibrated extent computed from the
manifest and excluded: --fp4-prefill costs +1.7%, --fp4-dense a further +5.2% on top of it,
--hc-bf16 +0.004%.

--no-rlp is the one flag these numbers cannot price: it reschedules chunking inside the
generator path, which the batched perplexity harness never enters. Through the harness where
it does execute it is bounded below that instrument's 0.035% resolution -- so "no measurable
cost", which is weaker than the "token-exact" this fork used to claim. That claim rested on a
greedy-token digest gate later found blind (identical digests for two packs with different
weights; its positive control never moved it) and is withdrawn.
"""

import os

# cli dest -> (env var, kind); kind: "bool" sets "1", "value" sets str(value)
SERVING_FLAGS = {
    "fp4_prefill":            ("EXL3_FP4_PREFILL", "bool"),
    "fp4_dense":              ("EXL3_FP4_DENSE", "bool"),
    "fp4_hook":               ("EXL3_FP4_HOOK", "value"),
    "hc_bf16":                ("EXL3_HC_BF16", "bool"),
    "no_rlp":                 ("EXL3_NO_RLP", "bool"),
    "prefill_graphs":         ("EXL3_PREFILL_GRAPHS", "bool"),
    "prefill_graphs_max_tokens": ("EXL3_P24_MAX_TOKENS", "value"),
    "prefill_stage":          ("EXL3_PREFILL_STAGE", "bool"),
    "gu_merge":               ("EXL3_GU_MERGE", "bool"),
}

CONVERT_FLAGS = {
    "strategy_json":  ("EXL3_STRATEGY_JSON", "value"),
    "score_json":     ("EXL3_SCORE_JSON", "value"),
    "qgroup":         ("EXL3_QGROUP", "value"),
    "snap_e2m1":      ("EXL3_SNAP_E2M1", "bool"),
    "calib_dir":      ("ANEMONE_CALIB_DIR", "value"),
    "calib_all":      ("ANEMONE_CALIB_ALL", "bool"),
    "cal_bf16":       ("EXL3_CAL_BF16", "bool"),
    "cal_tier":       ("EXL3_CAL_TIER", "bool"),
    "cal_vram_gb":    ("EXL3_CAL_VRAM_GB", "value"),
    "cal_ram_gb":     ("EXL3_CAL_RAM_GB", "value"),
    "cal_spill_dir":  ("EXL3_CAL_SPILL_DIR", "value"),
}

# Module-level constants captured at import time: (module, attribute, env var).
# Refreshed after the environment is updated so a CLI flag works regardless of import order.
_IMPORT_TIME_GATES = [
    ("exllamav3.modules.block_sparse_mlp", "BSM_FP4_PREFILL", "EXL3_FP4_PREFILL"),
    ("exllamav3.modules.dsv4", "prefill_graph_mode", "EXL3_PREFILL_GRAPHS"),
    ("exllamav3.modules.dsv4", "idx_static_mode", "EXL3_IDX_STATIC"),
]


def add_serving_args(parser):
    """Add the ANEMONE serving flags to an ArgumentParser. All default OFF."""
    g = parser.add_argument_group("ANEMONE (opt-in extensions; unflagged == upstream behavior)")
    g.add_argument("--fp4-prefill", action="store_true",
                   help="Run routed-expert prefill GEMMs through the block-scaled FP4 (e2m1) "
                        "path on Blackwell. Much faster prefill; costs ~+1.7%% perplexity "
                        "unless the pack was quantized snap-aware")
    g.add_argument("--fp4-dense", action="store_true",
                   help="Also route dense/attention prefill GEMMs through FP4. NOT recommended: "
                        "measured ~+5.2%% perplexity on top of --fp4-prefill, for ~13 ms")
    g.add_argument("--fp4-hook", type=str, default=None, metavar="PATH",
                   help="Path to the FP4 hook module (default: bundled exllamav3/anemone_fp4/fp4_hook.py)")
    g.add_argument("--hc-bf16", action="store_true",
                   help="Store hyperconnection residual streams in bf16 instead of fp32. Halves "
                        "stream traffic (large win at long context); ~+0.004%% perplexity")
    g.add_argument("--no-rlp", action="store_true",
                   help="Skip the recurrent-last-page split during prefill (one fewer forward "
                        "pass per prompt). No measurable cost; recommended for prompts under ~16k")
    g.add_argument("--prefill-graphs", action="store_true",
                   help="Capture and replay whole prefill chunks as CUDA graphs (<=128k context)")
    g.add_argument("--prefill-graphs-max-tokens", type=int, default=None, metavar="N",
                   help="Cache-size ceiling above which prefill graphs decline (default 131072)")
    g.add_argument("--prefill-stage", action="store_true",
                   help="Stage the next prefill chunk's embedding on a side stream")
    g.add_argument("--gu-merge", action="store_true",
                   help="Merge the gate and up expert GEMMs into one launch (uniform-K layers)")
    return parser


def add_convert_args(parser):
    """Add the ANEMONE conversion flags to an ArgumentParser."""
    g = parser.add_argument_group("ANEMONE (per-expert quantization)")
    g.add_argument("--strategy-json", type=str, default=None, metavar="PATH",
                   help="JSON map {tensor_key: K} forcing per-tensor bitrates; overrides the "
                        "allocator for the listed tensors (this is how per-expert hybrid packs "
                        "are built)")
    g.add_argument("--score-json", type=str, default=None, metavar="PATH",
                   help="JSON map of per-tensor scores used as an allocator tiebreak")
    g.add_argument("--qgroup", type=str, default=None, choices=["expert", "expert_split"],
                   help="Quantize routed experts as independent groups, so each expert may take "
                        "its own K")
    g.add_argument("--snap-e2m1", action="store_true",
                   help="Snap-aware trellis search: choose codes whose decoded values land on "
                        "the e2m1 grid, making a later FP4 cast lossless. Applies to K<=4 only")
    g.add_argument("--calib-dir", type=str, default=None, metavar="PATH",
                   help="Calibration corpus directory containing manifest.json; rows are "
                        "apportioned across its domains by weight_pct")
    g.add_argument("--calib-all", action="store_true",
                   help="Consume the WHOLE calibration corpus: every full window of every "
                        "domain file, ignoring weight caps and --cal_rows. Effective "
                        "weighting becomes proportional to each file's token count")
    g.add_argument("--cal-bf16", action="store_true",
                   help="Keep calibration state in bf16 (halves calibration RAM)")
    g.add_argument("--cal-tier", action="store_true",
                   help="Tiered calibration state (VRAM -> RAM -> disk spill), for models whose "
                        "calibration state exceeds host RAM")
    g.add_argument("--cal-vram-gb", type=float, default=None, metavar="GB",
                   help="VRAM budget for tiered calibration state")
    g.add_argument("--cal-ram-gb", type=float, default=None, metavar="GB",
                   help="Host RAM budget for tiered calibration state")
    g.add_argument("--cal-spill-dir", type=str, default=None, metavar="PATH",
                   help="Directory for tiered calibration disk spill")
    return parser


def _apply(args, table, quiet):
    applied = []
    for dest, (env, kind) in table.items():
        if not hasattr(args, dest):
            continue
        val = getattr(args, dest)
        if val is None or val is False:
            continue
        os.environ[env] = "1" if kind == "bool" else str(val)
        applied.append(f"--{dest.replace('_', '-')}")

    # Refresh module constants captured at import time
    import importlib
    import sys
    for mod_name, attr, env in _IMPORT_TIME_GATES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, attr):
            cur = os.environ.get(env) == "1"
            if getattr(mod, attr) != cur:
                setattr(mod, attr, cur)

    if applied and not quiet:
        print(f" -- ANEMONE: {' '.join(applied)}")
    return applied


def apply_serving_args(args, quiet=False):
    """Translate parsed serving flags into the environment. Call before the model loads."""
    return _apply(args, SERVING_FLAGS, quiet)


def apply_convert_args(args, quiet=False):
    """Translate parsed conversion flags into the environment. Call before conversion."""
    return _apply(args, CONVERT_FLAGS, quiet)
