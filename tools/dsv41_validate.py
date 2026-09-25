#!/usr/bin/env python3
"""
Validate the DeepSeek-V4.1-Flash exllamav3 port against captured vLLM prompt
logprobs (tools/dsv41_capture_ref.py; --ref, default DSV41_VLLM_REF), and its
cached path against itself.

Every mode except --dry-run loads the full model. Run it alone on the host,
under nothing else's GPU load. The whole body runs under main():
exllamav3's CPU-MoE worker is started with multiprocessing spawn, which
re-imports this file in the child.

Scoring lives in tools/dsv41_refcompare.py (pure numpy, loaded by file path so
--dry-run never imports torch): calibrated limits, each shifted by the reference's
own noise on the case's positions, measured on the capture's other reps and
lengths; a capture without isolation metadata is scored with a loud warning, or
failed with --strict. Logits never leave the device whole: each chunk
is reduced on the head's device, in slices of --logit-rows rows, to our top-5,
our logprob of the actual token and our logprobs at the reference's top-5 ids.

Engine settings go through the engine's own channels: --placement sets
EXL3_PLACEMENT before exllamav3 is imported, the other options set the config (config.dsv41_numerics) or its InferParams after
Config.from_directory, as model_init does. Environment-only settings
(EXL3_STABLE_ARITHMETIC and the rest of doc/env_vars.md) are taken from the
caller's environment and recorded in the report. Options a tree does not
implement (an older checkout) are refused before the model loads. See
doc/dsv41_tools.md.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import tempfile
import time
import uuid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(tempfile.gettempdir(), "dsv41_validate")
MODES = ("nc", "cached", "decode", "self")
PAGE_SIZE = 256         # exllamav3.constants.PAGE_SIZE (not imported: --dry-run stays torch-free)
CACHE_SLOTS = 2         # recurrent-state slots: the teacher-forced runs hold one, --generator's job the other


def _load_by_path(name: str, relative: str):
    """A torch-free source file by path; None when this tree does not have it."""
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(HERE, relative)
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def load_refcompare():
    """tools/dsv41_refcompare.py by path: importing the exllamav3 package would pull in torch."""
    return _load_by_path("dsv41_refcompare", os.path.join("tools", "dsv41_refcompare.py"))


def has_feature(relative: str) -> bool:
    """Whether this tree has an optional engine file (the tools also run on older checkouts)."""
    return os.path.isfile(os.path.join(HERE, relative))


def _num_layers(model) -> int | None:
    """num_hidden_layers of a checkpoint's config.json, or None"""
    try:
        with open(os.path.join(model, "config.json")) as f:
            d = json.load(f)
        return int(d.get("text_config", d)["num_hidden_layers"])
    except (TypeError, OSError, ValueError, KeyError):
        return None


def _epilog(rc) -> str:
    g = rc.GATES
    ab = "\n".join(f"  {a.token:<17} {' '.join(f'{k}={v}' for k, v in a.env.items()):<40} "
                   f"{a.owner:<9} fails at >= {a.affects_from}{' (isolated)' if a.isolated else ''}\n"
                   f"  {'':<17} {a.what}" for a in rc.ABLATIONS.values())
    reg = "\n".join(f"  [{lo},{hi})  {rc.REGION_NOTES.get((lo, hi), '')}" for lo, hi in rc.REGIONS)
    return f"""\
modes:
  nc      stateless flash_attn_nc forward of the whole prompt (n <= --nc-max)
  cached  Cache + chunked model.forward, uniform --chunk; logits reduced on the device in
          --logit-rows slices (512 keeps n=20000 transients small)
  decode  cached, token by token across 0-7, 512->513, 1025->1026, 16383->16384 (odd
          prefill lengths so the first step closes a carried rate-2 group) and the tail
  self    nc vs cached (--chunks) vs decode of the same sequence; exllamav3 only

gates vs vLLM, calibrated limits (set from vLLM's own rep0-vs-rep1 noise on the calibration
capture, at n=24 and 300 only; n=300: |dlp| p50 0.011, p90 0.216, top-1 293/299 with every flip
at margin <= 0.375, top-5 overlap 0.948):
  |dNLL| <= {g['dnll']}   top-1 >= {g['top1']:.0%} where the ref margin > {g['top1_margin']}
  |dlp| p50 <= {g['dlp_p50']} and p90 <= {g['dlp_p90']}   top-5 overlap >= {g['overlap']}
  |dNLL| <= {g['region_dnll']} in every region with >= {g['region_min_positions']} positions;
  distribution/ranking gates apply in each region too; |dlp| p99 <= {g['dlp_p99']}, max <= {g['dlp_max']}
noise-shifted limits: the calibrated limits fail vLLM against itself at other lengths (--dry-run
  prints its 'prefix noise': n=20000 vs 1500 fails region [1,128)). So each limit is also
  shifted by the reference's own noise on the same positions: the worst of vLLM against itself
  over the capture's other reps of the prompt, the prompt inside every longer capture and
  every shorter capture it starts with, plus the headroom the calibrated limit has over the
  n=300 noise it was set on (|dlp| max: position by position). --dry-run prints them per case.
  PASS                         within the calibrated limit
  FAIL within reference noise  beyond the calibrated limit, within the noise-shifted one: vLLM
                               differs from itself this much here; does not fail the run
  FAIL                         beyond both, or no comparison of vLLM against itself covers
                               these positions: fails the run
  verdicts: pass | within-noise (passes) | precision-suspect | fail (exit 1)
  Missing reference/candidate data, incomplete coverage, and a reference recorded as contended
  or with probe errors fail, and such a capture is never used as reference noise. A capture
  without that isolation metadata (the calibration capture among them) is scored with a loud
  WARNING; --strict fails it instead and keeps it out of the reference noise.
  A failure only in regions >= {rc.PRECISION_FROM} is labelled precision-suspect: the index top-k
  starts to select there (vLLM rounds its indexer to MXFP4, as deepseek:index does here, and its
  KV to FP8 at every position); re-run with --dsv41-numerics vllm before calling it a bug.
  vLLM's own logprob at position 1 moves 2.95-3.81 nats between consecutive captured lengths
  (6.05 between n=24 and 1500), enough to fail |dNLL| at n=24 alone: every |dNLL| failure is
  printed with its value without the single worst position.
gates, self mode: every position finite in both runs; top-1 identical except where the two
  tokens that traded places are tied (within --tie-eps, default exact at fp16 logit
  resolution) in one of the runs -- each hard flip is printed with the gap each run puts
  between the two tokens; |dlp| p99 <= {rc.SELF_GATES['dlp_p99']}; and |dlp| p90 <= {rc.SELF_GATES['boundary_dlp_p90']} over the
  first {rc.BOUNDARY_WINDOW} predictions after every chunk start of either run (lookback, ring and
  rate-2 carry bugs live there, and at chunk 2048 they are too few for the global p99)

regions (prompt position of the predicted token):
{reg}

ablations (--ablate TOKEN, or --ablate-sweep for a baseline plus each one in its own process:
the engine reads EXL3_DSV41_ABLATE once per process; each MUST fail the gates, beyond their
noise-shifted limits, in the regions it affects, otherwise the harness is blind to it):
{ab}

engine settings: --placement (EXL3_PLACEMENT rules), --dsv41-numerics (config.dsv41_numerics; default: the config's, EXL3_DSV41_NUMERICS or
  deepseek:index), --mcl/--mcs/--mct/--moe-cpu-mode and --fp32-logits (InferParams, as
  -mcl/-mcs/-mct/-mcm/-fp32_logits); everything else comes from the environment and the main
  settings are recorded in the report (REPORT_ENV), e.g. EXL3_STABLE_ARITHMETIC=1 for the
  reproducible-arithmetic profile. Before loading, the plan refuses what the loader would refuse
  of the placement and CPU-offload settings, options and environment together

examples:
  python3 tools/dsv41_validate.py --dry-run --mode self --cases 300,1500 --ref capture.json
  python3 tools/dsv41_validate.py --mode nc --cases 24,300 --model DIR --ref capture.json \\
      --placement '0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1' --gpu-split 63,91
  python3 tools/dsv41_validate.py --mode nc --cases 24,300 --gpu-split 88,56 --mcs 128
  python3 tools/dsv41_validate.py --mode cached --cases 20000 --dsv41-numerics precise
  EXL3_STABLE_ARITHMETIC=1 python3 tools/dsv41_validate.py --mode self --cases 1500 \\
      --placement '0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1'
  python3 tools/dsv41_validate.py --mode nc --cases 24,300 --ablate engram
  (--model defaults to DSV41_MODEL_DIR, --ref to DSV41_VLLM_REF)
"""


def build_parser(rc) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "dsv41_validate.py",
        description = "DeepSeek-V4.1-Flash exllamav3 port vs the captured vLLM logprobs.",
        epilog = _epilog(rc), formatter_class = argparse.RawDescriptionHelpFormatter,
        # an abbreviation (--ablate-sw) would survive _strip_args, and every sweep child would
        # start a sweep of its own
        allow_abbrev = False)
    ap.add_argument("--mode", choices = MODES, default = "nc")
    ap.add_argument("--cases", default = "24,300", help = "prompt lengths from the reference file")
    ap.add_argument("--ref", default = os.environ.get("DSV41_VLLM_REF"),
                    help = "vLLM capture (tools/dsv41_capture_ref.py; default: DSV41_VLLM_REF)")
    ap.add_argument("--ref-rep", type = int, default = 0,
                    help = "reference rep the gates use; other reps of the same n are reported")
    ap.add_argument("--strict", action = "store_true",
                    help = "fail a gated reference case that carries no isolation metadata (contended, "
                           "probe_errors) instead of scoring it with a warning, refused before loading; "
                           "no such capture serves as reference noise either")
    ap.add_argument("--model", default = os.environ.get("DSV41_MODEL_DIR"),
                    help = "checkpoint directory (default: DSV41_MODEL_DIR)")
    ap.add_argument("--out", default = None, help = f"JSON report (default {DEFAULT_OUT_DIR}/<mode>[-<ablate>].json)")
    ap.add_argument("--dry-run", action = "store_true", help = "parse the file, print the plan; no torch, no CUDA")
    ap.add_argument("--ablate", action = "append", default = [], metavar = "TOKEN",
                    help = f"one of {', '.join(rc.ABLATIONS)} (repeatable)")
    ap.add_argument("--ablate-sweep", action = "store_true",
                    help = "run a baseline and every --sweep-tokens ablation, each in its own process")
    ap.add_argument("--sweep-tokens", default = None,
                    help = "comma-separated ablations for --ablate-sweep (default: every ablation the mode can "
                           "exercise; engram_nocarry needs a cached mode and --engram-lookback state)")
    ap.add_argument("--chunk", type = int, default = None,
                    help = "cached/decode prefill chunk (default 2048; 512 above n=16384, which bounds the "
                           "chunk's logits at 512 x 129280 fp32)")
    ap.add_argument("--chunks", default = "2048,1000,7", help = "self mode: cached chunk sizes to compare")
    ap.add_argument("--nc-max", type = int, default = 1500, help = "longest prompt the nc path may run")
    ap.add_argument("--logit-rows", type = int, default = 512, help = "rows per on-device logit reduction")
    ap.add_argument("--bf16-logits", action = "store_true", help = "round our logits to bf16 as vLLM does")
    ap.add_argument("--greedy", type = int, default = 16, help = "greedy tokens to print after the prompt (0: off)")
    ap.add_argument("--generator", action = "store_true",
                    help = "also generate --greedy tokens through exllamav3's Generator (cached modes)")
    ap.add_argument("--prefill-check", action = "store_true",
                    help = "gate an eligible pipelined prefill against sequential prefill on continuation logits")
    ap.add_argument("--prefill-chunk", type = int, default = 16384,
                    help = "tokens per optimized prefill call (must exercise at least two pipeline subchunks)")
    ap.add_argument("--prefill-tail", type = int, default = 32,
                    help = "teacher-forced continuation rows after the independently prefetched prefix")
    ap.add_argument("--cache-slots", type = int, default = CACHE_SLOTS,
                    help = "recurrent slots; use 1 for a single long-context run")
    ap.add_argument("--tie-eps", type = float, default = rc.SELF_GATES["tie_eps"],
                    help = "self mode: top-2 gap that counts as a tie")
    ap.add_argument("--engram-lookback", choices = ("state", "harness"), default = "state",
                    help = "cached modes: 'state' leaves the 3-id n-gram lookback to the generation "
                           "path (what is being validated); 'harness' also passes the lookback it "
                           "computes from the prompt as params['dsv41_engram_lookback'], which the "
                           "engram checks against its state ring (a cross-check of the carry)")
    ap.add_argument("--visible", default = None, help = "CUDA_VISIBLE_DEVICES, with CUDA_DEVICE_ORDER=PCI_BUS_ID")
    ap.add_argument("--placement", default = None,
                    help = "explicit placement (EXL3_PLACEMENT rules), e.g. '0-11=cuda:0; 12-22=cuda:1 "
                           "experts=stream; 23-39=cuda:1'; checked against DeepSeek-V4.1's rule for its "
                           "changes of device when --model is given")
    ap.add_argument("--gpu-split", default = None, help = "GiB per device for model.load(use_per_device=...)")
    ap.add_argument("--reserve", default = None, help = "GiB per device for model.load(reserve_per_device=...)")
    ap.add_argument("--mcl", type = int, default = None, help = "infer_params.moe_cpu_offload (as -mcl)")
    ap.add_argument("--mcs", type = int, default = None, help = "infer_params.moe_cpu_split (as -mcs)")
    ap.add_argument("--mct", type = int, default = None, help = "infer_params.moe_cpu_threads (as -mct)")
    ap.add_argument("--moe-cpu-mode", choices = ("compute", "stream_only"), default = None,
                    help = "infer_params.moe_cpu_mode (as -mcm): compute (the CPU worker computes RAM-held "
                           "experts) or stream_only (every active RAM-held expert is streamed to its layer's GPU "
                           "and computed there; no CPU expert arithmetic)")
    ap.add_argument("--dsv41-numerics", default = None,
                    help = "config.dsv41_numerics for this run: precise, deepseek or vllm, optionally "
                           "':' and parts from index,window,compressed (default: the config's own, from "
                           "EXL3_DSV41_NUMERICS, else deepseek:index)")
    ap.add_argument("--fp32-logits", action = "store_true",
                    help = "infer_params.fp32_logits (as -fp32_logits): output-head logits in FP32")
    ap.add_argument("--max-chunk-size", type = int, default = None, help = "model.load max_chunk_size")
    ap.add_argument("--cache-tokens", type = int, default = None,
                    help = f"Cache max_num_tokens (default: {CACHE_SLOTS} x (longest case + greedy), page aligned, "
                           f">= 65536; each of the {CACHE_SLOTS} state slots gets 1/{CACHE_SLOTS} of it)")
    ap.add_argument("--min-avail-gib", type = float, default = 8.0,
                    help = "RAM watchdog: kill this run and its children below this MemAvailable (0: off)")
    ap.add_argument("--dump-positions", action = "store_true", help = "per-position arrays in the JSON")
    return ap


def _ints(s: str) -> list[int]:
    return [int(v) for v in str(s).split(",") if v.strip()]


def _floats(s):
    return None if s is None else [float(v) for v in str(s).split(",") if v.strip()]


# ---------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------

def chunk_for(args, n: int) -> int:
    if args.chunk:
        return args.chunk
    return 512 if n > 16384 else 2048


def ablation_errors(args, rc, token: str) -> list[str]:
    """Why this run cannot exercise an ablation (empty when it can)."""
    if token not in rc.ABLATIONS:
        return [f"unknown ablation '{token}' (known: {', '.join(rc.ABLATIONS)})"]
    if token == "engram_nocarry" and args.mode == "nc":
        return ["engram_nocarry needs cached/decode/self mode; stateless forward has no carried history"]
    if token == "engram_nocarry" and args.engram_lookback == "harness":
        # the ablation replaces the carried lookback, and the engram refuses an explicit lookback
        # that differs from the one it carried (dsv41_engram.history)
        return ["engram_nocarry needs --engram-lookback state: with 'harness' the engram refuses "
                "the ablated lookback at the first chunk start"]
    return []


def make_plan(args, rc, cases) -> dict:
    """What a run would do, per case; printed by --dry-run and stored in the report."""
    chunks = _ints(args.chunks)
    runs_by_case = []
    errors = []
    max_output_size = 1
    slots = getattr(args, "cache_slots", CACHE_SLOTS)
    if slots < 1:
        raise ValueError("cache slots must be positive")
    for name in ("chunk", "max_chunk_size", "cache_tokens", "logit_rows", "prefill_chunk", "prefill_tail"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not chunks or min(chunks) < 1 or not cases or any(c.n < 2 for c in cases):
        raise ValueError("need positive chunk sizes and nonempty reference cases with n >= 2")
    if args.greedy < 0 or not math.isfinite(args.tie_eps) or not math.isfinite(args.min_avail_gib) \
            or args.tie_eps < 0 or args.min_avail_gib < 0:
        raise ValueError("greedy count, tie epsilon and RAM floor must be nonnegative")
    if args.generator and (args.mode == "nc" or args.greedy < 1):
        errors.append("--generator requires a cached mode and --greedy >= 1")
    errors += engine_setting_errors(args)
    if getattr(args, "strict", False):
        bare = [c.name for c in cases if rc.isolation_missing(c)]
        if bare:
            errors.append(f"--strict: the gated reference cases {', '.join(bare)} carry no isolation metadata "
                          f"(contended, probe_errors), so nothing shows that vLLM served them alone; recapture "
                          f"with tools/dsv41_capture_ref.py, which records it, or run without --strict")
    if getattr(args, "prefill_check", False):
        if not has_feature("exllamav3/architecture/dsv41/pipeline.py"):
            errors.append("--prefill-check requires the optional pipelined-prefill implementation")
        subchunk = int(os.environ.get("EXL3_DSV41_PIPELINE_CHUNK", "4096"))
        if subchunk < 1 or args.prefill_tail < 2:
            errors.append("pipeline subchunk must be positive and --prefill-tail must be >= 2")
        if os.environ.get("EXL3_DSV41_PIPELINE", "0") == "0":     # the engine's own test (pipeline.py)
            errors.append("--prefill-check requires EXL3_DSV41_PIPELINE=1")
        if args.mode == "nc" or any(c.n < args.prefill_tail + 2 * subchunk for c in cases):
            errors.append("--prefill-check requires cached mode and a prefix long enough to exercise the pipeline")
        if subchunk > 0 and (args.prefill_chunk < 2 * subchunk or args.prefill_chunk % subchunk):
            errors.append("--prefill-chunk must be a multiple of and >= 2 pipeline subchunks")
        if args.max_chunk_size is not None and args.max_chunk_size < subchunk:
            errors.append("--max-chunk-size must cover the pipeline subchunk")
    for c in cases:
        runs = []
        if args.mode == "nc":
            runs.append({"name": "nc", "forwards": 1})
            if c.n > args.nc_max:
                errors.append(f"{c.name}: nc mode is limited to n <= {args.nc_max} (use cached)")
        elif args.mode == "cached":
            ch = chunk_for(args, c.n)
            s = rc.uniform_schedule(c.n, ch)
            runs.append({"name": f"cached{ch}", "forwards": len(s), "schedule": s})
        elif args.mode == "decode":
            s = rc.decode_schedule(c.n, chunk_for(args, c.n))
            runs.append({"name": "decode", "forwards": len(s), "single_steps": s.count(1), "schedule": s})
        else:
            if c.n <= args.nc_max:
                runs.append({"name": "nc", "forwards": 1})
            for ch in chunks:
                s = rc.uniform_schedule(c.n, ch)
                runs.append({"name": f"cached{ch}", "forwards": len(s), "schedule": s})
            s = rc.decode_schedule(c.n, chunks[0] if chunks else chunk_for(args, c.n))
            runs.append({"name": "decode", "forwards": len(s), "single_steps": s.count(1), "schedule": s})
        for index, run in enumerate(runs):
            # Collector.rows limits the reduction workspace, not the full logits
            # tensor returned by model.forward. Stateless greedy generation also
            # re-runs the growing prompt, unlike cached single-token decoding.
            output_rows = max(run["schedule"]) if "schedule" in run else \
                c.n + (max(0, args.greedy - 1) if index == 0 else 0)
            max_output_size = max(max_output_size, output_rows)
        regions = [f"[{lo},{hi})" for lo, hi in rc.REGIONS if lo < c.n]
        runs_by_case.append({"case": c.name, "n": c.n, "rep": c.rep, "runs": runs, "regions": regions,
                             "greedy": args.greedy})
    warnings = [f"{c['case']} {r['name']}: {r['forwards']} forwards (trim with --chunks)"
                for c in runs_by_case for r in c["runs"] if r["forwards"] > 1000]
    tokens = sorted({t for a in args.ablate for t in a.split(",") if t})
    for t in tokens:
        errors += ablation_errors(args, rc, t)
    longest = max([c.n for c in cases] or [0]) + max(0, args.greedy) + 1
    need_pages = -(-longest // PAGE_SIZE)
    cache_tokens = args.cache_tokens or max(65536, slots * need_pages * PAGE_SIZE)
    # The teacher-forced runs pass no block table, so the V4/V4.1 attention falls back to
    # CacheLayer_dsa.slot_bt, which gives each of the CACHE_SLOTS state slots only
    # num_pages // CACHE_SLOTS pages. A sequence longer than that runs off its slot's row.
    per_slot = cache_tokens // PAGE_SIZE // slots * PAGE_SIZE
    if (args.mode != "nc" or args.generator) and cache_tokens % PAGE_SIZE:
        errors.append(f"--cache-tokens {cache_tokens} is not a multiple of the page size {PAGE_SIZE}")
    if args.mode != "nc" and per_slot < longest:
        errors.append(f"--cache-tokens {cache_tokens}: each of the {slots} state slots holds {per_slot} "
                      f"tokens, {longest} needed (use >= {slots * need_pages * PAGE_SIZE})")
    if args.generator and cache_tokens < longest:
        errors.append(f"--cache-tokens {cache_tokens} < {longest} needed by --generator")
    if getattr(args, "prefill_check", False):
        max_output_size = max(max_output_size, args.prefill_tail)
    # The loader measures at most max_chunk_size rows even when a larger output
    # size is requested. Both dimensions must cover the actual forward plan.
    load_chunk = max(chunks + [args.chunk or 2048, max_output_size])
    if getattr(args, "prefill_check", False):
        load_chunk = max(load_chunk, subchunk)
    if args.max_chunk_size is not None:
        load_chunk = args.max_chunk_size
        if load_chunk < max_output_size:
            errors.append(f"--max-chunk-size {load_chunk} does not cover the largest "
                          f"logit-producing forward ({max_output_size} rows)")
    return {"mode": args.mode, "ref": args.ref, "ref_rep": args.ref_rep, "model": args.model,
            "cases": runs_by_case, "ablate": tokens, "cache_tokens": cache_tokens, "cache_slots": slots,
            "needs_cache": args.mode != "nc" or args.generator, "bf16_logits": args.bf16_logits,
            "logit_rows": args.logit_rows, "engram_lookback": args.engram_lookback, "errors": errors,
            "warnings": warnings, "max_output_size": max_output_size, "max_chunk_size": load_chunk}


def isolation_banner(rc, cases, strict: bool) -> list[str]:
    """The loud warning for captured cases without isolation metadata (empty when every case has it)."""
    bare = [c.name for c in cases if rc.isolation_missing(c)]
    if not bare:
        return []
    lines = [f"!!! WARNING: {len(bare)} of the capture's {len(cases)} cases carry no isolation metadata: "
             f"{', '.join(bare)}",
             "!!!   ('contended' and 'probe_errors' were not recorded): nothing shows that vLLM served these",
             "!!!   prompts alone, and a shared batch changes its numerics."]
    if strict:
        lines.append("!!!   --strict: such a case fails as a reference and is never used as reference noise.")
    else:
        lines += ["!!!   They are scored as if vLLM had served them alone (a WARNING line in every gated run)",
                  "!!!   and serve as reference noise. --strict fails them instead; tools/dsv41_capture_ref.py",
                  "!!!   records the metadata."]
    return lines


def print_noise_limits(rc, cases, all_cases, strict: bool) -> dict:
    """
    For --dry-run: the reference noise of each gated case and the limits it shifts (the calibrated
    limits elsewhere). Returns them for the dry-run report.
    """
    names = {"dnll": "|dNLL|", "top1_margin": "top-1@m", "dlp_p50": "p50", "dlp_p90": "p90",
             "dlp_p99": "p99", "overlap": "overlap"}
    out = {}
    print("\nreference noise of each gated case (noise-shifted limits; every limit not listed stays at "
          "its calibrated value, and |dlp| max is judged position by position):")
    for c in cases:
        noise, excluded = rc.reference_noise(all_cases, c, strict = strict)
        report = rc.attach_noise(rc.compare(c, *rc.as_ours(c)), noise, excluded)
        limits = rc.noise_limits(report)
        print(f"  {c.name}: {len(noise)} comparisons of vLLM against itself"
              + (f" ({'; '.join(r['label'] for r in noise)})" if noise else " (the calibrated limits alone)"))
        for x in excluded:
            print(f"    not used: {x}")
        rows = [("all", limits.get("global"))] + [(f"< {rc.PRECISION_FROM}", limits.get("head"))] + \
               [(f"[{x['lo']},{x['hi'] if x['hi'] is not None else 'end'})", x["limits"])
                for x in limits.get("regions", [])]
        for where, lim in rows:
            if lim:
                print(f"    {where:<14} " + "  ".join(f"{names[k]} {'>=' if k in ('top1_margin', 'overlap') else '<='} "
                                                      f"{v:.4f}" for k, (v, _) in lim.items()))
        out[c.name] = {"sources": [r["label"] for r in noise], "excluded": excluded,
                       "limits": {w: {k: {"limit": v, "source": src} for k, (v, src) in (lim or {}).items()}
                                  for w, lim in rows}}
    return out


def print_plan(plan: dict, rc):
    print(f"plan: mode {plan['mode']}  reference {plan['ref']} (rep {plan['ref_rep']})  model {plan['model']}")
    for c in plan["cases"]:
        rs = "; ".join(f"{r['name']}: {r['forwards']} forwards"
                       + (f" ({r['single_steps']} single-token)" if "single_steps" in r else "")
                       for r in c["runs"])
        print(f"  {c['case']:<10} n={c['n']:<6} {rs}")
        print(f"  {'':<10} regions {' '.join(c['regions'])}; greedy {c['greedy']}")
    if plan["ablate"]:
        for t in plan["ablate"]:
            a = rc.ABLATIONS.get(t)
            if a:
                print(f"  ablation {t}: {a.env} -- must FAIL at positions >= {a.affects_from} ({a.what})")
    if plan["needs_cache"]:
        print(f"  cache: max_num_tokens {plan['cache_tokens']}")
    print(f"  load: max_chunk_size {plan['max_chunk_size']}, max_output_size {plan['max_output_size']}")
    for w in plan["warnings"]:
        print(f"  note: {w}")
    for e in plan["errors"]:
        print(f"  ERROR {e}")


# ---------------------------------------------------------------------------------
# Environment, watchdog
# ---------------------------------------------------------------------------------

# split=<k>, the V4.1-only split variable's grammar, which the engine no longer has
_SPLIT_SPEC = re.compile(r"\s*split\s*=\s*(\S*?)\s*;?\s*", re.IGNORECASE)


def _setting(value, environ, name: str, default):
    """An option's value, else the caller's environment variable the config reads, else default."""
    if value is not None:
        return value
    raw = environ.get(name)
    return default if raw is None else raw


def engine_setting_errors(args, environ = None) -> list[str]:
    """
    Engine settings this tree lacks or refuses, found without importing torch: each option's own
    grammar, and the combinations the loader refuses of the placement and CPU-offload settings,
    taken from the options and the caller's environment together (run_env replaces only the
    variable --placement names).
    """
    environ = os.environ if environ is None else environ
    errors = []
    offload = {}
    for option, name in (("mcl", "EXL3_MOE_CPU_OFFLOAD"), ("mcs", "EXL3_MOE_CPU_SPLIT")):
        value = _setting(getattr(args, option, None), environ, name, 0)
        try:
            offload[option] = int(value)
        except ValueError:
            errors.append(f"{name}={value!r} is not an integer")
            offload[option] = 0
    if offload["mcl"] and offload["mcs"]:
        errors.append("--mcl and --mcs are mutually exclusive, as -mcl and -mcs are "
                      "(EXL3_MOE_CPU_OFFLOAD and EXL3_MOE_CPU_SPLIT count too)")
    # --placement sets EXL3_PLACEMENT (and overrides the caller's); the V4.1-only split variable
    # and its split=<k> grammar are gone from the engine, which would ignore them
    placement = getattr(args, "placement", None)
    from_option = placement is not None
    explicit = placement if from_option else environ.get("EXL3_PLACEMENT")
    m = _SPLIT_SPEC.fullmatch(placement) if from_option else None
    if m:
        k = m.group(1)
        n = _num_layers(getattr(args, "model", None))
        if k.isascii() and k.isdigit() and int(k) > 0:
            k = int(k)
            example = f"'0-{k - 1}=cuda:0; {k}-{n - 1}=cuda:1'" if n and k < n else f"'0-{k - 1}=cuda:0; *=cuda:1'"
        else:
            example = "'0-11=cuda:0; 12-39=cuda:1' (no placement is an empty value)"
        errors.append(f"--placement {placement!r}: split=<k> is not a placement rule; write each device's "
                      f"layers, e.g. --placement {example}")
        explicit = None
    old_split = environ.get("EXL3_DSV41_PLACEMENT", "")
    if old_split.strip():
        errors.append(f"EXL3_DSV41_PLACEMENT={old_split!r} is set, but the engine does not read it: the layer "
                      f"split comes from EXL3_PLACEMENT / --placement (e.g. '0-11=cuda:0; 12-39=cuda:1') or "
                      f"from the split budgets; unset it")
    has_generic = has_feature(os.path.join("exllamav3", "model", "placement.py"))
    if from_option and explicit is not None and explicit.strip() and not has_generic:
        errors.append("--placement needs the explicit placement (exllamav3/model/placement.py), which this "
                      "tree lacks; use --gpu-split")
    if explicit is not None and has_generic:
        gp = _load_by_path("_dsv41_validate_placement", os.path.join("exllamav3", "model", "placement.py"))
        where = "--placement" if from_option else "EXL3_PLACEMENT"
        try:
            parsed = gp.parse(explicit)
        except ValueError as e:
            errors.append(f"{where}: {e}")
            parsed = None
        if parsed is not None:
            import types
            ip = types.SimpleNamespace(
                moe_cpu_offload = offload["mcl"], moe_cpu_split = offload["mcs"],
                moe_cpu_mode = _setting(getattr(args, "moe_cpu_mode", None), environ, "EXL3_MOE_CPU_MODE",
                                        "compute"))
            try:
                bad = gp.conflicts(ip, environ)
            except ValueError as e:
                bad = [f"EXL3_MOE_CPU_SPLIT_LAYERS ({e})"]
            if bad:
                errors.append(f"{where}: the explicit placement replaces, and the loader refuses it "
                              f"together with, {', '.join(bad)}")
            # DeepSeek-V4.1's rule for its changes of device, checked on the model's topology
            vp = _load_by_path("_dsv41_validate_v41_placement", os.path.join("exllamav3", "architecture",
                                                                            "dsv41", "placement.py"))
            model = getattr(args, "model", None)
            if model and vp is not None and hasattr(vp, "DSV41Placement") \
                    and os.path.isfile(os.path.join(model, "config.json")):
                try:
                    with open(os.path.join(model, "config.json")) as f:
                        vp.DSV41Placement.from_generic(parsed, vp.Topology.from_config_dict(json.load(f)))
                except (ValueError, KeyError) as e:
                    errors.append(f"{where}: {e}")
    numerics = getattr(args, "dsv41_numerics", None)
    if numerics is not None:
        nm = _load_by_path("_dsv41_validate_numerics", os.path.join("exllamav3", "architecture", "dsv41",
                                                                   "numerics.py"))
        if nm is None:
            errors.append("--dsv41-numerics needs the V4.1 numerics setting (architecture/dsv41/numerics.py), "
                          "which this tree lacks")
        else:
            try:
                nm.parse(numerics)
            except ValueError as e:
                errors.append(f"--dsv41-numerics: {e}")
    return errors


def run_env(args, rc) -> dict:
    """Environment a GPU run needs; applied before torch/exllamav3 are imported."""
    env = {}
    if args.visible is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = args.visible
    if args.placement is not None:
        env["EXL3_PLACEMENT"] = args.placement
    ablate_tokens = []
    for tok in sorted({t for a in args.ablate for t in a.split(",") if t}):
        for k, v in rc.ABLATIONS[tok].env.items():
            if k == "EXL3_DSV41_ABLATE":
                ablate_tokens.append(v)
            else:
                env[k] = v
    if ablate_tokens:
        env["EXL3_DSV41_ABLATE"] = ",".join(ablate_tokens)
    return env


def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _descendants(pid: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
            ppid = int(st[st.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        for c in kids.get(p, []):
            out.append(c)
            todo.append(c)
    return out


def start_ram_watchdog(min_gib: float):
    """Kill this run (and the CPU-MoE worker) before the kernel's OOM killer picks a process: a
    full-model run keeps the routed experts in pinned host memory, which does not swap."""
    if min_gib <= 0:
        return

    def loop():
        while True:
            a = _mem_available()
            if a is not None and a < min_gib * 2**30:
                print(f"*** RAM WATCHDOG: MemAvailable {a / 2**30:.1f} GiB < {min_gib} GiB, killing this run",
                      flush = True)
                for c in _descendants(os.getpid()):
                    try:
                        os.kill(c, signal.SIGKILL)
                    except OSError:
                        pass
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(2)

    threading.Thread(target = loop, daemon = True, name = "ram-watchdog").start()


# ---------------------------------------------------------------------------------
# On-device reduction and the forward drivers (torch is passed in, never imported
# at module level)
# ---------------------------------------------------------------------------------

class Collector:
    """
    Reduces logits as they come out of the model. add(a, logits) takes the logits of
    input positions a .. a+L-1, which predict prompt positions a+1 .. a+L; the row that
    predicts position n (past the prompt) is kept as the next-token top-5.
    """

    def __init__(self, torch, case, k: int = 5, bf16: bool = False, rows: int = 512):
        import numpy as np
        self.torch, self.case, self.k, self.bf16, self.rows = torch, case, k, bf16, max(1, rows)
        n = case.n
        self.top_ids = np.full((n - 1, k), -1, dtype = np.int64)
        self.top_lp = np.full((n - 1, k), -np.inf, dtype = np.float64)
        self.actual_lp = np.full(n - 1, np.nan, dtype = np.float64)
        self.at_ref_lp = np.full((n - 1, 5), np.nan, dtype = np.float64)
        self.filled = np.zeros(n - 1, dtype = bool)
        self.nonfinite = 0
        self.next_ids = None
        self.next_lp = None
        self._ids = torch.as_tensor(case.prompt_ids, dtype = torch.long)
        self._ref = torch.as_tensor(case.ref_ids[:, :5], dtype = torch.long)

    def _prep(self, x):
        x = x.float()
        if self.bf16:
            x = x.to(self.torch.bfloat16).float()
        return x

    def add(self, a: int, logits):
        torch = self.torch
        n = self.case.n
        if logits.dim() == 3:
            logits = logits[0]
        L = logits.shape[0]
        for s in range(0, L, self.rows):
            x = self._prep(logits[s:s + self.rows])
            self.nonfinite += int((~torch.isfinite(x).all(dim = -1)).sum())
            lp = torch.log_softmax(x, dim = -1)
            tl, ti = torch.topk(lp, self.k, dim = -1)
            p0 = a + s + 1
            p1 = p0 + x.shape[0]
            q1 = min(p1, n)
            m = q1 - p0
            if m > 0:
                dev = lp.device
                act_ids = self._ids[p0:q1].to(dev)
                act = lp[:m].gather(1, act_ids[:, None])[:, 0]
                rid = self._ref[p0 - 1:q1 - 1].to(dev)
                atr = lp[:m].gather(1, rid.clamp(min = 0))
                atr = torch.where(rid >= 0, atr, torch.full_like(atr, float("nan")))
                r0, r1 = p0 - 1, q1 - 1
                self.top_ids[r0:r1] = ti[:m].cpu().numpy()
                self.top_lp[r0:r1] = tl[:m].double().cpu().numpy()
                self.actual_lp[r0:r1] = act.double().cpu().numpy()
                self.at_ref_lp[r0:r1] = atr.double().cpu().numpy()
                self.filled[r0:r1] = True
            if p1 > n:
                j = n - p0
                self.next_ids = ti[j].cpu().tolist()
                self.next_lp = tl[j].double().cpu().tolist()

    def argmax_last(self, logits) -> int:
        if logits.dim() == 3:
            logits = logits[0]
        return int(self._prep(logits[-1:]).argmax(dim = -1)[0])

    def reduced(self):
        return self.top_ids, self.top_lp, self.actual_lp


def _engram_hasher(model):
    for m in getattr(model, "modules", []):
        stack = [m]
        while stack:
            x = stack.pop()
            h = getattr(x, "hasher", None)
            if h is not None:
                return h
            stack.extend(getattr(x, "modules", []) or [])
    return None


def engram_lookback(torch, model, seq_ids: list[int], a: int):
    """params['dsv41_engram_lookback'] for a chunk starting at a: [1, ctx] compressed ids, UNK before 0."""
    h = _engram_hasher(model)
    if h is None:
        return None
    ctx = h.ctx
    lo = max(0, a - ctx)
    comp = h.compress(torch.tensor([seq_ids[lo:a]], dtype = torch.long)) if a > lo else \
        torch.empty((1, 0), dtype = torch.long)
    pad = torch.full((1, ctx - comp.shape[1]), -2, dtype = torch.long)   # engram_torch.UNK
    return torch.cat([pad, comp], dim = 1)


def run_stateless(torch, model, case, col, greedy: int) -> list[int]:
    ids = torch.tensor([case.prompt_ids.tolist()], dtype = torch.long)
    logits = model.forward(ids, {"attn_mode": "flash_attn_nc"})
    col.add(0, logits)
    del logits
    out = []
    if greedy > 0 and col.next_ids:
        out.append(int(col.next_ids[0]))
        seq = case.prompt_ids.tolist()
        while len(out) < greedy:
            logits = model.forward(torch.tensor([seq + out], dtype = torch.long), {"attn_mode": "flash_attn_nc"})
            out.append(col.argmax_last(logits))
            del logits
    return out


def run_cached(torch, model, cache, case, col, schedule, greedy: int, lookback: str) -> list[int]:
    seq = case.prompt_ids.tolist()
    ids = torch.tensor([seq], dtype = torch.long)
    state = cache.get_new_state()
    out = []
    try:
        def params_at(a, full_seq):
            p = {"attn_mode": "flash_attn", "recurrent_states": [state]}
            if lookback == "harness":
                lb = engram_lookback(torch, model, full_seq, a)
                if lb is not None:
                    p["dsv41_engram_lookback"] = lb
            return p

        a = 0
        for size in schedule:
            b = a + size
            logits = model.forward(ids[:, a:b], params_at(a, seq))
            col.add(a, logits)
            del logits
            a = b
        if greedy > 0 and col.next_ids:
            out.append(int(col.next_ids[0]))
            while len(out) < greedy:
                full = seq + out
                logits = model.forward(torch.tensor([[out[-1]]], dtype = torch.long),
                                       params_at(len(full) - 1, full))
                out.append(col.argmax_last(logits))
                del logits
    finally:
        cache.release_state(state)
    return out


def run_generator(torch, model, cache, cfg, case, n_tokens: int, chunk: int) -> list[int]:
    from exllamav3 import Generator, Job, Tokenizer
    from exllamav3.generator.sampler import ArgmaxSampler
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = 1,
                    max_chunk_size = chunk, recurrent_cache_size = 512 * 1024**2)
    job = Job(input_ids = torch.tensor([case.prompt_ids.tolist()], dtype = torch.long),
              max_new_tokens = n_tokens, sampler = ArgmaxSampler(), stop_conditions = [])
    gen.enqueue(job)
    out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            t = r.get("token_ids")
            if t is not None:
                out += [int(v) for v in t.flatten().tolist()]
    return out[:n_tokens]


def run_prefill_check(torch, rc, model, cache, case, args) -> dict:
    """Exercise actual optimized prefill, not forward(), then compare continuation rows.

    The sequential control uses the same internal row count. This checks accumulated
    state over the whole prefix, but only scores the explicitly reported tail; it is
    complementary to (not a replacement for) full teacher-forced reference validation.
    """
    import dataclasses
    from exllamav3.architecture.dsv41 import pipeline
    if not pipeline.PIPELINE:
        raise ValueError("--prefill-check requires EXL3_DSV41_PIPELINE=1")
    sub = pipeline.SUB_CHUNK
    if args.prefill_chunk < 2 * sub or args.prefill_chunk % sub:
        raise ValueError("--prefill-chunk must be a multiple of and >= 2 pipeline subchunks")
    tail = args.prefill_tail
    prefix = case.n - tail
    short = dataclasses.replace(case, n=tail, prompt_ids=case.prompt_ids[-tail:],
        ref_ids=case.ref_ids[-(tail-1):], ref_lp=case.ref_lp[-(tail-1):],
        ref_actual_lp=case.ref_actual_lp[-(tail-1):])
    ids = torch.tensor([case.prompt_ids.tolist()], dtype=torch.long)
    original = pipeline.prefill_pipelined
    calls = []

    def counted(m, tokens, params):
        calls.append(int(tokens.shape[1]))
        return original(m, tokens, params)

    runs = []
    pipeline.prefill_pipelined = counted
    try:
        for optimized in (False, True):
            state = cache.get_new_state()
            pages = cache.max_num_tokens // PAGE_SIZE // cache.num_slots
            bt = torch.arange(state.slot * pages, (state.slot + 1) * pages, dtype=torch.int32)[None, :]
            started = time.monotonic()
            before = len(calls)
            try:
                def params():
                    return {"attn_mode": "flash_attn", "recurrent_states": [state],
                            "cache_seqlens": torch.tensor([state.position], dtype=torch.int32),
                            "block_table": bt}
                a = 0
                while a < prefix:
                    size = min(args.prefill_chunk if optimized else sub, prefix - a)
                    if size < 2 * sub:
                        size = min(sub, size)
                    model.prefill(ids[:, a:a+size], params())
                    a += size
                if state.position != prefix:
                    raise RuntimeError(f"prefill advanced to {state.position}, expected {prefix}")
                col = Collector(torch, short, bf16=args.bf16_logits, rows=args.logit_rows)
                col.add(0, model.forward(ids[:, prefix:], params()))
                _sync(torch)
                if not col.filled.all() or col.nonfinite:
                    raise RuntimeError("prefill continuation is incomplete or nonfinite")
                runs.append((col, time.monotonic() - started, len(calls) - before))
            finally:
                cache.release_state(state)
    finally:
        pipeline.prefill_pipelined = original
    if runs[0][2] or not runs[1][2]:
        raise RuntimeError(f"incorrect path coverage: sequential={runs[0][2]}, optimized={runs[1][2]} calls")
    report = rc.compare_self(runs[0][0].reduced(), runs[1][0].reduced(), n=tail,
                             tie_eps=args.tie_eps, label="sequential vs pipelined prefill continuation")
    ok, messages = rc.self_gates(report)
    return {"ok": ok, "gates": messages, "report": report, "prefix_tokens": prefix,
            "continuation_rows": tail - 1, "pipeline_calls": runs[1][2], "pipeline_call_sizes": calls,
            "seconds": {"sequential": runs[0][1], "pipelined": runs[1][1]}}


def _sync(torch):
    # never the first CUDA call: synchronize() would initialise CUDA (and create a
    # context) on every visible device
    if torch.cuda.is_initialized():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def _summary_line(rc, rep, v) -> str:
    # every field may be None: a path that returns all-NaN logits (fp16 overflow, an eps
    # underflow) compares zero positions, and must be reported, not crash the run
    g = rep["global"]
    d = g["dlp"]
    tm = g["top1_margin"]
    f = rc._f
    return (f"{v.upper():<17} {rep['label']}: dNLL {f(g['dnll'], '+.4f')}  |dlp| p50 {f(d['p50'])} "
            f"p90 {f(d['p90'])} mean {f(d['mean'])}  top1 {g['top1']['agree']}/{g['top1']['total']} "
            f"(@margin: {tm['agree']}/{tm['total']})  ovl {f(g['overlap'], '.3f')}"
            + (f"  non-finite rows {rep['nonfinite']}" if rep.get("nonfinite") else ""))


def run_case(torch, rc, model, cache, cfg, case, others, args, tok, all_cases = None) -> dict:
    """
    Every run of one case, scored against the reference and (self mode) against each other.
    all_cases: the whole capture, whose other captures give the reference noise the gates are
    shifted by (rc.reference_noise); without it only the calibrated limits apply.
    """
    plan = make_plan(args, rc, [case])["cases"][0]
    rec = {"case": case.name, "n": case.n, "runs": {}, "self": []}
    strict = getattr(args, "strict", False)
    noise = {c.rep: rc.reference_noise(all_cases, c, strict = strict) for c in [case] + list(others)} \
        if all_cases is not None else {}
    if all_cases is not None:
        rec["reference_noise"] = {"sources": [r["label"] for r in noise[case.rep][0]],
                                  "excluded": noise[case.rep][1]}
    reduced = {}
    decode = (lambda ids: tok.decode(ids)) if tok is not None else (lambda ids: str(ids))
    for i, run in enumerate(plan["runs"]):
        name = run["name"]
        col = Collector(torch, case, bf16 = args.bf16_logits, rows = args.logit_rows)
        greedy = args.greedy if i == 0 else 0
        t0 = time.time()
        try:
            if name == "nc":
                gen = run_stateless(torch, model, case, col, greedy)
            else:
                gen = run_cached(torch, model, cache, case, col, run["schedule"], greedy, args.engram_lookback)
            _sync(torch)
        except Exception as e:     # one broken path must not cost the other runs a model reload
            import traceback
            traceback.print_exc()
            rec["runs"][name] = {"verdict": "error", "ok": False, "error": f"{type(e).__name__}: {e}",
                                 "forwards": run["forwards"]}
            print(f"\n== {case.name} {name}: ERROR {type(e).__name__}: {e}", flush = True)
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            continue
        secs = time.time() - t0
        missing = int((~col.filled).sum())
        rep = rc.compare(case, col.top_ids, col.top_lp, col.actual_lp, at_ref_lp = col.at_ref_lp,
                         label = f"{case.name} {name}", detail = args.dump_positions)
        rep["nonfinite"] = col.nonfinite
        rep["unfilled"] = missing
        if case.rep in noise:
            rc.attach_noise(rep, *noise[case.rep])
        ok, msgs = rc.gates(rep, strict = strict)
        v = rc.verdict(rep, strict = strict)
        entry = {"seconds": round(secs, 2), "forwards": run["forwards"], "verdict": v, "ok": ok,
                 "gates": msgs, "report": rep, "vs_other_reps": []}
        print(f"\n== {case.name} {name}: {run['forwards']} forwards in {secs:.1f}s", flush = True)
        for line in rc.format_report(rep):
            print(line)
        for m in msgs:
            print(f"    {m}")
        if missing:
            print(f"    WARNING {missing} positions never received logits")
        print(_summary_line(rc, rep, v), flush = True)
        for o in others:
            r2 = rc.compare(o, col.top_ids, col.top_lp, col.actual_lp, label = f"{case.name} {name} vs rep{o.rep}")
            r2["nonfinite"], r2["unfilled"] = rep["nonfinite"], rep["unfilled"]
            if o.rep in noise:
                rc.attach_noise(r2, *noise[o.rep])
            v2 = rc.verdict(r2, strict = strict)
            entry["vs_other_reps"].append({"rep": o.rep, "verdict": v2, "global": r2["global"]})
            print(f"    vs vLLM rep{o.rep}: {v2}  dNLL {rc._f(r2['global']['dnll'], '+.4f')}  "
                  f"|dlp| p90 {rc._f(r2['global']['dlp']['p90'])}")
        if col.next_ids:
            nt = rc.next_token_check(case, col.next_ids, decode if tok is not None else None)
            entry["next_token"] = nt
            print(f"    next token: ours {nt['our_text']!r} ({nt['our_id']})  vLLM {nt['ref']!r}  "
                  f"match {nt['match']}  top-5 overlap {nt['top5_overlap']}  (information only)")
        if gen:
            entry["greedy_ids"] = gen
            entry["greedy_text"] = decode(gen)
            print(f"    exllamav3 greedy: {entry['greedy_text']!r}")
            for c in [case] + list(others):
                print(f"    vLLM rep{c.rep}      : {c.gen_text!r}")
        rec["runs"][name] = entry
        reduced[name] = col
    if args.generator and cache is not None and args.greedy > 0:
        try:
            ids = run_generator(torch, model, cache, cfg, case, args.greedy, chunk_for(args, case.n))
            rec["generator_ids"] = ids
            rec["generator_text"] = decode(ids)
            expected = next((e.get("greedy_ids") for e in rec["runs"].values() if e.get("greedy_ids")), None)
            rec["generator_ok"] = len(ids) == args.greedy and expected is not None and ids == expected
            print(f"    Generator greedy: {rec['generator_text']!r}")
        except Exception as e:    # the report must survive a generator failure
            rec["generator_error"] = f"{type(e).__name__}: {e}"
            rec["generator_ok"] = False
            print(f"    Generator FAILED: {rec['generator_error']}")
    if getattr(args, "prefill_check", False):
        try:
            rec["prefill_check"] = run_prefill_check(torch, rc, model, cache, case, args)
        except Exception as e:
            rec["prefill_check"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"    prefill check: {rec['prefill_check']}", flush=True)
    names = [r["name"] for r in plan["runs"] if r["name"] in reduced]
    starts = {r["name"]: rc.chunk_starts(r.get("schedule") or []) for r in plan["runs"]}
    floor = None
    if args.mode == "self" and len(names) > 1 and "nc" in reduced and case.n > 24:
        # The row-count noise floor of this model on this case: the stateless forward vs a
        # stateless forward two thirds as long, over the positions both cover. No cache, no
        # state -- only a row-count change like the chunked runs make: the exl3 kernels change
        # tier and split with the row count and the MoE routing near-ties amplify it (dropping a
        # single row usually crosses no tier and understates this). The self gates are judged
        # against it.
        import dataclasses
        m = (2 * case.n) // 3
        short = dataclasses.replace(case, n = m, prompt_ids = case.prompt_ids[:m],
                                    ref_ids = case.ref_ids[:m - 1], ref_lp = case.ref_lp[:m - 1],
                                    ref_actual_lp = case.ref_actual_lp[:m - 1])
        cs = Collector(torch, short, bf16 = args.bf16_logits, rows = args.logit_rows)
        try:
            run_stateless(torch, model, short, cs, 0)
            _sync(torch)
            full = tuple(x[: m - 1] for x in reduced["nc"].reduced())
            floor = rc.compare_self(full, cs.reduced(), n = m, tie_eps = args.tie_eps,
                                    label = f"{case.name} floor: nc({case.n}) vs nc({m})")
            fg = floor["global"]
            rec["floor"] = floor
            print(f"\n== self {case.name}: row-count floor nc({case.n}) vs nc({m}) on {m - 1} positions: |dlp| p99 "
                  f"{rc._f(fg['dlp']['p99'])}  hard flips {fg['top1_diff_hard']}/{fg.get('compared', fg['count'])}",
                  flush = True)
        except Exception as e:     # without a floor the gates stay strict
            rec["floor_error"] = f"{type(e).__name__}: {e}"
            print(f"    floor run failed ({type(e).__name__}: {e}); self gates stay strict", flush = True)
    if args.mode == "self" and len(names) > 1:
        base = names[0]
        for other in names[1:]:
            sr = rc.compare_self(reduced[base].reduced(), reduced[other].reduced(), n = case.n,
                                 tie_eps = args.tie_eps, boundaries = starts[base] + starts[other],
                                 label = f"{case.name} {base} vs {other}")
            ok, msgs = rc.self_gates(sr, floor = floor)
            rec["self"].append({"base": base, "other": other, "ok": ok, "gates": msgs, "report": sr})
            print(f"\n== self {case.name}: {base} vs {other}: {'PASS' if ok else 'FAIL'}")
            for m in msgs:
                print(f"    {m}")
            for r in sr["regions"]:
                d = r["dlp"]
                print(f"    [{r['lo']},{r['hi']}) {r['count']} pos  |dlp| p99 {d['p99']}  max {d['max']}  "
                      f"top1 diff {r['top1_diff']} (ties {r['top1_diff_ties']})")
    return rec


def _write(path: str, obj: dict):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok = True)
    fd, tmp = tempfile.mkstemp(prefix=".dsv41-report-", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent = 1, default = lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# Explicitly non-sensitive tuning controls that can change the numbers or the layout (doc/env_vars.md):
# this series' own settings and the main dispatch and precision switches V4.1's default path reads,
# not every documented variable. Do not dump an arbitrary prefixed environment. F16ACC caches its
# GEMM accumulation choice on first use per device; NO_FUSED_RECONSTRUCT is read at Python import and
# disables long-row folded-Hadamard reconstruction; INT8_GEMV (default on) quantizes the activations
# of every mul1-codebook GEMV, i.e. V4.1's decode. Record explicit settings without claiming the
# environment alone proves which shape-dependent kernel was actually dispatched; the report also
# records the values the run resolved (numerics, arithmetic).
REPORT_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "EXLLAMA_NO_P2P_COPY",
              # arithmetic
              "EXL3_STABLE_ARITHMETIC", "EXL3_HGEMM_FIXED_ROWS", "EXL3_MOE_FUSED_PREFILL",
              "EXL3_HGEMM_F16ACC", "EXL3_HGEMM_FP32_REDUCTION", "EXL3_NO_FUSED_RECONSTRUCT",
              "EXL3_MOE_FUSED_DET", "EXL3_MOE_RECON_DET", "EXL3_MOE_RECON_FOLDED",
              # debug builds of the JIT kernels (Triton's debug mode, V4.1's kernels included)
              "EXL3_DSA_DEBUG_BOUNDS",
              # dispatch and activation precision of the exl3 GEMV/GEMM and the fused MoE
              "EXL3_INT8_GEMV", "EXL3_INT8_GEMV_MAX_K", "EXL3_GEMV", "EXL3_MGEMM_K_THRESHOLD",
              "EXL3_MGEMM_N_THRESHOLD", "EXL3_MOE_FUSED_ROWS", "EXL3_MOE_FUSED_ROWS_WIDE",
              "EXL3_MOE_BATCH_RECON",
              # placement and CPU MoE offload
              "EXL3_PLACEMENT", "EXL3_MOE_CPU_OFFLOAD", "EXL3_MOE_CPU_SPLIT", "EXL3_MOE_CPU_SPLIT_LAYERS",
              "EXL3_MOE_CPU_MODE", "EXL3_MOE_CPU_THREADS", "EXL3_MOE_CPU_SWAP", "EXL3_MOE_CPU_SPLIT_STATS",
              "EXL3_MOE_PINNED_ARENA", "EXL3_MOE_CPU_WSLOTS", "EXL3_MOE_CPU_SLOT_ROWS",
              "EXL3_MOE_CPU_WSLOT_MB", "EXL3_MOE_STREAM_T", "EXL3_MOE_STREAM_FUSED_T",
              "EXL3_MOE_STREAM_MIN_ROWS", "EXL3_MOE_STREAM_BATCH_RECON",
              # output
              "EXL3_FP32_LOGITS",
              # DeepSeek-V4.1
              "EXL3_DSV41_NUMERICS", "EXL3_DSV41_ENGRAM_PREFETCH",
              "EXL3_DSV41_PIPELINE", "EXL3_DSV41_PIPELINE_CHUNK", "EXL3_DSV41_XDEV_BF16",
              "EXL3_DSV41_ROUTER_BIAS", "EXL3_DSV41_FUSED_COMPRESS", "EXL3_DSV41_ACTIVATION",
              "EXL3_DSV41_ABLATE")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(args, torch) -> dict:
    def git(*values):
        try:
            return subprocess.check_output(["git", "-C", HERE, *values], stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return b"unavailable"
    paths = {"reference": args.ref, "model_config": os.path.join(args.model, "config.json")}
    router = os.environ.get("EXL3_DSV41_ROUTER_BIAS")
    if router:
        paths["router_bias"] = router
    source_hash = hashlib.sha256()
    for tree in ("exllamav3", "exllamav3_ext", "tools"):
        for path in sorted((Path(HERE) / tree).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".cuh", ".h", ".hpp"):
                source_hash.update(str(path.relative_to(HERE)).encode() + b"\0")
                source_hash.update(bytes.fromhex(_sha256(path)))
    # Header digests + sizes identify the tensor inventory without re-reading a
    # multi-hundred-GiB checkpoint. They are NOT full weight-content checksums.
    inventory = []
    header_limit = _load_by_path("_dsv41_validate_header", os.path.join(
        "exllamav3", "architecture", "dsv41", "checkpoint_header.py")).MAX_HEADER_SIZE
    for path in sorted(Path(args.model).glob("*.safetensors")):
        with path.open("rb") as stream:
            length_raw = stream.read(8)
            length = int.from_bytes(length_raw, "little")
            if len(length_raw) != 8 or length > header_limit or length + 8 > path.stat().st_size:
                raise ValueError(f"invalid safetensors header in {path.name}")
            header = stream.read(length)
        inventory.append({"file": path.name, "bytes": path.stat().st_size,
                          "header_sha256": hashlib.sha256(length_raw + header).hexdigest()})
    native = {"verified": False, "reason": "loaded native extension artifact not found"}
    extension = sys.modules.get("exllamav3_ext")
    extension_path = getattr(extension, "__file__", None)
    if extension_path and os.path.isfile(extension_path):
        native = {"verified": True, "file": os.path.basename(extension_path),
                  "sha256": _sha256(extension_path)}
        build = os.path.join(os.path.dirname(extension_path), "build.ninja")
        native["build_recipe_sha256"] = _sha256(build) if os.path.isfile(build) else None
    return {"source_commit": git("rev-parse", "HEAD").decode(),
            "source_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--")).hexdigest(),
            "source_content_sha256": source_hash.hexdigest(),
            "source_status": git("status", "--porcelain").decode(),
            "sha256": {name: _sha256(path) for name, path in paths.items()},
            "model_inventory": inventory, "model_full_weight_hashes_verified": False,
            "native_extension": native,
            "torch": torch.__version__, "torch_git": torch.version.git_version,
            "cuda_runtime": torch.version.cuda,
            "devices": [{"name": torch.cuda.get_device_name(i),
                         "capability": torch.cuda.get_device_capability(i)}
                        for i in range(torch.cuda.device_count())]}


def apply_settings(cfg, args):
    """
    The options that set the config or its InferParams, as model_init applies its flags after
    Config.from_directory. A setting this tree does not have is refused before the model is built.
    """
    ip = getattr(cfg, "infer_params", None)
    for name, attr in (("mcl", "moe_cpu_offload"), ("mcs", "moe_cpu_split"), ("mct", "moe_cpu_threads"),
                       ("moe_cpu_mode", "moe_cpu_mode"), ("fp32_logits", "fp32_logits")):
        value = getattr(args, name)
        if value is None or value is False:
            continue
        if not hasattr(ip, attr):
            raise ValueError(f"--{name.replace('_', '-')}: this tree's InferParams has no {attr}")
        setattr(ip, attr, value)
    if args.dsv41_numerics is not None:
        if not hasattr(cfg, "dsv41_numerics"):
            raise ValueError("--dsv41-numerics: this config has no dsv41_numerics (not a DeepSeek-V4.1 model?)")
        cfg.dsv41_numerics = args.dsv41_numerics


def resolved_arithmetic() -> dict | None:
    """The arithmetic policy the imported package resolved (model/math_policy.py), when it has one."""
    mp = sys.modules.get("exllamav3.model.math_policy")
    if mp is None:
        return None
    return {"stable_arithmetic": mp.STABLE_ARITHMETIC, "hgemm_fixed_rows": mp.HGEMM_FIXED_ROWS,
            "fused_prefill": mp.FUSED_PREFILL}


def case_ok(rec) -> bool:
    return (bool(rec.get("runs")) and all(e.get("ok", False) for e in rec["runs"].values())
            and all(e.get("ok", False) for e in rec.get("self", []))
            and rec.get("generator_ok", True) and not rec.get("generator_error")
            and rec.get("prefill_check", {"ok": True}).get("ok", False))


def run_gpu(args, rc, cases, all_cases, plan, out) -> int:
    session = uuid.uuid4().hex
    _write(out, {"tool": "dsv41_validate", "session": session, "status": "running", "ok": False,
                 "plan": plan, "args": vars(args), "cases": []})
    env = run_env(args, rc)
    if "torch" in sys.modules and ("CUDA_VISIBLE_DEVICES" in env):
        print("WARNING torch was imported before the device environment was applied")
    os.environ.update(env)
    start_ram_watchdog(args.min_avail_gib)
    import torch
    from exllamav3 import Cache, Config, Model

    exl3_env = {k: os.environ[k] for k in REPORT_ENV if k in os.environ}
    print(f"environment: {exl3_env}", flush = True)
    if os.environ.get("EXL3_DSV41_ABLATE") and not args.ablate:
        print("WARNING an ablation is set in the environment: this is NOT a baseline run", flush = True)

    t0 = time.time()
    cfg = Config.from_directory(args.model)
    apply_settings(cfg, args)
    numerics = getattr(cfg, "dsv41_numerics", None)
    print(f"dsv41 numerics: {numerics}", flush = True)
    model = Model.from_config(cfg)
    # two recurrent slots: the teacher-forced runs hold one, --generator's job the other
    cache = Cache(model, max_num_tokens = plan["cache_tokens"], max_batch_size = plan["cache_slots"]) \
        if plan["needs_cache"] else None
    load_kw = {"progressbar": False, "max_chunk_size": plan["max_chunk_size"],
               "max_output_size": plan["max_output_size"]}
    if args.gpu_split:
        load_kw["use_per_device"] = _floats(args.gpu_split)
    if args.reserve:
        load_kw["reserve_per_device"] = _floats(args.reserve)
    model.load(**load_kw)
    load_s = time.time() - t0
    print(f"LOADED in {load_s:.0f}s", flush = True)
    devs = {}
    for m in model.modules:
        devs.setdefault(str(getattr(m, "device", None)), []).append(m.key)
    for d, ks in devs.items():
        print(f"  {d}: {len(ks)} modules  {ks[0]} .. {ks[-1]}", flush = True)
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} allocated {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB", flush = True)

    tok = None
    try:
        from tokenizers import Tokenizer as _T
        tok = _T.from_file(os.path.join(args.model, "tokenizer.json"))
    except Exception as e:
        print(f"  (no tokenizer for printing: {e})")

    report = {"tool": "dsv41_validate", "session": session, "status": "running", "ok": False,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
              "plan": plan, "env": exl3_env, "numerics": None if numerics is None else str(numerics),
              "arithmetic": resolved_arithmetic(), "gates": rc.GATES, "self_gates": rc.SELF_GATES,
              "calibration_noise": rc.CALIBRATION_NOISE,
              "reference_isolation": {"missing": [c.name for c in all_cases if rc.isolation_missing(c)],
                                      "strict": bool(getattr(args, "strict", False))},
              "load": {"seconds": round(load_s, 1), "devices": {d: [ks[0], ks[-1], len(ks)] for d, ks in devs.items()},
                       "allocated_gib": [round(torch.cuda.memory_allocated(i) / 2**30, 2)
                                         for i in range(torch.cuda.device_count())]},
              "provenance": provenance(args, torch), "cases": []}
    for case in cases:
        others = [c for c in all_cases if c.n == case.n and c.rep != case.rep
                  and (c.prompt_ids == case.prompt_ids).all()]
        # Inference mode, as exllamav3's Generator runs: the Cache's recurrent-state tensors are
        # inference tensors, and the cached modes touch them directly (get_new_state clears a
        # slot in place). Model.forward carries its own decorator, which is why the nc mode
        # never needed this.
        with torch.inference_mode():
            rec = run_case(torch, rc, model, cache, cfg, case, others, args, tok, all_cases = all_cases)
        report["cases"].append(rec)
        _write(out, report)

    ok = True
    print("\nSUMMARY")
    verdicts = set()
    for rec in report["cases"]:
        ok &= case_ok(rec)
        for name, e in rec["runs"].items():
            print(f"  {rec['case']:<10} {name:<12} {e['verdict']}")
            verdicts.add(e["verdict"])
            ok &= e["ok"]
        for s in rec["self"]:
            print(f"  {rec['case']:<10} self {s['base']} vs {s['other']}: {'PASS' if s['ok'] else 'FAIL'}")
            ok &= s["ok"]
    if "within-noise" in verdicts:
        print("  within-noise: some check is beyond its calibrated limit but within its noise-shifted limit, "
              "i.e. vLLM differs from itself this much on those positions; it passes (doc/dsv41_tools.md, Gates)")
    for line in isolation_banner(rc, all_cases, getattr(args, "strict", False)):
        print(line)
    report["ok"] = ok
    report["status"] = "completed"
    _write(out, report)
    print(f"report: {out}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------
# Ablation sweep: one process per ablation, the parent never touches CUDA
# ---------------------------------------------------------------------------------

def _strip_args(argv: list[str], drop_flags: set[str], drop_with_value: set[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        key = a.split("=", 1)[0]
        if key in drop_flags:
            continue
        if key in drop_with_value:
            skip = "=" not in a
            continue
        out.append(a)
    return out


def ablation_integrity(report, baseline = None) -> list[str]:
    """An ablation must produce complete finite measurements, not a broken run.

    Numeric disagreement is intentional and is checked separately. Missing rows,
    invalid rankings, contaminated references or changed case/region coverage are
    invalid controls even if their failure would otherwise count as detection.
    Require the scorer's integrity fields rather than assuming absent flags mean
    success. This also rejects old/incomplete reports before interpreting them.
    """
    if not isinstance(report, dict):
        return ["missing measurement report"]
    n = report.get("n")
    if type(n) is not int or n < 2:
        return ["invalid prompt length"]
    expected = n - 1
    problems = []
    for key, value in (("positions", expected), ("coverage", 1.0), ("nonfinite", 0),
                       ("unfilled", 0), ("invalid_ranked_rows", 0),
                       ("reference_contended", False), ("reference_probe_errors", 0)):
        if key not in report or report[key] != value:
            problems.append(f"{key}: {report.get(key)!r}, expected {value!r}")
    global_report = report.get("global") or {}
    for key, value in (("count", expected), ("compared", expected), ("missing_actual", 0)):
        if global_report.get(key) != value:
            problems.append(f"global {key}: {global_report.get(key)!r}, expected {value}")
    regions = report.get("regions")
    if not isinstance(regions, list) or not regions:
        problems.append("missing region measurements")
        regions = []
    for region in regions:
        if region.get("compared") != region.get("count") or region.get("missing_actual") != 0:
            problems.append(f"incomplete region [{region.get('lo')},{region.get('hi')})")
    if baseline is not None:
        for key in ("case", "n", "rep", "positions"):
            if key not in report or report[key] != baseline.get(key):
                problems.append(f"{key} differs from baseline")
        coverage = lambda rows: [(r.get("lo"), r.get("hi"), r.get("count")) for r in rows]
        if coverage(regions) != coverage(baseline.get("regions") or []):
            problems.append("region coverage differs from baseline")
    return problems


def _plan_chunk_starts(rc, report, case, name):
    """The chunk starts of one run from the plan a report records ([] for a stateless run), or
    None when the report carries no plan for it."""
    for c in ((report or {}).get("plan") or {}).get("cases") or []:
        if c.get("case") == case:
            for run in c.get("runs") or []:
                if run.get("name") == name:
                    return rc.chunk_starts(run.get("schedule") or [])
    return None


def sweep_summary(rc, tokens, results, strict: bool = False) -> tuple[dict, list[str], bool]:
    """
    results: {None: baseline report, token: ablated report}, as written by run_gpu.
    Each ablated run is checked against the baseline run of the same case and mode, with the
    chunk starts of its plan (an ablation that acts at chunk starts is n/a on a run without one).
    Every run is judged as the gates judge it (noise-shifted limits where the run recorded its
    reference noise; strict: the --strict rule for isolation metadata).
    """
    base = results.get(None)
    summary = {"tokens": {}}
    lines = []
    all_ok = (bool(tokens) and base is not None and base.get("ok") is True
              and bool(base.get("cases")) and all(case_ok(rec) for rec in base["cases"]))
    if base is None:
        lines.append("  baseline wrote no report")
    elif not all_ok:
        lines.append("  baseline failed/incomplete, or no ablations were requested")
    base_runs = {} if base is None else {(rec["case"], name): e for rec in base["cases"]
                                           for name, e in rec["runs"].items()}
    for (case, name), entry in base_runs.items():
        problems = ablation_integrity(entry.get("report"))
        if problems or not rc.gates(entry["report"], strict = strict)[0]:
            all_ok = False
            lines.append(f"  baseline {case} {name}: invalid/failed measurements")
            lines += [f"      {problem}" for problem in problems]
    for t in tokens:
        r = results.get(t)
        summary["tokens"][t] = []
        if r is None or base is None:
            lines.append(f"  {t:<17} no report")
            all_ok = False
            continue
        seen = False
        actual_runs = {(rec["case"], name) for rec in r["cases"] for name in rec["runs"]}
        if actual_runs != set(base_runs):
            lines.append(f"  {t:<17} case/run coverage differs from the baseline")
            all_ok = False
        for rec in r["cases"]:
            for name, e in rec["runs"].items():
                be = base_runs.get((rec["case"], name))
                if "report" not in e:
                    lines.append(f"  {t:<17} {rec['case']:<10} {name:<12} error: {e.get('error')}")
                    all_ok = False
                    continue
                problems = ablation_integrity(e["report"], be.get("report") if be else None)
                if problems:
                    chk = {"token": t, "case": rec["case"], "run": name,
                           "status": "invalid", "ok": False, "msgs": problems}
                    summary["tokens"][t].append(chk)
                    lines.append(f"  {t:<17} {rec['case']:<10} {name:<12} invalid measurements")
                    lines += [f"      {problem}" for problem in problems]
                    all_ok = False
                    continue
                chk = rc.ablation_check(e["report"], t, be["report"] if be and "report" in be else None,
                                        chunk_starts = _plan_chunk_starts(rc, r, rec["case"], name))
                summary["tokens"][t].append({"case": rec["case"], "run": name, **chk})
                lines.append(f"  {t:<17} {rec['case']:<10} {name:<12} {chk['status']}")
                lines += [f"      {m}" for m in chk["msgs"]]
                seen |= chk["ok"] is not None
                if chk["ok"] is False:
                    all_ok = False
        if not seen:
            why = "no run has a chunk start" if rc.ABLATIONS[t].at_chunk_starts else \
                f"no case reaches position {rc.ABLATIONS[t].affects_from}"
            lines.append(f"  {t:<17} never exercised: {why}")
            all_ok = False
    summary["ok"] = all_ok
    return summary, lines, all_ok


def sweep_tokens(args, rc) -> tuple[list[str], list[str]]:
    """(the ablations to sweep, errors). Without --sweep-tokens: every ablation this run can
    exercise; tokens named explicitly must all be exercisable."""
    if args.sweep_tokens is None:
        tokens = [t for t in rc.ABLATIONS if not ablation_errors(args, rc, t)]
        for t in rc.ABLATIONS:
            if t not in tokens:
                print(f"sweep: not sweeping {t}: {ablation_errors(args, rc, t)[0]}")
        return tokens, []
    tokens = [t for t in args.sweep_tokens.split(",") if t]
    errors = [e for t in tokens for e in ablation_errors(args, rc, t)]
    if not tokens:
        errors.append("at least one ablation must be requested")
    return tokens, errors


def run_sweep(args, rc, argv, out) -> int:
    tokens, errors = sweep_tokens(args, rc)
    if errors:
        for e in errors:
            print(e)
        return 2
    base_argv = _strip_args(argv, {"--ablate-sweep", "--dry-run"}, {"--sweep-tokens", "--out", "--ablate"})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="dsv41-sweep-", dir=os.path.dirname(os.path.abspath(out)))
    stem = os.path.join(run_dir, "run")
    runs = [(None, f"{stem}-baseline.json")] + [(t, f"{stem}-{t}.json") for t in tokens]
    results = {}
    # the baseline must not inherit an ablation from the caller's environment
    child_env = {k: v for k, v in os.environ.items() if k != "EXL3_DSV41_ABLATE"}
    for t, path in runs:
        cmd = [sys.executable, os.path.abspath(__file__)] + base_argv + ["--out", path]
        if t:
            cmd += ["--ablate", t]
        print(f"\n### sweep: {'baseline' if t is None else t}: {' '.join(cmd)}", flush = True)
        rcode = subprocess.call(cmd, env = child_env)
        try:
            if rcode not in (0, 1):
                raise ValueError(f"child did not complete: exit {rcode}")
            with open(path) as f:
                results[t] = json.load(f)
            if "ok" not in results[t] or results[t].get("status") != "completed":
                raise ValueError("child wrote only a partial report")
        except (OSError, ValueError):
            print(f"### sweep: {t or 'baseline'} wrote no report (exit {rcode})")
            results[t] = None
    summary, lines, all_ok = sweep_summary(rc, tokens, results, strict = getattr(args, "strict", False))
    summary["baseline"] = runs[0][1]
    print("\nABLATION SWEEP")
    for line in lines:
        print(line)
    summary["ok"] = all_ok
    _write(out, summary)
    print(f"sweep report: {out}")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------------

def main(argv = None) -> int:
    rc = load_refcompare()
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser(rc).parse_args(argv)
    if not args.ref:
        print("no reference capture: pass --ref or set DSV41_VLLM_REF")
        return 2
    if not args.dry_run and not args.model:
        print("no model: pass --model or set DSV41_MODEL_DIR")
        return 2
    all_cases = rc.load_ref(args.ref)
    wanted = _ints(args.cases)
    cases = [c for c in all_cases if c.n in wanted and c.rep == args.ref_rep]
    missing = sorted(set(wanted) - {c.n for c in cases})
    if missing:
        print(f"no rep-{args.ref_rep} case for n={missing} in {args.ref} "
              f"(have: {sorted({(c.n, c.rep) for c in all_cases})})")
        return 2
    cases.sort(key = lambda c: wanted.index(c.n))
    for line in isolation_banner(rc, all_cases, args.strict):
        print(line)
    try:
        plan = make_plan(args, rc, cases)
    except ValueError as error:
        print(f"invalid plan: {error}")
        return 2
    print_plan(plan, rc)
    tag = "-".join(plan["ablate"])
    out = args.out or os.path.join(DEFAULT_OUT_DIR, f"{args.mode}{'-' + tag if tag else ''}.json")

    if args.dry_run:
        print("\nreference self-noise (vLLM rep vs rep0, what the calibrated limits were set on):")
        noise = []
        for r in rc.self_noise(all_cases):
            ok, _ = rc.gates(r)
            g = r["global"]
            print(f"  {r['label']}: {'PASS' if ok else 'FAIL'}  dNLL {g['dnll']:+.4f}  |dlp| p50 {g['dlp']['p50']:.4f} "
                  f"p90 {g['dlp']['p90']:.4f} p99 {g['dlp']['p99']:.4f}  top1 {g['top1']['agree']}/{g['top1']['total']} "
                  f"ovl {g['overlap']:.3f}")
            noise.append({"label": r["label"], "ok": ok, "global": g})
        print("\nreference prefix noise (vLLM, the same prefix inside a longer request, against the calibrated "
              "limits, which were not set on it: a FAIL here is vLLM failing against itself):")
        pnoise = []
        for r in rc.prefix_noise(all_cases):
            ok, msgs = rc.gates(r)
            g = r["global"]
            print(f"  {r['label']}: {'PASS' if ok else 'FAIL'}  dNLL {g['dnll']:+.4f}  |dlp| p50 {g['dlp']['p50']:.4f} "
                  f"p90 {g['dlp']['p90']:.4f} max {g['dlp']['max']:.2f}  top1 {g['top1']['agree']}/{g['top1']['total']} "
                  f"ovl {g['overlap']:.3f}")
            for m in msgs:
                if m.startswith("FAIL"):
                    print(f"      {m}")
            pnoise.append({"label": r["label"], "ok": ok, "gates": msgs, "global": g,
                           "regions": r["regions"]})
        case_noise = print_noise_limits(rc, cases, all_cases, args.strict)
        _write(out, {"tool": "dsv41_validate", "dry_run": True, "plan": plan, "ref_noise": noise,
                     "ref_prefix_noise": pnoise, "gates": rc.GATES, "calibration_noise": rc.CALIBRATION_NOISE,
                     "reference_noise": case_noise})
        print(f"\ndry run: plan written to {out}; torch imported: {'torch' in sys.modules}; CUDA untouched")
        return 2 if plan["errors"] else 0
    if plan["errors"]:
        return 2
    if args.ablate_sweep:
        return run_sweep(args, rc, argv, out)
    try:
        return run_gpu(args, rc, cases, all_cases, plan, out)
    except Exception as error:
        try:
            with open(out) as stream:
                failed = json.load(stream)
        except (OSError, ValueError):
            failed = {"tool": "dsv41_validate", "plan": plan}
        failed.update(status="failed", ok=False, error=f"{type(error).__name__}: {error}")
        _write(out, failed)
        print(f"run failed: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
