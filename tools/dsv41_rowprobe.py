#!/usr/bin/env python3
"""
Row-count probe for DeepSeek-V4.1-Flash: does a row of a 1+N-row verify forward get the bits a
one-row decode step gives it, and if not, which operation differs first?

The generator verifies a draft with ONE trunk forward of 1+N rows and keeps a draft token only
when it equals the trunk's own sample, so drafted output equals undrafted output exactly when
every row of that forward gets the bits a one-row step gives it. This tool measures that on the
cached path, in the arithmetic of the calling environment:

  1. Loads the model with a Cache, as the cached mode of tools/dsv41_validate.py does, and
     prefills --prefix tokens of one sequence (a case of a vLLM capture, or --text).
  2. For each row count K of --rows, from the SAME job state: (a) K one-row forwards of tokens
     t0 .. t(K-1) with the generator's decode params, (b) one forward of those K rows with the
     generator's verify params ("recurrent_history": True).
  3. Captures, in both arms and for every row, the inputs and outputs of the operations of each
     block (engram, mHC mix and apply, norms, projections, compressor, index keys, top-k
     selection, attention, router, MoE) and of the head, by wrapping the Python methods and the
     native calls that produce them.
  4. Reports per K and operation whether the rows are bit-equal, where they first differ, how
     many elements differ and by how much, whether id-valued outputs differ as SETS, and whether
     the operation is an ORIGIN (inputs bit-equal, outputs not) or only propagates a difference.
  5. Replays (unless --no-replay): inside the K-row forward every wrapped operation is run again
     one row at a time on the very inputs the K-row call had. That shows whether the operation
     itself depends on the row count, also below the first origin, where the in-situ comparison
     only sees propagated differences.

State between the arms is restored by the engine's own rewind (DSV41State.rewind, what the
generator's reject path calls) on the one job state, not by a second state prefilled the same
way: two prefills are separate GPU computations that nothing guarantees to be bit-equal, while a
rewind keeps the very bytes. It is exact because all V4.1 job state is addressed by position (the
window ring, the rate-2 carry ring, the engram id ring, the pool entries) and a forward writes
every row at or above its start position before any row reads it. The tool does not rely on that
argument. Before the first K-row forward it runs the one-row arm on the state as prefilled, twice
unhooked and once captured (the pristine reference), and it requires every later one-row run to
equal that reference: the logits of each, and every captured tensor of arm (a). A K-row forward
that left something behind for a later one-row step to read would fail that check. The K-row arm
is checked from the other side: a K-row forward that follows a K-row forward must equal one that
follows one-row steps. Not restored by a rewind: window_beg after a ring shift (the rows keep
their bytes at another ring offset; reported when it happens), the expert cache's policy state
and the n-gram row cache, which decide where bytes live, not their values.

Self-checks, all bit-exact, each printed and stored: unhooked run against unhooked run
(determinism), hooked against unhooked logits (the wrappers change nothing), the pristine
reference against every later one-row run, the repeat control (arm (a) again after arm (b), every
captured tensor), the replay run against the capture run on every captured tensor (the replays
change nothing), and coverage: every operation the layer facts call for was captured on every row
or pool entry, else it is listed as NOT CAPTURED.

The launch autotuner of the EXL3 kernels keeps its choices in a file (coop_autotune_v1.bin); by
default the tool runs on a private copy of it (--tune-cache), so that serving's file is read but
never written. Runs that are to be compared should share one copy (--tune-cache PATH): a row
bucket the file has no record for is tuned by timing, and two runs can tune it differently.

The whole body runs under main(): exllamav3's CPU-MoE worker is started with multiprocessing
spawn, which re-imports this file in the child. See doc/dsv41_tools.md.
"""

import argparse
import gc
import hashlib
import importlib.util
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(tempfile.gettempdir(), "dsv41_rowprobe")
PAGE_SIZE = 256         # exllamav3.constants.PAGE_SIZE (not imported: --stable-check stays torch-free)
MAX_ROWS = 8            # block_sparse_mlp.MAX_BSZN: the widest forward that still takes the decode kernels

# Captured operations in execution order within a block, then the head
OP_ORDER = (
    "engram.wkv", "engram.gate", "engram.out",
    "hc_attn.mix", "attn_norm",
    "attn.wq_a", "attn.q_norm", "attn.wq_b", "attn.wkv", "attn.kv_norm", "attn.rope",
    "comp.wkv", "comp.wgate", "comp.pool", "comp.norm", "comp.fused", "idx.wk", "idx.k_norm", "pool.store",
    "idx.wq_b", "idx.weights_proj", "idx.scores", "idx.topk",
    "attn.dsa_attn", "attn.wo_a", "attn.wo_b", "hc_attn.apply",
    "hc_ffn.mix", "ffn_norm", "moe.router", "moe.out", "hc_ffn.apply",
    "head.collapse", "norm", "head",
)
# Operations whose units are pool entries (closed groups), not token rows
ENTRY_OPS = frozenset(("comp.pool", "comp.norm", "comp.fused", "idx.wk", "idx.k_norm", "pool.store"))
# Call facts that differ between the arms by construction
META_SKIP = frozenset(("rows", "ec", "pos0", "bound", "est"))
# Environment switches that select arithmetic on this path (doc/env_vars.md), printed with the policy
POLICY_ENV = ("EXL3_STABLE_ARITHMETIC", "EXL3_HGEMM_FIXED_ROWS", "EXL3_MOE_FUSED_PREFILL", "EXL3_MOE_FUSED_DET",
              "EXL3_NO_FUSED_RECONSTRUCT", "EXL3_INT8_GEMV", "EXL3_INT8_GEMV_MAX_K", "EXL3_GEMV",
              "EXL3_DSV41_FUSED_COMPRESS", "EXL3_DSV41_NUMERICS", "EXL3_DSV41_XDEV_BF16", "EXL3_MOE_COOP_WIDE",
              "EXL3_MOE_COOP_KSPLIT", "EXL3_MOE_SHARED_COOP", "EXL3_FP32_LOGITS", "EXLLAMAV3_TUNE_CACHE")

# What the extension and Config take when EXL3_INT8_GEMV is not set (exl3_gemv_int8.cu,
# exl3_gemv_int8_mode; model/config.py): setting it to this value changes nothing
INT8_GEMV_DEFAULT = "2"
# Captured on every row of every block, whatever the layer's role
BLOCK_OPS = ("hc_attn.mix", "attn_norm", "attn.wq_a", "attn.q_norm", "attn.wq_b", "attn.wkv", "attn.kv_norm",
             "attn.rope", "attn.dsa_attn", "attn.wo_a", "attn.wo_b", "hc_attn.apply",
             "hc_ffn.mix", "ffn_norm", "moe.router", "moe.out", "hc_ffn.apply")
HEAD_OPS = ("head.collapse", "norm", "head")

TUNE_MAGIC = b"EX3ATUNE"
TUNE_FILE = "coop_autotune_v1.bin"
_M64 = (1 << 64) - 1
_FNV_PRIME = 1099511628211
_MISSING = object()


def _load_validate():
    """tools/dsv41_validate.py by path (torch-free at import): its loaders, RAM watchdog and report writer."""
    name = "dsv41_validate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "tools", "dsv41_validate.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "dsv41_rowprobe.py",
        description = "DeepSeek-V4.1-Flash: one-row decode steps against one multi-row verify forward, "
                      "operation by operation, bit for bit.",
        epilog = """\
output, per row count K:
  in-situ table   one line per captured output, in execution order: '=' or DIFF, the first layer and
                  row that differ, units and elements that differ, the largest absolute and relative
                  difference, how many units are an ORIGIN (inputs bit-equal, outputs not), and for
                  id-valued outputs (top-k indices, candidate blocks, expert ids) how many rows differ
                  as sets. Units are rows (r0 = the first token after the prefix) or pool entries (e<n>)
  FIRST           the first diverging operation of the whole forward, and the origins in order
  dispatch        call facts that differ between the arms (mHC chunk count, indexer kernel, attention
                  split count, dense or gathered pool, grouped-GEMM launch tag)
  logits          per row: bit equality, argmax of both arms, the top-2 gap
  replay table    each operation run one row at a time on the K-row call's own inputs, against its
                  K-row result: '=' means the operation gives a row the same bits at 1 and at K rows
then one table operation x K for the in-situ runs ('=', or the first differing layer; 'L3*' the
operation is an origin at that layer, 'L3/o20' it only propagates there and is first an origin at
layer 20) and one for the replays; the kernel each EXL3 linear takes at 1..8 rows (0 = int8 GEMV,
90 = FP16 GEMV, 1..4 = cooperative kernel shape) with its autotune records; and the self-checks.
exit status: 0 the probe ran, its self-checks passed and every expected operation was captured
(whatever it found), 1 a self-check failed, an operation was not captured or the run broke,
2 refused.

examples:
  python3 tools/dsv41_rowprobe.py --stable-check
  python3 tools/dsv41_rowprobe.py --model DIR --ref capture.json --case n1500r0 --visible 1,0,2 \\
      --placement '0-10=cuda:0; 11-13=cuda:2; 14-39=cuda:1 experts=cache; ram experts=all; disk experts=off'
  python3 tools/dsv41_rowprobe.py --model DIR --ref capture.json --prefix 40 --rows 2,8 --no-replay
  EXL3_STABLE_ARITHMETIC=1 python3 tools/dsv41_rowprobe.py --model DIR --ref capture.json --allow-stable
  (--model defaults to DSV41_MODEL_DIR, --ref to DSV41_VLLM_REF)
""", formatter_class = argparse.RawDescriptionHelpFormatter, allow_abbrev = False)
    ap.add_argument("--model", default = os.environ.get("DSV41_MODEL_DIR"),
                    help = "checkpoint directory (default: DSV41_MODEL_DIR)")
    ap.add_argument("--ref", default = os.environ.get("DSV41_VLLM_REF"),
                    help = "vLLM capture whose prompt ids are the token source (default: DSV41_VLLM_REF)")
    ap.add_argument("--case", default = "n1500r0", help = "case of --ref, as n<length>r<rep> (default n1500r0)")
    ap.add_argument("--text", default = None, help = "token source instead of --ref: this text, with BOS")
    ap.add_argument("--text-file", default = None, help = "token source instead of --ref: this file's text, with BOS")
    ap.add_argument("--prefix", type = int, default = 1200,
                    help = "tokens prefilled before the probed rows (default 1200: every compressed pool is past "
                           "its 512-entry threshold, so the top-512 selection is active; a short prefix such as "
                           "40 probes the dense regime). The candidate stage of the index sources above the "
                           "candidate source masks nothing until a row sees more than candidate_topk_blocks * "
                           "candidate_block_size entries (2048 * 8 = 16384 on V4.1-Flash): probing it needs a "
                           "prefix past that, e.g. 19000")
    ap.add_argument("--rows", default = "2,3,4,5,6,7,8", help = f"row counts K to probe, each 2..{MAX_ROWS}")
    ap.add_argument("--visible", default = None, help = "CUDA_VISIBLE_DEVICES, with CUDA_DEVICE_ORDER=PCI_BUS_ID")
    ap.add_argument("--placement", default = None, help = "explicit placement (EXL3_PLACEMENT rules)")
    ap.add_argument("--gpu-split", default = None, help = "GiB per device for model.load(use_per_device=...)")
    ap.add_argument("--reserve", default = None, help = "GiB per device for model.load(reserve_per_device=...)")
    ap.add_argument("--max-chunk-size", type = int, default = None, help = "model.load max_chunk_size (default 2048)")
    ap.add_argument("--cache-tokens", type = int, default = None,
                    help = "Cache max_num_tokens (default: the probed length, page aligned, >= 65536)")
    ap.add_argument("--prefill-chunk", type = int, default = None,
                    help = "rows per prefill call of the prefix (default: min(2048, max chunk size))")
    ap.add_argument("--dsv41-numerics", default = None, help = "config.dsv41_numerics for this run")
    ap.add_argument("--fp32-logits", action = "store_true", help = "infer_params.fp32_logits: head logits in FP32")
    ap.add_argument("--no-replay", action = "store_true", help = "in-situ comparison only, no per-row replays")
    ap.add_argument("--no-int8-split", action = "store_true",
                    help = "skip the replays that set EXL3_INT8_GEMV=0 in-process (they separate the int8-vs-FP16 "
                           "switch of the EXL3 linears from the row buckets of the launch autotuner) and the "
                           "kernel tags")
    ap.add_argument("--no-repeat", action = "store_true", help = "skip the repeat control (arm (a) run again)")
    ap.add_argument("--timing-reps", type = int, default = 5,
                    help = "unhooked repetitions of both arms per K (default 5, at least 2): all must agree bit "
                           "for bit, and the verify cost is reported as their median and minimum")
    ap.add_argument("--tune-cache", default = "copy", metavar = "copy|live|PATH",
                    help = "launch-autotune cache the extension uses: 'copy' (default) a private copy of the live "
                           "file, removed when the run ends, so the probe reads serving's records and never "
                           "writes that file; 'live' the file itself; or a path, created as a copy of the live "
                           "file when it does not exist and kept, so that several runs share the records one of "
                           "them tuned")
    ap.add_argument("--stable-check", action = "store_true",
                    help = "only print which switches of exllamav3/model/math_policy.py are on; no torch, no CUDA")
    ap.add_argument("--allow-stable", action = "store_true",
                    help = "run under EXL3_STABLE_ARITHMETIC=1 (refused otherwise: the probe is about the default "
                           "arithmetic; under the profile every operation should come out equal, which validates "
                           "the probe)")
    ap.add_argument("--min-avail-gib", type = float, default = 8.0,
                    help = "RAM watchdog: kill this run and its children below this MemAvailable (0: off)")
    ap.add_argument("--out", default = None, help = f"JSON report (default {DEFAULT_OUT_DIR}/rowprobe.json)")
    # read by dsv41_validate.apply_settings, which this tool reuses; not options here
    ap.set_defaults(mcl = None, mcs = None, mct = None, moe_cpu_mode = None)
    return ap


# ---------------------------------------------------------------------------------
# Math policy (torch-free)
# ---------------------------------------------------------------------------------

def math_policy(val) -> dict:
    """The switches of exllamav3/model/math_policy.py as this environment sets them (loaded by path)."""
    mp = val._load_by_path("_dsv41_rowprobe_math_policy", os.path.join("exllamav3", "model", "math_policy.py"))
    if mp is None:
        raise ValueError("this tree has no exllamav3/model/math_policy.py")
    return {"stable_arithmetic": bool(mp.STABLE_ARITHMETIC), "hgemm_fixed_rows": int(mp.HGEMM_FIXED_ROWS),
            "fused_prefill": bool(mp.FUSED_PREFILL)}


def print_policy(policy: dict):
    on = lambda v: "ON" if v else "off"
    print("math policy (exllamav3/model/math_policy.py, as this environment sets it):")
    print(f"  STABLE_ARITHMETIC  {on(policy['stable_arithmetic'])}")
    print(f"  HGEMM_FIXED_ROWS   {policy['hgemm_fixed_rows'] or 'off'}")
    print(f"  FUSED_PREFILL      {on(policy['fused_prefill'])}")
    env = [f"{k}={os.environ[k]}" for k in POLICY_ENV if k in os.environ]
    print(f"  arithmetic switches set in the environment: {' '.join(env) if env else 'none'}", flush = True)


# ---------------------------------------------------------------------------------
# Launch-autotune cache (exllamav3_ext/quant/coop_autotune.cu)
# ---------------------------------------------------------------------------------

def tune_cache_live_path(environ = None) -> str | None:
    """The file the extension persists its launch choices in (disk_cache_path), or None."""
    environ = os.environ if environ is None else environ
    override = environ.get("EXLLAMAV3_TUNE_CACHE")
    if override:
        return os.path.join(override, TUNE_FILE) if os.path.isdir(override) else override
    if os.name == "nt":
        base = environ.get("LOCALAPPDATA")
        if not base:
            profile = environ.get("USERPROFILE")
            if not profile:
                return None
            local = os.path.join(profile, "AppData", "Local")
            base = local if os.path.exists(local) else profile
    else:
        base = environ.get("XDG_CACHE_HOME") or (os.path.join(environ["HOME"], ".cache") if environ.get("HOME") else None)
        if not base:
            return None
    return os.path.join(base, "exllamav3", "autotune", TUNE_FILE)


def read_tune_cache(path) -> dict:
    """{salted key: (tag, block_dim, num_sms, concurrency)} of an autotune cache file; {} if absent or foreign."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except (OSError, TypeError):
        return {}
    if len(data) < 16 or data[:8] != TUNE_MAGIC:
        return {}
    fmt, size = struct.unpack_from("<II", data, 8)
    if fmt != 1 or size < 32:
        return {}
    out, off = {}, 16
    while off + size <= len(data):
        if data[off:off + 8] == TUNE_MAGIC:         # a second writer's header between records
            off += 16
            continue
        key, tag, block_dim, num_sms, concurrency = struct.unpack_from("<Qiiii", data, off)
        out[key] = (tag, block_dim, num_sms, concurrency)
        off += size
    return out


def _sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, TypeError):
        return None


def prepare_tune_cache(mode: str, out: str) -> tuple[dict, dict]:
    """Decide the autotune cache of this run: (info, the records the file holds before the run).
    info['env'] is the EXLLAMAV3_TUNE_CACHE value to set, or None."""
    live = tune_cache_live_path()
    live_records = read_tune_cache(live)
    info = {"mode": mode, "live_path": live, "live_sha256": _sha256(live), "live_records": len(live_records),
            "used_path": live, "env": None, "private": False, "seeded_from_live": False}
    if mode != "live":
        if mode == "copy":
            used = os.path.join(os.path.dirname(os.path.abspath(out)), f"rowprobe-autotune-{os.getpid()}.bin")
            os.makedirs(os.path.dirname(used), exist_ok = True)
            if os.path.exists(used):
                os.unlink(used)
            info["private"] = True
        else:
            used = os.path.join(mode, TUNE_FILE) if os.path.isdir(mode) else mode
        if not os.path.exists(used) and live and os.path.isfile(live):
            shutil.copyfile(live, used)
            info["seeded_from_live"] = True
        info["used_path"] = info["env"] = used
    # what the file holds before this run, and which of it the live file lacks (earlier probe runs tuned it)
    before = read_tune_cache(info["used_path"])
    info["records_before"] = len(before)
    info["carried_over"] = {f"{k:016x}": list(v) for k, v in before.items() if k not in live_records}
    return info, before


def _autotune_version() -> int:
    """COOP_AUTOTUNE_VERSION, the salt of the cache keys, from this tree's source (4 if it cannot be read)."""
    try:
        with open(os.path.join(HERE, "exllamav3", "exllamav3_ext", "quant", "coop_autotune.cu")) as f:
            m = re.search(r"COOP_AUTOTUNE_VERSION\s*=\s*(\d+)", f.read())
        return int(m.group(1)) if m else 4
    except OSError:
        return 4


def _fnv(values, h: int = 1469598103934665603) -> int:
    for v in values:
        h ^= int(v) & _M64
        h = (h * _FNV_PRIME) & _M64
    return h


def _pow2_bucket(m: int) -> int:
    return min(1 << max(m - 1, 0).bit_length(), 16)


def autotune_key(ent: dict, m: int, version: int) -> int:
    """
    The salted cache key of a linear at m rows, as exl3_gemm.cu forms it (gemm_autotune_hash,
    mgemm_autotune_hash; salt_hash in coop_autotune.cu). A grouped GEMM keys on m itself, a plain
    one on max(m, 2).
    """
    grouped = ent.get("groups")
    k_int = int(ent["K"])
    half_k = 0 if float(ent["K"]).is_integer() else 1
    cb = 2 if ent["mul1"] else (1 if ent["mcg"] else 0)
    h = _fnv((half_k, _pow2_bucket(m if grouped else max(m, 2)), ent["in"], ent["out"], k_int,
              1 if ent["fp32_out"] else 0, ent["device_index"], ent["cc"], ent["num_sms"], cb))
    if grouped:
        h = _fnv((min(grouped, 24), min(grouped, 24)), h)
    return ((h ^ version) * _FNV_PRIME) & _M64


# ---------------------------------------------------------------------------------
# Bit comparison
# ---------------------------------------------------------------------------------

def diff_stats(torch, a, b, ids: bool = False) -> dict:
    """
    a against b, bit for bit: equal, ndiff (differing elements; -1 when shape or dtype differ),
    numel, and for a difference max_abs / max_rel (float) over the differing elements; with ids,
    set_equal over the values other than the -1 padding. Bit-equal NaNs are equal; +0 and -0 differ.
    """
    n = int(b.numel())
    if a.shape != b.shape or a.dtype != b.dtype:
        st = {"equal": False, "ndiff": -1, "numel": n, "max_abs": None, "max_rel": None,
              "shapes": [list(a.shape), list(b.shape)], "dtypes": [str(a.dtype), str(b.dtype)]}
        if ids:
            sa, sb = set(a.reshape(-1).tolist()) - {-1}, set(b.reshape(-1).tolist()) - {-1}
            st["set_equal"] = sa == sb
        return st
    if n == 0:
        return {"equal": True, "ndiff": 0, "numel": 0}
    af, bf = a.reshape(-1).contiguous(), b.reshape(-1).contiguous()
    if af.is_floating_point():
        view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[af.element_size()]
        ne = af.view(view) != bf.view(view)
    else:
        ne = af != bf
    nd = int(ne.sum())
    if nd == 0:
        return {"equal": True, "ndiff": 0, "numel": n}
    st = {"equal": False, "ndiff": nd, "numel": n, "max_abs": None, "max_rel": None}
    if af.is_floating_point():
        ad, bd = af.double(), bf.double()
        d = torch.nan_to_num((ad - bd).abs(), nan = math.inf, posinf = math.inf)
        d = torch.where(ne, d, torch.zeros_like(d))
        den = torch.maximum(ad.abs(), bd.abs())
        rel = torch.where(ne & (den > 0), d / den, torch.zeros_like(d))
        st["max_abs"] = float(d.max())
        st["max_rel"] = float(torch.nan_to_num(rel, nan = math.inf, posinf = math.inf).max())
    else:
        st["max_abs"] = float((af.long() - bf.long()).abs().max())
    if ids:
        sa, sb = set(af.tolist()) - {-1}, set(bf.tolist()) - {-1}
        st["set_equal"] = sa == sb
    return st


def _one_row(x, r: int):
    """Row r of x (rows = every dim but the last) as a fresh tensor of the shape a one-row call passes."""
    row = x.reshape(-1, x.shape[-1])[r:r + 1].clone()
    return row.view(*([1] * (x.dim() - 1)), x.shape[-1])


def _op_index(op: str) -> float:
    if op in OP_ORDER:
        return float(OP_ORDER.index(op))
    if op.startswith("attn.wo_a."):
        return OP_ORDER.index("attn.wo_a") + 0.5
    if op == "attn.project_qkv":        # a replay of the whole projection front, listed after its parts
        return OP_ORDER.index("attn.rope") + 0.5
    return float(len(OP_ORDER))


class _Int8Off:
    """EXL3_INT8_GEMV=0 for the launches inside: the extension reads the variable on every launch
    (exl3_gemv_int8.cu, exl3_gemv_int8_mode), so the switch can be flipped in-process.

    Other threads are alive by then (the RAM watchdog, the engram prefetch executor, the expert
    tier's workers) and native code calls getenv at run time, so the variable is never added or
    removed here, only replaced: run_gpu sets it to its default before any thread starts, and a
    replaced value leaves the environment array as it is."""

    def __enter__(self):
        self.old = os.environ.get("EXL3_INT8_GEMV", INT8_GEMV_DEFAULT)
        os.environ["EXL3_INT8_GEMV"] = "0"

    def __exit__(self, *exc):
        os.environ["EXL3_INT8_GEMV"] = self.old
        return False


class Run:
    """
    What one arm captured: rec[(op, layer)] = {"units": {unit: {"in": {name: tensor}, "out":
    {name: tensor}}}, "meta": [dict per call], "ids": names of id-valued outputs}. A unit is an
    absolute token position, or a pool entry index for the operations of ENTRY_OPS. Tensors are
    CPU copies taken when the operation returned.
    """

    def __init__(self, label: str):
        self.label = label
        self.rec = {}
        self.logits = None


# ---------------------------------------------------------------------------------
# The wrappers
# ---------------------------------------------------------------------------------

class Probe:
    """
    Wraps the operations of a loaded DeepseekV41Model. With mode None every wrapper calls straight
    through; "capture" records inputs and outputs into self.run; "replay" also runs each operation
    again one row at a time on the same inputs and accumulates the comparison in self.replays.
    A wrapper never changes what the engine computes: it reads, copies, and (replay) restores every
    engine buffer a replay call wrote. The tool verifies that, bit for bit, on every run.
    """

    def __init__(self, torch, ext, model, position: int, replay_int8: bool = True, stable: bool = False):
        self.torch, self.ext, self.model, self.P = torch, ext, model, position
        self.int8_split = replay_int8
        self.stable = stable        # EXL3_STABLE_ARITHMETIC: mHC mixes in one chunk at every row count
        self.mode = None
        self.run = None
        self.layer = None           # current block (n_layers for the head), set by the block wrappers
        self.pos0 = 0               # absolute position of row 0 of the current forward
        self.rows = 0               # rows of the current forward
        self.suspend = 0            # > 0 inside a replay: nested wrappers pass through
        self.entry = None           # (first entry, rate) while a kv source pools and stores
        self.last = {}              # (layer, op) -> outputs this forward captured, for derived inputs
        self.gpu = {}               # (layer, "engram.wkv") -> that projection's device output, this forward
        self.in_woa = False
        self.moe_stage = None       # "real": capture the next routing call; "forced": overwrite its result
        self.route = None
        self.force = None
        self.replays = {}
        self.disabled = set()
        self.notes = []
        self.errors = []
        self.tags = {}
        self.settled = set()
        self.patches = []
        self.layers = {}
        self.linears = {}
        self.n_layers = 0
        self.fused_compress = False

    # -- bookkeeping --

    def on(self) -> bool:
        return self.mode is not None and self.suspend == 0

    def begin(self, run: Run, pos0: int, rows: int):
        """Before each forward of a hooked run."""
        self.run, self.pos0, self.rows = run, pos0, rows
        self.layer = self.entry = self.route = self.force = self.moe_stage = None
        self.last, self.gpu, self.in_woa = {}, {}, False

    def note(self, text: str):
        if text not in self.notes and len(self.notes) < 200:
            self.notes.append(text)

    def _error(self, text: str):
        if text not in self.errors and len(self.errors) < 200:
            self.errors.append(text)

    def guard(self, what: str, fn, *args):
        """Capture code must never break the forward it observes."""
        try:
            fn(*args)
        except Exception as e:
            self._error(f"capture {what} L{self.layer}: {type(e).__name__}: {e}")

    def replay(self, what: str, fn, restore = None):
        """Run the per-row replays of one operation with the nested wrappers passing through. An
        error disables this replay for the rest of the run; restore always runs."""
        if self.mode != "replay" or what in self.disabled:
            return
        self.suspend += 1
        try:
            fn()
        except Exception as e:
            self.disabled.add(what)
            self._error(f"replay {what} L{self.layer} disabled: {type(e).__name__}: {e}")
        finally:
            self.suspend -= 1
            if restore is not None:
                restore()

    def units(self, kind: str, n: int, op: str):
        if kind == "entry":
            if self.entry is None:
                self.note(f"{op} L{self.layer}: called outside a pool store; not captured")
                return None
            return [self.entry[0] + i for i in range(n)]
        if n != self.rows:
            self.note(f"{op} L{self.layer}: {n} rows in a {self.rows}-row forward; not captured")
            return None
        return [self.pos0 + r for r in range(n)]

    def record(self, op: str, units, ins: dict, outs: dict, ids = (), meta = None, append: bool = False):
        """Store CPU copies, unit by unit. Values are unit-major tensors or lists of per-unit tensors."""
        torch = self.torch
        ent = self.run.rec.get((op, self.layer))
        if ent is None:
            ent = self.run.rec[(op, self.layer)] = {"units": {}, "meta": [], "ids": set(ids)}
        if meta is not None:
            ent["meta"].append(meta)

        def snap(v):
            if isinstance(v, (list, tuple)):
                return [snap(t) for t in v]
            return v.detach().to("cpu") if v.is_cuda else v.detach().clone()

        ins = {k: snap(v) for k, v in ins.items() if v is not None}
        outs = {k: snap(v) for k, v in outs.items() if v is not None}
        for k, v in list(ins.items()) + list(outs.items()):
            if len(v) != len(units):
                raise ValueError(f"{op}:{k} holds {len(v)} units, expected {len(units)}")
        for i, u in enumerate(units):
            slot = ent["units"].setdefault(u, {"in": {}, "out": {}})
            for k, v in ins.items():
                slot["in"][k] = v[i]
            for k, v in outs.items():
                slot["out"][k] = torch.cat([slot["out"][k], v[i]]) if append and k in slot["out"] else v[i]
        self.last[(self.layer, op)] = outs

    def rep(self, label: str, row: int, one, many, ids: bool = False, note: str | None = None):
        """One replayed row: the one-row result against the same row of the multi-row result."""
        st = diff_stats(self.torch, one, many, ids)
        d = self.replays.get(label)
        if d is None:
            d = self.replays[label] = {"rows": 0, "rows_diff": 0, "ndiff": 0, "numel": 0, "max_abs": 0.0,
                                       "max_rel": 0.0, "set_diff": 0, "first": None, "layers_diff": [], "note": None}
        d["rows"] += 1
        d["numel"] += st["numel"]
        if note:
            d["note"] = note
        if st["equal"]:
            return
        d["rows_diff"] += 1
        d["ndiff"] += max(st["ndiff"], 0)
        for k in ("max_abs", "max_rel"):
            if st.get(k) is not None:
                d[k] = max(d[k], st[k])
        if st.get("set_equal") is False:
            d["set_diff"] += 1
        if d["first"] is None:
            d["first"] = {"layer": self.layer, "row": row}
        if self.layer not in d["layers_diff"]:
            d["layers_diff"].append(self.layer)

    # -- patching --

    def patch(self, obj, name: str, make):
        """obj.name = make(original); undone by uninstall()."""
        raw = vars(obj).get(name, _MISSING)
        new = make(getattr(obj, name))
        if isinstance(raw, staticmethod):
            new = staticmethod(new)
        setattr(obj, name, new)
        self.patches.append((obj, name, raw))

    def uninstall(self):
        for obj, name, raw in reversed(self.patches):
            if raw is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, raw)
        self.patches = []
        self.mode = None

    def install(self):
        from exllamav3.modules import dsv41 as m_dsv41, dsv41_cached as m_cached
        from exllamav3.modules.attention_fn import dsa_triton as m_triton
        from exllamav3.architecture.dsv41 import compressor as m_comp
        from exllamav3.cache.dsv41 import FUSED_COMPRESS
        self.fused_compress = bool(FUSED_COMPRESS)      # which carry rings the layer states were built with
        model = self.model
        self.n_layers = n = model.config.num_hidden_layers
        blocks = model.modules[model.first_block_idx: model.first_block_idx + n]
        assert len(blocks) == n and all(type(b).__name__ == "DSV41Block" for b in blocks), \
            "dsv41_rowprobe: not a DeepSeek-V4.1 module graph"
        for b in blocks:
            L = b.layer_idx
            at = b.attn
            self.layers[L] = {
                "device": str(b.device), "compress_ratio": at.compress_ratio, "kv_source": at.kv_source_layer,
                "index_source": at.index_source_layer, "is_kv_source": bool(at.is_kv_source),
                "is_index_source": bool(at.is_index_source), "engram": b.engram is not None,
                "index_topk": getattr(at, "index_topk", None),
                "comp_gate": at.compressor is not None and at.compressor.wgate is not None,
                "owns_index_k": at.indexer is not None and at.indexer.wk is not None,
                "moe_bc": getattr(b.mlp, "bc", None) is not None,
                "sh_coop": getattr(getattr(b.mlp, "bc", None), "sh_coop", None),
                "expert_cache": getattr(b.mlp, "tier", None) is not None,
            }
            self._hook_block(b, L)
            if b.engram is not None:
                self._hook_linear(b.engram.wkv, "engram.wkv")
                self._hook_engram(b.engram)
            self._hook_hc(b.attn_hc, "hc_attn")
            self._hook_hc(b.mlp_hc, "hc_ffn")
            self._hook_norm(b.attn_norm, "attn_norm")
            self._hook_norm(b.mlp_norm, "ffn_norm")
            self._hook_linear(at.q_a, "attn.wq_a")
            self._hook_norm(at.q_norm, "attn.q_norm")
            self._hook_linear(at.q_b, "attn.wq_b")
            self._hook_linear(at.wkv, "attn.wkv")
            self._hook_norm(at.kv_norm, "attn.kv_norm")
            self._hook_project(at)
            if at.compressor is not None:
                self._hook_linear(at.compressor.wkv, "comp.wkv")
                if at.compressor.wgate is not None:
                    self._hook_linear(at.compressor.wgate, "comp.wgate")
                self._hook_store(at)
            if at.indexer is not None:
                self._hook_linear(at.indexer.wq_b, "idx.wq_b")
                self._hook_linear(at.indexer.weights_proj, "idx.weights_proj")
                if at.indexer.wk is not None:
                    self._hook_linear(at.indexer.wk, "idx.wk", kind = "entry")
                    self._hook_norm(at.indexer.k_norm, "idx.k_norm", kind = "entry")
            for g, lin in enumerate(at.wo_a):
                self._hook_linear(lin, f"attn.wo_a.g{g}")       # the per-group path, when the grouped GEMM is off
            self._hook_woa(at)
            self._hook_linear(at.wo_b, "attn.wo_b")
            self._hook_moe(b.mlp)
        tail = {m.key: m for m in model.modules[model.first_block_idx + n:]}
        self._hook_head(tail["hc_head"])
        self._hook_norm(tail["norm"], "norm")
        self._hook_linear(tail["head"], "head")
        # module-level and native calls
        self.patch(m_dsv41, "dsa_attn", self._make_attn)
        self.patch(m_dsv41, "select_topk", self._make_select)
        self.patch(m_dsv41, "fused_compress", self._make_fused_compress)
        self.patch(m_triton, "dsa_indexer_scores", self._make_scores)
        self.patch(m_comp, "rms_norm", self._make_comp_norm)
        self.patch(m_comp, "group_pool", self._make_group_pool)
        self.patch(m_cached.CompressCarry, "step", self._make_carry_step)
        self.patch(self.ext, "exl3_mgemm", self._make_mgemm)
        self.patch(self.ext, "routing_ds3_nogroup", self._make_routing)

    # -- blocks, head --

    def _hook_block(self, block, L: int):
        def make(orig):
            def forward(x, params, out_dtype = None):
                if self.on():
                    self.layer = L
                return orig(x, params, out_dtype)
            return forward
        self.patch(block, "forward", make)

    def _hook_head(self, mod):
        def make(orig):
            def forward(x, params, out_dtype = None):
                if not self.on():
                    return orig(x, params, out_dtype)
                self.layer = self.n_layers
                out = orig(x, params, out_dtype)
                self.guard("head.collapse", after, orig, x, params, out_dtype, out)
                return out
            return forward

        def after(orig, x, params, out_dtype, out):
            units = self.units("row", x.shape[1], "head.collapse")
            if units is None or x.shape[0] != 1:
                return
            pre = params.get("dsv41_hc_pre")
            self.record("head.collapse", units, {"streams": x[0], "pre": None if pre is None else pre[0]}, {"x": out[0]})
            if pre is None:
                return

            def rows():
                for r in range(x.shape[1]):
                    p = dict(params)
                    p["dsv41_hc_pre"] = pre[:, r:r + 1].clone()
                    self.rep("head.collapse:x", r, orig(x[:, r:r + 1].clone(), p, out_dtype).reshape(-1),
                             out[:, r].reshape(-1))
            self.replay("head.collapse", rows)
        self.patch(mod, "forward", make)

    # -- linears and norms --

    def _hook_linear(self, mod, op: str, kind: str = "row"):
        inner = getattr(mod, "inner", None)
        ent = self.linears.setdefault(f"{op} @ {mod.device}", {
            "layers": 0, "quant": set(), "K": set(), "mul1": set(), "in": mod.in_features, "out": mod.out_features})
        ent["layers"] += 1
        ent["quant"].add(str(getattr(mod, "quant_type", None)))
        ent["K"].add(getattr(inner, "K", None))
        ent["mul1"].add(bool(getattr(inner, "mul1", False)))
        eligible = type(inner).__name__ == "LinearEXL3" and bool(getattr(inner, "mul1", False))

        def make(orig):
            def forward(x, params, out_dtype = None):
                if not self.on():
                    return orig(x, params, out_dtype)
                y = orig(x, params, out_dtype)
                self.guard(op, after, orig, x, params, out_dtype, y)
                return y
            return forward

        def after(orig, x, params, out_dtype, y):
            x2, y2 = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
            n = x2.shape[0]
            units = self.units(kind, n, op)
            if units is None:
                return
            self.record(op, units, {"x": x2}, {"y": y2})
            if op == "engram.wkv":
                self.gpu[(self.layer, op)] = y          # the engram wrapper derives the gate from it

            def rows():
                for r in range(n):
                    self.rep(f"{op}:y", r, orig(_one_row(x, r), params, out_dtype).reshape(-1), y2[r])
            self.replay(f"{op} rows", rows)
            if not (self.int8_split and eligible):
                return

            def split():
                # Both sides through the FP16 kernels: what remains is the launch config of the row
                # bucket (and, on the FP16 GEMV, its instance). The first launch of a bucket may be
                # the autotuner's timing run, so one call of each is made and dropped first
                with _Int8Off():
                    key = (op, str(x.device), n)
                    if key not in self.settled:
                        self.settled.add(key)
                        orig(x.clone(), params, out_dtype)
                        orig(_one_row(x, 0), params, out_dtype)
                    ym = orig(x.clone(), params, out_dtype).reshape(n, -1)
                    for r in range(n):
                        self.rep(f"{op}:y [int8 gemv off]", r,
                                 orig(_one_row(x, r), params, out_dtype).reshape(-1), ym[r])
            self.replay(f"{op} int8-off", split)
            self.replay(f"{op} kernel tags", lambda: self._kernel_tags(mod, op, x2, y2))
        self.patch(mod, "forward", make)

    def _device_facts(self, device) -> dict:
        idx = device.index if device.index is not None else self.torch.cuda.current_device()
        return {"device": str(device), "device_index": idx, "cc": int(self.ext.g_get_cc(idx)),
                "num_sms": int(self.ext.g_get_num_sms(idx))}

    def _kernel_tags(self, mod, op: str, x2, y2):
        """Which kernel ext.exl3_gemm launches for this linear at 1..8 rows, with the int8 GEMV on and
        off: 0 = int8, 90 = FP16 GEMV, else the cooperative kernel's shape. Once per (op, device)."""
        torch, inner = self.torch, mod.inner
        key = f"{op} @ {x2.device}"
        if key in self.tags or x2.dtype != torch.half or x2.shape[-1] != inner.in_features:
            return
        fp32 = y2.dtype == torch.float
        ent = self.tags[key] = dict(self._device_facts(x2.device), K = inner.K, mul1 = bool(inner.mul1),
                                    mcg = bool(inner.mcg), fp32_out = fp32, groups = 0, on = {}, off = {})
        ent["in"], ent["out"] = inner.in_features, inner.out_features
        for name in ("on", "off"):
            for m in range(1, MAX_ROWS + 1):
                A = x2[:1].expand(m, -1).contiguous()
                C = torch.empty((m, inner.out_features), dtype = torch.float if fp32 else torch.half, device = A.device)
                if name == "off":
                    with _Int8Off():
                        tag = self.ext.exl3_gemm(A, inner.trellis, C, inner.suh, torch.empty_like(A), inner.svh,
                                                 -1, bool(inner.mcg), bool(inner.mul1), 0)
                else:
                    tag = self.ext.exl3_gemm(A, inner.trellis, C, inner.suh, torch.empty_like(A), inner.svh,
                                             -1, bool(inner.mcg), bool(inner.mul1), 0)
                ent[name][m] = int(tag)

    def _hook_norm(self, mod, op: str, kind: str = "row"):
        def make(orig):
            def forward(x, params, out_dtype = None, residual = None, residual_in = None):
                if not self.on() or residual is not None or residual_in is not None:
                    return orig(x, params, out_dtype, residual, residual_in)
                y = orig(x, params, out_dtype)
                self.guard(op, after, orig, x, params, out_dtype, y)
                return y
            return forward

        def after(orig, x, params, out_dtype, y):
            x2, y2 = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
            n = x2.shape[0]
            units = self.units(kind, n, op)
            if units is None:
                return
            self.record(op, units, {"x": x2}, {"y": y2})

            def rows():
                for r in range(n):
                    self.rep(f"{op}:y", r, orig(_one_row(x, r), params, out_dtype).reshape(-1), y2[r])
            self.replay(f"{op} rows", rows)
        self.patch(mod, "forward", make)

    # -- hyper-connections --

    def _hook_hc(self, hc, tag: str):
        def make_mix(orig):
            def mix_delayed(streams, params, carried_pre, own_pre = False):
                if not self.on():
                    return orig(streams, params, carried_pre, own_pre)
                res = orig(streams, params, carried_pre, own_pre)
                self.guard(f"{tag}.mix", after_mix, orig, streams, params, carried_pre, own_pre, res)
                return res
            return mix_delayed

        def after_mix(orig, streams, params, carried_pre, own_pre, res):
            post, comb, coll, pre = res
            b, s, H, D = streams.shape
            units = self.units("row", s, f"{tag}.mix")
            if units is None or b != 1:
                return
            self.record(f"{tag}.mix", units,
                        {"streams": streams[0], "carried_pre": None if carried_pre is None else carried_pre[0]},
                        {"post": post[0], "comb": comb[0], "collapsed": coll[0], "pre": pre[0]},
                        meta = {"rows": s, "chunks": 1 if self.stable else int(self.ext.hc_mix_num_chunks(s, H * D))})
            # post and comb are views of static workspaces that a one-row call may share
            saved_post, saved_comb = post.clone(), comb.clone()

            def rows():
                for r in range(s):
                    cr = None if carried_pre is None else carried_pre[:, r:r + 1].clone()
                    p1, c1, o1, e1 = orig(streams[:, r:r + 1].clone(), params, cr, own_pre)
                    self.rep(f"{tag}.mix:post", r, p1.reshape(-1), saved_post[:, r].reshape(-1))
                    self.rep(f"{tag}.mix:comb", r, c1.reshape(-1), saved_comb[:, r].reshape(-1))
                    self.rep(f"{tag}.mix:collapsed", r, o1.reshape(-1), coll[:, r].reshape(-1))
                    self.rep(f"{tag}.mix:pre", r, e1.reshape(-1), pre[:, r].reshape(-1))

            def restore():
                post.copy_(saved_post)
                comb.copy_(saved_comb)
            self.replay(f"{tag}.mix", rows, restore)

        def make_apply(orig):
            def apply_(x, y, post, comb, params):
                if not self.on():
                    return orig(x, y, post, comb, params)
                before = x.clone()              # the update is in place
                out = orig(x, y, post, comb, params)
                self.guard(f"{tag}.apply", after_apply, orig, before, y, post, comb, params, out)
                return out
            return apply_

        def after_apply(orig, before, y, post, comb, params, out):
            b, s = before.shape[:2]
            units = self.units("row", s, f"{tag}.apply")
            if units is None or b != 1:
                return
            self.record(f"{tag}.apply", units, {"x": before[0], "y": y[0], "post": post[0], "comb": comb[0]},
                        {"x": out[0]})

            def rows():
                for r in range(s):
                    o1 = orig(before[:, r:r + 1].clone(), y[:, r:r + 1].clone(), post[:, r:r + 1].clone(),
                              comb[:, r:r + 1].clone(), params)
                    self.rep(f"{tag}.apply:x", r, o1.reshape(-1), out[:, r].reshape(-1))
            self.replay(f"{tag}.apply", rows)
        self.patch(hc, "mix_delayed", make_mix)
        self.patch(hc, "apply_", make_apply)

    # -- engram --

    def _engram_gate(self, eng, h, key):
        """The gate of DSV41Engram.forward (modules/dsv41_engram.py), the same expressions on the same
        shapes: (rstd, dot, gate), rstd and dot None under the row-local kernel of the stable profile."""
        from exllamav3.modules import dsv41_engram as m_engram
        torch = self.torch
        eps = eng.config.rms_norm_eps
        if m_engram.STABLE_ARITHMETIC and h.is_cuda:
            return None, None, m_engram.stable_engram_gate(h, key, eng.qk, eps)
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * eng.qk * key).sum(-1) * rstd * h.shape[-1] ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return rstd, dot, gate

    def _hook_engram(self, eng):
        def make(orig):
            def forward(x, params, out_dtype = None):
                if not self.on():
                    return orig(x, params, out_dtype)
                self.gpu.pop((self.layer, "engram.wkv"), None)
                out = orig(x, params, out_dtype)
                self.guard("engram", after, x, out)
                return out
            return forward

        def after(x, out):
            torch = self.torch
            B, n, H, D = x.shape
            units = self.units("row", n, "engram.out")
            if units is None or B != 1:
                return
            kv = self.gpu.get((self.layer, "engram.wkv"))
            wkv = self.last.get((self.layer, "engram.wkv"))
            # the gate is inline torch code in the engine: it is recomputed here from the same
            # operands, and checked against the engine's output
            if kv is not None:
                key = kv[..., :H * D].view(B, n, H, D)
                value = kv[..., H * D:].view(B, n, 1, D)
                h = x.float()
                rstd, dot, gate = self._engram_gate(eng, h, key)
                same = diff_stats(torch, h + gate.unsqueeze(-1) * value, out)["equal"]
                self.record("engram.gate", units, {"streams": h[0], "key": key[0]},
                            {"rstd": None if rstd is None else rstd[0], "dot": None if dot is None else dot[0],
                             "gate": gate[0]}, meta = {"probe_gate_matches_engine_output": same})
            self.record("engram.out", units, {"streams": x[0], "wkv.y": None if wkv is None else wkv["y"]},
                        {"streams": out[0]})
            if kv is None:
                return

            def rows():
                for r in range(n):
                    r1, d1, g1 = self._engram_gate(eng, h[:, r:r + 1].clone(), key[:, r:r + 1].clone())
                    if r1 is not None:
                        self.rep("engram.gate:rstd", r, r1.reshape(-1), rstd[:, r].reshape(-1))
                        self.rep("engram.gate:dot", r, d1.reshape(-1), dot[:, r].reshape(-1))
                    self.rep("engram.gate:gate", r, g1.reshape(-1), gate[:, r].reshape(-1))
            self.replay("engram.gate", rows)
        self.patch(eng, "forward", make)

    # -- attention front, compressor, pool --

    def _hook_project(self, at):
        def make(orig):
            def _project_qkv(x, params, position):
                if not self.on():
                    return orig(x, params, position)
                res = orig(x, params, position)
                self.guard("attn.rope", after, orig, x, params, position, res)
                return res
            return _project_qkv

        def after(orig, x, params, position, res):
            _, q, kv = res
            bsz, seq = x.shape[:2]
            units = self.units("row", seq, "attn.rope")
            if units is None or bsz != 1:
                return
            wq_b, kv_norm = self.last.get((self.layer, "attn.wq_b")), self.last.get((self.layer, "attn.kv_norm"))
            self.record("attn.rope", units,
                        {"wq_b.y": None if wq_b is None else wq_b["y"],
                         "kv_norm.y": None if kv_norm is None else kv_norm["y"]},
                        {"q": q[0], "kv": kv[0]})

            def rows():
                for r in range(seq):
                    _, q1, kv1 = orig(x[:, r:r + 1].clone(), params, position + r)
                    self.rep("attn.project_qkv:q (wq_a .. rope)", r, q1.reshape(-1), q[:, r].reshape(-1))
                    self.rep("attn.project_qkv:kv (wkv .. rope)", r, kv1.reshape(-1), kv[:, r].reshape(-1))
            self.replay("attn.project_qkv", rows)
        self.patch(at, "_project_qkv", make)

    def _make_carry_step(self, orig):
        def step(carry, kv, score, pos0, m, norm_weight = None, eps = 1e-6):
            if not self.on():
                return orig(carry, kv, score, pos0, m, norm_weight, eps)
            prev, self.entry = self.entry, (pos0 // m, m)
            try:
                return orig(carry, kv, score, pos0, m, norm_weight, eps)
            finally:
                self.entry = prev
        return step

    def _make_group_pool(self, orig):
        def group_pool(kv, score):
            out = orig(kv, score)
            if self.on() and self.entry is not None:
                self.guard("comp.pool", after, kv, score, out)
            return out

        def after(kv, score, out):
            n = kv.shape[0]
            self.record("comp.pool", self.units("entry", n, "comp.pool"), {"kv": kv, "score": score}, {"pooled": out})

            def rows():
                for i in range(n):
                    self.rep("comp.pool:pooled", i, orig(kv[i:i + 1].clone(), score[i:i + 1].clone()).reshape(-1), out[i])
            self.replay("comp.pool", rows)
        return group_pool

    def _make_comp_norm(self, orig):
        def rms_norm(x, weight, eps):
            out = orig(x, weight, eps)
            if self.on() and self.entry is not None:
                self.guard("comp.norm", after, x, weight, eps, out)
            return out

        def after(x, weight, eps, out):
            n = x.shape[0]
            self.record("comp.norm", self.units("entry", n, "comp.norm"), {"x": x}, {"latent": out},
                        meta = {"rows": n})

            def rows():
                for i in range(n):
                    self.rep("comp.norm:latent", i, orig(x[i:i + 1].clone(), weight, eps).reshape(-1), out[i])
            self.replay("comp.norm", rows)
        return rms_norm

    def _make_fused_compress(self, orig):
        def fused_compress(*args, **kwargs):
            res = orig(*args, **kwargs)
            if self.on():
                self.guard("comp.fused", after, res)
            return res

        def after(res):
            latent, first = res
            self.record("comp.fused", [first + i for i in range(latent.shape[0])], {}, {"latent": latent})
        return fused_compress

    def _hook_store(self, at):
        from exllamav3.modules.dsv41_cached import entry_rows
        torch = self.torch

        def make(orig):
            def _store_entries(latent, e0, kl, bt_row, pos0, seq, params):
                if not self.on():
                    return orig(latent, e0, kl, bt_row, pos0, seq, params)
                prev, self.entry = self.entry, (e0, at.compress_ratio)
                try:
                    res = orig(latent, e0, kl, bt_row, pos0, seq, params)
                    self.guard("pool.store", after, latent, e0, kl, bt_row)
                finally:
                    self.entry = prev
                return res
            return _store_entries

        def after(latent, e0, kl, bt_row):
            n = latent.shape[0]
            outs = {}
            if kl.quant:
                self.note(f"pool.store L{self.layer}: quantized pool, the stored rows are not read back")
            else:
                rows = entry_rows(bt_row, torch.arange(e0, e0 + n, device = latent.device), kl.epp)
                outs["pool_c"] = kl.pool_c.view(-1, kl.D_c).index_select(0, rows)
                outs["pool_r"] = kl.pool_r.view(-1, kl.D_r).index_select(0, rows)
                if kl.pool_idx is not None:
                    outs["pool_idx"] = kl.pool_idx.view(-1, kl.D_i).index_select(0, rows)
            self.record("pool.store", self.units("entry", n, "pool.store"), {"latent": latent}, outs)
        self.patch(at, "_store_entries", make)

    # -- selection --

    def _make_scores(self, orig):
        torch = self.torch

        def dsa_indexer_scores(q_idx, weights, k_idx, q_pos0, compress_rate, bound_max, *args, **kwargs):
            out = orig(q_idx, weights, k_idx, q_pos0, compress_rate, bound_max, *args, **kwargs)
            if self.on() and self.layer is not None and not args:
                self.guard("idx.scores", after, q_idx, weights, k_idx, q_pos0, compress_rate, bound_max, kwargs, out)
            return out

        def after(q_idx, weights, k_idx, q_pos0, m, bound_max, kwargs, out):
            R = q_idx.shape[0]
            units = self.units("row", R, "idx.scores")
            if units is None:
                return
            vis = [max(0, min((q_pos0 + r + 1) // max(m, 1), bound_max)) for r in range(R)]
            # a row's scores up to its own causal bound; a second tile of the same row is appended
            self.record("idx.scores", units, {"q_idx": q_idx, "wts": weights}, {"scores": [out[r, :vis[r]] for r in range(R)]},
                        append = True,
                        meta = {"rows": R, "kernel": "few-query" if R <= 4 and kwargs.get("few_query", True) else "query-tiled"})
            backing = kwargs.get("scores")

            def rows():
                for r in range(R):
                    if vis[r] == 0:
                        continue
                    kw = dict(kwargs)
                    if backing is not None:
                        kw["scores"] = torch.empty((1, backing.shape[1]), dtype = backing.dtype, device = backing.device)
                    o1 = orig(q_idx[r:r + 1].clone(), weights[r:r + 1].clone(), k_idx, q_pos0 + r, m, vis[r], **kw)
                    self.rep("idx.scores:scores", r, o1[0, :vis[r]], out[r, :vis[r]])
            self.replay("idx.scores", rows)
        return dsa_indexer_scores

    def _make_select(self, orig):
        torch = self.torch

        def select_topk(q_idx, wts, idx_pool, **kwargs):
            res = orig(q_idx, wts, idx_pool, **kwargs)
            if self.on() and self.layer is not None:
                self.guard("idx.topk", after, q_idx, wts, idx_pool, kwargs, res)
            return res

        def after(q_idx, wts, idx_pool, kwargs, res):
            seq = q_idx.shape[0]
            units = self.units("row", seq, "idx.topk")
            if units is None:
                return
            cand_in = kwargs.get("cand_in")
            self.record("idx.topk", units, {"q_idx": q_idx, "wts": wts, "cand_in": cand_in},
                        {"indices": res.indices, "cand": res.cand}, ids = ("indices", "cand"),
                        meta = {"rows": seq, "ec": kwargs.get("ec"), "dense": res.indices is None})
            if res.indices is None:
                return
            pos0, m, ec = kwargs["pos0"], kwargs["m"], kwargs["ec"]

            def rows():
                for r in range(seq):
                    kw = dict(kwargs)
                    kw["pos0"] = pos0 + r
                    kw["ec"] = min(ec, (pos0 + r + 1) // m)         # what a one-row call at this position passes
                    if cand_in is not None:
                        kw["cand_in"] = cand_in[r:r + 1].clone()
                    r1 = orig(q_idx[r:r + 1].clone(), wts[r:r + 1].clone(), idx_pool, **kw)
                    if r1.indices is None:
                        # one row is still dense here (every visible entry, ascending), the K-row call
                        # selected: equal when it selected exactly those entries
                        dense = torch.arange(kw["ec"], dtype = res.indices.dtype, device = res.indices.device)
                        picked = res.indices[r]
                        self.rep("idx.topk:indices", r, dense, picked[picked >= 0], ids = True,
                                 note = "a one-row call at some of these positions is dense (ec <= topk)")
                        continue
                    self.rep("idx.topk:indices", r, r1.indices[0], res.indices[r], ids = True)
                    if res.cand is not None and r1.cand is not None:
                        self.rep("idx.topk:cand", r, r1.cand[0], res.cand[r], ids = True)
            self.replay("idx.topk", rows)
        return select_topk

    # -- attention core, output projection --

    def _make_attn(self, orig):
        torch = self.torch

        def dsa_attn(q, pool_c, pool_r, block_table, **kwargs):
            out = orig(q, pool_c, pool_r, block_table, **kwargs)
            if self.on() and self.layer is not None:
                self.guard("attn.dsa_attn", after, q, pool_c, pool_r, block_table, kwargs, out)
            return out

        def after(q, pool_c, pool_r, block_table, kwargs, out):
            R = q.shape[0]
            units = self.units("row", R, "attn.dsa_attn")
            if units is None or out.dim() != 3 or out.shape[1] != R:
                return
            indices, kvc = kwargs.get("indices"), kwargs.get("kv_chunk")
            win, rate = kwargs.get("win_len", 0), max(kwargs.get("compress_rate", 1), 1)
            pool_len, p0 = kwargs.get("pool_len", 0), kwargs.get("q_pos0", 0)
            # the split count dsa_attn resolves for a call of up to 8 rows (dsa_triton.py, n_splits == 0)
            est = (win if kvc is not None and win > 0 else 0) + \
                  (kwargs.get("k_len", 0) if indices is not None else min(pool_len, (p0 + R) // rate))
            n_splits = kwargs.get("n_splits", 0) or ((16 if est > 256 else 8) if R <= 8 else 1)
            self.record("attn.dsa_attn", units, {"q": q, "kv_chunk": kvc, "indices": indices},
                        {"out": out.permute(1, 0, 2)},
                        meta = {"rows": R, "est": est, "n_splits": n_splits, "pool": "gathered" if indices is not None
                                else ("dense" if pool_len else "none")})
            ring, beg = kwargs.get("ring"), kwargs.get("ring_beg", 0)
            off = p0 - beg
            if ring is None or kvc is None or off < 0 or off + R > ring.shape[0]:
                return

            def rows():
                # A one-row step at position p reads the earlier rows of this call from the ring
                # (the engine appends them after attention): a private ring copy gets them
                tmp = ring.clone()
                for r in range(R):
                    if r:
                        tmp[off + r - 1].copy_(kvc[r - 1])
                    p = p0 + r
                    kw = dict(kwargs)
                    kw.update(ring = tmp, kv_chunk = kvc[r:r + 1].clone(), q_pos0 = p,
                              win_floor = p - min(win - 1, p - beg, p),
                              out = torch.empty((out.shape[0], 1, out.shape[2]), dtype = out.dtype, device = out.device))
                    if indices is not None:
                        kw["indices"] = indices[r:r + 1].clone()
                    else:
                        kw["pool_len"] = min(pool_len, (p + 1) // rate)
                    o1 = orig(q[r:r + 1].clone(), pool_c, pool_r, block_table, **kw)
                    self.rep("attn.dsa_attn:out", r, o1[:, 0].reshape(-1), out[:, r].reshape(-1))
            self.replay("attn.dsa_attn", rows)
        return dsa_attn

    def _hook_woa(self, at):
        def make(orig):
            def _project_o_grouped(o, params, out_dtype, mgemm_out = False):
                if not self.on():
                    return orig(o, params, out_dtype, mgemm_out)
                self.in_woa = True
                try:
                    return orig(o, params, out_dtype, mgemm_out)
                finally:
                    self.in_woa = False
            return _project_o_grouped
        self.patch(at, "_project_o_grouped", make)

    def _make_mgemm(self, orig):
        torch = self.torch

        def exl3_mgemm(*args, **kwargs):
            tag = orig(*args, **kwargs)
            if self.on() and self.in_woa and len(args) >= 12:
                self.guard("attn.wo_a", after, args, kwargs, tag)
            return tag

        def after(args, kwargs, tag):
            A, C = args[0], args[2]             # (groups, rows, k), (groups, rows, n)
            G, seq, k = A.shape
            units = self.units("row", seq, "attn.wo_a")
            if units is None:
                return
            self.record("attn.wo_a", units, {"A": A.permute(1, 0, 2)}, {"C": C.permute(1, 0, 2)},
                        meta = {"rows": seq, "tag": None if tag is None else int(tag)})
            key = f"attn.wo_a @ {A.device}"
            ent = self.tags.get(key)
            if ent is None:
                K = args[8]
                ent = self.tags[key] = dict(self._device_facts(A.device), K = K, mul1 = bool(args[11]),
                                            mcg = bool(args[10]), fp32_out = C.dtype == torch.float, groups = G, on = {})
                ent["in"], ent["out"] = k, C.shape[2]
            if tag is not None:
                ent["on"][seq] = int(tag)

            def rows():
                for r in range(seq):
                    a = list(args)
                    a[0] = A[:, r:r + 1].clone()
                    a[2] = torch.empty((G, 1, C.shape[2]), dtype = C.dtype, device = C.device)
                    a[4] = torch.empty_like(a[0])
                    t1 = orig(*a, **kwargs)
                    if t1 is not None:
                        ent["on"][1] = int(t1)
                    self.rep("attn.wo_a:C", r, a[2][:, 0].reshape(-1), C[:, r].reshape(-1))
            self.replay("attn.wo_a", rows)
        return exl3_mgemm

    # -- MoE --

    def _make_routing(self, orig):
        def routing_ds3_nogroup(*args, **kwargs):
            res = orig(*args, **kwargs)
            stage = self.moe_stage
            if stage == "real" and len(args) >= 6:
                # the layer's own routing call (a prefetch prediction of the expert cache may follow)
                self.moe_stage = "done"
                try:
                    n = args[0].shape[0]
                    self.route = {"z": args[0].clone(), "logits": args[2].reshape(n, -1).clone(),
                                  "sel": args[4].reshape(n, -1).clone(), "w": args[5].reshape(n, -1).clone()}
                except Exception as e:
                    self._error(f"capture moe.router L{self.layer}: {type(e).__name__}: {e}")
            elif stage == "forced" and len(args) >= 6:
                self.moe_stage = "done"
                sel, w = self.force
                args[4].copy_(sel.reshape(args[4].shape))
                args[5].copy_(w.reshape(args[5].shape))
            return res
        return routing_ds3_nogroup

    def _hook_moe(self, mlp):
        torch = self.torch

        def make(orig):
            def forward(x, params, out_dtype = None):
                if not self.on():
                    return orig(x, params, out_dtype)
                self.route, self.moe_stage = None, "real"
                try:
                    out = orig(x, params, out_dtype)
                finally:
                    self.moe_stage = None
                self.guard("moe", after, orig, x, params, out_dtype, out)
                return out
            return forward

        def after(orig, x, params, out_dtype, out):
            y = x.reshape(-1, x.shape[-1])
            n = y.shape[0]
            units = self.units("row", n, "moe.out")
            if units is None:
                return
            route = self.route
            ins = {"y": y}
            if route is not None and route["z"].shape[0] == n:
                self.record("moe.router", units, {"z": route["z"]},
                            {"logits": route["logits"], "sel": route["sel"], "w": route["w"]}, ids = ("sel",))
                ins.update(sel = route["sel"], w = route["w"])
            else:
                route = None
                self.note(f"moe.router L{self.layer}: routing did not go through ext.routing_ds3_nogroup; not captured")
            self.record("moe.out", units, ins, {"y": out.reshape(n, -1)})
            if self.mode != "replay":
                return
            # the routed sum is a view of a buffer every MoE layer of the device shares: a one-row
            # call overwrites its first row, so the K-row result is kept and written back
            saved = out.clone()
            rows_out = saved.reshape(n, -1)

            def restore():
                self.moe_stage = self.force = None
                out.copy_(saved)
            if route is None:
                return

            def router():
                cfg = mlp.routing_cfg
                for r in range(n):
                    sel1, w1 = mlp.routing_fn(1, cfg, route["z"][r:r + 1].clone(), params)
                    self.rep("moe.router:logits", r, cfg.router_logits_bsz1.reshape(-1), route["logits"][r])
                    self.rep("moe.router:sel", r, sel1.reshape(-1), route["sel"][r], ids = True)
                    self.rep("moe.router:w", r, w1.reshape(-1), route["w"][r])
            self.replay("moe.router", router)

            def forced():
                # the experts with the K-row call's own selection and weights, one row at a time
                for r in range(n):
                    self.force, self.moe_stage = (route["sel"][r:r + 1], route["w"][r:r + 1]), "forced"
                    o1 = orig(_one_row(x, r), params, out_dtype)
                    self.rep("moe.out:y [routing forced to the K-row call's]", r, o1.reshape(-1), rows_out[r])
            self.replay("moe.forced", forced, restore)

            def shared():
                # every routing weight 0: what remains is the shared expert's term
                zero = torch.zeros_like(route["w"])
                self.force, self.moe_stage = (route["sel"], zero), "forced"
                many = orig(x.clone(), params, out_dtype).reshape(n, -1).clone()
                for r in range(n):
                    self.force, self.moe_stage = (route["sel"][r:r + 1], zero[r:r + 1]), "forced"
                    o1 = orig(_one_row(x, r), params, out_dtype)
                    self.rep("moe.out:y [routing weights 0: the shared expert]", r, o1.reshape(-1), many[r])
            self.replay("moe.shared", shared, restore)
        self.patch(mlp, "forward", make)


# ---------------------------------------------------------------------------------
# Comparison of two captured runs
# ---------------------------------------------------------------------------------

def _run_keys(*runs):
    keys = set()
    for r in runs:
        keys |= set(r.rec)
    return sorted(keys, key = lambda k: (k[1] if k[1] is not None else -1, _op_index(k[0]), k[0]))


def same_runs(torch, x: Run, y: Run, subset: bool = False) -> dict:
    """Two runs of the SAME arm must agree on every captured tensor: {'equal', 'tensors', 'first'}.
    With subset, y ran fewer steps than x: every tensor of y must be in x and equal."""
    first, total, bad = None, 0, 0
    for key in _run_keys(y) if subset else _run_keys(x, y):
        ex, ey = x.rec.get(key), y.rec.get(key)
        ux, uy = (ex["units"] if ex else {}), (ey["units"] if ey else {})
        for u in sorted(uy if subset else set(ux) | set(uy)):
            for side in ("in", "out"):
                names = set(uy[u][side]) if subset else \
                    set(ux.get(u, {}).get(side, {})) | set(uy.get(u, {}).get(side, {}))
                for name in sorted(names):
                    total += 1
                    a, b = ux.get(u, {}).get(side, {}).get(name), uy.get(u, {}).get(side, {}).get(name)
                    if a is not None and b is not None and diff_stats(torch, a, b)["equal"]:
                        continue
                    bad += 1
                    if first is None:
                        first = {"op": key[0], "layer": key[1], "unit": u, "side": side, "name": name,
                                 "missing": a is None or b is None}
    return {"equal": bad == 0, "tensors": total, "differing": bad, "first": first}


def compare_arms(torch, a: Run, b: Run, probe: Probe) -> dict:
    """
    One-row arm a against K-row arm b. Returns the report rows (one per captured output, in
    execution order), the first divergence of the forward, the origins and the dispatch facts
    that differ. An output is an ORIGIN at a unit when it differs there while every input of its
    operation is bit-equal, the state it reads included: for attention the window rows and pool
    entries this call wrote, for the indexer the index keys this call wrote.
    """
    P, layers = probe.P, probe.layers
    keys = _run_keys(a, b)
    eq = {}
    for key in keys:
        ea, eb = a.rec.get(key), b.rec.get(key)
        ids = (eb or ea)["ids"]
        ua, ub = (ea["units"] if ea else {}), (eb["units"] if eb else {})
        d = eq[key] = {}
        for u in sorted(set(ua) | set(ub)):
            if u not in ua or u not in ub:
                d[u] = {"missing": "one-row" if u not in ua else "k-row", "names": list((ua.get(u) or ub.get(u))["out"])}
                continue
            ent = {"in": {}, "out": {}}
            for side in ("in", "out"):
                for name, tb in ub[u][side].items():
                    ta = ua[u][side].get(name)
                    ent[side][name] = {"equal": False, "ndiff": -1, "numel": int(tb.numel())} if ta is None else \
                        diff_stats(torch, ta, tb, side == "out" and name in ids)
            d[u] = ent

    def same(key, u, name) -> bool:
        st = eq.get(key, {}).get(u)
        if st is None or "missing" in st or name not in st["out"]:
            return True                 # not captured: no evidence of a difference
        return st["out"][name]["equal"]

    def state_equal(op, L, u) -> bool:
        info = layers.get(L)
        if info is None:
            return True
        ok = True
        if op == "attn.dsa_attn":
            ok = all(same(("attn.rope", L), q, "kv") for q in range(P, u))
        m = info["compress_ratio"]
        if m and op in ("attn.dsa_attn", "idx.scores", "idx.topk"):
            names = ("pool_c", "pool_r") if op == "attn.dsa_attn" else ("pool_idx",)
            src = ("pool.store", info["kv_source"])
            ok = ok and all(same(src, e, nm) for e in range(P // m, (u + 1) // m) for nm in names)
        return ok

    rows, first, origins = {}, None, []
    for key in keys:
        op, L = key
        kind = "entry" if op in ENTRY_OPS else "row"
        for u, ent in eq[key].items():
            names = ent["names"] if "missing" in ent else list(ent["out"])
            in_eq = "missing" not in ent and all(s["equal"] for s in ent["in"].values()) and state_equal(op, L, u)
            for name in names:
                label = f"{op}:{name}"
                row = rows.get(label)
                if row is None:
                    row = rows[label] = {"op": op, "out": name, "label": label, "kind": kind, "layers": 0, "_layers": set(),
                                         "units": 0, "units_diff": 0, "in_diff": 0, "origin": 0, "propagated": 0,
                                         "absorbed": 0, "ndiff": 0, "numel": 0, "max_abs": 0.0, "max_rel": 0.0,
                                         "set_diff": 0, "only_one_row": 0, "only_k_row": 0, "first_diff": None,
                                         "first_origin": None, "equal": True}
                row["_layers"].add(L)
                row["units"] += 1
                where = {"layer": L, "unit": u}
                if "missing" in ent:
                    row["only_k_row" if ent["missing"] == "one-row" else "only_one_row"] += 1
                    row["equal"] = False
                    row["units_diff"] += 1
                    if row["first_diff"] is None:
                        row["first_diff"] = where
                    differs, origin = True, False
                else:
                    st = ent["out"][name]
                    row["numel"] += st["numel"]
                    if not in_eq:
                        row["in_diff"] += 1
                    differs = not st["equal"]
                    origin = differs and in_eq and bool(ent["in"])
                    if not differs:
                        if not in_eq:
                            row["absorbed"] += 1
                    else:
                        row["equal"] = False
                        row["units_diff"] += 1
                        row["ndiff"] += max(st["ndiff"], 0)
                        for k in ("max_abs", "max_rel"):
                            if st.get(k) is not None:
                                row[k] = max(row[k], st[k])
                        if st.get("set_equal") is False:
                            row["set_diff"] += 1
                        if row["first_diff"] is None:
                            row["first_diff"] = where
                        if origin:
                            row["origin"] += 1
                            if row["first_origin"] is None:
                                row["first_origin"] = where
                        else:
                            row["propagated"] += 1
                if differs and first is None:
                    first = {"label": label, "layer": L, "unit": u, "kind": kind, "origin": origin,
                             "missing": ent.get("missing")}
                if origin:
                    if origins and origins[-1]["label"] == label and origins[-1]["layer"] == L:
                        origins[-1]["units"].append(u)
                    else:
                        origins.append({"label": label, "layer": L, "kind": kind, "units": [u]})
    for row in rows.values():
        row["layers"] = len(row.pop("_layers"))

    # call facts that differ between the arms, the same difference on several layers folded
    folded = {}
    for key in keys:
        if key not in a.rec or key not in b.rec:
            continue
        ma, mb = a.rec[key]["meta"], b.rec[key]["meta"]
        for field in sorted({f for m in ma + mb for f in m} - META_SKIP):
            va = sorted({repr(m.get(field)) for m in ma})
            vb = sorted({repr(m.get(field)) for m in mb})
            if va != vb:
                folded.setdefault((key[0], field, tuple(va), tuple(vb)), []).append(key[1])
    dispatch = [{"op": op, "field": field, "one_row": list(va), "k_row": list(vb), "layers": ls}
                for (op, field, va, vb), ls in folded.items()]
    ordered = sorted(rows.values(), key = lambda row: _op_index(row["op"]))      # stable: outputs keep their order
    return {"rows": ordered, "first": first, "origins": origins, "dispatch": dispatch}


def expected_units(probe: Probe, forwards) -> dict:
    """
    {(op, layer): units} the wrappers should capture over the forwards [(pos0, rows), ...] of one
    arm, from the layer facts, as the cached path runs them (modules/dsv41.py, _forward_cached_row
    and _compress_store; modules/dsv41_block.py, DSV41Block.forward): the block operations on
    every row, the engram's on its layers, a kv source's projections on every row and its pooling,
    index keys and store on the entries [pos0 // m, (pos0 + rows) // m) the forward closes, and an
    index source's selection on every row of a forward whose pool passes index_topk entries.
    The ablations of EXL3_DSV41_ABLATE are not modelled.
    """
    exp = {}

    def add(op, L, units):
        if units:
            exp.setdefault((op, L), set()).update(units)

    for pos0, rows in forwards:
        row_units = range(pos0, pos0 + rows)
        for L, info in probe.layers.items():
            for op in BLOCK_OPS:
                add(op, L, row_units)
            if info["engram"]:
                for op in ("engram.wkv", "engram.gate", "engram.out"):
                    add(op, L, row_units)
            m = info["compress_ratio"] or 0
            if m and info["is_kv_source"]:
                add("comp.wkv", L, row_units)
                if info["comp_gate"]:
                    add("comp.wgate", L, row_units)
                entries = range(pos0 // m, (pos0 + rows) // m)
                if m > 1 and probe.fused_compress:
                    add("comp.fused", L, entries)
                else:
                    if m > 1:
                        add("comp.pool", L, entries)
                    add("comp.norm", L, entries)
                if info["owns_index_k"]:
                    add("idx.wk", L, entries)
                    add("idx.k_norm", L, entries)
                add("pool.store", L, entries)
            if m and info["is_index_source"] and info["index_topk"] is not None \
                    and (pos0 + rows) // m > info["index_topk"]:
                for op in ("idx.wq_b", "idx.weights_proj", "idx.scores", "idx.topk"):
                    add(op, L, row_units)
        for op in HEAD_OPS:
            add(op, probe.n_layers, row_units)
    return exp


def coverage(probe: Probe, arms) -> dict:
    """
    arms: [(name, run, forwards)]. What expected_units calls for and the run did not capture, the
    same gap on several layers folded: {'complete', 'missing': [{op, arm, layers, units}]}. The
    grouped output projection counts as captured through ext.exl3_mgemm (attn.wo_a) or through
    its per-group linears (attn.wo_a.g<n>), whichever the engine took.
    """
    folded = {}
    for name, run, forwards in arms:
        for (op, L), units in expected_units(probe, forwards).items():
            got = set()
            for (o, l), ent in run.rec.items():
                if l == L and (o == op or (op == "attn.wo_a" and o.startswith("attn.wo_a."))):
                    got |= set(ent["units"])
            gap = len(units - got)
            if gap:
                ent = folded.setdefault((op, name), {"op": op, "arm": name, "layers": [], "units": 0})
                ent["layers"].append(L)
                ent["units"] += gap
    missing = sorted(folded.values(), key = lambda e: (_op_index(e["op"]), e["op"], e["arm"]))
    return {"complete": not missing, "missing": missing}


def logits_rows(torch, la, lb, P: int) -> list:
    """Per row: bit equality of the logits, both argmaxes and the gap between the two best logits."""
    out = []
    for r in range(lb.shape[0]):
        st = diff_stats(torch, la[r], lb[r])
        ta, tb = torch.topk(la[r].float(), 2), torch.topk(lb[r].float(), 2)
        out.append({"row": r, "position": P + r, "equal": st["equal"], "ndiff": st["ndiff"], "max_abs": st.get("max_abs"),
                    "argmax_one_row": int(ta.indices[0]), "argmax_k_row": int(tb.indices[0]),
                    "argmax_equal": int(ta.indices[0]) == int(tb.indices[0]),
                    "top2_gap_one_row": float(ta.values[0] - ta.values[1]),
                    "top2_gap_k_row": float(tb.values[0] - tb.values[1])})
    return out


# ---------------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------------

def _num(v, spec: str = ".3g") -> str:
    return "-" if v is None else format(v, spec)


def _layer_name(L, n_layers: int) -> str:
    return "head" if L == n_layers else f"L{L}"


def _where(w, P: int, kind: str, n_layers: int) -> str:
    if w is None:
        return "-"
    unit = f"e{w['unit']}" if kind == "entry" else f"r{w['unit'] - P}"
    return f"{_layer_name(w['layer'], n_layers)} {unit}"


def _ranges(values) -> str:
    out, values = [], sorted(set(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(str(values[i]) if i == j else f"{values[i]}-{values[j]}")
        i = j + 1
    return ",".join(out)


def print_insitu(cmp: dict, K: int, P: int, n_layers: int):
    print(f"\n== K = {K}: {K} one-row steps against one {K}-row forward, positions {P}..{P + K - 1} (in-situ)")
    print(f"   {'operation:output':<27} {'':<4} {'first diff':<11} {'units diff':>11} {'elements diff':>19} "
          f"{'max abs':>9} {'max rel':>9} {'origin':>7} {'first origin':<12} {'in-diff':>7} {'sets':>5}")
    for row in cmp["rows"]:
        if row["equal"]:
            print(f"   {row['label']:<27} =    {'':<11} {'0/' + str(row['units']):>11}")
            continue
        extra = ""
        if row["only_k_row"] or row["only_one_row"]:
            extra = f"  only in the K-row arm: {row['only_k_row']}, only in the one-row arm: {row['only_one_row']}"
        print(f"   {row['label']:<27} DIFF {_where(row['first_diff'], P, row['kind'], n_layers):<11} "
              f"{str(row['units_diff']) + '/' + str(row['units']):>11} "
              f"{str(row['ndiff']) + '/' + str(row['numel']):>19} {_num(row['max_abs']):>9} {_num(row['max_rel']):>9} "
              f"{row['origin']:>7} {_where(row['first_origin'], P, row['kind'], n_layers):<12} "
              f"{row['in_diff']:>7} {row['set_diff'] if row['op'] in ('idx.topk', 'moe.router') else '':>5}{extra}")
    f = cmp["first"]
    if f is None:
        print(f"   FIRST: none. Every captured output is bit-equal at K = {K}")
    else:
        why = "ORIGIN: its inputs are bit-equal" if f["origin"] else \
            (f"the {f['missing']} arm has no such output there" if f["missing"] else "its inputs already differ")
        print(f"   FIRST diverging operation of the forward: {f['label']} at "
              f"{_where(f, P, f['kind'], n_layers)} ({why})")
    if cmp["origins"]:
        labels = {}
        for o in cmp["origins"]:
            labels.setdefault(o["label"], []).append(o["layer"])
        print("   origins (inputs bit-equal, outputs not), in execution order:")
        for label, ls in labels.items():
            names = [_layer_name(L, n_layers) for L in sorted(set(ls))]
            print(f"     {label:<27} first {names[0]:<5} on {len(names)} layer(s): "
                  f"{' '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
    for d in cmp["dispatch"]:
        print(f"   dispatch: {d['op']} {d['field']}: one row {'/'.join(d['one_row'])}, {K} rows {'/'.join(d['k_row'])} "
              f"(layers {_ranges(d['layers'])})")


def print_logits(rows: list):
    print("   logits   row  bits   differing  max abs   argmax one-row / K-row   top-2 gap one-row / K-row")
    for r in rows:
        print(f"            r{r['row']:<3} {'=' if r['equal'] else 'DIFF':<6} {max(r['ndiff'], 0):>9}  {_num(r['max_abs']):>8}  "
              f"{r['argmax_one_row']:>7} / {r['argmax_k_row']:<7} {'' if r['argmax_equal'] else 'FLIP':<5} "
              f"{r['top2_gap_one_row']:.4g} / {r['top2_gap_k_row']:.4g}")


def print_replay(rep: dict, K: int, n_layers: int):
    print(f"   replay: each operation one row at a time on the {K}-row call's own inputs, against its {K}-row result")
    print(f"   {'operation:output':<52} {'':<4} {'first diff':<11} {'rows diff':>11} {'elements diff':>19} "
          f"{'max abs':>9} {'max rel':>9} {'layers'}")
    for label in sorted(rep, key = lambda s: (_op_index(s.split(":")[0]), s)):
        d = rep[label]
        if not d["rows_diff"]:
            print(f"   {label:<52} =    {'':<11} {'0/' + str(d['rows']):>11}")
            continue
        w = d["first"]
        print(f"   {label:<52} DIFF {_layer_name(w['layer'], n_layers) + ' r' + str(w['row']):<11} "
              f"{str(d['rows_diff']) + '/' + str(d['rows']):>11} {str(d['ndiff']) + '/' + str(d['numel']):>19} "
              f"{_num(d['max_abs']):>9} {_num(d['max_rel']):>9} "
              f"{_ranges(d['layers_diff'])}{'  sets differ: ' + str(d['set_diff']) if d['set_diff'] else ''}"
              f"{'  (' + d['note'] + ')' if d['note'] else ''}")


def print_table(title: str, table: dict, ks: list):
    print(f"\n{title}")
    width = max([len(s) for s in table] + [16])
    col = max([6] + [len(cell) for row in table.values() for cell in row.values()])
    print(f"   {'operation:output':<{width}} " + " ".join(f"{'K=' + str(k):>{col}}" for k in ks))
    for label in sorted(table, key = lambda s: (_op_index(s.split(":")[0]), s)):
        print(f"   {label:<{width}} " + " ".join(f"{table[label].get(k, '.'):>{col}}" for k in ks))


def insitu_cell(row: dict, n_layers: int) -> str:
    """Cell of the in-situ operation x K table: '=', or the first differing layer; '*' when the
    operation is an origin AT that layer, '/o<layer>' when its first origin is at another one."""
    if row["equal"]:
        return "="
    cell = _layer_name(row["first_diff"]["layer"], n_layers)
    origin = row["first_origin"]
    if origin is None:
        return cell
    if origin["layer"] == row["first_diff"]["layer"]:
        return cell + "*"
    return f"{cell}/o{_layer_name(origin['layer'], n_layers).lstrip('L')}"


def print_kernel_tags(tags: dict, records: dict, version: int) -> dict:
    """The kernel each EXL3 linear launches at 1..8 rows, and its autotune records per row bucket."""
    out = {}
    if not tags:
        return out
    print("\nkernel of each EXL3 linear by row count (0 = int8 GEMV, 90 = FP16 GEMV, 1..4 = cooperative kernel shape)"
          "\nand its launch-autotune records (kernel shape, blocks, concurrency) per row bucket; 'none' where the "
          "cache holds no record under the key this tool recomputes:")
    for key in sorted(tags, key = lambda s: (_op_index(s.split(" @ ")[0]), s)):
        ent = tags[key]
        buckets = (1, 2, 3, 5) if ent["groups"] else (1, 3, 5)
        names = {1: "1" if ent["groups"] else "1-2", 2: "2", 3: "3-4", 5: "5-8"}
        recs = {}
        for m in buckets:
            rec = records.get(autotune_key(ent, m, version))
            recs[names[m]] = None if rec is None else {"tag": rec[0], "num_sms": rec[2], "concurrency": rec[3]}
        distinct = {repr(v) for v in recs.values() if v is not None}
        on =" ".join(str(ent["on"].get(m, ".")) for m in range(1, MAX_ROWS + 1))
        off = " ".join(str(ent.get("off", {}).get(m, ".")) for m in range(1, MAX_ROWS + 1))
        shown = "  ".join(f"{n}: " + ("none" if r is None else f"({r['tag']}, {r['num_sms']}, {r['concurrency']})")
                          for n, r in recs.items())
        print(f"   {key:<28} K={ent['K']} {ent['in']}->{ent['out']} {'fp32' if ent['fp32_out'] else 'fp16'}"
              f"{' x' + str(ent['groups']) + ' grouped' if ent['groups'] else ''}  rows 1..8: {on}"
              + (f"  | int8 off: {off}" if ent.get("off") else "")
              + f"  | autotune {shown}{'  <- buckets differ' if len(distinct) > 1 else ''}")
        out[key] = dict(ent, autotune = recs, buckets_differ = len(distinct) > 1)
    return out


# ---------------------------------------------------------------------------------
# Token source, driver
# ---------------------------------------------------------------------------------

def reference_tokens(val, args) -> tuple[list, str]:
    rc = val.load_refcompare()
    cases = rc.load_ref(args.ref)
    for c in cases:
        if c.name == args.case:
            return [int(t) for t in c.prompt_ids.tolist()], f"{args.ref}:{c.name}"
    raise ValueError(f"no case {args.case!r} in {args.ref} (it has: {', '.join(c.name for c in cases)})")


class Driver:
    """The job state, the generator's params, and the two arms."""

    def __init__(self, torch, val, model, cache, ids: list, position: int):
        self.torch, self.val, self.model, self.cache, self.P = torch, val, model, cache, position
        self.ids = torch.tensor([ids], dtype = torch.long)
        self.state = cache.get_new_state()
        # the slot's pages, as a job's block table lists them (dsv41_validate.run_prefill_check)
        pages = cache.max_num_tokens // PAGE_SIZE // cache.num_slots
        self.block_table = torch.arange(self.state.slot * pages, (self.state.slot + 1) * pages,
                                        dtype = torch.int32)[None, :]
        self.probe = None

    def params(self, history: bool) -> dict:
        """What Generator.iterate_gen passes to model.forward (generator/generator.py): history is
        'recurrent_history', True for a verify forward of 1+N rows and False for a decode step."""
        return {
            "attn_mode": "flash_attn",
            "block_table": self.block_table,
            "cache": self.cache,
            "cache_seqlens": self.torch.tensor([self.state.position], dtype = self.torch.int32),
            "recurrent_states": [self.state],
            "indexed_embeddings": [],
            "positions": None,
            "recurrent_history": history,
            "pinned_staging": True,
        }

    def prefill(self, chunk: int):
        """The prefix, as a job prefills it (generator/job.py: model.prefill in chunks)."""
        torch, a = self.torch, 0
        while a < self.P:
            n = min(chunk, self.P - a)
            self.model.prefill(self.ids[:, a:a + n], {
                "attn_mode": "flash_attn", "block_table": self.block_table, "cache": self.cache,
                "cache_seqlens": torch.tensor([a], dtype = torch.int32),
                "recurrent_states": [self.state], "indexed_embeddings": []})
            a += n
        self.val._sync(torch)
        if self.state.position != self.P:
            raise RuntimeError(f"prefill left the state at {self.state.position}, expected {self.P}")

    def _logits(self, logits, rows: int):
        logits = logits[0] if logits.dim() == 3 else logits
        if logits.shape[0] != rows:
            raise RuntimeError(f"forward returned {logits.shape[0]} logit rows, expected {rows}")
        # the copy is also the sync point the generator's sampling gives between two forwards
        return logits.detach().to("cpu")

    def one_row(self, K: int, run: Run | None = None):
        """Arm (a): K decode steps. Returns the (K, vocab) logits."""
        out = []
        for r in range(K):
            if run is not None:
                self.probe.begin(run, self.P + r, 1)
            out.append(self._logits(self.model.forward(self.ids[:, self.P + r: self.P + r + 1], self.params(False)), 1))
        logits = self.torch.cat(out, dim = 0)
        if run is not None:
            run.logits = logits
        return logits

    def multi(self, K: int, run: Run | None = None):
        """Arm (b): one verify forward of K rows. Returns the (K, vocab) logits."""
        if run is not None:
            self.probe.begin(run, self.P, K)
        logits = self._logits(self.model.forward(self.ids[:, self.P: self.P + K], self.params(True)), K)
        if run is not None:
            run.logits = logits
        return logits

    def rewind(self, K: int):
        """Back to the prefix, by the engine's own rewind (the generator's reject path)."""
        capacity = self.state.rollback_capacity()
        if capacity < K:
            raise RuntimeError(f"the job state can rewind {capacity} tokens at position {self.state.position}, "
                               f"{K} needed; choose another --prefix or --prefill-chunk")
        self.state.rewind(K)
        if self.state.position != self.P:
            raise RuntimeError(f"rewind left the state at {self.state.position}, expected {self.P}")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], stderr = subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_gpu(val, args, policy: dict, ks: list, out: str) -> int:
    report = {"tool": "dsv41_rowprobe", "status": "running", "ok": False, "args": vars(args), "arithmetic": policy}
    val._write(out, _jsonable(report))
    env = {}
    if args.visible is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = args.visible
    if args.placement is not None:
        env["EXL3_PLACEMENT"] = args.placement
    tune, tune_before = prepare_tune_cache(args.tune_cache, out)
    try:
        return _run_gpu(val, args, ks, out, report, env, tune, tune_before)
    finally:
        if tune["private"]:
            # the private copy: what this run added to it is in the report
            try:
                os.unlink(tune["used_path"])
            except OSError:
                pass


def _run_gpu(val, args, ks: list, out: str, report: dict, env: dict, tune: dict, tune_before: dict) -> int:
    if tune["env"]:
        env["EXLLAMAV3_TUNE_CACHE"] = tune["env"]
    int8_pinned = not args.no_int8_split and "EXL3_INT8_GEMV" not in os.environ
    if int8_pinned:
        # Before any thread exists: the int8 split flips this variable in-process later, and
        # replacing a value leaves the environment array alone where adding or removing one
        # would move it under a concurrent getenv (see _Int8Off)
        env["EXL3_INT8_GEMV"] = INT8_GEMV_DEFAULT
    if "torch" in sys.modules and "CUDA_VISIBLE_DEVICES" in env:
        print("WARNING torch was imported before the device environment was applied")
    os.environ.update(env)
    val.start_ram_watchdog(args.min_avail_gib)
    import torch
    from exllamav3 import Cache, Config, Model
    from exllamav3.ext import exllamav3_ext as ext

    shown = {k: os.environ[k] for k in tuple(val.REPORT_ENV) + ("EXLLAMAV3_TUNE_CACHE",) if k in os.environ}
    print(f"environment: {shown}", flush = True)
    if int8_pinned:
        print(f"  EXL3_INT8_GEMV was not set: set to {INT8_GEMV_DEFAULT}, the value the extension takes when it "
              f"is unset, so that the int8 split only ever replaces it", flush = True)
    print(f"autotune cache: {tune['mode']}: {tune['used_path']}"
          f"{' (created as a copy of the live file)' if tune['seeded_from_live'] else ''}, "
          f"{tune['records_before']} records; live file {tune['live_path']}, {tune['live_records']} records",
          flush = True)
    if tune["mode"] == "live":
        print("  the live file is in use: a row bucket it has no record for is tuned by this run and WRITTEN to it",
              flush = True)
    if tune["carried_over"]:
        print(f"  {len(tune['carried_over'])} record(s) of this cache are not in the live file (tuned by earlier "
              f"probe runs on it): " + " ".join(sorted(tune["carried_over"])[:16])
              + (" ..." if len(tune["carried_over"]) > 16 else ""), flush = True)

    P, kmax = args.prefix, max(ks)
    t0 = time.time()
    cfg = Config.from_directory(args.model)
    val.apply_settings(cfg, args)
    if args.text is not None or args.text_file is not None:
        from exllamav3 import Tokenizer
        text = args.text
        if text is None:
            with open(args.text_file, encoding = "utf-8") as f:
                text = f.read()
        ids = Tokenizer.from_config(cfg).encode(text, add_bos = True, encode_special_tokens = True)[0].tolist()
        source = "--text" if args.text is not None else args.text_file
    else:
        ids, source = reference_tokens(val, args)
    if len(ids) < P + kmax:
        raise ValueError(f"the token source has {len(ids)} tokens, --prefix {P} plus {kmax} rows need {P + kmax}")
    ids = ids[:P + kmax]

    chunk = args.max_chunk_size or 2048
    cache_tokens = args.cache_tokens or max(65536, -(-(P + kmax + 1) // PAGE_SIZE) * PAGE_SIZE)
    if cache_tokens % PAGE_SIZE or cache_tokens < P + kmax + 1:
        raise ValueError(f"--cache-tokens {cache_tokens} must be a multiple of {PAGE_SIZE} and hold {P + kmax + 1} tokens")
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens = cache_tokens, max_batch_size = 1)
    load_kw = {"progressbar": False, "max_chunk_size": chunk, "max_output_size": 32}
    if args.gpu_split:
        load_kw["use_per_device"] = val._floats(args.gpu_split)
    if args.reserve:
        load_kw["reserve_per_device"] = val._floats(args.reserve)
    model.load(**load_kw)
    print(f"LOADED in {time.time() - t0:.0f}s", flush = True)
    devs = {}
    for m in model.modules:
        devs.setdefault(str(getattr(m, "device", None)), []).append(m.key)
    for d, keys in devs.items():
        print(f"  {d}: {len(keys)} modules  {keys[0]} .. {keys[-1]}", flush = True)
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)} sm_{''.join(map(str, torch.cuda.get_device_capability(i)))}"
              f" allocated {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB", flush = True)

    extension = getattr(ext, "__file__", None)
    report.update({
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": shown, "tune_cache": tune,
        "numerics": str(getattr(cfg, "dsv41_numerics", None)),
        "provenance": {"source_commit": _git_commit(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                       "native_extension": None if not extension else
                       {"file": os.path.basename(extension), "sha256": _sha256(extension)},
                       "devices": [{"name": torch.cuda.get_device_name(i),
                                    "capability": list(torch.cuda.get_device_capability(i))}
                                   for i in range(torch.cuda.device_count())]},
        "placement": {d: [keys[0], keys[-1], len(keys)] for d, keys in devs.items()},
        "tokens": {"source": source, "prefix": P, "probed_ids": ids[P:]},
        "restore": "engine rewind (DSV41State.rewind) on one job state, verified against a reference taken before "
                   "any K-row forward",
    })

    with torch.inference_mode():
        ok = _probe(torch, ext, val, args, model, cache, ids, ks, min(args.prefill_chunk or 2048, chunk), report,
                    tune, tune_before)
    report["ok"] = ok
    report["status"] = "completed"
    val._write(out, _jsonable(report))
    print(f"\nreport: {out}")
    return 0 if ok else 1


def _probe(torch, ext, val, args, model, cache, ids, ks, prefill_chunk, report, tune, tune_before) -> bool:
    P, kmax = args.prefix, max(ks)
    drv = Driver(torch, val, model, cache, ids, P)
    t0 = time.time()
    drv.prefill(prefill_chunk)
    st = drv.state
    window_beg = st.window_beg
    print(f"prefilled {P} tokens in {time.time() - t0:.1f}s (chunks of {prefill_chunk}); window_beg "
          f"{window_beg}, rollback capacity {st.rollback_capacity()}", flush = True)
    report["state"] = {"position": P, "window_beg": window_beg, "rollback_capacity": st.rollback_capacity()}
    eq = lambda a, b: diff_stats(torch, a, b)["equal"]
    stable = report["arithmetic"]["stable_arithmetic"]

    # The pristine reference: the one-row arm on the state as prefilled, before this process has
    # run any K-row verify forward. Twice unhooked (the first use of a launch bucket may be the
    # autotuner's timing run, so the second is the reference), then once captured, through
    # wrappers that are removed again. Every later one-row run must equal it
    first = drv.one_row(kmax)
    drv.rewind(kmax)
    ref = drv.one_row(kmax)
    drv.rewind(kmax)
    probe0 = Probe(torch, ext, model, P, replay_int8 = False, stable = stable)
    drv.probe = probe0
    probe0.install()
    try:
        probe0.mode = "capture"
        A0 = Run("one-row, before any K-row forward")
        drv.one_row(kmax, A0)
        drv.rewind(kmax)
    finally:
        probe0.uninstall()
        drv.probe = None
    pristine = {"first_use_equals_second": eq(first, ref), "captured_equals_unhooked": eq(A0.logits, ref)}
    del first
    report["pristine"] = pristine
    print(f"pristine reference: {kmax} one-row steps before any K-row forward; captured run equals the unhooked "
          f"one: {'yes' if pristine['captured_equals_unhooked'] else 'NO'}", flush = True)
    if not pristine["first_use_equals_second"]:
        print("   note: the FIRST one-row run of this process differs from the second, so a first use (the launch "
              "autotuner timing a bucket, a kernel compile) changes one-row results; the second run is the reference",
              flush = True)
    ok = pristine["captured_equals_unhooked"]

    # Unhooked K-row runs, on the engine as it is: a warm-up (the launch autotuner times a row
    # bucket on its first use), then one that directly follows a K-row forward and its rewind,
    # for the check that a K-row forward does not depend on which arm ran before it
    after_k = {}
    for K in ks:
        drv.multi(K)
        drv.rewind(K)
        after_k[K] = drv.multi(K)
        drv.rewind(K)
    # then both arms alternating, --timing-reps times: determinism and the verify cost
    reps = max(2, args.timing_reps)
    plain, det, timing = {}, {}, {}
    for K in ks:
        ta, tb = [], []
        det[K] = {"one_row": True, "k_row": True, "runs": reps}
        for i in range(reps):
            t = time.perf_counter()
            la = drv.one_row(K)
            ta.append(time.perf_counter() - t)
            drv.rewind(K)
            t = time.perf_counter()
            lb = drv.multi(K)
            tb.append(time.perf_counter() - t)
            drv.rewind(K)
            if i == 0:
                plain[K] = (la, lb)
            else:
                det[K]["one_row"] &= eq(la, plain[K][0])
                det[K]["k_row"] &= eq(lb, plain[K][1])
        ma, mb = statistics.median(ta), statistics.median(tb)
        timing[K] = {"reps": reps,
                     "one_row_steps_ms": {"median": round(1e3 * ma, 2), "min": round(1e3 * min(ta), 2)},
                     "k_row_forward_ms": {"median": round(1e3 * mb, 2), "min": round(1e3 * min(tb), 2)},
                     "k_row_over_one_step": {"median": round(mb / (ma / K), 2),
                                             "min": round(min(tb) / (min(ta) / K), 2)}}
    print(f"timing, unhooked, {reps} repetitions, median [minimum]: K one-row steps, one K-row forward, and that "
          f"forward in one-row steps (wall clock around forward and logits copy; other load on the host shows)")
    for K in ks:
        t = timing[K]
        print(f"   K={K}: {t['one_row_steps_ms']['median']:.1f} [{t['one_row_steps_ms']['min']:.1f}] ms"
              f" / {t['k_row_forward_ms']['median']:.1f} [{t['k_row_forward_ms']['min']:.1f}] ms"
              f"   {t['k_row_over_one_step']['median']:.2f}x [{t['k_row_over_one_step']['min']:.2f}x]", flush = True)
    report["timing"] = timing

    probe = Probe(torch, ext, model, P, replay_int8 = not args.no_int8_split, stable = stable)
    drv.probe = probe
    probe.install()
    print(f"wrapped {len(probe.patches)} methods and calls", flush = True)
    n_layers = probe.n_layers
    report["layers"] = probe.layers
    report["linears"] = probe.linears
    report["k"], report["table"] = {}, {"insitu": {}, "replay": {}}
    covered = True
    try:
        for K in ks:
            la0, lb0 = plain[K]
            checks = {"deterministic_unhooked": det[K]}
            probe.replays, probe.notes = {}, []
            probe.mode = "capture"
            A, B = Run("one-row"), Run(f"{K}-row")
            drv.one_row(K, A)
            drv.rewind(K)
            drv.multi(K, B)
            drv.rewind(K)
            checks["hooks_change_nothing"] = {"one_row": eq(A.logits, la0), "k_row": eq(B.logits, lb0)}
            # every one-row run so far followed K-row forwards; the reference followed none
            pr = {"unhooked": eq(la0, ref[:K]), "captured": eq(A.logits, ref[:K]),
                  "tensors": same_runs(torch, A0, A, subset = True)}
            checks["pristine_equals_post_multi"] = pr
            checks["k_row_after_k_row_equals_after_one_row"] = eq(after_k[K], lb0)
            if not args.no_repeat:
                A2 = Run("one-row, repeated")
                drv.one_row(K, A2)
                drv.rewind(K)
                checks["repeat_control"] = same_runs(torch, A, A2)
                pr["repeat"] = eq(A2.logits, ref[:K])
                del A2
            if not args.no_replay:
                probe.mode = "replay"
                B2 = Run(f"{K}-row, with replays")
                drv.multi(K, B2)
                drv.rewind(K)
                checks["replays_change_nothing"] = dict(same_runs(torch, B, B2), logits = eq(B2.logits, lb0))
                del B2
            probe.mode = None
            checks["coverage"] = coverage(probe, [("one-row", A, [(P + r, 1) for r in range(K)]),
                                                  (f"{K}-row", B, [(P, K)])])
            cmp = compare_arms(torch, A, B, probe)
            lrows = logits_rows(torch, A.logits, B.logits, P)
            print_insitu(cmp, K, P, n_layers)
            print_logits(lrows)
            if probe.replays:
                print_replay(probe.replays, K, n_layers)
            for n in probe.notes:
                print(f"   note: {n}")
            good = _print_checks(checks, K)
            ok &= good
            covered &= checks["coverage"]["complete"]
            for row in cmp["rows"]:
                report["table"]["insitu"].setdefault(row["label"], {})[K] = insitu_cell(row, n_layers)
            for label, d in probe.replays.items():
                report["table"]["replay"].setdefault(label, {})[K] = \
                    "=" if not d["rows_diff"] else _layer_name(d["first"]["layer"], n_layers)
            report["k"][K] = {"insitu": cmp["rows"], "first_divergence": cmp["first"], "origins": cmp["origins"][:200],
                              "dispatch": cmp["dispatch"], "logits": lrows, "replay": probe.replays,
                              "checks": checks, "notes": probe.notes, "checks_ok": good}
            sys.stdout.flush()
            del A, B, cmp
            gc.collect()
    finally:
        probe.uninstall()
        drv.probe = None
    # the engine as it was: one more unhooked run of the widest K must still give the first logits
    # and the pristine ones
    K = ks[-1]
    la = drv.one_row(K)
    drv.rewind(K)
    after = {"equals_first_unhooked": eq(la, plain[K][0]), "equals_pristine": eq(la, ref[:K])}
    ok &= all(after.values())
    report["unhooked_after"] = after
    report["state"]["window_beg_after"] = st.window_beg
    cache.release_state(drv.state)
    del A0

    print_table("in-situ: operation x K -> '=' or the first differing layer ('L3*': an origin at that layer, "
                "'L3/o20': propagated there, first an origin at layer 20; '.': not captured)",
                report["table"]["insitu"], ks)
    if report["table"]["replay"]:
        print_table("replay: operation x K -> '=' when one row at a time gives the K-row call's bits on the same inputs, "
                    "else the first differing layer", report["table"]["replay"], ks)
    records = read_tune_cache(tune["used_path"])
    report["kernel_tags"] = print_kernel_tags(probe.tags, records, _autotune_version())
    tune["records_after"] = len(records)
    added = {f"{k:016x}": list(v) for k, v in records.items() if tune_before.get(k) != v}
    tune["added_by_this_run"] = added
    if added:
        where = {"copy": "its private cache", "live": "the LIVE cache"}.get(tune["mode"], "the cache of --tune-cache")
        print(f"   this run added {len(added)} autotune record(s) to {where}: launches it held no record for "
              f"(the replays with the int8 GEMV off among them): " + " ".join(sorted(added)[:16])
              + (" ..." if len(added) > 16 else ""))
    report["errors"] = probe0.errors + [e for e in probe.errors if e not in probe0.errors]
    for e in report["errors"]:
        print(f"ERROR in the probe's own code (the forward was not affected): {e}")
    if st.window_beg != window_beg:
        print(f"\nnote: the window ring shifted during the probe (window_beg {window_beg} -> {st.window_beg}); a "
              f"rewind keeps the shifted ring, so later runs read the same rows at another ring offset")
    print(f"\nunhooked run after the probe equals the first unhooked run: "
          f"{'yes' if after['equals_first_unhooked'] else 'NO'}, the pristine reference: "
          f"{'yes' if after['equals_pristine'] else 'NO'}")
    print("self-checks: " + ("all passed" if ok else "FAILED (see the K sections): a difference reported above "
                                                     "may then not be due to the row count"))
    print("coverage: " + ("every expected operation was captured on every row and pool entry" if covered else
                          "INCOMPLETE (see the NOT CAPTURED lines): the tables miss those operations there. The "
                          "expected set follows the layer facts (expected_units); check it against the engine "
                          "before trusting either"))
    report["coverage_complete"] = covered
    return ok and covered and not report["errors"]


def _print_checks(checks: dict, K: int) -> bool:
    """Print the self-checks of one K; True when the validity checks passed. Coverage is printed
    here and counted by the caller: a gap makes the map incomplete, not a reported difference wrong."""
    yn = lambda v: "yes" if v else "NO"
    det, hook = checks["deterministic_unhooked"], checks["hooks_change_nothing"]
    pr, pred = checks["pristine_equals_post_multi"], checks["k_row_after_k_row_equals_after_one_row"]
    good = det["one_row"] and det["k_row"] and all(hook.values()) and pred
    line = [f"deterministic unhooked ({det['runs']} runs): one-row {yn(det['one_row'])}, K-row {yn(det['k_row'])}",
            f"hooked logits equal unhooked: one-row {yn(hook['one_row'])}, K-row {yn(hook['k_row'])}",
            f"K-row forward after a K-row forward equals one after one-row steps: {yn(pred)}"]

    def tensors(c) -> tuple[bool, str]:
        passed = c["equal"] and c.get("logits", True)
        where = ""
        if not passed and c["first"] is not None:
            f = c["first"]
            where = f" (first: {f['op']} L{f['layer']} unit {f['unit']} {f['side']} {f['name']}; {c['differing']} of {c['tensors']})"
        return passed, f"{yn(passed)}{where}"

    passed, text = tensors(pr["tensors"])
    logits = [f"{name} {yn(pr[name])}" for name in ("unhooked", "captured", "repeat") if name in pr]
    good &= passed and all(pr[name] for name in ("unhooked", "captured", "repeat") if name in pr)
    line.append(f"one-row arm equals the pristine reference (taken before any K-row forward): logits "
                f"{', '.join(logits)}; every captured tensor {text}")
    for name, label in (("repeat_control", "repeat of the one-row arm equal on every tensor"),
                        ("replays_change_nothing", "replay run equal to the capture run on every tensor")):
        c = checks.get(name)
        if c is None:
            continue
        passed, text = tensors(c)
        good &= passed
        line.append(f"{label}: {text}")
    print(f"   self-checks K = {K}: " + "; ".join(line))
    for miss in checks["coverage"]["missing"]:
        where = "the head" if miss["op"] in HEAD_OPS else f"layers {_ranges(miss['layers'])}"
        print(f"   NOT CAPTURED: {miss['op']} on {where} in the {miss['arm']} arm ({miss['units']} units)")
    return good


# ---------------------------------------------------------------------------------

def main(argv = None) -> int:
    val = _load_validate()
    args = build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        policy = math_policy(val)
    except ValueError as error:
        print(f"math policy: {error}")
        return 2
    print_policy(policy)
    if args.stable_check:
        return 0                        # it only shows the switches, in whatever environment
    if policy["stable_arithmetic"] and not args.allow_stable:
        print("refused: EXL3_STABLE_ARITHMETIC=1 is set. This probe measures the default arithmetic; pass "
              "--allow-stable to probe the profile (every operation should then come out equal)")
        return 2
    if not args.model:
        print("no model: pass --model or set DSV41_MODEL_DIR")
        return 2
    sources = sum(v is not None for v in (args.text, args.text_file))
    if sources > 1 or (not sources and not args.ref):
        print("one token source is needed: --ref (or DSV41_VLLM_REF) with --case, or --text, or --text-file")
        return 2
    try:
        ks = sorted({int(v) for v in args.rows.split(",") if v.strip()})
    except ValueError:
        ks = []
    if not ks or ks[0] < 2 or ks[-1] > MAX_ROWS:
        print(f"--rows needs row counts from 2 to {MAX_ROWS}: wider forwards leave the decode kernels "
              f"(block_sparse_mlp.MAX_BSZN) and are prefill, not a verify")
        return 2
    for name in ("prefix", "max_chunk_size", "cache_tokens", "prefill_chunk"):
        value = getattr(args, name)
        if value is not None and value < (0 if name == "prefix" else 1):
            print(f"--{name.replace('_', '-')} must be {'nonnegative' if name == 'prefix' else 'positive'}")
            return 2
    if args.timing_reps < 2:
        print("--timing-reps must be at least 2: the determinism check compares the repetitions")
        return 2
    if args.tune_cache not in ("copy", "live") and not os.path.isdir(os.path.dirname(os.path.abspath(args.tune_cache))):
        print(f"--tune-cache {args.tune_cache}: no such directory")
        return 2
    errors = val.engine_setting_errors(args)
    if errors:
        for e in errors:
            print(f"ERROR {e}")
        return 2
    out = args.out or os.path.join(DEFAULT_OUT_DIR, "rowprobe.json")
    try:
        return run_gpu(val, args, policy, ks, out)
    except Exception as error:
        import traceback
        traceback.print_exc()
        val._write(out, _jsonable({"tool": "dsv41_rowprobe", "status": "failed", "ok": False, "args": vars(args),
                                   "error": f"{type(error).__name__}: {error}"}))
        print(f"run failed: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
