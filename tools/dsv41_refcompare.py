"""
Offline reference comparison for the DeepSeek-V4.1-Flash port.

Scores exllamav3 logprobs against a vLLM prompt-logprob capture (the file
tools/dsv41_capture_ref.py writes; DSV41_VLLM_REF or --ref names it), and exllamav3
against itself (stateless vs cached vs decode). Pure python + numpy: no torch and no
CUDA, so tools/dsv41_validate.py can load this file by path for a --dry-run and the
unit tests run on any host. See doc/dsv41_tools.md.

Alignment. For every prompt position i >= 1 the reference holds the logprobs vLLM
gave token i from tokens [0, i): the top-K ids plus the actual token, which is an
extra (K+1-th) entry when it ranks below K. Our logits[i-1] predict the same token,
so row j of every per-position array here is prompt position i = j + 1. The prompts
carry no BOS; they must be fed exactly as captured.

Why statistics and not digests. vLLM's logits are bf16 and its prefill is not
bitwise reproducible: two captures of the same 300-token prompt differ by p90 0.216
nats on the actual token, their top-1 disagrees at 6 of 299 positions and their
greedy continuations diverge. Exact top-1 ties are common at bf16 resolution. Every
gate below is set against that measured self-noise, and tests/test_dsv41_refcompare_.py
recomputes the noise from the file so the calibration cannot drift silently. The
calibration covers n <= 300 only: at n >= 1500 the calibrated limits fail vLLM against
itself and nearly every recorded run of the port (see GATES). So each limit is also
shifted by the reference's own noise at the case's length, measured on the other
captures of the same file (reference_noise, attach_noise): a check beyond its
calibrated limit but within its noise-shifted limit is reported as a FAIL within the
reference's own noise, and only a check beyond both fails the run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

# Half-open ranges of prompt positions (the token being predicted). Each one is where
# a different part of V4.1's attention first carries weight, so a failure localizes.
REGIONS: tuple[tuple[int, int], ...] = (
    (1, 128), (128, 512), (512, 1025), (1025, 1500), (1500, 16384), (16384, 20000),
    (20000, 65536), (65536, 131072), (131072, 262144), (262144, 524288), (524288, 1048576),
)
REGION_NOTES = {
    (1, 128): "SWA window, pools in fill mode, engram, mHC",
    (128, 512): "compressed pools in fill mode (every entry selected)",
    (512, 1025): "rate-1 top-k active (> 512 entries)",
    (1025, 1500): "rate-2 top-k active (T/2 > 512 entries)",
    (1500, 16384): "top-k on every compressed layer",
    (16384, 20000): "candidate stage (T > 2048 blocks x 8)",
}

# Gates: the calibrated limits. Calibrated on vLLM rep0 vs rep1 of the calibration capture at
# n = 24 and 300 only (see MEASURED_NOISE): there the thresholds sit above that noise with room
# for the systematic differences we expect (fp16 KV here, and fp16 indexer operands under the
# precise numerics setting, vs fp8 KV / MXFP4 indexer / bf16 streams there). They were never
# calibrated at n >= 1500, and there they fail vLLM against itself: its prefix noise
# (prefix_noise, the same prompt inside a longer request) fails them at n = 20000 vs 1500 in
# region [1,128) (|dNLL| 0.052, |dlp| p99 1.24), as it does at n = 300 vs 24 on |dNLL| through
# position 1 alone. The recorded real-model runs failed them too: every nc and cached run at
# n = 1500 in region [1,128) (|dlp| p99 1.06-1.45; mostly also p90 0.43-0.48 and top-1 at
# margin 111/115, one flip short of 0.97; only the decode runs passed), every run at n = 20000
# on |dlp| max (5.64-6.12). That is why a report with attach_noise() also carries
# noise-shifted limits, each limit moved by as much as the capture shows vLLM differing from
# itself on the same positions beyond CALIBRATION_NOISE (see "The reference's own noise"
# below and doc/dsv41_tools.md, "Gates"). The calibrated limits themselves are constants.
GATES = {
    "dnll": 0.03,                   # |mean NLL(ours) - mean NLL(ref)|, nats
    "top1": 0.97,                   # top-1 agreement ...
    "top1_margin": 0.375,           # ... on positions whose ref top1-top2 margin exceeds this
    "dlp_p50": 0.05,                # |logprob(actual token) difference| quantiles
    "dlp_p90": 0.4,
    "dlp_p99": 1.0,
    "dlp_max": 5.0,                 # catches localized corruption hidden by quantiles
    "overlap": 0.90,                # mean |ours top-5 & ref top-5| / 5
    "region_dnll": 0.05,            # |dNLL| inside each region ...
    # The candidate's variance must never enlarge its own acceptance threshold.
    # dnll_se remains a diagnostic, not evidence of equivalence or independent noise.
    "region_min_positions": 8,      # regions with fewer positions are reported, not gated
}
# A failure confined to regions starting here is 'precision-suspect': the index top-k starts
# to select (drop entries) at 512, so the indexer's rounding matters only from here. vLLM
# rounds the index operands to MXFP4, as the default deepseek:index does here too, and its KV
# to FP8 at every position, which the port does only when its numerics setting names window
# and compressed (config.dsv41_numerics). Re-run with vLLM's contract (--dsv41-numerics vllm)
# before calling it a bug. The label needs no check below 512 to fail beyond its
# noise-shifted limit, so it cannot apply while region [1,128) fails that way.
PRECISION_FROM = 512
# exllamav3 vs itself (stateless vs cached vs teacher-forced decode). The boundary gate
# applies the same 0.02 to the first BOUNDARY_WINDOW predictions after every chunk start,
# at p90 so that one routing-flip outlier cannot fail it while a lookback, ring or
# carry bug (which hits most of those positions) cannot pass it.
SELF_GATES = {"dlp_p99": 0.02, "tie_eps": 0.0, "boundary_dlp_p90": 0.02, "boundary_min_positions": 8,
              "overlap": 0.90, "dlp_max": 5.0,
              # with a measured floor (self_gates(floor = ...)): limits relative to the noise that changing
              # only the forward's row count makes on the same model and case
              "floor_mult": 1.5, "floor_sigmas": 3.0, "boundary_vs_rest": 1.5}
BOUNDARY_WINDOW = 4

# vLLM's run-to-run noise on the calibration capture (two reps each of n = 24 and 300,
# plus one rep of n = 1500 and 20000 sharing their prefix; rep1 scored against rep0),
# recomputed by the unit tests from that file. The n=300 row is what the gates are
# calibrated on (CALIBRATION_NOISE); the single reps of 1500 and 20000 give no rep-to-rep
# noise, only prefix noise (prefix_noise). These figures belong to the calibration capture;
# another capture's own noise enters through reference_noise(), never here. Top-5 overlap is
# 0.9485 by vLLM's own ranks; the
# 0.949 recorded at planning time sorted each dict by value, which at position 166
# promoted the rank-6 actual token over a 5th-ranked token with the identical logprob.
# top1_margin: the top-1 agreement where the reference's top1-top2 margin exceeds
# GATES["top1_margin"] (every one of vLLM's 6 flips at n = 300 lies at a smaller margin).
MEASURED_NOISE = {
    24: {"dlp_mean": 0.016, "dlp_p90": 0.066, "dlp_max": 0.088, "top1": (23, 23)},
    300: {"dlp_p50": 0.011, "dlp_p90": 0.216, "dlp_p99": 0.556, "dlp_max": 0.744,
          "dlp_mean": 0.066, "top1": (293, 299), "top1_margin": (268, 268), "overlap": 0.9485,
          "dnll": 0.0089},
}


@dataclass(frozen = True)
class Ablation:
    token: str
    env: dict
    owner: str              # the module that implements it
    affects_from: int       # first prompt position the ablation can change
    isolated: bool          # positions below affects_from must stay within the gates
    what: str
    at_chunk_starts: bool = False   # acts only at a forward's start after position 0 (n/a without one)


# Each ablation removes one V4.1 feature. A run with it MUST fail the gates in the
# regions it affects; if it passes, the harness is blind to that feature.
ABLATIONS = {a.token: a for a in (
    Ablation("engram", {"EXL3_DSV41_ABLATE": "engram"}, "engram", 1, False,
             "the engram layers (1 and 14) do not run"),
    Ablation("engram_nocarry", {"EXL3_DSV41_ABLATE": "engram_nocarry"}, "engram", 1, False,
             "drop cached n-gram history at chunk starts (requires cached/decode/self mode)",
             at_chunk_starts = True),
    Ablation("v4mix", {"EXL3_DSV41_ABLATE": "v4mix"}, "block", 1, False,
             "V4's immediate hyper-connection mix instead of V4.1's delayed pre-mix"),
    Ablation("dense_consumers", {"EXL3_DSV41_ABLATE": "dense_consumers"}, "attention", 512, True,
             "consumers attend their whole pool instead of their index source's top-k"),
    Ablation("rope_consecutive", {"EXL3_DSV41_ABLATE": "rope_consecutive"}, "attention", 128, False,
             "rate-2 pool entries roped at consecutive positions instead of (first + i) * m"),
    Ablation("no_candidates", {"EXL3_DSV41_ABLATE": "no_candidates"}, "select", 16384, True,
             "no candidate-block stage above layer 20"),
)}


# ---------------------------------------------------------------------------------
# Reference file
# ---------------------------------------------------------------------------------

@dataclass
class Case:
    """One captured prompt. Row j of the ref_* arrays is prompt position j + 1."""
    n: int
    rep: int
    prompt_ids: np.ndarray                  # int64 [n]
    ref_ids: np.ndarray                     # int64 [n-1, K], by descending logprob; -1 = missing
    ref_lp: np.ndarray                      # float64 [n-1, K]; -inf = missing
    ref_actual_lp: np.ndarray               # float64 [n-1]; nan = missing
    top: int = 5
    gen_text: str = ""
    gen_tokens: list = field(default_factory = list)
    gen_top_logprobs: list = field(default_factory = list)
    seconds: float = float("nan")
    meta: dict = field(default_factory = dict)

    @property
    def name(self) -> str:
        return f"n{self.n}r{self.rep}"

    @property
    def actual_ids(self) -> np.ndarray:
        return self.prompt_ids[1:]

    @property
    def margin(self) -> np.ndarray:
        """Reference top-1 minus top-2 logprob per row (inf if only one entry)."""
        with np.errstate(invalid = "ignore"):
            m = self.ref_lp[:, 0] - self.ref_lp[:, 1]
        m[self.ref_ids[:, 0] < 0] = np.nan
        return m


def _entries(p: dict, actual: int, K: int) -> tuple[list[tuple[int, float]], float]:
    """(top-K (id, logprob) by descending logprob, logprob of the actual token)."""
    items = []
    for k, v in p.items():
        try:
            t = int(k)
        except (TypeError, ValueError):
            continue                        # text-keyed entries cannot be aligned by id
        if isinstance(v, dict):
            v = v.get("logprob")
        if v is None:
            continue
        v = float(v)
        if math.isnan(v):
            continue
        items.append((t, v))
    act = next((v for t, v in items if t == actual), math.nan)
    # vLLM appends the actual token when it ranks below K; it is not part of the top-K
    if len(items) > K and any(t == actual for t, _ in items):
        items = [(t, v) for t, v in items if t != actual]
    items.sort(key = lambda tv: -tv[1])     # stable: dict (= vLLM rank) order breaks ties
    return items[:K], act


def _reference_integer(value, name: str, minimum: int = 0) -> int:
    """JSON integer fields must not silently truncate floats or accept booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"reference {name} must be an integer >= {minimum}, not {value!r}")
    return int(value)


def parse_case(raw: dict, top: int = 5, meta: dict | None = None) -> Case:
    if not isinstance(raw, dict) or (meta is not None and not isinstance(meta, dict)):
        raise ValueError("reference case and metadata must be objects")
    top = _reference_integer(top, "top", 5)
    raw_ids = raw.get("prompt_ids")
    if not isinstance(raw_ids, (list, tuple, np.ndarray)) \
            or (isinstance(raw_ids, np.ndarray) and raw_ids.ndim != 1):
        raise ValueError("reference prompt_ids must be a one-dimensional integer sequence")
    n = _reference_integer(raw.get("n", len(raw_ids)), "n", 2)
    rep = _reference_integer(raw.get("rep", 0), "rep")
    if n != len(raw_ids):
        raise ValueError(f"case n={n} but {len(raw_ids)} prompt ids")
    # Validate before NumPy conversion: even an otherwise integer list can hide
    # a boolean, and a float/string conversion can change the prompt being tested.
    max_token_id = np.iinfo(np.int64).max
    for token in raw_ids:
        token = _reference_integer(token, "prompt token ID")
        if token > max_token_id:
            raise ValueError("reference prompt token ID exceeds signed int64")
    ids = np.asarray(raw_ids, dtype = np.int64)
    if "contended" in raw and not isinstance(raw["contended"], (bool, np.bool_)):
        raise ValueError("reference contended must be a boolean when recorded")
    probe_errors = _reference_integer(raw.get("probe_errors", 0), "probe_errors")
    missing_isolation = [key for key in ("contended", "probe_errors") if key not in raw]
    pl = raw["prompt_logprobs"]
    if not isinstance(pl, (list, tuple)) or any(p is not None and not isinstance(p, dict) for p in pl):
        raise ValueError("reference prompt_logprobs must be an array of objects or nulls")
    if len(pl) != n:
        raise ValueError(f"case n={n}: {len(pl)} prompt_logprobs entries, expected {n}")
    ref_ids = np.full((n - 1, top), -1, dtype = np.int64)
    ref_lp = np.full((n - 1, top), -np.inf, dtype = np.float64)
    act = np.full(n - 1, np.nan, dtype = np.float64)
    for i in range(1, n):
        p = pl[i]
        if not p:
            continue
        ent, a = _entries(p, int(ids[i]), top)
        act[i - 1] = a
        for j, (t, v) in enumerate(ent):
            ref_ids[i - 1, j] = t
            ref_lp[i - 1, j] = v
    return Case(n = n, rep = rep, prompt_ids = ids, ref_ids = ref_ids,
                ref_lp = ref_lp, ref_actual_lp = act, top = top,
                gen_text = raw.get("gen_text") or "", gen_tokens = list(raw.get("gen_tokens") or []),
                gen_top_logprobs = list(raw.get("gen_top_logprobs") or []),
                seconds = float(raw.get("seconds", math.nan)),
                meta = {**dict(meta or {}), "contended": bool(raw.get("contended", False)),
                        "probe_errors": probe_errors, "isolation_metadata_missing": missing_isolation})


def load_ref(path: str, top: int | None = None) -> list[Case]:
    """Every case of a capture file, in file order."""
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict) or not isinstance(d.get("meta", {}), dict) \
            or not isinstance(d.get("cases"), list) or not d["cases"]:
        raise ValueError("reference requires object metadata and a nonempty cases array")
    meta = d.get("meta", {})
    K = _reference_integer(top if top is not None else meta.get("top", 5), "top", 5)
    return [parse_case(c, K, meta) for c in d["cases"]]


def isolation_missing(case: Case) -> list[str]:
    """The isolation fields the capture did not record for this case ('contended', 'probe_errors'):
    without them nothing shows that vLLM served the prompt alone."""
    return list(case.meta.get("isolation_metadata_missing",
                              [key for key in ("contended", "probe_errors") if key not in case.meta]))


def select(cases: list[Case], ns = None, reps = None) -> list[Case]:
    return [c for c in cases if (ns is None or c.n in ns) and (reps is None or c.rep in reps)]


def as_ours(case: Case, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A reference case in the (top_ids, top_lp, actual_lp) form compare() scores."""
    return case.ref_ids[:, :k].copy(), case.ref_lp[:, :k].copy(), case.ref_actual_lp.copy()


# ---------------------------------------------------------------------------------
# Logit reduction (numpy twin of the on-device reduction in tools/dsv41_validate.py)
# ---------------------------------------------------------------------------------

def bf16_round(x) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), returned as float32."""
    x = np.ascontiguousarray(x, dtype = np.float32)
    b = x.view(np.uint32).astype(np.uint64)
    r = (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)
    return np.where(np.isnan(x), x, r)


def log_softmax(x) -> np.ndarray:
    x = np.asarray(x, dtype = np.float64)
    m = x.max(axis = -1, keepdims = True)
    return x - m - np.log(np.exp(x - m).sum(axis = -1, keepdims = True))


def reduce_logits(logits, actual_ids, ref_ids = None, k: int = 5, bf16: bool = False):
    """
    logits [rows, V] -> (top_ids [rows, k], top_lp [rows, k], actual_lp [rows],
    at_ref_lp [rows, R] or None). actual_ids < 0 and ref_ids < 0 give nan.
    """
    x = np.asarray(logits, dtype = np.float32)
    if bf16:
        x = bf16_round(x)
    lp = log_softmax(x)
    rows = np.arange(lp.shape[0])
    part = np.argpartition(-lp, k - 1, axis = -1)[:, :k]
    order = np.argsort(-np.take_along_axis(lp, part, -1), axis = -1, kind = "stable")
    top_ids = np.take_along_axis(part, order, -1).astype(np.int64)
    top_lp = np.take_along_axis(lp, top_ids, -1)
    a = np.asarray(actual_ids, dtype = np.int64)
    act = np.where(a >= 0, lp[rows, np.clip(a, 0, None)], np.nan)
    at_ref = None
    if ref_ids is not None:
        r = np.asarray(ref_ids, dtype = np.int64)
        at_ref = np.where(r >= 0, np.take_along_axis(lp, np.clip(r, 0, None), -1), np.nan)
    return top_ids, top_lp, act, at_ref


# ---------------------------------------------------------------------------------
# Scoring against the reference
# ---------------------------------------------------------------------------------

def _q(a: np.ndarray, f: float):
    # 'lower' = sorted[floor(f * (len - 1))], the convention of the recorded noise
    return float(np.quantile(a, f, method = "lower")) if a.size else None


def _dist(a: np.ndarray) -> dict:
    if not a.size:
        return {"mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {"mean": float(a.mean()), "p50": _q(a, 0.5), "p90": _q(a, 0.9),
            "p99": _q(a, 0.99), "max": float(a.max())}


def _region_list(n: int, pos: np.ndarray, regions) -> list[tuple[int, int | None]]:
    out = [(lo, hi) for lo, hi in regions if lo < n and hi > 1]
    last = max(hi for _, hi in regions)
    if pos.size and pos.max() >= last:
        out.append((last, None))            # beyond the named regions (longer captures)
    return out


def compare(case: Case, top5_ids, top5_lp, actual_lp, *, positions = None, at_ref_lp = None,
            label: str = "", regions = REGIONS, margin: float | None = None, worst: int = 20,
            detail: bool = False) -> dict:
    """
    Score our reduced logprobs against one reference case.

    top5_ids/top5_lp [m, >=5] are our top tokens by descending logprob, actual_lp [m]
    our logprob of the actual prompt token, for prompt positions `positions` (default
    1 .. n-1, i.e. from logits[0 .. n-2]). at_ref_lp [m, 5], our logprobs AT the
    reference's top-5 ids, makes the top-5 KL exact; without it a reference token
    missing from our top-5 is imputed at our 5th logprob (an upper bound).
    """
    thr = GATES["top1_margin"] if margin is None else margin
    n = case.n
    pos = np.arange(1, n, dtype = np.int64) if positions is None else np.asarray(positions, dtype = np.int64)
    m = pos.size
    o_ids = np.asarray(top5_ids, dtype = np.int64).reshape(m, -1)[:, :5]
    o_lp = np.asarray(top5_lp, dtype = np.float64).reshape(m, -1)[:, :5]
    o_act = np.asarray(actual_lp, dtype = np.float64).reshape(m)
    if o_ids.shape[1] < 5 or o_lp.shape != o_ids.shape:
        raise ValueError(f"need [m, 5] top ids/logprobs, got {o_ids.shape} / {o_lp.shape}")
    if pos.ndim != 1 or np.unique(pos).size != m:
        raise ValueError("positions must be a one-dimensional set without duplicates")
    if m and (pos.min() < 1 or pos.max() > n - 1):
        raise ValueError(f"positions must lie in [1, {n - 1}]")
    rows = pos - 1
    r_ids = case.ref_ids[rows, :5]
    r_lp = case.ref_lp[rows, :5]
    r_act = case.ref_actual_lp[rows]
    marg = case.margin[rows]

    o_valid = o_ids >= 0
    r_valid = r_ids >= 0
    invalid_ranked = (~np.isfinite(o_lp).all(1) | ~np.isfinite(r_lp).all(1)
                      | ~o_valid.all(1) | ~r_valid.all(1)
                      | (np.diff(np.sort(o_ids, axis=1), axis=1) == 0).any(1)
                      | (np.diff(np.sort(r_ids, axis=1), axis=1) == 0).any(1))
    both = np.isfinite(o_act) & np.isfinite(r_act)
    dlp = np.where(both, np.abs(o_act - r_act), np.nan)
    top1_eq = (o_ids[:, 0] == r_ids[:, 0]) & r_valid[:, 0]
    has_top1 = r_valid[:, 0] & o_valid[:, 0]
    confident = has_top1 & (marg > thr)

    # top-5 overlap over the entries both sides actually have
    hit = ((r_ids[:, :, None] == o_ids[:, None, :]) & r_valid[:, :, None] & o_valid[:, None, :]).any(-1)
    denom = np.maximum(1, np.minimum(r_valid.sum(1), o_valid.sum(1)))
    overlap = hit.sum(1) / denom

    # KL(ref || ours) restricted to, and renormalized over, the reference top-5
    imputed = np.zeros_like(r_valid)
    if at_ref_lp is not None:
        q = np.asarray(at_ref_lp, dtype = np.float64).reshape(m, -1)[:, :5].copy()
        q[~r_valid] = np.nan
    else:
        eq = (r_ids[:, :, None] == o_ids[:, None, :]) & o_valid[:, None, :]
        q = np.where(eq.any(-1), np.take_along_axis(o_lp, eq.argmax(-1), 1), np.nan)
        is_act = r_ids == case.prompt_ids[pos][:, None]
        q = np.where(np.isnan(q) & is_act & np.isfinite(o_act)[:, None], o_act[:, None], q)
        floor = np.where(o_valid, o_lp, np.inf).min(1)
        imputed = np.isnan(q) & r_valid
        q = np.where(imputed, floor[:, None], q)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        p_l = np.where(r_valid, r_lp, -np.inf)
        p_l = p_l - np.logaddexp.reduce(p_l, axis = 1, keepdims = True)
        q_l = np.where(r_valid & np.isfinite(q), q, -np.inf)
        q_l = q_l - np.logaddexp.reduce(q_l, axis = 1, keepdims = True)
        kl = np.where(r_valid, np.exp(p_l) * (p_l - q_l), 0.0).sum(1)
    kl_ok = r_valid.any(1) & np.isfinite(kl)

    def summ(sel: np.ndarray) -> dict:
        b = sel & both
        c = sel & confident
        nll_o = float(-o_act[b].mean()) if b.any() else None
        nll_r = float(-r_act[b].mean()) if b.any() else None
        low = sel & has_top1 & ~(marg > thr)
        # leave-one-out: a small-sample |dNLL| can be one position (see prefix_noise)
        worst_pos, dnll_loo = None, None
        if b.sum() > 1:
            k = np.flatnonzero(b)[np.argmax(dlp[b])]
            worst_pos = int(pos[k])
            d = r_act[b] - o_act[b]
            dnll_loo = float((d.sum() - (r_act[k] - o_act[k])) / (d.size - 1))
        dd = (o_act - r_act)[b]
        dnll_se = float(dd.std(ddof = 1) / math.sqrt(dd.size)) if dd.size > 1 else None
        return {
            "count": int(sel.sum()),
            "compared": int(b.sum()),
            "dnll_se": dnll_se,
            "nll_ours": nll_o,
            "nll_ref": nll_r,
            "dnll": (nll_o - nll_r) if b.any() else None,
            "worst_pos": worst_pos,
            "dnll_loo": dnll_loo,
            "dlp": _dist(dlp[b]),
            "top1": {"agree": int((sel & has_top1 & top1_eq).sum()), "total": int((sel & has_top1).sum()),
                     "rate": float(top1_eq[sel & has_top1].mean()) if (sel & has_top1).any() else None},
            "top1_margin": {"agree": int((c & top1_eq).sum()), "total": int(c.sum()),
                            "rate": float(top1_eq[c].mean()) if c.any() else None, "margin": thr},
            "flips_low_margin": int((low & ~top1_eq).sum()),
            "overlap": float(overlap[sel].mean()) if sel.any() else None,
            "kl_top5": _dist(kl[sel & kl_ok]),
            "imputed_frac": float(imputed[sel].sum() / max(1, r_valid[sel].sum())),
            "missing_actual": int((sel & ~both).sum()),
        }

    everything = np.ones(m, dtype = bool)
    missing_isolation = isolation_missing(case)
    rep = {
        "label": label or case.name,
        "case": case.name,
        "n": n,
        "rep": case.rep,
        "positions": int(m),
        "coverage": float(m / max(1, n - 1)),
        "invalid_ranked_rows": int(invalid_ranked.sum()),
        "reference_contended": bool(case.meta.get("contended")),
        "reference_probe_errors": int(case.meta.get("probe_errors", 0)),
        # Legacy captures remain scoreable, but absent probes are not evidence
        # that contention was measured and ruled out during capture.
        "reference_isolation_metadata": {"complete": not missing_isolation,
                                         "missing": missing_isolation},
        "global": summ(everything),
        "regions": [],
        "head": None,
        "worst": [],
    }
    for lo, hi in _region_list(n, pos, regions):
        sel = (pos >= lo) & ((pos < hi) if hi is not None else True)
        r = {"lo": lo, "hi": hi, "note": REGION_NOTES.get((lo, hi), "")}
        r.update(summ(sel))
        rep["regions"].append(r)
    if (pos >= PRECISION_FROM).any():
        rep["head"] = summ(pos < PRECISION_FROM)
    order = np.argsort(-np.nan_to_num(dlp, nan = -1.0), kind = "stable")[:worst]
    for j in order:
        if not both[j]:
            continue
        rep["worst"].append({
            "pos": int(pos[j]), "dlp": float(dlp[j]), "ours": float(o_act[j]), "ref": float(r_act[j]),
            "token": int(case.prompt_ids[pos[j]]), "ref_top1": int(r_ids[j, 0]), "our_top1": int(o_ids[j, 0]),
            "margin": None if not np.isfinite(marg[j]) else float(marg[j]),
        })
    if detail:
        rep["per_position"] = {
            "pos": pos.tolist(), "dlp": [None if not np.isfinite(v) else round(float(v), 5) for v in dlp],
            "top1_eq": top1_eq.astype(int).tolist(),
            "margin": [None if not np.isfinite(v) else round(float(v), 4) for v in marg],
            "overlap": np.round(overlap, 3).tolist(),
        }
    return rep


def _checks(s: dict, g: dict, dnll_limit: float) -> list[tuple]:
    """The same distribution/ranking gates globally and regionally (only the |dNLL| limit
    differs); no candidate-derived limits. Each check is (name, value, op, calibrated limit,
    passes it, key), key naming its measure in CALIBRATION_NOISE (None: never noise-shifted)."""
    if not s or s["compared"] == 0:
        return [("positions compared", 0, ">", 0, False, None)]
    lim = dnll_limit
    out = [("missing actual logprobs", s.get("missing_actual", 0), "==", 0,
            s.get("missing_actual", 0) == 0, None),
           ("|dNLL|", abs(s["dnll"]), "<=", lim, abs(s["dnll"]) <= lim, "dnll")]
    t = s["top1_margin"]
    if t["total"]:
        out.append((f"top-1 @margin>{t['margin']}", t["rate"], ">=", g["top1"], t["rate"] >= g["top1"],
                    "top1_margin"))
    d = s["dlp"]
    out.append(("|dlp| p50", d["p50"], "<=", g["dlp_p50"], d["p50"] <= g["dlp_p50"], "dlp_p50"))
    out.append(("|dlp| p90", d["p90"], "<=", g["dlp_p90"], d["p90"] <= g["dlp_p90"], "dlp_p90"))
    for key in ("p99", "max"):
        limit = g.get("dlp_" + key, GATES["dlp_" + key])
        out.append(("|dlp| " + key, d[key], "<=", limit, d[key] is not None and d[key] <= limit, "dlp_" + key))
    out.append(("top-5 overlap", s["overlap"], ">=", g["overlap"], s["overlap"] >= g["overlap"], "overlap"))
    return out


def _v(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _fmt(c: tuple, where: str = "", s: dict | None = None, status: str | None = None,
         detail: dict | None = None) -> str:
    """One check line. status/detail come from _judge when the report carries reference noise:
    'noise' prints as a FAIL within the reference's own noise, which does not fail the run."""
    name, v, op, lim, ok = c[:5]
    note = ""
    if not ok and name == "|dNLL|" and s and s.get("dnll_loo") is not None:
        note = f"  ({abs(s['dnll_loo']):.4f} without position {s['worst_pos']})"
    line = f"{where}{name} {_v(v)} {op} {lim}{note}"
    if ok:
        return f"PASS  {line}"
    if status is None or detail is None:
        return f"FAIL  {line}"
    if detail.get("why") == "none":
        return f"FAIL  {line}; no comparison of vLLM against itself covers these positions"
    if detail.get("why") == "unlisted":
        return (f"FAIL  {line}; more positions exceed {lim} than the report lists, so their own "
                f"reference noise cannot be checked")
    if "positions" in detail:                       # |dlp| max: position by position
        h = detail["headroom"]
        shown = sorted(detail["positions"], key = lambda w: -w["dlp"])[:3]
        each = "; ".join(f"position {w['pos']} {w['dlp']:.4f} <= "
                         + (f"{w['noise'] + h:.4f} (vLLM against itself {w['noise']:.4f} there, "
                            f"{w['source']})" if w["noise"] is not None else "? (no comparison of vLLM "
                                                                             "against itself there)")
                         for w in shown)
        more = f"; and {len(detail['positions']) - 3} more" if len(detail["positions"]) > 3 else ""
        head = "FAIL within reference noise" if status == "noise" else "FAIL"
        judged = "within" if status == "noise" else "not all within"
        return (f"{head}  {line}; each position beyond {lim} {judged} vLLM's own difference there "
                f"+ {h:.4f}: {each}{more}")
    shifted = detail["limit"]
    where_noise = (f"vLLM against itself {_v(detail['value'])} on these positions ({detail['source']}), "
                   f"{_v(detail['calibration'])} at calibration")
    if status == "noise":
        return f"FAIL within reference noise  {line}; noise-shifted limit {_v(shifted)}: {where_noise}"
    if (op == "<=" and shifted <= lim) or (op == ">=" and shifted >= lim):
        return f"FAIL  {line}; no wider here: {where_noise}"
    return f"FAIL  {line}; beyond the noise-shifted limit {_v(shifted)}: {where_noise}"


def _rname(r: dict) -> str:
    return f"[{r['lo']},{r['hi'] if r['hi'] is not None else 'end'})"


# ---------------------------------------------------------------------------------
# The reference's own noise: noise-shifted limits
# ---------------------------------------------------------------------------------

# What the calibrated limits (GATES) were set on: vLLM rep1 against rep0 at n = 300 of the
# calibration capture (MEASURED_NOISE[300]). Each limit sits a fixed headroom beyond it, left for
# the systematic differences expected between the two engines:
#     headroom = limit - calibration noise   (distances: |dNLL|, |dlp| quantiles and max)
#     headroom = calibration noise - limit   (agreement: top-1 at margin, top-5 overlap)
# At another length vLLM may differ from itself by more. attach_noise() records, for every
# summary of a report, the worst difference of vLLM against itself that the capture shows on
# exactly the same positions (reference_noise), and a check is then judged against
#     noise-shifted limit = that noise + headroom   (or that noise - headroom)
# never tighter than the calibrated limit. |dlp| max is judged position by position: every
# position beyond its calibrated limit must lie within vLLM's own difference at that position
# plus the headroom (vLLM's position 1, predicted from one token, moves by up to 6 nats between
# request lengths; that must not excuse a large difference at any other position).
_N300 = MEASURED_NOISE[300]
CALIBRATION_NOISE = {
    "dnll": _N300["dnll"],
    "top1_margin": _N300["top1_margin"][0] / _N300["top1_margin"][1],
    "dlp_p50": _N300["dlp_p50"], "dlp_p90": _N300["dlp_p90"], "dlp_p99": _N300["dlp_p99"],
    "dlp_max": _N300["dlp_max"],
    "overlap": _N300["overlap"],
}
_AT_LEAST = ("top1_margin", "overlap")          # agreement measures: larger is better


def _isolation_problem(case: Case, strict: bool) -> str | None:
    """Why a capture must not serve as reference noise, or None."""
    if case.meta.get("contended"):
        return "recorded as contended"
    if case.meta.get("probe_errors"):
        return f"recorded with {case.meta['probe_errors']} probe errors"
    missing = isolation_missing(case)
    if strict and missing:
        return f"no isolation metadata ({', '.join(missing)}; --strict)"
    return None


def reference_noise(cases: list[Case], case: Case, strict: bool = False) -> tuple[list[dict], list[str]]:
    """
    vLLM against itself on the positions of `case`, from the other captures of the same file:
    every other rep of the same prompt (rep noise), the same prompt inside every longer capture
    that starts with it (prefix noise), and every shorter capture that `case` starts with, on
    that capture's positions. Each is a compare() report of `case` with the other capture in the
    place of ours (per-position arrays included, for the |dlp| max check). A capture recorded as
    contended or with probe errors never serves as noise; with strict, neither does one without
    isolation metadata. Returns (reports, excluded), one line per capture left out and why.
    """
    reports, excluded = [], []
    for other in sorted(cases, key = lambda c: (c.n, c.rep)):
        if other is case or (other.n == case.n and other.rep == case.rep):
            continue
        m = min(other.n, case.n)
        if m < 2 or not np.array_equal(other.prompt_ids[:m], case.prompt_ids[:m]):
            continue
        why = _isolation_problem(other, strict)
        if why:
            excluded.append(f"{other.name}: {why}")
            continue
        ids, lp, act = as_ours(other)
        if other.n == case.n:
            label = f"vLLM rep{other.rep} vs rep{case.rep} n={case.n}"
            reports.append(compare(case, ids, lp, act, label = label, detail = True))
        elif other.n > case.n:
            k = case.n - 1
            label = f"vLLM n={other.n} rep{other.rep} prefix vs n={case.n} rep{case.rep}"
            reports.append(compare(case, ids[:k], lp[:k], act[:k], label = label, detail = True))
        else:
            label = f"vLLM n={other.n} rep{other.rep} vs n={case.n} rep{case.rep} positions 1-{other.n - 1}"
            reports.append(compare(case, ids, lp, act, positions = np.arange(1, other.n), label = label,
                                   detail = True))
    return reports, excluded


def _span_count(n: int, lo: int, hi: int | None) -> int:
    """Prompt positions of a case of length n in [lo, hi)."""
    return max(0, min(n if hi is None else hi, n) - max(lo, 1))


def _noise_value(s: dict, key: str):
    if key == "dnll":
        return None if s.get("dnll") is None else abs(s["dnll"])
    if key == "top1_margin":
        return s["top1_margin"]["rate"] if s["top1_margin"]["total"] else None
    if key == "overlap":
        return s.get("overlap")
    return s["dlp"][key[len("dlp_"):]]


def attach_noise(report: dict, noise: list[dict], excluded = ()) -> dict:
    """
    Record on a compare() report the reference's own noise on its positions (reference_noise of
    the same case), so that gates(), verdict() and ablation_check() judge it against the
    noise-shifted limits. For each summary (global, head, each region) and each measure: the worst
    value over the comparisons that cover exactly the summary's positions, and which one it was;
    a comparison that covers only part of them does not count (the global summary of a case
    longer than every other capture has none). For |dlp| max: vLLM's own worst difference at each
    position the report lists as its worst (report["worst"]), and down to which |dlp| that list
    is complete. Returns the report.
    """
    n = report["n"]

    def worst(s: dict | None, lo: int, hi: int | None, pick) -> dict:
        full = _span_count(n, lo, hi)
        out = {}
        if not s or s.get("count") != full or s.get("compared") != full:
            return out
        for r in noise:
            ns = pick(r)
            if not ns or ns.get("count") != full or ns.get("compared") != full:
                continue
            for key in CALIBRATION_NOISE:
                if key == "dlp_max":
                    continue
                value = _noise_value(ns, key)
                if value is None:
                    continue
                cur = out.get(key)
                if cur is None or (value < cur["value"] if key in _AT_LEAST else value > cur["value"]):
                    out[key] = {"value": value, "source": r["label"], "calibration": CALIBRATION_NOISE[key]}
        return out

    per_pos = []
    for r in noise:
        pp = r.get("per_position")
        if pp and pp["pos"]:
            per_pos.append((r["label"], pp["pos"][0], pp["dlp"]))
    listed = []
    for w in report.get("worst", []):
        best = None
        for label, first, dlp in per_pos:
            i = w["pos"] - first
            if 0 <= i < len(dlp) and dlp[i] is not None and (best is None or dlp[i] > best[0]):
                best = (dlp[i], label)
        listed.append({"pos": w["pos"], "dlp": w["dlp"], "noise": None if best is None else best[0],
                       "source": None if best is None else best[1]})
    compared = report["global"].get("compared", 0)
    floor = 0.0 if len(listed) >= compared else min((w["dlp"] for w in listed), default = 0.0)
    report["reference_noise"] = {
        "sources": [r["label"] for r in noise],
        "excluded": list(excluded),
        "global": worst(report["global"], 1, None, lambda r: r["global"]),
        "head": worst(report.get("head"), 1, PRECISION_FROM, lambda r: r.get("head"))
                if report.get("head") else {},
        "regions": [{"lo": x["lo"], "hi": x["hi"],
                     "noise": worst(x, x["lo"], x["hi"],
                                    lambda r, lo = x["lo"], hi = x["hi"]: next(
                                        (y for y in r["regions"] if y["lo"] == lo and y["hi"] == hi), None))}
                    for x in report["regions"]],
        "worst": listed,
        "worst_floor": floor,
    }
    return report


def _summary_noise(rn: dict | None, lo: int, hi: int | None, which: str) -> dict | None:
    if rn is None:
        return None
    if which in ("global", "head"):
        return rn.get(which) or {}
    return next((x["noise"] for x in rn.get("regions", []) if x["lo"] == lo and x["hi"] == hi), {})


def _judge(c: tuple, rn: dict | None, noise: dict | None, span: tuple) -> tuple[str, dict | None]:
    """
    ('pass' | 'noise' | 'fail', detail) of one check. 'noise': beyond the calibrated limit, within
    the noise-shifted one (a FAIL within the reference's own noise, which does not fail the run).
    Without reference noise on the report (rn None) every failure is a plain 'fail'.
    """
    name, v, op, lim, ok, key = c
    if ok:
        return "pass", None
    if rn is None or key is None:
        return "fail", None
    n0 = CALIBRATION_NOISE[key]
    headroom = lim - n0 if op == "<=" else n0 - lim
    if key == "dlp_max":
        if rn.get("worst_floor", math.inf) > lim:
            return "fail", {"why": "unlisted"}
        lo, hi = span
        over = [w for w in rn.get("worst", []) if w["pos"] >= lo and (hi is None or w["pos"] < hi)
                and w["dlp"] > lim]
        within = bool(over) and all(w["noise"] is not None and w["dlp"] <= w["noise"] + headroom for w in over)
        return ("noise" if within else "fail"), {"positions": over, "headroom": headroom}
    e = (noise or {}).get(key)
    if e is None or v is None:
        return "fail", {"why": "none"}
    shifted = e["value"] + headroom if op == "<=" else e["value"] - headroom
    within = v <= shifted if op == "<=" else v >= shifted
    return ("noise" if within else "fail"), {"limit": shifted, **e}


def noise_limits(report: dict, g: dict = GATES) -> dict:
    """
    The noise-shifted limits of a report with attach_noise(), for printing: {'global': {...},
    'head': {...}, 'regions': [{'lo', 'hi', 'limits': {...}}]}, each {measure: (limit, source)} for
    the measures whose limit the reference's noise widens. |dlp| max is judged per position and
    is not listed.
    """
    rn = report.get("reference_noise")
    if rn is None:
        return {}

    def lim_of(key, region):
        if key == "dnll":
            return g["region_dnll"] if region else g["dnll"]
        return g["top1"] if key == "top1_margin" else g[key]

    def limits(noise, region):
        out = {}
        for key, e in (noise or {}).items():
            lim = lim_of(key, region)
            n0 = CALIBRATION_NOISE[key]
            shifted = e["value"] + (lim - n0) if key not in _AT_LEAST else e["value"] - (n0 - lim)
            if (key in _AT_LEAST and shifted < lim) or (key not in _AT_LEAST and shifted > lim):
                out[key] = (shifted, e["source"])
        return out

    return {"global": limits(rn.get("global"), False), "head": limits(rn.get("head"), False),
            "regions": [{"lo": x["lo"], "hi": x["hi"], "limits": limits(x["noise"], True)}
                        for x in rn.get("regions", [])]}


def _gate_eval(report: dict, regions = REGIONS, g: dict = GATES, strict: bool = False):
    msgs, fails, within_noise, region_fails = [], 0, 0, []
    rn = report.get("reference_noise")
    isolation = report.get("reference_isolation_metadata")
    complete = isolation is not None and isolation.get("complete", False)
    if not complete:
        missing = ", ".join(isolation.get("missing", [])) if isolation is not None else "unknown"
        if strict:
            msgs.append(f"FAIL  reference isolation not recorded (no {missing}): nothing shows that vLLM "
                        f"served this prompt alone (--strict)")
            fails += 1
        else:
            msgs.append(f"WARNING reference isolation NOT RECORDED (no {missing}): nothing shows that vLLM "
                        f"served this prompt alone, and a shared batch changes its numerics; scored as if it "
                        f"did (default false/zero flags do not establish measured non-contention); --strict "
                        f"fails this")
    if rn is not None:
        if rn["sources"]:
            msgs.append(f"info  reference noise: {len(rn['sources'])} comparisons of vLLM against itself on "
                        f"this case's positions: {'; '.join(rn['sources'])}")
        else:
            msgs.append("info  reference noise: no other capture shares this case's prompt; the calibrated "
                        "limits alone apply")
        for x in rn.get("excluded", []):
            msgs.append(f"info  not used as reference noise: {x}")
    for key, expected in (("coverage", 1.0), ("invalid_ranked_rows", 0),
                          ("reference_contended", False), ("reference_probe_errors", 0)):
        if report.get(key, expected) != expected:
            msgs.append(f"FAIL  {key}: {report[key]} (expected {expected})")
            fails += 1
    if report.get("nonfinite"):
        msgs.append(f"FAIL  non-finite logits at {report['nonfinite']} positions")
        fails += 1
    if report.get("unfilled"):
        msgs.append(f"FAIL  {report['unfilled']} positions never received logits")
        fails += 1

    def judge(s, where, dnll_limit, noise, span):
        nonlocal within_noise
        failed = 0
        for c in _checks(s, g, dnll_limit):
            status, detail = _judge(c, rn, noise, span)
            msgs.append(_fmt(c, where, s, status, detail))
            failed += status == "fail"
            within_noise += status == "noise"
        return failed

    fails += judge(report["global"], "", g["dnll"], _summary_noise(rn, 1, None, "global"), (1, None))
    allowed = None if regions is None else set(regions)
    for r in report["regions"]:
        if allowed is not None and r["hi"] is not None and (r["lo"], r["hi"]) not in allowed:
            continue
        if r["compared"] < g["region_min_positions"]:
            msgs.append(f"info  region {_rname(r)}: {r['compared']} positions, not gated")
            continue
        failed = judge(r, f"region {_rname(r)} ", g["region_dnll"],
                       _summary_noise(rn, r["lo"], r["hi"], "region"), (r["lo"], r["hi"]))
        fails += failed
        if failed:
            region_fails.append(r)
    suspect = False
    if fails and not report.get("nonfinite") and not report.get("unfilled") and report.get("head") \
            and report.get("coverage", 1.0) == 1.0 and not report.get("invalid_ranked_rows") \
            and not report["global"].get("missing_actual") and not report.get("reference_contended") \
            and not report.get("reference_probe_errors") and (complete or not strict):
        head_noise = _summary_noise(rn, 1, PRECISION_FROM, "head")
        head_ok = all(_judge(c, rn, head_noise, (1, PRECISION_FROM))[0] != "fail"
                      for c in _checks(report["head"], g, g["dnll"]))
        suspect = head_ok and all(r["lo"] >= PRECISION_FROM for r in region_fails)
    if suspect:
        msgs.append(f"precision-suspect: no check on positions < {PRECISION_FROM} fails beyond its "
                    f"noise-shifted limit, and every failing region starts at >= {PRECISION_FROM}, where the "
                    f"index top-k starts to select (vLLM rounds its indexer to MXFP4 and its KV to FP8 at "
                    f"every position); re-run with vLLM's contract (--dsv41-numerics vllm) before calling "
                    f"it a bug")
    return fails == 0, msgs, suspect, within_noise


def gates(report: dict, regions = REGIONS, g: dict = GATES, strict: bool = False) -> tuple[bool, list[str]]:
    """
    (no check fails, one message per check). A check beyond its calibrated limit (GATES) but
    within its noise-shifted limit (a report with attach_noise) prints as 'FAIL within reference
    noise' and does not count. A reference without isolation metadata is scored with a WARNING,
    or fails with strict. See GATES, CALIBRATION_NOISE and PRECISION_FROM.
    """
    ok, msgs, _, _ = _gate_eval(report, regions, g, strict)
    return ok, msgs


def verdict(report: dict, regions = REGIONS, g: dict = GATES, strict: bool = False) -> str:
    """
    'pass' (every check within its calibrated limit), 'within-noise' (no check fails, and at least
    one is beyond its calibrated limit but within its noise-shifted limit), 'precision-suspect'
    (fails only at >= PRECISION_FROM) or 'fail'. gates() passes the first two.
    """
    ok, _, suspect, within_noise = _gate_eval(report, regions, g, strict)
    if ok:
        return "within-noise" if within_noise else "pass"
    return "precision-suspect" if suspect else "fail"


def self_noise(cases: list[Case]) -> list[dict]:
    """vLLM's own run-to-run noise: every extra rep scored against rep 0 of the same n."""
    out = []
    by: dict[int, list[Case]] = {}
    for c in cases:
        by.setdefault(c.n, []).append(c)
    for n, cs in sorted(by.items()):
        base = min(cs, key = lambda c: c.rep)
        for other in cs:
            if other is base:
                continue
            if not np.array_equal(other.prompt_ids, base.prompt_ids):
                continue
            r = compare(base, *as_ours(other), label = f"vLLM rep{other.rep} vs rep{base.rep} n={n}")
            out.append(r)
    return out


def prefix_noise(cases: list[Case]) -> list[dict]:
    """
    vLLM against itself across REQUEST SHAPES: a shorter capture scored against the same
    positions of a longer capture that shares its prefix. The logprobs should be identical;
    what differs is chunking, batch shape and fp8 KV rounding. Rep-vs-rep noise (self_noise)
    is the same request twice and is much tighter, and it is the only noise the calibrated
    limits were set on: they FAIL this prefix noise, i.e. vLLM against itself, which is why
    reference_noise() counts prefix pairs too (every longer and shorter capture, where this
    function takes the next longer one only) and the gates judge a report with it against
    noise-shifted limits. On the calibration
    capture the outlier is always POSITION 1 (predicted from token 0 alone, so causally
    independent of the rest of the prompt): vLLM gave it -13.50 at n=24, -10.40 at n=300,
    -7.45 at n=1500 and -11.26 at n=20000. That one position alone fails the n=24 |dNLL| gate
    (0.139 over 23 positions, 0.005 without it); gates() prints the leave-one-out value next to
    every |dNLL| failure for this reason. n=20000 against n=1500 fails region [1,128) (|dNLL|
    0.052, 0.022 without position 1; |dlp| p99 1.24), the size of the port's own failures there
    at n >= 1500 (|dlp| p99 1.06-1.45 on the recorded runs).
    """
    out = []
    firsts = sorted((c for c in cases if c.rep == min(x.rep for x in cases if x.n == c.n)), key = lambda c: c.n)
    for short in sorted(cases, key = lambda c: (c.n, c.rep)):
        for longer in firsts:
            if longer.n <= short.n or not np.array_equal(longer.prompt_ids[:short.n], short.prompt_ids):
                continue
            m = short.n - 1
            ids, lp, act = as_ours(longer)
            out.append(compare(short, ids[:m], lp[:m], act[:m],
                               label = f"vLLM n={longer.n} prefix vs n={short.n} rep{short.rep}"))
            break                           # the next longer capture only
    return out


# ---------------------------------------------------------------------------------
# exllamav3 vs itself
# ---------------------------------------------------------------------------------

def chunk_starts(schedule) -> list[int]:
    """Input positions a > 0 at which a forward of `schedule` starts, i.e. where state is carried in."""
    return np.cumsum(list(schedule)[:-1], dtype = np.int64).tolist()


def _gap_to(ids: np.ndarray, lp: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Per row: this run's top-1 logprob minus its logprob of token `other` (inf if not in its top-k)."""
    eq = (ids == other[:, None]) & (other[:, None] >= 0)
    at = np.take_along_axis(lp, eq.argmax(1)[:, None], 1)[:, 0]
    with np.errstate(invalid = "ignore"):
        return np.where(eq.any(1) & np.isfinite(at), lp[:, 0] - at, np.inf)


def compare_self(a, b, *, n: int, positions = None, tie_eps: float | None = None, boundaries = None,
                 label: str = "", regions = REGIONS, worst: int = 20) -> dict:
    """
    Two exllamav3 runs of the same sequence, each (top_ids [m, k>=5], top_lp, actual_lp [m]).

    A top-1 difference X (run a) vs Y (run b) is excused only when X and Y themselves are
    within tie_eps in at least one run (default 0: exact ties; our logits are fp16, so that
    is fp16 resolution, 1 ulp = 0.0156 for |logit| in [16, 32)). A tie between one run's
    top-1 and some third token excuses nothing. Every hard flip is listed with gap_a/gap_b,
    how far each run ranks the other run's top-1 below its own: their sum is the relative
    logit shift the flip implies, a few fp16 ulps for noise and far more for a bug.

    boundaries: input positions a > 0 where either run started a forward (chunk_starts of
    its schedule). Predicted positions a+1 .. a+BOUNDARY_WINDOW -- the logits of the first
    tokens of a chunk, whose n-gram lookback, ring and rate-2 carry come from the previous
    forward -- get their own summary. A bug there touches a few positions per chunk, which
    the global p99 cannot see at chunk 2048.
    """
    eps = SELF_GATES["tie_eps"] if tie_eps is None else tie_eps
    a_ids, a_lp, a_act = (np.asarray(x) for x in a)
    b_ids, b_lp, b_act = (np.asarray(x) for x in b)
    for ids, lp, actual in ((a_ids, a_lp, a_act), (b_ids, b_lp, b_act)):
        if ids.ndim != 2 or ids.shape[1] < 5 or lp.shape != ids.shape \
                or actual.ndim != 1 or actual.shape[0] != ids.shape[0] \
                or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("self-comparison requires aligned integer top-5 IDs/logprobs and actual logprobs")
    a_lp, b_lp = a_lp.astype(np.float64), b_lp.astype(np.float64)
    m = a_act.shape[0]
    if b_act.shape[0] != m:
        raise ValueError(f"runs cover {m} and {b_act.shape[0]} positions")
    pos = np.arange(1, n, dtype = np.int64) if positions is None else np.asarray(positions, dtype = np.int64)
    if pos.ndim != 1 or pos.size != m or np.unique(pos).size != m \
            or (m and (pos.min() < 1 or pos.max() >= n)):
        raise ValueError("self-comparison positions must be unique, aligned and inside the prompt")
    both = np.isfinite(a_act) & np.isfinite(b_act)
    dlp = np.where(both, np.abs(a_act - b_act), np.nan)
    diff = (a_ids[:, 0] != b_ids[:, 0]) & (a_ids[:, 0] >= 0) & (b_ids[:, 0] >= 0)
    gap_a = _gap_to(a_ids, a_lp, b_ids[:, 0])
    gap_b = _gap_to(b_ids, b_lp, a_ids[:, 0])
    tie = (gap_a <= eps) | (gap_b <= eps)
    hard = diff & ~tie
    k = min(5, a_ids.shape[1], b_ids.shape[1])
    hit = (a_ids[:, :k, None] == b_ids[:, None, :k]).any(-1).sum(1) / k
    edge = np.zeros(m, dtype = bool)
    starts = sorted({int(x) for x in (boundaries or []) if 0 < int(x) < n})
    if starts and m:
        mark = np.zeros(n + BOUNDARY_WINDOW + 1, dtype = bool)
        for s in starts:
            mark[s + 1:s + 1 + BOUNDARY_WINDOW] = True
        edge = mark[pos]

    def summ(sel):
        d = dlp[sel & both]
        return {"count": int(sel.sum()), "compared": int((sel & both).sum()), "missing": int((sel & ~both).sum()),
                "dlp": _dist(d), "top1_diff": int((sel & diff).sum()),
                "top1_diff_ties": int((sel & diff & tie).sum()), "top1_diff_hard": int((sel & hard).sum()),
                "overlap": float(hit[sel].mean()) if sel.any() else None}

    def fin(v):
        return None if not np.isfinite(v) else round(float(v), 5)

    invalid_ranked = (~np.isfinite(a_lp).all(1) | ~np.isfinite(b_lp).all(1)
                      | (a_ids < 0).any(1) | (b_ids < 0).any(1)
                      | (np.diff(np.sort(a_ids, axis=1), axis=1) == 0).any(1)
                      | (np.diff(np.sort(b_ids, axis=1), axis=1) == 0).any(1))
    rep = {"label": label, "n": n, "positions": int(m), "coverage": m / max(n - 1, 1),
           "invalid_ranked_rows": int(invalid_ranked.sum()), "global": summ(np.ones(m, dtype = bool)),
           "tie_eps": eps, "regions": [], "hard_positions": pos[hard][:worst].tolist(),
           "hard_flips": [{"pos": int(pos[j]), "a": int(a_ids[j, 0]), "b": int(b_ids[j, 0]),
                           "gap_a": fin(gap_a[j]), "gap_b": fin(gap_b[j])} for j in np.flatnonzero(hard)[:worst]],
           "boundary_starts": len(starts), "boundary_window": BOUNDARY_WINDOW,
           "boundary": summ(edge) if starts else None,
           "rest": summ(~edge) if starts else None,
           "worst": [{"pos": int(pos[j]), "dlp": float(dlp[j]), "boundary": bool(edge[j])}
                     for j in np.argsort(-np.nan_to_num(dlp, nan = -1.0), kind = "stable")[:worst] if both[j]]}
    for lo, hi in _region_list(n, pos, regions):
        sel = (pos >= lo) & ((pos < hi) if hi is not None else True)
        r = {"lo": lo, "hi": hi}
        r.update(summ(sel))
        rep["regions"].append(r)
    return rep


def self_gates(report: dict, g: dict = SELF_GATES, floor: dict | None = None) -> tuple[bool, list[str]]:
    """
    Gates for two exllamav3 runs of the same sequence (a compare_self report).

    Without a floor they are strict, for comparisons that should be (near) bitwise: no hard
    top-1 flip, |dlp| p99 and the chunk-boundary p90 within g's absolute limits.

    floor is compare_self of the SAME model on the same case with only the row count of one
    forward changed (the stateless forward vs a shorter prefix). On the full model two correct
    paths differ by far more than 0.02: the exl3 kernels pick tiers by row count and the MoE
    routing near-ties amplify a few ulps into different experts (p99 0.3-0.5 nats measured,
    at positions no cache is involved in). With a floor the limits are relative to it:

      hard flips   <= m * floor rate * floor_mult + floor_sigmas * sqrt(that)   (Poisson)
      |dlp| p99    <= max(dlp_p99, floor_mult * the floor's p99)
      boundary p90 <= max(boundary_dlp_p90, boundary_vs_rest * independent floor p90)

    The candidate cannot increase the boundary tolerance by corrupting non-boundary rows.
    The floor is still a limited control (same model, shorter request), not a universal
    precision bound. Fixed maximum-error and ranking gates apply as well.
    """
    s = report["global"]
    msgs = []
    compared = s.get("compared", s["count"])
    ok_fill = (s.get("missing", 0) == 0 and compared > 0
               and report.get("coverage", 1.0) == 1.0 and not report.get("invalid_ranked_rows"))
    msgs.append(f"{'PASS' if ok_fill else 'FAIL'}  {compared}/{s['count']} positions have a "
                f"finite logprob in both runs")
    fl = floor["global"] if floor else None
    if floor and (floor.get("coverage", 1.0) != 1.0 or floor.get("invalid_ranked_rows")
                  or fl.get("missing", 0) or not fl.get("compared", fl["count"])):
        return False, msgs + ["FAIL  calibration floor has incomplete/nonfinite coverage"]
    if fl is None:
        lim_top, how = 0, ""
    else:
        fc = max(fl.get("compared", fl["count"]), 1)
        expect = g["floor_mult"] * fl["top1_diff_hard"] / fc * compared
        lim_top = int(math.floor(expect + g["floor_sigmas"] * math.sqrt(expect)))
        how = f"; limit {lim_top} from the floor's {fl['top1_diff_hard']}/{fc}"
    ok_top = s["top1_diff_hard"] <= lim_top
    flips = [(f["pos"], f["gap_a"], f["gap_b"]) for f in report.get("hard_flips", [])[:6]]
    msgs.append(f"{'PASS' if ok_top else 'FAIL'}  top-1 differs at {s['top1_diff_hard']} positions where the two "
                f"tokens are not tied in either run ({s['top1_diff_ties']} tie flips excused{how})"
                + (f"; first (pos, gap_a, gap_b): {flips}" if not ok_top else ""))
    p99 = s["dlp"]["p99"]
    lim_p99 = g["dlp_p99"]
    if fl is not None and fl["dlp"]["p99"] is not None:
        lim_p99 = max(lim_p99, g["floor_mult"] * fl["dlp"]["p99"])
    ok_p = p99 is not None and p99 <= lim_p99
    overlap_limit = g.get("overlap", 0.90)
    if fl is not None and fl.get("overlap") is not None:
        overlap_limit = max(overlap_limit, 1.0 - g["floor_mult"] * (1.0 - fl["overlap"]))
    ok_overlap = s.get("overlap") is not None and s["overlap"] >= overlap_limit
    msgs.append(f"{'PASS' if ok_overlap else 'FAIL'}  top-5 overlap {s.get('overlap')} >= {overlap_limit}")
    max_limit = g.get("dlp_max", 5.0)
    ok_max = s["dlp"]["max"] is not None and s["dlp"]["max"] <= max_limit
    msgs.append(f"{'PASS' if ok_max else 'FAIL'}  |dlp| max {s['dlp']['max']} <= {max_limit}")
    msgs.append(f"{'PASS' if ok_p else 'FAIL'}  |dlp| p99 {p99 if p99 is None else round(p99, 5)} "
                f"<= {round(lim_p99, 5)} (max {s['dlp']['max']}"
                + (f"; floor p99 {round(fl['dlp']['p99'], 5)})" if fl is not None and fl["dlp"]["p99"] is not None else ")"))
    ok_b = True
    b = report.get("boundary")
    if b:
        where = (f"positions a+1..a+{report['boundary_window']} after {report['boundary_starts']} "
                 f"chunk starts a")
        bp = b["dlp"]["p90"]
        rest = report.get("rest")
        lim_b, vs = g["boundary_dlp_p90"], ""
        if fl is not None and rest and rest.get("compared", 0) >= g["boundary_min_positions"] \
                and rest["dlp"]["p90"] is not None:
            lim_b = max(lim_b, g["boundary_vs_rest"] * fl["dlp"]["p90"])
            vs = f"; independent row-count floor p90 {round(fl['dlp']['p90'], 5)}"
        if b["compared"] >= g["boundary_min_positions"]:
            ok_b = bp is not None and bp <= lim_b
            msgs.append(f"{'PASS' if ok_b else 'FAIL'}  chunk-boundary |dlp| p90 {round(bp, 5) if bp is not None else None} "
                        f"<= {round(lim_b, 5)} over {b['compared']} {where} (max {b['dlp']['max']}{vs})")
        else:
            msgs.append(f"info  chunk boundary: {b['compared']} {where}, not gated (max {b['dlp']['max']})")
    return ok_fill and ok_top and ok_p and ok_b and ok_overlap and ok_max, msgs


# ---------------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------------

def _failing(report: dict, s: dict, g: dict, span: tuple, which: str) -> set[str]:
    """Names of the checks that fail under the rule the verdict applies (_gate_eval): the same
    checks globally and in each region, with the global or the regional |dNLL| limit, each judged
    against its noise-shifted limit when the report carries reference noise (a FAIL within the
    reference's own noise does not count), so detection cannot claim a failure the verdict would
    pass."""
    lim = g["region_dnll"] if which == "region" else g["dnll"]
    rn = report.get("reference_noise")
    noise = _summary_noise(rn, span[0], span[1], which)
    return {c[0] for c in _checks(s, g, lim) if _judge(c, rn, noise, span)[0] == "fail"}


def ablation_check(report: dict, token: str, baseline: dict | None = None, g: dict = GATES,
                   chunk_starts = None) -> dict:
    """
    Did an ablated run fail where the ablation acts? A region counts as detected when a
    check the verdict applies there (distribution/ranking checks and regional NLL limit)
    fails that the baseline run passes, each run judged as gates() judges it: against the
    noise-shifted limits when the report carries reference noise, so a failure within the
    reference's own noise neither detects nor leaks. Isolated ablations must also
    leave the regions below them passing.
    status: 'detected' | 'blind' | 'leak' | 'n/a' (the case never reaches the region, or an
    ablation that acts at chunk starts runs without one).

    chunk_starts: the input positions a > 0 where the run started a forward (chunk_starts of
    its schedule; [] for a single stateless forward). With it, an ablation that acts only at
    chunk starts (Ablation.at_chunk_starts, engram_nocarry) is 'n/a' on a run that has none
    before its last scored prediction; without it (None) that cannot be known and the run is
    scored as any other.
    """
    ab = ABLATIONS[token]
    if ab.at_chunk_starts and chunk_starts is not None \
            and not any(0 < int(a) < report["n"] - 1 for a in chunk_starts):
        return {"token": token, "case": report.get("case"), "status": "n/a", "ok": None,
                "msgs": [f"n/a: the run has no chunk start inside n={report['n']}, where '{token}' acts"]}
    base_regions = {(r["lo"], r["hi"]): r for r in (baseline or {}).get("regions", [])}
    msgs, hit, leak, reached = [], [], [], False
    for r in report["regions"]:
        if r["compared"] < g["region_min_positions"]:
            continue
        span = (r["lo"], r["hi"])
        new = _failing(report, r, g, span, "region")
        if baseline is not None and span in base_regions:
            new -= _failing(baseline, base_regions[span], g, span, "region")
        if r["lo"] >= ab.affects_from:
            reached = True
            if new:
                hit.append(r)
                msgs.append(f"detected in {_rname(r)}: {', '.join(sorted(new))}")
        elif ab.isolated and r["hi"] is not None and r["hi"] <= ab.affects_from and new:
            leak.append(r)
            msgs.append(f"LEAK below {ab.affects_from} in {_rname(r)}: {', '.join(sorted(new))}")
    if ab.affects_from <= 1 and report["global"]["compared"] > 0:
        reached = True
        new = _failing(report, report["global"], g, (1, None), "global")
        if baseline is not None:
            new -= _failing(baseline, baseline["global"], g, (1, None), "global")
        if new:
            hit.append(None)
            msgs.append(f"detected globally: {', '.join(sorted(new))}")
    if not reached:
        status = "n/a"
        msgs.append(f"n/a: case n={report['n']} has no gated region at >= {ab.affects_from}")
    elif leak:
        status = "leak"
    elif hit:
        status = "detected"
    else:
        status = "blind"
        msgs.append(f"BLIND: the gates pass with '{token}' ablated ({ab.what}) -- "
                    f"the harness cannot see this feature at n={report['n']}")
    return {"token": token, "case": report.get("case"), "status": status,
            "ok": None if status == "n/a" else status == "detected", "msgs": msgs}


# ---------------------------------------------------------------------------------
# Schedules for the cached and decode modes
# ---------------------------------------------------------------------------------

DECODE_THRESHOLDS = (512, 1025, 16384)


def uniform_schedule(n: int, chunk: int) -> list[int]:
    if n < 1 or chunk < 1:
        raise ValueError("n and chunk must be >= 1")
    return [min(chunk, n - a) for a in range(0, n, chunk)]


def decode_schedule(n: int, chunk: int = 2048, thresholds = DECODE_THRESHOLDS, head: int = 8,
                    pre: int = 5, post: int = 6, tail: int = 8) -> list[int]:
    """
    Forward sizes covering [0, n). Single-token steps over positions 0 .. head-1 (the
    engram's padded n-grams and the first rate-2 group closures), over [t - pre, t + post)
    around each threshold t < n, and over the last `tail` positions; chunked prefill
    (at most `chunk`) everywhere else. Every single-step run after the head starts at
    an ODD position, i.e. after an odd prefill length, so its first step closes a
    rate-2 group whose first row was carried over from the prefill.
    """
    if n < 1 or chunk < 1 or min(head, pre, post, tail) < 0:
        raise ValueError("n and chunk must be positive; boundary windows must be nonnegative")
    single = np.zeros(n, dtype = bool)
    single[:min(head, n)] = True

    def odd_start(s):
        return max(0, s - 1 if s % 2 == 0 and s > 0 else s)

    for t in thresholds:
        if t < n:
            single[odd_start(t - pre):min(t + post, n)] = True
    single[odd_start(n - tail):] = True
    steps: list[int] = []
    pos = 0
    while pos < n:
        if single[pos]:
            steps.append(1)
            pos += 1
            continue
        e = pos
        while e < n and not single[e] and e - pos < chunk:
            e += 1
        steps.append(e - pos)
        pos = e
    return steps


# ---------------------------------------------------------------------------------
# Next token and printing
# ---------------------------------------------------------------------------------

def gen_token_text(tok) -> tuple[str | None, int | None]:
    """A captured generated token: (text, id) -- id only when captured as 'token_id:N'."""
    if isinstance(tok, str) and tok.startswith("token_id:"):
        try:
            return None, int(tok.split(":", 1)[1])
        except ValueError:
            return tok, None
    return tok, None


def next_token_check(case: Case, our_top_ids, decode = None) -> dict:
    """Information-only next-token comparison; unknown text is not a mismatch.

    Programmatic callers may omit a tokenizer. IDs can still be compared with
    ID-form references, but formatting an ID list is not token decoding.
    """
    ids = [int(t) for t in our_top_ids]
    ours = [decode([t]) for t in ids] if decode is not None else []
    out = {"our_id": ids[0] if ids else None, "our_text": ours[0] if ours else None,
           "ref": None, "match": None, "top5_overlap": None,
           "text_decoding_available": decode is not None}
    if case.gen_tokens:
        text, tid = gen_token_text(case.gen_tokens[0])
        out["ref"] = case.gen_tokens[0]
        if ids and tid is not None:
            out["match"] = ids[0] == tid
        elif ours:
            out["match"] = ours[0] == text
    if ids and case.gen_top_logprobs and case.gen_top_logprobs[0]:
        keys = list(case.gen_top_logprobs[0].keys())
        ref_set = set()
        for k in keys:
            t, i = gen_token_text(k)
            ref_set.add(("id", i) if i is not None else ("text", t))
        our_set = {("id", i) for i in ids} | {("text", t) for t in ours}
        if decode is not None or all(kind == "id" for kind, _ in ref_set):
            out["top5_overlap"] = len(ref_set & our_set) / max(1, len(ref_set))
    return out


def _f(v, spec: str = ".4f") -> str:
    return "-" if v is None else format(v, spec)


def format_report(rep: dict) -> list[str]:
    def row(name, s):
        if not s or not s.get("compared"):
            return f"  {name:<18} {s.get('count', 0) if s else 0:>6} pos   (nothing compared)"
        d, t, tm = s["dlp"], s["top1"], s["top1_margin"]
        tmr = f"{tm['agree']}/{tm['total']}" if tm["total"] else "-"
        return (f"  {name:<18} {s['count']:>6} pos  dNLL {s['dnll']:+.4f}  |dlp| mean {d['mean']:.3f} "
                f"p50 {d['p50']:.3f} p90 {d['p90']:.3f} max {d['max']:.2f}  top1 {t['agree']}/{t['total']} "
                f"(@m>{tm['margin']}: {tmr})  ovl {s['overlap']:.3f}  KL5 {_f(s['kl_top5']['mean'])}")
    out = [f"{rep['label']}: {rep['positions']} positions, coverage {rep['coverage']:.3f}",
           row("all", rep["global"])]
    if rep.get("head"):
        out.append(row(f"< {PRECISION_FROM}", rep["head"]))
    for r in rep["regions"]:
        out.append(row(_rname(r), r))
    return out
