"""
CPU tests for the V4.1 reference-validation harness (no CUDA, well under 2 GiB).

    DSV41_VLLM_REF=<capture.json> python3 tests/test_dsv41_refcompare_.py
    DSV41_VLLM_REF=<capture.json> python -m pytest tests/test_dsv41_refcompare_.py

Reads the vLLM capture the gates were calibrated on (DSV41_VLLM_REF; tools/dsv41_capture_ref.py
writes captures) and skips visibly without it. The harness must PASS vLLM against itself (the
prefix pairs the calibrated limits fail are within the reference's own noise), and FAIL on noise,
on a 5% top-1 swap and on an off-by-one alignment, beyond the noise-shifted limits too --
otherwise its verdicts mean nothing. The mock-model test drives tools/dsv41_validate.py's own forward loops (stateless,
chunked, token-by-token) through a fake model that replays the reference, so the
chunk/position bookkeeping is checked without a GPU. The noise figures (MEASURED_NOISE) belong
to the calibration capture; another capture fails the first test by design.
"""
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import skip, vllm_ref

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = vllm_ref()
VOCAB = 129280


def _load(rel, name):
    if name in sys.modules:
        return sys.modules[name]
    s = importlib.util.spec_from_file_location(name, os.path.join(_H, rel))
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


rc = _load("tools/dsv41_refcompare.py", "dsv41_refcompare")
_CASES = None


class Skip(Exception):
    pass


def cases():
    global _CASES
    if not REF or not os.path.exists(REF):
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("no vLLM capture (set DSV41_VLLM_REF)")
        raise Skip("no vLLM capture (set DSV41_VLLM_REF)")
    if _CASES is None:
        _CASES = rc.load_ref(REF)
    return _CASES


def case(n, rep = 0):
    return next(c for c in cases() if c.n == n and c.rep == rep)


def failing(msgs):
    return [m for m in msgs if m.startswith("FAIL")]


# ---------------------------------------------------------------------------------

def test_rep_vs_rep_passes_and_reproduces_the_measured_noise():
    for n in (24, 300):
        a, b = case(n, 0), case(n, 1)
        for base, other in ((a, b), (b, a)):
            r = rc.compare(base, *rc.as_ours(other))
            ok, msgs = rc.gates(r)
            assert ok, (n, failing(msgs))
            assert rc.verdict(r) == "pass"
    g = rc.compare(case(300, 0), *rc.as_ours(case(300, 1)))["global"]
    want = rc.MEASURED_NOISE[300]
    for key in ("p50", "p90", "p99", "max"):
        assert abs(g["dlp"][key] - want[f"dlp_{key}"]) < 5e-4, (key, g["dlp"][key])
    assert (g["top1"]["agree"], g["top1"]["total"]) == want["top1"], g["top1"]
    assert abs(g["overlap"] - want["overlap"]) < 5e-4, g["overlap"]
    assert abs(g["dnll"] - want["dnll"]) < 5e-4, g["dnll"]
    # every vLLM flip sits at margin <= 0.375: the margin-restricted top-1 is perfect
    assert g["top1_margin"]["agree"] == g["top1_margin"]["total"] > 200
    assert (g["top1_margin"]["agree"], g["top1_margin"]["total"]) == want["top1_margin"], g["top1_margin"]
    assert rc.CALIBRATION_NOISE["top1_margin"] == 1.0 and rc.CALIBRATION_NOISE["dlp_p90"] == want["dlp_p90"]
    assert g["flips_low_margin"] == 6
    g24 = rc.compare(case(24, 0), *rc.as_ours(case(24, 1)))["global"]
    assert abs(g24["dlp"]["mean"] - rc.MEASURED_NOISE[24]["dlp_mean"]) < 5e-4, g24["dlp"]
    assert (g24["top1"]["agree"], g24["top1"]["total"]) == (23, 23)
    pn = {r["label"]: r for r in rc.prefix_noise(cases())}
    assert set(pn) == {"vLLM n=300 prefix vs n=24 rep0", "vLLM n=300 prefix vs n=24 rep1",
                       "vLLM n=1500 prefix vs n=300 rep0", "vLLM n=1500 prefix vs n=300 rep1",
                       "vLLM n=20000 prefix vs n=1500 rep0"}, set(pn)
    assert pn["vLLM n=20000 prefix vs n=1500 rep0"]["positions"] == 1499
    # vLLM's own position-1 logprob depends on the request length; the leave-one-out
    # note shows the n=24 |dNLL| failure is that one position
    p24 = pn["vLLM n=300 prefix vs n=24 rep0"]
    assert p24["global"]["worst_pos"] == 1 and abs(p24["global"]["dnll"]) > 0.1 > 0.01 > abs(p24["global"]["dnll_loo"])
    assert [round(float(case(n).ref_actual_lp[0]), 2) for n in (24, 300, 1500, 20000)] == [-13.5, -10.4, -7.45, -11.26]
    assert any("without position 1" in m for m in rc.gates(p24)[1])
    print(f"  OK  rep0 vs rep1 passes; n=300 |dlp| p50 {g['dlp']['p50']:.4f} p90 {g['dlp']['p90']:.4f} "
          f"p99 {g['dlp']['p99']:.4f} max {g['dlp']['max']:.4f} top1 {g['top1']['agree']}/{g['top1']['total']} "
          f"ovl {g['overlap']:.3f} dNLL {g['dnll']:+.4f}; n=24 mean {g24['dlp']['mean']:.4f}")


def test_noise_shifted_limits_on_the_calibration_capture():
    by = {c.name: c for c in cases()}
    for c in cases():
        noise, excluded = rc.reference_noise(cases(), c)
        assert len(noise) == 5 and not excluded, (c.name, [r["label"] for r in noise])
    # vLLM against itself: the prefix pairs the calibrated limits fail are within its own noise
    for r in rc.prefix_noise(cases()) + rc.self_noise(cases()):
        c = by[r["case"]]
        calibrated = rc.verdict(r)
        judged = with_noise(r, c)
        ok, msgs = rc.gates(judged)
        assert ok and rc.verdict(judged) == ("pass" if calibrated == "pass" else "within-noise"), (r["label"], msgs)
    p = next(r for r in rc.prefix_noise(cases()) if r["label"] == "vLLM n=20000 prefix vs n=1500 rep0")
    msgs = rc.gates(with_noise(p, by["n1500r0"]))[1]
    assert any(m.startswith("FAIL within reference noise  region [1,128) |dlp| p99 1.2405 <= 1.0; "
                            "noise-shifted limit 1.6845") for m in msgs), msgs
    # the limits at n=1500 in region [1,128), where every recorded nc and cached run failed the
    # calibrated limits (p99 1.06-1.45, p90 up to 0.48, top-1 at margin 111/115)
    c = by["n1500r0"]
    lim = rc.noise_limits(with_noise(rc.compare(c, *rc.as_ours(c)), c))
    r1 = next(x["limits"] for x in lim["regions"] if x["lo"] == 1)
    got = {k: round(v[0], 4) for k, v in r1.items()}
    assert got == {"dnll": 0.0931, "top1_margin": 0.9439, "dlp_p50": 0.0869, "dlp_p90": 0.5453,
                   "dlp_p99": 1.6845, "overlap": 0.8665}, got
    assert 111 / 115 >= got["top1_margin"] and 1.45 <= got["dlp_p99"] and 0.48 <= got["dlp_p90"]
    # n=20000: one rep, and the longest capture: nothing measures vLLM against itself on its
    # positions >= 1500, so the whole-case and late-region limits stay calibrated
    c = by["n20000r0"]
    rep = with_noise(rc.compare(c, *rc.as_ours(c)), c)
    lim = rc.noise_limits(rep)
    assert lim["global"] == {} and rep["reference_noise"]["global"] == {}
    assert all(x["limits"] == {} for x in lim["regions"] if x["lo"] >= 1500)
    # position 1 is vLLM's noisiest: 6.05 nats between n=24 and n=1500, at most 1.37 anywhere else
    rep = with_noise(rc.compare(by["n24r0"], *rc.as_ours(by["n24r0"])), by["n24r0"])
    noise_at = {w["pos"]: w["noise"] for w in rep["reference_noise"]["worst"]}
    assert round(noise_at[1], 2) == 6.05 and max(v for p, v in noise_at.items() if p > 1) < 1.37, noise_at
    print(f"  OK  every case has 5 comparisons of vLLM against itself; the prefix pairs that fail the calibrated "
          f"limits are within-noise; n=1500 [1,128) limits {got}; n=20000 keeps the calibrated limits past 1500")


def _noisy(c, sigma, seed):
    """rep c with N(0, sigma) added per (position, token), top-5 re-ranked."""
    rng = np.random.default_rng(seed)
    ids, lp, act = rc.as_ours(c)
    out_ids, out_lp, out_act = ids.copy(), lp.copy(), act.copy()
    for j in range(ids.shape[0]):
        tok = {int(t): float(v) for t, v in zip(ids[j], lp[j]) if t >= 0}
        a = int(c.prompt_ids[j + 1])
        if np.isfinite(act[j]):
            tok.setdefault(a, float(act[j]))
        noisy = {t: v + rng.normal(0, sigma) for t, v in tok.items()}
        top = sorted((t for t in noisy if t in set(ids[j].tolist())), key = lambda t: -noisy[t])[:5]
        out_ids[j, :len(top)] = top
        out_lp[j, :len(top)] = [noisy[t] for t in top]
        out_act[j] = noisy.get(a, np.nan)
    return out_ids, out_lp, out_act


def with_noise(report, c):
    """report judged against the noise-shifted limits of case c, from the whole capture"""
    return rc.attach_noise(report, *rc.reference_noise(cases(), c))


def test_perturbed_copies_fail():
    for n in (24, 300):
        c = case(n, 0)
        r = rc.compare(c, *_noisy(c, 0.5, seed = n))
        ok, msgs = rc.gates(r)
        assert not ok and any("|dlp| p50" in m for m in failing(msgs)), msgs
        # beyond the reference's own noise as well
        ok, msgs = rc.gates(with_noise(r, c))
        assert not ok and any(m.startswith("FAIL  |dlp| p50") for m in msgs), msgs
    c = case(300, 0)
    ids, lp, act = rc.as_ours(c)
    rng = np.random.default_rng(1)
    rows = rng.choice(ids.shape[0], size = int(round(0.05 * ids.shape[0])), replace = False)
    for j in rows:                          # our top-1 and top-2 trade places
        ids[j, [0, 1]] = ids[j, [1, 0]]
        a = c.prompt_ids[j + 1]
        if a == ids[j, 0]:
            act[j] = lp[j, 0]
        elif a == ids[j, 1]:
            act[j] = lp[j, 1]
    r = rc.compare(c, ids, lp, act)
    ok, msgs = rc.gates(r)
    assert not ok and any("top-1" in m for m in failing(msgs)), msgs
    ok, msgs = rc.gates(with_noise(r, c))
    assert not ok and any(m.startswith("FAIL  top-1") for m in msgs), msgs
    print(f"  OK  sigma 0.5 noise fails (|dlp|); 5% top-1 swaps fail (top-1 "
          f"{r['global']['top1_margin']['agree']}/{r['global']['top1_margin']['total']} at margin > 0.375)")


def test_off_by_one_fails():
    for n in (24, 300):
        c = case(n, 0)
        ids, lp, act = rc.as_ours(case(n, 1))
        pos = np.arange(1, n - 1)
        for sh_ids, sh_lp, sh_act, what in ((ids[1:], lp[1:], act[1:], "late"),
                                            (ids[:-1], lp[:-1], act[:-1], "early")):
            p = pos if what == "late" else pos + 1
            ok, msgs = rc.gates(rc.compare(c, sh_ids, sh_lp, sh_act, positions = p))
            assert not ok, (n, what)
            assert not rc.gates(with_noise(rc.compare(c, sh_ids, sh_lp, sh_act, positions = p), c))[0], (n, what)
    print("  OK  alignment shifted by one position either way fails at n=24 and n=300")


def _synthetic_logits(c, positions, vocab = VOCAB):
    """Rows whose log_softmax reproduces the reference's logprobs at the given predicted positions."""
    out = np.empty((len(positions), vocab), dtype = np.float32)
    for r, p in enumerate(positions):
        j = p - 1
        tok = {int(t): float(v) for t, v in zip(c.ref_ids[j], c.ref_lp[j]) if t >= 0}
        a = int(c.prompt_ids[p])
        if np.isfinite(c.ref_actual_lp[j]):
            tok[a] = float(c.ref_actual_lp[j])
        rest = max(1e-30, 1.0 - sum(np.exp(v) for v in tok.values()))
        out[r] = np.log(rest / (vocab - len(tok)))
        for t, v in tok.items():
            out[r, t] = v
    return out


def test_synthetic_logits_reduce_and_align():
    c = case(300, 0)
    pos = np.arange(1, c.n)
    top_ids = np.empty((len(pos), 5), np.int64)
    top_lp = np.empty((len(pos), 5))
    act = np.empty(len(pos))
    at_ref = np.empty((len(pos), 5))
    for s in range(0, len(pos), 64):        # logits[i-1] predict position i
        p = pos[s:s + 64]
        lg = _synthetic_logits(c, p)
        t = rc.reduce_logits(lg, c.prompt_ids[p], c.ref_ids[p - 1, :5])
        top_ids[s:s + 64], top_lp[s:s + 64], act[s:s + 64], at_ref[s:s + 64] = t
    r = rc.compare(c, top_ids, top_lp, act, at_ref_lp = at_ref)
    g = r["global"]
    assert rc.gates(r)[0] and g["dlp"]["max"] < 1e-5 and g["kl_top5"]["max"] < 1e-8, g
    assert g["top1_margin"]["agree"] == g["top1_margin"]["total"]
    # top-1 may only differ where the reference itself has an exact tie
    diff = top_ids[:, 0] != c.ref_ids[:, 0]
    assert np.all(c.margin[diff] == 0), c.margin[diff]
    # the same rows one position off must fail
    r2 = rc.compare(c, top_ids[1:], top_lp[1:], act[1:], at_ref_lp = at_ref[1:], positions = pos[:-1])
    assert not rc.gates(r2)[0]
    # bf16 rounding of the logits stays inside the gates
    lg = _synthetic_logits(c, pos[:64])
    tb = rc.reduce_logits(lg, c.prompt_ids[pos[:64]], bf16 = True)
    rb = rc.compare(c, tb[0], tb[1], tb[2], positions = pos[:64])
    assert not rc.gates(rb)[0] and all("coverage" in m for m in failing(rc.gates(rb)[1]))
    # ties go to even: 1 + 2^-8 -> 1.0, 1 + 3 * 2^-8 -> 1 + 2^-6; past the bf16 max -> inf
    x = np.array([1.0, 1.00390625, 1.01171875, 1.005859375, -3.14159, 65504.0, 3.4e38, np.nan], np.float32)
    want = np.array([1.0, 1.0, 1.015625, 1.0078125, -3.140625, 65536.0, np.inf, np.nan], np.float32)
    got = rc.bf16_round(x)
    assert np.array_equal(got[:-1], want[:-1]) and np.isnan(got[-1]), got
    import torch
    y = np.random.default_rng(0).standard_normal(200000).astype(np.float32) * np.float32(30.0)
    assert np.array_equal(rc.bf16_round(y), torch.from_numpy(y).to(torch.bfloat16).float().numpy())
    print(f"  OK  synthetic full-vocab logits: |dlp| max {g['dlp']['max']:.1e}, KL5 max {g['kl_top5']['max']:.1e}, "
          f"{int(diff.sum())} top-1 differences all at exact reference ties; shifted rows fail; bf16 path passes")


def test_region_counts_n20000():
    c = case(20000, 0)
    r = rc.compare(c, *rc.as_ours(c))
    counts = {(x["lo"], x["hi"]): x["count"] for x in r["regions"]}
    want = {(1, 128): 127, (128, 512): 384, (512, 1025): 513, (1025, 1500): 475,
            (1500, 16384): 14884, (16384, 20000): 3616}
    assert counts == want, counts
    assert sum(counts.values()) == 19999 == r["positions"]
    assert r["head"]["count"] == 511
    assert rc.gates(r)[0] and r["global"]["dlp"]["max"] == 0.0
    short = rc.compare(case(300, 0), *rc.as_ours(case(300, 0)))
    assert [(x["lo"], x["hi"], x["count"]) for x in short["regions"]] == [(1, 128, 127), (128, 512, 172)]
    assert short["head"] is None
    print(f"  OK  n=20000 regions {list(want.values())} (sum 19999); n=300 -> [1,128)x127 [128,512)x172")


def test_missing_entries():
    ids = [5, 6, 7, 8, 9, 10, 11]
    raw = {"n": 7, "rep": 0, "prompt_ids": ids, "prompt_logprobs": [
        None,
        {"6": -0.1, "1": -2.0, "2": -3.0},                                          # 3 entries only
        None,                                                                        # position missing
        {"8": -9.0, "1": -0.5, "2": -1.5, "3": -2.5, "4": -3.5, "0": -4.0},          # actual ranks below 5
        {"1": -0.2, "2": -1.0, "3": -2.0, "4": -3.0, "0": -4.0},                     # actual absent
        {"x": -0.1, "10": -0.3, "1": -1.0, "2": -2.0, "3": -3.0, "4": -4.0},         # a text key
        {"11": -0.05, "1": -3.0, "2": -3.0, "3": -4.0, "4": -5.0}]}
    c = rc.parse_case(raw, 5)
    assert c.ref_ids.shape == (6, 5)
    assert c.ref_ids[0].tolist() == [6, 1, 2, -1, -1] and np.isneginf(c.ref_lp[0, 3:]).all()
    assert (c.ref_ids[1] == -1).all() and np.isnan(c.ref_actual_lp[1])
    assert c.ref_ids[2].tolist() == [1, 2, 3, 4, 0] and c.ref_actual_lp[2] == -9.0
    assert np.isnan(c.ref_actual_lp[3])
    assert c.ref_ids[4].tolist() == [10, 1, 2, 3, 4]
    assert np.isnan(c.margin[1]) and abs(c.margin[0] - 1.9) < 1e-12
    r = rc.compare(c, *rc.as_ours(c))
    g = r["global"]
    assert g["count"] == 6 and g["compared"] == 4 and g["missing_actual"] == 2
    assert g["top1"]["total"] == 5 and g["overlap"] == 5 / 6        # the empty row scores 0
    assert g["dlp"]["max"] == 0.0 and np.isfinite(g["kl_top5"]["mean"])
    # ours may be short of entries too
    oi, ol, oa = rc.as_ours(c)
    oi[0, 1:] = -1
    ol[0, 1:] = -np.inf
    r2 = rc.compare(c, oi, ol, oa)
    assert r2["global"]["overlap"] <= g["overlap"] and r2["global"]["kl_top5"]["mean"] >= 0
    lines = rc.format_report(r2)
    assert len(lines) >= 2
    # a run that left positions without logits, or produced non-finite ones, fails outright
    ok_rep = rc.compare(case(300, 0), *rc.as_ours(case(300, 0)))
    assert rc.gates(ok_rep)[0]
    for key in ("unfilled", "nonfinite"):
        bad = dict(ok_rep, **{key: 3})
        ok, msgs = rc.gates(bad)
        assert not ok and rc.verdict(bad) == "fail", msgs
    print("  OK  missing positions/entries/actual tokens and text keys are handled")


def test_self_compare_and_ties():
    c = case(300, 0)
    a = rc.as_ours(c)
    r = rc.compare_self(a, rc.as_ours(c), n = c.n)
    assert rc.self_gates(r)[0] and r["global"]["dlp"]["max"] == 0.0
    ties = np.where(c.margin == 0)[0]
    assert len(ties) > 0
    b = tuple(x.copy() for x in a)
    b[0][ties[0], [0, 1]] = b[0][ties[0], [1, 0]]     # a flip at an exact tie is excused
    r = rc.compare_self(a, b, n = c.n)
    assert rc.self_gates(r)[0] and r["global"]["top1_diff_ties"] == 1
    hard = np.where(c.margin > 1.0)[0][0]
    b[0][hard, [0, 1]] = b[0][hard, [1, 0]]
    r = rc.compare_self(a, b, n = c.n)
    ok, msgs = rc.self_gates(r)
    assert not ok and r["hard_positions"] == [hard + 1], (msgs, r["hard_positions"])
    b = tuple(x.copy() for x in a)
    b[2][:10] += 0.05                                   # 10/299 > 1% of positions off by 0.05
    ok, msgs = rc.self_gates(rc.compare_self(a, b, n = c.n))
    assert not ok and any("p99" in m for m in failing(msgs)), msgs
    print("  OK  self compare: identical passes, exact-tie flip excused, hard flip and p99 0.05 fail")


def test_self_compare_flip_pairs_missing_rows_and_chunk_boundaries():
    c = case(300, 0)
    a = rc.as_ours(c)
    j = int(np.where(c.margin > 1.0)[0][0])
    X, Z = int(a[0][j, 0]), int(a[0][j, 1])
    top = float(a[1][j, 0])
    # run a: X exactly tied with a THIRD token Z; run b: yet another token Y leads X by 0.9.
    # A tie that does not involve the two tokens that traded places excuses nothing.
    a1 = tuple(x.copy() for x in a)
    a1[1][j, 1] = top
    b = tuple(x.copy() for x in a)
    b[0][j] = [999999, X, Z, 999998, 999997]
    b[1][j] = [top + 0.9, top, top - 0.1, top - 20, top - 30]
    r = rc.compare_self(a1, b, n = c.n)
    ok, msgs = rc.self_gates(r)
    assert not ok and r["global"]["top1_diff_hard"] == 1 and r["hard_positions"] == [j + 1], msgs
    assert r["hard_flips"][0]["gap_a"] is None and abs(r["hard_flips"][0]["gap_b"] - 0.9) < 1e-9, r["hard_flips"]
    # the same flip with X and Y tied in run b is excused; with a 1-ulp gap it is not, unless --tie-eps
    b[1][j] = [top, top, top - 0.1, top - 20, top - 30]
    assert rc.self_gates(rc.compare_self(a1, b, n = c.n))[0]
    b[1][j] = [top + 2**-6, top, top - 0.1, top - 20, top - 30]
    assert not rc.self_gates(rc.compare_self(a1, b, n = c.n))[0]
    assert rc.self_gates(rc.compare_self(a1, b, n = c.n, tie_eps = 0.02))[0]
    # positions one run never produced (NaN / unfilled) fail instead of dropping out
    b = tuple(x.copy() for x in a)
    b[0][::2] = -1
    b[2][::2] = np.nan
    r = rc.compare_self(a, b, n = c.n)
    ok, msgs = rc.self_gates(r)
    assert not ok and r["global"]["missing"] == 150 and any("149/299" in m for m in failing(msgs)), msgs
    # a bug on the first 3 predictions of every 2048-chunk at n=20000 (a lookback lost at each
    # chunk start): 27 of 19999 positions, invisible to the global p99, caught at the boundary
    big = case(20000, 0)
    a = rc.as_ours(big)
    starts = rc.chunk_starts(rc.uniform_schedule(big.n, 2048))
    assert starts == list(range(2048, 20000, 2048))
    b = tuple(x.copy() for x in a)
    for s in starts:
        b[2][s:s + 3] -= 0.3                            # rows s .. s+2 = predicted positions s+1 .. s+3
    r = rc.compare_self(a, b, n = big.n, boundaries = starts)
    ok, msgs = rc.self_gates(r)
    assert r["global"]["dlp"]["p99"] == 0.0 and r["boundary"]["compared"] == 36, (r["global"]["dlp"], r["boundary"])
    assert not ok and [m for m in failing(msgs)] and all("chunk-boundary" in m for m in failing(msgs)), msgs
    assert rc.self_gates(rc.compare_self(a, b, n = big.n))[0]      # without boundaries: blind
    # ...while noise at 0.5% of positions, one of them on a boundary, passes
    b = tuple(x.copy() for x in a)
    rng = np.random.default_rng(7)
    b[2][rng.choice(big.n - 1, 100, replace = False)] += 0.3
    b[2][starts[0]] += 0.3
    ok, msgs = rc.self_gates(rc.compare_self(a, b, n = big.n, boundaries = starts))
    assert ok, msgs
    print("  OK  self compare: a tie with a third token does not excuse a flip; missing rows fail; a chunk-start "
          "bug on 27/19999 positions passes the global p99 and fails the boundary gate")


def _shifted(c, lo, delta):
    ids, lp, act = rc.as_ours(c)
    sel = np.arange(1, c.n) >= lo
    return ids, lp + delta * sel[:, None], act + delta * sel


def test_self_gates_against_a_floor():
    c = case(300, 0)
    a = rc.as_ours(c)
    # two correct paths of a chaotic model: every position moves (sigma 0.15); the floor is the same
    # kind of noise from a row-count change alone
    b = _noisy(c, 0.15, 11)
    floor = rc.compare_self(a, _noisy(c, 0.15, 12), n = c.n)
    r = rc.compare_self(a, b, n = c.n)
    assert not rc.self_gates(r)[0], "the strict gates must fail on floor-level noise"
    ok, msgs = rc.self_gates(r, floor = floor)
    assert ok, msgs
    # the same noise plus a state-carry bug: the first predictions after each chunk start are off by
    # 1 nat; the global numbers barely move, the boundary-vs-rest test must catch it
    starts = list(range(20, c.n, 20))
    bad = tuple(x.copy() for x in b)
    for s0 in starts:
        bad[2][s0:s0 + 3] -= 1.0          # rows s0.. predict positions s0+1.. (a+1..a+3)
    rb = rc.compare_self(a, bad, n = c.n, boundaries = starts)
    ok, msgs = rc.self_gates(rb, floor = floor)
    assert not ok and any("chunk-boundary" in m for m in failing(msgs)), msgs
    # without the bug the boundary positions are as noisy as the rest and pass
    rg = rc.compare_self(a, b, n = c.n, boundaries = starts)
    assert rc.self_gates(rg, floor = floor)[0]
    # hard flips far above the floor's rate fail
    many = tuple(x.copy() for x in b)
    confident = np.where(c.margin > 1.0)[0][:40]
    for j in confident:
        many[0][j, [0, 1]] = many[0][j, [1, 0]]
        many[1][j, [0, 1]] = many[1][j, [0, 0]] + np.array([1.0, 0.0])
    ok, msgs = rc.self_gates(rc.compare_self(a, many, n = c.n), floor = floor)
    assert not ok and any("top-1" in m for m in failing(msgs)), msgs
    print("  OK  floor-relative self gates: floor-level noise passes (strict fails), a boundary bug and "
          "excess flips fail")


def test_ablation_check():
    base = rc.compare(case(300, 0), *rc.as_ours(case(300, 1)))
    abl = rc.compare(case(300, 0), *_noisy(case(300, 1), 0.5, 3))
    assert rc.ablation_check(abl, "engram", base)["status"] == "detected"
    assert rc.ablation_check(base, "engram", base)["status"] == "blind"
    assert rc.ablation_check(abl, "no_candidates", base)["status"] == "n/a"
    big = case(20000, 0)
    b20 = rc.compare(big, *rc.as_ours(big))
    hit = rc.compare(big, *_shifted(big, 16384, -0.2))
    chk = rc.ablation_check(hit, "no_candidates", b20)
    assert chk["status"] == "detected" and chk["ok"], chk
    leak = rc.compare(big, *_shifted(big, 1500, -0.2))
    assert rc.ablation_check(leak, "no_candidates", b20)["status"] == "leak"
    assert rc.ablation_check(b20, "dense_consumers", b20)["status"] == "blind"
    assert rc.ablation_check(rc.compare(big, *_shifted(big, 512, -0.2)), "dense_consumers", b20)["status"] == "detected"
    # a late-only failure is precision-suspect, an early one is not
    assert rc.verdict(hit) == "precision-suspect" and rc.verdict(abl) == "fail"
    # Regional distribution gates detect balanced error even when mean NLL cancels.
    c = case(1500, 0)
    ids, lp, act = rc.as_ours(c)
    pos = np.arange(1, c.n)
    sel = (pos >= 512) & (pos < 1025)
    sign = np.where(np.random.default_rng(0).random(int(sel.sum())) < 0.5, -1.0, 1.0)
    act = act.copy()
    act[sel] += 0.1 * sign
    lp = lp.copy()
    for j in np.flatnonzero(sel):
        lp[j, ids[j] == c.prompt_ids[j + 1]] = act[j]
    quiet = rc.compare(c, ids, lp, act)
    b15 = rc.compare(c, *rc.as_ours(c))
    assert rc.verdict(quiet) == "precision-suspect", rc.gates(quiet)
    assert rc.ablation_check(quiet, "dense_consumers", b15)["status"] == "detected"
    for rep, tok, bl in ((quiet, "dense_consumers", b15), (abl, "engram", base), (hit, "no_candidates", b20),
                         (leak, "no_candidates", b20),
                         (rc.compare(big, *_shifted(big, 512, -0.2)), "dense_consumers", b20)):
        if rc.ablation_check(rep, tok, bl)["status"] == "detected":
            assert rc.verdict(rep) != "pass", (tok, rc.gates(rep))
    print("  OK  ablation check: detected / blind / n/a / leak; late-only failure labelled precision-suspect; "
          "detected implies a failing verdict")


def test_schedules():
    for n in (1, 5, 9, 24, 300, 1500, 20000, 1025, 1026, 513, 514, 16385, 16390, 5000):
        s = rc.decode_schedule(n, 2048)
        assert sum(s) == n and all(x >= 1 for x in s)
        starts = np.cumsum([0] + s[:-1])
        singles = set(starts[np.array(s) == 1].tolist())
        assert set(range(min(8, n))) <= singles
        assert set(range(max(0, n - 8), n)) <= singles
        for t in rc.DECODE_THRESHOLDS:
            if t < n:
                assert {t - 1, t} <= singles, (n, t)
        # every single-step run after the head follows an odd-length prefill
        for x in singles:
            if x > 0 and x - 1 not in singles:
                assert x % 2 == 1, (n, x)
        assert max(s) <= 2048
        assert rc.uniform_schedule(n, 7) == [7] * (n // 7) + ([n % 7] if n % 7 else [])
    print(f"  OK  decode schedule: n=1500 -> {len(rc.decode_schedule(1500))} forwards, "
          f"{rc.decode_schedule(1500).count(1)} single-token; n=20000 -> {len(rc.decode_schedule(20000))}")


# ---------------------------------------------------------------------------------
# tools/dsv41_validate.py
# ---------------------------------------------------------------------------------

def _run_tool(args, timeout = 60):
    env = dict(os.environ, PYTHONPATH = _H)
    return subprocess.run([sys.executable, os.path.join(_H, "tools/dsv41_validate.py")] + args,
                          capture_output = True, text = True, timeout = timeout, env = env)


def test_validate_dry_run_and_help():
    cases()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "plan.json")
        p = _run_tool(["--dry-run", "--mode", "self", "--cases", "24,300,1500,20000", "--ref", REF, "--out", out])
        assert p.returncode == 0, p.stdout + p.stderr
        assert "torch imported: False" in p.stdout, p.stdout
        assert "n20000r0" in p.stdout and "vLLM rep1 vs rep0 n=300: PASS" in p.stdout, p.stdout
        plan = json.load(open(out))["plan"]
        runs = {c["case"]: [r["name"] for r in c["runs"]] for c in plan["cases"]}
        assert runs["n300r0"] == ["nc", "cached2048", "cached1000", "cached7", "decode"]
        assert runs["n20000r0"] == ["cached2048", "cached1000", "cached7", "decode"]   # nc only to 1500
        p = _run_tool(["--dry-run", "--mode", "nc", "--cases", "20000", "--ref", REF, "--out", out])
        assert p.returncode == 2 and "nc mode is limited" in p.stdout
        p = _run_tool(["--dry-run", "--cases", "24", "--ablate", "bogus", "--ref", REF, "--out", out])
        assert p.returncode == 2 and "unknown ablation" in p.stdout
        # the loud isolation warning, the noise-shifted limits, and --strict refusing the bare capture
        p = _run_tool(["--dry-run", "--mode", "cached", "--cases", "1500", "--ref", REF, "--out", out])
        assert p.returncode == 0 and p.stdout.startswith("!!! WARNING: 6 of the capture's 6 cases carry no "
                                                         "isolation metadata"), p.stdout
        assert "reference noise of each gated case" in p.stdout and "p99 <= 1.6845" in p.stdout, p.stdout
        noise = json.load(open(out))["reference_noise"]["n1500r0"]
        assert len(noise["sources"]) == 5 and noise["limits"]["[1,128)"]["dlp_p99"]["limit"] > 1.68
        p = _run_tool(["--dry-run", "--strict", "--mode", "cached", "--cases", "1500", "--ref", REF, "--out", out])
        assert p.returncode == 2 and "ERROR --strict: the gated reference cases n1500r0 carry no isolation" in p.stdout
        assert "not used: n20000r0: no isolation metadata" in p.stdout, p.stdout
    h = _run_tool(["--help"])
    assert h.returncode == 0
    for tok in rc.ABLATIONS:
        assert tok in h.stdout, tok
    for s in ("|dNLL| <= 0.03", "top-1 >= 97%", "p90 <= 0.4", "overlap >= 0.9", "precision-suspect", "p99 <= 0.02",
              "noise-shifted limits", "FAIL within reference noise", "within-noise", "--strict"):
        assert s in h.stdout, s
    print("  OK  validate.py --dry-run plans every mode without importing torch; --help documents gates and ablations")


class _MockState:
    def __init__(self):
        self.position = 0


class _MockCache:
    def __init__(self):
        self.live = 0

    def get_new_state(self):
        self.live += 1
        return _MockState()

    def release_state(self, s):
        self.live -= 1


def _compact(*cs):
    """The same cases over a small vocabulary (ids renumbered), so a mock forward is cheap."""
    import dataclasses
    ids = np.unique(np.concatenate([x for c in cs for x in (c.prompt_ids, c.ref_ids[c.ref_ids >= 0])]))
    remap = np.full(int(ids.max()) + 1, -1, dtype = np.int64)
    remap[ids] = np.arange(len(ids))
    out = [dataclasses.replace(c, prompt_ids = remap[c.prompt_ids],
                               ref_ids = np.where(c.ref_ids >= 0, remap[np.clip(c.ref_ids, 0, None)], -1))
           for c in cs]
    return out, max(len(ids), 1010) + 8


class _MockModel:
    """Replays the reference: the row at absolute input position p predicts position p + 1."""

    def __init__(self, torch, c, lose_position = False, vocab = VOCAB, cached_raises = False):
        self.torch, self.c, self.lose, self.vocab, self.raises = torch, c, lose_position, vocab, cached_raises
        self.modules = []

    def forward(self, ids, params):
        L = ids.shape[1]
        rs = params.get("recurrent_states")
        if rs is not None and self.raises:
            raise RuntimeError("cached path not built")
        p0 = 0 if (rs is None or self.lose) else rs[0].position
        pos = np.arange(p0, p0 + L) + 1                 # predicted positions
        rows = np.full((L, self.vocab), -20.0, dtype = np.float32)
        known = pos < self.c.n
        if known.any():
            rows[known] = _synthetic_logits(self.c, pos[known], self.vocab)
        for r in np.where(~known)[0]:
            rows[r, 1000 + int(pos[r]) % 7] = 5.0       # deterministic continuation
        if rs is not None:
            rs[0].position += L                          # what advance_recurrent_states does
        return self.torch.from_numpy(rows)[None]


def test_validate_forward_loops_on_a_mock_model():
    import torch
    threads = torch.get_num_threads()
    torch.set_num_threads(2)            # tiny tensors: all-core OpenMP only adds contention
    try:
        _mock_loops(torch)
    finally:
        torch.set_num_threads(threads)
    assert not torch.cuda.is_initialized(), "a CPU test initialised CUDA"


def _mock_loops(torch):
    v = _load("tools/dsv41_validate.py", "dsv41_validate_tool")
    (c, c1), V = _compact(case(300, 0), case(300, 1))
    red = {}
    for name, sched in (("nc", None), ("cached64", rc.uniform_schedule(c.n, 64)),
                        ("cached7", rc.uniform_schedule(c.n, 7)), ("decode", rc.decode_schedule(c.n, 64))):
        model, cache = _MockModel(torch, c, vocab = V), _MockCache()
        col = v.Collector(torch, c, rows = 100)
        if sched is None:
            gen = v.run_stateless(torch, model, c, col, 4)
        else:
            gen = v.run_cached(torch, model, cache, c, col, sched, 4, "state")
            assert cache.live == 0
        assert col.filled.all() and col.nonfinite == 0
        assert gen == [1000 + (c.n + k) % 7 for k in range(4)], gen
        r = rc.compare(c, col.top_ids, col.top_lp, col.actual_lp, at_ref_lp = col.at_ref_lp)
        assert rc.gates(r)[0] and r["global"]["dlp"]["max"] < 1e-5, (name, r["global"]["dlp"])
        red[name] = col.reduced()
    for other in ("cached64", "cached7", "decode"):
        ok, msgs = rc.self_gates(rc.compare_self(red["nc"], red[other], n = c.n))
        assert ok, (other, msgs)
    # a cached path that forgets its position: the harness must catch it
    col = v.Collector(torch, c, rows = 100)
    v.run_cached(torch, _MockModel(torch, c, lose_position = True, vocab = V), _MockCache(), c, col,
                 rc.uniform_schedule(c.n, 64), 0, "state")
    r = rc.compare(c, col.top_ids, col.top_lp, col.actual_lp)
    assert not rc.gates(r)[0]
    assert not rc.self_gates(rc.compare_self(red["nc"], col.reduced(), n = c.n))[0]
    # run_case end to end (self mode: scoring, printing, other reps, next token, JSON)
    import contextlib, io
    args = v.build_parser(rc).parse_args(["--mode", "self", "--chunks", "64,37", "--greedy", "3",
                                          "--logit-rows", "100"])
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        rec = v.run_case(torch, rc, _MockModel(torch, c, vocab = V), _MockCache(), None, c, [c1], args, None)
    assert list(rec["runs"]) == ["nc", "cached64", "cached37", "decode"], list(rec["runs"])
    assert all(e["verdict"] == "pass" and e["vs_other_reps"][0]["verdict"] == "pass" for e in rec["runs"].values())
    # with the whole capture, every run is judged against the noise-shifted limits too
    (cc, cc1, c24, c1500), V2 = _compact(case(300, 0), case(300, 1), case(24, 0), case(1500, 0))
    with contextlib.redirect_stdout(io.StringIO()):
        rec2 = v.run_case(torch, rc, _MockModel(torch, cc, vocab = V2), _MockCache(), None, cc, [cc1], args, None,
                          all_cases = [cc, cc1, c24, c1500])
    assert len(rec2["reference_noise"]["sources"]) == 3, rec2["reference_noise"]
    for e in rec2["runs"].values():
        assert e["verdict"] == "pass" and e["report"]["reference_noise"]["sources"], e["gates"]
        assert any(m.startswith("info  reference noise: 3 comparisons") for m in e["gates"]), e["gates"]
    assert [x["ok"] for x in rec["self"]] == [True, True, True]
    assert rec["runs"]["nc"]["greedy_ids"] == [1000 + (c.n + k) % 7 for k in range(3)]
    json.dumps(rec, default = lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    assert "self n300r0: nc vs decode: PASS" in buf.getvalue()
    # a path that raises is recorded and the other runs still complete
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rec = v.run_case(torch, rc, _MockModel(torch, c, vocab = V, cached_raises = True), _MockCache(), None,
                         c, [], args, None)
    assert rec["runs"]["nc"]["verdict"] == "pass" and rec["runs"]["cached64"]["verdict"] == "error"
    assert rec["runs"]["decode"]["error"].startswith("RuntimeError") and rec["self"] == []
    print("  OK  validate.py loops (nc, cached 64/7, decode schedule, greedy) replay the reference exactly on a "
          "mock model; a cached path that loses its position fails both gates; run_case self mode end to end")


class _NanModel(_MockModel):
    """A path whose logits are all NaN, as an fp16 overflow or an eps underflow produces."""

    def forward(self, ids, params):
        out = super().forward(ids, params)
        return self.torch.full_like(out, float("nan"))


def test_validate_nan_logits_boundaries_and_cache_slots():
    import contextlib, io
    import torch
    v = _load("tools/dsv41_validate.py", "dsv41_validate_tool")
    (c, c1), V = _compact(case(300, 0), case(300, 1))
    args = v.build_parser(rc).parse_args(["--mode", "nc", "--greedy", "2"])
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        # an all-NaN path is recorded as a failed run; formatting its empty report used to raise
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rec = v.run_case(torch, rc, _NanModel(torch, c, vocab = V), _MockCache(), None, c, [c1], args, None)
        e = rec["runs"]["nc"]
        assert e["verdict"] == "fail" and not e["ok"] and e["report"]["nonfinite"] == c.n, e["gates"]
        assert any("non-finite" in m for m in e["gates"]) and e["vs_other_reps"][0]["verdict"] == "fail"
        assert "dNLL -" in buf.getvalue()
        # self mode hands each run's chunk starts to compare_self
        args = v.build_parser(rc).parse_args(["--mode", "self", "--chunks", "64,37", "--greedy", "0",
                                              "--logit-rows", "100"])
        with contextlib.redirect_stdout(io.StringIO()):
            rec = v.run_case(torch, rc, _MockModel(torch, c, vocab = V), _MockCache(), None, c, [], args, None)
    finally:
        torch.set_num_threads(threads)
    starts = {s["other"]: s["report"]["boundary_starts"] for s in rec["self"]}
    assert starts == {"cached64": 4, "cached37": 8, "decode": len(rc.decode_schedule(c.n, 64)) - 1}, starts
    assert all(s["ok"] and any("chunk-boundary" in m for m in s["gates"]) for s in rec["self"]), rec["self"]
    # a cached run's slot owns cache_tokens / CACHE_SLOTS pages (slot_bt): the plan must check that
    cases20k = select(20000)
    for extra, want in ((["--cache-tokens", "32768"], "state slots holds 16384"),
                        (["--cache-tokens", "65000"], "not a multiple of the page size"),
                        ([], None), (["--cache-tokens", "40448"], None)):
        p = v.make_plan(v.build_parser(rc).parse_args(["--mode", "cached", "--cases", "20000"] + extra), rc, cases20k)
        assert (want is None and not p["errors"]) or any(want in e for e in p["errors"]), (extra, p["errors"])
    p = v.make_plan(v.build_parser(rc).parse_args(["--mode", "nc", "--cache-tokens", "256"]), rc, [c])
    assert not p["errors"] and not p["needs_cache"]
    print("  OK  an all-NaN path is reported as a failed run; self mode gates the chunk-boundary positions; "
          "the plan rejects a cache whose per-slot share cannot hold the case")


def select(n, rep = 0):
    return [x for x in cases() if x.n == n and x.rep == rep]


def test_sweep_summary():
    v = _load("tools/dsv41_validate.py", "dsv41_validate_tool")
    c = case(300, 0)

    def res(r):
        r = dict(r, nonfinite = 0, unfilled = 0)
        ok = rc.gates(r)[0]
        return {"ok": ok, "cases": [{"case": c.name, "runs": {"nc": {"ok": ok, "report": r}}}]}

    base = rc.compare(c, *rc.as_ours(case(300, 1)))
    noisy = rc.compare(c, *_noisy(case(300, 1), 0.5, 5))
    summary, lines, ok = v.sweep_summary(rc, ["engram", "no_candidates"],
                                         {None: res(base), "engram": res(noisy), "no_candidates": res(base)})
    assert not ok and summary["tokens"]["engram"][0]["status"] == "detected"
    assert summary["tokens"]["no_candidates"][0]["status"] == "n/a"
    assert any("never exercised" in l for l in lines)
    assert v.sweep_summary(rc, ["engram"], {None: res(base), "engram": res(noisy)})[2]
    assert not v.sweep_summary(rc, [], {None: res(base)})[2]
    failed = res(base)
    failed["ok"] = False
    assert not v.sweep_summary(rc, ["engram"], {None: failed, "engram": res(noisy)})[2]
    summary, lines, ok = v.sweep_summary(rc, ["v4mix", "engram"], {None: res(base), "v4mix": res(base), "engram": None})
    assert not ok and summary["tokens"]["v4mix"][0]["status"] == "blind"
    assert any("engram" in l and "no report" in l for l in lines)
    print("  OK  sweep summary: detected / n/a (never exercised) / blind / missing report")


# ---------------------------------------------------------------------------------
# tools/dsv41_capture_ref.py, against a local stand-in for the service
# ---------------------------------------------------------------------------------

class _Fake(http.server.BaseHTTPRequestHandler):
    busy_polls = 2
    bodies: list = []

    def log_message(self, *a):
        pass

    def _send(self, obj, ctype = "application/json"):
        data = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/metrics":
            busy = 1 if _Fake.busy_polls > 0 else 0
            _Fake.busy_polls -= 1
            self._send(f'# HELP x\nvllm:num_requests_running{{engine="0",model_name="ds41"}} {busy}.0\n'
                       f'vllm:num_requests_waiting{{engine="0",model_name="ds41"}} 0.0 1700000000\n', "text/plain")
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "ds41"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Fake.bodies.append(body)
        ids = body["prompt"]
        pl = [None] + [{str(t): {"logprob": -0.5, "rank": 1, "decoded_token": "x"},
                        "7": {"logprob": -1.5, "rank": 2, "decoded_token": "y"}} for t in ids[1:]]
        self._send({"choices": [{"text": "z", "prompt_logprobs": pl, "token_ids": [42],
                                 "logprobs": {"tokens": ["token_id:42"], "top_logprobs": [{"token_id:42": -0.1}]}}]})


def test_capture_tool_against_a_fake_service():
    cap = _load("tools/dsv41_capture_ref.py", "dsv41_capture_tool")
    m = cap.parse_metrics('a{x="}"} 3\na{y="2"} 4 1700000000\nb 1.5\n# c 9\n')
    assert m == {"a": 7.0, "b": 1.5}, m
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target = srv.serve_forever, daemon = True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        assert cap.wait_idle(url, 10, poll = 0.01, log = lambda *a: None) >= 0 and _Fake.busy_polls < 0
        with tempfile.TemporaryDirectory() as d:
            ids_file = os.path.join(d, "ids.json")
            json.dump(list(range(100, 140)), open(ids_file, "w"))
            out = os.path.join(d, "cap.json")
            assert cap.main(["--url", url, "--lengths", "12,30", "--reps", "2", "--out", out,
                             "--ids-file", ids_file, "--idle-timeout", "10"]) == 0
            doc = json.load(open(out))
            assert [(c["n"], c["rep"]) for c in doc["cases"]] == [(12, 0), (12, 1), (30, 0), (30, 1)]
            assert doc["meta"]["top"] == 20 and all(b["max_tokens"] == 1 and b["prompt_logprobs"] == 20
                                                    and b["temperature"] == 0 for b in _Fake.bodies)
            loaded = rc.load_ref(out)
            assert loaded[0].top == 20 and loaded[0].ref_ids[0, :2].tolist() == [101, 7]
            assert rc.gen_token_text(loaded[0].gen_tokens[0]) == (None, 42)
            assert cap.main(["--url", url, "--lengths", "12", "--out", out, "--ids-file", ids_file]) == 2
            existing = os.path.join(d, "existing.json")
            with open(existing, "w") as f:
                f.write("{}")
            assert cap.main(["--url", url, "--lengths", "12", "--out", existing, "--ref", existing,
                             "--dry-run"]) == 2
    finally:
        srv.shutdown()
    print("  OK  capture tool: idle-gated, one request at a time, top-20 token-id case written to a new file "
          "that load_ref reads; refuses to overwrite")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    t0 = time.time()
    if not REF or not os.path.exists(REF):
        skip("V4.1 reference validation harness", "no vLLM capture (set DSV41_VLLM_REF)")
        return 0
    order = [test_rep_vs_rep_passes_and_reproduces_the_measured_noise,
             test_noise_shifted_limits_on_the_calibration_capture, test_perturbed_copies_fail,
             test_off_by_one_fails, test_synthetic_logits_reduce_and_align, test_region_counts_n20000,
             test_missing_entries, test_self_compare_and_ties,
             test_self_compare_flip_pairs_missing_rows_and_chunk_boundaries, test_self_gates_against_a_floor,
             test_ablation_check, test_schedules,
             test_validate_dry_run_and_help, test_validate_forward_loops_on_a_mock_model,
             test_validate_nan_logits_boundaries_and_cache_slots, test_sweep_summary,
             test_capture_tool_against_a_fake_service]
    assert set(order) == set(TESTS), "a test is missing from the run order"
    for t in order:
        t1 = time.time()
        t()
        if os.environ.get("DSV41_TEST_TIMES"):
            print(f"      {t.__name__}: {time.time() - t1:.2f}s")
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"ALL PASS ({len(order)} tests) in {time.time() - t0:.1f}s, peak RSS {rss:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
