"""CPU-only: the DeepSeek-V4.1 attention numerics (architecture/dsv41/numerics.py) and their
rounding kernels (modules/dsv41_rounding.py) against independent scalar oracles, the contract
dispatch, the setting's syntax, and the config attribute that holds it; the overflow policy (a
block that would overflow FP16, or whose operand is not finite, keeps its unrounded values and
is counted, with one warning; EXL3_DSV41_NUMERICS_STRICT refuses instead), bitwise equal to the
strict path on finite operands whose rounded values fit.

    python -m pytest tests/test_dsv41_numerics_.py      or      python tests/test_dsv41_numerics_.py

The rounding kernels load without the compiled extension. The config-attribute test imports
exllamav3 and fails, naming the cause, where the package cannot be imported (a missing or
broken compiled extension included): nothing in this file is skipped."""

import json
import os
import subprocess
import sys
import weakref
from fractions import Fraction

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_numerics

N, R = load_numerics()
P = N.parse

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_L = [(v, i % 2) for i, v in enumerate(E2M1)]       # index parity is the mantissa bit
# every finite E4M3 (fn) magnitude with its mantissa bits: e = 0 subnormal, e = 15 m = 7 is NaN
E4M3 = sorted({((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7), m)
               for e in range(16) for m in range(8) if not (e == 15 and m == 7)})
f32 = np.float32


def nearest(lattice, a):
    """Nearest lattice magnitude to a >= 0, ties to even mantissa. lattice: [(value, mantissa)]."""
    return min(lattice, key = lambda q: (abs(Fraction(q[0]) - Fraction(float(a))), q[1] % 2))[0]


def pow2_ceil(t):
    """kernel.py fast_round_scale on an fp32 value: 2^(exponent + (mantissa != 0))."""
    m, e = np.frexp(np.float64(t))
    return np.float64(2.0) ** (e - 1 if m == 0.5 else e)


def o_mxfp4(v):
    amax = max(np.max(np.abs(v)), f32(6 * 2.0 ** -126))
    s = pow2_ceil(f32(amax) * f32(1 / 6))
    return [np.sign(x) * nearest(E2M1_L, min(abs(np.float64(x)) / s, 6.0)) * s for x in v]


def o_nvfp4(v):
    amax = max(np.max(np.abs(v)), f32(6 * 2.0 ** -9))
    t = min(float(f32(f32(amax) / f32(6.0))), 448.0)
    s = nearest(E4M3, t)
    out = []
    for x in v:
        y = float(np.clip(f32(x) / f32(s), -6.0, 6.0))
        out.append(np.sign(y) * nearest(E2M1_L, abs(y)) * s)
    return out


def o_fp8_ue8m0(v):
    amax = max(np.max(np.abs(v)), f32(1e-4))
    s = pow2_ceil(f32(amax) * f32(1 / 448))
    return [np.sign(x) * nearest(E4M3, min(abs(np.float64(x)) / s, 448.0)) * s for x in v]


def o_ds_mla(v):
    peak = max(float(np.max(np.abs(v.astype(np.float64)))), 1e-4)
    e = int(np.frexp(peak / 448.0)[1]) - 1
    if 448.0 * 2.0 ** e < peak:
        e += 1
    s = 2.0 ** e
    return [np.sign(x) * nearest(E4M3, min(abs(np.float64(x)) / s, 448.0)) * s for x in v]


def blocks(rows, width, fn):
    return np.stack([np.concatenate([fn(r[i:i + width]) for i in range(0, len(r), width)]) for r in rows])


def check(name, got, want):
    bad = np.nonzero(got.numpy() != want.astype(np.float32))
    assert bad[0].size == 0, f"{name}: {bad[0].size} mismatches, first at {bad[0][0]},{bad[1][0]}"


def operand_rows():
    """24 random rows over five decades, plus a row of E2M1 tie values, scaled ties, a tiny row
    (scale floors) and saturation at 448."""
    rng = np.random.default_rng(11)
    rows = [rng.standard_normal(512) * 10 ** rng.uniform(-3, 2.5) for _ in range(24)]
    ties = np.resize(np.array([6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -2.5, 0, -0.0, 448,
                               1.0625, 1.1875]), 512)
    rows += [ties, ties * 2.0 ** -7, np.full(512, 1e-6), np.resize(np.array([448 * 2 ** -3, 1.0]), 512)]
    return np.stack(rows).astype(np.float32)


E2M1_MID = [(a + b) / 2 for a, b in zip(E2M1, E2M1[1:])]              # 7 midpoints
E4M3_V = [v for v, _ in E4M3]
E4M3_MID = [(a + b) / 2 for a, b in zip(E4M3_V, E4M3_V[1:])]          # 126 midpoints


def tie_rows():
    """
    Every lattice midpoint at scale 1, where it is a tie of the rounding itself. In operand_rows
    the tie values share their blocks with 448, which moves the block scale off 1. Here:
    E2M1 midpoints (+-) in 32-lane blocks with peak 6 (MXFP4 scale 2^ceil(log2(6 / 6)) = 1) and
    in 16-lane blocks with peak 6 (NVFP4 scale e4m3(6 / 6) = 1); the E4M3 midpoints, and their
    negations, in 32-lane blocks that each start with a +-448 peak (FP8 E8M0 scale 1, and 64-lane
    fp8_ds_mla scale 1).
    """
    mids = E2M1_MID
    mx = np.tile(np.resize(np.array([6.0] + mids + [-m for m in mids] + [0.0]), 32), 16)
    nv = np.tile(np.array([6.0] + mids + [-m for m in mids] + [-6.0]), 32)
    e4 = np.concatenate([np.concatenate([[448.0], E4M3_MID[i:i + 31]]) for i in range(0, len(E4M3_MID), 31)])
    e4 = np.concatenate([e4, np.full(-len(e4) % 32, 448.0)])            # whole 32-lane blocks
    e4 = np.resize(np.concatenate([e4, -e4]), 512)                       # every tie negated too
    return np.stack([mx, nv, e4]).astype(np.float32)


def _scale_hits(rows, width, scale, mids):
    """The midpoints of a lattice that some lane of `rows` lands on after its block's scale."""
    mids = set(mids)
    hit = set()
    for r in rows.astype(np.float64):
        for i in range(0, r.size, width):
            b = r[i:i + width]
            hit |= mids & set((np.abs(b) / scale(b)).tolist())
    return hit


def _s_mx(b):
    return pow2_ceil(f32(max(np.max(np.abs(b)), f32(6 * 2.0 ** -126))) * f32(1 / 6))


def _s_nv(b):
    return nearest(E4M3, min(float(f32(f32(max(np.max(np.abs(b)), f32(6 * 2.0 ** -9))) / f32(6.0))), 448.0))


def _s_ue(b):
    return pow2_ceil(f32(max(np.max(np.abs(b)), f32(1e-4))) * f32(1 / 448))


def _s_ml(b):
    peak = max(float(np.max(np.abs(b))), 1e-4)
    e = int(np.frexp(peak / 448.0)[1]) - 1
    return 2.0 ** (e + 1 if 448.0 * 2.0 ** e < peak else e)


def test_tie_rows_cover_every_midpoint():
    """The oracle comparison below sees every tie of every lattice, so a wrong tie rule fails it."""
    t = tie_rows()
    assert _scale_hits(t[:1], 32, _s_mx, E2M1_MID) == set(E2M1_MID), "MXFP4 ties"
    assert _scale_hits(t[1:2], 16, _s_nv, E2M1_MID) == set(E2M1_MID), "NVFP4 ties"
    assert _scale_hits(t[2:], 32, _s_ue, E4M3_MID) == set(E4M3_MID), "FP8 E8M0 ties"
    assert _scale_hits(t[2:, :448], 64, _s_ml, E4M3_MID) == set(E4M3_MID), "fp8_ds_mla ties"
    assert len(E2M1_MID) == 7 and len(E4M3_MID) == 126


@pytest.fixture(autouse = True)
def default_policy():
    """Every test starts under the default overflow policy, whatever EXL3_DSV41_NUMERICS_STRICT
    says, with nothing counted, and leaves the switch as it found it."""
    previous = R.set_strict(False)
    R.reset_fallbacks()
    yield
    R.set_strict(previous)
    R.reset_fallbacks()


@pytest.mark.parametrize("strict", (False, True), ids = ("default", "strict"))
def test_quantizers_match_scalar_oracles(strict):
    R.set_strict(strict)
    x = np.concatenate([operand_rows(), tie_rows()])
    check("mxfp4", R.mxfp4_(torch.tensor(x)), blocks(x, 32, o_mxfp4))
    check("nvfp4", R.nvfp4_(torch.tensor(x)), blocks(x, 16, o_nvfp4))
    check("fp8_ue8m0", R.fp8_ue8m0_(torch.tensor(x)), blocks(x, 32, o_fp8_ue8m0))
    check("fp8_ds_mla", R.fp8_ds_mla_nope_(torch.tensor(x[:, :448].copy())), blocks(x[:, :448], 64, o_ds_mla))
    assert torch.equal(R.bf16_(torch.tensor(x)), torch.tensor(x).to(torch.bfloat16).float())
    assert R.fallback_count() == 0


def test_rounding_in_place_idempotent_and_fp16():
    x = operand_rows()
    t = torch.tensor(x)
    assert R.mxfp4_(t) is t, "the kernels round in place"
    # power-of-two scales are idempotent; NVFP4's E4M3 scale is not (a max that rounds to 4s
    # rescales to e4m3(4s/6) on a second pass), exactly like the reference kernel
    for fn in (R.mxfp4_, R.fp8_ue8m0_):
        once = fn(torch.tensor(x))
        assert torch.equal(fn(once.clone()), once), f"{fn.__name__} is not idempotent"
    h = torch.tensor(x).half()
    assert torch.equal(R.mxfp4_(h.clone()).float(), R.mxfp4_(h.float()).half().float()), "fp16 write-back"


def refusal_cases():
    """(kernel, lanes, value, reason): a nonfinite operand for every kernel, and for every kernel
    that can overflow FP16 the smallest FP16 value it rounds past 65,504 (NVFP4 saturates at
    2,688 instead)."""
    kernels = ((R.mxfp4_, 64), (R.nvfp4_, 64), (R.fp8_ue8m0_, 64), (R.fp8_ds_mla_nope_, 448), (R.bf16_, 64))
    cases = [(fn, lanes, v, "nonfinite") for fn, lanes in kernels for v in (float("nan"), float("inf"))]
    cases += [(R.mxfp4_, 64, 57344.0, "overflows"), (R.fp8_ue8m0_, 64, 63488.0, "overflows"),
              (R.fp8_ds_mla_nope_, 448, 63488.0, "overflows"), (R.bf16_, 64, 65408.0, "overflows")]
    return cases


# the strict switch's refusal messages, checked word for word
NONFINITE = "dsv41 numerics: nonfinite operand"
OVERFLOW = "dsv41 numerics: rounded value overflows the operand dtype"

# (kernel, lanes per block, lanes per operand row): bf16_ has no shared scale, so each lane is a block
KERNELS = ((R.mxfp4_, 32, 64), (R.nvfp4_, 16, 64), (R.fp8_ue8m0_, 32, 64), (R.fp8_ds_mla_nope_, 64, 448),
           (R.bf16_, 1, 64))
# FP16 values each kernel rounds past 65,504: the smallest one and FP16's largest (NVFP4 saturates)
OVERFLOWING = {R.mxfp4_: (57344.0, 65504.0), R.nvfp4_: (), R.fp8_ue8m0_: (63488.0, 65504.0),
               R.fp8_ds_mla_nope_: (63488.0, 65504.0), R.bf16_: (65408.0, 65504.0)}


def ibits(t):
    """The bit patterns of t (NaN payloads included), for bitwise comparisons."""
    return t.view(torch.int16 if t.dtype == torch.half else torch.int32)


def test_strict_refusals_raise_and_leave_the_operand_unchanged():
    R.set_strict(True)
    for fn, lanes, value, why in refusal_cases():
        x = torch.ones(2, lanes, dtype = torch.half)
        x[1, 5] = value
        before = x.clone()
        with pytest.raises(ValueError) as e:
            fn(x)
        assert str(e.value) == (NONFINITE if why == "nonfinite" else OVERFLOW), (fn.__name__, value, str(e.value))
        assert torch.equal(x.view(torch.int16), before.view(torch.int16)), f"{fn.__name__} wrote before refusing"
    with pytest.raises(ValueError, match = "nonfinite"):
        R.bf16_(torch.tensor([1.0, float("nan")]))
    assert R.fallback_count() == 0, "a refusal was counted as a fallback"


@pytest.mark.parametrize("strict", (False, True), ids = ("default", "strict"))
def test_largest_values_that_stay_within_fp16(strict):
    """The largest FP16 values each kernel rounds without overflowing, and NVFP4's saturation:
    rounded under both policies, nothing kept or counted."""
    R.set_strict(strict)
    for fn, lanes, value, want in ((R.mxfp4_, 64, 57312.0, 49152.0), (R.fp8_ue8m0_, 64, 63456.0, 61440.0),
                                   (R.fp8_ds_mla_nope_, 448, 63456.0, 61440.0), (R.bf16_, 64, 65376.0, 65280.0),
                                   (R.nvfp4_, 64, 65504.0, 2688.0)):
        x = torch.ones(2, lanes, dtype = torch.half)
        x[0, 0] = value
        fn(x)
        assert x[0, 0] == want, (fn.__name__, value, float(x[0, 0]))
    assert R.fallback_count() == 0


@pytest.mark.parametrize("slice_blocks", (None, 7), ids = ("one-pass", "sliced"))
def test_fallback_keeps_the_unrounded_blocks(monkeypatch, slice_blocks):
    """
    Default policy: a block holding a nonfinite value, or a value its rounding carries past
    FP16, keeps its unrounded values bit for bit (NaN payloads included), every other block is
    rounded exactly as the strict path rounds it, and the count grows by the number of blocks
    kept. Several bad blocks per operand, in different rows (and slices).
    """
    if slice_blocks:
        monkeypatch.setattr(R, "SLICE_BLOCKS", slice_blocks)
    rng = np.random.default_rng(3)
    n_cases = 0
    for fn, block, lanes in KERNELS:
        for value in ("nan", float("inf"), float("-inf")) + OVERFLOWING[fn]:
            x = torch.tensor(rng.standard_normal((40, lanes)) * 30).half()
            pos = ((0, 0), (7, lanes - 1), (19, 3), (19, 3 + block), (39, lanes // 2))
            for r, c in pos:
                if value == "nan":
                    # quiet NaNs with payloads, negative in row 19: a scalar assignment would
                    # canonicalize them
                    bits = (0x7E01 + r + c) | (0x8000 if r == 19 else 0)
                    x.view(torch.int16)[r, c] = bits - 0x10000 if bits >= 0x8000 else bits
                else:
                    x[r, c] = -value if r == 19 else value
            kept = {(r, c // block) for r, c in pos}
            mask = torch.zeros_like(x, dtype = torch.bool)
            benign = x.clone()
            for r, b in kept:
                mask[r, b * block:(b + 1) * block] = True
                benign[r, b * block:(b + 1) * block] = 1.0
            R.set_strict(True)
            rounded = fn(benign)                 # blocks are independent: the good ones as strict rounds them
            R.set_strict(False)
            want = torch.where(mask, x, rounded)
            before = R.fallback_count()
            got = fn(x.clone())
            assert torch.equal(ibits(got), ibits(want)), (fn.__name__, value)
            assert R.fallback_count() - before == len(kept), (fn.__name__, value, R.fallback_count() - before)
            n_cases += 1
    assert n_cases == sum(3 + len(v) for v in OVERFLOWING.values()) == 23
    # the limits are the operand dtype's: in FP32 only the nonfinite block is kept
    x = torch.ones(2, 64)
    x[0, 0], x[1, 40] = 60000.0, float("nan")
    before = R.fallback_count()
    got = R.mxfp4_(x.clone())
    assert got[0, 0] == 65536.0 and torch.equal(ibits(got[1, 32:]), ibits(x[1, 32:]))
    assert R.fallback_count() - before == 1


def test_fallback_through_the_entry_points():
    """round_index_, round_window_ and round_compressed_ keep and count per block of the kernel
    that rounds each lane: under vllm the NoPE lanes in 64-lane groups and the RoPE lanes one
    by one, under deepseek 32-lane (index, window) or 16-lane (compressed) blocks; a block that
    NVFP4's saturation clamps is rounded, not kept. Every other block as the strict path rounds it."""
    x = torch.tensor(operand_rows()[:4]).half().contiguous()
    x[x.abs() > 1000] = 1.0
    x[1, 10] = 64000.0                  # NoPE lane: past FP16 under MXFP4, FP8 and fp8_ds_mla
    x[2, 470] = float("nan")            # RoPE lane
    x[3, 500] = 65440.0                 # RoPE lane: past FP16 under MXFP4, FP8 and BF16
    wide = [(1, 0, 32), (2, 448, 480), (3, 480, 512)]
    ds_mla = [(1, 0, 64), (2, 470, 471), (3, 500, 501)]
    cases = (("deepseek:window", wide), ("vllm:window", ds_mla), ("deepseek:compressed", [(2, 464, 480)]),
             ("vllm:compressed", ds_mla), ("deepseek:index", wide), ("vllm:index", wide), ("precise", []))

    def run(pol, a):
        if pol.compressed:
            return torch.cat(R.round_compressed_(pol, a[:, :448].contiguous(), a[:, 448:].contiguous()), 1)
        return R.round_index_(pol, a) if pol.index else R.round_window_(pol, a, 64)

    for setting, kept in cases:
        pol = P(setting)
        benign = x.clone()
        for r, a, b in kept:
            benign[r, a:b] = 1.0
        R.set_strict(True)
        want = run(pol, benign)
        R.set_strict(False)
        for r, a, b in kept:
            want[r, a:b] = x[r, a:b]
        before = R.fallback_count()
        got = run(pol, x.clone())
        assert torch.equal(ibits(got), ibits(want)), setting
        assert R.fallback_count() - before == len(kept), (setting, R.fallback_count() - before)


@pytest.mark.parametrize("slice_blocks", (None, 7), ids = ("one-pass", "sliced"))
def test_default_equals_strict_on_finite_operands(monkeypatch, slice_blocks):
    """
    On finite operands whose rounded values fit, the default policy gives bitwise the values of
    the strict path, which checks the whole operand at once: the oracle rows, and random rows at
    scales from 1e-9 to 2e4, clipped to +-50,000, in FP32 and FP16, through every kernel and
    every entry point of every setting. Nothing is counted.
    """
    if slice_blocks:
        monkeypatch.setattr(R, "SLICE_BLOCKS", slice_blocks)
    rng = np.random.default_rng(29)
    rand = np.stack([rng.standard_normal(512) * 10 ** rng.uniform(-9, 4.3) for _ in range(400)])
    x = np.concatenate([operand_rows(), tie_rows(), np.clip(rand, -50000, 50000)]).astype(np.float32)
    settings = [P(s) for s in ("deepseek", "vllm", "deepseek:index", "vllm:window", "deepseek:compressed")]
    n = 0
    for dtype in (torch.float32, torch.float16):
        t = torch.tensor(x).to(dtype)
        runs = [(fn.__name__, lambda a, fn = fn: fn(a[:, :448].contiguous() if fn is R.fp8_ds_mla_nope_ else a))
                for fn, _, _ in KERNELS]
        for pol in settings:
            runs += [(f"{pol} index", lambda a, pol = pol: R.round_index_(pol, a)),
                     (f"{pol} window", lambda a, pol = pol: R.round_window_(pol, a, 64)),
                     (f"{pol} compressed", lambda a, pol = pol: torch.cat(
                         R.round_compressed_(pol, a[:, :448].contiguous(), a[:, 448:].contiguous()), 1))]
        for name, run in runs:
            R.set_strict(True)
            want = run(t.clone())
            R.set_strict(False)
            got = run(t.clone())
            assert torch.equal(ibits(got), ibits(want)), (name, dtype)
            n += 1
    assert n == 2 * (5 + 3 * len(settings))
    assert R.fallback_count() == 0


def test_the_count_and_the_one_warning(capsys):
    """fallback_count() sums the blocks kept by every call; the first nonzero count prints one
    warning, never a second one until reset_fallbacks(); rounding without a fallback prints
    nothing."""
    x = torch.ones(4, 64, dtype = torch.half)
    for _ in range(3):
        R.mxfp4_(x.clone())
        R.bf16_(x.clone())
    assert R.fallback_count() == 0 and "!!" not in capsys.readouterr().out
    bad = x.clone()
    bad[0, 0], bad[2, 40], bad[3, 1], bad[3, 2] = float("nan"), 60000.0, float("inf"), 5.0
    R.mxfp4_(bad.clone())                                       # blocks (0, 0), (2, 1), (3, 0)
    assert R.fallback_count() == 3
    out = capsys.readouterr().out
    assert out.count(" !! DSV41 numerics:") == 1 and " 3 block(s) " in out and f"{N.ENV_STRICT}=1" in out, out
    for _ in range(4):
        R.mxfp4_(bad.clone())
        R.nvfp4_(bad.clone())                                   # 60000 saturates: 2 blocks
        R.bf16_(bad.clone())                                    # lanes: nan and inf, 2
    assert R.fallback_count() == 3 + 4 * (3 + 2 + 2)
    assert "!!" not in capsys.readouterr().out, "warned more than once"
    R.reset_fallbacks()
    assert R.fallback_count() == 0
    R.nvfp4_(bad.clone())
    assert R.fallback_count() == 2 and capsys.readouterr().out.count(" !! DSV41 numerics:") == 1
    # the forward runs under inference mode, tests and tools often outside it: a count created
    # in either mode keeps counting in the other
    for first, second in ((torch.inference_mode, torch.no_grad), (torch.no_grad, torch.inference_mode)):
        R.reset_fallbacks()
        with first():
            R.mxfp4_(bad.clone())
        with second():
            R.mxfp4_(bad.clone())
        R.mxfp4_(bad.clone())
        assert R.fallback_count() == 9, (first.__name__, R.fallback_count())


# a process whose last rounding call keeps one block ("kept", "lagged") or none ("clean"), under
# the default policy; "lagged" stubs out the looks at the count, as a GPU's count is unread
# between two looks
EXIT_SCRIPT = """
import sys
import torch
sys.path.insert(0, {tests!r})
from dsv41_ref import load_numerics
_, R = load_numerics()
mode, device = sys.argv[1], sys.argv[2]
if mode == "lagged":
    R._poll = lambda fb, device: None
x = torch.ones(2, 64, dtype = torch.half, device = device)
if mode != "clean":
    x[1, 3] = float("nan")
R.mxfp4_(x)
print("END", flush = True)
"""


def run_to_exit(mode: str, device: str = "cpu") -> list[str]:
    """The END line and the warning lines a fresh process prints, in order, exit included."""
    env = {k: v for k, v in os.environ.items() if k != N.ENV_STRICT}
    script = EXIT_SCRIPT.format(tests = os.path.dirname(os.path.abspath(__file__)))
    p = subprocess.run([sys.executable, "-c", script, mode, device], env = env, capture_output = True,
                       text = True, timeout = 600)
    assert p.returncode == 0, f"{mode} {device}: exit {p.returncode}: {p.stderr[-3000:]}"
    return [l for l in p.stdout.splitlines() if l == "END" or "!! DSV41 numerics:" in l]


def test_the_warning_at_exit():
    """
    A count that no look has read when the interpreter exits (blocks kept by the process's last
    rounding calls) is reported at exit, once: with the looks stubbed out, the warning follows
    the process's last output. A warning printed while the process ran is not printed again at
    exit, and a process that kept nothing prints none.
    """
    lagged = run_to_exit("lagged")
    assert len(lagged) == 2 and lagged[0] == "END" and " 1 block(s) " in lagged[1], lagged
    kept = run_to_exit("kept")
    assert len(kept) == 2 and kept[1] == "END" and " 1 block(s) " in kept[0], kept
    assert run_to_exit("clean") == ["END"]


def test_the_strict_switch_is_read_once(monkeypatch):
    """EXL3_DSV41_NUMERICS_STRICT: on for any value but '' and '0', read at the first call and
    not again; set_strict() (tests only) returns the previous setting, None re-reads."""
    assert N.ENV_STRICT == "EXL3_DSV41_NUMERICS_STRICT"
    for value, want in (("1", True), ("0", False), ("", False), (" 0 ", False), ("yes", True)):
        monkeypatch.setenv(N.ENV_STRICT, value)
        R.set_strict(None)
        assert R.strict() is want, value
        monkeypatch.setenv(N.ENV_STRICT, "0" if want else "1")
        assert R.strict() is want, f"{value!r}: read again after the first call"
    monkeypatch.delenv(N.ENV_STRICT)
    R.set_strict(None)
    assert R.strict() is False
    assert R.set_strict(True) is False and R.set_strict(None) is True and R.set_strict(False) is None
    # the variable itself switches the refusal on
    monkeypatch.setenv(N.ENV_STRICT, "1")
    R.set_strict(None)
    x = torch.ones(1, 32, dtype = torch.half)
    x[0, 3] = 60000.0
    with pytest.raises(ValueError) as e:
        R.mxfp4_(x)
    assert str(e.value) == OVERFLOW and x[0, 3] == 60000.0


def test_no_host_read_on_the_default_path(monkeypatch):
    """
    The default policy decides, keeps and counts on the device. Meta tensors have no values, so
    any host read (item, tolist, a branch on a tensor, boolean-mask indexing) fails on them:
    every kernel and every entry point runs on meta operands, in one pass and in slices, while
    the strict path, which reads its flags back once per call, fails there.
    """
    meta = lambda *shape: torch.empty(*shape, dtype = torch.half, device = "meta")
    kernels = [lambda: R.mxfp4_(meta(3, 5, 128)), lambda: R.nvfp4_(meta(7, 448)),
               lambda: R.fp8_ue8m0_(meta(2, 512)), lambda: R.fp8_ds_mla_nope_(meta(9, 448)),
               lambda: R.bf16_(meta(9, 64))]
    entries = []
    for s in ("deepseek", "vllm"):
        entries += [lambda s = s: R.round_index_(P(s), meta(1, 4, 32, 128)),
                    lambda s = s: R.round_window_(P(s), meta(1, 4, 512), 64),
                    lambda s = s: R.round_compressed_(P(s), meta(4, 448), meta(1, 4, 1, 64))]
    for slice_blocks in (R.SLICE_BLOCKS, 7):
        monkeypatch.setattr(R, "SLICE_BLOCKS", slice_blocks)
        for call in kernels + entries:
            call()
    assert R.fallback_count() == 0                  # a meta count holds no value and is skipped
    R.set_strict(True)
    for call in kernels:
        with pytest.raises((RuntimeError, NotImplementedError)):
            call()


def test_slices_equal_one_pass(monkeypatch):
    """Operands of more than SLICE_BLOCKS blocks are rounded slice by slice: same bits as one
    pass. A nonfinite block in a later slice is the only block kept; under the strict switch it
    is refused with the whole operand unchanged."""
    x = torch.tensor(np.concatenate([operand_rows(), tie_rows()])).half()
    whole = {fn: fn(x.clone()) for fn in (R.mxfp4_, R.nvfp4_, R.fp8_ue8m0_)}
    whole[R.fp8_ds_mla_nope_] = R.fp8_ds_mla_nope_(x[:, :448].contiguous())
    block = {R.mxfp4_: 32, R.nvfp4_: 16, R.fp8_ue8m0_: 32, R.fp8_ds_mla_nope_: 64}
    monkeypatch.setattr(R, "SLICE_BLOCKS", 7)
    for fn, want in whole.items():
        arg = x[:, :448].contiguous() if fn is R.fp8_ds_mla_nope_ else x.clone()
        assert torch.equal(fn(arg).view(torch.int16), want.view(torch.int16)), fn.__name__
        bad = x[:, :448].contiguous() if fn is R.fp8_ds_mla_nope_ else x.clone()
        bad[-1, -1] = float("inf")
        before = bad.clone()
        kept = want.clone()
        kept[-1, -block[fn]:] = before[-1, -block[fn]:]
        assert torch.equal(fn(bad.clone()).view(torch.int16), kept.view(torch.int16)), f"{fn.__name__} sliced fallback"
        R.set_strict(True)
        with pytest.raises(ValueError, match = "nonfinite"):
            fn(bad)
        R.set_strict(False)
        assert torch.equal(bad.view(torch.int16), before.view(torch.int16)), f"{fn.__name__} wrote a slice"


def test_parse():
    assert str(P(None)) == str(P("")) == str(P("  ")) == N.DEFAULT == "deepseek:index"
    assert str(P("precise")) == "precise" and str(P(" Precise ")) == "precise"
    assert str(P("DeepSeek")) == "deepseek" and str(P("vllm")) == "vllm"
    assert str(P("deepseek:index,window,compressed")) == "deepseek"
    assert str(P("vllm:compressed, index")) == "vllm:index,compressed"
    n = P("deepseek:window,compressed")
    assert (n.contract, n.index, n.window, n.compressed) == ("deepseek", False, True, True)
    assert P(n) is n
    for s in ("precise", "deepseek", "vllm", "deepseek:index", "vllm:window", "deepseek:index,compressed"):
        assert P(str(P(s))) == P(s), s
    for bad in ("fp4", "precise:index", "vllm:", "vllm:kv", "deepseek:index,foo", ":index", "deepseek;index"):
        with pytest.raises(ValueError):
            P(bad)
    # a hand-built setting is checked as a parsed one: the rounding treats every contract but
    # 'deepseek' as vLLM's, so these would round while printing as something else
    for bad in (dict(contract = "precise", index = True), dict(contract = "fp8", index = True, window = True),
                dict(contract = "deepseek")):
        with pytest.raises(ValueError):
            N.Numerics(**bad)


def test_contract_dispatch():
    x = operand_rows()
    kv = torch.tensor(x[:4])
    # window KV [.., 448 NoPE + 64 RoPE]
    assert torch.equal(R.round_window_(P("precise"), kv.clone(), 64), kv)
    assert torch.equal(R.round_window_(P("deepseek:index"), kv.clone(), 64), kv)
    assert torch.equal(R.round_window_(P("deepseek"), kv.clone(), 64), R.fp8_ue8m0_(kv.clone()))
    w = R.round_window_(P("vllm"), kv.clone(), 64)
    assert torch.equal(w[:, :448], R.fp8_ds_mla_nope_(kv[:, :448].contiguous()))
    assert torch.equal(w[:, 448:], kv[:, 448:].to(torch.bfloat16).float())
    # compressed entries, stored as NoPE and RoPE halves: DeepSeek's 16-lane blocks never
    # straddle the boundary (448 = 28 x 16), so rounding the halves equals the whole entry
    nope, rope = kv[:, :448].contiguous(), kv[:, 448:].contiguous()
    a, b = R.round_compressed_(P("deepseek"), nope.clone(), rope.clone())
    assert torch.equal(torch.cat([a, b], 1), R.nvfp4_(kv.clone())), "16-lane halves differ from the whole entry"
    a, b = R.round_compressed_(P("vllm"), nope.clone(), rope.clone())
    assert torch.equal(a, R.fp8_ds_mla_nope_(nope.clone())) and torch.equal(b, rope.to(torch.bfloat16).float())
    for s in ("precise", "deepseek:index", "vllm:index,window"):
        a, b = R.round_compressed_(P(s), nope.clone(), rope.clone())
        assert torch.equal(a, nope) and torch.equal(b, rope), f"{s}: a part outside the selection was rounded"
    # index Q/K
    q = torch.tensor(x[:2, :128])
    for s in ("deepseek", "deepseek:index", "vllm", "vllm:index"):
        assert torch.equal(R.round_index_(P(s), q.clone()), R.mxfp4_(q.clone())), s
    for s in ("precise", "deepseek:window", "vllm:window,compressed"):
        assert torch.equal(R.round_index_(P(s), q.clone()), q), s


def _minimal_config(path):
    t = {
        "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "vocab_size": 256,
        "q_lora_rank": 32, "o_lora_rank": 32, "max_position_embeddings": 4096,
        "moe_intermediate_size": 32, "n_routed_experts": 4, "num_experts_per_tok": 2,
        "index_n_heads": 2, "index_head_dim": 32, "index_topk": 16,
        "compress_ratios": [0, 2, 2, 1], "kv_source_layer_ids": [1, 3],
        "index_source_layer_ids": [1, 3],
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"architectures": ["DeepseekV41ForCausalLM"], "text_config": t}, f)


def test_config_attribute(tmp_path, monkeypatch):
    """config.dsv41_numerics: EXL3_DSV41_NUMERICS read when the config is built, validated on
    assignment, and the compressed part refused while a packed pool exists."""
    try:
        from exllamav3.architecture.deepseek_v41 import DeepseekV41Config
    except Exception as e:
        # a failure, never a skip: this test needs the real package, and a broken import (the
        # compiled extension included) must not pass as "nothing to test"
        raise AssertionError(
            f"test_dsv41_numerics_.py::test_config_attribute: importing exllamav3 failed "
            f"({type(e).__name__}: {e}), so config.dsv41_numerics cannot be tested. This test "
            f"needs the package with its compiled extension: build it, or put a prebuilt "
            f"exllamav3_ext on PYTHONPATH") from e
    _minimal_config(str(tmp_path))
    monkeypatch.delenv(N.ENV_NUMERICS, raising = False)
    cfg = DeepseekV41Config(str(tmp_path))
    assert str(cfg.dsv41_numerics) == "deepseek:index"
    monkeypatch.setenv(N.ENV_NUMERICS, "vllm:compressed,index")
    assert str(DeepseekV41Config(str(tmp_path)).dsv41_numerics) == "vllm:index,compressed"
    monkeypatch.setenv(N.ENV_NUMERICS, "precise:index")
    with pytest.raises(ValueError, match = "takes no parts"):
        DeepseekV41Config(str(tmp_path))
    monkeypatch.delenv(N.ENV_NUMERICS)

    cfg.dsv41_numerics = "precise"
    assert str(cfg.dsv41_numerics) == "precise" and not cfg.dsv41_numerics.index
    with pytest.raises(ValueError):
        cfg.dsv41_numerics = "fp8"
    assert str(cfg.dsv41_numerics) == "precise", "a refused value changed the setting"
    cfg.dsv41_numerics = None
    assert str(cfg.dsv41_numerics) == "deepseek:index"

    # a packed pool: refused under the compressed part, and blocks switching to it while alive
    class Pool:
        pass
    cfg.dsv41_numerics = "deepseek"
    with pytest.raises(ValueError, match = r"quantized Cache \(-cq\); use a setting without the compressed part \(e.g. deepseek:index,window\)"):
        cfg.register_packed_pool(Pool())
    cfg.dsv41_numerics = "vllm:compressed"
    with pytest.raises(ValueError, match = r"\(e.g. deepseek:index\)"):
        cfg.register_packed_pool(Pool())
    cfg.dsv41_numerics = "deepseek:index,window"
    pool = Pool()
    cfg.register_packed_pool(pool)
    with pytest.raises(ValueError, match = "quantized pool"):
        cfg.dsv41_numerics = "vllm"
    assert str(cfg.dsv41_numerics) == "deepseek:index,window"
    cfg.dsv41_numerics = "vllm:index,window"
    ref = weakref.ref(pool)
    del pool
    assert ref() is None
    cfg.dsv41_numerics = "deepseek"
    assert str(cfg.dsv41_numerics) == "deepseek"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rs"]))
