# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
Rounding kernels for the DeepSeek-V4.1 attention numerics (architecture/dsv41/numerics.py),
written as plain torch so each one can be checked element by element against DeepSeek's
kernel.py and vLLM's fp8_ds_mla layout. Every kernel rounds in place and writes the dequantized
values back in the operand's dtype.

Overflow and nonfinite operands. A block (the lanes that share one scale: 32, 16 or 64; one lane
for bf16_) whose operand is not finite, or whose rounded value the operand's dtype cannot hold
(in FP16: 57,344 or more under MXFP4, 63,488 or more under the FP8 roundings, 65,408 or more
under BF16), keeps its unrounded values, bit for bit the values 'precise' stores; the rest of
the operand is rounded as usual. Which blocks fall back is decided on the device, with no host
read. Each device keeps a count of these blocks, and the first nonzero count prints one warning
per process (see _poll: the host reads a GPU's count from a pinned copy only after that copy has
landed, and looks at most every POLL_SECONDS, so no call waits for the GPU; a count no look has
read by the time the interpreter exits is reported then, see _report_at_exit). fallback_count()
gives the total.

EXL3_DSV41_NUMERICS_STRICT=1 (strict(), read once) restores the refusal instead: a kernel checks
its operand and its result before it writes anything, raises ValueError with the operand
unchanged, and reads its finiteness flags back to the host once per call, which waits for the
device. The policy entry points that round two halves of an entry (round_window_ under vllm,
round_compressed_) run one kernel per half, so a refusal of the second half leaves the first
half rounded; every caller hands them a copy it drops on an error. On finite operands whose
rounded values fit, both modes give the same bits.

Memory: the arithmetic runs on fp32 copies of the operand's blocks, about ten of them alive at
once. The block kernels therefore work through the operand in slices of SLICE_BLOCKS blocks, and
round each slice in place (blocks are independent, so this does not change a bit). Under the
strict switch the slices go into one buffer of the operand's dtype instead, copied into the
operand only after every slice has passed its checks. A decode-sized operand is a single slice;
a prefill chunk's index query (4096 rows x 32 heads x 128 lanes, 32 MiB in FP16) peaks at about
the slice temporaries (plus that buffer under the strict switch) instead of about 17 times the
operand.

The vllm contract's fp8_ds_mla scale is the smallest power of two with 448 * 2^e >= the block
peak, computed exactly in float64 on the FP16 operand. vLLM's kernels compute the same scale in
FP32 with an approximate log2, on BF16 operands, and can pick 2^k where this picks 2^(k + 1) in a
band a few ulps wide just above a peak of 448 * 2^k; the oracles of tests/test_dsv41_numerics_.py
transcribe the exact form.
"""

from __future__ import annotations
import atexit
import os
import threading
import time
import torch

from ..architecture.dsv41.numerics import Numerics, NOPE, ENV_STRICT


# ---------------------------------------------------------------- shared lattices

def _pow2_ceil(t: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2 t) for positive normal fp32 t, on the IEEE fields exactly as kernel.py's
    fast_log2_ceil / fast_pow2 do (exponent, plus one when any mantissa bit is set). Integer
    bit operations give the same bits on every device (torch.frexp / torch.ldexp faulted on the
    GPU with at least one torch build)."""
    bits = t.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (e << 23).view(torch.float32)


def _pow2_ceil_f64(t: torch.Tensor) -> torch.Tensor:
    """The same on float64 (positive normal), returned as float64."""
    bits = t.double().contiguous().view(torch.int64)
    e = ((bits >> 52) & 0x7FF) + (bits & ((1 << 52) - 1)).ne(0).to(torch.int64)
    return (e << 52).view(torch.float64)


def _e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude (0, 0.5, 1, 1.5, 2, 3, 4, 6) of a in [0, 6], ties to even."""
    return torch.where(a < 2.0, torch.round(a * 2.0) * 0.5,
                       torch.where(a < 4.0, torch.round(a), torch.round(a * 0.5) * 2.0))


# blocks per slice of a block kernel (see Memory above): 2M lanes at 32 lanes per block, so
# each fp32 temporary of a slice is 8 MiB
SLICE_BLOCKS = 1 << 16


# ---------------------------------------------------------------- overflow policy

# EXL3_DSV41_NUMERICS_STRICT, or None until it has been read
_strict: bool | None = None


def strict() -> bool:
    """
    True when a rounder refuses (ValueError) instead of keeping a block unrounded. The first
    call reads EXL3_DSV41_NUMERICS_STRICT (on for any value other than '' and '0'); later calls
    return the same, or what set_strict() put in its place.
    """
    global _strict
    if _strict is None:
        _strict = os.environ.get(ENV_STRICT, "").strip() not in ("", "0")
    return _strict


def set_strict(value: bool | None) -> bool | None:
    """
    Tests only: set the strict switch, and return the previous setting, or None when the variable
    had not been read yet. None clears it, so the next call reads EXL3_DSV41_NUMERICS_STRICT
    again; restoring the returned value therefore restores exactly what was there.
    """
    global _strict
    previous = _strict
    _strict = None if value is None else bool(value)
    return previous


class _Fallbacks:
    """One device's count of blocks kept unrounded, and the pinned host copy it is read from."""

    def __init__(self, device: torch.device):
        cuda = device.type == "cuda"
        # never inference tensors: the count is updated in place inside and outside inference mode
        with torch.inference_mode(False):
            self.count = torch.zeros((), dtype = torch.int64, device = device)
            self.host = torch.zeros((), dtype = torch.int64, pin_memory = cuda)
        self.event = torch.cuda.Event() if cuda else None
        self.pending = False
        self.next_poll = 0.0


# a GPU's count is looked at no more often than this (seconds): a look costs an event query and,
# once the last copy has landed, a new copy and event record, about 20 us of host time
POLL_SECONDS = 0.25


_fallbacks: dict[torch.device, _Fallbacks] = {}
_warned = False
_exit_report_armed = False
_lock = threading.Lock()


def _warn(seen: int):
    global _warned
    with _lock:
        if _warned:
            return
        _warned = True
    print(f" !! DSV41 numerics: {seen} block(s) kept unrounded so far, because rounding them would "
          f"store a value FP16 cannot hold or their operand is not finite; they hold the values "
          f"'precise' stores. Shown once per process; {ENV_STRICT}=1 raises instead, and "
          f"exllamav3.modules.dsv41_rounding.fallback_count() gives the total", flush = True)


def _poll(fb: _Fallbacks, device: torch.device):
    """
    Warn once the count is nonzero, without waiting for the device. On a GPU the count is copied
    (non-blocking) into pinned host memory, and an event recorded after the copy; a later look
    reads the copy only when the event has completed (the generator's own synchronization after
    each forward sees to that) and then issues the next copy. Looks are at least POLL_SECONDS
    apart, so the warning comes one or two looks late, normally within a second (at exit when no
    later rounding call looks), and at most one small copy is in flight per device. Nothing is
    looked at while a CUDA graph is being captured. On the CPU every call reads the count
    directly.
    """
    if device.type == "cpu":
        seen = int(fb.count)
    elif device.type == "cuda":
        now = time.monotonic()
        if now < fb.next_poll or torch.cuda.is_current_stream_capturing():
            return
        fb.next_poll = now + POLL_SECONDS
        seen = 0
        if fb.pending:
            if not fb.event.query():
                return
            fb.pending = False
            seen = int(fb.host)
        if not seen:
            fb.host.copy_(fb.count, non_blocking = True)
            fb.event.record(torch.cuda.current_stream(device))
            fb.pending = True
            return
    else:
        return
    if seen:
        _warn(seen)


def _note(device: torch.device, n_bad: torch.Tensor):
    """Add n_bad (a device scalar: blocks kept unrounded by one call) to the device's count."""
    global _exit_report_armed
    fb = _fallbacks.get(device)
    if fb is None:
        fb = _fallbacks.setdefault(device, _Fallbacks(device))
        with _lock:
            if not _exit_report_armed:
                _exit_report_armed = True
                atexit.register(_report_at_exit)
    fb.count += n_bad
    if not _warned:
        _poll(fb, device)


def _read(fb: _Fallbacks) -> int:
    """One device's count, read on the host: on a GPU this waits for the device. Meta tensors
    (tests) hold no count."""
    return int(fb.count) if fb.count.device.type in ("cpu", "cuda") else 0


def _report_at_exit():
    """
    At interpreter exit (atexit, registered with the first count): print the warning if a count
    is nonzero that no look has read, i.e. blocks kept by the last rounding calls of the
    process, less than a look or two before it ended. Reads every count once, which waits for
    the GPUs; a device that cannot be read any more (a failed CUDA context) is skipped.
    """
    if _warned:
        return
    seen = 0
    for fb in list(_fallbacks.values()):
        try:
            seen += _read(fb)
        except Exception:
            pass
    if seen:
        _warn(seen)


def fallback_count() -> int:
    """
    Blocks kept unrounded since the process started (or since reset_fallbacks()), over every
    device. Reads each device's count, so on a GPU it waits for the device: for diagnostics
    and tests, never called by a forward pass.
    """
    return sum(_read(fb) for fb in list(_fallbacks.values()))


def reset_fallbacks():
    """Tests only: drop every count and arm the warning again."""
    global _warned
    _fallbacks.clear()
    _warned = False


def _check(ok_in: torch.Tensor, ok_out: torch.Tensor):
    """Strict switch: raise for a nonfinite operand or result. One host read for both flags."""
    finite = torch.stack((ok_in, ok_out)).tolist()
    if not finite[0]:
        raise ValueError("dsv41 numerics: nonfinite operand")
    if not finite[1]:
        raise ValueError("dsv41 numerics: rounded value overflows the operand dtype")


def _round_blocks_strict_(x: torch.Tensor, xb: torch.Tensor, fn) -> torch.Tensor:
    """_round_blocks_ under the strict switch: all or nothing, one host read."""
    n = xb.shape[0]
    if n <= SLICE_BLOCKS:
        v = xb.float()
        out = fn(v).to(x.dtype)
        _check(torch.isfinite(v).all(), torch.isfinite(out).all())
        xb.copy_(out)
        return x
    out = torch.empty_like(xb)
    ok_in = torch.ones((), dtype = torch.bool, device = x.device)
    ok_out = torch.ones((), dtype = torch.bool, device = x.device)
    for i in range(0, n, SLICE_BLOCKS):
        v = xb[i : i + SLICE_BLOCKS].float()
        o = fn(v).to(x.dtype)
        ok_in &= torch.isfinite(v).all()
        ok_out &= torch.isfinite(o).all()
        out[i : i + SLICE_BLOCKS].copy_(o)
        del v, o
    _check(ok_in, ok_out)
    xb.copy_(out)
    return x


def _round_blocks_(x: torch.Tensor, block: int, fn) -> torch.Tensor:
    """
    Apply fn (fp32 blocks -> their dequantized fp32 values) to x in `block`-lane blocks and store
    the result in x's dtype, in place, slice by slice. A block whose operand is not finite, or
    whose rounded values the dtype cannot hold, keeps its values and is counted (module
    docstring); under the strict switch either raises ValueError and leaves x unchanged. Values
    below the dtype's subnormal range flush toward zero (FP16 cannot hold what BF16 would keep
    there).
    """
    assert x.shape[-1] % block == 0, f"last dim {x.shape[-1]} is not a multiple of {block}"
    assert x.is_contiguous(), "dsv41 numerics: operand must be contiguous"
    xb = x.view(-1, block)
    if strict():
        return _round_blocks_strict_(x, xb, fn)
    n_bad = None
    for i in range(0, xb.shape[0], SLICE_BLOCKS):
        s = xb[i : i + SLICE_BLOCKS]
        v = s.float()
        o = fn(v).to(x.dtype)
        bad = (torch.isfinite(v).all(-1, keepdim = True) & torch.isfinite(o).all(-1, keepdim = True)).logical_not_()
        torch.where(bad, s, o, out = s)
        n_bad = bad.sum() if n_bad is None else n_bad + bad.sum()
        del v, o
    if n_bad is not None:
        _note(x.device, n_bad)
    return x


# ---------------------------------------------------------------- DeepSeek kernels

def _mxfp4(v: torch.Tensor) -> torch.Tensor:
    amax = v.abs().amax(-1, keepdim = True).clamp_min(6.0 * 2.0 ** -126)
    # a Python scalar: multiplied in fp32 like an fp32 tensor of 1/6, with no host-to-device copy
    s = _pow2_ceil(amax * (1.0 / 6.0))
    y = (v / s).clamp(-6.0, 6.0)
    return torch.sign(y) * _e2m1_rne(y.abs()) * s


def mxfp4_(x: torch.Tensor) -> torch.Tensor:
    """fp4_act_quant(x, 32, inplace=True), E8M0 scale: s = 2^ceil(log2(max(amax, 6*2^-126) / 6))."""
    return _round_blocks_(x, 32, _mxfp4)


def _nvfp4(v: torch.Tensor) -> torch.Tensor:
    amax = v.abs().amax(-1, keepdim = True).clamp_min(6.0 * 2.0 ** -9)
    s = (amax / 6.0).clamp(max = 448.0).to(torch.float8_e4m3fn).float()   # saturating, as cvt satfinite
    y = (v / s).clamp(-6.0, 6.0)
    return torch.sign(y) * _e2m1_rne(y.abs()) * s


def nvfp4_(x: torch.Tensor) -> torch.Tensor:
    """
    fp4_act_quant(x, 16, inplace=True, scale_dtype=float8_e4m3fn): s = e4m3(max(amax, 6*2^-9) / 6).
    The scale saturates at 448 as in the reference kernel, so a block whose peak exceeds
    6 * 448 = 2,688 is clamped to +-2,688 (it never overflows FP16); only a nonfinite operand
    keeps a block unrounded.
    """
    return _round_blocks_(x, 16, _nvfp4)


def _fp8_ue8m0(v: torch.Tensor) -> torch.Tensor:
    amax = v.abs().amax(-1, keepdim = True).clamp_min(1e-4)
    s = _pow2_ceil(amax * (1.0 / 448.0))
    return (v / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s


def fp8_ue8m0_(x: torch.Tensor) -> torch.Tensor:
    """act_quant(x, 32, 'ue8m0', inplace=True): s = 2^ceil(log2(max(amax, 1e-4) / 448)), E4M3 nearest."""
    return _round_blocks_(x, 32, _fp8_ue8m0)


# ---------------------------------------------------------------- vLLM fp8_ds_mla

def _fp8_ds_mla(v: torch.Tensor) -> torch.Tensor:
    peak = v.abs().amax(-1, keepdim = True).clamp_min(1e-4)
    s = _pow2_ceil_f64(peak.double() / 448.0).float()  # exact: smallest 2^e with 448 * 2^e >= peak
    return (v / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s


def fp8_ds_mla_nope_(x: torch.Tensor) -> torch.Tensor:
    """448 NoPE lanes in 64-lane groups, scale 2^ceil(log2(max(amax, 1e-4) / 448)), E4M3 nearest."""
    assert x.shape[-1] == NOPE, f"expected {NOPE} NoPE lanes, got {x.shape[-1]}"
    return _round_blocks_(x, 64, _fp8_ds_mla)


def bf16_(x: torch.Tensor) -> torch.Tensor:
    """
    Round to BF16 and widen back in place. Each lane is its own block: a nonfinite lane, and an
    FP16 value that rounds past FP16's range (65,408 or more rounds to 65,536), keep their values
    and are counted; under the strict switch either raises ValueError with x unchanged.
    """
    out = x.to(torch.bfloat16).to(x.dtype)
    if strict():
        _check(torch.isfinite(x).all(), torch.isfinite(out).all())
        x.copy_(out)
        return x
    bad = (torch.isfinite(x) & torch.isfinite(out)).logical_not_()
    torch.where(bad, x, out, out = x)
    _note(x.device, bad.sum())
    return x


# ---------------------------------------------------------------- policy entry points

def round_index_(policy: Numerics, x: torch.Tensor) -> torch.Tensor:
    """Index query or key vectors, post-RoPE, last dim a multiple of 32."""
    return mxfp4_(x) if policy.index else x


def round_window_(policy: Numerics, kv: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Sliding-window KV entries [..., NOPE + rope_dim], post-RoPE, in place."""
    if not policy.window:
        return kv
    if kv.shape[-1] != NOPE + rope_dim:
        raise ValueError(f"dsv41 numerics: window entries of {kv.shape[-1]} lanes, expected "
                         f"{NOPE} NoPE + {rope_dim} RoPE")
    if policy.contract == "deepseek":
        return fp8_ue8m0_(kv)
    nope = kv[..., :NOPE].contiguous()
    rope = kv[..., NOPE:].contiguous()
    kv[..., :NOPE] = fp8_ds_mla_nope_(nope)
    kv[..., NOPE:] = bf16_(rope)
    return kv


def round_compressed_(policy: Numerics, nope: torch.Tensor, rope: torch.Tensor):
    """Compressed-pool entries as the pools store them: NoPE lanes [n, NOPE] and post-RoPE lanes
    [..., rope_dim], both contiguous, in place. DeepSeek's 16-lane blocks never straddle the
    NoPE/RoPE boundary (448 = 28 * 16), so rounding the halves equals rounding the entry."""
    if not policy.compressed:
        return nope, rope
    if policy.contract == "deepseek":
        return nvfp4_(nope), nvfp4_(rope)
    return fp8_ds_mla_nope_(nope), bf16_(rope)
