"""GPU: the DeepSeek-V4.1 numerics rounding kernels (modules/dsv41_rounding.py) give, on every
visible GPU, bitwise the values the CPU computes (which test_dsv41_numerics_.py checks against
scalar oracles), for fp32 and fp16 operands: random rows over seven decades, the CPU test's
operand rows (scale floors, saturation) and tie rows (every lattice midpoint at scale 1), in one
pass and in slices, under the default overflow policy and under the strict switch. Every
refusal case of the CPU test (a nonfinite operand, an FP16 overflow) keeps its block unrounded
on each GPU, as on the CPU, and is counted; under the strict switch it raises the refusal's
error and leaves the operand unchanged.

The default policy never waits for the GPU: every default-path call, the fallback cases and the
poll that reads the count back included, runs under torch.cuda.set_sync_debug_mode("error"),
which turns any synchronizing call into an error. The warning comes one poll late (after a
synchronization outside the checked calls, as the generator's after each forward) and only once;
with the default POLL_SECONDS, not before the next look at the count. A process whose last
rounding call keeps a block on a GPU (its first look only copies the count) prints the warning
at exit, once.

    python tests/test_dsv41_numerics_gpu_.py

The kernels load without the compiled extension. Skipped when no GPU is visible."""
import contextlib
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_numerics, skip
from test_dsv41_numerics_ import operand_rows, refusal_cases, run_to_exit, tie_rows


@contextlib.contextmanager
def no_sync():
    """Any call that synchronizes the host with a GPU raises inside this block."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(0)


def main():
    _, R = load_numerics()
    previous, poll_seconds = R.set_strict(False), R.POLL_SECONDS
    R.reset_fallbacks()
    R.POLL_SECONDS = 0.0                     # look at the count on every call, except below
    try:
        return run(R)
    finally:
        R.set_strict(previous)
        R.POLL_SECONDS = poll_seconds
        R.reset_fallbacks()


def run(R):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(257, 512, generator = g) * torch.logspace(-4, 3, 257).unsqueeze(1)
    x[3, :64] = 0.0                          # an all-zero block hits the scale floors
    x = torch.cat([x, torch.tensor(operand_rows()), torch.tensor(tie_rows())])
    cases = [("mxfp4", R.mxfp4_, x), ("nvfp4", R.nvfp4_, x), ("fp8_ue8m0", R.fp8_ue8m0_, x),
             ("fp8_ds_mla", R.fp8_ds_mla_nope_, x[:, :448].contiguous()), ("bf16", R.bf16_, x)]
    n_dev = torch.cuda.device_count()
    ok = True
    slice_blocks = R.SLICE_BLOCKS
    n_cmp = 0
    for name, fn, arg in cases:
        for dtype in (torch.float32, torch.float16):
            a = arg.to(dtype)
            want = fn(a.clone())
            for d in range(n_dev):
                for strict in (False, True):
                    for sb in (slice_blocks, 7):             # one pass, and many slices
                        dev = a.to(f"cuda:{d}")
                        torch.cuda.synchronize(d)
                        R.SLICE_BLOCKS = sb
                        R.set_strict(strict)
                        try:
                            if strict:
                                got = fn(dev)
                            else:
                                with no_sync():
                                    got = fn(dev)
                        except RuntimeError as e:
                            ok = False
                            print(f"  FAIL {name} {dtype} cuda:{d}: the default path synchronized: {e}")
                            continue
                        finally:
                            R.SLICE_BLOCKS = slice_blocks
                            R.set_strict(False)
                        same = torch.equal(got.cpu(), want)
                        ok &= same
                        n_cmp += 1
                        if not same:
                            print(f"  FAIL {name} {dtype} cuda:{d} {'strict' if strict else 'default'} slices "
                                  f"of {sb} blocks: {(got.cpu() != want).sum().item()} values differ from the CPU")
    if R.fallback_count():
        ok = False
        print(f"  FAIL finite operands counted {R.fallback_count()} fallback block(s)")

    # the refusal cases: strict raises their errors, the default keeps the block (as the CPU)
    n_ref = n_kept = 0
    for fn, lanes, value, why in refusal_cases():
        t = torch.ones(2, lanes, dtype = torch.half)
        t[1, 5] = value
        want = fn(t.clone())                                 # the CPU, default policy
        for d in range(n_dev):
            dev = t.to(f"cuda:{d}")
            before = dev.clone()
            R.set_strict(True)
            try:
                fn(dev)
                refused = False
            except ValueError as e:
                refused = str(e) == ("dsv41 numerics: nonfinite operand" if why == "nonfinite" else
                                     "dsv41 numerics: rounded value overflows the operand dtype")
            finally:
                R.set_strict(False)
            same = torch.equal(dev.view(torch.int16), before.view(torch.int16))
            ok &= refused and same
            n_ref += 1
            if not (refused and same):
                print(f"  FAIL strict {fn.__name__} cuda:{d} value {value}: refused {refused}, unchanged {same}")
            dev = t.to(f"cuda:{d}")
            torch.cuda.synchronize(d)
            try:
                with no_sync():
                    fn(dev)
            except RuntimeError as e:
                ok = False
                print(f"  FAIL {fn.__name__} cuda:{d} value {value}: the fallback synchronized: {e}")
                continue
            kept = torch.equal(dev.cpu().view(torch.int16), want.view(torch.int16)) and \
                torch.equal(dev[1, 5:6].cpu().view(torch.int16), t[1, 5:6].view(torch.int16))
            ok &= kept
            n_kept += 1
            if not kept:
                print(f"  FAIL default {fn.__name__} cuda:{d} value {value}: not the CPU's values, or the "
                      f"bad block was rounded")
    counted = R.fallback_count()
    want_count = n_kept + len(refusal_cases())               # every GPU, and the CPU references
    ok &= counted == want_count
    if counted != want_count:
        print(f"  FAIL counted {counted} fallback blocks, expected {want_count}")

    # the warning: one poll late, once, with no synchronization in the rounding calls
    R.reset_fallbacks()
    one = torch.ones(2, 64, dtype = torch.half)
    bad = one.clone()
    bad[1, 40] = float("nan")
    lines = []
    for d in range(n_dev):
        ops = [t.to(f"cuda:{d}") for t in (bad, one, bad, one)]
        torch.cuda.synchronize(d)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                with no_sync():
                    R.mxfp4_(ops[0])                         # counted; the poll copies the count
                    first = out.getvalue()
                torch.cuda.synchronize(d)                    # as the generator after a forward
                with no_sync():
                    R.mxfp4_(ops[1])                         # the copy has landed: warns (device 0)
                    R.mxfp4_(ops[2])
                torch.cuda.synchronize(d)
                with no_sync():
                    R.mxfp4_(ops[3])
        except RuntimeError as e:
            ok = False
            print(f"  FAIL cuda:{d}: the poll synchronized: {e}")
            continue
        if d == 0 and first:
            ok = False
            print(f"  FAIL cuda:0 warned before the count was read back: {first!r}")
        lines += [l for l in out.getvalue().splitlines() if "!! DSV41 numerics:" in l]
    warned_once = len(lines) == 1 and " 1 block(s) " in lines[0]
    ok &= warned_once and R.fallback_count() == 2 * n_dev
    if not warned_once:
        print(f"  FAIL expected one warning naming 1 block, got {lines}")

    # looks at a GPU's count are POLL_SECONDS apart: the landed copy is read only at the next look
    R.reset_fallbacks()
    R.POLL_SECONDS = 1.0
    ops = [t.to("cuda:0") for t in (bad, one, one)]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            with no_sync():
                R.mxfp4_(ops[0])                             # the first look: copies the count
            torch.cuda.synchronize(0)
            with no_sync():
                R.mxfp4_(ops[1])                             # within the interval: no look
            early = out.getvalue()
            time.sleep(1.05)
            with no_sync():
                R.mxfp4_(ops[2])                             # the next look reads the copy: warns
    except RuntimeError as e:
        ok, early = False, ""
        print(f"  FAIL cuda:0: a rate-limited look synchronized: {e}")
    finally:
        R.POLL_SECONDS = 0.0
    late = [l for l in out.getvalue().splitlines() if "!! DSV41 numerics:" in l]
    limited = not early and len(late) == 1
    ok &= limited
    if not limited:
        print(f"  FAIL POLL_SECONDS: warned within the interval ({early!r}) or not after it ({late})")

    # a process whose last rounding call keeps a block on each GPU: reported at exit, once
    for d in range(n_dev):
        lines = run_to_exit("kept", f"cuda:{d}")
        at_exit = len(lines) == 2 and lines[0] == "END" and " 1 block(s) " in lines[1]
        ok &= at_exit
        if not at_exit:
            print(f"  FAIL cuda:{d}: expected the warning once, at exit: {lines}")

    names = ", ".join(f"cuda:{d} {torch.cuda.get_device_name(d)}" for d in range(n_dev))
    print(f"  {'OK ' if ok else 'FAIL'} {len(cases)} rounding kernels bitwise equal to the CPU on "
          f"{n_dev} GPU(s) ({names}), fp32 and fp16, one pass and sliced, default and strict "
          f"({n_cmp} comparisons, the default ones with no synchronization); {n_ref} strict refusals "
          f"raise their errors and leave the operand unchanged; {n_kept} fallbacks keep the block "
          f"as the CPU does and are counted; one warning, one poll late; looks POLL_SECONDS apart; "
          f"a count left unread reported at exit")
    return 0 if ok else 1


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        skip("V4.1 numerics on the GPU", "no CUDA device")
        sys.exit(0)
    sys.exit(main())
