"""
Opt-in GPU arithmetic policies (doc/env_vars.md). Each variable is parsed strictly and read once,
when exllamav3 is imported, into the module-level constants at the bottom; the native side reads
the same variables on first use, with the same rules. Set them before importing exllamav3. The
default of every policy is the ordinary dispatch.

EXL3_STABLE_ARITHMETIC=1 implies EXL3_HGEMM_FIXED_ROWS=128 and EXL3_MOE_FUSED_PREFILL=1: each may be
left unset or set to that value, anything else is refused.

EXL3_EXACT_ROWS is Python only: the extension does not read it.

Torch-free, so tests can load this file by path.
"""
import os


def stable_arithmetic_enabled(environ = None) -> bool:
    """
    EXL3_STABLE_ARITHMETIC: 1 selects GPU arithmetic whose results do not depend on how many rows
    a call has; unset or 0 keeps the ordinary dispatch. Mirrors stable_arithmetic() in
    stable_arithmetic.cpp. Refuses a contradicting EXL3_HGEMM_FIXED_ROWS or EXL3_MOE_FUSED_PREFILL,
    EXL3_MOE_FUSED_DET=0 and EXL3_NO_FUSED_RECONSTRUCT
    """
    environ = os.environ if environ is None else environ
    value = environ.get("EXL3_STABLE_ARITHMETIC", "0")
    if value not in ("0", "1"):
        raise ValueError(f"EXL3_STABLE_ARITHMETIC must be 0 or 1, got {value!r}")
    if value == "0":
        return False
    for name, implied in (("EXL3_HGEMM_FIXED_ROWS", "128"), ("EXL3_MOE_FUSED_PREFILL", "1")):
        if environ.get(name, implied) != implied:
            raise ValueError(f"EXL3_STABLE_ARITHMETIC=1 implies {name}={implied}, but it is set to "
                             f"{environ[name]!r}; unset it")
    if environ.get("EXL3_MOE_FUSED_DET", "1") == "0":
        raise ValueError("EXL3_STABLE_ARITHMETIC=1 needs the ordered expert accumulation that "
                         "EXL3_MOE_FUSED_DET=0 turns off")
    if environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0":
        raise ValueError("EXL3_STABLE_ARITHMETIC=1 reconstructs EXL3 weights in the original basis at "
                         "every row count, which EXL3_NO_FUSED_RECONSTRUCT turns off")
    return True


def hgemm_fixed_rows(environ = None) -> int:
    """
    EXL3_HGEMM_FIXED_ROWS: 128 runs the native GEMMs (hgemm.cu) as fixed 128-row cuBLAS tiles
    and routes FP16 x FP16 linears through them; unset or 0 keeps the default dispatch, and unset
    means 128 under EXL3_STABLE_ARITHMETIC=1. Mirrors hgemm_fixed_rows() in hgemm.cu: any other
    value is an error
    """
    environ = os.environ if environ is None else environ
    value = environ.get("EXL3_HGEMM_FIXED_ROWS")
    if value is None or value == "0":
        return 128 if stable_arithmetic_enabled(environ) else 0
    if value != "128":
        raise ValueError(f"EXL3_HGEMM_FIXED_ROWS must be 0 or 128, got {value!r}")
    return 128


def fused_prefill_enabled(environ = None) -> bool:
    """
    EXL3_MOE_FUSED_PREFILL: 1 computes every GPU-side routed expert of an MoE prefill with the
    fused kernel, in row stripes of its temp buffers, instead of switching experts with many rows
    to the reconstruct tiers; unset or 0 keeps the tiers, and unset means 1 under
    EXL3_STABLE_ARITHMETIC=1. Needs the slot accumulation of EXL3_MOE_FUSED_DET (default on)
    """
    environ = os.environ if environ is None else environ
    value = environ.get("EXL3_MOE_FUSED_PREFILL")
    if value is None or value == "0":
        enabled = stable_arithmetic_enabled(environ)
    elif value == "1":
        enabled = True
    else:
        raise ValueError(f"EXL3_MOE_FUSED_PREFILL must be 0 or 1, got {value!r}")
    if enabled and environ.get("EXL3_MOE_FUSED_DET", "1") == "0":
        raise ValueError("EXL3_MOE_FUSED_PREFILL=1 needs the ordered expert accumulation that "
                         "EXL3_MOE_FUSED_DET=0 turns off")
    return enabled


def exact_rows_enabled(environ = None) -> bool:
    """
    EXL3_EXACT_ROWS: 1 makes a cached DeepSeek-V4.1 forward of one sequence with 2 to EXACT_ROWS_MAX
    rows (a draft verification) compute every row with the arithmetic of a one-row decode step, so
    generation with a draft returns the tokens generation without one returns; unset or 0 keeps the
    ordinary dispatch. It keeps the one-row arithmetic that EXL3_STABLE_ARITHMETIC=1 replaces, and
    the per-pair kernel of EXL3_DSV41_FUSED_COMPRESS takes the rows of the call: both are refused
    """
    environ = os.environ if environ is None else environ
    value = environ.get("EXL3_EXACT_ROWS", "0")
    if value not in ("0", "1"):
        raise ValueError(f"EXL3_EXACT_ROWS must be 0 or 1, got {value!r}")
    if value == "0":
        return False
    if stable_arithmetic_enabled(environ):
        raise ValueError("EXL3_EXACT_ROWS=1 keeps the arithmetic of a one-row decode step, which "
                         "EXL3_STABLE_ARITHMETIC=1 replaces; unset one of them")
    if environ.get("EXL3_DSV41_FUSED_COMPRESS", "0") != "0":
        raise ValueError("EXL3_EXACT_ROWS=1 pools DeepSeek-V4.1's compressed entries one at a time, "
                         "which the fused kernel of EXL3_DSV41_FUSED_COMPRESS does not; unset one of them")
    return True


def exact_rows(rows: int, enabled: bool = None) -> bool:
    """
    Whether a call of `rows` rows of one sequence is in the row range of EXL3_EXACT_ROWS: 2 to
    EXACT_ROWS_MAX. A one-row call is the reference and is never changed; a longer one is prefill
    """
    enabled = EXACT_ROWS if enabled is None else enabled
    return bool(enabled) and 1 < rows <= EXACT_ROWS_MAX


def require_whole_k_moe(key: str, built: bool, stable: bool = None):
    """
    EXL3_STABLE_ARITHMETIC runs every fused MoE launch through the kernel's whole-K instances, which
    an extension built with EXLLAMA_NO_WHOLE_K_MOE leaves out: an MoE layer that would need them is
    refused when it loads (built: exllamav3_ext.exl3_moe_whole_k_built())
    """
    stable = STABLE_ARITHMETIC if stable is None else stable
    if stable and not built:
        raise ValueError(f"{key}: EXL3_STABLE_ARITHMETIC=1 computes MoE experts with the whole-K fused "
                         f"MoE kernels, which this build of exllamav3_ext leaves out (it was built with "
                         f"EXLLAMA_NO_WHOLE_K_MOE set); rebuild the extension without "
                         f"EXLLAMA_NO_WHOLE_K_MOE, or unset EXL3_STABLE_ARITHMETIC")


# Fused-only prefill: the fused tier's row limit. A selection bound, not a buffer size (the
# kernel stripes rows through its temp buffers and checks its int32 addressing itself)
FUSED_COUNT_LIMIT = (1 << 31) - 1

# EXL3_EXACT_ROWS: the widest call it covers. dsa_attn gives calls of up to this many query rows the
# split softmax a one-row step takes (attention_fn/dsa_triton.py, SPLIT_MAX_ROWS)
EXACT_ROWS_MAX = 8

STABLE_ARITHMETIC = stable_arithmetic_enabled()
HGEMM_FIXED_ROWS = hgemm_fixed_rows()
FUSED_PREFILL = fused_prefill_enabled()
EXACT_ROWS = exact_rows_enabled()
