"""
Opt-in GPU arithmetic policies (doc/env_vars.md). Each variable is parsed strictly and read once,
when exllamav3 is imported, into the module-level constants at the bottom; the native side reads
the same variables on first use, with the same rules. Set them before importing exllamav3. The
default of every policy is the ordinary dispatch.

EXL3_STABLE_ARITHMETIC=1 implies EXL3_HGEMM_FIXED_ROWS=128 and EXL3_MOE_FUSED_PREFILL=1: each may be
left unset or set to that value, anything else is refused.

EXL3_EXACT_ROWS is read in Python only. The extension does not read it: it reports which row-exact
entry points it has (exact_rows_caps, exact_rows.h), and Python passes a flagged call to them or
runs its own row loops (exact_rows_native). How an entry point gives the rows their one-row bits
(a launch per row, or one launch under the launch record of a one-row call,
EXACT_ROWS_CAP_ONE_LAUNCH) is the extension's choice per call; Python passes the same call either way.
Four operations of a flagged call have a batched form that is Python alone and runs with every
extension (exact_rows_forms, EXACT_ROWS_FORM_*).

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


def exact_rows_native(ext_module, enabled: bool = None) -> int:
    """
    The row-exact entry points of the extension a flagged call of EXL3_EXACT_ROWS may use: the OR of
    the EXACT_ROWS_CAP_* bits it reports (exact_rows_caps() in exact_rows.cpp), or 0 without the
    switch and for an extension built before these entry points. With a bit clear, the operation
    keeps its Python loop of one call per row, which gives the same results
    """
    enabled = EXACT_ROWS if enabled is None else enabled
    if not enabled:
        return 0
    caps = getattr(ext_module, "exact_rows_caps", None)
    return 0 if caps is None else int(caps())


def exact_rows_forms(enabled: bool = None) -> int:
    """
    EXL3_EXACT_ROWS: the batched forms of a flagged call that are Python alone, as the OR of their
    EXACT_ROWS_FORM_* bits: all of them with the switch, 0 without. They need no entry point of the
    extension, so they run with every extension, the ones built before the row-exact entry points
    included. With a bit clear the operation keeps the form that makes one call per row, entry or
    token (the extension's per-row entry points where it has them), which gives the same results.
    Not a setting: each consumer holds the value as ROWS_FORMS, and the tests and
    tools/dsv41_rowprobe.py (--exact-rows-forms) clear bits there to compare and to time the two
    forms of one operation
    """
    enabled = EXACT_ROWS if enabled is None else enabled
    return EXACT_ROWS_FORMS_ALL if enabled else 0


def exact_rows_sum(terms: int, width: int) -> bool:
    """
    EXL3_EXACT_ROWS: whether torch's sum over the middle dimension of a contiguous FP32 CUDA operand
    (rows, terms, width) may run over all rows of a flagged call at once and still give every row
    the bits of the sum over its own (1, terms, width) operand. torch has no argument that pins the
    layout of a reduction, so this is the bound inside which its layout (ATen/native/cuda/
    Reduce.cuh, setReduceConfig) lets ONE thread add all terms of an output, in an order fixed by
    the term count alone, whatever the number of outputs:

      * the reduced dimension is not the fastest-striding one (width >= 2), so outputs, not
        terms, are split across the lanes of a warp;
      * terms are split across warps only from min(16 * block height, 256) terms per thread on.
        Fewer than EXACT_ROWS_SUM_TERMS terms never reach that at any block height. Up to
        EXACT_ROWS_SUM_TERMS_WIDE the block is at least 16 threads high only when the outputs are
        not loaded as vectors, which an odd width guarantees (an even width may give a block 4 or
        8 threads high for some output counts and not others: a real row-count dependence).

    Everything else keeps the per-row form (terms: size of the reduced dimension, width: size of
    the operand's last dimension).

    The bound was read in the Reduce.cuh of torch 2.14.0 (setReduceConfig, set_block_dimension,
    get_output_vec_size, thread_reduce_impl; the header ships in the wheel under
    torch/include/ATen/native/cuda/). It is a statement about that source, and this tree holds
    none of it: for any other torch the layout is not shown here. So callers do not rest on the
    bound alone. They compare the two forms bitwise on the first use of every operand shape on
    every device, in the process that serves, and keep the per-row form where they differ
    (modules/dsv41_block.py, _sum_rows)
    """
    if terms < 2 or width < 2:
        return False
    if terms < EXACT_ROWS_SUM_TERMS:
        return True
    return terms < EXACT_ROWS_SUM_TERMS_WIDE and width % 2 == 1


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

# EXL3_EXACT_ROWS: one bit per row-exact entry point of the extension (exact_rows.h)
EXACT_ROWS_CAP_LINEAR = 1       # BC_LinearEXL3.run_alloc_rows
EXACT_ROWS_CAP_MGEMM = 2        # exl3_mgemm_rows
EXACT_ROWS_CAP_ROUTER = 4       # routing_ds3_nogroup_rows
EXACT_ROWS_CAP_MOE = 8          # BC_BlockSparseMLP.run_bszN_rows, rows_exact_ok
EXACT_ROWS_CAP_HC = 16          # hc_collapse_rows, hc_partials_rows
# Not an entry point, and nothing here selects by it: with EXL3_INT8_GEMV=0, run_alloc_rows and
# exl3_mgemm_rows make one launch for the rows where a one-row call is the cooperative FP16 kernel,
# on the GPU types that launch was verified on, and count those calls (exact_rows_one_launches).
# With the int8 path on they launch per row. Tests and tools/dsv41_rowprobe.py read both
EXACT_ROWS_CAP_ONE_LAUNCH = 32
EXACT_ROWS_CAP_HGEMM = 64       # hgemm_rows
# An argument of run_bszN_rows: on the GPU types that launch was verified on, the rows that picked
# one expert are grouped, as in an unflagged call, with the tile of a one-row call
# (BC_BlockSparseMLP.rows_grouped says for a layer, exact_rows_served counts the launch pairs)
EXACT_ROWS_CAP_MOE_GROUPED = 128
# An argument of routing_ds3_nogroup_rows: the rows are projected with one launch of the FMA GEMV
# where a one-row call is that GEMV (exact_rows_served counts the calls)
EXACT_ROWS_CAP_ROUTER_ONE_LAUNCH = 256

# EXL3_EXACT_ROWS: the term counts below which a torch sum over the rows of a flagged call is
# row-exact (exact_rows_sum): any width, and an odd width
EXACT_ROWS_SUM_TERMS = 16
EXACT_ROWS_SUM_TERMS_WIDE = 256

# EXL3_EXACT_ROWS: one bit per batched form that is Python alone (exact_rows_forms)
EXACT_ROWS_FORM_SELECT = 1      # DSV41Attention._index_select: one select_topk pass for the rows
EXACT_ROWS_FORM_SUMS = 2        # dsv41_block: the hyper-connection sums over the rows (_sum_rows)
EXACT_ROWS_FORM_COMPRESS = 4    # CompressCarry.step: the closed entries pooled and normalized at once
EXACT_ROWS_FORM_GATE = 8        # DSV41Engram._gate_rows: the reductions per token, the rest at once
EXACT_ROWS_FORMS_ALL = EXACT_ROWS_FORM_SELECT | EXACT_ROWS_FORM_SUMS | EXACT_ROWS_FORM_COMPRESS | EXACT_ROWS_FORM_GATE

STABLE_ARITHMETIC = stable_arithmetic_enabled()
HGEMM_FIXED_ROWS = hgemm_fixed_rows()
FUSED_PREFILL = fused_prefill_enabled()
EXACT_ROWS = exact_rows_enabled()
