"""
Opt-in GPU arithmetic policies (doc/env_vars.md). Each variable is parsed strictly and read once,
when exllamav3 is imported, into the module-level constants at the bottom; the native side reads
the same variables on first use, with the same rules. Set them before importing exllamav3. The
default of every policy is the ordinary dispatch.

Torch-free, so tests can load this file by path.
"""
import os


def hgemm_fixed_rows(environ = None) -> int:
    """
    EXL3_HGEMM_FIXED_ROWS: 128 runs the native GEMMs (hgemm.cu) as fixed 128-row cuBLAS tiles
    and routes FP16 x FP16 linears through them; unset or 0 keeps the default dispatch. Mirrors
    hgemm_fixed_rows() in hgemm.cu: any other value is an error
    """
    environ = os.environ if environ is None else environ
    value = environ.get("EXL3_HGEMM_FIXED_ROWS")
    if value is None or value == "0":
        return 0
    if value != "128":
        raise ValueError(f"EXL3_HGEMM_FIXED_ROWS must be 0 or 128, got {value!r}")
    return 128


HGEMM_FIXED_ROWS = hgemm_fixed_rows()
