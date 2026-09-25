#pragma once

// EXL3_STABLE_ARITHMETIC=1: GPU arithmetic whose results do not depend on how many rows a call has
// (doc/env_vars.md). Implies the fixed-row GEMM (hgemm_fixed_rows), selects the whole-K fused MoE
// kernel instances (exl3_moe) and the row-count-independent router projection (routing_gemv,
// rg_slices). Read once, on first use; unset or 0 = off, any other value is an error. Mirrors
// model/math_policy.py
bool stable_arithmetic();
