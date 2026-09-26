#pragma once

// EXL3_STABLE_ARITHMETIC=1: GPU arithmetic whose results do not depend on how many rows a call has
// (doc/env_vars.md). Implies the fixed-row GEMM (hgemm_fixed_rows) and selects the whole-K fused
// MoE kernel instances (exl3_moe). Read once, on first use; unset or 0 = off, any other value is an
// error. Mirrors model/math_policy.py
bool stable_arithmetic();
