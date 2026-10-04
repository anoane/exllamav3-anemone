#pragma once

#include <cstdint>

// EXL3_EXACT_ROWS=1 (doc/env_vars.md): the row-exact entry points. Each takes a call of 2 to
// EXACT_ROWS_MAX rows of one sequence and gives every row the bits of a one-row call, with no
// Python between the rows: the launches a one-row call makes, once per row, or, where the kernel of
// the one-row call computes a row the same way whatever the row count (EXACT_ROWS_CAP_ONE_LAUNCH),
// one launch for all rows under that call's launch record. The extension does not read the variable:
// Python passes a flagged call to these entry points, and keeps its own row loops for an extension
// that does not report them (model/math_policy.py, exact_rows_native)
#define EXACT_ROWS_MAX 8            // math_policy.EXACT_ROWS_MAX

// One bit per entry point (math_policy.EXACT_ROWS_CAP_*)
#define EXACT_ROWS_CAP_LINEAR 1     // BC_LinearEXL3::run_alloc_rows
#define EXACT_ROWS_CAP_MGEMM 2      // exl3_mgemm_rows
#define EXACT_ROWS_CAP_ROUTER 4     // routing_ds3_nogroup_rows
#define EXACT_ROWS_CAP_MOE 8        // BC_BlockSparseMLP::run_bszN_rows, rows_exact_ok
#define EXACT_ROWS_CAP_HC 16        // hc_collapse_rows, hc_partials_rows
// Not an entry point: run_alloc_rows and exl3_mgemm_rows make one launch for the rows, under the
// launch record of a one-row call, where that call is the cooperative kernel (quant/exl3_gemm.cuh,
// one_row_route), and the launch per row everywhere else
#define EXACT_ROWS_CAP_ONE_LAUNCH 32

// exl3_gemm_gr and exl3_mgemm_gr with one_row_route: nothing was launched, the caller makes the
// one-row launches itself. Never a launch tag (those are 0 and up)
#define EXACT_ROWS_NO_LAUNCH (-1)

// OR of the entry points this build has
int exact_rows_caps();

// Calls of run_alloc_rows and exl3_mgemm_rows served with one launch since the extension was
// loaded. Both routes give the same bits, so a test or the probe can tell them apart only here
uint64_t exact_rows_one_launches();
void exact_rows_count_one_launch();
