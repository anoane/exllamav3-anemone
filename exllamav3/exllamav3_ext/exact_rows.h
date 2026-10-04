#pragma once

// EXL3_EXACT_ROWS=1 (doc/env_vars.md): the row-exact entry points. Each takes a call of 2 to
// EXACT_ROWS_MAX rows of one sequence and gives every row the launches a one-row call makes, with no
// Python between the rows. The extension does not read the variable: Python passes a flagged call to
// these entry points, and keeps its own row loops for an extension that does not report them
// (model/math_policy.py, exact_rows_native)
#define EXACT_ROWS_MAX 8            // math_policy.EXACT_ROWS_MAX

// One bit per entry point (math_policy.EXACT_ROWS_CAP_*)
#define EXACT_ROWS_CAP_LINEAR 1     // BC_LinearEXL3::run_alloc_rows
#define EXACT_ROWS_CAP_MGEMM 2      // exl3_mgemm_rows
#define EXACT_ROWS_CAP_ROUTER 4     // routing_ds3_nogroup_rows
#define EXACT_ROWS_CAP_MOE 8        // BC_BlockSparseMLP::run_bszN_rows, rows_exact_ok
#define EXACT_ROWS_CAP_HC 16        // hc_collapse_rows, hc_partials_rows

// OR of the entry points this build has
int exact_rows_caps();
