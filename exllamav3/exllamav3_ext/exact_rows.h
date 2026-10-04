#pragma once

#include <cstdint>

// EXL3_EXACT_ROWS=1 (doc/env_vars.md): the row-exact entry points. Each takes a call of 2 to
// EXACT_ROWS_MAX rows of one sequence and gives every row the bits of a one-row call, with no
// Python between the rows: the launches a one-row call makes, once per row, or, with
// EXL3_INT8_GEMV=0, where the kernel of the one-row call computes a row the same way whatever the
// row count (EXACT_ROWS_CAP_ONE_LAUNCH), one launch for all rows under that call's launch record.
// The extension does not read EXL3_EXACT_ROWS: Python passes a flagged call to these entry points,
// and keeps its own row loops for an extension that does not report them (model/math_policy.py,
// exact_rows_native)
#define EXACT_ROWS_MAX 8            // math_policy.EXACT_ROWS_MAX

// One bit per entry point (math_policy.EXACT_ROWS_CAP_*)
#define EXACT_ROWS_CAP_LINEAR 1     // BC_LinearEXL3::run_alloc_rows
#define EXACT_ROWS_CAP_MGEMM 2      // exl3_mgemm_rows
#define EXACT_ROWS_CAP_ROUTER 4     // routing_ds3_nogroup_rows
#define EXACT_ROWS_CAP_MOE 8        // BC_BlockSparseMLP::run_bszN_rows, rows_exact_ok
#define EXACT_ROWS_CAP_HC 16        // hc_collapse_rows, hc_partials_rows
// Not an entry point: with EXL3_INT8_GEMV=0, run_alloc_rows and exl3_mgemm_rows make one launch for
// the rows, under the launch record of a one-row call, where that call is the cooperative kernel
// (quant/exl3_gemm.cuh, exl3_gemm_one_row_route_ok and one_row_route). The launch per row everywhere
// else, and for every call with the int8 path on
#define EXACT_ROWS_CAP_ONE_LAUNCH 32
#define EXACT_ROWS_CAP_HGEMM 64     // hgemm_rows
// An argument of run_bszN_rows (grouped): on the GPU types of exact_rows_device_ok, for a module
// without a shared-expert gate (BC_BlockSparseMLP::rows_grouped), the rows are launched as an
// unflagged call launches them (the rotation launch, the slots that picked one expert as rows of
// one tile), with the tile and the split-k factor of a one-row call (quant/exl3_moe_coop.cuh,
// exl3_moe_coop_launch). Without the argument, and everywhere else, a launch per slot, rotated
// in-block
#define EXACT_ROWS_CAP_MOE_GROUPED 128
// An argument of routing_ds3_nogroup_rows (one_launch): the rows are projected with one launch of
// the FMA GEMV where a one-row call is that GEMV (routing.cu, routing_gemv with one_row_route).
// Without the argument, and where the one-row call is anything else, a launch per row
#define EXACT_ROWS_CAP_ROUTER_ONE_LAUNCH 256

// exl3_gemm_gr and exl3_mgemm_gr with one_row_route: nothing was launched, the caller makes the
// one-row launches itself. Never a launch tag (those are 0 and up)
#define EXACT_ROWS_NO_LAUNCH (-1)

// OR of the entry points this build has
int exact_rows_caps();

// The GPU types on which several rows share one tensor-core tile under EXL3_EXACT_ROWS: compute
// capability 8.0, 8.9 and 12.0. The kernel sources show that the operations on a row and their
// order do not depend on the other rows. What they cannot show is a property of the instruction:
// that an element of a tensor-core product is computed from its own row of the first operand and
// its own column of the second, whatever the other rows and columns of the fragments hold and
// wherever the row sits in its m16 tile. It is assumed of three uses, each a separate assumption
// with a test of its own, and those are the capabilities on which each is compared bit for bit
// with one-row calls (tests/test_dsv41_exact_rows_gpu_.py):
// - the cooperative EXL3 linears (EXACT_ROWS_CAP_ONE_LAUNCH): the FP32-accumulate instruction
//   (ptx.cuh, f32.f16.f16.f32), a row among the other rows of its tile (test_one_launch_*)
// - the grouped MoE tile (EXACT_ROWS_CAP_MOE_GROUPED): the FP16-accumulate instruction
//   (quant/exl3_gemv_kernel.cuh, mma_ab_h, f16.f16.f16.f16, folded to FP32 every FOLD slices). A
//   slot is fragment row 0 with rows 1 to 15 zero in a one-row call, and row r among up to 8
//   non-zero rows when grouped: independence of the other rows and of the row's position
//   (test_native_moe_selections, whose "every row the same experts" case fills rows 0 .. K - 1 of
//   one tile, and test_moe_grouped_form, at every K on every GPU type)
// - the index scorer of a shared selection pass (Python: modules/dsv41_select.py, ROWS_PASS_SM,
//   the same list): the FP32-accumulate instruction as Triton emits it for tl.dot, a key column
//   among the other key columns of its tile (test_select_rows_equal_one_row_calls)
// A form is served on a GPU type only with its tests green there. False for every other device
// and for an index outside 0 .. MAX_DEVICES - 1
bool exact_rows_device_ok(int device);

// Calls of run_alloc_rows and exl3_mgemm_rows served with one launch since the extension was
// loaded. Both routes give the same bits, so a test or the probe can tell them apart only here
uint64_t exact_rows_one_launches();
void exact_rows_count_one_launch();

// Launches the other batched forms served since the extension was loaded, by capability bit:
// EXACT_ROWS_CAP_MOE_GROUPED, the launch pairs of run_bszN_rows made with the rows grouped (the
// shared expert's and the routed one count one each); EXACT_ROWS_CAP_ROUTER_ONE_LAUNCH, the calls
// of routing_ds3_nogroup_rows that projected their rows with one launch. 0 for a bit that counts
// nothing
uint64_t exact_rows_served(int cap);
void exact_rows_count_served(int cap);
