#pragma once

#include <ATen/Tensor.h>
#include "../graph.cuh"

// Whether a call of 2 to EXACT_ROWS_MAX rows with these widths on CUDA device `device` may be one
// launch under the launch record of a one-row call (one_row_route below). All of:
// - EXL3_INT8_GEMV=0, read on every call. With the int8 path on, the one-row call of a mul1 tensor
//   is offered to the int8 GEMV, and a call of any tensor keeps the launch per row
// - both widths multiples of 128: the Hadamard transforms work on 128-element blocks of the
//   flattened rows, and a block must not straddle two rows
// - a device of compute capability 8.0, 8.9 or 12.0. The kernel source shows that the operations
//   on a row and their order do not depend on the row count. That a tensor-core output row depends
//   on its own input row alone, whatever the other rows of its m16 tile hold, is a property of the
//   instruction: those are the capabilities on which the rows of such a launch are compared bit
//   for bit with one-row calls (tests/test_dsv41_exact_rows_gpu_.py). Every other device keeps the
//   launch per row, sm_86 with its FP16 accumulation (EXL3_GEMM_H_ACC) among them
// exl3_gemm_gr and exl3_mgemm_gr apply it themselves; a caller asks first only to spare the copies
// and the scratch of a call that would launch nothing
bool exl3_gemm_one_row_route_ok(int device, int64_t size_k, int64_t size_n);

// one_row_route (EXL3_EXACT_ROWS, exact_rows.h): a call of 2 to EXACT_ROWS_MAX rows is routed as
// a call of one row is, and launched only where exl3_gemm_one_row_route_ok holds and that route is
// the cooperative kernel with a launch-autotune record already in the process: one launch for all
// rows under that record (kernel shape, block count, concurrency), which gives every row the bits
// of a one-row call. The record is looked up, never tuned, loaded or stored by this call, and
// checked against the kernel of this weight before the launch. Everything else (the FP16 GEMV as
// the one-row route, no record yet, a record that is not this weight's) launches nothing and
// returns EXACT_ROWS_NO_LAUNCH; C and A_had are then untouched. Needs suh, A_had (m * k elements)
// and svh, no graph and no forced shape or block count
int exl3_gemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms,
    Graph* graph,
    bool one_row_route = false
);

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
);

int exl3_mgemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    float K,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {},
    // Sliced mode (see exl3_mgemm_gr): per-entry full row width of the slice's matrix, per-entry
    // source matrix index (suh and A_had are then per source), and the number of sources
    const c10::optional<at::Tensor>& n_stride_list = {},
    const c10::optional<at::Tensor>& had_src_list = {},
    int num_had_src = 0,
    // As exl3_gemm_gr's: A (G, m, k) and C (G, m, n) with 2 to EXACT_ROWS_MAX rows m are launched
    // once under the record of the m == 1 call, or not at all (EXACT_ROWS_NO_LAUNCH). Plain grouped
    // calls only: no weights, range filtering, per-matrix widths, sliced mode, graph or forced launch
    bool one_row_route = false
);

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    float K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {},
    const c10::optional<at::Tensor>& n_stride_list = {},
    const c10::optional<at::Tensor>& had_src_list = {},
    int num_had_src = 0
);

// EXL3_EXACT_ROWS (exact_rows.h): the grouped projection of 2 to EXACT_ROWS_MAX rows, every row with
// the bits of the exl3_mgemm call of a one-row step: that call once per row, or, where
// exl3_gemm_one_row_route_ok holds, one launch for the rows under that call's launch record. A
// (rows, G, k) and C (rows, G, n) are row-major, A_had is the one-row call's (G, 1, k) scratch,
// indices (1, G). Returns the launch tag the rows share
int exl3_mgemm_rows
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& indices,
    float K,
    bool mcg,
    bool mul1
);
