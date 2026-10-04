#pragma once

#include <ATen/Tensor.h>
#include "../graph.cuh"

// Whether exl3_gemm_gr offers a call to the int8-activation GEMV (exl3_gemv_int8.cuh) before any
// other kernel: the mul1 codebook with EXL3_INT8_GEMV not 0. The variable is read on every call
bool exl3_gemm_asks_int8(bool mul1);

// one_row_route (EXL3_EXACT_ROWS, exact_rows.h): a call of 2 to EXACT_ROWS_MAX rows is routed as
// a call of one row is, and launched only where that route is the cooperative kernel with a
// launch-autotune record already in the process: one launch for all rows under that record (kernel
// shape, block count, concurrency), which gives every row the bits of a one-row call. The record
// is looked up, never tuned, loaded or stored by this call. Every other route (the int8 GEMV, the
// FP16 GEMV, no record yet, a width that is not a multiple of 128) launches nothing and returns
// EXACT_ROWS_NO_LAUNCH; C and A_had are then untouched. Needs suh, A_had (m * k elements) and svh,
// no graph and no forced shape or block count
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
// the bits of the exl3_mgemm call of a one-row step: one launch for the rows under that call's
// launch record, or that call once per row. A (rows, G, k) and C (rows, G, n) are row-major, A_had
// is the one-row call's (G, 1, k) scratch, indices (1, G). Returns the launch tag the rows share
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
