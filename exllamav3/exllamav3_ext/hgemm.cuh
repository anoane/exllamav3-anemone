#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

// EXL3_HGEMM_FIXED_ROWS: 128 when the GEMMs of hgemm.cu run as fixed 128-row cuBLAS tiles (also
// implied by EXL3_STABLE_ARITHMETIC), else 0. Read once, on first use; the native Graph wrapper and
// BC_LinearFP16 consult it too
int hgemm_fixed_rows();

// EXL3_HGEMM_FP32_REDUCTION: 1 (default) when the cuBLAS GEMMs of hgemm.cu disallow reduced-precision
// (fp16) intermediate reductions, 0 when cuBLAS may use them. Read once, on first use
int hgemm_fp32_reduction();

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
);

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
);
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
);

// EXL3_EXACT_ROWS (exact_rows.h, EXACT_ROWS_CAP_HGEMM): a @ b for 2 to EXACT_ROWS_MAX rows of a, one
// GEMM per row, each the call hgemm makes for a one-row forward (M = 1, an output of its own), with
// no Python between the rows. cuBLAS picks its kernel and its reduction split from the whole
// problem shape, so the rows cannot share a call. Returns (rows, N), FP32 or FP16
at::Tensor hgemm_rows
(
    const at::Tensor& a,
    const at::Tensor& b,
    bool output_fp32
);

// fp16-accumulator tensor-core GEMM with fp32 accumulation across K (hgemm_f16acc.cu): full
// rate on GeForce, where the fp32-accumulator MMA runs at half rate. hgemm_f16acc_try runs it
// when the device gains (rate probe, EXL3_HGEMM_F16ACC=0/1 overrides) and the shape is covered
// (K % 64, N % 128, aligned packed input rows and nonoverlapping output rows/batches).
// Shape/batch heuristics decide whether it pays; hgemm_recon adds the cuBLAS fallback.
bool hgemm_f16acc_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c);
void hgemm_f16acc(at::Tensor a, at::Tensor b, at::Tensor c);
int hgemm_f16acc_status(int device);
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c);
