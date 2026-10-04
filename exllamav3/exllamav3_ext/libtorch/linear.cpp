#include <Python.h>
#include "linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../exact_rows.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../add.cuh"

void BC_LinearFP16::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    // EXL3_HGEMM_FIXED_ROWS: FP16 operands take the fixed-row GEMM (as LinearFP16 does in Python);
    // torch's matmul picks its algorithm by row count
    bool fixed_rows = hgemm_fixed_rows() && x.dtype() == at::kHalf && weight.dtype() == at::kHalf;
    if (x.dtype() == y.dtype() && !graph && !fixed_rows)
        at::matmul_out(y, x, weight);
    else
        hgemm_gr(x, weight, y, graph);

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearFP16::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

//void BC_LinearFP16::run_cublas(const at::Tensor& x, at::Tensor& y)
//{
//    hgemm(x, weight, y);
//    if (bias)
//        y.add_(bias.value());
//}

void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (x.numel() == x.size(-1))
    {
        exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0, graph);
    }
    else
    {
        TORCH_CHECK(!graph || graph->disabled, "BC_LinearEXL3 invoked with graph and bsz > 1");
        at::Tensor xh_ = at::empty_like(x);
        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);
    }

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

at::Tensor BC_LinearEXL3::run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32)
{
    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() = out_features;

    at::Tensor y = at::empty(
        out_shape,
        x.options().dtype(output_fp32 ? at::kFloat : at::kHalf)
    );
    if (out_features == 0) return y;

    at::Tensor x_flat = x.view({-1, x.size(-1)});
    at::Tensor y_flat = y.view({-1, out_features});
    run(x_flat, y_flat);
    return y;
}

// EXL3_EXACT_ROWS (exact_rows.h): run_alloc for a call of 2 to EXACT_ROWS_MAX rows. Every row is a
// call of run_gr of its own, on one-row views of x and of the result: its one-row arm, so the row
// gets the kernel, grid, workspace and launch-autotune record of a one-row call, and its own bias
// add. Nothing is hoisted out of the loop, and the rows share only what one-row calls reuse (xh, the
// int8 workspace), in stream order
at::Tensor BC_LinearEXL3::run_alloc_rows(const at::Tensor& x, int64_t out_features, bool output_fp32)
{
    TORCH_CHECK(x.dim() >= 2, "run_alloc_rows: x must have at least 2 dimensions");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "run_alloc_rows: x must be a contiguous CUDA tensor");
    TORCH_CHECK(out_features > 0, "run_alloc_rows: out_features must be positive");
    const int64_t in_features = x.size(-1);
    TORCH_CHECK(in_features > 0, "run_alloc_rows: x has no columns");
    const int64_t rows = x.numel() / in_features;
    TORCH_CHECK(rows >= 2 && rows <= EXACT_ROWS_MAX,
                "run_alloc_rows: a row-exact call takes 2 to ", EXACT_ROWS_MAX, " rows, got ", rows);

    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() = out_features;

    at::Tensor y = at::empty(
        out_shape,
        x.options().dtype(output_fp32 ? at::kFloat : at::kHalf)
    );

    at::Tensor x_flat = x.view({rows, in_features});
    at::Tensor y_flat = y.view({rows, out_features});
    for (int64_t r = 0; r < rows; ++r)
    {
        at::Tensor xr = x_flat.slice(0, r, r + 1);
        at::Tensor yr = y_flat.slice(0, r, r + 1);
        run_gr(xr, yr, nullptr);
    }
    return y;
}
