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

// EXL3_EXACT_ROWS (exact_rows.h): run_alloc for a call of 2 to EXACT_ROWS_MAX rows, every row with
// the bits of a one-row call. Where a one-row call is the cooperative kernel with a launch record in
// the process (exl3_gemm_gr, one_row_route: not offered to the int8 GEMV, so EXL3_INT8_GEMV=0 or a
// tensor outside the mul1 codebook), the rows are ONE launch under that record, on a Hadamard
// scratch of their own, and each row then gets the bias add of its one-row call. Everywhere else
// every row is a call of run_gr of its own, on one-row views of x and of the result: its one-row
// arm, so the row gets the kernel, grid, workspace and launch-autotune record of a one-row call, and
// its own bias add. Nothing is hoisted out of that loop, and its rows share only what one-row calls
// reuse (xh, the int8 workspace), in stream order
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
    // What exl3_gemm_gr checks for every row, before the result is allocated, and the device: the
    // rows are launched on the device of x with the module's weight and workspace pointers
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DIM(trellis, 3);
    TORCH_CHECK(in_features == trellis.size(0) * 16 && out_features == trellis.size(1) * 16,
                "run_alloc_rows: x or out_features does not match the quantized weight");
    TORCH_CHECK(x.device() == trellis.device(), "run_alloc_rows: x must be on the module's device");
    // The scales a launch reads through raw pointers: one element per input and per output column
    TORCH_CHECK(suh.defined() && svh.defined(), "run_alloc_rows: the module has no suh or svh");
    TORCH_CHECK_DTYPE(suh, kHalf);
    TORCH_CHECK_DTYPE(svh, kHalf);
    TORCH_CHECK(suh.numel() == in_features && svh.numel() == out_features && suh.device() == x.device() &&
                svh.device() == x.device(), "run_alloc_rows: suh or svh does not match the quantized weight");

    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() = out_features;

    at::Tensor y = at::empty(
        out_shape,
        x.options().dtype(output_fp32 ? at::kFloat : at::kHalf)
    );

    at::Tensor x_flat = x.view({rows, in_features});
    at::Tensor y_flat = y.view({rows, out_features});

    // One launch for the rows. exl3_gemm_gr decides (one_row_route); the test here only spares the
    // scratch where a one-row call is offered to the int8 path and the answer is known to be no
    if (!exl3_gemm_asks_int8(mul1))
    {
        at::Tensor xh_rows = at::empty_like(x_flat);
        if (exl3_gemm_gr(x_flat, trellis, y_flat, suh, xh_rows, svh, -1, mcg, mul1, 0, nullptr, true) != EXACT_ROWS_NO_LAUNCH)
        {
            if (bias)
            {
                for (int64_t r = 0; r < rows; ++r)
                {
                    at::Tensor yr = y_flat.slice(0, r, r + 1);
                    add_gr(yr, bias.value(), yr, nullptr);
                }
            }
            exact_rows_count_one_launch();
            return y;
        }
    }

    for (int64_t r = 0; r < rows; ++r)
    {
        at::Tensor xr = x_flat.slice(0, r, r + 1);
        at::Tensor yr = y_flat.slice(0, r, r + 1);
        run_gr(xr, yr, nullptr);
    }
    return y;
}
