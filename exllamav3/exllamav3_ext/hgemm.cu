#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>
#include <cstdlib>
#include <cstring>
#include <algorithm>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float32 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

// EXL3_HGEMM_FIXED_ROWS=128 runs every GEMM in this file as a sequence of 128-row cuBLAS calls
// and bypasses the fp16-accumulator kernel (hgemm_f16acc.cu); 0 or unset keeps the default
// dispatch, any other value is an error. Read once, on first use
int hgemm_fixed_rows()
{
    static const int rows = []()
    {
        const char* value = std::getenv("EXL3_HGEMM_FIXED_ROWS");
        if (!value || !std::strcmp(value, "0")) return 0;
        TORCH_CHECK(!std::strcmp(value, "128"), "EXL3_HGEMM_FIXED_ROWS must be 0 or 128");
        return 128;
    }();
    return rows;
}

// EXL3_HGEMM_FP32_REDUCTION: 1 (default) keeps the intermediate reductions of every cuBLAS GEMM
// in this file in fp32 (require_fp32_reductions, below); 0 leaves the handle's math mode as it is,
// so cuBLAS may reduce through an fp16 output type; any other value is an error. Read once, on
// first use
int hgemm_fp32_reduction()
{
    static const int on = []()
    {
        const char* value = std::getenv("EXL3_HGEMM_FP32_REDUCTION");
        if (!value || !std::strcmp(value, "1")) return 1;
        TORCH_CHECK(!std::strcmp(value, "0"), "EXL3_HGEMM_FP32_REDUCTION must be 0 or 1");
        return 0;
    }();
    return on;
}

// COMPUTE_32F alone does not prohibit split-K reductions through an fp16 output
// type. Keep intermediate reductions in the promised accumulator precision.
// The handle belongs to PyTorch's thread-local pool: restore its previous mode
// after launch, including when GEMM returns an error, rather than leaking policy
// into another user of the handle. This does not disable tensor cores. Returns
// the previous mode; with EXL3_HGEMM_FP32_REDUCTION=0 the mode is left unchanged
// (the caller's restore sets the same mode again)
static cublasMath_t require_fp32_reductions(cublasHandle_t handle)
{
    cublasMath_t previous;
    cublas_check(cublasGetMathMode(handle, &previous));
    if (!hgemm_fp32_reduction()) return previous;
    auto required = static_cast<cublasMath_t>(
        static_cast<int>(previous) | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    cublas_check(cublasSetMathMode(handle, required));
    return previous;
}

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream,
    bool allow_fixed_rows = true
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    const int64_t k64 = a.size(-1);
    TORCH_CHECK(k64 > 0, "hgemm: positive reduction dimension required");
    const int64_t m64 = a.numel() / k64;
    const int64_t n64 = b.size(-1);
    const auto int_limit = std::numeric_limits<int>::max();
    TORCH_CHECK(k64 <= int_limit && m64 <= int_limit && n64 <= int_limit, "hgemm: dimension exceeds int32");
    int size_k = k64, size_m = m64, size_n = n64;
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    if (allow_fixed_rows && hgemm_fixed_rows())
    {
        // Fixed row geometry: cuBLAS chooses its algorithm, including how the K reduction is
        // split, from the whole problem shape, so one input row can come out differently in a
        // 4096-row call than in a 2048-row one. Here every call is a series of identical
        // [128, K] x [K, N] GEMMs on a packed, zero-padded scratch tile, whatever M, the input
        // alignment or the output stride, so each output row depends only on its own input row
        // (for a given GPU, K and N). Not a guarantee across GPU types or of exact arithmetic.
        // Scratch is two tiles whatever M; allocations, copies and GEMMs use the given stream
        TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "hgemm: fixed-row GEMM requires packed inputs");
        // Reconstruct callers may pass a reusable output buffer with more rows than the
        // product; only the leading M rows are written
        TORCH_CHECK(c.numel() >= (int64_t) size_m * size_n, "hgemm: output too small for fixed-row GEMM");
        c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(stream, a.get_device()));
        const int rows = hgemm_fixed_rows();
        auto input = a.view({size_m, size_k});
        auto output = c.as_strided({size_m, size_n}, {c_stride_m, 1});
        auto a_tile = at::empty({rows, size_k}, a.options());
        auto c_tile = at::empty({rows, size_n}, c.options());
        for (int64_t begin = 0; begin < size_m; begin += rows)
        {
            const int count = std::min<int64_t>(rows, size_m - begin);
            a_tile.narrow(0, 0, count).copy_(input.narrow(0, begin, count));
            if (count < rows) a_tile.narrow(0, count, rows - count).zero_();
            hgemm_gemmex_impl(a_tile, b, c_tile, stream, false);
            output.narrow(0, begin, count).copy_(c_tile.narrow(0, 0, count));
        }
        return;
    }

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    const auto previous_math = require_fp32_reductions(cublas_handle);
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, size_k,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    const auto restore_status = cublasSetMathMode(cublas_handle, previous_math);
    cublas_check(r);
    cublas_check(restore_status);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS
    if (!hgemm_fixed_rows() && hgemm_f16acc_try(a, w, c)) return;

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    if (hgemm_fixed_rows())
    {
        // Fixed row geometry: one matrix at a time through the same 128-row tiles, so a
        // matrix's result does not depend on how many others share the batch
        for (int i = 0; i < batch; ++i)
            hgemm_gemmex_impl(a[i], w[i], c[i], stream);
        return;
    }

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    const auto previous_math = require_fp32_reductions(cublas_handle);
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    const auto restore_status = cublasSetMathMode(cublas_handle, previous_math);
    cublas_check(r);
    cublas_check(restore_status);
    cuda_check(cudaPeekAtLastError());
}
