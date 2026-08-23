#pragma once

#include "exl3_kernel_map.cuh"

#ifndef EXL3_RUNTIME_K_MIN
#define EXL3_RUNTIME_K_MIN 1
#endif
#ifndef EXL3_RUNTIME_K_MAX
#define EXL3_RUNTIME_K_MAX 8
#endif
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    // if (suh)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (A_, B, C_, MIN(size_m_, 16), size_k, size_n, locks, svh);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;

        if (size_m_ > 0 || svh)
            grid.sync();
    }

    // if (svh)
    /*
    {
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true>
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner<false, true>
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
     */
}

#define MAX_INDICES 128

__device__ int64_t v_indices[128];
__device__ half v_weights[128];
__device__ int bszm_sync;

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_mgemm_kernel(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    auto grid = cg::this_grid();

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
        int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;
    #endif

    // Pack indices within min_index <= idx < max_index

    if (min_index >= 0)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            int j = 0;
            for (int i = 0; i < bszm; ++i)
            {
                int idx = B_indices[i];
                if (idx >= min_index && idx < max_index)
                {
                    v_indices[j] = idx - min_index;
                    if (B_weights) v_weights[j] = B_weights[i];
                    j++;
                }
            }
            bszm_sync = j;
            for (; j < bszm; ++j)
            {
                v_indices[j] = -1;
            }
        }
        __threadfence();
        grid.sync();
        B_indices = v_indices;
        if (B_weights) B_weights = v_weights;
        bszm = bszm_sync;
    }

    for (int i = 0; i < bszm; i += gridDim.z)
    {
        int j = i + blockIdx.z;
        int mat_index = -1;
        int a_slot = j;      // input slab index (dual mode folds the two halves)
        const uint16_t* B = nullptr;
        if (j >= bszm) j = -1;
        else
        {
            // gu_dual maps slot j onto the interleaved gate/up table [g0,u0,g1,u1..]:
            // slots [0, bszm/2) = gate of B_indices[j], slots [bszm/2, bszm) = up of the same
            // expert — one cooperative launch covers both projections (same A row per pair)
            int jj = j, gu_off = 0;
            if (gu_dual)
            {
                int half = bszm >> 1;
                if (j >= half) { jj = j - half; gu_off = 1; }
                a_slot = jj;
            }
            mat_index = B_indices ? (int) B_indices[jj] : jj;
            if (mat_index >= 0)
            {
                if (gu_dual) mat_index = 2 * mat_index + gu_off;
                B = B_list[mat_index];
            }
        }

        // Had and input scales

        if (B)
        {
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* suh = suh_list[mat_index];
            const half* A_ = bszm_in == 1 ? A : A + a_slot * size_m * size_k;
            half* A_had_ = A_had + j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_hf_r_128_inner<true, false>
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh + (this_warp * 128) % size_k,
                    0.088388347648f  // 1/sqrt(128)
                );
        }

        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
        #else
            grid.sync();
        #endif

        // Matmul. Per-matrix output width/pointer when the caller supplies the lists
        // (size_n then only sizes the per-z-slice lock ranges and must be the max width);
        // resolved once per matrix, outside all inner loops

        int n_j = (size_n_list && mat_index >= 0) ? size_n_list[mat_index] : size_n;
        int size_m_ = size_m;
        half* A_ = A_had + j * size_m * size_k;
        void* C_;
        if (C_list && mat_index >= 0) C_ = C_list[mat_index];
        else if constexpr (c_fp32) C_ = (void*) (((float*) C) + j * size_m * size_n);
        else                       C_ = (void*) (((half*) C) + j * size_m * size_n);
        void* C_base = C_;

        // bits == 0 instances dispatch on the per-matrix K at runtime (same pattern
        // as exl3_moe_kernel's gemm lambdas); the host guarantees K_list is non-null for them
        [[maybe_unused]] int K_j = 0;
        if constexpr (!bits)
            if (B && K_list) K_j = K_list[mat_index];

        while (size_m_ > 0)
        {
            if (B)
            {
                int lock_offs = blockIdx.z * size_n / 128;

                #define MGEMM_INNER_ARGS \
                    (A_, B, C_, MIN(size_m_, 16), size_k, n_j, locks + lock_offs, nullptr)
                if constexpr (bits)
                    exl3_gemm_kernel_inner
                    <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false>
                    MGEMM_INNER_ARGS;
                else switch (K_j)
                {
#if EXL3_RUNTIME_K_MIN <= 1 && EXL3_RUNTIME_K_MAX >= 1
                    case 1: exl3_gemm_kernel_inner<1, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
#if EXL3_RUNTIME_K_MIN <= 2 && EXL3_RUNTIME_K_MAX >= 2
                    case 2: exl3_gemm_kernel_inner<2, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
#if EXL3_RUNTIME_K_MIN <= 3 && EXL3_RUNTIME_K_MAX >= 3
                    case 3: exl3_gemm_kernel_inner<3, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
#if EXL3_RUNTIME_K_MIN <= 4 && EXL3_RUNTIME_K_MAX >= 4
                    case 4: exl3_gemm_kernel_inner<4, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
#if EXL3_RUNTIME_K_MIN <= 5 && EXL3_RUNTIME_K_MAX >= 5
                    case 5: exl3_gemm_kernel_inner<5, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
#if EXL3_RUNTIME_K_MIN <= 6 && EXL3_RUNTIME_K_MAX >= 6
                    case 6: exl3_gemm_kernel_inner<6, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
#if EXL3_RUNTIME_K_MIN <= 7 && EXL3_RUNTIME_K_MAX >= 7
                    case 7: exl3_gemm_kernel_inner<7, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
#if EXL3_RUNTIME_K_MIN <= 8 && EXL3_RUNTIME_K_MAX >= 8
                    case 8: exl3_gemm_kernel_inner<8, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false> MGEMM_INNER_ARGS; break;
#endif
                    default: __trap();
                }
                #undef MGEMM_INNER_ARGS
            }

            A_ += 16 * size_k;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * n_j);
            else                  C_ = (void*) (((half*) C_) + 16 * n_j);
            size_m_ -= 16;

            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
                group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
            #else
                grid.sync();
            #endif
        }

        // Had and output scales

        if (B)
        {
            int total_warps = size_m * n_j / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* svh = svh_list[mat_index];
            float scale = 0.088388347648f;  // 1/sqrt(128)
            if (B_weights) scale *= __half2float(B_weights[j]);

            C_ = C_base;

            for(; this_warp < total_warps; this_warp += warps_grid)
            {
                if constexpr (c_fp32)
                    had_ff_r_128_inner<false, true>
                    (
                        ((const float*) C_) + this_warp * 128,
                        ((float*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % n_j,
                        scale
                    );
                else
                    had_hf_r_128_inner<false, true>
                    (
                        ((const half*) C_) + this_warp * 128,
                        ((half*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % n_j,
                        scale
                    );
            }
        }
    }

    if (B_weights)
        grid.sync();

    // Final reduction: each of the num_tokens groups of (bszm / num_tokens) contiguous slots is
    // summed into its own output row (row t for group t), instead of always collapsing into row
    // 0. num_tokens == 1 (the legacy single-token case) reduces to exactly the original
    // single-row behavior. Groups MUST be processed in increasing t order per column: row t is
    // only ever read by group floor(t / stride), which is <= t, so it has already been fully
    // read (and, if that group's index equals t, is only then correctly overwritten) by the time
    // group t's own write happens.
    if (B_weights && blockIdx.z == 0)
    {
        int total_warps = size_m * size_n / 32;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        int this_lane = threadIdx.x % 32;
        int stride = bszm / num_tokens;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            for (int t = 0; t < num_tokens; ++t)
            {
                int col = this_warp * 32 + this_lane;
                if constexpr (c_fp32)
                {
                    float* C___ = ((float*) C) + t * stride * size_m * size_n + col;
                    float sum = 0.0f;
                    for (int j = 0; j < stride; ++j)
                    {
                        sum += *C___;
                        C___ += size_m * size_n;
                    }
                    ((float*) C)[t * size_m * size_n + col] = sum;
                }
                else
                {
                    half* C___ = ((half*) C) + t * stride * size_m * size_n + col;
                    half sum = {};
                    for (int j = 0; j < stride; ++j)
                    {
                        sum = __hadd(sum, *C___);
                        C___ += size_m * size_n;
                    }
                    ((half*) C)[t * size_m * size_n + col] = sum;
                }
            }
        }
    }
}
