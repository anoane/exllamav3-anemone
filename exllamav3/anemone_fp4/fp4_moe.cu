// fused FP4 prefill MoE path: trellis -> decode -> hw e2m1 snap -> OMMA.
//
// Five stream-ordered kernels replace the cooperative exl3_moe kernel for prefill:
//   1. k_stage   gather + input hadamard (suh) + per-32 ue8m0 A-scales + e2m1 conversion
//   2. k_gemm    gate & up OMMA GEMMs (B = live trellis decode + hw snap; MTILE-chunked)
//   3. k_act     output hadamard + activation + down input hadamard + A-scales + e2m1
//   4. k_gemm    down OMMA GEMM
//   5. k_out     down output hadamard (svh) + routing weight + scatter-add to fp32 output
//
// v2 over v1: (a) REAL per-32 activation scales (mxfp4 semantics: x -> e2m1(x*2^-e) with
// SFA byte 127+e, e = ceil(log2(amax/6))) — quality now limited only by the un-requantized
// weight snap; (b) MTILE reuse — work items are 64-row chunks (up to 4 m16 tiles), B decoded
// once per k-step per chunk and reused across its tiles (M15's lever). B stays at scale 1.0
// (the snap semantic; B scales are the requant/runtime-derivation question).
//
// Launch-context notes (cost a crash each in v1): had_hf_r_128_d_inner needs 128 floats/warp
// of dynamic smem; the had inners index their scale vectors by blockIdx.y (selectors go in z).
//
// Compile: second torch extension, -gencode arch=compute_120a,code=sm_120a (see fp4_hook.py).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/hadamard_inner.cuh"
#include "quant/exl3_moe_common.cuh"

#define MT_CAP 4   // moe: m16 tiles per chunk (64 rows); retest with v3b pipeline
#define MT_D 4     // dense: deeper m-chunks (64 rows), reg-pipelined

#define CVT_E2M1X2(dst16, h2src) \
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f16x2 t, %1; cvt.u16.u8 %0, t; }" \
        : "=h"(dst16) : "r"(h2src))

#define OMMA_ACC(dd, a0, a1, a2, a3, b0, b1, sfa) \
    asm volatile( \
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 " \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};" \
      : "+f"(dd[0]),"+f"(dd[1]),"+f"(dd[2]),"+f"(dd[3]) \
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sfa),"r"(0x7F7F7F7Fu))

__device__ __forceinline__ uint32_t h2_as_u32(half a, half b)
{
    half2 h = __halves2half2(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

// per-32 mxfp4 scale + normalize + convert: sh[128] halves -> 64 nibble bytes + 4 scale bytes.
// lane covers halves 4l..4l+3; group g = l/8 covers one 32-block (lanes 8g..8g+7, so the
// shfl_xor tree with offsets 1/2/4 stays inside the group).
__device__ __forceinline__ uint8_t quant_seg_128
(
    const half* sh, uint8_t* out /*64B*/, const int lane
)
{
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) v[i] = __half2float(sh[lane * 4 + i]);
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    #pragma unroll
    for (int off = 1; off < 8; off <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    int e = 0;
    amax = fminf(amax, 65504.0f);   // Inf guard: (int)ceilf(log2f(inf)) is UB; NaN already
                                    // falls through the > 0 comparison to e = 0
    if (amax > 0.0f) e = max(-32, min(32, (int)ceilf(log2f(amax * (1.0f / 6.0f)))));
    const float mult = exp2f((float)-e);
    uint32_t h2a = h2_as_u32(__float2half_rn(v[0] * mult), __float2half_rn(v[1] * mult));
    uint32_t h2b = h2_as_u32(__float2half_rn(v[2] * mult), __float2half_rn(v[3] * mult));
    uint16_t b0a, b0b;
    CVT_E2M1X2(b0a, h2a);
    CVT_E2M1X2(b0b, h2b);
    out[lane * 2] = (uint8_t)b0a;
    out[lane * 2 + 1] = (uint8_t)b0b;
    return (uint8_t)(127 + e);
}


// GPU-side table building (P12): one block; replaces the host expert_count.tolist() sync +
// pageable table uploads. Chunk size is MT_CAP*16 BY CONSTRUCTION (the v2b CHUNK=64 vs
// MT_CAP=2 mismatch silently dropped rows 32..63 of oversized chunks — fixed here).
__global__ void k_build_tables
(
    const int64_t* __restrict__ expert_count,   // (num_experts+1,)
    int* __restrict__ row_expert,               // (rows,)
    int* __restrict__ wt_expert,
    int* __restrict__ wt_row0,
    int* __restrict__ wt_mrows,
    int* __restrict__ n_work,                   // device scalar, zeroed here
    const int num_experts
)
{
    __shared__ int s_start[1025];
    const int t = threadIdx.x;
    if (t == 0)
    {
        int acc = 0;
        for (int e = 0; e < num_experts; e++) { s_start[e] = acc; acc += (int)expert_count[e]; }
        s_start[num_experts] = acc;
        *n_work = 0;
    }
    __syncthreads();
    for (int e = t; e < num_experts; e += blockDim.x)
    {
        const int c = (int)expert_count[e];
        if (c == 0) continue;
        const int start = s_start[e];
        const int nch = (c + (MT_CAP * 16) - 1) / (MT_CAP * 16);
        const int base = atomicAdd(n_work, nch);
        for (int i = 0; i < nch; i++)
        {
            wt_expert[base + i] = e;
            wt_row0[base + i] = start + i * (MT_CAP * 16);
            wt_mrows[base + i] = min(MT_CAP * 16, c - i * (MT_CAP * 16));
        }
        for (int r = start; r < start + c; r++) row_expert[r] = e;
    }
}

// ---------------------------------------------------------------- 1. stage: had(suh) + quant
// grid: x = ceil(rows*segs/8), z = 2 (0=gate A, 1=up A); block 256 (8 warps)
__global__ __launch_bounds__(256)
void k_stage
(
    const half* __restrict__ hidden_state,
    const int64_t* __restrict__ token_sorted,
    const int* __restrict__ row_expert,
    const half** __restrict__ gate_suh,
    const half** __restrict__ up_suh,
    uint8_t* __restrict__ a_gate,
    uint8_t* __restrict__ a_up,
    uint8_t* __restrict__ sfa_gate,
    uint8_t* __restrict__ sfa_up,
    const int sfa_stride,
    const int rows,
    const int hidden_dim,
    const bool gated
)
{
    __shared__ half sh[8][128];
    const int segs = hidden_dim / 128;
    const int wi = blockIdx.x * 8 + threadIdx.x / 32;
    if (wi >= rows * segs) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int row = wi / segs, seg = wi % segs;
    const bool is_gate = blockIdx.z == 0;   // z, not y: the had inner indexes scales by blockIdx.y
    if (is_gate && !gated) return;
    const int e = row_expert[row];
    const half* suh = (is_gate ? gate_suh : up_suh)[e] + 128 * seg;
    const half* in_ptr = hidden_state + (size_t)token_sorted[row] * hidden_dim + 128 * seg;
    had_hf_r_128_inner<true, false>(in_ptr, sh[warp], suh, 0.088388347648f);
    __syncwarp();
    uint8_t* out = (is_gate ? a_gate : a_up) + (size_t)row * (hidden_dim / 2) + 64 * seg;
    uint8_t sc = quant_seg_128(sh[warp], out, lane);
    if ((lane & 7) == 0)   // transposed sidecar: [k32-block][row], read coalesced in k_gemm
        (is_gate ? sfa_gate : sfa_up)[(size_t)(4 * seg + lane / 8) * sfa_stride + row] = sc;
}

// ------------------------------------------------------- 2/4. OMMA GEMM, MTILE-chunked
template <int K>
__device__ __forceinline__ void gemm_kloop
(
    const uint16_t* __restrict__ trellis,
    const uint8_t* __restrict__ a_rows,   // [rows][size_k/2] nibbles, k-major
    const uint8_t* __restrict__ a_sfa,    // [size_k/32][sfa_stride] ue8m0, transposed
    const int sfa_stride,
    int row0, int mrows,                  // chunk: up to MT_CAP*16 rows
    int tiles_n, int strip,
    const int size_k,
    float d[MT_CAP][2][4],
    uint8_t* sh_b_raw, uint32_t* sh_t_raw
)
{
    constexpr int ps_u16 = 256 * K / 16;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    uint8_t (*sh_b)[32] = (uint8_t (*)[32])sh_b_raw;      // [16][32] per warp
    uint32_t* sh_tw = sh_t_raw;                            // [4][ps_u16/2] per warp
    const int ksteps = size_k / 64;
    const int mt = (mrows + 15) / 16;
    // SFA lane map (07_OMMA_LAYOUT_SPEC / packSFA): live lanes (l&2)==0, row rr=(l>>2)+8*(l&1)
    const bool sfa_live = (lane & 2) == 0;
    const int sfa_rr = (lane >> 2) + 8 * (lane & 1);
    for (int ks = 0; ks < ksteps; ks++)
    {
        #pragma unroll
        for (int t = 0; t < 4; t++)
        {
            const uint32_t* gp = (const uint32_t*)(trellis + ((size_t)(ks * 4 + t) * tiles_n + strip) * ps_u16);
            if (lane < ps_u16 / 2) sh_tw[t * (ps_u16 / 2) + lane] = gp[lane];
        }
        __syncwarp();
        #pragma unroll
        for (int t = 0; t < 4; t++)
        {
            FragB frag[2];
            dq_dispatch<K, 2>(sh_tw + t * (ps_u16 / 2), lane * 8, frag[0], frag[1]);
            const int kb = t * 16;
            const int n0 = 2 * (lane / 8) + ((lane >> 2) & 1);
            #pragma unroll
            for (int f = 0; f < 2; f++)
                #pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    uint32_t h2 = *reinterpret_cast<uint32_t*>(&frag[f][i]);
                    uint16_t by;
                    CVT_E2M1X2(by, h2);
                    int kk = kb + (lane % 4) * 2 + i * 8;
                    sh_b[n0 + 8 * f][kk >> 1] = (uint8_t)by;
                }
        }
        __syncwarp();
        uint32_t bfr[2][2];
        #pragma unroll
        for (int f = 0; f < 2; f++)
        {
            int col = (lane >> 2) + 8 * f;    // fragment f covers strip cols f*8..f*8+7
            bfr[f][0] = *(const uint32_t*)&sh_b[col][4 * (lane & 3)];
            bfr[f][1] = *(const uint32_t*)&sh_b[col][16 + 4 * (lane & 3)];
        }
        // all m-tiles of the chunk reuse the decoded B fragments
        #pragma unroll
        for (int mtile = 0; mtile < MT_CAP; mtile++)
        {
            if (mtile >= mt) break;
            uint32_t a[4];
            #pragma unroll
            for (int r = 0; r < 4; r++)
            {
                int row = mtile * 16 + (lane >> 2) + 8 * (r & 1);
                if (row < mrows)
                    a[r] = *(const uint32_t*)(a_rows + (size_t)(row0 + row) * (size_k / 2)
                                              + ks * 32 + 16 * (r >> 1) + 4 * (lane & 3));
                else a[r] = 0;
            }
            uint32_t sfa = 0x7F7F7F7Fu;
            if (sfa_live)
            {
                int row = mtile * 16 + sfa_rr;
                if (row < mrows)
                {
                    uint32_t s0 = a_sfa[(size_t)(ks * 2) * sfa_stride + row0 + row];
                    uint32_t s1 = a_sfa[(size_t)(ks * 2 + 1) * sfa_stride + row0 + row];
                    sfa = s0 | (s1 << 8) | 0x7F7F0000u;
                }
            }
            #pragma unroll
            for (int f = 0; f < 2; f++)
                OMMA_ACC(d[mtile][f], a[0], a[1], a[2], a[3], bfr[f][0], bfr[f][1], sfa);
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(256)
void k_gemm
(
    const uint16_t** __restrict__ trellis_ptrs,
    const int* __restrict__ K_tab,
    const uint8_t* __restrict__ a_rows,
    const uint8_t* __restrict__ a_sfa,
    const int sfa_stride,
    half* __restrict__ c_rows,                            // [rows][size_n] fp16
    const int* __restrict__ wt_expert,
    const int* __restrict__ wt_row0,
    const int* __restrict__ wt_mrows,
    const int* __restrict__ n_work,
    const int size_k,
    const int size_n
)
{
    const int w = blockIdx.x;
    if (w >= *n_work) return;
    const int warp = threadIdx.x / 32;
    const int strip = blockIdx.y * 8 + warp;
    if (strip * 16 >= size_n) return;
    const int lane = threadIdx.x & 31;
    const int e = wt_expert[w];
    const int row0 = wt_row0[w], mrows = wt_mrows[w];
    if (mrows > MT_CAP * 16) __trap();          // table/accumulator contract guard
    const uint16_t* trellis = trellis_ptrs[e];
    const int K = K_tab[e];
    const int tiles_n = size_n / 16;
    __shared__ uint8_t sh_b_all[8][16][32];
    __shared__ uint32_t sh_t_all[8][4][32];               // sized for K=4 (ps_u16/2 = 32)
    uint8_t* sh_b_raw = &sh_b_all[warp][0][0];
    uint32_t* sh_t_raw = &sh_t_all[warp][0][0];
    float d[MT_CAP][2][4] = {};
    switch (K)
    {
        case 2: gemm_kloop<2>(trellis, a_rows, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t_raw); break;
        case 3: gemm_kloop<3>(trellis, a_rows, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t_raw); break;
        case 4: gemm_kloop<4>(trellis, a_rows, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t_raw); break;
        default: __trap();
    }
    const int mt = (mrows + 15) / 16;
    #pragma unroll
    for (int mtile = 0; mtile < MT_CAP; mtile++)
    {
        if (mtile >= mt) break;
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int m = mtile * 16 + (lane >> 2) + 8 * (i >> 1);
                int n = 2 * (lane & 3) + (i & 1) + 8 * f;
                if (m < mrows)
                    c_rows[(size_t)(row0 + m) * size_n + strip * 16 + n] = __float2half_rn(d[mtile][f][i]);
            }
    }
}


// ---------------------------------------------------------- v3: block-tiled schedule port
// b12x-inspired: block-uniform k-loop, A + SFA staged ONCE per k-step in smem and shared by
// all 8 warps (v2 loaded identical A fragments 8x from global per k-step). Decode stays
// per-warp per-strip. Requires whole-block uniform work (guaranteed: dims %128, whole-block
// early returns only).
__device__ __forceinline__ void cpa16(void* smem, const void* gmem)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cpa4(void* smem, const void* gmem)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
}
#define CPA_COMMIT asm volatile("cp.async.commit_group;\n")
#define CPA_WAIT1  asm volatile("cp.async.wait_group 1;\n")
#define CPA_WAIT0  asm volatile("cp.async.wait_group 0;\n")

// v3b: 2-stage cp.async pipeline — stage k-step ks+1 (A, SFA, trellis) while computing ks.
template <int K>
__device__ __forceinline__ void gemm_kloop_v3
(    const uint16_t* __restrict__ trellis,
    const uint8_t* __restrict__ a_rows,
    const uint8_t* __restrict__ a_sfa,
    const int sfa_stride,
    int row0, int mrows,
    int tiles_n, int strip,
    const int size_k,
    float d[MT_CAP][2][4],
    uint8_t* sh_b_raw, uint32_t (*sh_t2)[4][32],
    uint8_t (*sh_a)[MT_CAP * 16][32],
    uint8_t (*sh_sf)[2][MT_CAP * 16]
)
{
    constexpr int ps_u16 = 256 * K / 16;
    constexpr int tile_b16 = (ps_u16 * 2) / 16;        // 16B units per trellis tile (K2:4 K3:6 K4:8)
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int tid = threadIdx.x;
    uint8_t (*sh_b)[32] = (uint8_t (*)[32])sh_b_raw;
    const int ksteps = size_k / 64;
    const int mt = (mrows + 15) / 16;
    const bool sfa_live = (lane & 2) == 0;
    const int sfa_rr = (lane >> 2) + 8 * (lane & 1);

    auto stage = [&](int buf, int ks)
    {
        // A: mrows x 32B, 16B units, block-cooperative
        const int na = mrows * 2;
        for (int i = tid; i < na; i += 256)
        {
            int row = i >> 1, off = (i & 1) * 16;
            cpa16(&sh_a[buf][row][off],
                  a_rows + (size_t)(row0 + row) * (size_k / 2) + ks * 32 + off);
        }
        // SFA: 2 x mrows bytes, plain sync loads (tiny)
        for (int i = tid; i < 2 * mrows; i += 256)
        {
            int blk = i / mrows, row = i % mrows;
            sh_sf[buf][blk][row] = a_sfa[(size_t)(ks * 2 + blk) * sfa_stride + row0 + row];
        }
        // trellis: this warp's strip, 4 tiles
        #pragma unroll
        for (int t = 0; t < 4; t++)
        {
            const uint8_t* gp = (const uint8_t*)(trellis + ((size_t)(ks * 4 + t) * tiles_n + strip) * ps_u16);
            if (lane < tile_b16)
                cpa16((uint8_t*)&sh_t2[buf][t][0] + lane * 16, gp + lane * 16);
        }
    };

    stage(0, 0);
    CPA_COMMIT;
    int buf = 0;
    for (int ks = 0; ks < ksteps; ks++)
    {
        if (ks + 1 < ksteps) { stage(buf ^ 1, ks + 1); CPA_COMMIT; CPA_WAIT1; }
        else                 { CPA_WAIT0; }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 4; t++)
        {
            FragB frag[2];
            dq_dispatch<K, 2>(&sh_t2[buf][t][0], lane * 8, frag[0], frag[1]);
            const int kb = t * 16;
            const int n0 = 2 * (lane / 8) + ((lane >> 2) & 1);
            #pragma unroll
            for (int f = 0; f < 2; f++)
                #pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    uint32_t h2 = *reinterpret_cast<uint32_t*>(&frag[f][i]);
                    uint16_t by;
                    CVT_E2M1X2(by, h2);
                    int kk = kb + (lane % 4) * 2 + i * 8;
                    sh_b[n0 + 8 * f][kk >> 1] = (uint8_t)by;
                }
        }
        __syncwarp();
        uint32_t bfr[2][2];
        #pragma unroll
        for (int f = 0; f < 2; f++)
        {
            int col = (lane >> 2) + 8 * f;
            bfr[f][0] = *(const uint32_t*)&sh_b[col][4 * (lane & 3)];
            bfr[f][1] = *(const uint32_t*)&sh_b[col][16 + 4 * (lane & 3)];
        }
        #pragma unroll
        for (int mtile = 0; mtile < MT_CAP; mtile++)
        {
            if (mtile >= mt) break;
            uint32_t a[4];
            #pragma unroll
            for (int r = 0; r < 4; r++)
            {
                int row = mtile * 16 + (lane >> 2) + 8 * (r & 1);
                a[r] = (row < mrows) ? *(const uint32_t*)&sh_a[buf][row][16 * (r >> 1) + 4 * (lane & 3)] : 0u;
            }
            uint32_t sfa = 0x7F7F7F7Fu;
            if (sfa_live)
            {
                int row = mtile * 16 + sfa_rr;
                if (row < mrows)
                    sfa = (uint32_t)sh_sf[buf][0][row] | ((uint32_t)sh_sf[buf][1][row] << 8) | 0x7F7F0000u;
            }
            #pragma unroll
            for (int f = 0; f < 2; f++)
                OMMA_ACC(d[mtile][f], a[0], a[1], a[2], a[3], bfr[f][0], bfr[f][1], sfa);
        }
        __syncthreads();
        buf ^= 1;
    }
}


// coalesced epilogue (P20): stage the warp's [mrows x 16] half output tile in (dead) pipeline
// smem, then flush as 4B-aligned row runs — replaces 2-byte scattered global stores.
template <int MT>
__device__ __forceinline__ void epilogue_coalesced
(
    float d[MT][2][4], half* __restrict__ c_rows, uint8_t* wsm,   // >= MT*16*16*2 bytes
    int row0, int mrows, int strip, int size_n, int lane
)
{
    half (*sh_o)[16] = (half (*)[16])wsm;
    const int mt = (mrows + 15) / 16;
    #pragma unroll
    for (int mtile = 0; mtile < MT; mtile++)
    {
        if (mtile >= mt) break;
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int m = mtile * 16 + (lane >> 2) + 8 * (i >> 1);
                int n = 2 * (lane & 3) + (i & 1) + 8 * f;
                sh_o[m][n] = __float2half_rn(d[mtile][f][i]);
            }
    }
    __syncwarp();
    const int n4 = mrows * 8;                       // 4B units (2 cols each)
    for (int i = lane; i < n4; i += 32)
    {
        int row = i >> 3, seg = i & 7;
        *(uint32_t*)(c_rows + (size_t)(row0 + row) * size_n + strip * 16 + seg * 2) =
            *(uint32_t*)&sh_o[row][seg * 2];
    }
}

__global__ __launch_bounds__(256)
void k_gemm_v3
(
    const uint16_t** __restrict__ trellis_ptrs,
    const int* __restrict__ K_tab,
    const uint8_t* __restrict__ a_rows,
    const uint8_t* __restrict__ a_sfa,
    const int sfa_stride,
    half* __restrict__ c_rows,
    const int* __restrict__ wt_expert,
    const int* __restrict__ wt_row0,
    const int* __restrict__ wt_mrows,
    const int* __restrict__ n_work,
    const int size_k,
    const int size_n
)
{
    const int w = blockIdx.x;
    if (w >= *n_work) return;
    const int warp = threadIdx.x / 32;
    const int strip = blockIdx.y * 8 + warp;
    const int lane = threadIdx.x & 31;
    const int e = wt_expert[w];
    const int row0 = wt_row0[w], mrows = wt_mrows[w];
    if (mrows > MT_CAP * 16) __trap();
    const uint16_t* trellis = trellis_ptrs[e];
    const int K = K_tab[e];
    const int tiles_n = size_n / 16;
    __shared__ uint8_t sh_b_all[8][16][32];
    __shared__ __align__(16) uint32_t sh_t2_all[8][2][4][32];   // warp-major: [warp][buf]
    __shared__ __align__(16) uint8_t sh_a2[2][MT_CAP * 16][32];
    __shared__ uint8_t sh_sf2[2][2][MT_CAP * 16];
    __shared__ __align__(4) half sh_out[8][MT_CAP * 16][16];    // P20 epilogue staging
    uint8_t* sh_b_raw = &sh_b_all[warp][0][0];
    uint32_t (*sh_t2)[4][32] = &sh_t2_all[warp][0];
    float d[MT_CAP][2][4] = {};
    switch (K)
    {
        case 2: gemm_kloop_v3<2>(trellis, a_rows, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t2, sh_a2, sh_sf2); break;
        case 3: gemm_kloop_v3<3>(trellis, a_rows, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t2, sh_a2, sh_sf2); break;
        case 4: gemm_kloop_v3<4>(trellis, a_rows, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t2, sh_a2, sh_sf2); break;
        default: __trap();
    }
    epilogue_coalesced<MT_CAP>(d, c_rows, (uint8_t*)&sh_out[warp][0][0], row0, mrows, strip, size_n, lane);
}

// ---------------------------------------------- 3. act: had+act+had(suh_d) + quant
__global__ __launch_bounds__(256)
void k_act
(
    const half* __restrict__ interm_g,
    const half* __restrict__ interm_u,
    const int* __restrict__ row_expert,
    const half** __restrict__ gate_svh,
    const half** __restrict__ up_svh,
    const half** __restrict__ down_suh,
    uint8_t* __restrict__ a_down,
    uint8_t* __restrict__ sfa_down,
    const int sfa_stride,
    const int rows,
    const int intermediate_dim,
    const float act_limit,
    const int act_function
)
{
    __shared__ half sh[8][128];
    const int segs = intermediate_dim / 128;
    const int wi = blockIdx.x * 8 + threadIdx.x / 32;
    if (wi >= rows * segs) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int row = wi / segs, seg = wi % segs;
    const int e = row_expert[row];
    const size_t off = (size_t)row * intermediate_dim + 128 * seg;
    had_hf_r_128_guad_inner
    (
        interm_g + off,
        interm_u + off,
        sh[warp],
        gate_svh[e] + 128 * seg,
        up_svh[e] + 128 * seg,
        down_suh[e] + 128 * seg,
        0.088388347648f,
        act_limit,
        act_function
    );
    __syncwarp();
    uint8_t* out = a_down + (size_t)row * (intermediate_dim / 2) + 64 * seg;
    uint8_t sc = quant_seg_128(sh[warp], out, lane);
    if ((lane & 7) == 0)
        sfa_down[(size_t)(4 * seg + lane / 8) * sfa_stride + row] = sc;
}

// --------------------------------------------------- 5. out: had(svh_d) + weight + scatter
__global__ __launch_bounds__(256)
void k_out
(
    const half* __restrict__ d_out,
    float* __restrict__ output_state,
    const int64_t* __restrict__ token_sorted,
    const half* __restrict__ weight_sorted,
    const int* __restrict__ row_expert,
    const half** __restrict__ down_svh,
    const int rows,
    const int hidden_dim
)
{
    const int segs = hidden_dim / 128;
    const int wi = blockIdx.x * 8 + threadIdx.x / 32;
    if (wi >= rows * segs) return;
    const int row = wi / segs, seg = wi % segs;
    const int e = row_expert[row];
    float* out_ptr = output_state + (size_t)token_sorted[row] * hidden_dim + 128 * seg;
    had_hf_r_128_d_inner
    (
        d_out + (size_t)row * hidden_dim + 128 * seg,
        out_ptr,
        down_svh[e] + 128 * seg,
        0.088388347648f * __half2float(weight_sorted[row])
    );
}

// ------------------------------------------------------------------------------- host
void fp4_moe
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& expert_count,   // (E+1,) int64, device
    const at::Tensor& row_expert,     // scratch (rows,) int32
    const at::Tensor& wt_expert,      // scratch (n_work_max,) int32
    const at::Tensor& wt_row0,
    const at::Tensor& wt_mrows,
    const at::Tensor& n_work_dev,     // scratch (1,) int32
    const at::Tensor& gate_ptrs_trellis, const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,   const at::Tensor& up_ptrs_suh,   const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis, const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,
    const at::Tensor& K_gate, const at::Tensor& K_up, const at::Tensor& K_down,
    const at::Tensor& a_gate, const at::Tensor& a_up, const at::Tensor& a_down,
    const at::Tensor& sfa_gate, const at::Tensor& sfa_up, const at::Tensor& sfa_down,
    const at::Tensor& interm_g, const at::Tensor& interm_u,
    const at::Tensor& d_out,
    const double act_limit,
    const int64_t act_function
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int H = (int)hidden_state.size(1);
    const int I = (int)interm_g.size(1);
    const int rows = (int)token_sorted.size(0);
    const int sfa_stride = (int)sfa_gate.size(1);
    const int num_experts = (int)expert_count.size(0) - 1;
    const int n_work_max = (int)wt_expert.size(0);
    if (!rows) return;
    TORCH_CHECK(num_experts <= 1024, "fp4_moe: too many experts for table builder");
    const bool gated = act_function != MOE_ACT_RELU2_NOGATE;

    #define P(t, ty) (ty)t.data_ptr()
    static const bool dbg_sync = [](){ const char* e = getenv("FP4_DEBUG_SYNC"); return e && e[0] == '1'; }();
    #define CK(nm) do { if (dbg_sync) cudaStreamSynchronize(stream); \
        TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "fp4_moe: ", nm, ": ", cudaGetErrorString(cudaGetLastError())); } while (0)

    k_build_tables<<<1, 256, 0, stream>>>(P(expert_count, const int64_t*), P(row_expert, int*),
        P(wt_expert, int*), P(wt_row0, int*), P(wt_mrows, int*), P(n_work_dev, int*), num_experts);
    CK("k_build_tables");

    dim3 g1((rows * (H / 128) + 7) / 8, 1, 2);
    k_stage<<<g1, 256, 0, stream>>>(P(hidden_state, const half*), P(token_sorted, const int64_t*),
        P(row_expert, const int*), P(gate_ptrs_suh, const half**), P(up_ptrs_suh, const half**),
        P(a_gate, uint8_t*), P(a_up, uint8_t*), P(sfa_gate, uint8_t*), P(sfa_up, uint8_t*),
        sfa_stride, rows, H, gated);
    CK("k_stage");

    const char* v3e = getenv("FP4_V3");
    const bool use_v3 = !(v3e && v3e[0] == '0');   // default ON; FP4_V3=0 reverts
    auto kg = use_v3 ? k_gemm_v3 : k_gemm;
    dim3 g2(n_work_max, I / 128);
    if (gated)
    {
        kg<<<g2, 256, 0, stream>>>(P(gate_ptrs_trellis, const uint16_t**), P(K_gate, const int*),
            P(a_gate, const uint8_t*), P(sfa_gate, const uint8_t*), sfa_stride, P(interm_g, half*),
            P(wt_expert, const int*), P(wt_row0, const int*), P(wt_mrows, const int*), P(n_work_dev, const int*), H, I);
        CK("k_gemm_gate");
    }
    kg<<<g2, 256, 0, stream>>>(P(up_ptrs_trellis, const uint16_t**), P(K_up, const int*),
        P(a_up, const uint8_t*), P(sfa_up, const uint8_t*), sfa_stride, P(interm_u, half*),
        P(wt_expert, const int*), P(wt_row0, const int*), P(wt_mrows, const int*), P(n_work_dev, const int*), H, I);
    CK("k_gemm_up");

    dim3 g3((rows * (I / 128) + 7) / 8);
    k_act<<<g3, 256, 0, stream>>>(P(interm_g, const half*), P(interm_u, const half*),
        P(row_expert, const int*), P(gate_ptrs_svh, const half**), P(up_ptrs_svh, const half**),
        P(down_ptrs_suh, const half**), P(a_down, uint8_t*), P(sfa_down, uint8_t*), sfa_stride, rows, I,
        (float)act_limit, (int)act_function);
    CK("k_act");

    dim3 g4(n_work_max, H / 128);
    kg<<<g4, 256, 0, stream>>>(P(down_ptrs_trellis, const uint16_t**), P(K_down, const int*),
        P(a_down, const uint8_t*), P(sfa_down, const uint8_t*), sfa_stride, P(d_out, half*),
        P(wt_expert, const int*), P(wt_row0, const int*), P(wt_mrows, const int*), P(n_work_dev, const int*), I, H);
    CK("k_gemm_down");

    dim3 g5((rows * (H / 128) + 7) / 8);
    k_out<<<g5, 256, 8 * 128 * sizeof(float), stream>>>(P(d_out, const half*), P(output_state, float*),
        P(token_sorted, const int64_t*), P(weight_sorted, const half*),
        P(row_expert, const int*), P(down_ptrs_svh, const half**), rows, H);
    CK("k_out");
    #undef P
    #undef CK
}


// ============================================================ dense (non-MoE) FP4 path (P11)
// Replaces reconstruct_hgemm (per-chunk W dequant + cutlass HGEMM, ~36 bits/weight of traffic)
// with the fused trellis->snap->OMMA pipeline at 4 bits/weight. Same basis math as the moe
// path: y = had_svh( had_suh(x) @ dequant(trellis) ), scale 1/sqrt(128) on each had.

__global__ __launch_bounds__(256)
void k_dense_stage
(
    const half* __restrict__ x,
    const half* __restrict__ suh,
    uint8_t* __restrict__ a_x,
    uint8_t* __restrict__ sfa,
    const int sfa_stride,
    const int rows,
    const int in_dim
)
{
    __shared__ half sh[8][128];
    const int segs = in_dim / 128;
    const int wi = blockIdx.x * 8 + threadIdx.x / 32;
    if (wi >= rows * segs) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int row = wi / segs, seg = wi % segs;
    had_hf_r_128_inner<true, false>(x + (size_t)row * in_dim + 128 * seg, sh[warp],
                                    suh + 128 * seg, 0.088388347648f);
    __syncwarp();
    uint8_t sc = quant_seg_128(sh[warp], a_x + (size_t)row * (in_dim / 2) + 64 * seg, lane);
    if ((lane & 7) == 0)
        sfa[(size_t)(4 * seg + lane / 8) * sfa_stride + row] = sc;
}

__global__ __launch_bounds__(256)
void k_dense_gemm
(
    const uint16_t* __restrict__ trellis,
    const uint8_t* __restrict__ a_x,
    const uint8_t* __restrict__ a_sfa,
    const int sfa_stride,
    half* __restrict__ c_rows,
    const int rows,
    const int size_k,
    const int size_n,
    const int K
)
{
    const int warp = threadIdx.x / 32;
    const int strip = blockIdx.y * 8 + warp;
    if (strip * 16 >= size_n) return;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * (MT_CAP * 16);
    const int mrows = min(MT_CAP * 16, rows - row0);
    if (mrows <= 0) return;
    const int tiles_n = size_n / 16;
    __shared__ uint8_t sh_b_all[8][16][32];
    __shared__ uint32_t sh_t_all[8][4][32];
    uint8_t* sh_b_raw = &sh_b_all[warp][0][0];
    uint32_t* sh_t_raw = &sh_t_all[warp][0][0];
    float d[MT_CAP][2][4] = {};
    switch (K)
    {
        case 2: gemm_kloop<2>(trellis, a_x, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t_raw); break;
        case 3: gemm_kloop<3>(trellis, a_x, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t_raw); break;
        case 4: gemm_kloop<4>(trellis, a_x, a_sfa, sfa_stride, row0, mrows, tiles_n, strip, size_k, d, sh_b_raw, sh_t_raw); break;
        default: __trap();
    }
    const int mt = (mrows + 15) / 16;
    #pragma unroll
    for (int mtile = 0; mtile < MT_CAP; mtile++)
    {
        if (mtile >= mt) break;
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int m = mtile * 16 + (lane >> 2) + 8 * (i >> 1);
                int n = 2 * (lane & 3) + (i & 1) + 8 * f;
                if (m < mrows)
                    c_rows[(size_t)(row0 + m) * size_n + strip * 16 + n] = __float2half_rn(d[mtile][f][i]);
            }
    }
}


// dense W dequant-to-e2m1 cache: decode+snap each tile once per call; the GEMM then streams
// packed nibbles with zero in-loop decode. w4 layout: [n][size_k/2] bytes, k-pairs per byte.
template <int K>
__global__ __launch_bounds__(256)
void k_dense_dq
(
    const uint16_t* __restrict__ trellis,
    uint8_t* __restrict__ w4,
    const int tiles_k,
    const int tiles_n
)
{
    constexpr int ps_u16 = 256 * K / 16;
    const int tile = blockIdx.x * 8 + threadIdx.x / 32;
    if (tile >= tiles_k * tiles_n) return;
    const int lane = threadIdx.x & 31;
    const int tk = tile / tiles_n, tn = tile % tiles_n;
    const int size_k2 = tiles_k * 8;                      // bytes per n-column
    __shared__ uint32_t sh_t[8][ps_u16 / 2];
    const int warp = threadIdx.x / 32;
    const uint32_t* gp = (const uint32_t*)(trellis + (size_t)tile * ps_u16);
    if (lane < ps_u16 / 2) sh_t[warp][lane] = gp[lane];
    __syncwarp();
    FragB frag[2];
    dq_dispatch<K, 2>(sh_t[warp], lane * 8, frag[0], frag[1]);
    const int n0 = 2 * (lane / 8) + ((lane >> 2) & 1);
    #pragma unroll
    for (int f = 0; f < 2; f++)
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            uint32_t h2 = *reinterpret_cast<uint32_t*>(&frag[f][i]);
            uint16_t by;
            CVT_E2M1X2(by, h2);
            int kk = tk * 16 + (lane % 4) * 2 + i * 8;
            w4[(size_t)(tn * 16 + n0 + 8 * f) * size_k2 + (kk >> 1)] = (uint8_t)by;
        }
}

// pure-streaming dense OMMA GEMM over the e2m1 cache (no in-loop decode, no smem)
__global__ __launch_bounds__(256)
void k_dense_gemm2
(
    const uint8_t* __restrict__ w4,       // [size_n][size_k/2]
    const uint8_t* __restrict__ a_x,
    const uint8_t* __restrict__ a_sfa,
    const int sfa_stride,
    half* __restrict__ c_rows,
    const int rows,
    const int size_k,
    const int size_n
)
{
    const int warp = threadIdx.x / 32;
    const int strip = blockIdx.y * 8 + warp;
    if (strip * 16 >= size_n) return;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * (MT_D * 16);
    const int mrows = min(MT_D * 16, rows - row0);
    if (mrows <= 0) return;
    const int ksteps = size_k / 64;
    const int size_k2 = size_k / 2;
    const bool sfa_live = (lane & 2) == 0;
    const int sfa_rr = (lane >> 2) + 8 * (lane & 1);
    const int mt = (mrows + 15) / 16;
    float d[MT_D][2][4] = {};
    const uint8_t* wcol0 = w4 + (size_t)(strip * 16 + (lane >> 2)) * size_k2 + 4 * (lane & 3);
    const uint8_t* wcol1 = wcol0 + (size_t)8 * size_k2;
    // two-stage register pipeline: fetch ks+1 operands while the MMAs consume ks
    uint32_t b[2][4], a[2][MT_D][4], sf[2][MT_D];
    #define FETCH(buf, ks_) do { \
        b[buf][0] = *(const uint32_t*)(wcol0 + (ks_) * 32); \
        b[buf][1] = *(const uint32_t*)(wcol0 + (ks_) * 32 + 16); \
        b[buf][2] = *(const uint32_t*)(wcol1 + (ks_) * 32); \
        b[buf][3] = *(const uint32_t*)(wcol1 + (ks_) * 32 + 16); \
        _Pragma("unroll") \
        for (int mtile = 0; mtile < MT_D; mtile++) { \
            if (mtile >= mt) break; \
            _Pragma("unroll") \
            for (int r = 0; r < 4; r++) { \
                int row = mtile * 16 + (lane >> 2) + 8 * (r & 1); \
                a[buf][mtile][r] = (row < mrows) ? \
                    *(const uint32_t*)(a_x + (size_t)(row0 + row) * (size_k / 2) \
                                       + (ks_) * 32 + 16 * (r >> 1) + 4 * (lane & 3)) : 0u; \
            } \
            sf[buf][mtile] = 0x7F7F7F7Fu; \
            if (sfa_live) { \
                int row = mtile * 16 + sfa_rr; \
                if (row < mrows) { \
                    uint32_t s0 = a_sfa[(size_t)((ks_) * 2) * sfa_stride + row0 + row]; \
                    uint32_t s1 = a_sfa[(size_t)((ks_) * 2 + 1) * sfa_stride + row0 + row]; \
                    sf[buf][mtile] = s0 | (s1 << 8) | 0x7F7F0000u; \
                } \
            } \
        } \
    } while (0)
    FETCH(0, 0);
    for (int ks = 0; ks < ksteps; ks++)
    {
        const int cur = ks & 1, nxt = cur ^ 1;
        if (ks + 1 < ksteps) FETCH(nxt, ks + 1);
        #pragma unroll
        for (int mtile = 0; mtile < MT_D; mtile++)
        {
            if (mtile >= mt) break;
            OMMA_ACC(d[mtile][0], a[cur][mtile][0], a[cur][mtile][1], a[cur][mtile][2], a[cur][mtile][3], b[cur][0], b[cur][1], sf[cur][mtile]);
            OMMA_ACC(d[mtile][1], a[cur][mtile][0], a[cur][mtile][1], a[cur][mtile][2], a[cur][mtile][3], b[cur][2], b[cur][3], sf[cur][mtile]);
        }
    }
    #undef FETCH
    const int mt2 = (mrows + 15) / 16;
    #pragma unroll
    for (int mtile = 0; mtile < MT_D; mtile++)
    {
        if (mtile >= mt2) break;
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int m = mtile * 16 + (lane >> 2) + 8 * (i >> 1);
                int n = 2 * (lane & 3) + (i & 1) + 8 * f;
                if (m < mrows)
                    c_rows[(size_t)(row0 + m) * size_n + strip * 16 + n] = __float2half_rn(d[mtile][f][i]);
            }
    }
}


// dense v3: same 2-stage cp.async pipeline as gemm_kloop_v3, no decode (w4 cache).
__global__ __launch_bounds__(256)
void k_dense_gemm3
(
    const uint8_t* __restrict__ w4,       // [size_n][size_k/2]
    const uint8_t* __restrict__ a_x,
    const uint8_t* __restrict__ a_sfa,
    const int sfa_stride,
    half* __restrict__ c_rows,
    const int rows,
    const int size_k,
    const int size_n
)
{
    const int warp = threadIdx.x / 32;
    const int strip = blockIdx.y * 8 + warp;
    const int lane = threadIdx.x & 31;
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * (MT_D * 16);
    const int mrows = min(MT_D * 16, rows - row0);
    if (mrows <= 0) return;
    const int ksteps = size_k / 64;
    const int size_k2 = size_k / 2;
    const int mt = (mrows + 15) / 16;
    const bool sfa_live = (lane & 2) == 0;
    const int sfa_rr = (lane >> 2) + 8 * (lane & 1);
    __shared__ __align__(16) uint8_t sh_w[2][8][16][32];      // [buf][warp][col][k64/2]
    __shared__ __align__(16) uint8_t sh_a[2][MT_D * 16][32];
    __shared__ uint8_t sh_sf[2][2][MT_D * 16];
    __shared__ __align__(4) half sh_out[8][MT_D * 16][16];    // P20 epilogue staging

    auto stage = [&](int buf, int ks)
    {
        const int na = mrows * 2;
        for (int i = tid; i < na; i += 256)
        {
            int row = i >> 1, off = (i & 1) * 16;
            cpa16(&sh_a[buf][row][off],
                  a_x + (size_t)(row0 + row) * size_k2 + ks * 32 + off);
        }
        for (int i = tid; i < 2 * mrows; i += 256)
        {
            int blk = i / mrows, row = i % mrows;
            sh_sf[buf][blk][row] = a_sfa[(size_t)(ks * 2 + blk) * sfa_stride + row0 + row];
        }
        // this warp's 16 cols x 32B, 2 x 16B per col
        for (int i = lane; i < 32; i += 32)
        {
            int col = i >> 1, off = (i & 1) * 16;
            cpa16(&sh_w[buf][warp][col][off],
                  w4 + (size_t)(strip * 16 + col) * size_k2 + ks * 32 + off);
        }
    };

    float d[MT_D][2][4] = {};
    stage(0, 0);
    CPA_COMMIT;
    int buf = 0;
    for (int ks = 0; ks < ksteps; ks++)
    {
        if (ks + 1 < ksteps) { stage(buf ^ 1, ks + 1); CPA_COMMIT; CPA_WAIT1; }
        else                 { CPA_WAIT0; }
        __syncthreads();
        uint32_t bfr[2][2];
        #pragma unroll
        for (int f = 0; f < 2; f++)
        {
            int col = (lane >> 2) + 8 * f;
            bfr[f][0] = *(const uint32_t*)&sh_w[buf][warp][col][4 * (lane & 3)];
            bfr[f][1] = *(const uint32_t*)&sh_w[buf][warp][col][16 + 4 * (lane & 3)];
        }
        #pragma unroll
        for (int mtile = 0; mtile < MT_D; mtile++)
        {
            if (mtile >= mt) break;
            uint32_t a[4];
            #pragma unroll
            for (int r = 0; r < 4; r++)
            {
                int row = mtile * 16 + (lane >> 2) + 8 * (r & 1);
                a[r] = (row < mrows) ? *(const uint32_t*)&sh_a[buf][row][16 * (r >> 1) + 4 * (lane & 3)] : 0u;
            }
            uint32_t sfa = 0x7F7F7F7Fu;
            if (sfa_live)
            {
                int row = mtile * 16 + sfa_rr;
                if (row < mrows)
                    sfa = (uint32_t)sh_sf[buf][0][row] | ((uint32_t)sh_sf[buf][1][row] << 8) | 0x7F7F0000u;
            }
            OMMA_ACC(d[mtile][0], a[0], a[1], a[2], a[3], bfr[0][0], bfr[0][1], sfa);
            OMMA_ACC(d[mtile][1], a[0], a[1], a[2], a[3], bfr[1][0], bfr[1][1], sfa);
        }
        __syncthreads();
        buf ^= 1;
    }
    epilogue_coalesced<MT_D>(d, c_rows, (uint8_t*)&sh_out[warp][0][0], row0, mrows, strip, size_n, lane);
}

__global__ __launch_bounds__(256)
void k_dense_out
(
    const half* __restrict__ c_rows,
    half* __restrict__ y,
    const half* __restrict__ svh,
    const int rows,
    const int out_dim
)
{
    const int segs = out_dim / 128;
    const int wi = blockIdx.x * 8 + threadIdx.x / 32;
    if (wi >= rows * segs) return;
    const int row = wi / segs, seg = wi % segs;
    const size_t off = (size_t)row * out_dim + 128 * seg;
    had_hf_r_128_inner<false, true>(c_rows + off, y + off, svh + 128 * seg, 0.088388347648f);
}

void fp4_dense
(
    const at::Tensor& x,          // (rows, in) half, contiguous
    const at::Tensor& y,          // (rows, out) half
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor& a_x,        // (rows_a, in/2) uint8
    const at::Tensor& sfa,        // (in/32, rows_a) uint8 transposed
    const at::Tensor& c_tmp,      // (rows_a, out) half
    const at::Tensor& w4,         // (out, in/2) uint8 e2m1 cache
    const int64_t K,
    const int64_t skip_dq         // 1 = w4 already holds this tensor (persistent cache hit)
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int rows = (int)x.size(0);
    const int in_dim = (int)x.size(1);
    const int out_dim = (int)y.size(1);
    const int sfa_stride = (int)sfa.size(1);
    if (!rows) return;
    #define P(t, ty) (ty)t.data_ptr()
    dim3 g1((rows * (in_dim / 128) + 7) / 8);
    k_dense_stage<<<g1, 256, 0, stream>>>(P(x, const half*), P(suh, const half*),
        P(a_x, uint8_t*), P(sfa, uint8_t*), sfa_stride, rows, in_dim);
    const int tiles_k = in_dim / 16, tiles_n = out_dim / 16;
    dim3 gdq((tiles_k * tiles_n + 7) / 8);
    if (!skip_dq) switch ((int)K)
    {
        case 2: k_dense_dq<2><<<gdq, 256, 0, stream>>>(P(trellis, const uint16_t*), P(w4, uint8_t*), tiles_k, tiles_n); break;
        case 3: k_dense_dq<3><<<gdq, 256, 0, stream>>>(P(trellis, const uint16_t*), P(w4, uint8_t*), tiles_k, tiles_n); break;
        case 4: k_dense_dq<4><<<gdq, 256, 0, stream>>>(P(trellis, const uint16_t*), P(w4, uint8_t*), tiles_k, tiles_n); break;
        default: TORCH_CHECK(false, "K out of range");
    }
    dim3 g2((rows + MT_D * 16 - 1) / (MT_D * 16), out_dim / 128);
    const char* dv3e = getenv("FP4_V3");
    if (!(dv3e && dv3e[0] == '0'))
        k_dense_gemm3<<<g2, 256, 0, stream>>>(P(w4, const uint8_t*), P(a_x, const uint8_t*),
            P(sfa, const uint8_t*), sfa_stride, P(c_tmp, half*), rows, in_dim, out_dim);
    else
        k_dense_gemm2<<<g2, 256, 0, stream>>>(P(w4, const uint8_t*), P(a_x, const uint8_t*),
            P(sfa, const uint8_t*), sfa_stride, P(c_tmp, half*), rows, in_dim, out_dim);
    dim3 g3((rows * (out_dim / 128) + 7) / 8);
    k_dense_out<<<g3, 256, 0, stream>>>(P(c_tmp, const half*), P(y, half*), P(svh, const half*),
        rows, out_dim);
    #undef P
    TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "fp4_dense launch failed: ",
                cudaGetErrorString(cudaGetLastError()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fp4_moe", &fp4_moe); m.def("fp4_dense", &fp4_dense); }
