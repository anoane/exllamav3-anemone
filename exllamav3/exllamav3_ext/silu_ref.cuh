#pragma once

#include <cuda_fp16.h>

// DeepSeek's reference SwiGLU (activation "silu_ref"): the raw gate is clamped from above and the
// raw up value symmetrically (a limit of 0 disables both), then silu(gate) * up is evaluated in
// FP32 and rounded once to FP16. The plain "silu" activation computes SiLU in the input precision
// and clamps the activated gate instead. Comparisons, not fminf/fmaxf, so a NaN stays NaN
__device__ __forceinline__ float silu_ref_f32(float g, float u, const float limit)
{
    if (limit > 0.0f)
    {
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        if (u < -limit) u = -limit;
    }
    const float sigmoid = __fdividef(1.0f, 1.0f + __expf(-g));
    const float activated = g * sigmoid;
    return activated * u;
}

__device__ __forceinline__ half2 silu_ref_h2(const half2 g, const half2 u, const float limit)
{
    const float2 gf = __half22float2(g);
    const float2 uf = __half22float2(u);
    return __floats2half2_rn(silu_ref_f32(gf.x, uf.x, limit), silu_ref_f32(gf.y, uf.y, limit));
}

// The FP16 rounding of the other paths, for kernels that keep the product in FP32
__device__ __forceinline__ float silu_ref_half_boundary(float g, float u, const float limit)
{
    return __half2float(__float2half_rn(silu_ref_f32(g, u, limit)));
}
