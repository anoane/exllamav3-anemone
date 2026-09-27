#pragma once

#include <cstdlib>
#include <cstring>

/*
__sinf/__cosf reduce their argument with a truncated fp32 multiply by 1/(2*pi), so their error grows
with |x|. With EXL3_ROPE_RANGE_REDUCE set, the angle is first reduced to about [-pi, pi] (Cody-Waite,
2*pi split over two floats). Shared by ext.rope and the fused DeepSeek-V4 compressor so both rotate
identically; see doc/env_vars.md.
*/

inline bool rope_range_reduce()
{
    static const bool enabled = [] { const char* e = getenv("EXL3_ROPE_RANGE_REDUCE"); return e && strcmp(e, "0") != 0; }();
    return enabled;
}

template <bool range_reduce>
__device__ __forceinline__ float rope_reduce_angle(float x)
{
    if constexpr (range_reduce)
    {
        float k = rintf(x * 0.15915494f);
        x = fmaf(k, -6.28318548f, x);
        x = fmaf(k, 1.74845553e-7f, x);
    }
    return x;
}
