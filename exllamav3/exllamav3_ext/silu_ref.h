#pragma once

#include <cstdint>
#include <cstring>

// Host translation units build with -Ofast. std::isfinite may be folded to true
// under finite-math assumptions, so validate the actual IEEE exponent bits.
inline bool silu_ref_valid_limit(const float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffu;
    return magnitude < 0x7f800000u && ((bits & 0x80000000u) == 0 || magnitude == 0);
}

inline bool silu_ref_valid_cpu_limit(const double value)
{
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint64_t magnitude = bits & UINT64_C(0x7fffffffffffffff);
    // Positive IEEE encodings are monotonically ordered. This is FP32_MAX
    // encoded as double; keep every check integer-only under fast-math.
    return magnitude <= UINT64_C(0x47efffffe0000000) &&
           ((bits & UINT64_C(0x8000000000000000)) == 0 || magnitude == 0);
}
