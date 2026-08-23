#pragma once

// runtime-K (bits = 0) instances, mgemm only (the plain gemm launcher infers K from
// the trellis tensor and never needs a runtime-K instance)
ALL_EXL3_MGEMM_KERNEL_EXTERNS(0)
