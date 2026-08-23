#define EXL3_RUNTIME_K_MIN 2
#define EXL3_RUNTIME_K_MAX 4
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../../ptx.cuh"
#include "../exl3_gemm_kernel.cuh"
#include "exl3_comp_unit_0.cuh"

EXL3_MGEMM_KERNEL_INSTANCES_CB_FP32(0, 0)
