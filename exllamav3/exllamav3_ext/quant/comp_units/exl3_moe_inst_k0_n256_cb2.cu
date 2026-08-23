#define EXL3_RUNTIME_K_MIN 2
#define EXL3_RUNTIME_K_MAX 4
#define EXL3_MOE_MIN_BLOCKS 1
#define EXL3_MOE_NOINLINE_ARMS 1
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k0_n256_cb2() { return exl3_moe_kernel<0, 256, 2>; }
