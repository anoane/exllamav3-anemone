#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k4_n128_cb2_st() { return exl3_moe_kernel<4, 128, 2, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_cb2_st() { return exl3_moe_kernel<4, 256, 2, MOE_TILESIZE_M, true>; }
