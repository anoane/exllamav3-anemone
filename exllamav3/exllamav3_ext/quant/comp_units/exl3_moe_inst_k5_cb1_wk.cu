#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k5_n128_cb1_wk() { return exl3_moe_kernel<5, 128, 1, MOE_TILESIZE_M, true, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k5_n256_cb1_wk() { return exl3_moe_kernel<5, 256, 1, MOE_TILESIZE_M, true, true>; }
