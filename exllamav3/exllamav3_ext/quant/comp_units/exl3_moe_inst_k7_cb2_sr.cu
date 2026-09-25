#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k7_n128_cb2_sr() { return exl3_moe_kernel<7, 128, 2, MOE_TILESIZE_M, true, false, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k7_n256_cb2_sr() { return exl3_moe_kernel<7, 256, 2, MOE_TILESIZE_M, true, false, true>; }
