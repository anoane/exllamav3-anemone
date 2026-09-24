#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k8_n128_cb1_st() { return exl3_moe_kernel<8, 128, 1, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k8_n256_cb1_st() { return exl3_moe_kernel<8, 256, 1, MOE_TILESIZE_M, true>; }
