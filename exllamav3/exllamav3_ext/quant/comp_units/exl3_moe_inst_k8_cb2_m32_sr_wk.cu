#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k8_n128_cb2_m32_sr_wk() { return exl3_moe_kernel<8, 128, 2, 32, true, true, true>; }
