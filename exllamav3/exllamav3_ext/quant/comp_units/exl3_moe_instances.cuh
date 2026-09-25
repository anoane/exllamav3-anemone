#pragma once

#include "../exl3_moe_common.cuh"

typedef void (*fp_exl3_moe_kernel) (EXL3_MOE_KERNEL_ARGS);

// Each instance has a row-striped twin (suffix _st, files *_st.cu, exl3_moe_kernel's tile_rows) for fused-only
// prefill (EXL3_MOE_FUSED_PREFILL) and a whole-K twin (suffix _wk, files *_wk.cu, row-striped as well) that
// partitions the GEMMs by whole output columns, used under EXL3_STABLE_ARITHMETIC (see exl3_gemm_kernel_inner).
// DeepSeek's reference swiglu (MOE_ACT_SILU_REF) has twins of the two (suffixes _sr and _sr_wk, both row-striped).
// A build with EXLLAMA_NO_WHOLE_K_MOE set compiles no *_wk.cu unit (neither _wk nor _sr_wk)
#define EXL3_MOE_DECLARE_GETTERS(K) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1_st(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1_st(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_st(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2_st(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1_sr(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1_sr(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_sr(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2_sr(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1_sr_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1_sr_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_sr_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2_sr_wk(); \

EXL3_MOE_DECLARE_GETTERS(0);
EXL3_MOE_DECLARE_GETTERS(1);
EXL3_MOE_DECLARE_GETTERS(2);
EXL3_MOE_DECLARE_GETTERS(3);
EXL3_MOE_DECLARE_GETTERS(4);
EXL3_MOE_DECLARE_GETTERS(5);
EXL3_MOE_DECLARE_GETTERS(6);
EXL3_MOE_DECLARE_GETTERS(7);
EXL3_MOE_DECLARE_GETTERS(8);

#undef EXL3_MOE_DECLARE_GETTERS

// 32 / 64-row tile instances: mul1 codebook, N = 128 tile shape only (the 64-row reduction
// scratch does not fit the N = 256 tile in shared memory, and the N = 256 32-row tile gains
// nothing on Ada / Ampere). Used for any dims that are multiples of 128
#define EXL3_MOE_DECLARE_MTILE_GETTERS(K) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m32(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m64(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m32_st(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m64_st(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m32_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m64_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m32_sr(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m64_sr(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m32_sr_wk(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_m64_sr_wk(); \

EXL3_MOE_DECLARE_MTILE_GETTERS(0);
EXL3_MOE_DECLARE_MTILE_GETTERS(1);
EXL3_MOE_DECLARE_MTILE_GETTERS(2);
EXL3_MOE_DECLARE_MTILE_GETTERS(3);
EXL3_MOE_DECLARE_MTILE_GETTERS(4);
EXL3_MOE_DECLARE_MTILE_GETTERS(5);
EXL3_MOE_DECLARE_MTILE_GETTERS(6);
EXL3_MOE_DECLARE_MTILE_GETTERS(7);
EXL3_MOE_DECLARE_MTILE_GETTERS(8);

#undef EXL3_MOE_DECLARE_MTILE_GETTERS

extern fp_exl3_moe_kernel exl3_moe_kernel_instances[];
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m32[];   // [K], N = 128
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m64[];   // [K], N = 128
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_st[];      // row-striped twins of the three above
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m32_st[];
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m64_st[];
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_sr[];      // reference swiglu, row-striped
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m32_sr[];
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m64_sr[];
#ifndef EXLLAMA_NO_WHOLE_K_MOE
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_wk[];      // whole-K twins
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m32_wk[];
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m64_wk[];
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_sr_wk[];   // reference swiglu, whole-K
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m32_sr_wk[];
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_m64_sr_wk[];
#endif
