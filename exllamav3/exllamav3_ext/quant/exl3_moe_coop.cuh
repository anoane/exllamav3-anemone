#pragma once

#include <ATen/Tensor.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused decode-shaped MoE path (bsz <= MAX_BSZN): two launches per layer run every (token, expert)
// slot through gate/up, activation and down, replacing the per-projection mgemm CUDA graphs of
// BC_BlockSparseMLP. See exl3_moe_coop_kernel.cuh / exl3_moe_coop.cu.

#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3
#define MOE_COOP_ACT_SILU_REF 5    // DeepSeek's reference swiglu (silu_ref.cuh)

#define MOE_COOP_THREADS 512
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp (32-column groups per block)
#define MOE_COOP_COLS (MOE_COOP_WNT * 16)

struct MoeCoopParams
{
    // Inputs
    const half* x;              // (bsz, H), row stride x_stride
    int x_stride;
    const int64_t* sel;         // (bsz, topk) global expert indices
    const half* rw;             // (bsz, topk) routing weights
    int bsz;
    int topk;
    int H;                      // x width (the experts' input width before padding)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: no expert-range filtering; else sel in [min, max) is local
    int max_expert;

    // Per-expert pointer tables (int64 device arrays indexed by local expert)
    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    const int64_t* g_bias;      // nullable
    const int64_t* u_bias;
    const int64_t* d_bias;
    int act;                    // MOE_COOP_ACT_*
    float act_limit;
    bool gated;                 // false: no gate projection, activation is relu(u) * u

    // Scratch (leading dim >= bsz * topk)
    half* had_g;                // (slots, Hi) rotated gate input (bsz > 1, written by the rot kernel)
    half* had_u;                // (slots, Hi) rotated up input
    bool a_global;              // true: the GEMV reads had_g / had_u; false (bsz 1): rotated in-block
    void* gu_g;                 // (slots, I) gate GEMV output, half or float (gu_f32)
    void* gu_u;                 // (slots, I)
    bool gu_f32;
    half* act_out;              // (slots, I) activation, rotated for the down projection
    float* d_out;               // (slots, Ho) down GEMV output before the output rotation
    int* ctr_a;                 // (slots_max, I / 128) completion counters of the gate/up stage
    int* ctr_b;                 // (MAX_BSZN, Ho / 128) completion counters of the down stage
    int ctr_a_len;
    int ctr_b_len;
    int ksplit_a;               // split-k blocks per chunk per stage (launcher; partial rows at slot + q * slots)
    int ksplit_b;
    int dbg;                    // diagnostics (EXL3_MOE_COOP_DBG): 1 skip A epilogue, 2 skip B reduction
    int* runs;                  // run table (bsz > 1): [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    int slots_max;
    int rows_max;               // token rows the output scratch holds
    int n_local;                // entries in the pointer tables
    int sh_gate_n;              // shared gate weight length (checked against H per call)

    // Output
    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;

    // Shared expert (nullable): out += gate * sh_out, gate = sigmoid(x . sh_gate_w) or 1
    const float* sh_out;        // (bsz, H), row stride H
    const half* sh_gate_w;      // (H,) or null
};

// Launch with plain device pointers (called from BC_BlockSparseMLP). K: gate/up and down bit
// widths (gate and up always share one, the converter allocates them as one group); cb: 0 default,
// 1 mcg, 2 mul1 codebook, uniform across the three projections.
// exact_rows (EXL3_EXACT_ROWS, exact_rows.h): the launch covers every row with the kernel instances
// of a one-row call: the tile and the split-k factor chosen from the topk slots of one row. With
// rows_grouped, where exl3_moe_coop_rows_grouped holds for the device, the slots are launched as an
// unflagged call launches them: the rotation launch, and the slots that picked one expert as the
// rows of one tile (a_global true).
// With the tile fixed, a slot's result does not depend on that:
// - the GEMV tiles and both epilogues are the code of the one-row call's own kernel instances. A
//   row of a run sits at a fragment row of its own (rows past the run and rows 8 to 15 are zero),
//   every accumulation, fold and cross-warp sum is per (row, column), and the epilogues are per
//   (slot, chunk) and per (token, chunk). That the FP16-accumulate tensor-core instruction gives a
//   row the same bits at fragment row r among other rows as at row 0 alone is a property of the
//   instruction, compared bit for bit on the GPU types of exact_rows_device_ok (exact_rows.h)
// - the input rotation is computed by the rotation kernel, another compilation of rotate_chunk
//   than the one inside kernel A. Its result does not depend on which products the build
//   contracts into FMAs in either (rotate_chunk, exl3_moe_coop_kernel.cuh): the only products are
//   of two FP16 values, exact in FP32, so a product contracted into the add or subtract that
//   follows rounds the same exact sum as the separate multiply and add
// - an empty token row (no active slot) is written by the rotation kernel instead of kernel A:
//   zeros, or a copy of the shared expert's row. With a shared-expert gate the two would each
//   compute the gate (write_empty_row_chunk: an FP32 dot product and an exponential, two
//   compilations again, and not exact): a caller must not ask for rows_grouped then
//   (BC_BlockSparseMLP::rows_grouped), and the launch below does not group
// Without rows_grouped, and on every other device, the slots stay ungrouped and rotated in-block
// (a_global false, no rot launch): a block then computes its slot exactly as the block of that slot
// does in the row's own one-row launch. rows_grouped without exact_rows changes nothing
void exl3_moe_coop_launch(const MoeCoopParams& p, float K_gu, float K_d, int cb, int device, cudaStream_t stream,
                          bool exact_rows = false, bool rows_grouped = false);

// Whether an exact_rows launch with rows_grouped on CUDA device `device` groups the rows (above),
// given parameters without a shared-expert gate: the GPU types on which rows sharing a tensor-core
// tile are compared bit for bit with one-row calls (exact_rows.h, exact_rows_device_ok). Counted
// per launch in exact_rows_served (EXACT_ROWS_CAP_MOE_GROUPED)
bool exl3_moe_coop_rows_grouped(int device);

// Whether an exact_rows launch of `rows` token rows of `topk` picks fits the scratch p was prepared
// with: the rows, their slots, and the split-k partials at the split factor of a one-row call
bool exl3_moe_coop_rows_ok(const MoeCoopParams& p, int topk, int rows);

// Static part of the parameter block, validated once from the module's tensors (BC construction);
// K_gu / K_d / cb come back through the out-params. Gate tables/scratch are ignored when
// gated == false but must still be valid tensors (pass the up ones)
MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    float& K_gu, float& K_d, int& cb
);

// Per-call part: input rows, routing, optional shared-expert output (bsz rows, width H), launch
// (exact_rows, rows_grouped: see exl3_moe_coop_launch)
void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out,
    bool exact_rows = false,
    bool rows_grouped = false
);

// Tensor front end (prepare + run; the test entry point)
void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w
);

// Length of the int32 scratch for a module: completion counters + run table (slots_max =
// MAX_BSZN * top_k)
inline int64_t exl3_moe_coop_ctr_len(int64_t slots_max, int64_t bsz_max, int64_t I, int64_t Ho)
{
    return slots_max * (I / 128) + bsz_max * (Ho / 128) + 2 + (slots_max + 1) + slots_max;
}
