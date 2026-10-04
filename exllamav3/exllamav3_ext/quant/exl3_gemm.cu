#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../exact_rows.h"
#include "exl3_gemm_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "bits_k.cuh"
#include "exl3_devctx.cuh"
#include "exl3_gemv.cuh"
#include "exl3_gemv_int8.cuh"
#include "coop_autotune.cuh"
#include <set>
#include <vector>

int exl3_gemm_tilesize_k_g[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n_g[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim_g[] = {EXL3_GEMM_BLOCKDIM};

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

uint64_t roundup_pow2(uint64_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;
    return x + 1;
}

uint64_t gemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    bool half_k
)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) (half_k ? 1 : 0));
    mix((uint64_t) MIN(roundup_pow2(size_m), 16));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) K);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) max_num_sms);
    mix((uint64_t) cb);
    return h;
}

uint64_t mgemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    int bszm_in,
    int bszm_out,
    bool half_k
)
{
    uint64_t h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb, half_k);
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(bszm_in, 24));
    mix((uint64_t) MIN(bszm_out, 24));
    return h;
}

bool exl3_gemm_asks_int8(bool mul1)
{
    return mul1 && exl3_gemv_int8_enabled();
}

int exl3_gemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms,
    Graph* graph,
    bool one_row_route
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    // TORCH_CHECK_SHAPES(A, 0, C, 0, 1);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    // Get SU, optionally
    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = nullptr;
    if (suh_ptr)
    {
        // TORCH_CHECK_SHAPES(suh.value(), 0, A, 1, 1);
        A_had_ptr = (half*) OPTPTR(A_had);
        // TORCH_CHECK(A_had_ptr, "Must supply A_had with suh");
        // TORCH_CHECK_SHAPES_FULL(A_had.value(), A);
    }

    // Get SV, optionally
    const half* svh_ptr = (const half*) OPTPTR(svh);
    // if (svh_ptr)
        // TORCH_CHECK_SHAPES(svh.value(), 0, B, 1, 16);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);
    // Turing allows only 64 KB of dynamic shared memory per block, so the fixed SMEM_MAX
    // request would fail the launch there. Ask for what this device actually permits.
    int smem_max = DevCtx::instance().get_smem_request(device);

    // Dispatch. 16 * K uint16 per tile for integer K; half-integer bitrates (K + 0.5, mul1 only) carry 16 * K + 8
    const int tile_u16 = B.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1), "exl3_gemm: half-integer bitrates require the mul1 codebook");
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;

    // Select kernel
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    // one_row_route (exl3_gemm.cuh): every routing decision below is taken for one row. The kernel
    // arguments keep the real row count
    const int route_m = one_row_route ? 1 : size_m;
    if (one_row_route)
    {
        // The rows of the call are one m16 tile of the kernel and take its small-row reduction, as
        // the one row does (exl3_gemm_inner.cuh, threadblock_reduce)
        static_assert(EXACT_ROWS_MAX <= 8, "one_row_route: rows must stay in the kernel's small-row reduction");
        TORCH_CHECK(!graph && force_shape_idx <= 0 && force_num_sms <= 0,
                    "exl3_gemm: one_row_route takes the autotuned, ungraphed launch");
        TORCH_CHECK(size_m >= 2 && size_m <= EXACT_ROWS_MAX,
                    "exl3_gemm: one_row_route takes 2 to ", EXACT_ROWS_MAX, " rows, got ", size_m);
        TORCH_CHECK(suh_ptr && A_had_ptr && svh_ptr, "exl3_gemm: one_row_route needs suh, A_had and svh");
        TORCH_CHECK(size_k > 0 && size_n > 0 && A.numel() == (int64_t) size_m * size_k &&
                    C.numel() == (int64_t) size_m * size_n,
                    "exl3_gemm: one_row_route: A or C does not hold the rows");
        TORCH_CHECK(A.is_contiguous() && C.is_contiguous() && A_had.value().is_contiguous() &&
                    A_had.value().dtype() == at::kHalf && A_had.value().numel() >= A.numel(),
                    "exl3_gemm: one_row_route: A, C and A_had must be contiguous, A_had FP16 with room for A");
        TORCH_CHECK(suh.value().dtype() == at::kHalf && svh.value().dtype() == at::kHalf &&
                    suh.value().numel() >= size_k && svh.value().numel() >= size_n,
                    "exl3_gemm: one_row_route: suh or svh does not cover the weight");
        TORCH_CHECK(A.is_cuda() && A.device() == C.device() && A.device() == A_had.value().device() &&
                    A.device() == B.device() && A.device() == suh.value().device() &&
                    A.device() == svh.value().device(),
                    "exl3_gemm: one_row_route: the operands must be on one CUDA device");
        // The input transform works on 128-element blocks of the flattened rows, and the output
        // transform on 128-column blocks: a block must not straddle two rows
        if (size_k % 128 || size_n % 128) return EXACT_ROWS_NO_LAUNCH;
    }

    // Experimental fused int8-activation GEMV path (EXL3_INT8_GEMV=1) for mul1 tensors. Rows are
    // processed as successive GEMV launches, so this is only sensible for small m (the reconstruct
    // threshold keeps m <= 144 in practice). Not graph-capturable yet; graphed callers fall through
    // to the regular kernel.
    if (exl3_gemm_asks_int8(mul1))
    {
        // one_row_route: the one-row call is offered to the int8 path, which decodes the weights per
        // call and whose one-row launch is not the launch of several rows
        if (one_row_route) return EXACT_ROWS_NO_LAUNCH;
        if (exl3_gemv_int8(A, B, C, suh, A_had, svh, stream, graph))
            return 0;
    }

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
            graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
            graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
            graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    // QTIP-style GEMV path for small m (exl3_gemv_kernel.cuh). Same kernel arguments, so graph
    // recording is identical; falls through to the regular kernel when the heuristic declines
    if (force_shape_idx <= 0 && force_num_sms <= 0)
    {
        void* gemv_kernel = nullptr;
        if (exl3_gemv_try_launch
        (
            kernelArgs, route_m, size_k, size_n, K, half_k, cb, c_fp32,
            suh_ptr && A_had_ptr && svh_ptr,
            device, stream, &gemv_kernel, false, one_row_route
        ))
        {
            // one_row_route: the one-row call takes the GEMV kernel (decided only, nothing launched),
            // whose one-row instance is not its instance for several rows
            if (one_row_route) return EXACT_ROWS_NO_LAUNCH;
            add_graph_args(gemv_kernel);
            cuda_check(cudaPeekAtLastError());
            return 90;
        }
    }

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = gemm_autotune_hash(MAX(route_m, 2), size_k, size_n, K, c_fp32, device, cc, num_sms, cb, half_k);
        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, smem_max, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            TORCH_CHECK(!one_row_route || (tuned.tag >= 1 && tuned.tag <= EXL3_GEMM_NUM_SHAPES),
                        "exl3_gemm: one_row_route launched under a record with kernel tag ", tuned.tag);
            return tuned.tag;
        }
        // one_row_route: the process has no record of the one-row call yet. This call does not make
        // one (no disk cache lookup, no tuning with its rows, no store): the first one-row call does
        if (one_row_route) return EXACT_ROWS_NO_LAUNCH;
        std::vector<CoopAutotuneCandidate> candidates;
        for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
        {
            if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K, half_k)) continue;

            fp_exl3_gemm_kernel candidate_kernel = get_gemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb, half_k);
            if (!candidate_kernel) continue;

            int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
            int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
            int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
            int max_candidate_sms = MAX(MIN(max_slices, num_sms), 1);

            candidates.push_back
            ({
                (void*) candidate_kernel,
                exl3_gemm_blockdim_g[candidate_shape_idx],
                max_candidate_sms,
                1,
                max_candidate_sms,
                candidate_shape_idx
            });
        }
        TORCH_CHECK(!candidates.empty(), "exl3_gemm autotune: no compatible kernel shapes");

        tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, smem_max, stream, (size_t) size_k * size_n);
        if (graph)
        add_graph_args((void*) tuned.kernel);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, half_k
    );
    if (!kernel) return 0;

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
        kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }
    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        smem_max,
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
)
{
    return exl3_gemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        force_shape_idx,
        mcg,
        mul1,
        force_num_sms,
        nullptr
    );
}

/*
EXL3 batched/multi-matrix matmul.

This is not a conventional batched A @ B. B, suh and svh are CUDA int64
tensors containing device addresses (one address per quantized matrix), rather
than the matrix data themselves. Entry q of each table describes one linear:

    B[q]   -> EXL3 trellis, logically (k / 16, n / 16, 16 * K) uint16
    suh[q] -> packed input scales/flips, logically (k / 16) float16
    svh[q] -> packed output scales/flips, logically (n / 16) float16

A is contiguous float16 [a_batches, m, k], C is contiguous float16 or
float32 [c_batches, m, n], and A_had is float16 scratch with room for every
active matrix. The kernel applies the input Hadamard transform into A_had,
performs the selected EXL3 matmul, then applies the output transform.

The active matrix/output slot j selects q = indices[j] when indices is given,
or q = j otherwise. This supports the following modes:

- Multiple inputs and outputs: A[j] @ B[q] -> C[j].
- One input, multiple outputs: when a_batches == 1, A[0] is broadcast and
  transformed separately for each selected B[q], producing C[j]. This is used
  for e.g. fused gate/up projections and MoE expert fan-out.
- Indexed matrices: indices is a contiguous int64 [*, num_indices] tensor;
  the kernel reads its first num_indices entries as q values. Negative indices
  skip that slot.
- Weighted MoE reduction: weights is a float16 tensor parallel to indices.
  Each transformed result is multiplied by weights[j], then all active C[j]
  are summed into C[0]. C therefore also serves as per-expert scratch; only
  C[0] is the reduced result.
- Expert-range filtering: with min_index >= 0, selections outside
  [min_index, max_index) are removed and retained indices are rebased by
  min_index. This allows B/suh/svh to be local pointer tables for an expert
  shard. At num_tokens == 1 the retained indices (and their weights) are
  compacted; at num_tokens > 1 out-of-range slots are instead masked to -1 in
  place, preserving the per-token slot groups the final reduction depends on
  (and, with bszm_in > 1, the slot -> input-row correspondence).

Without weights, every active C[j] is a separate output. The active slot count
is max(a_batches, c_batches), capped to num_indices when indices is present.

Limitations: k must be divisible by 16 and n by 128. Range filtering supports
at most 128 slots (the kernel's index-compaction capacity).
*/

int exl3_mgemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    float K_,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src,
    bool one_row_route
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    // num_tokens > 1 with expert-range filtering (min_index >= 0) uses position-preserving
    // masking in the kernel (out-of-range slots marked -1 in place) instead of index
    // compaction, so the grouped reduction's fixed per-token slot runs stay intact

    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(suh, kLong);
    TORCH_CHECK_DTYPE(svh, kLong);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK_DIM(A, 3);
    TORCH_CHECK_DIM(B, 1);
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DIM(C, 3);

    TORCH_CHECK_SHAPES(A, 1, C, 1, 1);
    if (!had_src_list) TORCH_CHECK_SHAPES(B, 0, suh, 0, 1);   // sliced mode: suh is per source
    TORCH_CHECK_SHAPES(B, 0, svh, 0, 1);

    int bsz = A.size(1);
    int bszm_in = A.size(0);
    int bszm_out = C.size(0);

    // Per-matrix output widths/pointers (uniform-width callers pass neither): C then only
    // provides the dtype and the max width (locks/shape sizing); outputs go to c_ptrs
    const int* size_n_list_ptr = nullptr;
    void** c_list_ptr = nullptr;
    if (size_n_list)
    {
        TORCH_CHECK(c_ptrs, "exl3_mgemm: size_n_list requires c_ptrs");
        TORCH_CHECK_DTYPE(size_n_list.value(), kInt);
        TORCH_CHECK_DTYPE(c_ptrs.value(), kLong);
        TORCH_CHECK(num_tokens == 1 && min_index < 0 && !weights,
                    "exl3_mgemm: per-matrix widths incompatible with multi-token/filtering/weights");
        size_n_list_ptr = (const int*) size_n_list.value().data_ptr();
        c_list_ptr = (void**) c_ptrs.value().data_ptr();
        bszm_out = (int) c_ptrs.value().size(0);
    }
    int bszm = MAX(bszm_in, bszm_out);

    // Sliced mode: the entries are equal-width column slices of num_had_src source matrices,
    // scheduled as independent z-groups so unequal matrices (e.g. Q vs K/V) can't leave groups
    // idle. B/svh/c_ptrs entries are pre-offset to the slice's first column, size_n_list holds
    // the slice width, n_stride_list the source's full width (the row stride of B and C), and
    // had_src_list maps each slice to its source: suh and the A_had slabs are per source
    const int* n_stride_list_ptr = nullptr;
    const int* had_src_list_ptr = nullptr;
    if (had_src_list)
    {
        TORCH_CHECK(size_n_list && c_ptrs && n_stride_list,
                    "exl3_mgemm: sliced mode requires size_n_list, c_ptrs and n_stride_list");
        TORCH_CHECK(bszm_in == 1 && !indices && !weights && num_tokens == 1 && min_index < 0,
                    "exl3_mgemm: sliced mode is single-input, unfiltered and single-token");
        TORCH_CHECK_DTYPE(had_src_list.value(), kInt);
        TORCH_CHECK_DTYPE(n_stride_list.value(), kInt);
        TORCH_CHECK(had_src_list.value().size(0) >= bszm_out && n_stride_list.value().size(0) >= bszm_out,
                    "exl3_mgemm: had_src_list / n_stride_list must have one entry per slice");
        TORCH_CHECK(num_had_src > 0 && suh.size(0) >= num_had_src,
                    "exl3_mgemm: sliced mode needs one suh entry per source");
        n_stride_list_ptr = (const int*) n_stride_list.value().data_ptr();
        had_src_list_ptr = (const int*) had_src_list.value().data_ptr();
    }
    else
    {
        TORCH_CHECK(!n_stride_list, "exl3_mgemm: n_stride_list requires had_src_list");
    }

    // The kernel writes one hadamard-transformed input slab PER MATRIX (A_had + j * m * k), or
    // per source in sliced mode; an undersized scratch is silent OOB corruption (found the hard way)
    int64_t had_slabs = had_src_list ? num_had_src : bszm;
    TORCH_CHECK(A_had.numel() >= had_slabs * A.size(1) * A.size(2),
                "exl3_mgemm: A_had must hold bszm * m * k elements");

    const int64_t* indices_ptr = (const int64_t*) OPTPTR(indices);
    const half* weights_ptr = (const half*) OPTPTR(weights);

    if (indices)
    {
        TORCH_CHECK_DIM(indices.value(), 2);
        int num_indices = indices.value().size(1);
        TORCH_CHECK(num_indices <= bszm_in || num_indices <= bszm_out, "mgemm: too many indices for tensor batch");
        if (bszm_in > num_indices) bszm_in = num_indices;
        if (bszm_out > num_indices) bszm_out = num_indices;
    }

    if (weights)
    {
        TORCH_CHECK_DIM(weights.value(), 2);
    }

    int size_m = A.size(1);
    int size_k = A.size(2);
    int size_n = C.size(2);

    // one_row_route (exl3_gemm.cuh): the launch-autotune record is looked up for one row. The kernel
    // arguments keep the real row count
    const int route_m = one_row_route ? 1 : size_m;
    if (one_row_route)
    {
        // As in exl3_gemm_gr: one m16 tile per matrix, the kernel's small-row reduction
        static_assert(EXACT_ROWS_MAX <= 8, "one_row_route: rows must stay in the kernel's small-row reduction");
        TORCH_CHECK(!graph && force_shape_idx <= 0 && force_num_sms <= 0 && !weights && num_tokens == 1 &&
                    min_index < 0 && !size_n_list && !c_ptrs && !n_stride_list && !had_src_list,
                    "exl3_mgemm: one_row_route takes the plain grouped launch");
        TORCH_CHECK(size_m >= 2 && size_m <= EXACT_ROWS_MAX,
                    "exl3_mgemm: one_row_route takes 2 to ", EXACT_ROWS_MAX, " rows, got ", size_m);
        TORCH_CHECK(size_k > 0 && size_n > 0 && bszm_in > 0 && bszm_out > 0,
                    "exl3_mgemm: one_row_route: A or C is empty");
        TORCH_CHECK(A.is_contiguous() && C.is_contiguous() && A_had.is_contiguous() && A_had.dtype() == at::kHalf,
                    "exl3_mgemm: one_row_route: A, C and A_had must be contiguous, A_had FP16");
        TORCH_CHECK(A.is_cuda() && A.device() == C.device() && A.device() == A_had.device(),
                    "exl3_mgemm: one_row_route: A, C and A_had must be on one CUDA device");
        // The transforms work on 128-element blocks of each matrix's flattened rows: a block must
        // not straddle two rows
        if (size_k % 128 || size_n % 128) return EXACT_ROWS_NO_LAUNCH;
    }

    // Device properties
    int device;
    cudaGetDevice(&device);
    int total_sms = DevCtx::instance().get_num_sms(device);
    int smem_max = DevCtx::instance().get_smem_request(device);
    int num_sms = force_num_sms ? force_num_sms : total_sms;
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    const half* A_ptr = (const half*) A.data_ptr();
    const uintptr_t* B_ptr_ptr = (const uintptr_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    const half* A_had_ptr = (const half*) A_had.data_ptr();
    const uintptr_t* suh_ptr_ptr = (const uintptr_t*) suh.data_ptr();
    const uintptr_t* svh_ptr_ptr = (const uintptr_t*) svh.data_ptr();

    // Select kernel. K_ is the bitrate (integer or half-integer, see bits_k.cuh)
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    const bool half_k = bk.half;
    TORCH_CHECK(!half_k || mul1, "exl3_mgemm: half-integer bitrates require the mul1 codebook");

    int shape_idx;
    int block_dim;
    fp_exl3_mgemm_kernel kernel;
    int concurrency;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr_ptr,
        (void*)& indices_ptr,
        (void*)& weights_ptr,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index,
        (void*)& num_tokens,
        (void*)& size_n_list_ptr,
        (void*)& c_list_ptr,
        (void*)& n_stride_list_ptr,
        (void*)& had_src_list_ptr,
        (void*)& num_had_src
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_mgemm_A, 0);
            graph->record_param(kernel_ptr, GP_mgemm_C, 2);
            graph->record_param(kernel_ptr, GP_mgemm_indices, 10);
            graph->record_param(kernel_ptr, GP_mgemm_weights, 11);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = mgemm_autotune_hash
        (
            route_m, size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in, bszm_out, half_k
        );
        if (had_src_list) autotune_key ^= 0x9e3779b97f4a7c15ull;   // sliced launches tune separately

        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, smem_max, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            TORCH_CHECK(!one_row_route || (tuned.tag >= 1 && tuned.tag <= EXL3_GEMM_NUM_SHAPES),
                        "exl3_mgemm: one_row_route launched under a record with kernel tag ", tuned.tag);
            return tuned.tag;
        }
        // one_row_route: no record of the one-row call in the process yet, and this call does not
        // make one (as in exl3_gemm_gr)
        if (one_row_route) return EXACT_ROWS_NO_LAUNCH;
        if (!graph)
        {
            std::vector<CoopAutotuneCandidate> candidates;
            for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
            {
                if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K, half_k)) continue;

                fp_exl3_mgemm_kernel candidate_kernel = get_mgemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb, half_k);
                if (!candidate_kernel) continue;

                int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
                int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
                int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
                int max_candidate_sms = MAX(MIN(max_slices, total_sms), 1);

                candidates.push_back
                ({
                    (void*) candidate_kernel,
                    exl3_gemm_blockdim_g[candidate_shape_idx],
                    max_candidate_sms,
                    bszm,
                    total_sms,
                    candidate_shape_idx
                });
            }
            TORCH_CHECK(!candidates.empty(), "exl3_mgemm autotune: no compatible kernel shapes");

            tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, smem_max, stream, (size_t) size_k * size_n * bszm);
            add_graph_args((void*) tuned.kernel);

            // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, tuned.tag, tuned.block_dim, tuned.num_sms, tuned.concurrency);

            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
    }

    kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out, half_k
    );
    int tilesize_k = exl3_gemm_tilesize_k_g[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n_g[shape_idx];
    int tiles = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
    num_sms = tiles;
    if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    concurrency = MIN(total_sms / num_sms, bszm);

    // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, shape_idx, block_dim, num_sms, concurrency);

    // Launch bigger grid if possible
    dim3 block_grid(num_sms, 1, concurrency);

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
        kernel_attr_set[device].insert((void*) kernel);
    }

    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        block_grid,
        block_dim,
        kernelArgs,
        smem_max,
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    float K_,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src
)
{
    return exl3_mgemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        indices,
        weights,
        K_,
        force_shape_idx,
        mcg_mult,
        mul1_mult,
        min_index,
        max_index,
        force_num_sms,
        nullptr,
        num_tokens,
        size_n_list,
        c_ptrs,
        n_stride_list,
        had_src_list,
        num_had_src,
        false
    );
}

// EXL3_EXACT_ROWS (exact_rows.h): the grouped projection of 2 to EXACT_ROWS_MAX rows. Row r of the
// row-major A and C is the contiguous (G, 1, k) and (G, 1, n) block a one-row call passes, so every
// row is exl3_mgemm_gr with that call's argument list: size_m 1, G inputs and outputs, and with them
// the launch-autotune record of the one-row call. The rows share A_had, in stream order, as
// successive one-row calls do
int exl3_mgemm_rows
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& indices,
    float K_,
    bool mcg,
    bool mul1
)
{
    // Everything a row's launch reads through a raw pointer is checked here, ahead of the first row;
    // exl3_mgemm_gr checks the rest (the dtypes of A, C and the pointer tables) before it launches
    TORCH_CHECK_DIM(A, 3);
    TORCH_CHECK_DIM(C, 3);
    TORCH_CHECK_DIM(indices, 2);
    TORCH_CHECK_DIM(B, 1);
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DTYPE(A_had, kHalf);
    TORCH_CHECK_DTYPE(indices, kLong);
    TORCH_CHECK(A.is_contiguous() && C.is_contiguous() && A_had.is_contiguous() && indices.is_contiguous(),
                "exl3_mgemm_rows: A, C, A_had and indices must be contiguous");
    TORCH_CHECK(B.is_contiguous() && suh.is_contiguous() && svh.is_contiguous(),
                "exl3_mgemm_rows: the pointer tables must be contiguous");
    const int64_t rows = A.size(0);
    const int64_t groups = A.size(1);
    const int64_t size_k = A.size(2);
    const int64_t size_n = C.size(2);
    TORCH_CHECK(rows >= 2 && rows <= EXACT_ROWS_MAX,
                "exl3_mgemm_rows: a row-exact call takes 2 to ", EXACT_ROWS_MAX, " rows, got ", rows);
    TORCH_CHECK(C.size(0) == rows && C.size(1) == groups, "exl3_mgemm_rows: A and C must have the same rows and groups");
    TORCH_CHECK(groups > 0 && size_k > 0 && size_n > 0, "exl3_mgemm_rows: A or C is empty");
    TORCH_CHECK(indices.size(0) >= 1 && indices.size(1) == groups, "exl3_mgemm_rows: indices must hold one entry per group");
    TORCH_CHECK(B.size(0) >= groups && suh.size(0) >= groups && svh.size(0) >= groups,
                "exl3_mgemm_rows: the pointer tables must hold one entry per group");
    TORCH_CHECK(A_had.numel() >= groups * size_k, "exl3_mgemm_rows: A_had must hold G * k elements");
    TORCH_CHECK(A.is_cuda() && A.device() == C.device() && A.device() == A_had.device() && A.device() == indices.device() &&
                A.device() == B.device() && A.device() == suh.device() && A.device() == svh.device(),
                "exl3_mgemm_rows: A, C, A_had, indices and the pointer tables must be on one CUDA device");

    int tag = 0;
    for (int64_t r = 0; r < rows; ++r)
    {
        at::Tensor a = A.slice(0, r, r + 1).view({groups, 1, size_k});
        at::Tensor c = C.slice(0, r, r + 1).view({groups, 1, size_n});
        int row_tag = exl3_mgemm_gr(a, B, c, suh, A_had, svh, indices, c10::nullopt, K_, -1, mcg, mul1, -1, -1, 0, nullptr, 1);
        TORCH_CHECK(r == 0 || row_tag == tag,
                    "exl3_mgemm_rows: row ", r, " took kernel ", row_tag, ", the rows before it kernel ", tag);
        tag = row_tag;
    }
    return tag;
}
