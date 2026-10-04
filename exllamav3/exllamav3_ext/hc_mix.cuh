#pragma once

#include <ATen/Tensor.h>

class Graph;

// Fused mHC HyperConnection mix / HyperHead collapse

int hc_mix_num_chunks(int R, int row_len);

void hc_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor collapsed
);

void hc_head
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    at::Tensor partials,
    at::Tensor collapsed
);

void hc_apply
(
    at::Tensor x,
    const at::Tensor& y,
    const at::Tensor& post,
    const c10::optional<at::Tensor>& comb
);

// EXL3_EXACT_ROWS (exact_rows.h): the reductions DeepSeek-V4.1's delayed mix makes in torch, for 2 to
// EXACT_ROWS_MAX rows of one sequence, every row with the operators and operand shapes of a one-row
// call (modules/dsv41_block.py)

// (pre.unsqueeze(-1) * streams).sum(dim = 2) per token: pre (1, s, H), streams (1, s, H, D), both
// float -> (1, s, D)
at::Tensor hc_collapse_rows(const at::Tensor& pre, const at::Tensor& streams);

// partials.sum(dim = 1) per row: partials (R, chunks, M + 1) float -> (R, M + 1)
at::Tensor hc_partials_rows(const at::Tensor& partials);

void gr_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& upt,
    const at::Tensor& w,
    double rms_eps,
    at::Tensor dots,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);

// Tiled deterministic GatedResidual mix for prefill row counts (hc_mix_tiled.cu)

int gr_mix_tiled_slices(int R, int D, int Mpad);

void gr_mix_tiled
(
    const at::Tensor& streams,
    const at::Tensor& w,
    const at::Tensor& proj_i8,
    const at::Tensor& proj_sb,
    const at::Tensor& up_i8,
    const at::Tensor& up_sb,
    double rms_eps,
    int M,
    at::Tensor dm_part,
    at::Tensor ss_part,
    at::Tensor rmr,
    at::Tensor t_i8,
    at::Tensor t_s,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);
