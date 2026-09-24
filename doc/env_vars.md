# Environment variables

Runtime and build-time toggles recognized by ExLlamaV3. All of these have sensible defaults;
they exist mainly for A/B testing, debugging and working around platform quirks.

Boolean-ish variables treat `0` as off and any other value as on unless noted. C++-side
variables are read once (on first use) and cached; Python-side variables are read at import
time. Either way, set them before loading a model.

## Attention

### `EXL3_BC_ATTN` (default: `1`)

Graph-captured C++ decode attention. For decode steps (bsz ≤ 8, q_len ≤ 16) the whole attention
block -- q/k/v projections, fused head norm + RoPE, cache append, flash-decoding attention and
o_proj -- runs as a single C++ call, captured as one CUDA graph per (bsz, q_len) shape and
replayed with only the input/output/position/block-table pointers patched. Removes effectively
all Python host time from the attention block; the largest gains are on host-bound setups
(small or hybrid models, fast GPUs, contended CPUs). The same flag covers the equivalent path
for MLA layers (BC_MLAttention: q projections, latent projection and staging, partial RoPE,
W_UK absorption, cache append, absorbed flash-decoding, W_UV unfold and o_proj as one graph).

Module or cache configurations the path does not support (TP, headwise gates, LayerNorm or
span-heads head norms, non-EXL3 projections, compander-enabled quant cache, ...) fall back to
the regular dispatch path by design. Unexpected errors while building the path are raised, not
swallowed. Set to `0` to disable the path entirely.

### `EXL3_BC_ATTN_TRACE` (default: `0`)

Print one line per attention module/cache-layer pair when the graph-captured decode path is
built or declined (module key, device). Activation check for A/B tests: a benchmark comparing
`EXL3_BC_ATTN` settings is only meaningful if the enabled run actually built the path.

### `EXL3_BC_DSA` (default: `1`)

DeepSeek-V4 counterpart of `EXL3_BC_ATTN`: for decode steps (bsz 1, q_len ≤ 16) the whole DSA
attention step runs as one graph-captured C++ call: the batched x-side projection fan, fused
head-norm RoPE, both compressor updates (compressed/indexer entry pools and rings), the
lightning-indexer scoring and capture-safe top-k selection (long-context regime), the
flash-decoding sparse attention with fused output de-rotation, the grouped o_proj and the
sliding-window ring append. One graph per (cache layer, job slot, q_len, dense/top-k regime);
position flows through shared device scalars (one 8-byte host write per step per job) plus a
handful of patched scalar node parameters (live pool width, causal bounds), so replays never
rebuild anything. Steps that need a host-side window ring shift or rebase (page-granular, rare)
decline to the eager path for that step, as do ineligible layer configurations (non-EXL3
projections, ...) permanently. Set to `0` to force the eager path everywhere.

### `EXL3_BC_MLA` (default: `1`)

MLA counterpart of `EXL3_BC_ATTN`: decode steps (q_len ≤ 16) of an MLA layer run as one
graph-captured C++ block (projections, absorb, latent attention, unfold, o_proj). Set to `0` to
force the eager dispatch path, for A/B testing.

### `EXL3_MOE_SHARED_COOP` (default: `1`)

MoE layers with a shared expert run it, at decode batch sizes, as a one-expert launch of the fused
expert kernels (at the shared expert's own bit width) instead of the three separate GEMV launches
of its own graph; the routed launch merges the result as before. Applies to EXL3-quantized gated
shared experts with 128-aligned widths and no post-norm. Under tensor parallelism such a shared
expert is placed whole on one rank (its contribution enters the all-reduce from that rank only)
rather than split across ranks. Set to `0` to keep the separate graph and the tensor split.

### `EXL3_GR_MIX_TILED` (default: `1`)

Prefill-sized mixes of the Qwen3.8-style gated residual (`GatedResidual`, the low-rank
hyper-connection) run a tiled CUDA kernel whose GEMMs use exact int8 tensor-core accumulation with
a fixed fp32 combination, so the result is bit-identical on every GPU architecture and
tensor-parallel ranks compute identical streams (a prerequisite for replicating decisions such as
MoE routing across ranks instead of broadcasting them). Precision matches the fp16 path and it is
faster than the cuBLAS path it replaces. Set to `0` to fall back to the cuBLAS GEMM path
(device-dependent kernel choice, not rank-consistent). Decode-sized mixes use the fused `gr_mix`
kernel either way. The same int8 scheme covers the MoE router projection for batched rows
(`routing_gemm.cu`), which has no switch.

### `EXL3_BC_GDN` (default: `1`)

Gated-delta-net (Qwen3-Next/3.5, KDA in GLM-5.3/Kimi Linear) counterpart of `EXL3_BC_ATTN`:
the decode step of a linear-attention layer runs as one graph-captured C++ call. Set to `0` to
force the torch path, for A/B testing.

### `EXL3_GDN_SUB_CHUNK` (default: `2048`)

Prefill of a gated-delta-net / KDA layer runs the fla chunked scan over consecutive sub-ranges
of this many tokens, carrying the fp32 recurrent state between them. The projections and the
convolution still run at the full chunk size; only the scan's temporaries (about 112 KB per
token at 48 value heads of 128) shrink from the chunk to the sub-range, e.g. 1.5 GB to 0.7 GB
per layer at a 16k chunk. The result is bit-identical to a single pass, and 2048 is also the
fastest setting measured (1024 costs about 40% more on the scan). Set to `0` to run the whole
chunk in one pass.

### `EXL3_GDN_PROJ_FP32` (default: `0`)

Prefill of a gated-delta-net / KDA layer takes its q/k/v projection out of the GEMM in fp16
instead of fp32; the values are converted to bf16 for the recurrence either way, and the
projections stay below |100| on natural text. Set to `1` for the fp32 projection (about
0.5-2.5e-3 relative difference in the layer's output, 1.5 GB less transient per 16k rows on
GLM-5.3).

### `EXL3_GDN_GATE_FP32` (default: `1`)

KDA's forget-gate and beta projections (the inputs to the per-channel decay) stay fp32. Set to
`0` for fp16: 256 MiB less per 16k rows on GLM-5.3, about 1e-3 relative difference in the
layer's output.

### `EXL3_GDN_CONV_TOKEN_MAJOR` (default: `1`)

The prefill short convolution reads the projection output in place (token-major, any float
dtype) instead of a transposed bf16 copy of it. Same kernel arithmetic; a bf16 projection is
bit-identical to the copy, an fp16 one skips its bf16 rounding. Set to `0` for the copy.

### `EXL3_PLE_SUB_CHUNK` (default: `1024`)

Prefill of the PLE (n-gram) layer runs its fused stream pass in row slabs of this many tokens,
carrying the short-conv state between slabs and adding each slab's result into the stream in
place. The pass holds several full-width fp32 working tensors of the stream stack so the slab 
bounds that. 1024 is the fastest slab measured. Set to `0` to run the chunk whole.

### `EXL3_BC_DSA_DEBUG` (default: `0`)

Raise errors encountered while building the graphed DSA path instead of silently declining to
the eager path. Diagnostic for why a configuration falls back.

### `EXL3_DSV4_NO_XFAN` (default: `0`)

Set to disable the eager DSA path's projection fans. With the fans, the x-side projections that
share the block input (q_a, wkv, and the compressor/indexer kv/gate pairs) run as a single
per-matrix-N batched MGEMM, and q_b pairs with the indexer query projection over q_res the same
way in the top-k regime: six to eight GEMV launches collapse into two. A/B switch for the
eager path only; the graphed path (`EXL3_BC_DSA`) builds its own fan and is not affected.
Layers whose projections mix quantization formats or bitrates decline the fan by themselves.

### `EXL3_QC_STAGING` (default: `1`)

How quantized K/V caches feed the attention kernels (replaces the former `EXL3_QC_ATTN`). Only
affects quantized caches.

- `0` - no staging: packed cache tensors feed the prefill and decode kernels directly, with
  dequantization fused into the kernel loads. Lowest memory: no staging scratch is ever
  allocated, and nothing extra is reserved during autosplit loading. Prefill pays for the
  in-kernel expansion (roughly 5–25% on the attention kernel depending on bitrate and GPU),
  which every kv tile repeats once per query block and sibling query head.
- `1` - prefill staging (default): prefill chunks of 256+ tokens dequantize the referenced
  cache window once into a shared fp16 scratch and run the fp16 kernel over it, putting
  quantized-cache prefill within ~1–3% of fp16. Decode stays on the direct path. The scratch is
  sized for the full cache at batch size 1 (`2 * max_num_tokens * num_kv_heads * head_dim`
  fp16 elements, shared across layers per device) and is allocated by the autosplit measuring
  pass, so the space is reserved at load time rather than discovered at the first long prefill.
  For very large caches this reservation is the tradeoff to weigh against `0` (e.g. ~4 GB at
  1M tokens with 8 kv heads of dim 128).
- `2` - full staging: legacy dequantize-then-attend path; whole cache layers are expanded into
  full-size fp16 temporaries before attention. Debug/A-B mode (same effect as the former
  `EXL3_QC_ATTN=0`); only affects decode if `EXL3_BC_ATTN` is also disabled, since the graphed
  decode path reads the packed cache directly.

### `EXL3_QC_PF_TWO_PASS_MIN_Q` (default: `256`)

Query-length threshold for the prefill staging pass at `EXL3_QC_STAGING=1`. Chunks shorter than
this keep the direct path, which reads less global memory (relevant for short trailing chunks
over long contexts at low cache bitrates). Tuning/testing knob.

### `EXL3_QC_PREFILL_NS` (default: `0` = measure)

Pipeline stage count for the direct quantized-cache prefill kernel. Unset/`0`, the best of
{1, 2} is measured once per (shape family, device) at first use; a nonzero value pins it,
skipping the measurement. Only relevant where the direct path still runs (`EXL3_QC_STAGING=0`,
or short chunks below the threshold above).

### `EXL3_TRITON_SMEM_LIMIT` (default: unset)

Caps the dynamic shared memory per block that the Triton attention kernels believe the device
grants. The kernels' tile configs are sized for Ampere-class budgets; each launch site lists a
ladder of smaller configs and takes the first whose compiled footprint fits the device (Turing:
64 KB). Setting a limit below the real one walks the ladders on any GPU, which is how the
stepped-down configs are exercised without the smaller hardware. Note the picks then reflect
this device's footprints, which differ from the target architecture's. Debugging only.

### `EXL3_TRITON_SMEM_DEBUG` (default: `0`)

Prints every ladder probe (kernel, config, measured footprint, fits or over) and every graph
kernel that declines to the eager path for lack of shared memory.

### `EXL3_MLA_PREFILL` (default: `mha`)

Prefill strategy for MLA layers: `mha` up-projects past latent tiles from the compressed cache
and attends in MHA form (~2.8× fewer FLOPs per query-past pair); `absorbed` restores the
single-kernel absorbed-form prefill for A/B testing.

### `EXL3_PREFER_FA2` (default: `0`)

Put the flash-attn-2 backends ahead of the built-in Triton attention kernels in the dispatch
order. flash-attn is an optional dependency; when it is not installed, this switch is ignored
(with a warning) and the built-in kernels serve everything. The Triton kernels match or beat
FA2 across supported hardware and cover more cases (quantized caches, head dims > 256,
attention sinks); this switch exists for A/B comparison.

## EXL3 GEMM / GEMV

### `EXL3_GEMV` (default: `1`)

QTIP-style small-m fp16 GEMV path, dispatched from the main GEMM entry point when the shape
heuristic applies. `0` disables, `1` uses the measured heuristic envelope (default), `2` forces
the path wherever its hard constraints allow (testing).

### `EXL3_GEMV_SMEM` (default: `-1`)

Weight-extraction strategy inside the fp16 GEMV kernel: `-1` picks per bitrate (default), `0`
forces shuffle extraction, `1` forces shared-memory staging. Testing only.

### `EXL3_INT8_GEMV` (default: `2`)

Fused int8-activation GEMV for tensors quantized with the mul1 codebook: one cooperative launch
covering the input Hadamard, activation quantization, dp4a GEMV and output Hadamard. `2`
(default) is the plain int8 mode, `1` the error-feedback residual mode (~15–16 bit effective
activation precision, slightly slower), `0` disables the path.

Tensors quantized with other codebooks are unaffected and keep their regular kernels. When the
mode is enabled, gate/up (and other same-input) tensor pairs that the int8 path can take are
also *unfused* from the batched MGEMM when each matrix is wide enough to fill the GPU on its
own. See the two thresholds below. The graphed decode paths (BC modules) handle both the fused
and unfused configurations.

### `EXL3_INT8_GEMV_MAX_K` (default: per-arch)

Highest bitrate K the int8 GEMV path accepts; above it the regular fp16 kernel runs instead.
The default is 6 on Hopper and Blackwell and 5 elsewhere: Ampere is DRAM-bound from K = 6 up,
where the int8 path's reduced per-weight compute no longer helps (and Ada is marginal there),
but on Hopper the fp16 kernel is throughput-bound at K = 6 as well. Values up to 8 can be forced
to test the crossover on unmeasured parts; the MGEMM unfusing threshold below follows this cap
automatically.

### `EXL3_MGEMM_K_THRESHOLD` (default: per-arch), `EXL3_MGEMM_N_THRESHOLD` (default: `8192`)

Unfusing heuristics applied when the int8 GEMV mode is enabled, to mul1 tensor pairs only: keep
the fused MGEMM when the bitrate K is at or above the K threshold (the int8 path declines those
anyway), or when the matrices are narrower than the N threshold (too narrow for separate GEMV
calls to fill the GPU; batching is what restores utilization there). The K threshold defaults
to one above the int8 path's per-arch K cap (see `EXL3_INT8_GEMV_MAX_K`); setting it explicitly
pins it on every device.

### `EXL3_NO_FUSED_RECONSTRUCT` (default: `0`)

Set to disable original-basis weight reconstruction on the hgemm prefill path. Long inputs to
EXL3 linears run reconstruct-then-GEMM; by default the reconstruct kernel emits the weights in
the original basis. Both 128-point Hadamard transforms and the sign vectors are folded into
the (memory-bound) reconstruct kernel's shared-memory epilogue, so the GEMM runs on the raw
input and the standalone input/output Hadamard launches disappear (previously ~14% of
long-chunk prefill GPU time; ~+6-7% prefill throughput on DeepSeek-V4-Flash). The fused kernel
does k·n-proportional extra work while the saved Hadamard traffic scales with rows·(k+n), so it
engages at 1024+ input rows (breakeven is ~400–900 depending on shape); below that, and for the
per-expert MoE dequant path (small row counts per expert), the rotated-basis pipeline
(input Hadamard → GEMM → output Hadamard) is kept. Set to `1` to force the rotated-basis
pipeline everywhere, for A/B testing.

### `EXLLAMAV3_TUNE_CACHE` (default: platform cache dir)

Override the path of the on-disk autotune cache for the cooperative GEMM kernels (kernel shape
selection results, persisted across runs).

## Sampling

### `EXL3_FUSED_SAMPLER` (default: `1`)

Collapse eligible sampler stacks into fused kernels at sampler construction. Stacks ending in
greedy or temperature/min-P/top-K/top-P/Gumbel steps (in the orders emitted by the preset
samplers, optionally preceded by repetition/presence/frequency penalties) run as a few custom
kernels working directly in logit space, instead of the step-by-step softmax/sort pipeline.
Collapsed temperature/min-P stacks sample the same token as the uncollapsed reference for the
same seed, up to float rounding at exact ties; top-K/top-P stacks keep the same token set as
the sort-based reference (ties at the exact cutoff are all kept) but draw their Gumbel noise by
token id rather than sorted position, so individual seeds map to different samples from the
same distribution. Stacks the collapse does not recognize fall back to the step-by-step path by
design. Set to `0` to disable collapsing entirely, e.g. for A/B validation against the
reference implementation.

## CPU MoE offload

Experimental: `-mcl`/`--moe_cpu_offload` (main model) and `-dmcl`/`--draft_moe_cpu_layers`
(draft model or MTP head) run the routed experts of the first N block-sparse MoE layers on the
CPU, expert weights resident in system RAM, freeing the VRAM those layers' experts would have
used. Layer-split mode only; requires mul1-codebook experts, K ≤ 8, and uniform per-expert
biases (all or none; ineligible layers fall back to the GPU as usual). A spawned worker process
per model component (main / draft / MTP) owns its own expert weights and a job ring in pinned
shared memory; the parent's forward pass never blocks on the CPU. During prefill, hot experts
additionally stream their weights to the GPU and run there (via the fused kernel or per-expert
dequant, by size) while the CPU works the remaining tail. See `-mclt`/`-dmclt` below for 
thread configuration, and the knobs below for tuning the split. `EXL3_MOE_CPU_MODE=stream_only`
keeps the experts in system RAM but computes all of them on the GPU instead (see below). To choose
the layers (not only the first N) and their mode per layer, use an explicit placement
(`EXL3_PLACEMENT`, under Model loading), which replaces `-mcl`, `-mcs` and `-mcm`.

These knobs are collected in `exllamav3/model/moe_cpu_host.py`'s `MoeCpuTuning` class (read once
from the environment at import); for a same-process sweep, mutate fields on the module-level
`TUNING` singleton before constructing a model instead of setting env vars.

### `EXL3_MOE_CPU_OFFLOAD` (default: `0`)

Fallback value for when `-mcl` is not set.

### `EXL3_MOE_CPU_MODE` (default: `compute`)

Default for `Config.infer_params.moe_cpu_mode`, which `-mcm` / `--moe_cpu_mode` sets in
`model_init`-based scripts: how the routed experts that `-mcl`, `-mcs` and `-dmcl` keep in
system RAM are computed. The weights stay in system RAM either way, owned and staged by the
worker process; only where the arithmetic runs changes.

- `compute`: the CPU worker computes the experts; during prefill the experts with at least
  `EXL3_MOE_STREAM_T` assignments in a chunk of at least `EXL3_MOE_STREAM_MIN_ROWS` rows are
  streamed to the layer's GPU and computed there (the behaviour described at the top of this
  section).
- `stream_only`: every expert a call routes at least one row to is streamed from system RAM to
  the layer's GPU and computed there, in every call, one-token decode included. No expert job is
  ever submitted to the worker for these layers, so there is no CPU expert arithmetic;
  `EXL3_MOE_STREAM_T` and `EXL3_MOE_STREAM_MIN_ROWS` do not apply, and the per-device bandwidth
  probe is skipped. In a `-mcs` split layer only the tail experts a call selects are streamed; a
  call that selects none of them adds nothing and transfers nothing.

Values: exactly `compute` or `stream_only`. Any other value of the variable raises a `ValueError`
when the `Config` is created, `-mcm` accepts only these two, and any other Python value raises
when the first offloaded layer loads. Refusing a value is never a silent fallback.

When it is read: every offloaded layer reads it once, when it registers with the worker during
`model.load()`, and keeps that mode until it is unloaded; set it before loading (changing the
attribute afterwards affects the next load). With an explicit placement (`EXL3_PLACEMENT`) each
layer's `experts=` decides instead (`stream` is this mode, `cpu` and `split` are `compute`), and
this setting must stay `compute`. An MTP head shares its model's config and so its setting.
`model_init` copies the setting to a draft model loaded from its own directory; a Python script
that builds the draft's `Config` itself sets the draft's attribute (or relies on the variable,
which every `Config` reads).

Refused at load under `stream_only`, before the worker starts, instead of computing on the CPU:

- an offloaded layer whose packed expert weights do not fit one weight staging slot
  (`EXL3_MOE_CPU_WSLOT_MB`, default 32 MiB, with `EXL3_MOE_CPU_WSLOTS` at least 1): a
  `RuntimeError` names both variables;
- a layer without the packed projection layout or its GPU-side scale tensors (every layer the
  worker accepts has them; this is a guard);
- a layer not on a CUDA device.

The worker's eligibility rules (mul1 codebook, K <= 8, uniform per-expert biases, supported
activation) apply unchanged: an ineligible layer keeps its experts in VRAM, as with `compute`.
At run time a CPU expert job for a `stream_only` layer, or a call on a GPU other than the one the
layer loaded on, raises a `RuntimeError`; neither happens on the supported paths.

Memory: no additional weights. The GPU side of the streamed path (per device: the VRAM weight
ring of `EXL3_MOE_CPU_WSLOTS x EXL3_MOE_CPU_WSLOT_MB`, 64 MiB by default, the reconstruct
scratch, the fused-tier and batched-reconstruct buffers, and per call the output and slot
scratch described under `EXL3_MOE_FUSED_DET`) is the one `compute` uses for its streamed
experts; with `stream_only` every chunk size uses it, and the autosplit's worst-case estimate
(`EXL3_AUTOSPLIT_WORSTCASE`) counts it also for chunks below `EXL3_MOE_STREAM_MIN_ROWS`.

Performance: every call moves the packed weights of every expert it selects over PCIe. One
expert of a gated layer is `3 x hidden x intermediate x bits / 8` bytes, e.g. 11.25 MiB at
hidden 5120, intermediate 2048 and 3 bits. A decoded token selects top-k experts per layer, so
at top-6 that is 67.5 MiB per offloaded layer per token, about 2.8 ms per layer over a link that
sustains 25 GB/s, before any compute: decode speed is bounded by the link, where `compute`
depends on the CPU. Besides the transfer, every call of a `stream_only` layer pays one host
synchronization (the per-expert counts, from which the host checks that no expert is left for the
CPU), where `compute` decode needs none; unless `EXL3_MOE_PINNED_ARENA=1`, the worker's stager first
copies every streamed expert into the pinned staging ring (the 67.5 MiB above, per layer and
token), so `EXL3_MOE_PINNED_ARENA=1` is recommended with `stream_only`; and a call whose experts
do not fit one staging slot is serialized through the `EXL3_MOE_CPU_WSLOTS` slots (at 11.25 MiB
per expert a 32 MiB slot holds two, so a top-6 token's experts pass through the two default
slots in three batches). Prefill chunks select most experts anyway and amortize the transfer
over their rows. Measured on DeepSeek-V4.1-Flash on two GPUs with the experts of 11 layers in
system RAM, 64K-token prefill: 796 tok/s with `stream_only` (and `EXL3_STABLE_ARITHMETIC=1`)
against 1040 tok/s with `compute` (default arithmetic). The two runs differ in more than
this setting, so the figure bounds its cost rather than isolating it; decode was not measured.

Determinism: `stream_only` removes the CPU's arithmetic, whose results depend on the CPU's
instruction set and thread count, and the split of each chunk between CPU and GPU, which depends
on the routing counts and the measured link bandwidth. With `EXL3_MOE_FUSED_DET=1` (default) the
streamed experts' contributions are then summed in routing order, and repeats of the same call
are bitwise identical for whole-layer offload (`-mcl`). For a `-mcs` split layer that holds only
with `EXL3_MOE_CPU_SWAP=0` or between swap sweeps: the layer's result is its GPU-resident
experts' partial sum plus the streamed experts' partial sum, added in FP32, and a sweep moves
experts between the two. Results still change with the chunk size, the staging-batch partition
and the GPU type, as described under `EXL3_MOE_FUSED_DET`.

When to use it: when the host CPU is slow relative to the PCIe link, or when results must not
depend on the CPU (reproducibility checks, comparisons across machines). With a fast CPU and a
narrow link, `compute` decodes faster. `EXL3_STABLE_ARITHMETIC=1` requires it for every
offloaded layer (refused at load otherwise).

```sh
python examples/chat.py -m /path/to/model -mcl 8 -mcm stream_only
EXL3_MOE_CPU_MODE=stream_only python eval/perf.py -m /path/to/model -mcs 64
```

```python
config = Config.from_directory(model_dir)
config.infer_params.moe_cpu_offload = 8
config.infer_params.moe_cpu_mode = "stream_only"
model = Model.from_config(config)
model.load()
```

The loader reports each such layer as `-- CPU-offloaded experts (streamed to cuda:1, no CPU
compute): <key>` (`-- CPU split experts (streamed to cuda:1, no CPU compute, dynamic): ...` for
`-mcs`).

### `-mclt` / `--moe_cpu_threads`, `-dmclt` / `--draft_moe_cpu_threads` (CLI, not env)

Worker thread count, set per component via `config.infer_params.moe_cpu_threads` /
`draft_moe_cpu_threads`. Takes precedence over `EXL3_MOE_CPU_THREADS` below when set.

### `EXL3_MOE_CPU_THREADS` (default: `cpu_count // 2`)

Fallback worker thread count when the component's `-mclt`/`-dmclt` config value is not set.

### `EXL3_MOE_CPU_SLOTS` (default: `4`), `EXL3_MOE_CPU_SLOT_ROWS` (default: `64`)

Compute job-ring depth and rows per slot (the CPU-tail chunk size). Each slot holds one
in-flight chunk of the D2H-staged input, selected experts and routing weights, and the
H2D-staged fp32 output.

### `EXL3_MOE_CPU_WSLOTS` (default: `2`), `EXL3_MOE_CPU_WSLOT_MB` (default: `32`)

Depth and per-slot size of the pinned/VRAM weight-staging ring used by GPU-streamed prefill.
Each slot must be large enough to hold a batch of streamed experts' packed weights (see
`EXL3_MOE_STREAM_BATCH_EXPERTS`); if not, the batch is capped by capacity instead.

### `EXL3_MOE_CPU_STAGE_THREADS` (default: `4`)

Memcpy threads used by the worker's dedicated stager (which packs streamed experts' weights
into the pinned staging ring, concurrently with the compute pool working the CPU tail). A few
threads saturate host memcpy bandwidth; raising this mainly helps wide streamed batches on
models with many small experts (see issue trace on Qwen3.6-35B-A3B).

### `EXL3_MOE_STREAM_T` (default: per-device, bandwidth-scaled from `16`)

Minimum per-expert token-assignment count (in a prefill chunk) for an expert's weights to be
streamed to the GPU instead of computed on the CPU tail. Unset, the effective threshold scales
inversely with the measured pinned→device bandwidth (probed once per device): a chipset-attached
x4 link needs a much hotter expert to justify the weight DMA than a CPU-direct x16 one. On
Windows the driver drops an idle link to Gen1, so the probe keeps traffic on it for at least
0.5 s and until the rate is steady. Setting this explicitly pins the threshold on every device
and disables the bandwidth scaling. Not used under `EXL3_MOE_CPU_MODE=stream_only`, which streams
every active expert.

### `EXL3_MOE_STREAM_FUSED_T` (default: `256`)

Maximum per-expert assignment count eligible for the fused `exl3_moe` GPU kernel (one launch
covers a whole batch of experts, up to three with the row tiles of `EXL3_MOE_MTILE`); above
this an expert still streams but runs through the batched reconstruct tier
(`EXL3_MOE_STREAM_BATCH_RECON`) or the per-expert reconstruct path instead. Same eligibility as
the GPU-resident fused path otherwise (mul1, silu/gelu gated or relu2 gateless, no per-expert
biases, no padded dims); ineligible layers use the reconstruct path for every streamed expert
regardless of count. The default was 512 while the alternative above it was the per-expert
loop; with the batched tier there, 128-256 measure best (Qwen3.8 4090 + 3090 split, 4k
chunks: 512 -> 256 +4%; mistral-small-4 119B full offload on the PRO 6000: +2.8%), and the
fused temp buffers (concurrency x T x (2 hidden + 2 intermediate) x 2 bytes per device) halve.
With `EXL3_MOE_FUSED_PREFILL=1` every streamed expert is fused and this is the height of the row
stripes instead of a tier limit (at least 64, or 16 with `EXL3_MOE_MTILE=0`).

### `EXL3_MOE_STREAM_MIN_ROWS` (default: `32`)

Prefill chunk size floor below which GPU streaming never engages and every expert runs on the
CPU tail as usual (decode, at 1 row per pass, always stays under this). Not used under
`EXL3_MOE_CPU_MODE=stream_only`, which streams at every row count.

### `EXL3_MOE_STREAM_BATCH_EXPERTS` (default: `24`, max `256`)

Experts packed per weight-staging batch (one stage job, one DMA, and, below
`EXL3_MOE_STREAM_FUSED_T, one fused-kernel launch). Further capped by staging-slot capacity
(`EXL3_MOE_CPU_WSLOT_MB` divided by one expert's packed byte size). The hard ceiling of 256 is
the structural size of the job descriptor's expert-id array; raising the ceiling itself costs
only a small amount of shared-memory overprovisioning, not runtime.

### `EXL3_MOE_CPU_MAX_ISA` (default: unset, auto-detect)

Caps the CPU kernel's runtime ISA detection at `scalar`, `avx2`, `bw`/`avx512bw`,
`vnni`/`avx512`, or `vbmi`, for testing a lower-tier kernel path on hardware that supports
better. The `bw` tier covers AVX-512F/BW/VL hardware without VNNI (Skylake-SP/X: 1st-gen Xeon
Scalable, Core-X), which previously fell through to `avx2`: the `vnni` dword kernel with the
AVX2 tier's vpmaddubsw/vpmaddwd accumulate, ~1.5x the `avx2` tier's cold-expert decode
throughput on a Xeon Gold 6148. The `vbmi` tier
(AVX512-VBMI byte-gather state extraction, Zen 4+ / Ice Lake+; Cascade/Cooper Lake have VNNI
without VBMI and stay on the `vnni` tier) is 15-70% faster than the dword scheme depending on
bitrate. Never upgrades past what the CPU actually supports; unrecognized values are ignored.
Read once per process (parent and worker independently), so it must be set before either is
started. Note that capping below `bw` also disables the swizzled weight layout (see
`EXL3_MOE_CPU_SWIZZLE`).

### `EXL3_MOE_CPU_SWIZZLE` (default: `1`)

Repack the CPU worker's expert trellis copies into a band-contiguous ("swizzled") layout at
load, so each GEMV band streams sequentially from DRAM instead of in short strided runs
(+45-75% cold decode GEMV throughput measured on a 7960X, reaching the sequential-read
roofline). Takes effect on every AVX-512 kernel tier: `vbmi`, whose byte-gather extraction
leaves the register headroom for the wide bands the swizzled layout wants at m > 1, `bw`
(+2-29% on Skylake-SP, where the sequential per-band k-stream beats 96-128 B strided reads)
and `vnni` (the dword kernel with the same band structure; +40% cold-expert decode measured
with the tier forced on a 7960X). The `avx2` and `scalar` tiers read the native layout. K8
tensors always stay in the native layout (they route to the dword kernel). The GPU-streaming
prefill path un-swizzles during staging, so staged bytes reaching the GPU dequant are
unaffected. Set to `0` to keep the native layout.

### `EXL3_MOE_MEMOPS` (default: `1`)

The parent enqueues its wait/publish handshake with the worker as CUDA stream memory operations
(`cuStreamWaitValue32`/`WriteValue32`, front-end executed: no SM occupancy, no per-op launch
cost) rather than the older spin-wait kernels. Set to `0` to force the kernel fallback, kept
around specifically because the memop path is not yet exercised on Windows. The kernel path's
30-second stall timeout does not apply to the memop path; a dead worker there is instead detected
by a host-side watchdog that unblocks any pending wait.

### `EXL3_MOE_STREAM_DEBUG` (default: `0`)

Print per-layer and per-batch engagement: streamed bandwidth probe result and threshold, expert
counts, streamed-vs-tail assignment split, and fused-vs-reconstruct tier split within each
streamed batch.

### `EXL3_MOE_CPU_PROF` (default: `0`)

Accumulate per-phase wall time in the CPU compute pool and report every 512 jobs. Enabled once
per worker at startup.

### `EXL3_MOE_ARENA_DEBUG` (default: `0`)

Print each hugepage-arena chunk allocation (size, running total) as the CPU worker loads expert
weights, and confirmation when the end-of-load `MADV_COLLAPSE` pass (see
`EXL3_MOE_ARENA_HUGEPAGE`) is issued. The worker copies loaded expert tensors into a small
number of large (1 GiB) anonymous mappings instead of leaving them as many separate small
(sub-2MB) allocations, confirmed via `/proc/<pid>/smaps` that the latter cannot be backed by
transparent huge pages even under system-wide THP=always, since each is its own VMA.

### `EXL3_MOE_ARENA_HUGEPAGE` (default: `1`)

Whether to attempt hugepage promotion for the arena chunks described above. This is done as a
single `MADV_COLLAPSE` (Linux 6.1+) pass over each chunk *after* all expert weights for every
offloaded layer have been loaded, deliberately not via a live `MADV_HUGEPAGE` hint during the
per-layer writes: on hosts where `/sys/kernel/mm/transparent_hugepage/defrag` is `madvise`, that
hint makes the kernel do *synchronous* compaction on first touch of a hinted region once
easily-compactable free memory runs low, which turns into multi-second stalls per offloaded
layer partway through a large model's load. The collapse pass runs on a background thread in
the worker after it has started serving: it copies the whole arena (about 4 GiB/s on a
7960X when the chunks were faulted as 4K pages, i.e. on `transparent_hugepage/enabled =
madvise` hosts, plus any compaction the kernel needs first), so it must not sit on the
startup path; the worker reads 4K pages until each chunk lands. `EXL3_MOE_ARENA_DEBUG=1`
prints how long it took. Set to `0` to skip hugepage promotion entirely.

### `EXL3_HGEMM_F16ACC` (default: auto)

GeForce parts run the fp32-accumulator tensor-core MMA at half the rate of the fp16-accumulator
form (measured 2.00x on the 3090, 4090 and 5090; 1.00x on the RTX PRO 6000). The reconstruct
(prefill) GEMMs, i.e. the dense `Linear` path above the reconstruct threshold, the per-expert
and batched MoE reconstruct tiers, run through a kernel (`hgemm_f16acc.cu`) that uses the
fp16-accumulator MMA and flushes the partial sums into fp32 every 32 elements of K, so the
accumulation across K stays fp32. `auto` runs a one-time per-device rate probe and enables the
kernel where the fp16-accumulator MMA is at least 1.5x faster; `1` forces it on, `0` forces
cuBLAS. Shapes the kernel does not cover (K not a multiple of 64, N not a multiple of 128,
unsupported strides/alignment) use cuBLAS regardless. Compute capability 12.x uses swizzled
128x128 or 128x64 tiles selected by shape, and a native mixed-precision add when folding
each 32-term FP16 partial into FP32.

### `EXL3_HGEMM_FP32_REDUCTION` (default: `1`)

Whether the cuBLAS GEMMs of `hgemm.cu` must keep their intermediate reductions in fp32. They
accumulate in fp32 (`CUBLAS_COMPUTE_32F`) for both fp16 and fp32 outputs, but with an fp16 output
cuBLAS may pick a split-K algorithm that reduces its partial sums in fp16, the output type: one
rounding per split, and partial sums beyond the fp16 range overflow even when the result is small.

Values:

- `1` (default): every call through `hgemm.cu` sets
  `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` on PyTorch's cuBLAS handle for the duration
  of the call, OR-ed into the handle's math mode, and restores the previous mode right after the
  launch, also when the GEMM fails, so the setting never leaks to other users of the handle.
  Tensor cores (and TF32 settings) stay enabled.
- `0`: the handle's math mode is left as it is, so cuBLAS may reduce through the output type, as
  it did before this setting existed. For comparisons only.
- Anything else, an empty value included, is refused at the first call that reads it, with a
  `RuntimeError`: `EXL3_HGEMM_FP32_REDUCTION must be 0 or 1`.

When it is read: once per process, by the first GEMM through `hgemm.cu` (or the first
`exllamav3_ext.hgemm_fp32_reduction()` call, which returns the value in use); changing the
variable afterwards has no effect, so an A/B comparison runs the two values in separate processes.

Which calls this reaches: every cuBLAS GEMM with an fp16 output that goes through `hgemm.cu`,
i.e. the cuBLAS path of `hgemm_recon` (the prefill of EXL3 linears above 144 rows, including the
fp16-output prefill projections of GatedDeltaNet / KDA) and of `hgemm_batched` (the batched MoE
reconstruct tier), the per-expert reconstruct path of streamed CPU-offload experts, the cuBLAS
fallback of the router scores, the fp16 gate projection of the native attention block, the
indexer projections of MLA / DSA and DeepSeek-V4 attention, the DeepSeek-V4 compressor,
convolutions lowered to a GEMM, and the native `BC_LinearFP16` while it runs inside a native CUDA
graph. Eager fp16 -> fp16 linears (Python `LinearFP16`, and the eager path of the native
`BC_LinearFP16`) do not reach it: they call `torch.matmul` / `at::matmul`, which follow PyTorch's
own `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` (default `True`), so the
same `BC_LinearFP16` layer may reduce differently eagerly and under a native graph. Calls with an
fp32 output should be unaffected by the flag's definition (a reduction may not use a type narrower
than the compute type, fp32 here); no fp32-output call was measured. The FP16-partial kernel of
`EXL3_HGEMM_F16ACC` (above) is not cuBLAS: this setting does not touch it, and it keeps its own
contract.

Whether a call's bits change depends on the algorithm cuBLAS picks for the GPU and the shape. On
the first attention projection of DeepSeek-V4.1-Flash (2048 x 5120 -> 1280, fp16), 2,927 of
10,240 sampled outputs (the first eight rows) disagreed with the FP64 product rounded once to
fp16 without this policy and 214 with it on an sm_80 GPU; on an sm_120 GPU nothing changed. Ten
launches of that projection at 2048 / 4096 rows took 0.174 / 0.311 ms without it and 0.161 /
0.310 ms with it on the sm_80 GPU, 0.082 / 0.140 ms and 0.076 / 0.141 ms on the sm_120 GPU.
Those "without" numbers come from an extension built without the policy, before this setting
existed; `0` leaves cuBLAS as that build did. Decode-shaped calls were not timed.

### `EXL3_HGEMM_FIXED_ROWS` (default: `0`)

Fixed row geometry for the native cuBLAS GEMMs, so that a row's result does not depend on how many
other rows share the call. cuBLAS chooses its kernel, and with it how the K reduction is split
and in what order the partial sums are added, from the whole problem: M, N, K, alignment and
strides. The same input row can therefore come out different in the last bits in a 4096-row call
than in a 2048-row or a one-row call, and `torch.matmul` behaves the same way. With `128`, every
call instead runs as a series of identical `[128, K] x [K, N]` GEMMs: the input rows are copied
into a packed 128-row scratch tile (the last tile zero-padded), cuBLAS multiplies the tile, and
the valid rows are copied to the output. For a given GPU, K and N, each output row then depends
only on its own input row: not on M, on where the row sits in the call, on the input's alignment
or the output's row stride, or on the other matrices of a batched call.

What it covers, every call that reaches the native GEMM (`exllamav3_ext/hgemm.cu`):

- `hgemm`: FP16 linears whose output dtype differs from the input (e.g. FP16 in, FP32 out), the
  cuBLAS fallback of the router scores, the FP16 gate projection of the native attention block,
  the indexer projections of MLA and DeepSeek-V4 attention, the DeepSeek-V4 compressor, the
  per-expert reconstruct path of streamed CPU-offload experts, and convolutions lowered to a
  GEMM.
- FP16 linears with FP16 input, weight and output: by default these take `torch.matmul` (Python
  `LinearFP16`) or `at::matmul` (the eager path of the native `BC_LinearFP16`); with this setting
  they take the fixed-row GEMM too. Other operand types (FP32, BF16) keep the matmul.
- `hgemm_recon`: the prefill path of EXL3 linears (`reconstruct_hgemm`, above 144 rows) and the
  per-expert MoE reconstruct path. The FP16-partial kernel of `EXL3_HGEMM_F16ACC` is bypassed
  whatever that variable says: the tiles always run on cuBLAS with FP32 accumulation, and
  `hgemm_f16acc_status()` reports `0` on every device.
- `hgemm_batched`: the batched MoE reconstruct tier, now one matrix at a time through the same
  tiles.

It does not change any other kernel: the EXL3 GEMV and small-batch GEMM kernels (the direct
quantized path at 144 rows or fewer), the fused MoE kernels, the int8 router projection,
attention, norms and the element-wise kernels keep their own dispatch, and several of them also
depend on the row count. On its own it therefore does not make a model's output independent of
the chunk size (on DeepSeek-V4.1-Flash, 4096- and 2048-token chunked prefills still differed with
it, p99 0.31 in actual-token log-probability); it removes one source of that dependence.

Native CUDA graphs: the per-call scratch tiles are ordinary PyTorch allocations, and the
engine's own graph wrapper (the internal CUDA graphs that the native attention, MLA, MLP,
block-sparse MLP, GatedDeltaNet / Mamba2 and DeepSeek-V4 attention blocks capture for decode)
records a raw stream capture without an allocator pool that would keep them alive for replay.
With this setting every native graph is therefore disabled and those blocks run eagerly on every
call; `Graph::capture_begin` refuses a disabled graph with a `RuntimeError` before any capture
starts, so a block without the eager fallback fails loudly instead of replaying freed memory.
Graphs captured from Python with `torch.cuda.graph` are unaffected (the scratch comes from their
pool). With the setting off, graph behaviour is unchanged.

Values: unset or `0`, the default dispatch; `128`, fixed rows. Anything else, including an
empty value, raises a `ValueError` when `exllamav3` is imported (`exllamav3.model.math_policy`),
and the native side refuses the same values with a `RuntimeError` on first use.
`EXL3_STABLE_ARITHMETIC=1` implies `128`; an explicit `0` with it is refused.

When it is read: once in Python, when `exllamav3` is imported, and once in the extension, on the
first GEMM or native block construction. Set it before importing `exllamav3`; changing it later
does not reach the Python side and can leave the two sides disagreeing. `ext.hgemm_fixed_rows()`
returns the native value.

Requirements and refusals: the input and weight of a fixed-row GEMM must be packed (contiguous);
a strided input raises `hgemm: fixed-row GEMM requires packed inputs` instead of being read
wrongly. Outputs may be strided views, and may have more rows than the product (reusable output
buffers): only the product's rows are written.

Memory: two scratch tiles per call, `128 x K` FP16 and `128 x N` in the output dtype, whatever M:
e.g. 1.25 MiB and 320 KiB (640 KiB for FP32 output) for K = 5120, N = 1280. They come from the
caching allocator and are released after the call. No other allocation.

Performance: a GEMM of M rows becomes `ceil(M / 128)` cuBLAS calls of a size that uses a large
GPU poorly, plus two copies per tile. Measured on the first attention projection of
DeepSeek-V4.1-Flash (4096 x 5120 x 1280, FP16), with the diagnostic form of this tiling: about
2.6x the default time on an sm_80 GPU and 3.1x on an sm_120 GPU. Decode additionally loses the
native CUDA graphs, i.e. pays the launch overhead of every kernel of every native block per
token; that cost was not measured separately and grows with the number of small kernels a model
launches. This is not a speed setting.

Determinism: what it guarantees is stated above (per GPU, K and N). Results differ from the
default dispatch, between GPU types, and from exact arithmetic (FP32 accumulation, one rounding to
the output dtype). Within the covered GEMMs, `tests/test_hgemm_fixed_rows_.py` checks bitwise that
1 to 4096-row calls, strided outputs with spare rows, and batched calls reproduce the rows of a
4096-row call, on each visible GPU.

When to use it: to take GEMM shape effects out of a comparison (e.g. when bisecting why two chunk
sizes disagree), or as part of the row-count-independent profile `EXL3_STABLE_ARITHMETIC`, which
sets it. Keep it unset for normal inference.

```sh
EXL3_HGEMM_FIXED_ROWS=128 python eval/ppl.py -m /path/to/model
```

```python
import os
os.environ["EXL3_HGEMM_FIXED_ROWS"] = "128"     # before importing exllamav3
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
assert ext.hgemm_fixed_rows() == 128
```

### `EXL3_MOE_COOP_KSPLIT` (default: unset)

Split-k factor of the fused decode MoE kernels: `n` runs every column chunk as `n` blocks over
disjoint k ranges whose partial sums the last-arriving block adds. Measured as neutral to harmful
on every GPU here, so the default is no split. Testing knob only.

### `EXL3_MOE_COOP_WIDE` (default: unset)

Tile geometry of the fused decode MoE kernels (the bsz <= 8 path of `BlockSparseMLP`): `0` forces
the narrow tile (32 columns per block, k split 16 ways), `1` the wide one (128 columns per block,
4 x 4 warps). Unset picks per stage: wide on Ampere/Ada at every shape, on Blackwell only for
k >= 4096 or k >= 2048 with 32 or more (token, expert) slots. Testing knob only.

### `EXL3_MOE_FUSED_DET` (default: `1`), `EXL3_MOE_RECON_DET` (default: follows `EXL3_MOE_FUSED_DET`)

Bit-reproducible MoE prefill. Without it the fused MoE kernel adds each expert's weighted
output into the token row with float atomics, in whatever order the expert groups finish, and
the batched reconstruct tier accumulates its padded slab with one atomic `index_add_`;
together these are the only sources of run-to-run nondeterminism on the GPU prefill path
(Qwen3.8-Flash-Next KL ~2e-2 between identical 4k-token runs, lfm2.5 ~1e-3, Qwen3-30B-A3B
~3e-4). With `EXL3_MOE_FUSED_DET=1` every assignment of the fused and batched tiers gets a
slot in one per-call fp32 scratch (`assignments x hidden`, ~320 MB per layer call on
Qwen3.8 at 4k tokens): the fused kernel stores its weighted outputs there, the batched
tier's down-projection GEMMs write straight into their slots, and one `exl3_moe_gather` per
layer sums each token's slots in k order with the routing weights. Identical runs are then
bit-identical (verified on all three models), and since the GEMM outputs are written once
and read once either way it costs nothing measurable: Qwen3.8 6198/6219 vs 6164/6178 tok/s,
lfm2.5 25.9k vs 26.2k, Qwen3-30B-A3B 7304 vs ~7100 (atomic).

The same switch covers CPU-offloaded layers (`-mcl`, `-mcs`) whose experts are streamed to the
GPU for a prefill chunk (see `EXL3_MOE_STREAM_T` and `EXL3_MOE_STREAM_MIN_ROWS`). Every
streamed assignment gets its own fp32 slot in one call-local scratch, whichever tier computes
it: the fused kernel and the per-expert path store weighted outputs (the per-expert path keeps
its fp32 routing weights), the batched reconstruct tier stores unweighted outputs that the
gather weights, as on the resident path. One `exl3_moe_gather` after the last staging batch
sums each token's slots in routing order, so neither the order in which experts finish nor
how they are packed into staging batches (`EXL3_MOE_STREAM_BATCH_EXPERTS`,
`EXL3_MOE_CPU_WSLOT_MB`) changes the order of the accumulation. Reconstruct groups keep their
padded slabs; padding rows are never gathered, and when the packed down-projection width
differs from the hidden size the group writes a temporary slab that is cropped into the slots.
The native gather table holds 32 routes, so top-k above 32 is reduced in fixed groups of 32
routes (an order that does not depend on the batching either). Memory per layer call:
`slots x hidden x 4` bytes of scratch, where `slots` is the number of streamed assignments plus
the reconstruct padding (bounded by `EXL3_MOE_RECON_PAD` and the group size), plus 18 bytes per
routed assignment for the index tables (26 with top-k above 32) and 24 bytes per expert. The
autosplit worst case of every offloaded layer includes all of it and the cropping slab, so an
`-mcl` / `-mcs` load may place its layers differently than without the slots, and a manual
split (`-gs`) that fit before may need the extra room: on a device with no resident MoE layer
(e.g. every MoE layer offloaded) the prefill peak grows by the scratch, up to about 295 MB per
streamed layer call for a hidden size of 2048 with top-8 routing at 4096-row chunks.

Measured on DeepSeek-V4.1-Flash on two GPUs, with the experts of 11 layers (12-22) held in RAM
and every expert those layers select streamed to the second GPU in every call, decode included:
`EXL3_PLACEMENT='0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1'` (below), which gives those
layers alone the `stream_only` execution of `EXL3_MOE_CPU_MODE` (`-mcl` offloads the first layers
and cannot express this), over a 4096-token prefill plus eight teacher-forced decode steps: two
identical runs differed by up to 0.59 in actual-token log-probability (p99 0.062, one top-1 change)
with atomic accumulation, and were bitwise identical with slots, also with 2048-token chunks.
Prefill time +0.40% (1105 vs 1100 tok/s) and decode +0.65%, two interleaved trials each, a
difference of the order of their noise. Allocated peaks were unchanged in that layout, where the
resident MoE layers of the same device already allocate a slot scratch of the same size.
In the default mode, an explicit `EXL3_MOE_STREAM_T=1` streams every selected expert of a
prefill chunk of at least `EXL3_MOE_STREAM_MIN_ROWS` rows; calls with fewer rows (decode) never
reach the streamed path, so the decode figure above applies only where decode streams too.

The switch orders the accumulation, nothing else. Results still differ between chunk sizes;
between tiers (an expert whose row count crosses `EXL3_MOE_STREAM_FUSED_T` or the reconstruct
row cap takes a different projection path; `EXL3_MOE_FUSED_PREFILL` removes the tiers); with the
staging-batch partition (`EXL3_MOE_STREAM_BATCH_EXPERTS`, `EXL3_MOE_CPU_WSLOT_MB`), which decides
the batched reconstruct groups and so the batch count and padded row count of their GEMMs (on
DeepSeek-V4.1-Flash, one-expert batches against the default batches differed by p99 0.047, max
0.22, in actual-token log-probability); between GPU types; on the CPU tail, whose arithmetic is
its own; and, in the default `compute` mode with `EXL3_MOE_STREAM_T` unset, between processes on
links slower than about 25 GB/s, where the bandwidth probed on a device's first streamed prefill
sets the streaming threshold and so which experts run on the CPU tail (set `EXL3_MOE_STREAM_T`
explicitly for reproducibility across processes; `stream_only` skips the probe).
`EXL3_MOE_FUSED_DET=0` restores atomic accumulation on the resident and the streamed path alike,
for A/B comparisons. `EXL3_MOE_RECON_DET` only matters where the batched tier does not write into
slot scratch, i.e. with `EXL3_MOE_FUSED_DET=0`: `1` accumulates one expert at a time, `0` with one
atomic `index_add_`. The GDN/KDA recurrent decode kernels (Qwen3.5, Qwen3.8, GLM-5.3) reduce their
per-slice partial dot products in a fixed order unconditionally (no switch, no cost), so greedy
decode on those models is reproducible as well.

### `EXL3_MOE_FUSED_PREFILL` (default: `0`)

Fused-only MoE prefill. By default a prefill chunk sorts its routed experts into tiers by how many
rows each one received: experts up to the fused kernel's row capacity (`EXL3_MOE_FUSED_ROWS`,
`EXL3_MOE_FUSED_ROWS_WIDE` for GPU-resident experts, `EXL3_MOE_STREAM_FUSED_T` for experts
streamed from system RAM) run through the fused `exl3_moe` kernel, which dequantizes the weights
inside the GEMM and applies the Hadamard transforms around it; hotter experts take the batched
reconstruct tier or the per-expert reconstruct path, which materialize FP16 weights and multiply
with cuBLAS. The tiers compute the same function with different arithmetic, so the result for one
row depends on how many rows its expert received in the same call: another chunk size, another
prompt in the same batch, or a longer prefill moves experts between tiers. With `1`, every routed
expert of every call that takes the fused MoE path (prefill chunks; calls of up to 8 rows keep the
decode kernels, see below) is computed by the fused kernel, whatever its row count. An expert with
more rows than the kernel's temp buffers hold runs in row stripes of the buffer height: each stripe
is finished before the next one reuses the buffers, the expert stays with one group of SMs, and
each assignment's output goes to its own slot as with `EXL3_MOE_FUSED_DET`, which is required. The
stripes are a compile-time option of the kernel: every fused MoE kernel instance has a row-striped
twin, which this setting selects for every row count of the fused path, and the default instances
that every MoE prefill runs with the setting off are compiled without the stripe code.

What it covers: GPU-resident experts (`BlockSparseMLP`) and experts held in system RAM and
streamed to the GPU (`-mcl`, `-mcs`, placement `experts=stream`, and the streamed hot experts of
`compute` layers). What it does not change: the bsz 1-8 decode kernels (`exl3_moe_coop` and the
native block-sparse MLP's graphs), the CPU worker's arithmetic, routing, shared experts, and which
experts a call selects. The temp buffers keep their size: the row capacities above become the
stripe height, and no longer limit the tier.

Values: `0` (default) or `1`. Anything else raises a `ValueError` when `exllamav3` is imported,
as does `1` together with `EXL3_MOE_FUSED_DET=0` (the setting exists for reproducible arithmetic,
which atomic accumulation would undo). `EXL3_STABLE_ARITHMETIC=1` implies `1`; an explicit `0`
with it is refused. Read once, at import; set it before importing `exllamav3`.
`MoeCpuTuning.fused_prefill` holds the streamed tier's copy.

Refused, never a silent fallback to the reconstruct tiers:

- at load, an MoE layer on a GPU whose experts the fused kernel does not implement: not all EXL3,
  codebooks other than one uniform `mcg` or `mul1`, an activation other than gated SiLU / GELU or
  gateless ReLU^2, per-expert biases, padded dims, or `infer_params.no_reconstruct`; a
  `ValueError` names the layer;
- at load, resident temp buffers smaller than the largest row tile (64 rows with `EXL3_MOE_MTILE`
  on mul1 experts, else 16; set by `EXL3_MOE_FUSED_ROWS_WIDE` / `EXL3_MOE_FUSED_ROWS`);
- `EXL3_MOE_STREAM_FUSED_T` below that tile (64, or 16 with `EXL3_MOE_MTILE=0`), when the CPU
  offload module is first imported, i.e. on the first load with experts held in system RAM;
- a streamed layer the fused kernel does not implement, when its worst case is measured at load
  (`EXL3_AUTOSPLIT_WORSTCASE`) or at the latest on its first streamed prefill.

Memory: no new buffers; the fused tier's temp buffers are the ones described under
`EXL3_MOE_FUSED_ROWS_WIDE` and `EXL3_MOE_STREAM_FUSED_T`, plus the slot scratch of
`EXL3_MOE_FUSED_DET` (`assignments x hidden x 4` bytes per call). The batched reconstruct tier's
slabs and dequantized weights and the per-expert path's intermediates are never allocated, and
the autosplit's worst-case estimate of a layer drops them.

Performance: the fused kernel is the faster tier below the row capacities above; for hot experts it
dequantizes the expert's weights again for every row tile (64 rows with `EXL3_MOE_MTILE`) where the
reconstruct tiers dequantize once per call, so long prefill chunks on models with few, hot experts
slow down most. A traced (not controlled) diagnostic 4096-token prefill of DeepSeek-V4.1-Flash, with
fixed-row GEMM tiles and a whole-column partition of the fused kernel's K reduction in both runs,
and 4096-row fused buffers instead of stripes, took 5.37 s with every expert fused against about
4.43 s with the tiers. The row stripes are compiled into instances of their own, the row-striped
twins of the 54 fused MoE instances (`quant/comp_units/exl3_moe_inst_*_st.cu`), which only this
setting selects. With it off every fused MoE launch runs the default instances, whose code the
stripes do not change: ptxas reports the same registers and spill bytes, and the SASS is identical,
for all 54 on sm_80, sm_86 and sm_120. The twins carry the stripe code. Their registers are the
default instances' except in six instance and GPU pairs (k2 and k3 N=128 16-row on sm_86, 124 and
122 -> 128; k5 and k6 mul1 N=128 16-row on sm_120, 128 -> 127), and ptxas reports more spilling in
160 of the 162 pairs: about 2.2 times the bytes in the median, from about 1.04 to 14 times, and 17
pairs that spilled nothing before spill now (bytes stored / loaded: k3 N=256 16-row 68/178 ->
164/422 on sm_80 and 76/184 -> 136/272 on sm_120; the 64-row tile on sm_86 320/1402 -> 344/1446; k3
N=128 16-row on sm_86 0/0 at 122 registers -> 52/148 at 128). Measured on its own on
DeepSeek-V4.1-Flash, in a 32K-token prefill split over an sm_80 and an sm_120 GPU (the experts of 11
layers computed by the CPU, the others resident): 904 and 906 tok/s with the setting against 997 and
996 tok/s without (-9%), and a 64K-token teacher-forced NLL of 0.250987 against 0.250437. Build
cost: 54 more instances, each about as costly to compile as its default unit, about 10 s of compiler
CPU time per compilation unit for sm_80, sm_86 and sm_120 (38 units) and 135 MiB of objects, 93 MiB
in an sm_80 + sm_120a build.

Determinism: removes the tier boundaries, so an expert's arithmetic no longer depends on which
tier its row count selects. On its own it does not make results independent of the row count:
the fused kernel splits each dot product's K reduction over the SMs of its expert group, and the
group width follows the number of active experts in the call. Row stripes themselves do not
change results: `tests/test_moe_stripe_layout_.py` checks bitwise that 64-, 128- and 256-row
buffers in stripes give the same slots as buffers holding all 700 rows of an expert. Real
DeepSeek-V4.1-Flash experts (17 to 1025 input rows, 16/64/256-row buffers, both supported row
tiles, two GPU types) and whole-model scores over 4,104 predictions were bitwise equal between
striped and full-height buffers, measured together with the other row-count-independent changes.

When to use it: as part of the row-count-independent profile `EXL3_STABLE_ARITHMETIC`, which
sets it, or to take the tier boundaries out of a comparison. Keep it off for throughput.

```sh
EXL3_MOE_FUSED_PREFILL=1 python eval/ppl.py -m /path/to/moe/model
```

```python
import os
os.environ["EXL3_MOE_FUSED_PREFILL"] = "1"      # before importing exllamav3
from exllamav3.model.math_policy import FUSED_PREFILL
assert FUSED_PREFILL
```

### `EXL3_MOE_PINNED_ARENA` (default: `0`, experimental)

Back the CPU worker's expert-weight arena with shared chunks that the parent process also maps
and page-locks (`cudaHostRegister`), and lay each expert's gate/up/down trellis tensors out as
one contiguous block. Streamed prefill (`EXL3_MOE_STREAM_T`) then DMAs an expert's block
straight out of the arena on the copy stream instead of having the worker's stager thread
memcpy it into the pinned handoff ring first; the stager is the prefill bottleneck on fully
offloaded models (mistral-small-4 119B, 54 GiB of experts: 4k-token prefill 700 -> 1850 tok/s
on a gen5 x16 link, decode unchanged within noise). Costs: every chunk is registered with CUDA
as it appears (~0.2 s per GiB, overlapping the load) and the arena is shared memory counted in
both processes' RSS. On Linux the chunks are `memfd`s passed over the worker pipe (resolved at
runtime through libc or the raw syscall when the interpreter was built without
`os.memfd_create`, as conda builds are); shmem pages
only get transparent huge pages where `/sys/kernel/mm/transparent_hugepage/shmem_enabled`
allows it at allocation time: `within_size` (or `always`) is what works, since the chunks are
preallocated with `fallocate` and then page-locked by the parent, so neither the `advise` hint
nor the later collapse pass can convert them (on the default `never` the CPU kernels run on 4K
pages, which cost a few percent of decode on some hosts). On Windows the chunks are named
pagefile-backed sections (4K pages); each must fit both free physical RAM and commit headroom
when it is created, or the load fails naming the chunk.

### `EXL3_HOST_MEM_RESERVE_MB` (default: `2048`)

Host-memory guard for the large CPU allocations (CPU MoE expert arena chunks, the n-gram table
held in RAM with `--ngram_ram`): before each one, `MemAvailable` (from `/proc/meminfo`, or
psutil where that is unavailable) must cover the allocation plus this reserve, or the load
fails with a message naming the allocation. Linux has no allocation-time failure for anonymous
or shmem memory: an oversized arena only fails once the machine has swapped itself into a
minutes-long stall and the OOM killer picks a victim, and pinned pages cannot be reclaimed at
all. `0` disables the check.

### `EXL3_MOE_ARENA_HUGE` (default: unset)

Linux only. With `EXL3_MOE_PINNED_ARENA=1`: `2m` or `1g` backs the memfd chunks with hugetlbfs
pages (`MFD_HUGETLB`) of that size instead of shmem. Requires reserved huge pages
(`vm.nr_hugepages`, or `hugepages-1048576kB` for `1g`) covering the whole arena; allocation
fails with a clear error otherwise. Rejected on Windows.

### `EXL3_MOE_MTILE` (default: `1`)

Row tiles for the fused MoE prefill kernel. The kernel dequantizes each expert's weights once
per 16-row tile, so an expert holding 100 rows re-runs the whole B pipeline seven times. With
this on, experts with 17-32 rows go through a 32-row-tile instance and experts with more than 32
rows through a 64-row one, each as its own launch over its expert range (up to three launches
per layer, largest tile first; a wide instance finishes an expert's remainder with the largest
smaller tile). The wider tiles are separate kernel instances rather than a runtime switch inside
one kernel: sharing one 128-register budget between the tiers costs the 16-row path 60-70% on
Ada / Ampere. Instantiated for the mul1 codebook as N = 128 tile-shape kernels: 30-47% less
fused-kernel time at 24-128 rows per expert on every GPU generation for Qwen3.8-class shapes,
and ~20% for dims that are multiples of 256 (gemma4, Qwen3-30B), where they replace the N = 256
16-row tiling for experts above 16 rows (the N = 256 instance stays for the small-expert
launch, where it is 10-20% faster). Model level, 4k chunks: Qwen3.8 +10% on the PRO 6000, +3%
on a 4090 + 3090 split; gemma4-26B +1-4%; Qwen3-30B neutral. Other codebooks and the
all-fused fast path (no host-side counts) keep the single launch. Outputs are bit-identical to
the 16-row tiling at equal group geometry (per-launch active counts widen the groups, which
reorders the fp32 k-slice reduction to rounding level). Set to `0` for the single launch.

### `EXL3_MOE_TILE_N` (default: `0` = automatic)

`128` keeps the N = 128 tile shape for the fused kernel's 16-row launches on dims that are
multiples of 256 (which otherwise take the N = 256 instances). Measurement knob; the N = 256
instance is faster for those launches.

### `EXL3_MOE_FUSED_ROWS_WIDE` (default: `256`)

Fused-tier row capacity per expert for layers that use the wide tiles (see `EXL3_MOE_MTILE`);
`EXL3_MOE_FUSED_ROWS` still applies to every other layer. With the wide tiles the fused kernel
beats the batched reconstruct tier up to 256 rows (Qwen3.8 4k chunk on the PRO 6000: 6.87k ->
7.06k tok/s over 128 rows), at 4 x concurrency x rows x (hidden + intermediate) x 2 bytes of
static buffers per device (Qwen3.8 on a 188-SM card: +38 MB over 128 rows). With
`EXL3_MOE_FUSED_PREFILL=1` this and `EXL3_MOE_FUSED_ROWS` set the height of the fused kernel's row
stripes instead of the tier limit (at least 64 rows with the wide tiles, 16 without).

### `EXL3_MOE_BATCH_RECON` (default: `1`), `EXL3_MOE_STREAM_BATCH_RECON` (default: `1`)

Batched reconstruct tier for prefill (`exllamav3/modules/moe_batch_recon.py`): experts with
more assigned rows than the fused MoE kernel's capacity (up to the `EXL3_MOE_RECON_TILES`
budget below) are dequantized and multiplied in count-sorted groups (one pointer-table reconstruct, one
strided-batched GEMM and one Hadamard launch per projection for the whole group, rows padded
to the largest expert of the group) instead of one expert at a time, cutting the per-expert
launch count that dominates many-expert models at large chunk sizes (Qwen3.8-Flash-Next 512
experts, 8k tokens: +16% prefill). The arithmetic is the same as the per-expert path (fp16
rotated-basis weights, fp16 activations, fp32 down-projection output; models with fp32
intermediates such as Gemma 4 keep fp32 gate/up outputs and the activation kernel writes the
fp16 input of the down projection, as in the per-expert path). `EXL3_MOE_BATCH_RECON`
covers GPU-resident experts, `EXL3_MOE_STREAM_BATCH_RECON` the streamed CPU experts (the
`EXL3_MOE_STREAM_FUSED_T` overflow tier). Set to `0` to restore the per-expert loops.

### `EXL3_MOE_RECON_TILES` (default: `64`), `EXL3_MOE_RECON_BATCH` (default: `16`), `EXL3_MOE_RECON_PAD` (default: `1.1`), `EXL3_MOE_RECON_ROWS` (default: `16384`), `EXL3_MOE_RECON_MB` (default: `256`), `EXL3_MOE_RECON_MAX_ROWS` (default: unset), `EXL3_MOE_RECON_FOLDED` (default: `1`)

Tuning for the batched reconstruct tier. Batching pays while a single expert's GEMM cannot
fill the GPU: an `m x n` GEMM launches about `m/128 * n/128` output tiles, and below roughly
the SM count the strided-batched kernel is much faster (PRO 6000, `k = 2048`: `n = 768` runs
1.45x faster batched at `m = 1024` and 2.9x at `m = 256`, `n = 1792` breaks even at
`m = 1024`), while above it the batched cuBLAS kernels are 10-15% slower than the single-GEMM
ones, and the padded slabs cost traffic that grows with the rows. `EXL3_MOE_RECON_TILES` is
that tile budget: an expert stays on the per-expert path once its narrowest projection
(`min(intermediate, hidden)`) would span more tiles, i.e. above `TILES * 16384 / n_min` rows
(1365 rows for `n = 768`, 512 for `n = 2048`). Measured at 8k tokens on a PRO 6000: lfm2.5
(`n` 1792/2048) loses 15% with everything batched and 4% at a 1024-row cap, parity at 512;
Qwen3.8-Flash-Next (`n = 768`) is within 2% for any cap. `0` batches
every expert; `EXL3_MOE_RECON_MAX_ROWS` overrides the derived row cap directly. The other
knobs: experts per group; `EXL3_MOE_RECON_PAD`, the padded-to-real row ratio a group may reach
before the planner starts a new one (groups fill largest expert first, so this splits only
groups of uneven experts: on Qwen3.8-Flash-Next at 4096 tokens a limit of 1.5 pads 32% of the
tier's rows, 1.1 pads 8% and is 2% faster overall, beating a batch of 8 at 1.5; lfm2.5 gains
2-5%); the padded-row budget per group; the dequantized-weight scratch budget, which shrinks
the group for large expert shapes. `EXL3_MOE_RECON_FOLDED=1` (default) folds both
Hadamards and the sign vectors into the dequantized weights (the formulation the dense
`Linear` prefill path uses above 1024 rows): fewer launches, about 4% faster prefill on
Qwen3.8-Flash-Next, at the cost of rounding the folded weights to fp16, which roughly doubles
the tier's error against an fp32 reference of the same quantized weights (9.6e-4 vs 4.6e-4).
Against the unquantized model that difference is invisible: lfm2.5 4.10bpw over 10 x 2048
tokens scores KL 0.0902 to the HF weights folded and 0.0907 unfolded (top-1 85.2% vs 84.9%,
perplexity 40.50 vs 40.56). Note that the two modes are far apart from each other on
routing-sensitive models (Qwen3.8 KL 2e-2, lfm2.5 6e-3) while equally far from the
baseline, so mode-to-mode distance is not a fidelity measure. `0` selects the activation-side
Hadamards.

### `EXL3_MOE_CPU_START_TIMEOUT` (default: `60`)

Seconds the parent waits for the CPU worker to signal ready after every offloaded layer has
been handed over. Startup is the shared-memory attach, layer registration and thread spawn,
so the default is only a safety net against a wedged worker; raise it on very slow hosts.

### `EXL3_MOE_CPU_PIN` (default: `1`)

Pin each worker thread (and the worker's own main thread) to a distinct physical CPU core,
SMT siblings last, instead of leaving placement to the OS scheduler. On an SMT host, unpinned
placement is a real source of run-to-run throughput variance, two workers can land on the same
physical core (contending for its execution resources) on one run and not the next; measured on
a 24-core/48-thread SMT2 box, this swung matrix-decode throughput 61–105 GB/s run to run,
pinned flat at ~105 GB/s (88% of the box's measured 24-thread DRAM read bandwidth). Set to `0`
to disable, e.g. on a shared/multi-tenant host where fixed placement may fight the scheduler's
own balancing across other processes. Falls back to no pinning if the CPU topology can't be
read.

### `EXL3_MOE_HANDOFF_PROF` (default: unset)

Enable GPU/CPU handoff profiling, for debug purposes. 

## Reproducible arithmetic

### `EXL3_STABLE_ARITHMETIC` (default: `0`)

Opt-in arithmetic profile for results that do not depend on how many rows a call processes. With
the default dispatch the same token can get slightly different logits depending on the prefill
chunk size, on whether it was decoded one token at a time or prefilled, or on how much of a prompt
was reused from the cache, because several operations switch kernels, algorithms or numeric
formulations by row count: cuBLAS picks its reduction per problem shape, EXL3 linears use direct
quantized kernels up to 144 rows and rotate the Hadamard transforms across an FP16 rounding below
1024 rows, MLPs and MoE layers have separate small-batch decode kernels, and routed experts move
between the fused kernel and the reconstruct tiers with their row count. The differences are
last-bit rounding, but they can flip a near-tie (a top-k selection, a greedy token) and then grow.
With `1` each operation below takes one arithmetic path at every row count, the one prefill uses:

- `EXL3_HGEMM_FIXED_ROWS=128` is implied: native GEMMs in fixed 128-row cuBLAS tiles, FP16
  linears through them, native CUDA graphs disabled (see that entry).
- `EXL3_MOE_FUSED_PREFILL=1` is implied: every routed expert of a fused-path call through the
  fused kernel, in row stripes (see that entry).
- EXL3 linears dispatched through `LinearEXL3.forward` reconstruct their weights at every row
  count, decode included, instead of using the direct quantized kernels up to 144 rows, and always
  in the original basis (Hadamard transforms folded into the weights) instead of the rotated
  basis below 1024 rows. Projections that native blocks or grouped kernels run themselves are not
  among them (see what it does not cover, below).
- MLPs and gated MLPs, shared experts included, run their separate linears at every row count:
  no native single-token or small-batch block (`run_bsz1`, `run_bszN`) and no grouped gate/up
  GEMM for up to 32 rows.
- Block-sparse MoE layers send every row count, decode included, to the fused routed-expert path
  instead of the 1-8-row decode kernels, and apply a shared-expert gate as its own linear instead
  of the fused gate projection used up to 32 rows.
- Experts held in system RAM must be computed on the GPU: offloaded layers must register in
  stream mode (`-mcm stream_only` / `EXL3_MOE_CPU_MODE=stream_only`, or `experts=stream` in a
  placement). The CPU worker's arithmetic depends on the host and on how rows split between CPU and
  GPU. With an explicit placement `-mcm` is refused, so the RAM-held experts of layers the
  placement does not cover (`-dmcl`: an MTP head or a draft model) stay CPU-computed and are
  refused under the profile: with a placement, keep those experts in VRAM.

Values: `0` (default) or `1`; anything else raises a `ValueError` when `exllamav3` is imported. It
may be combined with `EXL3_HGEMM_FIXED_ROWS` and `EXL3_MOE_FUSED_PREFILL` left unset or set to
the implied values (`128`, `1`), so scripts that set all three keep working.

When it is read: once in Python, when `exllamav3` is imported (the constants of
`exllamav3.model.math_policy`, which the modules above import), and once in the extension, on
first use (`ext.stable_arithmetic()`, `ext.hgemm_fixed_rows()`). Set it in the environment of the
process, before importing `exllamav3`. There is deliberately no command-line flag or `Config`
attribute: a switch applied after the import would change the native side and not the frozen
Python constants, and give a silently mixed profile.

Refused, as an error rather than a partial profile:

- at import, a value other than `0`/`1`, `EXL3_HGEMM_FIXED_ROWS=0`, `EXL3_MOE_FUSED_PREFILL=0`,
  `EXL3_MOE_FUSED_DET=0` (atomic expert accumulation) or `EXL3_NO_FUSED_RECONSTRUCT` set to
  anything but `0` (rotated-basis reconstruction), each with a `ValueError` naming the variable;
- at load, an EXL3 linear with `infer_params.no_reconstruct` set or dims that are not multiples
  of 128, an offloaded MoE layer whose experts would be computed by the CPU worker (before the
  worker starts; this includes `-dmcl` layers next to an explicit placement), and everything
  `EXL3_MOE_FUSED_PREFILL` refuses.

What it does not cover: operations not listed above keep their row-count-dependent dispatch,
among them the attention kernels of most architectures (prefill and decode kernels differ); the
projections inside the native decode blocks (attention and MLA with `EXL3_BC_ATTN` /
`EXL3_BC_MLA`, GatedDeltaNet / KDA with `EXL3_BC_GDN`, Mamba2), which run their EXL3 projections
through the direct quantized kernels themselves, and the grouped q/k/v(/z) `exl3_mgemm` that
attention, sliding-window attention and GatedDeltaNet use for calls of up to 32 rows, so on
those architectures decode and prefill projections still differ; the GatedDeltaNet / KDA prefill
path (its token-major convolution rounds the projection to BF16 only
for calls of up to 32 rows), the n-gram embedding (PLE) projections (`at::matmul` in 1024-row
slabs inside the extension), DeepSeek-V4's native attention block and hyper-connection head,
tensor parallelism, and the CPU. Results also still differ between GPU types. The profile has
only been validated end to end on DeepSeek-V4.1-Flash.

Memory: the fixed-row GEMM tiles (two tiles per GEMM call), no reconstruct-tier slabs. Decoding
through the reconstruct path allocates each linear's dequantized weights per call (`in x out`
FP16, in slices of at most 32768 columns), the same transient every long prefill already
allocates and the autosplit already measures.

Performance: slower, in prefill and much more in decode. Every decoded token reconstructs the
weights of every EXL3 linear that `LinearEXL3.forward` dispatches and runs the MoE layers through
the prefill kernels, without native CUDA graphs; prefill pays the fixed-row GEMM tiles (2.6x-3.1x on
a large projection, see `EXL3_HGEMM_FIXED_ROWS`) and fused-only experts. On DeepSeek-V4.1-Flash on
two GPUs with the experts of 11 layers in system RAM, a 64K-token scoring pass ran at 796 tok/s with
the complete profile (experts streamed) against 1040 tok/s with the default arithmetic (experts
computed by the CPU), so the two runs differ in more than this setting; decode speed was not
measured. Use it for validation and reproducibility work, not for serving.

Determinism: within the operations above, a row's result no longer depends on the row count of
the call, on the chunk boundaries or on decode versus prefill (for a given GPU type, model and
placement). Combined with `EXL3_MOE_FUSED_DET` (on by default), identical calls are also bitwise
repeatable. Whether a whole model becomes chunk-size invariant depends on every operation it uses
being covered, see above.

```sh
EXL3_STABLE_ARITHMETIC=1 python eval/ppl.py -m /path/to/model
EXL3_STABLE_ARITHMETIC=1 python examples/chat.py -m /path/to/moe/model -mcl 8 -mcm stream_only
```

```python
import os
os.environ["EXL3_STABLE_ARITHMETIC"] = "1"      # before importing exllamav3
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
assert ext.stable_arithmetic() and ext.hgemm_fixed_rows() == 128
```

## Model loading

### `EXL3_EXPANDABLE_SEGMENTS` (default: `1`)

Use expandable segments for all Torch allocations. Opt out with a value of 1 or by explicitly
setting `PYTORCH_CUDA_ALLOC_CONF`.

### `EXL3_LOAD_ARENA` (default: `1`)

Slab allocation for small weight tensors during (deferred) module loads: tensors up to 16 MB
are carved out of shared 128 MB per-device blocks (first-fit over the open blocks, so partially
filled tails are packed by later small tensors) instead of getting one CUDA caching-allocator
allocation each. MoE models with many small per-expert tensors otherwise shatter the allocator
into tens of thousands of segments with large reserved-but-unallocated overhead. Unloading a
module frees its blocks; at most one boundary block shared with a neighboring module stays
pinned. Set to `0` to fall back to per-tensor allocations.

### `EXL3_NGRAM_STREAM` (default: `1`)

Default for `Config.infer_params.ngram_stream_from_disk`: stream an n-gram embedding table
(PLE models, e.g. Qwen3.8-Flash-Next) from disk with per-forward row gathers (run-coalesced
positioned reads into pinned staging — threaded preads on Linux, overlapped `ReadFile` at high
queue depth on Windows) instead of loading the whole table into system RAM. The quantized table
is tens of GB, and streaming costs little on SSD-class storage (decode is latency-tolerant at
~30 rows/token; prefill gathers are batched). Set to `0` to hold the table in RAM — worthwhile
only when the table lives on high-latency storage (e.g. HDD, where per-row seeks make streaming
unusable). Also settable per load via `config.infer_params.ngram_stream_from_disk` or
`--ngram_ram` in `model_init`-based scripts.

### `EXL3_AUTOSPLIT_WORSTCASE` (default: `1`)

The layer-split autosplit loader measures each module's transient VRAM with one forward of a
dummy state and keeps that much headroom per device. Some transients don't show in that
forward: attention/MLA decode statics (QSA/DSA families) and, for block-sparse MoE layers,
everything that depends on how the real workload routes tokens: the deterministic slot scratch
(all assignments slotted, `EXL3_MOE_FUSED_DET`), the batched-reconstruct group temporaries,
and the CPU-offload host's GPU side (per-device weight ring, fused-tier buffers, batched tier
statics, padded fp32 outputs), which the measuring forward skips entirely. With the switch on,
these modules allocate and drop an upper bound of those transients inside the measuring window
(`autosplit_extra_measure`), so the split leaves room for them. `0` restores the plain
measured forward.

### `EXL3_AUTOSPLIT_PREPARE` (default: `1`)

Before measuring a module's transient memory, the autosplit loader lets the module allocate
the state it would otherwise create lazily inside the measuring forward and keep for good
(CPU-offload host buffers, batched-reconstruct tables, decode graph slot statics). Measured
inside the window, that state would be budgeted twice: as resident memory and again as
transient headroom. Set to `0` to skip the preparation step, for comparison. With verbose
loading, the loader prints a notice for any module that still leaves more than a small amount
resident during its measuring forward.

### `EXL3_AUTOSPLIT_MARGIN_MB` (default: `256`)

The layer-split loader closes a device when its remaining headroom no longer covers the largest
transient measured on it. Besides the `-gs` budget, the check is made against the device
itself (free VRAM plus the allocator's reserved-but-unallocated pool): a split value at or above
the card's size otherwise plans against the several hundred MiB the CUDA context, the loaded
kernels and other processes hold, and the load passes but the first real forward fails. This
margin is added on top of the measured transient in that physical check, covering what a real
forward keeps live around a module (recurrent test states, gathered embeddings, allocator
slack). `0` disables the margin.

### `EXL3_PLACEMENT` (default: unset)

Experimental. Default for `Config.infer_params.placement`, which `-placement` / `--placement`
sets in `model_init`-based scripts: an explicit placement of the text model, for any model in
layer-split mode. One rule per range of decoder layers gives their GPU and, for MoE layers, where
the routed experts live:

```
<layers>=cuda:<n> [experts=vram|stream|cpu|split] [cpu=<k>]
```

Layers are `7`, `0-11`, `0-3,8-11`, `*` (every layer no other rule lists), `embed` (the modules
before the first layer) or `head` (after the last); rules are separated by `;` or newlines, `#`
starts a comment. `experts=vram` (default) keeps the experts in the GPU's memory, `stream` keeps
them in system RAM and computes them on the layer's GPU only (as `-mcl` with
`EXL3_MOE_CPU_MODE=stream_only`), `cpu` computes them on the CPU worker (as `-mcl`), and
`split cpu=<k>` keeps `k` of every layer's routed experts on the worker (as `-mcs k`). The complete
grammar, the loader's behaviour and worked examples are in [placement.md](placement.md).

No placement: unset, empty, or only whitespace, `;` and `#` comments; `--placement ""` clears a
set variable; the Python attribute `None` (or `""`). The autosplit and `-mcl` / `-mcs` / `-mcm`
then decide, as before. A whole value of `none`, `off`, `auto`, `default`, `0` or `false` is
refused with a `ValueError`, not read as "no placement" (placement.md, "No placement", says why
and lists every spelling).

When it is read: the variable when a `Config` is created (a malformed value raises a `ValueError`
there); `--placement` when `model_init` builds the `Config`, overriding the variable; the Python
attribute (a string or a parsed `Placement`) when `model.load()` starts, or, for
DeepSeek-V4.1, already when the model object is built and again at load (placement.md,
"DeepSeek-V4.1"). Set it before loading, and for DeepSeek-V4.1 before `Model.from_config`.

What it does at load: every module loads on the device its rule gives, with the autosplit's
measuring forward and headroom checks. `-gs` / `use_per_device` / `reserve_per_device` still cap
each device's memory, but never move a module: a module that does not fit is a `RuntimeError`
naming the module, its device and the loader's reason (the headroom it needs includes the
transients of `EXL3_AUTOSPLIT_WORSTCASE`, among them a quantized cache's staging,
`EXL3_QC_STAGING`). An MoE layer whose rule asks for RAM-held experts but cannot hold them
(unsupported activation, not mul1, K > 8, mixed per-expert biases) is a `RuntimeError`, not a
silent fallback to VRAM.

Refused with a `ValueError` before anything loads: together with `-mcl` / `EXL3_MOE_CPU_OFFLOAD`,
`-mcs` / `EXL3_MOE_CPU_SPLIT`, `-mcm` / `EXL3_MOE_CPU_MODE` other than `compute` or
`EXL3_MOE_CPU_SPLIT_LAYERS` (all of which it replaces); with a single-device or tensor-parallel
load; when a rule names a layer the model does not have or a layer is placed by no rule or by
two; when a device is not visible or the budgets exclude it; and when one GPU holds both `stream`
layers and `cpu` / `split` layers (one kind of RAM-held experts per GPU). DeepSeek-V4.1 adds a
rule of its own on where the device may change (placement.md, "DeepSeek-V4.1").

Scope: the text component only. A draft model loaded from its own directory, an MTP head and a
vision tower load as without it; `model_init` clears the placement of a draft model's config, and
a script that builds a draft `Config` itself sets `draft_config.infer_params.placement = None`
(every `Config` reads the variable). `-dmcl` experts are computed by the CPU worker under a
placement, since `-mcm` is refused next to it.

Cost: none of its own. Each change of device between consecutive modules copies the running
state once per forward pass, as any split does; the expert modes cost what `-mcl`, `-mcs` and
`EXL3_MOE_CPU_MODE` cost. Determinism: the same placement gives the same devices on every load,
and it computes nothing differently from older settings that describe the same layout (on
DeepSeek-V4.1-Flash, an earlier revision of this code gave 64K-token scores through the placement
bitwise identical to the same layout set up with model-specific expert settings this code no
longer has; not repeated on this code).

When to use it: to fix a split independently of free memory, to choose which layers keep their
experts in system RAM (not only the first N) and how each is computed, or to order the GPUs.

```sh
python examples/chat.py -m /path/to/model --placement "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
EXL3_PLACEMENT="*=cuda:0; 50-59=cuda:0 experts=cpu" python eval/perf.py -m /path/to/model
```

```python
config = Config.from_directory(model_dir)
config.infer_params.placement = "0-23=cuda:0; 24-47=cuda:1 experts=split cpu=64"
model = Model.from_config(config)
model.load()
```

### `EXL3_VISION_PINNED` (default: `0`)

Default for `Config.infer_params.vision_pinned`: store the vision component's linear-layer
weights (fp16 or EXL3 trellis) in pinned host memory instead of VRAM, computing straight from
a zero-copy device alias. Trades vision-tower speed for VRAM. Set before loading the vision
component.

## Multi-GPU

### `EXLLAMA_NO_P2P_COPY` (default: unset)

Controls device-to-device tensor moves (the layer split boundary, draft/MTP heads reading the
target model's states, sparse-attention selections shared between layers). On some platforms
the driver reports peer-to-peer access that the PCIe fabric does not deliver, and a direct copy
silently yields garbage. Unset: the first move between each pair of GPUs probes it (a few
random floats there and back, checked on the host) and, if the probe fails, every later move
between that pair bounces through system memory, with a warning printed once. Set to `1`: always
bounce, no probing. Set to `0`: always copy directly, no probing.

### `EXLLAMA_MASTER_ADDR` (default: `127.0.0.1`), `EXLLAMA_MASTER_PORT` (default: auto)

Rendezvous address and port for the tensor-parallel backend. The port defaults to a free port
picked at startup.

### `EXL3_TP_ROUTING_CHECK` (default: `0`)

Debug for expert-parallel MoE layers. Routing is normally replicated: every rank holds the router
and selects experts for itself on the rank-identical residual stream (the deterministic int8 GEMM
and mix kernels make the streams bit-identical across ranks and architectures), which saves two
broadcasts per MoE layer per token. With this set, the layers fall back to routing on the output
rank and broadcasting the selection, while every other rank also routes locally and compares its
top-k selection and weights with the broadcast one; the mismatch counts (split by row class:
single-row, other decode-sized, prefill) are printed at exit. Any mismatch means the streams or
the router differ between ranks, so this is the acceptance test for changes to replicated paths.

### `EXL3_TP_STREAM_HASH` (default: `0`)

Debug: set to N to print a digest of the residual stream after every module on every rank for
the first N real forward passes (warmup passes excluded), to locate where ranks stop agreeing
bit for bit.

### `EXL3_TP_NO_FWD_BARRIER` (default: `1`)

Skip the pass-start barrier in tensor-parallel forward passes. The native collectives are each
ordered by their own stage counters, so the barrier is not required for correctness; skipping it
saves one spin-kernel launch per rank per pass. Set to `0` to restore the barrier (one aligned
sync point per pass at the cost of a small amount of GPU spin time).

### `EXL3_TP_NO_FP16_WIRE` (default: `0`)

The native backend's CPU-assisted all-reduce moves fp16 payloads over an fp16 wire when the CPU
supports F16C (universal on AVX2-era hardware, probed at runtime): exactly-rounded results for
two ranks, fp16-level rounding beyond, at the same PCIe traffic as the bf16 wire. fp32 payloads
always use the bf16 wire (fp16 lacks the range for residual-stream outliers). Set to `1` to
force the bf16 wire for fp16 payloads too, e.g. for A/B comparison.

### `EXL3_TP_NCCL_FP32` (default: `0`)

NCCL backend only: reduce fp32 payloads in fp32 instead of over a bf16 wire (which is what the
native backend always uses for fp32 sublayer outputs). Exact but roughly twice the reduction
traffic; for A/B testing of the wire rounding.

### `EXL3_TP_TRACE_WIRE` (default: `0`)

Print a line (once per process) when the fp16 all-reduce wire first activates. Activation check
for numerics A/B tests: whether the wire engages depends on the model's residual dtype, so a
comparison is only meaningful if the fp16-wire run actually used it.

### `EXL3_TP_REDUCE_THREADS` (default: number of participating ranks)

Number of threads slicing each large-payload accumulate in the native backend's CPU-reduce
helper (persistent workers, spin-parked between jobs; AVX-512 path only). The default of one
thread per participating rank covers the cases where a single thread's ~31 GB/s wire rate falls
behind: three or more ranks (multiple adds per chunk) and PCIe 5.0 links. Set to `1` to force
the single-threaded accumulate. Decode-size reduces are always single-threaded.

### `EXL3_TP_SPIN_RECV` (default: `0`)

Milliseconds each tensor-parallel child worker hot-polls its command pipe after finishing a
command before falling back to a blocking receive. A blocking receive pays scheduler wake
latency (tens to hundreds of microseconds, worse with deep C-states) at the start of every
forward pass; during decode the next command arrives within a few milliseconds, so a short spin
window (e.g. `4`) catches it with no wake cost, at the price of one busy core per rank for the
window. `0` disables the spin. Mostly useful on hosts where TP profiling shows a large stagger
between the main process and child workers reaching their first kernel launch.

## DeepSeek-V4.1

These variables apply only to DeepSeek-V4.1 checkpoints (`DeepseekV41ForCausalLM`); every other
architecture ignores them, with one exception: with `EXL3_DSV41_PIPELINE` on, a malformed
`EXL3_DSV41_PIPELINE_CHUNK` makes every `Config.from_directory` of the process fail for any model
until the value is fixed, because the architecture registry imports the DeepSeek-V4.1 module (a
module whose import failed is imported again, and fails again, on the next attempt). The first
entry is a loading rule that has no variable. exllamav3 runs DeepSeek-V4.1 from EXL3 checkpoints;
converting DeepSeek's original checkpoint is not supported yet, and `convert.py` refuses the
architecture before any work, saying why.

### Loading across GPUs (the layer split; no variable)

DeepSeek-V4.1 loads across several GPUs as a layer split, like every other model: split budgets
(`-gs` / `use_per_device`, or `reserve_per_device`, or neither, which lets the autosplit use every
visible device) and the autosplit, which fills the devices in layer order until each budget runs
out, or an explicit placement (`EXL3_PLACEMENT` / `--placement`, [placement.md](placement.md)),
which names every layer's device. V4.1 adds one rule about where the device may change. It has
no V4.1 setting: no variable, no config attribute, no option of its own.

Why the rule exists. A V4.1 compressed layer does not keep its own compressed KV. The kv source of
each group owns one paged pool and every other layer of the group reads it: on
DeepSeek-V4.1-Flash four pools, owned by layers 2, 8, 14 and 20, for the groups 2-7, 8-13, 14-19
and 20-39. Likewise each index source publishes a top-k selection that the layers above it reuse,
up to the next index source (2, 8, 14, 20, 24, 28, 32 and 36 on Flash), and the candidate source
(layer 20 on Flash) publishes candidate blocks for the index sources above it. The attention
kernels read the Cache's pools, the selections and the candidate blocks on the reading layer's own
device.

The rule. With a Cache attached, every change of device must fall at a free cut: a layer `c` such
that no layer at or past `c` reads a pool, a top-k selection or candidate blocks produced below
`c`. The free cuts follow from the checkpoint's config (`kv_source_layer_ids`,
`index_source_layer_ids`, `candidate_source_layer_id`); on DeepSeek-V4.1-Flash they are:

| Change of device at layer | Layers before the change | Layers from the change on |
|---|---|---|
| 1 | 0 | 1-39 |
| 2 | 0-1 | 2-39 |
| 8 | 0-7 | 8-39 |
| 14 | 0-13 | 14-39 |
| 20 | 0-19 | 20-39 |

Every other layer lies inside a kv group. Without a Cache any layer works, at a cost (below).

The one exception is the split of an explicit placement: a placement may put one change of
device inside a kv group, and the layers past it then read a replica of the group's pool, which
the Cache allocates when it is created. [placement.md](placement.md), "DeepSeek-V4.1", has that
rule, what each replica costs and the examples. The autosplit cannot add a replica: it finds
where a device fills up only while it loads, after the Cache may already have been sized.

Steering the autosplit. The change of device falls where a budget runs out, so place it with the
budgets: give each device room for the layers it should hold and their share of the Cache, and
not for one more layer. Sizes to plan with, for DeepSeek-V4.1-Flash at 3.0 bpw:

- a layer: about 4.9 GiB with every routed expert resident; about 2.5 GiB with `-mcs 192` (the
  tail 192 of each layer's 384 routed experts in system RAM, about 95 GiB of RAM for all 40
  layers). The 40 layers are about 195 GiB with every expert resident, more than two 96 GiB cards
  hold, so a two-card load keeps part of the routed experts in system RAM: `-mcs` (the tail
  experts of every layer, which relieves both devices) or `-mcl` (every routed expert of the first
  N layers, which relieves only the first);
- the Cache: each kv source's pool on its group's device, 640 MiB for a rate-2 source (layers 2, 8
  and 14) and 1.25 GiB for layer 20's rate-1 pool at `max_num_tokens` = 1M in FP16, in proportion
  for a smaller Cache; a 768 KiB sliding-window ring per layer and batch slot;
- headroom: the autosplit keeps the largest transient it measured plus `EXL3_AUTOSPLIT_MARGIN_MB`
  (256 MiB by default) free on each device. Each device that holds an index source also needs
  room for the index selection's transient, which grows with the context, up to about 160 MiB for
  a 2048-token chunk at 1M tokens (`modules/dsv41_select.py`, `transient_bytes`); the autosplit's
  measuring forward at position 0 sees about 90 MiB of it, and the difference, about 67 MiB, fits
  inside the default margin.

The embedding stays in system RAM.

What fails, and when:

- A Cache attached before the load (as `model_init` and the examples below do): the first layer
  past a change of device that is not a free cut refuses to load, with a `RuntimeError` raised
  before anything of its MoE loads, e.g.

  `DeepSeek-V4.1: the layer split put layers.12 on cuda:1, but layers 8-13 share the compressed-KV pool of layers.8, which is on cuda:0. With a Cache attached, a change of device can only fall where no pool, top-k selection or candidate list is shared across it: at layer 1, 2, 8, 14 or 20 on this model. Change the split budgets (-gs / use_per_device / reserve_per_device): less room on cuda:0, so that it holds layers up to 7 only, or room for layers 12-13 as well, so that the change falls at layer 14 (doc/env_vars.md, DeepSeek-V4.1, "Loading across GPUs"). An explicit placement (EXL3_PLACEMENT / --placement) can instead change device inside one kv group; the layers past it then read a replica of its pool (doc/placement.md, "DeepSeek-V4.1")`

  When no free cut lies above the change (layers 21-39 on Flash), the second way reads `room for
  layers <k>-39 as well, so that no change of device is left`. Load again with the budgets
  changed, or write an explicit placement with the split there.
- A Cache built after the load: the load succeeds and prints the note below, and the first cached
  forward of a layer whose pool is on another device raises a `RuntimeError`, e.g.
  `layers.12.attn: the compressed-KV pool it reads (layers.8) is on cuda:0, but this layer is on cuda:1. A Cache needs the layers that share a pool on one device: reload with every change of device at layer 1, 2, 8, 14 or 20 (doc/env_vars.md, DeepSeek-V4.1, "Loading across GPUs"). An explicit placement (EXL3_PLACEMENT / --placement) can instead change device inside one kv group; the layers past it then read a replica of its pool (doc/placement.md, "DeepSeek-V4.1")`.
- No Cache: the load succeeds and prints
  ` !! DSV41: layers 12-13 read a compressed-KV pool on another device; without a Cache each forward copies those pools whole, and a Cache would need every change of device at layer 1, 2, 8, 14 or 20. An explicit placement (EXL3_PLACEMENT / --placement) can instead change device inside one kv group; the layers past it then read a replica of its pool (doc/placement.md, "DeepSeek-V4.1")`.
  Each forward then copies the whole pool of a cut group to the other device, about 1.25 KiB per
  compressed entry of the sequence (about 40 MiB for a 64K-token sequence at rate 2), once per
  forward and device, plus the top-k selections (2 KiB per query row) and candidate blocks (8 KiB
  per row) that cross the change. A change of device at a free cut copies nothing but the hidden
  state, as for every model.
- After every load the compressed layers are checked against their pool's device. With a Cache
  attached a layer on another device than its pool is a `RuntimeError`:

  `DeepSeek-V4.1 load check failed: layers 12-13 read a compressed-KV pool on another device, which the cached path cannot do; reload with every change of device at layer 1, 2, 8, 14 or 20. An explicit placement (EXL3_PLACEMENT / --placement) can instead change device inside one kv group; the layers past it then read a replica of its pool (doc/placement.md, "DeepSeek-V4.1")`

  The check during the load refuses such a split before it completes, so this catches only a
  load that went around it, e.g. modules loaded one by one. Without a Cache it is the note above.

Three or more devices: the same rule holds for every change of device. Budgets that put layers
0-7, 8-19 and 20-39 on three devices load with a Cache; budgets that change device at 8 and 18
fail at layer 18. (An explicit placement may put one of its changes of device inside a kv group,
the others at free cuts; placement.md.)

Other loads: a load on one device (`model.load(device = ...)`, one budget, or a single visible
device) changes device nowhere and needs nothing. Tensor-parallel loading (`-tp`,
`tensor_p = True`) is refused before anything loads, with `NotImplementedError: DeepSeek-V4.1:
tensor-parallel loading is not implemented; load it as a layer split (-gs / use_per_device), or
with an explicit placement (EXL3_PLACEMENT / --placement)`. `-mcl` / `-mcs` (or the `experts=` of
an explicit placement) decide which routed experts stay in system RAM exactly as for every other
MoE model; they change the sizes above, not the rule. With an explicit placement the loader puts
every layer where the placement says, whatever the budgets would do; placement.md,
"DeepSeek-V4.1", describes when V4.1 reads it and what it checks at load.

Determinism: where the change of device falls changes no arithmetic of V4.1's own code. As at any
layer split, which GPU runs a layer can change the results of the kernels on that layer.

Example. Two cards, the tail 192 routed experts of every layer in system RAM, a 256K-token Cache:
by the sizes above (an estimate, not a measurement), 36 GiB on the first card holds layers 0-13
and their share of the Cache (about 35 GiB) but not layer 14 as well, so the change falls at the
free cut 14; the second card holds layers 14-39 and the head (about 66 GiB). If a load reports the
change at another layer, the message says which way to move the budget.

```sh
python examples/chat.py -m /path/to/DeepSeek-V4.1-Flash-exl3 -gs 36,90 -mcs 192 ...
```

```python
config = Config.from_directory(model_dir)
config.infer_params.moe_cpu_split = 192            # what -mcs 192 sets
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 262144)       # attached before the load: a cut kv group fails at once
model.load(use_per_device = [36, 90])
```

The same layout written as an explicit placement fixes the change of device whatever the budgets
leave room for: `--placement "0-13=cuda:0 experts=split cpu=192; 14-39=cuda:1 experts=split cpu=192"`
(the placement replaces `-mcs`, so the experts go in its rules). A change of device inside a kv
group, e.g. at 12, needs the placement's split and its pool replica; placement.md,
"DeepSeek-V4.1", has that example.

### `EXL3_DSV41_NUMERICS` (default: `deepseek:index`)

How DeepSeek-V4.1's attention rounds three operands that DeepSeek's reference inference code
rounds on purpose, the way the model was trained: the lightning indexer's query and key vectors,
the sliding-window KV, and the compressed KV. The indexer's top-512 selection and its candidate
blocks are discrete choices over up to a million positions, so these roundings change which
entries each query attends to, not only the last bits of its scores.

Python: `config.dsv41_numerics` on the `DeepseekV41Config`. The variable is read once, when the
config is built (`Config.from_directory`), and gives the attribute its starting value; after
that only the attribute counts. The attention layers read the attribute on every forward pass,
so one loaded model can be switched between settings (see Cache consistency below). Assigning
parses and validates the value: an invalid one raises `ValueError` and leaves the setting
unchanged, and `None` or `""` restores the default. Reading it returns the parsed setting, whose
`str()` is the canonical spelling (`"vllm:index,compressed"`, not `"vllm:compressed, index"`).

Settings. A contract, optionally restricted to some of its parts:

| Contract | `index`: index Q and K (post-RoPE) | `window`: sliding-window KV | `compressed`: compressed KV |
|---|---|---|---|
| `precise` | FP16 | FP16 | FP16 |
| `deepseek` | MXFP4: E2M1 values, one E8M0 (power-of-two) scale per 32 lanes, as DeepSeek's `fp4_act_quant` | FP8 E4M3, one E8M0 scale per 32 lanes, RoPE lanes included, as `act_quant(scale_fmt = "ue8m0")` | E2M1 values, one E4M3 scale per 16 lanes, RoPE lanes included, as `fp4_act_quant(scale_dtype = float8_e4m3fn)` |
| `vllm` | MXFP4, as `deepseek` | `fp8_ds_mla`: the 448 NoPE lanes E4M3 with one power-of-two scale per 64 lanes, the 64 RoPE lanes BF16 | `fp8_ds_mla`, as the window |

Syntax: `precise`, `deepseek` or `vllm`, optionally followed by `:` and a comma-separated list of
parts from `index`, `window` and `compressed`. A contract without a part list applies all
three parts; `precise` takes no parts. Names are case-insensitive and whitespace around them is
ignored. Examples: `deepseek:index` (the default), `deepseek` (= `deepseek:index,window,compressed`),
`vllm:window,compressed`, `deepseek:compressed`. Refused, with a `ValueError` when the config is
built (variable) or on assignment (attribute): any other contract name, `precise:<parts>`, an
empty part list (`vllm:`) and unknown parts (`deepseek:index,kv`).

Each rounding is computed from the FP16 (or FP32) operand, and the rounded values are widened
back into the operand and stored in the FP16 cache: a setting reproduces a contract's values,
not its memory savings. The kernels are checked bit for bit against independent scalar
transcriptions of DeepSeek's `kernel.py` and of the `fp8_ds_mla` rounding, on every lattice tie
(each midpoint of the E2M1 and E4M3 lattices, at a block scale of 1), scale boundary and
saturation case, and give the same bits on the CPU and on every GPU
(`tests/test_dsv41_numerics_.py`, `tests/test_dsv41_numerics_gpu_.py`). The `fp8_ds_mla` scale is
the exact one, the smallest power of two with 448 times it at or above the 64-lane peak; vLLM's
kernels compute it in FP32 with an approximate log2 on BF16 operands, which can keep the next
lower power of two for peaks a few ulps above a power of two times 448 (a negligible difference,
but not bit for bit vLLM's).

Overflow and nonfinite operands. Some values FP16 holds round past its range: in FP16, a
magnitude of 57,344 or more under MXFP4 (it rounds to 65,536), 63,488 or more under the FP8
roundings, and 65,408 or more under BF16 (the `vllm` RoPE lanes). A block whose rounding would
store such a value (a block is the lanes that share one scale: 32 under MXFP4 and the `deepseek`
FP8, 16 under E2M1, 64 under `fp8_ds_mla`, and a single lane under BF16), and a block whose
operand is not finite, is stored unrounded instead: it keeps exactly the values `precise` stores,
every other block is rounded as usual, and the forward goes on. The choice is made on the GPU,
with no host synchronization. Each device counts these blocks, and the first nonzero count
prints one warning per process:

```
 !! DSV41 numerics: 3 block(s) kept unrounded so far, because rounding them would store a value FP16 cannot hold or their operand is not finite; ...
```

The count is read back without waiting for the GPU: a copy into pinned host memory, read only
once it has landed, at most every 0.25 s per GPU (`POLL_SECONDS` in `modules/dsv41_rounding.py`),
so the warning comes one or two looks late, normally within a second of the first such block;
a count that no look has read when the interpreter exits (blocks kept by the process's last
rounding calls) is reported then, from an `atexit` handler that reads each device's count once.
`exllamav3.modules.dsv41_rounding.fallback_count()` returns the total over every device; it
waits for the GPUs, so it is for diagnostics, not for a serving loop. With
`EXL3_DSV41_NUMERICS_STRICT=1` such an operand raises `ValueError` instead (below). The E2M1
rounding of the `deepseek` compressed part never overflows: as in DeepSeek's kernel its E4M3
scale saturates at 448, so a 16-lane block whose peak exceeds 6 x 448 = 2,688 is clamped to
+-2,688; only a nonfinite operand keeps one of its blocks unrounded. On finite operands whose
rounded values fit, the default and the strict switch (below) store the same bits.

For maintainers: the fallback exists so that one extreme activation cannot abort a long
generation. The blocks it keeps hold the `precise` values, a valid FP16 state for every later
forward, for prefix reuse and for rewinds; only the rounding of those blocks is lost. The count
lives in `modules/dsv41_rounding.py` (one int64 per device, its pinned host copy and an event,
and the `atexit` handler above) for the life of the process, and the engine never resets it.
The default path must stay free of host reads, and the default and strict paths bitwise equal
on finite operands whose rounded values fit: `tests/test_dsv41_numerics_.py` checks both on the
CPU (the host reads on meta operands), `tests/test_dsv41_numerics_gpu_.py` on every GPU under
`torch.cuda.set_sync_debug_mode("error")`. If the warning appears, rerun with
`EXL3_DSV41_NUMERICS_STRICT=1` to stop at the first rounding call that meets such a value, with
its traceback.

Why the default is `deepseek:index`. On DeepSeek-V4.1-Flash (EXL3, 3.0 bpw), teacher-forced over
one long real text and scored at every position, the MXFP4 rounding of the index operands alone
gave the lowest error of the settings compared:

- negative log-likelihood of the real text, which needs no reference: lowest at 64K (0.249963
  against 0.250939 for `precise`) and 128K (0.256422 against 0.256955), tied on the first 364K
  predictions (0.294591 against 0.294598), and 0.1% higher over all 1,048,559 (0.321319 against
  0.320997);
- against two captures of vLLM's V4.1 path running the same checkpoint (FP8 KV, MXFP4 index): fewer
  confident flips than `precise` at every length (per 100K reference-confident positions: 150
  against 161 at 64K, 180 against 189 at 128K, 212 against 228 at 364K, 253 against 270 at 1M;
  `vllm`, which adds FP8 KV, is within 1-2 of it at 64K, 128K and 364K) and the fewest certain
  misses of the numerics settings at every length (7.6 per 100K at 64K, against 8.4 for `precise`,
  10.7 for `vllm` and 19.1 for `deepseek`; 1.6 against 2.9 for `precise` at 1M);
- against a released-precision baseline (the original checkpoint served by a hosted provider,
  its top-5 log-probabilities at 65,535 positions of the first 64K tokens of the same text): the
  top-1 token agreed at 98.07% of positions for `deepseek:index`, 98.04% for `precise`, 98.03%
  for `vllm` and 97.99% for `deepseek` (98.00-98.06% for the vLLM captures), against 98.89%
  between two identical requests to the provider; the top-5 KL divergence was 0.0176, 0.0176,
  0.0171 and 0.0179 against the provider's own 0.0059. The settings are within about 0.08
  points of each other there; the gap to the provider is the 3.0-bpw quantization, which none
  of them removes.

The rounding matters at near-ties of the top-512 cut: at one position of the 1M run the two
reference captures give the actual next token 93% and 96%, `deepseek:index` 97% and `precise`
0.06%, and exact replays trace the difference to the layer-2 index selections of the rows
256-512 tokens earlier.

Rounding the KV as well (`deepseek`, `vllm`, or only their `window`/`compressed` parts) raised
the negative log-likelihood at every length measured (64K: 0.251180 `vllm`, 0.251389
`vllm:window,compressed`, 0.251634 `deepseek:window,compressed`, 0.252312 `deepseek`). The
reference in these comparisons is itself an FP8/MXFP4 engine on the same 3-bit weights; the
ranking might differ against an FP8 reference captured with DeepSeek's official numerics path
on B200/B300 GPUs; no reference captured with that path was available.

Interactions and refusals:

- Quantized cache (`-cq`): rounding compressed entries is defined on the FP16 pool, while a
  quantized V4.1 pool stores the entries' NoPE part in the cache-quant format. A setting with the
  `compressed` part is therefore refused when a Cache with quantized pools is built
  (`Cache(model, layer_type = CacheLayer_quant, ...)`), and assigning such a setting is refused
  while a quantized Cache built for a model of this config exists (call
  `cache.detach_from_model()` and drop the Cache first). `index` and `window` combine with `-cq`
  freely. Both are configuration-time errors: neither refusal happens inside a forward pass (the
  operand checks above do). The error suggests the setting without its `compressed` part, or
  the default when no other part is left.
- A pool replica (the split of an explicit placement, placement.md, "DeepSeek-V4.1") copies the
  stored, already rounded entries bitwise.
- The setting applies to the cached (generation) path and to the stateless path alike.

Relation to other settings. `-cq` is the one setting that overlaps: it stores the compressed
pool's NoPE lanes (`pool_c`) packed in exllamav3's cache-quant format, and those are exactly the
values the `compressed` part rounds, so the two would define the same stored values in two
different ways; the combination is refused, both ways, as above. `-cq 8` is exllamav3's own
storage format, neither vLLM's `fp8_ds_mla` nor DeepSeek's FP8. Under `-cq` the index keys
(`pool_idx`), the compressed RoPE lanes (`pool_r`) and the sliding-window ring stay FP16, so the
`index` and `window` parts are independent of it. No other setting rounds these operands: the
`EXL3_DSA_*`, `EXL3_BC_DSA` and `EXL3_QC_*` variables choose kernels, graphs or cache staging, or
add bounds checks, and leave the values alone.

Cache consistency: entries already in a Cache keep the rounding they were stored with. Change
the setting between sequences, and do not let the generator reuse cached prefixes that were
stored under another setting (use a fresh Cache and Generator after switching).

Performance: the roundings are elementwise torch operations. With `deepseek:index`, per forward:
the index query of each index source that selects (8 layers on V4.1-Flash, once more than 512
compressed entries are visible) and the index keys each kv source stores (4 layers): under the
default about 10-11 rounding calls per decoded token (the 8 index queries, and the index keys
that layer 20 stores every token and layers 2, 8 and 14 every second token). The `window` part
adds one call per layer and decoded token under `deepseek` (40 on V4.1-Flash) and two under
`vllm` (NoPE and RoPE lanes, 80); the `compressed` part two per stored entry. No call waits for
the GPU: the overflow check adds a few elementwise kernels and a device-side add to the count,
and, until the warning has been printed, at most every 0.25 s per GPU an event query and one
8-byte copy of the count. Under `EXL3_DSV41_NUMERICS_STRICT=1` every call reads its checks back
to the host once, which waits for the GPU. Measured on a decode-sized index query (32 heads x 128
lanes, FP16), on the RTX PRO 6000 and on the CMP 170HX, in two runs: with the GPU idle a call
costs 0.14-0.20 ms of host time under either policy; with GPU work queued ahead of it, a default
call returns in 0.23-0.29 ms, while a strict call waits for the queue (2.8-2.9 ms on the RTX PRO
6000 and 8.0 ms on the CMP 170HX in that test). The effect on decode speed has not been measured
on an otherwise idle host.
Prefill of 64K tokens (single runs, measured while every call still read its checks back to the
host): under the stable-arithmetic profile, in the same setup, 796 tok/s for `deepseek:index`
against 798 for `precise` (and 792 for `vllm`, 794 for `deepseek`, which also round the window
KV of all 40 layers, against 793 for `deepseek:index` in their setup); in the serving setup
(routed experts in system RAM, ordinary arithmetic) 1007 against 1040, about 3% slower.

Memory: nothing persistent but the count (8 bytes per device, and 8 pinned bytes on the host
per GPU). A rounding call works on FP32 copies of the operand's blocks, one slice of 65,536
blocks at a time, and rounds each slice in place; under `EXL3_DSV41_NUMERICS_STRICT=1` the
slices go into one buffer of the operand's size instead, copied into the operand once every
slice has passed its checks. Measured on the CPU, the index query of a 4096-token prefill chunk
(32 MiB in FP16) needs about 90 MiB of transient memory under MXFP4 (about 120 MiB under the
strict switch), one of a 2048-token chunk about 90 MiB (about 100), and a decode step's under
1 MiB; the autosplit's measuring forward rounds a chunk as well, so it sees this.

Determinism: the kernels are deterministic, and each rounds, or keeps, blocks of 16, 32 or 64
lanes inside one entry or head (single lanes under BF16), so the result does not depend on chunk
size or batch composition. Different settings give different model outputs by design.

When to use which: keep the default. `precise` removes all rounding (for comparisons with FP16
arithmetic or with engines that do not round). The full `deepseek` or `vllm` contracts, and the
part lists, reproduce another engine's attention numerics for experiments and A/B comparisons.

```sh
EXL3_DSV41_NUMERICS=precise python eval/ppl.py -m /path/to/DeepSeek-V4.1-Flash-exl3
```

```python
config = Config.from_directory(model_dir)          # EXL3_DSV41_NUMERICS, if set, is read here
config.dsv41_numerics = "vllm:index,window"         # validated now
model = Model.from_config(config)
...
for setting in ("precise", "deepseek:index"):       # one loaded model, two settings
    config.dsv41_numerics = setting
    ...                                             # run each setting on a fresh Cache
```

### `EXL3_DSV41_NUMERICS_STRICT` (default: `0`)

For debugging the attention numerics (`EXL3_DSV41_NUMERICS`, above). By default a block whose
rounding would store a value FP16 cannot hold, or whose operand is not finite, is stored
unrounded, counted and reported once. With this switch on (any value other than `0` or empty),
the rounding raises `ValueError` inside the forward pass instead, with the operand unchanged:
`dsv41 numerics: nonfinite operand`, or `dsv41 numerics: rounded value overflows the operand
dtype`. The error stops the forward, and the generator's batch, at the call that met the value,
which is what the switch is for: finding where such a value comes from. Under a two-part contract
(`vllm` window or compressed entries, `deepseek` compressed entries) the first half may already
be rounded when the second is refused; every caller rounds a copy that it drops on the error, and
the cached path refuses before it writes the pool. The switch changes neither `-cq` refusal
(Interactions and refusals, above): those are configuration-time errors under either policy.

Cost: every rounding call reads its checks back to the host once, which waits for the GPU (under
`deepseek:index` about 10-11 times per decoded token, 40 more with the `deepseek` window part, 80
with `vllm`'s), and an operand of more than 65,536 blocks is rounded into a buffer of its own
size (see Memory above). On finite operands whose rounded values fit, both modes give the same
bits.

Read once, at the first rounding call of the process; set it before loading the model. Tests
switch it with `exllamav3.modules.dsv41_rounding.set_strict()`, which returns the previous
setting.

```sh
EXL3_DSV41_NUMERICS_STRICT=1 python eval/ppl.py -m /path/to/DeepSeek-V4.1-Flash-exl3
```

### `EXL3_DSV41_ENGRAM_PREFETCH` (default: `1`)

DeepSeek-V4.1's engram layers (layers 1 and 14 on Flash) add a gated embedding of each position's
n-gram hashes to the hidden streams. Their tables are about 94 GiB per layer and stay on disk:
each forward hashes its token ids, removes duplicate rows, reads the rows it needs from the
checkpoint shards (the extension's `ngram_gather_cpu` thread pool) and dequantizes them on the
GPU. With this switch on, for a prefill chunk of at least 256 positions (batch size times
tokens) the hash, the deduplication and the disk reads run on one worker thread per model that
starts before block 0, so the reads of layer 1 overlap block 0 and those of layer 14 overlap
blocks 0-13. The forward uses the staged rows only when their token history equals its own and
stages inline otherwise, so a stale prefetch costs time, never correctness. Decode-sized chunks,
the autosplit's measuring pass and forwards with `EXL3_DSV41_ABLATE=engram` are never prefetched.
`0` stages every chunk inline in the engram layer's own forward.

Results are bitwise identical either way. Memory, with the switch on or off (inline staging uses the
same sets): each engram layer on a GPU keeps at most two page-locked staging sets (one whose uploads
may still be in flight, one being staged), each sized for the largest chunk seen, 272 bytes per
hashed row (24 rows per token): 12.8 MiB requested at 2048 tokens and at most 25.5 MiB (4096
tokens). PyTorch's host allocator rounds each page-locked allocation up to a power of two, so a full
set holds about 34 MiB, and the blocks a set gives up when it grows stay cached, still page-locked.
Longer chunks stage through a transient pageable buffer instead. Read once per process, when the
first DeepSeek-V4.1 model object is built (`Model.from_config`); set it before that. Its effect on
prefill throughput has not been measured in isolation; it is an A/B switch.

### `EXL3_DSV41_PIPELINE` (default: `0`), `EXL3_DSV41_PIPELINE_CHUNK` (default: `4096`)

Two-stage pipelined prefill for DeepSeek-V4.1 split across two GPUs. In a plain layer-split
prefill the two GPUs take turns: the host issues a chunk's first-GPU layers, then its second-GPU
layers, and inside each half it waits at sync points (block-table uploads, MoE row counts, the
CPU-MoE stream plans), so it never queues the next chunk's first half while the second GPU is
still busy. With `EXL3_DSV41_PIPELINE=1`, a prefill call longer than one sub-chunk of
`EXL3_DSV41_PIPELINE_CHUNK` tokens runs as consecutive sub-chunks (the last may be shorter), and
the calling thread runs the first-GPU layers of sub-chunk k+1 while a worker thread runs the
second-GPU layers of sub-chunk k:

```
calling thread (first GPU):    S1(0)  S1(1)  S1(2)  ...
worker thread (second GPU):           S2(0)  S2(1)  S2(2)  ...
```

Stage 2 of a sub-chunk waits, on the GPU, for an event its stage 1 recorded, so it reads the first
half's products (the residual streams, the carried hyper-connection pre-mix, and across the split
of an explicit placement the pool replica rows and the top-k selections that cross it) only once
they are complete; tensors it reads from the first GPU are recorded on its side stream, so the
caching allocator cannot hand them out again before the read is done. Stage 2 is queued on the
caller's current stream of the second GPU, so work the caller queues after `prefill()` returns
(decode, cache reads) is ordered after it without a device-wide synchronization. On the first GPU,
the side stream may also read persistent Cache rows (the pool replica gathers, across the split of
an explicit placement), which the allocator records do not cover; so when the call returns, after a
failure too, the caller's current stream of the first GPU waits for the side stream, and first-GPU
work queued afterwards (a page-table defragmentation, for example) cannot overtake those reads. The
job state's position is advanced after each stage 1; stage 2 runs on a snapshot of the sub-chunk's
own start position.

Values: `EXL3_DSV41_PIPELINE`: `0` off (default), any other value on. `EXL3_DSV41_PIPELINE_CHUNK`:
the sub-chunk in tokens, a positive integer; it is read (and validated) only when the pipeline is
on, and a value that is not a positive integer then raises a `ValueError` whenever the architecture
registry is imported (every `Config.from_directory` until the value is fixed). Both are read once at
import; set them before loading the first model. For a same-process A/B, set `PIPELINE` and
`SUB_CHUNK` in `exllamav3.architecture.dsv41.pipeline` before loading the model, as for
`MoeCpuTuning` above; `PIPELINE` on with `SUB_CHUNK` unset (the variable was off at import) is
refused with a `ValueError` at load.

Which calls are pipelined: `Model.prefill` calls (the generator's prompt ingestion) of a single
sequence longer than one sub-chunk, with a job state (`recurrent_states`, one), `cache_seqlens`
and a `block_table`, and none of these params set: `last_tokens_only`, `export_state_layers`,
`capture`, `quant_preserve`, `autosplit_measure`, `indexed_embeddings` (a non-empty list:
multimodal input), `indexed_embeddings_required`, `inv_freq`, `position_ids`, `positions`,
`past_len`, `position`, `batch_shape`, `recurrent_history`, `dsv41_engram_lookback`. Tensors
count as set whatever their contents; other values by their truth value, so the generator's
empty embeddings list of a text-only job does not disqualify it. `Model.forward` is never
pipelined. Everything else takes the plain path unchanged.

Which loads are eligible (checked from the loaded modules, and re-checked after every
load/unload):

- a layer split (not `-tp`, which V4.1 refuses anyway) over exactly two CUDA devices, every
  module past the first second-GPU module on the second GPU, at least the first decoder layer
  before the split, and the split before the last layer that writes the cache (a split that
  leaves only the final norm and head on the second GPU gains nothing and is not pipelined).
  Every V4.1 layer owns a sliding-window ring whose shift both halves must agree on, so a first
  half without a layer cannot be pipelined; an autosplit without a placement whose first GPU
  cannot hold layer 0 leaves only the stream expansion there (the embedding is in system RAM);
- no layer before the split with experts in system RAM: both stages would feed the same CPU-MoE
  worker, which is not thread-safe. `-mcl` offloads the first N MoE layers and `-mcs` a share of
  every layer, so either one makes a split load ineligible. An explicit placement can keep
  RAM-held experts past the split only, e.g. `0-13=cuda:0; 14-22=cuda:1 experts=stream;
  23-39=cuda:1`, which stays eligible;
- no repeated layer instances (`layer_map`).

With the switch on, every load prints one line: whether it is pipelinable and, if not, why
(`!! DSV41 pipelined prefill: this load is not pipelinable (<reason>) ...`).

Refusals: at load, `EXL3_DSV41_PIPELINE_CHUNK` larger than the load's `max_chunk_size` raises a
`ValueError` naming both, because the autosplit sizes each device's prefill workspace for
`max_chunk_size` rows. `-chunk_size` in `model_init` defaults to 4096, but `Model.load`'s own
`max_chunk_size` defaults to 2048, below the default sub-chunk: with the switch on, every
DeepSeek-V4.1 load through the Python API must pass `max_chunk_size` of at least
`EXL3_DSV41_PIPELINE_CHUNK`, single-GPU loads included (the check runs before the layout is
known). A job state whose position disagrees with `cache_seqlens` is a caller error, refused
with a `ValueError` before any layer runs. If a pipelined prefill fails partway (an exception
in either stage, including out-of-memory), the job's layers may have written different amounts
of its state (its sliding-window rings, compressor carry and engram ring, and its pages); the
job state is then marked failed, and any later forward with it raises a `RuntimeError`. Those
writes stay within that job's slot and pages, as when a plain forward fails halfway, so the
Cache and the other jobs stay usable: the generator ends the failed job with an error result
(its consumer gets the exception), frees its pages and carries on with the other jobs, and a
direct caller starts the sequence again with a new job state.

Generator: it hands the model one prefill call per `max_chunk_size` tokens of the generator
(default 2048, `-gcs` in `examples/chat.py`), so the pipeline only engages with a generator
`max_chunk_size` above the sub-chunk, e.g. 16384 with 4096-token sub-chunks (four sub-chunks
per call). The generator's `max_chunk_size` may exceed the load's only because a pipelined call
never runs more than one sub-chunk of rows at once; a call that takes the plain path runs all
its rows together. So do not combine a generator chunk larger than the load's with anything
that keeps long calls on the plain path: an MTP head or draft model (their prefill passes
`last_tokens_only` or export params, or runs `forward`), multimodal input, or an ineligible
load (the line printed at load says which it is). A larger generator `max_chunk_size` has two
more costs: each prefill round interrupts the other active jobs for longer (their decode waits
for the whole call), and the recurrent checkpoints taken while a prompt is ingested become
coarser, since they fall only at call boundaries (the one at the end of the prompt is kept).
The generator path has not been benchmarked.

Memory: while the second GPU works on sub-chunk k, the first GPU already holds sub-chunk k+1's
working set, and sub-chunk k's first-GPU products stay allocated until both the next first half
and this second half have finished (the calling thread and the worker both hold them): its
residual streams, 80 KiB per token on DeepSeek-V4.1-Flash (4 streams x 5120 x FP32), 320 MiB at
4096 tokens, and its params, which hold the top-k selections of the index sources before the
split (2 KiB per row each), the candidate blocks when the candidate source (layer 20) is before
it (8 KiB per row), the rope tables and block-table copies: with the split at layer 22 and 4096
tokens, about 64 MiB more. Stage 2's own first-GPU buffers (the pool-replica staging of up to
8 MiB across the split of an explicit placement, index-select outputs) are allocated on its side
stream and come from that stream's own
allocator pool, which the default stream does not reuse. The autosplit reserves none of this;
leave that much headroom on the first GPU (its `-gs` budget or `EXL3_AUTOSPLIT_MARGIN_MB`). One
extra CUDA stream on the first GPU, and one worker thread per sub-chunk.

Performance: measured on DeepSeek-V4.1-Flash split at layer 12 across two GPUs, with the routed
experts of layers 12-22 held in system RAM and streamed to the second GPU (past the split, so
eligible: the explicit placement `0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1`; `-mcl`
and `-mcs` cannot place experts that way), 4096-token sub-chunks, 65,024 prompt tokens in four
direct `model.prefill` calls of up to 16,384 tokens followed by a
512-token continuation: 82.4 s plain and 67.2 s pipelined (1.23x) with the split crossing as in
this code (FP32 residual streams) and every GPU arithmetic path fixed to be independent of the
row count; 54.4 s and 40.6 s (1.34x) in a build with other options on as well, among them
BF16 rounding of the crossing. Single unpublished runs that include continuation scoring and
overlapped other host work, not a controlled prefill benchmark. The gain depends on how evenly
the layers divide the work between the GPUs and on how much of each half the host spends
waiting.

Determinism: each sub-chunk runs the same layers with the same inputs as a plain prefill in
sub-chunk-sized calls, only overlapped; with arithmetic independent of the row count
(`EXL3_STABLE_ARITHMETIC=1`), the six pipelined/plain continuation pairs over that 65K-token prefix
were bitwise equal. Against a
plain prefill of the whole call, results differ the way any change of chunk size does.

When to use it: long prompts on a two-GPU split whose first half holds no CPU-offloaded
experts, with direct `model.prefill` calls or a generator configured as above. With the variable
unset nothing changes: the plain path runs, at the cost of one failed-state check per forward.

The examples load every routed expert into VRAM, as the eligibility rules above require of the
first half, and change device at the free cut 14 (with a Cache attached the change must fall at a
free cut; "Loading across GPUs", above): at 3.0 bpw DeepSeek-V4.1-Flash's layers 0-13 take about 68
GiB and layers 14-39 about 127 GiB, so on two 96 GiB-class cards this layout does not fit. There
the second half has to keep part of its experts in system RAM without the first half doing so (the
layout measured above streams the experts of layers 12-22 from RAM to the second GPU), or the
checkpoint has to be smaller.

```sh
EXL3_DSV41_PIPELINE=1 EXL3_DSV41_PIPELINE_CHUNK=4096 \
    python examples/chat.py -m /path/to/DeepSeek-V4.1-Flash-exl3 -gs 70,130 -chunk_size 4096 -gcs 16384 ...
```

```python
import os
os.environ["EXL3_DSV41_PIPELINE"] = "1"         # before importing exllamav3
from exllamav3 import Config, Model, Cache, Generator

config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 131072)
model.load(max_chunk_size = 4096, use_per_device = [70, 130])   # layers 0-13 on the first GPU
generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = 16384)
```

### `EXL3_DSV41_ABLATE` (default: unset; tests and validation only)

A comma-separated list of deliberate errors in DeepSeek-V4.1's function. Each token makes the
model compute a known wrong function, so that a test or a validation run can show that it
detects the error it exists to detect: with the token set, the check must fail (a negative
control). Never set it for inference.

| Token | What it breaks |
|---|---|
| `engram` | The engram layers do not run. Their n-gram context is still committed, so the carried ids stay current if the ablation is turned off mid-sequence. |
| `engram_nocarry` | Each forward hashes its first n-grams without the ids carried from the previous forward, as at position 0: a state-carry bug that only a chunked run can show. |
| `v4mix` | Every sublayer collapses its input with its own pre-mix, as V4 does, instead of the pre-mix carried from the previous sublayer. |
| `dense_consumers` | Compressed layers that are not index sources attend over their whole pool instead of their index source's top-k selection. |
| `rope_consecutive` | Pool entries and index keys are rotated at `first_entry * m + j` instead of `(first_entry + j) * m`: one position apart, wrong at compression rate 2. |
| `no_candidates` | The candidate stage is off: the candidate source publishes no blocks and the index sources above it score every visible entry. Changes nothing until a query sees more than 16,384 compressed entries (2,048 blocks of 8). |

Syntax: tokens separated by commas; whitespace and empty entries are ignored; tokens are
case-sensitive. An unknown token is a `ValueError` at the first forward that consults the
variable (and at every later one), never silently ignored.

When it is read: once per process, at the first forward that checks for an ablation (the
first DeepSeek-V4.1 forward), and parsed then. Changing the variable afterwards has no effect;
set it before starting the process. Checking an ablation afterwards costs a set lookup. When
a token becomes active, one line is printed: ` !! EXL3_DSV41_ABLATE=<token>: deliberately NOT
DeepSeek-V4.1's function, for tests and validation only`, once per token per process. No
memory cost. With the variable unset nothing changes.

The V4.1 tests use the tokens as their negative controls (`test_dsv41_select_`,
`test_dsv41_attention_`, `test_dsv41_cached_`, `test_dsv41_cached_gpu_`,
`test_dsv41_block_hook_`, `test_dsv41_engram_state_`). To switch an ablation on and off around
single calls of one loaded model, they use the test-only setter
`exllamav3.modules.dsv41_ablation.set_ablations(value)`, which takes the variable's syntax
(`""` for none), replaces the parsed set, and returns the previous setting (or `None` when the
variable had not been read yet, which makes the next check read it again), so that
`set_ablations(previous)` restores it; `tests/dsv41_ref`'s `ablate()` context manager wraps it.
Inference code has no reason to call it.

A manual check that a perplexity run notices a missing engram:

```sh
python eval/ppl.py -m /path/to/DeepSeek-V4.1-Flash-exl3
EXL3_DSV41_ABLATE=engram python eval/ppl.py -m /path/to/DeepSeek-V4.1-Flash-exl3
```

## Debug

### `EXL3_NGRAM_GATHER_PROF` (default: unset)

Windows only: print per-gather statistics from the streamed n-gram table path (unique rows,
coalesced runs, reads completed synchronously vs left pending, span tasks drained by pool
workers vs the calling thread). Activation check for the overlapped-`ReadFile` gather when
validating a streamed-table model on Windows.

### `EXLLAMA_DEBUGLOG_<CATEGORY>` (default: unset)

Enables timestamped debug logging for the given category when the corresponding variable is
present in the environment. Categories are defined at the call sites (see
`exllamav3/util/debug.py`); mostly hooks for development.

## Build (JIT extension)

These only matter when the C++/CUDA extension is compiled at import time rather than installed
prebuilt.

### `CUDAHOSTCXX` (default: unset)

Host compiler passed to nvcc (`-ccbin`), for systems whose default compiler is too new for the
installed CUDA toolkit.

### `TORCH_CUDA_ARCH_LIST` (default: auto)

Standard PyTorch variable; overrides the compute architectures the extension is built for. When
unset, ExLlamaV3 derives the list from the GPUs present in the system.

## `EXL3_DSA_DEBUG_BOUNDS`

When set to `1`, the JIT DSA attention/indexer kernels compile with device-side bounds
asserts on every block-table page read and gathered pool index, and the DSA module range-
checks block-table contents on the host each forward. A violation traps at the faulting
kernel with the kernel name, source line and bad index instead of corrupting memory or
faulting asynchronously downstream. Debug tool for paged-pool issues; significant JIT
overhead (forces Triton debug mode globally), leave unset in production. AOT/BC graph
kernels are unaffected (compiled with asserts off).

## Quantization

### `EXL3_QT_OPTIMIZED`

Forces the quantizer's dense trellis specializations (`quantize_tiles_optimized.cuh`) on (`1`) or
off (`0`) regardless of the per-architecture dispatch (on for every K on sm_120; per K and codebook
on Ada and Ampere where measured faster; the original kernels elsewhere). For experiments only: the
choice also sizes the quantizer's scratch buffers.
