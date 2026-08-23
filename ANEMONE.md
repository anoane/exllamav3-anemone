# ANEMONE — per-expert mixed-bitrate quantization for exllamav3

This fork adds one capability to exllamav3 and a set of optimizations built around it:

> **Every routed expert in a mixture-of-experts model can carry its own bit-width.**

Upstream exllamav3 assigns one bit-width per layer and asserts uniformity at load time.
That is the right default, but it leaves capacity on the table in a model like
DeepSeek-V4-Flash, where routing concentrates traffic on a minority of the 11,008 experts:
the hot experts are under-served and the cold ones over-served at the same time. This fork
lets the allocator spend bits where the traffic actually is.

Everything here is **opt-in**. With no ANEMONE flag set, the engine behaves exactly as
upstream; the flags below are the entire surface.

---

## 1. Serving flags

| flag | what it does | speed | quality cost |
|---|---|---|---|
| `--no-rlp` | skip the recurrent-last-page split during prefill (one fewer forward per prompt) | −34% prefill @512, −20% @2k, nothing past ~16k | **none measurable** (see §6) |
| `--hc-bf16` | store hyperconnection residual streams in bf16 instead of fp32 | −645 ms @128k, −1.4 s @260k prefill | +0.004% perplexity |
| `--prefill-graphs` | capture/replay whole prefill chunks as CUDA graphs (≤128k) | small, depth-dependent | none |
| `--prefill-graphs-max-tokens N` | cap the chunk size that graph capture will attempt | — | none |
| `--prefill-stage` | stage the next chunk's embedding on a side stream | measured *slower* (+30 ms @8k) and +254 MiB — off | none |
| `--gu-merge` | merge the gate and up expert GEMMs into one launch (uniform-K layers only) | small | none |
| `--fp4-prefill` | run routed-expert prefill GEMMs through Blackwell's block-scaled FP4 (e2m1) path | ~1.45× prefill; ≤128k only | +1.7% perplexity on a trellis pack; ~0 on a snap-aware pack |
| `--fp4-dense` | also route dense/attention prefill through FP4 | ~13 ms | **+5.2% on top of `--fp4-prefill` — not recommended** |
| `--fp4-hook PATH` | override the FP4 compute module (defaults to the bundled `exllamav3/anemone_fp4/fp4_hook.py`) | — | — |

Quality costs are measured on a 364,544-token held-out corpus — text the model did not
calibrate on, with the calibrated extent computed per domain rather than assumed. They are
not estimates. `--no-rlp` is the one flag they cannot price: it does not execute in a
batched forward at all, so §6 explains how it was bounded instead.

The FP4 paths JIT-compile a second extension for `sm_120a` on first use; they are Blackwell
only, and the CUDA source ships in `exllamav3/anemone_fp4/`.

## 2. Conversion flags

| flag | what it does |
|---|---|
| `--qgroup expert` | quantize routed experts as independent groups, so each may take its own K |
| `--strategy-json PATH` | a `{tensor_key: K}` map that forces per-tensor bit-widths — this is how a hybrid pack is built |
| `--score-json PATH` | per-tensor scores used as an allocator tiebreak |
| `--snap-e2m1` | choose trellis codes whose decoded values land on the e2m1 grid, making a later FP4 cast lossless (K≤4 only) |
| `--calib-dir PATH` | calibration corpus directory containing `manifest.json`; rows are apportioned across its domains by `weight_pct` |
| `--calib-all` | consume the whole corpus instead: every full window of every domain file, ignoring weight caps and `--cal_rows` |
| `--cal-bf16`, `--cal-tier`, `--cal-vram-gb`, `--cal-ram-gb`, `--cal-spill-dir` | tiered calibration state (VRAM → RAM → disk), for models whose calibration state exceeds host RAM |

## 3. What changed, by subsystem

**Per-expert mixed-K MoE** — the core capability. The `exl3_moe` kernels take per-expert
`const int*` K tables instead of scalars and index them by `expert_idx`; a `K_uniform`
scalar keeps the templated fast path when a layer happens to be uniform, and `K=0` selects
runtime dispatch. `exl3_mgemm` gains a trailing `K_list` (`exl3_mgemm_pk`). Out-of-range K
traps loudly rather than silently producing garbage. Host side: `MultiLinear` gains
`ptrs_K` and an opt-in `allow_mixed_K`; the C++ block-sparse MLP derives per-expert K lists
and declines single-expert graph replay for mixed-K layers, where a captured graph would
otherwise bake one expert's K into all of them. CPU expert offload declines mixed-K layers
for the same reason.

**Codegen** — the runtime-K path initially cost 1.9× because `__launch_bounds__` without a
`minBlocks` hint sent ptxas chasing occupancy at 64 registers, and because a multi-arm
switch spilled at full register budget. Pinning `minBlocks=1` and giving each arm a
`__noinline__` ABI function brought the runtime-K path to parity with the templated one
*in the isolated GEMM benchmark*.

End to end it is not free: a mixed-K pack decodes about 6% slower than a uniform-K pack of
the same bitrate (83.5 vs 89.0 tok/s in a controlled comparison). Roughly two thirds of
that is the runtime-K kernel — its extra register pressure pushes the autotuner onto a
lower-occupancy tile shape — and the rest is dispatch. Four separate attempts to recover it
measured negative; a fifth recovered 58% of the kernel share, but only for packs built with
two K values rather than three.

**What that buys, measured rather than assumed.** On the reference pack, per-expert
allocation is worth roughly **0.3% perplexity** against a uniform-K pack of the same class,
and the best per-layer-uniform arrangement at an identical bit budget is projected to cost
about 0.15% while returning that ~6% of decode. A convert-time distortion proxy makes the
gap look far larger — around 90% — but it over-predicts what reaches the output by about two
orders of magnitude, so do not size the decision on it.

The honest case for this capability is therefore not a large quality margin. It is
*reachability*: per-expert allocation is how a model gets to a bitrate that fits the
hardware you have at the context you need. Where a uniform-K pack already fits, it is
usually the better choice, and it decodes faster.

**FP4 prefill** — routed-expert prefill GEMMs can run on Blackwell's native block-scaled
FP4 MMA. The weights are trellis-decoded, cast to e2m1 and fed to `mma.sync...kind::mxf4`.
This is a compute-format change only: storage stays trellis, and decode is untouched.

**Snap-aware quantization** — if you intend to serve with `--fp4-prefill`, the conversion
can choose trellis codes whose decoded values already lie on the e2m1 grid, so the runtime
cast is bit-exact and the FP4 quality cost disappears. LDLQ compensates the constraint.
Scoped to K≤4, so a 6-bit head — which no runtime path casts to FP4 — keeps its
unconstrained fit.

**bf16 hyperconnection streams** — the mHC residual streams are fp32 by default; storing
them in bf16 halves that traffic. The kernels are templated on stream dtype (the fp32
instantiations are byte-identical to before) and `rms_norm` gained bf16 dispatch rows.

**Host-side prefill** — a `torch.bincount` in the MoE dispatch was synchronizing on the
host every layer; `scatter_add_` is the sync-free equivalent. The recurrent-last-page split
costs a whole extra forward pass per prompt and is skippable at no measurable cost. Optional CUDA
graph capture and lookahead chunk staging are available for chunked prefill.

**Tiered calibration** — conversion of a ~300B MoE at 2000 calibration rows needs more
calibration state than a 62 GB host has; the state can now tier VRAM → RAM → disk spill.

## 4. Building

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
pip install -e . --no-build-isolation
```

Build for plain `12.0` on Blackwell, **not** `12.0a`: a single-arch `120a` build has been
observed to break the DSA decode graph path. (The optional FP4 prefill kernel is a separate
extension that does target `sm_120a`, isolated from the main build.)

## 5. Relationship to upstream

This is a fork of `turboderp-org/exllamav3`. ANEMONE commits sit on top of upstream `dev`,
each scoped to one subsystem so it can be reviewed, reverted or upstreamed independently.
One fix in here is not ANEMONE-specific and is worth upstreaming on its own: safetensors
shard globbing was unsorted, which makes load order filesystem-dependent.

**Where this fork meets upstream's own allocation work.** Upstream now ships an experimental
pipeline that measures per-tensor quantization sensitivity by noise injection and solves for
a bit allocation at a size budget, emitting a recipe that maps each tensor to a bitrate. That
recipe format can already express a per-expert allocation, because each routed expert is an
individual `Linear` with its own key — but upstream's runtime cannot load the result:
`MultiLinear` asserts that every expert in a layer shares one K. That assertion is exactly
what this fork lifts. The two mechanisms compose rather than compete, and the conversion path
here runs the ANEMONE override on top of whichever strategy the allocator or a recipe
produced, so a measured recipe can set the dense and attention tensors while the override
sets each expert.

Upstream's pipeline is documented as untested on sparse models, and this fork has not been
run against it.

## 6. What was verified, and how

Claims in this fork are measured rather than asserted. The load-bearing ones, and the
evidence behind each:

| claim | evidence |
|---|---|
| per-expert mixed-K kernels are numerically correct | three levels, because one number would overstate it: the dispatch **replays bitwise** (0.0 across repeated launches); runtime-K matches the templated path to **exactly one fp16 ulp**, deterministically; and against the per-expert reference GEMM it sits at the **pre-existing cross-launcher tiling floor** (~10 fp16 ulp, present upstream and unrelated to mixed K) with no per-K-class anomaly. Out-of-range K traps on device rather than degrading silently |
| the bf16 stream kernels do not disturb the fp32 path | the fp32 template instantiations are byte-identical to the originals; the fp32 timing guard was unchanged after the change |
| `--no-rlp` is free | bounded, not proven nil. It reschedules chunking inside the generator path, which the batched perplexity harness never enters — so a null there measures the instrument, not the flag. Through the generator-path harness on real text it sits below that instrument's 0.035% resolution. An earlier greedy-token digest gate appeared to prove it exact; that gate was withdrawn as blind (same digest for two packs with different weights, and its positive control failed to move it) |
| `--fp4-prefill` costs +1.7%, `--fp4-dense` +5.2%, `--hc-bf16` +0.004% | held-out perplexity over 364,544 tokens, one process per configuration, with the calibrated extent of every domain computed from the manifest and excluded |
| the rebase preserved behavior | the rebased engine reproduces the recorded pack numbers: perplexity 1.8938 vs 1.8938, prefill 507.6 ms @2k against 519 recorded and 2082.5 ms @8k against 2174, and 256k context still serves. (Both prefill figures are profile A, which is what was re-measured.) |
| an unflagged run equals upstream | applying the flag layer with no flags set writes no environment variables and touches no module state |

**On this branch's history.** The series was rebuilt from scratch onto current upstream
`dev` and grouped by subsystem, with each kernel landing before the host code that calls it,
so no commit in the middle is broken. An earlier arrangement captured straight from a working
tree had the kernel signature change landing before its Python callers, which would have
failed at runtime at every intermediate commit.

Two classes of measurement mistake are worth knowing about if you extend this work. First,
greedy-token comparison is not a valid equality test here: this stack is not run-to-run
deterministic even at temperature 0 with the optional paths off, so identical
configurations produce different tokens. Second, an A/B that repeats the same prompt in one process
measures almost nothing, because page-cache and checkpoint reuse mean the second run only
prefills the tail. Use perplexity, or tensor comparison against a measured noise floor, and
one process per configuration.

## 7. Debug and experimental switches (environment only)

Every user-facing option has a flag. A few knobs deliberately do not, because they exist
for development rather than deployment:

| variable | purpose |
|---|---|
| `EXL3_FORCE_K0=1` | force runtime-K dispatch even for uniform-K layers, to exercise that path in validation |
| `EXL3_IDX_STATIC=1` | capacity-stride statics for the eager indexer chain; measured slower (+26 ms @16k, +105 ms @32k) and kept only as a documented negative |
| `EXL3_P24_CHECK=1` | assert device-side scalars match their host values on every chunk |
| `EXL3_P27_CHECK=1` | assert staged chunk embeddings are bitwise equal to an eager recompute |
| `EXL3_PREFILL_GRAPHS_{DEBUG,STRICT,NOCAP,HOOK}` | graph capture diagnostics: log decisions, raise instead of falling back, skip capture, bypass the hook |

They are listed here so their existence is discoverable; none is needed to serve.
