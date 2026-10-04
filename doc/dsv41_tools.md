## DeepSeek-V4.1 checkpoint, layout and validation tools

Standalone scripts in `tools/` for working with a DeepSeek-V4.1 checkpoint: three checkpoint
and layout tools for before a load, and a validation harness that compares the loaded model
with a vLLM capture of the same prompts and with itself. Nothing in the engine imports them.

The checkpoint and layout tools never import the `exllamav3` package, torch or the compiled
extension: they read `config.json` and the safetensors headers with the standard library,
through the same bounded header reader the model uses for its engram tables
(`exllamav3/architecture/dsv41/checkpoint_header.py`), and load the few torch-free engine
modules they need by file path. They run on any host with Python 3.10 or newer, including one
without a GPU. The harness imports torch and `exllamav3` only for a GPU run; its `--dry-run`, its
scorer and the capture tool stay torch-free.

| Script | Answers | Reads |
|---|---|---|
| `tools/dsv41_keymap.py` | Does the checkpoint carry exactly the tensors the V4.1 graph claims? | config.json, shard headers |
| `tools/dsv41_topology_check.py` | Do the layers that carry compressors, indexers and engrams match the topology config.json declares? | config.json, shard headers |
| `tools/dsv41_fit.py` | How much GPU memory and host RAM does a layout need, and does it fit? | config.json, shard headers |
| `tools/dsv41_validate.py` | Does the loaded model reproduce vLLM's prompt logprobs, and do its cached, chunked and token-by-token paths agree with its stateless one? | the model, a vLLM capture |
| `tools/dsv41_rowprobe.py` | Does a row of a 1+N-row verify forward get the bits a one-row decode step gives it, and which operation differs first? | the model, a token sequence |
| `tools/dsv41_refcompare.py` | (library) The scorer and gates the harness uses | |
| `tools/dsv41_capture_ref.py` | Capture vLLM prompt logprobs to compare against | a vLLM server |

The checkpoint and layout tools exit with status 0 when the check passes (or the estimate fits),
1 when it fails and 2 on a usage error, so they can gate a script; the harness does the same for
its gates.

### `tools/dsv41_keymap.py`

    python tools/dsv41_keymap.py <checkpoint-dir>

Compares, in both directions and on every layer, the tensor names the V4.1 text graph asks for
with the names the shards carry. Names are compared as *stems*: the storage suffix
(`.trellis`, `.suh`, `.svh`, `.mul1`, `.mcg`, `.weight`, `.bias`, `.scale`,
`.weight_scale_inv`) is stripped, so a quantized and an unquantized checkpoint of the same
model have the same stems.

- **MISSING** lists stems the graph claims and the checkpoint lacks. A load would fail on the
  first of them, possibly minutes in.
- **UNCLAIMED** lists stems the checkpoint carries and no module claims: weights that would be
  silently left out of the forward.
- Prefixes the text model deliberately does not load are counted and reported, never failed:
  the MTP draft layers (`mtp.*`), the vision tower and aligner (`vision.*`, `aligner.*`) and
  the image marker embeddings.

What the graph claims is derived from config.json alone: per layer the attention projections
and norms, the attention sink, `o_groups` slices of `wo_a`, the six hyper-connection tensors,
the router, the shared expert and `n_routed_experts` routed experts; the compressor (plus its
gate on rate-2 layers) on every kv source, the indexer on every index source (plus its own K
projection and norm when the layer is also a kv source), and the engram projections and table
on every engram layer.

Output on a complete DeepSeek-V4.1-Flash checkpoint (3.0 bpw EXL3):

```
  checkpoint: 192,464 tensors, 48,644 distinct stems
  graph claims 47,206 stems | MISSING 0 | UNCLAIMED 0
  not loaded by the text-only port: aligner (2 stems), image_end (1 stems), image_newline (1 stems), image_start (1 stems), mtp (1239 stems), vision (194 stems)
```

### `tools/dsv41_topology_check.py`

    python tools/dsv41_topology_check.py <checkpoint-dir>

config.json declares V4.1's source topology (`kv_source_layer_ids`, `index_source_layer_ids`,
`compress_ratios`, `engram_layer_ids`); the checkpoint proves it, because compressors, indexers
and engrams are only materialised on the layers that own them. For every layer the tool checks:

- a compressor exactly on the kv sources, each with both `wkv` and `norm`, and a `wgate` exactly
  where the compression rate is 2;
- an indexer (`wq_b`, `weights_proj`) exactly on the index sources, and the indexer's own K
  projection and norm (`wk`, `k_norm`) exactly on index sources that are also kv sources (the
  others borrow K from their kv source);
- engram tensors exactly on the engram layers.

It prints the topology it found (kv sources, index sources and which of them own K, rate-1 and
rate-2 layers, engram layers) and at most ten failures.

### `tools/dsv41_fit.py`

    python tools/dsv41_fit.py --model <checkpoint-dir> --devices <GiB,...> [layout] [options]

A header-only estimate of what a layout needs on every CUDA device and in host RAM, compared
with the capacities given in `--devices`. **It is an estimate, not a bound**: it never loads
anything, and the transient terms are historical calibrations (listed below and in the
script's docstring). Before trusting a headroom of about a GiB or less, load the model once and
read `torch.cuda.memory_allocated`, then pass that snapshot back with
`--measured-resident-bytes`.

#### Describing the layout

The layout is described with the same settings the loader takes, and each option defaults to
the environment variable the loader reads. A shell that is about to load the model therefore
estimates the layout that load will build without repeating anything:

- **--placement *rules*** (default: `EXL3_PLACEMENT`): the explicit placement, in the grammar
  of `EXL3_PLACEMENT` / `--placement` (doc/placement.md and doc/env_vars.md): one rule per
  layer range, `<layers>=cuda:<n> [experts=vram|stream|cpu|split cpu=<k>]`, `*` for the
  remaining layers, `head=cuda:<n>` for the output head. Every layer's device and expert
  storage come from its rule. The tool refuses it together with either option below, with
  `EXL3_MOE_CPU_SPLIT_LAYERS` and with an `EXL3_MOE_CPU_MODE` other than `compute` (`-mcm`
  changes no memory, but the loader refuses it next to a placement), as the loader does. No
  placement is an empty value (or one with only separators and comments), as for the engine;
  the words the engine refuses in its place (`none`, `off`, ...) are refused here too. Needs a
  tree that has the explicit placement (`exllamav3/model/placement.py`).

- **--mcl *N*** (default: `EXL3_MOE_CPU_OFFLOAD`): as `-mcl`: the routed experts of the first
  *N* MoE layers, in load order, are kept in system RAM.

- **--mcs *K*** (default: `EXL3_MOE_CPU_SPLIT`): as `-mcs`: the last *K* routed experts of every
  MoE layer are kept in system RAM, capped at `E - 1` (the loader keeps at least one expert on
  the GPU). **--mcs-layers *L*** (default: `EXL3_MOE_CPU_SPLIT_LAYERS`) limits the split to the
  first *L* MoE layers; 0 means all.

`--mcl` and `--mcs` are mutually exclusive, as `-mcl` and `-mcs` are. Without a placement
every layer is estimated on `cuda:0` (with its experts in VRAM, or as `--mcl` / `--mcs` say),
which is what the autosplit does when the first device is large enough; a split made by the
budgets alone is not estimated (write it as a placement).

A placement's changes of device follow the engine's DeepSeek-V4.1 rule (doc/placement.md,
"DeepSeek-V4.1"): every one at a free cut (layers 1, 2, 8, 14 and 20 on Flash), except at most
one, the split, which may fall inside a kv group. A placement with two changes of device inside
kv groups is refused with the engine's message. The report lists the changes of device, the
split, its pool replica and the per-forward crossings (`changes of device 12 | split 12 |
replicas 12<-8 | top-k cross 8->12 8->13 | candidates cross none`).

`experts=cpu` and `experts=stream` keep the same bytes in system RAM and on the GPU, so the
estimate is the same for both; they differ in where the experts are computed.

#### Devices and budgets

- **-d / --devices *list*** (required): the capacity of every CUDA device the layout uses, in
  GiB, by CUDA index (`cuda:0` first), comma-separated; each entry is `<GiB>` or
  `<name>:<GiB>`, e.g. `64,96` or `first:64,second:95.59`. Use the physical capacity, not a
  `-gs` budget: the context, the transient reserve and the margin are added on top of the
  weights and compared with it.
- **--context-gib *list*** (default `0.5`): the CUDA context per device, `cuda:0` first; the last
  value repeats for the remaining devices. The context size depends on the GPU and driver;
  `nvidia-smi` after creating an empty context shows it.
- **--host-ram-gib *float*** (default: `MemTotal` from `/proc/meminfo`, none elsewhere):
  capacity of system RAM for the arena check. **--host-overhead-gib** (default 10) and
  **--host-reserve-gib** (default 8) are added to the arena and the embedding for the process,
  the CPU-MoE worker and headroom.

#### Workload

- **--max-seq-len *int*** (default 1048576): the Cache's `max_num_tokens`; a multiple of 256.
  Pools grow linearly with it.
- **--slots *int*** (default 1): recurrent state slots (the Cache's `max_batch_size`); the SWA
  rings and rate-2 compressor carries are per slot.
- **--cache-bits *int*** (default 0): the packed pool format of `-cq` (2 to 8 bits for the NoPE
  lanes); 0 is fp16.
- **--chunk *int*** (default 2048): the prefill chunk; activations and the head input scale with
  it.
- **--max-output-size *int*** (default 32), **--max-output-factor *int*** (default 1),
  **--output-dtype** `float16|bfloat16|float32` (default float16): the output geometry, as
  `Model.load(max_output_size=..., max_output_factor=...)`; the effective output rows are
  `min(max_output_size, chunk)`. Use `--output-dtype float32` with `EXL3_FP32_LOGITS=1`.
- **--validation-logit-rows *int*** (default 0): adds the three FP32 matrices the validation
  harness's logit collector keeps (tools/dsv41_validate.py `--logit-rows`); not part of
  ordinary serving.

#### Calibration terms

- **--transient-mib-per-token** (default 0.30), **--index-tile-gib** (0.5),
  **--load-overhead-gib** (`0.29,0.41,0.68`: resident only, with whole-layer RAM experts, with
  split layers; measured 294, 415 and 689 MiB, each rounded up),
  **--split-overhead-mib** (25 per extra split layer), **--margin-gib** (0.25,
  the autosplit's `EXL3_AUTOSPLIT_MARGIN_MB`): the historical slice measurements the reserve is
  made of. Each is listed item by item in the report; replace them with your own measurements
  when you have them.
- **--measured-resident-bytes *list***: one `torch.cuda.memory_allocated` value per device the
  layout uses (CUDA index order), taken after a real load of the same model, source, Cache,
  layout and options, and after transient cleanup. It replaces the header subtotal and the
  historical load overhead (it is not added to them). The tool does not verify where the
  numbers came from; the JSON output records them as unverified caller measurements.

#### Output

- The default report lists, per device: its layers split into resident, experts-in-RAM and
  split layers; the weights (engram projections and the head included), each pool (kv sources
  and replicas), the SWA rings and carries, the header/cache subtotal, the sequential phases of
  the head's device (body, head, output consumer; the maximum is taken, not the sum) and the
  check against the capacity with the estimated headroom. Then the host RAM check, the split
  with its replicas and crossings, and the warnings that apply.
- **--json** prints the same as one JSON object (the per-layer routes are left out).
- **--search** walks every split `k` of a two-device layout (`--devices` must list two) and,
  for each, the fewest layers past the split whose routed experts must be kept in RAM for
  everything to fit, taken from the split upwards. Each row is printed as the explicit placement
  that loads it, e.g. `0-11=cuda:0; 12-21=cuda:1 experts=cpu; 22-39=cuda:1`; replace `cpu` with
  `stream` to compute those experts on the GPU instead (same memory). `--measured-resident-bytes`
  is refused with `--search`, because a snapshot belongs to one layout.

#### Examples

A two-GPU layout with eleven layers' experts in RAM, computed by the CPU worker, at 1M tokens:

```sh
python tools/dsv41_fit.py -m /mnt/models/DeepSeek-V4.1-Flash-exl3 -d 64,96 \
    --placement "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1"
```

The same with the experts streamed to the second GPU and computed there, and the head placed
on the first GPU:

```sh
python tools/dsv41_fit.py -m /mnt/models/DeepSeek-V4.1-Flash-exl3 -d 64,96 \
    --placement "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1; head=cuda:0"
```

The change of device at 12, the last 128 routed experts of every layer in RAM (as `-mcs 128`,
which a placement writes per rule), 256K tokens and a 4-bit packed pool:

```sh
python tools/dsv41_fit.py -m /mnt/models/DeepSeek-V4.1-Flash-exl3 -d 64,96 \
    --placement "0-11=cuda:0 experts=split cpu=128; 12-39=cuda:1 experts=split cpu=128" \
    --max-seq-len 262144 --cache-bits 4
```

One GPU with the first 20 layers' experts in RAM (`-mcl 20`), estimated from the environment a
load would use:

```sh
EXL3_MOE_CPU_OFFLOAD=20 python tools/dsv41_fit.py -m /mnt/models/DeepSeek-V4.1-Flash-exl3 -d 96
```

The frontier of two-GPU layouts, for a 2048-row scoring forward:

```sh
python tools/dsv41_fit.py -m /mnt/models/DeepSeek-V4.1-Flash-exl3 -d 64,96 --search \
    --max-output-size 2048
```

Re-checking one layout against a measured load (two devices, bytes from
`torch.cuda.memory_allocated(i)` after `model.load()`):

```sh
python tools/dsv41_fit.py -m /mnt/models/DeepSeek-V4.1-Flash-exl3 -d 64,96 \
    --placement "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1" \
    --measured-resident-bytes 64204521472,85748002816
```

#### What is counted where

| Item | Where | Bytes |
|---|---|---|
| resident layer | its device | every `layers.{i}.*` tensor except the engram table and the stale FP8 `engram.wkv` copy |
| layer with its experts in RAM (`-mcl`, `experts=cpu`/`stream`) | its device | everything but the routed experts, plus the routed experts' `suh`/`svh` |
| | arena (host) | the routed experts |
| split layer (`-mcs`, `experts=split cpu=k`) | its device | experts `[0, E-k)`, plus the tail's `suh`/`svh` |
| | arena (host) | the last `k` experts |
| engram | the layer's device | the `wkv`, `q`, `k` projections (the table stays in the files, gathered by row) |
| head and final norm | the last layer's device, or `head=` | as stored |
| embedding | host RAM | as stored |
| pools | each kv source's device; replicas on their owner | `(D_c + D_r + D_i)` fp16 per entry, `D_i` only where index K is owned or carried |
| rings | every layer's device | `ceil((window + 512) / 256) * 256` rows of `head_dim` fp16 per slot, plus a `(256 + 2) x 2 * head_dim` FP32 carry on rate-2 sources |

Tensors that ship twice (the engram `q`/`k` in FP32 and BF16) are counted at the larger size.

### `tools/dsv41_validate.py`

    python tools/dsv41_validate.py --model <checkpoint-dir> --ref <capture.json> [--mode MODE] [options]

Loads the model and scores its prompt logprobs against a vLLM capture of the same prompts (the
file `tools/dsv41_capture_ref.py` writes), and its cached paths against its own stateless
forward. Every mode except `--dry-run` loads the full model; run it alone on the host, with
nothing else using the GPUs. Logits never leave the device whole: each forward's logits are
reduced on the head's device, `--logit-rows` rows at a time, to the top-5, the logprob of the
actual token and the logprobs at the reference's top-5 ids.

#### Modes

- **nc** (default): one stateless forward of the whole prompt (`attn_mode = "flash_attn_nc"`),
  for prompts up to `--nc-max` tokens (default 1500).
- **cached**: a Cache and a chunked `model.forward` with the job state carried between chunks,
  uniform `--chunk` (default 2048, 512 above 16384 tokens).
- **decode**: cached, token by token across the positions where V4.1's attention changes
  regime (0-7; 512->513; 1025->1026; 16383->16384; the tail), prefilled in chunks elsewhere.
  The single-step runs start after odd prefill lengths, so their first step closes a rate-2
  group whose first row was carried over.
- **self**: nc (when short enough), cached at each of `--chunks` (default `2048,1000,7`) and
  decode on the same sequence, scored against vLLM and against each other.

#### Gates

Against vLLM, each run is judged at two levels: the **calibrated limits**, and the same limits
**shifted by the reference's own noise** on the case's positions. Only a check beyond both fails
the run.

**Calibrated limits** (`GATES` in the scorer), set from vLLM's own rep-to-rep noise on the
calibration capture, at 24 and 300 tokens only (at 300 tokens its |dlp| p50 0.011, p90 0.216 and
top-1 293/299, with every flip at a margin of at most 0.375; `CALIBRATION_NOISE`):

- |dNLL| <= 0.03 overall and <= 0.05 in every region with at least 8 positions;
- top-1 agreement >= 97% where the reference's top-1/top-2 margin exceeds 0.375;
- |dlp| (logprob difference on the actual token) p50 <= 0.05, p90 <= 0.4, p99 <= 1.0,
  max <= 5.0, globally and per region;
- top-5 overlap >= 0.90;
- integrity, never shifted: complete coverage, no non-finite logits, no missing reference
  entries, and a reference not recorded as contended, or with probe errors, while it was
  captured (with `--strict`, also one that recorded its isolation at all; see "Isolation
  metadata" below).

**Why the limits move with the length.** The calibrated limits fail vLLM against itself at other
lengths. Its *prefix noise*, a shorter capture scored against the same positions of a longer one
(`--dry-run` prints it): 20000 against 1500 tokens fails region [1,128) (|dNLL| 0.052, |dlp| p99
1.24; across the prefix pairs that region's top-1 at margin is 0.983-0.991 and its p90
0.29-0.36), and 300 against 24 tokens fails |dNLL| through position 1 alone: vLLM's own logprob
at position 1 moves by 2.95-3.81 nats between consecutive captured lengths (6.05 between 24 and
1500 tokens). The recorded runs of the port (DeepSeek-V4.1-Flash 3.0 bpw) failed the calibrated
limits in the same places: every nc and cached run at 1500 tokens in region [1,128) (|dlp| p99
1.06-1.45; in most runs also p90 0.43-0.48 and top-1 at margin 111/115; the two decode runs
passed), and every run at 20000 tokens on |dlp| max (5.64-6.12). At 1M tokens two runs of the
same vLLM reference differ by up to 7.85 nats.

**Noise-shifted limits.** So the harness measures, for each gated case, how far vLLM differs from
itself on exactly that case's positions, from the other captures in the same file
(`reference_noise` in the scorer):

- every other rep of the same prompt (rep noise);
- the same prompt inside every longer capture that starts with it (prefix noise);
- every shorter capture the prompt starts with, on that capture's own positions.

A capture recorded as contended or with probe errors never counts; with `--strict`, neither does
one without isolation metadata. A comparison counts for a summary (the whole case, positions
below 512, each region) only when it covers all of that summary's positions: a shorter capture
covers the regions that end within it, never the whole case. For each measure the reference's
noise there is the worst value over those comparisons, and the limit moves by as much as that
noise exceeds the calibration noise:

    noise-shifted limit = noise here + (calibrated limit - calibration noise)   |dNLL|, |dlp| p50, p90, p99
    noise-shifted limit = noise here - (calibration noise - calibrated limit)   top-1 at margin, top-5 overlap

It is never tighter than the calibrated limit. Each limit keeps the headroom it has over the
noise it was set on, which leaves room for the systematic differences expected between the two
engines (FP16 KV here, FP8 KV and BF16 streams in vLLM):

| Measure | Calibrated limit | Calibration noise (300 tokens, rep1 vs rep0) | Headroom |
|---|---|---|---|
| \|dNLL\|, overall / per region | 0.03 / 0.05 | 0.0089 | 0.0211 / 0.0411 |
| top-1 at margin > 0.375 | >= 0.97 | 1.0 (268/268) | 0.03 |
| \|dlp\| p50 | 0.05 | 0.011 | 0.039 |
| \|dlp\| p90 | 0.4 | 0.216 | 0.184 |
| \|dlp\| p99 | 1.0 | 0.556 | 0.444 |
| \|dlp\| max | 5.0 | 0.744 | 4.256, per position |
| top-5 overlap | >= 0.90 | 0.9485 | 0.0485 |

|dlp| max is judged position by position: every position beyond 5.0 must lie within vLLM's own
difference at that same position plus 4.256. vLLM's position 1, predicted from one token, moves by
up to 6.05 nats between request lengths; that excuses a large difference at position 1 only,
never at another position. The report lists its 20 worst positions; when more positions than that
exceed 5.0, none of them is excused.

**What each line means.**

| Line | Meaning | Fails the run |
|---|---|---|
| `PASS` | within the calibrated limit | no |
| `FAIL within reference noise` | beyond the calibrated limit, within the noise-shifted limit; the line gives that limit, vLLM's own value on these positions, the comparison it comes from, and the calibration noise | no |
| `FAIL ...; beyond the noise-shifted limit` | beyond both limits | yes |
| `FAIL ...; no wider here` | beyond the calibrated limit; vLLM's own noise on these positions is no larger than at calibration | yes |
| `FAIL ...; no comparison of vLLM against itself covers these positions` | beyond the calibrated limit, and nothing measured vLLM's noise there | yes |
| `FAIL` without a note | an integrity check, or a report scored without reference noise (the scorer called without `attach_noise`) | yes |
| `WARNING reference isolation NOT RECORDED` | the capture did not record whether vLLM served the prompt alone | only with `--strict`, where it is a `FAIL` |
| `info` | the comparisons used as reference noise and those left out (and why); regions under 8 positions, reported but not gated | no |

A run's verdict is `pass` (every check within its calibrated limit), `within-noise` (nothing fails
the run, and at least one check is a FAIL within reference noise), `precision-suspect` (see below)
or `fail`. The harness exits 0 when every run is `pass` or `within-noise` and the self, Generator
and prefill checks pass, 1 otherwise, 2 when it refuses the plan.

**What they mean.** `pass`: the port agrees with this capture as closely as the calibrated limits
ask. `within-noise`: it differs from this capture by no more than vLLM differs from itself on the
same positions, plus the headroom above; it matches the reference as well as the reference
reproduces itself, which is not proof that it computes the same function (a bug smaller than
vLLM's own noise there passes). A FAIL of the run means a difference beyond what vLLM shows
against itself on those positions, or on positions where nothing measured vLLM's noise. That is
evidence of a difference, not by itself of a bug: compare it with a known-good run and with the
prefix noise `--dry-run` prints, and re-run with `--dsv41-numerics vllm`.

**On the calibration capture** (`--dry-run` prints the limits of each gated case; every case has 5
comparisons):

- 24 tokens: |dNLL| <= 0.269 overall (vLLM between 24 and 1500 tokens: 0.248, from position
  1's 6.05 nats), |dlp| p50 <= 0.086, p90 <= 0.503, p99 <= 1.81, overlap >= 0.847; |dlp| max
  up to 10.30 at position 1 and 5.62 at position 2, 5.0 elsewhere.
- 300 tokens: |dNLL| <= 0.049, top-1 at margin >= 0.966, p90 <= 0.492, p99 <= 1.418 overall;
  region [1,128) p90 <= 0.647, p99 <= 1.508.
- 1500 tokens: region [1,128) |dNLL| <= 0.093, top-1 at margin >= 0.944, p90 <= 0.545,
  p99 <= 1.685, where the recorded nc and cached runs had p99 1.06-1.45, p90 at most 0.48 and
  top-1 at margin 111/115 (0.965): all within.
- 20000 tokens: one rep, and no longer capture, so nothing covers the whole case or any position
  from 1500 on: the calibrated limits stand there, |dlp| max included (the recorded runs had
  5.64-6.12; their worst positions were not recorded). Region [1,128) gets p90 <= 0.647, p99 <=
  1.685 and top-1 at margin >= 0.961, which the recorded runs' values there (p90 0.34-0.50, p99
  0.76-1.40, top-1 112-113/114) meet. Capture reps at 20000 tokens (`tools/dsv41_capture_ref.py
  --lengths 20000 --reps 3`) to measure vLLM's noise over the rest.

The recorded runs' reports were not kept, so they were not rescored as a whole with these limits:
the figures above compare their recorded region [1,128) and whole-case summaries.

**Isolation metadata.** `tools/dsv41_capture_ref.py` records per case whether anyone else's request
shared vLLM's batch while it ran (`contended`) and how many of its probes failed (`probe_errors`).
A case recorded as contended or with probe errors fails as a reference and is never used as
reference noise. A case without these fields (all six of the calibration capture's) is scored as
if vLLM had served it alone: the harness prints a `!!! WARNING` banner when it starts and again
in its summary, and every gated run carries a `WARNING reference isolation NOT RECORDED` line.
`--strict` refuses such a gated case before anything loads (exit 2), fails it when a report is
scored (`FAIL  reference isolation not recorded ... (--strict)`), and keeps such captures out of
the reference noise; with the calibration capture it refuses every case.

A failure confined to regions from position 512 on is labelled **precision-suspect**: the index
top-k starts to select there (below 512 every entry is kept), so the indexer's rounding matters
only from there. vLLM rounds the index operands to MXFP4, which the default `deepseek:index`
does too, and its KV to FP8 (`fp8_ds_mla`) at every position; re-run with `--dsv41-numerics
vllm` (vLLM's whole contract) before calling it a bug. The label needs no check below 512 to
fail beyond its noise-shifted limit. Every |dNLL| failure is printed with its value without the
worst single position, because of vLLM's position 1.

Self mode compares two exllamav3 runs of the same sequence. Without a calibration floor the
gates are strict (no top-1 flip that is not an exact tie in one of the runs, |dlp| p99 <= 0.02,
and p90 <= 0.02 over the first 4 predictions after every chunk start, where n-gram lookback,
ring and carry bugs live). With a floor (the stateless forward against a stateless forward two
thirds as long, which only changes the row count) the limits are relative to it: at most
1.5 times the floor's rate of hard flips plus three Poisson standard deviations, |dlp| p99 at
most max(0.02, 1.5 x the floor's), the chunk-boundary p90 at most max(0.02, 1.5 x the floor's
p90); the top-5 overlap and |dlp| max limits apply as well.

The floor exists only for a case longer than 24 tokens and at most `--nc-max` (default 1500)
tokens, and only when both stateless forwards succeed. Otherwise (longer cases, the 24-token
case, a failed nc run) the strict gates apply, and a correct run without the
reproducible-arithmetic profile fails them: the exl3 kernels change tier with the row count and
the MoE routing near-ties amplify that, so two correct paths differ by far more than 0.02 (a
recorded run at 1500 tokens without a floor: |dlp| p99 0.06-0.50 between nc and the cached and
decode runs). Beyond `--nc-max`, run self mode with `EXL3_STABLE_ARITHMETIC=1`, or with a larger
`--nc-max` when a stateless forward of the whole case fits.

#### Engine settings

The harness applies engine settings through the engine's own channels and records everything
it used:

- **--placement *rules***: the explicit placement, set as `EXL3_PLACEMENT` before `exllamav3` is
  imported (an empty value is no placement and clears the caller's), and checked before
  anything loads: its grammar, and with `--model` DeepSeek-V4.1's rule for its changes of device
  on the model's topology (doc/placement.md, "DeepSeek-V4.1"). `split=<k>`, the grammar of the
  removed V4.1 split variable, is refused with the placement that means the same, e.g.
  `--placement 'split=12': split=<k> is not a placement rule; write each device's layers, e.g.
  --placement '0-11=cuda:0; *=cuda:1'`.
- **--dsv41-numerics *setting***: assigned to `config.dsv41_numerics` after
  `Config.from_directory` (`precise`, `deepseek`, `vllm`, optionally `:` and parts from `index`,
  `window`, `compressed`). Without it the config keeps its own value: `EXL3_DSV41_NUMERICS`,
  else `deepseek:index`. Checked before the model loads. Comparisons against earlier `precise`
  runs must pass `--dsv41-numerics precise` explicitly.
- **--mcl**, **--mcs**, **--mct**, **--moe-cpu-mode**, **--fp32-logits**: set
  `infer_params.moe_cpu_offload`, `moe_cpu_split`, `moe_cpu_threads`, `moe_cpu_mode` and
  `fp32_logits`, as model_init's `-mcl`, `-mcs`, `-mct`, `-mcm` and `-fp32_logits` do. `--mcl`
  and `--mcs` are mutually exclusive.
- **--gpu-split**, **--reserve**, **--max-chunk-size**: `model.load(use_per_device=...,
  reserve_per_device=..., max_chunk_size=...)`; **--cache-tokens**, **--cache-slots**: the
  Cache's `max_num_tokens` and `max_batch_size`.
- **--visible *list***: `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`), set before
  torch is imported.
- Every other setting (the reproducible-arithmetic profile `EXL3_STABLE_ARITHMETIC`, the
  pipelined prefill `EXL3_DSV41_PIPELINE`, the group of FP32 options, the CPU-offload tuning)
  comes from the caller's environment, as the engine reads it. There is deliberately no option
  for the arithmetic profile: it is read once when `exllamav3` is imported.

A setting the checked-out tree does not implement (an older checkout without the explicit
placement, `moe_cpu_mode` or `fp32_logits`) is refused before the model is built, so a run never
silently ignores it. The plan (and `--dry-run`) also refuses, before anything loads, the
combinations the loader would refuse, taking each setting from its option or, without one, from
the caller's environment as the config reads it: an explicit placement (`--placement`, or the
caller's `EXL3_PLACEMENT` without it) together with `-mcl`, `-mcs` or a `-mcm` other than
`compute` (the options, or `EXL3_MOE_CPU_OFFLOAD`, `EXL3_MOE_CPU_SPLIT`, `EXL3_MOE_CPU_MODE`), or
with `EXL3_MOE_CPU_SPLIT_LAYERS`; and `-mcl` together with `-mcs`. It also refuses a caller's
`EXL3_DSV41_PLACEMENT` with any value, because the engine no longer reads it and a run would
silently use another layout than the script meant: `EXL3_DSV41_PLACEMENT='split=12' is set, but
the engine does not read it: the layer split comes from EXL3_PLACEMENT / --placement (e.g.
'0-11=cuda:0; 12-39=cuda:1') or from the split budgets; unset it`.

#### Other options

- **--cases *list*** (default `24,300`): prompt lengths to run, from the capture; **--ref-rep**
  (default 0): the capture's rep the gates use (other reps of the same length are scored and
  reported, not gated, and serve as reference noise).
- **--strict**: a gated case without isolation metadata is refused before anything loads and fails
  when scored, and no capture without it serves as reference noise (see Gates, "Isolation
  metadata"). Without it such a case is scored with a loud warning.
- **--greedy *N*** (default 16): greedy tokens printed after the prompt; **--generator**: also
  generate them through exllamav3's Generator (cached modes) and require the same ids.
- **--prefill-check**: with `EXL3_DSV41_PIPELINE` on (any value but `0`, as the engine reads
  it), prefill the prompt (minus `--prefill-tail` tokens, default 32) once with sub-chunk-sized
  calls and once with `--prefill-chunk`-sized pipelined calls (default 16384, a multiple of at
  least two `EXL3_DSV41_PIPELINE_CHUNK` sub-chunks), then gate the continuation logits of the
  two against each other with the self gates. Fails unless the pipelined path actually ran, so
  it needs a load the pipeline is eligible for (doc/env_vars.md, `EXL3_DSV41_PIPELINE`): a layer
  split whose RAM-held experts, if any, all lie past the split. `-mcl` puts RAM-held experts
  before the split and `-mcs` on every layer, so either makes the load ineligible and the check
  fails on path coverage; use an explicit placement such as the one in the example below.
- **--engram-lookback** `state|harness` (default `state`): `harness` also passes the n-gram
  lookback the harness computes from the prompt, which the engram checks against its own carry.
  The `engram_nocarry` ablation needs `state` (the engram refuses a passed lookback that differs
  from the one it carried).
- **--logit-rows *N*** (default 512): rows per on-device logit reduction. The reduction's FP32
  workspace scales with it (`tools/dsv41_fit.py --validation-logit-rows` estimates it); the
  logits a forward returns do not.
- **--cache-slots *N*** (default 2): the Cache's recurrent-state slots (`max_batch_size`): the
  teacher-forced runs hold one, the `--generator` job the other. 1 suits a single long-context
  run.
- **--cache-tokens *N*** (default: `max(65536, slots x pages x 256)`, where `pages` is the
  number of 256-token pages the longest case plus `--greedy` plus one token needs): the Cache's
  `max_num_tokens`. The teacher-forced runs pass no block table, so each state slot owns
  `1/slots` of the pages (`CacheLayer_dsa.slot_bt`); the plan refuses a value that is not a
  multiple of 256, or whose per-slot share cannot hold that longest sequence.
- **--max-chunk-size *N*** (default: the largest of `--chunks`, `--chunk` (2048 when unset), the
  largest logit-producing forward of the plan and, with `--prefill-check`, the pipeline
  sub-chunk): `model.load(max_chunk_size=...)`. The load's `max_output_size` is that largest
  logit-producing forward (a cached run's largest chunk; the whole prompt plus `--greedy` - 1
  tokens for nc; at least `--prefill-tail` with `--prefill-check`). An explicit
  `--max-chunk-size` below it is refused.
- **--bf16-logits**: round our logits to BF16 before the reduction, as vLLM does.
- **--tie-eps** (default 0): the top-2 gap that counts as a tie in self mode.
- **--dump-positions**: per-position arrays in the report.
- **--min-avail-gib** (default 8): a watchdog kills the run and its CPU-MoE worker when
  `MemAvailable` falls below this, before the kernel's OOM killer picks a process (the routed
  experts in pinned host memory cannot be swapped). 0 turns it off.
- **--out *file***: the JSON report (default `<tmp>/dsv41_validate/<mode>[-<ablations>].json`).
- **--dry-run**: parse the capture, print the plan, vLLM's own rep-to-rep and prefix noise against
  the calibrated limits, and each gated case's reference noise and noise-shifted limits; write
  them with the plan; no torch, no CUDA.

#### Ablations

`--ablate TOKEN` sets `EXL3_DSV41_ABLATE` (doc/env_vars.md) before `exllamav3` is imported, which
makes the model compute a known wrong function; the run MUST then fail the gates in the regions
the ablation affects, otherwise the harness is blind to that feature. The engine reads the
variable once per process, so `--ablate-sweep` runs a baseline and every `--sweep-tokens`
ablation each in its own process, and passes only when the baseline passes (`pass` or
`within-noise`), every ablation is detected where it acts, nothing leaks below an isolated
ablation's first affected position, and every ablated run is a complete, finite measurement. A
check counts as detected (or as a leak) when it fails the run, beyond its noise-shifted limit,
where the baseline's does not: a FAIL within reference noise neither detects nor leaks, exactly
as it does not fail a run. Without
`--sweep-tokens` the sweep takes every ablation the run can exercise (`engram_nocarry` needs a
cached, decode or self mode and `--engram-lookback state`); a token named in `--sweep-tokens`
that the run cannot exercise is refused before any child starts.

| Token | Fails from position | What it breaks | Recorded sweep (cached, chunk 4096): 1500 / 20000 tokens |
|---|---|---|---|
| `engram` | 1 | the engram layers (1 and 14) do not run | detected / detected |
| `engram_nocarry` | after the first chunk start (cached, decode and self modes; `n/a` on a run without one) | the n-gram history is dropped at chunk starts | not in that sweep |
| `v4mix` | 1 | V4's immediate hyper-connection mix instead of V4.1's delayed pre-mix | detected / detected |
| `rope_consecutive` | 128 | rate-2 pool entries roped at consecutive positions | detected / detected |
| `dense_consumers` | 512 (isolated) | consumers attend their whole pool instead of their index source's top-k | blind / leak |
| `no_candidates` | 16384 (isolated) | no candidate-block stage above layer 20 | n/a / blind |

The last column is a sweep recorded on DeepSeek-V4.1-Flash 3.0 bpw with an earlier version of
the harness, rescored with the calibrated limits alone, before the noise-shifted limits existed;
its reports were not kept, so it has not been rescored with them, and a detection that lay
within the reference's noise would now read `blind`. Its baseline failed the calibrated limits at
both lengths (see Gates), so that sweep did not pass as a whole; the per-ablation statuses compare
each ablated run with the baseline and remain meaningful. At 1500 tokens the gates did not see
`dense_consumers`, and at 20000 they did not see `no_candidates`: a regression of either feature
can pass them at those lengths.

The `dense_consumers` leak at 20000 tokens (region [1,128) |dNLL| -0.060 against the
baseline's -0.041) is not run-to-run noise: two recorded sweeps agree to 1e-4. The positions
below 512, which the ablation cannot reach, are computed in the same 4096-row forward as those
it changes; there different routing changes how many rows each expert gets and so which kernel
tier and partition the MoE layers use for every row. Without `EXL3_STABLE_ARITHMETIC=1` a
`leak` verdict is therefore not evidence of a real leak. The profile makes every row's result
independent of the other rows, which should remove this coupling; the sweep has not been
recorded under it.

#### Report

The JSON report is written atomically (a new private file renamed into place) at the start of a
run, marked `running`, and rewritten after every case; it is marked `completed` or `failed` at
the end, so a stale success can never survive a new run. It records the arguments and the plan,
the allowlisted environment (`REPORT_ENV`: this series' own settings and the main dispatch and
precision switches V4.1's default path reads, from doc/env_vars.md, for example
`EXL3_INT8_GEMV`, the int8 activations of decode's GEMVs; not every documented variable, and
nothing else), the numerics setting and arithmetic policy
the run resolved, the gates (the calibrated limits and `calibration_noise`), the captured cases
without isolation metadata and whether `--strict` was set (`reference_isolation`), the load
(devices, allocated memory, seconds), per case the comparisons used as reference noise and those
left out, every run's scores, gate messages and verdict with the reference noise it was judged
against (`report.reference_noise`: per summary and measure the worst value of vLLM against
itself and its comparison, and vLLM's own difference at each of the run's worst positions), and
provenance: the source commit, a hash of the uncommitted diff and
of every source file, the capture's and config's SHA256 (and the router-bias sidecar's when
`EXL3_DSV41_ROUTER_BIAS` is set), each shard's size and header hash (not a full weight hash),
the loaded extension's SHA256, and the torch, CUDA and device identities.

#### Examples

Plan a self-mode run without loading anything:

```sh
python tools/dsv41_validate.py --dry-run --mode self --cases 300,1500 --ref capture.json
```

Two GPUs with eleven layers' experts streamed to the second one, against the 24- and 300-token
prompts:

```sh
python tools/dsv41_validate.py --model /mnt/models/DeepSeek-V4.1-Flash-exl3 --ref capture.json \
    --mode nc --cases 24,300 --gpu-split 63,91 \
    --placement "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
```

The chunked path at 20000 tokens under the precise numerics, then under vLLM's contract:

```sh
python tools/dsv41_validate.py --mode cached --cases 20000 --dsv41-numerics precise
python tools/dsv41_validate.py --mode cached --cases 20000 --dsv41-numerics vllm
```

Chunk invariance under the reproducible-arithmetic profile (the profile is environment-only):

```sh
EXL3_STABLE_ARITHMETIC=1 python tools/dsv41_validate.py --mode self --cases 1500 \
    --chunks 1024,256,7 --placement "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
```

The pipelined prefill against sequential prefill on a 20000-token prompt, on a layout the
pipeline is eligible for: a split with RAM-held experts only past it. With every expert in VRAM
a change of device at 12 does not fit a 64 + 96 GiB pair (`tools/dsv41_fit.py`), and
`-mcl`/`-mcs` would make the load ineligible:

```sh
EXL3_DSV41_PIPELINE=1 EXL3_DSV41_PIPELINE_CHUNK=4096 python tools/dsv41_validate.py \
    --mode cached --cases 20000 --prefill-check --prefill-chunk 16384 --gpu-split 63,91 \
    --placement "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
```

Every negative control the run can exercise, each in its own process, under the
reproducible-arithmetic profile, so that the positions an isolated ablation cannot reach do not
depend on the rows it changes (see Ablations):

```sh
EXL3_STABLE_ARITHMETIC=1 python tools/dsv41_validate.py --mode cached --cases 300,20000 \
    --ablate-sweep --placement "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
```

With the calibration capture this sweep can fail as a whole at 20000 tokens: nothing in it
measures vLLM's noise past position 1500 there, so the baseline's |dlp| max is held to the
calibrated 5.0 (the recorded baselines had 6.03-6.05, at positions that were not recorded; see
Gates). Read the per-ablation lines; at
300 tokens (one forward) `engram_nocarry`, `dense_consumers` and `no_candidates` are `n/a`, and
at 20000 all six are exercised.

(`--model` defaults to `DSV41_MODEL_DIR` and `--ref` to `DSV41_VLLM_REF`.)

### `tools/dsv41_rowprobe.py`

    python tools/dsv41_rowprobe.py --model <checkpoint-dir> --ref <capture.json> [--case n1500r0] [options]

The generator verifies a draft with one forward of 1+N rows and keeps a draft token only when it
equals the trunk's own sample, so drafted output equals undrafted output exactly when each row of
that forward gets the bits a one-row decode step gives it. This tool measures that, operation by
operation, in the arithmetic of the calling environment. It loads the full model with a Cache;
run it alone on the host.

After prefilling `--prefix` tokens (default 1200, past the 512-entry threshold of every pool, so
the top-512 selection is active; a short prefix such as 40 probes the dense regime), it runs, for
each row count K of `--rows` (default 2 to 8) and from the same job state, K one-row forwards
with the generator's decode params and one K-row forward with its verify params. The candidate
stage of the index sources above the candidate source masks nothing until a row sees more than
`candidate_topk_blocks * candidate_block_size` entries (16,384 on V4.1-Flash), so probing it
needs a prefix past that and a token source that long. Wrappers around
the Python methods and native calls of each block and of the head capture every row's inputs and
outputs in both arms. The state is restored between the arms by the engine's own rewind, the one
the generator uses to reject draft tokens.

Per K it prints, in execution order, each captured output with `=` or `DIFF`, the first layer
and row that differ, the count of differing units and elements, the largest absolute and relative
difference, and how many units are an origin: the output differs while every input of the
operation, the state it reads included, is bit-equal. Then the first diverging operation of the
forward, the call facts that differ between the arms (mHC chunk count, indexer kernel, attention
split count), and per row the logits' bit equality, both argmaxes and the top-2 gap. A final
table lists operation by K: `=`, or the first differing layer, as `L3*` when the operation is an
origin at that layer and as `L3/o20` when it only propagates there and is first an origin at
layer 20.

The replay table isolates each operation: inside the K-row forward the operation is run again one
row at a time on the inputs the K-row call had. `=` there means the operation gives a row the
same bits at 1 and at K rows, whatever differs upstream. For the EXL3 linears a second replay
with the int8 GEMV off (`EXL3_INT8_GEMV=0`, set in-process for those calls only) separates the
int8-versus-FP16 switch from the row buckets of the launch autotuner, and a table lists the
kernel each linear launches at 1 to 8 rows with its autotune records. `--no-replay` and
`--no-int8-split` turn these off. When `EXL3_INT8_GEMV` is not set the tool sets it to 2, the
value the extension takes when it is unset, before anything is loaded, so that the in-process
switch only ever replaces an existing variable while other threads run.

Every run checks itself, bit for bit:

- the unhooked repetitions of each arm agree (`--timing-reps`, default 5), and hooked logits
  equal unhooked ones;
- every one-row run equals a reference taken on the state as prefilled, before the first K-row
  forward of the process: the logits of each run, and every captured tensor of the one-row arm.
  This is what shows that a K-row forward and its rewind leave nothing behind that a one-row
  step reads;
- a K-row forward that follows a K-row forward equals one that follows one-row steps;
- the one-row arm repeated after the K-row arm equals its first run on every captured tensor,
  and the replay run equals the capture run on every captured tensor;
- coverage: every operation the layer facts call for was captured on every row or pool entry
  of both arms. A gap is printed as `NOT CAPTURED: <operation> on layers ...`; the tables miss
  that operation there.

Exit status 0 means the probe ran, these checks passed and nothing was missed, whatever it
found; 1 a check failed or an operation was not captured; 2 refused.

The verify cost is printed per K as the median and the minimum of those repetitions: K one-row
steps, one K-row forward, and that forward in one-row steps.

The launch autotuner keeps its choices in `coop_autotune_v1.bin` (see `EXLLAMAV3_TUNE_CACHE`),
and a row bucket with no record there is tuned by timing on its first use. By default the tool
works on a private copy next to its report (`--tune-cache copy`), removed when the run ends, so
it reads the records serving uses and never writes that file. Runs that are to be compared
should share one file, `--tune-cache <path>`: it is created as a copy of the live file when it
does not exist, and a bucket one run tuned is reused by the next. `live` uses the file itself.
The records a run added are listed in its report.

`--stable-check` only prints the switches of `exllamav3/model/math_policy.py` as the environment
sets them, without importing torch, under the profile too. Under `EXL3_STABLE_ARITHMETIC=1` the
tool refuses to run unless `--allow-stable` is given: every operation should then come out equal,
which validates the probe.

Under `EXL3_EXACT_ROWS=1` (see `doc/env_vars.md`) the tool is that mode's acceptance test: the
K-row forward then gives every row of the operations that follow the row count the launch of a
one-row call, from the extension's row-exact entry points or from Python, one call per row; the
wrappers follow both, and every in-situ and replay line is expected to read `=`, with bit-equal
logits; the index selection and the attention kernel have no other test. The tool prints the
entry points in use and the MoE layers they serve. `--exact-rows-caps MASK` runs with a subset of
them (a sum of 1 EXL3 linears, 2 grouped `wo_a`, 4 router, 8 MoE experts, 16 hyper-connection
sums; the MoE needs 4 and 8 together; `0` is the Python row loops alone), which gives the same
tables and the cost of each step. Bit 32 is not an entry point and cannot be masked: with
`EXL3_INT8_GEMV=0`, an extension that reports it makes one launch for the rows of an EXL3
linear and of the grouped `wo_a` inside the entry points of bits 1 and 2, and the timing table
prints, per K, how many calls of the K-row forward were served that way (`one-launch calls`; 0
with the int8 path on). The timing table then gives what a verify forward costs under
the mode, in one-row steps: the figure that decides whether a draft pays; its `host` column is
the part of each arm until `model.forward` returned, before the logits copy waits for the GPU.
Per K it says whether the forward is one the mode covers; a forward
whose rows cross a change of the attention plan (`--prefix 124`, `252`, `508` or `1020` with
enough rows on DeepSeek-V4.1-Flash) is not, and keeps the default arithmetic. The report holds
SHA-256 digests of the pristine one-row reference (its logits and every captured tensor), and
`--same-pristine A.json B.json` compares those of two reports without importing torch: a run with
the mode against one without it, on one `--tune-cache` file, shows that one-row steps keep their
bits (exit status 0 equal, 1 different, 2 not comparable).

```sh
python tools/dsv41_rowprobe.py --stable-check
python tools/dsv41_rowprobe.py --model /mnt/models/DeepSeek-V4.1-Flash-exl3 --ref capture.json \
    --case n1500r0 --placement "0-11=cuda:0; 12-39=cuda:1" --out rowprobe.json
# one-row steps keep their bits under EXL3_EXACT_ROWS: two runs on one autotune file
python tools/dsv41_rowprobe.py --model /mnt/models/DeepSeek-V4.1-Flash-exl3 --ref capture.json \
    --tune-cache tune.bin --out off.json
EXL3_EXACT_ROWS=1 python tools/dsv41_rowprobe.py --model /mnt/models/DeepSeek-V4.1-Flash-exl3 \
    --ref capture.json --tune-cache tune.bin --out on.json
python tools/dsv41_rowprobe.py --same-pristine off.json on.json
```

### `tools/dsv41_refcompare.py`

The scorer the harness uses, as a library: pure Python and numpy, no torch. Load it by path
(it is not part of the `exllamav3` package):

```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location("dsv41_refcompare", "tools/dsv41_refcompare.py")
rc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rc          # its dataclasses need the module registered
spec.loader.exec_module(rc)

cases = rc.load_ref("capture.json")                  # Case per captured prompt
report = rc.compare(case, top5_ids, top5_lp, actual_lp, at_ref_lp = at_ref)
rc.attach_noise(report, *rc.reference_noise(cases, case))   # the noise-shifted limits
ok, messages = rc.gates(report)                       # the gates above (strict = True: --strict)
verdict = rc.verdict(report)                          # pass / within-noise / precision-suspect / fail
```

Row `j` of every per-position array is prompt position `j + 1`: `logits[i - 1]` predicts token
`i`, and the captured prompts carry no BOS. `compare_self` and `self_gates` compare two exllamav3
runs, `ablation_check` scores a negative control, `uniform_schedule` and `decode_schedule` give
the forward sizes the harness runs, and `self_noise` / `prefix_noise` score vLLM against itself.
`reference_noise(cases, case, strict)` returns the comparisons of vLLM against itself on a
case's positions and the captures it left out, `attach_noise` records their worst values on a
report (without it, `gates` and `verdict` apply the calibrated limits alone), `noise_limits`
lists the limits they widen, and `isolation_missing(case)` names the isolation fields a case
lacks. `MEASURED_NOISE` holds the calibration capture's own noise, `tests/test_dsv41_refcompare_.py`
recomputes it from that file, and `CALIBRATION_NOISE` is its 300-token row, what the calibrated
limits were set on. `ablation_check(..., chunk_starts=...)` takes the run's chunk
starts (`chunk_starts(schedule)`, `[]` for a stateless run), without which it cannot tell that
`engram_nocarry` had nothing to act on.

### `tools/dsv41_capture_ref.py`

    python tools/dsv41_capture_ref.py --url <vllm-server> --out <new-file.json> [--lengths 1500,20000] [--reps 3]

Captures prompt logprobs from a vLLM server serving DeepSeek-V4.1-Flash, in the schema the
scorer reads. Run it while nothing else uses the server: a shared batch changes vLLM's numerics.
The harness takes the reference noise of a gated case from the other captures of the same file,
so several reps at a length give the rep noise at that length, and the longest length needs
them: nothing else measures vLLM against itself on its last positions (see Gates). Every case
records `contended` and `probe_errors`, the isolation metadata `--strict` requires.

- One request at a time, idle-gated: before each request it polls `/metrics` until
  `vllm:num_requests_running` and `vllm:num_requests_waiting` are both 0 (at most
  `--idle-timeout` seconds, default 1800); while the request runs a probe records whether anyone
  else's request shared the batch, and such a case is flagged `contended` (the harness then
  refuses it as a reference).
- Token-id prompts without BOS, `max_tokens` 1, temperature 0, `prompt_logprobs` and `logprobs`
  `--top` (default 20), the generated token returned as an id (`--no-token-ids` to not ask for
  that; it is also dropped automatically when the server refuses it).
- Prompts are prefixes of the longest prompt in an existing capture (`--ref`, default
  `DSV41_VLLM_REF`), so every new capture shares its prefix with it, or of a JSON list of token
  ids (`--ids-file`).
- **--url** (default `http://127.0.0.1:8000`, vLLM's own default), **--model** (the served
  model name; default the first of `/v1/models`), **--lengths** (default `1500,20000`),
  **--reps** (default 3), **--dry-run** (print the plan, touch no network).
- It writes a NEW file: it refuses an `--out` that exists and never writes `--ref`. The capture
  is written to a private directory as it goes (and kept there on failure), and linked into place
  atomically at the end.

### Test data

The tests of these tools, like the other DeepSeek-V4.1 tests, read their data from the
environment and skip visibly without it:

| Variable | Contents | Used by |
|---|---|---|
| `DSV41_MODEL_DIR` | a DeepSeek-V4.1-Flash checkpoint directory | test_dsv41_placement_ (part C), test_dsv41_build_ |
| `DSV41_VLLM_REF` | a vLLM capture (`tools/dsv41_capture_ref.py`); the harness's noise checks expect the calibration capture | test_dsv41_refcompare_ |

`test_dsv41_fit_inputs_`, `test_dsv41_checkpoint_header_`, `test_dsv41_reference_schema_` and
`test_dsv41_validation_safety_` need no data.
