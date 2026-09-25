## DeepSeek-V4.1 checkpoint and layout tools

Standalone scripts in `tools/` for working with a DeepSeek-V4.1 checkpoint before (and
around) loading it. None of them imports the `exllamav3` package, torch or the compiled
extension: they read `config.json` and the safetensors headers with the standard library,
through the same bounded header reader the model uses for its engram tables
(`exllamav3/architecture/dsv41/checkpoint_header.py`), and load the few torch-free engine
modules they need by file path. They run on any host with Python 3.10 or newer, including one
without a GPU.

| Script | Answers | Reads |
|---|---|---|
| `tools/dsv41_keymap.py` | Does the checkpoint carry exactly the tensors the V4.1 graph claims? | config.json, shard headers |
| `tools/dsv41_topology_check.py` | Do the layers that carry compressors, indexers and engrams match the topology config.json declares? | config.json, shard headers |
| `tools/dsv41_fit.py` | How much GPU memory and host RAM does a layout need, and does it fit? | config.json, shard headers |

All three exit with status 0 when the check passes (or the estimate fits), 1 when it fails and
2 on a usage error, so they can gate a script.

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
