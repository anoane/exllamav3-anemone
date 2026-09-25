# Expert tiers and RAM budgets

**Experimental.** A large MoE model rarely fits in GPU memory whole. Its routed experts can live
in three places: the GPU's memory (VRAM), system RAM, and the checkpoint on the SSD. This document
covers the budgets that bound how much system RAM a load may take for routed experts and for
n-gram tables, and how a load checks them against the host before it allocates anything.

It is the reference for:

| Knob | CLI (`model_init`) | Environment | Python (`config.infer_params`) | Default |
|---|---|---|---|---|
| RAM for the routed experts of the main model | `-er` / `--expert_ram <size>` | `EXL3_EXPERT_RAM` | `expert_ram` | no cap |
| RAM for the routed experts of a draft model or MTP head | `-der` / `--draft_expert_ram <size>` | none | `draft_expert_ram` | no cap |
| RAM for n-gram tables (PLE models) | `-ngr` / `--ngram_ram [<size>]` | `EXL3_NGRAM_RAM`, and the older `EXL3_NGRAM_STREAM` | `ngram_ram` (and the older `ngram_stream_from_disk`) | `0`: every row streams from disk |
| Host memory kept free by every check | none | `EXL3_HOST_MEM_RESERVE_MB` | none | `2048` |

The explicit placement ([placement.md](placement.md)) says which layers keep their experts in
system RAM. The budgets say how much RAM they may take in total.

## Sizes

Every budget takes a size, written the same way everywhere (flags, variables, Python attributes,
the placement's `ram` rule):

| Written | Bytes | Notes |
|---|---|---|
| `48GiB`, `48Gi`, `48gib` | 48 x 2^30 | binary units: `KiB MiB GiB TiB` (or `Ki Mi Gi Ti`) |
| `48GB`, `48gb` | 48 x 10^9 | decimal units: `KB MB GB TB` |
| `48` | 48 x 2^30 | a bare number is GiB, as `-gs` and `use_per_device` read theirs |
| `1.5`, `0.5GiB` | 1.5 x 2^30, 2^29 | decimals are exact: `0.1GB` is 100,000,000 bytes |
| `4096B` | 4096 | bytes |
| `0` | 0 | nothing |
| `48G`, `512m`, `1T` | refused | ambiguous: `'48G' is ambiguous: write 48GiB (2^30 bytes) or 48GB (10^9 bytes)` |
| `48 GiB`, `48GIGS`, `-1GiB`, `1e3`, non-ASCII digits | refused | `not a size (use 0, auto, all, or a number with B, KiB, MiB, GiB, TiB, KB, MB, GB or TB (a bare number is GiB))` |

Units are case-insensitive. A size prints in the largest unit that divides it exactly, binary
first at equal magnitude (`1.5` prints `1536MiB`, `49152MiB` prints `48GiB`, `48GB` stays
`48GB`), and what it prints parses back to the same size.

Besides sizes, `--ngram_ram` takes `all` (every table whole). `--expert_ram` and
`--draft_expert_ram` take sizes only: they are caps, and no cap is the default.

## The budgets

### `-er` / `--expert_ram` (main model), `-der` / `--draft_expert_ram` (draft model or MTP head)

The most system RAM the routed experts of one component may take. What counts is what the CPU
worker's expert arena holds (`model/moe_cpu_host.py`): the experts of

- `-mcl N` / `--moe_cpu_offload N` (the first `N` eligible MoE layers, whole),
- `-mcs K` / `--moe_cpu_split K` (the last `K` routed experts of every eligible layer),
- a placement's `experts=stream`, `experts=cpu` and `experts=split cpu=<k>` layers,
- `-dmcl N` / `--draft_moe_cpu_layers N` (the draft model's or MTP head's first `N` eligible
  layers), which `--draft_expert_ram` caps.

The main model's text component takes `--expert_ram`; every other component (an MTP head sharing
the main model's config) takes `--draft_expert_ram`. A draft model loaded from its own directory
(`-dm`) is the text component of its own config: `model_init` sets that config's `expert_ram` from
`--draft_expert_ram` (and clears the `EXL3_EXPERT_RAM` its config read). `-der` without a draft
model or `--mtp` is refused.

Per expert, the arena holds the gate / up / down trellis (int16) and the fp16 `suh`, `svh` and
bias vectors of each projection, each rounded up to 64 bytes:

```
bytes per expert = sum over projections of  align64(trellis) + align64(2 x in) + align64(2 x out) [+ align64(2 x out) with a bias]
```

DeepSeek-V4.1-Flash 3.0 bpw (hidden 5120, intermediate 2304, K = 3): 3 x 4,423,680 + 44,544 =
13,315,584 bytes per expert, 4.76 GiB per layer of 384 experts; `-mcl 11` holds 52.4 GiB.

With a cap, a load that needs more is refused before anything loads (see "How a load checks
them"), with the numbers:

```
-mcl 11 keeps 52.4 GiB of routed experts in RAM, more than --expert_ram 40GiB; raise the cap or offload fewer layers
placement: the stream / cpu layers keep 52.4 GiB of routed experts in RAM, more than --expert_ram 40GiB; raise the cap or offload fewer layers
```

The CPU worker checks the cap again as each layer registers, from the layer's real dims
(cumulative per component, a rollback retry of the same layer is not counted twice), so a layer
that reaches the worker another way cannot exceed it either:

```
CPU MoE: model.layers.20.mlp would bring the routed experts held in RAM by this component to 57.1 GiB, more than --expert_ram 56GiB; raise the cap or offload fewer layers
```

### `-ngr` / `--ngram_ram [<size>]`

The system RAM for n-gram embedding tables (PLE models, e.g. Qwen3.8-Flash-Next, whose quantized
table is tens of GB). A table held in RAM serves its rows from memory; any other table streams its
rows from disk with per-forward gathers (run-coalesced positioned reads into pinned staging),
which costs little on SSD-class storage.

| Value | Meaning |
|---|---|
| unset (default), `0` | every table streams from disk |
| `-ngr` alone, `all` | every table whole in RAM (the flag's meaning before it took a size) |
| a size | the tables that fit, whole, smallest first (tables read by every gathered row would go first; PLE models have none); the others stream |

A table is held whole or not at all: a 12 GiB budget next to a 41 GiB table holds nothing and
the table streams. As `-ngr` takes an optional value, a bare `-ngr` right before a word that is
not a size takes that word and is refused (`ngram_ram=prompt.txt: not a size ...`), never
silently. `NGramEmbedding(stream_from_disk = True / False)` still overrides the budget for one
module.

DeepSeek-V4.1's engram tables (two of about 94.6 GiB each) are always read from disk, whatever
`--ngram_ram` says; a non-zero budget next to them is reported by the load (`engram tables
stream from disk (ngram=6GiB holds no DeepSeek-V4.1 engram table)`), not refused.

`EXL3_NGRAM_RAM` is the default of `--ngram_ram`. The older `EXL3_NGRAM_STREAM=0` still means
every table in RAM (`all`); next to an `EXL3_NGRAM_RAM` other than `all` it is refused as a
contradiction. `config.infer_params.ngram_stream_from_disk` is now derived: `False` exactly when
`ngram_ram` is `all`; setting it to `False` sets `ngram_ram = "all"`, setting it to `True` clears
`ngram_ram` (every row streams).

### Where each value comes from

| Knob | Order (first set wins) |
|---|---|
| `expert_ram` | `-er` > `EXL3_EXPERT_RAM` > no cap |
| `draft_expert_ram` | `-der` > no cap |
| `ngram_ram` | `-ngr` > `EXL3_NGRAM_RAM` > `EXL3_NGRAM_STREAM=0` (= `all`) > `0` |

The variables are read when a `Config` is created (a malformed value raises a `ValueError`
there); the flags when `model_init` builds it (argparse refuses a malformed value with the size
error); the Python attributes accept a size string, a number (GiB), `None` and, for `ngram_ram`,
`"all"`, `True` and `False`, and are parsed when set:

```python
config = Config.from_directory(model_dir)
config.infer_params.expert_ram = "52GiB"      # cap the -mcl / placement arena of the main model
config.infer_params.ngram_ram = "12GiB"       # whole n-gram tables up to 12 GiB, smallest first
config.infer_params.moe_cpu_offload = 11
model = Model.from_config(config)
model.load()
```

## How a load checks them

When `model.load()` (or `load_gen`) starts, after the placement is resolved and before any module
loads, the loader makes one plan of what this component will hold in host memory
(`model/ram_budget.py`, `plan_component`):

1. **RAM-held experts**, from the checkpoint headers (no tensor data is read): the layers the
   placement puts in RAM, or those `-mcl` / `-mcs` / `-dmcl` would take, chosen the way the
   loader chooses them (the first eligible layers in load order; eligible = an activation the
   worker implements, mul1 experts with K <= 8, per-expert biases on all experts or none; `-mcs`
   capped at E - 1 and limited by `EXL3_MOE_CPU_SPLIT_LAYERS`), counted per expert as above.
2. **Caps**: the arena against `--expert_ram` / `--draft_expert_ram`; refused with the messages
   above.
3. **N-gram tables**: which tables the budget holds whole; the others stream.
4. **One host check** of everything together (below). The result is kept for the n-gram modules
   (`infer_params.ngram_ram_plan`), which load in RAM exactly the tables the plan holds.

A load with anything in RAM, or with a budget set, prints one line, e.g.

```
 -- RAM budget (text): routed experts 52.4 GiB (-mcl 11: 11 layers), cap 56GiB; 118.3 GiB available
 -- RAM budget (text): n-gram tables 41.2 GiB in RAM (1 of 1 whole); 150.0 GiB available
```

### Host memory

What the host can still give the process is `MemAvailable` (`/proc/meminfo`; psutil elsewhere),
lowered to the room of the tightest memory cgroup (v2) on the process's path when one has a limit:
`min(memory.max, memory.high) - memory.current`, plus that cgroup's file LRU (`active_file +
inactive_file` of `memory.stat`: page cache the kernel drops before it refuses a charge; the
loader reads whole checkpoints through it). Shared memory and page-locked (pinned) memory are not
on the file LRU and stay counted. This is how a box that must leave room for other workloads can
cap a load, e.g. at 128 GiB:

```sh
systemd-run --scope -p MemoryMax=128G -p MemorySwapMax=0 python examples/chat.py -m $MODEL ...
```

`EXL3_HOST_MEM_RESERVE_MB` (default 2048) is kept free on top of every check; `0` disables the
checks. When the total does not fit, the load stops with every item named:

```
-mcl 11: ram experts=52.4 GiB needs 52.4 GiB of host memory, but 50.0 GiB is available and 2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: pinned and anonymous memory cannot be reclaimed); offload fewer layers, lower --ngram_ram, or free host memory
placement: ram experts=110.0 GiB needs 110.0 GiB of host memory, but 108.0 GiB is available (the memory cgroup /system.slice/run-x.scope allows 108.0 GiB more; MemAvailable is 150.0 GiB) and 2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: pinned and anonymous memory cannot be reclaimed); lower the budgets or use auto
```

Without swap, anonymous memory (the expert arena, n-gram tables) and pinned memory can never be
reclaimed: an oversized load does not fail at allocation, it drives the machine into the OOM
killer. The check before the load is the only clean refusal. The per-allocation guard of the
arena chunks and of a table held in RAM (`check_host_memory`) still runs as each is allocated,
with the same view of the cgroup.

## Refusals

| When | Message |
|---|---|
| a malformed size (any budget) | `expert_ram=48G: '48G' is ambiguous: write 48GiB (2^30 bytes) or 48GB (10^9 bytes)`; `ngram_ram=prompt.txt: not a size (use 0, auto, all, or a number with ...)` |
| a word a budget does not take | `expert_ram=all: all is not allowed here (use a size)`; `ngram_ram=auto: auto is not allowed here (use all or a size)` |
| `EXL3_NGRAM_STREAM=0` with `EXL3_NGRAM_RAM=6GiB` | `EXL3_NGRAM_STREAM=0 (every n-gram table in RAM) contradicts EXL3_NGRAM_RAM=6GiB; unset EXL3_NGRAM_STREAM` |
| `-der` without a draft model | `--draft_expert_ram requires a draft model (or --mtp)` |
| arena above the cap (legacy) | `-mcl 11 keeps 52.4 GiB of routed experts in RAM, more than --expert_ram 40GiB; raise the cap or offload fewer layers` |
| arena above the cap (placement) | `placement: the stream / cpu layers keep 52.4 GiB of routed experts in RAM, more than --expert_ram 40GiB; raise the cap or offload fewer layers` |
| the worker's guard | `CPU MoE: <layer> would bring the routed experts held in RAM by this component to <x>, more than --expert_ram <cap>; raise the cap or offload fewer layers` |
| the host cannot hold the total | the ledger message above |

Refusals of sizes and flags are `ValueError`s (argparse exits with the message for the flags);
the host check and the worker's guard raise `RuntimeError`.
