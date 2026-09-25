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
system RAM. The budgets say how much RAM they may take in total. The placement's storage rules
(`cuda:<n>`, `ram`, `disk`, below) extend it with the three levels of the expert tier; the flags
are caps, the placement's `ram` rule is the request.

**In this build.** The budgets, the `ram` rule's `experts=` / `ngram=` / `pagecache=` for
`stream` / `cpu` / `split` layers and n-gram tables, `hot=` on `stream` / `cpu`, `disk ngram=off`
and the dict / JSON / file forms work. The expert tier itself (`experts=cache`, the `cuda:<n>`
rule, the RAM tier's `policy=` / `demote=` / `evict=`, `disk experts=`) and `prefetch=`,
`profile=`, `disk ngram=<dir>`, `disk io=` are parsed and checked in full, then refused as not
available yet (`placement_storage.PENDING`).

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

With a placement, its `ram` rule is the request and the flags stay caps:

| | `ram experts=` (request) | `--expert_ram` (cap) | Result |
|---|---|---|---|
| neither | | | the layers take what they need |
| cap only | | `48GiB` | the layers' need, refused above 48 GiB |
| request only | `64GiB` | | refused when the `stream` / `cpu` / `split` layers need more than 64 GiB |
| both | `64GiB` | `48GiB` | refused: `placement asks ram experts=64GiB (64.0 GiB) but --expert_ram caps this component at 48GiB` |
| `all` / `auto` | `all` | `48GiB` | `all` is what the layers need: refused when that exceeds the cap |

`ram ngram=` works the same way against `--ngram_ram` (`placement asks ram ngram=8GiB but
--ngram_ram caps this component at 6GiB`; `-ngr all` caps nothing). `ram ngram=auto` takes what
the host has left after `EXL3_HOST_MEM_RESERVE_MB`, the page cache kept free (`ram pagecache=`,
default `auto` = max(8 GiB, MemTotal / 8)) and the RAM-held experts, and fills it with whole
tables as a size would. The placement names the main model's layers, so its `ram` rule applies to
the main model's text component only.

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

## The placement's storage rules

Three rules, headed by the resource they govern, join the layer rules of a placement:

```
0-11=cuda:0; 12-39=cuda:1 experts=cache hot=16; cuda:1 cache=75GiB spare=12; ram experts=48GiB ngram=6GiB; disk experts=model
```

### Grammar

```ebnf
placement   = [ rule ] , { separator , [ rule ] } ;        (* or a JSON object, or "@" path *)
separator   = ";" | newline ;
comment     = "#" , { char - newline } ;                  (* outside quotes; ignored *)

rule        = layer-rule | module-rule | device-rule | ram-rule | disk-rule ;
layer-rule  = layers , [ws] , "=" , [ws] , device , { ws , layer-attr } ;
module-rule = ( "embed" | "head" ) , [ws] , "=" , [ws] , device ;
device-rule = device , ws , device-attr , { ws , device-attr } ;
ram-rule    = "ram" , ws , ram-attr , { ws , ram-attr } ;
disk-rule   = "disk" , ws , disk-attr , { ws , disk-attr } ;

layers      = "*" | range , { [ws] , "," , [ws] , range } ;
range       = uint , [ "-" , uint ] ;
device      = "cuda:" , uint ;

layer-attr  = "experts=" , ( "vram" | "cache" | "stream" | "cpu" | "split" | "hybrid" )
            | "hot=" , ( uint | share )                          (* cache, stream, cpu, hybrid *)
            | "cpu=" , uint                                      (* split only *)
            | "prefetch=" , ( "auto" | "off" | method , { "+" , method } )
            | "profile=" , value ;
method      = ( "layer" | "router" ) , [ ":" , uint ] ;          (* depth, default 1 *)

device-attr = "cache=" , ( "auto" | size )
            | "spare=" , uint
            | "evict=" , ( "lru" | "lfu" )
            | "admit=" , ( "adaptive" | "heat" | "always" ) ;

ram-attr    = "experts=" , budget
            | "ngram=" , budget
            | "pagecache=" , ( "auto" | size )
            | "policy=" , ( "lazy-exclusive" | "exclusive" | "inclusive" )
            | "demote=" , ( "swap" | "heat" | "all" | "off" )
            | "evict=" , ( "lfu" | "lru" ) ;

disk-attr   = "experts=" , source
            | "ngram=" , source
            | "io=" , ( "auto" | "direct" | "buffered" ) ;
source      = "model" | "off" | value ;                         (* value: a directory *)

budget      = "auto" | "all" | size ;                           (* "0" is a size *)
size        = number , [ unit ] ;                               (* no unit: GiB *)
unit        = "B" | "KiB" | "Ki" | "KB" | "MiB" | "Mi" | "MB"
            | "GiB" | "Gi" | "GB" | "TiB" | "Ti" | "TB" ;        (* case-insensitive; bare K/M/G/T refused *)
share       = number , "%" ;
number      = uint , [ "." , digit , { digit } ] ;
uint        = digit , { digit } ;                               (* ASCII digits *)
value       = bare | quoted ;
bare        = vchar , { vchar } ;                               (* no space, ";", "#", '"' *)
quoted      = '"' , { char - ( '"' | "\" ) | "\" , char } , '"' ;
ws          = ( " " | tab ) , { " " | tab } ;
```

Keywords and keys are case-insensitive; values that name things (paths, profiles) keep their
case.

### `cuda:<n>`: the expert cache of one GPU

| Attribute | Default | Meaning |
|---|---|---|
| `cache=` | `auto` | the VRAM pool of this GPU's `experts=cache` layers, shared by all of them (one global least-recently-used pool per GPU). `auto` = what the device has left inside its `-gs` budget after the placed modules, the Cache, the generator's statics, the prefill staging, `EXL3_AUTOSPLIT_MARGIN_MB` and `EXL3_MOE_TIER_HEADROOM_MB`. `0` = no cache (every miss streams through staging). A size is at least top-k + `spare` slots. `all` is refused: write `experts=vram` |
| `spare=` | `8` | free slots kept ready, so a promotion never waits on a demotion's copy |
| `evict=` | `lru` | which slot a promotion replaces: `lru` (least recently used, the slots of the current step protected) or `lfu` (lowest decayed heat) |
| `admit=` | `adaptive` | which decode misses enter the cache: `adaptive` (with a probability adapted so the VRAM and RAM tiers' cache lives match), `heat` (when warmer than the slot it would replace), `always` |

### `ram`: the component's RAM budgets and the RAM tier

| Attribute | Default | Meaning |
|---|---|---|
| `experts=` | the `--expert_ram` cap; else `auto` for `cache` layers; else what the layers need | system RAM for the component's routed experts: the static arenas of its `stream` / `cpu` / `split` layers first, the rest is the RAM tier of its `cache` layers. `all` = what the layers need (with `cache` layers: every cached expert outside VRAM, so none is read from disk after load); `auto` = the smaller of `all` and what the host has left after the reserve, the page cache and the other budgets; `0` = no RAM tier |
| `ngram=` | the `--ngram_ram` cap, else `0` | system RAM for n-gram tables: `0` streams every row, `all` holds every table whole, a size the tables that fit (whole, smallest first), `auto` what the host has left |
| `pagecache=` | `auto` = max(8 GiB, MemTotal / 8) | RAM that `auto` budgets leave unpinned (the page cache is the n-gram rows' warm tier); it does not limit explicit sizes. `all` is refused |
| `policy=` | `lazy-exclusive` | the RAM tier against the VRAM cache: `lazy-exclusive` (a promotion leaves the RAM copy as a reclaimable duplicate), `exclusive` (a promotion frees the RAM copy), `inclusive` (RAM keeps what it holds) |
| `demote=` | `swap` | what happens to a VRAM victim with no RAM copy: `swap` (copied back only into the RAM slot its replacement left, and only when warmer than RAM's coldest), `heat` (into any free slot, or over RAM's coldest when warmer), `all` (every victim), `off` (dropped; the disk still has it) |
| `evict=` | `lfu` | RAM tier victims: `lfu` (lowest heat of a 16-entry sample) or `lru` |

### `disk`: where experts and n-gram rows are read

| Attribute | Default | Meaning |
|---|---|---|
| `experts=` | `model` | `model`: the checkpoint's shards, read in place. `<dir>`: a copy on another drive. `off`: never read an expert at runtime; the load refuses unless VRAM and RAM hold every cached expert, and every victim returns to RAM |
| `ngram=` | `model` | the same for n-gram tables; `off` needs `ram ngram=all` |
| `io=` | `auto` | `auto`: experts `O_DIRECT` (buffered if refused), n-gram rows buffered; `direct` / `buffered` force one mode |

### Checks the grammar cannot express (parse time)

- Coverage as for layer rules: every layer placed exactly once.
- One kind of host-held experts per GPU: GPU-computed (`stream`, `cache`) and CPU-computed
  (`cpu`, `split`) layers on different GPUs.
- `hot=` applies to `cache`, `stream`, `cpu` (and is required by `hybrid`); `cpu=` only to
  `split`; `prefetch=` only to `cache` and `stream`; `profile=` not to `vram`.
- One `cuda:<n>` rule per GPU, one `ram` rule, one `disk` rule. A `cuda:<n>` rule needs an
  `experts=cache` layer on that GPU; `cache=0` takes no `spare=` / `evict=` / `admit=`;
  `spare=0` needs `ram demote=off` or `policy=inclusive` (no free slot would take a demoted
  victim).
- `ram experts=` needs a layer that keeps experts in RAM. `policy=`, `demote=`, `evict=` need a
  `cache` layer and a RAM tier (`experts=0` has none). `demote=` has no effect with
  `policy=inclusive` and is refused there. At most one of `experts=` and `ngram=` can be `auto`.
- `disk experts=<dir|off>` needs a `cache` layer. `disk experts=off` refuses `demote=off` unless
  the policy is `inclusive`. `disk ngram=off` needs `ngram=all`. `io=` with both `experts=off` and
  `ngram=off` has nothing to configure.
- Storage rules need layer rules: without a placement, the flags carry the budgets.

### Canonical form

Rules in the order `embed`, layer rules by first layer, `*`, `head`, `cuda:<n>` by index, `ram`,
`disk`; attributes in a fixed order (layer `experts hot cpu prefetch profile`; device `cache
spare evict admit`; ram `experts ngram pagecache policy demote evict`; disk `experts ngram io`);
defaults omitted, and a storage rule left with only defaults dropped; sizes in the largest exact
unit; `hybrid hot=K` as `cpu hot=K`; values quoted only when needed. Placements are equal when
their canonical texts are, and the canonical text, the dict form and the JSON form all parse back
to an equal placement. Placements without storage rules print exactly as before.

### Examples

| # | Meaning | Canonical form | In this build |
|---|---|---|---|
| T1 | one GPU, every expert in VRAM | `*=cuda:0` | yes |
| T3 | one GPU, 32 of 128 experts per layer resident, the rest on the CPU worker (`-mcs 96`) | `*=cuda:0 experts=cpu hot=32` | yes |
| T3c | the same, in the older spelling (`experts=hybrid hot=32`, normalized) | `*=cuda:0 experts=cpu hot=32` | yes |
| T4 | DeepSeek-V4.1 serving layout, CMP + PRO | `0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1` | yes |
| T5 / (c) | V4.1, CMP + PRO: layers 12-39 cached on the PRO over RAM that holds the rest; the disk never read for experts | `0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off` | no |
| (a) | the PRO alone, RAM for every expert outside VRAM | `*=cuda:0 experts=cache; ram experts=all; disk experts=off` | no |
| (b) | the PRO alone, 96 GiB of RAM tier, the SSD for the rest | `*=cuda:0 experts=cache; ram experts=96GiB` | no |
| T6 | the PRO alone: 75 GiB cache, 96 GiB RAM tier, n-gram budget | `*=cuda:0 experts=cache; cuda:0 cache=75GiB; ram experts=96GiB ngram=6GiB` | no |
| T7 | a RAM-limited box: 16 experts per layer pinned; the tier takes what is free after 24 GiB of page cache | `*=cuda:0 experts=cache hot=16; ram experts=auto pagecache=24GiB` | no |
| T8 | no RAM tier: the VRAM cache straight over the SSD | `*=cuda:0 experts=cache; ram experts=0` | no |
| T10 | a big-RAM host: inclusive RAM, whole n-gram tables | `*=cuda:0 experts=cache; ram experts=all ngram=all policy=inclusive` | no |
| T11 | every policy knob | `*=cuda:0 experts=cache; cuda:0 cache=40GiB spare=12 evict=lfu admit=heat; ram experts=64GiB pagecache=16GiB policy=exclusive demote=heat evict=lru; disk io=direct` | no |
| T12 | no D2H copies at all | `*=cuda:0 experts=cache; ram experts=48GiB demote=off` | no |
| T13 | CPU-computed layers on one GPU, cached layers on the other | `0-9=cuda:0 experts=cpu; 10-39=cuda:1 experts=cache; cuda:1 cache=40GiB; ram experts=96GiB` | no |
| T14 | experts and n-gram tables read from other drives | `*=cuda:0 experts=cache; ram experts=48GiB; disk experts=/nvme1/v41 ngram="/mnt/engram disk/v41"` | no |
| T16 | GPU streaming with a resident slice (`-mcs k -mcm stream_only`) | `*=cuda:1 experts=stream hot=64` | yes |
| T17 | units: `cache=1.5`, `ram experts=48GB ngram=512mi` | `*=cuda:0 experts=cache; cuda:0 cache=1536MiB; ram experts=48GB ngram=512MiB` | no |
| T18 | an n-gram budget only, for a PLE model | `*=cuda:0; ram ngram=12GiB` | yes |
| R1 | the RAM of stream layers capped by the placement | `0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1; ram experts=64GiB` | yes |
| R2 | whole n-gram tables, never read from disk | `*=cuda:0; ram ngram=all; disk ngram=off` | yes |

A multi-line value with comments; the written-out defaults vanish from the canonical form
(`0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=52GiB`):

```sh
export EXL3_PLACEMENT="
0-11  = cuda:0                                  # CMP: resident
12-39 = cuda:1 experts=cache                    # PRO: cached
cuda:1 cache=auto spare=8 evict=lru admit=adaptive
ram experts=52GiB pagecache=auto policy=lazy-exclusive demote=swap evict=lfu
disk experts=model ngram=model io=auto
"
```

### Parse-time refusals

Each is a `ValueError`, prefixed with `placement rule <n> '<rule>': ` where one rule is at fault:

| Written | Message |
|---|---|
| `rams experts=48GiB` | `unknown rule 'rams' (did you mean 'ram'?) (a rule starts with layers (7, 0-11, 0-3,8-11), '*', 'embed', 'head', 'cuda:<n>', 'ram' or 'disk')` |
| `...; cuda:0=75GiB` | `cuda:0 takes attributes, not '=<value>': write e.g. 'cuda:0 cache=75GiB'` |
| `*=cuda:0 experts=cache cache=75GiB` | `unknown attribute 'cache' for a layer rule (expected experts, hot, cpu, prefetch, profile); cache= belongs in a cuda:<n> rule` |
| `*=cuda:0 experts=cache hott=16` | `unknown attribute 'hott' for a layer rule (expected experts, hot, cpu, prefetch, profile) (did you mean 'hot'?)` |
| `...; ram expert=48GiB` | `unknown attribute 'expert' for the ram rule (expected experts, ngram, pagecache, policy, demote, evict) (did you mean 'experts'?)` |
| `...; ram experts=48 GiB` | `'GiB' must be key=value (experts, ngram, pagecache, policy, demote, evict) (sizes are written without spaces: 48GiB)` |
| `...; ram experts=48G` | `experts=48G: '48G' is ambiguous: write 48GiB (2^30 bytes) or 48GB (10^9 bytes)` |
| `*=cuda:0 experts=cahce` | `experts='cahce' must be one of vram, cache, stream, cpu (or the older split cpu=<k> / hybrid hot=<k>) (did you mean 'cache'?)` |
| `*=cuda:0 hot=16` | `hot= applies to experts=cache, stream and cpu (experts=vram keeps every routed expert in VRAM)` |
| `*=cuda:0 experts=split cpu=96 hot=32` | `hot= applies to experts=cache, stream and cpu (experts=split counts the other side: cpu=<k>)` |
| `*=cuda:0 experts=hybrid` | `experts=hybrid needs hot=<routed experts per layer kept in VRAM> (the same as experts=cpu hot=<k>)` |
| `*=cuda:0 experts=cpu cpu=96` | `cpu= only applies to experts=split (write experts=cpu hot=<k> to keep k routed experts per layer in VRAM)` |
| `*=cuda:0 experts=cache hot=100%` | `hot=100%: a share must lie between 0% and 100%, exclusive` |
| `*=cuda:0 experts=cpu prefetch=layer` | `prefetch= applies to experts=cache and stream (experts=cpu computes on the CPU worker)` |
| `*=cuda:0 profile=code` | `profile= chooses hot or cached experts; experts=vram has none to choose` |
| `...; cuda:0 cache=all` | `cache=all would hold every expert of its layers: write experts=vram on them` |
| `...; cuda:0 cache=0 spare=4` | `cache=0 has no slots for spare= / evict= / admit= to govern` |
| `...; cuda:0 spare=0` | `placement: cuda:0 spare=0 leaves no free slot for a demoted victim; spare=0 needs ram demote=off or policy=inclusive` |
| `*=cuda:0 experts=cache; cuda:1 cache=8GiB` | `placement: 'cuda:1 cache=8GiB' configures an expert cache, but no experts=cache layer is placed on cuda:1` |
| `...; cuda:0 cache=8GiB; cuda:0 spare=4` | `cuda:0 is set twice (rules 2 and 3); merge them into one rule` |
| `*=cuda:0; ram experts=48GiB` | `placement: 'ram experts=48GiB' but no layer keeps routed experts in system RAM (every rule is experts=vram)` |
| `*=cuda:0 experts=cpu; ram experts=64GiB policy=exclusive` | `placement: ram policy= governs the RAM tier of experts=cache layers, and there are none` |
| `...; ram experts=0 policy=inclusive` | `policy= governs the RAM tier, which experts=0 disables` |
| `...; ram experts=auto ngram=auto` | `experts=auto and ngram=auto: at most one RAM budget can be auto` |
| `...; ram experts=48GiB policy=inclusive demote=swap` | `demote= has no effect with policy=inclusive (every VRAM-cached expert keeps its RAM copy, so a victim is simply dropped); remove it` |
| `...; ram experts=auto pagecache=all` | `pagecache=all: all is not allowed here (use auto or a size)` |
| `*=cuda:0 experts=cpu; disk experts=/nvme1/v41` | `placement: 'disk experts=/nvme1/v41' applies to experts=cache layers (stream and cpu layers keep every expert in RAM), and there are none` |
| `*=cuda:0; ram ngram=8GiB; disk ngram=off` | `placement: 'disk ngram=off' never reads n-gram rows from disk, which needs ngram=all, not ngram=8GiB` |
| `...; ram experts=all demote=off; disk experts=off` | `placement: 'disk experts=off' keeps no copy of an expert outside VRAM and RAM, so demote=off would lose VRAM victims; drop demote=off (with the disk off, every victim moves back to the RAM slot its replacement left) or use policy=inclusive with ram experts=all` |
| `0-19=cuda:0 experts=cpu; 20-39=cuda:0 experts=cache` | `placement: cuda:0 holds both GPU-computed (experts=stream / cache) and CPU-computed (experts=cpu / split) layers; one kind of host-held experts per GPU is supported` |
| `ram ngram=6GiB` | `placement: storage rules (cuda:<n>, ram, disk) need layer rules to go with; without a placement, set RAM budgets with --expert_ram / --ngram_ram` |
| `*=cuda:0 experts=cache` (in this build) | `placement '*=cuda:0 experts=cache': experts=cache is not available in this build yet; it needs the expert tier runtime (a VRAM expert cache per GPU over the RAM tier and the disk) (see doc/expert_tiers.md)` |

`tests/test_placement_tiers_.py` holds every case with its full text.
