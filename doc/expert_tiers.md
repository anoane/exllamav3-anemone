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

## The expert tier

**Not available in this build yet**: `experts=cache` is parsed and checked, then refused. This
section describes what the words ask for, how a load sizes the tier and what it refuses, as
`model/expert_tier_config.py` implements it (its tests, `tests/test_expert_tier_config_.py`, run
every number below).

### Where a routed expert of a `cache` layer lives

| Place | Holds | Sized by |
|---|---|---|
| VRAM, `hot=` pins | `hot` experts per layer, never evicted | the layer (counted as its weights) |
| VRAM, the GPU's cache | a changing set, shared by every `cache` layer on that GPU (one least-recently-used pool per GPU) | `cuda:<n> cache=` |
| VRAM, prefill staging | the next layer's experts during a long prefill | `EXL3_MOE_TIER_STAGING` |
| RAM, the tier | the warmest experts VRAM does not hold (`policy=lazy-exclusive` / `exclusive`), or a copy of what it holds too (`inclusive`) | `ram experts=` |
| RAM, the disk slab | SSD reads that bypass the tier | `EXL3_MOE_TIER_DISK_SLAB` |
| SSD | every expert: the checkpoint's shards, read in place, never written | `disk experts=` |

With `disk experts=off` nothing is read from the SSD after the load: VRAM and RAM must hold every
cached expert, and every VRAM victim moves back to RAM (demotion is then unconditional).

### Sizing a tiered load

Per GPU with `cache` layers, `n` = its cached experts (E - hot, summed over its `cache` layers),
`vram_slot` = one expert's three trellis tensors (13,271,040 bytes for DeepSeek-V4.1 3.0 bpw):

```
staging = (2 with EXL3_MOE_TIER_STAGING=double, else 1) x max over its cache layers of (E - hot) slots
room    = what the GPU has left inside its -gs budget after the placed modules, the Cache and the
          loader's margins  -  staging  -  EXL3_MOE_TIER_HEADROOM_MB
S       = min(n, room / vram_slot)             cache=auto
S       = min(n, size / vram_slot)             cache=<size>, refused when size > room
held    = n if S == n (everything fits: no spare needed), else S - spare
```

Then for the component, `N` = cached experts on every GPU, `ram_slot` = one expert's extent in the
checkpoint rounded up to 4 KiB plus one page for an unaligned start (13,320,192 bytes):

```
tier_all = N - sum(held)          lazy-exclusive, exclusive (inclusive: N)
want     = static arenas + tier_all x ram_slot                  ram experts=all
         = min(that, available - reserve - pagecache - ngram)   ram experts=auto (the default), within --expert_ram
         = the size                                             ram experts=<size>
R        = min(tier_all, (want - static arenas) / ram_slot)     RAM tier slots
disk-only experts = N - sum(held) - R                           (inclusive: N - R)
```

`ram experts=` covers the `stream` / `cpu` / `split` layers' arenas first; the tier gets the rest.
The disk slab (8 slots by default) is pinned too when `disk experts=` is not `off`. The host check
covers the static arenas, the tier, the n-gram tables and the slab together.

A load checks what it can before any module loads (explicit `cache=` and `ram experts=` sizes,
the settings, the host memory they need), and sizes the pools at the end of the layer split, with
each GPU's room, before any of the tier is allocated. It prints:

```
 -- expert tiers: cuda:1 cache 6553 slots (81.0 GiB, 8 spare, evict=lru admit=adaptive; staging 768 slots (9.5 GiB, double))
    ram experts 52.2 GiB pinned (static 0 MiB + tier 4207 slots), policy lazy-exclusive demote=swap, evict=lfu
    ngram 0 MiB (every row streams from disk)
    disk experts=off io=auto: 0 experts on disk only; 73.8 GiB of RAM left unpinned
```

### Where an expert is read from: extents and slot geometry

The tier never repacks the checkpoint. Each routed expert of a `cache` layer is read in place
from its shard (`model/expert_extents.py`, `build_extent_index`, from the headers alone):

- **One extent per expert.** An expert's tensors (per projection: `suh`, `svh`, `mul1`,
  `trellis`, and `bias` when present) are written back to back by `convert.py`, so the extent from
  the first byte of the first to the last byte of the last is one read: 13,315,596 bytes for a
  DeepSeek-V4.1 3.0 bpw expert. The trellis tensors lie inside it at offsets that are not sector
  aligned; the index records them.
- **Split experts.** When the tensors of one expert are in different files (an override
  collection replaced one projection) or another tensor lies between them, the expert is read as
  one piece per projection, each piece its trellis alone. Such an expert costs three reads instead
  of one; nothing else changes.
- **Slot geometry**, one per set of cache layers on a GPU (their experts must have the same
  trellis sizes):

| Name | What it holds | V4.1 3.0 bpw |
|---|---|---|
| VRAM slot | the gate, up and down trellis back to back (the compact layout) | 13,271,040 B (3 x 4,423,680) |
| projection offsets in a compact slot | `(0, gate, gate + up)` | `(0, 4423680, 8847360)` |
| RAM slot | one extent read: each piece rounded up to 4 KiB, plus 4 KiB for a start that is not 4 KiB aligned | 13,320,192 B (3,252 pages), 80 per 1 GiB chunk |

The disk engine reads the 4 KiB-aligned superset of an extent into a 4 KiB-aligned slot with
`O_DIRECT` and reports where the payload landed (the file offset modulo 4 KiB); the trellis of
projection `p` is then at `slot + payload offset + sub-offset[p]`. A promotion copies those three
ranges into the compact VRAM slot.

### Policies: what the tier decides

Every decision of the tier is made by one set of rules, written once as a reference
(`model/expert_tier_policy.py`) and reproduced exactly by the native core
(`exllamav3_ext/tier/tier_policy.cpp`). The rules depend only on the sequence of calls, never on
timing, so a replay of the same calls gives the same records, the same copies and the same final
contents. Tier state changes where an expert's bytes come from, never the arithmetic: a layer
computes the same logits whatever the cache holds.

**Keys and sizes.** A cached expert is `lc * E + e`: `lc` numbers the `cache` layers of one GPU in
forward order, `e` is the routed expert. `S` = the GPU's pool slots, `spare` its spare slots,
`C = S - spare` the experts the pool holds in steady state, `R` = the RAM tier's slots.

**One call of a cache layer** (the lookup):

1. Calls are numbered (`seq`), one per call of any cache layer on the GPU. The call's routed
   experts are deduplicated in order of first occurrence (`U`), with their counts.
2. **Heat** (below) grows by the count: 1 per decode assignment, `EXL3_MOE_HEAT_PREFILL` per
   prefill assignment.
3. **Protection** (decode calls only): every expert of the call that the pool holds gets
   `stamp = seq` before any victim is chosen, so a call never evicts an expert it uses.
4. For each expert of `U` in order:
   - held by the pool: a **hit**;
   - a prefill call (routed mode, fewer rows than `EXL3_MOE_TIER_PREFILL_ROWS`): a
     **transient**: it is copied into the staging for this call only;
   - a decode call: **admitted** while the pool holds fewer than `C` experts, otherwise as
     `admit=` says. An admitted expert takes the lowest free slot; with none free it becomes a
     transient (counted as *starved*). When the pool then holds more than `C` experts, one
     **victim** is retired and paired with the admission.
5. The call's record lists every expert of `U`: hit, admit (with its victim), or transient.

| `cuda:<n> admit=` | A decode miss enters the pool when |
|---|---|
| `adaptive` (default) | a draw keyed on the access counter falls under the probability `p`; `p` adapts every `EXL3_MOE_TIER_ADAPT_EVERY` accesses so the VRAM and RAM tiers' cache lives match (Gill's PROMOTE), between `EXL3_MOE_TIER_ADMIT_PMIN` and `C / (C + R)`; `EXL3_MOE_TIER_ADMIT_P` pins it |
| `heat` | its heat is at least the would-be victim's |
| `always` | always |

| `cuda:<n> evict=` | The victim is the pool slot, among those not used by this call, with |
|---|---|
| `lru` (default) | the oldest stamp, then the lowest slot |
| `lfu` | the least heat, then the oldest stamp, then the lowest slot |

A victim's bytes stay intact while it is *retiring*: its slot returns to the free list only after
its copy back to RAM (if any) has landed, so a spare slot is always ready for the next admission
and an admission never waits on a demotion. `spare=0` (allowed with `ram demote=off` or
`policy=inclusive`) replaces victims in place instead.

**What the host does with a record**, entry by entry (every miss, then its paired victim):

| A miss is read from | when |
|---|---|
| a retiring VRAM slot that still holds it (device to device) | its demotion is still in flight |
| its RAM tier slot (host to device, three copies from an SSD-filled slot, one from a demoted one) | RAM holds it |
| the SSD (a class-1 read of the disk engine, into a RAM slot or the disk slab) | otherwise |

| What RAM keeps | `exclusive` | `lazy-exclusive` (default) | `inclusive` |
|---|---|---|---|
| admitted, was in RAM | the RAM slot is freed | the RAM copy becomes a reclaimable duplicate | the RAM copy becomes a duplicate |
| admitted, read from the SSD | nothing (read into the slab) | a duplicate, in a free slot or over a reclaimed duplicate; else nothing | a duplicate, in a free slot or over the coldest owner |
| transient, read from the SSD (decode) | RAM admission: a free slot, else a reclaimed duplicate, else over the coldest sampled owner (`EXL3_MOE_TIER_RAM_ADMIT=heat`: only when warmer); else the slab | same | same, without reclaiming duplicates |
| transient, read from the SSD (prefill) | the slab; a refill candidate (below) | same | same |

| The victim (`ram demote=`) | Rule |
|---|---|
| has a duplicate in RAM | the duplicate becomes the owner; nothing is copied |
| `swap` (default) | copied back only into the RAM slot its admission freed or made a duplicate, and only when it is warmer than RAM's coldest sampled owner; else dropped: at most one copy back per promotion from RAM |
| `heat` | copied into a free slot, else over a reclaimed duplicate, else over the coldest sampled owner when warmer; else dropped |
| `all` | always copied back (a free slot, else a reclaimed duplicate, else over the coldest sampled owner) |
| `off`, or `EXL3_MOE_TIER_DEMOTE=0` | dropped (the SSD has it) |
| any, with `disk experts=off` | always copied back: into the slot its admission left, else a free slot, else a reclaimed duplicate |

`inclusive` never copies a victim back: RAM always has it. A copied-back victim's RAM slot holds
its three trellis tensors back to back (compact layout).

- **Coldest sampled owner**: the least heat (`ram evict=lfu`, default) or the oldest RAM stamp
  (`ram evict=lru`; set when an expert enters RAM and when a decode call reads it), then the
  lowest key, over `EXL3_MOE_TIER_SAMPLE` draws from the owners (all of them when there are no
  more than that).
- **Reclaimed duplicate**: among sampled duplicates, the one whose VRAM copy was used last (the
  last to lose it), then the lowest key.
- **Refills** (`EXL3_MOE_TIER_REFILL`): an expert a prefill call read from the SSD past the RAM
  tier is read again into RAM in the background (class 3) when a free slot or a duplicate is
  available, or when it is warmer than RAM's coldest sampled owner; at most
  `EXL3_MOE_TIER_REFILL_INFLIGHT` in flight (refills of earlier calls that are still reading
  count; in deterministic mode they have all landed), the others are dropped, never queued. Decode
  misses already went through RAM admission.
- **Layer mode** (prefill calls from `EXL3_MOE_TIER_PREFILL_ROWS` rows): every cached expert of
  the layer that the pool does not hold is staged (from RAM, or read from the SSD into the slab);
  no stamps, no admission, no demotion; heat grows.

**Heat** is a decayed routing count per key, kept in 16.16 fixed point with the epoch it was last
written in. The epoch is `decode tokens // EXL3_MOE_HEAT_HALFLIFE` (one decode pass = one token);
reading a key halves its value once per epoch since (lazily, no sweep). A decode assignment adds
1.0, a prefill assignment `EXL3_MOE_HEAT_PREFILL` (1/16): a 4K-token prompt adds about 4 to every
expert of a layer, a few tokens of decode.

**Cold fill** (end of the load): in priority order (round robin across layers by expert index:
expert 0 of every layer, then expert 1, ...), the first `C` experts fill the pool (all of them when
they fit), the next `R` the RAM tier (`inclusive`: RAM first takes a duplicate of every pool
expert). Nothing else is read after the load unless a decode or prefill call misses.

**Checked in the tests** (`tests/test_expert_tier_policy_.py`): nine micro-scenarios with every
record, counter and final state; invariants after every call of 200 random configurations x 5,000
calls; agreement with the policy simulator the defaults were chosen on; the adaptive rule on
scripted cache lives, exactly. The native core (`tests/test_expert_tier_native_.py`) makes the
same decisions as the reference on the micro-scenarios and 200 random configurations: the same
records, actions, counters and final state, heat, random draws and the adaptive probability
included; its standalone driver (`tests/expert_tier/build.sh`, `tier_policy_test`) checks the
micro-scenarios and 10^6 random calls with every invariant and a mirror directory that applies
the records (the tier thread's view of the lookup kernel's directory), under AddressSanitizer,
UndefinedBehaviorSanitizer and ThreadSanitizer.

### The three test layouts (DeepSeek-V4.1-Flash 3.0 bpw on the AI VM)

| | (a) PRO + RAM for every expert outside VRAM | (b) PRO + 96 GiB RAM tier + SSD | (c) CMP resident + PRO cache + RAM, no expert SSD reads |
|---|---|---|---|
| Devices | `CUDA_VISIBLE_DEVICES=0` | same | `1,0` (cuda:0 = CMP, cuda:1 = PRO) |
| Placement | `*=cuda:0 experts=cache; ram experts=all; disk experts=off` | `*=cuda:0 experts=cache; ram experts=96GiB` | `0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off` |
| Pool (65.5 / 71.5 GiB estimated) | 5,299 slots | 5,299 slots | 5,784 slots |
| RAM tier | 10,069 slots, 124.9 GiB | 7,738 slots, 96 GiB | 4,976 slots, 61.7 GiB |
| SSD-only experts | 0 | 2,331 (28.9 GiB) | 0 |
| Under a 128 GiB cgroup | refused (the refusal names the cgroup) | fits | fits |
| Engram tables | disk | disk | disk |

The CMP 170HX's x1 link (~0.4 GB/s) is below `EXL3_MOE_TIER_MIN_LINK_GBS`: it holds resident
layers only, as in (c).

### Load-time refusals of the tier

| Situation | Message |
|---|---|
| `disk experts=off`, tiers too small | `placement: disk experts=off needs every cached expert in VRAM or RAM, but 1628 of 10752 fit in neither (6545 VRAM slots besides the spare ones, 2579 RAM slots); raise ram experts= by 20.2 GiB or allow disk reads` |
| budgets above the host memory | `placement: ram experts=96.0 GiB + ngram=6.0 GiB + disk slab=102 MiB need 102.1 GiB of host memory, but 53.0 GiB is available and 2.0 GiB is kept free (EXL3_HOST_MEM_RESERVE_MB) (no swap: pinned and anonymous memory cannot be reclaimed); lower the budgets or use auto` (with the cgroup when it is the limit) |
| cache below its minimum | `placement: cuda:0 cache=100MiB holds 7 expert slots; experts=cache needs at least top-k (6) + spare (8) = 14 slots (177 MiB), or cache=0` |
| cache above the device's room | `placement: cuda:0 cache=90GiB does not fit: 80.0 GiB is free on cuda:0 after the placed modules, the cache and the margins (the prefill staging, 9.5 GiB with EXL3_MOE_TIER_STAGING=double, and EXL3_MOE_TIER_HEADROOM_MB=1024 are set aside first)` |
| staging leaves too little | `placement: cuda:0 has 100 MiB left for the expert cache after the placed modules, the margins and the prefill staging (9.5 GiB, EXL3_MOE_TIER_STAGING=double); an expert cache needs at least top-k + spare = 14 slots (177 MiB); set EXL3_MOE_TIER_STAGING=single, raise -gs on cuda:0, or move layers` |
| request above the flag | `placement asks ram experts=64GiB (64.0 GiB) but --expert_ram caps this component at 48GiB` |
| static arenas above the request | `placement: the stream / cpu layers keep 47.6 GiB of routed experts in RAM, more than ram experts=40GiB; raise it or move layers to experts=cache` |
| a tier smaller than one decode step | `placement: ram experts=50MiB leaves the RAM tier 3 slots, fewer than one decode step uses (top-k = 6); raise it, or use ram experts=0 (a VRAM cache straight over the disk)` |
| `inclusive` too small | `placement: policy=inclusive keeps a RAM copy of every expert the VRAM cache holds, so ram experts= must hold more than the cache's 6472 + 8 slots (80.4 GiB); it holds 5159 (64.0 GiB)` |
| a slow host link | `placement: cuda:0 reaches host memory at 0.38 GB/s (measured at load), below the 2.0 GB/s an expert cache needs (EXL3_MOE_TIER_MIN_LINK_GBS); keep its layers experts=vram, or set EXL3_MOE_TIER_MIN_LINK_GBS=0 to allow it` |
| `EXL3_MOE_TIER_DEMOTE=0` with `disk experts=off` | `EXL3_MOE_TIER_DEMOTE=0 would drop VRAM victims, but disk experts=off keeps no other copy of them` |
| two expert shapes on one GPU | `placement: the experts=cache layers on cuda:1 have different expert shapes (...); one expert slot size per GPU is supported` |
| a layer the tier cannot serve | `placement: <layer> cannot use experts=cache: it needs the fused MoE kernel (EXL3 mul1 codebook shared by gate, up and down; silu, silu_ref or gelu gated, or relu2 gateless; no per-expert biases; unpadded dims; at most 512 experts); use experts=vram, stream or cpu for this layer` |
| not Linux | `placement: experts=cache needs the disk engine, which is Linux-only` |

### Tuning

The placement says what the tier does; `EXL3_MOE_TIER_*` and `EXL3_MOE_HEAT_*` say how. Every
variable is described in [env_vars.md](env_vars.md) ("Expert tiers"); a load checks them all
when it starts (a malformed value or an unknown name under those prefixes is refused, with a
did-you-mean hint) and prints the ones that differ from their default.

| Variable | Default | Values |
|---|---|---|
| `EXL3_MOE_TIER_DEMOTE` | `1` | `0` drops every VRAM victim (master switch of `ram demote=`) |
| `EXL3_MOE_TIER_STAGING` | `double` | `double`, `single` |
| `EXL3_MOE_TIER_PREFILL_ROWS` | `256` | rows from which a call copies whole layers (2 .. 2^20, above the decode rows) |
| `EXL3_MOE_TIER_DECODE_ROWS` | `8` | rows up to which a call may admit (1 .. 8) |
| `EXL3_MOE_HEAT_HALFLIFE` | `256` | decode tokens per halving of the heat |
| `EXL3_MOE_HEAT_PREFILL` | `0.0625` | heat of a prefill assignment (0 .. 1) |
| `EXL3_MOE_HEAT_FILE` | unset | path |
| `EXL3_MOE_TIER_ADMIT_P` | unset | pins the adaptive probability (0 .. 1) |
| `EXL3_MOE_TIER_ADAPT_EVERY` | `2048` | accesses per adaptation |
| `EXL3_MOE_TIER_ADMIT_PMIN` | `0.05` | lower bound of the probability |
| `EXL3_MOE_TIER_SAMPLE` | `16` | RAM entries sampled for the coldest |
| `EXL3_MOE_TIER_RAM_ADMIT` | `always` | `always`, `heat` |
| `EXL3_MOE_TIER_REFILL` | `1` | `0` turns refills off |
| `EXL3_MOE_TIER_REFILL_INFLIGHT` | `2` | 1 .. 64 |
| `EXL3_MOE_TIER_DISK_SLAB` | `auto` | slots, 1 .. 4096 |
| `EXL3_MOE_TIER_HEADROOM_MB` | `1024` | MiB |
| `EXL3_MOE_TIER_MIN_LINK_GBS` | `2.0` | GB/s; `0` allows any link |
| `EXL3_MOE_TIER_SPIN_US` | `-1` | `-1` while a pass is active, `0` never, N us |
| `EXL3_MOE_TIER_AFFINITY` | unset | CPU list |
| `EXL3_MOE_TIER_DETERMINISTIC` | `0` | `1` applies every call before the next |
| `EXL3_MOE_TIER_VERIFY` | `0` | `1` checks invariants after every call |
| `EXL3_MOE_TIER_TRACE` | unset | path of the call-record trace |

The policies themselves (`admit=`, `evict=`, `policy=`, `demote=`) are words of the placement, so
one string says what a load does; the defaults (`admit=adaptive`, `evict=lru`,
`policy=lazy-exclusive`, `demote=swap`, `ram evict=lfu`, `spare=8`) are those of the policy
simulation until measurements on the target host replace them.
