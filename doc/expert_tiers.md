# Expert tiers and RAM budgets

A large MoE model rarely fits in GPU memory whole. Its routed experts can live in three places: the
GPU's memory (VRAM), system RAM, and the checkpoint on the SSD. This document covers the budgets
that bound how much system RAM a load may take for routed experts and for n-gram tables, how a load
checks them against the host before it allocates anything, and the expert tier: a VRAM cache per
GPU over a RAM tier over the SSD, for the `experts=cache` layers of a placement.

It is the reference for:

| Knob | CLI (`model_init`) | Environment | Python (`config.infer_params`) | Default |
|---|---|---|---|---|
| RAM for the routed experts of the main model | `-er` / `--expert_ram <size>` | `EXL3_EXPERT_RAM` | `expert_ram` | no cap |
| RAM for the routed experts of a draft model or MTP head | `-der` / `--draft_expert_ram <size>` | none | `draft_expert_ram` | no cap |
| RAM for n-gram tables (PLE models, DeepSeek-V4.1's engram tables) | `-ngr` / `--ngram_ram [<size>]` | `EXL3_NGRAM_RAM`, and the older `EXL3_NGRAM_STREAM` | `ngram_ram` (and the older `ngram_stream_from_disk`) | `0`: every row streams from disk |
| Host memory kept free by every check | none | `EXL3_HOST_MEM_RESERVE_MB` | none | `2048` |

The explicit placement ([placement.md](placement.md)) says which layers keep their experts in
system RAM. The budgets say how much RAM they may take in total. The placement's storage rules
(`cuda:<n>`, `ram`, `disk`, below) extend it with the three levels of the expert tier; the flags
are caps, the placement's `ram` rule is the request. Every word of the grammar runs.

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

The system RAM for n-gram embedding tables: those of PLE models (e.g. Qwen3.8-Flash-Next, whose
quantized table is tens of GB) and DeepSeek-V4.1's engram tables (per engram layer, 384,006,168 rows
of 256 fp8 bytes, 91.5 GiB, and their 8-byte e8m0 scales, 2.86 GiB; two layers). A table held whole
in RAM serves its rows from memory; any other table streams its rows from disk with per-forward
gathers (run-coalesced positioned reads into pinned staging, through the disk engine), and what the
budget has left after the whole tables caches the rows the streamed tables deliver.

| Value | Meaning |
|---|---|
| unset (default), `0` | every table streams from disk; no row cache (the page cache is the rows' only warm tier) |
| `-ngr` alone, `all` | every table whole in RAM (the flag's meaning before it took a size) |
| a size | the tables that fit, whole, smallest first (tables every gathered row reads go first: the engram's scales); what is left is split evenly into a row cache per module whose tables stream (none below 64 MiB) |
| `auto` (the placement's `ram ngram=auto`) | a size of what the host has left after the reserve, the page cache kept free and the RAM-held experts |

A table is held whole or not at all: a 12 GiB budget next to a 41 GiB table holds no table, and
the 12 GiB become that table's row cache. For DeepSeek-V4.1 with `-ngr 16GiB`, both scale tables
are held whole (5.7 GiB) and each engram layer gets a 5.1 GiB cache of its fp8 rows (about 20
million rows each); `-ngr 100GiB` holds one layer's two tables whole (94.4 GiB) and caches the
other layer's rows in the rest. `-ngr all` next to the engram asks 189 GiB, which the host check
refuses on any machine without that much RAM free.

**The row cache** (`modules/ngram_row_cache.py`) is direct-mapped: row `u` lives in slot
`hash(u) mod slots` (the top bits of `u` times the golden-ratio constant), tagged with `u`; a row
that hashes to an occupied slot replaces what is there. One slot holds the row of every cached
table of the module (the engram's fp8 row and its scale share their row ids), plus an 8-byte tag.
A gather looks every row up at once (vectorized, torch on the CPU): hits are copied into the
staging rows the forward uploads, misses are read from disk into scratch rows and, once the read
completed, copied into their staging rows and their slots. A row is served from the cache only
after a completed read stored it whole, so the cache can only change where a row comes from, never
its bytes; the prefetch worker and the forward's thread may use one module's cache at once (a lock
around each lookup and insert). A table's rows are the table's bytes whatever the cache holds: the
logits are the same with any budget.

As `-ngr` takes an optional value, a bare `-ngr` right before a word that is not a size takes that
word and is refused (`ngram_ram=prompt.txt: not a size ...`), never silently.
`NGramEmbedding(stream_from_disk = True / False)` still overrides the budget for one module.

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
tables as a size would. An expert tier's RAM counts as RAM-held experts: a size (`ram experts=48GiB`,
or the `-er` cap) at the start of the load; `ram experts=all` only at its end, when the tier is
sized, so the row caches `auto` made at the start (still empty) are shrunk to what is left then,
never grown. A row cache's rows are faulted in as rows arrive: the end-of-load host check counts
them as not yet held, so a budget the host cannot keep is refused at the load, not met by the
memory cgroup or the OOM killer later. The placement names the main model's layers, so its `ram`
rule applies to the main model's text component only.

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
3. **N-gram tables**: which tables the budget holds whole; the others stream, and what is left of
   the budget becomes their row caches.
4. **One host check** of everything together (below). The result is kept for the n-gram modules
   (`infer_params.ngram_ram_plan`, `infer_params.ngram_row_cache_plan`), which load in RAM exactly
   the tables the plan holds and make exactly the row caches it sizes.

A load with anything in RAM, or with a budget set, prints one line, e.g.

```
 -- RAM budget (text): routed experts 52.4 GiB (-mcl 11: 11 layers), cap 56GiB; 118.3 GiB available
 -- RAM budget (text): n-gram tables 41.2 GiB in RAM (1 of 1 whole, 41.2 GiB); 150.0 GiB available
 -- RAM budget (text): n-gram tables 16.0 GiB in RAM (2 of 4 whole, 5.7 GiB; row caches 10.3 GiB for 2 modules); 142.6 GiB available
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
            | "prefetch=" , ( "auto" | "off" | method , { "+" , method } )   (* cache *)
            | "profile=" , value ;                               (* cache, split; stream / cpu with hot= *)
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
| `ngram=` | the `--ngram_ram` cap, else `0` | system RAM for n-gram tables: `0` streams every row, `all` holds every table whole, a size the tables that fit (whole, smallest first) and a row cache of the others in the rest, `auto` what the host has left (see `-ngr`) |
| `pagecache=` | `auto` = max(8 GiB, MemTotal / 8) | RAM that `auto` budgets leave unpinned (the page cache is the n-gram rows' warm tier); it does not limit explicit sizes. `all` is refused |
| `policy=` | `exclusive` | the RAM tier against the VRAM cache: `lazy-exclusive` (a promotion leaves the RAM copy as a reclaimable duplicate), `exclusive` (a promotion frees the RAM copy), `inclusive` (RAM keeps what it holds) |
| `demote=` | `heat` | what happens to a VRAM victim with no RAM copy: `swap` (copied back only into the RAM slot its replacement left, and only when warmer than RAM's coldest), `heat` (into any free slot, or over RAM's coldest when warmer), `all` (every victim), `off` (dropped; the disk still has it) |
| `evict=` | `lfu` | RAM tier victims: `lfu` (lowest heat of a 16-entry sample) or `lru` |

### `disk`: where experts and n-gram rows are read

| Attribute | Default | Meaning |
|---|---|---|
| `experts=` | `model` | `model`: the checkpoint's shards, read in place. `<dir>`: a copy of the shards in another directory (another drive), same file names; every shard the tier reads must be there with the model's size and safetensors header, byte for byte, or the load is refused naming the file (`model/disk_source.py`); every expert read (cold fill, misses, read-ahead, refills) then goes to the copy. `off`: never read an expert at runtime; the load refuses unless VRAM and RAM hold every cached expert, and every victim returns to RAM |
| `ngram=` | `model` | the same for n-gram tables (PLE tables and DeepSeek-V4.1's engram tables): `<dir>` reads their rows from a checked copy; `off` needs `ram ngram=all` |
| `io=` | `auto` | the page-cache mode of the component's expert and n-gram reads through the disk engine: `auto` leaves it to the engine (`EXL3_DISK_DIRECT`: with `io_uring`, expert extents `O_DIRECT` and rows buffered); `direct` reads everything `O_DIRECT` (buffered where the file system refuses it; rows through the engine's bounce slots); `buffered` reads everything through the page cache. Needs the disk engine: refused with `EXL3_DISK_BACKEND=original` when n-gram rows stream |

### Checks the grammar cannot express (parse time)

- Coverage as for layer rules: every layer placed exactly once.
- One kind of host-held experts per GPU: GPU-computed (`stream`, `cache`) and CPU-computed
  (`cpu`, `split`) layers on different GPUs.
- `hot=` applies to `cache`, `stream`, `cpu` (and is required by `hybrid`); `cpu=` only to
  `split`; `prefetch=` only to `cache` (it reads experts ahead from the disk; `stream` layers hold
  all theirs in RAM); `profile=` to `cache`, `split`, and `stream` / `cpu` with `hot=` (it chooses
  which experts stay in VRAM, and without `hot=` those layers keep none there).
- One `cuda:<n>` rule per GPU, one `ram` rule, one `disk` rule. A `cuda:<n>` rule needs an
  `experts=cache` layer on that GPU; `cache=0` takes no `spare=` / `evict=` / `admit=`;
  `spare=0` needs `ram demote=off` or `policy=inclusive` (no free slot would take a demoted
  victim).
- `ram experts=` needs a layer that keeps experts in RAM. `policy=`, `demote=`, `evict=` need a
  `cache` layer and a RAM tier (`experts=0` has none). `demote=` has no effect with
  `policy=inclusive` and is refused there. At most one of `experts=` and `ngram=` can be `auto`.
- `disk experts=<dir|off>` needs a `cache` layer. `disk experts=off` refuses `demote=off` unless
  the policy is `inclusive`, and an explicit `prefetch=layer` / `router` (nothing is read from the
  disk to read ahead). `disk ngram=off` needs `ngram=all`. `io=` with both `experts=off` and
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

| # | Meaning | Canonical form |
|---|---|---|
| T1 | one GPU, every expert in VRAM | `*=cuda:0` |
| T3 | one GPU, 32 of 128 experts per layer resident, the rest on the CPU worker (`-mcs 96`) | `*=cuda:0 experts=cpu hot=32` |
| T3c | the same, in the older spelling (`experts=hybrid hot=32`, normalized) | `*=cuda:0 experts=cpu hot=32` |
| T4 | DeepSeek-V4.1 serving layout, CMP + PRO | `0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1` |
| T5 / (c) | V4.1, CMP + PRO: layers 12-39 cached on the PRO over RAM that holds the rest; the disk never read for experts | `0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off` |
| (a) | the PRO alone, RAM for every expert outside VRAM | `*=cuda:0 experts=cache; ram experts=all; disk experts=off` |
| (b) | the PRO alone, 96 GiB of RAM tier, the SSD for the rest | `*=cuda:0 experts=cache; ram experts=96GiB` |
| P1 | read two layers ahead at prefill, predict the next layer's experts at decode | `*=cuda:0 experts=cache prefetch=layer:2+router` |
| P2 | hot pins and the cold fill from two weighted profiles | `*=cuda:0 experts=cache hot=16 profile=code:3,chat:1` |
| P3 | the resident slice of streamed layers from a profile | `*=cuda:1 experts=stream hot=64 profile=chat` |
| T6 | the PRO alone: 75 GiB cache, 96 GiB RAM tier, n-gram budget | `*=cuda:0 experts=cache; cuda:0 cache=75GiB; ram experts=96GiB ngram=6GiB` |
| T7 | a RAM-limited box: 16 experts per layer pinned; the tier takes what is free after 24 GiB of page cache | `*=cuda:0 experts=cache hot=16; ram experts=auto pagecache=24GiB` |
| T8 | no RAM tier: the VRAM cache straight over the SSD | `*=cuda:0 experts=cache; ram experts=0` |
| T10 | a big-RAM host: inclusive RAM, whole n-gram tables | `*=cuda:0 experts=cache; ram experts=all ngram=all policy=inclusive` |
| T11 | every policy knob | `*=cuda:0 experts=cache; cuda:0 cache=40GiB spare=12 evict=lfu admit=heat; ram experts=64GiB pagecache=16GiB policy=lazy-exclusive demote=swap evict=lru; disk io=direct` |
| T12 | no D2H copies at all | `*=cuda:0 experts=cache; ram experts=48GiB demote=off` |
| T13 | CPU-computed layers on one GPU, cached layers on the other | `0-9=cuda:0 experts=cpu; 10-39=cuda:1 experts=cache; cuda:1 cache=40GiB; ram experts=96GiB` |
| T14 | experts and n-gram tables read from other drives | `*=cuda:0 experts=cache; ram experts=48GiB; disk experts=/nvme1/v41 ngram="/mnt/engram disk/v41"` |
| T16 | GPU streaming with a resident slice (`-mcs k -mcm stream_only`) | `*=cuda:1 experts=stream hot=64` |
| T17 | units: `cache=1.5`, `ram experts=48GB ngram=512mi` | `*=cuda:0 experts=cache; cuda:0 cache=1536MiB; ram experts=48GB ngram=512MiB` |
| T18 | an n-gram budget only, for a PLE model | `*=cuda:0; ram ngram=12GiB` |
| R1 | the RAM of stream layers capped by the placement | `0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1; ram experts=64GiB` |
| R2 | whole n-gram tables, never read from disk | `*=cuda:0; ram ngram=all; disk ngram=off` |

A multi-line value with comments; the written-out defaults vanish from the canonical form
(`0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=52GiB`):

```sh
export EXL3_PLACEMENT="
0-11  = cuda:0                                  # CMP: resident
12-39 = cuda:1 experts=cache                    # PRO: cached
cuda:1 cache=auto spare=8 evict=lru admit=adaptive
ram experts=52GiB pagecache=auto policy=exclusive demote=heat evict=lfu
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
| `*=cuda:0 experts=cpu prefetch=layer` | `prefetch= applies to experts=cache (experts=cpu computes on the CPU worker)` |
| `*=cuda:0 experts=stream prefetch=router` | `prefetch= applies to experts=cache (experts=stream holds every routed expert in RAM and streams each call's experts, with nothing to read ahead from the disk)` |
| `*=cuda:0 experts=stream profile=chat` | `profile= chooses which routed experts stay in VRAM, and experts=stream without hot= keeps none there; add hot=<k> (or use experts=cache)` |
| `*=cuda:0 experts=cache prefetch=layer:2; ram experts=all; disk experts=off` | `placement: prefetch=layer:2 reads experts ahead from the disk, which 'disk experts=off' never reads; remove it (or write prefetch=off)` |
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

`tests/test_placement_tiers_.py` holds every case with its full text.

## The expert tier

The `experts=cache` layers of a placement keep their routed experts in three levels: a VRAM cache
per GPU (a lookup kernel inside the MoE dispatch decides each call's slots; a tier thread turns the
decisions into copies; whole layers are staged on the copy engines during a long prefill), a RAM
tier of the warmest experts the cache does not hold, and the model's own shards on the SSD, read in
place through the disk engine. This section describes what the words ask for, how a load sizes the
tier and what it refuses (`model/expert_tier_config.py`; its tests,
`tests/test_expert_tier_config_.py`, run every number below), the extent index, the policy
(reference and native), the RAM tier host, the VRAM cache, the layer-mode prefill, the read-ahead,
and the profiles and the heat file.

### Where a routed expert of a `cache` layer lives

| Place | Holds | Sized by |
|---|---|---|
| VRAM, `hot=` pins | `hot` experts per layer, never evicted | the layer (counted as its weights) |
| VRAM, the GPU's cache | a changing set, shared by every `cache` layer on that GPU (one least-recently-used pool per GPU) | `cuda:<n> cache=` |
| VRAM, staging | a decode or routed call's transients (half 0); in a layer-mode prefill, the whole layer being computed and the next one being copied (two halves) | `EXL3_MOE_TIER_STAGING` |
| RAM, the tier | the warmest experts VRAM does not hold (`policy=lazy-exclusive` / `exclusive`), or a copy of what it holds too (`inclusive`) | `ram experts=` |
| RAM, the disk slab | SSD reads that bypass the tier: misses read ahead of their call (`prefetch=`), a record's reads in flight together, a layer's disk-only experts being staged | `EXL3_MOE_TIER_DISK_SLAB` |
| SSD | every expert: the checkpoint's shards (or a checked copy of them, `disk experts=<dir>`), read in place, never written | `disk experts=` |

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
When `disk experts=` is not `off`, each cache GPU has a disk slab of `EXL3_MOE_TIER_DISK_SLAB`
slots (RAM slots, pinned), `auto` by default:

```
per_layer = min(E, ceil(1.25 x disk-only experts / cache layers))
slab      = 8 + (D + 1) x per_layer + Dr x top-k        D: deepest prefetch=layer, Dr: deepest prefetch=router
```

8 for one decode call's misses read together, `D + 1` layers' shares of the disk-only experts for a
layer-mode prefill (the layer being staged and the `D` read ahead; a quarter more for layers that
hold more than their share), `Dr x top-k` for the router's predictions; 8 alone when VRAM and RAM
hold every cached expert. The host check covers the static arenas, the tier, the n-gram tables and
the slabs together.

A load checks what it can before any module loads (explicit `cache=` and `ram experts=` sizes,
the settings, the host memory they need), and sizes the pools at the end of the layer split, with
each GPU's room, before any of the tier is allocated. It prints:

```
 -- expert tiers: cuda:1 cache 6553 slots (81.0 GiB, 8 spare, evict=lru admit=adaptive; staging 768 slots (9.5 GiB, double))
    ram experts 52.2 GiB pinned (static 0 MiB + tier 4207 slots), policy exclusive demote=heat, evict=lfu
    ngram 0 MiB (every row streams from disk)
    disk experts=off io=auto: 0 experts on disk only; 73.8 GiB of RAM left unpinned
```

### Where an expert is read from: extents and slot geometry

The tier never repacks the checkpoint. Each routed expert of a `cache` layer is read in place
from its shard (`model/expert_extents.py`, `build_extent_index`, from the headers alone), or from
the same shard in the directory `disk experts=<dir>` names, once that copy is checked to have the
model's size and safetensors header (`model/disk_source.py`: the offsets the index took from the
model's headers must hold in the copy):

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

| What RAM keeps | `exclusive` (default) | `lazy-exclusive` | `inclusive` |
|---|---|---|---|
| admitted, was in RAM | the RAM slot is freed | the RAM copy becomes a reclaimable duplicate | the RAM copy becomes a duplicate |
| admitted, read from the SSD | nothing (read into the slab) | a duplicate, in a free slot or over a reclaimed duplicate; else nothing | a duplicate, in a free slot or over the coldest owner |
| transient, read from the SSD (decode) | RAM admission: a free slot, else a reclaimed duplicate, else over the coldest sampled owner (`EXL3_MOE_TIER_RAM_ADMIT=heat`: only when warmer); else the slab | same | same, without reclaiming duplicates |
| transient, read from the SSD (prefill) | the slab; a refill candidate (below) | same | same |

| The victim (`ram demote=`) | Rule |
|---|---|
| has a duplicate in RAM | the duplicate becomes the owner; nothing is copied |
| `swap` | copied back only into the RAM slot its admission freed or made a duplicate, and only when it is warmer than RAM's coldest sampled owner; else dropped: at most one copy back per promotion from RAM |
| `heat` (default) | copied into a free slot, else over a reclaimed duplicate, else over the coldest sampled owner when warmer; else dropped |
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
  the layer that the pool does not hold is staged, in expert order (from RAM, or read from the SSD
  into the slab); no stamps, no admission, no demotion; heat grows by the call's assignments, and
  the experts it used that only the disk holds are refill candidates (in expert order). The GPU
  runtime runs the two halves apart (`layer_plan`: the staging, before the layer's routing is known;
  `layer_record`: heat and refills, from the call's record), which give exactly the actions of the
  whole (`layer_call`, the CPU replay; tested).

**Heat** is a decayed routing count per key, kept in 16.16 fixed point with the epoch it was last
written in. The epoch is `decode tokens // EXL3_MOE_HEAT_HALFLIFE` (one decode pass = one token);
reading a key halves its value once per epoch since (lazily, no sweep). A decode assignment adds
1.0, a prefill assignment `EXL3_MOE_HEAT_PREFILL` (1/16): a 4K-token prompt adds about 4 to every
expert of a layer, a few tokens of decode.

**Cold fill** (end of the load): in priority order (a profile's or the heat file's, see "Profiles
and the heat file"; else round robin across layers by expert index: expert 0 of every layer, then
expert 1, ...), the first `C` experts fill the pool (all of them when they fit), the next `R` the RAM
tier (`inclusive`: RAM first takes a duplicate of every pool expert). Nothing else is read after the
load unless a call misses, a refill or a read-ahead reads.

**Checked in the tests** (`tests/test_expert_tier_policy_.py`): nine micro-scenarios with every
record, counter and final state; invariants after every call of 200 random configurations x 5,000
calls; agreement with the policy simulator the first defaults were chosen on; the adaptive rule on
scripted cache lives, exactly. The native core (`tests/test_expert_tier_native_.py`) makes the
same decisions as the reference on the micro-scenarios and 200 random configurations: the same
records, actions, counters and final state, heat, random draws and the adaptive probability
included; its standalone driver (`tests/expert_tier/build.sh`, `tier_policy_test`) checks the
micro-scenarios and 10^6 random calls with every invariant and a mirror directory that applies
the records (the tier thread's view of the lookup kernel's directory), under AddressSanitizer,
UndefinedBehaviorSanitizer and ThreadSanitizer.

### The RAM tier: memory, reads and copies

The RAM tier and everything that moves bytes for it live in the parent process
(`exllamav3_ext/tier/tier_host.cpp`, Python `model/expert_tier_host.py`, class `TierHost`). No
child process and no copy of the checkpoint are involved: experts are read in place from the
model's own shards.

**Memory.**

| Part | What it is | V4.1 3.0 bpw |
|---|---|---|
| RAM tier | `R` slots of the RAM slot size, in anonymous mappings of whole slots (one chunk = at most 1 GiB; transparent huge pages only with `EXL3_MOE_TIER_HUGEPAGE=1`); each chunk is one registered io_uring buffer of the disk engine, so reads into it are `READ_FIXED` (up to 1,023 chunks) | 80 slots (1,065,615,360 bytes) per chunk; `R` = 4,976 slots = 63 chunks for layout (c) |
| disk slab | `EXL3_MOE_TIER_DISK_SLAB` slots (`auto`, see "Sizing"), same layout, registered too | 13.3 MB per slot |
| pool and staging | the GPU's; the CPU replay (`TierHost`) stands in a host buffer for them | |

Every slot is filled at load, so the chunks are faulted in right away by several threads
(`prefault_threads`, 8), then registered with io_uring, which pins them; the GPU runtime registers
them with CUDA once they are filled (pin after fill). Registering alone would fault them in one
chunk after the other: on the AI VM, 8 GiB took 1.4 s that way and 0.5-0.8 s faulted in by
threads; with transparent huge pages 9-12 s (compaction next to a large page cache, hence
`EXL3_MOE_TIER_HUGEPAGE=0` by default). The cold fill then reads at 10-12 GB/s. All of it counts in
the load's host-memory check (`ram experts=`, `disk slab=`).

**What a RAM slot holds.** Either an expert as it was read from the SSD (*extent layout*: its
pieces at 4 KiB-aligned bases, each payload at its file offset modulo 4 KiB, the three trellis
tensors at the recorded sub-offsets), or an expert copied back from VRAM by a demotion (*compact
layout*: the three trellis tensors back to back, as in a VRAM slot). A promotion copies three
ranges from an extent-layout slot and one from a compact one.

**Reads** go through the process-wide disk engine ([disk_engine.md](disk_engine.md)), with the
extent API, one ticket per expert, in the page-cache mode `disk io=` says (`auto`: the engine's,
`O_DIRECT` for extents with `io_uring`; the payload offset is the same either way), in the engine's
request classes:

| Class | Tier reads |
|---|---|
| 1 expert demand | a decode or routed call's miss that only the SSD holds; a layer-mode staging read that was not read ahead |
| 2 prefetch | the cold fill; the layer-mode read-ahead (`prefetch=layer`: `prefetch_layer`, deadline its layer's expected start, promoted to class 1 when its layer is staged); the router's predictions (`prefetch=router`: `read_ahead`, deadline 1 ms) |
| 3 refill | refills (the policy's refill decisions); the RAM slot of a miss served from the slab |

A read the engine fails (an I/O error, a short read, a file that changed) is retried once with a
plain `pread` of the same bytes into the same place. If that fails too, the tier fails: the call
raises `RuntimeError` with `expert tier: reading layer <l> expert <e> (<first projection key>) from
<file> at offset <n> failed: ... (errno <n>)`, and every later call raises the same message.

**Ordering.** The policy's decisions are final when a record is processed; the bytes follow, and
every slot keeps what is still reading or writing it: a read into a RAM slot waits for the copies
still reading its old contents and for a demotion still writing it; a promotion from a RAM slot
waits for the read or the demotion that fills it; a demotion into a slot is ordered after the
promotion that last read it (the swap case: the victim goes into the slot its admission's copy is
reading from). A pool slot returns to the free list only after the demotion that reads it has
landed. `spare=0` replaces victims in place, so it needs `ram demote=off` or `policy=inclusive`
(the runtime refuses it otherwise, as the grammar does).

**A record's reads go out together.** Before any copy of a call's record is issued, every SSD read
the record needs is submitted (`presubmit`): the reads into the slab while it has a slot to spare,
and a read into a RAM slot when no earlier action of the record copies from or into that slot (its
old bytes are still needed first; that read waits for its turn). A call with six SSD misses then
waits for about the slowest read, not for six in a row (`presubmitted` counts them).

**Disk slab.** A slab slot is free, reading or read for a demand (a read `presubmit` started, or
the one a miss makes), a read-ahead (`prefetch=`, of layer `lc`), or copied: its copy to VRAM was
issued and its bytes stay valid, so a later miss of the same expert copies it again without a read
(`slab_reused`) until the slot is taken for another read, oldest first once its copy has landed.
A slot is taken, in order: a free one; the oldest copied one whose copy has landed; the read-ahead
farthest ahead (counted as dropped); a demand may also wait for the oldest copied slot's copy
(`slab_waits`). The slots a record needs are pinned while it runs, so its own reads never take
each other's slots, unless the record needs more slab reads than the slab has slots (then a later
action reads its expert again). A decode miss whose expert the slab holds (read ahead) is served
from there even when the policy gives it a RAM slot: the copy to VRAM comes from the slab at once,
and the RAM slot is filled by a background read of its own (class 3, `slab_to_vram`). Read-ahead
of an expert that reached RAM or VRAM another way before its layer is dropped when the layer is
staged.

**Cold fill** (`cold_fill`, end of the load): the RAM tier's experts are read straight into their
slots, `fill_inflight` (8) at a time; the hot pins' and the pool's experts are copied from their RAM
copy when RAM holds one (`policy=inclusive`), else read through the slab, as many reads in flight as
the slab has slots. The order is the seed's (see "Profiles and the heat file"). On the AI VM, 44
real V4.1 experts (586 MB) fill at 8.9 GB/s.

**Deterministic mode** (`TierHost(deterministic = True)`, the CPU replay's default;
`EXL3_MOE_TIER_DETERMINISTIC=1` on the GPU): a call returns only when every read and copy it
started, refills included, has landed. Otherwise refills complete in the background and land on
`poll()` or the next access of their slot. Either way the decisions are the same; only timing
differs.

**Statistics** (`TierHost.stats()`): `ssd_reads`, `ssd_bytes`, `pread_fallbacks`, `h2d` / `d2h` /
`d2d` (copies, a three-piece promotion counts once) and their bytes, `staged`, `prefetch_issued`,
`prefetch_used`, `prefetch_starved`, `prefetch_dropped`, `refill_reads`, `hazard_waits` (a write
waited for a reader or writer of its slot), `slab_waits`, `cold_ram`, `cold_vram`, `cold_bytes`,
`cold_ns`, `arena_ns` (mapping, faulting in and registering the tier), `compacted`, `presubmitted`
(SSD reads of a record submitted before its first copy), `slab_reused`, `predicted_reads`
(`prefetch=router`), `slab_to_vram`, and the policy's counters (`policy_hits`, `policy_admits`, ...
as in "Policies").

**Python.**

```python
from exllamav3.model.expert_extents import build_extent_index
from exllamav3.model.expert_tier_policy import PolicyConfig, round_robin_order, DECODE, ROUTED
from exllamav3.model.expert_tier_host import TierHost

index = build_extent_index(stc, keys)             # keys[lc * E + e] = (gate, up, down) projection keys
tier = TierHost(index, layers = L, experts = E, policy = PolicyConfig(spare = 8), pool_slots = S,
                ram_slots = R, slab_slots = 8)
tier.cold_fill()                                  # round robin across layers
tier.tick()                                       # one decode pass
entries = tier.call(lc, ids)                      # [(key, kind, dst, cnt, heat, vkey, vslot, ...)]
tier.prefetch(lc + 1); tier.promote(lc + 1)       # layer-mode read-ahead, then its layer is next
tier.layer(lc + 1, ids, half = 1)                 # stage the layer into staging half 1 (a layer call)
keys = tier.stage(lc + 1, half = 1)               # ... or its halves, as the GPU runtime runs them:
tier.process((seq, lc + 1, LAYER, entries))       #     the staging, then the call's record
tier.read_ahead((seq, lc + 2, 3, entries))        # prefetch=router: a prediction record
tier.stats(); tier.state(); tier.arena()
# or from a load's placement and sizing: TierHost.from_placement(stc, cache_modules, placement,
#     "cuda:1", tier_config, pool_slots = S, ram_slots = R)
```

`call()` runs the policy's own lookup (the CPU replay of a routing trace); `process(record)` takes a
record decided elsewhere (the GPU lookup kernel's) and applies it to the host's mirror of the
directory before the same host step.

**Checked in the tests** (`tests/test_expert_tier_ram_.py`, `tests/expert_tier/tier_host_test.cpp`):
after every call of scripted and random traces over every policy x demote x admit x evict, with the
disk on and off, `spare=0`, deferred returns and asynchronous refills, the host's records, actions
and state equal the reference policy's, and every pool slot, retiring slot, RAM tier slot and
staging slot holds exactly the checkpoint bytes of the expert the policy says it holds; the cold
fill with every engine backend; the read-ahead, and a read-ahead serving a decode miss; the router's
predictions; layer calls split into their halves (the same actions, bytes and state as whole);
refills; every engine read failing (the `pread` fallback serves them); a shard cut short (the error
names the expert); the arena's chunks and registrations; `disk experts=<dir>` (every read from the
copy, the model's shards moved away meanwhile; a copy with another header refused) and `disk
io=direct` / `buffered`; the real V4.1 shards. The C++ driver runs under AddressSanitizer,
UndefinedBehaviorSanitizer and ThreadSanitizer.

### The VRAM cache: decode and routed calls

A `cache` layer is a resident layer in every respect but one: its routed experts' trellis tensors
are not loaded with it. Its expert `Linear`s get their EXL3 inner at load with `suh`, `svh`, `mcg` /
`mul1` and `bias` resident (about 44.5 KB per V4.1 expert) and a trellis that is a view of one
zeroed *sentinel* slot per GPU (`modules/block_sparse_mlp_tier.py`), so the layer builds its
`MultiLinear` pointer tables, bound classes and fused buffers exactly as a resident layer does. The
tier then owns one thing of the layer: its per-expert trellis pointer tables (gate, up, down), which
every quantized MoE path reads on the device (the fused decode kernels, the fused MoE kernel, the
batched reconstruct tier). The per-expert dequant path (an expert with more rows than the batched
tier takes, 455 for V4.1) gets the trellis tensors where the tables point
(`BC_BlockSparseMLP.run_single_expert_dq_views`, same kernels and arguments as
`run_single_expert_dq`). The graph and torch per-expert paths read an expert's own trellis tensor:
a `cache` layer must be one the fused MoE kernel covers (refused at load otherwise) and raises should
one of those paths be reached anyway. A tiered layer therefore computes exactly what the same layer
computes resident, bit for bit: same kernels, same arguments, same trellis bytes, only where the
bytes live differs.

A call's mode follows its rows: up to `EXL3_MOE_TIER_DECODE_ROWS` (8) a **decode** call (stamps,
admission), from `EXL3_MOE_TIER_PREFILL_ROWS` (256) a **layer-mode** call (next section), in between
a **routed** call (hits keep their slots, misses are staged as transients for the call; no stamps,
no admission, no demotion). A forward pass has one row count, so all its calls on a GPU have one
mode.

**One decode or routed call** (`BlockSparseMLP.forward`, right after routing), two kernels on the
compute stream:

```
compute stream                                   tier thread (native, one per GPU, no CUDA calls)
--------------------------------------------     -------------------------------------------------
lookup kernel (1 CTA of 1,024 threads):          reads the record from the ring in mapped memory
  harvest the slots the host returned            applies it to its mirror of the directory
  dedup, heat, stamp the hits (decode)           runs the policy's host step (sources, RAM side,
  decide every miss (admit / transient),           demotions, RAM admission, refills), submits the
  retire victims                                   record's SSD reads together
  point the layer's tables at the slots          appends the call's copy commands to the plan ring:
  publish the call's record                        promotions (RAM -> VRAM, slab -> VRAM after an
fetch kernel (cooperative, half the SMs + 1):      SSD read), rescues (a retiring VRAM slot ->
  all hits: returns at once                        VRAM), demotions (VRAM -> RAM), then END
  else runs the call's commands with the SMs  <--  polls the completion flags; a demoted slot goes
    until END                                      back through the return ring once its copy is done
the unchanged MoE kernels
```

- The tier thread **never calls the CUDA API** once the model runs: records, commands, completion
  flags and returned slots are plain loads and stores on mapped memory. This is what makes the
  design safe next to any other code in the process: a thread that holds the driver's locks in a
  synchronous call (a pageable copy such as `.tolist()` waiting for the stream) cannot hold up the
  copies the stream is waiting for. (A first version issued the copies with `cudaMemcpyAsync` from
  the tier thread and deadlocked exactly there: the caller's `.tolist()` waited for the stream while
  holding a driver lock the tier thread's next copy needed.) The copy engines serve the cold fill,
  before the tier thread starts, and the layer-mode prefill, from the caller's thread.
- The SMs copy as fast as the copy engines here: on the AI VM (RTX PRO 6000) 13.3 MB from
  page-locked memory to VRAM at 50-52 GB/s with 94 CTAs of 256 threads (the copy engines: 52.7),
  VRAM to page-locked memory at 30-37 GB/s (the copy engines: 55). Each thread keeps four 16-byte
  loads in flight. Host memory is read uncached (`ld.global.cv`): a RAM slot the host rewrote is
  never served stale. A source at an odd offset (the extent layout of an SSD fill) is read as
  aligned words, each once, through shared memory, and every output word is assembled from two
  neighbours: every source byte crosses the link once, as for an aligned source.
- An all-hit call's fetch returns at once: the decode path has **no host synchronization**, and a
  call with misses waits on the GPU for its own copies only.
- The lookup decides exactly what the policy core decides (`VramDirectory::lookup`): the same
  records for the same calls. The tier thread's mirror applies those records; it never re-decides.
- Commands run in order per worker CTA; a command that must wait for another (a demotion into the
  RAM slot a promotion of the same call reads, the in-place replacement of `spare=0`) names it and
  waits for it and for every command before it: a command's chunks are spread over the CTAs, so one
  command being done says nothing of the ones before it (a promotion is three copies, one per
  projection, and a demotion into its RAM slot that waited for the last one alone could overwrite
  the slot while the first two still read it). A victim is only ever an expert the call does not use; a pool slot is only reused
  after its demotion is done; a copy for call `n` runs in `n`'s fetch, after every earlier kernel
  on the stream, so nothing a kernel may still read is overwritten. Command tokens and completion
  flags are 64-bit (a command index never wraps).
- **Hot pins** (`hot=<k>` or `hot=<p>%` on `cache` layers): `k` experts of each layer (the first
  `k`, or the `k` a profile ranks highest) are placed in pin slots after the pool at load and are
  hits forever (never victims, never in the RAM tier).
- **Heat** advances one decode token per decode call of the GPU's first cache layer.
- The host side (policy state, RAM tier, statistics) is read and changed under one lock: by the
  tier thread while it handles a record or returns slots, by the caller's thread while it plans a
  layer-mode prefill, and by whoever reads statistics.

**Memory** (per GPU with `cache` layers):

| Part | Where | V4.1 3.0 bpw |
|---|---|---|
| pool | `S` slots of the compact VRAM slot (256-byte aligned), in allocations of at most 4 GiB, allocated by torch at the end of the layer split (still under the `-gs` fraction) | 13,271,040 B per slot |
| pin slots | `hot` per layer, after the pool | |
| staging | two halves (`EXL3_MOE_TIER_STAGING=double`) or one (`single`) of `max over cache layers of (E - hot)` slots | 384 slots per half, 4.75 GiB |
| sentinel | one slot, zeroed | 12.7 MiB |
| directory | slot state, key, stamp, address; per key slot, heat; free bitmap; counters (`cudaMalloc`) | ~0.3 MB at S = 6,000 |
| control page | mapped pinned host memory: flags, the return ring (a power of two above `S + pins`), the record ring (4 MiB), the plan ring (65,536 commands of 40 bytes) and its completion flags | 7.1 MiB |
| fetch state | the plan's device mirror, completion flags and counters (`cudaMalloc`) | 3.9 MiB |
| layer-mode tables | pinned host memory: every cache layer's three pointer tables | 9 KiB per layer |

The RAM tier's chunks are page-locked and mapped (`cudaHostRegister`, portable and mapped: the fetch
kernel reads and writes them at their host addresses; the copy engines read them) once the cold fill
has filled them; the disk slab is page-locked at creation, and released after the cold fill when
`disk experts=off` (nothing reads the SSD afterwards).

**Load.** At the end of the layer split, per GPU with `cache` layers: its room (what is left inside
the `-gs` budget and physically, after the placed modules, the Cache, the largest transient and
`EXL3_AUTOSPLIT_MARGIN_MB`; minus the pin slots), its host link (a 64 MiB page-locked copy; the
`EXL3_MOE_TIER_MIN_LINK_GBS` refusal), then `size_tiers` with those rooms (the sizing summary is
printed), the extents of every expert of its cache layers (`build_for_modules`, from the model's
shards or a checked copy), the native runtime, the seed (hot pins, cold-fill order and heat), the
cold fill, and the tier thread. With cache layers on several GPUs the component's RAM tier is split
between them by what each pool does not hold. The load prints:

```
 -- expert cache cuda:<n>: cold fill of <pool + pins> VRAM and <R> RAM experts, <bytes> read in <t> s; <bytes> of RAM page-locked in <t> s; seeded from <source>
```

**Modes.** Production (default): an all-hit call's fetch returns at once, the thread returns demoted
slots when their copy is done, and a slot returned late only makes an admission use a transient
(counted as *starved*). `EXL3_MOE_TIER_DETERMINISTIC=1`: every decode and routed call's fetch waits
until the thread has applied its record, run its copies and returned its slots, so the directory is
exactly the policy core's after every call (the tests; a debugging aid). Logits never depend on the
mode. `EXL3_MOE_TIER_VERIFY=1` synchronizes after every call and checks that the device directory
equals the mirror. The tier thread spins while calls keep coming (`EXL3_MOE_TIER_SPIN_US=-1`: until
2 ms after the last call; `0`: never, a sleeping thread is woken by the next call; `N`: N us) and runs
on `EXL3_MOE_TIER_AFFINITY` when set. `EXL3_MOE_TIER_TRACE=<path>` writes every record the thread
handles to a file: a header line (`{"format": "exl3-tier-trace", "version": 1, "device", "layers",
"experts", "slots", "pins", "ram_slots", "spare"}`), then one JSON line per call (`{"seq", "lc",
"mode", "tokens", "e": [[key, kind, count, dst, victim key, victim slot], ...]}`; kind 0 hit, 1
admitted, 2 transient or staged; mode 0 decode, 1 routed, 2 layer), with `.cuda<n>` appended to the
path per GPU when several hold cache layers. A trace replays through the policy cores (each call's
experts, each repeated `count` times in the listed order, give the same record).

**Failures.** A read that fails twice (disk engine, then `pread`), a policy error or a lookup error
(an expert id outside the layer, more transients than staging slots) puts the tier in an error
state: the fetch kernels stop waiting (the abort flag), and the next call raises `RuntimeError` with
the message. At most the token in flight is affected. A fetch that waits a minute without a command
gives up the same way.

**Statistics** (`TierDevice.stats()`, `ExpertTierSet.stats()`): `records`, `host_records` (calls
with misses), `returns`, `predictions`, `max_lag` (records waiting when the thread picked one up),
`idle_sleeps`, `work_ns`, `overflow` (all-hit records dropped on a full ring: heat and stamps of the
mirror only), `device_error`, the fetch kernel's copy counts and bytes (`h2d_*`, `d2h_*`, `d2d_*`),
`plan_commands`, `plan_full_waits`, `plan_deps_done` (dependencies already met when a command was
written), the layer mode's (`layer_plans`, `layer_calls`, `layer_catch_ups`, `layer_h2d_bytes`,
`layer_h2d_copies`, `layer_batches`), the load's (`cold_h2d_bytes`, `pinned_bytes`, `pin_ns`,
`compacted`, `compact_ns`), the RAM tier host's (`ssd_reads`, `ssd_bytes`, `presubmitted`,
`prefetch_*`, `predicted_reads`, `slab_to_vram`, `cold_*`, ...) and the policy's (`policy_hits`,
`policy_admits`, `policy_d2h`, ...).

**Checked in the tests** (GPU; on the AI VM through `/root/rnd/gpu_window.sh`):
`tests/expert_tier/tier_kernel_gpu_.py` (the kernel's records, the mirror state and the device
directory against the policy core's replay, exactly: the micro-scenarios, hot pins, `spare=0`,
routed calls, heat across ticks, two-projection experts, and seeded random replays up to the V4.1
shape at batch sizes 1 to 8 with every `admit` x `evict`, the adaptive probability live and
pinned; the layer's pointer tables after the calls); `tests/expert_tier/tier_runtime_gpu_.py` (a
synthetic checkpoint with unaligned and split experts: records against the replay and every pool,
staging and RAM slot byte-checked against the checkpoint after every call, over every `policy` x
`demote` x `admit`, the disk off, hot pins, `spare=0`; production mode with the thread, with stream
memory operations and with the fallback kernels, byte checks after a drain, and the data path of
1,500 calls checked as the MoE kernels would read it; the V4.1 promotion cost);
`tests/expert_tier/tier_bitwise_gpu_.py` (real V4.1 layers resident and cached give bit-identical
outputs for decode, routed and layer-mode calls under every policy, over RAM and over the SSD, with
the read-ahead, in production mode through thousands of calls); `tests/expert_tier/tier_model_gpu_.py`
(the whole model: the same logits with a layer's experts resident, streamed and cached, under the
stable arithmetic profile).

### Prefill: layer mode

A prefill chunk of a few thousand rows uses nearly every expert of every layer, so a call from
`EXL3_MOE_TIER_PREFILL_ROWS` (256) rows stages the whole layer instead of its misses: every cached
expert the pool does not hold, from RAM or read from the SSD through the slab, into a staging half,
on the **copy engines**, one layer ahead, while the previous layer computes. Nothing is admitted,
stamped or demoted: decode-hot contents survive any prompt. The caller's thread (the Python thread
that runs the forward, or the V4.1 pipeline's second-stage thread) drives it; the tier thread keeps
its role (the record: heat and refills) and still never calls CUDA.

```
caller's thread (compute stream C, copy stream K)        GPU
-----------------------------------------------------     -----------------------------------------
layer L's call (resolve):                                 K: copies of L's staging (planned earlier)
  first layer of the pass: layer_begin, layer_plan(L)
  layer_run(L): C waits for L's copies; L's tables        C: waits, uploads L's tables (9 KiB),
    uploaded; heat record (lookup kernel, mode 2)             heat record, L's MoE
  L's MoE enqueued
after L's compute (after_compute):
  layer_after(L): C records "half(L) consumed"            K: waits "half(L+1) consumed" (by L-1),
  layer_plan(L+1): into the other half, K waits for           then L+1's copies, then a completion
    L-1's compute, then L+1's copies; tables built            flag
```

- **Planning** (`TierGpu::layer_plan`, native, GIL released): under the host lock, the policy's
  `layer_plan` lists every expert of the layer the pool does not hold, in expert order; the host
  submits their SSD reads together (into slab slots, or found there read ahead), then issues the
  copies on the copy stream: three per expert from an extent-layout RAM slot or a slab slot, one
  from a compact RAM slot, in the order the staging index gives. The layer's pointer tables are
  built on the host (VRAM-held experts at their pool or pin slot, the others at their staging slot)
  into pinned memory the compute stream uploads right before the layer. An SSD read the plan needs
  and that is still reading makes the caller wait (the GPU still has the previous layer queued);
  the read-ahead (`prefetch=layer:D`, next section) starts those reads `D` layers earlier.
- **Double buffering** (`EXL3_MOE_TIER_STAGING=double`, default): layer `L+1` is copied into the
  half layer `L-1` used, as soon as `L-1`'s compute is done, so the copies overlap `L`'s compute
  (and whatever runs between two cache layers, e.g. layers on another GPU). `single`: one half; the
  copies of `L+1` wait for `L`'s compute; half the staging memory, no overlap.
- **Transitions.** A layer-mode pass after decode or routed calls first waits until the GPU ran
  them and the tier thread applied their records (the plan reads the directory as of them:
  `layer_catch_ups` counts these waits); consecutive prefill chunks need none. The copy stream
  starts after everything enqueued on the compute stream before the pass; a decode or routed call
  after a pass waits for the copy stream's last batch (a pass may end before its last planned layer
  ran). A layer's pinned table is reused only after its previous upload landed. One thread drives a
  pass: a cache layer called from another thread in the middle of a pass is refused
  (`RuntimeError`), and the next layer-mode call begins a new pass.
- **Copy completion without the CUDA API on the tier thread**: every plan's copies end with a
  one-thread kernel on the copy stream that writes a completion flag in the control page; a RAM or
  slab slot a staging copy reads keeps that batch's token, and whoever must write the slot (a
  refill, a demotion, another read) waits for the flag with plain loads.
- **The record** of a layer-mode call (the lookup kernel in mode 2: dedup, heat, no stamps, no table
  writes) reaches the tier thread like any other: heat, and refills of the used experts only the
  disk holds. The policy decisions are those of the CPU replay's layer call, split in two
  (tested).
- **DQ views.** An expert above the batched reconstruct tier's row cap (455 rows for V4.1) takes the
  per-expert dequant path with the addresses the plan chose (host-known, no device read).

### Read-ahead: `prefetch=`

`prefetch=` (on `experts=cache` layer rules) says what the tier reads from the disk ahead of the
call that needs it; transfer only, it never changes routing or what the cache holds:

| Value | What is read ahead |
|---|---|
| `auto` (default) | `layer:1` when the disk holds experts (`disk experts=` not `off`), else nothing |
| `layer[:D]` | layer-mode prefill: planning cache layer `L` also reads the disk-only experts of the layers up to `L + D` (a layer's own depth: it is read `D` layers ahead of its call) into free slab slots, class 2 with a deadline of their expected start (the time between two plans), promoted to class 1 when their layer is planned; unused ones are dropped then. `D` defaults to 1 |
| `router[:D]` | decode: each decode call of cache layer `L` applies the router of cache layer `L + D` (the same GPU) to `L`'s router input and reads the picks that neither VRAM nor RAM holds from the disk into the slab (class 2, deadline 1 ms); when `L + D`'s call misses one of them, its copy comes from the slab (its RAM slot, if the policy gives it one, is filled by a background read). DeepSeek's sqrt-softplus router runs its own selection kernel into private buffers (the picks `L + D` would make for that input); other routers select on the gate logits plus the selection bias. `D` defaults to 1 |
| `layer:D+router:D2` | both |
| `off` | nothing |

With `disk experts=off` nothing is read from the disk: an explicit `prefetch=layer` or `router` is
refused there (`auto` means nothing). The slab (`EXL3_MOE_TIER_DISK_SLAB=auto`) is sized for the
deepest read-ahead; a read-ahead that finds no free slot is dropped (`prefetch_starved`), a demand
read takes the slot of the read-ahead farthest ahead (`prefetch_dropped`). Counters: layer mode
`prefetch_issued`, `prefetch_used`, `prefetch_dropped`, `prefetch_starved`; router `predictions`
(records), `predicted_reads`, `slab_to_vram`.

### Profiles and the heat file

A profile is how often each routed expert of each MoE layer was selected on some workload, measured
ahead of time and shipped as a file (`model/moe_profile.py`). `profile=<spec>` on a layer rule names
one or several, with weights (`profile=code`, `profile=code:3,chat:1`, `profile="/data/p/x.exl3moe"`);
counts are normalized per layer before the weighting. Files are found by name in
`<model dir>/moe_profiles/`, then each directory of `EXL3_MOE_PROFILE_DIR` (`os.pathsep`-separated),
then `~/.cache/exllamav3/moe_profiles/`, with the extensions `.exl3moe`, `.safetensors`, `.npz`,
`.json` in that order (a path is taken as it is):

| Format | Contents |
|---|---|
| `.exl3moe`, `.safetensors` | tensors `counts` `[layers, experts]` (any number type) and/or `ranking` `[layers, experts]` (expert ids, most selected first); metadata `layer_keys` (JSON list of the MoE modules' keys) and `fingerprint` (JSON) |
| `.npz` | `counts_decode` (preferred), `counts` or `counts_prefill`: `[layers, experts]`, or `[prompts, layers, experts]` summed; layer keys and fingerprint from a `<name>.meta.json` sidecar when there is one |
| `.json` | `{"<layer key>": [count per expert], ...}` |

Rows are matched to layers by key when the file names them, else by MoE-layer ordinal (the n-th MoE
layer of the model). A profile whose fingerprint names another model (architecture, layer count,
experts per layer, hidden or expert width) is refused; one of another quantization of the same
model is used (a seed, not a placement). What it seeds:

- **`experts=cache`**: the hot pins (`hot=<k>` pins the layer's `k` most selected experts, not its
  first `k`); the cold-fill order (every expert by its share of its layer's selections, highest
  first, across the GPU's cache layers; layers the profile lacks after, in the round robin); and the
  heat (each expert at the decayed count its share would reach in steady decode: share x top-k x
  `EXL3_MOE_HEAT_HALFLIFE` / ln 2). It seeds; it never freezes: from the first call the policies move
  experts by the live heat, which takes over within a few half-lives.
- **`stream` / `cpu` with `hot=`, `split`**: which experts stay resident in VRAM (the hottest `E - k`)
  and which go to system RAM, through the split's expert permutation; the dynamic placement sweeps
  (`EXL3_MOE_CPU_SWAP`) keep moving them by live traffic. A profile takes precedence over
  `EXL3_MOE_CPU_SPLIT_STATS` for its layers.

**The heat file.** `EXL3_MOE_HEAT_FILE=<path>`: the tier's heat (each expert's decayed routing
count) is written there at unload and between generations (the generator's queue drained, at most
once a minute), atomically (a temporary file renamed), in the `.exl3moe` format (`counts` in
decode-assignment units, `layer_keys` the cache layers' keys, `fingerprint`, `format`). A later load
whose placement names no profile reads it back as the seed: the same order and hot pins as a
profile, and the heat exactly as saved. A missing file is simply no seed; an unreadable one, or one
whose fingerprint names another model, is reported and ignored.

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
| cache above the device's room | `placement: cuda:0 cache=90GiB does not fit: 80.0 GiB is free on cuda:0 after the placed modules, the cache and the margins (the prefill staging, 9.5 GiB with EXL3_MOE_TIER_STAGING=double, and EXL3_MOE_TIER_HEADROOM_MB=4096 are set aside first)` |
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
| `disk experts=` / `ngram=<dir>`: no such directory | `placement: disk experts=/nvme1/v41: no such directory` |
| a shard missing from the copy | `placement: disk experts=/nvme1/v41: model-00007-of-00041.safetensors is not there (the copy must hold every shard the reads touch, under the model's file names)` |
| a copy of another size | `placement: disk experts=/nvme1/v41: /nvme1/v41/model-00007-of-00041.safetensors is 4989123 bytes, the model's /models/v41/model-00007-of-00041.safetensors is 4989124; the copy must be the same file` |
| a copy with another header | `placement: disk ngram=/mnt/e: the safetensors header of /mnt/e/model-00002-of-00041.safetensors differs from the model's /models/v41/model-00002-of-00041.safetensors; the copy must be the same file` |
| `disk io=` next to the original gather pool | `placement: disk io=direct sets the page-cache mode of the disk engine's reads, but DeepSeek-V4.1's engram rows are read by the original gather pool (EXL3_DISK_BACKEND=original); unset EXL3_DISK_BACKEND or name a backend, or drop io=` |
| a profile that is not there | `expert profile 'code' not found: looked in /models/v41/moe_profiles, /root/.cache/exllamav3/moe_profiles for code{.exl3moe,.safetensors,.npz,.json}` |
| a profile of another model | `expert profile /data/p/code.exl3moe was built for another model (experts: profile 288, model 384)` (also: `... 288 experts per layer, the model's MoE layers have 384; it was built for another model`) |
| a positional profile of another depth | `expert profile /data/p/code.npz names no layers and has 42 rows, the model has 40 MoE layers` |

### Tuning

The placement says what the tier does; `EXL3_MOE_TIER_*` and `EXL3_MOE_HEAT_*` say how. Every
variable is described in [env_vars.md](env_vars.md) ("Expert tiers"); a load checks them all
when it starts (a malformed value or an unknown name under those prefixes is refused, with a
did-you-mean hint) and prints the ones that differ from their default.

| Variable | Default | Values |
|---|---|---|
| `EXL3_MOE_TIER_DEMOTE` | `1` | `0` drops every VRAM victim (master switch of `ram demote=`) |
| `EXL3_MOE_TIER_STAGING` | `double` | `double` (layer mode overlaps the next layer's copies with this layer's compute), `single` (half the staging VRAM, no overlap) |
| `EXL3_MOE_TIER_PREFILL_ROWS` | `256` | rows from which a call runs in layer mode (2 .. 2^20, above the decode rows; 1048576 keeps every prefill call routed) |
| `EXL3_MOE_TIER_DECODE_ROWS` | `8` | rows up to which a call may admit (1 .. 8) |
| `EXL3_MOE_HEAT_HALFLIFE` | `256` | decode tokens per halving of the heat |
| `EXL3_MOE_HEAT_PREFILL` | `0.0625` | heat of a prefill assignment (0 .. 1) |
| `EXL3_MOE_HEAT_FILE` | unset | path of the heat file (see "Profiles and the heat file") |
| `EXL3_MOE_TIER_ADMIT_P` | unset | pins the adaptive probability (0 .. 1) |
| `EXL3_MOE_TIER_ADAPT_EVERY` | `2048` | accesses per adaptation |
| `EXL3_MOE_TIER_ADMIT_PMIN` | `0.05` | lower bound of the probability |
| `EXL3_MOE_TIER_SAMPLE` | `16` | RAM entries sampled for the coldest |
| `EXL3_MOE_TIER_RAM_ADMIT` | `always` | `always`, `heat` |
| `EXL3_MOE_TIER_REFILL` | `1` | `0` turns refills off |
| `EXL3_MOE_TIER_REFILL_INFLIGHT` | `2` | 1 .. 64 |
| `EXL3_MOE_TIER_DISK_SLAB` | `auto` | slots per cache GPU, 1 .. 4096 (`auto`: see "Sizing a tiered load") |
| `EXL3_MOE_TIER_HUGEPAGE` | `0` | `1`: transparent huge pages for the RAM tier and the slab |
| `EXL3_MOE_TIER_COMPACT` | `1` | `0` keeps the cold-filled RAM slots in the extent layout |
| `EXL3_MOE_TIER_HEADROOM_MB` | `4096` | MiB |
| `EXL3_MOE_TIER_MIN_LINK_GBS` | `2.0` | GB/s; `0` allows any link |
| `EXL3_MOE_TIER_SPIN_US` | `-1` | `-1` while a pass is active, `0` never, N us |
| `EXL3_MOE_TIER_AFFINITY` | unset | CPU list |
| `EXL3_MOE_TIER_DETERMINISTIC` | `0` | `1` applies every call before the next |
| `EXL3_MOE_TIER_VERIFY` | `0` | `1` checks invariants after every call |
| `EXL3_MOE_TIER_TRACE` | unset | path of the call-record trace (JSON lines) |
| `EXL3_MOE_PROFILE_DIR` | unset | directories searched for `profile=` names (`os.pathsep`-separated) |

The policies themselves (`admit=`, `evict=`, `policy=`, `demote=`) are words of the placement, so
one string says what a load does. The defaults are `admit=adaptive`, `evict=lru`, `policy=exclusive`,
`demote=heat`, `ram evict=lfu` and `spare=8`: `admit=`, `evict=` and `spare=` those of the policy
simulation, `policy=` and `demote=` the fastest of every combination measured on the test host
(below; the simulation had chosen `lazy-exclusive` and `swap`).

**The measurement** (DeepSeek-V4.1-Flash 3.0 bpw, the RTX PRO 6000 alone, `CUDA_VISIBLE_DEVICES=0`,
cudaMallocAsync, a 64K-token Cache; one process per combination, loaded fresh; a 32,768-token prompt
of the canonical ids, then 256 greedy tokens, twice on fresh generators, after an untimed prefill of
the same prompt; the second run, warm, is the one below). Configuration (a): `*=cuda:0
experts=cache; ram experts=102GiB`, the largest RAM tier within a 120 GB host budget (the pool
held 5,889 experts, RAM 8,222, the SSD alone 1,249); (b): `ram experts=93GiB` (99.9 GB), 7,496 in
RAM and 1,975 on the SSD alone. Every combination had the same decode hit rate (0.961 for (a), 0.962 for (b)): the
policies differ in where the misses come from and in how much prefill reads from the SSD.

| `policy=` / `demote=` | (a) decode tok/s | (a) prefill tok/s | (b) decode tok/s | (b) prefill tok/s | prefill SSD GB (a / b) | decode SSD MB/token (a / b) |
|---|---|---|---|---|---|---|
| `exclusive` / `heat` (default) | **43.5** | **1,393** | 39.5 | **1,308** | 142 / 224 | 0.3 / 1.0 |
| `exclusive` / `all` | 42.7 | 1,395 | 39.3 | 1,299 | 142 / 224 | 0.2 / 0.8 |
| `exclusive` / `swap` | 41.3 | 1,356 | 41.2 | 1,277 | 167 / 238 | 0.2 / 1.6 |
| `exclusive` / `off` | 38.3 | 819 | 37.7 | 825 | 397 / 450 | 16.3 / 17.6 |
| `lazy-exclusive` / `swap` | 42.3 | 1,302 | 39.7 | 1,277 | 168 / 239 | 0.2 / 1.5 |
| `lazy-exclusive` / `heat` | 38.5 | 1,391 | 41.2 | 1,300 | 142 / 224 | 0.3 / 1.0 |
| `lazy-exclusive` / `all` | 39.7 | 1,397 | 38.5 | 1,291 | 142 / 224 | 0.2 / 0.8 |
| `lazy-exclusive` / `off` | 38.8 | 832 | 40.5 | 844 | 390 / 441 | 8.3 / 9.3 |
| `inclusive` | refused (a) | | 35.9 | 444 | - / 874 | - / 43.5 |

`exclusive` / `heat` has the best mean over (a) and (b) of both decode (41.5 tok/s) and prefill
(1,351 tok/s). With one run per combination the decode differences of the first seven rows (38.5 to
43.5 tok/s, at equal hit rates) are within the run-to-run spread; the prefill ones are not: `swap`
stages about 25 GB (a) and 14 GB (b) more from the SSD per prompt than `heat` and `all`, `off` more
than twice as much, and `inclusive`, whose RAM tier holds only copies of what VRAM holds besides the
rest, reads the SSD for most of every prefill. (a) with `inclusive` was refused at load: its disk
slab is larger (5.7 GiB) and the budgets no longer fit the 120 GB. On the same host the serving
layout without the tier (`0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1`, CMP + PRO, 1M Cache)
decodes 23.2 tok/s and prefills 1,000 tok/s on the same prompt.
