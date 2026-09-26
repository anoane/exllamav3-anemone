# Explicit placement

**Experimental.** An explicit placement says, for every decoder layer of a model, which GPU it
loads on and, for MoE layers, where the routed experts are stored and computed. It works for any
model loaded in layer-split mode, and is set with any one of:

- `EXL3_PLACEMENT`, read when a `Config` is created;
- `-placement` / `--placement` in `model_init`-based scripts (`examples/chat.py`, `eval/perf.py`,
  ...), which overrides the variable;
- `config.infer_params.placement` in Python, set before loading (a string or a parsed
  `Placement`).

```sh
python examples/chat.py -m /path/to/model --placement "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
```

Without a placement (the default) nothing changes: the autosplit fills the GPUs in order until
each one's budget runs out, `-mcl N` keeps the routed experts of the first N eligible MoE layers
in system RAM and `-mcs N` the last N experts of every eligible layer. A placement replaces those
two decisions with one explicit layout: which layers go on which GPU, and which layers' experts
stay in GPU memory, move to system RAM, or split between the two. How to write "no placement"
is described under [No placement](#no-placement).

## Syntax

One rule per layer range, plus optional storage rules. Rules are separated by `;` or newlines,
and `#` starts a comment that runs to the end of the line. Double quotes protect a value that
holds spaces, `;`, `#` or `"` (`\"` and `\\` escape inside them):

```
<layers>=cuda:<n> [experts=<mode>] [hot=<k>|<p>%] [cpu=<k>] [prefetch=...] [profile=...]
cuda:<n> <cache attributes>     # the expert cache of one GPU (experts=cache layers)
ram <attributes>                # RAM budgets of the component, the RAM tier's policy
disk <attributes>               # where experts and n-gram rows are read from at runtime
```

| Part | Values |
|---|---|
| `<layers>` | Decoder-layer indices counted from 0, as the model numbers them (the `N` of `layers.N` in its tensor names): a single layer `7`, an inclusive range `0-11`, or a list of both `0-3,8-11` (ASCII digits). `*` places every layer that no other rule lists. `embed` places the modules before the first layer, `head` the modules after the last one (final norm, output head). |
| `cuda:<n>` | The device, a logical CUDA index after `CUDA_VISIBLE_DEVICES` remapping (as `torch` numbers it, not necessarily as `nvidia-smi` does). |
| `experts=<mode>` | Where an MoE layer's routed experts live. Default `vram`. See below. |
| `hot=<k>` or `hot=<p>%` | With `experts=cache`, `stream` or `cpu`: routed experts per layer kept resident in VRAM, a count or a share of the layer's routed experts (rounded half up). `hot=0` is the default. At least one expert must stay non-resident (`hot=E` asks for `experts=vram`). |
| `cpu=<k>` | With `experts=split` only, and required there: routed experts per layer held in system RAM. |
| `prefetch=` | With `experts=cache` only: the disk read-ahead of the layer's experts. `auto` (the default, `layer:1`), `off`, or methods joined by `+`: `layer[:<depth>]` (during a layer-mode prefill, read the disk-only experts of the next `depth` layers while the current one computes), `router[:<depth>]` (during decode, run the router of the cache layer `depth` ahead on the current layer's input and read the experts it predicts). Refused with `disk experts=off`, which never reads the disk. See [expert_tiers.md](expert_tiers.md), "Read-ahead: prefetch=". |
| `profile=` | Not with `vram`, and with `stream` / `cpu` only next to `hot=`: an expert profile, one name or path or several with weights (`code:3,wiki:1`), looked up in `<model dir>/moe_profiles/`, `EXL3_MOE_PROFILE_DIR` and `~/.cache/exllamav3/moe_profiles/`. On `cache` layers it seeds the `hot=` pins, the order of the cold fill and the heat; on `stream` / `cpu` / `split` layers it chooses the experts that stay in VRAM. See [expert_tiers.md](expert_tiers.md), "Profiles and the heat file". |

The expert modes, with the older settings each one corresponds to:

| `experts=` | Storage | Computation | Older equivalent |
|---|---|---|---|
| `vram` | GPU memory of the layer's device | that GPU | no offload |
| `cache` | the GPU's expert cache, over the RAM tier and the disk ([expert_tiers.md](expert_tiers.md)); `hot=<k>` pins `k` experts per layer | the layer's GPU, from VRAM, bit-identical to `vram` | new |
| `stream` | system RAM | every expert a call selects is streamed to the layer's GPU and computed there, one-token decode included; no CPU expert arithmetic | `-mcl` with `-mcm stream_only` (`EXL3_MOE_CPU_MODE`) |
| `stream hot=<k>` | `k` routed experts per layer in GPU memory, the rest in system RAM | the resident ones on the GPU, the others streamed to it; dynamic placement (`EXL3_MOE_CPU_SWAP`) re-chooses the resident ones between generations | `-mcs E-k` with `-mcm stream_only` |
| `cpu` | system RAM | the CPU expert worker; during prefill the experts with at least `EXL3_MOE_STREAM_T` assignments in a chunk of at least `EXL3_MOE_STREAM_MIN_ROWS` rows are streamed to the GPU | `-mcl` |
| `cpu hot=<k>` | `k` routed experts per layer in GPU memory, the rest in system RAM | the worker computes the RAM share, overlapping the GPU's share; dynamic placement as for `split` | `-mcs E-k` |
| `split cpu=<k>` | the last `k` routed experts in system RAM, the others in GPU memory | the worker computes the RAM share, overlapping the GPU's share; with dynamic placement (`EXL3_MOE_CPU_SWAP`, on by default) the most-selected experts move to the GPU slots between generations | `-mcs k` |
| `hybrid hot=<k>` | the older spelling of `cpu hot=<k>` (`hot=` required), printed as `cpu hot=<k>` | | |

`cpu hot=<k>` and `split cpu=<E-k>` are the same layout, counted from the two sides; both
spellings stay as written (turning one into the other needs the layer's number of experts).

The storage rules, in short (each is described in full, with its defaults and checks, in
[expert_tiers.md](expert_tiers.md)):

| Rule | Attributes | Applies to |
|---|---|---|
| `cuda:<n>` | `cache=auto\|0\|<size>`, `spare=<k>`, `evict=lru\|lfu`, `admit=adaptive\|heat\|always`: the expert cache of that GPU | `experts=cache` layers on that GPU (refused without them) |
| `ram` | `experts=auto\|all\|<size>`, `ngram=auto\|all\|<size>`, `pagecache=auto\|<size>`: the component's RAM budgets; `policy=`, `demote=`, `evict=`: the RAM tier of `cache` layers | `experts=` the RAM-held experts of `stream` / `cpu` / `split` layers and the RAM tier of `cache` layers; `ngram=` the n-gram and engram tables; the tier words need `experts=cache` layers |
| `disk` | `experts=model\|off\|<dir>`, `ngram=model\|off\|<dir>`, `io=auto\|direct\|buffered` | `experts=` the reads of `cache` layers (the only experts read at runtime); `ngram=` the n-gram and engram rows (`off` needs `ram ngram=all`); `<dir>` is a checked copy of the model's shards; `io=` the page-cache mode of both kinds of reads |

Sizes are written as for the RAM budgets: `48GiB` (binary), `48GB` (decimal), `48` (GiB), never
the ambiguous `48G`; see [expert_tiers.md](expert_tiers.md).

Details:

- Layers must be placed exactly once: a layer in two rules, or a layer of the model that no rule
  places (and no `*` rule), is an error. A rule naming a layer the model does not have is an
  error too, so a placement written for one model fails loudly on another.
- `embed` and `head` default to the device of the first and of the last layer. Modules the
  loader always keeps in system RAM (the token embedding) stay there whatever `embed` says; the
  rule then applies to the other modules before the first layer. `embed` and `head` take no
  `experts=`.
- A module between two layers (rare) follows the module before it. A module that belongs to a
  layer without being one follows that layer: the per-layer embedding modules of PLE models
  (e.g. Qwen3.8-Flash-Next), which sit right before their layer, go where the layer goes.
- `experts=` applies to layers with routed experts. On other layers (the dense layers of a model
  that mixes both, or every layer of a dense model) it has no effect, so a `*` rule may carry it
  over a mixed model.
- Case does not matter for keywords and keys (`CUDA:1`, `Experts=Stream`, `RAM Experts=48gib`);
  values that name things (paths, profiles) keep their case. Whitespace around rules and around
  the first `=` of a layer rule is ignored; attributes are written without spaces
  (`experts=stream`, not `experts = stream`, and `48GiB`, not `48 GiB`), and so are ranges
  (`0-11`); whitespace around a comma in a list is ignored (`0-3, 8-11` and `0-3 ,8-11` read as
  `0-3,8-11`). Numbers are ASCII digits.
- Each storage rule appears at most once (one `cuda:<n>` rule per GPU, one `ram`, one `disk`); a
  rule that only restates defaults is dropped (`ram pagecache=auto` says nothing).
- The placement prints, and compares, in one canonical order: `embed`, the explicit layer rules
  by first layer, `*`, `head`, then the `cuda:<n>` rules by index, `ram`, `disk`; attributes in a
  fixed order (layer: `experts hot cpu prefetch profile`; device: `cache spare evict admit`;
  ram: `experts ngram pagecache policy demote evict`; disk: `experts ngram io`), defaults
  omitted, sizes in the largest exact unit, `hybrid hot=K` as `cpu hot=K`, values quoted only
  when they need it. `str()` of a parsed placement is that text, and parses back to an equal
  placement; placements are equal when their canonical texts are.
- One kind of RAM-held experts per GPU: GPU-computed layers (`stream`, `cache`) and CPU-computed
  layers (`cpu`, `split`) must be placed on different GPUs. The CPU worker keeps one
  streamed-prefill state per GPU (its bandwidth probe and streaming threshold), which the two
  kinds use differently.
- Other forms of the same placement: a dict (`Placement.to_dict()`: `{"rules": [{"layers": "0-11",
  "device": "cuda:0", "experts": "cpu"}, ...], "devices": {"cuda:1": {...}}, "ram": {...},
  "disk": {...}}`), the same as a JSON object in a string, or `@<path>` to a file holding the rule
  text (comments and newlines included) or the JSON. `parse()` takes all of them, and
  `config.infer_params.placement` accepts a dict too.
- Every word of the grammar runs. The words that name files (`profile=`, `disk experts=<dir>`,
  `disk ngram=<dir>`) and `disk io=` are checked again at load, before anything loads, against the
  files they name and the disk backend: a profile that is missing or was built for another model,
  a copy whose shards differ from the model's, `io=` with the original n-gram gather pool
  ([expert_tiers.md](expert_tiers.md), "Load-time refusals of the tier").

## No placement

"No placement" has exactly one meaning and one representation: `config.infer_params.placement`
is `None`, and the model loads as it would without this feature (the autosplit places the layers,
`-mcl` / `-mcs` / `-mcm` place the experts). Every way of saying it:

| Spelling | Where | Result |
|---|---|---|
| `EXL3_PLACEMENT` not set | environment | `None` |
| `EXL3_PLACEMENT=` (empty) | environment | `None` |
| a value with only whitespace, `;` separators and `#` comments, e.g. `EXL3_PLACEMENT="  ; # later"` | environment, `--placement`, attribute | `None` |
| no `--placement` option | `model_init` scripts | the variable decides (`None` when it is unset or empty) |
| `--placement ""` | `model_init` scripts | `None`, also when `EXL3_PLACEMENT` is set: the option overrides the variable, so this clears it |
| `config.infer_params.placement = None` | Python | `None` |
| `config.infer_params.placement = ""` | Python | `None` (parsed when the model is built or loaded) |
| `@<path>` to a file holding only whitespace, `;` separators and `#` comments (e.g. `@/dev/null`) | environment, `--placement`, attribute | `None` |
| a dict, or a JSON object in a string, whose `rules` list is empty and that has no storage rule (`devices`, `ram` and `disk` absent or empty), e.g. `{"rules": []}` | attribute (dict); environment, `--placement`, attribute (JSON) | `None` |

Refused with a `ValueError`, whatever the case: a whole value of `none`, `off`, `auto`,
`default`, `0` or `false` (also with surrounding whitespace, `;` or a `#` comment), e.g.
`placement 'none': no placement is written as an empty value, not 'none': leave EXL3_PLACEMENT
unset or empty, pass --placement "", or set config.infer_params.placement = None, and the
autosplit decides as usual`. These words are refused rather than accepted as a second spelling
of "no placement": a placement is a layout, and a value that reads like a switch is more likely a
mistake (a variable meant for another setting, a half-edited script) than a request for the
autosplit. The variable's value is refused when a `Config` is created, `--placement`'s when
`model_init` builds the `Config`, and the attribute's when the model is built (DeepSeek-V4.1) or
loaded.
The same words are refused as the whole content of an `@<path>` file; the message then names
the `@<path>` value.

A placement that puts every layer on one GPU (`*=cuda:0`) is a placement, not "no placement": it
fixes the device (the budgets no longer choose it), replaces `-mcl` / `-mcs` / `-mcm`, and is
refused next to them like any other placement. Storage rules without layer rules (`ram
experts=48GiB` alone) are not "no placement" either: they are refused (without a placement, RAM
budgets are set with `--expert_ram` / `--ngram_ram`).

## Examples

One GPU, experts of the last ten of 60 layers in system RAM, computed by the CPU (`-mcl` can only
take the first layers):

```
*=cuda:0; 50-59=cuda:0 experts=cpu
```

One GPU, 96 of every layer's routed experts on the CPU worker, as `-mcs 96`:

```
*=cuda:0 experts=split cpu=96
```

Two GPUs with a fixed split at layer 24, whatever the budgets would pick:

```
0-23=cuda:0; 24-47=cuda:1
```

Two GPUs, the experts of layers 12-22 kept in system RAM and streamed to the second GPU, which
has the faster link, while the first GPU keeps all of its experts in its own memory:

```
0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1
```

The same layout with those experts computed by the CPU worker instead:

```
0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1
```

GPUs in reverse order, the output head on the GPU that holds the first layers:

```
0-19=cuda:1; 20-39=cuda:0; head=cuda:1
```

Four GPUs, streamed experts on one of them and CPU-computed experts on another (two kinds of
RAM-held experts, on different GPUs):

```
0-14=cuda:0; 15-29=cuda:1; 30-44=cuda:2 experts=stream; 45-59=cuda:3 experts=cpu
```

One GPU, 32 of every layer's 128 routed experts resident, the rest on the CPU worker (the same as
`experts=split cpu=96`, i.e. `-mcs 96`):

```
*=cuda:0 experts=cpu hot=32
```

GPU streaming with a resident slice: 64 experts per layer in VRAM, the rest streamed from RAM
(`-mcs E-64 -mcm stream_only`):

```
*=cuda:1 experts=stream hot=64
```

The serving layout of DeepSeek-V4.1 with the RAM of its CPU-computed layers capped at 56 GiB,
and a PLE model's n-gram tables held in RAM up to 12 GiB:

```
0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1; ram experts=56GiB
*=cuda:0; ram ngram=12GiB
```

A multi-line value, e.g. in a shell script:

```sh
export EXL3_PLACEMENT="
0-11  = cuda:0              # first GPU: every expert in VRAM
12-22 = cuda:1 experts=cpu  # experts of these layers in system RAM
*     = cuda:1
"
```

Python:

```python
from exllamav3 import Config, Model, Cache

config = Config.from_directory(model_dir)
config.infer_params.placement = "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 65536)
model.load(use_per_device = [22, 44])       # budgets cap memory; they do not move layers
```

The same as a dict, as JSON, or from a file:

```python
config.infer_params.placement = {"rules": [{"layers": "0-11", "device": "cuda:0"},
                                           {"layers": "12-22", "device": "cuda:1", "experts": "stream"},
                                           {"layers": "23-39", "device": "cuda:1"}]}
```

```sh
python examples/chat.py -m /path/to/model --placement @/etc/exl3/v41.placement
```

`exllamav3.model.placement.parse(text)` returns the parsed `Placement` (or `None` for an empty
value, see [No placement](#no-placement)) and raises `ValueError` on a malformed one;
`placement.layer_map(range(num_layers))` checks that it covers a model's layers, without loading
anything; `placement.to_dict()` gives the dict form.

## How a load applies it

When `model.load()` starts, the placement is parsed and checked against the model and the load
arguments before anything loads (see the refusals below). Then the layer-split loader loads the
modules in order, each on the device the placement gives it, with the same measuring forward and
headroom checks as the autosplit:

- Budgets (`-gs` / `use_per_device`, `reserve_per_device`, or neither, which leaves every
  visible GPU usable with the default 0.5 GB reserve) cap each device's memory, set once per
  device per load. They never move a module. Every GPU the placement names must have a budget:
  a non-zero `use_per_device` entry, or a `reserve_per_device` entry that is not negative.
- A module that does not fit on its device, including room for the largest transient the loader
  measured there (see `EXL3_AUTOSPLIT_WORSTCASE` and `EXL3_AUTOSPLIT_MARGIN_MB` in
  [env_vars.md](env_vars.md); a quantized cache's prefill staging, `EXL3_QC_STAGING`, counts
  there as well), fails the load with a `RuntimeError` that names the module, the device and the
  loader's reason, e.g. `placement: model.layers.12 does not fit on cuda:0 with the modules
  already placed there (autosplit: cuda:0 has no headroom left for the largest transient measured
  on it (812 MiB)); ...`. Where the autosplit would move the module to the next device, a
  placement stops: move layers to another device, move their experts to `stream` / `cpu` /
  `split`, or raise that device's budget.
- Each MoE layer takes its storage and computation from its rule, through the same worker and
  streaming machinery as `-mcl` / `-mcs` / `-mcm`. A layer whose rule asks for RAM-held experts
  must be able to hold them: a gated `silu` / `gelu` / `swiglu_oai` or ungated `relu2` activation,
  mul1-codebook experts with K <= 8, and uniform per-expert biases. A layer that cannot is a
  `RuntimeError` that asks for `experts=vram`, never a silent fallback to GPU memory: the device
  was planned without those experts. For `split`, `k` must also lie between 1 and the layer's
  number of routed experts minus 1 (a `ValueError` otherwise). With `-mcl` / `-mcs` an ineligible layer still falls back silently, as before.
- The loader reports RAM-held layers as with `-mcl` / `-mcs`, e.g. `-- CPU-offloaded experts
  (streamed to cuda:1, no CPU compute): model.layers.12.mlp` or `-- CPU-offloaded experts
  (worker): model.layers.12.mlp`.
- `hot=<k>` on `stream` / `cpu` layers runs through the split machinery (as `-mcs E-k`, with
  `-mcm stream_only` for `stream`), so the layer's resident slice holds `k` experts.
- The `ram` rule's requests are checked before anything loads, against the flags (`ram
  experts=64GiB` next to `--expert_ram 48GiB` is refused: the flags are caps), against what the
  `stream` / `cpu` / `split` layers take (`ram experts=48GiB` below it is refused), and with the
  n-gram tables against the host memory; `ram ngram=` decides which n-gram tables load in RAM.
  See [expert_tiers.md](expert_tiers.md).

## Refusals

When the value is parsed (at `Config` creation for `EXL3_PLACEMENT`, in `model_init` for
`--placement`, at `Model.from_config` for a DeepSeek-V4.1 model, at `model.load()` otherwise), a
`ValueError` for:

- a rule without `=`, or without a device after it;
- a device other than `cuda:<n>` (there is no `cpu` device: RAM-held experts are `experts=stream`,
  `cpu` or `split`, and the token embedding already lives in system RAM);
- malformed or backwards layer ranges (digits other than ASCII included), a layer listed twice in
  one rule or placed by two rules, `*`, `embed` or `head` placed twice;
- an unknown attribute or `experts=` value (with a did-you-mean hint, or the rule a misplaced
  key belongs to), an attribute given twice, `experts=` on `embed` / `head`, `split` without
  `cpu=`, `cpu=` without `split`, `cpu=` not a positive integer of ASCII digits;
- `hot=` on `vram` or `split`, `hybrid` without `hot=`, a share outside (0%, 100%), `prefetch=`
  on anything but `cache` (or a malformed method list), `profile=` on `vram` or on `stream` / `cpu`
  without `hot=`;
- GPU-computed (`stream`, `cache`) and CPU-computed (`cpu`, `split`) layers on the same GPU;
- an unknown rule (`rams ...`), a storage rule given twice or with `=<value>`, a malformed size
  (`48G` is ambiguous), and every storage check of [expert_tiers.md](expert_tiers.md) (e.g. `ram
  experts=` with every layer in VRAM, a storage rule without layer rules, `disk ngram=off`
  without `ram ngram=all`, `prefetch=` with `disk experts=off`);
- a whole value of `none`, `off`, `auto`, `default`, `0` or `false`, also as the whole content of
  an `@<path>` file (see [No placement](#no-placement)).

At `model.load()`, before anything loads, a `ValueError` for:

- any of the settings a placement replaces: `-mcl` / `--moe_cpu_offload` /
  `EXL3_MOE_CPU_OFFLOAD`, `-mcs` / `--moe_cpu_split` / `EXL3_MOE_CPU_SPLIT`, `-mcm` /
  `--moe_cpu_mode` / `EXL3_MOE_CPU_MODE` other than `compute`, and `EXL3_MOE_CPU_SPLIT_LAYERS`.
  The error lists every one that is set; there is no precedence rule. (An invalid
  `moe_cpu_mode` value is reported as invalid first.);
- a single-device load (`model.load(device = ...)`) or tensor-parallel loading (`-tp`);
- a rule naming a layer the model does not have, or a layer no rule places;
- a device beyond the visible GPUs, or one the budgets exclude;
- a `ram` request above its flag (`placement asks ram experts=64GiB (64.0 GiB) but --expert_ram
  caps this component at 48GiB`, the same for `ngram=` and `--ngram_ram`), and `stream` / `cpu`
  / `split` layers that keep more experts in RAM than `ram experts=` or `--expert_ram` allows
  (`placement: the stream / cpu layers keep 52.4 GiB of routed experts in RAM, more than ram
  experts=48GiB; ...`).

Then a `RuntimeError` when the host cannot hold what the load keeps in RAM (the RAM-held experts
and n-gram tables together, plus `EXL3_HOST_MEM_RESERVE_MB`, against MemAvailable or the memory
cgroup's limit; see [expert_tiers.md](expert_tiers.md)).

While the layers load: a `ValueError` for a `split` layer whose `cpu=` is not below its number of
routed experts, and the two `RuntimeError`s described above (a module that does not fit its
device, a layer that cannot hold its RAM-held experts).

DeepSeek-V4.1 adds refusals of its own, when the model object is built and at load; see
[DeepSeek-V4.1](#deepseek-v41).

## What it applies to

- The text component only: the placement names its layers. A draft model loaded from its own
  directory, an MTP head and a vision tower load as without a placement (an MTP head sharing the
  model's config still takes `-dmcl`). `model_init` clears the placement of a draft model's
  config; a Python script that builds a draft's `Config` itself gets `EXL3_PLACEMENT` there too,
  and sets `draft_config.infer_params.placement = None` (or the draft's own placement). The
  RAM-held experts of `-dmcl` layers are computed by the CPU worker under a placement: `-mcm` is
  refused next to it, so their mode stays `compute`.
- Relayered models (`--layer_map`): the indices are the model's own layers; a repeated layer runs
  on its one device every time.
- The cache: each attention layer's cache lives on that layer's device, as with the autosplit.
- The tuning of RAM-held experts (`EXL3_MOE_CPU_THREADS` / `-mct`, `EXL3_MOE_CPU_WSLOT_MB`,
  `EXL3_MOE_CPU_WSLOTS`, `EXL3_MOE_STREAM_*`, `EXL3_MOE_PINNED_ARENA`, `EXL3_MOE_CPU_SWAP`, ...)
  applies unchanged.
- DeepSeek-V4.1 reads the placement when the model object is built and restricts where it may
  change device; see [DeepSeek-V4.1](#deepseek-v41).

## Memory, performance and determinism

The placement adds no work of its own. Each module runs where it is placed, and every change of
device between consecutive modules copies the running state to the next device once per forward
pass (rows x hidden size x element size), as a split made by the autosplit does; a placement that
returns to an earlier device pays that copy at each change. GPU memory per device is what is
placed there plus the loader's transients. System RAM holds every expert of the `stream` and
`cpu` layers and the RAM share of the `split` layers, plus the worker's pinned staging ring, as
with `-mcl` / `-mcs`. `-er` / `--expert_ram` (`EXL3_EXPERT_RAM`) caps that RAM, and before
anything loads the loader counts it from the checkpoint headers and checks it, with the n-gram
tables `--ngram_ram` holds, against the host memory (MemAvailable, or the memory cgroup's limit):
see [expert_tiers.md](expert_tiers.md). The cost of the expert modes is theirs: see `EXL3_MOE_CPU_MODE` for
`stream` (the transfer per decoded token), and the CPU MoE offload section of
[env_vars.md](env_vars.md) for `cpu` and `split`.

The same placement puts the same modules on the same devices on every load, and a placement
computes nothing differently from the older settings that express the same layout (`-mcl`,
`-mcs`, `-mcm`, which it drives through the same machinery). On DeepSeek-V4.1-Flash, an earlier
revision of this code gave per-token log-probabilities of a 64K-token scoring run loaded through
`0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1` bitwise identical to the same layout set
up with model-specific expert settings that this code no longer has (its V4.1 expert claims now
take the generic path); that comparison has not been repeated on this code. Which GPU runs a
layer can change that layer's results, as with any split.

## When to use it

- The split must not depend on free memory, e.g. to compare runs.
- DeepSeek-V4.1 across GPUs with a change of device inside a kv group (its pool replica needs
  the explicit split, see below).
- The layers whose experts go to system RAM should not be the first N: next to the GPU with the
  faster host link, away from a GPU that should stay resident, or wherever there is room.
- Different layers need different expert modes, e.g. GPU-streamed experts on one GPU and
  CPU-computed ones on another.
- The GPUs should be used in another order than their indices.

For a plain split that the budgets already produce, or for a single GPU without offloading, the
autosplit does the same with less to write.

## DeepSeek-V4.1

DeepSeek-V4.1 applies a placement like every other model (each layer on its device, the routed
experts of each MoE layer as its rule says), with three differences, because its compressed
layers share state across layers.

Background. A V4.1 compressed layer does not keep its own compressed KV: the first layer of each
kv group (its kv source) owns one paged pool, and every other layer of the group reads it (on
DeepSeek-V4.1-Flash the groups are layers 2-7, 8-13, 14-19 and 20-39). Each index source (2, 8,
14, 20, 24, 28, 32 and 36 on Flash) publishes a top-k selection that the layers above it reuse,
up to the next index source, and the candidate source (layer 20 on Flash) publishes candidate
blocks for the index sources above it. The attention kernels read all of these on the reading
layer's own device. A change of device between layers `c - 1` and `c` is a *free cut* when
nothing of this is shared across it; on Flash the free cuts are layers 1, 2, 8, 14 and 20
("Loading across GPUs" in the DeepSeek-V4.1 section of [env_vars.md](env_vars.md) derives them
and covers loading without a placement).

### When it is read

When the model object is built (`Model.from_config`), not only at load: a split inside a kv group
needs a pool replica (below), which the Cache allocates when it is created, and a Cache can be
created before the model loads. Set `EXL3_PLACEMENT`, `--placement` or
`config.infer_params.placement` before `Model.from_config`; a placement set only afterwards does
not add a replica (and is refused at load if it needs one, below).

### Where it may change device

Every change of device between decoder layers must be a free cut, except at most one, the
*split*, which may fall anywhere else. At the split:

- the first layer past it owns a replica of its kv group's pool, on its own device, allocated by
  the Cache like any other pool, and every forward copies the entries the kv source added into
  the replica;
- the other layers of the group past the split read that replica;
- the top-k selections and candidate blocks produced below the split and read past it are copied
  to the other device once per forward.

No layer that reads across the split reads across another change of device: nothing is shared
across a free cut. So one split among any number of free cuts, on any number of GPUs, is exact.

Examples on DeepSeek-V4.1-Flash:

| Placement | Changes of device | Result |
|---|---|---|
| `0-13=cuda:0; 14-39=cuda:1` | 14 (free) | no replica |
| `0-7=cuda:0; 8-19=cuda:1; 20-39=cuda:2` | 8 and 20 (free) | no replica |
| `0-11=cuda:0; 12-39=cuda:1` | 12 (the split) | layer 12 owns a replica of layer 8's pool; layers 12-13 read it |
| `0-11=cuda:1; 12-39=cuda:0` | 12 (the split), GPUs reversed | the same, on the other GPUs |
| `0-11=cuda:0; 12-19=cuda:1; 20-39=cuda:2` | 12 (the split) and 20 (free) | replica of 8 on layer 12 |
| `0-5=cuda:0; 6-11=cuda:1; 12-39=cuda:0` | 6 and 12, both inside a kv group | refused |
| `*=cuda:0` | none | no replica |

A placement with more than one change of device outside the free cuts is refused with a
`ValueError` when the model object is built, e.g.

`DeepSeek-V4.1: the explicit placement '0-5=cuda:0; 6-11=cuda:1; 12-39=cuda:0' changes device in 2 places where a pool, a top-k selection or candidate blocks are shared across the change: at layer 6 (layers 2-7 share the compressed-KV pool of layers.2) and at layer 12 (layers 8-13 share the compressed-KV pool of layers.8). At most one change of device may fall there (the layers past it then read a replica of the pool); every other one must fall at layer 1, 2, 8, 14 or 20 on this model (doc/placement.md, "DeepSeek-V4.1")`

What each split costs on DeepSeek-V4.1-Flash. Replica sizes are for `max_num_tokens` = 1M in
FP16 and scale linearly with the Cache size; a quantized Cache (`-cq`) shrinks the replica's NoPE
part as it does the source's.

| Change of device at layer | Replica | Replica size at 1M | Also crosses every forward |
|---|---|---|---|
| 1, 2, 8, 14, 20 (a free cut) | none | 0 | nothing |
| 3 .. 7 | layer k, of source 2 (rate 2, no index keys) | 512 MiB | layer 2's top-k selection |
| 9 .. 13 | layer k, of source 8 (rate 2, no index keys) | 512 MiB | layer 8's top-k selection |
| 15 .. 19 | layer k, of source 14 (rate 2, no index keys) | 512 MiB | layer 14's top-k selection |
| 21 .. 36 | layer k, of source 20 (rate 1, with its index keys, which index sources 24/28/32/36 past the split borrow) | 1.25 GiB | the top-k selection of the last index source below k (nothing when k is itself an index source); layer 20's candidate blocks for the index sources at or past k |
| 37 .. 39 | layer k, of source 20 (rate 1, no index keys) | 1 GiB | layer 36's top-k selection |

Per forward, the entries a forward adds to the source pool are copied to the replica, 1 KiB per
entry without index keys and 1.25 KiB with them (1 MiB for a 2048-token prefill chunk at rate 2;
1 KiB for every second decoded token at rate 2, 1.25 KiB for every decoded token at rate 1, 1 KiB
for splits 37-39, whose replica has no index keys). A top-k selection crosses once per forward
per device, 2 KiB per query row, and the candidate blocks 8 KiB per row. As at any change of
device, the hidden state crosses into the first block past it too: for V4.1 the residual streams,
4 x 5120 FP32 values per token (80 KiB per token, 160 MiB for a 2048-token chunk), the largest
item here (`EXL3_DSV41_XDEV_BF16` halves it).

Replica copies go directly between GPUs whose driver reports peer access and that pass
`EXLLAMA_NO_P2P_COPY`'s check; otherwise (no peer access, `EXLLAMA_NO_P2P_COPY=1`, or a failed
probe) they go through one page-locked host buffer per device pair, grown as needed up to 8 MiB
(longer ranges move in pieces) and ordered with CUDA events, so the host never waits for them.
`EXLLAMA_NO_P2P_COPY=0` does not force a direct replica copy between GPUs without peer access.
The top-k selections and candidate blocks move with the engine's ordinary device copy
(`util/device_copy.py`, `to_device`): a driver copy, or, when `EXLLAMA_NO_P2P_COPY` or the probe
says the pair must go through the host, a blocking copy through pageable host memory.

### At load

The placement set at load is parsed and checked again, and the pool routes it needs are compared
with those the model was built with:

- the same routes: the load goes on. Devices, expert storage and free cuts may differ from the
  placement the model was built with, and a placement that had only free cuts may even be
  cleared (the routes are then the default ones either way);
- other routes: a `ValueError` before anything loads, e.g.
  `DeepSeek-V4.1: the model was built for no pool replica, but config.infer_params.placement is now '0-11=cuda:0; 12-39=cuda:1', which needs a pool replica for a change of device at layer 12. The cross-device pool routes are fixed when the model object is built (the Cache is sized from them): set the placement before Model.from_config, or build a new model to change it`.

Tensor-parallel loading (`-tp`) is refused for V4.1 with or without a placement. The general
checks of this page (settings the placement replaces, layer coverage, visible and budgeted GPUs)
apply as for every model.

After a load with a placement, one line reports where the modules landed and what crosses
between the devices, e.g.
`-- DSV41 placement [0-11=cuda:0; 12-39=cuda:1]: cpu embed | cuda:0 hc_expand,layers.0-11 | cuda:1 layers.12-39,hc_head,norm,head | replicas 12<-8 | top-k cross 8->12-13 | candidates cross none`
(after a real load each `cuda:N` is followed by the card's name in parentheses). A split at 22
reports `replicas 22<-20+idxK | top-k cross 20->22-23 | candidates cross 20->24,28,32,36`
(`+idxK`: the replica carries index keys).

### Examples

Two cards, the tail 192 of each layer's 384 routed experts in system RAM on the CPU worker (as
`-mcs 192`, about 95 GiB of RAM), the change of device inside kv group 8-13:

```sh
python examples/chat.py -m /path/to/DeepSeek-V4.1-Flash-exl3 -gs 62,90 \
    --placement "0-11=cuda:0 experts=split cpu=192; 12-39=cuda:1 experts=split cpu=192" ...
```

```python
config = Config.from_directory(model_dir)
config.infer_params.placement = "0-11=cuda:0 experts=split cpu=192; 12-39=cuda:1 experts=split cpu=192"
model = Model.from_config(config)                   # the placement is read here
cache = Cache(model, max_num_tokens = 262144)       # allocates layer 12's replica of 8 on cuda:1
model.load(use_per_device = [62, 90])
```

The layout of the long-context measurements in this series: the experts of layers 12-22 held in
system RAM and streamed to the second GPU (computed there), the first GPU keeping all of its
experts:

```
0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1
```

Determinism: the replica is a bitwise copy of the source's entries, and the placement changes no
arithmetic of V4.1's own code. As with any split, which GPU runs a layer can change the results of
the kernels on that layer.

See also `EXL3_DSV41_PIPELINE` (pipelined prefill across a change of device) and
`EXL3_DSV41_XDEV_BF16` (the residual streams' crossing) in [env_vars.md](env_vars.md).
