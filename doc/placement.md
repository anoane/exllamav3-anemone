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

One rule per layer range. Rules are separated by `;` or newlines, and `#` starts a comment that
runs to the end of the line:

```
<layers>=cuda:<n> [experts=<mode>] [cpu=<k>]
```

| Part | Values |
|---|---|
| `<layers>` | Decoder-layer indices counted from 0, as the model numbers them (the `N` of `layers.N` in its tensor names): a single layer `7`, an inclusive range `0-11`, or a list of both `0-3,8-11` (ASCII digits). `*` places every layer that no other rule lists. `embed` places the modules before the first layer, `head` the modules after the last one (final norm, output head). |
| `cuda:<n>` | The device, a logical CUDA index after `CUDA_VISIBLE_DEVICES` remapping (as `torch` numbers it, not necessarily as `nvidia-smi` does). |
| `experts=<mode>` | Where an MoE layer's routed experts live. Default `vram`. See below. |
| `cpu=<k>` | With `experts=split` only, and required there: routed experts per layer held in system RAM. |

The expert modes, with the older settings each one corresponds to:

| `experts=` | Storage | Computation | Older equivalent |
|---|---|---|---|
| `vram` | GPU memory of the layer's device | that GPU | no offload |
| `stream` | system RAM | every expert a call selects is streamed to the layer's GPU and computed there, one-token decode included; no CPU expert arithmetic | `-mcl` with `-mcm stream_only` (`EXL3_MOE_CPU_MODE`) |
| `cpu` | system RAM | the CPU expert worker; during prefill the experts with at least `EXL3_MOE_STREAM_T` assignments in a chunk of at least `EXL3_MOE_STREAM_MIN_ROWS` rows are streamed to the GPU | `-mcl` |
| `split` | the last `k` routed experts in system RAM, the others in GPU memory | the worker computes the RAM share, overlapping the GPU's share; with dynamic placement (`EXL3_MOE_CPU_SWAP`, on by default) the most-selected experts move to the GPU slots between generations | `-mcs k` |

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
- Case does not matter (`CUDA:1`, `Experts=Stream`). Whitespace around rules and around the
  first `=` is ignored; attributes are written without spaces (`experts=stream`, not
  `experts = stream`), and so are ranges (`0-11`); whitespace around a comma in a list is ignored
  (`0-3, 8-11` and `0-3 ,8-11` read as `0-3,8-11`). Numbers are ASCII digits.
- The placement prints, and compares, in one canonical order: `embed`, the explicit layer rules
  by first layer, `*`, `head`. `str()` of a parsed placement is that text, and parses back to an
  equal placement.
- One kind of RAM-held experts per GPU: `stream` layers and `cpu` / `split` layers must be placed
  on different GPUs. The CPU worker keeps one streamed-prefill state per GPU (its bandwidth probe
  and streaming threshold), which the two kinds use differently.

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

A placement that puts every layer on one GPU (`*=cuda:0`) is a placement, not "no placement": it
fixes the device (the budgets no longer choose it), replaces `-mcl` / `-mcs` / `-mcm`, and is
refused next to them like any other placement.

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

`exllamav3.model.placement.parse(text)` returns the parsed `Placement` (or `None` for an empty
value, see [No placement](#no-placement)) and raises `ValueError` on a malformed one;
`placement.layer_map(range(num_layers))` checks that it covers a model's layers, without loading
anything.

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

## Refusals

When the value is parsed (at `Config` creation for `EXL3_PLACEMENT`, in `model_init` for
`--placement`, at `Model.from_config` for a DeepSeek-V4.1 model, at `model.load()` otherwise), a
`ValueError` for:

- a rule without `=`, or without a device after it;
- a device other than `cuda:<n>` (there is no `cpu` device: RAM-held experts are `experts=stream`,
  `cpu` or `split`, and the token embedding already lives in system RAM);
- malformed or backwards layer ranges (digits other than ASCII included), a layer listed twice in
  one rule or placed by two rules, `*`, `embed` or `head` placed twice;
- an unknown attribute or `experts=` value, an attribute given twice, `experts=` on `embed` /
  `head`, `split` without `cpu=`, `cpu=` without `split`, `cpu=` not a positive integer of ASCII
  digits;
- `stream` and `cpu` / `split` layers on the same GPU;
- a whole value of `none`, `off`, `auto`, `default`, `0` or `false` (see
  [No placement](#no-placement)).

At `model.load()`, before anything loads, a `ValueError` for:

- any of the settings a placement replaces: `-mcl` / `--moe_cpu_offload` /
  `EXL3_MOE_CPU_OFFLOAD`, `-mcs` / `--moe_cpu_split` / `EXL3_MOE_CPU_SPLIT`, `-mcm` /
  `--moe_cpu_mode` / `EXL3_MOE_CPU_MODE` other than `compute`, and `EXL3_MOE_CPU_SPLIT_LAYERS`.
  The error lists every one that is set; there is no precedence rule. (An invalid
  `moe_cpu_mode` value is reported as invalid first.);
- a single-device load (`model.load(device = ...)`) or tensor-parallel loading (`-tp`);
- a rule naming a layer the model does not have, or a layer no rule places;
- a device beyond the visible GPUs, or one the budgets exclude.

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
- DeepSeek-V4.1 checks the placement when the model object is built and at load, and restricts
  where it may change device; see [DeepSeek-V4.1](#deepseek-v41).

## Memory, performance and determinism

The placement adds no work of its own. Each module runs where it is placed, and every change of
device between consecutive modules copies the running state to the next device once per forward
pass (rows x hidden size x element size), as a split made by the autosplit does; a placement that
returns to an earlier device pays that copy at each change. GPU memory per device is what is
placed there plus the loader's transients. System RAM holds every expert of the `stream` and
`cpu` layers and the RAM share of the `split` layers, plus the worker's pinned staging ring, as
with `-mcl` / `-mcs`. The cost of the expert modes is theirs: see `EXL3_MOE_CPU_MODE` for
`stream` (the transfer per decoded token), and the CPU MoE offload section of
[env_vars.md](env_vars.md) for `cpu` and `split`.

The same placement puts the same modules on the same devices on every load, and a placement
computes nothing differently from the older settings that express the same layout (`-mcl`,
`-mcs`, `-mcm`, which it drives through the same machinery). On DeepSeek-V4.1-Flash, an earlier
revision of this code gave per-token log-probabilities of a 64K-token scoring run loaded through
`0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1` bitwise identical to the same layout set
up with model-specific expert settings that this code no longer has (its V4.1 expert claims now
take the generic path); that comparison has not been repeated on this code, which refuses that
layout for DeepSeek-V4.1 (its change of device at 12 is not a free cut; see
[DeepSeek-V4.1](#deepseek-v41)). Which GPU runs a layer can change that layer's results, as with
any split.

## When to use it

- The split must not depend on free memory, e.g. to compare runs.
- The layers whose experts go to system RAM should not be the first N: next to the GPU with the
  faster host link, away from a GPU that should stay resident, or wherever there is room.
- Different layers need different expert modes, e.g. GPU-streamed experts on one GPU and
  CPU-computed ones on another.
- The GPUs should be used in another order than their indices.

For a plain split that the budgets already produce, or for a single GPU without offloading, the
autosplit does the same with less to write.

## DeepSeek-V4.1

DeepSeek-V4.1 applies a placement like every other model (each layer on its device, the routed
experts of each MoE layer as its rule says), with two differences, because its compressed layers
share state across layers.

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

When the model object is built (`Model.from_config`), and again when `model.load()` starts: a
placement that breaks the rule below is refused before a Cache can be built for the model, and
the placement set at load, which may have changed since, is checked again. Set
`EXL3_PLACEMENT`, `--placement` or `config.infer_params.placement` before `Model.from_config`.

### Where it may change device

Every change of device between decoder layers must be a free cut. A change anywhere else would
separate layers from a pool, a top-k selection or candidate blocks that they read on their own
device, and is refused with a `ValueError`, e.g.

`DeepSeek-V4.1: the explicit placement '0-11=cuda:0; 12-39=cuda:1' changes device at layer 12, but layers 8-13 share the compressed-KV pool of layers.8. On this model a change of device can only fall where no pool, top-k selection or candidate list is shared across it: at layer 1, 2, 8, 14 or 20 (doc/placement.md, "DeepSeek-V4.1")`

Examples on DeepSeek-V4.1-Flash:

| Placement | Changes of device | Result |
|---|---|---|
| `0-13=cuda:0; 14-39=cuda:1` | 14 (free) | loads |
| `0-13=cuda:1; 14-39=cuda:0` | 14 (free), GPUs reversed | loads |
| `0-7=cuda:0; 8-19=cuda:1; 20-39=cuda:2` | 8 and 20 (free) | loads |
| `0-7=cuda:0; 8-13=cuda:1; 14-39=cuda:0` | 8 and 14 (free), back to the first GPU | loads |
| `0-11=cuda:0; 12-39=cuda:1` | 12, inside kv group 8-13 | refused |
| `*=cuda:0` | none | loads |

As at any change of device, the hidden state crosses into the first block past it: for V4.1 the
residual streams, 4 x 5120 FP32 values per token (80 KiB per token, 160 MiB for a 2048-token
chunk). Nothing else crosses a free cut.

### At load

The placement set at load is checked again with the same rule, before anything loads.
Tensor-parallel loading (`-tp`) is refused for V4.1 with or without a placement. The general
checks of this page (settings the placement replaces, layer coverage, visible and budgeted GPUs)
apply as for every model, and after the load every compressed layer is checked against the
device of the pool it reads, as for a load without a placement.

### Examples

Two cards, the tail 192 of each layer's 384 routed experts in system RAM on the CPU worker (as
`-mcs 192`, about 95 GiB of RAM), the change of device at the free cut 14:

```sh
python examples/chat.py -m /path/to/DeepSeek-V4.1-Flash-exl3 -gs 40,90 \
    --placement "0-13=cuda:0 experts=split cpu=192; 14-39=cuda:1 experts=split cpu=192" ...
```

```python
config = Config.from_directory(model_dir)
config.infer_params.placement = "0-13=cuda:0 experts=split cpu=192; 14-39=cuda:1 experts=split cpu=192"
model = Model.from_config(config)                   # the placement is checked here
cache = Cache(model, max_num_tokens = 262144)
model.load(use_per_device = [40, 90])
```

Determinism: the placement changes no arithmetic of V4.1's own code. As with any split, which GPU
runs a layer can change the results of the kernels on that layer.

See also `EXL3_DSV41_PIPELINE` (pipelined prefill across a change of device) in
[env_vars.md](env_vars.md).
