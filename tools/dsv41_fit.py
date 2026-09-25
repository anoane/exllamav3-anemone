#!/usr/bin/env python3
"""
Header-only memory estimate for a DeepSeek-V4.1 layout.

Reads nothing but the safetensors headers (stdlib json) and config.json, and
prints, per CUDA device, the weights, the paged pools (sources and replicas),
the SWA rings and an ESTIMATED capacity check, plus the CPU-MoE arena in host
RAM. Exits 1 when the estimate exceeds a device/host budget. A successful
estimate is not a conservative bound or a guarantee that a workload will fit.
Full description: doc/dsv41_tools.md.

    python tools/dsv41_fit.py --model DIR --devices 64,96 \\
        --placement "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1"
    python tools/dsv41_fit.py --model DIR --devices 64,96 \
        --placement "0-11=cuda:0 experts=split cpu=128; 12-39=cuda:1 experts=split cpu=128"
    python tools/dsv41_fit.py --model DIR --devices 64,96 --search     # the split/CPU frontier

The layout is described with the loader's own settings, and each defaults to
the environment variable the loader reads, so a shell that is about to load
the model estimates the same layout:

  --placement        the explicit placement (EXL3_PLACEMENT, model/placement.py):
                     a device per layer and experts=vram|stream|cpu|split cpu=K
  --mcl N            -mcl: the routed experts of the first N MoE layers in RAM
                     (EXL3_MOE_CPU_OFFLOAD)
  --mcs K            -mcs: the last K routed experts of every MoE layer in RAM
                     (EXL3_MOE_CPU_SPLIT; EXL3_MOE_CPU_SPLIT_LAYERS limits it to
                     the first layers), capped at E - 1 as the loader caps it

The explicit placement replaces the other two and is refused with them, with
EXL3_MOE_CPU_SPLIT_LAYERS and with an EXL3_MOE_CPU_MODE other than compute, as
the loader refuses it; --mcl and --mcs are mutually exclusive. Without a
placement the model is estimated on cuda:0 alone. A placement's changes of
device follow the engine's DeepSeek-V4.1 rule (architecture/dsv41/placement.py,
DSV41Placement): each at a free cut except at most one, the split, whose pool
replica and crossings are counted; any other placement is refused with the
engine's message. --devices gives the capacity of every CUDA device the layout
uses, by CUDA index (cuda:0 first).

Output geometry defaults to Model.load's max_output_size=32, max_output_factor=1.
Use --max-output-size 2048 for a 2048-row scoring forward. As in Model.load,
the effective output budget is min(max_output_size, chunk). Output dtype is a
memory-accounting assumption, not a switch that changes the model's dtype.
The head's device takes the MAXIMUM of the historical block estimate, the head
phase and the output-consumer phase; sequential phases are not added together.
The head phase includes one output, its input (and a separate padded copy when
needed) and the packed head's reconstruct slice for long rows, plus one
transformed-input allowance (the default folded path can avoid that copy at
>=1024 rows; disabling fused reconstruction cannot). The consumer phase
includes max_output_factor output-sized copies, but no reconstruction slice,
which is already freed. Optional --validation-logit-rows models three FP32
matrices retained by the default validation collector's loop
(tools/dsv41_validate.py); it is NOT ordinary-serving memory or a complete
bound on top-k/library workspaces (nor the --bf16-logits collector path).

--measured-resident-bytes accepts one positive memory_allocated snapshot per
device the layout uses, AFTER load and transient cleanup, from the same
model/source/cache/layout/options. It REPLACES header total + historical load
overhead, rather than adding to them. The caller must establish that
compatibility; the tool does not import or authenticate measurement reports.
Do not pass reserved or driver-used memory: those include different terms.
JSON records the supplied values as unverified caller measurements alongside
the raw header subtotal.

What is counted where (the same rules the loader follows):
  resident layer     every layers.{i}.* tensor except the engram table and
                     the stale fp8 engram.wkv copy (Linear.load takes exl3)
  RAM layer          the GPU keeps everything but the routed experts, plus
  (-mcl, experts=    the routed experts' suh/svh (the streamed-prefill aux
   cpu or stream)    tensors); the arena holds the routed experts
  split layer        the GPU keeps experts [0, E-k) plus the tail's suh/svh;
  (-mcs, experts=    the arena holds the tail k
   split cpu=k)
  engram             wkv/q/k on the layer's device; the table stays in the
                     checkpoint files, gathered by row
  head + norm        on the device of the last layer (or the placement's
                     head= device); embed in host RAM
  duplicated keys    (engram q/k ship twice, F32 and BF16) count the larger
  pools              per kv source on its device; one replica per cut group
                     on its owner; (D_c + D_r + D_i) fp16 per entry, D_i only
                     where index K is owned (or carried by the replica)
  rings              every layer: ceil((window + 2*256)/256)*256 rows x
                     head_dim fp16 per slot; rate-2 sources add a
                     (256+2) x 2*head_dim fp32 compressor carry per slot

The transient reserve is listed item by item. The figures below are historical
slice calibrations, not a guarantee for another revision, full-model output width,
allocator or option combination; compare them with current measured peaks:

  activations     chunk x 0.30 MiB/token. Measured: one V4.1 block's forward
                  peaks 0.212-0.216 MiB/token above its weights at chunk 2048
                  (435-442 MiB; layers 0/2/3/20/21/24, resident, stateless
                  path, one GPU), plus the live fp32 hc streams entering
                  the block, hc_mult x hidden x 4 B = 0.078 MiB/token
  engram wkv out  chunk x hidden x (hc_mult+1) x 4 B on devices with an engram
                  layer (200 MiB at 2048; measured: engram blocks 1/14 peak
                  595-600 MiB, i.e. +160 MiB over the others)
  indexer tiles   0.5 GiB on devices with index sources: the tiled selection's
                  major-tensor allowance at T = 1M (transient_bytes <= 512 MiB),
                  not an exhaustive bound on all context-dependent allocations
  load overhead   what a loaded device holds beyond the header bytes, measured
                  as torch.cuda.memory_allocated after a real Model.load of
                  block slices: 289-294 MiB with resident layers only, 410-415
                  MiB with whole-layer CPU offload (flat for 1, 3 and 5 layers:
                  the CPU-MoE streamed-prefill buffers are per device), 689 MiB
                  + 25 MiB per extra layer with -mcs split layers; the defaults
                  0.29 / 0.41 / 0.68 GiB round each measurement up
  margin          0.25 GiB, the autosplit's own EXL3_AUTOSPLIT_MARGIN_MB

The CUDA context (0.5 GiB per device by default) is an estimate, and the slices
do not explain everything: one two-GPU smoke load with the engram layers
skipped and -mcs 128 logged 84.5 GiB on its first device against 82.26 GiB of
header bytes, ~1 GiB more than the slice figures predict. Read memory_allocated
after a real load before trusting a headroom of about a GiB or less.
In particular, weight-slab tails, CPU-offload slot/swizzle/fused-tier buffers,
allocator history, optional pipeline in-flight chunks and library workspaces
are not separately predicted by the historical overhead. No per-device constant
is adjusted automatically to match a particular measurement.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys

GiB = float(1 << 30)
MiB = float(1 << 20)
PAGE_SIZE = 256

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, relative: str):
    """One torch-free exllamav3 source file by path (the package __init__ needs the extension)."""
    path = os.path.join(_HERE, relative)
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod         # dataclasses resolve their module's annotations through it
    spec.loader.exec_module(mod)
    return mod


# The V4.1 rule for a placement's changes of device and its pool routes, and the engine's explicit
# placement grammar (absent from a tree without it: then only -mcl and -mcs describe a layout)
P = _load("_dsv41_fit_placement", "exllamav3/architecture/dsv41/placement.py")
G = _load("_dsv41_fit_generic_placement", "exllamav3/model/placement.py")
read_header = _load("_dsv41_fit_checkpoint_header", "exllamav3/architecture/dsv41/checkpoint_header.py").read_header


# ---------------------------------------------------------------------------
# headers

_LAYER = re.compile(r"layers\.(\d+)\.(.+)")
_EXPERT = re.compile(r"ffn\.experts\.(\d+)\.")


class Sizes:
    """Byte counts from the safetensors headers, bucketed the way the loader places them."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        self.text = self.config.get("text_config", self.config)
        self.topo = P.Topology.from_config_dict(self.config)
        n = self.topo.num_hidden_layers

        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise SystemExit(f"no *.safetensors in {model_dir}")
        ts: dict[str, int] = {}
        for fn in files:
            hdr, _, _ = read_header(fn)
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                b = v["data_offsets"][1] - v["data_offsets"][0]
                # Separate shards intentionally duplicate some gate weights.
                # Taking the larger representation is a conservative byte
                # estimate, not a claim about loader override order or content.
                ts[k] = max(ts.get(k, 0), b)
        self.num_files = len(files)
        self.num_tensors = len(ts)
        self.total = sum(ts.values())
        self.head_format = "exl3" if "head.trellis" in ts else \
            "dense" if "head.weight" in ts else "unknown"

        self.rest = [0] * n              # non-routed device weights per layer
        self.routed = [dict() for _ in range(n)]   # expert -> bytes
        self.aux = [dict() for _ in range(n)]      # expert -> suh/svh bytes
        self.engram_dev = [0] * n
        self.engram_table = [0] * n
        self.stale = [0] * n
        self.embed = 0
        self.head = 0
        self.skipped: dict[str, int] = {}
        for k, b in ts.items():
            m = _LAYER.match(k)
            if not m:
                top = k.split(".")[0]
                if top == "embed":
                    self.embed += b
                elif top in ("head", "norm"):
                    self.head += b
                else:
                    self.skipped[top] = self.skipped.get(top, 0) + b
                continue
            i, rest = int(m.group(1)), m.group(2)
            if i >= n:
                self.skipped["layers>=n"] = self.skipped.get("layers>=n", 0) + b
                continue
            if rest.startswith("engram."):
                if rest.startswith("engram.embed."):
                    self.engram_table[i] += b
                elif rest in ("engram.wkv.weight", "engram.wkv.scale"):
                    self.stale[i] += b
                else:
                    self.engram_dev[i] += b
                continue
            e = _EXPERT.match(rest)
            if e:
                ex = int(e.group(1))
                self.routed[i][ex] = self.routed[i].get(ex, 0) + b
                if rest.endswith((".suh", ".svh")):
                    self.aux[i][ex] = self.aux[i].get(ex, 0) + b
                continue
            self.rest[i] += b

    def routed_total(self, i):
        return sum(self.routed[i].values())

    def aux_total(self, i):
        return sum(self.aux[i].values())

    def num_experts(self):
        return [len(r) for r in self.routed]


# ---------------------------------------------------------------------------
# layout

class Layout:
    """
    Where every decoder layer loads and where its routed experts are kept, as the loader
    places them: a CUDA index per layer, per layer one of ("vram", 0, None), ("ram", 0, mode)
    (every routed expert in system RAM, mode "cpu" or "stream") or ("split", k, "cpu") (the
    last k routed experts in system RAM), the head's CUDA index, and what the engine makes of
    its changes of device (P.DSV41Placement): the split, the one of them allowed inside a kv
    group, with the pool routes it implies.
    """

    def __init__(self, topo, devices, experts, head_device = None, spec = ""):
        n = topo.num_hidden_layers
        if len(devices) != n or len(experts) != n:
            raise ValueError(f"layout: {n} layers, {len(devices)} devices and {len(experts)} expert plans")
        self.device = [int(d) for d in devices]
        self.experts = [tuple(e) for e in experts]
        self.head_device = self.device[-1] if head_device is None else int(head_device)
        changes = [i for i in range(1, n) if self.device[i] != self.device[i - 1]]
        # the engine's rule: every change of device at a free cut, except at most one (the split)
        dsv = P.DSV41Placement(topo, changes, spec = spec or None)
        self.cuts, self.split_at = dsv.cuts, dsv.split_at
        self.routes, self.topk_cross, self.cand_cross = dsv.routes, dsv.topk_cross, dsv.cand_cross
        self.replicas = dsv.replicas
        self.spec = spec or "cuda:0 only"

    def devices(self) -> list[int]:
        """The CUDA indices the layout uses, ascending."""
        return sorted(set(self.device) | {self.head_device})

    def layers_on(self, index: int, storage = None) -> list[int]:
        return [i for i, d in enumerate(self.device)
                if d == index and (storage is None or self.experts[i][0] == storage)]

    def format(self) -> str:
        return self.spec


def _count(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def build_layout(topo, num_experts, *, placement = None, mcl = 0, mcs = 0,
                 mcs_layers = 0, moe_cpu_mode = "compute") -> Layout:
    """
    The layout the loader builds from these settings (see the module docstring).

    placement        explicit placement: a string in the EXL3_PLACEMENT grammar or a parsed one
    mcl              -mcl / EXL3_MOE_CPU_OFFLOAD: the first mcl MoE layers keep every routed
                     expert in system RAM
    mcs              -mcs / EXL3_MOE_CPU_SPLIT: every MoE layer keeps its last mcs routed experts
                     in system RAM (at most E - 1)
    mcs_layers       EXL3_MOE_CPU_SPLIT_LAYERS: when nonzero, only the first mcs_layers MoE
                     layers are split
    moe_cpu_mode     -mcm / EXL3_MOE_CPU_MODE: changes no memory, only where RAM-held experts
                     are computed, but the loader refuses any value other than "compute"
                     next to an explicit placement (model/placement.conflicts), so this does too
    num_experts      routed experts per layer (0 for a layer without them)
    """
    n = topo.num_hidden_layers
    mcl, mcs, mcs_layers = _count(mcl, "mcl"), _count(mcs, "mcs"), _count(mcs_layers, "mcs_layers")
    if len(num_experts) != n:
        raise ValueError(f"num_experts has {len(num_experts)} entries for {n} layers")
    if isinstance(placement, str) and not placement.strip():
        placement = None
    if placement is not None and G is None:
        raise ValueError("this tree has no explicit placement (exllamav3/model/placement.py); describe "
                         "the layout with --mcl or --mcs")
    # the engine's reading: None, an empty value or only separators and comments is no placement,
    # and the words it refuses are refused here too
    if G is not None:
        placement = G.parse(placement)

    if placement is not None:
        other = [name for name, value in (("--mcl / EXL3_MOE_CPU_OFFLOAD", mcl),
                                          ("--mcs / EXL3_MOE_CPU_SPLIT", mcs),
                                          ("EXL3_MOE_CPU_SPLIT_LAYERS", mcs_layers),
                                          ("-mcm / EXL3_MOE_CPU_MODE", moe_cpu_mode != "compute")) if value]
        if other:
            raise ValueError(f"the explicit placement replaces {', '.join(other)}; the loader refuses them "
                             f"together, so unset them")
        pl = G.parse(placement)
        rules = pl.layer_map(range(n))
        devices = [rules[i].cuda_index for i in range(n)]
        experts = []
        for i in range(n):
            if not num_experts[i]:
                experts.append(("vram", 0, None))
                continue
            storage, _, k = G.expert_plan(pl, i, num_experts[i])
            mode = pl.experts_for_layer(i).mode
            experts.append((storage, k, None if storage == "vram" else ("stream" if mode == "stream" else "cpu")))
        head = pl.head_device()
        return Layout(topo, devices, experts, None if head is None else int(head[5:]), str(pl))

    if mcl and mcs:
        raise ValueError("--mcl and --mcs are mutually exclusive, as -mcl and -mcs are")
    devices = [0] * n
    experts, offloaded, split_count = [], 0, 0
    for i in range(n):
        e = num_experts[i]
        if e and mcl and offloaded < mcl:
            experts.append(("ram", 0, "cpu"))
            offloaded += 1
        elif e > 1 and mcs and (not mcs_layers or split_count < mcs_layers):
            experts.append(("split", min(mcs, e - 1), "cpu"))
            split_count += 1
        else:
            experts.append(("vram", 0, None))
    parts = []
    if mcl:
        parts.append(f"-mcl {mcl}")
    if mcs:
        parts.append(f"-mcs {mcs}" + (f" (first {mcs_layers} layers)" if mcs_layers else ""))
    return Layout(topo, devices, experts, None, " ".join(parts) or "cuda:0 only")


def split_layout(topo, num_experts, split: int, cpu_layers: int, mode: str = "cpu") -> Layout:
    """
    Layers [0, split) on cuda:0, [split, n) on cuda:1, the routed experts of the first
    cpu_layers layers past the split in system RAM: the layouts search() walks, named by the
    explicit placement that loads them.
    """
    n = topo.num_hidden_layers
    hi = split + cpu_layers
    devices = [0 if i < split else 1 for i in range(n)]
    experts = [("ram", 0, mode) if split <= i < hi and num_experts[i] else ("vram", 0, None) for i in range(n)]
    fmt = P.format_layer_ranges
    rules = [f"{fmt(range(0, split))}=cuda:0"]
    if cpu_layers:
        rules.append(f"{fmt(range(split, hi))}=cuda:1 experts={mode}")
    if hi < n:
        rules.append(f"{fmt(range(hi, n))}=cuda:1")
    return Layout(topo, devices, experts, None, "; ".join(rules))


# ---------------------------------------------------------------------------
# the fit

def compute(sizes: Sizes, layout: Layout, *, max_seq_len: int, chunk: int, slots: int = 1,
            cache_bits: int = 0, devices = None, context_gib = None,
            max_output_size: int = 32, max_output_factor: int = 1,
            output_dtype: str = "float16", validation_logit_rows: int = 0,
            measured_resident_bytes = None,
            transient_mib_per_token: float = 0.30, index_tile_gib: float = 0.5,
            load_overhead_gib = (0.29, 0.41, 0.68), split_overhead_mib: float = 25.0,
            margin_gib: float = 0.25,
            host_ram_gib: float | None = None,
            host_overhead_gib: float = 10.0, host_reserve_gib: float = 8.0) -> dict:
    """All figures in bytes unless named *_gib. This is an estimate, not a peak measurement.

    devices is one (name, GiB capacity) per CUDA index, cuda:0 first, covering every device
    the layout uses; context_gib one CUDA context estimate per CUDA index (the last value
    repeats; default 0.5 GiB each); measured_resident_bytes one snapshot per device the layout
    uses, in CUDA index order.

    ``total`` remains the raw header/cache subtotal, not all resident allocation.
    Compare ``need_gib`` with a measured
    end-to-end peak plus the driver/context allocation. Reset peak statistics after
    loading and measure representative prefill, decode and concurrent-job shapes;
    allocator reservation, fragmentation and other processes are not header bytes.
    """
    for name, value in (("max_seq_len", max_seq_len), ("chunk", chunk), ("slots", slots),
                        ("max_output_size", max_output_size), ("max_output_factor", max_output_factor)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(validation_logit_rows, int) or isinstance(validation_logit_rows, bool) or validation_logit_rows < 0:
        raise ValueError("validation_logit_rows must be a nonnegative integer")
    if output_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("output_dtype must be float16, bfloat16 or float32")
    if measured_resident_bytes is not None:
        if not isinstance(measured_resident_bytes, (list, tuple)) or not measured_resident_bytes:
            raise ValueError("measured_resident_bytes must contain one positive integer byte count per device")
        for value in measured_resident_bytes:
            if not isinstance(value, int) or isinstance(value, bool) or not 0 < value < (1 << 63):
                raise ValueError("measured_resident_bytes values must be positive integer byte counts below 2^63")
    if not isinstance(cache_bits, int) or isinstance(cache_bits, bool) or cache_bits not in (0, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError("cache_bits must be 0 (fp16) or an integer from 2 to 8")
    if len(load_overhead_gib) != 3:
        raise ValueError("load_overhead_gib must have resident, CPU and split estimates")
    estimates = dict(transient_mib_per_token = transient_mib_per_token,
                     index_tile_gib = index_tile_gib, split_overhead_mib = split_overhead_mib,
                     margin_gib = margin_gib, host_overhead_gib = host_overhead_gib,
                     host_reserve_gib = host_reserve_gib)
    estimates.update({f"load_overhead_gib[{i}]": v for i, v in enumerate(load_overhead_gib)})
    estimates.update({f"context_gib[{i}]": v for i, v in enumerate(context_gib or ())})
    for name, value in estimates.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not devices:
        raise ValueError("devices must list a (name, GiB) capacity for every CUDA device, cuda:0 first")
    for name, value in [(name or f"cuda:{j}", value) for j, (name, value) in enumerate(devices)] + \
            ([] if host_ram_gib is None else [("host RAM", host_ram_gib)]):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} capacity must be finite and positive")
    t, topo = sizes.text, sizes.topo
    n = topo.num_hidden_layers
    if max_seq_len % PAGE_SIZE:
        raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of {PAGE_SIZE}")
    context_gib = list(context_gib or [0.5])
    used = layout.devices()
    if used[-1] >= len(devices):
        raise ValueError(f"the layout uses cuda:{used[-1]} but devices lists {len(devices)} "
                         f"device{'s' if len(devices) != 1 else ''}")
    if measured_resident_bytes is not None and len(measured_resident_bytes) != len(used):
        raise ValueError(f"the layout uses {len(used)} devices: exactly {len(used)} measured resident byte "
                         f"counts are needed")

    head_dim = int(t.get("head_dim", 512))
    d_r = int(t.get("qk_rope_head_dim", 64))
    d_c = head_dim - d_r
    d_i_full = int(t.get("index_head_dim", 0))
    hidden = t["hidden_size"]
    if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden <= 0:
        raise ValueError("hidden_size must be a positive integer for the output estimate")
    hc_mult = int(t.get("hc_mult", 4))          # DeepseekV41Config's default
    window = int(t.get("sliding_window", 128))
    engram_ids = set(t.get("engram_layer_ids", []) or [])
    vocab = t.get("vocab_size")
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 0:
        raise ValueError("vocab_size must be a positive integer for the output estimate")
    # Linear pads both dimensions to 128; the head does not trim padded columns.
    padded_hidden = -(-hidden // 128) * 128
    padded_vocab = -(-vocab // 128) * 128
    output_rows = min(max_output_size, chunk)
    output_bytes = output_rows * padded_vocab * (4 if output_dtype == "float32" else 2)
    retained_output_bytes = output_bytes * max_output_factor
    if retained_output_bytes >= (1 << 63):
        raise ValueError("modeled retained output exceeds the supported 63-bit byte budget")
    head_format = getattr(sizes, "head_format", "unknown")
    # Match the default packed LinearEXL3 route (AUTO_RECONSTRUCT_THRESHOLD=144,
    # MAX_RECONSTRUCT_SLICE_N=32768). Other reconstruction overrides stay advisory.
    head_reconstruct = padded_hidden * min(padded_vocab, 32768) * 2 \
        if head_format == "exl3" and output_rows > 144 else 0
    head_input = chunk * hidden * 2  # final norm's unpadded backing allocation
    # Linear's zero-padding copy coexists with forward_ls's original input.
    head_input_padding = output_rows * padded_hidden * 2 if hidden != padded_hidden else 0
    # Non-folded reconstruction keeps a separate Hadamard-transformed xh. The
    # default path needs it at 145..1023 rows; an override can need it above that
    # range too. Include one possible copy for all reconstructed head shapes,
    # rather than assuming a shape/environment-dependent dispatcher decision.
    head_transform_input = output_rows * padded_hidden * 2 if head_reconstruct else 0
    head_phase = head_input + head_input_padding + output_bytes + head_reconstruct + head_transform_input
    collector_rows = min(validation_logit_rows, output_rows)
    collector_matrices = 3 * collector_rows * padded_vocab * 4
    consumer_phase = retained_output_bytes + collector_matrices
    warnings = [
        "Estimate only, not a conservative bound: validate the exact loaded workload and concurrent shapes.",
        "Persistent slab tails, CPU-offload slot/swizzle/fused-tier buffers, pipeline overlap and library workspaces are not individually modeled.",
        "Historical block/load allowances are slice calibrations, not current full-model measurements.",
    ]
    if max_output_size > chunk:
        warnings.append("max_output_size is capped to chunk, matching Model.load's effective output budget.")
    if head_format == "unknown":
        warnings.append("Head storage format is unknown; no reconstruction scratch is included in the head phase.")
    if validation_logit_rows:
        warnings.append("Validation collector term covers three FP32 matrices only; excludes bf16-logits conversion and top-k/library workspace.")
    if measured_resident_bytes is not None:
        warnings.append("Caller-supplied allocated resident bytes replace header subtotal plus load overhead; provenance and workload compatibility are not verified.")

    def entry_bytes(d_i):
        if cache_bits:
            g = d_c // 32
            return g * cache_bits * 4 + (g + d_r + d_i) * 2
        return (d_c + d_r + d_i) * 2

    def entries(m):
        return (max_seq_len // PAGE_SIZE) * (PAGE_SIZE // m)

    ring_rows = -(-(window + 2 * PAGE_SIZE) // PAGE_SIZE) * PAGE_SIZE
    ring_b = ring_rows * head_dim * 2 * slots
    carry_b = (PAGE_SIZE + 2) * 2 * head_dim * 4 * slots

    dev = {j: dict(weights = 0, pools = 0, rings = 0, items = [], layers = [], resident = [],
                   cpu = [], split = [], pool_items = [], has_index = False, has_engram = False)
           for j in used}
    arena = 0
    arena_layers = []
    for i in range(n):
        d = dev[layout.device[i]]
        d["layers"].append(i)
        E = len(sizes.routed[i])
        storage, k, _ = layout.experts[i]
        if storage == "ram" and E:
            d["weights"] += sizes.rest[i] + sizes.aux_total(i)
            arena += sizes.routed_total(i)
            arena_layers.append(i)
            d["cpu"].append(i)
        elif storage == "split" and E > 1:
            k = min(k, E - 1)
            first = E - k
            gpu_part = sum(b for e, b in sizes.routed[i].items() if e < first)
            tail_aux = sum(b for e, b in sizes.aux[i].items() if e >= first)
            d["weights"] += sizes.rest[i] + gpu_part + tail_aux
            arena += sizes.routed_total(i) - gpu_part
            arena_layers.append(i)
            d["split"].append(i)
        else:
            d["weights"] += sizes.rest[i] + sizes.routed_total(i)
            d["resident"].append(i)
        d["weights"] += sizes.engram_dev[i]
        d["rings"] += ring_b
        if topo.compress_ratios[i] == 2 and topo.is_kv_source(i):
            d["rings"] += carry_b
        if topo.is_index_source(i):
            d["has_index"] = True
        if i in engram_ids:
            d["has_engram"] = True
    dev[layout.head_device]["weights"] += sizes.head

    for s in topo.kv_source_layer_ids:
        m = topo.compress_ratios[s]
        b = entries(m) * entry_bytes(d_i_full if topo.index_owns_k(s) else 0)
        dev[layout.device[s]]["pools"] += b
        dev[layout.device[s]]["pool_items"].append((f"src {s}", b))
    for o, (owner, rep, wik) in sorted(layout.routes.items()):
        if rep is None:
            continue
        m = topo.compress_ratios[rep]
        b = entries(m) * entry_bytes(d_i_full if wik else 0)
        dev[layout.device[o]]["pools"] += b
        dev[layout.device[o]]["pool_items"].append((f"replica {o}<-{rep}{'+idxK' if wik else ''}", b))

    ok = True
    out_dev = []
    for pos, j in enumerate(used):
        d = dev[j]
        name, phys = devices[j]
        ctx = context_gib[j] if j < len(context_gib) else context_gib[-1]
        total = d["weights"] + d["pools"] + d["rings"]
        res = [("activations", chunk * transient_mib_per_token * MiB)]
        if d["has_engram"]:
            res.append(("engram wkv out", chunk * hidden * (hc_mult + 1) * 4))
        if d["has_index"]:
            res.append(("indexer tiles", index_tile_gib * GiB))
        lo_res, lo_cpu, lo_split = load_overhead_gib
        if d["split"]:
            lo = lo_split * GiB + (len(d["split"]) - 1) * split_overhead_mib * MiB
        elif d["cpu"]:
            lo = lo_cpu * GiB
        else:
            lo = lo_res * GiB
        body_phase = sum(b for _, b in res)
        phases = {"body_estimate": body_phase}
        if j == layout.head_device:
            phases.update(head = head_phase, output_consumer = consumer_phase)
            res.append(("output/consumer phase uplift", max(phases.values()) - body_phase))
        resident = total if measured_resident_bytes is None else measured_resident_bytes[pos]
        applied_lo = lo if measured_resident_bytes is None else 0
        res.append(("load overhead" if measured_resident_bytes is None else
                    "load overhead (included in measured resident)", applied_lo))
        res.append(("margin", margin_gib * GiB))
        reserve = sum(b for _, b in res)
        need = resident / GiB + ctx + reserve / GiB
        fits = need <= phys
        ok &= fits
        out_dev.append(dict(
            index = j, name = name, physical_gib = phys, context_gib = ctx,
            layers = d["layers"], resident = d["resident"], cpu = d["cpu"], split = d["split"],
            weights = d["weights"], pools = d["pools"], rings = d["rings"],
            pool_items = d["pool_items"], total = total, total_gib = total / GiB,
            resident_baseline = resident, resident_estimate = resident + applied_lo,
            resident_baseline_kind = "header subtotal" if measured_resident_bytes is None else
                "caller-supplied allocated bytes (unverified)",
            historical_load_overhead = lo, phases = phases,
            reserve_items = res, reserve = reserve, need_gib = need,
            headroom_gib = phys - need, fits = fits,
        ))

    host_total = arena + sizes.embed
    host_need = host_total / GiB + host_overhead_gib + host_reserve_gib
    host_ok = host_ram_gib is None or host_need <= host_ram_gib
    ok &= host_ok
    return dict(
        placement = layout.format(), cuts = list(layout.cuts), split = layout.split_at,
        max_seq_len = max_seq_len, chunk = chunk,
        slots = slots, cache_bits = cache_bits,
        estimate_only = True, conservative_bound = False, warnings = warnings,
        max_output_size = max_output_size, effective_output_rows = output_rows,
        max_output_factor = max_output_factor, output_dtype = output_dtype,
        output = dict(device = layout.head_device, padded_vocab = padded_vocab,
                      bytes = output_bytes, retained_bytes = retained_output_bytes,
                      head_format = head_format, head_input_bytes = head_input,
                      head_input_padding_bytes = head_input_padding,
                      head_reconstruct_bytes = head_reconstruct,
                      head_transform_input_allowance = head_transform_input,
                      validation_logit_rows = validation_logit_rows,
                      effective_validation_rows = collector_rows,
                      validation_fp32_matrix_allowance = collector_matrices),
        measured_resident_bytes = measured_resident_bytes,
        devices = out_dev, arena = arena, arena_gib = arena / GiB, arena_layers = arena_layers,
        embed = sizes.embed, host_total_gib = host_total / GiB, host_need_gib = host_need,
        host_ram_gib = host_ram_gib, host_ok = host_ok, ok = ok,
        routes = layout.routes, replicas = layout.replicas,
        topk_cross = sorted(layout.topk_cross), cand_cross = sorted(layout.cand_cross),
    )


# ---------------------------------------------------------------------------
# report

def _g(b):
    return f"{b / GiB:.4f}"


def _host_ram_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / GiB
    except OSError:
        pass
    return None


def report(sizes: Sizes, r: dict) -> str:
    fmt = P.format_layer_ranges
    topo = sizes.topo
    L = []
    L.append(f"  {os.path.basename(os.path.normpath(sizes.model_dir))}: {sizes.num_files} shards, "
             f"{sizes.num_tensors:,} tensors ({sizes.total / GiB:.2f} GiB), headers only")
    pool_kind = "fp16" if not r["cache_bits"] else "%d-bit nope" % r["cache_bits"]
    L.append(f"  layout {r['placement']} | max_seq_len {r['max_seq_len']:,} | chunk {r['chunk']} "
             f"| slots {r['slots']} | pools {pool_kind}")
    L.append(f"  output: {r['effective_output_rows']} rows (requested {r['max_output_size']}), "
             f"{r['output_dtype']}, retention factor {r['max_output_factor']}; "
             f"validation collector rows {r['output']['effective_validation_rows']}")
    L.append("  ESTIMATE ONLY -- not a conservative bound or a guarantee of fit")
    res_layer = [sizes.rest[i] + sizes.routed_total(i) for i in range(topo.num_hidden_layers)]
    plain = [i for i in range(topo.num_hidden_layers)
             if not topo.is_kv_source(i) and not topo.is_index_source(i)
             and sizes.engram_dev[i] == 0]
    if plain:
        i = plain[0]
        L.append(f"  per layer (layer {i}): resident {res_layer[i]:,} B ({_g(res_layer[i])} GiB) = "
                 f"routed {sizes.routed_total(i):,} + rest {sizes.rest[i]:,}; with its experts in RAM it keeps "
                 f"{sizes.rest[i] + sizes.aux_total(i):,} B ({_g(sizes.rest[i] + sizes.aux_total(i))} GiB, "
                 f"suh/svh {sizes.aux_total(i):,} included)")
    for d in r["devices"]:
        L.append("")
        L.append(f"  cuda:{d['index']}{': ' + d['name'] if d['name'] else ''} ({d['physical_gib']:.2f} GiB) -- "
                 f"layers {fmt(d['layers'])}")
        parts = []
        if d["resident"]:
            parts.append(f"resident {fmt(d['resident'])} ({len(d['resident'])})")
        if d["cpu"]:
            parts.append(f"experts in RAM {fmt(d['cpu'])} ({len(d['cpu'])})")
        if d["split"]:
            parts.append(f"experts split {fmt(d['split'])} ({len(d['split'])})")
        L.append(f"    {' | '.join(parts) or 'no decoder layers'}")
        extras = []
        for i in d["layers"]:
            if sizes.engram_dev[i]:
                extras.append(f"engram-{i} {sizes.engram_dev[i]:,}")
        if d["index"] == r["output"]["device"]:
            extras.append(f"head+norm {sizes.head:,}")
        L.append(f"    weights {_g(d['weights'])} GiB" + (f"  (incl. {', '.join(extras)} B)" if extras else ""))
        pi = ", ".join(f"{k} {_g(b)}" for k, b in d["pool_items"]) or "none"
        L.append(f"    pools   {_g(d['pools'])} GiB  ({pi})")
        L.append(f"    rings   {_g(d['rings'])} GiB  ({len(d['layers'])} layers x SWA ring"
                 f"{' + rate-2 compressor carry' if any(topo.compress_ratios[i] == 2 and topo.is_kv_source(i) for i in d['layers']) else ''})")
        L.append(f"    HEADER/CACHE SUBTOTAL {d['total_gib']:.2f} GiB")
        if r["measured_resident_bytes"] is not None:
            L.append(f"    resident baseline {_g(d['resident_baseline'])} GiB "
                     "(caller allocated snapshot; replaces subtotal + load overhead)")
        L.append("    sequential phases " + ", ".join(f"{k} {_g(b)} GiB" for k, b in d["phases"].items())
                 + " (take maximum, not sum)")
        ri = " + ".join(f"{k} {b / GiB:.2f}" for k, b in d["reserve_items"])
        L.append(f"    check   {d['resident_baseline'] / GiB:.2f} + context {d['context_gib']:.2f} + reserve "
                 f"{d['reserve'] / GiB:.2f} ({ri}) = {d['need_gib']:.2f} vs {d['physical_gib']:.2f} "
                 f"-> {'ESTIMATED FIT' if d['fits'] else 'ESTIMATE EXCEEDS BUDGET'} "
                 f"(estimated headroom {d['headroom_gib']:+.2f} GiB)")
    L.append("")
    L.append(f"  host RAM: CPU-MoE arena {r['arena_gib']:.2f} GiB ({len(r['arena_layers'])} layers "
             f"{fmt(r['arena_layers'])}) + embed {r['embed'] / GiB:.2f} = {r['host_total_gib']:.2f} GiB")
    if r["host_ram_gib"] is not None:
        L.append(f"    check   {r['host_total_gib']:.2f} + parent/worker overhead + reserve = "
                 f"{r['host_need_gib']:.2f} vs {r['host_ram_gib']:.2f} -> "
                 f"{'OK' if r['host_ok'] else 'DOES NOT FIT'}")
    rep = " ".join(f"{o}<-{s}" for o, s in sorted(r["replicas"].items())) or "none"
    tk = " ".join(f"{a}->{b}" for a, b in r["topk_cross"]) or "none"
    cc = " ".join(f"{a}->{b}" for a, b in r["cand_cross"]) or "none"
    cuts = ",".join(str(c) for c in r["cuts"]) or "none"
    L.append(f"  changes of device {cuts} | split {r['split'] if r['split'] is not None else 'none'} | "
             f"replicas {rep} | top-k cross {tk} | candidates cross {cc}")
    L.append(f"  RESULT: {'estimated fit' if r['ok'] else 'estimate exceeds budget'}")
    L += [f"  WARNING: {w}" for w in r["warnings"]]
    return "\n".join(L)


def search(sizes: Sizes, **kw) -> list[dict]:
    """
    For every change of device s, the fewest layers past it (taken from s upwards) whose routed
    experts must stay in system RAM for the layout to fit: layers [0, s) on cuda:0, the rest on
    cuda:1. Each row is named by the explicit placement (EXL3_PLACEMENT) that loads it; one change
    of device is always within the engine's rule (a free cut, or the one split its pool replica
    allows).
    """
    if kw.get("measured_resident_bytes") is not None:
        raise ValueError("measured resident bytes apply only to their measured placement, not search")
    n = sizes.topo.num_hidden_layers
    num_experts = sizes.num_experts()
    rows = []
    for s in range(1, n):
        for k in range(0, n - s + 1):
            r = compute(sizes, split_layout(sizes.topo, num_experts, s, k), **kw)
            if not r["devices"][0]["fits"]:
                break          # cuda:0 only grows with s; more layers in RAM past the split do not help it
            if r["ok"]:
                rows.append(r)
                break
    return rows


def _int_env(name: str) -> int:
    value = os.environ.get(name, "").strip()
    try:
        return int(value) if value else 0
    except ValueError:
        raise SystemExit(f"{name}={value!r} is not an integer") from None


def _devices(spec: str) -> list[tuple[str, float]]:
    out = []
    for j, part in enumerate(p for p in spec.split(",") if p.strip()):
        name, sep, gib = part.rpartition(":")
        try:
            out.append((name.strip() if sep else "", float(gib)))
        except ValueError:
            raise SystemExit(f"--devices: {part.strip()!r} is not <GiB> or <name>:<GiB>") from None
    return out


def main(argv = None):
    ap = argparse.ArgumentParser(description = __doc__.split("\n\n")[0].strip(),
                                 epilog = "See doc/dsv41_tools.md for every option and worked examples.")
    ap.add_argument("--model", "-m", default = os.environ.get("DSV41_MODEL_DIR"),
                    help = "checkpoint directory (config.json and *.safetensors; default: DSV41_MODEL_DIR)")
    ap.add_argument("--devices", "-d", default = None,
                    help = "capacity of every CUDA device in GiB, cuda:0 first, as <GiB> or <name>:<GiB>, "
                           "comma-separated, e.g. 64,96 (required)")
    ap.add_argument("--placement", default = os.environ.get("EXL3_PLACEMENT"),
                    help = "explicit placement, EXL3_PLACEMENT grammar (default: EXL3_PLACEMENT)")
    ap.add_argument("--mcl", type = int, default = _int_env("EXL3_MOE_CPU_OFFLOAD"),
                    help = "-mcl: routed experts of the first N MoE layers in RAM (default: EXL3_MOE_CPU_OFFLOAD)")
    ap.add_argument("--mcs", type = int, default = _int_env("EXL3_MOE_CPU_SPLIT"),
                    help = "-mcs: last K routed experts of every MoE layer in RAM (default: EXL3_MOE_CPU_SPLIT)")
    ap.add_argument("--mcs-layers", type = int, default = _int_env("EXL3_MOE_CPU_SPLIT_LAYERS"),
                    help = "split only the first N MoE layers (default: EXL3_MOE_CPU_SPLIT_LAYERS, 0 = all)")
    ap.add_argument("--max-seq-len", type = int, default = 1048576)
    ap.add_argument("--chunk", type = int, default = 2048)
    ap.add_argument("--max-output-size", type = int, default = 32,
                    help = "requested output rows; capped to chunk, as in Model.load (default 32)")
    ap.add_argument("--max-output-factor", type = int, default = 1,
                    help = "output-consumer retention in output-sized copies, after head scratch is freed")
    ap.add_argument("--output-dtype", choices = ("float16", "bfloat16", "float32"), default = "float16",
                    help = "accounted output storage dtype, not a model configuration switch")
    ap.add_argument("--validation-logit-rows", type = int, default = 0,
                    help = "opt-in default validation collector FP32 matrix allowance; 0 = ordinary engine")
    ap.add_argument("--measured-resident-bytes", default = None,
                    help = "comma-separated post-load allocated bytes per used device; replaces header + load "
                           "overhead; caller verifies source/model/options")
    ap.add_argument("--slots", type = int, default = 1, help = "recurrent state slots (Cache max_batch_size)")
    ap.add_argument("--cache-bits", type = int, default = 0, help = "packed nope pool bits (-cq); 0 = fp16")
    ap.add_argument("--context-gib", default = "0.5",
                    help = "CUDA context per device, cuda:0 first; the last value repeats (default 0.5)")
    ap.add_argument("--transient-mib-per-token", type = float, default = 0.30,
                    help = "activations per chunk token (measured 0.29, see the docstring)")
    ap.add_argument("--margin-gib", type = float, default = 0.25)
    ap.add_argument("--index-tile-gib", type = float, default = 0.5)
    ap.add_argument("--load-overhead-gib", default = "0.29,0.41,0.68",
                    help = "per device: resident only, with whole-layer RAM experts, with split layers")
    ap.add_argument("--split-overhead-mib", type = float, default = 25.0,
                    help = "per extra split layer on a device")
    ap.add_argument("--host-ram-gib", type = float, default = None, help = "default: /proc/meminfo")
    ap.add_argument("--host-overhead-gib", type = float, default = 10.0)
    ap.add_argument("--host-reserve-gib", type = float, default = 8.0)
    ap.add_argument("--search", action = "store_true",
                    help = "print the split / layers-in-RAM frontier over two devices")
    ap.add_argument("--json", action = "store_true")
    a = ap.parse_args(argv)
    if not a.model:
        ap.error("--model is required (or DSV41_MODEL_DIR)")
    if not a.devices:
        ap.error("--devices is required: the capacity in GiB of every CUDA device, cuda:0 first")

    devices = _devices(a.devices)
    kw = dict(
        max_seq_len = a.max_seq_len, chunk = a.chunk, slots = a.slots, cache_bits = a.cache_bits,
        max_output_size = a.max_output_size, max_output_factor = a.max_output_factor,
        output_dtype = a.output_dtype, validation_logit_rows = a.validation_logit_rows,
        measured_resident_bytes = None if a.measured_resident_bytes is None else
            [int(value) for value in a.measured_resident_bytes.split(",")],
        devices = devices,
        context_gib = [float(x) for x in a.context_gib.split(",")],
        transient_mib_per_token = a.transient_mib_per_token, index_tile_gib = a.index_tile_gib,
        load_overhead_gib = tuple(float(x) for x in a.load_overhead_gib.split(",")),
        split_overhead_mib = a.split_overhead_mib, margin_gib = a.margin_gib,
        host_ram_gib = a.host_ram_gib if a.host_ram_gib is not None else _host_ram_gib(),
        host_overhead_gib = a.host_overhead_gib, host_reserve_gib = a.host_reserve_gib,
    )
    sizes = Sizes(a.model)

    if a.search:
        if a.measured_resident_bytes is not None:
            ap.error("--measured-resident-bytes applies only to its measured layout, not --search")
        if len(devices) < 2:
            ap.error("--search splits the model over cuda:0 and cuda:1; --devices must list both")
        rows = search(sizes, **kw)
        names = ", ".join(f"{n or 'cuda:%d' % j}:{g}" for j, (n, g) in enumerate(devices))
        print(f"  frontier at max_seq_len {a.max_seq_len:,}, chunk {a.chunk} (fewest layers with their "
              f"experts in RAM per split; devices {names})")
        print(f"  {'placement':<58} {'cuda:0':>7} {'head0':>7} {'cuda:1':>7} {'head1':>7} {'arena':>7}")
        for r in rows:
            d0, d1 = r["devices"][0], r["devices"][1]
            print(f"  {r['placement']:<58} {d0['total_gib']:7.2f} {d0['headroom_gib']:+7.2f} "
                  f"{d1['total_gib']:7.2f} {d1['headroom_gib']:+7.2f} {r['arena_gib']:7.2f}")
        if rows:
            best = min(rows, key = lambda r: (len(r["arena_layers"]), -r["devices"][0]["total_gib"]))
            print(f"  fewest layers in RAM: {best['placement']} ({len(best['arena_layers'])} layers, "
                  f"arena {best['arena_gib']:.2f} GiB)")
        print("  ESTIMATE ONLY: not a conservative bound; inspect one placement's report for coverage warnings")
        return 0 if rows else 1

    try:
        layout = build_layout(sizes.topo, sizes.num_experts(), placement = a.placement,
                              mcl = a.mcl, mcs = a.mcs,
                              mcs_layers = a.mcs_layers,
                              moe_cpu_mode = os.environ.get("EXL3_MOE_CPU_MODE", "compute"))
        r = compute(sizes, layout, **kw)
    except ValueError as e:
        ap.error(str(e))
    if a.json:
        print(json.dumps({k: v for k, v in r.items() if k != "routes"}, default = str, indent = 1))
    else:
        print(report(sizes, r))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
