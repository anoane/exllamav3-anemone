"""
Optional FP32 router selection bias for DeepSeek-V4.1 (EXL3_DSV41_ROUTER_BIAS).

Every routed MoE layer adds a per-expert selection bias (layers.N.ffn.gate.bias) to its router
scores before the top-k choice; the bias steers which experts are chosen, not their weights.
DeepSeek keeps it in FP32. An EXL3 checkpoint may store it in FP16, and on DeepSeek-V4.1-Flash
its values are large next to their spread (about 9 to 16 on layers 0-35 and 31 to 58 on layers
36-39, where FP16 steps are 0.008 to 0.031, against a spread of 0.16 to 2.2 within a layer), so
the FP16 copy merges nearby biases (layer 39's 384 fall on 11 distinct values, 10 to 11 on each
of layers 36-39).

The loader already keeps a selection bias wider than FP16 at its stored precision (the GPU and
CPU-offload loads of BlockSparseMLP) and gives the FP16 top-k kernels a mean-centered FP16 copy
(block_sparse_mlp_routing._esb_h): selection does not change when every bias moves by the same
amount, and centering leaves steps of about 1e-3 at these magnitudes. attach() provides FP32
values for it: it checks a safetensors file holding the original biases against the checkpoint
and adds the file to the config's tensor collection, where it takes precedence over the
checkpoint's tensors of the same names (as conversion/ngram.py adds a quantized n-gram table).
Every later load of those layers, GPU-resident, CPU-offloaded or split, then reads the FP32
values through the unchanged loader, which also applies the expert order of a split load.

This cannot recover precision from the FP16 checkpoint: the file must hold the original values,
and each must round to the checkpoint's copy, which ties the file to the checkpoint it was
taken for.
"""

from __future__ import annotations

import copy
import os

import torch
from safetensors.torch import load_file

ENV_ROUTER_BIAS = "EXL3_DSV41_ROUTER_BIAS"


def bias_key(layer_idx: int) -> str:
    return f"layers.{layer_idx}.ffn.gate.bias"


def attach(config, path: str) -> str:
    """
    Validate the safetensors file at `path` against the checkpoint of `config` (a
    DeepseekV41Config, before Model.from_config) and add it to config.stc. Returns the resolved
    path. Raises ValueError, leaving config.stc unchanged, unless all of these hold:

      - the file holds exactly the checkpoint's router biases, one per routed layer
        (layers.N.ffn.gate.bias for every N the checkpoint has one for), and nothing else
      - each is FP32 with shape (n_routed_experts,) and finite
      - each rounds to the checkpoint's copy exactly (FP32 -> the stored dtype)
    """
    name = ENV_ROUTER_BIAS
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        raise ValueError(f"{name}: {path} is not a file")
    try:
        data = load_file(path, device = "cpu")
    except Exception as e:
        raise ValueError(f"{name}: cannot read {path} as a safetensors file ({e})") from None

    expected = [bias_key(i) for i in range(config.num_hidden_layers) if config.stc.has_tensor(bias_key(i))]
    if not expected:
        raise ValueError(f"{name}: the checkpoint has no router selection bias to replace")
    missing = sorted(set(expected) - set(data))
    extra = sorted(set(data) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{name}: {path} must hold exactly the checkpoint's {len(expected)} router biases "
            f"(layers.N.ffn.gate.bias); missing {missing}, unexpected {extra}")

    shape = (config.n_routed_experts,)
    stc = config.stc
    # The stored copies are read through the collection (its key mapping and name fixes), but the
    # check is not part of a model load: the load timer and byte count stay as they were, and the
    # files it opened are released again (the load reopens them)
    first_open, metrics = stc.first_open_time, copy.copy(stc.metrics)
    was_open = {f for f, h in stc.handles.items() if h}
    try:
        for key in expected:
            value = data[key]
            if value.dtype != torch.float32 or tuple(value.shape) != shape:
                raise ValueError(
                    f"{name}: {key} must be FP32 of shape {shape}, got {value.dtype} "
                    f"{tuple(value.shape)}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name}: {key} contains nonfinite values")
            stored = stc.get_tensor(key, "cpu", allow_bf16 = True, no_defer = True)
            if tuple(stored.shape) != shape or not torch.equal(value.to(stored.dtype), stored):
                raise ValueError(
                    f"{name}: {key} does not round to the checkpoint's {stored.dtype} bias; the file "
                    f"was taken from another model or revision")
    finally:
        stc.first_open_time, stc.metrics = first_open, metrics
        for f in [f for f, h in stc.handles.items() if h and f not in was_open]:
            stc.release_file(f)

    config.stc.add_tensor_files(path, warn_if_override = False)
    print(f" !! {name}: FP32 router selection bias of {len(expected)} layers from {path}")
    return path
