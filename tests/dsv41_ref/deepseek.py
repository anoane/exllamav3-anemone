"""
Loaders for DeepSeek's reference inference code for DeepSeek-V4.1-Flash, published with the
model (model.py, engram.py, kernel.py; MIT, see LICENSE-DEEPSEEK). The tests use its functions
as-is as oracles. Set DSV41_DEEPSEEK_REF to the directory that holds the files.
"""

import importlib.util
import os
import sys
import types

ENV = "DSV41_DEEPSEEK_REF"


def ref_dir() -> str | None:
    """DSV41_DEEPSEEK_REF, or None when it is not set."""
    return os.environ.get(ENV) or None


def _require(filename: str) -> str:
    d = ref_dir()
    if d is None:
        raise FileNotFoundError(f"{ENV} is not set")
    path = os.path.join(d, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{ENV}={d} has no {filename}")
    return path


def load_engram():
    """DeepSeek's engram.py (EngramLayout, NgramHashState), imported as-is. Needs sympy."""
    mod = sys.modules.get("ds_ref_engram")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("ds_ref_engram", _require("engram.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ds_ref_engram"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules["ds_ref_engram"]
        raise
    return mod


def load_model():
    """
    DeepSeek's model.py, imported as-is with its tilelang kernel module (used only for the FP8
    and FP4 GEMMs, activation quantization and sparse attention, none of which the CPU checks
    run) and its vision and image-processor modules stubbed. fp4_act_quant is a no-op: the
    port keeps fp16/fp32 where the reference quantizes, and the checks compare the math around
    it. default_dtype is float32, so every linear is a plain F.linear. Needs sympy: model.py
    imports the reference's engram.py, which is dropped from sys.modules again afterwards (the
    loaded module keeps the names it bound).
    """
    mod = sys.modules.get("ds_ref_model")
    if mod is not None:
        return mod
    import torch
    path = _require("model.py")
    ref = os.path.dirname(path)
    k = types.ModuleType("kernel")
    for n in ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn"):
        setattr(k, n, lambda *a, **kw: None)
    v = types.ModuleType("vision")
    v.Aligner = v.ViT = object
    ip = types.ModuleType("image_processor")
    ip.IMAGE = ip.IMAGE_END = ip.IMAGE_NEW_LINE = ip.IMAGE_START = None
    saved = {n: sys.modules.get(n) for n in ("kernel", "vision", "image_processor", "engram")}
    sys.modules.update({"kernel": k, "vision": v, "image_processor": ip})
    sys.path.insert(0, ref)
    spec = importlib.util.spec_from_file_location("ds_ref_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ds_ref_model"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules["ds_ref_model"]
        raise
    finally:
        sys.path.remove(ref)
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    mod.default_dtype = torch.float32
    return mod
