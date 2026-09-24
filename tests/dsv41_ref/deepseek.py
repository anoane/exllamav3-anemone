"""
Loaders for DeepSeek's reference inference code for DeepSeek-V4.1-Flash, published with the
model (model.py, engram.py, kernel.py; MIT, see LICENSE-DEEPSEEK). The tests use its functions
as-is as oracles. Set DSV41_DEEPSEEK_REF to the directory that holds the files.
"""

import importlib.util
import os
import sys

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
