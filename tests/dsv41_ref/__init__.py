"""
Reference implementations and helpers shared by the DeepSeek-V4.1 tests.

The modules here are test oracles, never used at inference time:

    engram_hash      the engram n-gram hash, numpy
    engram_gate      the engram gate into the hyper-connection streams, numpy
    engram_forward   hash -> gather -> wkv -> gate, numpy
    engram_tables    a pread/mmap reader for the file-backed engram tables
    deepseek         loaders for DeepSeek's own reference inference code

Tests import them as dsv41_ref.<module> with the tests directory on sys.path (pytest puts it
there; a test run as a script inserts it itself).

Data the tests read, from the environment (a test that needs one and finds it unset says so
and skips):

    DSV41_MODEL_DIR      a DeepSeek-V4.1-Flash checkpoint directory (config.json,
                         tokenizer.json, safetensors shards including the engram tables)
    DSV41_DEEPSEEK_REF   DeepSeek's reference inference code for V4.1-Flash (model.py,
                         engram.py, kernel.py)
"""

import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_package_file(relative: str, name: str):
    """
    Execute one exllamav3 source file by path, without importing the exllamav3 package, whose
    __init__ loads the compiled extension. Only for files that import nothing from the package
    at module scope. The module is cached in sys.modules under `name`.
    """
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, relative))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def model_dir() -> str | None:
    """DSV41_MODEL_DIR, or None when it is not set."""
    return os.environ.get("DSV41_MODEL_DIR") or None


def skip(test: str, reason: str):
    """Report a skipped script-style test."""
    print(f"  --  {test}: {reason}, skipped", flush = True)
