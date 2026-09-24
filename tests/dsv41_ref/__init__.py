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

import contextlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_package_file(relative: str, name: str, package: str | None = None):
    """
    Execute one exllamav3 source file by path, without importing the exllamav3 package, whose
    __init__ loads the compiled extension. The module is cached in sys.modules under `name`.

    A dotted `name` already makes the file's relative imports resolve inside its parent, a
    stand-in package the caller has put in sys.modules together with every module those
    imports name. `package` sets the file's package explicitly; only an undotted `name` needs
    it (a file loaded under an undotted name with no `package` can import nothing relative).
    """
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, relative))
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
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
    """
    Report a skipped check. Run as a script, prints one line and returns (the caller returns
    or exits 0); inside a pytest test function, raises pytest's skip so the report shows it.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(f"{test}: {reason}")
    print(f"  --  {test}: {reason}, skipped", flush = True)


@contextlib.contextmanager
def ablate(value: str, registry = None):
    """
    Run the block with the ablations `value` active (EXL3_DSV41_ABLATE's syntax, "" for none),
    then restore the previous setting. The engine reads the variable once, so the negative
    controls switch the parsed set through the registry's test-only set_ablations()
    (modules/dsv41_ablation.py), never the environment. `registry` is the dsv41_ablation module
    to switch: by default the package's own; a test that loads the V4.1 modules by path under a
    stand-in package passes the copy those modules import.
    """
    if registry is None:
        from exllamav3.modules import dsv41_ablation as registry
    previous = registry.set_ablations(value)
    try:
        yield
    finally:
        registry.set_ablations(previous)
