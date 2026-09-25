"""
FP32 logits (EXL3_FP32_LOGITS, -fp32_logits / --fp32_logits, config.infer_params.fp32_logits).
CPU only.

  * Model._apply_logits_dtype, compiled from the source and run against stand-in modules: every
    logits-output module with an output dtype gets FP32 when the flag is set, every other module
    keeps its own, and clearing the flag restores each module's own dtype;
  * Model.load_gen applies it before anything loads;
  * the environment default and the command-line flag, which import the exllamav3 package: where
    it does not import, each of those tests fails, naming the test and the cause. Nothing in this
    file is skipped.

    python -m pytest tests/test_fp32_logits_.py
"""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch

spec = importlib.util.spec_from_file_location("stable_helpers", Path(__file__).with_name("test_stable_arithmetic_.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

apply_logits_dtype = helpers.method("exllamav3/model/model.py", "Model", "_apply_logits_dtype", dict(torch = torch))


def fake_model(flag):
    head = types.SimpleNamespace(caps = {"logits_output": True}, out_dtype = None)
    head_half = types.SimpleNamespace(caps = {"logits_output": True}, out_dtype = torch.half)
    norm = types.SimpleNamespace(caps = {}, out_dtype = torch.half)
    block = types.SimpleNamespace(caps = {"kv_cache": True})
    params = types.SimpleNamespace(fp32_logits = flag)
    return types.SimpleNamespace(modules = [norm, block, head, head_half],
                                 config = types.SimpleNamespace(infer_params = params))


def test_logits_modules_take_fp32_and_get_their_own_dtype_back():
    model = fake_model(True)
    norm, block, head, head_half = model.modules
    apply_logits_dtype(model)
    assert head.out_dtype == torch.float and head_half.out_dtype == torch.float
    assert norm.out_dtype == torch.half and not hasattr(block, "out_dtype")
    apply_logits_dtype(model)                                 # a second load
    assert head.out_dtype == torch.float
    model.config.infer_params.fp32_logits = False
    apply_logits_dtype(model)
    assert head.out_dtype is None and head_half.out_dtype == torch.half and norm.out_dtype == torch.half


def test_flag_off_changes_nothing():
    model = fake_model(False)
    apply_logits_dtype(model)
    assert [getattr(m, "out_dtype", "none") for m in model.modules] == [torch.half, "none", None, torch.half]


def test_applied_before_anything_loads():
    fn = helpers.function("exllamav3/model/model.py", "Model", "load_gen")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    names = [ast.unparse(n.func) for n in calls]
    assert names.count("self._apply_logits_dtype") == 1
    applied = next(n for n in calls if ast.unparse(n.func) == "self._apply_logits_dtype")
    loads = [n for n in calls if ast.unparse(n.func) in ("self._load_single", "self._load_autosplit", "self._load_tp")]
    assert len(loads) == 3 and all(applied.lineno < n.lineno for n in loads)


@pytest.fixture
def package(request):
    try:
        import exllamav3.model_init as model_init
        from exllamav3.model.config import InferParams
    except Exception as e:
        # a failure, never a skip: these tests need the real package, and a broken import (the
        # compiled extension included) must not pass as "nothing to test"
        raise AssertionError(
            f"test_fp32_logits_.py::{request.node.name}: importing exllamav3 failed "
            f"({type(e).__name__}: {e}), so InferParams and model_init cannot be tested. This test "
            f"needs the package with its compiled extension: build it, or put a prebuilt "
            f"exllamav3_ext on PYTHONPATH") from e
    return model_init, InferParams


def test_environment_default(package):
    _, InferParams = package
    for value, expected in ((None, False), ("0", False), ("1", True), ("yes", True)):
        env = {k: v for k, v in os.environ.items() if k != "EXL3_FP32_LOGITS"}
        if value is not None:
            env["EXL3_FP32_LOGITS"] = value
        with patch.dict(os.environ, env, clear = True):
            assert InferParams().fp32_logits is expected, value


def test_command_line(package):
    import argparse
    model_init, _ = package
    parser = argparse.ArgumentParser()
    model_init.add_args(parser)
    for argv, expected in ((["-m", "x"], False), (["-m", "x", "--fp32_logits"], True), (["-m", "x", "-fp32_logits"], True)):
        assert parser.parse_args(argv).fp32_logits is expected
    # init applies the flag to the config it builds; without it the environment default stands
    source = ast.unparse(helpers.function("exllamav3/model_init.py", None, "init"))
    assert "config.infer_params.fp32_logits = True" in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
