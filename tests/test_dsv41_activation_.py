"""
EXL3_DSV41_ACTIVATION (architecture/dsv41/activation.py): the two settings and their spelling,
refusals, the limit check of the reference form, and that DeepseekV41Model builds its routed and
shared experts with the one activation it resolves when the model object is built. CPU only; the
parser is loaded by path and the model constructor is read from the source.

    python -m pytest tests/test_dsv41_activation_.py
"""

import ast
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file

act = load_package_file("exllamav3/architecture/dsv41/activation.py", "_dsv41_activation")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value, expected", [
    (None, "silu"), ("", "silu"), ("  ", "silu"), ("silu", "silu"), ("SiLU", "silu"),
    ("reference", "silu_ref"), (" Reference\n", "silu_ref"),
])
def test_settings(value, expected):
    assert act.activation_fn(value, 10.0) == expected


@pytest.mark.parametrize("value", ["legacy", "silu_ref", "ref", "reference,silu", "1", "gelu"])
def test_unknown_settings_are_refused(value):
    with pytest.raises(ValueError, match = act.ENV_ACTIVATION):
        act.activation_fn(value, 10.0)


def test_reference_limit_must_be_usable():
    for limit in (0.0, 10.0, float.fromhex("0x1.fffffep+127")):
        assert act.activation_fn("reference", limit) == "silu_ref"
    for limit in (-1.0, float("nan"), float("inf"), 1e39):
        with pytest.raises(ValueError, match = "swiglu_limit"):
            act.activation_fn("reference", limit)
        # the default keeps the engine's own handling of the limit
        assert act.activation_fn("silu", limit) == "silu"


def test_one_activation_for_routed_and_shared_experts():
    tree = ast.parse((ROOT / "exllamav3/architecture/deepseek_v41.py").read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41Model")
    init = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("DSV41MoE", "GatedMLP")]
    assert sorted(n.func.id for n in calls) == ["DSV41MoE", "GatedMLP"]
    for call in calls:
        value = next(k.value for k in call.keywords if k.arg == "activation_fn")
        assert isinstance(value, ast.Name) and value.id == "activation_fn", ast.unparse(value)
    resolved = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "activation_fn" for t in n.targets)]
    assert len(resolved) == 1
    text = ast.unparse(resolved[0].value)
    assert text == "activation.activation_fn(os.environ.get(activation.ENV_ACTIVATION), c.swiglu_limit)", text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
