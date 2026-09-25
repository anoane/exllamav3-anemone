"""CPU-only checks of tools/dsv41_refcompare.py: the reference capture schema and the optional
text-decoding diagnostics. No capture, model or torch needed."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("schema_refcompare", ROOT / "tools/dsv41_refcompare.py")
rc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rc
spec.loader.exec_module(rc)


def valid_case():
    probabilities = {str(i): -float(i) for i in range(1, 6)}
    return {"n": 3, "rep": 0, "prompt_ids": [1, 2, 3],
            "prompt_logprobs": [None, probabilities, probabilities.copy()]}


@pytest.mark.parametrize("tokens", [
    [1.0, 2, 3], [1.9, 2, 3], [True, 2, 3], ["1", 2, 3], [-1, 2, 3],
    [2**63, 2, 3], [[1], [2], [3]], "123", None,
    np.array([1, 2, 3], dtype=np.float32), np.array([True, False, True]),
    np.array([2**63, 2, 3], dtype=np.uint64),
])
def test_prompt_ids_never_coerce_to_a_different_prompt(tokens):
    raw = valid_case()
    raw["prompt_ids"] = tokens
    with pytest.raises(ValueError, match="reference"):
        rc.parse_case(raw)


@pytest.mark.parametrize("field, values", [
    ("n", [3.0, 3.9, True, "3", None, -3]),
    ("rep", [0.0, 0.9, False, "0", None, -1]),
    ("probe_errors", [0.0, 0.9, False, "0", None, -1]),
])
def test_geometry_and_probe_counts_require_integers(field, values):
    for value in values:
        raw = valid_case()
        raw[field] = value
        with pytest.raises(ValueError, match=field):
            rc.parse_case(raw)


@pytest.mark.parametrize("top", [5.0, 5.9, True, "5", None, 4])
def test_top_requires_integer_without_truncation(tmp_path, top):
    with pytest.raises(ValueError, match="top"):
        rc.parse_case(valid_case(), top=top)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"meta": {"top": top}, "cases": [valid_case()]}))
    with pytest.raises(ValueError, match="top"):
        rc.load_ref(path)
    if top is not None:
        path.write_text(json.dumps({"cases": [valid_case()]}))
        with pytest.raises(ValueError, match="top"):
            rc.load_ref(path, top=top)


@pytest.mark.parametrize("contended", [0, 1, "false", "true", None, []])
def test_recorded_contention_requires_boolean(contended):
    raw = valid_case()
    raw["contended"] = contended
    with pytest.raises(ValueError, match="contended"):
        rc.parse_case(raw)


@pytest.mark.parametrize("logprobs", [{}, [None, [], {}], [None, {}, "value"]])
def test_logprob_container_geometry(logprobs):
    raw = valid_case()
    raw["prompt_logprobs"] = logprobs
    with pytest.raises(ValueError, match="prompt_logprobs"):
        rc.parse_case(raw)


def test_integer_numpy_ids_and_int64_boundary_are_preserved():
    raw = valid_case()
    raw["prompt_ids"] = np.array([np.iinfo(np.int64).max, 2, 3], dtype=np.int64)
    raw["n"], raw["rep"] = np.int64(3), np.int32(0)
    parsed = rc.parse_case(raw, top=np.int64(5))
    assert parsed.prompt_ids.tolist() == [np.iinfo(np.int64).max, 2, 3]


def test_legacy_missing_isolation_loads_warns_and_preserves_numeric_scores(tmp_path):
    raw = valid_case()
    del raw["n"], raw["rep"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"cases": [raw]}))
    legacy = rc.load_ref(path)[0]
    explicit = rc.parse_case({**raw, "contended": False, "probe_errors": 0})
    assert (legacy.n, legacy.rep, legacy.top) == (3, 0, 5)
    old_report = rc.compare(legacy, *rc.as_ours(legacy))
    new_report = rc.compare(explicit, *rc.as_ours(explicit))
    assert old_report["global"] == new_report["global"]
    assert old_report["regions"] == new_report["regions"]
    assert old_report["reference_isolation_metadata"] == {
        "complete": False, "missing": ["contended", "probe_errors"]}
    assert new_report["reference_isolation_metadata"] == {"complete": True, "missing": []}
    legacy_ok, legacy_messages = rc.gates(old_report)
    explicit_ok, explicit_messages = rc.gates(new_report)
    assert legacy_ok and explicit_ok
    assert any(message.startswith("WARNING") for message in legacy_messages)
    assert not any(message.startswith("WARNING") for message in explicit_messages)
    assert [message for message in legacy_messages if not message.startswith("WARNING")] == explicit_messages


def test_partial_probe_metadata_is_not_complete_and_contamination_still_fails():
    for flags, missing, ok in (({"contended": False}, ["probe_errors"], True),
                               ({"probe_errors": 0}, ["contended"], True),
                               ({"contended": True, "probe_errors": 0}, [], False),
                               ({"contended": False, "probe_errors": 1}, [], False)):
        c = rc.parse_case({**valid_case(), **flags})
        report = rc.compare(c, *rc.as_ours(c))
        assert report["reference_isolation_metadata"] == {"complete": not missing, "missing": missing}
        assert rc.gates(report)[0] == ok


@pytest.mark.parametrize("document", [[], {"meta": [], "cases": [valid_case()]},
    {"cases": {}}, {"cases": []}, {"cases": [None]}])
def test_reference_document_geometry_is_explicit(tmp_path, document):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="reference"):
        rc.load_ref(path)


@pytest.mark.parametrize("tokens, ranked", [
    (["x"], {"x": -0.1, "y": -2.0}),
    (["x"], {"token_id:1": -0.1, "y": -2.0}),
])
def test_missing_tokenizer_does_not_report_text_mismatches(tokens, ranked):
    case = rc.parse_case({**valid_case(), "gen_tokens": tokens,
                          "gen_top_logprobs": [ranked]})
    result = rc.next_token_check(case, [1, 2])
    assert result["our_id"] == 1
    assert result["our_text"] is None
    assert result["match"] is None
    assert result["top5_overlap"] is None
    assert not result["text_decoding_available"]


@pytest.mark.parametrize("first, expected", [(1, True), (3, False)])
def test_id_reference_comparison_needs_no_tokenizer(first, expected):
    case = rc.parse_case({**valid_case(), "gen_tokens": ["token_id:1"],
        "gen_top_logprobs": [{"token_id:1": -0.1, "token_id:2": -2.0}]})
    result = rc.next_token_check(case, [first, 2])
    assert result["match"] is expected
    assert result["top5_overlap"] == (1.0 if expected else 0.5)


def test_available_tokenizer_retains_text_comparison():
    case = rc.parse_case({**valid_case(), "gen_tokens": ["x"],
                          "gen_top_logprobs": [{"x": -0.1, "y": -2.0}]})
    result = rc.next_token_check(case, [1, 2], lambda ids: {1: "x", 2: "y"}[ids[0]])
    assert result["our_text"] == "x"
    assert result["match"] is True
    assert result["top5_overlap"] == 1.0


@pytest.mark.parametrize("decode", [None, lambda ids: str(ids)])
def test_empty_next_token_prediction_is_unknown(decode):
    case = rc.parse_case({**valid_case(), "gen_tokens": ["token_id:1"],
                          "gen_top_logprobs": [{"token_id:1": -0.1}]})
    result = rc.next_token_check(case, [], decode)
    assert result["our_id"] is None
    assert result["match"] is None
    assert result["top5_overlap"] is None
