"""
EXL3_DSV41_ABLATE parsing (modules/dsv41_ablation.py): the token set, whitespace and empty
entries, unknown tokens refused, the variable read once at the first call, the test-only
set_ablations() and its restore round trip, one warning per token. CPU only; the compiled
extension is not imported.

    python -m pytest tests/test_dsv41_ablation_.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file

ab = load_package_file("exllamav3/modules/dsv41_ablation.py", "_dsv41_ablation")
ENV = "EXL3_DSV41_ABLATE"


@pytest.fixture(autouse = True)
def unread():
    """Every test starts and ends with the variable not yet read."""
    ab.set_ablations(None)
    yield
    ab.set_ablations(None)


def test_unset_and_empty_ablate_nothing(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    assert ab.ablations() == frozenset()
    for value in ("", " ", ",", " , ,"):
        ab.set_ablations(None)
        monkeypatch.setenv(ENV, value)
        assert ab.ablations() == frozenset()
        assert not any(ab.ablated(t) for t in ab.ABLATE_TOKENS)


def test_comma_list_with_spaces(monkeypatch):
    monkeypatch.setenv(ENV, " engram , no_candidates,,")
    assert ab.ablations() == {"engram", "no_candidates"}
    assert ab.ablated("engram") and ab.ablated("no_candidates")
    assert not ab.ablated("v4mix") and not ab.ablated("dense_consumers")


def test_every_token_is_accepted(monkeypatch):
    monkeypatch.setenv(ENV, ",".join(ab.ABLATE_TOKENS))
    assert ab.ablations() == frozenset(ab.ABLATE_TOKENS)
    assert len(ab.ABLATE_TOKENS) == 6


def test_read_once(monkeypatch):
    monkeypatch.setenv(ENV, "v4mix")
    assert ab.ablated("v4mix")
    # later changes to the environment do not reach the parsed set
    monkeypatch.setenv(ENV, "rope_consecutive")
    assert ab.ablated("v4mix") and not ab.ablated("rope_consecutive")
    monkeypatch.delenv(ENV)
    assert ab.ablated("v4mix")
    # ablated() and ablations() share the one reading
    assert ab.ablations() == {"v4mix"}


def test_set_ablations_round_trip(monkeypatch):
    monkeypatch.setenv(ENV, "engram")
    # before the first call there is nothing to return: None, which restores "not read yet"
    assert ab.set_ablations("dense_consumers") is None
    assert ab.ablations() == {"dense_consumers"}
    assert ab.set_ablations(None) == "dense_consumers"
    assert ab.ablations() == {"engram"}, "None must make the next call read the variable"
    # the returned value is the previous setting in the variable's syntax, and restores it
    previous = ab.set_ablations(" v4mix , no_candidates ")
    assert previous == "engram" and ab.ablations() == {"v4mix", "no_candidates"}
    assert ab.set_ablations("") == "no_candidates,v4mix"
    assert ab.ablations() == frozenset()
    assert ab.set_ablations(previous) == ""
    assert ab.ablations() == {"engram"}


def test_set_ablations_refuses_unknown_tokens(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    ab.set_ablations("engram")
    with pytest.raises(ValueError, match = ENV):
        ab.set_ablations("engram,typo")
    assert ab.ablations() == {"engram"}, "a refused value changed the setting"


@pytest.mark.parametrize("value", ["engram,typo", "Engram", "skip_engram", "engram;v4mix"])
def test_unknown_tokens_are_refused(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match = ENV):
        ab.ablations()
    # still refused at the next call: a bad value never turns into "no ablation"
    with pytest.raises(ValueError):
        ab.ablated("engram")


def test_querying_an_unknown_token_is_an_error(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    with pytest.raises(ValueError):
        ab.ablated("no_candidate")


def test_one_warning_per_token(monkeypatch, capsys):
    monkeypatch.setattr(ab, "_warned", set())
    monkeypatch.setenv(ENV, "engram_nocarry")
    ab.ablated("engram_nocarry")
    ab.ablated("engram_nocarry")
    ab.set_ablations("engram_nocarry,dense_consumers")
    ab.ablated("dense_consumers")
    ab.set_ablations("")
    ab.set_ablations("dense_consumers")
    out = capsys.readouterr().out
    assert out.count(f" !! {ENV}=engram_nocarry:") == 1
    assert out.count(f" !! {ENV}=dense_consumers:") == 1
