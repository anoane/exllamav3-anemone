"""
Deliberate errors in DeepSeek-V4.1's function, for tests and validation runs only.

EXL3_DSV41_ABLATE is a comma-separated list of tokens. Each token makes the model compute a
known WRONG function, so a test or a validation gate can show that it detects the error it
exists to detect (a negative control). Never set it for inference: every token is announced
with a warning when it becomes active.

    engram            the engram layers do not run; their n-gram context is still committed,
                      so the ids stay current if the ablation is turned off mid-sequence
    engram_nocarry    each forward hashes its first n-grams without the ids carried from the
                      previous forward (as at position 0): a state-carry bug that only a
                      chunked run can show
    v4mix             every sublayer collapses its input with its own pre-mix, as V4 does,
                      instead of the pre-mix carried from the previous sublayer
    dense_consumers   compressed layers that are not index sources attend over their whole
                      pool instead of their index source's top-k selection
    rope_consecutive  pool and index keys are rotated at first_entry * m + j instead of
                      (first_entry + j) * m: one position apart, wrong at compression rate 2
    no_candidates     the candidate source publishes no candidate blocks and the index sources
                      above it score every visible entry

The variable is read once, at the first ablated() or ablations() call, and parsed then; later
changes to the environment have no effect. An unknown token raises ValueError at that call
(and at every later one) rather than silently computing the real function.

Tests switch ablations on one loaded model with set_ablations(), the test-only setter, which
replaces the parsed set and returns the previous setting so it can be restored (the pattern of
util/device_copy.py, which reads EXLLAMA_NO_P2P_COPY once into a module-level setting that
tests replace).
"""

import os

ENV_ABLATE = "EXL3_DSV41_ABLATE"

ABLATE_TOKENS = (
    "engram",
    "engram_nocarry",
    "v4mix",
    "dense_consumers",
    "rope_consecutive",
    "no_candidates",
)

# The active tokens, or None until EXL3_DSV41_ABLATE has been read
_active: frozenset | None = None
_warned = set()


def _parse(value: str) -> frozenset:
    tokens = frozenset(t.strip() for t in value.split(",") if t.strip())
    unknown = sorted(tokens.difference(ABLATE_TOKENS))
    if unknown:
        raise ValueError(
            f"{ENV_ABLATE}: unknown token(s) {', '.join(unknown)}; "
            f"expected a comma-separated list of {', '.join(ABLATE_TOKENS)}"
        )
    return tokens


def _activate(tokens: frozenset) -> frozenset:
    """Make `tokens` the active set, warning once per process for each token."""
    global _active
    for token in sorted(tokens.difference(_warned)):
        print(f" !! {ENV_ABLATE}={token}: deliberately NOT DeepSeek-V4.1's function, "
              f"for tests and validation only", flush = True)
        _warned.add(token)
    _active = tokens
    return tokens


def ablations() -> frozenset:
    """
    The set of active ablation tokens. The first call reads and parses EXL3_DSV41_ABLATE;
    later calls return the same set, or what set_ablations() put in its place.
    """
    active = _active
    if active is None:
        active = _activate(_parse(os.environ.get(ENV_ABLATE, "")))
    return active


def ablated(token: str) -> bool:
    """
    True when the ablation `token` is active (EXL3_DSV41_ABLATE, read at the first call).
    """
    if token not in ABLATE_TOKENS:
        raise ValueError(f"{ENV_ABLATE}: {token!r} is not an ablation token")
    return token in ablations()


def set_ablations(value: str | None) -> str | None:
    """
    Tests only: make `value`, in EXL3_DSV41_ABLATE's syntax ("" for none), the active set, and
    return the previous setting in the same syntax, or None when the variable had not been read
    yet. None clears the setting, so the next call reads EXL3_DSV41_ABLATE again. Restoring
    the returned value therefore restores exactly what was there:

        previous = set_ablations("dense_consumers")
        try:
            ...
        finally:
            set_ablations(previous)

    An unknown token raises ValueError and leaves the setting unchanged.
    """
    global _active
    tokens = None if value is None else _parse(value)
    previous = None if _active is None else ",".join(sorted(_active))
    if tokens is None:
        _active = None
    else:
        _activate(tokens)
    return previous
