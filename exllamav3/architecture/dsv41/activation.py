"""
DeepSeek-V4.1's routed and shared expert activation (EXL3_DSV41_ACTIVATION), chosen when the model
object is built. Both are SwiGLU with the checkpoint's swiglu_limit; they differ in what the limit
clamps and in which precision SiLU and the product are evaluated:

    silu        (default) the engine's gated SiLU, as for every other model: SiLU of the gate in
                the precision of the projections (FP16 for V4.1's experts), then the ACTIVATED
                gate clamped from above and the up value symmetrically, product in that precision
    reference   DeepSeek's reference form (activation "silu_ref", exllamav3_ext/silu_ref.cuh): the
                RAW gate clamped from above and the up value symmetrically, then silu(gate) * up
                in FP32, rounded once to FP16

Syntax: one of the two names, case-insensitive, surrounding whitespace ignored; unset or empty
means silu. Anything else is refused, never silently replaced by the default.
"""

from __future__ import annotations

import math

ENV_ACTIVATION = "EXL3_DSV41_ACTIVATION"
DEFAULT = "silu"
# setting -> the activation_fn of BlockSparseMLP / GatedMLP
ACTIVATIONS = {"silu": "silu", "reference": "silu_ref"}
_FLT_MAX = float.fromhex("0x1.fffffep+127")


def activation_fn(value: str | None, act_limit: float) -> str:
    """
    The activation_fn for the setting `value` (None or "" for the default) and the checkpoint's
    swiglu_limit. Raises ValueError for another value, and for "reference" with a limit that is not
    finite, nonnegative and representable in FP32 (0 disables the clamps).
    """
    name = (value or "").strip().lower() or DEFAULT
    if name not in ACTIVATIONS:
        raise ValueError(f"{ENV_ACTIVATION}={value!r}: expected one of {', '.join(ACTIVATIONS)}")
    fn = ACTIVATIONS[name]
    if fn == "silu_ref" and not (math.isfinite(act_limit) and 0.0 <= act_limit <= _FLT_MAX):
        raise ValueError(f"{ENV_ACTIVATION}={name}: swiglu_limit {act_limit} must be finite, nonnegative and "
                         f"representable in FP32")
    return fn
