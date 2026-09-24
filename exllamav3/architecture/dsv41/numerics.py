"""
DeepSeek-V4.1 attention numerics: how the lightning-indexer operands and the attention KV are
rounded. This module only parses and names the setting (no torch); the rounding itself is in
modules/dsv41_rounding.py.

DeepSeek's reference inference code (inference/model.py and kernel.py) rounds three attention
operands on purpose, the way the model was trained. Each setting is a contract, optionally
restricted to some of its parts:

    precise    FP16 index operands, FP16 window and compressed KV: no rounding beyond the
               port's own FP16 storage.
    deepseek   DeepSeek's reference contract:
                 index        index Q/K, post-RoPE: MXFP4, E2M1 with one E8M0 (power-of-two)
                              scale per 32 lanes (fp4_act_quant)
                 window       sliding-window KV: FP8 E4M3 with one E8M0 scale per 32 lanes,
                              RoPE lanes included (act_quant, scale_fmt ue8m0)
                 compressed   compressed KV: E2M1 with one E4M3 scale per 16 lanes, RoPE lanes
                              included (fp4_act_quant, scale_dtype float8_e4m3fn)
    vllm       the contract of vLLM's V4.1 path:
                 index        MXFP4, as above
                 window and   fp8_ds_mla: the 448 NoPE lanes E4M3 with one power-of-two scale
                 compressed   per 64 lanes, the 64 RoPE lanes BF16

Syntax: a contract name, optionally followed by ':' and a comma-separated list of parts, e.g.
"deepseek:index" or "vllm:window,compressed". A contract without parts applies all three.
Names are case-insensitive and whitespace around them is ignored. "precise" takes no parts.

DEFAULT is "deepseek:index", the MXFP4 rounding of the index operands alone: of the settings
compared on DeepSeek-V4.1-Flash it gave the lowest error (doc/env_vars.md, EXL3_DSV41_NUMERICS).

Rounded values are widened back into the FP16 operands and cache: a setting reproduces a
contract's values, not its memory savings. Rounding compressed entries is defined on the FP16
pool, so a setting with the compressed part cannot be combined with a quantized (packed) pool
(-cq); DeepseekV41Config refuses the combination when the Cache is built and when the setting
is changed while such a Cache exists.

A block whose rounding would store a value FP16 cannot hold, or whose operand is not finite,
keeps its unrounded values (those 'precise' stores) and is counted, with one warning per process;
EXL3_DSV41_NUMERICS_STRICT=1 raises ValueError instead (modules/dsv41_rounding.py).
"""

from __future__ import annotations
from dataclasses import dataclass

ENV_NUMERICS = "EXL3_DSV41_NUMERICS"
# read by modules/dsv41_rounding.py: raise on an overflowing or nonfinite operand instead of
# keeping that block unrounded
ENV_STRICT = "EXL3_DSV41_NUMERICS_STRICT"
CONTRACTS = ("precise", "deepseek", "vllm")
PARTS = ("index", "window", "compressed")
DEFAULT = "deepseek:index"

# NoPE lanes of a V4.1 attention entry (head_dim 512 - rope_head_dim 64)
NOPE = 448


@dataclass(frozen = True)
class Numerics:
    """A parsed setting. Built by parse(); a hand-built one is checked the same way, since the
    rounding treats every contract other than 'deepseek' as vLLM's."""
    contract: str = "precise"
    index: bool = False
    window: bool = False
    compressed: bool = False

    def __post_init__(self):
        parts = [p for p in PARTS if getattr(self, p)]
        if self.contract not in CONTRACTS:
            raise ValueError(f"dsv41 numerics: contract {self.contract!r} is not one of "
                             f"{', '.join(CONTRACTS)}")
        if self.contract == "precise" and parts:
            raise ValueError(f"dsv41 numerics: 'precise' takes no parts, got {', '.join(parts)}")
        if self.contract != "precise" and not parts:
            raise ValueError(f"dsv41 numerics: contract {self.contract!r} needs at least one of "
                             f"{', '.join(PARTS)}")

    def __str__(self):
        if self.contract == "precise":
            return "precise"
        parts = [p for p in PARTS if getattr(self, p)]
        return self.contract if len(parts) == len(PARTS) else f"{self.contract}:{','.join(parts)}"


def parse(value) -> Numerics:
    """
    A Numerics, an unset or empty value (the default), or a string: 'precise', 'deepseek' or
    'vllm', optionally ':' and a nonempty comma-separated subset of 'index', 'window' and
    'compressed'. Anything else raises ValueError.
    """
    if isinstance(value, Numerics):
        return value
    s = ("" if value is None else str(value)).strip().lower() or DEFAULT
    contract, sep, parts = s.partition(":")
    contract = contract.strip()
    if contract not in CONTRACTS:
        raise ValueError(f"dsv41 numerics {value!r}: expected one of {', '.join(CONTRACTS)}, "
                         f"optionally ':' and parts from {', '.join(PARTS)}")
    if contract == "precise":
        if sep:
            raise ValueError(f"dsv41 numerics {value!r}: 'precise' takes no parts")
        return Numerics()
    sel = set(PARTS) if not sep else {p.strip() for p in parts.split(",")}
    if not sel or sel - set(PARTS):
        raise ValueError(f"dsv41 numerics {value!r}: parts must be a nonempty subset of "
                         f"{', '.join(PARTS)}")
    return Numerics(contract, "index" in sel, "window" in sel, "compressed" in sel)
