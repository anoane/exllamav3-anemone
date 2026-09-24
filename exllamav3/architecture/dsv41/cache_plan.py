"""
Which V4.1 layers own attention state, and which pool each compressed layer
reads.

V4 gives every compressed layer its own pool. V4.1 publishes compressed KV
only on kv_source_layer_ids; every other compressed layer reads the most
recent source at or below it. Two kinds of state are involved, and they are
owned differently:

    SWA ring      EVERY layer -- sliding, source and consumer alike -- keeps
                  its own ring of the last window of raw rows (recurrent
                  state; ~768 KiB per layer per slot)
    paged pool    only kv sources (compressed KV + rope part, plus index K
                  where the source is also an index source). Consumers own no
                  pool; they read their source's pool on their own device, so
                  a layer split must not separate them from it
                  (architecture/dsv41/placement.py).

    sliding layer   -> ring only
    kv source       -> ring + pool
    consumer        -> ring only; reads its kv source's pool

For DeepSeek-V4.1-Flash that is 4 pools and 40 rings instead of 38
per-layer pools: about 3.15 GiB at 1M tokens (fp16, one slot) instead of
~29 GiB.

Getting this wrong is silent: a consumer that allocates its own pool still
runs, it just attends over an empty cache and produces plausible nonsense.
"""

from __future__ import annotations


def _ranges(layers) -> str:
    """[0, 1, 2, 5] -> '0-2,5': runs of consecutive layers (placement.format_layer_ranges; not
    imported, so tests can load this file by path on its own)."""
    parts, s = [], sorted(layers)
    a = b = s[0]
    for i in s[1:] + [None]:
        if i is not None and i == b + 1:
            b = i
            continue
        parts.append(f"{a}" if a == b else f"{a}-{b}")
        if i is not None:
            a = b = i
    return ",".join(parts)


class CachePlan:
    """Per-layer state ownership derived from the V4.1 topology."""

    def __init__(self, compress_ratios, kv_source_layer_ids):
        self.compress_ratios = list(compress_ratios)
        self.kv_sources = tuple(sorted(kv_source_layer_ids))
        # kv source whose pool each layer reads (None: sliding, no pool at all)
        self.source: list[int | None] = []
        for i, r in enumerate(self.compress_ratios):
            if r == 0:
                self.source.append(None)
            else:
                src = [s for s in self.kv_sources if s <= i]
                if not src:
                    raise ValueError(
                        f"layer {i} is compressed (ratio {r}) but no kv source "
                        f"is published at or below it")
                self.source.append(src[-1])

    # -- ownership --

    @property
    def owner(self) -> list[int | None]:
        """Layer whose cache layer each layer reads (None: sliding): its kv source."""
        return list(self.source)

    def owns_ring(self, layer_idx: int) -> bool:
        """Every layer keeps its own SWA ring."""
        return 0 <= layer_idx < len(self.compress_ratios)

    def owns_pool(self, layer_idx: int) -> bool:
        """True for kv sources: the layers that allocate a paged pool."""
        return 0 <= layer_idx < len(self.source) and self.source[layer_idx] == layer_idx

    def allocation_count(self) -> int:
        """Paged pools (one per kv source); rings are one per layer on top."""
        return sum(1 for i in range(len(self.compress_ratios)) if self.owns_pool(i))

    def groups(self) -> list[tuple[int, list[int]]]:
        """(source, members) for every kv group, in layer order."""
        g: dict[int, list[int]] = {}
        for i, s in enumerate(self.source):
            if s is not None:
                g.setdefault(s, []).append(i)
        return sorted(g.items())

    def describe(self) -> str:
        sliding = [i for i, s in enumerate(self.source) if s is None]
        parts = []
        if sliding:
            parts.append(f"{_ranges(sliding)} sliding (ring only)")
        for src, members in self.groups():
            parts.append(f"{_ranges(members)} -> {src} (ratio-{self.compress_ratios[src]}, "
                         f"{len(members)} layers)")
        return "; ".join(parts)
