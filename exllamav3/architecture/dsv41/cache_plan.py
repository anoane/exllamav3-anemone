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
                  where the source is also an index source) and, when the split
                  of an explicit placement cuts a kv group, the replica owner
                  past the split (architecture/dsv41/placement.py). Consumers
                  own no pool.

    sliding layer   -> ring only
    kv source       -> ring + pool
    replica owner   -> ring + replica of its source's pool (same geometry;
                       index K only when an index source past the split
                       borrows it)
    consumer        -> ring only; reads its route owner's pool

For DeepSeek-V4.1-Flash that is 4 pools and 40 rings instead of 38
per-layer pools: about 3.15 GiB at 1M tokens (fp16, one slot) instead of
~29 GiB. A split inside a kv group adds one replica, 512 MiB for a rate-2
source whose replica carries no index K.

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
    """Per-layer state ownership derived from the V4.1 topology and pool routes."""

    def __init__(self, compress_ratios, kv_source_layer_ids, routes: dict | None = None):
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
        # (owner, replica_of, with_index_k) per compressed layer; the default
        # route reads the source itself
        self.routes: dict[int, tuple[int, int | None, bool]] = {
            i: (s, None, False) for i, s in enumerate(self.source) if s is not None
        }
        if routes is not None:
            self.set_routes(routes)

    def set_routes(self, routes: dict):
        """
        Replace the default routes (placement.compute_routes gives them for a split). Each
        compressed layer must read its own source's pool: from the source itself, or from a
        replica OF THAT SOURCE owned by an earlier member of the group, or it owns that
        replica. A route that reads another group's pool, or a kv source routed away from
        its own pool, is refused: at run time it would attend over the wrong entries without
        an error.
        """
        compressed = {i for i, s in enumerate(self.source) if s is not None}
        if set(routes) != compressed:
            raise ValueError(f"routes cover layers {sorted(routes)}, expected the compressed "
                             f"layers {sorted(compressed)}")
        for i, (owner, rep, wik) in routes.items():
            src = self.source[i]
            if i == src:
                if (owner, rep, wik) != (i, None, False):
                    raise ValueError(f"layer {i}: a kv source reads its own pool, route {routes[i]} "
                                     f"must be ({i}, None, False)")
                continue
            if rep is not None:
                if owner != i or rep != src:
                    raise ValueError(f"layer {i}: replica route {routes[i]} must be owned by the "
                                     f"layer itself and copy its own source {src}")
                continue
            if wik:
                raise ValueError(f"layer {i}: route {routes[i]} carries index K but owns no replica")
            if owner == src:
                continue
            r = routes.get(owner)
            if r is None or r[:2] != (owner, src) or not src < owner < i:
                raise ValueError(f"layer {i}: route owner {owner} neither is its source {src} nor "
                                 f"owns a replica of it below layer {i}")
        self.routes = dict(routes)

    # -- ownership --

    @property
    def owner(self) -> list[int | None]:
        """Layer whose cache layer each layer reads on its own device (None: sliding)."""
        return [self.routes[i][0] if i in self.routes else None
                for i in range(len(self.compress_ratios))]

    def owns_ring(self, layer_idx: int) -> bool:
        """Every layer keeps its own SWA ring."""
        return 0 <= layer_idx < len(self.compress_ratios)

    def owns_pool(self, layer_idx: int) -> bool:
        """True for kv sources and replica owners: the layers that allocate a paged pool."""
        r = self.routes.get(layer_idx)
        return r is not None and r[0] == layer_idx

    def replica_of(self, layer_idx: int) -> int | None:
        r = self.routes.get(layer_idx)
        return r[1] if r is not None else None

    def borrows_from(self, layer_idx: int) -> int | None:
        """Layer whose pool this one reads, or None when it owns a pool or has none."""
        r = self.routes.get(layer_idx)
        return None if r is None or r[0] == layer_idx else r[0]

    def allocation_count(self) -> int:
        """Paged pools (sources + replicas); rings are one per layer on top."""
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
            span = _ranges(members)
            reps = [i for i in members if self.replica_of(i) is not None]
            rep = "".join(f", replica on {i}" for i in reps)
            parts.append(f"{span} -> {src} (ratio-{self.compress_ratios[src]}, "
                         f"{len(members)} layers{rep})")
        return "; ".join(parts)
