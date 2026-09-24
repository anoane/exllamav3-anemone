"""Host-only layout of streamed expert contributions for ordered GPU reduction."""
import math


def slot_layout(counts, batches):
    """Return (base, kind, rows) for (expert ids, reconstruct groups, singles) batches.

    Fused and single-expert outputs are already weighted (kind 1). Reconstructed
    outputs are unweighted (kind 2), laid out as contiguous [B, cmax, hidden]
    slabs. Padding is never gathered. Inactive/CPU-tail experts remain kind 0.
    Every assignment has its own row, even if a token selects an expert twice.
    """
    bases, kinds = [0] * len(counts), [0] * len(counts)
    rows = 0
    for batch, groups, _ in batches:
        reconstructed = {e for group in groups for e in group}
        for e in batch:
            if e not in reconstructed:
                bases[e], kinds[e] = rows, 1
                rows += counts[e]
        for group in groups:
            cmax = max(counts[e] for e in group)
            for e in group:
                bases[e], kinds[e] = rows, 2
                rows += cmax
    return bases, kinds, rows


def slot_rows_bound(assignments, padding, group_cap):
    """Bound total slots, including reconstruct padding, without knowing routing.

    Each group has at most group_cap experts and at most padding * real_rows
    padded rows (singletons need no padding). The cap also bounds non-finite
    padding settings, for which the group planner's ratio test is ineffective.
    """
    ratio = max(1, group_cap)
    if math.isfinite(padding):
        ratio = max(1, min(ratio, padding))
    return math.ceil(assignments * ratio)
