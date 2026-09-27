"""
Row-tile launch plan of the fused MoE kernel (fused_row_tiers, shared by the GPU-resident tier in
block_sparse_mlp and the streamed tier in moe_cpu_host). exl3_moe refuses a tile taller than its
temp buffers (TORCH_CHECK in exl3_moe.cu), whose row count is EXL3_MOE_FUSED_ROWS_WIDE /
EXL3_MOE_STREAM_FUSED_T. Regression: the plan always put experts above 32 rows on the 64-row tile
and experts above 16 on the 32-row one, so a row capacity of 17-31 or 33-63 raised mid-prefill
as soon as an expert got that many rows. Every plan must stay within the buffers and cover each
fused expert exactly once; at 64 rows and up (the default is 256) it is unchanged.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from exllamav3.modules.block_sparse_mlp import fused_row_tiers


def _check(counts, rows, mtile):
    tiers = fused_row_tiers(counts, rows, mtile)
    tiles = [mt for _, _, _, mt in tiers]
    assert tiles == sorted(tiles, reverse = True)
    covered = 0
    for n_act, lo, hi, mt in tiers:
        assert mt in (16, 32, 64)
        # exl3_moe.cu: tiles above 16 rows need temp buffers at least that tall
        assert mt <= 16 or rows >= mt, f"rows {rows}, counts {counts}: {mt}-row tile"
        assert mt <= 16 or mtile
        assert n_act == sum(1 for c in counts if lo <= c <= hi)
        covered += n_act
    assert covered == len(counts)
    for c in counts:
        assert sum(1 for _, lo, hi, _ in tiers if lo <= c <= hi) == 1
    return tiers


@pytest.mark.parametrize("mtile", [True, False])
def test_tiles_fit_buffers(mtile):
    for rows in range(1, 300):
        for c in range(1, rows + 1):
            _check([c], rows, mtile)
            _check([c] + [x for x in (1, 3, 16, 17, 32, 33, 64, 65) if x <= rows], rows, mtile)


def test_wide_plan_unchanged():
    counts = [1, 3, 16, 17, 20, 32, 33, 64, 200, 256]
    assert _check(counts, 256, True) == [(4, 33, 256, 64), (3, 17, 32, 32), (3, 1, 16, 16)]
    assert _check([3, 40], 64, True) == [(1, 33, 64, 64), (1, 1, 16, 16)]
    assert _check([20], 32, True) == [(1, 17, 32, 32)]
    assert _check(counts, 256, False) == [(10, 1, 256, 16)]
    assert _check([1, 16], 256, True) == [(2, 1, 256, 16)]
    assert _check([], 256, True) == [(0, 1, 256, 16)]


def test_narrow_buffers_step_down():
    assert _check([3, 18, 35, 40], 40, True) == [(3, 17, 40, 32), (1, 1, 16, 16)]
    assert _check([3, 18, 20], 20, True) == [(3, 1, 20, 16)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
