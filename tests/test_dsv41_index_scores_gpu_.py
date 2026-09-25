"""GPU: with few_query = False, a lightning-indexer row's scores must not depend on how many rows
share the dsa_indexer_scores call (the setting DeepSeek-V4.1's selector uses under
EXL3_STABLE_ARITHMETIC=1).

By default calls of up to 4 rows take the few-query kernel and longer calls the query-tiled one,
which reduce over the heads in different orders. On every visible GPU, for a contiguous and a
paged pool and compression rates 1 and 4 (V4.1's index sources use 4, where the causal bound
(pos0 + r + 1) // m changes row by row inside a tile), rows scored in calls of 1, 3 and 4 rows
with few_query = False must equal the same rows of one 64-row call bitwise. How many rows the
default few-query kernel scores differently is reported, not required.

    python tests/test_dsv41_index_scores_gpu_.py

Skipped when no GPU is visible."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import skip


def check_device(dev, dsa_indexer_scores, m):
    g = torch.Generator(device = dev).manual_seed(7)
    R, H, D, T, epp = 64, 32, 128, 7937, 64
    q = (torch.randn(R, H, D, generator = g, device = dev) * 2).half()
    w = torch.randn(R, H, generator = g, device = dev).half()
    k = torch.randn(T, D, generator = g, device = dev).half()
    pos0 = m * T - R                              # every row sees nearly the whole pool
    pages = -(-T // epp)
    perm = torch.randperm(pages, generator = torch.Generator().manual_seed(3)).to(dev).int()
    paged = torch.zeros(pages * epp, D, dtype = torch.half, device = dev)
    for p in range(pages):
        n = min(epp, T - p * epp)
        paged[perm[p] * epp : perm[p] * epp + n] = k[p * epp : p * epp + n]
    ok = True
    for layout, pool, kw in (("contiguous", k, {}), ("paged", paged, {"block_table": perm, "epp": epp})):
        layout_ok = True
        full = dsa_indexer_scores(q, w, pool, pos0, m, T, **kw)[:, :T].clone()
        for rows in (1, 3, 4):
            for r in range(0, R - rows + 1, rows):
                part = dsa_indexer_scores(q[r : r + rows], w[r : r + rows], pool, pos0 + r, m, T,
                                          few_query = False, **kw)[:, :T]
                if not torch.equal(part, full[r : r + rows]):
                    print(f"  FAIL {dev} {layout} m={m}: rows {r}-{r + rows - 1} scored in a {rows}-row "
                          f"call differ from the {R}-row call")
                    layout_ok = False
                    break
        differ = 0
        for r in range(R):
            one = dsa_indexer_scores(q[r : r + 1], w[r : r + 1], pool, pos0 + r, m, T, **kw)[:, :T]
            differ += int(not torch.equal(one[0], full[r]))
        print(f"  {'OK ' if layout_ok else 'FAIL'} {dev} {layout} m={m}: rows in 1-, 3- and 4-row calls "
              f"with few_query = False bitwise equal to one {R}-row call; the default few-query kernel "
              f"scores {differ}/{R} single rows differently")
        ok &= layout_ok
    return ok


def main():
    from exllamav3.modules.attention_fn.dsa_triton import dsa_indexer_scores
    ok = True
    for d in range(torch.cuda.device_count()):
        for m in (1, 4):
            ok &= check_device(torch.device("cuda", d), dsa_indexer_scores, m)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        skip("V4.1 index scores on the GPU", "no CUDA device")
        sys.exit(0)
    sys.exit(main())
