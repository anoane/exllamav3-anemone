import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.modules.attention_fn.dsa_triton import dsa_indexer_scores


device = os.environ.get("EXL3_TEST_DEVICE", "cuda:0")
GiB = 1024**3


def _require_cuda_memory(min_free_bytes: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < min_free_bytes:
        pytest.skip(f"test requires at least {min_free_bytes / GiB:.1f} GiB free CUDA memory")


def _empty_cuda_cache():
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_indexer_score_row_offset_past_int32():
    """Score store offsets are row * S_stride + col. Without a backing S_stride is
    next_pow2(ceil128(T)), which is 2**18 once a CSA pool passes 131072 entries (~524K context),
    so the int32 boundary sits at row 2**31 / 2**18 = 8192 and a prefill chunk of 8448 rows
    crosses it. Rows past it wrapped to 2**32 elements below their slot before the widening:
    the store lands in the guard region placed there and the real row keeps its NaN sentinel.
    ~8.2 GiB: the score buffer and the guard must each span 2**31 elements (4 GiB fp16)."""
    _require_cuda_memory(10 * GiB)

    R = 8448  # one 256-token page past the 8192-row boundary
    T = 128
    H_i = 64
    D_i = 128
    m = 4
    s_stride = 2**18
    guard_len = 2**31

    torch.manual_seed(0)
    q_idx = torch.randn((R, H_i, D_i), dtype = torch.half, device = device)
    w = torch.randn((R, H_i), dtype = torch.half, device = device)
    k_idx = torch.randn((T, D_i), dtype = torch.half, device = device)

    # Score backing preceded by a guard exactly 2**31 elements long, so every wrapped row
    # offset lands inside the allocation instead of faulting
    buf = torch.full((guard_len + R * s_stride,), torch.nan, dtype = torch.half, device = device)
    guard = buf[:guard_len].view(-1, s_stride)
    backing = buf[guard_len:].view(R, s_stride)

    # q_pos0 = T * m: every row's causal bound covers the whole pool, no -inf in the checked span
    got = dsa_indexer_scores(q_idx, w, k_idx, T * m, m, T, scores = backing)

    # 0 and 8191 sit below the boundary as controls; the rest overflow without the fix
    rows = torch.tensor([0, 8191, 8192, 8300, R - 1], device = device)
    logits = torch.einsum("rhd,td->rht", q_idx[rows].double(), k_idx.double())
    ref = torch.einsum("rht,rh->rt", torch.relu(logits), w[rows].double()) * (D_i ** -0.5 * H_i ** -0.5)
    torch.testing.assert_close(got[rows].double(), ref, atol = 2e-3, rtol = 2e-3)
    assert guard[:R - 8192, :T].isnan().all()

    del got, backing, guard, buf, k_idx, w, q_idx
    _empty_cuda_cache()
