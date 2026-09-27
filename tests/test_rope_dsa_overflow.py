import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn.dsa_triton import dsa_attn
from exllamav3.util.rope import RopeStyle


device = os.environ.get("EXL3_TEST_DEVICE", "cuda:0")
GiB = 1024**3

# DeepSeek-V4 query geometry: 64 heads x 512 dim (448 nope | 64 rope). Element offsets
# (row * 64 + head) * 512 cross 2**31 at row 2**31 / 32768 = 65536
H = 64
D = 512
D_R = 64
Q_LEN = 65600  # 64 rows past the boundary


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
def test_rope_q_offsets_past_int32():
    """ext.rope addresses head (row * num_heads + head) * head_stride; rows past 65535 wrapped
    negative before the int64 row term: the in-place rotation reads and writes out of bounds
    and the row is left unrotated. ~4 GiB: q must span > 2**31 elements."""
    _require_cuda_memory(6 * GiB)

    q = torch.zeros((1, Q_LEN, H, D), dtype = torch.half, device = device)
    head = torch.arange(H, dtype = torch.float32, device = device)
    q[0, :, :, D - D_R] = ((head + 1) / H).half()

    # theta = pos / 65536 stays within [0, 1] so the fast sin/cos are accurate
    inv_freq = torch.full((D_R // 2,), 2.0 ** -16, dtype = torch.float32, device = device)
    ext.rope(
        q, q, None, None, inv_freq, 0, None, None,
        int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, D - D_R,
    )

    # 65535 sits below the boundary as a control; the rest overflow without the fix
    for pos in (65535, 65536, 65568, 65599):
        theta = torch.tensor(pos * 2.0 ** -16, dtype = torch.float64)
        v = ((head + 1) / H).half().double()
        torch.testing.assert_close(q[0, pos, :, D - D_R].double(), v * theta.cos(), atol = 1e-3, rtol = 1e-3)
        torch.testing.assert_close(q[0, pos, :, D - D_R + 1].double(), v * theta.sin(), atol = 1e-3, rtol = 1e-3)
        rest = torch.cat((q[0, pos, :, :D - D_R], q[0, pos, :, D - D_R + 2:]), dim = -1)
        torch.testing.assert_close(rest, torch.zeros_like(rest))

    del q
    _empty_cuda_cache()


@torch.inference_mode()
def test_dsa_attn_q_and_out_offsets_past_int32():
    """_dsa_attn_kernel addresses q at (row * H + head) * D and the group-major out at
    (head // HPG) * R * HPG * D + row * HPG * D in int32 without the WIDE_INDEX widening. The
    q load wraps negative from row 65536; the out store wraps earlier in the last group, from
    row 65088 at R = 65600, and never lands, leaving the NaN sentinel in place. ~9 GiB: q and
    out must each span > 2**31 elements."""
    _require_cuda_memory(10 * GiB)

    groups = 8
    hpg = H // groups
    q = torch.zeros((Q_LEN, H, D), dtype = torch.half, device = device)
    kv = torch.zeros((Q_LEN, D), dtype = torch.half, device = device)
    out = torch.full((groups, Q_LEN, hpg * D), torch.nan, dtype = torch.half, device = device)

    token = torch.arange(Q_LEN, dtype = torch.float32, device = device)
    head = torch.arange(H, dtype = torch.float32, device = device)
    q[:, :, 0] = ((token.remainder(127) - 63)[:, None] / 16 + head[None, :] / 32).half()
    kv[:, 0] = torch.where(token.remainder(2) == 0, 1.0, -1.0).half()
    kv[:, 1] = (token.remainder(251) / 251).half()

    # Two-row sliding window inside the chunk, empty dense pool
    pool_c = torch.zeros((1, D - D_R), dtype = torch.half, device = device)
    pool_r = torch.zeros((1, D_R), dtype = torch.half, device = device)
    bt = torch.zeros((1, 1), dtype = torch.int32, device = device)
    actual = dsa_attn(
        q, pool_c, pool_r, bt,
        kv_chunk = kv, win_len = 2, pool_len = 0, q_pos0 = 0,
        groups = groups, group_major = True, out = out,
    )

    # 65000 sits below every boundary as a control; the last group's out store crosses 2**31
    # from row 65088 at R = 65600, and the q load from row 65536
    for pos in (65000, 65088, 65536, 65599):
        got = actual[:, pos].reshape(H, D)
        keys = kv[pos - 1:pos + 1].float()
        weights = (q[pos, :, 0].float()[:, None] * keys[None, :, 0] * D ** -0.5).softmax(dim = -1)
        expected = weights @ keys[:, :2]
        torch.testing.assert_close(got[:, :2].float(), expected, atol = 2e-3, rtol = 2e-3)
        torch.testing.assert_close(got[:, 2:], torch.zeros_like(got[:, 2:]))

    del actual, out, kv, q
    _empty_cuda_cache()
