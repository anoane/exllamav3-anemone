"""Operand validation for ext.hgemm (cuBLAS path), without a model checkpoint."""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires CUDA")
device = "cuda:0"


def check_matches(a, b, c):
    ref = (a.float().view(-1, a.shape[-1]) @ b.float()).view(c.shape)
    rel_rms = ((c.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert float(rel_rms) < (1e-5 if c.dtype == torch.float else 1e-3)


@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
@pytest.mark.parametrize("a_shape", [(1, 256), (37, 256), (2, 19, 256)])
@torch.inference_mode()
def test_hgemm_valid(out_dtype, a_shape):
    torch.manual_seed(0)
    N = 384
    a = torch.randn(a_shape, dtype = torch.half, device = device)
    b = torch.randn(a_shape[-1], N, dtype = torch.half, device = device)
    rows = a.numel() // a_shape[-1]

    # Packed c, same leading shape as a
    c = torch.empty(a_shape[:-1] + (N,), dtype = out_dtype, device = device)
    ext.hgemm(a, b, c)
    check_matches(a, b, c)

    # Column slice of a wider output (exl3 reconstruct writes y_[:, n_start:n_end]); columns
    # outside the slice must be left alone
    y = torch.full((rows, N + 64), 123, dtype = out_dtype, device = device)
    ext.hgemm(a, b, y[:, 32 : 32 + N])
    check_matches(a, b, y[:, 32 : 32 + N])
    assert bool((y[:, :32] == 123).all()) and bool((y[:, 32 + N:] == 123).all())

    # c with spare rows: only the first rows are written
    y = torch.full((rows + 3, N), 123, dtype = out_dtype, device = device)
    ext.hgemm(a, b, y)
    check_matches(a, b, y[:rows])
    assert bool((y[rows:] == 123).all())


@torch.inference_mode()
def test_hgemm_empty():
    K, N = 128, 256
    b = torch.randn(K, N, dtype = torch.half, device = device)
    c = torch.full((0, N), 123, dtype = torch.float, device = device)
    ext.hgemm(torch.empty(0, K, dtype = torch.half, device = device), b, c)
    c = torch.full((8, 0), 123, dtype = torch.float, device = device)
    ext.hgemm(torch.randn(8, K, dtype = torch.half, device = device), b[:, :0].contiguous(), c)
    # Empty M with batched c, packed and as a row slice
    c = torch.empty(2, 0, N, dtype = torch.float, device = device)
    ext.hgemm(torch.empty(2, 0, K, dtype = torch.half, device = device), b, c)
    c = torch.empty(3, 5, N, dtype = torch.float, device = device)[:, :0]
    ext.hgemm(torch.empty(3, 0, K, dtype = torch.half, device = device), b, c)
    torch.cuda.synchronize()


@pytest.mark.parametrize("case, match", [
    ("zero_k", "nonzero inner dimension"),
    ("strided_a", "a and b must be contiguous"),
    ("transposed_b", "a and b must be contiguous"),
    ("short_c", "fewer rows than a"),
    ("strided_c_batch", "leading dimensions must collapse"),
])
@torch.inference_mode()
def test_hgemm_rejects(case, match):
    M, K, N = 32, 128, 256
    a = torch.randn(M, K, dtype = torch.half, device = device)
    b = torch.randn(K, N, dtype = torch.half, device = device)
    c = torch.empty(M, N, dtype = torch.float, device = device)
    if case == "zero_k":
        a = torch.empty(M, 0, dtype = torch.half, device = device)
        b = torch.empty(0, N, dtype = torch.half, device = device)
    elif case == "strided_a":
        a = torch.randn(M, 2 * K, dtype = torch.half, device = device)[:, :K]
    elif case == "transposed_b":
        b = torch.randn(N, K, dtype = torch.half, device = device).T
    elif case == "short_c":
        c = torch.empty(M - 1, N, dtype = torch.float, device = device)
    elif case == "strided_c_batch":
        a = a.view(2, M // 2, K)
        c = torch.empty(2, M, N, dtype = torch.float, device = device)[:, : M // 2]
    with pytest.raises(RuntimeError, match = match):
        ext.hgemm(a, b, c)
    torch.cuda.synchronize()
