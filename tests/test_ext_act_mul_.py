"""
Tests for the act(x) * y -> z kernels (activation.cu: silu_mul, silu_oai_mul, gelu_mul, relu2_mul, relu_mul).
The kernels read x and y and write z as flat buffers of x.numel() elements in aligned half2/float2 pairs, so
the host wrappers refuse any other layout before launching. The refusals are host-side and run on CPU
tensors too; the reference comparisons need a GPU.
"""
import sys, os
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

has_cuda = torch.cuda.is_available()
devices = ["cpu"] + (["cuda:0"] if has_cuda else [])


def ref_oai(x, y, limit):
    g = x.clamp(max = limit) if limit != 0.0 else x
    u = y.clamp(-limit, limit) if limit != 0.0 else y
    return (u + 1.0) * g * torch.sigmoid(1.702 * g)


act_fns = {
    "silu_mul": (ext.silu_mul, lambda x, y, l: F.silu(x) * y),
    "gelu_mul": (ext.gelu_mul, lambda x, y, l: F.gelu(x, approximate = "tanh") * y),
    "relu2_mul": (ext.relu2_mul, lambda x, y, l: F.relu(x).square() * y),
    "relu_mul": (ext.relu_mul, lambda x, y, l: F.relu(x) * y),
    "silu_oai_mul": (ext.silu_oai_mul, ref_oai),
}


def bad_layout(case, dtype, dev):
    rows, dim = 4, 256
    x = torch.randn(rows, dim, dtype = dtype, device = dev)
    y = torch.randn(rows, dim, dtype = dtype, device = dev)
    z = torch.full((rows, dim), 123.0, dtype = torch.half, device = dev)
    match case:
        case "odd":
            # odd width: the kernels would leave the last element of z unwritten
            return x[:1, 1:].contiguous(), y[:1, 1:].contiguous(), z[:1, 1:].contiguous(), "must be even"
        case "empty":
            return x[:0], y[:0], z[:0], "must not be empty"
        case "strided_x":
            return x.view(dim, rows).t(), y, z, "contiguous"
        case "strided_z":
            return x, y, torch.full((dim, rows), 123.0, dtype = torch.half, device = dev).t(), "contiguous"
        case "short_y":
            return x, y[:rows - 1], z, "incompatible shapes"
        case "short_z":
            return x, y, z[:rows - 1], "incompatible shapes"
        case "misaligned_x":
            # contiguous, but starting one element into the buffer: half2/float2 loads would be misaligned
            return torch.randn(rows * dim + 1, dtype = dtype, device = dev)[1:].view(rows, dim), y, z, "aligned"
        case "misaligned_z":
            return x, y, torch.full((rows * dim + 1,), 123.0, dtype = torch.half, device = dev)[1:].view(rows, dim), "aligned"
        case "not_cuda":
            return x, y, z, "same CUDA device"


@pytest.mark.parametrize("fn", list(act_fns.keys()))
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("case", ["odd", "empty", "strided_x", "strided_z", "short_y", "short_z", "misaligned_x", "misaligned_z", "not_cuda"])
@pytest.mark.parametrize("dev", devices)
@torch.inference_mode()
def test_act_mul_rejects_layout(fn, dtype, case, dev):
    if case == "not_cuda" and dev != "cpu":
        pytest.skip("CPU tensors only")
    x, y, z, match = bad_layout(case, dtype, dev)
    z_before = z.clone()
    with pytest.raises(RuntimeError, match = match):
        act_fns[fn][0](x, y, z, 0.0)
    assert torch.equal(z, z_before)


@pytest.mark.skipif(not has_cuda, reason = "needs CUDA")
@pytest.mark.parametrize("fn", list(act_fns.keys()))
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("rows, dim", [(1, 2), (1, 256), (7, 1536), (33, 2880)])
@pytest.mark.parametrize("layout", ["separate", "gu_slices", "in_place"])
@torch.inference_mode()
def test_act_mul(fn, dtype, rows, dim, layout):
    # The layouts the model code passes: separate outputs, gate/up halves of one [2, rows, dim] buffer,
    # and the in-place forms (z is x or y, half input only)
    torch.manual_seed(0)
    device = "cuda:0"
    call, ref_fn = act_fns[fn]
    limit = 7.0 if fn == "silu_oai_mul" else 0.0
    if layout == "gu_slices":
        gu = torch.randn(2, rows, dim, dtype = dtype, device = device) * 3
        x, y = gu[0], gu[1]
    else:
        x = torch.randn(rows, dim, dtype = dtype, device = device) * 3
        y = torch.randn(rows, dim, dtype = dtype, device = device) * 3
    ref = ref_fn(x.float(), y.float(), limit).clamp(-65504, 65504).half()
    if layout == "in_place":
        if dtype != torch.half:
            pytest.skip("z is half, so in-place needs half input")
        for target in ("x", "y"):
            xc, yc = x.clone(), y.clone()
            z = xc if target == "x" else yc
            call(xc, yc, z, limit)
            torch.testing.assert_close(z, ref, rtol = 1e-2, atol = 2e-2)
    else:
        z = torch.empty(rows, dim, dtype = torch.half, device = device)
        call(x, y, z, limit)
        torch.testing.assert_close(z, ref, rtol = 1e-2, atol = 2e-2)
