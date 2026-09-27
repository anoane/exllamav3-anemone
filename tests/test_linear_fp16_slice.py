"""LinearFP16 slicing of full FP16 weights (MLP layers split above MAX_MLP_INTERMEDIATE), on CPU."""
import os
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.model.config import InferParams
from exllamav3.modules.mlp import MLP
from exllamav3.modules.quant.fp16 import LinearFP16

H = 256  # hidden size


def rand_half(*shape):
    return (torch.randn(*shape) / 16).half()


@torch.inference_mode()
def test_out_feature_slice():
    # Up/gate slice: sliced Linears load the full (in, out) weight and bias, LinearFP16 cuts the columns
    torch.manual_seed(0)
    I, a, S = 512, 128, 256
    w, b = rand_half(H, I), rand_half(I)
    lin = LinearFP16(H, S, w, b, H, I, 0, a)
    assert torch.equal(lin.weight, w[:, a : a + S])
    assert torch.equal(lin.bias, b[a : a + S])
    assert lin.weight.is_contiguous()
    assert lin.weight.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()

    x = rand_half(1, 3, H)
    y = lin.forward(x, {})
    assert y.shape == (1, 3, S)
    ref = x.float() @ w[:, a : a + S].float() + b[a : a + S].float()
    torch.testing.assert_close(y.float(), ref, atol = 2e-3, rtol = 2e-3)


@torch.inference_mode()
def test_in_feature_slice():
    # Down slice: LinearFP16 cuts the rows of the full (in, out) weight
    torch.manual_seed(1)
    I, a, S = 512, 256, 256
    w = rand_half(I, H)
    lin = LinearFP16(S, H, w, None, I, H, a, 0)
    assert torch.equal(lin.weight, w[a : a + S])
    assert lin.weight.is_contiguous()

    x = rand_half(1, 3, S)
    y = lin.forward(x, {})
    assert y.shape == (1, 3, H)
    torch.testing.assert_close(y.float(), x.float() @ w[a : a + S].float(), atol = 2e-3, rtol = 2e-3)


@torch.inference_mode()
def test_padded_weight_is_not_sliced():
    # Unsliced layers get their weight padded to (in, out) while full_* may be the unpadded sizes
    torch.manual_seed(2)
    for full_in, full_out in [(H, 512), (H - 64, 512 - 96)]:
        w, b = rand_half(H, 512), rand_half(512)
        lin = LinearFP16(H, 512, w, b, full_in, full_out, 0, 0)
        assert lin.weight is w
        assert lin.bias is b


@torch.inference_mode()
def test_split_mlp_load(tmp_path):
    # MLP split above intermediate_split_size, loaded from BF16 weights: each slice keeps only its part
    torch.manual_seed(3)
    I = 640
    w_up, b_up, w_down = rand_half(I, H).bfloat16(), rand_half(I).bfloat16(), rand_half(H, I).bfloat16()
    save_file({"m.up.weight": w_up, "m.up.bias": b_up, "m.down.weight": w_down}, str(tmp_path / "model.safetensors"))

    stc = SafetensorsCollection(str(tmp_path))
    config = SimpleNamespace(stc = stc, infer_params = InferParams())
    mlp = MLP(config, "m", H, I, key_up = "up", key_down = "down", intermediate_split_size = 256)
    try:
        mlp.load(torch.device("cpu"))
    finally:
        stc.close()

    assert [up.out_features for up in mlp.ups] == [128, 128, 384]
    for up, down in zip(mlp.ups, mlp.downs):
        a, b = up.first_out_feature, up.first_out_feature + up.out_features
        assert torch.equal(up.inner.weight, w_up[a : b].T.half())
        assert torch.equal(up.inner.bias, b_up[a : b].half())
        assert torch.equal(down.inner.weight, w_down[:, a : b].T.half())

    x = rand_half(1, 3, H)
    ref = torch.nn.functional.silu(x.float() @ w_up.float().T + b_up.float()) @ w_down.float().T
    torch.testing.assert_close(mlp.forward(x, {}).float(), ref, atol = 5e-3, rtol = 5e-3)
