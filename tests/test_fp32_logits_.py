"""
Optional FP32 logits (config.infer_params.fp32_logits, default from EXL3_FP32_LOGITS). The CPU
tests check that the option defaults to off and changes nothing then, that Model.load gives the
logits output layer FP32 before any module loads and restores the layer's own dtype when the
option is cleared, that the TP planner's output estimate follows, and that the consumers which
dispatch on the logits dtype (categorical sampling without the fused sampler, the TP argmax of a
rank without a head slice) take FP32, and that the temperature step never scales the caller's
logits in place, so the logits a job returns stay unscaled. The CUDA test compares the FP32 logits of an FP16 head
with its default FP16 output.

    python -m pytest tests/test_fp32_logits_.py
"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

from exllamav3 import Model
from exllamav3.model.config import InferParams
from exllamav3.model import model_tp_fn
from exllamav3.modules import Linear
from exllamav3.generator.sampler import custom
from exllamav3.generator.sampler.custom import SS, SS_Sample, SS_Temperature, SS_Argmax, CustomSampler

VOCAB = 1000
VOCAB_PADDED = (VOCAB + 127) // 128 * 128
HIDDEN = 256


def _config(fp32_logits):
    stc = SimpleNamespace(has_tensor_group = lambda *args: False)
    return SimpleNamespace(stc = stc, infer_params = SimpleNamespace(fp32_logits = fp32_logits))


def _stub_model(fp32_logits, head_dtype = None):
    config = _config(fp32_logits)
    model = Model.__new__(Model)
    model.config = config
    model.cache_weakrefs = {}
    model.modules = [
        Linear(config, "down_proj", HIDDEN, HIDDEN, out_dtype = torch.float),
        Linear(config, "lm_head", HIDDEN, VOCAB, caps = {"logits_output": True}, out_dtype = head_dtype),
    ]
    return model


@pytest.mark.parametrize("value, expect", [(None, False), ("0", False), ("1", True)])
def test_env_default(monkeypatch, value, expect):
    if value is None:
        monkeypatch.delenv("EXL3_FP32_LOGITS", raising = False)
    else:
        monkeypatch.setenv("EXL3_FP32_LOGITS", value)
    assert InferParams().fp32_logits is expect


@pytest.mark.parametrize("head_dtype", [None, torch.half])
def test_load_sets_and_restores_logits_dtype(monkeypatch, head_dtype):
    model = _stub_model(True, head_dtype)
    down, head = model.modules
    seen = []

    def fake_load_single(progressbar, device, config, modules, verbose):
        # What the modules look like when they load: inner layers and output buffers take the dtype
        seen.append([m.out_dtype for m in modules])
        for m in modules:
            m.device = torch.device(device)

    monkeypatch.setattr(model, "_load_single", fake_load_single)

    model.load(device = "cpu")
    model.load(device = "cpu")
    assert seen == [[torch.float, torch.float]] * 2

    # Cleared before the next load: the layer's own dtype comes back, other layers never change
    model.config.infer_params.fp32_logits = False
    model.load(device = "cpu")
    assert seen[-1] == [torch.float, head_dtype]
    assert down.out_dtype == torch.float and head.out_dtype == head_dtype


def test_default_leaves_modules_alone(monkeypatch):
    model = _stub_model(False)
    monkeypatch.setattr(model, "_load_single", lambda *args: None)
    model.load(device = "cpu")
    assert [m.out_dtype for m in model.modules] == [torch.float, None]


def test_tp_output_estimate_follows():
    def overhead(fp32):
        model = _stub_model(fp32)
        model._apply_logits_dtype()
        tpa, = model.modules[-1].make_tp_allocation({})
        return tpa.overhead_per_device, tpa.overhead_to_split

    half_d, half_s = overhead(False)
    assert (half_d, half_s) == (VOCAB_PADDED * 2, VOCAB_PADDED * 2)
    assert overhead(True) == (half_d * 2, half_s * 2)


@pytest.mark.parametrize("dtype, kernel", [(torch.half, "gumbel_noise_f16"), (torch.float, "gumbel_noise_f32")])
def test_categorical_sampling_takes_logits_dtype(monkeypatch, dtype, kernel):
    calls = []

    def noise(name):
        def fn(logits_in, logits, rand_u32):
            calls.append((name, logits_in.dtype, logits.dtype))
            logits.copy_(logits_in)
        return fn

    monkeypatch.setattr(custom, "ext", SimpleNamespace(
        gumbel_noise_f16 = noise("gumbel_noise_f16"),
        gumbel_noise_f32 = noise("gumbel_noise_f32"),
    ))
    logits = torch.zeros((2, VOCAB), dtype = dtype)
    logits[0, 7] = 1.0
    logits[1, 3] = 1.0
    state = SimpleNamespace(state = SS.INIT, in_logits = logits, rand_u32 = 1)
    SS_Sample().run(state)
    assert calls == [(kernel, dtype, dtype)]
    assert state.sample.tolist() == [7, 3] and state.state == SS.DONE


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
def test_temperature_copies_logits(dtype):
    # in_logits is a view of the model's logits, which the generator reads again after sampling
    # (return_logits, return_probs); .float() would alias FP32 logits and scale them in place
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]], dtype = dtype)
    before = logits.clone()
    state = SimpleNamespace(state = SS.INIT, in_logits = logits)
    SS_Temperature(0.5).run(state)
    assert torch.equal(logits, before)
    assert state.logits.untyped_storage().data_ptr() != logits.untyped_storage().data_ptr()
    assert state.logits.dtype == torch.float and state.state == SS.LOGITS
    assert torch.equal(state.logits, before.float() / 0.5)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
def test_unfused_temperature_stack_keeps_logits(monkeypatch, dtype):
    # As with EXL3_FUSED_SAMPLER=0 (read at import), which keeps the stack unfused
    monkeypatch.setattr(custom, "fused_sampler_enable", False)
    logits = torch.tensor([[1.0, 2.0, 4.0]], dtype = dtype)
    before = logits.clone()
    sample = CustomSampler([SS_Temperature(0.5), SS_Argmax()]).forward(logits)
    assert sample.flatten().tolist() == [2]
    assert torch.equal(logits, before)


@pytest.mark.parametrize("head_dtype", [None, torch.float])
def test_tp_argmax_rank_without_head_slice(head_dtype):
    # A rank that holds no slice of the head still contributes a (zero-width) maxima tensor to
    # the gather, whose element size must match the logits of the ranks that do
    x = torch.zeros((1, 3, HIDDEN), dtype = torch.half)
    local_context = {
        "inf_consumer": SimpleNamespace(recv = lambda shared: x),
        "device": 1,
        "output_device": 0,
        "backend": None,
        "logits_module": SimpleNamespace(out_dtype = head_dtype),
    }
    v, i = model_tp_fn.mp_model_forward_lm_head_argmax(local_context, None, {}, -1, None, None, VOCAB)
    assert v.dtype == (head_dtype or x.dtype) and i.dtype == torch.long
    assert v.shape == i.shape == (1, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@torch.inference_mode()
def test_fp32_logits_match_fp16_head():
    from exllamav3.modules.quant import LinearFP16
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    weight = (torch.randn((HIDDEN, VOCAB_PADDED), device = device) * 0.2).half()
    x = torch.randn((1, 64, HIDDEN), device = device).half()

    def head(out_dtype):
        # As loaded: the inner layer takes the head's output dtype
        h = Linear(None, "lm_head", HIDDEN, VOCAB, caps = {"logits_output": True}, out_dtype = out_dtype)
        h.device = device
        h.inner = LinearFP16(HIDDEN, VOCAB_PADDED, weight, None, HIDDEN, VOCAB_PADDED, 0, 0, h.out_dtype)
        return h

    ref = x.float() @ weight.float()
    y16 = head(None).forward(x, {})
    y32 = head(torch.float).forward(x, {})
    assert y16.dtype == torch.half and y32.dtype == torch.float
    # FP32 accumulation either way; the FP16 output adds the rounding to FP16
    assert (y32 - ref).abs().max().item() < 1e-3
    assert (y16.float() - y32).abs().max().item() <= (y32.abs().max().item() * 2 ** -11) + 1e-3
    assert (y16.float() - ref).abs().max().item() > (y32 - ref).abs().max().item()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
