"""
EXL3_DSV41_ROUTER_BIAS (architecture/dsv41/router_bias.py): the checks on the FP32 sidecar, that
a refused file leaves the tensor collection unchanged, and that an accepted one is added to it.
CPU only. The validation runs on a stand-in config, loaded by path without the package; the
last test builds a real DeepseekV41Config from a small synthetic checkpoint (config.json plus
the router biases, read with the pure-Python loader) and fails, naming the test and the cause,
where the exllamav3 package does not import. Nothing in this file is skipped.

    python -m pytest tests/test_dsv41_router_bias_.py
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file

rb = load_package_file("exllamav3/architecture/dsv41/router_bias.py", "_dsv41_router_bias")
ENV = rb.ENV_ROUTER_BIAS

ORIGINAL = {
    "layers.0.ffn.gate.bias": torch.tensor([10.001, 10.002, 9.9951]),
    "layers.1.ffn.gate.bias": torch.tensor([31.017, 31.031, 30.990]),
}
STORED = {key: value.half() for key, value in ORIGINAL.items()}


class STC:
    def __init__(self, stored):
        self.stored = stored
        self.added = []
        self.first_open_time, self.metrics = None, SimpleNamespace(bytes_loaded = 0)
        self.handles, self.released = {"model.safetensors": None}, []

    def has_tensor(self, key):
        return key in self.stored

    def get_tensor(self, key, device, **kwargs):
        # as the real collection: the first read starts the load timer, opens the file, counts bytes
        if self.first_open_time is None:
            self.first_open_time = 1.0
        self.handles["model.safetensors"] = "handle"
        self.metrics.bytes_loaded += self.stored[key].numel() * 2
        return self.stored[key].clone()

    def release_file(self, filename):
        self.released.append(filename)
        self.handles[filename] = None

    def add_tensor_files(self, path, warn_if_override = True):
        self.added.append((path, warn_if_override))


def config(stored = STORED):
    return SimpleNamespace(num_hidden_layers = 2, n_routed_experts = 3, stc = STC(stored))


def write(tmp_path, data, name = "bias.safetensors"):
    path = tmp_path / name
    save_file(data, str(path))
    return str(path)


def test_the_fp16_copy_loses_what_the_file_restores():
    # the premise: nearby FP32 biases share one FP16 value
    assert STORED["layers.0.ffn.gate.bias"][0] == STORED["layers.0.ffn.gate.bias"][1]
    assert ORIGINAL["layers.0.ffn.gate.bias"][0] != ORIGINAL["layers.0.ffn.gate.bias"][1]


def test_accepted_file_is_added_to_the_collection(tmp_path, capsys):
    cfg = config()
    path = write(tmp_path, ORIGINAL)
    assert rb.attach(cfg, path) == os.path.realpath(path)
    assert cfg.stc.added == [(os.path.realpath(path), False)]
    assert ENV in capsys.readouterr().out


@pytest.mark.parametrize("accepted", [True, False])
def test_the_check_leaves_no_trace_of_a_load(tmp_path, accepted):
    # Reading the stored copies must not start the model load's timer, count bytes, or keep the
    # files open, whether the file is accepted or refused
    cfg = config()
    data = dict(ORIGINAL) if accepted else {k: v + 1 for k, v in ORIGINAL.items()}
    path = write(tmp_path, data)
    if accepted:
        rb.attach(cfg, path)
    else:
        with pytest.raises(ValueError):
            rb.attach(cfg, path)
    assert cfg.stc.first_open_time is None
    assert cfg.stc.metrics.bytes_loaded == 0
    assert cfg.stc.released == ["model.safetensors"] and cfg.stc.handles["model.safetensors"] is None


@pytest.mark.parametrize("case", [
    "missing_layer", "extra_tensor", "empty", "fp16", "bf16", "wrong_shape", "other_values",
    "nan", "inf",
])
def test_invalid_files_are_refused_and_nothing_is_added(tmp_path, case):
    data = {key: value.clone() for key, value in ORIGINAL.items()}
    k0 = "layers.0.ffn.gate.bias"
    match case:
        case "missing_layer": del data["layers.1.ffn.gate.bias"]
        case "extra_tensor": data["layers.0.ffn.gate.weight"] = torch.ones(3)
        case "empty": data = {"unrelated": torch.ones(1)}
        case "fp16": data = STORED
        case "bf16": data = {key: value.bfloat16() for key, value in ORIGINAL.items()}
        case "wrong_shape": data[k0] = data[k0][:2].contiguous()
        case "other_values": data[k0] = data[k0] + 1
        case "nan": data[k0][1] = float("nan")
        case "inf": data[k0][1] = float("inf")
    cfg = config()
    with pytest.raises(ValueError, match = ENV):
        rb.attach(cfg, write(tmp_path, data))
    assert cfg.stc.added == []


def test_file_must_exist_and_be_safetensors(tmp_path):
    cfg = config()
    with pytest.raises(ValueError, match = "is not a file"):
        rb.attach(cfg, str(tmp_path / "absent.safetensors"))
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match = "cannot read"):
        rb.attach(cfg, str(bad))
    with pytest.raises(ValueError, match = "is not a file"):
        rb.attach(cfg, str(tmp_path))
    assert cfg.stc.added == []


def test_file_is_tied_to_its_checkpoint(tmp_path):
    # the same file against a checkpoint whose stored biases differ
    other = {key: (value + 0.5).half() for key, value in ORIGINAL.items()}
    cfg = config(other)
    with pytest.raises(ValueError, match = "does not round"):
        rb.attach(cfg, write(tmp_path, ORIGINAL))
    assert cfg.stc.added == []


def test_only_the_checkpoints_router_biases(tmp_path):
    # a checkpoint without a bias on layer 1 takes a file without it, and refuses one with it
    stored = {"layers.0.ffn.gate.bias": STORED["layers.0.ffn.gate.bias"]}
    only0 = {"layers.0.ffn.gate.bias": ORIGINAL["layers.0.ffn.gate.bias"]}
    cfg = config(stored)
    rb.attach(cfg, write(tmp_path, only0, "only0.safetensors"))
    assert len(cfg.stc.added) == 1
    cfg = config(stored)
    with pytest.raises(ValueError, match = "unexpected"):
        rb.attach(cfg, write(tmp_path, ORIGINAL))
    with pytest.raises(ValueError, match = "no router selection bias"):
        rb.attach(config({}), write(tmp_path, only0, "only0.safetensors"))


def test_config_reads_fp32_values_through_the_loader(tmp_path, monkeypatch):
    try:
        from exllamav3 import Config
    except Exception as e:
        # a failure, never a skip: this test needs the real package, and a broken import (the
        # compiled extension included) must not pass as "nothing to test"
        raise AssertionError(
            f"test_dsv41_router_bias_.py::test_config_reads_fp32_values_through_the_loader: importing "
            f"exllamav3 failed ({type(e).__name__}: {e}), so the sidecar cannot be read through a real "
            f"DeepseekV41Config and the tensor loader. This test needs the package with its compiled "
            f"extension: build it, or put a prebuilt exllamav3_ext on PYTHONPATH") from e
    # a small synthetic V4.1 checkpoint: config.json and four FP16 router biases
    import random
    random.seed(1)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg_dict = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64, "qk_rope_head_dim": 16, "vocab_size": 32,
            "q_lora_rank": 16, "o_lora_rank": 16, "o_groups": 1, "max_position_embeddings": 1024,
            "moe_intermediate_size": 32, "n_routed_experts": 8, "num_experts_per_tok": 2,
            "compress_ratios": [0, 0, 0, 0], "hc_mult": 4,
        },
    }
    (ckpt / "config.json").write_text(json.dumps(cfg_dict))
    original = {rb.bias_key(i): torch.tensor([random.uniform(9, 16) for _ in range(8)])
                for i in range(4)}
    save_file({k: v.half() for k, v in original.items()}, str(ckpt / "model.safetensors"))
    sidecar = write(tmp_path, original)

    monkeypatch.delenv(ENV, raising = False)
    plain = Config.from_directory(str(ckpt), load_method = "python")
    assert plain.dsv41_router_bias is None
    assert plain.stc.get_tensor(rb.bias_key(3), "cpu", allow_bf16 = True).dtype == torch.half

    monkeypatch.setenv(ENV, sidecar)
    cfg = Config.from_directory(str(ckpt), load_method = "python")
    assert cfg.dsv41_router_bias == os.path.realpath(sidecar)
    for key, value in original.items():
        got = cfg.stc.get_tensor(key, "cpu", allow_bf16 = True, no_defer = True)
        assert got.dtype == torch.float32 and torch.equal(got, value)

    monkeypatch.setenv(ENV, str(tmp_path / "absent.safetensors"))
    with pytest.raises(ValueError, match = ENV):
        Config.from_directory(str(ckpt), load_method = "python")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
