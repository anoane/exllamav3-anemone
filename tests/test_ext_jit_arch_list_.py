"""
JIT build of the extension (ext.py, no precompiled exllamav3_ext) with no visible CUDA device: when
TORCH_CUDA_ARCH_LIST is unset (or "native", on torch versions that accept it) there is nothing to
derive the target architectures from, and torch's cpp_extension fails inside its own arch detection
with an IndexError. That error is now reported as a RuntimeError naming TORCH_CUDA_ARCH_LIST. With
the variable set, or a CUDA flag that already names an arch (torch then skips its detection), the
build goes ahead as before. No GPU and no compiler run: load() is replaced by torch's arch-flag step
alone.
"""
import os, sys, subprocess
import pytest
import torch
from torch.torch_version import TorchVersion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.version.cuda, reason = "CUDA build of torch")

# Hide any installed extension so ext.py takes the JIT path, report no visible device, and stand in
# for torch's load() with the step of its build that consumes the architecture list
_PROBE = r"""
import importlib.util, torch, torch.utils.cpp_extension as cpp
_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: \
    None if name == "exllamav3_ext" else _find_spec(name, *args, **kwargs)
torch.cuda.device_count = lambda: 0
def load(**kwargs):
    cpp._get_cuda_arch_flags(kwargs["extra_cuda_cflags"])
    print("reached load")
    raise SystemExit(0)
cpp.load = load
import exllamav3.ext
"""


def _jit_import(arch_list, cuda_host_cxx = None):
    env = {k: v for k, v in os.environ.items() if k not in ("TORCH_CUDA_ARCH_LIST", "CUDAHOSTCXX")}
    env.update(PYTHONPATH = ROOT, CUDA_VISIBLE_DEVICES = "")
    if arch_list is not None:
        env["TORCH_CUDA_ARCH_LIST"] = arch_list
    if cuda_host_cxx is not None:
        env["CUDAHOSTCXX"] = cuda_host_cxx
    return subprocess.run([sys.executable, "-B", "-c", _PROBE], capture_output = True, text = True,
                          env = env, timeout = 600)


@pytest.mark.parametrize("arch_list", [
    None,
    "",
    pytest.param("native", marks = pytest.mark.skipif(
        TorchVersion(torch.__version__) < "2.9", reason = "TORCH_CUDA_ARCH_LIST=native needs torch 2.9"
    )),
], ids = ["unset", "empty", "native"])
def test_no_visible_device_asks_for_arch_list(arch_list):
    r = _jit_import(arch_list)
    assert r.returncode != 0
    last = r.stderr.strip().splitlines()[-1]
    assert last.startswith("RuntimeError:") and "TORCH_CUDA_ARCH_LIST" in last, r.stderr
    assert "reached load" not in r.stdout


def test_arch_list_set_reaches_load_without_visible_device():
    r = _jit_import("8.6;8.9+PTX")
    assert r.returncode == 0, r.stderr
    assert "reached load" in r.stdout


@pytest.mark.skipif(sys.platform == "win32", reason = "ext.py ignores CUDAHOSTCXX on Windows")
def test_arch_named_in_cuda_flags_reaches_load_without_visible_device():
    # "-ccbin <path>" with "arch" in the path makes torch skip its detection and leave the target to
    # nvcc's default, so the build must go ahead as before
    r = _jit_import(None, "/usr/bin/aarch64-linux-gnu-g++")
    assert r.returncode == 0, r.stderr
    assert "reached load" in r.stdout
