"""
CUDAHOSTCXX in the JIT extension build (ext.py): passed to nvcc as -ccbin off Windows; on Windows it
is left out as before (nvcc uses the same cl.exe as the C++ sources) and a notice says so. Runs
ext.py in a fresh interpreter with torch's load() stubbed out, so nothing is compiled and no GPU is
needed.
"""
import ast, os, sys, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loads exllamav3/ext.py without the package __init__ (which would import the extension), with no
# prebuilt exllamav3_ext visible, os.name forced for ext.py's platform checks and load() recording
# its arguments. The helper modules are imported before os.name is changed
_HARNESS = r"""
import importlib.util, os, subprocess, sys, types
import torch.utils.cpp_extension as cpp_extension
root, os_name = sys.argv[1], sys.argv[2]
package = types.ModuleType("exllamav3")
package.__path__ = [os.path.join(root, "exllamav3")]
sys.modules["exllamav3"] = package
import exllamav3.util.arch_list, exllamav3.util.cuda_flags
calls = []
cpp_extension.load = lambda **kwargs: calls.append(kwargs)
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a: None if name == "exllamav3_ext" else find_spec(name, *a)
subprocess.check_output = lambda *a, **k: b""
os.name = os_name
import exllamav3.ext
assert len(calls) == 1
print(repr(calls[0]["extra_cuda_cflags"]))
"""


def _jit_flags(os_name, **env):
    env = {k: v for k, v in os.environ.items() if k != "CUDAHOSTCXX"} | env
    # Keep nvcc (the --compress-mode probe) and the GPU arch query out of the subprocess
    env |= {"EXLLAMA_EXT_COMPRESS": "0", "TORCH_CUDA_ARCH_LIST": "8.0"}
    r = subprocess.run([sys.executable, "-B", "-c", _HARNESS, ROOT, os_name],
                       capture_output = True, text = True, env = env, timeout = 300)
    assert r.returncode == 0, r.stderr
    return ast.literal_eval(r.stdout.strip().splitlines()[-1]), r.stderr


@pytest.mark.parametrize("os_name", ["posix", "nt"])
def test_unset_adds_no_ccbin_and_no_notice(os_name):
    flags, err = _jit_flags(os_name)
    assert "-ccbin" not in flags
    assert "CUDAHOSTCXX" not in err


def test_posix_passes_cudahostcxx_as_ccbin():
    flags, err = _jit_flags("posix", CUDAHOSTCXX = "/usr/bin/g++-13")
    assert "-DWIN32_LEAN_AND_MEAN" not in flags
    i = flags.index("-ccbin")
    assert flags[i + 1] == "/usr/bin/g++-13"
    assert flags.count("-ccbin") == 1
    assert "CUDAHOSTCXX" not in err


def test_windows_leaves_ccbin_out_and_says_so_once():
    flags, err = _jit_flags("nt", CUDAHOSTCXX = r"C:\VS\bin\Hostx64\x64\cl.exe")
    assert "-DWIN32_LEAN_AND_MEAN" in flags  # ext.py took its Windows branch
    assert "-ccbin" not in flags
    assert err.count("CUDAHOSTCXX is not used by the JIT build on Windows") == 1
