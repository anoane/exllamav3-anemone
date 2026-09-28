"""
EXL3_MOE_RECON_DET (batched reconstruct accumulation outside the slot scratch, i.e. the streamed
CPU-offload tier) resolves as doc/env_vars.md documents it: its own value when set, else the value
of EXL3_MOE_FUSED_DET when that is set explicitly, else 0 (atomic), although EXL3_MOE_FUSED_DET
itself defaults to 1. Both are read at import, so every case runs in a fresh interpreter. No GPU
needed.
"""
import os, sys, subprocess
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
import pytest

_code = (
    "from exllamav3.modules import moe_batch_recon as r, block_sparse_mlp as b; "
    "print(r.DETERMINISTIC, b.FUSED_DET)"
)


@pytest.mark.parametrize("fused, recon, expect", [
    (None, None, "False True"),    # neither set: streamed groups atomic, slot mode on
    ("1", None, "True True"),
    ("0", None, "False False"),
    ("0", "1", "True False"),
    (None, "1", "True True"),
])
def test_recon_det_resolution(fused, recon, expect):
    env = dict(os.environ)
    for k, v in (("EXL3_MOE_FUSED_DET", fused), ("EXL3_MOE_RECON_DET", recon)):
        env.pop(k, None)
        if v is not None: env[k] = v
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_root, env.get("PYTHONPATH")) if p)
    out = subprocess.run([sys.executable, "-B", "-c", _code], env = env, cwd = _root,
                         capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == expect, (fused, recon, out.stdout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
