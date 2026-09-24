"""V4.1 rotations through ext.rope's angle-table mode, on the GPU, at positions up to 1M.

    PYTHONPATH=$PWD python3 tests/test_dsv41_rope_gpu_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

ext.rope evaluates __sinf / __cosf(position * inv_freq). Those intrinsics lose accuracy as the
argument grows, so at long context a rotated head is visibly off the rotation DeepSeek applies
(the fp32 angle t * freq, with accurate trig), while dsa_attn's de-rotation uses accurate Triton
trig: rotation and de-rotation stop cancelling. Every V4.1 rotation therefore hands ext.rope a
per-position table of angles formed in fp32 and reduced to [-pi, pi) in fp64
(modules/dsv41.py rope_angle_table), where the intrinsics are accurate.

Reference: a float64 rotation (GPT-J pairs) by the fp32 angle t * inv_freq, of the same fp16
input. Gate: max |error| <= 4e-3 on N(0, 1) heads, i.e. fp16 rounding of the result, at every
position. Checked shapes, each at start positions 0, 1000, 262,000 and 1,048,320 (the last row
is 1,048,575):

  1. the attention front: q (bsz 2, 64 heads x 512, rope on the trailing 64) and the shared kv
     head in one call, with the table repeated per batch row (the kernel strides it by batch);
     the nope part is untouched;
  2. pool entries at (first + i) * m, rate 2 and rate 1, through the entry table;
  3. the indexer query, a trailing-slice view of (32 heads x 128).

Plain ext.rope (an inv_freq vector and a start position) must FAIL the same gate at 1M, or
the check could not tell the two apart. The tables are the ones the model builds from the
checkpoint (DSV4Attention.load): the main table at theta 10,000 without scaling, which the
sliding layers rotate with, and the compressed table at theta 160,000 with the YaRN scaling
(factor 16 over 65,536); without a checkpoint DeepSeek-V4.1-Flash's values are used.
"""
import json, os, sys
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip

# The extension is imported once CUDA is known to be there (main, rotate), so that collecting
# this file without a GPU skips its test instead of failing to import

GATE = 4e-3
STARTS = (0, 1000, 262_000, 1_048_320)
N = 256


def rope_ref(x, inv, pos):
    """float64 GPT-J rotation of x (..., n, heads, rd) by the fp32 angles pos * inv."""
    ang = (pos.float().unsqueeze(-1) * inv.float().unsqueeze(0)).double()          # (n, rd/2)
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    xd = x.double()
    x0, x1 = xd[..., 0::2], xd[..., 1::2]
    out = torch.empty_like(xd)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def rotate(x, tab_or_inv, position = 0):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.util.rope import RopeStyle
    ext.rope(x, x, None, None, tab_or_inv, position, None, None,
             int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0)


def main(path = None):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.util.rope import RopeStyle, yarn_inv_freq
    from exllamav3.modules.dsv41 import rope_angle_table
    dev = torch.device("cuda:0")
    c = {"rope_theta": 10000.0, "compress_rope_theta": 160000.0, "qk_rope_head_dim": 64,
         "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                          "original_max_position_embeddings": 65536}}
    if path:
        t = json.load(open(os.path.join(path, "config.json")))
        t = t.get("text_config", t)
        c.update({k: t[k] for k in c if k in t})
    rd = c["qk_rope_head_dim"]
    inv_main = yarn_inv_freq(rd, c["rope_theta"], dev)
    inv_comp = yarn_inv_freq(rd, c["compress_rope_theta"], dev, rope_scaling = c["rope_scaling"])
    torch.manual_seed(0)
    worst = {}

    def gate(tag, err):
        worst[tag] = max(worst.get(tag, 0.0), err)
        assert err <= GATE, f"{tag}: max |error| {err:.2e} > {GATE}"

    for p0 in STARTS:
        pos = torch.arange(p0, p0 + N, device = dev)

        # 1. attention front: q and the kv head together, bsz 2, rope on the trailing rd
        for inv, which in ((inv_main, "main"), (inv_comp, "compress")):
            q = torch.randn(2, N, 64, 512, device = dev).half()
            kv = torch.randn(2, N, 1, 512, device = dev).half()
            q0, kv0 = q.clone(), kv.clone()
            tab = rope_angle_table(inv, pos, bsz = 2)
            assert tuple(tab.shape) == (2, N, rd // 2) and tab.is_contiguous()
            ext.rope(q, q, kv, kv, tab, 0, None, None, int(RopeStyle.GPTJ), 1.0, None, None,
                     1e-20, 0.0, 0.0, 0, 1, 512 - rd)
            assert torch.equal(q[..., :512 - rd], q0[..., :512 - rd]) and \
                   torch.equal(kv[..., :512 - rd], kv0[..., :512 - rd]), "the nope part moved"
            for b in range(2):
                e_q = (q[b, ..., -rd:].double() - rope_ref(q0[b, ..., -rd:], inv, pos)).abs().max().item()
                e_k = (kv[b, ..., -rd:].double() - rope_ref(kv0[b, ..., -rd:], inv, pos)).abs().max().item()
                gate(f"q/kv {which}", max(e_q, e_k))

        # 2. pool entries at (first + i) * m
        for m in (2, 1):
            first = p0 // m
            epos = (first + torch.arange(N, device = dev)) * m
            rot = torch.randn(1, N, 1, rd, device = dev).half()
            r0 = rot.clone()
            rotate(rot, rope_angle_table(inv_comp, epos))
            gate(f"entries m={m}", (rot[0].double() - rope_ref(r0[0], inv_comp, epos)).abs().max().item())

        # 3. indexer query: the trailing rd of 32 x 128 heads, a strided view
        qi = torch.randn(1, N, 32, 128, device = dev).half()
        qi0 = qi.clone()
        rotate(qi[..., -rd:], rope_angle_table(inv_comp, pos))
        assert torch.equal(qi[..., :-rd], qi0[..., :-rd])
        gate("index q", (qi[0, ..., -rd:].double() - rope_ref(qi0[0, ..., -rd:], inv_comp, pos)).abs().max().item())

    # plain ext.rope at 1M: __sinf / __cosf of the raw angle
    p0 = STARTS[-1]
    pos = torch.arange(p0, p0 + N, device = dev)
    q = torch.randn(1, N, 1, rd, device = dev).half()
    q0 = q.clone()
    rotate(q, inv_comp, p0)
    e_plain = (q[0].double() - rope_ref(q0[0], inv_comp, pos)).abs().max().item()
    assert e_plain > GATE, f"plain ext.rope passes the gate at 1M ({e_plain:.2e}): the check is blind"

    print(f"  table path, max |error| vs float64 rotation of the fp32 angle "
          f"({N}-row windows starting at {STARTS}; rate-2 entries use doubled spacing): " +
          ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
    print(f"  plain ext.rope at position {p0:,}: max |error| {e_plain:.2e} (> {GATE})")
    print("  OK  V4.1 rotations meet the explicit tolerance in sampled windows through 1M")


def test_rope_gpu():
    """pytest entry point: the same checks, with the tables of DSV41_MODEL_DIR if set."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("V4.1 rope on the GPU: no CUDA device")
    main(model_dir())


if __name__ == "__main__":
    if not torch.cuda.is_available():
        skip("V4.1 rope on the GPU", "no CUDA device")
        sys.exit(0)
    main(sys.argv[1] if len(sys.argv) > 1 else model_dir())
