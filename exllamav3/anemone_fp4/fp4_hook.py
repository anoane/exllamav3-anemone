"""P9b/P12 hook — FP4 prefill moe + dense paths. Loaded lazily by block_sparse_mlp / exl3.

P12: fully sync-free moe path — routing tables built on GPU (k_build_tables) directly from
the expert_count device tensor; no tolist, no numpy, no pageable uploads. Chunk size is
MT_CAP*16 by construction (fixes the v2b CHUNK=64/MT_CAP=2 silent row-drop bug).
"""
import os

import torch

_ext = None
_buf = {}


def _get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="exl3_fp4_moe",
            sources=[os.path.join(here, "fp4_moe.cu")],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_120a,code=sm_120a"],
            # the engine tree this file ships inside, so a checkout resolves it
            extra_include_paths=[os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3_ext")],
            verbose=False,
        )
    return _ext


def _scratch(device, rows, H, I, n_work_max=0):
    """Phase-aliased pool: [a_gate | a_up | interm_g | interm_u | sfa_gate | sfa_up | sfa_down].
    a_down overlays the dead a_gate region (phases 3+); d_out overlays a_up..interm_u (4+).
    sfa buffers are live across phases and never overlaid. Plus GPU table scratch (P12)."""
    key = device
    b = _buf.get(key)
    if b is None or b["rows"] < rows or b.get("n_work_max", 0) < n_work_max:
        rows_a = rows            # grow-only; no floor (a 12288 floor doubled the pool at 1024-token chunks)
        nwm = max(n_work_max, 2048)
        s_ag, s_au, s_ig, s_iu = H // 2, H // 2, 2 * I, 2 * I
        s_sg, s_su, s_sd = H // 32, H // 32, I // 32
        assert s_au + s_ig + s_iu >= 2 * H, "d_out overlay does not fit"
        per_row = s_ag + s_au + s_ig + s_iu + s_sg + s_su + s_sd
        pool = torch.empty(rows_a * per_row, dtype=torch.uint8, device=device)
        o = [0]
        for s in (s_ag, s_au, s_ig, s_iu, s_sg, s_su):
            o.append(o[-1] + rows_a * s)
        b = {
            "rows": rows_a,
            "n_work_max": nwm,
            "pool": pool,
            "a_gate": pool[o[0]:o[1]].view(rows_a, s_ag),
            "a_up": pool[o[1]:o[2]].view(rows_a, s_au),
            "interm_g": pool[o[2]:o[3]].view(torch.half).view(rows_a, I),
            "interm_u": pool[o[3]:o[4]].view(torch.half).view(rows_a, I),
            "sfa_gate": pool[o[4]:o[5]].view(s_sg, rows_a),    # transposed [k32][row]
            "sfa_up": pool[o[5]:o[6]].view(s_su, rows_a),
            "sfa_down": pool[o[6]:o[6] + rows_a * s_sd].view(s_sd, rows_a),
            "a_down": pool[: rows_a * (I // 2)].view(rows_a, I // 2),
            "d_out": pool[o[1] : o[1] + rows_a * 2 * H].view(torch.half).view(rows_a, H),
            "row_expert": torch.empty(rows_a, dtype=torch.int32, device=device),
            "wt": torch.empty(3, nwm, dtype=torch.int32, device=device),
            "n_work": torch.zeros(1, dtype=torch.int32, device=device),
        }
        _buf[key] = b
    return b


def run(mod, y, final_hidden_states, expert_count, token_sorted, weight_sorted, num_ex):
    """Sync-free FP4 moe path (P12): expert_count is the (E+1,) int64 DEVICE tensor.
    Returns True if handled (caller skips the tolist, fused path and per-expert loop)."""
    if not mod.gated:
        return False
    if getattr(mod, "num_local_experts", None) not in (None, mod.num_experts):
        return False   # TP expert sharding adds sentinel rows; decline
    ok = getattr(mod, "_fp4_ok", None)
    if ok is None:
        ks = torch.cat([mod.multi_gate.ptrs_K, mod.multi_up.ptrs_K, mod.multi_down.ptrs_K])
        ok = bool(((ks >= 2) & (ks <= 4)).all().item())
        mod._fp4_ok = ok
    if not ok:
        return False

    rows = int(token_sorted.size(0))          # no sentinels on single device
    if rows == 0:
        return True
    dev = y.device
    H = mod.hidden_size
    I = mod.multi_up.out_features
    n_work_max = num_ex + rows // 32 + 8
    b = _scratch(dev, rows, H, I, n_work_max)

    _get_ext().fp4_moe(
        y, final_hidden_states,
        token_sorted, weight_sorted,
        expert_count,
        b["row_expert"], b["wt"][0][:n_work_max], b["wt"][1][:n_work_max], b["wt"][2][:n_work_max],
        b["n_work"],
        mod.multi_gate.ptrs_trellis, mod.multi_gate.ptrs_suh, mod.multi_gate.ptrs_svh,
        mod.multi_up.ptrs_trellis, mod.multi_up.ptrs_suh, mod.multi_up.ptrs_svh,
        mod.multi_down.ptrs_trellis, mod.multi_down.ptrs_suh, mod.multi_down.ptrs_svh,
        mod.multi_gate.ptrs_K, mod.multi_up.ptrs_K, mod.multi_down.ptrs_K,
        b["a_gate"], b["a_up"], b["a_down"],
        b["sfa_gate"], b["sfa_up"], b["sfa_down"],
        b["interm_g"], b["interm_u"], b["d_out"],
        float(mod.act_limit), int(mod.activation_fn_idx),
    )
    return True


# ---------------------------------------------------------------- dense (P11, env-off)
_dbuf = {}
_w4cache = {}   # P21: persistent per-tensor e2m1 cache (EXL3_FP4_DENSE_CACHE=1, ~1.5GB all-dense)
_w4seen = set() # populate cache only on the SECOND encounter, so the autosplit measuring
                # pass (each tensor once) never allocates it


def _dense_scratch(device, rows, in_f, out_f):
    b = _dbuf.get(device)
    need_rows = max(rows, 2048)
    if b is None or b["rows"] < need_rows or b["in"] < in_f or b["out"] < out_f:
        in_a = max(in_f, b["in"] if b else 0, 4096)
        out_a = max(out_f, b["out"] if b else 0, 4096)
        b = {
            "rows": need_rows, "in": in_a, "out": out_a,
            "a_x": torch.empty(need_rows, in_a // 2, dtype=torch.uint8, device=device),
            "sfa": torch.empty(in_a // 32, need_rows, dtype=torch.uint8, device=device),
            "c_tmp": torch.empty(need_rows, out_a, dtype=torch.half, device=device),
            "w4": torch.empty(out_a, in_a // 2, dtype=torch.uint8, device=device),
        }
        _dbuf[device] = b
    return b


def dense(lin, x, rows):
    """FP4 dense GEMM for one exl3 Linear at prefill; returns y (rows, out) half or None."""
    if getattr(lin, "mcg", False) or not getattr(lin, "mul1", False):
        return None
    K = int(lin.K)
    if K < 2 or K > 4 or lin.bias is not None:
        return None
    in_f, out_f = lin.in_features, lin.out_features
    if in_f % 128 or out_f % 128:
        return None
    b = _dense_scratch(x.device, rows, in_f, out_f)
    y = torch.empty((rows, out_f), dtype=torch.half, device=x.device)
    nr = b["rows"]
    a_x = b["a_x"].view(-1)[: nr * (in_f // 2)].view(nr, in_f // 2)
    c_tmp = b["c_tmp"].view(-1)[: nr * out_f].view(nr, out_f)
    sfa = b["sfa"][: in_f // 32]
    skip_dq = 0
    w4 = None
    if os.environ.get("EXL3_FP4_DENSE_CACHE") == "1":
        key = lin.trellis.data_ptr()
        w4 = _w4cache.get(key)
        if w4 is not None:
            skip_dq = 1
        elif key in _w4seen:
            w4 = torch.empty(out_f, in_f // 2, dtype=torch.uint8, device=x.device)
            _w4cache[key] = w4
        else:
            _w4seen.add(key)
    if w4 is None:
        w4 = b["w4"].view(-1)[: out_f * (in_f // 2)].view(out_f, in_f // 2)
    _get_ext().fp4_dense(x, y, lin.trellis, lin.suh, lin.svh, a_x, sfa, c_tmp, w4, K, skip_dq)
    return y
