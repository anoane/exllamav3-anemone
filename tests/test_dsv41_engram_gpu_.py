# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Transcribes parts of DeepSeek's reference inference code as a test oracle; see NOTICE.

"""DSV41Engram on a GPU, real weights and real table rows, against a float64 host reference.

Run on each card:  PYTHONPATH=$PWD python3 tests/test_dsv41_engram_gpu_.py [checkpoint-dir] [cuda:N]
(checkpoint-dir defaults to $DSV41_MODEL_DIR)

  1. One-shot forward of both engram layers over real text vs a float64 transcription of
     DeepSeek's Engram.forward. The reference gets its rows by a DIFFERENT path (the
     loader's Python DiskTensorHandle.read_rows, one read per hash id, no dedup), and its
     wkv is the exl3 weight reconstructed to a dense matrix (LinearEXL3.get_weight_tensor)
     and applied in float64; the gate uses the bf16-rounded q/k product. Gated on the
     injected term (out - x), which is what the engram contributes: rel-L2 <= 2e-3.
  2. The generation path on the device: random prefill chunks + decode through the real
     job state (DSV41State on a stand-in Cache) feed wkv exactly the one-shot rows
     (bitwise) and give the one-shot streams to within the exl3 GEMM's shape-dependent
     rounding.
  3. prefetch() then forward() -- the staged set is taken (a hit) and the result is
     bitwise the inline one; pinned staging sets are reused across forwards.
  4. The autosplit measuring pass (params["autosplit_measure"]) reads nothing from disk.
One engram layer at a time: ~0.2 GiB of VRAM."""
import os, random, sys, types
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip


class FakeCache:
    def __init__(self):
        self.layers = {}
        self.model = types.SimpleNamespace(loaded_tp = False)
    def get_recurrent_layer(self, key): return self.layers[key]
    def get_all_recurrent_layers(self): return self.layers
    def release_state(self, s): pass


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def reference64(eg, ids, x, dev):
    """DeepSeek's Engram.forward in float64 on the host, rows via read_rows."""
    from exllamav3.architecture.dsv41.engram_torch import UNK
    B, L = ids.shape
    H, D = eg.hc_mult, eg.hidden_size
    hid = eg.hasher.hash(eg.hasher.compress(ids), torch.full((B, eg.ctx), UNK))[:, :, eg.table_index]
    w = eg.w_handle.read_rows(hid.reshape(-1)).view(torch.uint8)
    sc = eg.s_handle.read_rows(hid.reshape(-1)).view(torch.uint8)
    vals = w.view(torch.float8_e4m3fn).double().unflatten(-1, (-1, 32))
    scl = torch.pow(2.0, sc.double() - 127)
    rows = (vals * scl.unsqueeze(-1)).flatten(-2)                           # [B*L*24, 256] f64
    W = eg.wkv.inner.get_weight_tensor().double().cpu()                     # [6144, 25600]
    bias = eg.wkv.inner.get_bias_tensor()
    kv = rows.view(B * L, -1) @ W
    if bias is not None:
        kv = kv + bias.double().cpu()
    key = kv[:, :H * D].view(B, L, H, D)
    value = kv[:, H * D:].view(B, L, 1, D)
    qk = (eg.q_weight.data.to(torch.bfloat16).double() * eg.k_weight.data.to(torch.bfloat16).double()).cpu()
    h = x.double().cpu()
    eps = eg.config.rms_norm_eps
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * qk * key).sum(-1) * rstd * D ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return h + gate.unsqueeze(-1) * value, gate


def main(checkpoint_dir, dev):
    from exllamav3 import Config, Model
    from exllamav3.cache.dsv41 import DSV41State
    from exllamav3.cache.recurrent_util import advance_recurrent_states
    from exllamav3.modules import dsv41_engram as dm
    from tokenizers import Tokenizer

    dev = torch.device(dev)
    cfg = Config.from_directory(checkpoint_dir)
    model = Model.from_config(cfg)
    tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    text = open(os.path.join(checkpoint_dir, "README.md")).read()
    all_ids = torch.tensor(tok.encode(text, add_special_tokens = False).ids, dtype = torch.long)
    L = 300
    ids = all_ids[None, :L]
    H, D = cfg.hc_mult, cfg.hidden_size
    x = torch.randn(1, L, H, D, generator = torch.Generator().manual_seed(0)) * 0.7
    x_d = x.to(dev)
    print(f"  --  {torch.cuda.get_device_name(dev)} ({dev}), {L} README tokens")

    for idx in sorted(model.engram_modules):
        eg = model.engram_modules[idx]
        cache = FakeCache()
        st = eg.layer_state_cls(eg, 4, 8, id(cache))
        eg.recurrent_layers.append(st)
        cache.layers[eg.state_key()] = st
        eg.load(dev)
        assert st.device is not None, "load() did not allocate the engram state"

        # ---- 1. one shot vs float64 reference ----
        got = eg.forward(x_d.clone(), {"input_ids": ids}).cpu()
        want, gate = reference64(eg, ids, x, dev)
        e_out = rel(got, want)
        e_inj = rel(got - x, want - x.double())
        inj = ((want - x.double()).norm() / x.double().norm()).item()
        print(f"  layer {idx}: injected term vs f64 reference rel-L2 {e_inj:.2e} (streams {e_out:.2e}); "
              f"engram moves the streams by rel {inj:.3f}, gate mean {gate.mean():.3f}")
        assert e_inj <= 2e-3, f"layer {idx}: engram differs from the float64 reference by {e_inj:.2e}"
        assert inj > 1e-2, "engram injected almost nothing -- the check would not see a skipped engram"

        # ---- 2. generation path: chunks + decode through the state ----
        seen = []
        orig_fwd = eg.wkv.forward
        def rec(t, params, out_dtype = None, _f = orig_fwd):
            seen.append(t.clone())
            return _f(t, params, out_dtype = out_dtype)
        eg.wkv.forward = rec
        y1 = eg.forward(x_d.clone(), {"input_ids": ids})
        emb1 = seen.pop()
        s = DSV41State(cache, 0, 0)
        rnd = random.Random(idx)
        pos = 0; ys = []; n = 0
        small = torch.zeros(L, dtype = torch.bool)      # positions run with <= 2 rows (decode)
        while pos < L:
            n_tok = min(rnd.randint(3, 90), L - 20 - pos) if pos < L - 20 else 1   # then 20 decodes
            params = {"input_ids": ids[:, pos:pos + n_tok], "recurrent_states": [s]}
            ys.append(eg.forward(x_d[:, pos:pos + n_tok].clone(), params))
            advance_recurrent_states(params["input_ids"], params, None)
            small[pos:pos + n_tok] = n_tok <= 2
            pos += n_tok; n += 1
        emb2 = torch.cat(seen, dim = 1); seen.clear()
        y2 = torch.cat(ys, dim = 1).cpu()
        assert torch.equal(emb1, emb2), f"layer {idx}: chunked wkv input differs from one shot"
        # Each path against the float64 reference. Decode steps run wkv with 1 row, where
        # the exl3 kernel is measurably less precise than at >= 3 rows (the same holds for
        # every exl3 linear in this checkpoint); gate each on its own
        inj2, inj64 = (y2 - x)[0], (want - x.double())[0]
        e_pre = rel(inj2[~small], inj64[~small])
        e_dec = rel(inj2[small], inj64[small])
        print(f"  layer {idx}: {n} chunks/decodes through the state: wkv input bitwise == one shot; "
              f"injected term vs f64: chunk positions {e_pre:.2e}, 1-row decode positions {e_dec:.2e}")
        assert e_pre <= 2e-3, f"layer {idx}: chunked positions differ from the reference by {e_pre:.2e}"
        # (1-row exl3 GEMV: 7.0e-3 .. 7.8e-3 vs float64 on random input for this checkpoint's
        # K=5 linears, measured on an sm_120 GPU, vs 5.0e-4 at >= 3 rows)
        assert e_dec <= 2e-2, f"layer {idx}: decode positions differ from the reference by {e_dec:.2e}"

        # ---- 3. prefetch: hit, bitwise the inline result, pinned sets reused ----
        # prefetch() itself is under test, so it is switched on whatever
        # EXL3_DSV41_ENGRAM_PREFETCH the runner set (read at call time)
        prefetch_was = dm.PREFETCH_ENABLED
        dm.PREFETCH_ENABLED = True
        try:
            eg.prefetch_stats.update(hit = 0, miss = 0, retired = 0)
            for rep in range(3):
                p = {"input_ids": ids}
                eg.prefetch(ids, p)
                assert len(eg._pending) == 1, f"prefetch queued {len(eg._pending)} entries"
                yp = eg.forward(x_d.clone(), p)
                seen.clear()
                assert torch.equal(yp, y1), f"layer {idx}: prefetched forward differs from inline"
            assert eg.prefetch_stats["hit"] == 3 and eg.prefetch_stats["miss"] == 0, eg.prefetch_stats
            pins = [pp for pp in eg._pins]
            assert 1 <= len(pins) <= dm.MAX_PIN_SETS and all(pp.pinned and pp.inv.is_pinned() for pp in pins)
            eg.prefetch(all_ids[None, :64], {"input_ids": all_ids[None, :64]})    # decode-sized: declined
            assert not eg._pending
        finally:
            dm.PREFETCH_ENABLED = prefetch_was
        print(f"  layer {idx}: prefetch hit x3, bitwise == inline; {len(pins)} pinned staging "
              f"set(s) reused; a 64-token chunk is not prefetched")

        # ---- 4. autosplit measuring pass: no disk reads ----
        calls = []
        real_ext = dm.ext
        class CountExt:
            def __getattr__(self, name):
                f = getattr(real_ext, name)
                if name == "ngram_gather_cpu":
                    def g(*a, _f = f):
                        calls.append(1); return _f(*a)
                    return g
                return f
        dm.ext = CountExt()
        try:
            eg.forward(x_d.clone(), {"input_ids": torch.zeros_like(ids), "autosplit_measure": True})
            n_measure = len(calls)
            eg.forward(x_d.clone(), {"input_ids": ids})
            n_real = len(calls) - n_measure
        finally:
            dm.ext = real_ext
        assert n_measure == 0 and n_real == 2, (n_measure, n_real)
        print(f"  layer {idx}: autosplit measuring forward made {n_measure} gather calls (a real one: {n_real})")

        eg.wkv.forward = orig_fwd
        eg.unload()
        assert st.device is None and not eg._pins
        torch.cuda.synchronize(dev)

    print(f"  OK  engram on {dev}: == float64 reference, generation path == one shot, prefetch exact")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not torch.cuda.is_available():
        skip("engram on the GPU", "no CUDA device")
    elif not d:
        skip("engram on the GPU", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
    else:
        main(d, sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
