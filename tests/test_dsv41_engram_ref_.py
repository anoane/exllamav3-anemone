# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# Transcribes parts of DeepSeek's reference inference code as a test oracle; see NOTICE.

"""The numpy engram reference stages (tests/dsv41_ref: engram_hash / engram_gate /
engram_forward) vs DeepSeek's reference, and the module vs them.

    PYTHONPATH=$PWD python3 tests/test_dsv41_engram_ref_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

What a wrong stage would miss: trailing wkv blocks taken as keys, no value injection,
positions 0-2 masked, eps 1e-6, clamp 0, the raw pad id. Checked here, CPU only (step 1
needs DSV41_DEEPSEEK_REF and is skipped without it):
  1. engram_hash_reference == DeepSeek NgramHashState one-shot, bitwise, both layers,
     on real text through the real token map -- with and without a dead span.
  2. engram_gate_reference == a float64 transcription of DeepSeek's Engram.forward gate
     (copysign / where-sign, clamp 1e-6, eps 1e-20), incl. dot == 0 rows, all-zero
     hidden rows (finite, no NaN) and masked rows (gate 0).
  3. EngramForward (numpy, rows from EngramTable's pread reader) == DSV41Engram.forward
     (torch, rows from the C++ gather + fp16 dequant) on the same ids, streams, q/k and a
     stand-in wkv; the forward's own staging (hash, dedup, C++ gather) of 700 real tokens
     gives the Python reader's bytes for the reference hash ids, and the dequant is exact
     (fp16 rows == fp32 rows); the measuring forward stages one row per n-gram.
  4. load_gate reads the bf16 copy of q_weight / k_weight exactly, whichever copy (fp32
     converted, bf16 original) the collection kept, and its product equals DeepSeek's
     q_weight.float() * k_weight.float() in every element; the fp32 copies are reported
     against it.
  5. The shared prefetch worker stops with the last engram module (no checkpoint needed;
     also a pytest test).
  6. The same read on synthetic shards, with values that fp16 staging would change (no
     checkpoint needed; also a pytest test)."""
import os, sys, types
import numpy as np
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip
from dsv41_ref import deepseek


def gate64(hidden, key, q, k, eps, mask = None):
    """DeepSeek Engram.forward's gate, float64."""
    h = hidden.astype(np.float64); key = key.astype(np.float64)
    D = h.shape[-1]
    w = q.astype(np.float64) * k.astype(np.float64)
    rstd = 1 / np.sqrt((h * h).mean(-1) + eps) / np.sqrt((key * key).mean(-1) + eps)
    dot = (h * w * key).sum(-1) * rstd * D ** -0.5
    g = 1 / (1 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)))
    if mask is not None:
        g = np.where(mask[:, None], g, 0.0)
    return g


def check_hash_vs_deepseek(cfg, tok, ids, layout, token_map):
    """engram_hash_reference == DeepSeek's NgramHashState one-shot, with and without a dead span."""
    from dsv41_ref.engram_hash import engram_hash_reference
    T = ids.size
    ref = deepseek.load_engram()
    args = types.SimpleNamespace(
        engram_layer_ids = cfg.engram_layer_ids, engram_max_ngram_size = cfg.engram_max_ngram_size,
        engram_n_heads = cfg.engram_n_heads, engram_head_dim = cfg.engram_head_dim,
        engram_vocab_size = cfg.engram_vocab_size, engram_num_embeddings = cfg.engram_num_embeddings,
        engram_compressed_vocab_size = cfg.engram_compressed_vocab_size,
        engram_pad_id = cfg.engram_pad_token_id, max_batch_size = 1, max_seq_len = T)
    class Shim:
        def __init__(self, b): self.backend_tokenizer = b
        def __len__(self): return self.backend_tokenizer.get_vocab_size(with_added_tokens = True)
    rs = ref.NgramHashState(args, ref.EngramLayout.from_args(args), Shim(tok))
    dead = np.zeros(T, dtype = bool); dead[200:215] = True
    for dm in (None, dead):
        want = rs(torch.from_numpy(ids)[None], 0,
                  None if dm is None else torch.from_numpy(~dm)[None])[0].numpy()
        got = engram_hash_reference(ids, layout, token_map, dead_mask = dm)
        assert np.array_equal(got, want), \
            f"hash reference differs from DeepSeek at {int((got != want).any(-1).any(-1).sum())} positions"
    print(f"  OK  engram_hash_reference == DeepSeek NgramHashState on {T} real tokens, both layers, "
          f"with and without a dead span")


def main(checkpoint_dir):
    from exllamav3 import Config, Model
    from dsv41_ref.engram_hash import engram_hash_reference
    from dsv41_ref.engram_gate import engram_gate_reference
    from dsv41_ref.engram_forward import EngramForward
    from dsv41_ref.engram_tables import EngramTable
    from exllamav3.architecture.dsv41.engram_torch import dequant_rows
    from tokenizers import Tokenizer

    cfg = Config.from_directory(checkpoint_dir)
    model = Model.from_config(cfg)
    egs = [model.engram_modules[i] for i in sorted(model.engram_modules)]
    for eg in egs:
        eg.open_table()
    layout = egs[0].layout
    token_map = egs[0].hasher.token_map.numpy()
    tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    ids = np.array(tok.encode(open(os.path.join(checkpoint_dir, "README.md")).read(),
                              add_special_tokens = False).ids[:700], dtype = np.int64)
    T = ids.size

    # ---- 1. hash reference vs DeepSeek ----
    if deepseek.ref_dir() is None:
        skip("engram_hash_reference vs DeepSeek NgramHashState", "DSV41_DEEPSEEK_REF not set")
    else:
        check_hash_vs_deepseek(cfg, tok, ids, layout, token_map)

    # ---- 2. gate reference, edge rows included ----
    rng = np.random.default_rng(0)
    H, D = cfg.hc_mult, cfg.hidden_size
    eps = cfg.rms_norm_eps
    assert eps == 1e-20, eps
    n = 64
    hidden = rng.standard_normal((n, H, D)).astype(np.float32)
    key = rng.standard_normal((n, H, D)).astype(np.float32)
    q = rng.standard_normal((H, D)).astype(np.float32)
    k = rng.standard_normal((H, D)).astype(np.float32)
    hidden[3] = 0.0                                   # all-zero stream stack: dot = 0
    key[5, 1] = 0.0                                   # all-zero key for one stream
    hidden[7, 2, :] = 0.0; hidden[7, 2, 0] = 1.0      # dot = h0 * w0 * key0 ...
    key[7, 2, 0] = 0.0                                # ... = 0 exactly
    mask = np.ones(n, dtype = bool); mask[10:14] = False
    got = engram_gate_reference(hidden, key, q, k, eps = eps, token_mask = mask)
    want = gate64(hidden, key, q, k, eps, mask)
    assert np.isfinite(got).all(), "gate reference produced NaN/inf"
    e = np.abs(got - want).max()
    assert e < 1e-5, f"gate reference differs from DeepSeek's formula by {e:.2e}"
    assert (got[10:14] == 0).all() and abs(got[3] - 1 / (1 + np.exp(-1e-3))).max() < 1e-6
    print(f"  OK  engram_gate_reference == DeepSeek gate (float64) to {e:.1e}; zero rows give "
          f"sigmoid(sqrt(1e-6)), masked rows 0, nothing non-finite")

    # ---- 3. module vs EngramForward reference, and the dequant ----
    class FakeWkv:
        def __init__(self):
            g = np.random.default_rng(1)
            self.a = (g.standard_normal((6144, 16)) / 6144 ** 0.5).astype(np.float32)
            self.b = g.standard_normal((16, (H + 1) * D)).astype(np.float32)
        def __call__(self, x): return (x.astype(np.float32) @ self.a) @ self.b
        def forward(self, x, params, out_dtype = None):
            return ((x.float() @ torch.from_numpy(self.a)) @ torch.from_numpy(self.b)).to(out_dtype or torch.half)
    L = 257
    x = rng.standard_normal((L, H, D)).astype(np.float32) * 0.7
    for eg in egs:
        tbl = EngramTable(checkpoint_dir, eg.backbone_idx, layout.head_dim)
        eg.load_gate(torch.device("cpu"))
        wkv = FakeWkv()
        eg.wkv = wkv
        qb = eg.q_weight.data.to(torch.bfloat16).float().numpy()
        kb = eg.k_weight.data.to(torch.bfloat16).float().numpy()
        fwd = EngramForward(layout, eg.table_index, tbl, wkv, qb, kb, H, D, eps = eps)
        want, _ = fwd(ids[:L], token_map, x)
        got = eg.forward(torch.from_numpy(x)[None].clone(), {"input_ids": torch.from_numpy(ids[:L])[None]})[0].numpy()
        e = np.linalg.norm(got - want) / np.linalg.norm(want)
        inj = np.linalg.norm(want - x) / np.linalg.norm(x)
        assert e < 1e-5, f"layer {eg.backbone_idx}: module differs from the numpy reference by {e:.2e}"

        # dead tokens (ids outside the text vocab; DeepSeek's token_mask False): their hash is
        # padded AND their gate is shut, so those positions pass through untouched
        dm = np.zeros(L, dtype = bool); dm[100:110] = True
        want_d, gate_d = fwd(ids[:L], token_map, x, dead_mask = dm)
        ids_d = ids[:L].copy(); ids_d[dm] = -1
        got_d = eg.forward(torch.from_numpy(x)[None].clone(), {"input_ids": torch.from_numpy(ids_d)[None]})[0].numpy()
        e_d = np.linalg.norm(got_d - want_d) / np.linalg.norm(want_d)
        assert (gate_d[dm] == 0).all() and np.array_equal(got_d[dm], x[dm]), \
            f"layer {eg.backbone_idx}: a dead token was not passed through untouched"
        assert e_d < 1e-5, f"layer {eg.backbone_idx}: with a dead span the module differs by {e_d:.2e}"

        # the forward's own staging (hash, dedup, C++ gather) over the 700 real tokens: its
        # rows == the Python reader's for the reference hash ids, its inverse map rebuilds
        # them, and the fp16 dequant == the fp32 dequant
        hid = engram_hash_reference(ids, layout, token_map)[:, eg.table_index].reshape(-1)
        uids = np.unique(hid)
        hist, _ = eg.history(torch.from_numpy(ids)[None], {})
        n = hid.size
        pin = eg._acquire_pin(n)
        try:
            U, _ = eg._stage(hist, pin, False)
            w, s, inv = pin.w[:U].clone(), pin.s[:U].clone(), pin.inv[:n].clone()
        finally:
            pin.held = False
        assert U == uids.size, f"staged {U} unique rows, the reference hash has {uids.size}"
        assert torch.equal(torch.from_numpy(uids)[inv], torch.from_numpy(hid)), "inverse map"
        w_py = eg.w_handle.read_rows(torch.from_numpy(uids)).view(torch.uint8)
        s_py = eg.s_handle.read_rows(torch.from_numpy(uids)).view(torch.uint8)
        assert torch.equal(w, w_py) and torch.equal(s, s_py), "C++ gather bytes differ from the Python reader"
        r16 = dequant_rows(w, s).float()
        r32 = torch.from_numpy(tbl.gather(uids))
        assert torch.equal(r16, r32), f"fp16 dequant is not exact: max |d| {(r16 - r32).abs().max():.2e}"
        codes = torch.unique(s)
        # the measuring forward stages the worst case instead: every n-gram its own row
        pin = eg._acquire_pin(n)
        try:
            assert eg._stage(hist, pin, True) == (n, None) and torch.equal(pin.inv[:n], torch.arange(n))
        finally:
            pin.held = False
        print(f"  OK  layer {eg.backbone_idx}: DSV41Engram.forward == EngramForward reference, rel-L2 "
              f"{e:.1e} (injection rel {inj:.2f}), with a dead span {e_d:.1e} (dead positions untouched); "
              f"staged {U} real rows: C++ gather == Python reader, "
              f"fp16 dequant == fp32 exactly (scale codes {int(codes.min())}..{int(codes.max())}); "
              f"the measuring forward stages {n} rows")
        tbl.close()
    check_gate_copies(cfg, egs)
    print("  OK  engram reference stages certify DeepSeek's function")


def _read_copy(stc, key, filename):
    """The copy of `key` in `filename`, exactly as stored (no loader staging)."""
    from exllamav3.loader.safetensors import DiskTensorHandle, convert_dtype
    h = stc.file_headers[filename]
    meta = h[key]
    d = DiskTensorHandle(key, filename, h["_header_offset"] + meta["data_offsets"][0], meta["shape"],
                         convert_dtype(meta["dtype"])[0])
    try:
        return d.read_range(0, d.num_rows)
    finally:
        d.close()


def _bits(t):
    return t.float().contiguous().view(torch.int32)


def _gate_weight_any_order(eg, name):
    """
    _gate_weight with the collection pointed at each copy of the tensor in turn (the copy its
    directory scan would keep, in every possible order): all of them must give the same bits.
    Returns (that tensor, copies sorted by path).
    """
    key = f"{eg.key}.{name}"
    stc = eg.config.stc.find_stc(key)
    copies = sorted(f for f, h in stc.file_headers.items() if isinstance(h.get(key), dict))
    kept = stc.tensor_file_map[key]
    got = []
    try:
        for f in copies:
            stc.tensor_file_map[key] = f
            got.append(eg._gate_weight(name, torch.device("cpu")))
    finally:
        stc.tensor_file_map[key] = kept
    for f, g in zip(copies, got):
        assert g.dtype == torch.float32 and torch.equal(_bits(g), _bits(got[0])), \
            f"{key}: the result depends on which copy the collection kept ({os.path.basename(f)})"
    return got[0], copies


def check_gate_copies(cfg, egs):
    """
    A converted checkpoint carries q_weight / k_weight twice under the same name (fp32
    converted, bf16 original) and the collection keeps whichever file its scan lists last.
    load_gate must read the bf16 copy exactly whichever that is, and its product must equal
    DeepSeek's q_weight.float() * k_weight.float() in every element. The fp32 copies of the
    V4.1-Flash conversion are the bf16 values after the loader's fp16 staging; each fp32 copy
    must be one of the two, and the elements and products the staging changes are reported.
    """
    report = []
    for eg in egs:
        eg.load_gate(torch.device("cpu"))
        exact = {}
        for name in ("q_weight", "k_weight"):
            key = f"{eg.key}.{name}"
            stc = cfg.stc.find_stc(key)
            got, copies = _gate_weight_any_order(eg, name)
            assert torch.equal(_bits(got), _bits(getattr(eg, name).data)), f"{key}: load_gate differs"
            vals = {f: _read_copy(stc, key, f) for f in copies}
            bf16 = [f for f in copies if vals[f].dtype == torch.bfloat16]
            assert bf16, f"{key}: no bf16 copy (DeepSeek's weights as released) in the checkpoint"
            for f in bf16:
                assert torch.equal(_bits(vals[f]), _bits(got)), f"{key}: the bf16 copy in {os.path.basename(f)} differs"
            staged = vals[bf16[0]].half().float()
            notes = []
            for f in copies:
                if f in bf16:
                    continue
                v = vals[f]
                n = int((_bits(v) != _bits(got)).sum())
                assert n == 0 or torch.equal(_bits(v), _bits(staged)), \
                    f"{key}: the {v.dtype} copy in {os.path.basename(f)} is neither the bf16 values " \
                    f"nor their fp16 staging"
                notes.append(f"{os.path.basename(f)} {'exact' if n == 0 else f'staged, {n} elements differ'}")
            exact[name] = vals[bf16[0]]
            report.append(f"{key.split('.')[1]}.{name} x{len(copies)}" + (f" ({'; '.join(notes)})" if notes else ""))
        want = exact["q_weight"].float() * exact["k_weight"].float()      # DeepSeek's Engram.forward
        assert torch.equal(_bits(eg.qk), _bits(want)), \
            f"layer {eg.backbone_idx}: {int((_bits(eg.qk) != _bits(want)).sum())} products differ from DeepSeek's"
        staged = (exact["q_weight"].half().float().to(torch.bfloat16).float()
                  * exact["k_weight"].half().float().to(torch.bfloat16).float())
        d = (staged - want).abs()
        report.append(f"layer {eg.backbone_idx}: fp16 staging would change {int((d > 0).sum())} of "
                      f"{want.numel()} products, max {float(d.max()):.2e}")
    print(f"  OK  engram gate weights read exactly from the bf16 copy whichever copy the collection "
          f"kept, product == DeepSeek's in every element ({', '.join(report)})")


def _write_safetensors(path, tensors: dict):
    """A minimal safetensors file holding `tensors` ({name: bf16 / fp16 / fp32 tensor})."""
    import json, struct
    names = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_gate_weight_exact_whatever_the_file_order(tmp_path):
    """
    Synthetic shards laid out as a converted checkpoint (no checkpoint needed): the fp32 copy of
    q_weight / k_weight is the fp16-staged bf16 one, as in the V4.1-Flash conversion, and the
    values include some that the staging changes (below 2^-17) or flushes to zero (below 2^-25).
    load_gate reads the bf16 values exactly whichever copy the collection kept, and whichever
    file sorts first; without a bf16 copy it reads the first copy by path, exactly; a missing
    tensor is the loader's error.
    """
    import types
    from exllamav3.loader.safetensors import SafetensorsCollection
    from exllamav3.modules.dsv41_engram import DSV41Engram
    H, D = 2, 8
    q = torch.tensor([[1.0, -0.5, 3e-6, -1e-6, 1e-9, 7e-5, 1.5 * 2.0 ** -20, -2.0 ** -26],
                      [0.25, 5e-6, -4e-6, 2e-7, 1.0 / 3, -1e-8, 6.1e-5, 12.0]]).to(torch.bfloat16)
    k = torch.tensor([[-3e-6, 2.0, 1e-7, 0.75, -6e-6, 1e-3, -2e-5, 4.5],
                      [1e-6, -1e-6, 0.125, 9e-6, -7e-7, 3.0, 1e-10, -0.01]]).to(torch.bfloat16)
    assert not torch.equal(q.half().float(), q.float()) and not torch.equal(k.half().float(), k.float())
    want_qk = q.float() * k.float()                     # DeepSeek's Engram.forward

    def engram(directory):
        stc = SafetensorsCollection(str(directory))
        eg = DSV41Engram.__new__(DSV41Engram)
        eg.key, eg.hc_mult, eg.hidden_size = "layers.1.engram", H, D
        eg.config = types.SimpleNamespace(stc = stc)
        return eg

    staged = {"layers.1.engram.q_weight": q.half().float(), "layers.1.engram.k_weight": k.half().float()}
    exact = {"layers.1.engram.q_weight": q, "layers.1.engram.k_weight": k}
    # both sort orders of the bf16 and the fp32 file
    for i, (a, b) in enumerate(((staged, exact), (exact, staged))):
        d = tmp_path / f"order{i}"
        d.mkdir()
        _write_safetensors(d / "model-00001.safetensors", a)
        _write_safetensors(d / "model-00002.safetensors", b)
        eg = engram(d)
        for name, t in (("q_weight", q), ("k_weight", k)):
            got, copies = _gate_weight_any_order(eg, name)
            assert len(copies) == 2 and torch.equal(_bits(got), _bits(t)), f"order {i}: {name} is not the bf16 copy"
        eg.load_gate(torch.device("cpu"))
        assert torch.equal(_bits(eg.qk), _bits(want_qk)), f"order {i}: the product differs from DeepSeek's"

    # no bf16 copy: the first fp32 copy by path, exactly (a second one would not be read)
    d = tmp_path / "fp32only"
    d.mkdir()
    other = {n: torch.full_like(t, 0.5) for n, t in staged.items()}
    _write_safetensors(d / "model-00001.safetensors", staged)
    _write_safetensors(d / "model-00002.safetensors", other)
    eg = engram(d)
    got, _ = _gate_weight_any_order(eg, "q_weight")
    assert torch.equal(_bits(got), _bits(q.half().float())), "without a bf16 copy: not the first fp32 copy"

    # a missing tensor: the loader's own error
    d = tmp_path / "missing"
    d.mkdir()
    _write_safetensors(d / "model-00001.safetensors", {"layers.1.engram.q_weight": q})
    eg = engram(d)
    try:
        eg._gate_weight("k_weight", torch.device("cpu"))
    except ValueError as e:
        assert "Required tensor layers.1.engram.k_weight not found" in str(e), e
    else:
        raise AssertionError("a missing k_weight did not raise")
    print("  OK  engram gate weights: bf16 values read exactly whatever the copy kept and the file "
          "order, product == DeepSeek's; fp32-only and missing copies handled")


def test_engram_worker_lifetime():
    """The shared prefetch worker stops when the last engram module releases the context and
    starts again on the next use (no checkpoint needed)."""
    import weakref
    from exllamav3.modules.dsv41_engram import _EngramCtx
    class Module:
        pass
    ctx = _EngramCtx.__new__(_EngramCtx)          # no hasher needed for the worker
    ctx._executor = None
    ctx._users = weakref.WeakSet()
    a, b = Module(), Module()
    ctx.attach(a); ctx.attach(b)
    ex = ctx.executor
    assert ex.submit(lambda: 7).result() == 7
    ctx.release(a)
    assert ctx._executor is ex, "stopped while a module still holds the table"
    ctx.release(b)
    assert ctx._executor is None and ex._shutdown
    assert ctx.executor is not ex and ctx.executor.submit(lambda: 8).result() == 8
    ctx.executor.shutdown(wait = True)
    print("  OK  engram prefetch worker stops with the last module and restarts on demand")


if __name__ == "__main__":
    import pathlib, tempfile
    test_engram_worker_lifetime()
    with tempfile.TemporaryDirectory() as tmp:
        test_gate_weight_exact_whatever_the_file_order(pathlib.Path(tmp))
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("engram reference stages", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
