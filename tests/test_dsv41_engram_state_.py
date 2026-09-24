"""Engram n-gram context carry (DSV41EngramState) vs DeepSeek's one-shot NgramHashState.

    PYTHONPATH=$PWD python3 tests/test_dsv41_engram_state_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

The oracle is DeepSeek's own inference/engram.py (DSV41_DEEPSEEK_REF), run ONE-SHOT
over the whole text: its full-sequence id cache is the ground truth that a chunked /
decoding / rewinding run through the ring must reproduce bit for bit. Without
DSV41_DEEPSEEK_REF only the token-map and graph checks run. Everything goes through the real module
(DSV41Engram.history -> engram_hash_chunk -> the end-of-forward commit), the model's
job state (cache/dsv41.py DSV41State: clear, stash, unstash, rewind) and the real
advance_recurrent_states, on a stand-in Cache that only maps keys to layer states.

Text: the checkpoint's README.md + LICENSE, tokenized with its own tokenizer.json.
Covers, for both engram layers:
  - random prefill chunks of 1..300 tokens, then token-by-token decode;
  - speculative verify of 4 positions, 2 accepted, state rewound by 2, repeated;
  - checkpoint (stash) mid-sequence, restored into a cleared slot, continued;
  - a batch of two rows at DIFFERENT positions in one decode forward;
  - a rewind past the ring raises instead of hashing with overwritten ids, and so do
    two rewinds that are each inside max_rewind but together reach overwritten rows;
  - after a checkpoint is restored, an in-place rewind BELOW the checkpoint position
    (which the job state allows: it restores window_beg) still hashes exactly;
  - dead spans of 1-5 alias ids placed at and across chunk boundaries, chunked and
    decoded through the ring, 40 schedules, against DeepSeek's one-shot with its token
    mask.
Negative controls: the same chunking WITHOUT the carried lookback must differ, and
only within 3 positions of a chunk boundary, both in the hash ids and in the rows the
module gathers under EXL3_DSV41_ABLATE=engram_nocarry.
Also: the token map (size, pad, image ids; its build time is printed), the Cache keys of the
engram states, checkpoint size, and weights_numel excluding the table.
CPU only: nothing here allocates on a GPU."""
import os, random, sys, time, types
import torch

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import ablate, model_dir, skip
from dsv41_ref import deepseek


class FakeCache:
    """What the job state and the engram use of a Cache: layer states by key."""
    def __init__(self):
        self.layers = {}
        self.model = types.SimpleNamespace(loaded_tp = False)
    def get_recurrent_layer(self, key): return self.layers[key]
    def get_all_recurrent_layers(self): return self.layers
    def release_state(self, s): pass


def main(checkpoint_dir):
    from exllamav3 import Config, Model
    from exllamav3.cache.dsv41 import DSV41State as JobState
    from exllamav3.cache.recurrent_util import advance_recurrent_states
    from exllamav3.architecture.dsv41.engram_torch import (
        build_compressed_token_map, engram_hash_chunk, UNK)
    from exllamav3.architecture.dsv41.engram_state import (
        DSV41EngramState, RING_ROWS, set_lookback, commit_chunk)
    from tokenizers import Tokenizer

    # ---- token map ----
    t0 = time.time()
    token_map, size = build_compressed_token_map(os.path.join(checkpoint_dir, "tokenizer.json"))
    dt = time.time() - t0
    assert size == 99092 and token_map.numel() == 129280, (size, token_map.numel())
    assert int(token_map[2]) == 2, "pad id 2 must compress to 2"
    assert int(token_map[129264]) == 99076 and int(token_map[129265]) == 99077
    print(f"  OK  token map: {token_map.numel()} ids -> {size}, map[2]=2, "
          f"129264/129265 -> 99076/99077, built in {dt:.2f} s")

    # ---- model graph (not loaded) and the engram modules' host side ----
    cfg = Config.from_directory(checkpoint_dir)
    model = Model.from_config(cfg)
    egs = [model.engram_modules[i] for i in sorted(model.engram_modules)]
    for eg in egs:
        eg.open_table()
    hasher = egs[0].hasher
    assert torch.equal(hasher.token_map, token_map)

    keys = [(m.layer_idx, 0) for m in model.get_recurrent_layers()]
    assert len(keys) == len(set(keys)), "recurrent-state keys collide"
    eg_keys = sorted(eg.state_key() for eg in egs)
    assert eg_keys == [(-1014, 0), (-1001, 0)], eg_keys
    attn_keys = {k for k in keys if k[0] >= 0}
    assert {(1, 0), (14, 0)} <= attn_keys and not (set(eg_keys) & attn_keys)
    print(f"  OK  {len(keys)} recurrent-state keys, all distinct; engram states at {eg_keys}, "
          f"attention keeps (1, 0) / (14, 0)")

    for eg in egs:
        n = eg.weights_numel()
        # wkv (6144 x 25600) + q/k (2 x 4 x 5120); the table alone is ~1e11 elements
        assert n == 6144 * 25600 + 2 * 4 * 5120, f"{eg.key}: weights_numel {n}"
        assert eg.table_numel() == cfg.engram_num_embeddings[eg.table_index] * 264
    print(f"  OK  weights_numel {egs[0].weights_numel():,} per engram (wkv + q/k); the table "
          f"({egs[0].table_numel():,} elements) is reported only by table_numel")

    # ---- text ----
    tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    text = open(os.path.join(checkpoint_dir, "README.md")).read() + "\n" + open(os.path.join(checkpoint_dir, "LICENSE")).read()
    ids = torch.tensor(tok.encode(text, add_special_tokens = False).ids, dtype = torch.long)
    N = ids.numel()
    assert N >= 700, N

    # ---- oracle: DeepSeek's NgramHashState, one shot ----
    if deepseek.ref_dir() is None:
        skip("engram context carry vs DeepSeek NgramHashState", "DSV41_DEEPSEEK_REF not set")
        return
    ref = deepseek.load_engram()
    t = cfg
    args = types.SimpleNamespace(
        engram_layer_ids = t.engram_layer_ids, engram_max_ngram_size = t.engram_max_ngram_size,
        engram_n_heads = t.engram_n_heads, engram_head_dim = t.engram_head_dim,
        engram_vocab_size = t.engram_vocab_size, engram_num_embeddings = t.engram_num_embeddings,
        engram_compressed_vocab_size = t.engram_compressed_vocab_size,
        engram_pad_id = t.engram_pad_token_id, max_batch_size = 2, max_seq_len = N + 16)
    class Shim:
        def __init__(self, b): self.backend_tokenizer = b
        def __len__(self): return self.backend_tokenizer.get_vocab_size(with_added_tokens = True)
    ref_state = ref.NgramHashState(args, ref.EngramLayout.from_args(args), Shim(tok))
    want = ref_state(ids[None], 0)[0].long()                              # [N, n_layers, 24]
    print(f"  --  oracle: DeepSeek NgramHashState one-shot over {N} README/LICENSE tokens")

    # ---- the job-state harness ----
    cache = FakeCache()
    for eg in egs:
        st = DSV41EngramState(eg, max_batch_size = 4, max_history = 8, cache_id = id(cache))
        st.alloc(torch.device("cpu"))
        cache.layers[eg.state_key()] = st

    def step(states, chunk_ids):
        """One forward over chunk_ids [B, L] for rows `states`, as the model runs it:
        each engram layer builds its history from ITS state, hashes, and commits at the
        end of its forward; then advance_recurrent_states moves the positions."""
        params = {"input_ids": chunk_ids, "recurrent_states": states}
        out = {}
        for eg in egs:
            hist, st = eg.history(chunk_ids, params)
            out[eg.table_index] = engram_hash_chunk(hist, eg.table_index, eg.hasher)
            eg._commit(hist, st, params)
        advance_recurrent_states(chunk_ids, params, None)
        return out

    def check(got, what):
        for eg in egs:
            li = eg.table_index
            g = got[li]
            assert g.shape == want[:, li].shape, (g.shape, want[:, li].shape)
            bad = (g != want[:, li]).any(-1)
            assert not bad.any(), f"{what}, layer {eg.backbone_idx}: {int(bad.sum())} positions differ, " \
                                  f"first at {int(bad.nonzero()[0])}"

    # (1) random prefill chunks, then decode (several chunkings)
    rnd = random.Random(0)
    got = {eg.table_index: torch.empty((N, 24), dtype = torch.long) for eg in egs}
    split = N - 120
    n_chunks = 0
    for seed in range(4):
        s = JobState(cache, 0, 0)
        pos = 0
        while pos < split:
            L = min(rnd.randint(1, 300), split - pos)
            o = step([s], ids[None, pos:pos + L])
            for li in got: got[li][pos:pos + L] = o[li][0]
            pos += L; n_chunks += 1
        while pos < N:
            o = step([s], ids[None, pos:pos + 1])
            for li in got: got[li][pos] = o[li][0, 0]
            pos += 1
        assert s.position == N
        check(got, f"chunked prefill + decode, pass {seed}")
    print(f"  OK  4 passes, {n_chunks} random prefill chunks (1..300) + {N - split} decode steps "
          f"each == one-shot, bitwise, both layers")

    # (2) speculative decoding: verify 4 positions, accept 2, rewind 2
    s = JobState(cache, 1, 0)
    pre = 257
    o = step([s], ids[None, :pre])
    for li in got: got[li][:pre] = o[li][0]
    pos = pre; rounds = 0
    while pos + 4 <= N:
        draft = ids[pos:pos + 4].clone()
        draft[2:] = (draft[2:] + 7919) % 129280                          # two wrong draft tokens
        o = step([s], draft[None])
        for li in got: got[li][pos:pos + 2] = o[li][0, :2]               # accepted positions only
        s.rewind(2)
        pos += 2; rounds += 1
    while pos < N:
        o = step([s], ids[None, pos:pos + 1])
        for li in got: got[li][pos] = o[li][0, 0]
        pos += 1
    check(got, "speculative accept-2-of-4")
    print(f"  OK  {rounds} speculative rounds (verify 4, accept 2, rewind 2) == one-shot, bitwise")

    # (3) checkpoint mid-sequence, restored into a cleared slot
    s = JobState(cache, 2, 0)
    P = 511
    o = step([s], ids[None, :P])
    for li in got: got[li][:P] = o[li][0]
    stashed = s.stash()
    assert stashed["checkpoint_size"] == 2 * (2 * RING_ROWS * 8), stashed["checkpoint_size"]
    for k in eg_keys:
        n = min(P, RING_ROWS)
        assert stashed[k].shape == (2, n), stashed[k].shape
        assert torch.equal(stashed[k][0], hasher.compress(ids[P - n:P])), k
        assert torch.equal(stashed[k][1], torch.arange(P - n, P)), k
    for st in cache.layers.values():                                     # garbage in the target slot
        st.ids[3].random_(0, 99092); st.pos[3].random_(0, 4096); st.hi[3] = 10 ** 6
    s2 = JobState(cache, 3, P, clear = False, stashed = stashed)
    pos = P
    while pos < N:
        L = min(rnd.randint(1, 64), N - pos)
        o = step([s2], ids[None, pos:pos + L])
        for li in got: got[li][pos:pos + L] = o[li][0]
        pos += L
    check(got, "stash / unstash")
    print(f"  OK  checkpoint at {P} ({stashed['checkpoint_size']} bytes: the ring's ids x 2 layers), "
          f"unstashed into a dirty slot, continued == one-shot")

    # (3b) restored checkpoint, then an in-place rewind BELOW it. The job state restores the
    # stashed window_beg, so its rollback_capacity allows this right after unstash; the
    # lookback of the rewound chunk lies before the checkpoint position
    s = JobState(cache, 2, 0)
    P = 1024
    pos = 0
    while pos < P:
        L = min(rnd.randint(1, 300), P - pos)
        step([s], ids[None, pos:pos + L]); pos += L
    stashed = s.stash()
    s2 = JobState(cache, 3, P, clear = False, stashed = stashed)
    step([s2], ids[None, P:P + 40])
    back = 100
    assert s2.rollback_capacity() >= back, s2.rollback_capacity()
    s2.rewind(back)
    pos = s2.position
    assert pos < P
    while pos < N:
        L = min(rnd.randint(1, 64), N - pos)
        o = step([s2], ids[None, pos:pos + L])
        for li in got: got[li][pos:pos + L] = o[li][0]
        pos += L
    got_ok = all(torch.equal(got[eg.table_index][P - back + 40:], want[P - back + 40:, eg.table_index])
                 for eg in egs)
    assert got_ok, "rewind below a restored checkpoint: hashes differ from one-shot"
    print(f"  OK  checkpoint at {P} restored, advanced 40, rewound {back} in place (below the "
          f"checkpoint), continued == one-shot")

    # (4) a batch of two rows at different positions in one forward
    sa, sb = JobState(cache, 0, 0), JobState(cache, 1, 0)
    Pa, Pb = 300, 200
    step([sa], ids[None, :Pa]); step([sb], ids[None, :Pb])
    for k in range(40):
        o = step([sa, sb], torch.stack([ids[Pa + k:Pa + k + 1], ids[Pb + k:Pb + k + 1]]))
        for li in got:
            assert torch.equal(o[li][0, 0], want[Pa + k, li]) and torch.equal(o[li][1, 0], want[Pb + k, li]), \
                f"batched decode, row positions {Pa + k}/{Pb + k}"
    print(f"  OK  batched decode, two rows at positions {Pa}.. and {Pb}.. in one forward, 40 steps")

    # (5) the exported functions: set_lookback / commit_chunk agree with the module
    s = JobState(cache, 2, 0)
    step([s], ids[None, :100])
    params = {"input_ids": ids[None, 100:130], "recurrent_states": [s]}
    lb = set_lookback(params)
    assert torch.equal(lb[0], hasher.compress(ids[97:100])) and params["dsv41_engram_lookback"] is lb
    commit_chunk(params)
    advance_recurrent_states(params["input_ids"], params, None)
    o = step([s], ids[None, 130:131])
    assert all(torch.equal(o[li][0, 0], want[130, li]) for li in o)
    print(f"  OK  set_lookback / commit_chunk: lookback = ids 97..99, a committed chunk carries on")

    # (6) a rewind past the ring must raise, not hash with overwritten ids
    s = JobState(cache, 3, 0)
    step([s], ids[None, :RING_ROWS + 50])
    s.position = 40                                                      # deeper than any legal rewind
    try:
        step([s], ids[None, 40:41])
        raise AssertionError("lookback over an overwritten ring entry did not raise")
    except RuntimeError as e:
        assert "no longer holds" in str(e) and "overwritten by position" in str(e), e
    print(f"  OK  a {RING_ROWS + 10}-token rewind raises (ring {RING_ROWS}, max_rewind "
          f"{cache.layers[eg_keys[0]].max_rewind})")

    # (6b) two rewinds, EACH inside max_rewind, that together reach rows the first pass
    # overwrote: 0..1499 written, rewind 600 to 900, re-advance 101 other tokens, rewind 601
    # to 400. The frontier is then 1000, 603 tokens ahead, but position 397's row holds
    # position 1421. Must raise, never hash with 1421's id
    mr = cache.layers[eg_keys[0]].max_rewind
    s = JobState(cache, 3, 0)
    pos = 0
    while pos < 1500:
        step([s], ids[None, pos:pos + 100]); pos += 100
    s.rewind(600)
    assert 600 <= mr
    step([s], ((ids[None, 900:1001] + 7919) % 129280))
    s.rewind(601)
    assert 601 <= mr and s.position == 400
    try:
        step([s], ids[None, 400:401])
        raise AssertionError("two rewinds past the ring hashed with overwritten ids instead of raising")
    except RuntimeError as e:
        assert "position 397" in str(e) and "overwritten by position 1421" in str(e), e
    print(f"  OK  two rewinds of 600 and 601 (each <= max_rewind {mr}) that reach overwritten rows raise")

    # (6c) dead tokens (ids outside the text vocab: image spans in the reference) at and across
    # chunk boundaries: the ring carries DEAD like any id, so a later chunk's lookback still
    # blocks on it. DeepSeek's one-shot NgramHashState with its token mask is the oracle
    from exllamav3.architecture.dsv41.engram_torch import DEAD
    bad = 0
    for trial in range(40):
        rt = random.Random(trial)
        L = 400
        tid = torch.randint(0, 129000, (1, L), generator = torch.Generator().manual_seed(trial))
        mask = torch.ones(1, L, dtype = torch.bool)
        cuts = sorted(rt.sample(range(5, L - 5), 12))
        for c in cuts[::2]:
            a = c - rt.randint(0, 3)
            mask[0, a:a + rt.randint(1, 5)] = False
        tid_port = torch.where(mask, tid, 129280 + 7)                 # alias ids past the vocab
        oracle = ref.NgramHashState(args, ref.EngramLayout.from_args(args), Shim(tok))
        want_d = oracle(tid, 0, mask).long()[0, :, 0]
        st = DSV41EngramState(types.SimpleNamespace(ctx = hasher.ctx), 1, 8, 0)
        st.alloc()
        parts, bounds = [], [0] + cuts + [L]
        for a, b in zip(bounds[:-1], bounds[1:]):
            for x0, x1 in ([(a, b)] if rt.random() < 0.5 else [(q, q + 1) for q in range(a, b)]):
                c = hasher.compress(tid_port[:, x0:x1])
                assert (c == DEAD).sum() == (~mask[:, x0:x1]).sum()
                parts.append(hasher.hash_hist(torch.cat([st.lookback(0, x0)[None], c], 1), 0)[0])
                st.write(0, x0, c[0])
        bad += not torch.equal(torch.cat(parts, 0), want_d)
    assert bad == 0, f"dead spans across chunk boundaries: {bad}/40 schedules differ from one-shot"
    print("  OK  dead spans of 1-5 ids at and across chunk boundaries, chunked and decoded through the "
          "ring: 40/40 schedules == DeepSeek one-shot with its token mask")

    # (7) the whole module forward through the state: chunked prefill + decode must feed
    # wkv the same gathered rows as one shot (bitwise) and give the same streams
    class FakeWkv:
        """rank-16 fp32 stand-in for the exl3 wkv, recording its input"""
        def __init__(self):
            g = torch.Generator().manual_seed(1)
            self.a = torch.randn(6144, 16, generator = g) / 6144 ** 0.5
            self.b = torch.randn(16, 5 * 5120, generator = g)
            self.inputs = []
        def forward(self, x, params, out_dtype = None):
            self.inputs.append(x.clone())
            return ((x.float() @ self.a) @ self.b).to(out_dtype or torch.half)
    M = 640
    x = torch.randn(1, M, 4, 5120, generator = torch.Generator().manual_seed(2)) * 0.5
    for eg in egs:
        eg.load_gate(torch.device("cpu"))
        eg.wkv = FakeWkv()
        y1 = eg.forward(x.clone(), {"input_ids": ids[None, :M]})          # stateless, one shot
        emb1 = eg.wkv.inputs.pop()
        s = JobState(cache, 0, 0)
        pos = 0; ys = []
        for L in [200, 1, 77, 150, 3] + [1] * 40:
            params = {"input_ids": ids[None, pos:pos + L], "recurrent_states": [s]}
            ys.append(eg.forward(x[:, pos:pos + L].clone(), params))
            advance_recurrent_states(params["input_ids"], params, None)
            pos += L
        params = {"input_ids": ids[None, pos:M], "recurrent_states": [s]}
        ys.append(eg.forward(x[:, pos:M].clone(), params))
        emb2 = torch.cat(eg.wkv.inputs, dim = 1)
        y2 = torch.cat(ys, dim = 1)
        assert torch.equal(emb1, emb2), f"layer {eg.backbone_idx}: gathered rows differ chunked vs one-shot"
        e = ((y1 - y2).norm() / y1.norm()).item()
        assert e < 1e-6, f"layer {eg.backbone_idx}: streams differ by rel-L2 {e:.2e}"
        print(f"  OK  layer {eg.backbone_idx} forward, 46 chunks/decodes through the state vs one shot: "
              f"wkv input bitwise equal, streams rel-L2 {e:.1e}, engram moved them by rel-L2 "
              f"{((y1 - x).norm() / x.norm()).item():.3f}")

        # negative control through the module: with engram_nocarry each forward hashes its
        # first n-grams without the carried ids, so the rows gathered for the first 3
        # positions of every later chunk change, and no others
        eg.wkv.inputs.clear()
        s = JobState(cache, 0, 0)
        pos = 0; starts = []
        with ablate("engram_nocarry"):
            for L in [200, 1, 77, 150, 3] + [1] * 40 + [M - 471]:
                params = {"input_ids": ids[None, pos:pos + L], "recurrent_states": [s]}
                eg.forward(x[:, pos:pos + L].clone(), params)
                advance_recurrent_states(params["input_ids"], params, None)
                starts.append(pos)
                pos += L
        emb3 = torch.cat(eg.wkv.inputs, dim = 1)
        eg.wkv.inputs.clear()
        bad = (emb3 != emb1).reshape(M, -1).any(-1)
        near = torch.zeros(M, dtype = torch.bool)
        for a in starts[1:]:
            near[a:a + 3] = True
        assert bad.any(), f"layer {eg.backbone_idx}: engram_nocarry changed nothing"
        assert not (bad & ~near).any(), f"layer {eg.backbone_idx}: engram_nocarry changed rows away " \
                                        f"from a chunk start"
        print(f"  OK  layer {eg.backbone_idx} negative control: with EXL3_DSV41_ABLATE=engram_nocarry "
              f"{int(bad.sum())} positions gather other rows, all within 3 of one of "
              f"{len(starts) - 1} chunk starts")

    # negative control: same chunking, lookback dropped -> must differ at boundaries only
    pos = 0; diff = 0; boundaries = 0
    while pos < N:
        L = min(rnd.randint(1, 300), N - pos)
        hist = torch.cat([torch.full((1, 3), UNK), hasher.compress(ids[None, pos:pos + L])], 1)
        h = engram_hash_chunk(hist, 0, hasher)[0]
        bad = (h != want[pos:pos + L, 0]).any(-1)
        if pos > 0:
            boundaries += 1
            assert not bad[3:].any(), "no-lookback hashes differ away from a boundary"
        diff += int(bad.sum())
        pos += L
    assert diff > 0, "dropping the lookback changed nothing: the check would not see it"
    print(f"  OK  negative control: without the carried lookback {diff} positions differ, all "
          f"within 3 of one of {boundaries} chunk boundaries")

    print("  OK  engram context carry == DeepSeek one-shot hashing in every generation pattern")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("engram context carry", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
