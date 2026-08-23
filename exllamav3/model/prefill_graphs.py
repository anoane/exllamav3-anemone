"""P25 v3 — whole-chunk CUDA-graph capture for prefill (EXL3_PREFILL_GRAPHS=1).

Deploys to exllamav3/model/prefill_graphs.py. Captures modules[1:last_kv] of a
graph-eligible prefill chunk once per (seq, regime, cache, slot) and replays it for every
later eligible chunk. Requires P24 v3 (device-arg DSA kernels + device positions in the
non-fan projection/compressor front): with it, every kernel argument that varies across
chunks lives in device memory refreshed by the pre-replay host phase.

v3 after adversarial review: the pre-phase primes ONE MODULE PER COMPRESSOR CLASS (csa and
hca have different compress rates and separate arr slots) so no P24 refresh copy is ever
recorded inside a capture; all pinned staging is a 4-deep event-fenced ring (prefill
iterations have no per-iteration sync, so the host can run ahead of its own H2D copies);
block-table statics are keyed by width; multi-device layer splits decline.

Any capture failure permanently disables the harness for the session and the chunk reruns
eager (capture does not execute work; P24's per-class self-priming keeps the layer-level
device-arg path correct even with the harness dead).
"""

import os

import torch

from ..constants import PAGE_SIZE

N_WARMUP = 2
MAX_GRAPHS = 16     # distinct tail lengths each key a graph; past the cap they run eager
PIN_RING = 4
_debug = os.environ.get("EXL3_PREFILL_GRAPHS_DEBUG", "0") == "1"
_strict = os.environ.get("EXL3_PREFILL_GRAPHS_STRICT", "0") == "1"
_nocap = os.environ.get("EXL3_PREFILL_GRAPHS_NOCAP", "0") == "1"   # bisect: phases, no graphs

_EXCLUDE_PARAMS = ("capture", "quant_preserve", "ovr", "reconstruct", "last_tokens_only")


def _walk(module):
    yield module
    for ch in getattr(module, "modules", []) or []:
        yield from _walk(ch)


def _dev(d):
    return torch.device(f"cuda:{d}") if isinstance(d, int) else torch.device(d)


class PrefillGraphs:

    def __init__(self, model):
        from ..modules.dsv4 import DSV4Attention
        self.model = model
        self.dead = False
        self.graphs = {}       # key -> dict(g, x_ptr, y, params)
        self.warm = {}         # key -> eager runs completed at this key
        self.pool = None       # shared graph memory pool across keys
        self.stat = {}         # staging buffers/rings, keyed by name

        self.dsv4 = []
        for module, instance, idx in model.fwd_modules:
            for m in _walk(module):
                if isinstance(m, DSV4Attention):
                    self.dsv4.append(m)
        self.csa = [m for m in self.dsv4 if m.compressor is not None and m.indexer is not None]
        ws = {m.sliding_window for m in self.dsv4}
        self.sw = ws.pop() if len(ws) == 1 else None   # mixed windows => never eligible
        devs = {_dev(m.device) for m in self.dsv4}
        self.dev = devs.pop() if len(devs) == 1 else None   # multi-device => never eligible

        # one primer module per compressor class (compress_rate, has_indexer) + one
        # universal fallback; priming all classes in the pre-phase guarantees the
        # per-layer refreshes inside the captured region are mirror-skipped no-ops
        self.primers = []
        seen = set()
        for m in self.dsv4:
            if m.compressor is None:
                continue
            ck = (m.compress_rate, m.indexer is not None)
            if ck not in seen:
                seen.add(ck)
                self.primers.append(m)
        if not self.primers and self.dsv4:
            self.primers.append(self.dsv4[0])

    def _decline(self, seq, why):
        if _debug:
            k = (seq, why)
            if k not in self.stat.setdefault("_declines", set()):
                self.stat["_declines"].add(k)
                print(f" -- p25: decline seq={seq}: {why}")
        return None

    # ------------------------------------------------------------------ eligibility
    def _eligible(self, x, params):
        seq = x.shape[1]
        if self.dead or self.sw is None or self.dev is None or not self.csa \
                or x.shape[0] != 1:
            return self._decline(seq, "dead/config/multidev/bsz")
        if params.get("indexed_embeddings"):
            return self._decline(seq, "embeddings")
        if any(k in params for k in _EXCLUDE_PARAMS):
            return self._decline(seq, "excluded param")
        rsg = params.get("recurrent_states")
        if not rsg or len(rsg) != 1:
            return self._decline(seq, "recurrent_states")
        rs = rsg[0]
        cm = self.csa[0]
        rsl = cm._get_rsl(rs, (cm.layer_idx, 0))
        if not cm._p24_eligible(rs, rsl, seq):
            return self._decline(seq, f"p24_eligible pos={rs.position} wb={rs.window_beg}")
        # long-context memory budget: must mirror the dsv4 layer gate, else a capture
        # would bake host scalars for declined layers
        from ..modules.dsv4 import _p24_max_tokens
        if rs.cache.max_num_tokens > _p24_max_tokens:
            return self._decline(seq, f"max_tokens budget ({rs.cache.max_num_tokens})")
        # regime of the csa layer class (uniform m): flips once per prefill; each regime
        # gets its own graph (dense has no indexer/topk launches, topk has constant K_pad)
        regime = 1 if (rs.position + seq) // cm.compress_rate > cm.index_topk else 0
        return (seq, regime, id(rs.cache), rs.slot), rs, seq

    # ------------------------------------------------------------------ staging
    def _ring(self, name, shape, dtype):
        r = self.stat.get(name)
        if r is None:
            r = dict(
                pins = [torch.zeros(shape, dtype = dtype, pin_memory = True)
                        for _ in range(PIN_RING)],
                evs = [torch.cuda.Event() for _ in range(PIN_RING)],
                i = 0,
            )
            self.stat[name] = r
        assert r["pins"][0].shape == tuple(shape), f"p25 {name}: shape changed"
        i = r["i"]
        r["i"] = (i + 1) % PIN_RING
        r["evs"][i].synchronize()      # host may not rewrite a pin with a pending H2D
        return r["pins"][i], r["evs"][i]

    def _stage(self, name, src, dev):
        """Event-fenced pinned-ring upload of a CPU tensor into a persistent device
        static. Name must encode anything the shape depends on (e.g. seq)."""
        dev_k = name + "_dev"
        dst = self.stat.get(dev_k)
        if dst is None:
            dst = torch.zeros(src.shape, dtype = src.dtype, device = dev)
            self.stat[dev_k] = dst
        assert dst.shape == src.shape, f"p25 {name}: shape changed {dst.shape} vs {src.shape}"
        pin, ev = self._ring(name + "_ring", src.shape, src.dtype)
        pin.copy_(src)
        dst.copy_(pin, non_blocking = True)
        ev.record()
        return dst

    def _stage_table(self, name, src, dev, width):
        """(1, npr) int table into a fixed (1, width) static; content beyond npr unused
        (all in-bounds pool indices stay below the entry count)."""
        name = f"{name}{width}"
        dev_k = name + "_dev"
        dst = self.stat.get(dev_k)
        if dst is None:
            dst = torch.zeros((1, width), dtype = src.dtype, device = dev)
            self.stat[dev_k] = dst
        npr = min(src.shape[1], dst.shape[1])
        pin, ev = self._ring(name + "_ring", (1, width), src.dtype)
        pin[:, :npr].copy_(src[:, :npr])
        dst[:, :npr].copy_(pin[:, :npr], non_blocking = True)
        ev.record()
        return dst

    def _pre_phase(self, x, params, rs, seq):
        """Embedding eager + all host->device staging + P24 priming (one refresh per
        compressor class so nothing refreshes inside the captured region). Returns x_st."""
        model = self.model
        m0, inst0, _ = model.fwd_modules[0]
        params["layer_instance"] = inst0
        h = m0.prepare_for_device(x, params)
        h = m0.forward(h, params)
        dev = self.dev

        if h.device.type == "cpu":
            x_st = self._stage(f"x{seq}", h, dev)
        else:
            x_k = f"x{seq}_dev"
            x_st = self.stat.get(x_k)
            if x_st is None:
                x_st = torch.zeros_like(h)
                self.stat[x_k] = x_st
            x_st.copy_(h)

        # tables/ids must live in OUR persistent statics regardless of source device: a
        # CPU source is pinned-staged; a device source (P27's per-job bt or recycled ring
        # tensors) is D2D-copied — a captured graph may only ever bake harness-owned
        # pointers (P27-interplay review finding)
        bt = params.get("block_table")
        if bt is not None:
            width = rs.cache.max_num_tokens // PAGE_SIZE
            if bt.device.type == "cpu":
                params["block_table"] = self._stage_table("bt", bt, dev, width)
            else:
                dst = self.stat.get(f"bt{width}_dev")
                if dst is None:
                    dst = torch.zeros((1, width), dtype = bt.dtype, device = dev)
                    self.stat[f"bt{width}_dev"] = dst
                npr = min(bt.shape[1], width)
                dst[:, :npr].copy_(bt[:, :npr])
                params["block_table"] = dst
        ids = params.get("input_ids")
        if ids is not None and torch.is_tensor(ids):
            if ids.device.type == "cpu":
                params["input_ids"] = self._stage(f"ids{seq}", ids, dev)
            else:
                dst = self.stat.get(f"ids{seq}_dev")
                if dst is None:
                    dst = torch.zeros(ids.shape, dtype = ids.dtype, device = dev)
                    self.stat[f"ids{seq}_dev"] = dst
                dst.copy_(ids)
                params["input_ids"] = dst
        cs = params.get("cache_seqlens")
        if cs is not None and torch.is_tensor(cs) and cs.device.type == "cpu":
            params["cache_seqlens"] = self._stage("cs", cs, dev)

        st_bt = params.get("block_table")
        for pm in self.primers:
            pm._p24_refresh(rs, pm._p24_state(rs, dev), seq, st_bt)
        return x_st

    def _run_stack(self, x_st, params):
        """The prefill_ls module loop over modules[1:], identical break condition."""
        model = self.model
        x = x_st
        for module, instance, idx in model.fwd_modules[1:]:
            params["layer_instance"] = instance
            pf = (idx, instance) == model.last_kv_module_idx_instance
            params["prefill"] = pf
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if pf:
                break
        params.pop("prefill", None)
        return x

    def _post_phase(self, rs, seq):
        """The eager ring-rebase host bookkeeping (idempotent across layers)."""
        new_beg = max(rs.position + seq - (self.sw - 1), 0) // PAGE_SIZE * PAGE_SIZE
        rs.wshift = new_beg - rs.window_beg

    # ------------------------------------------------------------------ main entry
    def try_chunk(self, x, params):
        """True if the chunk was fully handled (replay, warmup, or fail-safe eager)."""
        el = self._eligible(x, params)
        if el is None:
            return False
        key, rs, seq = el

        try:
            x_st = self._pre_phase(x, params, rs, seq)

            ent = self.graphs.get(key)
            if ent is not None:
                assert x_st.data_ptr() == ent["x_ptr"], "p25: graph input static moved"
                ent["g"].replay()
                self._post_phase(rs, seq)
                return True

            warm = self.warm.get(key, 0)
            if warm < N_WARMUP or len(self.graphs) >= MAX_GRAPHS or _nocap:
                self._run_stack(x_st, params)
                self.warm[key] = warm + 1
                return True
        except torch.OutOfMemoryError as e:
            # statics/warmup ran out of headroom (long-context margins are tens of MB).
            # Partial chunk work is idempotent (pool/ring destinations are deterministic
            # and recomputed identically), so the caller's eager loop can rerun it.
            torch.cuda.synchronize()
            self.dead = True
            self.graphs.clear()
            if _strict:
                raise
            print(f" !! p25: OOM in prefill-graph path, permanent eager fallback: {e}")
            return False

        try:
            torch.cuda.synchronize()
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            cap_params = dict(params)
            with torch.cuda.graph(g, pool = self.pool):
                y = self._run_stack(x_st, cap_params)
            g.replay()                       # capture records; this executes the chunk
            self._post_phase(rs, seq)
            self.graphs[key] = dict(g = g, x_ptr = x_st.data_ptr(), y = y,
                                    params = cap_params)
            if _debug:
                print(f" -- p25: captured prefill graph key={key}")
            return True
        except Exception as e:
            torch.cuda.synchronize()
            self.dead = True
            self.graphs.clear()
            if _strict:
                raise
            print(f" !! p25: prefill graph capture failed, eager fallback: {e}")
            self._run_stack(x_st, params)    # capture did not execute; rerun is safe
            self._post_phase(rs, seq)
            return True


def get_harness(model):
    h = getattr(model, "_p25_harness", None)
    if h is None:
        h = PrefillGraphs(model)
        model._p25_harness = h
    return h
