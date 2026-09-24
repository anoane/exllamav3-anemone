"""
Cache state for DeepSeek-V4.1 attention, added alongside V4's (cache/dsa.py is untouched).

V4 gives every compressed layer its own paged pool (CacheLayer_dsa) and every layer a fixed
set of compressor rings (DSV4LayerState). Neither fits V4.1:

  * Only the kv SOURCES compress (layers 2, 8, 14, 20 on DeepSeek-V4.1-Flash). Everything
    else in a kv group reads the source's pool, so Cache builds exactly one
    CacheLayer_dsv41 per source -- 3 x 640 MiB + 1.25 GiB at 1M tokens -- instead of 38
    per-layer pools (~29 GiB). The pools keep V4's page-major geometry
    (epp = PAGE_SIZE // m entries per token page, addressed through the job's ordinary block
    table), so prefix reuse, defrag, copy_page and the CPU tier apply unchanged.

  * Index keys belong to the source too. V4 allocates pool_idx only for its 'csa' kind; a
    V4.1 source that also owns index K (all four do) gets a 128-wide pool_idx, which index
    sources above it (24/28/32/36 read layer 20's) share.

  * The compressor has no overlap and no position bias: a rate-2 group is two raw rows,
    pooled when the second one arrives. The only cross-chunk state is therefore the raw
    [kv | score] row of a group still open at a chunk boundary. It is kept in a
    position-indexed fp32 ring of PAGE_SIZE + m rows per slot (comp_carry), so a rewind is
    pure cursor arithmetic: the row a rewound position needs is still in the ring for any
    rollback this state class reports as safe.

The SWA ring (768 rows, raw roped K=V) is identical to V4's and exists on every layer,
sliding and compressed alike: each V4.1 layer keeps its own window.

Also here: check_pool_addressing, which refuses a pool the int32 kernels cannot address, and
DeviceMemo, which moves the small per-forward selections of the stateless path to a reading
layer on another device, once per forward.
"""

from __future__ import annotations

import torch

from ..constants import PAGE_SIZE
from ..util.device_copy import to_device
from .dsa import CacheLayer_dsa, DSV4LayerState, DSV4State


# The DSA kernels address a pool row as row * width in int32
INT32_LIMIT = 2 ** 31


def check_pool_addressing(layer: CacheLayer_dsa, what: str):
    """
    Refuse a pool the int32 kernels cannot address. dsa_attn, the indexer scorer and the
    pool scatters compute a row's offset as row * width in int32, so a pool whose rows times
    its widest row reach 2^31 (about 4.79M entries of 448 fp16 values) would read and write
    another allocation past that point, without an error. A quantized (-cq) pool is checked
    at its packed widths (pool_q, pool_s); the prefill staging of dsa_triton dequantizes up
    to EXL3_DSA_QC_STAGE_MAX_ENTRIES entries (1M by default) into a D_c-wide fp16 buffer,
    which the same limit caps at about 4.79M entries if that variable is raised.
    """
    widths = [layer.D_r, layer.D_i]
    widths += [layer.G * layer.k_bits, layer.G] if layer.k_bits else [layer.D_c]
    width = max(widths)
    if layer.capacity * width >= INT32_LIMIT:
        pages = ((INT32_LIMIT - 1) // width) // layer.epp
        raise ValueError(
            f"{what}: {layer.capacity:,} pool entries x {width} values reach the kernels' int32 "
            f"row addressing (2^31); max_num_tokens can be at most {pages * PAGE_SIZE:,} for "
            f"this pool")


def _nbytes(obj) -> int:
    """Bytes held by the tensors of a stash entry (tensor, or nested lists/tuples/dicts)."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(o) for o in obj)
    if isinstance(obj, dict):
        return sum(_nbytes(o) for o in obj.values())
    return 0


class CacheLayer_dsv41(CacheLayer_dsa):
    """
    Paged pools of ONE V4.1 kv source: pool_c (nope, 448), pool_r (rope, 64) and, when the
    source owns index K, pool_idx (128), each (num_pages, PAGE_SIZE // m, D).

    Everything but the index width is inherited from CacheLayer_dsa: alloc/free,
    pool_c_view/qc, the k_bits packed path, copy_page, get_tensors, storage_size, slot_bt.
    The DSA kernels address a pool row as row * width in int32, so a pool whose rows times
    its widest row reach 2^31 is refused here (fp16 at rate 1: max_num_tokens 4,793,344).
    """

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int = 0,
        v_bits: int | None = None,
        compand_a: float = 0.0,
        **kwargs
    ):
        assert attention.is_kv_source, \
            f"CacheLayer_dsv41: layer {attention.layer_idx} is not a kv source; consumers " \
            f"read their source's pool and must not own one"
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits, compand_a)
        # V4 sets D_i only for its 'csa' kind (dsa.py); V4.1 drives the base as 'hca'
        self.D_i = attention.index_head_dim if attention.owns_index_k else 0
        check_pool_addressing(self, f"pool of kv source {attention.layer_idx}")

    def tp_export(self, plan):
        raise NotImplementedError("Tensor-parallel loading is not supported for DeepSeek-V4.1 pools")


class DSV41LayerState(DSV4LayerState):
    """
    Per-layer, per-cache recurrent state of one V4.1 attention layer.

    Replaces V4's __init__ (which dereferences compressor.wkv on every compressed layer and
    would crash on consumers) rather than calling it:

      ring          (B, 768, 512) fp16   every layer: raw roped K=V of the sliding window
      comp_carry    (B, PAGE_SIZE + 2, 1024) fp32   rate-2 kv sources: raw [kv | score]
                                         rows, row = abs position % rows

    Fields of V4's layer state that a V4.1 layer does not allocate stay None. _tensors and
    _set_tensors skip None fields, and the methods inherited from V4 (alloc, free, clear,
    rewind, storage_size) go through them. stash() keeps only the ring rows that hold
    history, and get_checkpoint_size() bounds what it keeps.
    """

    _NAMES = ("ring", "comp_buf_kv", "comp_buf_gate", "idx_buf_kv", "idx_buf_gate",
              "comp_ovl", "idx_ovl", "comp_carry")

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.cache_id = cache_id
        self.max_batch_size = max_batch_size
        self.max_history = max_history
        self.device = None

        D = module.head_dim
        D_r = module.rope_head_dim
        self.layer_type = module.layer_type
        self.window = module.sliding_window
        overp = 2 * PAGE_SIZE
        self.ring_rows = -(-(self.window + overp) // PAGE_SIZE) * PAGE_SIZE
        self.D_c = D - D_r
        self.D_r = D_r

        mk = lambda *shape, dtype = torch.half: torch.zeros(shape, dtype = dtype, device = "meta")
        B = max_batch_size
        self.ring = mk(B, self.ring_rows, D)

        self.comp_buf_kv = self.comp_buf_gate = None
        self.idx_buf_kv = self.idx_buf_gate = None
        self.comp_ovl = self.idx_ovl = None
        self.comp_carry = None
        self.buf_rows = 0

        m = getattr(module, "compress_ratio", 0) or 0
        if getattr(module, "is_kv_source", False) and m > 1:
            # One open group spans at most m - 1 raw rows; the ring is sized for rollback:
            # after a forward ending at P it holds rows [P - buf_rows, P), so a rewind to T
            # finds row T - 1 while P - T <= buf_rows - 1 (see rollback_limit)
            self.buf_rows = PAGE_SIZE + m
            self.comp_carry = mk(B, self.buf_rows, 2 * D, dtype = torch.float)

    def _tensors(self):
        return [t for t in (getattr(self, n) for n in self._NAMES) if t is not None]

    def _set_tensors(self, ts):
        it = iter(ts)
        for n in self._NAMES:
            if getattr(self, n) is not None:
                setattr(self, n, next(it))

    def rollback_limit(self) -> int | None:
        """How far below the job's HIGH-WATER position (the furthest position ever reached;
        DSV41State.high_water) a rewind may land, or None when unbounded. The carry ring is
        position-modulo: it holds rows [hw - buf_rows, hw), and a rewind to T needs row T - 1
        when T falls mid-group, so hw - T <= buf_rows - 1. Measured from the current position
        instead, two rewinds in a row (with a short replay between them) could reach below
        the ring -- rows the replay never rewrote."""
        return self.buf_rows - 1 if self.buf_rows else None

    def get_checkpoint_size(self):
        """Upper bound of what stash() keeps: every row of every ring. V4's figure counts one
        window (128 rows) of the 768-row ring, but its stash keeps up to 768, so a V4.1
        checkpoint would be 4.1x its declared size and the generator's host-RAM budget for
        checkpoints (RecurrentCache.max_size) 4.1x too small. DSV41State.stash declares the
        exact bytes of each checkpoint; this bound is what the base class sums at init."""
        return sum(t[0].numel() * t.element_size() for t in self._tensors())

    def stash(self, slot, position, window_beg = None):
        """One slot's checkpoint. Ring row i holds position window_beg + i, so only rows
        [0, position - window_beg) carry history: the rest are future rows the next forward
        writes before it reads them. DSV41State.stash passes window_beg; a caller that knows
        only the position keeps every row below it, as V4 does."""
        n = position if window_beg is None else position - window_beg
        out = [self.ring[slot, :max(0, min(self.ring_rows, n))].cpu()]
        out += [t[slot].cpu() for t in self._tensors()[1:]]
        return out

    def unstash(self, slot, stashed, position):
        it = iter(stashed)
        ring = next(it)
        self.ring[slot].zero_()  # never leave stale rows below the restored window
        self.ring[slot, :ring.shape[0]].copy_(ring)
        for t in self._tensors()[1:]:
            t[slot].copy_(next(it))

    def tp_export(self, plan):
        raise NotImplementedError("Tensor-parallel loading is not supported for DeepSeek-V4.1 layer states")


class DSV41State(DSV4State):
    """
    Job-level recurrent state for DeepSeek-V4.1 (the model's recurrent_state_cls).

    V4's DSV4State bookkeeping (position, page-aligned window_beg, wshift applied in
    post_advance, stash/unstash over every recurrent layer) carries over unchanged; it
    aggregates whatever layer states the Cache holds -- DSV41LayerState on the 40 attention
    layers and, once registered, the engram's DSV41EngramState -- through the common
    layer-state API (clear, get_checkpoint_size, stash, unstash, rewind). Three things differ:

      * rollback_capacity() is the largest rewind that every layer state survives (the
        minimum of their limits). V4 reports position - window_beg, the rows still present
        in the SWA ring; but after a window rebase that can be as little as w - 1, and
        rewinding by all of it would leave the next query with a truncated window. Here the
        ring must keep a FULL window below the rewound position (unless window_beg is 0,
        when all history is present). Layer states with a position-modulo ring (the rate-2
        compressor carry) report rollback_limit(), a bound below the high-water position:
        rows below hw - ring size were overwritten no matter where the job has since rewound
        to. The engram's id ring needs no limit of its own: no capacity reported here reaches
        more than the SWA ring's rows (768 on V4.1-Flash) below the high-water position, and
        DSV41EngramState refuses a window whose ring it could not cover with its lookback.
        The generator falls back to a stashed checkpoint and replays whenever a rewind
        exceeds this, so a conservative figure only costs time. guaranteed_rollback, inherited
        from V4 (PAGE_SIZE), is therefore not a floor of this capacity: right after the window
        moves up a page it can be anywhere from 0 to PAGE_SIZE - 1 (a 2175-token prefill in one
        chunk leaves window_beg 2048 and a capacity of 127 - 127 = 0). The generator reads
        guaranteed_rollback only for its banned-string length margin, so a banned-string rewind
        right after a large chunk takes the checkpoint-and-replay path.

      * rewind() also forwards to every layer state's rewind(), as the recurrent-layer
        contract expects (all V4.1 layer states are position-indexed, so these are no-ops
        today; the call keeps a future stateful one correct).

      * stash() declares the bytes it actually stashed as checkpoint_size, the figure the
        generator's checkpoint cache budgets host RAM by.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # furthest position whose rows the position-modulo rings have seen (set already when
        # the base constructor restored a checkpoint through unstash)
        self.__dict__.setdefault("high_water", self.position)

    def post_advance(self):
        super().post_advance()
        self.high_water = max(self.high_water, self.position)

    def stash(self):
        """V4's checkpoint with two changes. The SWA rings keep only the rows that hold
        history (see DSV41LayerState.stash). And checkpoint_size is the byte count actually
        stashed: RecurrentCache evicts against it to stay within its host-RAM budget, and the
        per-layer estimates V4 sums would declare a V4.1 checkpoint at a quarter of its size
        (4.1x too small). V4.1 refuses tensor-parallel loading, so there is no TP branch."""
        stashed = {"position": self.position, "window_beg": self.window_beg}
        size = 0
        for k, l in self.cache.get_all_recurrent_layers().items():
            if isinstance(l, DSV41LayerState):
                s = l.stash(self.slot, self.position, self.window_beg)
                size += _nbytes(s)
            else:
                # another package's layer state: its own figure if the walk misses a tensor
                s = l.stash(self.slot, self.position)
                size += max(_nbytes(s), l.get_checkpoint_size())
            stashed[k] = s
        stashed["checkpoint_size"] = size
        # the carry rings are stashed as they are, so their valid span is still measured from
        # the high-water position
        stashed["high_water"] = self.high_water
        return stashed

    def unstash(self, stashed: dict):
        super().unstash(stashed)
        self.high_water = max(stashed.get("high_water", self.position), self.position)

    def reset(self):
        super().reset()
        self.high_water = 0

    def _window(self) -> int:
        w = 1
        for l in self.cache.get_all_recurrent_layers().values():
            w = max(w, getattr(l, "window", 1) or 1)
        return w

    def rollback_capacity(self):
        cap = self.position - self.window_beg
        if self.window_beg > 0:
            cap -= self._window() - 1
        behind = max(self.high_water - self.position, 0)
        for l in self.cache.get_all_recurrent_layers().values():
            lim = getattr(l, "rollback_limit", None)
            lim = lim() if callable(lim) else None
            if lim is not None:
                cap = min(cap, lim - behind)
        return max(cap, 0)

    def rewind(self, num_tokens: int):
        assert num_tokens <= self.rollback_capacity(), \
            f"DSV41State: rewind {num_tokens} exceeds capacity {self.rollback_capacity()}"
        last_history = self.last_history
        self.position -= num_tokens
        self.last_history = 0
        for l in self.cache.get_all_recurrent_layers().values():
            l.rewind(self.slot, last_history, num_tokens)


# ---------------------------------------------------------------------------------------------
# Per-forward device memo

class DeviceMemo:
    """
    Small per-forward tensors that a layer on another device than their producer needs in the
    stateless path (no Cache, where a layer split may fall anywhere) -- the top-k selection of
    its index source (2 KiB per row, 4 MiB per 2048-row chunk) and the candidate blocks
    (8 KiB per row) -- moved once per (key, device) and shared by every reader on that
    device. The memo lives in params["dsv41_dev_memo"] and is only valid for one forward:
    prepare_inputs must reset it together with dsv41_topk and the other per-forward keys.

    An entry remembers the tensor it was made from and is remade when a different tensor is
    passed under the same key (a producer that republished). That catches a stale memo
    against a freshly allocated selection, but not against one buffer object refilled in
    place, which is why the per-forward reset is still required.
    """

    PARAMS_KEY = "dsv41_dev_memo"

    @staticmethod
    def _norm(device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    @staticmethod
    def _move(t: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Stream-ordered when both ends are CUDA; a CPU destination must block
        return to_device(t, device, non_blocking = t.device.type == "cuda" and device.type == "cuda")

    @staticmethod
    def get(params: dict, key: tuple, t: torch.Tensor | None, device) -> torch.Tensor | None:
        if t is None:
            return None
        device = DeviceMemo._norm(device)
        if t.device == device:
            return t
        memo = params.get(DeviceMemo.PARAMS_KEY)
        if memo is None:
            memo = params[DeviceMemo.PARAMS_KEY] = {}
        mk = (key, device)
        hit = memo.get(mk)
        if hit is not None and hit[0] is t:
            return hit[1]
        moved = DeviceMemo._move(t, device)
        memo[mk] = (t, moved)
        return moved
