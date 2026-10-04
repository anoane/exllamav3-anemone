# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
DeepSeek-V4.1 attention pieces that differ from V4.

Derived in part from vLLM vllm/models/deepseek_v4_1/{attention,compressor}.py
(Apache-2.0); see NOTICE.

Two structural differences from V4, both confirmed against the tensors of the
DeepSeek-V4.1-Flash checkpoint:

1. Compressors and indexers are materialised only on the layers that own
   them -- kv_source_layer_ids and index_source_layer_ids respectively.
   Consumers carry no such tensors and read the source's published state.

2. The indexer has no local compressor. V4 builds
   ``{key}.indexer.compressor.*`` over hidden states; V4.1 instead derives
   the index key from the kv-source compressor's latent:

       k = k_norm(wk(latent))        wk: head_dim -> index_head_dim

   so there is no hidden-state K GEMM here. An index source that is not also
   a kv source therefore owns no ``wk``/``k_norm`` at all and borrows the kv
   source's paged K.

Observed shapes (DeepSeek-V4.1-Flash, hidden 5120, q_lora 1280, head_dim 512,
index_n_heads 32, index_head_dim 128):

    compressor.wkv          5120 -> 512      every kv source
    compressor.wgate        5120 -> 512      ratio-2 kv sources only
    compressor.norm         RMSNorm(512)
    indexer.wq_b            1280 -> 4096     every index source
    indexer.weights_proj    5120 -> 32       every index source
    indexer.wk               512 -> 128      only index sources that own K
    indexer.k_norm          RMSNorm(128)     only index sources that own K
"""

from __future__ import annotations

import math

import torch

from .linear import Linear
from .rmsnorm import RMSNorm
from .dsv4 import DSV4Attention, _ext_rope
from ..ext import exllamav3_ext as ext
from ..util.rope import RopeStyle
from .attention_fn.dsa_triton import dsa_attn, auto_splits, SPLIT_MAX_ROWS
from ..util.tensor import get_for_device, g_tensor_cache
from ..util.device_copy import to_device
from ..constants import PAGE_SIZE
from ..model.math_policy import (
    STABLE_ARITHMETIC, EXACT_ROWS, EXACT_ROWS_MAX, EXACT_ROWS_CAP_MGEMM, EXACT_ROWS_FORM_SELECT,
    exact_rows_native, exact_rows_forms,
)
from ..cache.dsv41 import CacheLayer_dsv41, DSV41LayerState, DeviceMemo
from ..cache.dsv41_replica import sync_pool_replica
from .dsv41_select import select_topk, SelectResult, INT32_LIMIT
from .dsv41_ablation import ablated
from .dsv41_compress import fused_compress
from . import dsv41_rounding as rounding
from ..architecture.dsv41 import numerics as dsv41_numerics
from .dsv41_cached import emission_range, entry_rows, rope_positions, CompressCarry, ring_update

# EXL3_EXACT_ROWS leaves dsa_attn one call for the rows of a verify forward: every such call must
# take the split softmax of a one-row step, which dsa_attn gives calls of up to SPLIT_MAX_ROWS rows
assert EXACT_ROWS_MAX <= SPLIT_MAX_ROWS, \
    f"EXACT_ROWS_MAX {EXACT_ROWS_MAX} exceeds dsa_attn's split-softmax row limit {SPLIT_MAX_ROWS}"

# EXL3_EXACT_ROWS: the row-exact entry points of the extension (0 without the switch, and with an
# extension built before them: the Python row loops)
ROWS_NATIVE = exact_rows_native(ext)
# EXL3_EXACT_ROWS: the batched forms that are Python alone (math_policy.exact_rows_forms): here
# EXACT_ROWS_FORM_SELECT, one selection pass for the rows of a flagged call (_index_select)
ROWS_FORMS = exact_rows_forms()


def check_row_limit(key: str, rows: int, num_heads: int, head_dim: int):
    """
    Refuse a forward whose query tensor the int32 kernels cannot address: ext.rope and dsa_attn
    index q (and dsa_attn its output) as (row * n_heads + head) * head_dim in int32, so past
    2^31 elements (65,536 rows at 64 x 512) they would read and write below the tensor.
    """
    if rows * num_heads * head_dim >= INT32_LIMIT:
        raise ValueError(
            f"{key}: {rows} query rows x {num_heads} heads x {head_dim} exceed the int32 "
            f"addressing of ext.rope and dsa_attn (at most {(INT32_LIMIT - 1) // (num_heads * head_dim)} "
            f"rows per forward); run long prompts through the cached path in chunks")


def rope_angle_table(inv_freq: torch.Tensor, positions: torch.Tensor, bsz: int = 1) -> torch.Tensor:
    """
    (bsz, n, rd / 2) fp32 rotation angles for ext.rope's table mode.

    ext.rope evaluates __sinf / __cosf(position * inv_freq), whose error grows with the
    argument: against an accurate rotation of the same fp32 angle, a host emulation of the
    intrinsics' argument reduction puts a rotated unit vector about 1.3e-4 off on the
    highest-frequency pair at position 1000 and 0.12 off at 1,048,575, while dsa_attn
    de-rotates with accurate trig (tests/test_dsv41_rope_gpu_.py requires plain ext.rope to
    fail its gate at 1M). Here the angle is formed in fp32 as DeepSeek forms it (outer(t,
    freqs) in fp32), then reduced to [-pi, pi) in fp64, where __sinf / __cosf are accurate to
    about 2^-21: the rotation then carries only fp16 rounding at any position. The kernel
    strides the table by batch row, so a batch gets one copy per row.
    """
    ang = positions.to(torch.float).unsqueeze(-1) * inv_freq.to(torch.float).unsqueeze(0)
    ang = torch.remainder(ang.double() + math.pi, 2 * math.pi) - math.pi
    t = ang.float().unsqueeze(0)
    if bsz > 1:
        t = t.expand(bsz, -1, -1).contiguous()
    return t


class DSV41Indexer:
    """
    V4.1 sparse-attention indexer.

    Present only on ``index_source_layer_ids``. ``owns_k`` is true when the
    layer is also a kv source: only then does it project its own index key
    from the compressor latent. Non-owning sources share the kv source's K.
    """

    def __init__(
        self,
        config,
        key: str,
        *,
        hidden_size: int,
        q_lora_rank: int,
        head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        owns_k: bool,
        rms_norm_eps: float = 1e-6,
        qmap: str | None = None,
        qbits_key: str = "bits",
    ):
        self.key = key
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.owns_k = owns_k

        # query side: q_lora_rank -> index_n_heads * index_head_dim
        self.wq_b = Linear(
            config, f"{key}.wq_b",
            q_lora_rank, index_n_heads * index_head_dim,
            qmap=qmap, out_dtype=torch.half, qbits_key=qbits_key,
        )
        # one scalar weight per index head, from the hidden state. Linear pads its output to
        # 128 columns by default (zero-filled); the scoring kernel indexes the weights as a
        # contiguous (seq, index_n_heads) tensor, so the padding must not reach it -- with it,
        # row r read row r / 4's weights when r % 4 == 0 and zeros otherwise
        self.weights_proj = Linear(
            config, f"{key}.weights_proj",
            hidden_size, index_n_heads,
            qmap=qmap, out_dtype=torch.float, trim_padded_out=True,
        )
        # key side exists only where this layer owns K
        if owns_k:
            self.wk = Linear(
                config, f"{key}.wk",
                head_dim, index_head_dim,
                qmap=qmap, out_dtype=torch.half, qbits_key=qbits_key,
            )
            self.k_norm = RMSNorm(config, f"{key}.k_norm", rms_norm_eps)
        else:
            self.wk = None
            self.k_norm = None

    def produce_k(self, latent, params):
        """
        Index keys from the compressor latent: ``k_norm(wk(latent))``, pre-RoPE.

        Only an owner runs this. The reference projects the full-length latent
        buffer and lets the store kernel skip non-boundary rows; here the
        caller has already densified the closed groups, so every row is live.

        latent   [n_groups, head_dim]
        returns  [n_groups, index_head_dim] fp32
        """
        assert self.owns_k, f"{self.key} does not own index K"
        k = self.wk.forward(latent.half(), params)
        return self.k_norm.forward(k, params, out_dtype = torch.float)

    def modules(self) -> list:
        m = [self.wq_b, self.weights_proj]
        if self.owns_k:
            m += [self.wk, self.k_norm]
        return m

    def expected_keys(self) -> list[str]:
        k = [f"{self.key}.wq_b", f"{self.key}.weights_proj"]
        if self.owns_k:
            k += [f"{self.key}.wk", f"{self.key}.k_norm"]
        return k


def compressor_expected_keys(key: str, compress_ratio: int) -> list[str]:
    """
    Tensor stems a V4.1 compressor materialises.

    ``wgate`` exists only at compression rate 2. A ratio-1 layer keeps a
    full-length compressed cache and needs no gate -- verified on the
    checkpoint, where layer 20, the only rate-1 kv source, carries norm and
    wkv but no wgate.
    """
    k = [f"{key}.norm", f"{key}.wkv"]
    if compress_ratio == 2:
        k.append(f"{key}.wgate")
    return k


class SourcePools:
    """
    Compressed state one kv source publishes for the layers that share it.

    exllamav3 has no equivalent of vLLM's static forward context, so the
    source deposits this in ``params`` under ``dsv41_pools`` keyed by its own
    layer index and its consumers look it up. ``entries`` is the number of
    valid rows, i.e. how many groups have closed.

    ``pool_idx`` is present only when the source owns index K, which for
    DeepSeek-V4.1-Flash is every kv source; an index source that is not a kv
    source (24/28/32/36) reads the pool of the kv source below it (20).
    """

    __slots__ = ("pool_c", "pool_r", "pool_idx", "entries", "compress_ratio", "first_group")

    def __init__(self, pool_c, pool_r, pool_idx, entries, compress_ratio, first_group = 0):
        self.pool_c = pool_c
        self.pool_r = pool_r
        self.pool_idx = pool_idx
        self.entries = entries
        self.compress_ratio = compress_ratio
        self.first_group = first_group


class DSV41Compressor:
    """
    V4.1 KV compressor, materialised only on ``kv_source_layer_ids``.

    Holds the weights; the pooling itself lives in
    ``architecture/dsv41/compressor.py`` so it can be checked on CPU.
    ``wgate`` exists only at rate 2 -- at rate 1 the softmax is over one
    element and cancels, and the checkpoint has no such tensor.
    """

    def __init__(
        self,
        config,
        key: str,
        *,
        hidden_size: int,
        head_dim: int,
        compress_ratio: int,
        rms_norm_eps: float = 1e-6,
        qmap: str | None = None,
        qbits_key: str = "bits",
        select_hq_bits = None,
    ):
        assert compress_ratio in (1, 2), \
            f"V4.1 compressor supports rate 1 and 2, got {compress_ratio}"
        self.key = key
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.rms_norm_eps = rms_norm_eps

        self.wkv = Linear(
            config, f"{key}.wkv", hidden_size, head_dim,
            qmap = qmap, out_dtype = torch.float, qbits_key = qbits_key,
            select_hq_bits = select_hq_bits,
        )
        self.wgate = Linear(
            config, f"{key}.wgate", hidden_size, head_dim,
            qmap = qmap, out_dtype = torch.float, qbits_key = qbits_key,
            select_hq_bits = select_hq_bits,
        ) if compress_ratio == 2 else None
        # Pool raw projections in FP32. Narrowing a projection first can overflow or
        # erase differences between pair logits before the softmax. This preserves
        # the output range of the engine's quantized Linear implementation; it does
        # not recover original unquantized weights or change its GEMM policy.
        self.norm = RMSNorm(config, f"{key}.norm", rms_norm_eps)

    def modules(self) -> list:
        return [m for m in (self.wkv, self.wgate, self.norm) if m is not None]

    def expected_keys(self) -> list[str]:
        return compressor_expected_keys(self.key, self.compress_ratio)

    def forward(self, x, params, position: int, state = None):
        """
        Pool one chunk into pre-RoPE latents.

        x         [seq, hidden]
        returns   (latent [n_closed, head_dim] fp32, first_group)

        At rate 1 every row closes its own group and no state is touched, so
        the whole chunk comes back. At rate 2 a trailing odd row is carried in
        ``state`` when one is given and dropped when it is not, matching the
        HF ``cache = None`` semantics V4's stateless path already follows.
        """
        from ..architecture.dsv41.compressor import compress_chunk

        kv = self.wkv.forward(x, params).float()
        score = self.wgate.forward(x, params).float() if self.wgate is not None else None
        return compress_chunk(
            kv, score, position, self.compress_ratio,
            self.norm.weight, self.rms_norm_eps, state,
        )


class DSV41Attention(DSV4Attention):
    """
    DeepSeek-V4.1 attention.

    The projection front (q_a / q_norm / q_b, wkv, kv_norm), the attention
    sink, the eight-slice grouped wo_a and the dsa_attn epilogue are identical
    to V4 and are inherited untouched. What differs is everything around the
    compressed cache:

    * ``compress_ratios[i]`` is a rate, not a kind. 0 is a sliding-window
      layer with no compressed state at all; 1 keeps a full-length compressed
      cache (one entry per token, no gate); 2 pools token pairs. V4's three
      layer kinds do not survive, so the base class is driven with the kind
      that gives the right rope table and caps -- "sliding" at rate 0,
      otherwise "hca", which is V4's non-overlapping compressed kind.

    * Compressors exist only on ``kv_source_layer_ids`` and indexers only on
      ``index_source_layer_ids``. Consumers hold no such weights; they read
      the pools their source published. ``tp_defer_compressors`` suppresses
      V4's own compressor/indexer construction so this class can build the
      V4.1 shapes instead.

    * V4's indexer is a second full compressor over hidden states. V4.1
      derives index keys from the kv source's latent with one small GEMM, so
      an index source that is not also a kv source owns no wk/k_norm and
      reads the kv source's index pool (layers 24/28/32/36 read layer 20's).

    Two paths share the projection front (_project_qkv, no per-head q norm),
    the selection and the grouped epilogue:

    * stateless (attn_mode flash_attn_nc, _forward_nc): each kv source
      rebuilds its pools for the whole prompt and publishes them through
      ``params["dsv41_pools"]``; consumers on another device get a mirror.

    * cached (attn_mode flash_attn, _forward_cached): the pools are the kv
      source's paged CacheLayer_dsv41 -- the ONLY pool in its group -- and
      every layer keeps its own sliding ring in a DSV41LayerState. A source
      compresses and stores this chunk's closed entries before attention
      (a rate-2 group left open at a chunk boundary waits as one raw fp32 row
      in a position-indexed carry ring); index sources select and publish
      ``params["dsv41_topk"][(layer, row)]``, consumers reuse it. Every layer
      that reads a pool, a selection or candidate blocks sits on the device of
      their producer (the load guard, architecture/dsv41/placement.py, refuses
      a layer split that would separate them while a Cache is attached), except
      across the split of an explicit placement: a consumer past it reads a
      replica of its source's pool that the first layer past the split owns
      and refreshes (set_pool_route), and the selections and candidate blocks
      it reads move once per forward.

    Both paths apply the attention numerics (config.dsv41_numerics, architecture/dsv41/
    numerics.py), read on every call: the index query and keys are rounded after RoPE, the
    window KV after the projection front, and compressed entries before they are stored.
    """

    def __init__(
        self,
        config,
        key: str,
        layer_idx: int,
        *,
        compress_ratio: int,
        is_kv_source: bool,
        is_index_source: bool,
        index_owns_k: bool,
        kv_source_layer: int | None,
        index_source_layer: int | None,
        uses_candidates: bool = False,
        hidden_size: int,
        q_lora_rank: int,
        head_dim: int,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        rms_norm_eps: float = 1e-6,
        qmap: str | None = None,
        qbits_key: str = "bits",
        select_hq_bits = None,
        **kwargs,
    ):
        assert compress_ratio in (0, 1, 2), \
            f"layer {layer_idx}: compress_ratio {compress_ratio}, expected 0, 1 or 2"
        if compress_ratio == 0:
            assert not is_kv_source and not is_index_source, \
                f"layer {layer_idx}: a sliding layer cannot be a source"
        else:
            assert kv_source_layer is not None and index_source_layer is not None, \
                f"layer {layer_idx}: compressed layer with no source"
        assert index_owns_k == (is_index_source and is_kv_source), \
            f"layer {layer_idx}: index K is owned exactly by index sources that are kv sources"

        super().__init__(
            config, key, layer_idx,
            "sliding" if compress_ratio == 0 else "hca",
            hidden_size = hidden_size,
            q_lora_rank = q_lora_rank,
            head_dim = head_dim,
            compress_rate = compress_ratio if compress_ratio else None,
            index_n_heads = index_n_heads,
            index_head_dim = index_head_dim,
            index_topk = index_topk,
            rms_norm_eps = rms_norm_eps,
            qmap = qmap,
            qbits_key = qbits_key,
            select_hq_bits = select_hq_bits,
            tp_defer_compressors = True,
            **kwargs,
        )

        self.compress_ratio = compress_ratio
        self.is_kv_source = is_kv_source
        self.is_index_source = is_index_source
        self.owns_index_k = index_owns_k
        self.kv_source_layer = kv_source_layer
        self.index_source_layer = index_source_layer
        self.uses_candidates = uses_candidates
        self.compressor = None
        self.indexer = None

        if is_kv_source:
            self.compressor = DSV41Compressor(
                config, f"{key}.compressor",
                hidden_size = hidden_size,
                head_dim = head_dim,
                compress_ratio = compress_ratio,
                rms_norm_eps = rms_norm_eps,
                qmap = qmap,
                qbits_key = qbits_key,
                select_hq_bits = select_hq_bits,
            )
            for m in self.compressor.modules():
                self.register_submodule(m)

        if is_index_source:
            self.indexer = DSV41Indexer(
                config, f"{key}.indexer",
                hidden_size = hidden_size,
                q_lora_rank = q_lora_rank,
                head_dim = head_dim,
                index_n_heads = index_n_heads,
                index_head_dim = index_head_dim,
                index_topk = index_topk,
                owns_k = index_owns_k,
                rms_norm_eps = rms_norm_eps,
                qmap = qmap,
                qbits_key = qbits_key,
            )
            for m in self.indexer.modules():
                self.register_submodule(m)
            # The inherited top-k scoring reads these; V4.1 keeps the same two
            # tensor names under {key}.indexer, so aliasing is enough to reuse it
            self.idx_wq_b = self.indexer.wq_b
            self.idx_weights = self.indexer.weights_proj

        for m in self.modules:
            m.q_half_bits = False

        # Two-level selection: the candidate source publishes coarse blocks that
        # the indexers above it mask to (DeepSeek: candidate_source_layer = 20)
        cand_src = getattr(config, "candidate_source_layer_id", -1) if config is not None else -1
        self.is_candidate_source = is_index_source and cand_src == layer_idx
        self.candidate_block_size = (getattr(config, "candidate_block_size", 0) or 8) if config is not None else 8
        self.candidate_topk_blocks = (getattr(config, "candidate_topk_blocks", 0) or 2048) if config is not None else 2048

        # Cached-path state. Every layer keeps its own sliding ring (V4.1's state class,
        # which never dereferences a compressor on consumers); only kv sources own a
        # pool. pool_owner_layer is the layer whose Cache entry this one reads: its kv
        # source by default, or a replica owner past a device split (set_pool_route)
        self.layer_state_cls = DSV41LayerState
        self.caps["kv_cache"] = bool(is_kv_source)
        self.pool_owner_layer = kv_source_layer
        self.replica_of = None
        self.replica_with_index_k = False

    def set_pool_route(self, owner_layer: int, replica_of: int | None = None,
                       with_index_k: bool = False) -> None:
        """
        Which Cache pool this compressed layer reads. Called before the Cache is built
        (Cache discovers owners through caps['kv_cache'], and the model memoizes that list
        on first use).

          set_pool_route(L, replica_of = S)   layer L owns a replica of kv source S's pool,
                                              on L's device, refreshed by L each forward
                                              with the entries S wrote in that forward
          set_pool_route(L)                   read layer L's pool (a replica owner's, or the
                                              kv source's when L is the source)

        with_index_k: the replica also mirrors pool_idx (needed when an index source sits
        past the split and borrows the source's index K; with the split at 12 on
        DeepSeek-V4.1-Flash, layers 12 and 13 only consume and the replica carries no index K).
        """
        assert self.compress_ratio, f"{self.key}: a sliding layer reads no pool"
        if replica_of is not None:
            assert owner_layer == self.layer_idx, \
                f"{self.key}: a replica is owned by the layer that refreshes it"
            assert replica_of == self.kv_source_layer, \
                f"{self.key}: replica of {replica_of}, but the kv source is {self.kv_source_layer}"
            assert not self.is_kv_source, f"{self.key}: a kv source reads its own pool"
        elif owner_layer == self.layer_idx:
            assert self.is_kv_source, \
                f"{self.key}: routes to itself without a replica, but it owns no pool"
        self.pool_owner_layer = owner_layer
        self.replica_of = replica_of
        self.replica_with_index_k = bool(with_index_k) if replica_of is not None else False
        self.caps["kv_cache"] = bool(self.is_kv_source or replica_of is not None)

    def cache_layer_type(self, default, kwargs: dict):
        """
        CacheLayer_dsv41 for kv sources, CacheLayer_dsv41_replica for a replica owner.
        A quantized Cache request packs the pool's nope part at k_bits, as V4 does.
        """
        from ..cache import CacheLayer_quant
        kw = {}
        if issubclass(default, CacheLayer_quant) and (self.head_dim - self.rope_head_dim) % 32 == 0:
            kw = {"k_bits": kwargs["k_bits"], "v_bits": kwargs.get("v_bits")}
        if self.replica_of is not None:
            from ..cache.dsv41_replica import CacheLayer_dsv41_replica
            return CacheLayer_dsv41_replica, dict(kw, with_index_k = self.replica_with_index_k)
        assert self.is_kv_source, f"{self.key}: only kv sources and replica owners own a pool"
        return CacheLayer_dsv41, kw

    def load(self, device: torch.device, **kwargs):
        """
        V4's load(), minus its V4-compressor branch.

        DSV4Attention.load() treats a non-None ``self.compressor`` /
        ``self.indexer`` as V4 objects: it fetches ``{key}.ape`` (the V4
        within-window position bias, absent from V4.1 checkpoints) and calls
        ``make_bc`` (a DSV4Compressor method). Both raise for V4.1. The weights
        themselves are registered submodules and load through Module.load
        regardless, so hiding the two attributes for the duration of the call
        skips exactly that branch and nothing else -- and anything upstream
        later adds to V4's load() still applies here.
        """
        comp, idx = self.compressor, self.indexer
        self.compressor = self.indexer = None
        try:
            super().load(device, **kwargs)
        finally:
            self.compressor, self.indexer = comp, idx

    def unload(self):
        """Mirror of load(): V4's unload() without its ape/unmake_bc branch."""
        comp, idx = self.compressor, self.indexer
        self.compressor = self.indexer = None
        try:
            super().unload()
        finally:
            self.compressor, self.indexer = comp, idx

    def get_tensors(self):
        """V4's version adds {compressor,indexer}.ape, which V4.1 has no equivalent of."""
        return {f"{self.key}.attn_sink": self.sinks.contiguous()}

    def weights_numel(self):
        """Submodules plus the per-head sinks. V4's version adds its ape tables and reads
        V4-only compressor fields."""
        return sum(m.weights_numel() for m in self.modules) + self.num_q_heads

    def make_tp_allocation(self, options: dict):
        raise NotImplementedError("Tensor-parallel loading is not supported for DeepSeek-V4.1 attention")

    def tp_export(self, plan, producer):
        raise NotImplementedError("Tensor-parallel loading is not supported for DeepSeek-V4.1 attention")

    def _numerics(self) -> dsv41_numerics.Numerics:
        """
        The attention numerics (config.dsv41_numerics, validated when it is set), read on every
        call so that one loaded model can switch between settings. A config without the
        attribute gets the default.
        """
        return dsv41_numerics.parse(getattr(self.config, "dsv41_numerics", None))

    def _project_qkv(self, x, params, position):
        """
        V4's projection front WITHOUT the per-head q norm.

        DeepSeek's V4.1 reference (inference/model.py, Attention.forward):

            qr = q_norm(wq_a(x))                   # RMSNorm over the q_lora latent
            q  = wq_b(qr).unflatten(heads)         # no per-head norm
            kv = kv_norm(wkv(x))                   # weighted, over head_dim
            rope(q[..., -rd:]); rope(kv[..., -rd:])

        and its sparse_attn is a plain q.k * scale. V4's _project_qkv also applies an
        UNWEIGHTED per-head RMSNorm to q (a ones weight in its fused rope call);
        for V4.1 that forces every head's query to a fixed length and discards the
        magnitude the attention sinks were trained against -- wrong at every
        length. vLLM's V4.1 passes False for the same step to its fused kernel.

        ext.rope takes its q and k norm weights together or not at all, so kv_norm
        runs as its own step first (the reference's order: norm, then rotate) and
        the rope call is norm-free. Its angles come from rope_angle_table.
        """
        bsz, seq, _ = x.shape
        rd = self.rope_head_dim
        q_res = self.q_norm.forward(self.q_a.forward(x, params), params, out_dtype = torch.half)
        q = self.q_b.forward(q_res, params).view(bsz, seq, self.num_q_heads, self.head_dim)
        kv = self.wkv.forward(x, params)
        kv = self.kv_norm.forward(kv, params, out_dtype = torch.half).view(bsz, seq, 1, self.head_dim)
        # the rotation reads its angles from a per-position table (rope_angle_table)
        tab = self._range_table(params, self.layer_type != "sliding", position, seq, bsz, x.device)
        ext.rope(
            q, q, kv, kv,
            tab, 0, None, None,
            int(RopeStyle.GPTJ), 1.0, None, None,
            self.rms_norm_eps, 0.0, 0.0, 0, 1, self.head_dim - rd,
        )
        kv = kv.view(bsz, seq, self.head_dim)
        policy = self._numerics()
        if policy.window:
            # a new tensor: never round a projection output or scratch someone else reads
            kv = rounding.round_window_(policy, kv.clone(), rd)
        return q_res, q, kv

    def expected_keys(self) -> list[str]:
        """Tensor stems this layer's V4.1-specific pieces materialise."""
        k = []
        if self.compressor is not None:
            k += self.compressor.expected_keys()
        if self.indexer is not None:
            k += self.indexer.expected_keys()
        return k

    def _entry_rope_positions(self, first_entry: int, n: int, device):
        """(1, n) int32 rotation positions of entries first_entry .. first_entry + n - 1:
        (first_entry + j) * m. EXL3_DSV41_ABLATE=rope_consecutive deliberately uses the
        wrong first_entry * m + j instead (validation harness only)."""
        m = self.compress_ratio
        j = torch.arange(n, device = device, dtype = torch.int)
        pos = first_entry * m + j if ablated("rope_consecutive") else \
            rope_positions(first_entry + j, m)
        return pos.unsqueeze(0).contiguous()

    @staticmethod
    def _rope_memo(params: dict, key: tuple, build):
        """Angle tables are built once per forward per (device, table, positions) and shared
        by every layer that rotates at those positions (params["dsv41_rope"])."""
        memo = params.get("dsv41_rope")
        if memo is None:
            memo = params["dsv41_rope"] = {}
        t = memo.get(key)
        if t is None:
            t = memo[key] = build()
        return t

    def _range_table(self, params, compress: bool, pos0: int, n: int, bsz: int, device):
        """Angle table for rows at positions pos0 .. pos0 + n - 1 (queries, window kv, index q),
        from the compressed layers' table or the sliding layers' one."""
        inv = self.inv_freq_compress if compress else self.inv_freq_main
        key = (str(device), "compress" if compress else "main", "range", pos0, n, bsz)
        return self._rope_memo(params, key, lambda: rope_angle_table(
            inv, torch.arange(pos0, pos0 + n, device = device), bsz))

    def _entry_table(self, params, first_entry: int, n: int, device):
        """Angle table for pool entries first_entry .. first_entry + n - 1 at
        _entry_rope_positions (the compressed table)."""
        key = (str(device), "compress", "entries", self.compress_ratio, first_entry, n,
               ablated("rope_consecutive"))
        return self._rope_memo(params, key, lambda: rope_angle_table(
            self.inv_freq_compress, self._entry_rope_positions(first_entry, n, device)[0]))

    def publish_pools(self, x, params, position, out_pools = None):
        """
        Run the compressor and deposit this source's pools for its group.

        Returns the SourcePools the group's consumers will read. Only a kv
        source calls this; the rope split follows V4's layout so dsa_attn can
        consume the result unchanged -- nope part in pool_c, rotated part in
        pool_r, rotated at the group's first token position.
        """
        assert self.compressor is not None, f"{self.key} is not a kv source"
        bsz, seq, _ = x.shape
        assert bsz == 1, "V4.1 pool publication is per sequence"
        m = self.compress_ratio
        rd = self.rope_head_dim
        hd = self.head_dim

        latent, first_group = self.compressor.forward(x[0], params, position)
        n = latent.shape[0]
        dev = x.device

        # dsa_attn addresses the pool through a block table, so the allocation
        # is rounded up to whole pages even though only n rows are live. The
        # pools are allocated per forward and live as long as params holds them.
        # Nothing drops them, or the selections, candidate blocks, mirrors and
        # memo copies kept next to them, while the forward runs: on the stateless
        # path they stay allocated after their last reader until params is reset
        # for the next forward. Tensor-cache scratch keyed by shape would instead
        # keep every prompt length's pools for the process's life
        cap = max(-(-n // PAGE_SIZE), 1) * PAGE_SIZE
        pool_c = torch.empty((cap, hd - rd), dtype = torch.half, device = dev)
        pool_r = torch.empty((cap, rd), dtype = torch.half, device = dev)
        pool_idx = None
        if self.owns_index_k:
            pool_idx = torch.empty((cap, self.index_head_dim), dtype = torch.half, device = dev)

        if n:
            # Entry g covers tokens [g*m, (g+1)*m) and rotates at the FIRST of them,
            # (p // m) * m, as the reference kernel indexes its cos/sin table. So
            # consecutive entries sit m positions apart: entry i of this chunk is at
            # (first_group + i) * m. Passing a scalar start position instead would
            # rotate them 1 apart -- wrong for every rate-2 entry after the first.
            tab = self._entry_table(params, first_group, n, dev)
            rot = latent[:, hd - rd:].half().view(1, n, 1, rd).contiguous()
            _ext_rope(rot, tab)
            nope = latent[:, :hd - rd].half().contiguous()
            policy = self._numerics()
            nope, rot = rounding.round_compressed_(policy, nope, rot)
            pool_c[:n].copy_(nope)
            pool_r[:n].copy_(rot.view(n, rd))
            if pool_idx is not None:
                k = self.indexer.produce_k(latent, params)
                k = k.half().view(1, n, 1, self.index_head_dim).contiguous()
                _ext_rope(k[..., -rd:], tab)
                rounding.round_index_(policy, k)
                pool_idx[:n].copy_(k.view(n, self.index_head_dim))

        pools = SourcePools(pool_c, pool_r, pool_idx, n, m, first_group)
        if out_pools is None:
            out_pools = params.setdefault("dsv41_pools", {})
        out_pools[self.layer_idx] = pools
        return pools

    def resolve_pools(self, params) -> SourcePools | None:
        """
        The pools this layer attends over, published by its kv source.

        A missing entry means the source did not run before this layer in this
        forward. A source on another device is fine: its pools are mirrored here.
        """
        if self.compress_ratio == 0:
            return None
        pools = params.get("dsv41_pools")
        if pools is None or self.kv_source_layer not in pools:
            raise RuntimeError(
                f"{self.key}: compressed-KV source layer {self.kv_source_layer} has "
                f"not published its pools this forward pass: the source did not run "
                f"before this layer in this forward (check that it did not fail)."
            )
        src = pools[self.kv_source_layer]
        assert src.compress_ratio == self.compress_ratio, \
            f"{self.key}: rate {self.compress_ratio} reading a rate-{src.compress_ratio} pool"
        return self._pools_on_my_device(src, params)

    def _pools_on_my_device(self, src: SourcePools, params) -> SourcePools:
        """
        The source's pools, on this layer's device.

        A layer split without a Cache may put a consumer on a different card
        from its kv source. In the stateless path the pool is rebuilt every forward, so
        the fix is a copy -- made once per (source, device) per forward and
        shared by every consumer on that device. At prefill scale that is
        small (entries x 1.3 KiB), but it is NOT the answer for decode at long
        context: a rate-1 pool at 1M tokens is ~1.3 GiB, and recopying it each
        step would dominate. The cached path instead keeps every pool on its
        readers' device, and across the split of an explicit placement mirrors
        only the entries appended since the last step (cache/dsv41_replica.py).
        """
        dev = self.device
        if dev is None or src.pool_c.device == dev:
            return src
        mirrors = params.setdefault("dsv41_pool_mirrors", {})
        key = (self.kv_source_layer, str(dev))
        m = mirrors.get(key)
        if m is None:
            mv = lambda t: None if t is None else to_device(t, dev, non_blocking = True)
            m = SourcePools(mv(src.pool_c), mv(src.pool_r), mv(src.pool_idx),
                            src.entries, src.compress_ratio, src.first_group)
            mirrors[key] = m
        return m

    def resolve_index_pool(self, params):
        """
        The index-key pool this layer scores against.

        Owners read their own; a non-owning index source reads the pool of the
        kv source below it, which is the same layer its compressed KV comes
        from, so one lookup answers both.
        """
        src = self.resolve_pools(params)
        if src is None or src.pool_idx is None:
            raise RuntimeError(
                f"{self.key}: index source has no K pool; kv source "
                f"{self.kv_source_layer} did not publish one"
            )
        return src.pool_idx

    def _forward_nc(self, x, params, out_dtype):
        """
        Stateless single-shot pass, V4.1 topology.

        Same shape as V4's: project, fill or find the compressed pool, pick
        top-k index entries, one dsa_attn over [sliding window ++ selected
        pool entries], grouped o_proj. Three things differ, all of them about
        who owns what:

        * only a kv source runs a compressor, and it publishes for its group
        * only an index source scores, and it reads its source's K pool
        * rate 1 means one entry per token, so the pool is as long as the
          context and the top-k is over every token, not every window

        The trailing partial group is dropped, matching the HF cache = None
        semantics V4's stateless path already follows.

        The stateless path has no history: its pools hold only this forward's
        entries while visibility is absolute, so it runs from position 0 only.
        """
        bsz, seq, _ = x.shape
        device = x.device
        position = params.get("position", 0)
        if position != 0:
            raise ValueError(
                f"{self.key}: the stateless path starts at position 0 (got {position}); it keeps "
                f"no history, so later rows would attend entries of the wrong tokens. Continue a "
                f"sequence through the cached path (a Cache and recurrent states)")
        if bsz != 1:
            # refused here, on every layer, rather than at the first kv source: publish_pools
            # compresses one sequence, so the rows of a batch would attend row 0's pools
            raise ValueError(
                f"{self.key}: the stateless path runs one sequence at a time (got a batch of "
                f"{bsz}); it publishes one pool per kv source per forward. Batch sequences "
                f"through the cached path (a Cache and recurrent states)")
        check_row_limit(self.key, bsz * seq, self.num_q_heads, self.head_dim)

        q_res, q, kv = self._project_qkv(x, params, position)

        w = self.sliding_window
        hpg = self.num_q_heads // self.o_groups
        hd = self.head_dim
        rd = self.rope_head_dim

        if self.is_kv_source:
            self.publish_pools(x, params, position)
        src = self.resolve_pools(params)

        if src is None:
            # Sliding-window layer: no compressed state, one dummy row so the
            # block table and pool arguments stay well formed
            pool_c = x.new_empty((1, hd - rd), dtype = torch.half)
            pool_r = x.new_empty((1, rd), dtype = torch.half)
            T, m = 0, 1
            bt = torch.zeros((1, 1), dtype = torch.int32, device = device)
        else:
            pool_c, pool_r = src.pool_c, src.pool_r
            T, m = src.entries, src.compress_ratio
            bt = torch.arange(pool_c.shape[0] // PAGE_SIZE, dtype = torch.int32,
                              device = device).unsqueeze(0)

        # Every compressed layer -- source or consumer -- attends over its own
        # sliding window plus the top-k pool entries chosen by the NEAREST INDEX
        # SOURCE at or below it (the reference's shared topk_indices_buffer). Only
        # index sources score; consumers reuse their source's selection. Below
        # index_topk entries every entry is selected, which is the dense path.
        indices, k_len = None, 0
        if T > self.index_topk:
            if self.is_index_source:
                indices, k_len = self._indexer_topk(
                    x, params, q_res, self.resolve_index_pool(params)[:T], T, position)
                params.setdefault("dsv41_topk", {})[(self.layer_idx, 0)] = (indices, k_len)
            elif not ablated("dense_consumers"):
                sel = params.get("dsv41_topk", {}).get((self.index_source_layer, 0))
                if sel is None:
                    raise RuntimeError(
                        f"{self.key}: index source layer {self.index_source_layer} has not "
                        f"published a top-k selection this forward pass")
                indices, k_len = sel
                # moved once per device and forward, shared by every consumer there
                indices = DeviceMemo.get(params, ("dsv41_topk", self.index_source_layer, 0),
                                         indices, device)

        outs = []
        for b in range(bsz):
            out = dsa_attn(
                q[b].half().contiguous(), pool_c, pool_r, bt, sinks = self.sinks,
                kv_chunk = kv[b].contiguous(), win_len = w, win_floor = position,
                indices = indices, k_len = k_len, pool_len = T, q_pos0 = position,
                compress_rate = m, scale = self.sm_scale,
                derot_inv_freq = self._rope_type_neg(), groups = self.o_groups,
                group_major = True,
                # dsa_attn's automatic choice gives calls of up to 8 query rows (decode) a split
                # softmax over key partitions, whose partial results combine in another order
                # than the one-pass softmax of longer calls; EXL3_STABLE_ARITHMETIC=1 keeps one
                # partition at every row count
                n_splits = 1 if STABLE_ARITHMETIC else 0,
                out = torch.empty((self.o_groups, seq, hpg * hd), dtype = torch.half,
                                  device = device),
                nc_chunk = bool(params.get("nc_chunk", False)),
            )
            outs.append(self._project_o_grouped(out.unsqueeze(1), params, out_dtype))
        return torch.cat(outs, dim = 0) if bsz > 1 else outs[0]

    def _indexer_topk(self, x, params, q_res, idx_pool, ec, pos0, q_idx_pre = None,
                      block_table = None, epp = 0):
        """
        Stateless-path selection (V4's hook, called by _forward_nc): the same
        _index_select the cached path uses, over the dense published pool, with
        the candidate stage in (V4's version has no candidates).
        """
        r = self._index_select(x, params, q_res, idx_pool, block_table, epp, pos0, ec, 0)
        return r.indices, r.k_len

    def _index_select(self, x, params, q_res, idx_pool, bt_row, epp, pos0, ec, b):
        """
        Lightning-indexer top-k of one sequence (row b), as DeepSeek's Indexer.forward:

            q      = wq_b(qr), roped (compress table) at the query positions
            w      = weights_proj(x) * index_head_dim^-0.5 * n_heads^-0.5
            score  = sum_h w_h * relu(q_h . k_j)       over visible entries j
            top-k  = the index_topk best, ascending, -1 padded

        The candidate source (layer 20) also publishes its coarse block list from the
        UNMASKED scores in params["dsv41_candidates"][b]; indexers above it mask their
        scores to those blocks first. Scoring and top-k are select_topk (modules/
        dsv41_select.py), tiled so that no (seq, pool) score matrix is ever built.
        """
        seq = x.shape[1]
        H, D, rd = self.index_n_heads, self.index_head_dim, self.rope_head_dim
        q_idx = self.idx_wq_b.forward(q_res, params).view(1, seq, H, D).contiguous()
        _ext_rope(q_idx[..., -rd:], self._range_table(params, True, pos0, seq, 1, q_idx.device))
        rounding.round_index_(self._numerics(), q_idx)
        # D^-0.5 * H^-0.5 = 2^-6 here: an exact rescale, identical to scaling in-kernel. The
        # scoring kernel reads (seq, H) contiguous: never hand it a padded projection
        wts = self.idx_weights.forward(x, params)[0]
        assert wts.shape[-1] == H, f"{self.key}: index head weights {tuple(wts.shape)}, expected (seq, {H})"
        wts = wts.float().contiguous() * (D ** -0.5 * H ** -0.5)

        cand_in = None
        if self.uses_candidates and not ablated("no_candidates"):
            cands = params.get("dsv41_candidates", {})
            if b not in cands:
                raise RuntimeError(
                    f"{self.key}: candidate source layer has not published its blocks for "
                    f"row {b} this forward pass (it must run, and run first, in every forward)")
            if cands[b] is not None:
                cand_in = DeviceMemo.get(params, ("dsv41_candidates", b), cands[b], q_idx.device)

        if EXACT_ROWS and params.get("exact_rows") and seq > 1:
            # EXL3_EXACT_ROWS: every row gets the selection a one-row step at its position makes
            # (its own entry count, hence its tile width, score stride and scoring kernel: the
            # scorer takes another kernel from 5 rows on). A flagged call never mixes dense and
            # selecting rows (DeepseekV41Model.exact_rows_span), and a dense row selects nothing.
            # The rows share one pass where their one-row calls share the tile width and need one
            # tile, on the GPU types that pass is verified on, with the scorer pinned to the kernel
            # of a one-row call (select_topk, one_row). Otherwise (the rows straddle a tile width,
            # more than one tile, another GPU type) select_topk returns None before it launches
            # anything and each row is the one-row call itself, as without EXACT_ROWS_FORM_SELECT
            m = self.compress_ratio
            if (pos0 + 1) // m <= self.index_topk:
                raise RuntimeError(
                    f"{self.key}: EXL3_EXACT_ROWS=1: a draft verification of {seq} rows at position "
                    f"{pos0} selects although its first row is still dense ({(pos0 + 1) // m} entries, "
                    f"index_topk {self.index_topk}); such a call must not carry params['exact_rows']")
            r = select_topk(
                q_idx[0], wts, idx_pool, bt_row = bt_row, epp = epp, pos0 = pos0, m = m,
                ec = (pos0 + seq) // m, topk = self.index_topk, cand_in = cand_in,
                want_cand = self.is_candidate_source, block = self.candidate_block_size,
                n_blocks = self.candidate_topk_blocks, one_row = True,
            ) if ROWS_FORMS & EXACT_ROWS_FORM_SELECT else None
            if r is None:
                rows = [select_topk(
                    q_idx[0][j:j + 1], wts[j:j + 1], idx_pool, bt_row = bt_row, epp = epp, pos0 = pos0 + j,
                    m = m, ec = (pos0 + j + 1) // m, topk = self.index_topk,
                    cand_in = None if cand_in is None else cand_in[j:j + 1],
                    want_cand = self.is_candidate_source, block = self.candidate_block_size,
                    n_blocks = self.candidate_topk_blocks,
                ) for j in range(seq)]
                r = SelectResult(
                    torch.cat([row.indices for row in rows], dim = 0), rows[0].k_len,
                    None if rows[0].cand is None else torch.cat([row.cand for row in rows], dim = 0))
        else:
            r = select_topk(
                q_idx[0], wts, idx_pool, bt_row = bt_row, epp = epp, pos0 = pos0,
                m = self.compress_ratio, ec = ec, topk = self.index_topk, cand_in = cand_in,
                want_cand = self.is_candidate_source, block = self.candidate_block_size,
                n_blocks = self.candidate_topk_blocks,
            )
        if self.is_candidate_source:
            params.setdefault("dsv41_candidates", {})[b] = r.cand
        return r

    # ------------------------------------------------------------------------------------
    # Cached path (attn_mode flash_attn)

    def row_plan(self, position: int) -> tuple[bool, int]:
        """
        What a one-row cached step at `position` decides from its position: (dense pool, softmax
        splits). Dense is _select_cached's choice for its ec = (position + 1) // m, the split count
        dsa_attn's automatic one (auto_splits) for the keys of that row: the window plus the
        row's visible entries when dense, plus index_topk selected ones otherwise. A call of
        several rows takes ONE such plan, from its last row, so only a call whose rows all have
        the same plan gives each row its one-row attention (EXL3_EXACT_ROWS).
        """
        w = self.sliding_window
        m = self.compress_ratio
        if not m:
            return True, auto_splits(w)
        ec = (position + 1) // m
        dense = ec <= self.index_topk or (not self.is_index_source and ablated("dense_consumers"))
        return dense, auto_splits(w + (ec if dense else self.index_topk))

    def _forward_cached(self, x, params, out_dtype):
        """
        Generation path: chunked prefill and token-by-token decode over the paged pools and
        per-slot rings. V4's forward() dispatch routes attn_mode flash_attn here. bsz > 1 is
        a per-row loop over params["recurrent_states"]; block-table rows correspond 1:1 to
        batch rows, as in V4.
        """
        if self.tp_mode:
            raise NotImplementedError("Tensor-parallel forward is not supported for DeepSeek-V4.1 attention")
        rsg = params["recurrent_states"]
        bsz = x.shape[0]
        assert len(rsg) >= bsz, f"{self.key}: {bsz} rows but {len(rsg)} recurrent states"
        inst = params.get("layer_instance", 0)
        # the autosplit's measuring forward reaches here without prepare_inputs
        params.setdefault("dsv41_topk", {})
        params.setdefault("dsv41_candidates", {})

        kl = bt = src_kl = bt_src = None
        if self.compress_ratio:
            kl = self._pool_layer(rsg[0], inst)
            bt = self._block_table(params, kl, rsg, bsz)
            if self.replica_of is not None:
                src_kl = rsg[0].cache.layers[(self.replica_of, inst)]
                bt_src = self._block_table(params, src_kl, rsg, bsz)

        outs = []
        for b in range(bsz):
            rs = rsg[b]
            rsl = self._get_rsl(rs, (self.layer_idx, inst))
            outs.append(self._forward_cached_row(
                x[b:b + 1], params, rs, rsl, out_dtype, b,
                kl, bt[b:b + 1] if bt is not None else None,
                src_kl, bt_src[b:b + 1] if bt_src is not None else None,
            ))
        return torch.cat(outs, dim = 0) if bsz > 1 else outs[0]

    def _pool_layer(self, rs, inst):
        """This layer's pool: its route owner's Cache entry, which must be on this device."""
        kl = rs.cache.layers.get((self.pool_owner_layer, inst))
        if kl is None:
            raise RuntimeError(
                f"{self.key}: the Cache has no pool for layer {self.pool_owner_layer} (instance "
                f"{inst}); only kv sources and replica owners allocate one")
        if kl.device is None or torch.device(kl.device) != torch.device(self.device):
            from ..architecture.dsv41.placement import DOC_REF, REPLICA_HINT, format_cuts, free_cuts
            raise RuntimeError(
                f"{self.key}: the compressed-KV pool it reads (layers.{self.pool_owner_layer}) is on "
                f"{kl.device}, but this layer is on {self.device}. A Cache needs the layers that "
                f"share a pool on one device: reload with every change of device at "
                f"{format_cuts(free_cuts(self.config))} ({DOC_REF}). {REPLICA_HINT}")
        return kl

    @staticmethod
    def _block_table(params, kl, rsg, bsz):
        """(bsz, P) int32 block table on kl's device: the generator's, or the slot-partitioned
        identity for direct forwards without a page table (tests, benchmarks)."""
        if "block_table" in params:
            return get_for_device(params, "block_table", kl.device)
        sbt = kl.slot_bt(rsg[0].cache.num_slots)
        return sbt[[rsg[i].slot for i in range(bsz)]]

    def _forward_cached_row(self, x, params, rs, rsl, out_dtype, b, kl, bt_row, src_kl, bt_src):
        _, seq, _ = x.shape
        device = x.device
        pos0 = rs.position
        slot = rs.slot
        m = self.compress_ratio
        w = self.sliding_window
        hd, rd = self.head_dim, self.rope_head_dim
        hpg = self.num_q_heads // self.o_groups

        check_row_limit(self.key, seq, self.num_q_heads, self.head_dim)
        q_res, q, kv = self._project_qkv(x, params, pos0)

        # Window: this chunk's rows plus the ring's rows at abs - window_beg; the ring is
        # updated only after attention (its shift/rebase moves rows the kernel reads)
        n_prev = min(w - 1, pos0 - rs.window_beg, pos0)
        win_floor = pos0 - n_prev
        ring = rsl.ring[slot]
        ring_beg = rs.window_beg

        indices, k_len = None, 0
        if m:
            ec = (pos0 + seq) // m
            assert ec <= bt_row.shape[1] * kl.epp, \
                f"{self.key}: pool entry {ec} beyond the block table ({bt_row.shape[1]} pages)"
            # entries closed by this chunk are visible to the chunk's own queries (a group
            # closing at p is visible to p), so they are stored before attention
            if self.is_kv_source:
                self._compress_store(x, params, rsl, slot, kl, bt_row, pos0)
            elif self.replica_of is not None:
                e0, e1 = emission_range(pos0, seq, m)
                if e1 > e0:
                    sync_pool_replica(src_kl, kl, bt_src, bt_row, e0, e1)
            indices, k_len = self._select_cached(x, params, q_res, kl, bt_row, ec, pos0, b)
            pool_c, pool_r, bt, qc, page, pool_len = \
                kl.pool_c_view(), kl.pool_r, bt_row, kl.qc(), kl.epp, ec
        else:
            # sliding layer: V4's one-row dummy pool keeps the kernel arguments well formed
            pool_c = x.new_empty((1, hd - rd), dtype = torch.half)
            pool_r = x.new_empty((1, rd), dtype = torch.half)
            bt = torch.zeros((1, 1), dtype = torch.int32, device = device)
            qc, page, pool_len = None, PAGE_SIZE, 0

        exact = EXACT_ROWS and bool(params.get("exact_rows")) and seq > 1
        if exact:
            # EXL3_EXACT_ROWS leaves dsa_attn one call for the rows of the forward. The call takes
            # ONE softmax split count, from its last row, and partitions every row's own keys by
            # it: each row's one-row count only if the rows share one plan. The model flags only
            # such forwards (exact_rows_span, on one layer per position rule); every layer checks
            # its own here, against what the call selected. The first and last row suffice: the
            # entry count only grows with the position, so a plan that changed does not come back
            plan = self.row_plan(pos0)
            if plan != self.row_plan(pos0 + seq - 1) or plan[0] != (indices is None) \
                    or (indices is not None and k_len != self.index_topk):
                raise RuntimeError(
                    f"{self.key}: EXL3_EXACT_ROWS=1: a draft verification of {seq} rows at position "
                    f"{pos0} does not give every row the attention of a one-row step (plan of the "
                    f"first row {plan}, of the last {self.row_plan(pos0 + seq - 1)}; the call "
                    f"{'selects ' + str(k_len) if indices is not None else 'is dense'}, index_topk "
                    f"{self.index_topk}); such a call must not carry params['exact_rows']")

        out = dsa_attn(
            q[0].half().contiguous(), pool_c, pool_r, bt, sinks = self.sinks,
            ring = ring, kv_chunk = kv[0], win_len = w,
            win_floor = win_floor, ring_beg = ring_beg,
            indices = indices, k_len = k_len, pool_len = pool_len, q_pos0 = pos0,
            compress_rate = m or 1, scale = self.sm_scale,
            derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
            # one softmax partition at every row count under EXL3_STABLE_ARITHMETIC=1, as in
            # the stateless path
            n_splits = 1 if STABLE_ARITHMETIC else 0,
            page_size = page, qc = qc,
            out = torch.empty((self.o_groups, seq, hpg * hd), dtype = torch.half, device = device),
        )

        # every layer computes the same shift for the same forward; post_advance applies it
        shift = ring_update(ring, kv[0], pos0, rs.window_beg, w)
        if shift is not None:
            rs.wshift = shift

        if exact:
            return self._project_o_rows(out, params, out_dtype)
        return self._project_o_grouped(out.unsqueeze(1), params, out_dtype)

    def _project_o_rows(self, out, params, out_dtype):
        """
        EXL3_EXACT_ROWS: the grouped output projection of dsa_attn's group-major (groups, seq,
        width) output one row per call, each the seq == 1 call of a decode step: the grouped
        GEMM's launch configuration follows the row count, and wo_b behind it is a Linear. A row
        of the group-major output is strided across the groups, hence the contiguous copy.
        Returns (1, seq, hidden), as _project_o_grouped on the whole output does.

        Where the extension has the row-exact entry point and a one-row call takes the grouped
        GEMM (use_mg in _project_o_grouped), one native call gives every row the bits of that
        call: row j of the row-major copy is the (groups, 1, width) block the one-row call passes,
        and row j of C the (1, groups * n) row its wo_b reads. The entry point makes that call's
        launch once per row, with that call's scratch, or (an extension that reports
        EXACT_ROWS_CAP_ONE_LAUNCH, with EXL3_INT8_GEMV=0) one launch for the rows under that
        call's launch record. wo_b then gets the rows as one flagged call.
        """
        G, rows, width = out.shape
        if ROWS_NATIVE & EXACT_ROWS_CAP_MGEMM:
            if not self.woa_multi_ready and self.device is not None:
                self._build_woa_multi()
            mu = self.wo_a_multi
            if (
                mu is not None and not STABLE_ARITHMETIC and 1 < rows <= EXACT_ROWS_MAX
                and G == self.o_groups and width == mu.in_features
                and not any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
            ):
                A = out.transpose(0, 1).contiguous()
                ah = g_tensor_cache.get(out.device, (G, 1, width), torch.half, "dsv4_woa_had")
                C = torch.empty((rows, G, mu.out_features), dtype = torch.half, device = out.device)
                ext.exl3_mgemm_rows(
                    A, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh,
                    self.woa_indices, mu.K, mu.mcg, mu.mul1
                )
                return self.wo_b.forward(
                    C.view(1, rows, G * mu.out_features), params, out_dtype = out_dtype or self.out_dtype)
        return torch.cat([
            self._project_o_grouped(out[:, j:j + 1].contiguous().unsqueeze(1), params, out_dtype)
            for j in range(out.shape[1])], dim = 1)

    def _compress_store(self, x, params, rsl, slot, kl, bt_row, pos0):
        """
        kv source: pool this chunk's closed groups and write them into the paged pools.

        Rate 1: latent = rms_norm(wkv(x)) per token, no state. Rate 2: raw wkv/wgate rows,
        the open group's pending row prepended from the fp32 carry ring, column softmax over
        each pair, rms_norm -- in torch (CompressCarry) by default, in one fused FP32 kernel
        (dsv41_compress.fused_compress) when the layer state was built with
        EXL3_DSV41_FUSED_COMPRESS=1 and so holds that kernel's rings. Both yield the PRE-RoPE
        latents of entries [pos0 // m, (pos0 + seq) // m); _store_entries ropes and stores them.
        """
        comp = self.compressor
        m = self.compress_ratio
        seq = x.shape[1]
        e0, e1 = emission_range(pos0, seq, m)
        # EXL3_EXACT_ROWS: every closed entry pooled and normalized by a call of its own
        per_row = EXACT_ROWS and bool(params.get("exact_rows"))
        if m == 1:
            kvr = comp.wkv.forward(x, params)[0].float()
            latent, first = CompressCarry.step(None, kvr, None, pos0, 1, comp.norm.weight, comp.rms_norm_eps,
                                               per_row)
        elif rsl.comp_buf_kv is not None:
            kvr = comp.wkv.forward(x, params)[0].float()
            gr = comp.wgate.forward(x, params)[0].float()
            latent, first = fused_compress(
                kvr, gr, pos0, comp.norm.weight, comp.rms_norm_eps, rsl.comp_buf_kv[slot], rsl.comp_buf_gate[slot])
        else:
            kvr = comp.wkv.forward(x, params)[0].float()
            gr = comp.wgate.forward(x, params)[0].float()
            latent, first = CompressCarry.step(
                rsl.comp_carry[slot], kvr, gr, pos0, m, comp.norm.weight, comp.rms_norm_eps, per_row)
        assert first == e0 and latent.shape[0] == e1 - e0
        if e1 > e0:
            self._store_entries(latent, e0, kl, bt_row, pos0, seq, params)

    def _store_entries(self, latent, e0, kl, bt_row, pos0, seq, params):
        """
        Rope and store n pre-RoPE latents as entries e0 .. e0 + n - 1: nope part to pool_c,
        rope part (GPT-J at e * m, the group's first token) to pool_r, and -- where this
        source owns index K -- k_norm(wk(latent)) roped at the same positions to pool_idx.
        The arithmetic is publish_pools' (the stateless path), so the two paths store the
        same values.
        """
        n = latent.shape[0]
        hd, rd, m = self.head_dim, self.rope_head_dim, self.compress_ratio
        dev = latent.device
        tab = self._entry_table(params, e0, n, dev)
        # index K first: it is derived from the PRE-RoPE latent
        k = None
        policy = self._numerics()
        if kl.pool_idx is not None:
            k = self.indexer.produce_k(latent, params)
            k = k.half().view(1, n, 1, self.index_head_dim).contiguous()
            _ext_rope(k[..., -rd:], tab)
            rounding.round_index_(policy, k)
        lat16 = latent.half()
        # clone, never .contiguous(): at n == 1 the (1, rd) tail slice already counts as
        # contiguous, so .contiguous() would return a view and the in-place rope would rotate
        # the latent itself when the caller hands in an FP16 latent
        rot = lat16[:, hd - rd:].clone().view(1, n, 1, rd)
        _ext_rope(rot, tab)

        rows = entry_rows(bt_row, torch.arange(e0, e0 + n, device = dev), kl.epp)
        if policy.compressed:
            # defined on the FP16 pool: DeepseekV41Config refuses the compressed part for a
            # packed (quantized) pool when the Cache is built and when the setting changes
            nope, _ = rounding.round_compressed_(policy, lat16[:, :hd - rd].contiguous(), rot)
            lat16 = torch.cat([nope, lat16[:, hd - rd:]], dim = 1)
        if kl.quant:
            stage = torch.cat([lat16[:, :hd - rd], rot.view(n, rd)], dim = 1).contiguous()
            ext.dsv4_pool_quant_scatter(
                stage, kl.pool_c_view(), kl.pool_s.view(-1, kl.G), kl.pool_r.view(-1, kl.D_r),
                bt_row, pos0, None, m, seq, kl.epp)
        else:
            kl.pool_c.view(-1, kl.D_c).index_copy_(0, rows, lat16[:, :hd - rd])
            kl.pool_r.view(-1, kl.D_r).index_copy_(0, rows, rot.view(n, rd))
        if k is not None:
            kl.pool_idx.view(-1, kl.D_i).index_copy_(0, rows, k.view(n, self.index_head_dim))

    def _select_cached(self, x, params, q_res, kl, bt_row, ec, pos0, b):
        """
        (indices, k_len) for row b. At ec <= index_topk every visible entry is selected, which
        is dsa_attn's dense-pool mode (vLLM's short-context fill). Past it, index sources
        select and publish params["dsv41_topk"][(layer, b)]; consumers reuse their index
        source's entry (moved once per forward when it lives on another device).
        """
        if ec <= self.index_topk:
            return None, 0
        reg = params.setdefault("dsv41_topk", {})
        if self.is_index_source:
            if kl.pool_idx is None:
                raise RuntimeError(f"{self.key}: index source reads a pool with no index keys "
                                   f"(layer {self.pool_owner_layer})")
            r = self._index_select(x, params, q_res, kl.pool_idx, bt_row, kl.epp, pos0, ec, b)
            reg[(self.layer_idx, b)] = (r.indices, r.k_len)
            return r.indices, r.k_len
        if ablated("dense_consumers"):
            return None, 0
        sel = reg.get((self.index_source_layer, b))
        if sel is None:
            raise RuntimeError(
                f"{self.key}: index source layer {self.index_source_layer} has not published a "
                f"top-k selection for row {b} this forward pass")
        indices, k_len = sel
        if indices is not None:
            indices = DeviceMemo.get(
                params, ("dsv41_topk", self.index_source_layer, b), indices, self.device)
        return indices, k_len
