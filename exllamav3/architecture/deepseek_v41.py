# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
DeepSeek-V4.1-Flash configuration.

Derived in part from vLLM's vllm/models/deepseek_v4_1/attention.py
(Apache-2.0); see NOTICE. The compressed-attention topology below follows
that implementation's documented semantics.

V4.1 differs from V4 in the meaning of ``compress_ratios``. In V4 the list
selects an attention *kind* per layer (0 -> sliding, 4 -> CSA, 128 -> HCA)
and the rates live in separate keys. In V4.1 the value IS the compression
rate, in tokens per compressed cache state:

    0   pure sliding window (no compressed cache)
    1   full-length compressed cache   (1 token  per state)
    2   ratio-2 compressed cache       (2 tokens per state)

DeepSeek-V4.1-Flash ships:

    layers  0-1    ratio 0    sliding window (128)
    layers  2-19   ratio 2
    layers 20-39   ratio 1

The list may run past num_hidden_layers to describe the draft (MTP) blocks,
which this text model does not build; those entries are ignored.
"""

from __future__ import annotations

import inspect
import os
import weakref

import torch
from typing_extensions import override

from ..util.file import no_default
from ..model.config import Config
from ..model.model import Model
from ..model.placement import parse as parse_placement
from ..model.math_policy import STABLE_ARITHMETIC
from .dsv41 import activation, numerics, pipeline, router_bias

# V4.1: compress_ratios[i] is the compression rate, not a kind selector.
V41_VALID_RATIOS = (0, 1, 2)

# Why convert.py refuses this architecture (caps "can_quantize" / "quantize_refusal")
QUANTIZE_REFUSAL = (
    "DeepSeek-V4.1: converting (quantizing) this architecture is not supported yet. This version "
    "of exllamav3 runs DeepSeek-V4.1 from EXL3 checkpoints, but its V4.1 modules do not define "
    "how DeepSeek's original weights are quantized (the lightning indexer's projections, for "
    "one, would be calibrated against the wrong inputs), so a conversion would not produce a "
    "correct model. Conversion from DeepSeek's original checkpoint will come in a later change.")


def _without_compressed(n) -> str:
    """The setting a -cq refusal suggests: the chosen parts without 'compressed', or the default
    when no other part is left."""
    if n.index or n.window:
        return str(numerics.Numerics(n.contract, n.index, n.window, False))
    return numerics.DEFAULT


def _nested(key: str) -> list[str]:
    """V4.1 nests the text config; accept both nested and flat placements."""
    return [f"text_config->{key}", key]


class DeepseekV41Config(Config):
    arch_string = "DeepseekV41ForCausalLM"

    def __init__(self, directory: str, **kwargs):
        # Text only. The DSpark draft (blocks mtp.0-2, reading target layers
        # 37-39, with expert counts of its own) has no model class yet, so no
        # "mtp" component is offered rather than one that would build wrong.
        super().__init__(directory, {"text": DeepseekV41Model}, **kwargs)

        self.hidden_size = self.read_cfg(int, _nested("hidden_size"), no_default)
        self.num_hidden_layers = self.read_cfg(int, _nested("num_hidden_layers"), no_default)
        self.num_q_heads = self.read_cfg(int, _nested("num_attention_heads"), no_default)
        self.num_kv_heads = self.read_cfg(int, _nested("num_key_value_heads"), 1)
        if self.num_kv_heads != 1:
            # a ValueError, not an assert: python -O must not strip a refusal of a config this
            # graph would run wrong
            raise ValueError(
                f"DeepSeek-V4.1: the checkpoint's config sets num_key_value_heads = "
                f"{self.num_kv_heads}, but this implementation supports only 1. V4.1 attention "
                f"is multi-query: every query head reads one shared latent KV head, and the "
                f"compressed-KV pools, the sliding-window rings and the attention kernels store "
                f"and read exactly that one head, so a checkpoint with {self.num_kv_heads} KV "
                f"heads would be computed wrong.")

        self.head_dim = self.read_cfg(int, _nested("head_dim"), 512)
        self.qk_rope_head_dim = self.read_cfg(int, _nested("qk_rope_head_dim"), 64)
        self.sliding_window = self.read_cfg(int, _nested("sliding_window"), 128)
        # DeepSeek's default; the engram gate and every norm use it
        self.rms_norm_eps = self.read_cfg(float, _nested("rms_norm_eps"), 1e-20)

        # Shapes the graph needs that V4 reads under the same names. Kept here
        # rather than inherited because V4.1 nests them under text_config.
        self.vocab_size = self.read_cfg(int, _nested("vocab_size"), no_default)
        self.q_lora_rank = self.read_cfg(int, _nested("q_lora_rank"), no_default)
        self.o_groups = self.read_cfg(int, _nested("o_groups"), 8)
        self.o_lora_rank = self.read_cfg(int, _nested("o_lora_rank"), no_default)
        self.index_n_heads = self.read_cfg(int, _nested("index_n_heads"), 0)
        self.index_head_dim = self.read_cfg(int, _nested("index_head_dim"), 0)
        self.index_topk = self.read_cfg(int, _nested("index_topk"), 0)
        self.rope_theta = self.read_cfg(float, _nested("rope_theta"), 10000.0)
        self.compress_rope_theta = self.read_cfg(float, _nested("compress_rope_theta"), 160000.0)
        self.rope_scaling = self.read_cfg(dict, _nested("rope_scaling"), None)
        self.max_position_embeddings = self.read_cfg(int, _nested("max_position_embeddings"), no_default)
        self.tie_word_embeddings = self.read_cfg(bool, _nested("tie_word_embeddings"), False)

        # mHC hyper-connection streams
        self.hc_mult = self.read_cfg(int, _nested("hc_mult"), 4)
        self.hc_sinkhorn_iters = self.read_cfg(int, _nested("hc_sinkhorn_iters"), 20)
        self.hc_eps = self.read_cfg(float, _nested("hc_eps"), 1e-6)

        # MoE. V4 bootstraps its first layers with hash routing; V4.1 ships no
        # num_hash_layers, so every layer routes the same way. The function's scalars default to
        # DeepSeek's reference ModelArgs (swiglu_limit 0: no clamp; routed_scaling_factor 1),
        # and a missing rope_scaling means no YaRN; the released config sets all three
        self.moe_intermediate_size = self.read_cfg(int, _nested("moe_intermediate_size"), no_default)
        self.routed_scaling_factor = self.read_cfg(float, _nested("routed_scaling_factor"), 1.0)
        self.swiglu_limit = self.read_cfg(float, _nested("swiglu_limit"), 0.0)
        self.num_hash_layers = self.read_cfg(int, _nested("num_hash_layers"), 0)
        # The graph implements one router: sqrt-softplus scores, noaux_tc top-k with the
        # selection bias, normalised weights. A config naming another one would build a
        # model that runs and routes wrong
        for key, want in (("scoring_func", "sqrtsoftplus"), ("topk_method", "noaux_tc"),
                          ("norm_topk_prob", True)):
            got = self.read_cfg(None, _nested(key), None)
            if got is not None and got != want:
                raise ValueError(
                    f"DeepSeek-V4.1: {key}={got!r}; the graph implements {key}={want!r} only.")

        # ---- compressed-attention topology ----
        ratios = self.read_cfg(list, _nested("compress_ratios"), None)
        if ratios is None:
            # a missing schedule would build an all-sliding model that runs without an error
            raise ValueError(
                "DeepSeek-V4.1: compress_ratios is missing; the config must give every layer's "
                "compression rate.")
        if len(ratios) < self.num_hidden_layers:
            raise ValueError(
                f"DeepSeek-V4.1: compress_ratios has {len(ratios)} entries for "
                f"{self.num_hidden_layers} layers.")
        # entries past num_hidden_layers describe the draft blocks
        self.compress_ratios = [int(r) for r in ratios[:self.num_hidden_layers]]

        for idx, r in enumerate(self.compress_ratios):
            if r not in V41_VALID_RATIOS:
                raise ValueError(
                    f"DeepSeek-V4.1 layer {idx} has compress_ratio={r}; only "
                    f"{V41_VALID_RATIOS} are supported (0 = sliding window, "
                    f"1 = full-length compressed, 2 = ratio-2 compressed)."
                )

        # "sliding" layers keep only the local window; "compressed" layers read
        # a compressed KV cache published by a source layer.
        self.layer_types = [
            "sliding" if r == 0 else "compressed" for r in self.compress_ratios
        ]

        # ---- KV / index source topology ----
        # Compressors and compressed-KV caches live only on kv_source layers;
        # indexers only on index_source layers. A consumer reuses the most
        # recently published source at or below its own index. The two lists
        # have different granularity: DeepSeek-V4.1-Flash publishes KV at
        # [2, 8, 14, 20] but indices at [2, 8, 14, 20, 24, 28, 32, 36].
        self.kv_source_layer_ids = tuple(
            int(i) for i in (self.read_cfg(list, _nested("kv_source_layer_ids"), None) or ())
        )
        self.index_source_layer_ids = tuple(
            int(i) for i in (self.read_cfg(list, _nested("index_source_layer_ids"), None) or ())
        )

        if any(r > 0 for r in self.compress_ratios):
            if not self.kv_source_layer_ids or not self.index_source_layer_ids:
                raise ValueError(
                    "DeepSeek-V4.1 requires kv_source_layer_ids and "
                    "index_source_layer_ids for compressed layers."
                )
            for what, ids in (("kv", self.kv_source_layer_ids),
                              ("index", self.index_source_layer_ids)):
                for s in ids:
                    if not 0 <= s < self.num_hidden_layers:
                        raise ValueError(f"{what} source {s} is not a backbone layer.")
                    if self.compress_ratios[s] == 0:
                        raise ValueError(
                            f"{what} source {s} is a sliding-window layer (rate 0); "
                            f"only compressed layers publish.")
            for s in self.kv_source_layer_ids:
                # the index key is derived from the kv source's latent, so a kv source
                # that does not index leaves its group's index K without an owner
                if s not in self.index_source_layer_ids:
                    raise ValueError(
                        f"kv source {s} is not an index source; V4.1 derives index K "
                        f"from the kv source's latent.")
            for idx, r in enumerate(self.compress_ratios):
                if r == 0:
                    continue
                if not any(s <= idx for s in self.kv_source_layer_ids):
                    raise ValueError(
                        f"layer {idx} is compressed (ratio {r}) but no "
                        f"kv source is published at or below it."
                    )
                if not any(s <= idx for s in self.index_source_layer_ids):
                    raise ValueError(
                        f"layer {idx} is compressed (ratio {r}) but no "
                        f"index source is published at or below it."
                    )
                src = max(s for s in self.kv_source_layer_ids if s <= idx)
                if self.compress_ratios[src] != r:
                    raise ValueError(
                        f"layer {idx} has compression rate {r} but reads kv source "
                        f"{src} at rate {self.compress_ratios[src]}.")
            for key in ("index_topk", "index_n_heads", "index_head_dim"):
                if getattr(self, key) <= 0:
                    raise ValueError(
                        f"DeepSeek-V4.1: {key} is missing or not positive; compressed "
                        f"layers select through the indexer.")

        # ---- two-level candidate filtering ----
        # The indexer at candidate_source_layer_id publishes the top
        # candidate_topk_blocks blocks of candidate_block_size compressed
        # positions. Indexers on layers strictly above it mask their scores to
        # those blocks before running their own top-k, so selection is coarse
        # (blocks) then fine (index_topk positions). V4 has no such stage.
        self.candidate_source_layer_id = self.read_cfg(
            int, _nested("candidate_source_layer_id"), -1
        )
        self.candidate_block_size = self.read_cfg(int, _nested("candidate_block_size"), 0)
        self.candidate_topk_blocks = self.read_cfg(int, _nested("candidate_topk_blocks"), 0)

        if self.candidate_source_layer_id < 0 and (self.candidate_block_size or
                                                   self.candidate_topk_blocks):
            raise ValueError(
                "candidate_block_size / candidate_topk_blocks are set but "
                "candidate_source_layer_id is not; the candidate stage has no source.")
        if self.candidate_source_layer_id >= 0:
            if self.candidate_source_layer_id >= self.num_hidden_layers:
                raise ValueError(
                    f"candidate_source_layer_id="
                    f"{self.candidate_source_layer_id} is not a backbone layer "
                    f"(num_hidden_layers={self.num_hidden_layers})."
                )
            if self.candidate_source_layer_id not in self.index_source_layer_ids:
                raise ValueError(
                    f"candidate_source_layer_id="
                    f"{self.candidate_source_layer_id} must also be an index "
                    f"source; candidates are published by an indexer."
                )
            if self.candidate_block_size <= 0 or self.candidate_topk_blocks <= 0:
                raise ValueError(
                    "candidate_block_size and candidate_topk_blocks must both "
                    "be positive when candidate_source_layer_id is set."
                )
            candidate_rate = self.compress_ratios[self.candidate_source_layer_id]
            for idx in self.index_source_layer_ids:
                if idx > self.candidate_source_layer_id and self.compress_ratios[idx] != candidate_rate:
                    raise ValueError(
                        f"candidate source {self.candidate_source_layer_id} uses compression "
                        f"rate {candidate_rate}, but index source {idx} uses rate "
                        f"{self.compress_ratios[idx]}; candidate block IDs require the same "
                        "token-to-entry compression grid.")

        # ---- MoE (target) ----
        self.n_routed_experts = self.read_cfg(int, _nested("n_routed_experts"), no_default)
        self.num_experts_per_tok = self.read_cfg(int, _nested("num_experts_per_tok"), no_default)
        self.n_shared_experts = self.read_cfg(int, _nested("n_shared_experts"), 1)

        # ---- Engram ----
        # N-gram hash lookups gated into the hyper-connection stream, present
        # only on engram_layer_ids. The tables are enormous (~384M rows of
        # engram_head_dim each) and ship as separate checkpoint shards, so they
        # are never fully resident; see architecture/dsv41/engram.py for the
        # bucket layout. V4 has no equivalent.
        self.engram_layer_ids = tuple(
            int(i) for i in (self.read_cfg(list, _nested("engram_layer_ids"), None) or ())
        )
        self.engram_num_embeddings = tuple(
            int(i) for i in (self.read_cfg(list, _nested("engram_num_embeddings"), None) or ())
        )
        self.engram_max_ngram_size = self.read_cfg(int, _nested("engram_max_ngram_size"), 0)
        self.engram_n_heads = self.read_cfg(int, _nested("engram_n_heads"), 0)
        self.engram_head_dim = self.read_cfg(int, _nested("engram_head_dim"), 0)
        self.engram_vocab_size = self.read_cfg(int, _nested("engram_vocab_size"), 0)
        self.engram_compressed_vocab_size = self.read_cfg(
            int, _nested("engram_compressed_vocab_size"), 0
        )
        self.engram_pad_token_id = self.read_cfg(int, _nested("engram_pad_token_id"), None)

        if self.engram_layer_ids:
            if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
                raise ValueError(
                    "engram_layer_ids and engram_num_embeddings must have the "
                    f"same length ({len(self.engram_layer_ids)} vs "
                    f"{len(self.engram_num_embeddings)})."
                )
            for i in self.engram_layer_ids:
                if not 0 <= i < self.num_hidden_layers:
                    raise ValueError(
                        f"engram_layer_ids contains {i}, not a backbone layer."
                    )
            if self.engram_max_ngram_size < 2:
                raise ValueError("engram_max_ngram_size must be >= 2.")
            if self.engram_pad_token_id is None:
                # a default of -1 would pad every blocked n-gram with the last vocabulary entry
                raise ValueError("engram_pad_token_id is required when engram layers exist.")

        # ---- attention numerics ----
        # How the indexer operands and the attention KV are rounded (architecture/dsv41/
        # numerics.py). EXL3_DSV41_NUMERICS gives the value this config starts with; the
        # attention layers read the attribute on every forward, so it can be changed between
        # sequences on a loaded model
        self._dsv41_packed_pools = weakref.WeakSet()
        self.dsv41_numerics = os.environ.get(numerics.ENV_NUMERICS)

        # ---- router selection bias ----
        # EXL3_DSV41_ROUTER_BIAS: a safetensors file with the original FP32 selection bias of
        # every routed layer, checked against this checkpoint and added to its tensors, so the
        # routers load FP32 values instead of the stored copy (architecture/dsv41/router_bias.py).
        # The resolved path, or None
        self.dsv41_router_bias = None
        path = os.environ.get(router_bias.ENV_ROUTER_BIAS)
        if path:
            self.dsv41_router_bias = router_bias.attach(self, path)

    @property
    def dsv41_numerics(self) -> numerics.Numerics:
        return self._dsv41_numerics

    @dsv41_numerics.setter
    def dsv41_numerics(self, value):
        """
        Parsed and validated on assignment ('precise', 'deepseek', 'vllm', optionally ':' and
        parts; None or '' for the default). The compressed part rounds entries in the FP16
        pool, so it is refused while a packed (quantized) pool built from this config exists.
        """
        n = numerics.parse(value)
        if n.compressed and len(self._dsv41_packed_pools):
            raise ValueError(
                f"dsv41 numerics {n}: rounding compressed entries is defined on the FP16 pool, "
                f"and a Cache with a quantized pool (-cq) exists for this model; use a setting "
                f"without the compressed part (e.g. {_without_compressed(n)}) or an FP16 Cache")
        self._dsv41_numerics = n

    def register_packed_pool(self, layer):
        """
        Called by each V4.1 kv-source pool built in the packed (quantized) format, when the
        Cache is constructed. Refuses the pool under EXL3_STABLE_ARITHMETIC or if the current
        numerics round compressed entries, and keeps a weak reference so that the setting cannot
        switch to them while it exists.
        """
        if STABLE_ARITHMETIC:
            # dsa_attn stages a packed pool back to FP16 in the original domain for calls of 64 or
            # more query rows (EXL3_DSA_QC_STAGE_MIN_R) and reads it packed, in the rotated domain,
            # below that: the two round different quantities, so decode would differ from prefill
            raise ValueError(
                "EXL3_STABLE_ARITHMETIC=1: DeepSeek-V4.1's attention reads a quantized Cache (-cq) "
                "with arithmetic that depends on the number of query rows (from 64 rows the packed "
                "pool is staged back to FP16, below that it is read packed), so decode would differ "
                "from prefill; use an FP16 Cache")
        n = self._dsv41_numerics
        if n.compressed:
            raise ValueError(
                f"dsv41 numerics {n}: rounding compressed entries is defined on the FP16 pool and "
                f"cannot be combined with a quantized Cache (-cq); use a setting without the "
                f"compressed part (e.g. {_without_compressed(n)}) or an FP16 Cache")
        self._dsv41_packed_pools.add(layer)

    # -- topology queries --

    def is_kv_source(self, layer_idx: int) -> bool:
        return layer_idx in self.kv_source_layer_ids

    def is_index_source(self, layer_idx: int) -> bool:
        return layer_idx in self.index_source_layer_ids

    def kv_source_for(self, layer_idx: int) -> int | None:
        """Layer owning the compressed KV cache this layer reads (None if sliding)."""
        if self.compress_ratios[layer_idx] == 0:
            return None
        return max(s for s in self.kv_source_layer_ids if s <= layer_idx)

    def index_source_for(self, layer_idx: int) -> int | None:
        if self.compress_ratios[layer_idx] == 0:
            return None
        return max(s for s in self.index_source_layer_ids if s <= layer_idx)

    def kv_groups(self) -> list[tuple[int, list[int]]]:
        """
        (source, members) per compressed-KV group, in layer order.

        Every member reads the source's pool on its own device, so with a Cache
        a layer split must not cut a group, except at the split of an explicit
        placement, past which a replica of the pool is read
        (architecture/dsv41/placement.py).
        For DeepSeek-V4.1-Flash this yields
        {2-7} {8-13} {14-19} {20-39}, the last being 20 layers.
        """
        groups: dict[int, list[int]] = {}
        for idx in range(self.num_hidden_layers):
            src = self.kv_source_for(idx)
            if src is not None:
                groups.setdefault(src, []).append(idx)
        return sorted(groups.items())

    def uses_candidates(self, layer_idx: int) -> bool:
        """Indexers strictly above the source mask their scores to its blocks."""
        return 0 <= self.candidate_source_layer_id < layer_idx

    def index_owns_k(self, layer_idx: int) -> bool:
        """
        Whether an index source owns its paged K cache.

        V4.1 derives the index key from the kv-source compressor latent
        (k = k_norm(wk(latent))) rather than an indexer-local compressor over
        hidden states, so an index source that is not also a kv source has no
        K of its own and shares the kv source's cache. For
        DeepSeek-V4.1-Flash, index sources 2/8/14/20 own K while 24/28/32/36
        borrow layer 20's.
        """
        return self.is_index_source(layer_idx) and self.is_kv_source(layer_idx)

    def has_engram(self, layer_idx: int) -> bool:
        return layer_idx in self.engram_layer_ids

    def engram_layout(self):
        """Bucket layout for the engram tables (None when the model has none)."""
        from .dsv41.engram import EngramLayout
        return EngramLayout.from_config(self)


class DeepseekV41Model(Model):
    """
    DeepSeek-V4.1 module graph.

    Mirrors DeepseekV4Model, with the V4.1 differences wired in:

      - layer_types come from compress_ratios read as a RATE (0 sliding,
        1 full-length compressed, 2 ratio-2), not a kind selector
      - compressors exist only on kv_source_layer_ids; consumers reference
        the source's pool (see dsv41/cache_plan.py)
      - indexers exist only on index_source_layer_ids, and own K only where
        they are also a kv source (see modules/dsv41.py)
      - the indexer above candidate_source_layer_id masks to its published
        blocks before its own top-k (see dsv41/candidates.py)
      - engram sits on engram_layer_ids only, gating into the hc streams
        (see modules/dsv41_engram.py). It is a submodule of its block; its
        ~94 GiB-per-layer table stays on disk and is not a weight anywhere,
        so nothing that sizes or places weights counts it
      - hyper-connections are delayed and there is no learned hc_head
        (modules/dsv41_block.py)
      - the routed MoE is DSV41MoE, whose load asks the load guard
        (dsv41/placement.py) whether a change of device falls where nothing
        is shared across it, or at the split of the explicit placement, where
        a pool replica takes over; every compressed layer gets its pool route
        before any Cache exists. Which experts are kept in system RAM is -mcl /
        -mcs or the explicit placement (EXL3_PLACEMENT), as for every MoE model
    """

    config_class = DeepseekV41Config

    def __init__(self, config: DeepseekV41Config, **kwargs):
        super().__init__(config, **kwargs)
        from .dsv41.cache_plan import CachePlan
        from .dsv41.placement import DSV41Placement, DSV41LoadGuard, default_routes
        from ..modules import Embedding, RMSNorm, Linear, GatedMLP, ExpandStreams
        from ..modules.dsv41 import DSV41Attention
        from ..modules.dsv41_engram import DSV41Engram
        # V4.1's hyper-connections are DELAYED (each sublayer collapses its input
        # with the previous sublayer's pre-mix) and there is no learned hc_head;
        # these are added alongside V4's classes, which stay untouched
        from ..modules.dsv41_block import DSV41HyperConnection, DSV41Block, DSV41HeadCollapse
        from ..modules.dsv41_moe import DSV41MoE

        c = config

        # The routed and shared experts' activation, the same for both (EXL3_DSV41_ACTIVATION,
        # architecture/dsv41/activation.py), fixed for the model's lifetime: the fused and graph
        # kernels are selected when the layers load
        activation_fn = activation.activation_fn(os.environ.get(activation.ENV_ACTIVATION), c.swiglu_limit)

        # The explicit placement (EXL3_PLACEMENT / --placement / config.infer_params.placement),
        # read now, or None: its changes of device must be free cuts, except at most one, the
        # split, past which the first layer owns a replica of its kv group's pool. The Cache sizes
        # itself from those pool routes, so they must exist before any Cache is built
        # (architecture/dsv41/placement.py). Without one every layer reads its own kv source
        self.placement = DSV41Placement.from_generic(
            parse_placement(getattr(c.infer_params, "placement", None)), c)
        self.pool_routes = self.placement.routes if self.placement is not None \
            else default_routes(c)
        self.cache_plan = CachePlan(c.compress_ratios, c.kv_source_layer_ids,
                                    routes = self.pool_routes)
        # Shared by every DSV41MoE: while load_gen runs, fails fast when the autosplit
        # changes device where a pool, a selection or candidate blocks would be shared
        # across the change and a Cache is attached (the placement's split excepted,
        # whose replica the Cache holds); inert outside it
        self_ref = weakref.ref(self)
        self.load_guard = DSV41LoadGuard(
            c, cache_attached = lambda: (m := self_ref()) is not None and m._cache_attached(),
            split_at = self.placement.split_at if self.placement is not None else None,
        )
        # Pipelined prefill (dsv41/pipeline.py): the cached split plan, and the max_chunk_size
        # of the current load, which bounds its sub-chunk. Both are reset by every load/unload
        self._dsv41_pipeline_plan = None
        self._dsv41_loaded_max_chunk = None

        self.modules += [
            Embedding(
                config = c, key = "embed",
                vocab_size = c.vocab_size, hidden_size = c.hidden_size,
            ),
            ExpandStreams(
                config = c, key = "hc_expand", hc_mult = c.hc_mult,
            ),
        ]

        self.first_block_idx = len(self.modules)
        self.engram_modules = {}

        for idx in range(c.num_hidden_layers):
            key = f"layers.{idx}"
            ratio = c.compress_ratios[idx]

            attn = DSV41Attention(
                config = c,
                key = f"{key}.attn",
                layer_idx = idx,
                compress_ratio = ratio,
                is_kv_source = c.is_kv_source(idx),
                is_index_source = c.is_index_source(idx),
                index_owns_k = c.index_owns_k(idx),
                kv_source_layer = c.kv_source_for(idx),
                index_source_layer = c.index_source_for(idx),
                uses_candidates = c.uses_candidates(idx),
                hidden_size = c.hidden_size,
                num_q_heads = c.num_q_heads,
                head_dim = c.head_dim,
                rope_head_dim = c.qk_rope_head_dim,
                q_lora_rank = c.q_lora_rank,
                o_groups = c.o_groups,
                o_lora_rank = c.o_lora_rank,
                sliding_window = c.sliding_window,
                index_n_heads = c.index_n_heads,
                index_head_dim = c.index_head_dim,
                index_topk = c.index_topk,
                rope_theta = c.rope_theta,
                compress_rope_theta = c.compress_rope_theta,
                rope_scaling = c.rope_scaling,
                rms_norm_eps = c.rms_norm_eps,
                qmap = "block.attn",
                out_dtype = torch.float,
                select_hq_bits = 2,
            )

            # V4 routes its first layers by hash to bootstrap the router; V4.1
            # ships no num_hash_layers, so this stays empty and every layer
            # uses the same sqrt-softplus router
            is_hash = idx < c.num_hash_layers
            # BlockSparseMLP's own arguments, plus the load guard that checks where the
            # layer split falls
            mlp = DSV41MoE(
                layer_idx = idx,
                load_guard = self.load_guard,
                config = c,
                key = f"{key}.ffn",
                hidden_size = c.hidden_size,
                intermediate_size = c.moe_intermediate_size,
                num_experts = c.n_routed_experts,
                num_experts_per_tok = c.num_experts_per_tok,
                key_up = "experts.{expert_idx}.w3",
                key_gate = "experts.{expert_idx}.w1",
                key_down = "experts.{expert_idx}.w2",
                key_routing_gate = "gate",
                key_e_score_bias = "gate.bias",
                key_tid2eid = "gate.tid2eid" if is_hash else None,
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                activation_fn = activation_fn,
                act_limit = c.swiglu_limit,
                router_type = "sqrtsp_hash" if is_hash else "sqrtsp",
                routed_scaling_factor = c.routed_scaling_factor,
                shared_experts = GatedMLP(
                    config = c,
                    key = f"{key}.ffn.shared_experts",
                    hidden_size = c.hidden_size,
                    intermediate_size = c.moe_intermediate_size * c.n_shared_experts,
                    key_up = "w3", key_gate = "w1", key_down = "w2",
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    activation_fn = activation_fn,
                    act_limit = c.swiglu_limit,
                    select_hq_bits = 2,
                ) if c.n_shared_experts else None,
            )

            def _hc(tag: str, k = key):
                return DSV41HyperConnection(
                    config = c,
                    key = f"{k}.hc_{tag}",
                    hc_mult = c.hc_mult,
                    hidden_size = c.hidden_size,
                    sinkhorn_iters = c.hc_sinkhorn_iters,
                    hc_eps = c.hc_eps,
                    rms_norm_eps = c.rms_norm_eps,
                )

            # Engram: a registered submodule of its block, called at block entry.
            # Its table stays on disk and is excluded from weights_numel, so
            # nothing that sizes or places weights counts it
            eg = None
            if c.has_engram(idx):
                eg = DSV41Engram(
                    config = c,
                    key = f"{key}.engram",
                    layer_idx = idx,
                    hidden_size = c.hidden_size,
                    hc_mult = c.hc_mult,
                    qmap = "block.engram",
                )
                self.engram_modules[idx] = eg

            block = DSV41Block(
                config = c,
                key = key,
                layer_idx = idx,
                attn_norm = RMSNorm(c, f"{key}.attn_norm", c.rms_norm_eps),
                attn = attn,
                attn_hc = _hc("attn"),
                mlp_norm = RMSNorm(c, f"{key}.ffn_norm", c.rms_norm_eps),
                mlp = mlp,
                mlp_hc = _hc("ffn"),
                engram = eg,
            )

            self.modules += [block]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = "embed" if (c.tie_word_embeddings
                                   and not c.stc.has_tensor("head")) else None
        self.modules += [
            # no learned hc_head in V4.1: collapse with the last FFN's pre-mix
            DSV41HeadCollapse(config = c, key = "hc_head", hc_mult = c.hc_mult),
            RMSNorm(
                config = c, key = "norm",
                rms_norm_eps = c.rms_norm_eps, out_dtype = torch.half,
            ),
            Linear(
                config = c, key = "head", qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = c.hidden_size, out_features = c.vocab_size,
                qmap = "block", caps = {"logits_output": True},
            ),
        ]

        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True

        # As in V4, all attention state is recurrent-style (sliding rings plus
        # compressed pools); V4.1 adds that a pool is shared by a whole kv
        # group, so with a Cache a layer split must not cut one, except at the
        # split of an explicit placement, with a pool replica (placement.py)
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            # the V4.1 modules have no TP export; load_gen refuses tensor_p too
            "supports_tp": False,
            # convert.py stops before any work, with this reason
            "can_quantize": False,
            "quantize_refusal": QUANTIZE_REFUSAL,
        })
        # V4.1's own job-level state: V4's bookkeeping plus a rewind limit that keeps a
        # full sliding window below the rewound position, and checkpoints that include
        # the rate-2 compressor carry and the engram lookback ring (cache/dsv41.py)
        from ..cache.dsv41 import DSV41State
        self.recurrent_state_cls = DSV41State

        # Pool routes go to the attention modules now, before any Cache exists:
        # Cache() sizes its layers from the modules' caps at construction, and a
        # replica owner must already be one by then
        self._apply_pool_routes()

    def module_plan(self) -> list[dict]:
        """
        The graph this model will build, as data.

        Returned rather than constructed so it can be checked against a
        checkpoint before anything is allocated.
        """
        c = self.config
        plan = [
            {"key": "embed", "kind": "Embedding"},
            {"key": "hc_expand", "kind": "ExpandStreams"},
        ]
        for i in range(c.num_hidden_layers):
            k = f"layers.{i}"
            entry = {
                "key": k,
                "kind": "DSV41Block",
                "layer_type": c.layer_types[i],
                "compress_ratio": c.compress_ratios[i],
                "attn": f"{k}.attn",
                "attn_norm": f"{k}.attn_norm",
                "ffn_norm": f"{k}.ffn_norm",
                "mlp": f"{k}.ffn",
                "mlp_kind": "DSV41MoE",
                "hc": [f"{k}.hc_attn", f"{k}.hc_ffn"],
                # every layer owns an SWA ring; only kv sources and replica owners
                # own a paged pool, and a consumer reads its route owner's
                "owns_ring": self.cache_plan.owns_ring(i),
                "cache_owner": self.cache_plan.owner[i],
                "allocates_cache": self.cache_plan.owns_pool(i),
                "pool_route": self.pool_routes.get(i),
                "is_kv_source": c.is_kv_source(i),
                "is_index_source": c.is_index_source(i),
                "index_owns_k": c.index_owns_k(i),
                "uses_candidates": c.uses_candidates(i),
                "has_engram": c.has_engram(i),
            }
            if entry["is_kv_source"]:
                entry["compressor"] = f"{k}.attn.compressor"
            if entry["is_index_source"]:
                entry["indexer"] = f"{k}.attn.indexer"
            if entry["has_engram"]:
                entry["engram"] = f"{k}.engram"
            plan.append(entry)
        plan += [
            {"key": "hc_head", "kind": "DSV41HeadCollapse"},
            {"key": "norm", "kind": "RMSNorm"},
            {"key": "head", "kind": "Linear"},
        ]
        return plan

    # Keys DeepseekV41Model.prepare_inputs sets fresh for every forward. A value left in a
    # reused params dict by the previous pass would describe the wrong tokens
    PER_FORWARD_KEYS = (
        "dsv41_pools", "dsv41_hc_pre", "dsv41_pool_mirrors", "dsv41_topk",
        "dsv41_candidates", "dsv41_dev_memo", "dsv41_rope", "dsv41_engram_lookback_fwd",
    )

    @override
    @torch.inference_mode
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        # Two-stage pipelined prefill across the layer split (opt-in, see dsv41/pipeline.py)
        if params is not None and pipeline.eligible(self, input_ids, params):
            return pipeline.prefill_pipelined(self, input_ids, params)
        return super().prefill(input_ids, params)

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        from ..modules.attn import prepare_for_attn
        # A job state left partly advanced by a failed pipelined prefill is refused
        pipeline.check_state(params)
        # Engram layers hash the raw token ids, so they must survive into params
        params["input_ids"] = input_ids
        # A compressed pool published by one layer is read by its whole kv
        # group; the dict is per forward pass, so it starts empty each time
        params["dsv41_pools"] = {}
        # the delayed mHC pre-mix is per token and starts from the identity at
        # layer 0 of EVERY forward -- a stale one would leak into the next pass
        params["dsv41_hc_pre"] = None
        # per-forward like the pools: a mirror or a top-k selection left over from
        # a previous pass would describe the wrong tokens
        params["dsv41_pool_mirrors"] = {}
        params["dsv41_topk"] = {}
        # likewise the candidate blocks layer 20 publishes for the indexers above
        # it, and the per-forward memo of small tensors moved across a layer split
        params["dsv41_candidates"] = {}
        params["dsv41_dev_memo"] = {}
        # rope angle tables, per (device, table, positions)
        params["dsv41_rope"] = {}
        # An explicit engram lookback is an input to THIS forward only: it moves into a
        # per-forward slot, so a dict reused for the next sequence cannot hash that
        # sequence's first positions with this chunk's ids
        params["dsv41_engram_lookback_fwd"] = params.pop("dsv41_engram_lookback", None)
        return prepare_for_attn(input_ids, params)

    # -- layer split --

    def _blocks(self) -> list:
        n = self.config.num_hidden_layers
        return self.modules[self.first_block_idx : self.first_block_idx + n]

    def _cache_attached(self) -> bool:
        return any(ref() is not None for ref in self.cache_weakrefs.values())

    def _apply_pool_routes(self):
        """Hand every compressed layer its pool route."""
        attns = [b.attn for b in self._blocks()]
        for i, (owner, replica_of, with_index_k) in sorted(self.pool_routes.items()):
            attns[i].set_pool_route(owner, replica_of = replica_of, with_index_k = with_index_k)

    @override
    def load_gen(self, *args, **kwargs):
        """
        Model.load_gen (Model.load goes through here too), with V4.1's load checks around
        it: tensor-parallel loading and an explicit placement that needs other pool routes than
        the model was built with are refused before anything loads; while the autosplit runs,
        the load guard refuses a change of device that is neither a free cut nor the explicit
        placement's split when a Cache is attached (architecture/dsv41/placement.py); afterwards
        every compressed layer is checked against the device of the pool it reads.
        """
        from .dsv41.placement import TP_REFUSAL, DSV41Placement, default_routes
        a = inspect.signature(Model.load_gen).bind(self, *args, **kwargs)
        a.apply_defaults()
        a = a.arguments
        # A reload can change modules, devices, CPU hosts and streams
        self._dsv41_pipeline_plan = None
        self._dsv41_loaded_max_chunk = None
        if pipeline.PIPELINE:
            if pipeline.SUB_CHUNK is None:
                # the switch turned on in this process after import, without a sub-chunk
                raise ValueError(
                    "DSV41 pipelined prefill: pipeline.PIPELINE is on but pipeline.SUB_CHUNK is "
                    "not set; for an in-process A/B set both before loading")
            chunk = pipeline._positive_chunk(pipeline.SUB_CHUNK)
            if chunk > a["max_chunk_size"]:
                # The load sizes its prefill workspace for max_chunk_size rows
                raise ValueError(
                    f"EXL3_DSV41_PIPELINE_CHUNK={chunk} exceeds this load's "
                    f"max_chunk_size={a['max_chunk_size']}: load with max_chunk_size >= "
                    f"{chunk}, or lower EXL3_DSV41_PIPELINE_CHUNK")
        if a["tensor_p"]:
            raise NotImplementedError(TP_REFUSAL)
        # The pool routes were fixed when the model was built, from the explicit placement set
        # then, and the Cache is sized from them. The placement set now is checked again (it
        # may have changed since) and must need the same routes; its devices, expert storage
        # and free cuts may differ
        now = DSV41Placement.from_generic(
            parse_placement(getattr(self.config.infer_params, "placement", None)), self.config)
        now_routes = now.routes if now is not None else default_routes(self.config)
        if now_routes != self.pool_routes:
            was = self.placement.needs() if self.placement is not None else "no pool replica"
            raise ValueError(
                f"DeepSeek-V4.1: the model was built for {was}, but config.infer_params.placement "
                f"is now {repr(now.spec) if now is not None else 'None'}, which needs "
                f"{now.needs() if now is not None else 'no pool replica'}. The cross-device pool "
                f"routes are fixed when the model object is built (the Cache is sized from them): "
                f"set the placement before Model.from_config, or build a new model to change it")
        # the placement this load follows, whose split the guard lets through
        self.placement = now
        self.load_guard.split_at = now.split_at if now is not None else None
        self.load_guard.begin()
        try:
            yield from super().load_gen(*args, **kwargs)
        finally:
            self.load_guard.end()
        self._check_split()
        self._dsv41_loaded_max_chunk = a["max_chunk_size"]
        if pipeline.PIPELINE:
            pipeline.report(self)

    def unload(self):
        self._dsv41_pipeline_plan = None
        self._dsv41_loaded_max_chunk = None
        super().unload()

    def _check_split(self):
        """
        After a load: the compressed layers that read a pool on another device than their own
        (the pool of their route's owner: their kv source, or past an explicit placement's
        split the replica owner). With a Cache attached that is an error (the cached path reads
        every pool on the reading layer's device; the load guard refuses such a split while it
        loads, so this is a safety net); without one, a note, since each forward then copies
        those pools whole. With an explicit placement, first one line: where the modules landed
        and what crosses between the devices.
        """
        from .dsv41.placement import REPLICA_HINT, format_cuts, format_layer_ranges
        c = self.config
        blocks = self._blocks()
        if len(blocks) != c.num_hidden_layers:
            return          # not the full graph (a test loading a slice of it)
        devs = [None if b.device is None else torch.device(b.device) for b in blocks]
        if any(d is None for d in devs):
            return
        if self.placement is not None:
            print(f" -- DSV41 placement [{self.placement.spec}]: {self._where()} | "
                  f"{self.placement.crossings()}", flush = True)
        crossing = [i for i, (owner, _, _) in sorted(self.pool_routes.items())
                    if devs[owner] != devs[i]]
        if not crossing:
            return
        free = format_cuts(self.load_guard.free)
        ranges = format_layer_ranges(crossing)
        if self._cache_attached():
            raise RuntimeError(
                f"DeepSeek-V4.1 load check failed: layers {ranges} read a compressed-KV pool on "
                f"another device, which the cached path cannot do; reload with every change of "
                f"device at {free}. {REPLICA_HINT}")
        print(f" !! DSV41: layers {ranges} read a compressed-KV pool on another device; without a "
              f"Cache each forward copies those pools whole, and a Cache would need every change "
              f"of device at {free}. {REPLICA_HINT}", flush = True)

    def _where(self) -> str:
        """Where the top-level modules landed, as runs of one device: 'cpu embed | cuda:0 ...'"""
        from .dsv41.placement import format_layer_ranges
        runs = []
        for m in self.modules:
            d = "none" if m.device is None else str(torch.device(m.device))
            if runs and runs[-1][0] == d:
                runs[-1][1].append(m.key)
            else:
                runs.append((d, [m.key]))
        def span(names):
            out, lay = [], []
            for x in names + [None]:
                if x is not None and x.startswith("layers."):
                    lay.append(int(x.split(".")[1]))
                    continue
                if lay:
                    out.append(f"layers.{format_layer_ranges(lay)}")
                    lay = []
                if x is not None:
                    out.append(x)
            return ",".join(out)
        def devname(d):
            # CUDA_VISIBLE_DEVICES can reorder the cards, so name them. Only asked of an
            # initialized runtime, so a check run without a load stays off the GPU
            try:
                dv = torch.device(d)
                if dv.type == "cuda" and torch.cuda.is_initialized():
                    return f"{d} ({torch.cuda.get_device_name(dv)})"
            except Exception:
                pass
            return d
        return " | ".join(f"{devname(d)} {span(names)}" for d, names in runs)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        """
        One user turn as the checkpoint's chat_template.jinja renders it with its defaults:
        BOS, the system block with the default reasoning effort (75), the fullwidth role
        markers, and thinking on. V4's ASCII markers tokenize as plain text on V4.1.
        """
        p = ("<｜begin▁of▁sentence｜><｜System｜>"
             "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough "
             "the reasoning)\n\n")
        if system_prompt:
            p += system_prompt
        p += f"<｜User｜>{prompt}<｜Assistant｜><think>"
        return p
