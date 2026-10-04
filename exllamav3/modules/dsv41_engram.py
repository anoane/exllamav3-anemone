# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
DeepSeek-V4.1 engram layer as an exllamav3 module.

Derived in part from vLLM vllm/models/deepseek_v4_1/common/engram.py
(Apache-2.0); see NOTICE.

An engram layer looks up n-gram hashes of the input token ids in a very large
embedding table and gates the result into the hyper-connection streams. In
DeepSeek-V4.1-Flash there are two of them, layers 1 and 14; DSV41Block calls
them at block entry. Stages: hash (host) -> dedup -> gather (disk, host) ->
dequant -> wkv -> gate (device).

THE TABLE DOES NOT GO ON A DEVICE. Per layer the checkpoint carries

    engram.embed.weight   [384,006,168, 256]  fp8_e4m3    91.5 GiB
    engram.embed.scale    [384,006,168,   8]  fp8_e8m0     2.9 GiB

which is ~94.4 GiB for one layer and ~189 GiB for the two: more than any
single GPU, and more host RAM than most machines have to spare. It stays on disk:
the unique rows a chunk needs are pread out of the checkpoint shards
(exllamav3's C++ ngram_gather_cpu pool, FADV_RANDOM), shipped raw -- 264 bytes
a row -- and dequantised on the device. ``load`` allocates nothing for the table,
``weights_numel`` does not count it, and ``table_numel`` reports it separately.
The module is a registered submodule of its block, so load/unload, the Cache
(recurrent state) and Model.prefetch all find it. The autosplit's measuring
forward reads no rows: it stages the worst case of one distinct row per hashed
n-gram, so it sees the VRAM a forward of its size can allocate.

The small tensors are ordinary:

    engram.wkv            6144 -> 25600, exl3   (suh / svh / trellis / mul1)
    engram.q_weight       [hc_mult, hidden]  fp32 (bf16 in the original shards)
    engram.k_weight       [hc_mult, hidden]  fp32 (bf16 in the original shards)

The wkv output is (hc_mult + 1) * hidden: the LEADING hc_mult blocks are one
gate key per stream, the trailing block is the value all streams share
(DeepSeek's Engram.forward: kv.split([hc_mult * dim, dim])).

DUPLICATE WEIGHTS. A converted checkpoint keeps the original engram shards next
to the converted ones (the tables live only there), so it carries some engram
tensors twice:

  * engram.wkv as exl3 (suh / svh / trellis / mul1) and as the original fp8 pair
    (weight / scale). The names do not collide; Linear loads the exl3 tensors and
    the fp8 pair is stale.
  * engram.q_weight / k_weight under the SAME names: fp32 in the converted shards,
    bf16 in the original ones. The safetensors collection keeps whichever file its
    directory scan lists last, which differs between layers and machines, and says
    so with one " !! Overriding ..." line per tensor (four on V4.1-Flash, all
    expected). load_gate therefore picks the copy itself, from the file headers and
    never from the scan order: the bf16 copy of the original shards when the
    checkpoint carries one (DeepSeek's weights as released; of several bf16 copies,
    the first by file path), else the first copy by file path. It reads that copy's
    exact values into fp32, as the reference's .float() does, and not through the
    loader's bf16 -> fp16 staging. The fp32 copies in the V4.1-Flash conversion equal
    the bf16 values after that staging, not the bf16 values: they differ in some of
    the elements below fp16's normal range. They are not read while a bf16 copy
    exists; a conversion saves what load_gate loaded, so one made with the exact read
    saves the exact values.

GENERATION. Each position hashes the 3 compressed ids before it, so a chunk
past position 0 needs ids from an earlier forward. They come from a per-slot
ring kept as recurrent state (architecture/dsv41/engram_state.py): read at the
chunk start, written at the end of this module's forward. The module's
layer_idx is -(1000 + backbone layer) so that state's key never collides with
the attention's (backbone, 0); the backbone index is ``backbone_idx``.

PREFETCH. Hash, dedup and gather depend only on the token ids and the carried
lookback, so Model.forward/prefill stages them for a prefill-sized chunk on a
worker thread before block 0 is issued (``prefetch``): layer 1's rows then
overlap block 0 and layer 14's blocks 0-13. forward() takes the staged set
whose history equals its own and stages inline otherwise, so a stale prefetch
can only cost time. One worker per model, so layer 1 is always staged first.
"""

from __future__ import annotations

import os
import weakref
from concurrent.futures import ThreadPoolExecutor

from typing_extensions import override

import numpy as np
import torch
from torch import nn

from . import Module
from .dsv41_ablation import ablated
from .dsv41_engram_math import stable_engram_gate
from .linear import Linear
from ..ext import exllamav3_ext as ext
from ..loader.safetensors import DiskTensorHandle, convert_dtype
from ..model.math_policy import STABLE_ARITHMETIC, EXACT_ROWS
from ..architecture.dsv41.engram_state import DSV41EngramState, state_lookback
from ..architecture.dsv41.engram_torch import DEAD, UNK, dequant_rows, engram_hash_chunk
from .ngram_row_cache import Landing, CachedTicket

PREFETCH_ENABLED = os.environ.get("EXL3_DSV41_ENGRAM_PREFETCH", "1") != "0"   # A/B switch
PREFETCH_MIN_TOKENS = 256   # positions (bsz * seq) below which prefetch() declines: decode
                            # gathers are ~48 parallel preads, cheaper than the thread hop
MAX_PIN_SETS = 2            # one with the last forward's uploads in flight, one being staged
PIN_MAX_ROWS = 24 * 4096    # largest pinned staging set kept: 25.5 MiB requested, about 34 MiB
                            # page-locked after the host allocator's power-of-two rounding (and
                            # the blocks a growing set gives up stay cached, page-locked);
                            # bigger chunks stage through a transient pageable set --
                            # page-locked memory cannot be swapped out, so it is kept small


def _disk_engine() -> bool:
    """
    Whether the rows are read by the disk engine (doc/disk_engine.md): when EXL3_DISK_BACKEND
    names a backend (or disk_engine_configure did), or is unset / auto and the extension's auto
    choice (disk/disk_auto.h) routes auto to the engine; never with EXL3_DISK_BACKEND=original,
    and only where the engine exists (not on Windows, where ngram_gather_cpu keeps its
    overlapped path whatever the variable says).
    The extension decides, as it does for ngram_gather_cpu; an invalid value raises here, at
    the first gather. With the engine, one native call per layer reads both tables, with a
    request class: the prefetch worker's look-ahead is class 2 until its forward waits for it
    (then class 0); inline gathers are class 0, and decode-sized ones hold bulk reads back.
    Otherwise: two ngram_gather_cpu calls.
    """
    return ext.disk_ngram_route() == "engine"


class _PinSet:
    """
    Staging buffers for one chunk's gather: the inverse map and the raw fp8 / e8m0 rows.
    Pinned when the module lives on a GPU (non-blocking uploads); `held` while a queued
    prefetch or a running forward owns it; `event` marks the last uploads read from it.
    """

    def __init__(self, pinned: bool):
        self.pinned = pinned
        self.cap = 0
        self.inv = self.w = self.s = None
        self.outs = None
        self.event = None
        self.held = False

    def grow(self, n: int, row_bytes: int, scale_bytes: int):
        if self.cap < n:
            p = self.pinned
            self.inv = torch.empty(n, dtype = torch.long, pin_memory = p)
            self.w = torch.empty((n, row_bytes), dtype = torch.uint8, pin_memory = p)
            self.s = torch.empty((n, scale_bytes), dtype = torch.uint8, pin_memory = p)
            self.outs = [self.w, self.s]    # disk_gather_rows' outputs: built once per growth
            self.cap = n


class _EngramCtx:
    """Per-model shared pieces: the hasher (token map + layout tensors, ~0.3 s to build)
    and the single prefetch worker. The worker thread stops once no engram module of the
    model has its table open (attach / release), and a later load starts a new one."""

    def __init__(self, config, layout):
        from ..architecture.dsv41.engram_torch import EngramHasher, build_compressed_token_map
        token_map, size = build_compressed_token_map(os.path.join(config.directory, "tokenizer.json"))
        if size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"engram: the tokenizer's compressed vocab has {size} ids, the config says "
                f"{config.engram_compressed_vocab_size}; every hash multiplier derives from it, so "
                f"a mismatch would hash every n-gram into the wrong table rows")
        if token_map.numel() != config.vocab_size:
            raise ValueError(f"engram: the tokenizer's token map covers {token_map.numel()} ids, "
                             f"the config's vocab_size is {config.vocab_size}")
        self.hasher = EngramHasher(layout, token_map, size, config.engram_pad_token_id)
        self._executor = None
        self._users = weakref.WeakSet()

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "dsv41_engram")
        return self._executor

    def attach(self, module):
        """A module has opened its table (DSV41Engram.open_table)."""
        self._users.add(module)

    def release(self, module):
        """A module unloads, after draining its own prefetches. The last one stops the worker
        thread, as upstream's n-gram embedding shuts its executor down on unload."""
        self._users.discard(module)
        if not self._users and self._executor is not None:
            self._executor.shutdown(wait = True)
            self._executor = None

    @staticmethod
    def get(config, layout) -> "_EngramCtx":
        c = getattr(config, "_dsv41_engram_ctx", None)
        if c is None:
            c = _EngramCtx(config, layout)
            config._dsv41_engram_ctx = c
        return c


class DSV41Engram(Module):
    """One engram layer. Present only on ``engram_layer_ids``."""

    def __init__(
        self,
        config,
        key: str,
        *,
        layer_idx: int,
        hidden_size: int,
        hc_mult: int,
        qmap: str | None = None,
        qbits_key: str = "bits",
    ):
        super().__init__(config, key, None)
        self.module_name = "DSV41Engram"
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult

        layout = config.engram_layout()
        assert layout is not None and layer_idx in layout.layer_ids, \
            f"{key}: layer {layer_idx} is not an engram layer"
        self.layout = layout
        self.backbone_idx = layer_idx
        self.table_index = layout.layer_ids.index(layer_idx)
        self.ctx = layout.max_ngram_size - 1

        # gather width -> (hc_mult + 1) streams; both read from the checkpoint
        # rather than assumed, since the +1 is the part that surprises
        self.gather_width = layout.n_hash_cols * layout.head_dim
        self.wkv_out = hidden_size * (hc_mult + 1)
        self.wkv = Linear(
            config, f"{key}.wkv", self.gather_width, self.wkv_out,
            qmap = qmap, out_dtype = torch.half, qbits_key = qbits_key,
        )
        self.register_submodule(self.wkv)

        # Recurrent state: the n-gram lookback carried between forwards. The negative index
        # keeps the Cache's state key apart from the attention's (backbone, 0), as PLE does
        self.layer_idx = -(1000 + layer_idx)
        self.caps.update({"recurrent_cache": True, "prefetch_ids": True})
        self.layer_state_cls = DSV41EngramState
        self.recurrent_layers = []

        self.q_weight = None
        self.k_weight = None
        self.qk = None
        self.w_handle = self.s_handle = None
        self.engram_ctx = None
        self._pins = []
        self._pending = []          # queued prefetches, oldest first: {"hist", "pin", "future"}
        self._disk_argv = None      # (fds, offsets, row_bytes) for disk_gather_rows, per fd pair
        self.io_direct = -1         # the placement's disk io= (-1 the engine's choice, 0 buffered, 1 O_DIRECT)
        self.ram_tables = {}        # table index (0 rows, 1 scales) -> the table held whole in RAM
        self.streamed = [0, 1]      # the tables read from disk
        self.row_cache = None       # RowCache of the streamed tables' rows
        self.prefetch_stats = {"hit": 0, "miss": 0, "retired": 0}

    @property
    def hasher(self):
        return self.engram_ctx.hasher if self.engram_ctx is not None else None

    @override
    def optimizer_targets(self):
        # The table is not quantized here and the gate weights are fp32;
        # only wkv is an optimizer target, and it reports itself
        return self.wkv.optimizer_targets()

    def _gate_weight(self, name: str, device: torch.device) -> torch.Tensor:
        """
        q_weight or k_weight as fp32 on `device`: the exact values of the copy the module
        docstring's DUPLICATE WEIGHTS rule names (the bf16 one when the checkpoint carries one,
        else the first by file path), read from that copy's own file whichever copy the
        collection kept. bf16, fp16 and fp32 all convert to fp32 exactly; nothing goes through
        the loader's bf16 -> fp16 staging.
        """
        key = f"{self.key}.{name}"
        # the collection that serves this key (a variant collection, --override, holds none of
        # its own file headers)
        stc = self.config.stc.find_stc(key)
        copies = sorted(f for f, h in stc.file_headers.items() if isinstance(h.get(key), dict))
        new = getattr(stc, "new_tensors", None)
        if not copies or (new and key in new):
            # a tensor the conversion has just produced for this module comes first, as in
            # get_tensor; no copy at all ends in get_tensor's missing-tensor error
            return stc.get_tensor(key, device, allow_bf16 = True, no_defer = True).float()
        bf16 = [f for f in copies if stc.file_headers[f][key].get("dtype") == "BF16"]
        filename = (bf16 or copies)[0]
        header = stc.file_headers[filename]
        meta = header[key]
        handle = DiskTensorHandle(key, filename, header["_header_offset"] + meta["data_offsets"][0],
                                  meta["shape"], convert_dtype(meta["dtype"])[0])
        try:
            t = handle.read_range(0, handle.num_rows)
        finally:
            handle.close()
        return t.float().to(device)

    def load_gate(self, device: torch.device):
        """q_weight / k_weight, and their product as the forward uses it."""
        self.q_weight = nn.Parameter(self._gate_weight("q_weight", device), requires_grad = False)
        self.k_weight = nn.Parameter(self._gate_weight("k_weight", device), requires_grad = False)
        if self.q_weight.shape != (self.hc_mult, self.hidden_size) or \
                self.k_weight.shape != self.q_weight.shape:
            raise ValueError(f"{self.key}: q_weight {tuple(self.q_weight.shape)} and k_weight "
                             f"{tuple(self.k_weight.shape)}, expected {(self.hc_mult, self.hidden_size)}")
        # The reference's Parameters are bf16 and only their fp32 product is used
        # (q_weight.float() * k_weight.float()). Read exactly, the weights are those bf16
        # values, so the product below equals the reference's in every element; the loader's
        # fp16 staging would change 103 (layer 1) and 71 (layer 14) of V4.1-Flash's 20,480
        # products, by at most 4.6e-13. The bf16 rounding changes nothing for bf16 values; it
        # brings a checkpoint that holds only fp32 copies to the reference's precision
        self.qk = (self.q_weight.data.to(torch.bfloat16).float()
                   * self.k_weight.data.to(torch.bfloat16).float()).contiguous()

    def open_table(self):
        """
        Table handles and the shared hasher -- everything the host side of the forward
        needs, and nothing on a device. Rows stay on disk: handles, not tensors.
        """
        stc = self.config.stc
        self.w_handle = stc.get_tensor_handle(f"{self.key}.embed.weight")
        self.s_handle = stc.get_tensor_handle(f"{self.key}.embed.scale")
        # The placement's disk rule (model/disk_source.py): the rows from a checked copy of the
        # table's shards (disk ngram=<dir>), and the page-cache mode of the reads (disk io=)
        from ..model.placement import parse as parse_placement
        from ..model.disk_source import source_dir, copy_path, io_direct, check_route
        placement = parse_placement(getattr(self.config.infer_params, "placement", None))
        copy_dir = source_dir(placement, "ngram")
        if copy_dir:
            self.w_handle, self.s_handle = (
                DiskTensorHandle(h.key, copy_path(h.filename, copy_dir, "ngram"), h.abs_offset, h.shape, h.dtype)
                for h in (self.w_handle, self.s_handle))
        self.io_direct = io_direct(placement)
        if self.io_direct >= 0:
            check_route(placement, "engine" if _disk_engine() else "original", "DeepSeek-V4.1's engram rows")
        # The n-gram budget (--ngram_ram, the placement's ram ngram=; model/ram_budget.py): tables
        # held whole in RAM (smallest first: the scales), a cache of the rows of those that stream
        from ..model.ram_budget import ngram_in_ram, ngram_row_cache_bytes, register_row_cache
        from ..util.memory import check_host_memory
        from .ngram_row_cache import RowCache, read_whole
        self.ram_tables = {}
        tables = self.ngram_tables()
        for t, (name, nbytes, _, _) in enumerate(tables):
            if ngram_in_ram(self.config, f"{self.key}.{name}", nbytes):
                check_host_memory(nbytes, f"{self.key}.{name} held in RAM (--ngram_ram)")
                self.ram_tables[t] = read_whole((self.w_handle, self.s_handle)[t])
        self.streamed = [t for t in range(len(tables)) if t not in self.ram_tables]
        self.row_cache = None
        cache_bytes = ngram_row_cache_bytes(self.config, self.key) if self.streamed else 0
        if cache_bytes:
            check_host_memory(cache_bytes, f"the row cache of {self.key} (--ngram_ram)")
            self.row_cache = RowCache(self.key, [tables[t][2] for t in self.streamed], cache_bytes)
            register_row_cache(self.config, self.key, self.row_cache)
        # Equal byte widths do not establish the encoding: BF16[D/2] has the
        # same stride as FP8[D], but the raw gather would silently reinterpret it.
        # Check original dtype tags, exact shapes and file ranges without reading
        # payloads or modifying the engine's shared safetensors loader.
        from ..architecture.dsv41.checkpoint_header import validate_engram_handles
        validate_engram_handles(self.w_handle, self.s_handle, self.layout.head_dim,
                                self.layout.num_embeddings[self.table_index])
        # ngram_gather_cpu reads whatever row it is given: the largest bucket row the hash
        # can produce must lie inside the table
        t = self.table_index
        top = int(self.layout.offsets[t][-1]) + int(self.layout.primes[t][-1][-1])
        if top > self.w_handle.num_rows:
            raise ValueError(f"{self.key}: bucket rows reach {top}, table has {self.w_handle.num_rows}")
        self.engram_ctx = _EngramCtx.get(self.config, self.layout)
        self.engram_ctx.attach(self)

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)          # wkv
        self.load_gate(device)
        self.open_table()
        for rl in self.recurrent_layers:
            rl.alloc(device)

    @override
    def unload(self):
        self._drain_prefetch()                  # queued workers still read the table; first
        for rl in self.recurrent_layers:
            rl.free()
        self._pins = []
        if self.engram_ctx is not None:
            self.engram_ctx.release(self)       # the last engram module stops the worker
        self.engram_ctx = None
        self.qk = None
        for h in (self.w_handle, self.s_handle):
            if h is not None:
                # the disk engine keeps its own descriptors of the shards: let them go (a no-op
                # without an engine; a shard another layer still reads is simply reopened)
                try:
                    ext.disk_engine_forget(h.filename)
                except Exception:
                    pass
        self._disk_argv = None
        self.w_handle = self.s_handle = None
        self.ram_tables = {}
        self.streamed = [0, 1]
        self.row_cache = None
        self.q_weight = None
        self.k_weight = None
        super().unload()

    @override
    def get_tensors(self):
        t = {
            f"{self.key}.q_weight": self.q_weight.data,
            f"{self.key}.k_weight": self.k_weight.data,
        }
        t.update(self.wkv.get_tensors())
        return t

    def ngram_tables(self) -> list:
        """[(name, bytes, row bytes, read by every gathered row)] of the two tables, from the layout (the
        n-gram budget, model/ram_budget.py): the fp8 rows and their e8m0 scales, gathered together"""
        n = self.layout.num_embeddings[self.table_index]
        hd = self.layout.head_dim
        return [("embed.weight", n * hd, hd, True), ("embed.scale", n * (hd // 32), hd // 32, True)]

    def table_numel(self) -> int:
        """Elements of the disk-resident table, weight and scale together."""
        n = self.layout.num_embeddings[self.table_index]
        return n * (self.layout.head_dim + self.layout.head_dim // 32)

    @override
    def weights_numel(self):
        # What this layer places on a device: wkv and the two gate vectors. The table is
        # excluded: ~98G elements that never leave the disk. Counting it would make every
        # block-level sum (conversion accounting, anything sizing a block) see 94 GiB
        return self.wkv.weights_numel() + 2 * self.hc_mult * self.hidden_size

    def tp_export(self, plan, producer):
        raise NotImplementedError("DSV41Engram: tensor-parallel loading is not supported")

    # ---- hashing history and the carried lookback ----

    def compress(self, ids: torch.Tensor) -> torch.Tensor:
        return self.hasher.compress(ids)

    def state_key(self, instance: int = 0) -> tuple:
        """This module's key in Cache.recurrent_layers."""
        return self.layer_idx, instance

    def _state(self, params: dict) -> DSV41EngramState:
        rs = params["recurrent_states"]
        st = rs[0].cache.get_recurrent_layer(self.state_key(params.get("layer_instance", 0)))
        assert isinstance(st, DSV41EngramState), f"{self.key}: recurrent layer is {type(st).__name__}"
        return st

    @staticmethod
    def explicit_lookback(params: dict) -> torch.Tensor | None:
        """
        An explicit lookback for this forward: params["dsv41_engram_lookback"] on a direct
        module call, or the per-forward slot DeepseekV41Model.prepare_inputs moves it to
        (so a params dict reused for the next forward does not carry it along).
        """
        lb = params.get("dsv41_engram_lookback")
        return lb if lb is not None else params.get("dsv41_engram_lookback_fwd")

    def history(self, ids: torch.Tensor, params: dict):
        """
        (hist [B, ctx + L] compressed ids behind their lookback, state or None). The
        lookback comes from this module's recurrent state when params carries
        recurrent_states (row r reads at its chunk start r.position); an explicit lookback
        given as well must equal it. A stateless forward takes the explicit lookback if
        given, else must start at position 0.
        """
        cids = self.compress(ids)
        B = cids.shape[0]
        state = None
        explicit = self.explicit_lookback(params)
        if explicit is not None:
            explicit = explicit.to("cpu", torch.long)
            if explicit.shape != (B, self.ctx):
                raise ValueError(f"{self.key}: dsv41_engram_lookback {tuple(explicit.shape)}, "
                                 f"expected {(B, self.ctx)}")
        if params.get("recurrent_states"):
            state = self._state(params)
            lookback = state_lookback(params, state)
            if ablated("engram_nocarry"):
                # negative control (EXL3_DSV41_ABLATE): lose the carried ids at every chunk start
                lookback = torch.full_like(lookback, UNK)
            if explicit is not None and not torch.equal(explicit, lookback):
                raise ValueError(
                    f"{self.key}: the explicit engram lookback {explicit.tolist()} differs from "
                    f"the ids the recurrent state carried {lookback.tolist()}")
        elif explicit is None:
            if params.get("position", 0) != 0 or params.get("past_len", 0):
                raise NotImplementedError(
                    f"{self.key}: engram needs the {self.ctx} token ids before this chunk "
                    f"(position {params.get('position', params.get('past_len'))}): pass "
                    f"recurrent_states, or params['dsv41_engram_lookback'] for a stateless chunk")
            lookback = torch.full((B, self.ctx), UNK, dtype = torch.long)
        else:
            lookback = explicit
        return torch.cat([lookback, cids], dim = 1), state

    def _commit(self, hist: torch.Tensor, state: DSV41EngramState | None, params: dict):
        """Write this chunk's ids into the state ring, after they have been hashed."""
        if state is not None:
            for i, r in enumerate(params["recurrent_states"]):
                state.write(r.slot, r.position, hist[i, self.ctx:])

    def commit_only(self, params: dict):
        """Keep the carried ids current for a forward in which the engram does not run
        (EXL3_DSV41_ABLATE=engram)."""
        if self.engram_ctx is None or not params.get("recurrent_states"):
            return
        ids = params.get("input_ids")
        if ids is None:
            return
        hist, state = self.history(ids, params)
        self._commit(hist, state, params)

    # ---- staging: hash + dedup + gather, inline or on the prefetch worker ----

    def _fds(self, engine: bool | None = None):
        if engine is None:
            engine = _disk_engine()
        fds = []
        for h in (self.w_handle, self.s_handle):
            fd = h._ensure_open()           # stc.close() at the end of load closes it; reopen
            if hasattr(os, "posix_fadvise") and not engine:     # the engine advises its own
                # 256-byte rows scattered over 91 GiB: readahead only evicts useful pages
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            fds.append(fd)
        return fds

    def _disk_args(self, fw: int, fs: int):
        """disk_gather_rows' per-table lists, built once per descriptor pair (a reopen after
        stc.close() may change the numbers)."""
        a = self._disk_argv
        if a is None or a[0][0] != fw or a[0][1] != fs:
            a = self._disk_argv = ([fw, fs],
                                   [self.w_handle.abs_offset, self.s_handle.abs_offset],
                                   [self.w_handle.row_bytes, self.s_handle.row_bytes])
        return a

    def _acquire_pin(self, n: int) -> _PinSet:
        """A staging set not held by a queued prefetch, grown to n rows."""
        rb, sb = self.layout.head_dim, self.layout.head_dim // 32
        pinned = self.device is not None and self.device.type == "cuda"
        free = []
        if pinned and n <= PIN_MAX_ROWS:
            free = [p for p in self._pins if not p.held]
            if not free and len(self._pins) < MAX_PIN_SETS:
                free = [_PinSet(pinned = True)]
                self._pins += free
            if not free:
                # every set is held by a queued prefetch: retire one, preferably one whose
                # staging already finished (a stale guess), else the oldest
                held = [e for e in self._pending if e["pin"].pinned]
                done = [e for e in held if e["future"].done()]
                if held:
                    self._retire(done[0] if done else held[0])
                    free = [p for p in self._pins if not p.held]
        if not free:
            # CPU module, an oversized chunk, or nothing to reclaim: a transient pageable
            # set, dropped after the forward (its uploads are synchronous)
            free = [_PinSet(pinned = False)]
        pin = free[0]
        pin.grow(n, rb, sb)
        pin.held = True
        return pin

    @torch.inference_mode()
    def _stage(self, hist: torch.Tensor, pin: _PinSet, measure: bool, fds = None,
               cls: int = 0, hold: bool = False, wait: bool = True):
        """Hash, dedup and gather one chunk into pin; returns (unique row count, ticket). Runs
        on the prefetch worker or inline (inference mode is thread-local). cls / hold / wait:
        the disk engine's request class, hold rule, and whether to wait for the rows; with
        wait=False and the engine, ticket is the DiskTicket still reading them (else None),
        which the consumer promotes, waits for and releases (_finish)."""
        if pin.event is not None:
            # the previous forward's non-blocking uploads read this set
            pin.event.synchronize()
            pin.event = None
        if measure:
            # The autosplit's measuring forward reads no rows, and its token ids are stand-ins
            # (zeros hash to a few dozen rows), so it stages the worst case instead: every
            # hashed n-gram a distinct row (U = n), rows of zeros with scale 2^0. The forward
            # then allocates what a chunk of distinct n-grams would
            n = hist.shape[0] * (hist.shape[1] - self.ctx) * self.layout.n_hash_cols
            torch.arange(n, out = pin.inv[:n])
            pin.w[:n].zero_()
            pin.s[:n].fill_(127)
            return n, None
        idx = engram_hash_chunk(hist, self.table_index, self.hasher)         # [B, L, 24]
        # numpy's unique: single-threaded, and bitwise the same sorted ids and inverse as
        # torch.unique, whose parallel sort shares the intra-op pool with everything else on
        # the host (a 197k-id dedup took 179 ms with the cores busy, 8.9 ms here)
        u, iv = np.unique(idx.reshape(-1).numpy(), return_inverse = True)
        uids, inv = torch.from_numpy(u), torch.from_numpy(iv.reshape(-1).astype(np.int64, copy = False))
        U, n = uids.numel(), inv.numel()
        pin.inv[:n].copy_(inv)
        if U:
            # tables held whole in RAM (--ngram_ram)
            for t, table in self.ram_tables.items():
                torch.index_select(table, 0, uids, out = pin.outs[t][:U])
            if not self.streamed:
                return U, None
            # the row cache: hits straight into the staging rows, the misses read into scratch rows
            # and landed (staging and cache) once the read completes
            ids, landing = uids, None
            if self.row_cache is not None:
                outs = [pin.outs[t] for t in self.streamed]
                mpos, mslots = self.row_cache.lookup(uids, outs)
                if not mpos.numel():
                    return U, None
                ids = uids.index_select(0, mpos)
                scratch = [torch.empty((ids.numel(), pin.outs[t].shape[1]), dtype = torch.uint8) for t in self.streamed]
                landing = Landing(self.row_cache, ids, mslots, mpos, scratch, outs)
            engine = _disk_engine()
            fw, fs = fds if fds is not None else self._fds(engine)
            if engine:
                # rows land at row i of the whole staging tensors: no slicing in Python, and
                # positional arguments (pybind's keyword path costs ~1.5 us more per call)
                fl, offs, rbs = self._disk_args(fw, fs)
                if landing is None and len(self.streamed) == 2:
                    t = ext.disk_gather_rows(uids, 0, fl, offs, rbs, pin.outs, cls, hold, wait, None, 0, 1, 0,
                                             self.io_direct)
                    return U, t
                sel = self.streamed
                outs = landing.scratch if landing is not None else [pin.outs[t] for t in sel]
                t = ext.disk_gather_rows(ids, 0, [fl[i] for i in sel], [offs[i] for i in sel], [rbs[i] for i in sel],
                                         outs, cls, hold, wait, None, 0, 1, 0, self.io_direct)
                if landing is None:
                    return U, t
                if t is None:
                    landing.land()
                    return U, None
                return U, CachedTicket(t, landing)
            handles = (self.w_handle, self.s_handle)
            fdl = (fw, fs)
            for j, t in enumerate(self.streamed):
                out = landing.scratch[j] if landing is not None else pin.outs[t][:U]
                ext.ngram_gather_cpu(fdl[t], handles[t].abs_offset, handles[t].row_bytes, ids, 0, out)
            if landing is not None:
                landing.land()
        return U, None

    @staticmethod
    def _finish(ticket):
        """The forward needs a staged chunk's rows now: whatever the look-ahead has not read
        yet becomes class 0 (full depth, ahead of bulk), then wait and release."""
        if ticket is None:
            return
        try:
            ticket.promote(0)
            ticket.wait()
        finally:
            ticket.release()                # cancels what is queued, waits out what is in flight

    def _match(self, hist: torch.Tensor) -> dict | None:
        for e in self._pending:
            if e["hist"].shape == hist.shape and torch.equal(e["hist"], hist):
                return e
        return None

    def _forget(self, entry: dict):
        # by identity: list.remove would compare the entries' history tensors
        self._pending = [e for e in self._pending if e is not entry]

    def _retire(self, entry: dict):
        # Drop a queued prefetch no forward will take. Its worker may still be writing the
        # staging set, so wait it out unless it hasn't started, and release its disk ticket
        # (cancels the queued reads, waits for those in flight) before the set is reused
        self._forget(entry)
        self.prefetch_stats["retired"] += 1
        f = entry["future"]
        if not f.cancel():
            try:
                _, ticket = f.result()
                if ticket is not None:
                    ticket.release()
            except Exception:
                pass
        entry["pin"].held = False

    def _drain_prefetch(self):
        for e in list(self._pending):
            self._retire(e)

    def prefetch(self, x: torch.Tensor, params: dict):
        """
        Called by Model.forward/prefill before block 0: stage this chunk's rows on the
        shared worker. Declines decode-sized chunks, the autosplit's measuring pass, and
        anything it cannot build a history for (the forward then raises or stages inline).
        """
        if not PREFETCH_ENABLED or self.engram_ctx is None or params.get("autosplit_measure"):
            return
        if ablated("engram"):
            return
        ids = params.get("input_ids", x)
        if ids is None or ids.dim() != 2 or ids.shape[0] * ids.shape[1] < PREFETCH_MIN_TOKENS:
            return
        try:
            hist, _ = self.history(ids, params)
        except Exception:
            return
        if self._match(hist) is not None:
            return
        pin = self._acquire_pin(hist.shape[0] * (hist.shape[1] - self.ctx) * self.layout.n_hash_cols)
        fds = self._fds()                   # lazy open isn't thread-safe; do it here
        # class 2 without waiting: the rows are read while block 0 runs, behind any demand
        # reads, and forward() promotes what is left to class 0 when it needs them
        self._pending.append({
            "hist": hist,
            "pin": pin,
            "future": self.engram_ctx.executor.submit(self._stage, hist, pin, False, fds, 2,
                                                      False, False),
        })

    # ---- forward ----

    def _gate(self, h, key, eps):
        """
        The gate of the tokens of h and key (B, L, H, D) fp32, (B, L, H): torch reductions over
        the call, whose layout follows its shape. forward's docstring has the formula.
        """
        D = h.shape[-1]
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * self.qk * key).sum(-1) * rstd * D ** -0.5
        return torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))

    def _gate_rows(self, h, key, eps):
        """
        EXL3_EXACT_ROWS: _gate one token per call, on the token's (B, 1, H, D) slices of h and
        key: the operands of a one-token step, whose reductions run over freshly allocated
        pointwise results.
        """
        return torch.cat([self._gate(h[:, l:l + 1], key[:, l:l + 1], eps) for l in range(h.shape[1])], dim = 1)

    def forward(self, x, params, out_dtype = None):
        """
        streams (B, L, H, D) fp32 -> streams with the gated n-gram value added.

        DeepSeek's reference Engram.forward:
            kv = wkv(rows(hash_ids).flatten)              -> [.., (H + 1) * D]
            key, value = kv.split([H * D, D])             # one key per stream, one shared value
            rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)   # per (token, stream)
            dot  = sum(h * q_w * k_w * key) * rstd * D^-0.5
            gate = sigmoid(copysign(sqrt(clamp(|dot|, 1e-6)), dot))
            gate = 0 where token_mask is False    (dead tokens)
            out  = h + gate * value
        with eps = rms_norm_eps (1e-20).
        """
        B, L, H, D = x.shape
        ids = params.get("input_ids")
        if ids is None:
            # Only the autosplit's measuring pass runs without token ids (it measures VRAM and
            # reads no rows). Anywhere else, hashing stand-in ids would inject a plausible,
            # finite and wrong n-gram term into every position
            if not params.get("autosplit_measure"):
                raise KeyError(f"{self.key}: params['input_ids'] is required -- the engram hashes "
                               f"the token ids (DeepseekV41Model.prepare_inputs sets them)")
            ids = torch.zeros((B, L), dtype = torch.long)
        hist, state = self.history(ids, params)
        assert hist.shape == (B, self.ctx + L), f"{self.key}: ids {tuple(hist.shape)} vs streams {tuple(x.shape)}"
        n = B * L * self.layout.n_hash_cols

        measure = bool(params.get("autosplit_measure"))
        entry = None if measure else self._match(hist)
        if entry is not None:
            self._forget(entry)
            self.prefetch_stats["hit"] += 1
            pin = entry["pin"]
            try:
                U, ticket = entry["future"].result()
                self._finish(ticket)
            except BaseException:
                pin.held = False            # don't strand the set on a failed gather
                raise
        else:
            self.prefetch_stats["miss"] += 1
            pin = self._acquire_pin(n)
            try:
                # decode-sized: rows the next block needs now, bulk reads held back meanwhile
                U, _ = self._stage(hist, pin, measure, None, 0, B * L < PREFETCH_MIN_TOKENS)
            except BaseException:
                pin.held = False
                raise

        dev = x.device
        nb = pin.pinned
        w_d = pin.w[:U].to(dev, non_blocking = nb)
        s_d = pin.s[:U].to(dev, non_blocking = nb)
        inv_d = pin.inv[:n].to(dev, non_blocking = nb)
        if nb and dev.type == "cuda":
            pin.event = torch.cuda.Event()
            pin.event.record(torch.cuda.current_stream(dev))
        pin.held = False

        rows = dequant_rows(w_d, s_d)                                        # [U, 256] fp16, exact
        emb = rows.index_select(0, inv_d).view(B, L, -1)                     # [B, L, 24 * 256]
        kv = self.wkv.forward(emb, params, out_dtype = torch.float)
        key = kv[..., :H * D].view(B, L, H, D)
        value = kv[..., H * D:].view(B, L, 1, D)
        h = x.float()
        eps = self.config.rms_norm_eps
        if STABLE_ARITHMETIC and h.is_cuda:
            # torch picks its reduction by shape, so a token's gate would change in the last bit
            # with the token count of the call; the row-local kernel reduces each (token, stream)
            # alone (the same gate, scaled against overflow: dsv41_engram_math.py)
            gate = stable_engram_gate(h, key, self.qk, eps)
        elif EXACT_ROWS and params.get("exact_rows") and L > 1:
            gate = self._gate_rows(h, key, eps)
        else:
            gate = self._gate(h, key, eps)
        dead = hist[:, self.ctx:] == DEAD
        if dead.any():
            # DeepSeek's token_mask: a dead token (an id outside the text vocab; image spans in
            # the reference) takes no engram contribution -- its gate is shut, not just its hash
            # padded
            gate = gate.masked_fill(dead.to(dev).unsqueeze(-1), 0.0)
        out = h + gate.unsqueeze(-1) * value

        # after the hash has consumed the lookback: the next chunk's context
        self._commit(hist, state, params)
        return out
