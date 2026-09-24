"""
Two-stage pipelined prefill for a DeepSeek-V4.1 layer split across two GPUs.

With a layer split (e.g. layers 0-11 on the first GPU, 12-39 on the second) a chunk's
forward runs its two halves back to back, and inside each half the host blocks at sync
points (block-table uploads, the MoE row counts, the CPU layers' stream plans). So the
host does not queue chunk k+1's first half while the second GPU is still on chunk k:
the GPUs take turns. Whether overlap improves throughput depends on the placement,
transfer policy, chunk geometry and CPU-expert workload; measure the prepared model.

This driver runs one prefill call as sub-chunks and overlaps them:

    calling thread (first GPU, caller stream):        S1(k+1)  S1(k+2)  ...
    worker thread (second GPU, caller stream):        S2(k)    S2(k+1)  ...

Causality allows it: S1 of chunk k+1 reads only the first-half layers' caches, which S1 of
chunk k finished; S2 of chunk k reads the second-half caches and the chunk-k products of
the first half (streams and the carried pre-mix), all ordered after the event S1(k) records on
the first GPU.

S1 keeps the caller's first-device stream (normally the default stream), avoiding a
separate allocation pool for stage 1. S2 reads the first GPU's products on a side stream
that waits on that event; its destination work uses the caller's second-device stream.
When the call returns, the caller's first-device stream waits for the side stream.
Source allocation ownership is recorded separately from producer-event ordering. The
residual streams cross the split exactly as in the plain forward (the block's
prepare_for_device), so both paths compute the same function.

The job state's scalar bookkeeping (position, window_beg, wshift) is shared by every layer
and read live by the layer forwards, so the two halves need different values at the same
time. S1 advances the real state right after its half; S2 runs with a _StageView, a
snapshot of the chunk-start values over the same slot, cache and layer states. Both halves
compute the same SWA ring shift for a chunk; S2 checks it.

Each engram layer is prefetched and consumed by the thread of its own half (its staging
sets are not thread-safe), and the CPU-MoE worker (second-half layers only) begins a pass
per S2.

Opt-in: EXL3_DSV41_PIPELINE=1. EXL3_DSV41_PIPELINE_CHUNK sets the sub-chunk (default 4096,
at most the load's max_chunk_size, checked when the model loads). A call longer than one
sub-chunk is pipelined, as pieces of one sub-chunk (the last may be shorter), so no forward
on this path sees more rows than the load was sized for; anything else takes the plain path
(see eligible()). The generator hands the model one call per max_chunk_size tokens
(default 2048), so the pipeline only engages with a generator max_chunk_size above the
sub-chunk (e.g. 16384 with 4096-token sub-chunks).
"""

from __future__ import annotations

import os
import threading

import torch

from ...cache.recurrent_util import advance_recurrent_states

PIPELINE = os.environ.get("EXL3_DSV41_PIPELINE", "0") != "0"


def _positive_chunk(value) -> int:
    try:
        chunk = int(value)
    except (TypeError, ValueError):
        raise ValueError("EXL3_DSV41_PIPELINE_CHUNK must be a positive integer") from None
    if chunk <= 0:
        raise ValueError("EXL3_DSV41_PIPELINE_CHUNK must be a positive integer")
    return chunk


# Read only when the path is enabled, so a malformed value cannot affect the default path
SUB_CHUNK = _positive_chunk(os.environ.get("EXL3_DSV41_PIPELINE_CHUNK", "4096")) if PIPELINE else None

# params keys that change the prefill contract; any of them sends the call down the plain path
_PLAIN_KEYS = ("last_tokens_only", "export_state_layers", "capture", "quant_preserve",
               "autosplit_measure", "indexed_embeddings_required", "indexed_embeddings",
               "dsv41_engram_lookback", "dsv41_engram_lookback_fwd", "position_ids",
               "positions", "past_len", "position", "inv_freq", "batch_shape")


def _present(value) -> bool:
    """
    Whether a params value is set. A tensor counts as set whatever its contents (its truth
    value is ambiguous or data-dependent); anything else by its truth value, so the generator's
    empty indexed_embeddings list and a zero last_tokens_only are absent, as for the layers.
    """
    if isinstance(value, torch.Tensor):
        return True
    return bool(value)


def _record_source_tensors(value, stream, seen = None):
    """Protect CUDA allocations used on the source side stream until its work finishes.

    An event orders producers before consumers, but it does not tell the caching
    allocator that a tensor allocated on the default stream is still being read
    here. Copies between GPUs are host-asynchronous even with non_blocking=False.
    Do not rely on later CPU-offload or metadata uploads to incidentally synchronize.
    Only walk tensor containers; recurrent/cache objects retain their own storage.
    """
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        if value.device == stream.device:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_source_tensors(item, stream, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _record_source_tensors(item, stream, seen)


def _invalidate_state(state):
    """
    A failed overlapping forward can leave the job's layers at different positions: cursor
    rollback cannot restore overwritten rings or undo page writes. Refuse further use of this
    job state. The partial writes stay within the job's own slot (sliding-window rings,
    compressor carry, engram ring) and its own pages, exactly as when a plain forward fails
    halfway, and the generator discards those with the job (reap_failed_job); page contents are
    only taken as filled after a prefill call returns. So the Cache and other jobs stay usable.
    """
    state._dsv41_pipeline_failed = True


def check_state(params):
    for state in params.get("recurrent_states") or ():
        if getattr(state, "_dsv41_pipeline_failed", False):
            raise RuntimeError("DSV41 pipelined prefill failed for this job state; start the job "
                               "again with a new state (the generator does this for a new job)")


class _StageView:
    """
    The second half's view of a job state: slot, cache and every other attribute are the
    job's; position, window_beg, wshift and last_history are the chunk's own.
    """

    def __init__(self, state, position: int, window_beg: int):
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "window_beg", window_beg)
        object.__setattr__(self, "wshift", 0)
        object.__setattr__(self, "last_history", 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_state"), name)

    def __setattr__(self, name, value):
        if name in ("position", "window_beg", "wshift", "last_history"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_state"), name, value)


class _Plan:
    """
    Per-model split: the two module lists, their devices, prefetching layers, stream. ok says
    whether the load is pipelinable, reason why not.
    """

    def __init__(self, model):
        self.ok, self.reason = False, None
        fwd = model.fwd_modules
        devs = []
        for m, _, _ in fwd:
            d = getattr(m, "device", None)
            if d is not None and torch.device(d).type == "cuda" and torch.device(d) not in devs:
                devs.append(torch.device(d))
        if len(devs) != 2:
            self.reason = f"the modules are on {len(devs)} CUDA device(s), not two"
            return
        self.dev1, self.dev2 = devs
        split = next(i for i, (m, _, _) in enumerate(fwd)
                     if getattr(m, "device", None) is not None and torch.device(m.device) == self.dev2)
        self.st1, self.st2 = fwd[:split], fwd[split:]
        # every module of the second list must live on the second GPU (or be device-less)
        if not all(getattr(m, "device", None) is None or torch.device(m.device) == self.dev2
                   for m, _, _ in self.st2):
            self.reason = "a module past the split is not on the second GPU"
            return
        # The first half must hold at least the first decoder layer: every V4.1 layer owns a
        # sliding-window ring, so only then does stage 1 compute the chunk's ring shift that stage
        # 2 is checked against. An unplaced autosplit can put layer 0 on the second GPU and leave
        # only the stream expansion on the first (the embedding is in system RAM)
        first = model.modules[model.first_block_idx]
        if not any(m is first for m, _, _ in self.st1):
            self.reason = "no decoder layer before the split"
            return
        # A prefill stops at the last module that writes the cache. The split must come before
        # it: otherwise the first stage would be the whole prefill, and the second would run the
        # modules past the last cache writer (norm, head) for every sub-chunk
        last_kv = next((i for i, (_, instance, idx) in enumerate(fwd)
                        if (idx, instance) == model.last_kv_module_idx_instance), None)
        if last_kv is None or last_kv < split:
            self.reason = "the split is not before the last cache-writing layer"
            return
        # CPU-MoE queues/slots are shared by component, not by device. The normal
        # placement confines them to stage 2, but unrestricted autosplit need not.
        # Require this property from the actual loaded modules before using threads.
        def descendants(module):
            yield module
            for child in getattr(module, "modules", ()):
                yield from descendants(child)
        if any(getattr(child, "cpu_host", None) is not None
               for module, _, _ in self.st1 for child in descendants(module)):
            self.reason = "a layer before the split holds experts in system RAM"
            return
        # Repeated layer instances can share prefetch/module scratch state.
        if len({id(m) for m, _, _ in fwd}) != len(fwd):
            self.reason = "repeated layer instances"
            return
        self.ok = True
        pf = model._get_prefetch_layers
        self.pf1 = [m for m in pf if torch.device(m.device) == self.dev1]
        self.pf2 = [m for m in pf if torch.device(m.device) == self.dev2]
        self.hosts = list(getattr(model.config, "moe_cpu_hosts", {}).values())
        self.side1 = torch.cuda.Stream(device = self.dev1)


def _plan(model) -> _Plan:
    signature = (
        tuple((id(m), str(getattr(m, "device", None)), instance, idx)
              for m, instance, idx in model.fwd_modules),
        model.last_kv_module_idx_instance,
        tuple(id(h) for h in getattr(model.config, "moe_cpu_hosts", {}).values()),
    )
    p = getattr(model, "_dsv41_pipeline_plan", None)
    if p is None or p.signature != signature:
        p = _Plan(model)
        p.signature = signature
        model._dsv41_pipeline_plan = p
    return p


def report(model):
    """
    One line after a load with the switch on: whether this load's prefill calls can be
    pipelined, and if not, why. An ineligible load runs every call on the plain path, so a
    generator max_chunk_size above the load's max_chunk_size must not be used with it
    """
    plan = _plan(model)
    if plan.ok:
        print(f" -- DSV41 pipelined prefill: eligible, {len(plan.st1)} modules on {plan.dev1}, "
              f"{len(plan.st2)} on {plan.dev2}, sub-chunk {SUB_CHUNK}", flush = True)
    else:
        print(f" !! DSV41 pipelined prefill: this load is not pipelinable ({plan.reason}); every "
              f"prefill call takes the plain path, so keep the generator's max_chunk_size within "
              f"the load's", flush = True)


def _run(model, mods, x, params):
    for module, instance, idx in mods:
        params["layer_instance"] = instance
        last = (idx, instance) == model.last_kv_module_idx_instance
        params["prefill"] = last
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        if last:
            break
    params.pop("prefill", None)
    return x


def eligible(model, input_ids: torch.Tensor, params: dict) -> bool:
    """
    Whether this prefill call takes the pipelined path: the path is enabled, the model is
    loaded as a layer split (not tensor-parallel) over exactly two GPUs with at least the first
    decoder layer before the split and the split before the last cache-writing layer, no
    CPU-MoE layer in the first half and no repeated layer instance, and the call is one
    sequence longer than one sub-chunk, with a job state and a paged cache, and none of the
    options in _PLAIN_KEYS set. With the switch off it returns at once.
    """
    if not PIPELINE or model.loaded_tp:
        return False
    check_state(params)
    chunk = _positive_chunk(SUB_CHUNK)
    loaded_chunk = getattr(model, "_dsv41_loaded_max_chunk", None)
    if loaded_chunk is None:
        return False
    if chunk > loaded_chunk:
        raise ValueError(f"DSV41 pipeline chunk {chunk} exceeds the load-time maximum {loaded_chunk}")
    if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= chunk:
        return False
    rs = params.get("recurrent_states")
    if not rs or len(rs) != 1 or params.get("cache_seqlens") is None or "block_table" not in params:
        return False
    if any(_present(params.get(k)) for k in _PLAIN_KEYS) or params.get("recurrent_history"):
        return False
    return _plan(model).ok


def prefill_pipelined(model, input_ids: torch.Tensor, params: dict) -> None:
    """Prefill input_ids (1, L) as SUB_CHUNK pieces, first and second halves overlapped."""
    if not eligible(model, input_ids, params):
        raise ValueError("DSV41 pipelined prefill called with unsupported parameters or placement")
    plan = _plan(model)
    # CUDA current streams are thread-local. A fresh worker otherwise queues stage
    # 2 on its default stream, even when the caller uses another stream for this
    # device. Joining that thread only waits for enqueueing: later caller work
    # could then read partially updated caches. Capture per call (not per plan)
    # and queue stage 2 on the same destination stream as subsequent caller work.
    destination_stream = torch.cuda.current_stream(plan.dev2)
    rs = params["recurrent_states"][0]
    base = int(params["cache_seqlens"][0])
    # A caller error, found before any layer runs: nothing is written, nothing is marked failed
    if rs.position != base:
        raise ValueError(f"DSV41 pipelined prefill: job state at {rs.position}, cache_seqlens at {base}")
    L = input_ids.shape[1]
    spans = [(a, min(a + SUB_CHUNK, L)) for a in range(0, L, SUB_CHUNK)]
    shared = {k: v for k, v in params.items() if k not in ("cache_seqlens", "recurrent_states")}

    def stage1(a: int, b: int):
        with torch.cuda.device(plan.dev1):
            # holds by construction after the check above: stage 1 advances the job by b - a
            if rs.position != base + a:
                raise RuntimeError(f"DSV41 pipelined prefill: job at {rs.position}, chunk at {base + a}")
            p = dict(shared)
            p["cache_seqlens"] = torch.tensor([base + a], dtype = torch.int32)
            p["recurrent_states"] = [rs]
            ids = input_ids[:, a:b]
            x = model.prepare_inputs(ids, p)
            for m in plan.pf1:
                m.prefetch(x, p)
            x = _run(model, plan.st1, x, p)
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(plan.dev1))
            # the second half runs this chunk from its start; the job moves on for the next S1
            view = _StageView(rs, rs.position, rs.window_beg)
            shift1 = rs.wshift
            advance_recurrent_states(ids, p, model)
            p["recurrent_states"] = [view]
            return ids, x, p, view, shift1, done

    def stage2(ids, x, p, view, shift1, done):
        with torch.inference_mode(), torch.cuda.stream(plan.side1), torch.cuda.stream(destination_stream):
            plan.side1.wait_event(done)
            _record_source_tensors((ids, x, p), plan.side1)
            for m in plan.pf2:
                m.prefetch(x, p)
            for h in plan.hosts:
                h.begin_pass()
            _run(model, plan.st2, x, p)
            if view.wshift != shift1:
                raise RuntimeError(
                    f"DSV41 pipelined prefill: the second half shifted the SWA ring by {view.wshift}, "
                    f"the first by {shift1} (chunk at {view.position}); the split assumes one shift per chunk")

    worker, box = None, {}

    def run2(args):
        try:
            stage2(*args)
        except BaseException as e:               # re-raised on the calling thread
            box["e"] = e

    try:
        for a, b in spans:
            if "e" in box:
                raise box["e"]
            args = stage1(a, b)                      # overlaps the worker's S2 of the previous chunk
            if worker is not None:
                worker.join()
                if "e" in box:
                    raise box["e"]
            worker = threading.Thread(target = run2, args = (args,), name = "dsv41-prefill-s2")
            worker.start()
    except BaseException:
        _invalidate_state(rs)
        raise
    finally:
        if worker is not None:
            worker.join()
        # Order the caller's later work on the first GPU (e.g. a page-table defragmentation) after
        # everything stage 2 read there on the side stream, after a failure too
        torch.cuda.current_stream(plan.dev1).wait_stream(plan.side1)
    if "e" in box:
        _invalidate_state(rs)
        raise box["e"]
