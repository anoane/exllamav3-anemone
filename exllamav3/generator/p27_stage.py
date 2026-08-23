"""P27 — lookahead chunk staging (EXL3_PREFILL_STAGE=1).

Deploys to exllamav3/generator/p27_stage.py. After a prefill chunk's kernels are enqueued
(the GPU has ~300 ms of work), gather the NEXT chunk's embedding on the CPU, stage it
through a 2-deep pinned ring, and upload it on a side copy stream behind an event. The
embedding module returns the device tensor for an identity-matched chunk (misprediction
falls through to the eager path), which removes the per-chunk serial host phase
measured (~59 MB fp32 CPU gather + pageable upload, the dominant part of the 76 ms @8k
wall-kernel gap). The chunk ids are staged alongside so hash-MoE's params["input_ids"]
consumer gets a device tensor too.

Slot-reuse safety: the side stream waits on the main stream before overwriting a device
buffer (all consumers of that buffer were enqueued on the main stream at least one full
chunk earlier), and the slot event is host-synchronized before its pinned buffer is
rewritten.
"""

import torch

PAGE_SIZE = 256

_state = {}


def _dev_of(d):
    return torch.device(f"cuda:{d}") if isinstance(d, int) else torch.device(d)


MAX_RINGS = 4     # distinct chunk lengths kept staged; oldest evicted beyond this


def _get(gen):
    st = gen.__dict__.get("_p27_state")   # on the generator: no id-recycling resurrection
    if st is None:
        m0 = gen.model.fwd_modules[0][0]
        dev = None
        for module, instance, idx in gen.model.fwd_modules[1:]:
            d = module.device
            if d is not None and _dev_of(d).type == "cuda":
                dev = _dev_of(d)
                break
        st = dict(m0 = m0, dev = dev,
                  stream = torch.cuda.Stream(dev) if dev is not None else None,
                  bufs = {})
        gen._p27_state = st
    return st


def stage_next(gen, seq, next_start):
    """Called right after model.prefill() returns for the current chunk. Predicts the
    next chunk's span the way Job.prefill slices it; a misprediction is harmless (the
    identity check in Embedding.forward falls through to the eager path)."""
    st = _get(gen)
    m0 = st["m0"]
    if st["dev"] is None:
        return                       # no CUDA consumer module found
    if m0.device is None or torch.device(m0.device).type != "cpu":
        return                       # GPU-resident embedding: nothing to hide
    next_end = (next_start + gen.max_chunk_size) // PAGE_SIZE * PAGE_SIZE
    next_end = min(next_end, len(seq.sequence_ids) - 1)
    # mirror Job.prefill's recurrent-last-page clamp (default env; EXL3_NO_RLP=1 skips it)
    import os as _os
    if gen.recurrent_cache is not None and _os.environ.get("EXL3_NO_RLP") != "1":
        last_page_b = (len(seq.sequence_ids) - 1) // PAGE_SIZE * PAGE_SIZE
        if next_start < last_page_b <= next_end:
            next_end = last_page_b
    if next_end <= next_start:
        return
    ids = seq.sequence_ids.torch_slice(next_start, next_end).clone()

    # Gather STRAIGHT INTO pinned memory in the table's dtype (no intermediate tensor, no
    # 59 MB host memcpy), upload at table width, and do the out-dtype cast + any scaling
    # on the GPU side stream -- the host cost collapses to the fp16 gather itself, so the
    # stage stays profitable even in deep-context chunks with little host slack.
    w = m0.embedding.weight          # CPU table
    out_dtype = m0.out_dtype or w.dtype
    L = ids.shape[1]
    hidden = w.shape[1]
    ring = st["bufs"].pop(L, None)   # re-insert = move to the back (LRU order)
    if ring is None:
        while len(st["bufs"]) >= MAX_RINGS:
            old = next(iter(st["bufs"]))
            for s in st["bufs"].pop(old)["slots"]:
                s["ev"].synchronize()          # no pending copies from freed pins
        ring = dict(slots = [], i = 0)
        for _ in range(2):
            ring["slots"].append(dict(
                pin_h = torch.empty((1, L, hidden), dtype = w.dtype, pin_memory = True),
                dev_t = torch.empty((1, L, hidden), dtype = w.dtype, device = st["dev"]),
                dev_h = torch.empty((1, L, hidden), dtype = out_dtype, device = st["dev"]),
                pin_i = torch.empty((1, L), dtype = ids.dtype, pin_memory = True),
                dev_i = torch.empty((1, L), dtype = ids.dtype, device = st["dev"]),
                ev = torch.cuda.Event(),
                gen = 0,
            ))
    st["bufs"][L] = ring
    slot = ring["slots"][ring["i"]]
    ring["i"] ^= 1
    slot["gen"] += 1                 # invalidates any staged dict still pointing here

    slot["ev"].synchronize()         # prior copy from this slot's pins has executed
    torch.index_select(w, 0, ids[0], out = slot["pin_h"].view(L, hidden))
    slot["pin_i"].copy_(ids)
    side = st["stream"]
    side.wait_stream(torch.cuda.current_stream(st["dev"]))   # after all enqueued consumers
    with torch.cuda.stream(side):
        slot["dev_t"].copy_(slot["pin_h"], non_blocking = True)
        if m0.multiplier != 1.0:
            slot["dev_t"].mul_(m0.multiplier)                # table dtype, BEFORE cast (eager order)
        slot["dev_h"].copy_(slot["dev_t"])                   # cast to out_dtype on GPU
        if m0.normalize:
            slot["dev_h"].mul_(hidden ** 0.5)
        slot["dev_i"].copy_(slot["pin_i"], non_blocking = True)
        slot["ev"].record()
    seq._p27_staged = dict(ids = ids, ids_dev = slot["dev_i"], h_dev = slot["dev_h"],
                           ev = slot["ev"], slot = slot, gen = slot["gen"])


def bt_device(seq, gen):
    """Once-per-job device copy of the (constant during prefill) block-index table,
    replacing a per-chunk pageable synchronous upload. Re-uploads if the source tensor
    object is ever rebuilt (identity guard against stale pages)."""
    bt = seq.block_index_tensor
    if bt is None or bt.device.type != "cpu":
        return bt
    st = _get(gen)
    if st["dev"] is None:
        return bt
    if getattr(seq, "_p27_bt_src", None) is not bt:
        seq._p27_bt_dev = bt.to(st["dev"])
        seq._p27_bt_src = bt
    return seq._p27_bt_dev
