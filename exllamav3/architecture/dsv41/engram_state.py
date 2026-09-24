"""
N-gram context carry for the DeepSeek-V4.1 engram across forwards.

The engram hashes every position together with the max_ngram_size - 1 = 3
compressed token ids before it (architecture/dsv41/engram_torch.py). A forward
that starts at position p > 0 therefore needs the compressed ids of positions
p-3 .. p-1, which belong to an EARLIER forward: the previous prefill chunk, the
previous decode step, or the accepted part of a speculative verify. DeepSeek's
reference keeps a full-sequence cache of compressed ids (NgramHashState.cache,
indexed by start_pos); here each sequence slot keeps a small position-indexed
ring of them instead, as exllamav3 recurrent state.

    DSV41EngramState      one per engram module per Cache, created by the Cache
                          from DSV41Engram.layer_state_cls like every other
                          recurrent layer state (the module's layer_idx is
                          -(1000 + backbone layer), so its key never collides
                          with the attention's own state key).

The ring holds compressed ids -- DEAD included, since a dead token must keep
blocking the lookback of the positions after it -- at row position % R.
Entries are only ever read for positions below the chunk start, and every
forward rewrites the positions it covers, so:

  * rewind is free: DSV4State.rewind only moves the job position back, and the
    ids for the new chunk's lookback are still in the ring (see RING_ROWS);
    the rejected positions are overwritten when the sequence re-advances.
  * every row is tagged with its position, so a lookback is exact or raises:
    never another position's id, however the rewinds were split up.
  * a checkpoint is the ring's ids up to the stash position (16 KiB), since
    the job state may rewind in place below a restored checkpoint.
  * nothing lives on a device: hashing runs on the host, where the token ids
    already are.

Job-level protocol (what DSV4State / a V4.1 job state calls, via
cache.get_all_recurrent_layers()): alloc, free, clear, rewind, stash, unstash,
get_checkpoint_size, storage_size, tp_export. No extra wiring is needed in the
job state: it clears, stashes and unstashes every recurrent layer already.

Forward protocol. DSV41Engram.forward is self-sufficient: with
params["recurrent_states"] present it reads each row's lookback from its own
state (rows are (r.slot, r.position), position = chunk start, since
advance_recurrent_states runs after the forward) and writes the chunk's ids
back at the END of its forward. The two functions below expose the same two
steps for a caller that wants them outside the module (a model/State forward,
a harness, a pass that skips the engram but must keep the carry current):

    set_lookback(params)   before the forward: params["dsv41_engram_lookback"]
                           = [B, 3] from the states (also returned)
    commit_chunk(params)   after the forward: write the chunk's compressed ids
                           into every engram state of the cache

Both are idempotent with what the module does itself (writing the same ids at
the same positions twice is a no-op), so calling them in addition is safe.
"""

from __future__ import annotations

import torch

from ...constants import PAGE_SIZE
from .engram_torch import UNK

# Ring rows per slot. The generator rewinds a recurrent job IN PLACE whenever the
# rewind fits the job state's rollback_capacity() (generator/job.py), and for the
# DSA job state that is position - window_beg, bounded by the attention ring
# (DSV4LayerState.ring_rows: window + 2 pages of overprovision, rounded up to pages;
# 768 rows for V4.1's 128-token window). guaranteed_rollback (PAGE_SIZE) is not a floor of
# that capacity: right after the window moves up a page it can be smaller (V4.1's job
# state can report 0), and the generator checks rollback_capacity() before every in-place
# rewind. 4 pages cover ring_rows + the 3-id lookback with margin at 16 KiB per slot.
# Every row is tagged with the position it holds, so a lookback the ring can no longer
# serve raises in lookback() instead of hashing with another position's id -- however
# the rewinds that led there were split up.
RING_ROWS = 4 * PAGE_SIZE


class DSV41EngramState:
    """
    Per-slot host ring of the compressed token ids an engram layer has seen,
    position-indexed. Conforms to the recurrent layer-state protocol of
    DSV4LayerState / PLELayerState.
    """

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        from ...cache.dsa import DSV4State
        self.module = module
        self.cache_id = cache_id
        self.max_batch_size = max_batch_size
        self.max_history = max_history
        self.ctx = int(module.ctx)
        self.R = RING_ROWS
        need = self.ctx + max(DSV4State.guaranteed_rollback, max_history + 1)
        if self.R < need:
            raise ValueError(
                f"DSV41EngramState: max_history = {max_history} needs a {need - self.ctx}-token "
                f"rollback plus the {self.ctx}-id lookback, {need} rows, but the engram ring holds "
                f"{self.R}; max_history can be at most {self.R - self.ctx - 1}")
        # The deepest in-place rewind the job state allows is the attention ring (see RING_ROWS),
        # so this ring must hold that plus the lookback. It does for V4.1-Flash (768 + 3 rows);
        # a model with a sliding window above 256 (an attention ring of 1024 rows or more) would
        # otherwise fail in lookback() partway through a generation. A stand-in module without
        # a config (a unit test) skips this
        window = getattr(getattr(module, "config", None), "sliding_window", None)
        if window is not None:
            ring_rows = -(-(int(window) + 2 * PAGE_SIZE) // PAGE_SIZE) * PAGE_SIZE
            if ring_rows + self.ctx > self.R:
                raise ValueError(
                    f"DSV41EngramState: the attention ring of sliding_window = {window} holds "
                    f"{ring_rows} rows, which an in-place rewind can reach, plus the {self.ctx}-id "
                    f"lookback is more than the engram ring's {self.R} rows")
        self.ids = torch.empty((max_batch_size, self.R), dtype = torch.long, device = "meta")
        # The position each row holds (-1: none), checked on every read. The frontier alone
        # cannot tell whether a row still holds an old position: after a rewind the ring also
        # holds the rejected positions past it, and those overwrote the rows of positions a
        # later, deeper rewind may need again (two rewinds of ~600 tokens, each well inside
        # max_rewind, reach ids the first pass overwrote)
        self.pos = torch.empty((max_batch_size, self.R), dtype = torch.long, device = "meta")
        # End of the last write per slot (-1: nothing since clear). Positions past it were
        # never written in this timeline and read as UNK (a test state created past 0)
        self.hi = [-1] * max_batch_size
        self.device = None

    # ---- layer-state protocol ----

    def alloc(self, device = None):
        # Host-side like PLELayerState.id_state: the hash runs on the CPU ids. `device`
        # is the module's device, recorded only to mark the state as live
        self.ids = torch.full((self.max_batch_size, self.R), UNK, dtype = torch.long)
        self.pos = torch.full((self.max_batch_size, self.R), -1, dtype = torch.long)
        self.hi = [-1] * self.max_batch_size
        self.device = device if device is not None else torch.device("cpu")

    def free(self):
        self.ids = torch.empty((self.max_batch_size, self.R), dtype = torch.long, device = "meta")
        self.pos = torch.empty((self.max_batch_size, self.R), dtype = torch.long, device = "meta")
        self.hi = [-1] * self.max_batch_size
        self.device = None

    def clear(self, slot: int):
        # A job state clears every recurrent layer on creation, possibly before the model
        # (and so this state) is loaded
        if self.device is not None:
            self.ids[slot].fill_(UNK)
            self.pos[slot].fill_(-1)
            self.hi[slot] = -1

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        pass  # position-indexed: the job state moves the position, stale ids are overwritten

    def stash(self, slot: int, position: int) -> torch.Tensor:
        """
        [2, n] int64: the ids the ring holds for positions position-n .. position-1 and
        the positions they belong to (-1 where the ring no longer holds one), n =
        min(position, R). Not only the ctx ids the next chunk needs: the job state restores
        the attention's window_beg with the checkpoint, so right after unstash it allows an
        in-place rewind BELOW the checkpoint position, and that chunk's lookback must still
        be there. 16 KiB, next to the attention's ring slice in the same checkpoint.
        """
        self.lookback(slot, position)           # the next forward's ids must be exact
        n = min(position, self.R)
        p = torch.arange(position - n, position)
        rows = p % self.R
        never = p > self.hi[slot]               # never written: UNK, as lookback reads them
        held = (self.pos[slot, rows] == p) | never
        ids = torch.where(never, UNK, self.ids[slot, rows])
        return torch.stack([torch.where(held, ids, UNK), torch.where(held, p, -1)])

    def unstash(self, slot: int, stashed: torch.Tensor, position: int):
        self.ids[slot].fill_(UNK)
        self.pos[slot].fill_(-1)
        ids, p = stashed.to(torch.long)
        ok = p >= 0
        rows = p[ok] % self.R
        self.ids[slot, rows] = ids[ok]
        self.pos[slot, rows] = p[ok]
        self.hi[slot] = position - 1

    def get_checkpoint_size(self) -> int:
        return 2 * self.R * torch.long.itemsize

    def storage_size(self) -> int:
        # host memory, not VRAM
        return 2 * self.max_batch_size * self.R * torch.long.itemsize

    def tp_export(self, plan):
        raise NotImplementedError("DSV41EngramState: tensor-parallel loading is not supported")

    # ---- engram API ----

    @property
    def max_rewind(self) -> int:
        """Deepest in-place rewind, below the furthest position the slot has written, that
        lookback() always serves. Deeper ones are served while the ring still holds the
        ids and raise once it does not."""
        return self.R - self.ctx

    def lookback(self, slot: int, pos0: int) -> torch.Tensor:
        """
        [ctx] compressed ids of positions pos0-ctx .. pos0-1, oldest first. UNK for
        positions before the sequence start and for positions never written since the
        slot was cleared (a test state created past position 0). Raises for a position
        the ring no longer holds: overwritten by a later one (rewinds that went deeper than
        the ring, in one step or several), or older than a restored checkpoint's window.
        """
        assert self.device is not None, "DSV41EngramState: lookback on a state that is not allocated"
        p = torch.arange(pos0 - self.ctx, pos0)
        rows = p % self.R
        live = (p >= 0) & (p <= self.hi[slot])
        tag = self.pos[slot, rows]
        lost = live & (tag != p)
        if lost.any():
            i = int(lost.nonzero()[0])
            q, t = int(p[i]), int(tag[i])
            why = f"overwritten by position {t}" if t >= 0 else "not in the restored checkpoint"
            raise RuntimeError(
                f"engram lookback for position {pos0} (slot {slot}) needs the id of position {q}, "
                f"which the {self.R}-row ring no longer holds ({why}): the rewind exceeds "
                f"max_rewind {self.max_rewind}")
        return torch.where(live, self.ids[slot, rows], UNK)

    def write(self, slot: int, pos0: int, cids: torch.Tensor):
        """Record the compressed ids of positions pos0 .. pos0+len-1 (the last R of them)."""
        s = cids.shape[-1]
        n = min(s, self.R)
        p = torch.arange(pos0 + s - n, pos0 + s)
        rows = p % self.R
        self.ids[slot, rows] = cids.reshape(-1)[s - n:].to("cpu", torch.long)
        self.pos[slot, rows] = p
        # The frontier is the end of THIS write, not the max ever seen: after a rewind the
        # ids past it belong to rejected positions, and re-advancing overwrites them
        self.hi[slot] = pos0 + s - 1


# ---- functions for a caller outside the module ----

def engram_states(cache) -> list[DSV41EngramState]:
    """Every engram state a Cache holds (one per engram module and layer instance)."""
    return [l for l in cache.get_all_recurrent_layers().values() if isinstance(l, DSV41EngramState)]


def _rows(params: dict):
    rs = params.get("recurrent_states")
    if not rs:
        return None
    if getattr(rs[0], "exported", False):
        raise NotImplementedError("DSV41 engram: tensor-parallel recurrent states are not supported")
    assert all(r is not None for r in rs), "DSV41 engram: a batch row has no recurrent state"
    return rs


def state_lookback(params: dict, state: DSV41EngramState) -> torch.Tensor | None:
    """[B, ctx] lookback of every row of params["recurrent_states"] from `state`, or None
    for a stateless forward."""
    rs = _rows(params)
    if rs is None:
        return None
    return torch.stack([state.lookback(r.slot, r.position) for r in rs])


def set_lookback(params: dict, cache = None) -> torch.Tensor | None:
    """
    (a) Before a forward: set params["dsv41_engram_lookback"] to the [B, ctx] ids that
    precede each row's chunk, read from the recurrent states. Returns it, or None (and
    leaves params alone) for a stateless forward. DSV41Engram.forward re-derives the same
    thing from its own state, so this is for callers that want it explicitly.
    """
    rs = _rows(params)
    if rs is None:
        return None
    states = engram_states(cache if cache is not None else rs[0].cache)
    assert states, "set_lookback: the cache holds no engram state"
    lb = state_lookback(params, states[0])
    params["dsv41_engram_lookback"] = lb
    return lb


def commit_chunk(params: dict, cids: torch.Tensor | None = None, cache = None):
    """
    (b) After a forward: write the chunk's compressed ids ([B, L]; computed from
    params["input_ids"] when not given) into every engram state of the cache, at each
    row's chunk start. No-op for a stateless forward. Must run before
    advance_recurrent_states moves the positions.
    """
    rs = _rows(params)
    if rs is None:
        return
    states = [st for st in engram_states(cache if cache is not None else rs[0].cache)
              if st.device is not None]
    if not states:
        return
    if cids is None:
        cids = states[0].module.compress(params["input_ids"])
    assert cids.shape[0] == len(rs), f"{cids.shape[0]} id rows for {len(rs)} states"
    for st in states:
        for i, r in enumerate(rs):
            st.write(r.slot, r.position, cids[i])
