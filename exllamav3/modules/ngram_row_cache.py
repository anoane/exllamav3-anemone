"""
A RAM cache of n-gram / engram table rows that stream from disk (--ngram_ram <size>: the part of
the budget no whole table takes; doc/expert_tiers.md, "-ngr / --ngram_ram").

Direct-mapped: row id u lives in slot hash(u) mod slots, tagged with u; a later row that hashes
to the same slot replaces it. One slot holds the row of every cached table of the module (the
engram's fp8 weights and its e8m0 scales share their row ids), so one tag check serves them all.
Everything is vectorized over a gather's (sorted, unique) row ids with torch on the CPU: the lookup
is one hash, one index and one compare; hits are copied into the staging rows with index_select,
and the misses the disk delivered are copied into their slots once they land (insert), so a row
enters the cache only complete.

Thread-safe (a lock around lookup, fill and insert): a module stages on its prefetch worker and
inline on the forward's thread. What a gather reads is always the table's bytes: a hit copies what
a completed read stored, a miss reads the disk.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
import threading

import torch

_MUL = 0x9E3779B97F4A7C15 - (1 << 64)       # Fibonacci hashing multiplier as a signed int64


class RowCache:

    def __init__(self, key: str, row_bytes: list[int], nbytes: int):
        """row_bytes: the cached tables' row widths; nbytes: the cache's whole budget (tags included)"""
        self.key = key
        self.row_bytes = list(row_bytes)
        per = sum(self.row_bytes) + 8
        self.slots = max(0, nbytes // per)
        if self.slots < 1024:
            raise ValueError(f"{key}: a row cache of {nbytes} bytes holds {self.slots} rows; too small to help")
        self.shift = 64 - max(1, (self.slots - 1).bit_length())
        self.tags = torch.full((self.slots,), -1, dtype = torch.long)
        self.rows = [torch.empty((self.slots, rb), dtype = torch.uint8) for rb in self.row_bytes]
        self.nbytes = self.slots * per
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def resize(self, nbytes: int):
        """
        Give the cache a new budget (tags included), dropping what it holds: the end of a load
        shrinks the caches a ram ngram=auto budget sized before the expert tier's RAM was known
        (model/ram_budget.py, refit_row_caches). Below 1024 rows the cache holds nothing and every
        lookup misses. Only where a row comes from changes, never its bytes
        """
        per = sum(self.row_bytes) + 8
        slots = max(0, nbytes // per)
        if slots < 1024:
            slots = 0
        with self.lock:
            self.tags, self.rows = None, []
            self.slots = slots
            self.shift = 64 - max(1, (slots - 1).bit_length()) if slots else 0
            self.tags = torch.full((slots,), -1, dtype = torch.long)
            self.rows = [torch.empty((slots, rb), dtype = torch.uint8) for rb in self.row_bytes]
            self.nbytes = slots * per

    def slot_of(self, uids: torch.Tensor) -> torch.Tensor:
        """Slot of each row id: the top bits of uid x the golden-ratio constant (int64 wraparound), mod slots"""
        h = (uids * _MUL) >> self.shift
        return torch.remainder(h, self.slots)

    def lookup(self, uids: torch.Tensor, outs: list[torch.Tensor]):
        """Copy the cached rows of uids (int64, CPU) into outs (one [>= len(uids), row_bytes] uint8 per
        cached table) at their positions; returns (positions of the misses, their slots)"""
        if not self.slots:
            n = uids.numel()
            with self.lock:
                self.misses += n
            return torch.arange(n, dtype = torch.long), torch.zeros((n,), dtype = torch.long)
        slots = self.slot_of(uids)
        with self.lock:
            hit = self.tags.index_select(0, slots) == uids
            hpos = hit.nonzero().squeeze(1)
            if hpos.numel():
                hs = slots.index_select(0, hpos)
                for out, rows in zip(outs, self.rows):
                    out[:uids.numel()].index_copy_(0, hpos, rows.index_select(0, hs))
            mpos = (~hit).nonzero().squeeze(1)
            self.hits += int(hpos.numel())
            self.misses += int(mpos.numel())
        return mpos, slots.index_select(0, mpos)

    def insert(self, uids: torch.Tensor, slots: torch.Tensor, rows: list[torch.Tensor]):
        """Store the rows the disk delivered (rows[t][i] of uids[i]) in their slots: rows first, tags
        after, under the lock (a lookup never sees a tag before its row). Two misses of one gather
        hashing to one slot: the later one stays"""
        n = uids.numel()
        if not n or not self.slots:
            return
        # one row per slot (the later of two that collide): index_copy_ with repeated indices may
        # take the rows of one and the tag of the other
        order = torch.argsort(slots, stable = True)
        ss = slots.index_select(0, order)
        last = torch.ones((n,), dtype = torch.bool)
        last[:-1] = ss[1:] != ss[:-1]
        sel = order[last]
        slots, uids = slots.index_select(0, sel), uids.index_select(0, sel)
        with self.lock:
            if int(slots.max()) >= self.slots:
                return                  # looked up before a resize: its slots are not this cache's
            for cache_rows, r in zip(self.rows, rows):
                cache_rows.index_copy_(0, slots, r[:n].index_select(0, sel))
            self.tags.index_copy_(0, slots, uids)

    def stats(self) -> dict:
        return {"slots": self.slots, "bytes": self.nbytes, "hits": self.hits, "misses": self.misses}


class Landing:
    """The misses of one gather, read into scratch rows: land() copies them to their staging
    positions and into the cache (once)"""

    def __init__(self, cache: RowCache, uids, slots, pos, scratch: list, outs: list):
        self.cache, self.uids, self.slots, self.pos = cache, uids, slots, pos
        self.scratch, self.outs = scratch, outs
        self.landed = False

    def land(self):
        if self.landed:
            return
        for out, sc in zip(self.outs, self.scratch):
            out.index_copy_(0, self.pos, sc)
        self.cache.insert(self.uids, self.slots, self.scratch)
        self.landed = True


class CachedTicket:
    """A disk ticket of a gather's misses and its Landing: waiting for it also lands the rows (the
    DiskTicket calls the consumer makes: promote, wait, release)"""

    def __init__(self, ticket, landing: Landing):
        self.ticket = ticket
        self.landing = landing

    def promote(self, cls: int):
        self.ticket.promote(cls)

    def wait(self):
        self.ticket.wait()
        self.landing.land()

    def release(self):
        self.ticket.release()


def read_whole(handle, threads: int = 8, chunk: int = 64 << 20) -> torch.Tensor:
    """A table held whole in RAM: [rows, row_bytes] uint8, read with positioned reads on `threads`
    threads (the n-gram budget's whole tables)"""
    n = handle.num_rows * handle.row_bytes
    out = torch.empty((handle.num_rows, handle.row_bytes), dtype = torch.uint8)
    view = memoryview(out.numpy()).cast("B")
    fd = os.open(handle.filename, os.O_RDONLY)
    try:
        def read(a: int):
            b = min(n, a + chunk)
            while a < b:
                got = os.preadv(fd, [view[a:b]], handle.abs_offset + a)
                if got <= 0:
                    raise OSError(f"{handle.filename}: read {handle.key} ended at byte {a} of {n}")
                a += got
        with ThreadPoolExecutor(max_workers = threads) as ex:
            list(ex.map(read, range(0, n, chunk)))
    finally:
        os.close(fd)
    return out
