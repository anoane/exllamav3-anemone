"""
The expert tier's policy, as the reference every implementation reproduces exactly
(doc/expert_tiers.md, "Policies"): the native core (exllamav3_ext/tier/tier_policy.cpp), and the
GPU lookup kernel with the tier thread. Torch-free; no bytes move here, only decisions.

Keys are lc * E + e: lc = the index of a cache layer among the tier's cache layers on its GPU, in
forward order, e = the routed expert. S = the pool's slots, spare = the slots kept free for
promotions, C = S - spare the experts the pool holds in steady state; R = the RAM tier's slots.

One call (lookup(), section 3.1 of doc/expert_tiers_impl_plan.md):

  1. seq numbers every tier call on a GPU; U = the call's unique experts in order of first
     occurrence, cnt = their occurrences.
  2. heat[e] += cnt * inc (decode: 1.0, prefill: EXL3_MOE_HEAT_PREFILL), 16.16 fixed point,
     halved every EXL3_MOE_HEAT_HALFLIFE decode tokens (lazily, per key).
  3. Phase 1, decode calls only: every hit of the call gets stamp = seq, before any victim is
     chosen, so a call never evicts an expert it uses.
  4. Phase 2, e in U order (access += 1 each): a hit; in routed mode (prefill) every miss is a
     transient; in decode mode a miss is admitted when the pool holds fewer than C experts, else as
     admit= says (always; heat: at least as warm as the would-be victim; adaptive: with the
     probability p, a draw keyed on the access counter). An admitted miss takes the lowest FREE
     slot (none: a transient, counted starved); when the pool then holds more than C, one victim
     is retired (evict=lru: oldest stamp, then lowest slot; lfu: coldest, then oldest, then lowest)
     and paired with the admission in the record. With spare=0 there is never a free slot once the
     pool is full: the victim's slot is reused in place.
  5. The record lists every unique expert of the call: hit, admit (with its victim) or transient.

The host, per record in order (host(), section 3.2): where each miss comes from (a VRAM slot still
retiring, the RAM tier, or the SSD), what RAM keeps of it (policy=), what happens to the victim
(demote=), RAM admission of SSD reads, then refills. It returns the list of actions the runtime
executes (copies, reads, demotions, returned slots); the decisions never depend on timing.

Deterministic mode (the default here, EXL3_MOE_TIER_DETERMINISTIC=1 in the engine): the host
applies every record before the next lookup, demotions complete at once. defer_returns=True
emulates production: the slot of a demoted victim stays RETIRING until complete() is called.
"""

from __future__ import annotations
from bisect import bisect_left, insort
from dataclasses import dataclass, field
from heapq import heappush, heappop, heapify

MASK64 = (1 << 64) - 1
U32 = (1 << 32) - 1
GOLDEN = 0x9E3779B97F4A7C15         # access draws: seed ^ (access * GOLDEN)
DRAW_MUL = 0xD1B54A32D192ED03       # RAM samples: seed ^ (draw * DRAW_MUL)
Q16 = 1 << 16

POLICIES = ("lazy-exclusive", "exclusive", "inclusive")
DEMOTES = ("swap", "heat", "all", "off")
ADMITS = ("adaptive", "heat", "always")
EVICTS = ("lru", "lfu")
RAM_ADMITS = ("always", "heat")

FREE, VALID, RETIRING, PINNED = 0, 1, 2, 3
STATE_NAMES = ("FREE", "VALID", "RETIRING", "PINNED")
HIT, ADMIT, TRANSIENT = 0, 1, 2
KIND_NAMES = ("hit", "admit", "transient")
DECODE, ROUTED, LAYER = 0, 1, 2

COUNTERS = ("calls", "hits", "admits", "admits_from_ram", "transients", "starved", "retires", "inplace",
            "drops", "d2h", "h2d_ram", "ssd", "rescued", "ram_evict", "dup_reclaim", "ram_admit_refused",
            "refills", "refill_dropped", "prefill_h2d", "prefill_ssd", "adapts")


def splitmix64(x: int) -> int:
    """SplitMix64's output function on x + golden (Steele et al.), 64-bit wraparound"""
    z = (x + GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def p_q32(p: float) -> int:
    """An admission probability as the threshold a 32-bit draw is compared against (2^32 = always)"""
    return min(1 << 32, max(0, int(p * 4294967296.0)))


class TierError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------- config

@dataclass(frozen = True)
class PolicyConfig:
    """What the tier decides. The placement's words, plus the EXL3_MOE_TIER_* / EXL3_MOE_HEAT_* tuning"""
    policy: str = "lazy-exclusive"          # ram policy=
    demote: str = "swap"                    # ram demote=
    admit: str = "adaptive"                 # cuda:<n> admit=
    evict: str = "lru"                      # cuda:<n> evict= (the VRAM pool)
    ram_evict: str = "lfu"                  # ram evict=
    ram_admit: str = "always"               # EXL3_MOE_TIER_RAM_ADMIT
    demote_on: bool = True                  # EXL3_MOE_TIER_DEMOTE
    disk_off: bool = False                  # disk experts=off
    spare: int = 8                          # cuda:<n> spare=
    sample: int = 16                        # EXL3_MOE_TIER_SAMPLE
    seed: int = 0
    halflife: int = 256                     # EXL3_MOE_HEAT_HALFLIFE (decode tokens)
    prefill_inc: int = 4096                 # EXL3_MOE_HEAT_PREFILL in 16.16 (1/16)
    adapt_every: int = 2048                 # EXL3_MOE_TIER_ADAPT_EVERY (accesses)
    pmin: float = 0.05                      # EXL3_MOE_TIER_ADMIT_PMIN
    p: float | None = None                  # EXL3_MOE_TIER_ADMIT_P (pinned), None: adaptive
    refill: bool = True                     # EXL3_MOE_TIER_REFILL
    refill_inflight: int = 2                # EXL3_MOE_TIER_REFILL_INFLIGHT

    def __post_init__(self):
        for name, value, allowed in (("policy", self.policy, POLICIES), ("demote", self.demote, DEMOTES),
                                     ("admit", self.admit, ADMITS), ("evict", self.evict, EVICTS),
                                     ("ram_evict", self.ram_evict, EVICTS), ("ram_admit", self.ram_admit, RAM_ADMITS)):
            if value not in allowed:
                raise ValueError(f"expert tier policy: {name}={value!r} is not one of {', '.join(allowed)}")
        if self.spare < 0 or self.sample < 1 or self.halflife < 1 or self.adapt_every < 1 or self.refill_inflight < 1:
            raise ValueError("expert tier policy: spare >= 0, sample, halflife, adapt_every and refill_inflight >= 1")

    @property
    def demotes(self) -> bool:
        """Whether a victim without a RAM copy may be copied back (policy, demote= and the master switch)"""
        return self.policy != "inclusive" and self.demote != "off" and self.demote_on

    @classmethod
    def from_placement(cls, device_store, ram_store, disk_store, tier_config, seed: int = 0) -> "PolicyConfig":
        """From the placement's cuda:<n> / ram / disk rules (placement_storage) and a TierConfig"""
        tc = tier_config
        return cls(policy = ram_store.policy, demote = ram_store.demote, admit = device_store.admit,
                   evict = device_store.evict, ram_evict = ram_store.evict, ram_admit = tc.ram_admit,
                   demote_on = tc.demote, disk_off = disk_store.experts == "off", spare = device_store.spare,
                   sample = tc.sample, seed = seed, halflife = tc.heat_halflife, prefill_inc = tc.heat_prefill_q16,
                   adapt_every = tc.adapt_every, pmin = tc.admit_pmin, p = tc.admit_p, refill = tc.refill,
                   refill_inflight = tc.refill_inflight)


# ---------------------------------------------------------------------------------------- heat

class Heat:
    """
    Per key: a u32 in 16.16 fixed point and the epoch it was last written in. The global epoch is
    decode_tokens // halflife; reading a key shifts its value right by the epochs since (at most 31),
    so every key halves every halflife decode tokens without a sweep. Adding saturates at 2^32 - 1.
    """

    def __init__(self, nkeys: int, halflife: int):
        self.v = [0] * nkeys
        self.ep = [0] * nkeys
        self.halflife = halflife
        self.tokens = 0

    @property
    def epoch(self) -> int:
        return self.tokens // self.halflife

    def read(self, k: int) -> int:
        d = self.epoch - self.ep[k]
        return self.v[k] >> min(31, d) if d > 0 else self.v[k]

    def add(self, k: int, inc: int):
        self.v[k] = min(U32, self.read(k) + inc)
        self.ep[k] = self.epoch

    def set(self, k: int, value: int):
        self.v[k] = min(U32, value)
        self.ep[k] = self.epoch


class AdaptiveP:
    """
    admit=adaptive's probability (Gill, FAST'08: PROMOTE). Every adapt_every accesses the host
    compares the two tiers' cache lives, in calls: lv = seq - the VRAM LRU slot's stamp, lr = seq -
    the RAM LRU entry's stamp, and moves p toward equal lives:
        p = min(pmax, max(pmin, p * (1 + (lv / max(1, lv + lr) - 0.5))))
    pmax = C / (C + R), the start value. A pinned p (EXL3_MOE_TIER_ADMIT_P) never moves.
    """

    def __init__(self, capacity: int, ram_slots: int, pmin: float, pinned: float | None = None):
        self.pmax = capacity / (capacity + ram_slots) if capacity + ram_slots > 0 else 1.0
        self.pmin = pmin
        self.pinned = pinned is not None
        self.p = float(pinned) if pinned is not None else self.pmax

    def update(self, lv: int, lr: int) -> float:
        if not self.pinned:
            ratio = lv / max(1, lv + lr)
            self.p = min(self.pmax, max(self.pmin, self.p * (1.0 + (ratio - 0.5))))
        return self.p

    @property
    def q32(self) -> int:
        return p_q32(self.p)


# ---------------------------------------------------------------------------------------- records

@dataclass
class Entry:
    key: int
    kind: int                   # HIT, ADMIT, TRANSIENT
    dst: int                    # VRAM slot (hit, admit) or staging slot (transient)
    cnt: int
    heat: int                   # the key's heat after this call's increment (16.16)
    vkey: int = -1              # paired victim (admit)
    vslot: int = -1
    vstamp: int = 0
    vheat: int = 0
    inplace: bool = False       # spare=0: the admission took its victim's slot

    def astuple(self):
        return (self.key, self.kind, self.dst, self.cnt, self.heat, self.vkey, self.vslot, self.vstamp,
                self.vheat, int(self.inplace))


@dataclass
class Record:
    seq: int
    lc: int
    mode: int
    entries: list = field(default_factory = list)

    def short(self):
        """[(kind, expert[, victim expert])] with keys as given (tests)"""
        out = []
        for x in self.entries:
            t = (KIND_NAMES[x.kind], x.key)
            out.append(t + ((x.vkey,) if x.vkey >= 0 else ()))
        return out


# ---------------------------------------------------------------------------------------- VRAM

class VramDirectory:
    """
    The pool of one GPU (S slots) plus its hot pins (after the pool), as the lookup kernel keeps it:
    per slot a state, a key and a stamp; per key its slot (or -1). occupied counts VALID pool slots.
    """

    def __init__(self, nkeys: int, slots: int, spare: int, pins: int = 0):
        self.S, self.spare, self.pins = slots, spare, pins
        n = slots + pins
        self.state = [FREE] * slots + [PINNED] * pins
        self.key = [-1] * n
        self.stamp = [0] * n
        self.slot_of = [-1] * nkeys
        self.free = list(range(slots))          # sorted
        self.occupied = 0
        self.access = 0
        self.retiring = {}                      # key -> RETIRING slot that still holds it
        self.lru = []                           # heap of (stamp, slot); stale entries skipped lazily

    def _stamp(self, slot: int, stamp: int):
        self.stamp[slot] = stamp
        if self.state[slot] == VALID:
            heappush(self.lru, (stamp, slot))
            if len(self.lru) > 4 * self.S + 64:
                self.lru = [(self.stamp[s], s) for s in range(self.S) if self.state[s] == VALID]
                heapify(self.lru)

    @property
    def capacity(self) -> int:
        return self.S - self.spare

    def place(self, key: int, slot: int, stamp: int = 0):
        """Seed a VALID pool slot (cold fill, tests), or a hot pin (slot >= S)"""
        if slot >= self.S:
            if self.state[slot] != PINNED or self.key[slot] >= 0:
                raise TierError(f"pin slot {slot} is taken")
        else:
            if self.state[slot] != FREE:
                raise TierError(f"slot {slot} is not free")
            self.free.remove(slot)
            self.state[slot] = VALID
            self.occupied += 1
        if self.slot_of[key] >= 0:
            raise TierError(f"key {key} is already in slot {self.slot_of[key]}")
        self.key[slot] = key
        self._stamp(slot, stamp)
        self.slot_of[key] = slot

    def victim(self, now: int, heat: Heat | None = None) -> int:
        """The slot the eviction rule picks now: VALID with a stamp before now; lru: (stamp, slot);
        lfu: (heat, stamp, slot). -1 when every VALID slot was used by this call"""
        if heat is None:
            h = self.lru
            while h:
                st, s = h[0]
                if self.state[s] != VALID or self.stamp[s] != st:
                    heappop(h)
                    continue
                return s if st < now else -1
            return -1
        best, bk = -1, None
        for s in range(self.S):
            if self.state[s] == VALID and self.stamp[s] < now:
                k = (self.stamp[s], s) if heat is None else (heat.read(self.key[s]), self.stamp[s], s)
                if bk is None or k < bk:
                    best, bk = s, k
        return best

    def retire(self, slot: int):
        k = self.key[slot]
        self.state[slot] = RETIRING
        self.slot_of[k] = -1
        self.occupied -= 1
        self.retiring[k] = slot

    def return_slot(self, slot: int):
        if self.state[slot] != RETIRING:
            raise TierError(f"slot {slot} returned while {STATE_NAMES[self.state[slot]]}")
        if self.retiring.get(self.key[slot]) == slot:
            del self.retiring[self.key[slot]]
        self.state[slot], self.key[slot], self.stamp[slot] = FREE, -1, 0
        insort(self.free, slot)

    def retiring_slot_of(self, key: int) -> int:
        """A RETIRING slot that still holds key (its bytes are intact until returned), or -1"""
        return self.retiring.get(key, -1)

    def lookup(self, lc: int, ids: list, mode: int, seq: int, E: int, heat: Heat, cfg: PolicyConfig,
               pq: int, inc: int) -> tuple:
        """One call of cache layer lc (3.1). ids: the call's routed experts (row-major, any count).
        Returns (the record, how many admissions found no free slot and became transients)"""
        now = seq
        uniq, cnt = [], {}
        for e in ids:
            e = int(e)
            if e not in cnt:
                cnt[e] = 0
                uniq.append(e)
            cnt[e] += 1
        keys = [lc * E + e for e in uniq]
        for k, e in zip(keys, uniq):
            heat.add(k, cnt[e] * inc)
        if mode == DECODE:
            for k in keys:
                s = self.slot_of[k]
                if s >= 0:
                    self._stamp(s, now)
        rec = Record(seq, lc, mode)
        lfu = heat if cfg.evict == "lfu" else None
        staging = starved = 0
        for k, e in zip(keys, uniq):
            self.access += 1
            s = self.slot_of[k]
            if s >= 0:
                rec.entries.append(Entry(k, HIT, s, cnt[e], heat.read(k)))
                continue
            adm = False
            if mode == DECODE:
                if self.occupied < self.capacity:
                    adm = True
                elif cfg.admit == "always":
                    adm = True
                elif cfg.admit == "heat":
                    v = self.victim(now, lfu)
                    adm = v >= 0 and heat.read(k) >= heat.read(self.key[v])
                else:
                    adm = (splitmix64(cfg.seed ^ ((self.access * GOLDEN) & MASK64)) >> 32) < pq
            ent = None
            if adm:
                if self.free:
                    slot = self.free.pop(0)
                    self.state[slot], self.key[slot] = VALID, k
                    self._stamp(slot, now)
                    self.slot_of[k] = slot
                    self.occupied += 1
                    ent = Entry(k, ADMIT, slot, cnt[e], heat.read(k))
                    if self.occupied > self.capacity:
                        vs = self.victim(now, lfu)
                        if vs >= 0:
                            vk = self.key[vs]
                            ent.vkey, ent.vslot, ent.vstamp, ent.vheat = vk, vs, self.stamp[vs], heat.read(vk)
                            self.retire(vs)
                elif self.spare == 0:
                    # no spare slots: the victim's slot is reused in place (its bytes are dropped
                    # or copied back before the admission's copy lands)
                    vs = self.victim(now, lfu)
                    if vs >= 0:
                        vk = self.key[vs]
                        ent = Entry(k, ADMIT, vs, cnt[e], heat.read(k), vk, vs, self.stamp[vs], heat.read(vk), True)
                        self.slot_of[vk] = -1
                        self.key[vs] = k
                        self._stamp(vs, now)
                        self.slot_of[k] = vs
                if ent is None:
                    starved += 1
            if ent is None:
                rec.entries.append(Entry(k, TRANSIENT, staging, cnt[e], heat.read(k)))
                staging += 1
            else:
                rec.entries.append(ent)
        return rec, starved

    def apply(self, rec: Record):
        """Replay a record decided elsewhere (the lookup kernel) on this mirror"""
        for x in rec.entries:
            if x.kind == HIT:
                if rec.mode == DECODE:
                    self._stamp(x.dst, rec.seq)
            elif x.kind == ADMIT:
                if x.inplace:
                    self.slot_of[x.vkey] = -1
                    self.key[x.dst] = x.key
                    self._stamp(x.dst, rec.seq)
                    self.slot_of[x.key] = x.dst
                    self.access += 1
                    continue
                self.free.remove(x.dst)
                self.state[x.dst], self.key[x.dst] = VALID, x.key
                self._stamp(x.dst, rec.seq)
                self.slot_of[x.key] = x.dst
                self.occupied += 1
                if x.vkey >= 0:
                    self.retire(x.vslot)
            self.access += 1


# ---------------------------------------------------------------------------------------- RAM

class RamTier:
    """
    R slots. Per slot its key (-1 empty), whether it is a duplicate of a VRAM-held expert (DUP) or
    the owner (OWN), its layout (extent: as read from the SSD; compact: written by a demotion) and
    its recency stamp (the seq of its insertion or last decode read). OWN and DUP keys are kept in
    two lists with swap-remove, so a sample is O(sample); free slots are taken lowest first.
    """

    def __init__(self, slots: int, nkeys: int, sample: int, seed: int):
        self.R = slots
        self.key = [-1] * slots
        self.dup = [False] * slots
        self.compact = [False] * slots
        self.stamp = [0] * slots
        self.of = [-1] * nkeys
        self.free = list(range(slots))          # sorted
        self.lists = {False: [], True: []}      # OWN, DUP
        self.pos = [-1] * nkeys
        self.sample_n = sample
        self.seed = seed
        self.draws = 0

    def __len__(self):
        return self.R - len(self.free)

    def owns(self, k: int) -> bool:
        s = self.of[k]
        return s >= 0 and not self.dup[s]

    def is_dup(self, k: int) -> bool:
        s = self.of[k]
        return s >= 0 and self.dup[s]

    def _list_add(self, k: int, dup: bool):
        lst = self.lists[dup]
        self.pos[k] = len(lst)
        lst.append(k)

    def _list_del(self, k: int, dup: bool):
        lst = self.lists[dup]
        i = self.pos[k]
        last = lst[-1]
        lst[i] = last
        self.pos[last] = i
        lst.pop()
        self.pos[k] = -1

    def add(self, k: int, dup: bool, now: int, slot: int = -1, compact: bool = False) -> int:
        if self.of[k] >= 0:
            raise TierError(f"key {k} is already in RAM slot {self.of[k]}")
        if slot < 0:
            if not self.free:
                raise TierError("RAM tier full")
            slot = self.free.pop(0)
        else:
            i = bisect_left(self.free, slot)
            if i == len(self.free) or self.free[i] != slot:
                raise TierError(f"RAM slot {slot} is not free")
            self.free.pop(i)
        self.key[slot], self.dup[slot], self.compact[slot], self.stamp[slot] = k, dup, compact, now
        self.of[k] = slot
        self._list_add(k, dup)
        return slot

    def remove(self, k: int) -> int:
        slot = self.of[k]
        self._list_del(k, self.dup[slot])
        self.key[slot], self.dup[slot], self.compact[slot], self.stamp[slot] = -1, False, False, 0
        self.of[k] = -1
        insort(self.free, slot)
        return slot

    def set_dup(self, k: int, dup: bool):
        slot = self.of[k]
        if self.dup[slot] != dup:
            self._list_del(k, self.dup[slot])
            self.dup[slot] = dup
            self._list_add(k, dup)

    def rand(self, n: int) -> int:
        self.draws += 1
        return splitmix64(self.seed ^ ((DRAW_MUL * self.draws) & MASK64)) % n

    def sample(self, dup: bool) -> list:
        lst = self.lists[dup]
        n = len(lst)
        if n <= self.sample_n:
            return list(lst)
        return [lst[self.rand(n)] for _ in range(self.sample_n)]

    def min_stamp(self) -> int | None:
        st = [self.stamp[s] for s in range(self.R) if self.key[s] >= 0]
        return min(st) if st else None


# ---------------------------------------------------------------------------------------- core

class TierCore:
    """
    The whole policy of one GPU's cache layers over one RAM tier: VRAM directory, RAM tier, heat,
    the adaptive probability and the counters. call() = lookup() then host() (deterministic mode).
    """

    def __init__(self, cfg: PolicyConfig, layers: int, experts: int, slots: int, ram_slots: int,
                 pins: int = 0, defer_returns: bool = False):
        self.cfg = cfg
        self.L, self.E = layers, experts
        self.K = layers * experts
        self.heat = Heat(self.K, cfg.halflife)
        self.vram = VramDirectory(self.K, slots, cfg.spare, pins)
        self.ram = RamTier(ram_slots, self.K, cfg.sample, cfg.seed)
        self.adapt = AdaptiveP(self.vram.capacity, ram_slots, cfg.pmin, cfg.p)
        self.seq = 0
        self.next_adapt = cfg.adapt_every
        self.c = {k: 0 for k in COUNTERS}
        self.defer_returns = defer_returns
        self.refill_busy = 0                # refills of earlier calls still in flight (the runtime's)
        self.pending = []                   # (vram slot) of demotions not yet complete (defer_returns)
        self.log = []                       # (Record, actions) per call

    # ---- seeding (cold fill, tests)

    def key(self, lc: int, e: int) -> int:
        return lc * self.E + e

    def cold_fill(self, order: list) -> tuple:
        """Place keys in priority order: the pool's held slots first (lowest slots, stamp 0), then
        the RAM tier (inclusive: a duplicate of every pool key first). Returns (vram keys, ram keys)"""
        n = len(order)
        held = self.vram.S if self.vram.S >= n else self.vram.capacity
        vk = list(order[:held])
        for i, k in enumerate(vk):
            self.vram.place(k, i, 0)
        if self.cfg.policy == "inclusive":
            rk = list(order[:min(n, self.ram.R)])
            for k in rk:
                self.ram.add(k, self.vram.slot_of[k] >= 0, 0)
        else:
            rk = list(order[held:held + self.ram.R])
            for k in rk:
                self.ram.add(k, False, 0)
        return vk, rk

    def tick(self, tokens: int = 1):
        """A decode pass began: tokens advance the heat clock"""
        self.heat.tokens += tokens

    # ---- calls

    def call(self, lc: int, ids: list, mode: int = DECODE) -> tuple:
        rec = self.lookup(lc, ids, mode)
        acts = self.host(rec)
        return rec, acts

    def lookup(self, lc: int, ids: list, mode: int = DECODE) -> Record:
        if mode not in (DECODE, ROUTED):
            raise ValueError("lookup: decode or routed mode")
        self.seq += 1
        inc = Q16 if mode == DECODE else self.cfg.prefill_inc
        rec, starved = self.vram.lookup(lc, ids, mode, self.seq, self.E, self.heat, self.cfg, self.adapt.q32, inc)
        c = self.c
        c["calls"] += 1
        c["starved"] += starved
        for x in rec.entries:
            if x.kind == HIT:
                c["hits"] += 1
            elif x.kind == ADMIT:
                c["admits"] += 1
                if x.vkey >= 0:
                    c["retires"] += 1
                if x.inplace:
                    c["inplace"] += 1
            else:
                c["transients"] += 1
        return rec

    def layer_call(self, lc: int, ids: list) -> list:
        """A layer-mode prefill call: heat only (no stamps, no admission, no demotion); every
        cached expert of the layer that VRAM does not hold is staged (from RAM, or read from the
        SSD into the slab). Returns the staging actions, then any refills"""
        self.seq += 1
        self.c["calls"] += 1
        cnt = {}
        for e in ids:
            cnt[int(e)] = cnt.get(int(e), 0) + 1
        for e, n in cnt.items():
            self.heat.add(self.key(lc, e), n * self.cfg.prefill_inc)
        acts, bypassed, i = [], [], 0
        for e in range(self.E):
            k = self.key(lc, e)
            s = self.vram.slot_of[k]
            if s >= 0:
                continue
            r = self.ram.of[k]
            if r >= 0:
                self.c["prefill_h2d"] += 1
                acts.append(("stage_ram", k, r, i))
            else:
                if self.cfg.disk_off:
                    raise TierError(f"disk experts=off but key {k} is in neither VRAM nor RAM")
                self.c["prefill_ssd"] += 1
                acts.append(("stage_ssd", k, i))
                if e in cnt:
                    bypassed.append(k)
            i += 1
        self.refill(bypassed, acts)
        self.maybe_adapt()
        self.log.append((Record(self.seq, lc, LAYER), acts))
        return acts

    # ---- the host side of one record (3.2)

    def release(self, vslot: int, acts: list, demoted: bool):
        if demoted and self.defer_returns:
            self.pending.append(vslot)
            return
        self.vram.return_slot(vslot)
        acts.append(("return", vslot))

    def complete(self) -> list:
        """Production emulation: every demotion in flight has landed; their slots return"""
        acts = []
        for s in self.pending:
            self.vram.return_slot(s)
            acts.append(("return", s))
        self.pending = []
        return acts

    def metric(self, k: int) -> int:
        """ram evict=: lfu compares heat, lru the RAM recency stamp"""
        return self.heat.read(k) if self.cfg.ram_evict == "lfu" else self.ram.stamp[self.ram.of[k]]

    def coldest_own(self) -> int:
        c = self.ram.sample(False)
        return min(c, key = lambda k: (self.metric(k), k)) if c else -1

    def reclaimable_dup(self) -> int:
        """The duplicate whose VRAM copy was used last (the last to lose it), ties to the lowest key"""
        c = self.ram.sample(True)
        if not c:
            return -1

        def last_use(k):
            s = self.vram.slot_of[k]
            return self.vram.stamp[s] if s >= 0 else -1
        return max(c, key = lambda k: (last_use(k), -k))

    def evict_own(self, u: int, acts: list) -> int:
        slot = self.ram.remove(u)
        self.c["ram_evict"] += 1
        acts.append(("ram_evict", u, slot))
        return slot

    def reclaim(self, d: int, acts: list) -> int:
        slot = self.ram.remove(d)
        self.c["dup_reclaim"] += 1
        acts.append(("dup_reclaim", d, slot))
        return slot

    def room(self, e: int, gated: bool, acts: list) -> int:
        """A RAM slot for a new owner e: a free slot, else a reclaimed duplicate (never under
        inclusive), else the coldest sampled owner's (when gated, only if e is warmer). -1: none"""
        if self.ram.R <= 0:
            return -1
        if self.ram.free:
            return self.ram.free[0]
        if self.cfg.policy != "inclusive":
            d = self.reclaimable_dup()
            if d >= 0:
                return self.reclaim(d, acts)
        u = self.coldest_own()
        if u >= 0 and (not gated or self.heat.read(e) > self.heat.read(u)):
            return self.evict_own(u, acts)
        return -1

    def host(self, rec: Record) -> list:
        cfg, ram, now = self.cfg, self.ram, rec.seq
        acts, bypassed = [], []
        for x in rec.entries:
            if x.kind == HIT:
                continue
            e = x.key
            r = -1
            rescue = self.vram.retiring_slot_of(e)
            if rescue >= 0:
                self.c["rescued"] += 1
                acts.append(("d2d", e, rescue, x.kind, x.dst))
            if ram.of[e] >= 0:
                r = ram.of[e]
                if rescue < 0:
                    self.c["h2d_ram"] += 1
                    acts.append(("h2d", e, r, x.kind, x.dst))
                if rec.mode == DECODE:
                    ram.stamp[r] = now
                if x.kind == ADMIT:
                    self.c["admits_from_ram"] += 1
                    if cfg.policy == "exclusive":
                        ram.remove(e)
                        acts.append(("ram_free", e, r))
                    else:
                        ram.set_dup(e, True)
                        acts.append(("ram_dup", e, r))
            elif rescue < 0:
                if cfg.disk_off:
                    raise TierError(f"disk experts=off but key {e} is in neither VRAM nor RAM")
                self.c["ssd"] += 1
                placed = -1
                if x.kind == TRANSIENT:
                    if rec.mode == DECODE:
                        slot = self.room(e, cfg.ram_admit == "heat", acts)
                        if slot >= 0:
                            placed = ram.add(e, False, now, slot)
                        elif ram.R > 0:
                            self.c["ram_admit_refused"] += 1
                elif cfg.policy == "inclusive":
                    slot = ram.free[0] if ram.free else -1
                    if slot < 0:
                        u = self.coldest_own()
                        if u < 0:
                            raise TierError("policy=inclusive: the RAM tier is not larger than the VRAM pool")
                        slot = self.evict_own(u, acts)
                    placed = ram.add(e, True, now, slot)
                elif cfg.policy == "lazy-exclusive" and ram.R > 0:
                    if ram.free:
                        placed = ram.add(e, True, now)
                    else:
                        d = self.reclaimable_dup()
                        if d >= 0:
                            placed = ram.add(e, True, now, self.reclaim(d, acts))
                acts.append(("ssd", e, placed, x.kind, x.dst))
                if placed < 0 and x.kind == TRANSIENT and rec.mode != DECODE:
                    bypassed.append(e)
            if x.vkey >= 0:
                self.victim(x, e, r, acts, now)
        self.refill(bypassed, acts)
        self.maybe_adapt()
        self.log.append((rec, acts))
        return acts

    def victim(self, x: Entry, e: int, r: int, acts: list, now: int):
        cfg, ram = self.cfg, self.ram
        v, vs = x.vkey, x.vslot
        if ram.is_dup(v):
            ram.set_dup(v, False)                   # its RAM copy becomes the owner: nothing to copy
            self.c["drops"] += 1
            acts.append(("ram_own", v, ram.of[v]))
            if not x.inplace:
                self.release(vs, acts, False)
            return
        target = -1
        if cfg.disk_off:
            # no other copy: the victim always moves to RAM, into the slot the promotion left,
            # else a free slot, else a reclaimed duplicate (the sizing guarantees one)
            if ram.is_dup(e):
                target = ram.remove(e)
                acts.append(("ram_overwrite", e, target))
            elif ram.free:
                target = ram.free[0]
            else:
                d = self.reclaimable_dup()
                if d < 0:
                    raise TierError("disk experts=off: no RAM slot for a VRAM victim")
                target = self.reclaim(d, acts)
        elif not cfg.demotes or ram.R <= 0:
            pass
        elif cfg.demote == "swap":
            freed = r >= 0
            if freed:
                u = self.coldest_own()
                if u >= 0 and self.heat.read(v) <= self.heat.read(u):
                    freed = False
            if freed:
                if cfg.policy == "lazy-exclusive" and ram.is_dup(e):
                    target = ram.remove(e)
                    acts.append(("ram_overwrite", e, target))
                elif r in ram.free:
                    target = r
                elif ram.free:
                    target = ram.free[0]
        elif cfg.demote == "heat":
            if ram.free:
                target = ram.free[0]
            else:
                d = self.reclaimable_dup()
                if d >= 0:
                    target = self.reclaim(d, acts)
                else:
                    u = self.coldest_own()
                    if u >= 0 and self.heat.read(v) > self.heat.read(u):
                        target = self.evict_own(u, acts)
        else:  # all
            target = self.room(v, False, acts)
        if target < 0:
            self.c["drops"] += 1
            acts.append(("drop", v, vs))
            if not x.inplace:
                self.release(vs, acts, False)
            return
        ram.add(v, False, now, target, compact = True)
        self.c["d2h"] += 1
        acts.append(("d2h", v, vs, target))
        if not x.inplace:
            self.release(vs, acts, True)

    def refill(self, bypassed: list, acts: list):
        """Class-3 reads of warm experts that prefill read from the SSD past the RAM tier (3.3):
        into a free slot or a reclaimed duplicate, or over the coldest sampled owner when warmer; at
        most refill_inflight in flight (refill_busy: those of earlier calls still reading, which the
        runtime sets; 0 in deterministic mode), the others dropped (never queued)"""
        cfg, ram = self.cfg, self.ram
        if not cfg.refill or ram.R <= 0:
            return
        n = self.refill_busy
        for e in bypassed:
            if self.vram.slot_of[e] >= 0 or ram.of[e] >= 0:
                continue
            u = -1
            ok = bool(ram.free) or (cfg.policy != "inclusive" and bool(ram.lists[True]))
            if not ok:
                u = self.coldest_own()
                ok = u >= 0 and self.heat.read(e) > self.heat.read(u)
            if not ok:
                continue
            if n >= cfg.refill_inflight:
                self.c["refill_dropped"] += 1
                continue
            if ram.free:
                slot = ram.free[0]
            elif u < 0:
                slot = self.reclaim(self.reclaimable_dup(), acts)
            else:
                slot = self.evict_own(u, acts)
            ram.add(e, False, self.seq, slot)
            self.c["refills"] += 1
            acts.append(("refill", e, slot))
            n += 1

    def maybe_adapt(self):
        """Every adapt_every accesses: move p toward equal VRAM and RAM cache lives (3.5)"""
        if self.vram.access < self.next_adapt:
            return
        while self.next_adapt <= self.vram.access:
            self.next_adapt += self.cfg.adapt_every
        if self.adapt.pinned or self.cfg.admit != "adaptive":
            return
        vst = [self.vram.stamp[s] for s in range(self.vram.S) if self.vram.state[s] == VALID]
        rst = self.ram.min_stamp()
        if not vst or rst is None:
            return
        self.adapt.update(self.seq - min(vst), self.seq - rst)
        self.c["adapts"] += 1

    # ---- checks and views

    def check(self, deterministic: bool = True):
        """Every invariant of the tier; raises AssertionError naming the first that fails"""
        v, ram, cfg = self.vram, self.ram, self.cfg
        n = v.S + v.pins
        counts = [0, 0, 0, 0]
        for s in range(n):
            counts[v.state[s]] += 1
            if v.state[s] in (VALID, PINNED) and v.key[s] >= 0:
                assert v.slot_of[v.key[s]] == s, f"slot {s} holds {v.key[s]} but slot_of says {v.slot_of[v.key[s]]}"
            if v.state[s] == FREE:
                assert v.key[s] == -1, f"free slot {s} holds a key"
        assert sum(counts) == n
        assert counts[VALID] == v.occupied, f"occupied {v.occupied} != {counts[VALID]} VALID"
        assert v.free == sorted(s for s in range(v.S) if v.state[s] == FREE), "free list"
        for k in range(self.K):
            s = v.slot_of[k]
            if s >= 0:
                assert v.state[s] in (VALID, PINNED) and v.key[s] == k, f"key {k} -> slot {s}"
        if deterministic and not self.defer_returns:
            assert counts[RETIRING] == 0, f"{counts[RETIRING]} slots still RETIRING"
        assert len(ram.lists[False]) + len(ram.lists[True]) + len(ram.free) == ram.R, "RAM slots"
        for dup in (False, True):
            for i, k in enumerate(ram.lists[dup]):
                assert ram.pos[k] == i and ram.of[k] >= 0 and ram.dup[ram.of[k]] == dup and ram.key[ram.of[k]] == k
        for k in range(self.K):
            s = ram.of[k]
            if s >= 0:
                assert ram.key[s] == k
            if cfg.policy == "exclusive":
                assert not ram.is_dup(k), f"exclusive: {k} is a duplicate"
            if cfg.policy == "lazy-exclusive" and ram.is_dup(k):
                assert v.slot_of[k] >= 0, f"lazy-exclusive: duplicate {k} is not in VRAM"
            if cfg.policy == "inclusive" and v.slot_of[k] >= 0 and v.state[v.slot_of[k]] == VALID:
                assert s >= 0, f"inclusive: VRAM key {k} has no RAM copy"
        if cfg.demote == "swap" and not cfg.disk_off:
            assert self.c["d2h"] <= self.c["admits_from_ram"], "swap: more demotions than promotions from RAM"

    def disk_off_complete(self):
        """disk experts=off: every key is in VRAM or RAM (call after the cold fill)"""
        for k in range(self.K):
            assert self.vram.slot_of[k] >= 0 or self.ram.of[k] >= 0 or self.vram.retiring_slot_of(k) >= 0, \
                f"disk experts=off: key {k} is nowhere"

    def vram_view(self) -> dict:
        return {s: (STATE_NAMES[self.vram.state[s]], self.vram.key[s], self.vram.stamp[s])
                for s in range(self.vram.S + self.vram.pins)}

    def ram_view(self) -> dict:
        return {k: ("dup" if self.ram.dup[s] else "own", s) for k, s in enumerate(self.ram.of) if s >= 0}

    def counters(self) -> dict:
        return {k: v for k, v in self.c.items() if v}


def round_robin_order(layers: int, experts: int, pinned: set | None = None) -> list:
    """The cold-fill order without a heat file or profile: expert 0 of every layer, then expert 1..."""
    pinned = pinned or set()
    return [lc * experts + e for e in range(experts) for lc in range(layers) if lc * experts + e not in pinned]


# ---------------------------------------------------------------------------------------- replay

def replay(core: TierCore, script: list) -> list:
    """
    Run a script of operations on a core; returns [(record or None, actions)] per operation:
      ("tick", n)                          a decode pass of n tokens began
      ("call", lc, ids, mode)              lookup + host (mode DECODE or ROUTED)
      ("layer", lc, ids)                   a layer-mode prefill call
      ("complete",)                        demotions in flight landed (defer_returns)
    """
    out = []
    for op in script:
        if op[0] == "tick":
            core.tick(op[1])
            out.append((None, []))
        elif op[0] == "call":
            out.append(core.call(op[1], op[2], op[3] if len(op) > 3 else DECODE))
        elif op[0] == "layer":
            out.append((None, core.layer_call(op[1], op[2])))
        elif op[0] == "complete":
            out.append((None, core.complete()))
        else:
            raise ValueError(f"replay: unknown operation {op[0]!r}")
    return out
