"""
The policy simulator the expert tier's defaults were chosen on (disk-tier/sim_presets.py and
disk-tier/sim_demotion.py of the design work), reduced to what tests/test_expert_tier_policy_.py
compares the reference core against: the synthetic routing (Zipf popularity under a random
per-layer permutation, top-6 by Gumbel top-k, optional drift) and simulate_preset(), the per-access
model of policy= / demote= / admit= with one global LRU VRAM pool and a sampled-LFU RAM pool. The
functions are copied unchanged; `swapheat` is the heat-gated swap the tier ships as demote=swap.
Test support only (numpy, CPU).
"""
from __future__ import annotations

import random
from collections import OrderedDict

import numpy as np

E = 384                  # routed experts per layer
TOPK = 6
VRAM_SLOT = 13_271_040
RAM_SLOT = 13_320_192
GIB = 1 << 30
SAMPLE = 16
DECAY_EVERY = 256
POLICIES = ("inclusive", "exclusive", "lazy-exclusive")
DEMOTES = ("off", "swap", "swapheat", "heat", "all")
ADMITS = ("always", "heat", "adaptive")

# the 24 cases of the design's grid: (name, tiered layers, VRAM GiB, RAM GiB list); Zipf s; drift
GRID = [("C", 28, 81, [16, 32, 52]), ("D", 40, 75, [32, 64, 96])]
ZIPF = [0.6, 1.0]
DRIFTS = [(0, 0.0), (1500, 0.25)]
TOKENS, WARM, SEED = 9000, 3000, 1234


def gen_trace(layers, tokens, s, drift_every, drift_frac, seed):
    rng = np.random.default_rng(seed)
    base = -s * np.log(np.arange(1, E + 1, dtype = np.float64))
    logw = np.stack([base[rng.permutation(E)] for _ in range(layers)])
    out = np.empty((tokens, layers, TOPK), dtype = np.int32)
    block = 128
    t = 0
    while t < tokens:
        n = min(block, tokens - t)
        if drift_every:
            nxt = (t // drift_every + 1) * drift_every
            n = min(n, nxt - t)
        g = rng.gumbel(size = (n, layers, E)) + logw[None]
        out[t:t + n] = np.argpartition(-g, TOPK, axis = 2)[..., :TOPK]
        t += n
        if drift_every and t % drift_every == 0:
            m = int(round(drift_frac * E))
            for l in range(layers):
                idx = rng.choice(E, m, replace = False)
                logw[l, idx] = logw[l, rng.permutation(idx)]
    return out


class Pool:
    """RAM pool: key -> dup flag, with O(1) random sampling of uniques and of duplicates."""

    def __init__(self, cap):
        self.cap = cap
        self.dup = {}
        self.lists = {False: [], True: []}
        self.pos = {}

    def __len__(self):
        return len(self.dup)

    def __contains__(self, k):
        return k in self.dup

    def add(self, k, dup):
        self.dup[k] = dup
        lst = self.lists[dup]
        self.pos[k] = len(lst)
        lst.append(k)

    def remove(self, k):
        d = self.dup.pop(k)
        lst = self.lists[d]
        i = self.pos.pop(k)
        last = lst.pop()
        if last != k:
            lst[i] = last
            self.pos[last] = i

    def set_dup(self, k, dup):
        if self.dup[k] != dup:
            self.remove(k)
            self.add(k, dup)

    def sample(self, dup, rnd):
        lst = self.lists[dup]
        if not lst:
            return []
        return [lst[rnd.randrange(len(lst))] for _ in range(min(SAMPLE, len(lst)))]

    def sample_any(self, rnd):
        a, b = self.lists[False], self.lists[True]
        n = len(a) + len(b)
        out = []
        for _ in range(min(SAMPLE, n)):
            i = rnd.randrange(n)
            out.append(a[i] if i < len(a) else b[i - len(a)])
        return out


def simulate_preset(policy, demote, admit, trace, warm, vram_slots, ram_slots, seed = 0,
                    pmin = 0.05):
    assert policy in POLICIES and demote in DEMOTES and admit in ADMITS
    tokens, layers, _ = trace.shape
    rnd = random.Random(seed)
    freq = np.zeros(layers * E, dtype = np.float64)
    last = np.zeros(layers * E, dtype = np.int64)
    vram = OrderedDict()                  # LRU order, oldest first
    ram = Pool(ram_slots)                 # dup flag + O(1) sampling
    ram_rec = OrderedDict()               # RAM recency, for the adaptive cache-life estimate
    c = {"h2d": 0, "ssd": 0, "d2h": 0, "hits_vram": 0, "hits_ram": 0, "drops": 0, "backinv": 0,
         "transient": 0}
    pmax = vram_slots / max(1, vram_slots + ram_slots)
    p, clock, adapt_every = pmax, 0, 2048
    cap = ram_slots

    def r_add(k, dup):
        ram.add(k, dup)
        ram_rec[k] = True

    def r_del(k):
        ram.remove(k)
        del ram_rec[k]

    def coldest_unique():
        smp = ram.sample(False, rnd)
        return min(smp, key = lambda q: freq[q]) if smp else None

    def reclaim_dup():
        # the duplicate whose VRAM copy was used most recently: the last to lose its VRAM copy
        smp = ram.sample(True, rnd)
        return max(smp, key = lambda q: last[q]) if smp else None

    def room_for_unique():
        """make room for one unique entry: free slot, else a duplicate, else the coldest unique"""
        if cap <= 0:
            return False
        if len(ram) < cap:
            return True
        d = reclaim_dup()
        if d is not None:
            r_del(d)
            return True
        u = coldest_unique()
        if u is not None:
            r_del(u)
            return True
        return False

    def room_incl():
        u = coldest_unique()
        if u is not None:
            r_del(u)
            return
        d = reclaim_dup()
        r_del(d)
        vram.pop(d, None)
        c["backinv"] += 1

    for t in range(tokens):
        if t == warm:
            c = {k: 0 for k in c}
        if t % DECAY_EVERY == 0 and t:
            freq *= 0.5
        row = trace[t]
        for l in range(layers):
            base = l * E
            for e in row[l]:
                k = base + int(e)
                clock += 1
                freq[k] += 1.0
                last[k] = clock
                if admit == "adaptive" and clock % adapt_every == 0 and vram and ram_rec:
                    lv = clock - last[next(iter(vram))]
                    lr = clock - last[next(iter(ram_rec))]
                    ratio = lv / max(1, lv + lr)
                    p = min(pmax, max(pmin, p * (1.0 + (ratio - 0.5))))
                if k in vram:
                    vram.move_to_end(k)
                    c["hits_vram"] += 1
                    continue
                c["h2d"] += 1
                in_ram = k in ram
                c["hits_ram" if in_ram else "ssd"] += 1
                if in_ram:
                    ram_rec.move_to_end(k)

                # ---- VRAM admission
                if len(vram) < vram_slots or admit == "always":
                    enter = vram_slots > 0
                elif admit == "adaptive":
                    enter = rnd.random() < p
                else:  # heat
                    enter = freq[k] >= freq[next(iter(vram))]
                if not enter:
                    c["transient"] += 1
                    if not in_ram and room_for_unique():
                        r_add(k, False)
                    continue

                # ---- RAM side of a promotion
                freed = False
                if policy == "inclusive":
                    if in_ram:
                        ram.set_dup(k, True)
                    elif cap > 0:
                        if len(ram) >= cap:
                            room_incl()
                        r_add(k, True)
                elif policy == "exclusive":
                    if in_ram:
                        r_del(k)
                        freed = True
                else:  # lazy-exclusive
                    if in_ram:
                        ram.set_dup(k, True)
                        freed = True
                    elif cap > 0:
                        if len(ram) < cap:
                            r_add(k, True)
                        else:
                            d = reclaim_dup()
                            if d is not None:
                                r_del(d)
                                r_add(k, True)

                # ---- VRAM victim
                if len(vram) >= vram_slots:
                    v, _ = vram.popitem(last = False)
                    if v in ram:
                        ram.set_dup(v, False)          # its RAM copy becomes the owner: free
                    elif policy == "inclusive" or demote == "off" or cap <= 0:
                        c["drops"] += 1
                    elif demote in ("swap", "swapheat"):
                        if freed and demote == "swapheat":
                            u = coldest_unique()
                            if u is not None and freq[v] <= freq[u]:
                                freed = False          # colder than anything RAM keeps: drop
                        if freed:
                            if policy == "lazy-exclusive" and k in ram:
                                r_del(k)               # overwrite k's reclaimable duplicate
                            if len(ram) < cap:
                                r_add(v, False)
                                c["d2h"] += 1
                            else:
                                c["drops"] += 1
                        else:
                            c["drops"] += 1
                    elif demote == "heat":
                        if len(ram) < cap:
                            r_add(v, False)
                            c["d2h"] += 1
                        else:
                            d = reclaim_dup()
                            if d is not None:
                                r_del(d)
                                r_add(v, False)
                                c["d2h"] += 1
                            else:
                                u = coldest_unique()
                                if u is not None and freq[v] > freq[u]:
                                    r_del(u)
                                    r_add(v, False)
                                    c["d2h"] += 1
                                else:
                                    c["drops"] += 1
                    else:  # all
                        if room_for_unique():
                            r_add(v, False)
                            c["d2h"] += 1
                        else:
                            c["drops"] += 1
                vram[k] = True
    n = tokens - warm
    out = {k: v / n for k, v in c.items()}
    out["p_final"] = p
    return out
