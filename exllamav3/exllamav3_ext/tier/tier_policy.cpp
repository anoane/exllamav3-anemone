// Expert tier policy core; see tier_policy.h. Every function mirrors its namesake in
// exllamav3/model/expert_tier_policy.py line for line: the two must make the same decisions, draw
// the same samples in the same order and break ties the same way.

#include "tier_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace exl3_tier
{

namespace
{

constexpr uint64_t kU32 = 0xFFFFFFFFull;

[[noreturn]] void fail(const std::string& m) { throw TierError("expert tier: " + m); }

void expect(bool ok, const std::string& m)
{
    if (!ok) fail("invariant: " + m);
}

}  // namespace

uint64_t splitmix64(uint64_t x)
{
    uint64_t z = x + kGolden;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint64_t p_q32(double p)
{
    double x = p * 4294967296.0;
    if (!(x > 0.0)) return 0;
    if (x >= 4294967296.0) return 1ull << 32;
    return (uint64_t) x;
}

void PolicyConfig::validate() const
{
    if (policy < 0 || policy > 2 || demote < 0 || demote > 3 || admit < 0 || admit > 2 || evict < 0 || evict > 1 ||
        ram_evict < 0 || ram_evict > 1 || ram_admit < 0 || ram_admit > 1)
        fail("a policy setting is out of range");
    if (spare < 0 || sample < 1 || halflife < 1 || adapt_every < 1 || refill_inflight < 1)
        fail("spare >= 0, sample, halflife, adapt_every and refill_inflight >= 1");
    if (p_pinned && !(p >= 0.0 && p <= 1.0)) fail("the pinned admission probability must be in [0, 1]");
}

const char* action_name(int op)
{
    static const char* n[kActCount] = { "d2d", "h2d", "ram_free", "ram_dup", "ssd", "ram_evict", "dup_reclaim",
                                        "ram_own", "ram_overwrite", "d2h", "drop", "return", "refill", "stage_ram",
                                        "stage_ssd" };
    return op >= 0 && op < kActCount ? n[op] : "?";
}

int action_arity(int op)
{
    switch (op)
    {
        case kActD2D: case kActH2D: case kActSsd: return 4;
        case kActD2H: case kActStageRam: return 3;
        case kActReturn: return 1;
        default: return 2;
    }
}

const char* counter_name(int c)
{
    static const char* n[kNumCounters] = { "calls", "hits", "admits", "admits_from_ram", "transients", "starved",
                                           "retires", "inplace", "drops", "d2h", "h2d_ram", "ssd", "rescued",
                                           "ram_evict", "dup_reclaim", "ram_admit_refused", "refills",
                                           "refill_dropped", "prefill_h2d", "prefill_ssd", "adapts" };
    return c >= 0 && c < kNumCounters ? n[c] : "?";
}

// ---------------------------------------------------------------------------------------- heat

Heat::Heat(int64_t nkeys, uint32_t halflife) : v((size_t) nkeys, 0), ep((size_t) nkeys, 0), halflife_(halflife) {}

uint32_t Heat::read(int64_t k) const
{
    int64_t d = (int64_t) epoch() - (int64_t) ep[(size_t) k];
    uint32_t x = v[(size_t) k];
    return d > 0 ? x >> std::min<int64_t>(31, d) : x;
}

void Heat::add(int64_t k, uint64_t inc)
{
    uint64_t x = (uint64_t) read(k) + inc;
    v[(size_t) k] = (uint32_t) std::min(kU32, x);
    ep[(size_t) k] = epoch();
}

void Heat::set(int64_t k, uint64_t value)
{
    v[(size_t) k] = (uint32_t) std::min(kU32, value);
    ep[(size_t) k] = epoch();
}

AdaptiveP::AdaptiveP(int64_t capacity, int64_t ram_slots, double pmin_, bool pinned_, double p_)
    : pmax(capacity + ram_slots > 0 ? (double) capacity / (double) (capacity + ram_slots) : 1.0),
      pmin(pmin_), p(pinned_ ? p_ : 0.0), pinned(pinned_)
{
    if (!pinned) p = pmax;
}

double AdaptiveP::update(int64_t lv, int64_t lr)
{
    if (!pinned)
    {
        double ratio = (double) lv / (double) std::max<int64_t>(1, lv + lr);
        p = std::min(pmax, std::max(pmin, p * (1.0 + (ratio - 0.5))));
    }
    return p;
}

// ---------------------------------------------------------------------------------------- VRAM

VramDirectory::VramDirectory(int64_t nkeys, int slots, int spare_, int pins_)
    : S(slots), spare(spare_), pins(pins_), state((size_t) (slots + pins_), kFree), key((size_t) (slots + pins_), -1),
      stamp((size_t) (slots + pins_), 0), slot_of((size_t) nkeys, -1), retiring_of((size_t) nkeys, -1)
{
    for (int s = 0; s < slots; ++s) free.insert(free.end(), s);
    for (int s = slots; s < slots + pins_; ++s) state[(size_t) s] = kPinned;
}

void VramDirectory::set_stamp(int32_t slot, uint32_t st)
{
    if (state[(size_t) slot] == kValid) lru_.erase({ stamp[(size_t) slot], slot });
    stamp[(size_t) slot] = st;
    if (state[(size_t) slot] == kValid) lru_.insert({ st, slot });
}

void VramDirectory::place(int32_t k, int32_t slot, uint32_t st)
{
    if (slot < 0 || slot >= S + pins) fail("place: slot " + std::to_string(slot) + " out of range");
    if (slot >= S)
    {
        if (state[(size_t) slot] != kPinned || key[(size_t) slot] >= 0) fail("pin slot " + std::to_string(slot) + " is taken");
    }
    else
    {
        if (state[(size_t) slot] != kFree) fail("slot " + std::to_string(slot) + " is not free");
        free.erase(slot);
        state[(size_t) slot] = kValid;
        lru_.insert({ stamp[(size_t) slot], slot });
        ++occupied;
    }
    if (slot_of[(size_t) k] >= 0) fail("key " + std::to_string(k) + " is already in slot " + std::to_string(slot_of[(size_t) k]));
    key[(size_t) slot] = k;
    set_stamp(slot, st);
    slot_of[(size_t) k] = slot;
}

int32_t VramDirectory::victim(uint32_t now, const Heat* lfu) const
{
    if (!lfu)
    {
        if (lru_.empty()) return -1;
        const auto& b = *lru_.begin();
        return b.first < now ? b.second : -1;
    }
    int32_t best = -1;
    uint64_t bh = 0;
    uint32_t bs = 0;
    for (int32_t s = 0; s < S; ++s)
    {
        if (state[(size_t) s] != kValid || stamp[(size_t) s] >= now) continue;
        uint64_t h = lfu->read(key[(size_t) s]);
        uint32_t st = stamp[(size_t) s];
        if (best < 0 || h < bh || (h == bh && st < bs))       // then the lowest slot (scan order)
        {
            best = s;
            bh = h;
            bs = st;
        }
    }
    return best;
}

void VramDirectory::retire(int32_t slot)
{
    int32_t k = key[(size_t) slot];
    lru_.erase({ stamp[(size_t) slot], slot });
    state[(size_t) slot] = kRetiring;
    slot_of[(size_t) k] = -1;
    --occupied;
    retiring_of[(size_t) k] = slot;
}

void VramDirectory::return_slot(int32_t slot)
{
    if (slot < 0 || slot >= S || state[(size_t) slot] != kRetiring)
        fail("slot " + std::to_string(slot) + " returned while not retiring");
    int32_t k = key[(size_t) slot];
    if (k >= 0 && retiring_of[(size_t) k] == slot) retiring_of[(size_t) k] = -1;
    state[(size_t) slot] = kFree;
    key[(size_t) slot] = -1;
    stamp[(size_t) slot] = 0;
    free.insert(slot);
}

int32_t VramDirectory::retiring_slot_of(int32_t k) const
{
    return retiring_of[(size_t) k];
}

Record VramDirectory::lookup(int32_t lc, const int32_t* ids, int64_t n, int mode, uint32_t seq, int64_t E, Heat& heat,
                             const PolicyConfig& cfg, uint64_t pq, uint32_t inc, int64_t* starved)
{
    const uint32_t now = seq;
    std::vector<int32_t> uniq;
    std::vector<uint32_t> cnt;
    uniq.reserve((size_t) std::min<int64_t>(n, 64));
    for (int64_t i = 0; i < n; ++i)
    {
        int32_t e = ids[i];
        if (e < 0 || e >= E) fail("lookup: expert " + std::to_string(e) + " out of range");
        size_t j = 0;
        while (j < uniq.size() && uniq[j] != e) ++j;
        if (j == uniq.size())
        {
            uniq.push_back(e);
            cnt.push_back(0);
        }
        ++cnt[j];
    }
    std::vector<int32_t> keys(uniq.size());
    for (size_t j = 0; j < uniq.size(); ++j)
    {
        keys[j] = (int32_t) (lc * E + uniq[j]);
        heat.add(keys[j], (uint64_t) cnt[j] * inc);
    }
    if (mode == kDecode)
        for (int32_t k : keys)
        {
            int32_t s = slot_of[(size_t) k];
            if (s >= 0) set_stamp(s, now);
        }
    Record rec;
    rec.seq = seq;
    rec.lc = lc;
    rec.mode = mode;
    rec.entries.reserve(keys.size());
    const Heat* lfu = cfg.evict == kLfu ? &heat : nullptr;
    int32_t staging = 0;
    for (size_t j = 0; j < keys.size(); ++j)
    {
        const int32_t k = keys[j];
        ++access;
        Entry x;
        x.key = k;
        x.cnt = cnt[j];
        int32_t s = slot_of[(size_t) k];
        if (s >= 0)
        {
            x.kind = kHit;
            x.dst = s;
            x.heat = heat.read(k);
            rec.entries.push_back(x);
            continue;
        }
        bool adm = false;
        if (mode == kDecode)
        {
            if (occupied < capacity()) adm = true;
            else if (cfg.admit == kAdmitAlways) adm = true;
            else if (cfg.admit == kAdmitHeat)
            {
                int32_t v = victim(now, lfu);
                adm = v >= 0 && heat.read(k) >= heat.read(key[(size_t) v]);
            }
            else adm = (splitmix64(cfg.seed ^ (access * kGolden)) >> 32) < pq;
        }
        bool placed = false;
        if (adm)
        {
            if (!free.empty())
            {
                int32_t slot = *free.begin();
                free.erase(free.begin());
                state[(size_t) slot] = kValid;
                key[(size_t) slot] = k;
                lru_.insert({ stamp[(size_t) slot], slot });
                set_stamp(slot, now);
                slot_of[(size_t) k] = slot;
                ++occupied;
                x.kind = kAdmitted;
                x.dst = slot;
                x.heat = heat.read(k);
                if (occupied > capacity())
                {
                    int32_t vs = victim(now, lfu);
                    if (vs >= 0)
                    {
                        int32_t vk = key[(size_t) vs];
                        x.vkey = vk;
                        x.vslot = vs;
                        x.vstamp = stamp[(size_t) vs];
                        x.vheat = heat.read(vk);
                        retire(vs);
                    }
                }
                placed = true;
            }
            else if (spare == 0)
            {
                int32_t vs = victim(now, lfu);
                if (vs >= 0)
                {
                    int32_t vk = key[(size_t) vs];
                    x.kind = kAdmitted;
                    x.dst = vs;
                    x.heat = heat.read(k);
                    x.vkey = vk;
                    x.vslot = vs;
                    x.vstamp = stamp[(size_t) vs];
                    x.vheat = heat.read(vk);
                    x.inplace = 1;
                    slot_of[(size_t) vk] = -1;
                    key[(size_t) vs] = k;
                    set_stamp(vs, now);
                    slot_of[(size_t) k] = vs;
                    placed = true;
                }
            }
            if (!placed)
            {
                ++*starved;
                x.starved = 1;
            }
        }
        if (!placed)
        {
            x.kind = kTransient;
            x.dst = staging++;
            x.heat = heat.read(k);
        }
        rec.entries.push_back(x);
    }
    return rec;
}

void VramDirectory::apply(const Record& rec)
{
    if (rec.mode == kLayer) return;             // layer mode never changes the directory
    for (const Entry& x : rec.entries)
    {
        if (x.kind == kHit)
        {
            if (rec.mode == kDecode) set_stamp(x.dst, rec.seq);
        }
        else if (x.kind == kAdmitted)
        {
            if (x.inplace)
            {
                slot_of[(size_t) x.vkey] = -1;
                key[(size_t) x.dst] = x.key;
                set_stamp(x.dst, rec.seq);
                slot_of[(size_t) x.key] = x.dst;
                ++access;
                continue;
            }
            if (!free.erase(x.dst)) fail("apply: slot " + std::to_string(x.dst) + " is not free");
            state[(size_t) x.dst] = kValid;
            key[(size_t) x.dst] = x.key;
            lru_.insert({ stamp[(size_t) x.dst], x.dst });
            set_stamp(x.dst, rec.seq);
            slot_of[(size_t) x.key] = x.dst;
            ++occupied;
            if (x.vkey >= 0) retire(x.vslot);
        }
        ++access;
    }
}

// ---------------------------------------------------------------------------------------- RAM

RamTier::RamTier(int64_t slots, int64_t nkeys, int sample_, uint64_t seed_)
    : R(slots), key((size_t) slots, -1), dup((size_t) slots, 0), compact((size_t) slots, 0), stamp((size_t) slots, 0),
      of((size_t) nkeys, -1), pos((size_t) nkeys, -1), sample_n(sample_), seed(seed_)
{
    for (int64_t s = 0; s < slots; ++s) free.insert(free.end(), (int32_t) s);
}

void RamTier::list_add(int32_t k, bool d)
{
    auto& lst = lists[d ? 1 : 0];
    pos[(size_t) k] = (int32_t) lst.size();
    lst.push_back(k);
}

void RamTier::list_del(int32_t k, bool d)
{
    auto& lst = lists[d ? 1 : 0];
    int32_t i = pos[(size_t) k];
    int32_t last = lst.back();
    lst[(size_t) i] = last;
    pos[(size_t) last] = i;
    lst.pop_back();
    pos[(size_t) k] = -1;
}

int32_t RamTier::add(int32_t k, bool d, uint32_t now, int32_t slot, bool comp)
{
    if (of[(size_t) k] >= 0) fail("key " + std::to_string(k) + " is already in RAM slot " + std::to_string(of[(size_t) k]));
    if (slot < 0)
    {
        if (free.empty()) fail("RAM tier full");
        slot = *free.begin();
        free.erase(free.begin());
    }
    else if (!free.erase(slot)) fail("RAM slot " + std::to_string(slot) + " is not free");
    key[(size_t) slot] = k;
    dup[(size_t) slot] = d;
    compact[(size_t) slot] = comp;
    stamp[(size_t) slot] = now;
    of[(size_t) k] = slot;
    list_add(k, d);
    return slot;
}

int32_t RamTier::remove(int32_t k)
{
    int32_t slot = of[(size_t) k];
    list_del(k, dup[(size_t) slot]);
    key[(size_t) slot] = -1;
    dup[(size_t) slot] = 0;
    compact[(size_t) slot] = 0;
    stamp[(size_t) slot] = 0;
    of[(size_t) k] = -1;
    free.insert(slot);
    return slot;
}

void RamTier::set_dup(int32_t k, bool d)
{
    int32_t slot = of[(size_t) k];
    if ((bool) dup[(size_t) slot] != d)
    {
        list_del(k, dup[(size_t) slot]);
        dup[(size_t) slot] = d;
        list_add(k, d);
    }
}

uint64_t RamTier::rand(uint64_t n)
{
    ++draws;
    return splitmix64(seed ^ (kDrawMul * draws)) % n;
}

void RamTier::sample(bool d, std::vector<int32_t>& out)
{
    const auto& lst = lists[d ? 1 : 0];
    out.clear();
    size_t n = lst.size();
    if (n <= (size_t) sample_n)
    {
        out.assign(lst.begin(), lst.end());
        return;
    }
    for (int i = 0; i < sample_n; ++i) out.push_back(lst[(size_t) rand(n)]);
}

bool RamTier::min_stamp(uint32_t* out) const
{
    bool any = false;
    uint32_t m = 0;
    for (int64_t s = 0; s < R; ++s)
        if (key[(size_t) s] >= 0 && (!any || stamp[(size_t) s] < m))
        {
            m = stamp[(size_t) s];
            any = true;
        }
    if (any) *out = m;
    return any;
}

// ---------------------------------------------------------------------------------------- core

TierCore::TierCore(const PolicyConfig& cfg_, int64_t layers, int64_t experts, int slots, int64_t ram_slots, int pins,
                   bool defer)
    : cfg(cfg_), L(layers), E(experts), K(layers * experts), heat(layers * experts, cfg_.halflife),
      vram(layers * experts, slots, cfg_.spare, pins), ram(ram_slots, layers * experts, cfg_.sample, cfg_.seed),
      adapt(slots - cfg_.spare, ram_slots, cfg_.pmin, cfg_.p_pinned, cfg_.p), next_adapt((uint64_t) cfg_.adapt_every),
      defer_returns(defer)
{
    cfg.validate();
    if (K > (int64_t) std::numeric_limits<int32_t>::max()) fail("too many keys");
}

void TierCore::cold_fill(const std::vector<int32_t>& order, std::vector<int32_t>* vk, std::vector<int32_t>* rk)
{
    size_t n = order.size();
    size_t held = (size_t) vram.S >= n ? (size_t) vram.S : (size_t) std::max(0, vram.capacity());
    held = std::min(held, n);
    for (size_t i = 0; i < held; ++i)
    {
        vram.place(order[i], (int32_t) i, 0);
        if (vk) vk->push_back(order[i]);
    }
    if (cfg.policy == kInclusive)
    {
        size_t m = std::min(n, (size_t) ram.R);
        for (size_t i = 0; i < m; ++i)
        {
            ram.add(order[i], vram.slot_of[(size_t) order[i]] >= 0, 0, -1, false);
            if (rk) rk->push_back(order[i]);
        }
    }
    else
    {
        size_t end = std::min(n, held + (size_t) ram.R);
        for (size_t i = held; i < end; ++i)
        {
            ram.add(order[i], false, 0, -1, false);
            if (rk) rk->push_back(order[i]);
        }
    }
}

Record TierCore::lookup(int32_t lc, const int32_t* ids, int64_t n, int mode)
{
    if (mode != kDecode && mode != kRouted) fail("lookup: decode or routed mode");
    if (lc < 0 || lc >= L) fail("lookup: layer " + std::to_string(lc) + " out of range");
    ++seq;
    uint32_t inc = mode == kDecode ? kQ16 : cfg.prefill_inc;
    int64_t starved = 0;
    Record rec = vram.lookup(lc, ids, n, mode, seq, E, heat, cfg, adapt.q32(), inc, &starved);
    ++c[kCalls];
    c[kStarved] += starved;
    for (const Entry& x : rec.entries)
    {
        if (x.kind == kHit) ++c[kHits];
        else if (x.kind == kAdmitted)
        {
            ++c[kAdmits];
            if (x.vkey >= 0) ++c[kRetires];
            if (x.inplace) ++c[kInplace];
        }
        else ++c[kTransients];
    }
    return rec;
}

void TierCore::process(const Record& rec, std::vector<Action>& acts)
{
    if (rec.lc < 0 || rec.lc >= L) fail("record of layer " + std::to_string(rec.lc) + " out of range");
    if (rec.mode == kLayer)
    {
        layer_record(rec, acts);
        return;
    }
    seq = rec.seq;
    const uint32_t inc = rec.mode == kDecode ? kQ16 : cfg.prefill_inc;
    ++c[kCalls];
    for (const Entry& x : rec.entries)
    {
        heat.add(x.key, (uint64_t) x.cnt * inc);
        if (x.kind == kHit) ++c[kHits];
        else if (x.kind == kAdmitted)
        {
            ++c[kAdmits];
            if (x.vkey >= 0) ++c[kRetires];
            if (x.inplace) ++c[kInplace];
        }
        else
        {
            ++c[kTransients];
            if (x.starved) ++c[kStarved];
        }
    }
    vram.apply(rec);
    host(rec, acts);
}

void TierCore::layer_plan(int32_t lc, std::vector<Action>& acts)
{
    if (lc < 0 || lc >= L) fail("layer call: layer " + std::to_string(lc) + " out of range");
    int32_t i = 0;
    for (int32_t e = 0; e < (int32_t) E; ++e)
    {
        int32_t k = (int32_t) (lc * E + e);
        if (vram.slot_of[(size_t) k] >= 0) continue;
        int32_t r = ram.of[(size_t) k];
        Action a;
        if (r >= 0)
        {
            ++c[kPrefillH2D];
            a.op = kActStageRam;
            a.a = k;
            a.b = r;
            a.c = i;
        }
        else
        {
            if (cfg.disk_off) fail("disk experts=off but key " + std::to_string(k) + " is in neither VRAM nor RAM");
            ++c[kPrefillSsd];
            a.op = kActStageSsd;
            a.a = k;
            a.b = i;
        }
        acts.push_back(a);
        ++i;
    }
}

void TierCore::layer_record(const Record& rec, std::vector<Action>& acts)
{
    if (rec.lc < 0 || rec.lc >= L) fail("layer call: layer " + std::to_string(rec.lc) + " out of range");
    seq = rec.seq;
    ++c[kCalls];
    std::vector<int32_t> bypassed;
    for (const Entry& x : rec.entries)
    {
        if (x.key < rec.lc * E || x.key >= (rec.lc + 1) * E)
            fail("layer call: key " + std::to_string(x.key) + " outside layer " + std::to_string(rec.lc));
        heat.add(x.key, (uint64_t) x.cnt * cfg.prefill_inc);
        // used, and staged from the disk (neither VRAM nor RAM holds it now): a refill candidate
        if (x.cnt && vram.slot_of[(size_t) x.key] < 0 && ram.of[(size_t) x.key] < 0) bypassed.push_back(x.key);
    }
    std::sort(bypassed.begin(), bypassed.end());
    refill(bypassed, acts);
    maybe_adapt();
}

void TierCore::layer_call(int32_t lc, const int32_t* ids, int64_t n, std::vector<Action>& acts)
{
    if (lc < 0 || lc >= L) fail("layer call: layer " + std::to_string(lc) + " out of range");
    ++seq;
    Record rec;
    rec.seq = seq;
    rec.lc = lc;
    rec.mode = kLayer;
    std::vector<int32_t> slot((size_t) E, -1);
    for (int64_t i = 0; i < n; ++i)
    {
        int32_t e = ids[i];
        if (e < 0 || e >= E) fail("layer call: expert " + std::to_string(e) + " out of range");
        if (slot[(size_t) e] < 0)
        {
            slot[(size_t) e] = (int32_t) rec.entries.size();
            Entry x;
            x.key = (int32_t) (lc * E + e);
            rec.entries.push_back(x);
        }
        ++rec.entries[(size_t) slot[(size_t) e]].cnt;
    }
    layer_plan(lc, acts);
    layer_record(rec, acts);
}

void TierCore::release(int32_t vslot, std::vector<Action>& acts, bool demoted, int entry)
{
    if (demoted && defer_returns)
    {
        pending.push_back(vslot);
        return;
    }
    vram.return_slot(vslot);
    Action a;
    a.op = kActReturn;
    a.a = vslot;
    a.entry = entry;
    acts.push_back(a);
}

void TierCore::complete(std::vector<Action>& acts)
{
    for (int32_t s : pending)
    {
        vram.return_slot(s);
        Action a;
        a.op = kActReturn;
        a.a = s;
        acts.push_back(a);
    }
    pending.clear();
}

uint64_t TierCore::metric(int32_t k) const
{
    return cfg.ram_evict == kLfu ? heat.read(k) : ram.stamp[(size_t) ram.of[(size_t) k]];
}

int32_t TierCore::coldest_own()
{
    ram.sample(false, scratch_);
    int32_t best = -1;
    uint64_t bm = 0;
    for (int32_t k : scratch_)
    {
        uint64_t m = metric(k);
        if (best < 0 || m < bm || (m == bm && k < best))
        {
            best = k;
            bm = m;
        }
    }
    return best;
}

int32_t TierCore::reclaimable_dup()
{
    ram.sample(true, scratch_);
    int32_t best = -1;
    int64_t bu = 0;
    for (int32_t k : scratch_)
    {
        int32_t s = vram.slot_of[(size_t) k];
        int64_t u = s >= 0 ? (int64_t) vram.stamp[(size_t) s] : -1;
        if (best < 0 || u > bu || (u == bu && k < best))
        {
            best = k;
            bu = u;
        }
    }
    return best;
}

static Action act(int op, int32_t a, int32_t b, int32_t c, int32_t d, int entry)
{
    Action x;
    x.op = op;
    x.a = a;
    x.b = b;
    x.c = c;
    x.d = d;
    x.entry = entry;
    return x;
}

int32_t TierCore::evict_own(int32_t u, std::vector<Action>& acts, int entry)
{
    int32_t slot = ram.remove(u);
    ++c[kRamEvict];
    acts.push_back(act(kActRamEvict, u, slot, -1, -1, entry));
    return slot;
}

int32_t TierCore::reclaim(int32_t d, std::vector<Action>& acts, int entry)
{
    int32_t slot = ram.remove(d);
    ++c[kDupReclaim];
    acts.push_back(act(kActDupReclaim, d, slot, -1, -1, entry));
    return slot;
}

int32_t TierCore::room(int32_t e, bool gated, std::vector<Action>& acts, int entry)
{
    if (ram.R <= 0) return -1;
    if (!ram.free.empty()) return *ram.free.begin();
    if (cfg.policy != kInclusive)
    {
        int32_t d = reclaimable_dup();
        if (d >= 0) return reclaim(d, acts, entry);
    }
    int32_t u = coldest_own();
    if (u >= 0 && (!gated || heat.read(e) > heat.read(u))) return evict_own(u, acts, entry);
    return -1;
}

void TierCore::host(const Record& rec, std::vector<Action>& acts)
{
    const uint32_t now = rec.seq;
    std::vector<int32_t> bypassed;
    for (size_t i = 0; i < rec.entries.size(); ++i)
    {
        const Entry& x = rec.entries[i];
        const int en = (int) i;
        if (x.kind == kHit) continue;
        const int32_t e = x.key;
        int32_t r = -1;
        int32_t rescue = vram.retiring_slot_of(e);
        if (rescue >= 0)
        {
            ++c[kRescued];
            acts.push_back(act(kActD2D, e, rescue, x.kind, x.dst, en));
        }
        if (ram.of[(size_t) e] >= 0)
        {
            r = ram.of[(size_t) e];
            if (rescue < 0)
            {
                ++c[kH2DRam];
                acts.push_back(act(kActH2D, e, r, x.kind, x.dst, en));
            }
            if (rec.mode == kDecode) ram.stamp[(size_t) r] = now;
            if (x.kind == kAdmitted)
            {
                ++c[kAdmitsFromRam];
                if (cfg.policy == kExclusive)
                {
                    ram.remove(e);
                    acts.push_back(act(kActRamFree, e, r, -1, -1, en));
                }
                else
                {
                    ram.set_dup(e, true);
                    acts.push_back(act(kActRamDup, e, r, -1, -1, en));
                }
            }
        }
        else if (rescue < 0)
        {
            if (cfg.disk_off) fail("disk experts=off but key " + std::to_string(e) + " is in neither VRAM nor RAM");
            ++c[kSsd];
            int32_t placed = -1;
            if (x.kind == kTransient)
            {
                if (rec.mode == kDecode)
                {
                    int32_t slot = room(e, cfg.ram_admit == kRamAdmitHeat, acts, en);
                    if (slot >= 0) placed = ram.add(e, false, now, slot, false);
                    else if (ram.R > 0) ++c[kRamAdmitRefused];
                }
            }
            else if (cfg.policy == kInclusive)
            {
                int32_t slot = ram.free.empty() ? -1 : *ram.free.begin();
                if (slot < 0)
                {
                    int32_t u = coldest_own();
                    if (u < 0) fail("policy=inclusive: the RAM tier is not larger than the VRAM pool");
                    slot = evict_own(u, acts, en);
                }
                placed = ram.add(e, true, now, slot, false);
            }
            else if (cfg.policy == kLazyExclusive && ram.R > 0)
            {
                if (!ram.free.empty()) placed = ram.add(e, true, now, -1, false);
                else
                {
                    int32_t d = reclaimable_dup();
                    if (d >= 0) placed = ram.add(e, true, now, reclaim(d, acts, en), false);
                }
            }
            acts.push_back(act(kActSsd, e, placed, x.kind, x.dst, en));
            if (placed < 0 && x.kind == kTransient && rec.mode != kDecode) bypassed.push_back(e);
        }
        if (x.vkey >= 0) victim(x, e, r, acts, now, en);
    }
    refill(bypassed, acts);
    maybe_adapt();
}

void TierCore::victim(const Entry& x, int32_t e, int32_t r, std::vector<Action>& acts, uint32_t now, int en)
{
    const int32_t v = x.vkey, vs = x.vslot;
    if (ram.is_dup(v))
    {
        ram.set_dup(v, false);
        ++c[kDrops];
        acts.push_back(act(kActRamOwn, v, ram.of[(size_t) v], -1, -1, en));
        if (!x.inplace) release(vs, acts, false, en);
        return;
    }
    int32_t target = -1;
    if (cfg.disk_off)
    {
        if (ram.is_dup(e))
        {
            target = ram.remove(e);
            acts.push_back(act(kActRamOverwrite, e, target, -1, -1, en));
        }
        else if (!ram.free.empty()) target = *ram.free.begin();
        else
        {
            int32_t d = reclaimable_dup();
            if (d < 0) fail("disk experts=off: no RAM slot for a VRAM victim");
            target = reclaim(d, acts, en);
        }
    }
    else if (!cfg.demotes() || ram.R <= 0)
    {
    }
    else if (cfg.demote == kDemoteSwap)
    {
        bool freed = r >= 0;
        if (freed)
        {
            int32_t u = coldest_own();
            if (u >= 0 && heat.read(v) <= heat.read(u)) freed = false;
        }
        if (freed)
        {
            if (cfg.policy == kLazyExclusive && ram.is_dup(e))
            {
                target = ram.remove(e);
                acts.push_back(act(kActRamOverwrite, e, target, -1, -1, en));
            }
            else if (ram.free.count(r)) target = r;
            else if (!ram.free.empty()) target = *ram.free.begin();
        }
    }
    else if (cfg.demote == kDemoteHeat)
    {
        if (!ram.free.empty()) target = *ram.free.begin();
        else
        {
            int32_t d = reclaimable_dup();
            if (d >= 0) target = reclaim(d, acts, en);
            else
            {
                int32_t u = coldest_own();
                if (u >= 0 && heat.read(v) > heat.read(u)) target = evict_own(u, acts, en);
            }
        }
    }
    else target = room(v, false, acts, en);
    if (target < 0)
    {
        ++c[kDrops];
        acts.push_back(act(kActDrop, v, vs, -1, -1, en));
        if (!x.inplace) release(vs, acts, false, en);
        return;
    }
    ram.add(v, false, now, target, true);
    ++c[kD2H];
    acts.push_back(act(kActD2H, v, vs, target, -1, en));
    if (!x.inplace) release(vs, acts, true, en);
}

void TierCore::refill(const std::vector<int32_t>& bypassed, std::vector<Action>& acts)
{
    if (!cfg.refill || ram.R <= 0) return;
    int n = refill_busy;
    for (int32_t e : bypassed)
    {
        if (vram.slot_of[(size_t) e] >= 0 || ram.of[(size_t) e] >= 0) continue;
        int32_t u = -1;
        bool ok = !ram.free.empty() || (cfg.policy != kInclusive && !ram.lists[1].empty());
        if (!ok)
        {
            u = coldest_own();
            ok = u >= 0 && heat.read(e) > heat.read(u);
        }
        if (!ok) continue;
        if (n >= cfg.refill_inflight)
        {
            ++c[kRefillDropped];
            continue;
        }
        int32_t slot;
        if (!ram.free.empty()) slot = *ram.free.begin();
        else if (u < 0) slot = reclaim(reclaimable_dup(), acts, -1);
        else slot = evict_own(u, acts, -1);
        ram.add(e, false, seq, slot, false);
        ++c[kRefills];
        acts.push_back(act(kActRefill, e, slot, -1, -1, -1));
        ++n;
    }
}

void TierCore::maybe_adapt()
{
    if (vram.access < next_adapt) return;
    while (next_adapt <= vram.access) next_adapt += (uint64_t) cfg.adapt_every;
    if (adapt.pinned || cfg.admit != kAdmitAdaptive) return;
    uint32_t vmin = 0, rmin = 0;
    bool any = false;
    for (int32_t s = 0; s < vram.S; ++s)
        if (vram.state[(size_t) s] == kValid && (!any || vram.stamp[(size_t) s] < vmin))
        {
            vmin = vram.stamp[(size_t) s];
            any = true;
        }
    if (!any || !ram.min_stamp(&rmin)) return;
    adapt.update((int64_t) seq - (int64_t) vmin, (int64_t) seq - (int64_t) rmin);
    ++c[kAdapts];
}

void TierCore::check(bool deterministic) const
{
    const VramDirectory& v = vram;
    int64_t n = v.S + v.pins;
    int64_t counts[4] = { 0, 0, 0, 0 };
    for (int64_t s = 0; s < n; ++s)
    {
        uint8_t st = v.state[(size_t) s];
        expect(st <= kPinned, "slot state");
        ++counts[st];
        int32_t k = v.key[(size_t) s];
        if ((st == kValid || st == kPinned) && k >= 0)
            expect(v.slot_of[(size_t) k] == s, "slot " + std::to_string(s) + " holds " + std::to_string(k) +
                                              " but slot_of says " + std::to_string(v.slot_of[(size_t) k]));
        if (st == kFree) expect(k == -1, "free slot " + std::to_string(s) + " holds a key");
        // a retiring slot keeps its key (its bytes stay valid until it returns); retiring_of names the
        // latest retiring slot of a key, so an older one of the same key is not always listed
        if (st == kRetiring) expect(k >= 0, "retiring slot " + std::to_string(s) + " without a key");
    }
    expect(counts[kValid] == v.occupied, "occupied " + std::to_string(v.occupied) + " != VALID " + std::to_string(counts[kValid]));
    expect((int64_t) v.free.size() == counts[kFree], "free list size");
    for (int32_t s : v.free) expect(s < v.S && v.state[(size_t) s] == kFree, "free list entry " + std::to_string(s));
    for (int64_t k = 0; k < K; ++k)
    {
        int32_t s = v.slot_of[(size_t) k];
        if (s >= 0)
            expect((v.state[(size_t) s] == kValid || v.state[(size_t) s] == kPinned) && v.key[(size_t) s] == k,
                  "key " + std::to_string(k) + " -> slot " + std::to_string(s));
    }
    if (deterministic && !defer_returns) expect(counts[kRetiring] == 0, std::to_string(counts[kRetiring]) + " slots still RETIRING");
    expect((int64_t) (ram.lists[0].size() + ram.lists[1].size() + ram.free.size()) == ram.R, "RAM slots");
    for (int d = 0; d < 2; ++d)
        for (size_t i = 0; i < ram.lists[d].size(); ++i)
        {
            int32_t k = ram.lists[d][i];
            int32_t s = ram.of[(size_t) k];
            expect(ram.pos[(size_t) k] == (int32_t) i && s >= 0 && (int) ram.dup[(size_t) s] == d && ram.key[(size_t) s] == k,
                  "RAM list entry " + std::to_string(k));
        }
    for (int64_t k = 0; k < K; ++k)
    {
        int32_t s = ram.of[(size_t) k];
        if (s >= 0) expect(ram.key[(size_t) s] == k, "RAM key " + std::to_string(k));
        if (cfg.policy == kExclusive) expect(!ram.is_dup((int32_t) k), "exclusive: " + std::to_string(k) + " is a duplicate");
        if (cfg.policy == kLazyExclusive && ram.is_dup((int32_t) k))
            expect(v.slot_of[(size_t) k] >= 0, "lazy-exclusive: duplicate " + std::to_string(k) + " is not in VRAM");
        if (cfg.policy == kInclusive && v.slot_of[(size_t) k] >= 0 && v.state[(size_t) v.slot_of[(size_t) k]] == kValid)
            expect(s >= 0, "inclusive: VRAM key " + std::to_string(k) + " has no RAM copy");
    }
    if (cfg.demote == kDemoteSwap && !cfg.disk_off)
        expect(c[kD2H] <= c[kAdmitsFromRam], "swap: more demotions than promotions from RAM");
}

void TierCore::check_disk_off() const
{
    for (int64_t k = 0; k < K; ++k)
        expect(vram.slot_of[(size_t) k] >= 0 || ram.of[(size_t) k] >= 0 || vram.retiring_of[(size_t) k] >= 0,
              "disk experts=off: key " + std::to_string(k) + " is nowhere");
}

// ---------------------------------------------------------------------------------------- replay

void replay(TierCore& core, const std::vector<ScriptOp>& script, ReplayOut& out)
{
    for (size_t i = 0; i < script.size(); ++i)
    {
        const ScriptOp& op = script[i];
        switch (op.op)
        {
            case ScriptOp::kTick: core.tick((uint64_t) op.a); break;
            case ScriptOp::kCall:
            {
                Record rec = core.lookup((int32_t) op.a, op.ids.data(), (int64_t) op.ids.size(), (int) op.b);
                out.actions.emplace_back();
                out.action_ops.push_back((int) i);
                core.host(rec, out.actions.back());
                out.records.push_back(std::move(rec));
                break;
            }
            case ScriptOp::kLayer:
                out.actions.emplace_back();
                out.action_ops.push_back((int) i);
                core.layer_call((int32_t) op.a, op.ids.data(), (int64_t) op.ids.size(), out.actions.back());
                break;
            case ScriptOp::kComplete:
                out.actions.emplace_back();
                out.action_ops.push_back((int) i);
                core.complete(out.actions.back());
                break;
            case ScriptOp::kPlace: core.vram.place((int32_t) op.a, (int32_t) op.b, (uint32_t) op.c); break;
            case ScriptOp::kRamAdd: core.ram.add((int32_t) op.a, op.b != 0, (uint32_t) op.d, (int32_t) op.c, false); break;
            case ScriptOp::kHeatSet: core.heat.set(op.a, (uint64_t) op.b); break;
            case ScriptOp::kSeq: core.seq = (uint32_t) op.a; break;
            case ScriptOp::kColdFill: core.cold_fill(op.ids, nullptr, nullptr); break;
            case ScriptOp::kCheck: core.check(op.a != 0); break;
            default: fail("replay: unknown operation " + std::to_string(op.op));
        }
    }
}

}  // namespace exl3_tier
