#pragma once

// Expert tier policy: every decision of the expert tier (the VRAM cache's lookup, the host's
// handling of each call record, the RAM tier's ownership, demotions, RAM admission, refills),
// as exllamav3/model/expert_tier_policy.py defines it. This core reproduces that reference
// exactly (same records, actions, counters and final state for the same calls); the GPU lookup
// kernel and the tier thread reproduce this core. doc/expert_tiers.md, "Policies".
//
// Torch-free and CUDA-free: the standalone driver (tests/expert_tier/tier_policy_test.cpp)
// builds it with g++ alone.

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace exl3_tier
{

enum Policy : int { kLazyExclusive = 0, kExclusive = 1, kInclusive = 2 };
enum Demote : int { kDemoteSwap = 0, kDemoteHeat = 1, kDemoteAll = 2, kDemoteOff = 3 };
enum Admit : int { kAdmitAdaptive = 0, kAdmitHeat = 1, kAdmitAlways = 2 };
enum Evict : int { kLru = 0, kLfu = 1 };
enum RamAdmit : int { kRamAdmitAlways = 0, kRamAdmitHeat = 1 };
enum SlotState : uint8_t { kFree = 0, kValid = 1, kRetiring = 2, kPinned = 3 };
enum Kind : uint8_t { kHit = 0, kAdmitted = 1, kTransient = 2 };
enum Mode : int { kDecode = 0, kRouted = 1, kLayer = 2 };

constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ull;      // access draws: seed ^ (access * kGolden)
constexpr uint64_t kDrawMul = 0xD1B54A32D192ED03ull;     // RAM samples: seed ^ (draw * kDrawMul)
constexpr uint32_t kQ16 = 1u << 16;

class TierError : public std::runtime_error
{
public:
    explicit TierError(const std::string& m) : std::runtime_error(m) {}
};

uint64_t splitmix64(uint64_t x);
// An admission probability as the threshold a 32-bit draw is compared against (2^32 = always)
uint64_t p_q32(double p);

struct PolicyConfig
{
    int policy = kLazyExclusive;
    int demote = kDemoteSwap;
    int admit = kAdmitAdaptive;
    int evict = kLru;                   // the VRAM pool
    int ram_evict = kLfu;
    int ram_admit = kRamAdmitAlways;
    bool demote_on = true;              // EXL3_MOE_TIER_DEMOTE
    bool disk_off = false;              // disk experts=off
    int spare = 8;
    int sample = 16;
    uint64_t seed = 0;
    uint32_t halflife = 256;            // decode tokens
    uint32_t prefill_inc = 4096;        // 16.16
    int64_t adapt_every = 2048;         // accesses
    double pmin = 0.05;
    bool p_pinned = false;
    double p = 0.0;                     // when pinned
    bool refill = true;
    int refill_inflight = 2;

    bool demotes() const { return policy != kInclusive && demote != kDemoteOff && demote_on; }
    void validate() const;
};

// Per key: u32 16.16 value and the epoch it was written in; epoch = tokens / halflife
class Heat
{
public:
    Heat(int64_t nkeys, uint32_t halflife);
    uint32_t epoch() const { return (uint32_t) (tokens / halflife_); }
    uint32_t read(int64_t k) const;
    void add(int64_t k, uint64_t inc);
    void set(int64_t k, uint64_t value);
    uint64_t tokens = 0;
    std::vector<uint32_t> v, ep;
private:
    uint32_t halflife_;
};

class AdaptiveP
{
public:
    AdaptiveP(int64_t capacity, int64_t ram_slots, double pmin, bool pinned, double p);
    double update(int64_t lv, int64_t lr);
    uint64_t q32() const { return p_q32(p); }
    double pmax, pmin, p;
    bool pinned;
};

struct Entry
{
    int32_t key = -1;
    uint8_t kind = kHit;
    uint8_t inplace = 0;
    uint32_t cnt = 0;
    int32_t dst = -1;                   // VRAM slot (hit, admit) or staging slot (transient)
    uint32_t heat = 0;
    int32_t vkey = -1;                  // paired victim
    int32_t vslot = -1;
    uint32_t vstamp = 0;
    uint32_t vheat = 0;
};

struct Record
{
    uint32_t seq = 0;
    int32_t lc = 0;
    int mode = kDecode;
    std::vector<Entry> entries;
};

// What the runtime does for a record, in order. a..d as the Python tuples (-1 when absent)
enum ActionOp : int
{
    kActD2D = 0,            // (key, retiring slot, kind, dst)       rescue from a retiring slot
    kActH2D,                // (key, RAM slot, kind, dst)
    kActRamFree,            // (key, RAM slot)                       exclusive: freed after the read
    kActRamDup,             // (key, RAM slot)                       now a duplicate
    kActSsd,                // (key, RAM slot or -1 (slab), kind, dst)
    kActRamEvict,           // (key, RAM slot)                       an owner evicted for room
    kActDupReclaim,         // (key, RAM slot)                       a duplicate reclaimed for room
    kActRamOwn,             // (key, RAM slot)                       duplicate of a victim becomes owner
    kActRamOverwrite,       // (key, RAM slot)                       the promoted expert's duplicate, overwritten
    kActD2H,                // (victim key, VRAM slot, RAM slot)     demotion (compact layout)
    kActDrop,               // (victim key, VRAM slot)
    kActReturn,             // (VRAM slot)
    kActRefill,             // (key, RAM slot)                       class-3 read
    kActStageRam,           // (key, RAM slot, staging index)        layer mode
    kActStageSsd,           // (key, staging index)                  layer mode
    kActCount
};
const char* action_name(int op);
int action_arity(int op);

struct Action
{
    int op;
    int32_t a = -1, b = -1, c = -1, d = -1;
    int32_t entry = -1;                 // index of the record entry it belongs to (-1: refill, stage)
};

enum Counter : int
{
    kCalls = 0, kHits, kAdmits, kAdmitsFromRam, kTransients, kStarved, kRetires, kInplace, kDrops, kD2H,
    kH2DRam, kSsd, kRescued, kRamEvict, kDupReclaim, kRamAdmitRefused, kRefills, kRefillDropped,
    kPrefillH2D, kPrefillSsd, kAdapts, kNumCounters
};
const char* counter_name(int c);

class VramDirectory
{
public:
    VramDirectory(int64_t nkeys, int slots, int spare, int pins);
    int capacity() const { return S - spare; }
    void place(int32_t key, int32_t slot, uint32_t stamp);
    int32_t victim(uint32_t now, const Heat* lfu) const;
    void retire(int32_t slot);
    void return_slot(int32_t slot);
    int32_t retiring_slot_of(int32_t key) const;
    // One call; returns the record and adds the admissions that found no free slot to *starved
    Record lookup(int32_t lc, const int32_t* ids, int64_t n, int mode, uint32_t seq, int64_t E, Heat& heat,
                  const PolicyConfig& cfg, uint64_t pq, uint32_t inc, int64_t* starved);
    // Replay a record decided elsewhere (the lookup kernel) on this mirror
    void apply(const Record& rec);

    int S, spare, pins;
    std::vector<uint8_t> state;
    std::vector<int32_t> key;
    std::vector<uint32_t> stamp;
    std::vector<int32_t> slot_of;
    std::vector<int32_t> retiring_of;   // key -> RETIRING slot that still holds it, or -1
    std::set<int32_t> free;
    int64_t occupied = 0;
    uint64_t access = 0;
private:
    void set_stamp(int32_t slot, uint32_t st);
    std::set<std::pair<uint32_t, int32_t>> lru_;     // (stamp, slot) of VALID pool slots
};

class RamTier
{
public:
    RamTier(int64_t slots, int64_t nkeys, int sample, uint64_t seed);
    int64_t size() const { return R - (int64_t) free.size(); }
    bool owns(int32_t k) const { int32_t s = of[(size_t) k]; return s >= 0 && !dup[(size_t) s]; }
    bool is_dup(int32_t k) const { int32_t s = of[(size_t) k]; return s >= 0 && dup[(size_t) s]; }
    int32_t add(int32_t k, bool is_dup, uint32_t now, int32_t slot, bool compact);
    int32_t remove(int32_t k);
    void set_dup(int32_t k, bool d);
    uint64_t rand(uint64_t n);
    void sample(bool d, std::vector<int32_t>& out);
    bool min_stamp(uint32_t* out) const;

    int64_t R;
    std::vector<int32_t> key;
    std::vector<uint8_t> dup, compact;
    std::vector<uint32_t> stamp;
    std::vector<int32_t> of, pos;
    std::vector<int32_t> lists[2];      // owners, duplicates (swap-remove)
    std::set<int32_t> free;
    int sample_n;
    uint64_t seed, draws = 0;
private:
    void list_add(int32_t k, bool d);
    void list_del(int32_t k, bool d);
};

class TierCore
{
public:
    TierCore(const PolicyConfig& cfg, int64_t layers, int64_t experts, int slots, int64_t ram_slots, int pins,
             bool defer_returns);

    void cold_fill(const std::vector<int32_t>& order, std::vector<int32_t>* vram_keys,
                   std::vector<int32_t>* ram_keys);
    void tick(uint64_t tokens) { heat.tokens += tokens; }
    Record lookup(int32_t lc, const int32_t* ids, int64_t n, int mode);
    void host(const Record& rec, std::vector<Action>& acts);
    void layer_call(int32_t lc, const int32_t* ids, int64_t n, std::vector<Action>& acts);
    void complete(std::vector<Action>& acts);
    // Throws TierError naming the first invariant that fails
    void check(bool deterministic) const;
    void check_disk_off() const;

    PolicyConfig cfg;
    int64_t L, E, K;
    Heat heat;
    VramDirectory vram;
    RamTier ram;
    AdaptiveP adapt;
    uint32_t seq = 0;
    uint64_t next_adapt;
    int64_t c[kNumCounters] = {};
    bool defer_returns;
    std::vector<int32_t> pending;       // demoted slots whose copy has not landed (defer_returns)
    int refill_busy = 0;                // refills of earlier calls still in flight (set by the runtime)

private:
    uint64_t metric(int32_t k) const;
    int32_t coldest_own();
    int32_t reclaimable_dup();
    int32_t evict_own(int32_t u, std::vector<Action>& acts, int entry);
    int32_t reclaim(int32_t d, std::vector<Action>& acts, int entry);
    int32_t room(int32_t e, bool gated, std::vector<Action>& acts, int entry);
    void victim(const Entry& x, int32_t e, int32_t r, std::vector<Action>& acts, uint32_t now, int entry);
    void release(int32_t vslot, std::vector<Action>& acts, bool demoted, int entry);
    void refill(const std::vector<int32_t>& bypassed, std::vector<Action>& acts);
    void maybe_adapt();
    std::vector<int32_t> scratch_;
};

// Scripted replay (tests): the operations of expert_tier_policy.replay() plus seeding
struct ScriptOp
{
    enum Op : int { kTick = 0, kCall, kLayer, kComplete, kPlace, kRamAdd, kHeatSet, kSeq, kColdFill, kCheck };
    int op = kTick;
    int64_t a = 0, b = 0, c = 0, d = 0;
    std::vector<int32_t> ids;
};

struct ReplayOut
{
    std::vector<Record> records;        // one per kCall (lookup records)
    std::vector<std::vector<Action>> actions;   // one per operation that produces actions
    std::vector<int> action_ops;        // index of the script op each actions list belongs to
};

void replay(TierCore& core, const std::vector<ScriptOp>& script, ReplayOut& out);

}  // namespace exl3_tier
