#pragma once

// The host side of the expert tier (doc/expert_tiers.md, "The RAM tier"): the RAM tier's pinned
// slot arena, registered with the disk engine, the disk slab, the extent table (where each expert
// lies in the checkpoint, from model/expert_extents.py), and the execution of the policy's actions
// (tier_policy.h): SSD reads of the model's own shards in place through the disk engine's extent
// API (O_DIRECT into 4 KiB-aligned slots), promotions and demotions through a Copier, refills
// (class 3), the prefill read-ahead (class 2) and the cold fill at load.
//
// Copies go through Copier: HostCopier moves bytes with memcpy into a host buffer that stands in
// for the GPU's pool and staging (CPU tests, and the CPU replay of a trace); the GPU runtime
// implements Copier with the copy engines. Every hazard between an SSD read, a promotion reading a
// RAM slot and a demotion writing one is ordered here (per-slot tokens), whatever the Copier.
//
// Torch-free and CUDA-free. Linux-only (the disk engine): elsewhere the constructor throws.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../disk/disk_engine.h"
#include "tier_policy.h"

namespace exl3_tier
{

// Where one expert lies: up to three pieces (one when its tensors are contiguous) and, per
// projection, the piece and offset of its trellis
struct ExpertExtent
{
    std::string name;
    int npieces = 1;
    int32_t file[3] = { -1, -1, -1 };
    int64_t offset[3] = { 0, 0, 0 };            // absolute file offset of each piece
    int64_t length[3] = { 0, 0, 0 };
    int32_t tpiece[3] = { 0, 0, 0 };            // piece holding projection p's trellis
    int64_t trel[3] = { 0, 0, 0 };              // offset of that trellis in its piece
    // placement in a RAM slot, computed at construction from the engine's geometry
    int64_t base[3] = { 0, 0, 0 };              // piece start in the slot (4 KiB aligned)
    int64_t payload[3] = { 0, 0, 0 };           // where the piece's first byte lands in its sub-slot
    int64_t span[3] = { 0, 0, 0 };
};

struct Geometry
{
    int projections = 3;
    int64_t proj_bytes[3] = { 0, 0, 0 };        // trellis bytes per projection (gate, up, down)
    int64_t proj_off[3] = { 0, 0, 0 };          // compact offsets
    int64_t vram_slot_bytes = 0;
    int64_t ram_slot_bytes = 0;
};

struct HostConfig
{
    int64_t chunk_bytes = 1ll << 30;            // RAM arena chunk: one registered io_uring buffer
    bool hugepage = false;                      // MADV_HUGEPAGE on arena and slab chunks
    int prefault_threads = 8;                   // threads that fault the chunks in before they are
                                                // registered (0: registration faults them, one by one)
    int slab_slots = 8;                         // disk slab: reads that bypass the RAM tier
    int fill_inflight = 8;                      // cold fill reads in flight
    int staging_slots = 0;                      // per staging half (E - hot); 0 = experts per layer
    bool deterministic = true;                  // finish every read and copy before a call returns
    bool compact_fill = false;                  // move the cold-filled RAM slots to the compact layout (one
                                                // aligned copy per promotion instead of three unaligned ones)
    int io_direct = -1;                         // expert reads: -1 as EXL3_DISK_DIRECT says, 0 buffered,
                                                // 1 O_DIRECT (the placement's disk io=)
};

// Device-side copies. dst / src device addresses are uint64_t (HostCopier: host pointers). `after`:
// a token the copy is ordered after (0: none). Returns a token (0: complete already)
class Copier
{
public:
    virtual ~Copier() = default;
    virtual uint64_t h2d(uint64_t dst, const uint8_t* src, int64_t n, uint64_t after) = 0;
    virtual uint64_t d2h(uint8_t* dst, uint64_t src, int64_t n, uint64_t after) = 0;
    virtual uint64_t d2d(uint64_t dst, uint64_t src, int64_t n, uint64_t after) = 0;
    virtual bool done(uint64_t token) = 0;
    virtual void wait(uint64_t token) = 0;
    // Whether copies with tokens a and b complete in order (the later token covers the earlier one)
    virtual bool same_order(uint64_t a, uint64_t b) { (void) a; (void) b; return true; }
    // A RAM arena chunk has been filled (the GPU runtime registers it with CUDA: pin after fill)
    virtual void chunk_filled(uint8_t* p, size_t n) { (void) p; (void) n; }
    // Addresses of the pool slots (pool, then hot pins) and staging slots (half 0, then half 1)
    virtual uint64_t pool_addr(int32_t slot) = 0;
    virtual uint64_t staging_addr(int half, int32_t i) = 0;
};

// memcpy into a host buffer standing in for the GPU's memory
class HostCopier : public Copier
{
public:
    HostCopier(int64_t pool_slots, int64_t staging_per_half, int64_t slot_bytes);
    uint64_t h2d(uint64_t dst, const uint8_t* src, int64_t n, uint64_t after) override;
    uint64_t d2h(uint8_t* dst, uint64_t src, int64_t n, uint64_t after) override;
    uint64_t d2d(uint64_t dst, uint64_t src, int64_t n, uint64_t after) override;
    bool done(uint64_t) override { return true; }
    void wait(uint64_t) override {}
    uint64_t pool_addr(int32_t slot) override;
    uint64_t staging_addr(int half, int32_t i) override;
    const uint8_t* pool(int32_t slot) const;
    const uint8_t* staging(int half, int32_t i) const;
    int64_t h2d_bytes = 0, d2h_bytes = 0, d2d_bytes = 0;
    int64_t h2d_copies = 0, d2h_copies = 0, d2d_copies = 0;
private:
    std::vector<uint8_t> mem_;
    int64_t pool_slots_, staging_, slot_;
};

// Anonymous memory in chunks of whole slots, each chunk registered with the disk engine (READ_FIXED)
class Arena
{
public:
    Arena(int64_t slots, int64_t slot_bytes, int64_t chunk_bytes, bool hugepage, exl3_disk::Engine* eng,
          int prefault_threads = 0);
    ~Arena();
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    uint8_t* slot(int64_t i) const;
    int64_t chunk_of(int64_t i) const { return i / chunk_slots; }
    int64_t slots, slot_bytes, chunk_slots;
    std::vector<uint8_t*> base;
    std::vector<size_t> bytes;
    std::vector<int> reg;                       // registered buffer index per chunk, -1 none
private:
    void release();
    exl3_disk::Engine* eng_;
};

struct HostStats
{
    int64_t ssd_reads = 0, ssd_bytes = 0, pread_fallbacks = 0;
    int64_t h2d = 0, d2h = 0, d2d = 0;          // copies issued (a 3-piece promotion counts once)
    int64_t staged = 0;                         // layer-mode staging copies
    int64_t prefetch_issued = 0, prefetch_used = 0, prefetch_starved = 0, prefetch_dropped = 0;
    int64_t presubmitted = 0;                   // SSD reads of a record submitted before its first copy
    int64_t predicted_reads = 0;                // prefetch=router reads into the slab
    int64_t slab_to_vram = 0;                   // misses served from the slab (read ahead), their RAM
                                                // slot then filled by a background read
    int64_t slab_reused = 0;                    // SSD reads saved: the slab still held the expert
    int64_t refill_reads = 0;
    int64_t hazard_waits = 0;                   // a write into a slot waited for a read or write of it
    int64_t slab_waits = 0;                     // the slab was full and a read waited for a slot
    int64_t cold_ram = 0, cold_vram = 0, cold_bytes = 0, cold_ns = 0;
    int64_t compacted = 0, compact_ns = 0;      // RAM slots moved to the compact layout at the cold fill
    int64_t arena_ns = 0;                       // mapping, faulting in and registering the RAM tier
};

class TierHost
{
public:
    TierHost(const PolicyConfig& pc, int64_t layers, int64_t experts, int pool_slots, int64_t ram_slots, int pins,
             const Geometry& g, const std::vector<std::string>& files, const std::vector<ExpertExtent>& extents,
             const HostConfig& hc, Copier* copier, bool defer_returns = false);
    ~TierHost();
    TierHost(const TierHost&) = delete;
    TierHost& operator=(const TierHost&) = delete;

    // Load: place the keys in priority order (pool, then RAM tier) and read them: RAM slots in
    // place, pool slots through the slab (or from their RAM copy when RAM holds one too). `pins`
    // go to the hot-pin slots after the pool first (read through the slab); `order` leaves them out
    void cold_fill(const std::vector<int32_t>& order, const std::vector<int32_t>& pins = {});
    // Give the disk slab back once nothing reads the SSD at runtime (disk experts=off, after the
    // cold fill)
    void release_slab();
    // One call decided by the reference directory (CPU replay): lookup, then process
    Record call(int32_t lc, const int32_t* ids, int64_t n, int mode);
    // The host side of a record decided elsewhere (the lookup kernel): apply it to the mirror, add
    // its heat and counters, then run the policy's host step and execute its actions
    void process(const Record& rec);
    // Production mode (defer_returns): the demoted pool slots whose copy to RAM has landed, handed
    // back to the directory mirror; the caller returns them to the lookup kernel
    std::vector<int32_t> return_landed();
    // A layer-mode prefill call: stage every cached expert of the layer that the pool does not
    // hold into staging half `half` (RAM copies, prefetched or fresh SSD reads), then refills
    void layer_call(int32_t lc, const int32_t* ids, int64_t n, int half);
    // Layer mode on the GPU: stage every cached expert of layer lc that the pool does not hold into
    // staging half `half` (the policy's layer_plan, executed through the current Copier: the copy
    // engines of the caller's thread). keys (optional) receives the staged keys in staging order
    void stage_layer(int32_t lc, int half, std::vector<int32_t>* keys);
    // Class-2 reads of the layer's SSD-only experts into the slab, ahead of its layer call
    void prefetch_layer(int32_t lc, int64_t deadline_ns);
    // prefetch=router: class-2 reads into the slab of the experts a prediction record lists (a later
    // layer's router applied to an earlier layer's state) that neither VRAM nor RAM holds
    void read_ahead(const Record& rec);
    // Promote the layer's prefetch reads still queued to class 1 (its call is next)
    void promote_layer(int32_t lc);
    void tick(uint64_t tokens) { core.tick(tokens); }
    // Deferred returns (production emulation): demotions in flight have landed
    void complete();
    // Wait for every read and copy in flight (refills and prefetches included)
    void drain();
    // Retire what completed without waiting (refills)
    void poll();

    // Hand the copies to another Copier (the runtime: from the copy engines of the load to the fetch
    // kernel's plan). Only when nothing is in flight (after cold_fill() or drain())
    void set_copier(Copier* c) { copier_ = c; }

    // The trellis bytes of a RAM slot in projection order (compact), for checks
    void ram_trellis(int32_t r, uint8_t* out) const;
    const uint8_t* ram_slot(int32_t r) const { return arena_->slot(r); }
    int64_t ram_slot_offset(int32_t r, int p) const;
    const Arena& arena() const { return *arena_; }
    const Arena& slab() const { return *slab_; }
    const std::string& error() const { return error_; }
    // The actions of the last call, layer call or completion (tests, tracing)
    const std::vector<Action>& last_actions() const { return last_; }

    TierCore core;
    Geometry geo;
    HostConfig hcfg;
    HostStats stats;

private:
    struct SlotIo                               // per RAM or slab slot
    {
        exl3_disk::TicketId ticket = 0;         // a read writing it
        int32_t ticket_key = -1;
        uint64_t read_tok = 0;                  // the last copy reading it
        uint64_t write_tok = 0;                 // a demotion writing it
        int layout = 0;                         // 0 empty, 1 extent of layout_key, 2 compact
        int32_t layout_key = -1;
    };
    // A disk slab slot: FREE; READING / READY for a demand read of `key` (submitted ahead of its
    // copy); PREFETCH (a read-ahead of layer `lc`, dropped first when the slab runs out); COPIED (its
    // copy to VRAM was issued, SlotIo::read_tok; the bytes stay valid and are reused by a later read
    // of the same expert until the slot is reclaimed, oldest first)
    enum SlabKind : int { kSlabFree = 0, kSlabDemand = 1, kSlabPrefetch = 2, kSlabCopied = 3 };
    struct Slab
    {
        int32_t key = -1;                       // the expert read (or being read) into it
        int kind = kSlabFree;
        int32_t lc = -1;
        uint64_t age = 0;                       // COPIED: order of its copy (reclaim oldest first)
        bool pinned = false;                    // an action of the record being executed reads it:
                                                // never reclaimed or dropped until that action runs
    };

    void execute(const Record* rec, std::vector<Action>& acts);
    void run_action(const Action& a, const Record* rec, uint64_t* entry_after);
    uint64_t dst_addr(int kind, int32_t dst, int half) ;
    void promote_from_ram(int32_t r, uint64_t dst, uint64_t after, bool staging);
    void stage_from_ssd(int32_t key, uint64_t dst, uint64_t after, bool staging, int cls);
    void read_into(int32_t key, uint8_t* slot, int cls, int64_t deadline, exl3_disk::TicketId* ticket);
    void presubmit(const std::vector<Action>& acts);
    bool slab_submit(int32_t key, int cls, int64_t deadline, int kind, int32_t lc, bool may_wait);
    void finish_read(int32_t key, uint8_t* slot, exl3_disk::TicketId t);
    void pread_extent(int32_t key, uint8_t* slot);
    void ram_ready_for_write(int32_t r);
    void compact_ram(const std::vector<int32_t>& slots);
    void ram_ready_for_read(int32_t r);
    int32_t slab_take(bool may_wait, int32_t lc);
    uint64_t later(uint64_t a, uint64_t b);
    void slab_release(int32_t i);
    int32_t slab_of(int32_t key) const;
    std::string where(int32_t key) const;
    void fail_read(int32_t key, int err, const std::string& what);
    void throw_if_failed();

    std::shared_ptr<exl3_disk::Engine> eng_;
    Copier* copier_;
    std::vector<std::string> files_;
    std::vector<int> fds_;
    std::vector<ExpertExtent> ext_;
    std::unique_ptr<Arena> arena_, slab_;
    std::vector<SlotIo> ram_io_, slab_io_;
    std::vector<Slab> slab_state_;
    std::vector<int32_t> slab_free_;
    uint64_t slab_age_ = 0;
    std::vector<int32_t> presub_ram_;           // RAM slot -> key whose read presubmit() started, -1
    std::vector<uint64_t> pool_busy_;           // a demotion reading a pool slot
    std::vector<std::pair<int32_t, exl3_disk::TicketId>> refills_;  // RAM slot, ticket
    std::string error_;
    int cur_half_ = 0;
    std::vector<Action> last_;
};

}  // namespace exl3_tier
