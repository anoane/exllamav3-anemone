#pragma once

// The VRAM expert cache of one GPU (doc/expert_tiers.md, "The VRAM cache"): the lookup and fetch
// kernels' directory, control page and plan ring (tier_device.h), the RAM tier host (tier_host.h),
// and the tier thread that turns every call record into copy commands.
//
//   one MoE call (compute stream)          tier thread (no CUDA calls)
//   -----------------------------          --------------------------------------------------------
//   lookup kernel: hits, admissions,       record -> policy (TierCore::process): sources, RAM side,
//     victims, pointer tables, record        demotions -> copy commands into the plan ring (PlanCopier),
//   fetch kernel: all hits -> returns;       then the call's END; slots whose demotion is done go back
//     else runs the call's commands  <-----  through the return ring
//   the unchanged MoE kernels
//
// Nothing on the decode path waits for the host but a call with misses, and only on the GPU. The
// copy engines (CudaCopier) serve the cold fill only, before the tier thread starts.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "tier_device.h"
#include "tier_host.h"
#include "tier_policy.h"

namespace exl3_tier
{

// The copy engines as the host's Copier: lane 0 (high priority) carries promotions H2D and
// rescues D2D, lane 1 (lowest priority) the demotions D2H. A token is (lane << 62) | its lane's
// sequence number; an event per copy, from a ring per lane: an event slot reused by a later copy of
// the same lane only makes a check conservative (a lane completes in order)
class CudaCopier : public Copier
{
public:
    CudaCopier(int device, std::vector<uint64_t> pool, std::vector<uint64_t> staging, int64_t staging_per_half);
    ~CudaCopier() override;
    uint64_t h2d(uint64_t dst, const uint8_t* src, int64_t n, uint64_t after) override;
    uint64_t d2h(uint8_t* dst, uint64_t src, int64_t n, uint64_t after) override;
    uint64_t d2d(uint64_t dst, uint64_t src, int64_t n, uint64_t after) override;
    bool done(uint64_t token) override;
    void wait(uint64_t token) override;
    void chunk_filled(uint8_t* p, size_t n) override;
    uint64_t pool_addr(int32_t slot) override;
    uint64_t staging_addr(int half, int32_t i) override;

    // Page-lock host memory for the copy engines (the RAM tier after its fill, the slab)
    void pin(uint8_t* p, size_t n);
    void unpin(uint8_t* p);
    void unpin_all();
    void sync();
    cudaStream_t stream(int lane) const { return lanes_[lane].s; }

    int64_t h2d_bytes = 0, d2h_bytes = 0, d2d_bytes = 0;
    int64_t h2d_copies = 0, d2h_copies = 0, d2d_copies = 0;
    int64_t pinned_bytes = 0;
    int64_t pin_ns = 0;

private:
    struct Lane
    {
        cudaStream_t s = nullptr;
        std::vector<cudaEvent_t> ev;
        std::vector<uint64_t> ev_seq;
        uint64_t seq = 0, done_seq = 0;
    };
    uint64_t record(int lane);
    void order_after(int lane, uint64_t after);
    int device_;
    Lane lanes_[2];
    std::vector<uint64_t> pool_, staging_;
    int64_t staging_per_half_;
    std::vector<std::pair<uint8_t*, size_t>> pinned_;
};

// The fetch kernel as the host's Copier: every copy is a command appended to the plan ring (plain
// stores into mapped memory, no CUDA call); a token is the command's index + 1; done() reads the
// completion flags the fetch kernel writes. The commands of a call run while that call's fetch runs
class PlanCopier : public Copier
{
public:
    PlanCopier(PlanCmd* ring, uint32_t* done_flags, TierCtl* ctl, uint32_t n, std::vector<uint64_t> pool,
               std::vector<uint64_t> staging, int64_t staging_per_half);
    uint64_t h2d(uint64_t dst, const uint8_t* src, int64_t n, uint64_t after) override;
    uint64_t d2h(uint8_t* dst, uint64_t src, int64_t n, uint64_t after) override;
    uint64_t d2d(uint64_t dst, uint64_t src, int64_t n, uint64_t after) override;
    bool done(uint64_t token) override;
    void wait(uint64_t token) override;
    uint64_t pool_addr(int32_t slot) override;
    uint64_t staging_addr(int half, int32_t i) override;
    void end(uint32_t seq);
    void set_abort(const std::atomic<bool>* a) { abort_ = a; }
    uint64_t tail() const { return tail_; }

    int64_t h2d_bytes = 0, d2h_bytes = 0, d2d_bytes = 0;
    int64_t h2d_copies = 0, d2h_copies = 0, d2d_copies = 0;
    int64_t commands = 0, full_waits = 0, dep_dropped = 0;

private:
    uint64_t push(uint64_t src, uint64_t dst, int64_t n, uint64_t after, uint32_t kind, uint32_t seq);
    void scan();
    PlanCmd* ring_;
    uint32_t* done_;
    TierCtl* ctl_;
    uint32_t n_;
    uint64_t tail_ = 0;
    uint64_t contig_ = 0;               // every command below this index is complete
    std::vector<uint64_t> pool_, staging_;
    int64_t staging_per_half_;
    const std::atomic<bool>* abort_ = nullptr;
};

struct GpuConfig
{
    int device = 0;
    int projections = 3;
    int64_t proj_off[3] = { 0, 0, 0 };
    std::vector<uint64_t> pool;                 // S + pins slot addresses
    std::vector<uint64_t> staging;              // staging slot addresses (half 0, then half 1 if any)
    int64_t staging_per_half = 0;
    std::vector<uint64_t> tables;               // [L * P] device addresses of the int64 pointer tables
    bool deterministic = false;                 // the tier thread applies every record before its call runs
    int64_t spin_us = -1;                       // tier thread: -1 while calls come, 0 never, N us after the last
    std::vector<int> affinity;                  // CPUs of the tier thread
    int64_t rec_ring_bytes = 4 << 20;
    int64_t plan_n = 1 << 16;                   // plan ring commands (a power of two)
    int workers = 0;                            // fetch worker CTAs; 0: half the SMs
    int64_t trace = 0;                          // keep the last N processed records (tests)
};

struct GpuStats
{
    int64_t records = 0, host_records = 0, returns = 0;
    int64_t max_lag = 0;                        // records waiting when the thread picked one up
    int64_t idle_sleeps = 0;
    int64_t work_ns = 0;                        // tier thread time spent on records
};

class TierGpu
{
public:
    // host == nullptr: policy only, no bytes move (the lookup kernel's tests against the core)
    TierGpu(const PolicyConfig& pc, int64_t layers, int64_t experts, int pool_slots, int64_t ram_slots, int pins,
            const GpuConfig& gc, std::unique_ptr<TierHost> host_or_null, std::unique_ptr<CudaCopier> copier);
    ~TierGpu();
    TierGpu(const TierGpu&) = delete;
    TierGpu& operator=(const TierGpu&) = delete;

    TierCore& core() { return host_ ? host_->core : *core_; }
    TierHost* host() { return host_.get(); }
    CudaCopier* copier() { return copier_.get(); }
    PlanCopier* plan() { return plan_.get(); }

    // Load: fill the pool, the pins and the RAM tier, then the device directory from the result
    void cold_fill(const std::vector<int32_t>& order, const std::vector<int32_t>& pins);
    // Directory, heat and counters from the host's mirror to the device (after seeding in tests)
    void upload();
    void start();
    void stop();

    // One MoE call of cache layer lc on `stream` (the caller's): the lookup, then the wait
    void lookup(int lc, const int64_t* ids, int n, int mode, uint32_t seq, uint64_t tokens, cudaStream_t stream);
    // Without the thread: process every record available (wait for at least one when `wait`)
    int step(bool wait);
    // Every record published so far processed, every copy landed, every demoted slot returned
    void drain();
    // The device directory equals the mirror (after drain); throws naming the first difference
    void verify();

    std::string error() const;
    uint32_t device_error() const;
    uint32_t overflow() const;
    GpuStats stats;
    std::deque<Record> trace;
    std::mutex trace_mu;

    // device arrays (for tests: download through the handle)
    struct Dev
    {
        uint8_t* state = nullptr;
        int32_t* key = nullptr;
        uint32_t* stamp = nullptr;
        int64_t* addr = nullptr;
        int32_t* slot_of = nullptr;
        uint32_t* heat_v = nullptr;
        uint32_t* heat_ep = nullptr;
        uint32_t* free_bits = nullptr;
        uint64_t* dctr = nullptr;
        int64_t* staging = nullptr;
        int64_t* tables = nullptr;
        FetchCmd* dring = nullptr;
        uint32_t* ddone = nullptr;
        uint32_t* dcnt = nullptr;
    } dev;
    int S, pins, K, W;

private:
    bool next_record(Record& rec, bool& needs_host, uint64_t& tokens);
    void handle(const Record& rec, bool needs_host, uint64_t tokens);
    void push_returns(const std::vector<int32_t>& slots);
    void publish_p();
    void loop();
    void fail(const std::string& what);
    void check_ok();

    PolicyConfig pc_;
    GpuConfig gc_;
    std::unique_ptr<TierHost> host_;
    std::unique_ptr<TierCore> core_;
    std::unique_ptr<CudaCopier> copier_;
    std::unique_ptr<PlanCopier> plan_;
    TierKParams kp_ {};
    FetchParams fp_ {};
    void* dev_mem_ = nullptr;                   // one allocation for every device array
    uint8_t* ctl_mem_ = nullptr;                // mapped control page, return ring, record ring
    TierCtl* ctl_ = nullptr;                    // host view
    TierCtl* ctl_dev_ = nullptr;                // device view
    uint32_t* ret_ring_ = nullptr;
    uint8_t* rec_ring_ = nullptr;
    PlanCmd* plan_ring_ = nullptr;
    uint32_t* plan_done_ = nullptr;
    uint32_t ret_n_ = 0;
    uint32_t plan_n_ = 0;
    uint64_t rec_bytes_ = 0;
    uint64_t rec_tail_ = 0;
    uint32_t ret_tail_ = 0;
    std::atomic<uint32_t> last_seq_ { 0 };
    uint64_t last_p_ = ~0ull;

    std::thread th_;
    std::atomic<bool> quit_ { false };
    std::atomic<bool> running_ { false };
    std::atomic<uint64_t> kicks_ { 0 };
    std::atomic<bool> drain_req_ { false };
    std::mutex mu_;
    std::condition_variable cv_, drained_cv_;
    bool drained_ = false;
    std::string error_;
    mutable std::mutex err_mu_;
    std::atomic<bool> failed_ { false };
    std::mutex step_mu_;                        // step() and the thread never run at once
};

}  // namespace exl3_tier
