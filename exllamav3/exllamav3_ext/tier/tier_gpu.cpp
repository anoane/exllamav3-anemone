// The VRAM expert cache's runtime; see tier_gpu.h.

#include "tier_gpu.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif


namespace exl3_tier
{

namespace
{

constexpr int kEventsPerLane = 4096;
constexpr uint64_t kSeqMask = (1ull << 62) - 1;

[[noreturn]] void fail_tier(const std::string& m) { throw TierError("expert tier: " + m); }

void cuda_ok(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) fail_tier(std::string(what) + ": " + cudaGetErrorString(e));
}

int64_t mono_ns()
{
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void cpu_relax()
{
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

struct DeviceGuard
{
    int prev = -1;
    explicit DeviceGuard(int d)
    {
        cudaGetDevice(&prev);
        if (prev != d) cudaSetDevice(d);
    }
    ~DeviceGuard()
    {
        int cur = -1;
        cudaGetDevice(&cur);
        if (prev >= 0 && cur != prev) cudaSetDevice(prev);
    }
};

const char* dev_error_text(uint32_t e)
{
    switch (e)
    {
        case kErrExpertRange: return "a selected expert is outside the layer's range";
        case kErrStaging: return "a call has more transient experts than the staging area holds";
        case kErrRing: return "the tier thread stopped taking call records";
        default: return "unknown device error";
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------- copier

CudaCopier::CudaCopier(int device, std::vector<uint64_t> pool, std::vector<uint64_t> staging, int64_t staging_per_half)
    : device_(device), pool_(std::move(pool)), staging_(std::move(staging)), staging_per_half_(staging_per_half)
{
    DeviceGuard g(device_);
    int least = 0, greatest = 0;
    cuda_ok(cudaDeviceGetStreamPriorityRange(&least, &greatest), "stream priorities");
    for (int l = 0; l < 2; ++l)
    {
        Lane& L = lanes_[l];
        cuda_ok(cudaStreamCreateWithPriority(&L.s, cudaStreamNonBlocking, l == 0 ? greatest : least), "copy stream");
        L.ev.resize(kEventsPerLane);
        L.ev_seq.assign(kEventsPerLane, 0);
        for (auto& e : L.ev) cuda_ok(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "copy event");
    }
}

CudaCopier::~CudaCopier()
{
    DeviceGuard g(device_);
    for (auto& L : lanes_)
    {
        if (L.s) cudaStreamSynchronize(L.s);
        for (auto& e : L.ev) cudaEventDestroy(e);
        if (L.s) cudaStreamDestroy(L.s);
    }
    unpin_all();
}

uint64_t CudaCopier::record(int lane)
{
    Lane& L = lanes_[lane];
    ++L.seq;
    size_t i = (size_t) (L.seq % kEventsPerLane);
    cuda_ok(cudaEventRecord(L.ev[i], L.s), "event record");
    L.ev_seq[i] = L.seq;
    return ((uint64_t) lane << 62) | L.seq;
}

bool CudaCopier::done(uint64_t token)
{
    if (!token) return true;
    Lane& L = lanes_[token >> 62];
    uint64_t seq = token & kSeqMask;
    if (seq <= L.done_seq) return true;
    size_t i = (size_t) (seq % kEventsPerLane);
    cudaError_t q = cudaEventQuery(L.ev[i]);
    if (q == cudaSuccess)
    {
        L.done_seq = std::max(L.done_seq, L.ev_seq[i]);
        return true;
    }
    if (q != cudaErrorNotReady) cuda_ok(q, "event query");
    return false;
}

void CudaCopier::wait(uint64_t token)
{
    if (done(token)) return;
    Lane& L = lanes_[token >> 62];
    size_t i = (size_t) ((token & kSeqMask) % kEventsPerLane);
    cuda_ok(cudaEventSynchronize(L.ev[i]), "event wait");
    L.done_seq = std::max(L.done_seq, L.ev_seq[i]);
}

void CudaCopier::order_after(int lane, uint64_t after)
{
    if (!after || done(after)) return;
    int al = (int) (after >> 62);
    if (al == lane) return;                         // a lane runs in order
    Lane& A = lanes_[al];
    size_t i = (size_t) ((after & kSeqMask) % kEventsPerLane);
    cuda_ok(cudaStreamWaitEvent(lanes_[lane].s, A.ev[i], 0), "stream wait");
}

uint64_t CudaCopier::h2d(uint64_t dst, const uint8_t* src, int64_t n, uint64_t after)
{
    order_after(0, after);
    cuda_ok(cudaMemcpyAsync(reinterpret_cast<void*>(dst), src, (size_t) n, cudaMemcpyHostToDevice, lanes_[0].s), "H2D");
    h2d_bytes += n;
    ++h2d_copies;
    return record(0);
}

uint64_t CudaCopier::d2h(uint8_t* dst, uint64_t src, int64_t n, uint64_t after)
{
    order_after(1, after);
    cuda_ok(cudaMemcpyAsync(dst, reinterpret_cast<const void*>(src), (size_t) n, cudaMemcpyDeviceToHost, lanes_[1].s),
            "D2H");
    d2h_bytes += n;
    ++d2h_copies;
    return record(1);
}

uint64_t CudaCopier::d2d(uint64_t dst, uint64_t src, int64_t n, uint64_t after)
{
    order_after(0, after);
    cuda_ok(cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), (size_t) n,
                            cudaMemcpyDeviceToDevice, lanes_[0].s), "D2D");
    d2d_bytes += n;
    ++d2d_copies;
    return record(0);
}

void CudaCopier::chunk_filled(uint8_t* p, size_t n)
{
    pin(p, n);
}

void CudaCopier::pin(uint8_t* p, size_t n)
{
    if (!n) return;
    int64_t t0 = mono_ns();
    // mapped: the fetch kernel reads and writes it directly, at the same address (unified addressing)
    cuda_ok(cudaHostRegister(p, n, cudaHostRegisterPortable | cudaHostRegisterMapped),
            "page-locking the RAM tier (cudaHostRegister)");
    void* dp = nullptr;
    if (cudaHostGetDevicePointer(&dp, p, 0) != cudaSuccess || dp != (void*) p)
    {
        cudaHostUnregister(p);
        fail_tier("page-locked host memory is not visible to the GPU at its host address (unified addressing)");
    }
    pinned_.push_back({ p, n });
    pinned_bytes += (int64_t) n;
    pin_ns += mono_ns() - t0;
}

void CudaCopier::unpin(uint8_t* p)
{
    for (size_t i = 0; i < pinned_.size(); ++i)
        if (pinned_[i].first == p)
        {
            cudaHostUnregister(p);
            pinned_bytes -= (int64_t) pinned_[i].second;
            pinned_.erase(pinned_.begin() + (long) i);
            return;
        }
}

void CudaCopier::unpin_all()
{
    for (auto& x : pinned_) cudaHostUnregister(x.first);
    pinned_.clear();
    pinned_bytes = 0;
}

void CudaCopier::sync()
{
    for (auto& L : lanes_) cuda_ok(cudaStreamSynchronize(L.s), "copy stream sync");
    for (auto& L : lanes_) L.done_seq = L.seq;
}

uint64_t CudaCopier::pool_addr(int32_t slot)
{
    if (slot < 0 || (size_t) slot >= pool_.size()) fail_tier("pool slot " + std::to_string(slot) + " out of range");
    return pool_[(size_t) slot];
}

uint64_t CudaCopier::staging_addr(int half, int32_t i)
{
    int64_t idx = (int64_t) half * staging_per_half_ + i;
    if (half < 0 || i < 0 || i >= staging_per_half_ || idx >= (int64_t) staging_.size())
        fail_tier("staging slot " + std::to_string(i) + " of half " + std::to_string(half) + " out of range");
    return staging_[(size_t) idx];
}

// ---------------------------------------------------------------------------------------- plan

PlanCopier::PlanCopier(PlanCmd* ring, uint32_t* done_flags, TierCtl* ctl, uint32_t n, std::vector<uint64_t> pool,
                       std::vector<uint64_t> staging, int64_t staging_per_half)
    : ring_(ring), done_(done_flags), ctl_(ctl), n_(n), pool_(std::move(pool)), staging_(std::move(staging)),
      staging_per_half_(staging_per_half)
{
}

void PlanCopier::scan()
{
    while (contig_ < tail_)
    {
        const uint64_t i = contig_;
        if (ring_[i & (n_ - 1)].kind == kPlanEnd || __atomic_load_n(&done_[i & (n_ - 1)], __ATOMIC_ACQUIRE) == (uint32_t) (i + 1))
            ++contig_;
        else break;
    }
}

bool PlanCopier::done(uint64_t token)
{
    if (!token) return true;
    if (token <= contig_) return true;
    scan();
    return token <= contig_;
}

void PlanCopier::wait(uint64_t token)
{
    int64_t t0 = 0;
    while (!done(token))
    {
        if (abort_ && abort_->load()) fail_tier("the tier stopped while a copy was pending");
        if (!t0) t0 = mono_ns();
        else if (mono_ns() - t0 > 120000000000ll) fail_tier("a copy command was not run within 120 s (no fetch kernel ran it)");
        cpu_relax();
    }
}

uint64_t PlanCopier::push(uint64_t src, uint64_t dst, int64_t n, uint64_t after, uint32_t kind, uint32_t seq)
{
    if (kind == kPlanCopy && (n <= 0 || n % 16 || dst % 16 || n >= (1ll << 32)))
        fail_tier("a copy of " + std::to_string(n) + " bytes to an address that is not 16-byte aligned, or not whole "
                  "16-byte units");
    if (after && done(after))
    {
        after = 0;
        ++dep_dropped;
    }
    if (tail_ - contig_ >= n_)
    {
        ++full_waits;
        int64_t t0 = mono_ns();
        while (scan(), tail_ - contig_ >= n_)
        {
            if (abort_ && abort_->load()) fail_tier("the tier stopped while the plan ring was full");
            if (mono_ns() - t0 > 120000000000ll) fail_tier("the plan ring stayed full for 120 s");
            cpu_relax();
        }
    }
    PlanCmd& c = ring_[tail_ & (n_ - 1)];
    c.src = src;
    c.dst = dst;
    c.bytes = (uint32_t) n;
    c.seq = seq;
    c.dep = (uint32_t) after;
    c.kind = kind;
    // the fetch kernel checks a slot's flag against index + 1: a stale one from n commands back never
    // matches, but the host's own scan must not see it either
    __atomic_store_n(&done_[tail_ & (n_ - 1)], 0u, __ATOMIC_RELAXED);
    ++tail_;
    __atomic_store_n(&ctl_->plan_tail, tail_, __ATOMIC_RELEASE);
    ++commands;
    return tail_;
}

uint64_t PlanCopier::h2d(uint64_t dst, const uint8_t* src, int64_t n, uint64_t after)
{
    h2d_bytes += n;
    ++h2d_copies;
    return push(reinterpret_cast<uint64_t>(src), dst, n, after, kPlanCopy, 0);
}

uint64_t PlanCopier::d2h(uint8_t* dst, uint64_t src, int64_t n, uint64_t after)
{
    d2h_bytes += n;
    ++d2h_copies;
    return push(src, reinterpret_cast<uint64_t>(dst), n, after, kPlanCopy, 0);
}

uint64_t PlanCopier::d2d(uint64_t dst, uint64_t src, int64_t n, uint64_t after)
{
    d2d_bytes += n;
    ++d2d_copies;
    return push(src, dst, n, after, kPlanCopy, 0);
}

void PlanCopier::end(uint32_t seq)
{
    push(0, 0, 0, 0, kPlanEnd, seq);
}

uint64_t PlanCopier::pool_addr(int32_t slot)
{
    if (slot < 0 || (size_t) slot >= pool_.size()) fail_tier("pool slot " + std::to_string(slot) + " out of range");
    return pool_[(size_t) slot];
}

uint64_t PlanCopier::staging_addr(int half, int32_t i)
{
    int64_t idx = (int64_t) half * staging_per_half_ + i;
    if (half < 0 || i < 0 || i >= staging_per_half_ || idx >= (int64_t) staging_.size())
        fail_tier("staging slot " + std::to_string(i) + " of half " + std::to_string(half) + " out of range");
    return staging_[(size_t) idx];
}

// ---------------------------------------------------------------------------------------- runtime

TierGpu::TierGpu(const PolicyConfig& pc, int64_t layers, int64_t experts, int pool_slots, int64_t ram_slots, int pins_,
                 const GpuConfig& gc, std::unique_ptr<TierHost> host_or_null, std::unique_ptr<CudaCopier> copier)
    : S(pool_slots), pins(pins_), K((int) (layers * experts)), W((pool_slots + 31) / 32), pc_(pc), gc_(gc),
      host_(std::move(host_or_null)), copier_(std::move(copier))
{
    if (!host_) core_.reset(new TierCore(pc, layers, experts, pool_slots, ram_slots, pins_, false));
    if (experts > kLookupMaxExperts) fail_tier("at most " + std::to_string(kLookupMaxExperts) + " experts per layer");
    if (gc_.projections < 2 || gc_.projections > 3) fail_tier("an expert has 2 or 3 projections");
    if ((int64_t) gc_.pool.size() != (int64_t) S + pins)
        fail_tier(std::to_string(gc_.pool.size()) + " slot addresses for " + std::to_string(S + pins) + " slots");
    if ((int64_t) gc_.tables.size() != layers * gc_.projections)
        fail_tier("one pointer table per layer and projection");
    if (gc_.staging_per_half < 1 || (int64_t) gc_.staging.size() < gc_.staging_per_half)
        fail_tier("the staging area needs a slot per expert a call can miss");
    if (host_ && host_->hcfg.deterministic != gc_.deterministic) fail_tier("host and runtime disagree on determinism");
    if (host_ && host_->core.defer_returns == gc_.deterministic)
        fail_tier("production mode returns demoted slots when their copy lands (defer_returns)");

    DeviceGuard g(gc_.device);
    const size_t n = (size_t) (S + pins);
    auto al = [](size_t x) { return (x + 255) & ~(size_t) 255; };
    size_t off_state = 0;
    size_t off_key = al(off_state + n);
    size_t off_stamp = al(off_key + 4 * n);
    size_t off_addr = al(off_stamp + 4 * n);
    size_t off_slot_of = al(off_addr + 8 * n);
    size_t off_hv = al(off_slot_of + 4 * (size_t) K);
    size_t off_he = al(off_hv + 4 * (size_t) K);
    size_t off_free = al(off_he + 4 * (size_t) K);
    size_t off_dctr = al(off_free + 4 * (size_t) W);
    size_t off_staging = al(off_dctr + 8 * kDcCount);
    size_t off_tables = al(off_staging + 8 * (size_t) gc_.staging_per_half);
    uint32_t pn = 1024;
    while ((int64_t) pn < gc_.plan_n) pn <<= 1;
    plan_n_ = pn;
    size_t off_dring = al(off_tables + 8 * gc_.tables.size());
    size_t off_ddone = al(off_dring + sizeof(FetchCmd) * pn);
    size_t off_dcnt = al(off_ddone + 4 * (size_t) pn);
    size_t total = al(off_dcnt + 4 * (size_t) pn);
    cuda_ok(cudaMalloc(&dev_mem_, total), "expert tier directory");
    cuda_ok(cudaMemset(dev_mem_, 0, total), "expert tier directory");
    uint8_t* b = static_cast<uint8_t*>(dev_mem_);
    dev.state = b + off_state;
    dev.key = reinterpret_cast<int32_t*>(b + off_key);
    dev.stamp = reinterpret_cast<uint32_t*>(b + off_stamp);
    dev.addr = reinterpret_cast<int64_t*>(b + off_addr);
    dev.slot_of = reinterpret_cast<int32_t*>(b + off_slot_of);
    dev.heat_v = reinterpret_cast<uint32_t*>(b + off_hv);
    dev.heat_ep = reinterpret_cast<uint32_t*>(b + off_he);
    dev.free_bits = reinterpret_cast<uint32_t*>(b + off_free);
    dev.dctr = reinterpret_cast<uint64_t*>(b + off_dctr);
    dev.staging = reinterpret_cast<int64_t*>(b + off_staging);
    dev.tables = reinterpret_cast<int64_t*>(b + off_tables);
    dev.dring = reinterpret_cast<FetchCmd*>(b + off_dring);
    dev.ddone = reinterpret_cast<uint32_t*>(b + off_ddone);
    dev.dcnt = reinterpret_cast<uint32_t*>(b + off_dcnt);
    cuda_ok(cudaMemcpy(dev.addr, gc_.pool.data(), 8 * n, cudaMemcpyHostToDevice), "slot addresses");
    cuda_ok(cudaMemcpy(dev.staging, gc_.staging.data(), 8 * (size_t) gc_.staging_per_half, cudaMemcpyHostToDevice),
            "staging addresses");
    cuda_ok(cudaMemcpy(dev.tables, gc_.tables.data(), 8 * gc_.tables.size(), cudaMemcpyHostToDevice), "table addresses");

    uint32_t rn = 1024;
    while (rn < (uint32_t) (S + pins + 1)) rn <<= 1;
    ret_n_ = rn;
    rec_bytes_ = (uint64_t) std::max<int64_t>(1 << 16, gc_.rec_ring_bytes) & ~(uint64_t) 7;
    const size_t off_plan = kCtlBytes + 4 * (size_t) ret_n_ + (size_t) rec_bytes_;
    const size_t ctl_bytes = off_plan + sizeof(PlanCmd) * plan_n_ + 4 * (size_t) plan_n_;
    void* hp = nullptr;
    cuda_ok(cudaHostAlloc(&hp, ctl_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "expert tier control page");
    std::memset(hp, 0, ctl_bytes);
    ctl_mem_ = static_cast<uint8_t*>(hp);
    void* dp = nullptr;
    cuda_ok(cudaHostGetDevicePointer(&dp, hp, 0), "expert tier control page (device address)");
    ctl_ = reinterpret_cast<TierCtl*>(ctl_mem_);
    ctl_dev_ = reinterpret_cast<TierCtl*>(dp);
    ret_ring_ = reinterpret_cast<uint32_t*>(ctl_mem_ + kCtlBytes);
    rec_ring_ = ctl_mem_ + kCtlBytes + 4 * (size_t) ret_n_;
    plan_ring_ = reinterpret_cast<PlanCmd*>(ctl_mem_ + off_plan);
    plan_done_ = reinterpret_cast<uint32_t*>(ctl_mem_ + off_plan + sizeof(PlanCmd) * plan_n_);

    kp_.state = dev.state;
    kp_.key = dev.key;
    kp_.stamp = dev.stamp;
    kp_.addr = dev.addr;
    kp_.slot_of = dev.slot_of;
    kp_.heat_v = dev.heat_v;
    kp_.heat_ep = dev.heat_ep;
    kp_.free_bits = dev.free_bits;
    kp_.dctr = dev.dctr;
    kp_.staging = dev.staging;
    kp_.tables = dev.tables;
    for (int q = 0; q < 3; ++q) kp_.proj_off[q] = gc_.proj_off[q];
    kp_.ctl = ctl_dev_;
    kp_.ret_ring = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(dp) + kCtlBytes);
    kp_.rec_ring = static_cast<uint8_t*>(dp) + kCtlBytes + 4 * (size_t) ret_n_;
    kp_.rec_bytes = rec_bytes_;
    kp_.ret_n = ret_n_;
    kp_.P = gc_.projections;
    kp_.S = S;
    kp_.pins = pins;
    kp_.spare = pc.spare;
    kp_.nstage = (int) gc_.staging_per_half;
    kp_.E = (int) experts;
    kp_.L = (int) layers;
    kp_.admit = pc.admit;
    kp_.evict = pc.evict;
    kp_.seed = pc.seed;
    kp_.halflife = pc.halflife;
    kp_.prefill_inc = pc.prefill_inc;
    kp_.deterministic = gc_.deterministic ? 1 : 0;

    int sms = 0;
    cuda_ok(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gc_.device), "device attributes");
    int coop = 0;
    cuda_ok(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, gc_.device), "device attributes");
    if (!coop) fail_tier("the device cannot launch cooperative kernels (the fetch kernel needs it)");
    fp_.ctl = ctl_dev_;
    fp_.plan = reinterpret_cast<const PlanCmd*>(static_cast<uint8_t*>(dp) + off_plan);
    fp_.done_host = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(dp) + off_plan + sizeof(PlanCmd) * plan_n_);
    fp_.dring = dev.dring;
    fp_.ddone = dev.ddone;
    fp_.dcnt = dev.dcnt;
    fp_.dctr = dev.dctr;
    fp_.plan_n = plan_n_;
    fp_.workers = gc_.workers > 0 ? gc_.workers : std::max(1, std::min(128, sms / 2));
    fp_.deterministic = gc_.deterministic ? 1 : 0;
    plan_.reset(new PlanCopier(plan_ring_, plan_done_, ctl_, plan_n_, gc_.pool, gc_.staging, gc_.staging_per_half));
    plan_->set_abort(&failed_);
    publish_p();
    upload();
}

TierGpu::~TierGpu()
{
    DeviceGuard g(gc_.device);
    if (ctl_) __atomic_store_n(&ctl_->abort, 1u, __ATOMIC_RELEASE);
    failed_ = true;
    stop();
    cudaDeviceSynchronize();
    if (copier_) copier_->sync();
    if (copier_) copier_->unpin_all();              // before the host unmaps the RAM tier
    host_.reset();
    copier_.reset();
    if (dev_mem_) cudaFree(dev_mem_);
    if (ctl_mem_) cudaFreeHost(ctl_mem_);
}

void TierGpu::publish_p()
{
    uint64_t q = core().adapt.q32();
    if (q != last_p_)
    {
        __atomic_store_n(&ctl_->p_q32, q, __ATOMIC_RELEASE);
        last_p_ = q;
    }
}

void TierGpu::upload()
{
    DeviceGuard g(gc_.device);
    TierCore& c = core();
    const VramDirectory& v = c.vram;
    const size_t n = (size_t) (S + pins);
    std::vector<uint32_t> freeb((size_t) W, 0);
    for (int32_t s : v.free) freeb[(size_t) (s >> 5)] |= 1u << (s & 31);
    uint64_t dctr[kDcCount] = {};
    dctr[kDcAccess] = v.access;
    dctr[kDcOccupied] = (uint64_t) v.occupied;
    dctr[kDcRetHead] = ret_tail_;
    dctr[kDcRecHead] = __atomic_load_n(&ctl_->rec_head, __ATOMIC_ACQUIRE);
    cuda_ok(cudaMemcpy(dev.state, v.state.data(), n, cudaMemcpyHostToDevice), "upload");
    cuda_ok(cudaMemcpy(dev.key, v.key.data(), 4 * n, cudaMemcpyHostToDevice), "upload");
    cuda_ok(cudaMemcpy(dev.stamp, v.stamp.data(), 4 * n, cudaMemcpyHostToDevice), "upload");
    cuda_ok(cudaMemcpy(dev.slot_of, v.slot_of.data(), 4 * (size_t) K, cudaMemcpyHostToDevice), "upload");
    cuda_ok(cudaMemcpy(dev.heat_v, c.heat.v.data(), 4 * (size_t) K, cudaMemcpyHostToDevice), "upload");
    cuda_ok(cudaMemcpy(dev.heat_ep, c.heat.ep.data(), 4 * (size_t) K, cudaMemcpyHostToDevice), "upload");
    cuda_ok(cudaMemcpy(dev.free_bits, freeb.data(), 4 * (size_t) W, cudaMemcpyHostToDevice), "upload");
    cuda_ok(cudaMemcpy(dev.dctr, dctr, sizeof(dctr), cudaMemcpyHostToDevice), "upload");
    publish_p();
}

void TierGpu::cold_fill(const std::vector<int32_t>& order, const std::vector<int32_t>& pin_keys)
{
    if (running_) fail_tier("cold fill while the tier thread runs");
    DeviceGuard g(gc_.device);
    if (host_)
    {
        // the copy engines for the load; the fetch kernel's plan from the first call on
        host_->set_copier(copier_.get());
        host_->cold_fill(order, pin_keys);
        copier_->sync();
        host_->set_copier(plan_.get());
    }
    else
    {
        if ((int64_t) pin_keys.size() > pins) fail_tier("more hot pins than pin slots");
        for (size_t i = 0; i < pin_keys.size(); ++i) core_->vram.place(pin_keys[i], S + (int32_t) i, 0);
        core_->cold_fill(order, nullptr, nullptr);
    }
    upload();
}

void TierGpu::lookup(int lc, const int64_t* ids, int n, int mode, uint32_t seq, uint64_t tokens, cudaStream_t stream)
{
    check_ok();
    DeviceGuard g(gc_.device);
    last_seq_ = seq;
    kicks_.fetch_add(1, std::memory_order_relaxed);
    if (running_) cv_.notify_one();
    tier_lookup_launch(kp_, ids, n, lc, mode, seq, tokens, stream);
    cuda_ok(tier_fetch_launch(fp_, seq, stream), "launching the expert cache's fetch kernel");
}

bool TierGpu::next_record(Record& rec, bool& needs_host, uint64_t& tokens)
{
    const uint64_t head = __atomic_load_n(&ctl_->rec_head, __ATOMIC_ACQUIRE);
    if (rec_tail_ >= head) return false;
    uint64_t off = rec_tail_ % rec_bytes_;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(rec_ring_ + off);
    if (w[0] == kRecWrap)
    {
        rec_tail_ += rec_bytes_ - off;
        if (rec_tail_ >= head) return false;
        off = 0;
        w = reinterpret_cast<const uint32_t*>(rec_ring_);
    }
    const uint32_t nbytes = w[0];
    const uint32_t n = w[4];
    if (nbytes != (uint32_t) (kRecHeaderWords + kRecEntryWords * n) * 4u || off + nbytes > rec_bytes_)
        fail_tier("a corrupt call record in the ring (" + std::to_string(nbytes) + " bytes, " + std::to_string(n) +
                  " entries)");
    rec.seq = w[1];
    rec.lc = (int32_t) w[2];
    rec.mode = (int) (w[3] & 0xFF);
    needs_host = (w[3] >> 8) & 1;
    tokens = (uint64_t) w[5] | ((uint64_t) w[6] << 32);
    rec.entries.resize(n);
    for (uint32_t j = 0; j < n; ++j)
    {
        const uint32_t* x = w + kRecHeaderWords + j * kRecEntryWords;
        Entry& e = rec.entries[j];
        e.key = (int32_t) x[0];
        e.kind = (uint8_t) (x[1] & 0xFF);
        e.inplace = (uint8_t) ((x[1] >> 8) & 1);
        e.starved = (uint8_t) ((x[1] >> 9) & 1);
        e.cnt = x[2];
        e.dst = (int32_t) x[3];
        e.heat = x[4];
        e.vkey = (int32_t) x[5];
        e.vslot = (int32_t) x[6];
        e.vstamp = x[7];
        e.vheat = x[8];
    }
    rec_tail_ += nbytes;
    __atomic_store_n(&ctl_->rec_tail, rec_tail_, __ATOMIC_RELEASE);
    return true;
}

void TierGpu::push_returns(const std::vector<int32_t>& slots)
{
    for (int32_t s : slots)
    {
        ret_ring_[ret_tail_ & (ret_n_ - 1)] = (uint32_t) s;
        ++ret_tail_;
    }
    __atomic_store_n(&ctl_->ret_tail, ret_tail_, __ATOMIC_RELEASE);
    stats.returns += (int64_t) slots.size();
}

void TierGpu::handle(const Record& rec, bool needs_host, uint64_t tokens)
{
    int64_t t0 = mono_ns();
    TierCore& c = core();
    c.heat.tokens = tokens;
    std::vector<int32_t> returns;
    if (host_)
    {
        host_->process(rec);
        for (const Action& a : host_->last_actions())
            if (a.op == kActReturn) returns.push_back(a.a);
    }
    else
    {
        std::vector<Action> acts;
        core_->process(rec, acts);
        for (const Action& a : acts)
            if (a.op == kActReturn) returns.push_back(a.a);
    }
    if (!returns.empty()) push_returns(returns);
    publish_p();
    // the call's fetch runs the commands above until this (an all-hit call's fetch does not wait,
    // except in deterministic mode)
    if (gc_.deterministic || needs_host) plan_->end(rec.seq);
    ++stats.records;
    if (needs_host) ++stats.host_records;
    if (gc_.trace > 0)
    {
        std::lock_guard<std::mutex> lk(trace_mu);
        trace.push_back(rec);
        while ((int64_t) trace.size() > gc_.trace) trace.pop_front();
    }
    stats.work_ns += mono_ns() - t0;
}

int TierGpu::step(bool wait)
{
    if (running_) fail_tier("step() while the tier thread runs");
    std::lock_guard<std::mutex> lk(step_mu_);
    DeviceGuard g(gc_.device);
    int done = 0;
    Record rec;
    bool nh;
    uint64_t tk;
    int64_t t0 = mono_ns();
    while (true)
    {
        if (next_record(rec, nh, tk))
        {
            try { handle(rec, nh, tk); }
            catch (const std::exception& e) { fail(e.what()); throw; }
            ++done;
            continue;
        }
        if (host_ && !gc_.deterministic)
        {
            auto r = host_->return_landed();
            if (!r.empty()) push_returns(r);
        }
        if (done || !wait) break;
        if (mono_ns() - t0 > 30000000000ll) fail_tier("no call record arrived within 30 s");
        cpu_relax();
    }
    return done;
}

void TierGpu::loop()
{
    cudaSetDevice(gc_.device);
#if defined(__linux__)
    if (!gc_.affinity.empty())
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : gc_.affinity) CPU_SET(c, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
    const int64_t spin_ns = gc_.spin_us < 0 ? 2000000 : gc_.spin_us * 1000;
    int64_t last = mono_ns();
    uint64_t seen_kicks = kicks_.load();
    Record rec;
    bool nh;
    uint64_t tk;
    try
    {
        while (!quit_.load(std::memory_order_relaxed))
        {
            bool did = false;
            int64_t lag = 0;
            while (next_record(rec, nh, tk))
            {
                if (!did)
                {
                    uint64_t head = __atomic_load_n(&ctl_->rec_head, __ATOMIC_ACQUIRE);
                    lag = (int64_t) (head - rec_tail_);
                }
                handle(rec, nh, tk);
                did = true;
            }
            if (lag > stats.max_lag) stats.max_lag = lag;
            bool busy = false;
            if (host_ && !gc_.deterministic)
            {
                auto r = host_->return_landed();
                if (!r.empty())
                {
                    push_returns(r);
                    did = true;
                }
                busy = !host_->core.pending.empty();
            }
            if (drain_req_.load())
            {
                while (next_record(rec, nh, tk)) handle(rec, nh, tk);
                if (host_)
                {
                    host_->drain();
                    copier_->sync();
                    if (!gc_.deterministic)
                    {
                        auto r = host_->return_landed();
                        if (!r.empty()) push_returns(r);
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    drain_req_ = false;
                    drained_ = true;
                }
                drained_cv_.notify_all();
                did = true;
            }
            int64_t now = mono_ns();
            uint64_t k = kicks_.load(std::memory_order_relaxed);
            if (did || busy || k != seen_kicks)
            {
                if (did || k != seen_kicks) last = now;
                seen_kicks = k;
                cpu_relax();
                continue;
            }
            if (gc_.spin_us != 0 && now - last < spin_ns)
            {
                cpu_relax();
                continue;
            }
            std::unique_lock<std::mutex> lk(mu_);
            ++stats.idle_sleeps;
            cv_.wait_for(lk, std::chrono::microseconds(gc_.spin_us == 0 ? 50 : 500), [&]
            {
                return quit_.load() || drain_req_.load() || kicks_.load(std::memory_order_relaxed) != seen_kicks;
            });
        }
    }
    catch (const std::exception& e)
    {
        fail(e.what());
        std::lock_guard<std::mutex> lk(mu_);
        drain_req_ = false;
        drained_ = true;
        drained_cv_.notify_all();
    }
}

void TierGpu::start()
{
    if (running_) return;
    quit_ = false;
    running_ = true;
    th_ = std::thread([this] { loop(); });
}

void TierGpu::stop()
{
    if (!running_) return;
    quit_ = true;
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    running_ = false;
}

void TierGpu::drain()
{
    DeviceGuard g(gc_.device);
    if (running_)
    {
        std::unique_lock<std::mutex> lk(mu_);
        drained_ = false;
        drain_req_ = true;
        cv_.notify_all();
        drained_cv_.wait(lk, [&] { return drained_ || failed_.load(); });
    }
    else
    {
        step(false);
        if (host_)
        {
            host_->drain();
            copier_->sync();
            if (!gc_.deterministic)
            {
                auto r = host_->return_landed();
                if (!r.empty()) push_returns(r);
            }
        }
    }
    check_ok();
}

void TierGpu::fail(const std::string& what)
{
    {
        std::lock_guard<std::mutex> lk(err_mu_);
        if (failed_.load()) return;
        error_ = what;
        failed_ = true;
    }
    __atomic_store_n(&ctl_->abort, 1u, __ATOMIC_RELEASE);
}

uint32_t TierGpu::device_error() const
{
    return __atomic_load_n(&ctl_->error, __ATOMIC_ACQUIRE);
}

uint32_t TierGpu::overflow() const
{
    return __atomic_load_n(&ctl_->overflow, __ATOMIC_ACQUIRE);
}

std::string TierGpu::error() const
{
    std::lock_guard<std::mutex> lk(err_mu_);
    return error_;
}

void TierGpu::check_ok()
{
    if (failed_.load()) throw TierError(error());
    uint32_t e = device_error();
    if (e)
    {
        fail(std::string("expert tier: lookup kernel: ") + dev_error_text(e));
        throw TierError(error());
    }
}

void TierGpu::verify()
{
    DeviceGuard g(gc_.device);
    cuda_ok(cudaDeviceSynchronize(), "verify");
    const size_t n = (size_t) (S + pins);
    std::vector<uint8_t> st(n);
    std::vector<int32_t> key(n), slot_of((size_t) K);
    std::vector<uint32_t> stamp(n), hv((size_t) K), he((size_t) K), fb((size_t) W);
    uint64_t dctr[kDcCount];
    cuda_ok(cudaMemcpy(st.data(), dev.state, n, cudaMemcpyDeviceToHost), "verify");
    cuda_ok(cudaMemcpy(key.data(), dev.key, 4 * n, cudaMemcpyDeviceToHost), "verify");
    cuda_ok(cudaMemcpy(stamp.data(), dev.stamp, 4 * n, cudaMemcpyDeviceToHost), "verify");
    cuda_ok(cudaMemcpy(slot_of.data(), dev.slot_of, 4 * (size_t) K, cudaMemcpyDeviceToHost), "verify");
    cuda_ok(cudaMemcpy(hv.data(), dev.heat_v, 4 * (size_t) K, cudaMemcpyDeviceToHost), "verify");
    cuda_ok(cudaMemcpy(he.data(), dev.heat_ep, 4 * (size_t) K, cudaMemcpyDeviceToHost), "verify");
    cuda_ok(cudaMemcpy(fb.data(), dev.free_bits, 4 * (size_t) W, cudaMemcpyDeviceToHost), "verify");
    cuda_ok(cudaMemcpy(dctr, dev.dctr, sizeof(dctr), cudaMemcpyDeviceToHost), "verify");
    // returns the device has not harvested yet count as done
    for (uint32_t i = (uint32_t) dctr[kDcRetHead]; i != ret_tail_; ++i)
    {
        int32_t s = (int32_t) ret_ring_[i & (ret_n_ - 1)];
        st[(size_t) s] = kFree;
        key[(size_t) s] = -1;
        stamp[(size_t) s] = 0;
        fb[(size_t) (s >> 5)] |= 1u << (s & 31);
    }
    TierCore& c = core();
    const VramDirectory& v = c.vram;
    auto bad = [](const std::string& m) { fail_tier("verify: the device directory and the host mirror differ: " + m); };
    for (size_t s = 0; s < n; ++s)
    {
        if (st[s] != v.state[s]) bad("slot " + std::to_string(s) + " state " + std::to_string(st[s]) + " vs " +
                                     std::to_string(v.state[s]));
        if (key[s] != v.key[s]) bad("slot " + std::to_string(s) + " key " + std::to_string(key[s]) + " vs " +
                                    std::to_string(v.key[s]));
        if (stamp[s] != v.stamp[s]) bad("slot " + std::to_string(s) + " stamp " + std::to_string(stamp[s]) + " vs " +
                                        std::to_string(v.stamp[s]));
        if (s < (size_t) S && (bool) ((fb[s >> 5] >> (s & 31)) & 1) != (v.state[s] == kFree))
            bad("slot " + std::to_string(s) + " free bit");
    }
    for (size_t k = 0; k < (size_t) K; ++k)
    {
        if (slot_of[k] != v.slot_of[k]) bad("key " + std::to_string(k) + " slot " + std::to_string(slot_of[k]) +
                                            " vs " + std::to_string(v.slot_of[k]));
        if (hv[k] != c.heat.v[k] || he[k] != c.heat.ep[k]) bad("key " + std::to_string(k) + " heat");
    }
    if (dctr[kDcOccupied] != (uint64_t) v.occupied) bad("occupied");
    if (dctr[kDcAccess] != v.access) bad("access");
}

}  // namespace exl3_tier
