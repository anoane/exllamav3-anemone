// The expert tier's host side; see tier_host.h.

#include "tier_host.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <thread>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace exl3_tier
{

namespace
{

[[noreturn]] void fail(const std::string& m) { throw TierError("expert tier: " + m); }

int64_t round_up(int64_t n, int64_t a) { return (n + a - 1) / a * a; }

int64_t mono_ns()
{
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string errtext(int e)
{
    return std::string(std::strerror(e)) + " (errno " + std::to_string(e) + ")";
}

}  // namespace

// ---------------------------------------------------------------------------------------- copier

HostCopier::HostCopier(int64_t pool_slots, int64_t staging_per_half, int64_t slot_bytes)
    : mem_((size_t) ((pool_slots + 2 * staging_per_half) * slot_bytes), 0), pool_slots_(pool_slots),
      staging_(staging_per_half), slot_(slot_bytes)
{
}

uint64_t HostCopier::h2d(uint64_t dst, const uint8_t* src, int64_t n, uint64_t)
{
    std::memcpy(reinterpret_cast<void*>(dst), src, (size_t) n);
    h2d_bytes += n;
    ++h2d_copies;
    return 0;
}

uint64_t HostCopier::d2h(uint8_t* dst, uint64_t src, int64_t n, uint64_t)
{
    std::memcpy(dst, reinterpret_cast<const void*>(src), (size_t) n);
    d2h_bytes += n;
    ++d2h_copies;
    return 0;
}

uint64_t HostCopier::d2d(uint64_t dst, uint64_t src, int64_t n, uint64_t)
{
    std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), (size_t) n);
    d2d_bytes += n;
    ++d2d_copies;
    return 0;
}

uint64_t HostCopier::pool_addr(int32_t slot)
{
    if (slot < 0 || slot >= pool_slots_) fail("pool slot " + std::to_string(slot) + " out of range");
    return reinterpret_cast<uint64_t>(mem_.data() + (size_t) slot * (size_t) slot_);
}

uint64_t HostCopier::staging_addr(int half, int32_t i)
{
    if (half < 0 || half > 1 || i < 0 || i >= staging_) fail("staging slot " + std::to_string(i) + " out of range");
    return reinterpret_cast<uint64_t>(mem_.data() + (size_t) (pool_slots_ + half * staging_ + i) * (size_t) slot_);
}

const uint8_t* HostCopier::pool(int32_t slot) const
{
    return mem_.data() + (size_t) slot * (size_t) slot_;
}

const uint8_t* HostCopier::staging(int half, int32_t i) const
{
    return mem_.data() + (size_t) (pool_slots_ + half * staging_ + i) * (size_t) slot_;
}

// ---------------------------------------------------------------------------------------- arena

#if defined(__linux__)

// Fault a range in (write): MADV_POPULATE_WRITE (Linux 5.14), else one byte per page
static void prefault(uint8_t* p, size_t n)
{
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
    if (::madvise(p, n, MADV_POPULATE_WRITE) == 0) return;
    for (size_t i = 0; i < n; i += 4096) *(volatile uint8_t*) (p + i) = 0;
}

Arena::Arena(int64_t slots_, int64_t slot_bytes_, int64_t chunk_bytes, bool hugepage, exl3_disk::Engine* eng,
             int prefault_threads)
    : slots(slots_), slot_bytes(slot_bytes_), chunk_slots(std::max<int64_t>(1, chunk_bytes / slot_bytes_)), eng_(eng)
{
    if (slot_bytes <= 0 || slot_bytes % 4096) fail("slot size " + std::to_string(slot_bytes) + " is not whole pages");
    try
    {
        for (int64_t first = 0; first < slots; first += chunk_slots)
        {
            size_t n = (size_t) (std::min(chunk_slots, slots - first) * slot_bytes);
            void* p = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED)
                fail("mapping a RAM tier chunk of " + std::to_string(n) + " bytes failed: " + errtext(errno));
            base.push_back(static_cast<uint8_t*>(p));
            bytes.push_back(n);
            reg.push_back(-1);
#ifdef MADV_HUGEPAGE
            if (hugepage) (void) ::madvise(p, n, MADV_HUGEPAGE);
#else
            (void) hugepage;
#endif
        }
        // Registering a buffer with io_uring pins its pages, faulting in and zeroing whatever is not
        // there yet, one chunk after the other; faulting them in first on several threads is
        // several times faster (the memory is the tier's budget: every slot is filled at load)
        if (prefault_threads > 0 && !base.empty())
        {
            std::atomic<size_t> next { 0 };
            std::vector<std::thread> th;
            int nt = (int) std::min<size_t>((size_t) prefault_threads, base.size());
            for (int t = 0; t < nt; ++t)
                th.emplace_back([&]
                {
                    for (size_t i = next++; i < base.size(); i = next++) prefault(base[i], bytes[i]);
                });
            for (auto& x : th) x.join();
        }
        for (size_t i = 0; i < base.size(); ++i)
            if (eng_ && bytes[i] <= ((size_t) 1 << 30))
            {
                try { reg[i] = eng_->register_buffer(base[i], bytes[i]); }
                catch (const exl3_disk::Error&) { reg[i] = -1; }
            }
    }
    catch (...)
    {
        release();
        throw;
    }
}

Arena::~Arena()
{
    release();
}

void Arena::release()
{
    for (size_t i = 0; i < base.size(); ++i)
    {
        if (reg[i] >= 0 && eng_)
        {
            try { eng_->unregister_buffer(reg[i]); }
            catch (const exl3_disk::Error&) {}
        }
        if (base[i]) ::munmap(base[i], bytes[i]);
        base[i] = nullptr;
        reg[i] = -1;
    }
    base.clear();
    bytes.clear();
    reg.clear();
}

#else

Arena::Arena(int64_t, int64_t, int64_t, bool, exl3_disk::Engine*, int) { fail("the RAM tier needs Linux"); }
Arena::~Arena() {}
void Arena::release() {}

#endif

uint8_t* Arena::slot(int64_t i) const
{
    if (i < 0 || i >= slots) fail("arena slot " + std::to_string(i) + " out of range");
    return base[(size_t) (i / chunk_slots)] + (size_t) ((i % chunk_slots) * slot_bytes);
}

// ---------------------------------------------------------------------------------------- host

TierHost::TierHost(const PolicyConfig& pc, int64_t layers, int64_t experts, int pool_slots, int64_t ram_slots, int pins,
                   const Geometry& g, const std::vector<std::string>& files, const std::vector<ExpertExtent>& extents,
                   const HostConfig& hc, Copier* copier, bool defer_returns)
    : core(pc, layers, experts, pool_slots, ram_slots, pins, defer_returns), geo(g), hcfg(hc), copier_(copier),
      files_(files), ext_(extents)
{
#if !defined(__linux__)
    fail("the RAM tier needs the disk engine, which is Linux-only");
#else
    if (!copier_) fail("no copier");
    if (pc.spare == 0 && (pc.demotes() || pc.disk_off))
        fail("spare=0 needs ram demote=off or policy=inclusive: an admission that replaces its victim in place "
             "cannot copy the victim back into the RAM slot its own copy reads from");
    if (geo.projections < 2 || geo.projections > 3) fail("an expert has 2 or 3 projections");
    int64_t acc = 0;
    for (int p = 0; p < geo.projections; ++p)
    {
        if (geo.proj_bytes[p] <= 0) fail("empty projection");
        geo.proj_off[p] = acc;
        acc += geo.proj_bytes[p];
    }
    geo.vram_slot_bytes = acc;
    if (geo.ram_slot_bytes <= 0 || geo.ram_slot_bytes % 4096) fail("the RAM slot size must be whole 4 KiB pages");
    if ((int64_t) ext_.size() != core.K)
        fail(std::to_string(ext_.size()) + " extents for " + std::to_string(core.K) + " cached experts");
    if (hcfg.staging_slots <= 0) hcfg.staging_slots = (int) experts;
    eng_ = exl3_disk::default_engine();
    for (const std::string& f : files_)
    {
        int fd = ::open(f.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
        {
            int e = errno;
            for (int x : fds_) ::close(x);
            fail("opening " + f + ": " + errtext(e));
        }
        fds_.push_back(fd);
    }
    try
    {
        for (size_t k = 0; k < ext_.size(); ++k)
        {
            ExpertExtent& x = ext_[k];
            if (x.npieces < 1 || x.npieces > 3) fail(x.name + ": 1 to 3 pieces");
            int64_t at = 0;
            for (int i = 0; i < x.npieces; ++i)
            {
                if (x.file[i] < 0 || x.file[i] >= (int32_t) fds_.size()) fail(x.name + ": bad file index");
                exl3_disk::ExtentGeom eg = eng_->extent_geometry(fds_[(size_t) x.file[i]], x.offset[i], x.length[i]);
                x.base[i] = round_up(at, 4096);
                x.payload[i] = eg.payload_offset;
                x.span[i] = eg.span_bytes;
                at = x.base[i] + eg.span_bytes;
            }
            if (at > geo.ram_slot_bytes)
                fail(x.name + " needs " + std::to_string(at) + " bytes in a RAM slot, more than its " +
                     std::to_string(geo.ram_slot_bytes) + " (the O_DIRECT alignment of " +
                     files_[(size_t) x.file[0]] + " is above 4 KiB)");
            for (int p = 0; p < geo.projections; ++p)
            {
                int tp = x.tpiece[p];
                if (tp < 0 || tp >= x.npieces || x.trel[p] < 0 || x.trel[p] + geo.proj_bytes[p] > x.length[tp])
                    fail(x.name + ": projection " + std::to_string(p) + " lies outside its piece");
            }
        }
        int64_t t0 = mono_ns();
        arena_.reset(new Arena(core.ram.R, geo.ram_slot_bytes, hcfg.chunk_bytes, hcfg.hugepage, eng_.get(),
                               hcfg.prefault_threads));
        slab_.reset(new Arena(hcfg.slab_slots, geo.ram_slot_bytes, hcfg.chunk_bytes, hcfg.hugepage, eng_.get(),
                              hcfg.prefault_threads));
        stats.arena_ns = mono_ns() - t0;
    }
    catch (...)
    {
        for (int x : fds_) ::close(x);
        fds_.clear();
        throw;
    }
    ram_io_.resize((size_t) core.ram.R);
    slab_io_.resize((size_t) hcfg.slab_slots);
    slab_state_.resize((size_t) hcfg.slab_slots);
    for (int32_t i = hcfg.slab_slots - 1; i >= 0; --i) slab_free_.push_back(i);
    pool_busy_.assign((size_t) (core.vram.S + core.vram.pins), 0);
#endif
}

TierHost::~TierHost()
{
    // every read still in flight lands (or is cancelled) before its memory goes away
    for (auto& s : slab_io_)
        if (s.ticket) eng_->release(s.ticket);
    for (auto& s : ram_io_)
        if (s.ticket) eng_->release(s.ticket);
    slab_.reset();
    arena_.reset();
#if defined(__linux__)
    for (int fd : fds_) ::close(fd);
#endif
}

std::string TierHost::where(int32_t key) const
{
    return "layer " + std::to_string(key / core.E) + " expert " + std::to_string(key % core.E) + " (" +
           ext_[(size_t) key].name + ")";
}

void TierHost::fail_read(int32_t key, int err, const std::string& what)
{
    if (!error_.empty()) return;
    const ExpertExtent& x = ext_[(size_t) key];
    error_ = "expert tier: reading " + where(key) + " from " + files_[(size_t) x.file[0]] + " at offset " +
             std::to_string(x.offset[0]) + " failed: " + what + ": " + errtext(err);
}

void TierHost::throw_if_failed()
{
    if (!error_.empty()) throw TierError(error_);
}

void TierHost::pread_extent(int32_t key, uint8_t* slot)
{
#if defined(__linux__)
    const ExpertExtent& x = ext_[(size_t) key];
    ++stats.pread_fallbacks;
    for (int i = 0; i < x.npieces; ++i)
    {
        uint8_t* dst = slot + x.base[i] + x.payload[i];
        int64_t off = x.offset[i], left = x.length[i];
        while (left > 0)
        {
            ssize_t r = ::pread(fds_[(size_t) x.file[i]], dst, (size_t) left, (off_t) off);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0)
            {
                fail_read(key, r < 0 ? errno : EIO, r < 0 ? "the disk engine's read and a plain pread"
                                                          : "the disk engine's read, and a plain pread met the end of the file");
                return;
            }
            dst += r;
            off += r;
            left -= r;
        }
    }
#else
    (void) key; (void) slot;
#endif
}

void TierHost::read_into(int32_t key, uint8_t* slot, int cls, int64_t deadline, exl3_disk::TicketId* ticket)
{
    const ExpertExtent& x = ext_[(size_t) key];
    exl3_disk::ExtentReq req[3];
    int64_t pay[3] = { 0, 0, 0 };
    for (int i = 0; i < x.npieces; ++i)
    {
        req[i].fd = fds_[(size_t) x.file[i]];
        req[i].offset = x.offset[i];
        req[i].length = x.length[i];
        req[i].slot = slot + x.base[i];
        req[i].slot_bytes = x.span[i];
        stats.ssd_bytes += x.length[i];
    }
    ++stats.ssd_reads;
    exl3_disk::Options o;
    o.cls = cls;
    o.deadline_ns = deadline;
    *ticket = 0;
    try
    {
        *ticket = eng_->submit_extents(req, x.npieces, o, pay);
    }
    catch (const exl3_disk::Error&)
    {
        pread_extent(key, slot);            // the file changed, or the engine refused: read it plainly
        return;
    }
    for (int i = 0; i < x.npieces; ++i)
        if (pay[i] != x.payload[i]) fail(x.name + ": the disk engine placed a piece at another offset");
}

void TierHost::finish_read(int32_t key, uint8_t* slot, exl3_disk::TicketId t)
{
    if (!t) return;
    int rc = eng_->wait(t);
    eng_->release(t);
    if (rc != 0) pread_extent(key, slot);
}

void TierHost::ram_ready_for_read(int32_t r)
{
    SlotIo& io = ram_io_[(size_t) r];
    if (io.ticket)
    {
        finish_read(io.ticket_key, arena_->slot(r), io.ticket);
        io.ticket = 0;
        io.ticket_key = -1;
    }
}

void TierHost::ram_ready_for_write(int32_t r)
{
    SlotIo& io = ram_io_[(size_t) r];
    bool waited = false;
    if (io.ticket)
    {
        finish_read(io.ticket_key, arena_->slot(r), io.ticket);
        io.ticket = 0;
        io.ticket_key = -1;
        waited = true;
    }
    if (io.read_tok)
    {
        if (!copier_->done(io.read_tok)) waited = true;
        copier_->wait(io.read_tok);
        io.read_tok = 0;
    }
    if (io.write_tok)
    {
        if (!copier_->done(io.write_tok)) waited = true;
        copier_->wait(io.write_tok);
        io.write_tok = 0;
    }
    if (waited) ++stats.hazard_waits;
}

int64_t TierHost::ram_slot_offset(int32_t r, int p) const
{
    const SlotIo& io = ram_io_[(size_t) r];
    if (io.layout == 2) return geo.proj_off[p];
    if (io.layout != 1) fail("RAM slot " + std::to_string(r) + " is empty");
    const ExpertExtent& x = ext_[(size_t) io.layout_key];
    int tp = x.tpiece[p];
    return x.base[tp] + x.payload[tp] + x.trel[p];
}

void TierHost::ram_trellis(int32_t r, uint8_t* out) const
{
    for (int p = 0; p < geo.projections; ++p)
        std::memcpy(out + geo.proj_off[p], arena_->slot(r) + ram_slot_offset(r, p), (size_t) geo.proj_bytes[p]);
}

uint64_t TierHost::dst_addr(int kind, int32_t dst, int half)
{
    return kind == kTransient ? copier_->staging_addr(half, dst) : copier_->pool_addr(dst);
}

// Copy the trellis of RAM slot r (extent or compact layout) to a device slot
void TierHost::promote_from_ram(int32_t r, uint64_t dst, uint64_t after, bool staging)
{
    ram_ready_for_read(r);
    SlotIo& io = ram_io_[(size_t) r];
    if (io.write_tok)
    {
        if (after) copier_->wait(after);
        after = io.write_tok;
    }
    uint64_t tok = 0;
    const uint8_t* s = arena_->slot(r);
    if (io.layout == 2) tok = copier_->h2d(dst, s, geo.vram_slot_bytes, after);
    else
        for (int p = 0; p < geo.projections; ++p)
            tok = copier_->h2d(dst + (uint64_t) geo.proj_off[p], s + ram_slot_offset(r, p), geo.proj_bytes[p], after);
    io.read_tok = tok;
    if (staging) ++stats.staged;
    else ++stats.h2d;
}

int32_t TierHost::slab_of(int32_t key) const
{
    for (size_t i = 0; i < slab_state_.size(); ++i)
        if (slab_state_[i].key == key) return (int32_t) i;
    return -1;
}

int32_t TierHost::slab_take()
{
    if (slab_free_.empty())
    {
        // every slot holds a prefetch: a demand read goes first, the prefetch farthest ahead yields
        int32_t best = -1;
        for (size_t i = 0; i < slab_state_.size(); ++i)
            if (slab_state_[i].prefetch && (best < 0 || slab_state_[i].lc > slab_state_[(size_t) best].lc)) best = (int32_t) i;
        if (best < 0) fail("the disk slab has no free slot");
        ++stats.slab_waits;
        ++stats.prefetch_dropped;
        slab_release(best);
    }
    int32_t i = slab_free_.back();
    slab_free_.pop_back();
    return i;
}

void TierHost::slab_release(int32_t i)
{
    SlotIo& io = slab_io_[(size_t) i];
    if (io.ticket) eng_->release(io.ticket);        // cancels what is queued, waits for reads in flight
    if (io.read_tok) copier_->wait(io.read_tok);
    io = SlotIo();
    slab_state_[(size_t) i] = Slab();
    slab_free_.push_back(i);
}

// An SSD read that bypasses the RAM tier (a prefetched slab slot when there is one), then its copy
void TierHost::stage_from_ssd(int32_t key, uint64_t dst, uint64_t after, bool staging, int cls)
{
    int32_t i = slab_of(key);
    if (i >= 0)
    {
        ++stats.prefetch_used;
        if (slab_io_[(size_t) i].ticket) eng_->promote(slab_io_[(size_t) i].ticket, exl3_disk::kExpert);
    }
    else
    {
        i = slab_take();
        exl3_disk::TicketId t = 0;
        read_into(key, slab_->slot(i), cls, 0, &t);
        slab_io_[(size_t) i].ticket = t;
        slab_io_[(size_t) i].ticket_key = key;
        slab_state_[(size_t) i].key = key;
    }
    SlotIo& io = slab_io_[(size_t) i];
    finish_read(key, slab_->slot(i), io.ticket);
    io.ticket = 0;
    const ExpertExtent& x = ext_[(size_t) key];
    const uint8_t* s = slab_->slot(i);
    uint64_t tok = 0;
    for (int p = 0; p < geo.projections; ++p)
    {
        int tp = x.tpiece[p];
        tok = copier_->h2d(dst + (uint64_t) geo.proj_off[p], s + x.base[tp] + x.payload[tp] + x.trel[p],
                           geo.proj_bytes[p], after);
    }
    io.read_tok = tok;
    if (staging) ++stats.staged;
    else ++stats.h2d;
    slab_release(i);
}

void TierHost::run_action(const Action& a, const Record* rec, uint64_t* entry_after)
{
    switch (a.op)
    {
        case kActD2D:
        {
            uint64_t t = copier_->d2d(dst_addr(a.c, a.d, 0), copier_->pool_addr(a.b), geo.vram_slot_bytes, *entry_after);
            (void) t;
            ++stats.d2d;
            break;
        }
        case kActH2D:
        {
            SlotIo& io = ram_io_[(size_t) a.b];
            if (io.layout_key != a.a) fail("RAM slot " + std::to_string(a.b) + " does not hold " + where(a.a));
            promote_from_ram(a.b, dst_addr(a.c, a.d, 0), *entry_after, false);
            break;
        }
        case kActRamFree:
        case kActRamEvict:
        case kActDupReclaim:
        case kActRamOverwrite:
        {
            // the slot's bytes are no longer anyone's; the next write into it waits for its readers
            SlotIo& io = ram_io_[(size_t) a.b];
            io.layout = 0;
            io.layout_key = -1;
            break;
        }
        case kActRamDup:
        case kActRamOwn:
        case kActDrop:
            break;
        case kActSsd:
        {
            uint64_t dst = dst_addr(a.c, a.d, 0);
            if (a.b >= 0)
            {
                ram_ready_for_write(a.b);
                SlotIo& io = ram_io_[(size_t) a.b];
                exl3_disk::TicketId t = 0;
                read_into(a.a, arena_->slot(a.b), exl3_disk::kExpert, 0, &t);
                io.layout = 1;
                io.layout_key = a.a;
                io.ticket = t;
                io.ticket_key = a.a;
                promote_from_ram(a.b, dst, *entry_after, false);
            }
            else stage_from_ssd(a.a, dst, *entry_after, false, exl3_disk::kExpert);
            break;
        }
        case kActD2H:
        {
            SlotIo& io = ram_io_[(size_t) a.c];
            if (io.ticket)
            {
                finish_read(io.ticket_key, arena_->slot(a.c), io.ticket);
                io.ticket = 0;
            }
            uint64_t after = io.read_tok;
            if (io.write_tok)
            {
                if (after) copier_->wait(after);
                after = io.write_tok;
            }
            uint64_t t = copier_->d2h(arena_->slot(a.c), copier_->pool_addr(a.b), geo.vram_slot_bytes, after);
            io.read_tok = 0;
            io.write_tok = t;
            io.layout = 2;
            io.layout_key = a.a;
            pool_busy_[(size_t) a.b] = t;
            ++stats.d2h;
            if (rec && a.entry >= 0 && rec->entries[(size_t) a.entry].inplace) *entry_after = t;
            break;
        }
        case kActReturn:
        {
            // the slot may take a new expert once the demotion that read it has landed
            if (pool_busy_[(size_t) a.a]) copier_->wait(pool_busy_[(size_t) a.a]);
            pool_busy_[(size_t) a.a] = 0;
            break;
        }
        case kActRefill:
        {
            ram_ready_for_write(a.b);
            SlotIo& io = ram_io_[(size_t) a.b];
            exl3_disk::TicketId t = 0;
            read_into(a.a, arena_->slot(a.b), exl3_disk::kRefill, 0, &t);
            io.layout = 1;
            io.layout_key = a.a;
            io.ticket = t;
            io.ticket_key = a.a;
            if (t) refills_.push_back({ a.b, t });
            ++stats.refill_reads;
            break;
        }
        case kActStageRam:
        {
            SlotIo& io = ram_io_[(size_t) a.b];
            if (io.layout_key != a.a) fail("RAM slot " + std::to_string(a.b) + " does not hold " + where(a.a));
            promote_from_ram(a.b, copier_->staging_addr(cur_half_, a.c), 0, true);
            break;
        }
        case kActStageSsd:
            stage_from_ssd(a.a, copier_->staging_addr(cur_half_, a.b), 0, true, exl3_disk::kExpert);
            break;
        default:
            fail("unknown action " + std::to_string(a.op));
    }
}

void TierHost::execute(const Record* rec, std::vector<Action>& acts)
{
    last_ = acts;
    // An in-place admission (spare=0) writes its victim's slot: the victim's demotion (if any) must
    // read it first, so it runs ahead of the other actions of its entry
    std::vector<uint8_t> done(acts.size(), 0);
    int cur_entry = -2;
    uint64_t entry_after = 0;
    for (size_t i = 0; i < acts.size(); ++i)
    {
        if (done[i]) continue;
        const Action& a = acts[i];
        if (a.entry != cur_entry)
        {
            cur_entry = a.entry;
            entry_after = 0;
            if (rec && a.entry >= 0 && rec->entries[(size_t) a.entry].inplace)
                for (size_t j = i; j < acts.size() && acts[j].entry == a.entry; ++j)
                    if (acts[j].op == kActD2H)
                    {
                        run_action(acts[j], rec, &entry_after);
                        done[j] = 1;
                    }
        }
        run_action(a, rec, &entry_after);
        done[i] = 1;
    }
}

Record TierHost::call(int32_t lc, const int32_t* ids, int64_t n, int mode)
{
    throw_if_failed();
    Record rec = core.lookup(lc, ids, n, mode);
    std::vector<Action> acts;
    poll();
    core.refill_busy = (int) refills_.size();
    core.host(rec, acts);
    execute(&rec, acts);
    if (hcfg.deterministic) drain();
    else poll();
    throw_if_failed();
    return rec;
}

void TierHost::process(const Record& rec)
{
    throw_if_failed();
    std::vector<Action> acts;
    poll();
    core.refill_busy = (int) refills_.size();
    core.process(rec, acts);
    execute(&rec, acts);
    if (hcfg.deterministic) drain();
    else poll();
    throw_if_failed();
}

std::vector<int32_t> TierHost::return_landed()
{
    std::vector<int32_t> out;
    std::vector<int32_t>& p = core.pending;
    size_t w = 0;
    for (int32_t s : p)
    {
        uint64_t t = pool_busy_[(size_t) s];
        if (!t || copier_->done(t))
        {
            pool_busy_[(size_t) s] = 0;
            core.vram.return_slot(s);
            out.push_back(s);
        }
        else p[w++] = s;
    }
    p.resize(w);
    return out;
}

void TierHost::layer_call(int32_t lc, const int32_t* ids, int64_t n, int half)
{
    throw_if_failed();
    if (half < 0 || half > 1) fail("staging half " + std::to_string(half));
    std::vector<Action> acts;
    poll();
    core.refill_busy = (int) refills_.size();
    core.layer_call(lc, ids, n, acts);
    cur_half_ = half;
    try
    {
        execute(nullptr, acts);
    }
    catch (...)
    {
        cur_half_ = 0;
        throw;
    }
    cur_half_ = 0;
    // prefetches of this layer that were not needed (the expert reached RAM or VRAM meanwhile)
    for (size_t i = 0; i < slab_state_.size(); ++i)
        if (slab_state_[i].prefetch && slab_state_[i].lc == lc)
        {
            ++stats.prefetch_dropped;
            slab_release((int32_t) i);
        }
    if (hcfg.deterministic) drain();
    else poll();
    throw_if_failed();
}

void TierHost::prefetch_layer(int32_t lc, int64_t deadline_ns)
{
    throw_if_failed();
    if (lc < 0 || lc >= core.L) fail("prefetch of layer " + std::to_string(lc) + " out of range");
    for (int64_t e = 0; e < core.E; ++e)
    {
        int32_t k = (int32_t) (lc * core.E + e);
        if (core.vram.slot_of[(size_t) k] >= 0 || core.ram.of[(size_t) k] >= 0 || slab_of(k) >= 0) continue;
        if (slab_free_.empty())
        {
            ++stats.prefetch_starved;
            continue;
        }
        int32_t i = slab_free_.back();
        slab_free_.pop_back();
        exl3_disk::TicketId t = 0;
        read_into(k, slab_->slot(i), exl3_disk::kPrefetch, deadline_ns, &t);
        slab_io_[(size_t) i].ticket = t;
        slab_io_[(size_t) i].ticket_key = k;
        slab_state_[(size_t) i].key = k;
        slab_state_[(size_t) i].prefetch = true;
        slab_state_[(size_t) i].lc = lc;
        ++stats.prefetch_issued;
    }
}

void TierHost::promote_layer(int32_t lc)
{
    for (size_t i = 0; i < slab_state_.size(); ++i)
        if (slab_state_[i].prefetch && slab_state_[i].lc == lc && slab_io_[i].ticket)
            eng_->promote(slab_io_[i].ticket, exl3_disk::kExpert);
}

void TierHost::complete()
{
    std::vector<Action> acts;
    core.complete(acts);
    execute(nullptr, acts);
}

void TierHost::drain()
{
    for (auto& r : refills_)
    {
        SlotIo& io = ram_io_[(size_t) r.first];
        if (io.ticket == r.second)
        {
            finish_read(io.ticket_key, arena_->slot(r.first), io.ticket);
            io.ticket = 0;
            io.ticket_key = -1;
        }
    }
    refills_.clear();
    for (size_t i = 0; i < slab_io_.size(); ++i)
    {
        SlotIo& io = slab_io_[i];
        if (io.ticket)
        {
            finish_read(io.ticket_key, slab_->slot((int64_t) i), io.ticket);
            io.ticket = 0;
        }
        if (io.read_tok) copier_->wait(io.read_tok);
        io.read_tok = 0;
    }
    for (size_t r = 0; r < ram_io_.size(); ++r)
    {
        SlotIo& io = ram_io_[r];
        if (io.ticket)
        {
            finish_read(io.ticket_key, arena_->slot((int64_t) r), io.ticket);
            io.ticket = 0;
            io.ticket_key = -1;
        }
        if (io.read_tok) copier_->wait(io.read_tok);
        if (io.write_tok) copier_->wait(io.write_tok);
        io.read_tok = io.write_tok = 0;
    }
    for (auto& t : pool_busy_)
    {
        if (t) copier_->wait(t);
        t = 0;
    }
}

void TierHost::poll()
{
    size_t w = 0;
    for (size_t i = 0; i < refills_.size(); ++i)
    {
        auto r = refills_[i];
        SlotIo& io = ram_io_[(size_t) r.first];
        if (io.ticket != r.second) continue;                // finished by a reader already
        if (eng_->done(r.second))
        {
            finish_read(io.ticket_key, arena_->slot(r.first), io.ticket);
            io.ticket = 0;
            io.ticket_key = -1;
            continue;
        }
        refills_[w++] = r;
    }
    refills_.resize(w);
}

// Extent-layout RAM slots (their reads landed, nothing reads them) to the compact layout: the three
// trellis tensors back to back from the slot's start, as in a VRAM slot. The sources lie at odd
// offsets and in the checkpoint's tensor order, which need not be the slot order, so each slot goes
// through a scratch buffer. Several threads (prefault_threads)
void TierHost::compact_ram(const std::vector<int32_t>& slots)
{
    int64_t t0 = mono_ns();
    std::atomic<size_t> next { 0 };
    auto work = [&]
    {
        std::vector<uint8_t> tmp((size_t) geo.vram_slot_bytes);
        for (size_t i = next++; i < slots.size(); i = next++)
        {
            int32_t r = slots[i];
            SlotIo& io = ram_io_[(size_t) r];
            if (io.layout != 1 || io.ticket || io.read_tok || io.write_tok) continue;
            uint8_t* s = arena_->slot(r);
            for (int p = 0; p < geo.projections; ++p)
                std::memcpy(tmp.data() + geo.proj_off[p], s + ram_slot_offset(r, p), (size_t) geo.proj_bytes[p]);
            std::memcpy(s, tmp.data(), (size_t) geo.vram_slot_bytes);
            io.layout = 2;
        }
    };
    int nt = std::max(1, std::min(hcfg.prefault_threads, (int) std::max<size_t>(1, slots.size() / 16)));
    std::vector<std::thread> th;
    for (int t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto& x : th) x.join();
    for (int32_t r : slots)
        if (ram_io_[(size_t) r].layout == 2) ++stats.compacted;
    stats.compact_ns += mono_ns() - t0;
}

void TierHost::release_slab()
{
    for (size_t i = 0; i < slab_io_.size(); ++i) slab_release((int32_t) i);
    slab_free_.clear();
    slab_io_.clear();
    slab_state_.clear();
    slab_.reset(new Arena(0, geo.ram_slot_bytes, hcfg.chunk_bytes, false, eng_.get(), 0));
    hcfg.slab_slots = 0;
}

void TierHost::cold_fill(const std::vector<int32_t>& order, const std::vector<int32_t>& pins)
{
    throw_if_failed();
    int64_t t0 = mono_ns(), bytes0 = stats.ssd_bytes;
    if ((int64_t) pins.size() > core.vram.pins)
        fail(std::to_string(pins.size()) + " hot pins for " + std::to_string(core.vram.pins) + " pin slots");
    for (size_t i = 0; i < pins.size(); ++i) core.vram.place(pins[i], core.vram.S + (int32_t) i, 0);
    std::vector<int32_t> vk, rk;
    core.cold_fill(order, &vk, &rk);
    // the RAM tier, read in place (class 2), fill_inflight at a time
    std::deque<int32_t> inflight;
    auto finish_one = [&]()
    {
        int32_t r = inflight.front();
        inflight.pop_front();
        ram_ready_for_read(r);
    };
    for (int32_t k : rk)
    {
        int32_t r = core.ram.of[(size_t) k];
        SlotIo& io = ram_io_[(size_t) r];
        exl3_disk::TicketId t = 0;
        read_into(k, arena_->slot(r), exl3_disk::kPrefetch, 0, &t);
        io.layout = 1;
        io.layout_key = k;
        io.ticket = t;
        io.ticket_key = k;
        inflight.push_back(r);
        if ((int) inflight.size() >= std::max(1, hcfg.fill_inflight)) finish_one();
        ++stats.cold_ram;
    }
    while (!inflight.empty()) finish_one();
    if (hcfg.compact_fill)
    {
        std::vector<int32_t> slots;
        for (int32_t k : rk) slots.push_back(core.ram.of[(size_t) k]);
        compact_ram(slots);
    }
    for (size_t c = 0; c < arena_->base.size(); ++c) copier_->chunk_filled(arena_->base[c], arena_->bytes[c]);
    // the hot pins, through the slab
    for (size_t i = 0; i < pins.size(); ++i)
    {
        stage_from_ssd(pins[i], copier_->pool_addr(core.vram.S + (int32_t) i), 0, false, exl3_disk::kPrefetch);
        ++stats.cold_vram;
    }
    // the pool: from the RAM copy when RAM holds one (inclusive), else through the slab
    for (int32_t k : vk)
    {
        int32_t s = core.vram.slot_of[(size_t) k];
        int32_t r = core.ram.of[(size_t) k];
        if (r >= 0) promote_from_ram(r, copier_->pool_addr(s), 0, false);
        else stage_from_ssd(k, copier_->pool_addr(s), 0, false, exl3_disk::kPrefetch);
        ++stats.cold_vram;
    }
    drain();
    stats.cold_ns += mono_ns() - t0;
    stats.cold_bytes += stats.ssd_bytes - bytes0;
    throw_if_failed();
}

}  // namespace exl3_tier
