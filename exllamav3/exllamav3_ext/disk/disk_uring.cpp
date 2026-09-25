// io_uring backend of the disk engine, driven with raw syscalls (no liburing): a synchronous
// class-0 caller submits its own reads inline, one reaper thread submits everything else and
// completes, fixed files, registered buffers (the O_DIRECT bounce arena and caller slots, on
// every ring), optional SQPOLL, optional IOPOLL ring for O_DIRECT reads on devices with poll
// queues. Request ownership rules: disk_internal.h. Design: doc/disk_engine.md.

#if defined(__linux__)

#include "disk_internal.h"

#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#endif
#include <sys/syscall.h>

#if defined(IORING_ENTER_EXT_ARG) && defined(IORING_FEAT_EXT_ARG) && \
    defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter) && defined(__NR_io_uring_register)
#define EXL3_DISK_HAVE_URING 1
#else
#define EXL3_DISK_HAVE_URING 0
#endif

#if EXL3_DISK_HAVE_URING

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

// Sparse tables and in-place updates of registered files / buffers (Linux 5.19 headers)
#if defined(IORING_RSRC_REGISTER_SPARSE)
#define EXL3_DISK_URING_RSRC 1
#else
#define EXL3_DISK_URING_RSRC 0
#endif

namespace exl3_disk
{
namespace detail
{

namespace
{

std::string errstr(int e)
{
    char buf[128];
    const char* s = strerror_r(e, buf, sizeof buf);
    return std::string(s ? s : "unknown error") + " (errno " + std::to_string(e) + ")";
}

void warn(const std::string& msg)
{
    std::fprintf(stderr, " !! disk engine: %s\n", msg.c_str());
    std::fflush(stderr);
}

inline void cpu_relax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

int sys_setup(unsigned entries, io_uring_params* p)
{
    return (int) syscall(__NR_io_uring_setup, entries, p);
}

int sys_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags, const void* arg,
              size_t argsz)
{
    return (int) syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, arg, argsz);
}

int sys_register(int fd, unsigned op, const void* arg, unsigned nr)
{
    return (int) syscall(__NR_io_uring_register, fd, op, arg, nr);
}

constexpr uint32_t kNoSlot = UINT32_MAX;

// One ring: its kernel-shared memory and the local submission tail
struct Ring
{
    int fd = -1;
    unsigned entries = 0;
    unsigned cq_entries = 0;
    unsigned features = 0;
    bool sqpoll = false;
    bool iopoll = false;
    void* sq_map = MAP_FAILED;
    size_t sq_map_sz = 0;
    void* cq_map = MAP_FAILED;
    size_t cq_map_sz = 0;
    io_uring_sqe* sqes = nullptr;
    size_t sqes_sz = 0;
    unsigned* k_sq_head = nullptr;
    unsigned* k_sq_tail = nullptr;
    unsigned* k_sq_flags = nullptr;
    unsigned* k_sq_array = nullptr;
    unsigned sq_mask = 0;
    unsigned* k_cq_head = nullptr;
    unsigned* k_cq_tail = nullptr;
    unsigned* k_cq_overflow = nullptr;
    io_uring_cqe* cqes = nullptr;
    unsigned cq_mask = 0;
    unsigned sq_tail = 0;                       // sq_mx_ (+ Core::mx to change it)
    std::vector<uint32_t> sq_slot;              // SQ index -> slot, same locks
    bool buf_table = false;                     // sparse buffer table registered (init only)
    bool bounce_reg = false;                    // bounce arena registered as buffer 0

    bool setup(unsigned n, unsigned flags, unsigned sq_idle_ms, int sq_cpu, std::string* why)
    {
        io_uring_params p;
        unsigned base = flags | IORING_SETUP_CQSIZE;
#ifdef IORING_SETUP_SUBMIT_ALL
        unsigned extra = IORING_SETUP_SUBMIT_ALL;
#else
        unsigned extra = 0;
#endif
        int r = -1;
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            std::memset(&p, 0, sizeof p);
            p.flags = base | (attempt == 0 ? extra : 0);
            p.cq_entries = n * 2;
            if (flags & IORING_SETUP_SQPOLL)
            {
                p.sq_thread_idle = sq_idle_ms;
                if (sq_cpu >= 0)
                {
                    p.flags |= IORING_SETUP_SQ_AFF;
                    p.sq_thread_cpu = (unsigned) sq_cpu;
                }
            }
            r = sys_setup(n, &p);
            if (r >= 0 || errno != EINVAL || !extra) break;   // retry without SUBMIT_ALL
        }
        if (r < 0)
        {
            *why = "io_uring_setup: " + errstr(errno);
            return false;
        }
        fd = r;
        features = p.features;
        sqpoll = (flags & IORING_SETUP_SQPOLL) != 0;
        iopoll = (flags & IORING_SETUP_IOPOLL) != 0;
        if (!(features & IORING_FEAT_EXT_ARG))
        {
            *why = "the kernel lacks IORING_FEAT_EXT_ARG (Linux 5.11 or later is needed)";
            teardown();
            return false;
        }
        entries = p.sq_entries;
        cq_entries = p.cq_entries;
        sq_map_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_map_sz = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        bool single = (features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) sq_map_sz = cq_map_sz = std::max(sq_map_sz, cq_map_sz);
        sq_map = mmap(nullptr, sq_map_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                      IORING_OFF_SQ_RING);
        if (sq_map == MAP_FAILED)
        {
            *why = "mmap of the SQ ring: " + errstr(errno);
            teardown();
            return false;
        }
        if (single) cq_map = sq_map;
        else
        {
            cq_map = mmap(nullptr, cq_map_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                          fd, IORING_OFF_CQ_RING);
            if (cq_map == MAP_FAILED)
            {
                *why = "mmap of the CQ ring: " + errstr(errno);
                teardown();
                return false;
            }
        }
        sqes_sz = p.sq_entries * sizeof(io_uring_sqe);
        void* s = mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                       IORING_OFF_SQES);
        if (s == MAP_FAILED)
        {
            *why = "mmap of the SQEs: " + errstr(errno);
            teardown();
            return false;
        }
        sqes = static_cast<io_uring_sqe*>(s);
        uint8_t* sq = static_cast<uint8_t*>(sq_map);
        uint8_t* cq = static_cast<uint8_t*>(cq_map);
        k_sq_head = reinterpret_cast<unsigned*>(sq + p.sq_off.head);
        k_sq_tail = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
        k_sq_flags = reinterpret_cast<unsigned*>(sq + p.sq_off.flags);
        k_sq_array = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
        sq_mask = *reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask);
        k_cq_head = reinterpret_cast<unsigned*>(cq + p.cq_off.head);
        k_cq_tail = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
        k_cq_overflow = reinterpret_cast<unsigned*>(cq + p.cq_off.overflow);
        cqes = reinterpret_cast<io_uring_cqe*>(cq + p.cq_off.cqes);
        cq_mask = *reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask);
        sq_tail = __atomic_load_n(k_sq_tail, __ATOMIC_RELAXED);
        sq_slot.assign(entries, kNoSlot);
        return true;
    }

    void teardown()
    {
        if (sqes) munmap(sqes, sqes_sz);
        if (cq_map != MAP_FAILED && cq_map != sq_map) munmap(cq_map, cq_map_sz);
        if (sq_map != MAP_FAILED) munmap(sq_map, sq_map_sz);
        if (fd >= 0) ::close(fd);
        sqes = nullptr;
        cq_map = sq_map = MAP_FAILED;
        fd = -1;
    }

    // sq_mx_ held
    unsigned sq_used() const { return sq_tail - __atomic_load_n(k_sq_head, __ATOMIC_ACQUIRE); }

    io_uring_sqe* get_sqe(uint32_t slot)
    {
        unsigned idx = sq_tail & sq_mask;
        io_uring_sqe* sqe = &sqes[idx];
        std::memset(sqe, 0, sizeof *sqe);
        k_sq_array[idx] = idx;
        sq_slot[idx] = slot;
        ++sq_tail;
        return sqe;
    }

    void publish()
    {
        // seq_cst: with SQPOLL the tail store must be ordered before the later load of the
        // NEED_WAKEUP flag (store -> load, the full barrier liburing puts there)
        __atomic_store_n(k_sq_tail, sq_tail, __ATOMIC_SEQ_CST);
    }

    bool cq_ready() const
    {
        return __atomic_load_n(k_cq_head, __ATOMIC_ACQUIRE) !=
               __atomic_load_n(k_cq_tail, __ATOMIC_ACQUIRE);
    }
};

thread_local std::vector<io_uring_cqe> t_cqes;
// Completed reads whose bounce copy / FADV_DONTNEED runs after Core::mx is released
struct Post
{
    uint32_t slot;
    Op* op;
};
thread_local std::vector<Post> t_post;
// EXL3_DISK_AFFINITY for the io-wq of this thread: the executor it was last applied for
thread_local uint64_t t_aff_token = 0;
std::atomic<uint64_t> g_aff_tokens { 0 };
// Buffered ops above this size that are not class 0 go to io-wq (IOSQE_ASYNC): inline, their
// page-cache copies would run inside io_uring_enter under the submission lock
constexpr uint32_t kInlineCopyMax = 64 * 1024;

class UringExec final : public Executor
{
public:
    explicit UringExec(Core& c) : c_(c) {}

    ~UringExec() override
    {
        stop();
        if (bounce_) munmap(bounce_, bounce_total_);
        if (evfd_ >= 0) ::close(evfd_);
        poll_.teardown();
        main_.teardown();
    }

    bool init(std::string* why)
    {
        const Config& cfg = c_.cfg;
        unsigned n = 32;
        while (n < (unsigned) cfg.qd + 16) n <<= 1;

        evfd_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (evfd_ < 0)
        {
            *why = "eventfd: " + errstr(errno);
            return false;
        }

        // Main ring (buffered reads, and O_DIRECT reads of devices that do not poll)
        unsigned flags = cfg.sqpoll ? IORING_SETUP_SQPOLL : 0;
        std::string w;
        if (!main_.setup(n, flags, (unsigned) cfg.sqpoll_idle_ms, cfg.sqpoll_cpu, &w))
        {
            if (!cfg.sqpoll)
            {
                *why = w;
                return false;
            }
            note("EXL3_DISK_SQPOLL refused (" + w + "): submitting from the calling threads");
            main_.teardown();
            if (!main_.setup(n, 0, 0, -1, why)) return false;
        }
        if (main_.entries < (unsigned) cfg.qd + 8 || main_.cq_entries < 2 * main_.entries)
        {
            *why = "ring smaller than requested (" + std::to_string(main_.entries) + " entries)";
            return false;
        }
        slots_.resize(main_.entries);
        free_.reserve(main_.entries);
        resub_.reserve(main_.entries);
        for (uint32_t i = main_.entries; i-- > 0; ) free_.push_back(i);

        // IOPOLL ring for O_DIRECT reads on devices that poll
        if (cfg.iopoll && (cfg.direct_rows || cfg.direct_extents))
        {
            unsigned pf = IORING_SETUP_IOPOLL | (main_.sqpoll ? IORING_SETUP_SQPOLL : 0);
            if (!poll_.setup(n, pf, (unsigned) cfg.sqpoll_idle_ms, cfg.sqpoll_cpu, &w))
            {
                note("EXL3_DISK_IOPOLL ring refused (" + w + "): no polled completions");
                poll_.teardown();
            }
        }

        // Bounce arena for O_DIRECT rows: one slot per ring slot
        if (cfg.direct_rows)
        {
            bounce_total_ = (size_t) main_.entries * kBounceBytes;
            void* p = mmap(nullptr, bounce_total_, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
            if (p == MAP_FAILED)
            {
                *why = "bounce arena: " + errstr(errno);
                bounce_ = nullptr;
                return false;
            }
            bounce_ = static_cast<uint8_t*>(p);
            (void) madvise(bounce_, bounce_total_, MADV_DONTFORK);
        }

        if (cfg.reg_buffers)
        {
            register_buffer_table(main_, "");
            if (poll_.fd >= 0) register_buffer_table(poll_, "IOPOLL ring: ");
        }
        if (cfg.reg_files) register_file_table();

#if EXL3_DISK_URING_RSRC
        if (cfg.iowq_workers > 0)
        {
            unsigned vals[2] = { (unsigned) cfg.iowq_workers, 0 };
            if (sys_register(main_.fd, IORING_REGISTER_IOWQ_MAX_WORKERS, vals, 2) < 0)
                note("EXL3_DISK_IOWQ_WORKERS refused: " + errstr(errno));
        }
        if (!cfg.affinity.empty())
        {
            // io-wq belongs to each submitting task, so every other submitting thread applies
            // the mask again before its first submission (apply_iowq_affinity); this covers
            // the creating thread and, with SQPOLL, the ring's poller and its workers
            CPU_ZERO(&aff_set_);
            for (int cpu : cfg.affinity) CPU_SET(static_cast<size_t>(cpu), &aff_set_);   // < 1024
            aff_token_ = g_aff_tokens.fetch_add(1) + 1;
            if (sys_register(main_.fd, IORING_REGISTER_IOWQ_AFF, &aff_set_, sizeof aff_set_) < 0)
                note("EXL3_DISK_AFFINITY for io_uring workers refused: " + errstr(errno));
            else t_aff_token = aff_token_;
        }
#else
        if (cfg.iowq_workers > 0 || !cfg.affinity.empty())
            note("io-wq limits need Linux 5.19 headers at build time: not applied");
#endif

        stop_.store(false);
        reaper_ = std::thread(&UringExec::reaper_main, this);
        started_ = true;
        if (poll_.fd >= 0)
        {
            try { poller_ = std::thread(&UringExec::poller_main, this); }
            catch (...)
            {
                stop();
                throw;
            }
        }
        return true;
    }

    const char* name() const override { return "io_uring"; }

    // A class-0 caller about to wait for t submits t's reads itself: no hand-off, and inline
    // page-cache completions it reaps itself. Anything else goes to the reaper (will_wait is
    // only true for a class-0 ticket; enqueue decides under Core::mx)
    void submitted(Ticket* t, bool will_wait) override
    {
        if (t && will_wait) pump(t);
        else wake();
    }

    bool wait(Ticket* t, std::unique_lock<std::mutex>& lk, int64_t deadline) override
    {
        if (t->complete.load(std::memory_order_acquire)) return true;
        // Only a waiter that stays until t completes may own (submit) t's reads, and only for
        // class 0 (bulk submitted by a caller would keep the submission lock from the reaper)
        Ticket* own = deadline < 0 && t->cls == kEngram ? t : nullptr;
        const int64_t spin = (int64_t) c_.cfg.spin_us * 1000;
        if (spin > 0)
        {
            // Completions of reads this thread submitted are posted by its own task work, which
            // runs while it spins here; draining them here saves the reaper's wake-up
            lk.unlock();
            int64_t end = now_ns() + spin;
            if (deadline >= 0) end = std::min(end, deadline);
            while (!t->complete.load(std::memory_order_acquire))
            {
                if (reap(main_, true))
                {
                    after_inline_reap(own);
                    continue;
                }
                if (now_ns() >= end) break;
                cpu_relax();
            }
            lk.lock();
        }
        while (!t->complete.load(std::memory_order_acquire))
        {
            if (deadline < 0) t->cv.wait(lk);
            else if (t->cv.wait_until(lk, std::chrono::steady_clock::time_point(
                                              std::chrono::nanoseconds(deadline))) ==
                         std::cv_status::timeout &&
                     !t->complete.load(std::memory_order_acquire))
                return false;
        }
        return true;
    }

    void stop() override
    {
        if (!started_) return;
        stop_.store(true, std::memory_order_release);
        wake();
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            cv_poll_.notify_all();
        }
        if (reaper_.joinable()) reaper_.join();
        if (poller_.joinable()) poller_.join();
        started_ = false;
    }

    void file_opened(FileEnt* f) override
    {
        if (!files_reg_) return;
        fixed_set(f->fd_buf, &f->fixed_buf);
        fixed_set(f->fd_dir, &f->fixed_dir);
    }

    void file_closing(FileEnt* f) override
    {
        if (!files_reg_) return;
        fixed_clear(&f->fixed_buf);
        fixed_clear(&f->fixed_dir);
    }

    int register_buffer(void* p, size_t n) override
    {
#if EXL3_DISK_URING_RSRC
        if (!main_.buf_table) return -1;
        if (n > (size_t) 1 << 30)
            throw Error(EINVAL, "disk engine: a registered buffer is at most 1 GiB");
        int idx = -1;
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            for (int i = 1; i < kMaxUserBuffers; ++i)
                if (ubuf_[i].state == 0)
                {
                    idx = i;
                    ubuf_[i].state = 1;
                    break;
                }
        }
        if (idx < 0) return -1;
        // on every ring that has a table: READ_FIXED on a ring without the buffer is EFAULT
        int err = 0;
        bool on_main = update_buffer(main_, idx, p, n, &err);
        bool ok = on_main && (!poll_.buf_table || update_buffer(poll_, idx, p, n, &err));
        if (on_main && !ok) (void) update_buffer(main_, idx, nullptr, 0, nullptr);
        std::lock_guard<std::mutex> lk(c_.mx);
        if (!ok)
        {
            ubuf_[idx].state = 0;
            if (!buf_warned_)
            {
                buf_warned_ = true;
                warn("registering a buffer with io_uring failed (" + errstr(err) +
                     "): reading into it without READ_FIXED");
            }
            return -1;
        }
        ubuf_[idx].p = static_cast<uint8_t*>(p);
        ubuf_[idx].n = n;
        ubuf_[idx].refs = 0;
        ubuf_[idx].state = 2;
        ++nreg_;
        index_add(idx);
        return idx;
#else
        (void) p;
        (void) n;
        return -1;
#endif
    }

    void unregister_buffer(int idx) override
    {
#if EXL3_DISK_URING_RSRC
        if (idx < 1 || idx >= kMaxUserBuffers)
            throw Error(EINVAL, "disk engine: no registered buffer " + std::to_string(idx));
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            if (ubuf_[idx].state != 2)
                throw Error(EINVAL, "disk engine: no registered buffer " + std::to_string(idx));
            if (ubuf_[idx].refs > 0)
                throw Error(EBUSY, "disk engine: registered buffer " + std::to_string(idx) +
                                   " is still the target of unfinished reads");
            ubuf_[idx].state = 1;       // find_buffer no longer returns it
            --nreg_;
            index_remove(idx);
        }
        (void) update_buffer(main_, idx, nullptr, 0, nullptr);
        if (poll_.buf_table) (void) update_buffer(poll_, idx, nullptr, 0, nullptr);
        std::lock_guard<std::mutex> lk(c_.mx);
        ubuf_[idx] = UBuf();
#else
        throw Error(EINVAL, "disk engine: no registered buffer " + std::to_string(idx));
#endif
    }

    int user_buffers() const override { return nreg_; }

    // The registered buffer that holds [p, p + n) whole, or -1. Registered buffers are kept
    // sorted by address, so this is a binary search (a RAM tier registers one buffer per 1 GiB
    // chunk, over a hundred of them, and every destination range of every ticket is looked up).
    // Buffers that overlap each other (the same memory registered twice) fall back to the scan
    // in index order, which returns the lowest index that holds the range
    int find_buffer(const uint8_t* p, size_t n) override
    {
        if (nreg_ == 0) return -1;
        uintptr_t a = reinterpret_cast<uintptr_t>(p);
        if (overlap_)
        {
            for (int i = 1; i < kMaxUserBuffers; ++i)
                if (ubuf_[i].state == 2 && holds(ubuf_[i], a, n)) return i;
            return -1;
        }
        auto it = std::upper_bound(sorted_.begin(), sorted_.end(), a,
                                   [](uintptr_t x, const std::pair<uintptr_t, int>& e) { return x < e.first; });
        if (it == sorted_.begin()) return -1;
        --it;
        return holds(ubuf_[it->second], a, n) ? it->second : -1;
    }

    void buffer_ref(int idx, int d) override
    {
        if (idx >= 1 && idx < kMaxUserBuffers) ubuf_[idx].refs += d;
    }

    std::string describe() const override
    {
        std::string s = "io_uring: " + std::to_string(main_.entries) + " SQ / " +
                        std::to_string(main_.cq_entries) + " CQ entries";
        s += main_.sqpoll ? ", sqpoll" : "";
        s += poll_.fd >= 0 ? ", iopoll ring" : "";
        s += files_reg_ ? ", fixed files" : "";
        s += bounce_ ? (main_.bounce_reg ? ", registered bounce arena" : ", bounce arena") : "";
        s += main_.buf_table ? ", buffer table" : "";
        s += poll_.buf_table ? " (both rings)" : "";
        for (const std::string& n : notes_) s += " | " + n;
        return s;
    }

private:
    struct Slot
    {
        Op* op = nullptr;
        uint32_t gen = 0;
        uint8_t ring = 0;       // 0 main, 1 poll
    };
    struct UBuf
    {
        uint8_t* p = nullptr;
        size_t n = 0;
        int64_t refs = 0;
        int state = 0;          // 0 free, 1 changing, 2 registered
    };

    static bool holds(const UBuf& b, uintptr_t a, size_t n)
    {
        uintptr_t base = reinterpret_cast<uintptr_t>(b.p);
        return a >= base && a - base <= b.n && n <= b.n - (a - base);
    }

    // Core::mx held: keep sorted_ (registered buffers by address) and overlap_ current
    void index_add(int idx)
    {
        std::pair<uintptr_t, int> e(reinterpret_cast<uintptr_t>(ubuf_[idx].p), idx);
        sorted_.insert(std::lower_bound(sorted_.begin(), sorted_.end(), e), e);
        index_check();
    }

    void index_remove(int idx)
    {
        for (auto it = sorted_.begin(); it != sorted_.end(); ++it)
            if (it->second == idx)
            {
                sorted_.erase(it);
                break;
            }
        index_check();
    }

    void index_check()
    {
        overlap_ = false;
        for (size_t i = 1; i < sorted_.size(); ++i)
        {
            const UBuf& prev = ubuf_[sorted_[i - 1].second];
            if (reinterpret_cast<uintptr_t>(prev.p) + prev.n > sorted_[i].first) overlap_ = true;
        }
    }

    void note(const std::string& s)
    {
        notes_.push_back(s);
        c_.fallbacks.push_back(s);
        warn(s);
    }

    void wake()
    {
        uint64_t one = 1;
        ssize_t r = ::write(evfd_, &one, sizeof one);   // EAGAIN: the counter is already set
        (void) r;
    }

    // Buffer table of one ring: sparse (caller buffers can be added later) with the bounce
    // arena at index 0, else the bounce arena alone. Every ring that reads with READ_FIXED
    // needs its own registration
    void register_buffer_table(Ring& r, const char* which)
    {
#if EXL3_DISK_URING_RSRC
        io_uring_rsrc_register rr;
        std::memset(&rr, 0, sizeof rr);
        rr.nr = kMaxUserBuffers;
        rr.flags = IORING_RSRC_REGISTER_SPARSE;
        if (sys_register(r.fd, IORING_REGISTER_BUFFERS2, &rr, sizeof rr) == 0)
        {
            r.buf_table = true;
            if (bounce_)
            {
                int err = 0;
                if (update_buffer(r, 0, bounce_, bounce_total_, &err)) r.bounce_reg = true;
                else note(std::string(which) + "registering the bounce arena failed: " + errstr(err));
            }
            return;
        }
        note(std::string(which) + "sparse buffer table refused (" + errstr(errno) +
             "): no registered caller buffers");
#endif
        if (bounce_)
        {
            struct iovec iov { bounce_, bounce_total_ };
            if (sys_register(r.fd, IORING_REGISTER_BUFFERS, &iov, 1) == 0) r.bounce_reg = true;
            else note(std::string(which) + "registering the bounce arena failed: " + errstr(errno));
        }
    }

    // Set (p, n) or clear (nullptr) buffer idx of a ring's sparse table
    bool update_buffer(Ring& r, int idx, void* p, size_t n, int* err)
    {
#if EXL3_DISK_URING_RSRC
        struct iovec iov { p, n };
        io_uring_rsrc_update2 up;
        std::memset(&up, 0, sizeof up);
        up.offset = (unsigned) idx;
        up.data = (uint64_t) (uintptr_t) &iov;
        up.nr = 1;
        int rc = sys_register(r.fd, IORING_REGISTER_BUFFERS_UPDATE, &up, sizeof up);
        if (rc < 0 && err) *err = errno;
        return rc >= 0;
#else
        (void) r; (void) idx; (void) p; (void) n;
        if (err) *err = EINVAL;
        return false;
#endif
    }

    void register_file_table()
    {
        constexpr unsigned kFiles = 512;
        bool ok = false;
#if EXL3_DISK_URING_RSRC
        io_uring_rsrc_register rr;
        std::memset(&rr, 0, sizeof rr);
        rr.nr = kFiles;
        rr.flags = IORING_RSRC_REGISTER_SPARSE;
        ok = sys_register(main_.fd, IORING_REGISTER_FILES2, &rr, sizeof rr) == 0;
#endif
        if (!ok)
        {
            std::vector<int> fds(kFiles, -1);
            ok = sys_register(main_.fd, IORING_REGISTER_FILES, fds.data(), kFiles) == 0;
        }
        if (!ok)
        {
            note("fixed-file table refused (" + errstr(errno) + "): plain descriptors");
            return;
        }
        files_reg_ = true;
        for (unsigned i = kFiles; i-- > 0; ) fixed_free_.push_back((int) i);
    }

    void fixed_set(int fd, int* slot)
    {
        if (fd < 0 || *slot >= 0) return;
        std::lock_guard<std::mutex> g(fixed_mx_);
        if (fixed_free_.empty()) return;
        int idx = fixed_free_.back();
        io_uring_files_update up;
        std::memset(&up, 0, sizeof up);
        up.offset = (unsigned) idx;
        up.fds = (uint64_t) (uintptr_t) &fd;
        if (sys_register(main_.fd, IORING_REGISTER_FILES_UPDATE, &up, 1) == 1)
        {
            fixed_free_.pop_back();
            *slot = idx;
        }
    }

    void fixed_clear(int* slot)
    {
        if (*slot < 0) return;
        std::lock_guard<std::mutex> g(fixed_mx_);
        int none = -1;
        io_uring_files_update up;
        std::memset(&up, 0, sizeof up);
        up.offset = (unsigned) *slot;
        up.fds = (uint64_t) (uintptr_t) &none;
        if (sys_register(main_.fd, IORING_REGISTER_FILES_UPDATE, &up, 1) == 1)
            fixed_free_.push_back(*slot);
        *slot = -1;
    }

    Ring& ring_of(uint32_t s) { return slots_[s].ring ? poll_ : main_; }

    // sq_mx_ and Core::mx held
    void prep(Ring& r, Op& op, uint32_t s)
    {
        io_uring_sqe* sqe = r.get_sqe(s);
        if (op.bounce) op.dst = bounce_ + (size_t) s * kBounceBytes;
        sqe->opcode = IORING_OP_READ;
        // READ_FIXED only against this ring's own registrations (user buffers are registered
        // on every ring with a table, or on none)
        if (op.bounce && r.bounce_reg)
        {
            sqe->opcode = IORING_OP_READ_FIXED;
            sqe->buf_index = 0;
        }
        else if (!op.bounce && op.buf_index >= 0 && r.buf_table)
        {
            // never a READ_FIXED the kernel would mis-describe (fixed_read_ok, disk_engine.h);
            // then a plain READ into the same memory
            const UBuf& b = ubuf_[op.buf_index];
            if (fixed_read_ok(reinterpret_cast<uintptr_t>(b.p), b.n,
                              reinterpret_cast<uintptr_t>(op.dst + op.got), page_))
            {
                sqe->opcode = IORING_OP_READ_FIXED;
                sqe->buf_index = (uint16_t) op.buf_index;
            }
            else ++c_.fixed_plain;
        }
        if (&r == &main_ && op.fixed >= 0)
        {
            sqe->fd = op.fixed;
            sqe->flags |= IOSQE_FIXED_FILE;
        }
        else sqe->fd = op.fd;
        sqe->addr = (uint64_t) (uintptr_t) (op.dst + op.got);
        sqe->len = op.dev_len - op.got;
        sqe->off = (uint64_t) (op.dev_off + op.got);
        // Large buffered bulk reads go to io-wq: inline, the kernel would copy them out of the
        // page cache inside io_uring_enter, with the submission lock held, and a class-0 batch
        // would wait for it before its own reads are even submitted
        if (op.async || (!op.direct && op.cls_issued != kEngram && op.dev_len > kInlineCopyMax))
            sqe->flags |= IOSQE_ASYNC;
        sqe->user_data = ((uint64_t) slots_[s].gen << 32) | (uint64_t) (s + 1);
    }

    // Admit ops in class order, prepare their SQEs and submit them from this thread, which then
    // owns them. only != nullptr: that ticket's ops alone (its class-0 caller waits for it
    // without a timeout), and only if the submission lock is free: a caller never waits for
    // another submitter's io_uring_enter, it hands its ticket to the reaper, which admits class
    // 0 first; nullptr: everything (reaper and poller threads only)
    void pump(Ticket* only)
    {
        std::unique_lock<std::mutex> sq(sq_mx_, std::defer_lock);
        if (only)
        {
            if (!sq.try_lock())
            {
                wake();
                return;
            }
        }
        else sq.lock();
        bool wake_reaper = false;
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            if (c_.broken.load(std::memory_order_relaxed))
            {
                timer_.store(0, std::memory_order_relaxed);   // nothing is admitted again
                return;
            }
            int64_t now = now_ns();
            bool any_poll = false;
            // resumed reads (short, EAGAIN, EINTR) and submissions the kernel did not take
            for (size_t i = 0; i < resub_.size(); )
            {
                uint32_t s = resub_[i];
                Op* op = slots_[s].op;
                Ring& r = ring_of(s);
                if ((only && op->t != only) || r.sq_used() >= r.entries)
                {
                    if (only) wake_reaper = true;
                    ++i;
                    continue;
                }
                prep(r, *op, s);
                if (slots_[s].ring) any_poll = true;
                resub_[i] = resub_.back();
                resub_.pop_back();
            }
            while (!free_.empty())
            {
                if (main_.sq_used() >= main_.entries) break;
                if (poll_.fd >= 0 && poll_.sq_used() >= poll_.entries) break;
                Op* op = c_.admit_one(only, now);
                if (!op) break;
                uint32_t s = free_.back();
                free_.pop_back();
                bool to_poll = op->poll && poll_.fd >= 0;
                slots_[s].op = op;
                slots_[s].ring = to_poll ? 1 : 0;
                prep(to_poll ? poll_ : main_, *op, s);
                if (to_poll)
                {
                    ++poll_inflight_;
                    any_poll = true;
                }
            }
            main_.publish();
            if (poll_.fd >= 0) poll_.publish();
            if (any_poll) cv_poll_.notify_one();
            if (only)
            {
                int64_t mine = only->q_cls >= 0 ? (int64_t) (only->ops.size() - only->next_issue) : 0;
                if (c_.queued_ops > mine) wake_reaper = true;
            }
            timer_.store(c_.next_timer(now), std::memory_order_relaxed);
        }
        enter(main_);
        if (poll_.fd >= 0) enter(poll_);
        if (wake_reaper) wake();
    }

    // sq_mx_ held: hand what this thread prepared to the kernel
    void enter(Ring& r)
    {
        if (c_.broken.load(std::memory_order_relaxed)) return;
        if (r.sqpoll)
        {
            if (__atomic_load_n(r.k_sq_flags, __ATOMIC_SEQ_CST) & IORING_SQ_NEED_WAKEUP)
                (void) sys_enter(r.fd, 0, 0, IORING_ENTER_SQ_WAKEUP, nullptr, 0);
            return;
        }
        if (r.sq_tail == __atomic_load_n(r.k_sq_head, __ATOMIC_ACQUIRE)) return;
        apply_iowq_affinity(r, false);
        for (int guard = 0; guard < 64; ++guard)
        {
            unsigned head = __atomic_load_n(r.k_sq_head, __ATOMIC_ACQUIRE);
            unsigned n = r.sq_tail - head;
            if (n == 0) return;
            int rc;
            if (c_.cfg.fault_enter_fatal > 0 &&
                fault_enters_.fetch_add(1, std::memory_order_relaxed) + 1 ==
                    (uint64_t) c_.cfg.fault_enter_fatal)
            {
                rc = -1;                        // test: the ring refuses submissions for good
                errno = EBADFD;
            }
            else rc = sys_enter(r.fd, n, 0, 0, nullptr, 0);
            if (rc > 0)
            {
                c_.enters.fetch_add(1, std::memory_order_relaxed);
                apply_iowq_affinity(r, true);
                continue;
            }
            int e = rc == 0 ? EAGAIN : errno;
            if (e == EINTR) continue;
            if (e == EAGAIN || e == EBUSY || e == ENOMEM) break;
            fatal(e);
            return;
        }
        // The kernel did not take everything (out of memory for requests, or an overflowing
        // completion queue): the rest goes back to the reaper, which retries shortly
        rollback(r);
        retry_.store(true, std::memory_order_relaxed);
        wake();
    }

    // sq_mx_ held: take back SQEs the kernel did not consume
    void rollback(Ring& r)
    {
        std::lock_guard<std::mutex> lk(c_.mx);
        unsigned head = __atomic_load_n(r.k_sq_head, __ATOMIC_ACQUIRE);
        for (unsigned i = head; i != r.sq_tail; ++i)
        {
            uint32_t s = r.sq_slot[i & r.sq_mask];
            if (s != kNoSlot) resub_.push_back(s);
        }
        r.sq_tail = head;
        r.publish();
    }

    // sq_mx_ held: this thread's io_uring workers get EXL3_DISK_AFFINITY. io-wq belongs to
    // each submitting task (created with its io_uring task context), and the mask applies to
    // the workers created after it is set (Linux 6.x no longer moves running workers). So it
    // is set before the thread's first submission: a temporarily registered ring descriptor
    // creates the task context without submitting anything. Where that is refused (before
    // Linux 5.18) the mask is set right after the first submission instead (after_submit)
    void apply_iowq_affinity(Ring& r, bool after_submit)
    {
#if EXL3_DISK_URING_RSRC
        if (!aff_token_ || t_aff_token == aff_token_ || r.sqpoll) return;
        if (!after_submit)
        {
            // IORING_REGISTER_RING_FDS (5.18) is an enumerator, not a macro: the 5.19 headers
            // this block needs (EXL3_DISK_URING_RSRC) have it
            io_uring_rsrc_update reg;
            std::memset(&reg, 0, sizeof reg);
            reg.offset = UINT32_MAX;            // any free slot; the kernel writes it back
            reg.data = (__u64) (unsigned) r.fd;     // r.fd >= 0
            bool registered = sys_register(r.fd, IORING_REGISTER_RING_FDS, &reg, 1) == 1;
            int rc = sys_register(r.fd, IORING_REGISTER_IOWQ_AFF, &aff_set_, sizeof aff_set_);
            int err = errno;
            if (registered)
            {
                io_uring_rsrc_update un;
                std::memset(&un, 0, sizeof un);
                un.offset = reg.offset;
                (void) sys_register(r.fd, IORING_UNREGISTER_RING_FDS, &un, 1);
            }
            if (rc == 0 || err != EINVAL)
            {
                t_aff_token = aff_token_;
                if (rc < 0 && !aff_warned_.exchange(true))
                    warn("EXL3_DISK_AFFINITY for a thread's io_uring workers refused: " +
                         errstr(err));
            }
            return;                             // EINVAL: no task context yet, retry after
        }
        t_aff_token = aff_token_;
        if (sys_register(r.fd, IORING_REGISTER_IOWQ_AFF, &aff_set_, sizeof aff_set_) < 0 &&
            !aff_warned_.exchange(true))
            warn("EXL3_DISK_AFFINITY for a thread's io_uring workers refused: " + errstr(errno));
#else
        (void) r;
        (void) after_submit;
#endif
    }

    // sq_mx_ held. A ring refused submissions for good: fail what no ring consumed (a pump
    // prepares SQEs on both rings before it enters either) and everything still queued, so no
    // waiter hangs
    void fatal(int e)
    {
        warn("io_uring_enter failed: " + errstr(e) + "; failing the queued reads");
        std::lock_guard<std::mutex> lk(c_.mx);
        int expected = 0;
        c_.broken.compare_exchange_strong(expected, -e);
        timer_.store(0, std::memory_order_relaxed);
        int64_t now = now_ns();
        std::vector<uint32_t> dead;
        for (Ring* rp : { &main_, &poll_ })
        {
            Ring& r = *rp;
            if (r.fd < 0) continue;
            unsigned head = __atomic_load_n(r.k_sq_head, __ATOMIC_ACQUIRE);
            for (unsigned i = head; i != r.sq_tail; ++i)
            {
                uint32_t s = r.sq_slot[i & r.sq_mask];
                if (s != kNoSlot) dead.push_back(s);
            }
            r.sq_tail = head;
            r.publish();
        }
        dead.insert(dead.end(), resub_.begin(), resub_.end());
        resub_.clear();
        for (uint32_t s : dead)
        {
            if (s >= slots_.size() || !slots_[s].op) continue;
            Op* op = slots_[s].op;
            release_slot(s);
            c_.finish_op(*op, -e, now);
        }
        c_.fail_all_queued(-e, now);
    }

    // Core::mx held
    void release_slot(uint32_t s)
    {
        if (slots_[s].ring) --poll_inflight_;
        slots_[s].op = nullptr;
        ++slots_[s].gen;
        free_.push_back(s);
    }

    // Core::mx held. A read that completed and still needs its bounce copy or FADV_DONTNEED
    // goes to post: it stays in flight (its slot, its ticket, its file) until reap() did that
    // work without the lock
    void handle(const io_uring_cqe& cqe, int64_t now, std::vector<Post>& post)
    {
        uint64_t ud = cqe.user_data;
        uint32_t s = (uint32_t) (ud & 0xffffffffu) - 1u;
        uint32_t gen = (uint32_t) (ud >> 32);
        if (s >= slots_.size() || !slots_[s].op || slots_[s].gen != gen)
        {
            ++c_.stray_cqes;
            if (!stray_warned_)
            {
                stray_warned_ = true;
                warn("completion with an unknown tag ignored (this is a bug)");
            }
            return;
        }
        Op& op = *slots_[s].op;
        int res = cqe.res;
        // after a fatal ring error nothing is submitted again: what would be resumed fails
        const int broken = c_.broken.load(std::memory_order_relaxed);
        if (res >= 0)
        {
            int fe = c_.fault_errno();
            if (fe) res = -fe;
            else if (res > 1 && c_.fault_short())
            {
                int sh = op.direct ? (int) (((unsigned) res / 2) & ~(op.align - 1)) : res / 2;
                if (sh > 0) res = sh;
            }
        }
        if (res < 0)
        {
            if ((res == -EAGAIN || res == -EINTR) && op.retries < kMaxRetries && !broken)
            {
                ++op.retries;
                ++c_.resubmits;
                resub_.push_back(s);
                return;
            }
            release_slot(s);
            c_.finish_op(op, res, now);
            return;
        }
        if (res == 0 || (uint32_t) res > op.dev_len - op.got)
        {
            release_slot(s);
            c_.finish_op(op, -EIO, now);                            // EOF before the payload end
            return;
        }
        op.got += (uint32_t) res;
        if (op.got < op.need)
        {
            if (op.got % op.align || op.shorts >= kMaxShort || broken)
            {
                release_slot(s);
                c_.finish_op(op, broken ? broken : -EIO, now);
                return;
            }
            ++op.shorts;
            ++c_.resubmits;
            resub_.push_back(s);
            return;
        }
        if (op.bounce || op.dontneed)
        {
            post.push_back({ s, &op });         // reserved: no allocation under the lock
            return;
        }
        release_slot(s);
        c_.finish_op(op, (int) op.pay_len, now);
    }

    // Drain one ring's completions and process them (no submission). try_only: give up if
    // another thread is draining. Returns true if anything was processed
    bool reap(Ring& r, bool try_only)
    {
        if (!r.cq_ready()) return false;
        std::unique_lock<std::mutex> cq(cq_mx_, std::defer_lock);
        if (try_only)
        {
            if (!cq.try_lock()) return false;
        }
        else cq.lock();
        std::vector<io_uring_cqe>& buf = t_cqes;
        buf.clear();
        unsigned head = __atomic_load_n(r.k_cq_head, __ATOMIC_RELAXED);
        unsigned tail = __atomic_load_n(r.k_cq_tail, __ATOMIC_ACQUIRE);
        while (head != tail)
        {
            buf.push_back(r.cqes[head & r.cq_mask]);
            ++head;
        }
        __atomic_store_n(r.k_cq_head, head, __ATOMIC_RELEASE);
        cq.unlock();
        if (buf.empty()) return false;
        std::vector<Post>& post = t_post;
        post.clear();
        post.reserve(buf.size());
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            int64_t now = now_ns();
            for (const io_uring_cqe& e : buf) handle(e, now, post);
            if (try_only) c_.inline_reaped += buf.size();
            else c_.reaper_reaped += buf.size();
        }
        if (post.empty()) return true;
        // Page-cache invalidation and bounce copies without Core::mx, which every class-0
        // submission needs. The ops are still in flight: their slots, bounce buffers,
        // destinations and files stay theirs until they are finished below
        for (const Post& x : post)
        {
            Op& op = *x.op;
            if (op.bounce) std::memcpy(op.copy_dst, op.dst + op.pay_off, op.pay_len);
            if (op.dontneed)
                (void) ::posix_fadvise(op.fd_advise, (off_t) op.dev_off, (off_t) op.dev_len,
                                       POSIX_FADV_DONTNEED);
        }
        std::lock_guard<std::mutex> lk(c_.mx);
        int64_t now = now_ns();
        for (const Post& x : post)
        {
            release_slot(x.slot);
            c_.finish_op(*x.op, (int) x.op->pay_len, now);
        }
        post.clear();
        return true;
    }

    // A waiter processed completions: capacity freed. It submits more of its own ticket; the
    // reaper gets the rest
    void after_inline_reap(Ticket* own)
    {
        if (own)
        {
            pump(own);
            return;
        }
        bool others;
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            others = c_.queued_ops > 0 || !resub_.empty();
        }
        if (others) wake();
    }

    void reaper_main()
    {
        set_thread_name("exl3-disk-reap");
        set_thread_affinity(c_.cfg.affinity, "reaper");
        while (!stop_.load(std::memory_order_acquire))
        {
            int64_t now = now_ns();
            int64_t wait_ns = 50000000;
            if (retry_.load(std::memory_order_relaxed)) wait_ns = 1000000;
            int64_t tm = timer_.load(std::memory_order_relaxed);
            // after a fatal error nothing is admitted again: a timer left behind must not turn
            // this loop into a spin
            if (tm > 0 && !c_.broken.load(std::memory_order_relaxed))
                wait_ns = std::max<int64_t>(0, std::min(wait_ns, tm - now));
            if (!main_.cq_ready() && wait_ns > 0)
            {
                // the ring polls readable when completions are posted; the eventfd is the wake
                // for work that has none (queued tickets, resumptions)
                struct pollfd pf[2];
                pf[0].fd = main_.fd;
                pf[0].events = POLLIN;
                pf[0].revents = 0;
                pf[1].fd = evfd_;
                pf[1].events = POLLIN;
                pf[1].revents = 0;
                struct timespec ts;
                ts.tv_sec = (time_t) (wait_ns / 1000000000);
                ts.tv_nsec = (long) (wait_ns % 1000000000);
                (void) ::ppoll(pf, 2, &ts, nullptr);            // EINTR is expected
            }
            uint64_t v;
            while (::read(evfd_, &v, sizeof v) == (ssize_t) sizeof v) {}
            retry_.store(false, std::memory_order_relaxed);
            reap(main_, false);
            pump(nullptr);
        }
    }

    void poller_main()
    {
        set_thread_name("exl3-disk-poll");
        set_thread_affinity(c_.cfg.affinity, "poller");
        for (;;)
        {
            {
                std::unique_lock<std::mutex> lk(c_.mx);
                cv_poll_.wait(lk, [&] { return stop_.load() || poll_inflight_ > 0; });
                if (poll_inflight_ == 0 && stop_.load()) break;
            }
            // polls the device in the kernel until at least one read completed
            (void) sys_enter(poll_.fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
            if (reap(poll_, false)) pump(nullptr);
        }
    }

    Core& c_;
    Ring main_, poll_;
    int evfd_ = -1;
    std::mutex sq_mx_, cq_mx_, fixed_mx_;
    std::vector<Slot> slots_;                   // Core::mx
    std::vector<uint32_t> free_;                // Core::mx
    std::vector<uint32_t> resub_;               // Core::mx: slots whose op must be (re)submitted
    uint8_t* bounce_ = nullptr;
    size_t bounce_total_ = 0;
    const size_t page_ = (size_t) sysconf(_SC_PAGESIZE);
    bool buf_warned_ = false;                   // Core::mx
    bool files_reg_ = false;
    std::vector<int> fixed_free_;               // fixed_mx_
    UBuf ubuf_[kMaxUserBuffers];                // Core::mx
    int nreg_ = 0;                              // Core::mx: registered caller buffers
    std::vector<std::pair<uintptr_t, int>> sorted_;   // Core::mx: (address, index), registered only
    bool overlap_ = false;                      // Core::mx: two registered buffers overlap
    cpu_set_t aff_set_;                         // EXL3_DISK_AFFINITY (init only)
    uint64_t aff_token_ = 0;                    // 0: no affinity
    std::atomic<bool> aff_warned_ { false };
    std::atomic<uint64_t> fault_enters_ { 0 };
    std::thread reaper_, poller_;
    std::atomic<bool> stop_ { false };
    std::atomic<bool> retry_ { false };
    std::atomic<int64_t> timer_ { 0 };
    int poll_inflight_ = 0;                     // Core::mx
    std::condition_variable cv_poll_;
    bool started_ = false;
    bool stray_warned_ = false;                 // Core::mx
    std::vector<std::string> notes_;
};

}  // namespace

std::unique_ptr<Executor> make_uring(Core& c, std::string* why)
{
    if (c.cfg.fault_no_uring)
    {
        *why = "refused by EXL3_DISK_FAULT_NO_URING";
        return nullptr;
    }
    std::unique_ptr<UringExec> u(new UringExec(c));
    if (!u->init(why)) return nullptr;
    return std::unique_ptr<Executor>(u.release());
}

}  // namespace detail
}  // namespace exl3_disk

#else  // !EXL3_DISK_HAVE_URING

namespace exl3_disk
{
namespace detail
{

std::unique_ptr<Executor> make_uring(Core&, std::string* why)
{
    *why = "built without io_uring support (linux/io_uring.h from Linux 5.11 or later is needed)";
    return nullptr;
}

}  // namespace detail
}  // namespace exl3_disk

#endif  // EXL3_DISK_HAVE_URING
#endif  // __linux__
