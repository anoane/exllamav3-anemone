// io_uring backend of the disk engine, driven with raw syscalls (no liburing): a synchronous
// caller submits its own reads inline, one reaper thread submits everything else and completes,
// fixed files, registered buffers (the O_DIRECT bounce arena and caller slots), optional SQPOLL,
// optional IOPOLL ring for O_DIRECT reads on devices with poll queues. Request ownership rules:
// disk_internal.h. Design: doc/disk_engine.md.

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

        if (cfg.reg_buffers) register_buffer_table();
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
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int cpu : cfg.affinity) CPU_SET(static_cast<size_t>(cpu), &set);   // < 1024
            if (sys_register(main_.fd, IORING_REGISTER_IOWQ_AFF, &set, sizeof set) < 0)
                note("EXL3_DISK_AFFINITY for io_uring workers refused: " + errstr(errno));
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

    // A caller about to wait for t submits t's reads itself: no hand-off, and inline page-cache
    // completions it reaps itself. Anything else goes to the reaper
    void submitted(Ticket* t, bool will_wait) override
    {
        if (t && will_wait) pump(t);
        else wake();
    }

    bool wait(Ticket* t, std::unique_lock<std::mutex>& lk, int64_t deadline) override
    {
        if (t->complete.load(std::memory_order_acquire)) return true;
        // Only a waiter that stays until t completes may own (submit) t's reads
        Ticket* own = deadline < 0 ? t : nullptr;
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
        if (!buf_table_) return -1;
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
        struct iovec iov { p, n };
        io_uring_rsrc_update2 up;
        std::memset(&up, 0, sizeof up);
        up.offset = (unsigned) idx;
        up.data = (uint64_t) (uintptr_t) &iov;
        up.nr = 1;
        int rc = sys_register(main_.fd, IORING_REGISTER_BUFFERS_UPDATE, &up, sizeof up);
        int err = errno;
        std::lock_guard<std::mutex> lk(c_.mx);
        if (rc < 0)
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
            ubuf_[idx].state = 1;
        }
        struct iovec iov { nullptr, 0 };
        io_uring_rsrc_update2 up;
        std::memset(&up, 0, sizeof up);
        up.offset = (unsigned) idx;
        up.data = (uint64_t) (uintptr_t) &iov;
        up.nr = 1;
        (void) sys_register(main_.fd, IORING_REGISTER_BUFFERS_UPDATE, &up, sizeof up);
        std::lock_guard<std::mutex> lk(c_.mx);
        ubuf_[idx] = UBuf();
#else
        throw Error(EINVAL, "disk engine: no registered buffer " + std::to_string(idx));
#endif
    }

    int find_buffer(const uint8_t* p, size_t n) override
    {
        uintptr_t a = reinterpret_cast<uintptr_t>(p);
        for (int i = 1; i < kMaxUserBuffers; ++i)
        {
            const UBuf& b = ubuf_[i];
            uintptr_t base = reinterpret_cast<uintptr_t>(b.p);
            if (b.state == 2 && a >= base && a - base <= b.n && n <= b.n - (a - base)) return i;
        }
        return -1;
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
        s += bounce_ ? (bounce_reg_ ? ", registered bounce arena" : ", bounce arena") : "";
        s += buf_table_ ? ", buffer table" : "";
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

    void register_buffer_table()
    {
#if EXL3_DISK_URING_RSRC
        io_uring_rsrc_register rr;
        std::memset(&rr, 0, sizeof rr);
        rr.nr = kMaxUserBuffers;
        rr.flags = IORING_RSRC_REGISTER_SPARSE;
        if (sys_register(main_.fd, IORING_REGISTER_BUFFERS2, &rr, sizeof rr) == 0)
        {
            buf_table_ = true;
            if (bounce_)
            {
                struct iovec iov { bounce_, bounce_total_ };
                io_uring_rsrc_update2 up;
                std::memset(&up, 0, sizeof up);
                up.offset = 0;
                up.data = (uint64_t) (uintptr_t) &iov;
                up.nr = 1;
                if (sys_register(main_.fd, IORING_REGISTER_BUFFERS_UPDATE, &up, sizeof up) >= 0)
                    bounce_reg_ = true;
                else note("registering the bounce arena failed: " + errstr(errno));
            }
            return;
        }
        note("sparse buffer table refused (" + errstr(errno) + "): no registered caller buffers");
#endif
        if (bounce_)
        {
            struct iovec iov { bounce_, bounce_total_ };
            if (sys_register(main_.fd, IORING_REGISTER_BUFFERS, &iov, 1) == 0) bounce_reg_ = true;
            else note("registering the bounce arena failed: " + errstr(errno));
        }
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
        if (op.bounce && bounce_reg_)
        {
            sqe->opcode = IORING_OP_READ_FIXED;
            sqe->buf_index = 0;
        }
        else if (!op.bounce && op.buf_index >= 0)
        {
            sqe->opcode = IORING_OP_READ_FIXED;
            sqe->buf_index = (uint16_t) op.buf_index;
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
        if (op.async) sqe->flags |= IOSQE_ASYNC;
        sqe->user_data = ((uint64_t) slots_[s].gen << 32) | (uint64_t) (s + 1);
    }

    // Admit ops in class order, prepare their SQEs and submit them from this thread, which then
    // owns them. only != nullptr: that ticket's ops alone (its caller waits for it without a
    // timeout); nullptr: everything (reaper and poller threads only)
    void pump(Ticket* only)
    {
        std::lock_guard<std::mutex> sq(sq_mx_);
        bool wake_reaper = false;
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            if (c_.broken.load(std::memory_order_relaxed)) return;
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
        for (int guard = 0; guard < 64; ++guard)
        {
            unsigned head = __atomic_load_n(r.k_sq_head, __ATOMIC_ACQUIRE);
            unsigned n = r.sq_tail - head;
            if (n == 0) return;
            int rc = sys_enter(r.fd, n, 0, 0, nullptr, 0);
            if (rc > 0)
            {
                c_.enters.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            int e = rc == 0 ? EAGAIN : errno;
            if (e == EINTR) continue;
            if (e == EAGAIN || e == EBUSY || e == ENOMEM) break;
            fatal(r, e);
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

    // sq_mx_ held. The ring refused submissions for good: fail what it never consumed and
    // everything still queued, so no waiter hangs
    void fatal(Ring& r, int e)
    {
        warn("io_uring_enter failed: " + errstr(e) + "; failing the queued reads");
        std::lock_guard<std::mutex> lk(c_.mx);
        int expected = 0;
        c_.broken.compare_exchange_strong(expected, -e);
        int64_t now = now_ns();
        std::vector<uint32_t> dead;
        unsigned head = __atomic_load_n(r.k_sq_head, __ATOMIC_ACQUIRE);
        for (unsigned i = head; i != r.sq_tail; ++i)
        {
            uint32_t s = r.sq_slot[i & r.sq_mask];
            if (s != kNoSlot) dead.push_back(s);
        }
        r.sq_tail = head;
        r.publish();
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

    // Core::mx held
    void handle(const io_uring_cqe& cqe, int64_t now)
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
        if (op.bounce) std::memcpy(op.copy_dst, op.dst + op.pay_off, op.pay_len);
        if (op.dontneed)
            (void) ::posix_fadvise(op.fd_advise, (off_t) op.dev_off, (off_t) op.dev_len,
                                   POSIX_FADV_DONTNEED);
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
        std::lock_guard<std::mutex> lk(c_.mx);
        int64_t now = now_ns();
        for (const io_uring_cqe& e : buf) handle(e, now);
        if (try_only) c_.inline_reaped += buf.size();
        else c_.reaper_reaped += buf.size();
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
            if (tm > 0) wait_ns = std::max<int64_t>(0, std::min(wait_ns, tm - now));
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
    bool bounce_reg_ = false;
    bool buf_table_ = false;
    bool buf_warned_ = false;                   // Core::mx
    bool files_reg_ = false;
    std::vector<int> fixed_free_;               // fixed_mx_
    UBuf ubuf_[kMaxUserBuffers];                // Core::mx
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
