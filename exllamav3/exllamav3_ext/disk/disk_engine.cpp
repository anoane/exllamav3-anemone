// Disk I/O engine: scheduler, file table, thread-pool backends (pread, odirect) and the public
// API. The io_uring backend is in disk_uring.cpp. Design: doc/disk_engine.md.

#include "disk_engine.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>

#if defined(__linux__)

#include "disk_internal.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>
#include <thread>

#include <fcntl.h>
#include <linux/fs.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>
#include <unistd.h>

namespace exl3_disk
{
namespace detail
{

int64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static inline void cpu_relax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

static std::string errstr(int e)
{
    char buf[128];
    // GNU strerror_r returns a pointer that may or may not be buf
    const char* s = strerror_r(e, buf, sizeof buf);
    return std::string(s ? s : "unknown error") + " (errno " + std::to_string(e) + ")";
}

static void warn(const std::string& msg)
{
    std::fprintf(stderr, " !! disk engine: %s\n", msg.c_str());
    std::fflush(stderr);
}

void set_thread_name(const char* name)
{
    (void) pthread_setname_np(pthread_self(), name);
}

void set_thread_affinity(const std::vector<int>& cpus, const char* who)
{
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) if (c >= 0 && c < CPU_SETSIZE) CPU_SET(static_cast<size_t>(c), &set);
    int r = pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    if (r != 0)
    {
        static std::atomic<bool> once { false };
        if (!once.exchange(true))
            warn(std::string("EXL3_DISK_AFFINITY could not be applied to the ") + who + " thread: " +
                 errstr(r));
    }
}

static bool mul_ok(int64_t a, int64_t b, int64_t* r) { return !__builtin_mul_overflow(a, b, r); }
static bool add_ok(int64_t a, int64_t b, int64_t* r) { return !__builtin_add_overflow(a, b, r); }
static bool sub_ok(int64_t a, int64_t b, int64_t* r) { return !__builtin_sub_overflow(a, b, r); }

// a: a power of two <= 65536; v in [0, kMaxFileEnd + 2^40] (callers check against kMaxFileEnd
// and kMaxExtentBytes first), so v + a - 1 cannot overflow
static inline int64_t round_up(int64_t v, int64_t a) { return (v + a - 1) & ~(a - 1); }

// ---- files ------------------------------------------------------------------------------------

static constexpr size_t kMaxFiles = 256;
static constexpr uint64_t kSweepEvery = 64;     // acquires between checks for unlinked files

static int reopen(int fd, int flags)
{
    char path[64];
    std::snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    return ::open(path, flags | O_CLOEXEC);
}

static std::string fd_name(int fd)
{
    char path[64], target[512];
    std::snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    ssize_t n = ::readlink(path, target, sizeof target - 1);
    if (n <= 0) return "fd " + std::to_string(fd);
    target[n] = '\0';
    return std::string(target);
}

static bool read_sysfs(const std::string& path, std::string* out)
{
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    char buf[128];
    ssize_t n = ::read(fd, buf, sizeof buf - 1);
    ::close(fd);
    if (n <= 0) return false;
    buf[n] = '\0';
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    *out = s;
    return true;
}

// sysfs directory holding queue/ for the block device behind dev (the parent for a partition)
static std::string queue_dir(dev_t dev, std::string* devname)
{
    std::string base = "/sys/dev/block/" + std::to_string(major(dev)) + ":" +
                       std::to_string(minor(dev));
    char target[512];
    ssize_t n = ::readlink(base.c_str(), target, sizeof target - 1);
    if (n > 0)
    {
        target[n] = '\0';
        std::string t(target);
        size_t slash = t.find_last_of('/');
        *devname = slash == std::string::npos ? t : t.substr(slash + 1);
    }
    std::string dummy;
    if (read_sysfs(base + "/partition", &dummy)) return base + "/..";
    return base;
}

static void verify_same(int fd, const struct stat& want, const std::string& name)
{
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_dev != want.st_dev || st.st_ino != want.st_ino)
    {
        ::close(fd);
        throw Error(ESTALE, "disk engine: " + name + " changed while it was being opened");
    }
}

static void open_direct(FileEnt& e, const Config& cfg, Executor* ex)
{
    int dfd = reopen(e.fd_buf, O_RDONLY | O_DIRECT);
    if (dfd < 0)
    {
        int err = errno;
        e.fd_dir = -2;
        if (!e.direct_warned.exchange(true))
        {
            warn("O_DIRECT refused for " + e.name + " (" + errstr(err) +
                 "): reading it through the page cache");
        }
        return;
    }
    struct stat st;
    if (::fstat(dfd, &st) != 0 || st.st_dev != e.dev || st.st_ino != e.ino)
    {
        ::close(dfd);
        e.fd_dir = -2;
        return;
    }
    e.fd_dir = dfd;
    (void) cfg;
    if (ex) ex->file_opened(&e);
}

FileRef FileTable::acquire(int fd, bool want_direct, int advice, const Config& cfg, Executor* ex)
{
    if (fd < 0) throw Error(EBADF, "disk engine: invalid file descriptor " + std::to_string(fd));
    struct stat st;
    if (::fstat(fd, &st) != 0)
    {
        int e = errno;
        throw Error(e, "disk engine: fstat(fd " + std::to_string(fd) + ") failed: " + errstr(e));
    }
    bool blk = S_ISBLK(st.st_mode);
    if (!S_ISREG(st.st_mode) && !blk)
        throw Error(EINVAL, "disk engine: fd " + std::to_string(fd) +
                            " is not a regular file or a block device");

    std::lock_guard<std::mutex> lk(mx);
    Key key { st.st_dev, st.st_ino };
    auto it = map.find(key);
    FileEnt* e = nullptr;
    if (it == map.end())
    {
        std::unique_ptr<FileEnt> ne(new FileEnt());
        ne->dev = st.st_dev;
        ne->ino = st.st_ino;
        ne->is_blk = blk;
        ne->name = fd_name(fd);
        int b = reopen(fd, O_RDONLY);
        if (b < 0)
        {
            // no /proc: share the caller's open file description
            b = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
            if (b < 0)
            {
                int err = errno;
                throw Error(err, "disk engine: cannot open " + ne->name + ": " + errstr(err));
            }
        }
        verify_same(b, st, ne->name);
        ne->fd_buf = b;
        if (blk)
        {
            uint64_t sz = 0;
            if (::ioctl(b, BLKGETSIZE64, &sz) != 0 || sz > (uint64_t) INT64_MAX)
            {
                ::close(b);
                throw Error(EIO, "disk engine: cannot size block device " + ne->name);
            }
            ne->blk_size = (int64_t) sz;
        }

        // O_DIRECT alignment: statx, then the logical block size of a block device, then 4 KiB
        bool known = false;
#ifdef STATX_DIOALIGN
        struct statx sx;
        std::memset(&sx, 0, sizeof sx);
        if (::statx(b, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 &&
            (sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align)
        {
            ne->mem_align = std::max<uint32_t>(1, sx.stx_dio_mem_align);
            ne->off_align = sx.stx_dio_offset_align;
            known = true;
        }
#endif
        if (!known && blk)
        {
            int ss = 0;
            if (::ioctl(b, BLKSSZGET, &ss) == 0 && ss > 0)
            {
                ne->mem_align = (uint32_t) ss;
                ne->off_align = (uint32_t) ss;
                known = true;
            }
        }
        auto pow2 = [](uint32_t v) { return v && !(v & (v - 1)); };
        if (!known || !pow2(ne->off_align) || !pow2(ne->mem_align) || ne->off_align > 65536 ||
            ne->mem_align > 65536)
        {
            ne->mem_align = 4096;
            ne->off_align = 4096;
        }
        if (cfg.align)
        {
            if (known && cfg.align < ne->off_align)
            {
                ::close(b);
                throw Error(EINVAL, "EXL3_DISK_ALIGN=" + std::to_string(cfg.align) +
                                    " is below the O_DIRECT alignment of " + ne->name + " (" +
                                    std::to_string(ne->off_align) + ")");
            }
            ne->off_align = cfg.align;
        }
        ne->geom_align = std::max<uint32_t>(4096, ne->off_align);

        // Bulk chunk: the device's max_sectors_kb (the block layer splits there anyway)
        std::string devname;
        dev_t bdev = blk ? st.st_rdev : st.st_dev;
        std::string qdir = queue_dir(bdev, &devname);
        ne->chunk = cfg.chunk_bytes;
        if (!ne->chunk)
        {
            std::string v;
            int64_t kb = 0;
            if (read_sysfs(qdir + "/queue/max_sectors_kb", &v))
            {
                errno = 0;
                char* end = nullptr;
                long long r = std::strtoll(v.c_str(), &end, 10);
                if (errno == 0 && end && *end == '\0' && r > 0) kb = r;
            }
            int64_t c = kb ? kb * 1024 : 1280 * 1024;
            c = std::max<int64_t>(256 * 1024, std::min<int64_t>(4 << 20, c));
            ne->chunk = c & ~(int64_t) 65535;
        }
        if (ne->chunk % ne->geom_align) ne->chunk = round_up(ne->chunk, ne->geom_align);

        // Polled completions need poll queues on the device (NVMe with nvme.poll_queues > 0)
        if (cfg.iopoll && cfg.fault_force_iopoll) ne->poll_ok = true;
        else if (cfg.iopoll)
        {
            std::string v;
            if (!read_sysfs(qdir + "/queue/io_poll", &v))
                ne->poll_why = "it is not on a block device with a queue (" + qdir + ")";
            else if (v != "1")
                ne->poll_why = "device " + (devname.empty() ? qdir : devname) +
                               " has queue/io_poll = " + v + " (virtio and SCSI disks have no "
                               "poll queues; NVMe needs nvme.poll_queues > 0)";
            else ne->poll_ok = true;
        }
        e = ne.get();
        map.emplace(key, std::move(ne));
        if (ex) ex->file_opened(e);
    }
    else
    {
        e = it->second.get();
        if (e->doomed)
        {
            // forgotten while busy, and wanted again before it went idle: keep it
            e->doomed = false;
            --n_doomed;
        }
    }

    e->refs.fetch_add(1, std::memory_order_acq_rel);
    e->last_use = ++clock;
    if (map.size() > kMaxFiles) evict_locked(ex);
    else if (n_doomed > 0 || clock % kSweepEvery == 0) sweep_locked(ex);

    if (want_direct && e->fd_dir == -1) open_direct(*e, cfg, ex);
    if (cfg.iopoll && want_direct && !e->poll_ok && !e->poll_warned.exchange(true) &&
        std::find(poll_refused.begin(), poll_refused.end(), e->dev) == poll_refused.end())
    {
        poll_refused.push_back(e->dev);
        warn("EXL3_DISK_IOPOLL=1 refused for " + e->name + ": " + e->poll_why +
             "; reading it with interrupt-driven completions");
    }
    if (advice >= 0 && e->advice != advice)
    {
        (void) ::posix_fadvise(e->fd_buf, 0, 0, advice);
        e->advice = advice;
    }

    FileRef r;
    r.ent = e;
    r.size = blk ? e->blk_size : (int64_t) st.st_size;
    r.fd_buf = e->fd_buf;
    r.fd_dir = e->fd_dir >= 0 ? e->fd_dir : -1;
    r.fixed_buf = e->fixed_buf;
    r.fixed_dir = e->fixed_dir;
    r.mem_align = e->mem_align;
    r.off_align = e->off_align;
    r.geom_align = e->geom_align;
    r.chunk = e->chunk;
    r.poll_ok = e->poll_ok && cfg.iopoll;
    return r;
}

bool FileTable::acquire_recent_direct(const Config& cfg, Executor* ex, FileRef* out)
{
    std::lock_guard<std::mutex> lk(mx);
    FileEnt* e = nullptr;
    for (auto& kv : map)
    {
        FileEnt* f = kv.second.get();
        if (f->doomed || f->fd_buf < 0 || f->fd_dir == -2) continue;
        if (!e || f->last_use > e->last_use) e = f;
    }
    if (!e) return false;
    if (e->fd_dir == -1) open_direct(*e, cfg, ex);
    if (e->fd_dir < 0) return false;                   // refused: another file next time
    int64_t size = e->blk_size;
    if (!e->is_blk)
    {
        struct stat st;
        if (::fstat(e->fd_buf, &st) != 0) return false;
        size = (int64_t) st.st_size;
    }
    if (size < (int64_t) e->off_align) return false;
    e->refs.fetch_add(1, std::memory_order_acq_rel);
    FileRef r;
    r.ent = e;
    r.size = size;
    r.fd_buf = e->fd_buf;
    r.fd_dir = e->fd_dir;
    r.mem_align = e->mem_align;
    r.off_align = e->off_align;
    r.geom_align = e->geom_align;
    r.chunk = e->chunk;
    *out = r;
    return true;
}

void FileTable::sweep_doomed(Executor* ex)
{
    std::lock_guard<std::mutex> lk(mx);
    if (n_doomed > 0) sweep_locked(ex);
}

void FileTable::close_ent(FileEnt* e, Executor* ex)
{
    if (ex) ex->file_closing(e);
    if (e->fd_dir >= 0) ::close(e->fd_dir);
    if (e->fd_buf >= 0) ::close(e->fd_buf);
    e->fd_dir = e->fd_buf = -1;
}

void FileTable::evict_locked(Executor* ex)
{
    sweep_locked(ex);
    std::vector<std::pair<uint64_t, Key>> idle;
    for (auto& kv : map)
        if (kv.second->refs.load(std::memory_order_acquire) == 0)
            idle.push_back({ kv.second->last_use, kv.first });
    std::sort(idle.begin(), idle.end(),
              [](const std::pair<uint64_t, Key>& a, const std::pair<uint64_t, Key>& b)
              { return a.first < b.first; });
    size_t target = kMaxFiles * 3 / 4;
    for (auto& p : idle)
    {
        if (map.size() <= target) break;
        auto it = map.find(p.second);
        close_ent(it->second.get(), ex);
        map.erase(it);
    }
}

// Idle files that were forgotten while busy, or whose last name was removed (an unloaded or
// replaced model file must not keep its blocks allocated through the engine's descriptors)
void FileTable::sweep_locked(Executor* ex)
{
    for (auto it = map.begin(); it != map.end(); )
    {
        FileEnt* e = it->second.get();
        bool close = false;
        if (e->refs.load(std::memory_order_acquire) == 0)
        {
            struct stat st;
            if (e->doomed) close = true;
            else if (!e->is_blk && e->fd_buf >= 0 && ::fstat(e->fd_buf, &st) == 0 &&
                     st.st_nlink == 0)
                close = true;
        }
        if (!close)
        {
            ++it;
            continue;
        }
        if (e->doomed) --n_doomed;
        close_ent(e, ex);
        it = map.erase(it);
    }
}

int FileTable::forget(dev_t dev, ino_t ino, Executor* ex)
{
    std::lock_guard<std::mutex> lk(mx);
    auto it = map.find(Key { dev, ino });
    if (it == map.end()) return 0;
    FileEnt* e = it->second.get();
    if (e->refs.load(std::memory_order_acquire) != 0)
    {
        if (!e->doomed)
        {
            e->doomed = true;
            ++n_doomed;
        }
        return -1;
    }
    if (e->doomed) --n_doomed;
    close_ent(e, ex);
    map.erase(it);
    return 1;
}

void FileTable::close_all(Executor* ex)
{
    std::lock_guard<std::mutex> lk(mx);
    for (auto& kv : map)
    {
        if (kv.second->refs.load() != 0)
            warn("closing " + kv.second->name + " with references outstanding");
        close_ent(kv.second.get(), ex);
    }
    map.clear();
    n_doomed = 0;
}

int64_t FileTable::size()
{
    std::lock_guard<std::mutex> lk(mx);
    return (int64_t) map.size();
}

// ---- scheduler --------------------------------------------------------------------------------

Core::Core(const Config& c) : cfg(c)
{
    pid = ::getpid();
    for (int k = 0; k < kClasses; ++k) cap[k] = c.qd;
    cap[kEngram] = c.engram_qd;
    if (c.trace > 0) trace.resize((size_t) c.trace);
}

Ticket* Core::find(uint64_t id)
{
    auto it = tickets.find(id);
    return it == tickets.end() ? nullptr : it->second.get();
}

bool Core::hold_active(int64_t now) const
{
    return hold_pending > 0 || (hold_armed_until != 0 && now < hold_armed_until);
}

bool Core::gate(int c, int64_t bytes, bool hold) const
{
    if (inflight_total >= cfg.qd) return false;
    if (c > 0 && inflight_total >= cfg.qd - cfg.reserve0) return false;
    if (inflight[c] >= cap[c]) return false;
    int64_t w = (hold && c >= kPrefetch) ? cfg.window_lo[c] : cfg.window_hi[c];
    if (w == 0) return false;
    if (w > 0 && inflight_bytes[c] > 0 && inflight_bytes[c] + bytes > w) return false;
    return true;
}

void Core::q_insert(Ticket* t, int c, int64_t now)
{
    t->cls = c;
    t->q_cls = c;
    t->t_queued = now;
    TicketQueue& Q = q[c];
    Ticket* after = Q.tail;
    if (c == kPrefetch)
    {
        // earliest deadline first; no deadline sorts last; FIFO among equals
        auto key = [](const Ticket* x) { return x->deadline > 0 ? x->deadline : INT64_MAX; };
        int64_t k = key(t);
        while (after && key(after) > k) after = after->q_prev;
    }
    t->q_prev = after;
    t->q_next = after ? after->q_next : Q.head;
    if (t->q_next) t->q_next->q_prev = t;
    else Q.tail = t;
    if (after) after->q_next = t;
    else Q.head = t;
}

void Core::q_remove(Ticket* t)
{
    if (t->q_cls < 0) return;
    TicketQueue& Q = q[t->q_cls];
    if (t->q_prev) t->q_prev->q_next = t->q_next;
    else Q.head = t->q_next;
    if (t->q_next) t->q_next->q_prev = t->q_prev;
    else Q.tail = t->q_prev;
    t->q_prev = t->q_next = nullptr;
    t->q_cls = -1;
}

void Core::age_refill(int64_t now)
{
    if (cfg.refill_age_ms <= 0) return;
    int64_t age = (int64_t) cfg.refill_age_ms * 1000000ll;
    while (Ticket* t = q[kRefill].head)
    {
        if (now - t->t_queued < age) break;
        q_remove(t);
        t->deadline = 0;
        q_insert(t, kPrefetch, now);
        ++cc[kRefill].aged;
    }
}

int64_t Core::next_timer(int64_t now) const
{
    int64_t tm = 0;
    if (hold_pending == 0 && hold_armed_until > now && (q[kPrefetch].head || q[kRefill].head))
        tm = hold_armed_until;
    if (cfg.refill_age_ms > 0 && q[kRefill].head)
    {
        int64_t a = q[kRefill].head->t_queued + (int64_t) cfg.refill_age_ms * 1000000ll;
        if (!tm || a < tm) tm = a;
    }
    return tm;
}

Op* Core::admit_one(Ticket* only, int64_t now)
{
    if (broken.load(std::memory_order_relaxed)) return nullptr;
    age_refill(now);
    bool hold = hold_active(now);
    for (int c = 0; c < kClasses; ++c)
    {
        Ticket* t = only ? (only->q_cls == c ? only : nullptr) : q[c].head;
        if (!t) continue;
        Op& op = t->ops[t->next_issue];
        if (!gate(c, op.dev_len, hold))
        {
            ++cc[c].held;
            if (inflight_total >= cfg.qd) return nullptr;
            continue;
        }
        ++t->next_issue;
        --queued_ops;
        if (t->next_issue == t->ops.size()) q_remove(t);
        ++t->n_inflight;
        if (!t->t_first_issue) t->t_first_issue = now;
        op.cls_issued = (uint8_t) c;
        op.t_issue = now;
        ++inflight_total;
        ++inflight[c];
        inflight_bytes[c] += op.dev_len;
        cc[c].max_inflight = std::max<int64_t>(cc[c].max_inflight, inflight[c]);
        cc[c].max_inflight_bytes = std::max<int64_t>(cc[c].max_inflight_bytes, inflight_bytes[c]);
        return &op;
    }
    return nullptr;
}

bool Core::admit_span(Ticket* only, Span& sp, int64_t now)
{
    if (broken.load(std::memory_order_relaxed)) return false;
    age_refill(now);
    bool hold = hold_active(now);
    for (int c = 0; c < kClasses; ++c)
    {
        Ticket* t = only ? (only->q_cls == c ? only : nullptr) : q[c].head;
        if (!t) continue;
        size_t avail = t->ops.size() - t->next_issue;
        const Op& first = t->ops[t->next_issue];
        size_t k = 1;
        int64_t bytes = first.dev_len;
        if (first.dev_len <= kSpanOpMax && span_div > 0)
        {
            size_t lim = std::min<size_t>(kSpanMax, std::max<size_t>(1, avail / (size_t) span_div));
            lim = std::min(lim, avail);
            while (k < lim && t->ops[t->next_issue + k].dev_len <= kSpanOpMax)
            {
                bytes = std::max<int64_t>(bytes, t->ops[t->next_issue + k].dev_len);
                ++k;
            }
        }
        if (!gate(c, bytes, hold))
        {
            ++cc[c].held;
            if (inflight_total >= cfg.qd) return false;
            continue;
        }
        sp.t = t;
        sp.first = t->next_issue;
        sp.count = k;
        sp.cls = c;
        sp.bytes = bytes;
        for (size_t i = 0; i < k; ++i)
        {
            t->ops[sp.first + i].cls_issued = (uint8_t) c;
            t->ops[sp.first + i].t_issue = now;
        }
        t->next_issue += k;
        queued_ops -= (int64_t) k;
        if (t->next_issue == t->ops.size()) q_remove(t);
        t->n_inflight += k;
        if (!t->t_first_issue) t->t_first_issue = now;
        ++inflight_total;
        ++inflight[c];
        inflight_bytes[c] += bytes;
        cc[c].max_inflight = std::max<int64_t>(cc[c].max_inflight, inflight[c]);
        cc[c].max_inflight_bytes = std::max<int64_t>(cc[c].max_inflight_bytes, inflight_bytes[c]);
        return true;
    }
    return false;
}

void Core::op_done(Op& op, int res, int64_t now)
{
    Ticket* t = op.t;
    int c = op.cls_issued;
    op.res = res;
    op.t_done = now;
    --t->n_inflight;
    ++t->n_finished;
    Counters& k = cc[c];
    ++k.ops;
    if (res < 0)
    {
        ++k.op_errors;
        if (!t->err) t->err = res;
    }
    else k.bytes += op.pay_len;
    ++k.hq[hist_bucket_of(op.t_issue - t->t_submit)];
    ++k.hs[hist_bucket_of(now - op.t_issue)];
    if (!trace.empty())
    {
        TraceRec& r = trace[trace_pos];
        r.ticket = t->id;
        r.cls = c;
        r.res = res;
        r.dev_off = op.dev_off;
        r.dev_len = op.dev_len;
        r.t_submit = t->t_submit;
        r.t_issue = op.t_issue;
        r.t_done = now;
        if (++trace_pos == trace.size())
        {
            trace_pos = 0;
            trace_wrapped = true;
        }
    }
    if (t->n_finished == t->ops.size()) complete_ticket(t, now);
}

void Core::op_done_inline(Ticket* t, size_t n, int64_t now)
{
    Counters& k = cc[t->cls];
    for (size_t i = 0; i < n; ++i)
    {
        Op& op = t->ops[i];
        op.cls_issued = (uint8_t) t->cls;
        ++k.ops;
        k.bytes += op.pay_len;
        ++k.hq[0];                                  // never queued
        ++k.hs[hist_bucket_of(op.t_done - op.t_issue)];
        if (!t->t_first_issue || op.t_issue < t->t_first_issue) t->t_first_issue = op.t_issue;
    }
    (void) now;
    t->next_issue = n;
    t->n_finished = n;
}

void Core::unit_done(int c, int64_t bytes)
{
    --inflight_total;
    --inflight[c];
    inflight_bytes[c] -= bytes;
    if (stopping && inflight_total == 0) cv_drained.notify_all();
}

void Core::finish_op(Op& op, int res, int64_t now)
{
    int c = op.cls_issued;
    int64_t b = op.dev_len;
    op_done(op, res, now);
    unit_done(c, b);
}

void Core::finish_span(const Span& sp, int64_t now)
{
    (void) now;
    int c = sp.cls;
    int64_t b = sp.bytes;
    Ticket* t = sp.t;
    for (size_t i = 0; i < sp.count; ++i)
    {
        Op& op = t->ops[sp.first + i];
        op_done(op, op.res, op.t_done);       // the last one may complete (and hand back) t
    }
    unit_done(c, b);
}

void Core::complete_ticket(Ticket* t, int64_t now)
{
    t->t_done = now;
    last_io_ns.store(now, std::memory_order_relaxed);
    if (t->hold && --hold_pending == 0) hold_armed_until = 0;
    for (FileEnt* f : t->files) FileTable::unref(f);
    t->files.clear();
    for (int b : t->buffers) if (ex) ex->buffer_ref(b, -1);
    t->buffers.clear();
    Counters& k = cc[t->cls_submit];
    ++k.tickets;
    // a synchronous caller's inline reads started before the ticket reached the scheduler
    int64_t start = t->t_first_issue && t->t_first_issue < t->t_submit ? t->t_first_issue
                                                                       : t->t_submit;
    ++k.ht[hist_bucket_of(now - start)];
    if (t->flag) __atomic_store_n(t->flag, t->flag_value, __ATOMIC_RELEASE);
    t->complete.store(true, std::memory_order_release);
    t->cv.notify_all();
}

void Core::cancel_queued(Ticket* t, int err, int64_t now)
{
    q_remove(t);
    size_t n = t->ops.size() - t->next_issue;
    if (!n) return;
    for (size_t i = t->next_issue; i < t->ops.size(); ++i) t->ops[i].res = err;
    t->next_issue = t->ops.size();
    queued_ops -= (int64_t) n;
    t->n_finished += n;
    cc[t->cls].ops_cancelled += n;
    if (!t->err) t->err = err;
    if (t->n_finished == t->ops.size()) complete_ticket(t, now);
}

void Core::fail_all_queued(int err, int64_t now)
{
    for (int c = 0; c < kClasses; ++c)
        while (Ticket* t = q[c].head) cancel_queued(t, err, now);
}

// Fault injection draws: a hash of a counter, so a fault hits one attempt in N on average
// without lining up with the order in which concurrent reads complete (a plain n % N would
// hit the same read again and again when completions rotate among N reads)
static uint64_t fault_draw(std::atomic<uint64_t>& ctr, uint64_t salt)
{
    uint64_t x = ctr.fetch_add(1, std::memory_order_relaxed) + salt;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

int Core::fault_errno()
{
    if (!cfg.fault_eio && !cfg.fault_eintr) return 0;
    uint64_t h = fault_draw(fault_ctr, 0x9e3779b97f4a7c15ull);
    if (cfg.fault_eintr && h % (uint64_t) cfg.fault_eintr == 0) return EINTR;
    if (cfg.fault_eio && (h >> 20) % (uint64_t) cfg.fault_eio == 0) return EIO;
    return 0;
}

bool Core::fault_short()
{
    if (!cfg.fault_short) return false;
    return fault_draw(fault_ctr, 0x7f4a7c159e3779b9ull) % (uint64_t) cfg.fault_short == 0;
}

// ---- thread-pool backends (pread, odirect) -----------------------------------------------------

namespace
{

struct Bounce
{
    uint8_t* p = nullptr;
    ~Bounce() { std::free(p); }
    uint8_t* get()
    {
        if (!p)
        {
            void* q = nullptr;
            if (posix_memalign(&q, 4096, kBounceBytes) != 0) return nullptr;
            p = static_cast<uint8_t*>(q);
        }
        return p;
    }
};

thread_local Bounce t_bounce;

class PoolExec final : public Executor
{
public:
    explicit PoolExec(Core& c) : c_(c) {}

    ~PoolExec() override { stop(); }

    void start()
    {
        int n = c_.cfg.threads;
        threads_.reserve((size_t) n);
        try
        {
            for (int i = 0; i < n; ++i) threads_.emplace_back(&PoolExec::worker, this, i);
        }
        catch (...)
        {
            stop();
            throw;
        }
    }

    const char* name() const override { return backend_name(c_.cfg.backend); }

    void submitted(Ticket*, bool) override {}

    // A caller that waits reads what the page cache holds itself, without waking anyone
    // (RWF_NOWAIT: EAGAIN for a page that is not cached, after the kernel started reading it).
    // Only for decode-sized tickets, where waking workers costs more than the copies; larger
    // warm tickets go faster across the pool. It stops once misses dominate, so a cold batch
    // reaches the workers after a handful of probes
    size_t prepass(Ticket* t) override
    {
#ifdef RWF_NOWAIT
        if (c_.cfg.fault_eio || c_.cfg.fault_short || c_.cfg.fault_eintr || c_.cfg.fault_delay_us)
            return 0;                   // fault injection exercises the normal path only
        if (t->ops.size() > kPrepassMaxOps) return 0;
        size_t done = 0, misses = 0;
        for (Op& op : t->ops)
        {
            if (op.direct) continue;
            if (misses > kPrepassMisses && misses > done) break;
            op.t_issue = now_ns();
            while (op.got < op.need)
            {
                struct iovec iov { op.dst + op.got, (size_t) (op.dev_len - op.got) };
                ssize_t r = ::preadv2(op.fd, &iov, 1, (off_t) (op.dev_off + op.got), RWF_NOWAIT);
                if (r <= 0 || (size_t) r > iov.iov_len) break;   // EAGAIN, EOF, errors: normal path
                op.got += (uint32_t) r;
            }
            if (op.got < op.need)
            {
                ++misses;
                continue;
            }
            if (op.dontneed)
                (void) ::posix_fadvise(op.fd_advise, (off_t) op.dev_off, (off_t) op.dev_len,
                                       POSIX_FADV_DONTNEED);
            op.res = (int) op.pay_len;
            op.t_done = now_ns();
            ++done;
        }
        if (!done) return 0;
        // every op has need >= 1, and direct ops were not touched
        std::stable_partition(t->ops.begin(), t->ops.end(),
                              [](const Op& o) { return !o.direct && o.got >= o.need; });
        return done;
#else
        (void) t;
        return 0;
#endif
    }

    // Wake a few workers; each one that finds work wakes the next (spread_wake), so the
    // submitter pays a few futex wakes instead of one per idle thread (waking all 24 at once
    // cost it ~70 us here), and the pool still fans out within a few wake-ups
    void kick(std::unique_lock<std::mutex>&) override
    {
        int64_t n = std::min<int64_t>({ idle_, c_.queued_ops, (int64_t) kWakeFirst });
        for (int64_t i = 0; i < n; ++i) cv_work_.notify_one();
    }

    bool wait(Ticket* t, std::unique_lock<std::mutex>& lk, int64_t deadline) override
    {
        const int64_t spin = (int64_t) c_.cfg.spin_us * 1000;
        bool spun = false;
        while (!t->complete.load(std::memory_order_acquire))
        {
            int64_t now = now_ns();
            if (deadline >= 0 && now >= deadline) return false;
            Span sp;
            if (c_.admit_span(t, sp, now))
            {
                // the caller reads its own ops while it waits, as the original gather does
                lk.unlock();
                run_span(sp);
                lk.lock();
                c_.finish_span(sp, now_ns());
                spread_wake();
                continue;
            }
            if (spin > 0 && !spun)
            {
                spun = true;
                lk.unlock();
                int64_t end = now + spin;
                if (deadline >= 0) end = std::min(end, deadline);
                while (!t->complete.load(std::memory_order_acquire) && now_ns() < end) cpu_relax();
                lk.lock();
                continue;
            }
            if (deadline >= 0)
                t->cv.wait_until(lk, std::chrono::steady_clock::time_point(
                                         std::chrono::nanoseconds(deadline)));
            else t->cv.wait(lk);
        }
        return true;
    }

    void stop() override
    {
        {
            std::lock_guard<std::mutex> lk(c_.mx);
            stop_ = true;
            cv_work_.notify_all();
        }
        for (auto& th : threads_) if (th.joinable()) th.join();
        threads_.clear();
    }

    std::string describe() const override
    {
        return std::string(name()) + " pool: " + std::to_string(c_.cfg.threads) + " threads";
    }

private:
    static constexpr int kWakeFirst = 4;
    static constexpr size_t kPrepassMaxOps = 1024;
    static constexpr size_t kPrepassMisses = 4;

    // Core::mx held: more work queued and a worker asleep -> wake one
    void spread_wake()
    {
        if (c_.queued_ops > 0 && idle_ > 0) cv_work_.notify_one();
    }

    void worker(int idx)
    {
        char nm[16];
        std::snprintf(nm, sizeof nm, "exl3-disk-%d", idx);
        set_thread_name(nm);
        set_thread_affinity(c_.cfg.affinity, "pool");
        std::unique_lock<std::mutex> lk(c_.mx);
        while (!stop_)
        {
            int64_t now = now_ns();
            Span sp;
            if (c_.admit_span(nullptr, sp, now))
            {
                spread_wake();
                lk.unlock();
                run_span(sp);
                lk.lock();
                c_.finish_span(sp, now_ns());
                spread_wake();
                continue;
            }
            int64_t tm = c_.next_timer(now);
            ++idle_;
            if (tm > now) cv_work_.wait_for(lk, std::chrono::nanoseconds(tm - now));
            else cv_work_.wait(lk);
            --idle_;
        }
    }

    // No locks held: the span's ops belong to this thread until finish_span
    void run_span(const Span& sp)
    {
        for (size_t i = 0; i < sp.count; ++i)
        {
            Op& op = sp.t->ops[sp.first + i];
            op.res = run_op(op);
        }
    }

    int run_op(Op& op)
    {
        if (c_.cfg.fault_delay_us > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(c_.cfg.fault_delay_us));
        uint8_t* buf = op.dst;
        if (op.bounce)
        {
            buf = t_bounce.get();
            if (!buf)
            {
                op.t_done = now_ns();
                return -ENOMEM;
            }
        }
        int res = read_fully(op, buf);
        if (res == 0)
        {
            if (op.bounce) std::memcpy(op.copy_dst, buf + op.pay_off, op.pay_len);
            if (op.dontneed)
                (void) ::posix_fadvise(op.fd_advise, (off_t) op.dev_off, (off_t) op.dev_len,
                                       POSIX_FADV_DONTNEED);
            res = (int) op.pay_len;
        }
        op.t_done = now_ns();
        return res;
    }

    int read_fully(Op& op, uint8_t* buf)
    {
        int again = 0, shorts = 0;
        while (op.got < op.need)
        {
            size_t want = (size_t) (op.dev_len - op.got);
            off_t off = (off_t) (op.dev_off + op.got);
            ssize_t r;
            int fe = c_.fault_errno();
            if (fe)
            {
                r = -1;
                errno = fe;
            }
            else
            {
#ifdef RWF_HIPRI
                if (op.poll)
                {
                    struct iovec iov { buf + op.got, want };
                    r = ::preadv2(op.fd, &iov, 1, off, RWF_HIPRI);
                }
                else
#endif
                r = ::pread(op.fd, buf + op.got, want, off);
                if (r > 1 && c_.fault_short())
                {
                    ssize_t s = op.direct ? (ssize_t) ((r / 2) & ~(ssize_t) (op.align - 1)) : r / 2;
                    if (s > 0) r = s;
                }
            }
            if (r < 0)
            {
                int e = errno;
                if ((e == EINTR || e == EAGAIN) && ++again <= kMaxRetries) continue;
                return -e;
            }
            if (r == 0) return -EIO;                     // EOF before the payload end
            if ((size_t) r > want) return -EIO;          // cannot happen; never trust it
            op.got += (uint32_t) r;
            if (op.got < op.need)
            {
                if (op.got % op.align || ++shorts > kMaxShort) return -EIO;
            }
        }
        return 0;
    }

    Core& c_;
    std::vector<std::thread> threads_;
    std::condition_variable cv_work_;
    int64_t idle_ = 0;
    bool stop_ = false;
};

}  // namespace

std::unique_ptr<Executor> make_pool(Core& c)
{
    std::unique_ptr<PoolExec> p(new PoolExec(c));
    p->start();
    return std::unique_ptr<Executor>(p.release());
}

// ---- request building -------------------------------------------------------------------------

namespace
{

int advice_for(Fadvise f, bool rows)
{
    switch (f)
    {
        case Fadvise::Auto: return rows ? POSIX_FADV_RANDOM : -1;
        case Fadvise::None: return -1;
        case Fadvise::Random: return POSIX_FADV_RANDOM;
        case Fadvise::Normal: return POSIX_FADV_NORMAL;
        case Fadvise::Sequential: return POSIX_FADV_SEQUENTIAL;
    }
    return -1;
}

void check_cls(int cls)
{
    if (cls < 0 || cls >= kClasses)
        throw Error(EINVAL, "disk engine: class " + std::to_string(cls) + " is not in [0, 3]");
}

// Drops the file references of a ticket that never got queued. It refers to the owning
// pointer, not to the ticket: enqueue() moves the ticket into the table only after its last
// throw point, so on every unwind the ticket is still owned here (and alive) when this runs,
// and after a successful enqueue the pointer is empty and there is nothing to drop. Declared
// after the pointer, so it runs first.
struct RefGuard
{
    std::unique_ptr<Ticket>& t;
    ~RefGuard()
    {
        if (!t) return;
        for (FileEnt* f : t->files) FileTable::unref(f);
        t->files.clear();
    }
};

thread_local std::vector<Op> t_ops_cache;   // reused op storage for back-to-back calls

Op base_op(Ticket* t, const FileRef& r, bool direct)
{
    Op op;
    op.t = t;
    op.f = r.ent;
    op.direct = direct ? 1 : 0;
    op.fd = direct ? r.fd_dir : r.fd_buf;
    op.fixed = direct ? r.fixed_dir : r.fixed_buf;
    op.fd_advise = r.fd_buf;
    op.align = direct ? std::max(r.mem_align, r.off_align) : 1;
    op.poll = direct && r.poll_ok ? 1 : 0;
    return op;
}

// File bytes [off, off + len) to dst: buffered in chunks, or O_DIRECT through bounce slots
void build_segment(std::vector<Op>& ops, Ticket* t, const FileRef& r, bool direct, int64_t off,
                   int64_t len, uint8_t* dst)
{
    if (!direct)
    {
        while (len > 0)
        {
            int64_t piece = std::min(len, r.chunk);
            Op op = base_op(t, r, false);
            op.dst = dst;
            op.dev_off = off;
            op.dev_len = (uint32_t) piece;
            op.need = (uint32_t) piece;
            op.pay_len = (uint32_t) piece;
            ops.push_back(op);
            off += piece;
            dst += piece;
            len -= piece;
        }
        return;
    }
    const int64_t A = r.off_align;
    const int64_t P = (int64_t) kBounceBytes - 2 * A;    // payload per bounce slot
    while (len > 0)
    {
        int64_t piece = std::min(len, P);
        int64_t a0 = off & ~(A - 1);
        int64_t a1 = round_up(off + piece, A);
        Op op = base_op(t, r, true);
        op.bounce = 1;
        op.copy_dst = dst;
        op.dev_off = a0;
        op.dev_len = (uint32_t) (a1 - a0);
        op.pay_off = (uint32_t) (off - a0);
        op.pay_len = (uint32_t) piece;
        op.need = (uint32_t) (op.pay_off + piece);
        ops.push_back(op);
        off += piece;
        dst += piece;
        len -= piece;
    }
}

bool ranges_overlap(std::vector<std::pair<uintptr_t, uintptr_t>>& v)
{
    std::sort(v.begin(), v.end());
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i].first < v[i - 1].second) return true;
    return false;
}

std::atomic<bool> g_forked { false };

void install_fork_handler()
{
    static std::once_flag once;
    std::call_once(once, []
    {
        (void) pthread_atfork(nullptr, nullptr, +[] { g_forked.store(true); });
    });
}

bool forked_child(pid_t owner)
{
    return g_forked.load(std::memory_order_relaxed) && ::getpid() != owner;
}

// Every Engine entry point: a forked child must not touch the parent's engine at all (its
// mutexes may have been held by threads that do not exist in the child)
void check_not_forked(pid_t owner)
{
    if (forked_child(owner))
        throw Error(ECHILD, "disk engine: an engine does not survive fork(); this process is a "
                            "child of the one that created it");
}

}  // namespace

static void stop_keepalive(Core& c)
{
    {
        std::lock_guard<std::mutex> lk(c.ka_mx);
        c.ka_stop = true;
    }
    c.ka_cv.notify_all();
    if (c.ka_thread.joinable()) c.ka_thread.join();
}

// One small O_DIRECT read of the most recently used file (its first aligned block; the device
// is all that needs waking, which block does not matter), unless the engine has reads in
// flight or queued. It bypasses the scheduler: one block per period cannot crowd anything.
static void keepalive_ping(Core& c, uint8_t* buf)
{
    {
        std::lock_guard<std::mutex> lk(c.mx);
        if (c.stopping || c.inflight_total > 0 || c.queued_ops > 0) return;
    }
    FileRef r;
    if (!c.files.acquire_recent_direct(c.cfg, c.ex.get(), &r)) return;
    size_t len = std::max<size_t>(512, r.off_align);   // off_align <= 65536: fits the buffer
    ssize_t n;
    do n = ::pread(r.fd_dir, buf, len, 0);
    while (n < 0 && errno == EINTR);
    FileTable::unref(r.ent);
    c.files.sweep_doomed(c.ex.get());       // forgotten while this read held it: close it now
    c.last_io_ns.store(now_ns(), std::memory_order_relaxed);
    if (n >= 0) c.keepalive_reads.fetch_add(1, std::memory_order_relaxed);
    // a failed read is not reported here: the next real read of the file reports its error
}

static void keepalive_loop(Core& c)
{
    set_thread_name("exl3-disk-ka");
    set_thread_affinity(c.cfg.affinity, "keep-alive");
    const int64_t period = (int64_t) c.cfg.keepalive_ms * 1000000;
    const int64_t idle = (int64_t) c.cfg.keepalive_idle_s * 1000000000;
    void* mem = nullptr;
    if (::posix_memalign(&mem, 65536, 65536) != 0)
    {
        warn("EXL3_DISK_KEEPALIVE_MS: no memory for the read buffer; keep-alive off");
        return;
    }
    std::unique_ptr<void, decltype(&std::free)> hold(mem, &std::free);
    std::unique_lock<std::mutex> lk(c.ka_mx);
    while (!c.ka_stop)
    {
        int64_t now = now_ns();
        int64_t last = c.last_submit_ns.load();
        if (last == 0 || now - last > idle)
        {
            // Nothing submitted lately: let the device sleep until the next submission, which
            // clears ka_dormant. The stores to ka_dormant and last_submit_ns and the loads of
            // the other are sequentially consistent on both sides, so a submission racing with
            // this is seen by one of them.
            c.ka_dormant.store(true);
            last = c.last_submit_ns.load();
            if (last != 0 && now_ns() - last <= idle)
            {
                c.ka_dormant.store(false);
                continue;
            }
            c.ka_cv.wait(lk, [&] { return c.ka_stop || !c.ka_dormant.load(); });
            continue;
        }
        int64_t since = now - c.last_io_ns.load(std::memory_order_relaxed);
        if (since >= period)
        {
            lk.unlock();
            keepalive_ping(c, static_cast<uint8_t*>(mem));
            lk.lock();
            since = 0;
        }
        c.ka_cv.wait_for(lk, std::chrono::nanoseconds(std::max<int64_t>(period - since, 100000)),
                         [&] { return c.ka_stop; });
    }
}

// Nothing may escape a std::thread (std::terminate): a failure (an allocation, opening the
// O_DIRECT descriptor) turns the keep-alive off and leaves the engine as it was. The loop drops
// every file reference before anything that can throw, and ka_mx is released by unwinding.
static void keepalive_main(Core* cp)
{
    try
    {
        keepalive_loop(*cp);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, " !! disk engine: EXL3_DISK_KEEPALIVE_MS: %s; keep-alive off\n", e.what());
    }
    catch (...)
    {
        std::fprintf(stderr, " !! disk engine: EXL3_DISK_KEEPALIVE_MS: unknown error; keep-alive off\n");
    }
}

// Every submission: stamp the clocks, and wake a dormant keep-alive
static void keepalive_touch(Core& c)
{
    if (!c.cfg.keepalive_ms) return;
    int64_t now = now_ns();
    c.last_submit_ns.store(now);
    c.last_io_ns.store(now, std::memory_order_relaxed);
    if (c.ka_dormant.load())
    {
        {
            std::lock_guard<std::mutex> lk(c.ka_mx);
            c.ka_dormant.store(false);
        }
        c.ka_cv.notify_one();
    }
}

static void shutdown_core(Core& c)
{
    stop_keepalive(c);
    {
        std::unique_lock<std::mutex> lk(c.mx);
        c.stopping = true;
        c.fail_all_queued(-ECANCELED, now_ns());
        while (c.inflight_total > 0)
        {
            if (c.cv_drained.wait_for(lk, std::chrono::seconds(10)) == std::cv_status::timeout &&
                c.inflight_total > 0)
                warn("shutdown is waiting for " + std::to_string(c.inflight_total) +
                     " reads still in flight (their buffers must outlive them)");
        }
    }
    if (c.ex) c.ex->stop();
    c.files.close_all(c.ex.get());
    c.ex.reset();
}

}  // namespace detail

using namespace detail;

// ---- Engine -----------------------------------------------------------------------------------

Engine::Engine(const Config& cfg)
{
    install_fork_handler();
    core_.reset(new Core(cfg));
    Core& c = *core_;
    for (const std::string& n : cfg.notes) warn(n);
    if (cfg.backend == Backend::IoUring)
    {
        std::string why;
        c.ex = make_uring(c, &why);
        if (!c.ex)
        {
            std::string msg = "io_uring unavailable (" + why + "): using the thread pool";
            c.fallbacks.push_back(msg);
            warn(msg);
            c.cfg.backend = (cfg.direct_rows || cfg.direct_extents) ? Backend::ODirect
                                                                      : Backend::Pread;
        }
    }
    if (!c.ex)
    {
        c.span_div = 2 * (cfg.threads + 1);
        c.ex = make_pool(c);
    }
    if (cfg.verbose)
        std::printf(" -- disk engine: %s | %s\n", c.cfg.describe().c_str(),
                    c.ex->describe().c_str());
    // Last: nothing after it can throw, so an unwinding constructor never destroys a joinable
    // std::thread
    if (cfg.keepalive_ms > 0)
    {
        try
        {
            c.ka_thread = std::thread(keepalive_main, &c);
        }
        catch (const std::exception& e)
        {
            std::fprintf(stderr, " !! disk engine: EXL3_DISK_KEEPALIVE_MS: cannot start the "
                                 "thread (%s); keep-alive off\n", e.what());
        }
    }
}

Engine::~Engine()
{
    if (!core_) return;
    if (forked_child(core_->pid))
    {
        // The threads and the ring belong to the parent: leave everything alone
        (void) core_.release();
        return;
    }
    shutdown_core(*core_);
}

static void check_usable(Core& c)
{
    check_not_forked(c.pid);
    int b = c.broken.load();
    if (b) throw Error(-b, "disk engine: stopped after a fatal I/O error: " + errstr(-b));
}

// Queue a fully built ticket. On success t is empty (the ticket table owns it); on a throw t
// still owns the ticket, nothing of it was queued and no reference was taken (the caller's
// RefGuard drops the file references). Everything that can throw happens before the ticket
// moves into the table; everything after is noexcept.
static TicketId enqueue(Core& c, std::unique_ptr<Ticket>& t, bool will_wait)
{
    Ticket* tp = t.get();
    size_t pre = will_wait ? c.ex->prepass(tp) : 0;
    // at most one registered buffer per destination range, and at most kMaxUserBuffers
    tp->buffers.reserve(std::min<size_t>(tp->dst.size(), kMaxUserBuffers));
    std::unique_lock<std::mutex> lk(c.mx);
    if (c.stopping) throw Error(ESHUTDOWN, "disk engine: shutting down");
    int b = c.broken.load();
    if (b) throw Error(-b, "disk engine: stopped after a fatal I/O error: " + errstr(-b));
    uint64_t id = c.next_id;
    auto slot = c.tickets.emplace(id, nullptr);       // may throw: t is untouched
    if (!slot.second) throw Error(EEXIST, "disk engine: ticket id reused");   // 64-bit counter
    ++c.next_id;
    slot.first->second = std::move(t);               // noexcept from here on
    int64_t now = now_ns();
    tp->id = id;
    tp->t_submit = now;     // when the scheduler saw it (the hold rule starts here)
    // io_uring: destinations inside a registered buffer are read with READ_FIXED. One lookup
    // per destination range (a table's output, an extent's slot), and none while no buffer
    // is registered. (A prepass reorders ops; only the pools have one, and they register no
    // buffers.)
    if (pre == 0 && c.ex->user_buffers() > 0)
    {
        for (const DstRange& d : tp->dst)
        {
            int bi = c.ex->find_buffer(d.p, d.n);
            if (bi < 0) continue;
            for (size_t i = d.op_begin; i < d.op_end; ++i)
                if (!tp->ops[i].bounce) tp->ops[i].buf_index = bi;
            if (std::find(tp->buffers.begin(), tp->buffers.end(), bi) == tp->buffers.end())
            {
                // distinct buffers <= ranges, so this stays within the capacity reserved above
                tp->buffers.push_back(bi);
                c.ex->buffer_ref(bi, 1);
            }
        }
    }
    if (c.cfg.uring_async > 0 && tp->ops.size() >= (size_t) c.cfg.uring_async)
        for (Op& op : tp->ops) if (!op.direct) op.async = 1;
    if (tp->hold) ++c.hold_pending;
    if (pre) c.op_done_inline(tp, pre, now);
    if (pre == tp->ops.size()) c.complete_ticket(tp, now);
    else
    {
        c.q_insert(tp, tp->cls, now);
        c.queued_ops += (int64_t) (tp->ops.size() - pre);
        c.ex->kick(lk);
    }
    bool queued = tp->q_cls >= 0;
    // io_uring: only a class-0 caller submits its own reads (read here: cls changes under mx)
    bool submit_inline = will_wait && tp->cls == kEngram;
    lk.unlock();
    if (queued)
    {
        // tp stays valid: nobody knows its id yet. The ticket is queued, so a failure to
        // submit here only delays it: the reaper admits queued work on its own (<= 50 ms)
        try
        {
            c.ex->submitted(tp, submit_inline);
        }
        catch (...)
        {
        }
    }
    return id;
}

static std::unique_ptr<Ticket> new_ticket(const Options& o)
{
    check_cls(o.cls);
    if (o.hold && o.cls >= kPrefetch)
        throw Error(EINVAL, "disk engine: hold applies to class 0 and 1 tickets only (a class " +
                            std::to_string(o.cls) + " ticket would be held by its own rule)");
    if (o.flag && (reinterpret_cast<uintptr_t>(o.flag) & 3))
        throw Error(EINVAL, "disk engine: the completion flag must be 4-byte aligned");
    std::unique_ptr<Ticket> t(new Ticket());
    t->cls = t->cls_submit = o.cls;
    t->hold = o.hold;
    t->deadline = o.deadline_ns > 0 ? o.deadline_ns : 0;
    t->flag = o.flag;
    t->flag_value = o.flag_value;
    t->ops.swap(t_ops_cache);
    t->ops.clear();
    return t;
}

static TicketId submit_rows_impl(Core& c, const int64_t* uids, int64_t n, int64_t uid_base,
                                 const RowTable* tables, int ntables, const Options& o,
                                 bool will_wait)
{
    check_usable(c);
    keepalive_touch(c);
    if (n < 0 || (n > 0 && !uids)) throw Error(EINVAL, "disk engine: bad row id list");
    if (ntables < 1 || ntables > kMaxTables || !tables)
        throw Error(EINVAL, "disk engine: between 1 and " + std::to_string(kMaxTables) +
                            " tables per call");
    std::unique_ptr<Ticket> t = new_ticket(o);
    RefGuard guard { t };
    int advice = advice_for(c.cfg.fadvise, true);
    const bool want_direct = o.direct < 0 ? c.cfg.direct_rows : o.direct == 1;

    std::vector<FileRef> refs((size_t) ntables);
    std::vector<std::pair<uintptr_t, uintptr_t>> outs;
    for (int k = 0; k < ntables; ++k)
    {
        const RowTable& tb = tables[k];
        std::string what = "table " + std::to_string(k);
        if (tb.row_bytes <= 0 || tb.row_bytes > kMaxOpBytes)
            throw Error(EINVAL, "disk engine: " + what + ": row_bytes " +
                                std::to_string(tb.row_bytes) + " out of range");
        if (tb.base_offset < 0)
            throw Error(EINVAL, "disk engine: " + what + ": negative base offset");
        int64_t need = 0;
        if (!mul_ok(n, tb.row_bytes, &need) || (n > 0 && !tb.out) || tb.out_bytes < need)
            throw Error(EINVAL, "disk engine: " + what + ": output holds " +
                                std::to_string(tb.out_bytes) + " bytes, " + std::to_string(n) +
                                " rows need more");
        refs[(size_t) k] = c.files.acquire(tb.fd, want_direct, advice, c.cfg, c.ex.get());
        t->files.push_back(refs[(size_t) k].ent);
        if (n > 0)
            outs.push_back({ reinterpret_cast<uintptr_t>(tb.out),
                             reinterpret_cast<uintptr_t>(tb.out) + (uintptr_t) need });
    }
    if (ranges_overlap(outs)) throw Error(EINVAL, "disk engine: output buffers overlap");

    t->ops.reserve((size_t) n * (size_t) ntables);
    t->dst.reserve((size_t) ntables);
    for (int k = 0; k < ntables; ++k)
    {
        const RowTable& tb = tables[k];
        const FileRef& r = refs[(size_t) k];
        size_t first_op = t->ops.size();
        bool direct = want_direct && r.fd_dir >= 0;
        if (direct && (r.off_align > kBounceBytes / 4 || r.mem_align > 4096))
        {
            direct = false;
            FileEnt* e = r.ent;
            if (!e->bounce_warned.exchange(true))
            {
                warn("O_DIRECT rows of " + e->name + " need " + std::to_string(r.off_align) +
                     "-byte alignment, more than the bounce slots allow: reading them buffered");
            }
        }
        int64_t i = 0;
        while (i < n)
        {
            int64_t j = i + 1;
            while (j < n && uids[j - 1] < INT64_MAX && uids[j] == uids[j - 1] + 1) ++j;
            int64_t row = 0, off = 0, len = 0, end = 0;
            if (!sub_ok(uids[i], uid_base, &row) || row < 0)
                throw Error(ERANGE, "disk engine: table " + std::to_string(k) + ": row id " +
                                    std::to_string(uids[i]) + " is below uid_base " +
                                    std::to_string(uid_base));
            if (!mul_ok(row, tb.row_bytes, &off) || !add_ok(off, tb.base_offset, &off) ||
                !mul_ok(j - i, tb.row_bytes, &len) || !add_ok(off, len, &end) || end > r.size ||
                end > kMaxFileEnd)
                // the ids as given (their difference to uid_base may not fit in int64_t)
                throw Error(ERANGE, "disk engine: table " + std::to_string(k) + ": row ids " +
                                    std::to_string(uids[i]) + ".." + std::to_string(uids[j - 1]) +
                                    " (uid_base " + std::to_string(uid_base) +
                                    ") end beyond the file (" + std::to_string(r.size) +
                                    " bytes)");
            build_segment(t->ops, t.get(), r, direct, off, len, tb.out + i * tb.row_bytes);
            i = j;
        }
        if (n > 0)
            t->dst.push_back({ first_op, t->ops.size(), tb.out,
                               (size_t) n * (size_t) tb.row_bytes });
    }
    return enqueue(c, t, will_wait);
}

static TicketId submit_extents_impl(Core& c, const ExtentReq* ex, int64_t n, const Options& o,
                                    int64_t* payload_offsets, bool will_wait)
{
    check_usable(c);
    keepalive_touch(c);
    if (n < 0 || (n > 0 && !ex)) throw Error(EINVAL, "disk engine: bad extent list");
    std::unique_ptr<Ticket> t = new_ticket(o);
    RefGuard guard { t };
    int advice = advice_for(c.cfg.fadvise, false);
    const bool want_direct = o.direct < 0 ? c.cfg.direct_extents : o.direct == 1;

    std::vector<std::pair<int, FileRef>> seen;    // per-call cache: fd -> file
    std::vector<std::pair<uintptr_t, uintptr_t>> slots;
    for (int64_t i = 0; i < n; ++i)
    {
        const ExtentReq& e = ex[i];
        std::string what = "extent " + std::to_string(i);
        if (e.offset < 0 || e.length < 0 || e.length > kMaxExtentBytes)
            throw Error(EINVAL, "disk engine: " + what + ": bad offset or length");
        const FileRef* r = nullptr;
        for (auto& s : seen) if (s.first == e.fd) { r = &s.second; break; }
        if (!r)
        {
            seen.push_back({ e.fd, c.files.acquire(e.fd, want_direct, advice, c.cfg, c.ex.get()) });
            t->files.push_back(seen.back().second.ent);
            r = &seen.back().second;
        }
        int64_t G = r->geom_align;
        int64_t delta = e.offset & (G - 1);
        int64_t end = 0, span = 0;
        if (!add_ok(e.offset, e.length, &end) || end > r->size || end > kMaxFileEnd)
            throw Error(ERANGE, "disk engine: " + what + " [" + std::to_string(e.offset) + ", +" +
                                std::to_string(e.length) + ") ends beyond the file (" +
                                std::to_string(r->size) + " bytes)");
        if (!add_ok(delta, e.length, &span)) throw Error(ERANGE, "disk engine: " + what);
        span = round_up(span, G);
        if (payload_offsets) payload_offsets[i] = delta;
        if (e.length == 0) continue;
        uintptr_t sa = reinterpret_cast<uintptr_t>(e.slot);
        uintptr_t malign = std::max<uintptr_t>(4096, r->mem_align);
        if (!e.slot || (sa & (malign - 1)))
            throw Error(EINVAL, "disk engine: " + what + ": slot must be " +
                                std::to_string(malign) + "-byte aligned");
        if (e.slot_bytes < span)
            throw Error(EINVAL, "disk engine: " + what + ": slot holds " +
                                std::to_string(e.slot_bytes) + " bytes, the extent spans " +
                                std::to_string(span));
        slots.push_back({ sa, sa + (uintptr_t) span });
        size_t first_op = t->ops.size();
        t->dst.push_back({ first_op, first_op, e.slot, (size_t) span });   // end set below

        bool direct = want_direct && r->fd_dir >= 0;
        if (!direct)
        {
            // same geometry as O_DIRECT, so the payload offset does not depend on the backend
            build_segment(t->ops, t.get(), *r, false, e.offset, e.length, e.slot + delta);
            if (c.cfg.extent_dontneed)
                for (size_t k = first_op; k < t->ops.size(); ++k) t->ops[k].dontneed = 1;
            t->dst.back().op_end = t->ops.size();
            continue;
        }
        // O_DIRECT: aligned superset [a0, aend) straight into the slot, in chunks
        const int64_t A = r->off_align;
        int64_t a0 = e.offset - delta;
        int64_t aend = round_up(end, A);
        for (int64_t d = a0; d < end; d += r->chunk)
        {
            int64_t dl = std::min<int64_t>(r->chunk, aend - d);
            Op op = base_op(t.get(), *r, true);
            op.dst = e.slot + (d - a0);
            op.dev_off = d;
            op.dev_len = (uint32_t) dl;
            op.need = (uint32_t) std::min<int64_t>(dl, end - d);
            op.pay_len = (uint32_t) std::max<int64_t>(
                0, std::min<int64_t>(d + dl, end) - std::max<int64_t>(d, e.offset));
            t->ops.push_back(op);
        }
        t->dst.back().op_end = t->ops.size();
    }
    if (ranges_overlap(slots)) throw Error(EINVAL, "disk engine: extent slots overlap");
    return enqueue(c, t, will_wait);
}

TicketId Engine::submit_rows(const int64_t* uids, int64_t n, int64_t uid_base,
                             const RowTable* tables, int ntables, const Options& o)
{
    return submit_rows_impl(*core_, uids, n, uid_base, tables, ntables, o, false);
}

TicketId Engine::submit_extents(const ExtentReq* ex, int64_t n, const Options& o,
                                int64_t* payload_offsets)
{
    return submit_extents_impl(*core_, ex, n, o, payload_offsets, false);
}

ExtentGeom Engine::extent_geometry(int fd, int64_t offset, int64_t length)
{
    Core& c = *core_;
    check_usable(c);
    // the same bounds as submit_extents: every rounding below then stays inside int64_t
    if (offset < 0 || length < 0 || length > kMaxExtentBytes || offset > kMaxFileEnd)
        throw Error(EINVAL, "disk engine: bad offset or length");
    FileRef r = c.files.acquire(fd, false, -1, c.cfg, c.ex.get());
    FileTable::unref(r.ent);
    ExtentGeom g;
    int64_t G = r.geom_align;
    g.payload_offset = offset & (G - 1);
    g.span_bytes = round_up(g.payload_offset + length, G);   // < 2^40 + 2^17
    return g;
}

namespace
{
// A thread inside Engine::wait / release for a ticket (Core::mx held on construction and
// destruction): the ticket is not destroyed before it leaves
struct Waiter
{
    Ticket* t;
    explicit Waiter(Ticket* x) : t(x) { ++t->waiters; }
    ~Waiter()
    {
        if (--t->waiters == 0 && t->releasing) t->cv.notify_all();
    }
    Waiter(const Waiter&) = delete;
    Waiter& operator=(const Waiter&) = delete;
};
}

int Engine::wait(TicketId id, int64_t timeout_ns)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    std::unique_lock<std::mutex> lk(c.mx);
    Ticket* t = c.find(id);
    if (!t) throw Error(ENOENT, "disk engine: unknown ticket " + std::to_string(id));
    if (!t->complete.load(std::memory_order_acquire))
    {
        Waiter w(t);        // ex->wait may drop and retake the lock; t outlives it
        int64_t deadline = timeout_ns < 0 ? -1 : now_ns() + timeout_ns;
        if (!c.ex->wait(t, lk, deadline)) return -ETIMEDOUT;
    }
    return t->err;
}

bool Engine::done(TicketId id)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    std::lock_guard<std::mutex> lk(c.mx);
    Ticket* t = c.find(id);
    if (!t) throw Error(ENOENT, "disk engine: unknown ticket " + std::to_string(id));
    return t->complete.load(std::memory_order_acquire);
}

TicketTimes Engine::times(TicketId id)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    std::lock_guard<std::mutex> lk(c.mx);
    Ticket* t = c.find(id);
    if (!t) throw Error(ENOENT, "disk engine: unknown ticket " + std::to_string(id));
    TicketTimes r;
    r.t_submit = t->t_submit;
    r.t_first_issue = t->t_first_issue;
    r.t_done = t->complete.load(std::memory_order_acquire) ? t->t_done : 0;
    r.ops = (int64_t) t->ops.size();
    return r;
}

void Engine::release(TicketId id)
{
    Core& c = *core_;
    if (forked_child(c.pid)) return;
    std::unique_ptr<Ticket> owned;
    {
        std::unique_lock<std::mutex> lk(c.mx);
        Ticket* t = c.find(id);
        if (!t) return;
        if (t->releasing)
        {
            // another thread releases it: return once it completed (no read lands after this
            // returns), and let that thread destroy it
            Waiter w(t);
            if (!t->complete.load(std::memory_order_acquire)) c.ex->wait(t, lk, -1);
            return;
        }
        t->releasing = true;
        c.cancel_queued(t, -ECANCELED, now_ns());
        {
            Waiter w(t);
            if (!t->complete.load(std::memory_order_acquire)) c.ex->wait(t, lk, -1);
        }
        // threads still inside wait() / release() for it leave once they see it complete
        while (t->waiters > 0) t->cv.wait(lk);
        auto it = c.tickets.find(id);
        if (it == c.tickets.end() || it->second.get() != t) return;   // cannot happen
        owned = std::move(it->second);
        c.tickets.erase(it);
    }
    if (t_ops_cache.capacity() < owned->ops.capacity() && owned->ops.capacity() <= (1u << 22))
    {
        owned->ops.clear();
        t_ops_cache.swap(owned->ops);
    }
}

void Engine::cancel(TicketId id)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    std::lock_guard<std::mutex> lk(c.mx);
    Ticket* t = c.find(id);
    if (!t) throw Error(ENOENT, "disk engine: unknown ticket " + std::to_string(id));
    c.cancel_queued(t, -ECANCELED, now_ns());
}

void Engine::promote(TicketId id, int cls)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    check_cls(cls);
    {
        std::unique_lock<std::mutex> lk(c.mx);
        Ticket* t = c.find(id);
        if (!t) throw Error(ENOENT, "disk engine: unknown ticket " + std::to_string(id));
        if (cls >= t->cls) return;
        if (t->q_cls >= 0)
        {
            c.q_remove(t);
            c.q_insert(t, cls, now_ns());
            c.ex->kick(lk);
        }
        else t->cls = cls;
    }
    c.ex->submitted(nullptr, false);
}

void Engine::arm_hold()
{
    Core& c = *core_;
    check_not_forked(c.pid);
    std::lock_guard<std::mutex> lk(c.mx);
    if (c.cfg.hold_arm_ms > 0) c.hold_armed_until = now_ns() + (int64_t) c.cfg.hold_arm_ms * 1000000;
}

namespace
{
struct Releaser
{
    Engine* e;
    TicketId id;
    ~Releaser() { e->release(id); }
};
}

int Engine::gather_rows(const int64_t* uids, int64_t n, int64_t uid_base,
                        const RowTable* tables, int ntables, const Options& o,
                        int64_t* t_submitted)
{
    TicketId id = submit_rows_impl(*core_, uids, n, uid_base, tables, ntables, o, true);
    if (t_submitted) *t_submitted = now_ns();
    Releaser rel { this, id };
    return wait(id, -1);
}

int Engine::read_extents(const ExtentReq* ex, int64_t n, const Options& o,
                         int64_t* payload_offsets, int64_t* t_submitted)
{
    TicketId id = submit_extents_impl(*core_, ex, n, o, payload_offsets, true);
    if (t_submitted) *t_submitted = now_ns();
    Releaser rel { this, id };
    return wait(id, -1);
}

int Engine::register_buffer(void* p, size_t n)
{
    Core& c = *core_;
    check_usable(c);
    if (!p || !n) throw Error(EINVAL, "disk engine: empty buffer");
    return c.ex->register_buffer(p, n);
}

void Engine::unregister_buffer(int idx)
{
    Core& c = *core_;
    check_usable(c);
    c.ex->unregister_buffer(idx);
}

int Engine::find_registered(const void* p, size_t n)
{
    Core& c = *core_;
    check_usable(c);
    std::lock_guard<std::mutex> lk(c.mx);
    if (c.ex->user_buffers() == 0) return -1;
    return c.ex->find_buffer(static_cast<const uint8_t*>(p), n);
}

int Engine::forget(int fd)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    struct stat st;
    if (fd < 0 || ::fstat(fd, &st) != 0)
        throw Error(EBADF, "disk engine: forget: invalid file descriptor " + std::to_string(fd));
    return c.files.forget(st.st_dev, st.st_ino, c.ex.get());
}

int Engine::forget_path(const std::string& path)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    struct stat st;
    if (::stat(path.c_str(), &st) != 0)
    {
        int e = errno;
        if (e == ENOENT) return 0;          // gone: an unlinked file is swept once idle
        throw Error(e, "disk engine: forget: stat(" + path + ") failed: " + errstr(e));
    }
    return c.files.forget(st.st_dev, st.st_ino, c.ex.get());
}

Stats Engine::stats(bool reset)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    Stats s;
    s.files_open = c.files.size();
    std::lock_guard<std::mutex> lk(c.mx);
    for (int k = 0; k < kClasses; ++k)
    {
        Counters& x = c.cc[k];
        ClassStats& y = s.cls[k];
        y.tickets = x.tickets;
        y.ops = x.ops;
        y.op_errors = x.op_errors;
        y.ops_cancelled = x.ops_cancelled;
        y.bytes = x.bytes;
        y.held = x.held;
        y.aged = x.aged;
        y.max_inflight = x.max_inflight;
        y.max_inflight_bytes = x.max_inflight_bytes;
        y.hist_queue.assign(x.hq, x.hq + kHistBuckets);
        y.hist_service.assign(x.hs, x.hs + kHistBuckets);
        y.hist_ticket.assign(x.ht, x.ht + kHistBuckets);
        if (reset) x = Counters();
    }
    s.inline_reaped = c.inline_reaped;
    s.reaper_reaped = c.reaper_reaped;
    s.enters = c.enters.load();
    s.resubmits = c.resubmits;
    s.fixed_plain = c.fixed_plain;
    s.stray_cqes = c.stray_cqes;
    s.inflight_now = c.inflight_total;
    s.queued_ops_now = c.queued_ops;
    s.tickets_live = (int64_t) c.tickets.size();
    s.keepalive_reads = c.keepalive_reads.load();
    if (reset)
    {
        c.inline_reaped = c.reaper_reaped = c.resubmits = c.fixed_plain = 0;
        c.enters.store(0);
        c.keepalive_reads.store(0);
    }
    return s;
}

std::vector<TraceRec> Engine::trace(bool clear)
{
    Core& c = *core_;
    check_not_forked(c.pid);
    std::lock_guard<std::mutex> lk(c.mx);
    std::vector<TraceRec> out;
    if (c.trace.empty()) return out;
    if (c.trace_wrapped)
        out.insert(out.end(), c.trace.begin() + (ptrdiff_t) c.trace_pos, c.trace.end());
    out.insert(out.end(), c.trace.begin(), c.trace.begin() + (ptrdiff_t) c.trace_pos);
    if (clear)
    {
        c.trace_pos = 0;
        c.trace_wrapped = false;
    }
    return out;
}

const Config& Engine::config() const { return core_->cfg; }

std::string Engine::describe() const
{
    Core& c = *core_;
    std::string s = c.cfg.describe() + " | " + (c.ex ? c.ex->describe() : std::string("stopped"));
    for (const std::string& f : c.fallbacks) s += " | " + f;
    return s;
}

std::vector<std::string> Engine::fallbacks() const { return core_->fallbacks; }

// ---- process-wide default engine --------------------------------------------------------------

namespace
{
std::mutex g_mx;
std::shared_ptr<Engine>* g_engine = nullptr;     // leaked on purpose: never destroyed at exit
pid_t g_engine_pid = 0;
std::atomic<int> g_route { -1 };                 // -1 unknown, 0 original gather, 1 engine
thread_local int t_class = kEngram;

// g_mx is held across fork(), so a child never inherits it locked by a thread it does not
// have (it only guards a pointer swap and engine creation, never anything that waits for the
// forking thread)
void install_default_fork_handler()
{
    static std::once_flag once;
    std::call_once(once, []
    {
        (void) pthread_atfork(+[] { g_mx.lock(); }, +[] { g_mx.unlock(); },
                              +[] { g_mx.unlock(); });
    });
}
}

std::shared_ptr<Engine> default_engine_if_created()
{
    install_default_fork_handler();
    std::lock_guard<std::mutex> lk(g_mx);
    if (g_engine && g_engine_pid == ::getpid()) return *g_engine;
    return nullptr;
}

std::shared_ptr<Engine> default_engine()
{
    install_default_fork_handler();
    std::lock_guard<std::mutex> lk(g_mx);
    pid_t me = ::getpid();
    if (g_engine && g_engine_pid != me) g_engine = nullptr;   // forked child: leave it alone
    if (g_engine && *g_engine) return *g_engine;
    Config cfg = config_from_env();
    std::shared_ptr<Engine> e = std::make_shared<Engine>(cfg);
    if (!g_engine) g_engine = new std::shared_ptr<Engine>();
    *g_engine = e;
    g_engine_pid = me;
    return e;
}

std::shared_ptr<Engine> configure_default(const Overrides& ov)
{
    install_default_fork_handler();
    Config cfg = config_from_env(ov);
    std::shared_ptr<Engine> e = std::make_shared<Engine>(cfg);
    std::shared_ptr<Engine> old;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        pid_t me = ::getpid();
        if (g_engine && g_engine_pid != me) g_engine = nullptr;
        if (!g_engine) g_engine = new std::shared_ptr<Engine>();
        old = std::move(*g_engine);
        *g_engine = e;
        g_engine_pid = me;
        g_route.store(cfg.ngram_engine ? 1 : 0);
    }
    old.reset();    // outside the lock: waits for the old engine's reads in flight
    return e;
}

void shutdown_default()
{
    install_default_fork_handler();
    std::shared_ptr<Engine> old;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        if (g_engine && g_engine_pid == ::getpid()) old = std::move(*g_engine);
        g_route.store(-1);
    }
    old.reset();
}

bool ngram_route_engine()
{
    int r = g_route.load(std::memory_order_acquire);
    if (r < 0)
    {
        // Only EXL3_DISK_BACKEND decides the route; the other knobs are read when the engine
        // is created
        r = ngram_engine_from_env() ? 1 : 0;
        int expected = -1;
        g_route.compare_exchange_strong(expected, r);
        r = g_route.load();
    }
    return r == 1;
}

int thread_class() { return t_class; }

int set_thread_class(int cls)
{
    check_cls(cls);
    int prev = t_class;
    t_class = cls;
    return prev;
}

static thread_local int t_direct = -1;

int thread_direct() { return t_direct; }

int set_thread_direct(int direct)
{
    if (direct < -1 || direct > 1) throw Error(EINVAL, "disk engine: direct must be -1, 0 or 1");
    int prev = t_direct;
    t_direct = direct;
    return prev;
}

int ngram_gather(int fd, int64_t base_offset, int64_t row_bytes, const int64_t* uids, int64_t n,
                 int64_t uid_base, uint8_t* out, int64_t out_bytes)
{
    std::shared_ptr<Engine> e = default_engine();
    RowTable tb { fd, base_offset, row_bytes, out, out_bytes };
    Options o;
    o.cls = t_class;
    o.direct = t_direct;
    return e->gather_rows(uids, n, uid_base, &tb, 1, o);
}

}  // namespace exl3_disk

#else  // !__linux__: the engine is Linux-only

namespace exl3_disk
{

namespace detail { struct Core {}; }

[[noreturn]] static void unsupported()
{
    throw Error(ENOSYS, "the disk engine is Linux-only (Windows keeps the overlapped ReadFile "
                        "n-gram gather)");
}

Engine::Engine(const Config&) { unsupported(); }
Engine::~Engine() {}
TicketId Engine::submit_rows(const int64_t*, int64_t, int64_t, const RowTable*, int,
                             const Options&) { unsupported(); }
TicketId Engine::submit_extents(const ExtentReq*, int64_t, const Options&, int64_t*)
{ unsupported(); }
ExtentGeom Engine::extent_geometry(int, int64_t, int64_t) { unsupported(); }
int Engine::wait(TicketId, int64_t) { unsupported(); }
bool Engine::done(TicketId) { unsupported(); }
TicketTimes Engine::times(TicketId) { unsupported(); }
void Engine::release(TicketId) {}
void Engine::cancel(TicketId) { unsupported(); }
void Engine::promote(TicketId, int) { unsupported(); }
void Engine::arm_hold() { unsupported(); }
int Engine::gather_rows(const int64_t*, int64_t, int64_t, const RowTable*, int, const Options&,
                        int64_t*)
{ unsupported(); }
int Engine::read_extents(const ExtentReq*, int64_t, const Options&, int64_t*, int64_t*)
{ unsupported(); }
int Engine::register_buffer(void*, size_t) { unsupported(); }
void Engine::unregister_buffer(int) { unsupported(); }
int Engine::find_registered(const void*, size_t) { unsupported(); }
int Engine::forget(int) { return 0; }
int Engine::forget_path(const std::string&) { return 0; }
Stats Engine::stats(bool) { unsupported(); }
std::vector<TraceRec> Engine::trace(bool) { unsupported(); }
const Config& Engine::config() const { unsupported(); }
std::string Engine::describe() const { unsupported(); }
std::vector<std::string> Engine::fallbacks() const { unsupported(); }

std::shared_ptr<Engine> default_engine() { unsupported(); }
std::shared_ptr<Engine> default_engine_if_created() { return nullptr; }
std::shared_ptr<Engine> configure_default(const Overrides&) { unsupported(); }
void shutdown_default() {}
bool ngram_route_engine() { return false; }
static thread_local int t_class = kEngram;
int thread_class() { return t_class; }
int set_thread_class(int cls)
{
    if (cls < 0 || cls >= kClasses) throw Error(EINVAL, "disk engine: class out of range");
    int p = t_class;
    t_class = cls;
    return p;
}
static thread_local int t_direct = -1;
int thread_direct() { return t_direct; }
int set_thread_direct(int direct)
{
    if (direct < -1 || direct > 1) throw Error(EINVAL, "disk engine: direct must be -1, 0 or 1");
    int p = t_direct;
    t_direct = direct;
    return p;
}
int ngram_gather(int, int64_t, int64_t, const int64_t*, int64_t, int64_t, uint8_t*, int64_t)
{ unsupported(); }

}  // namespace exl3_disk

#endif
