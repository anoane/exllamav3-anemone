// Standalone test driver for the disk engine (exllamav3_ext/disk), no torch needed.
//
//   tests/disk_engine/build.sh            builds release, ASan/UBSan and TSan variants and runs them
//   disk_engine_test --dir DIR [--quick] [--only NAME] [--seed N]
//
// Everything is checked byte for byte against a reference pread on the caller's own descriptor:
// rows (1 to 3 tables, any row size, sorted / unsorted / duplicate ids, runs, rows ending at
// EOF), extents (unaligned head and tail, ending at EOF, multi-chunk), with guard bytes around
// every destination. Scheduler tests read the engine's trace: priorities, windows, the reserve,
// the hold rule, deadlines, aging, cancellation, promotion. Fault injection covers EIO, short
// reads and EINTR in every backend; real faults cover closed and reused descriptors, bad ranges,
// a file truncated under queued reads, and engine shutdown with reads in flight.

#include "disk/disk_engine.h"
#include "disk/disk_auto.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__SANITIZE_THREAD__)
#define EXL3_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define EXL3_TEST_TSAN 1
#endif
#endif
#ifndef EXL3_TEST_TSAN
#define EXL3_TEST_TSAN 0
#endif

// GCC's AddressSanitizer runtime (GCC 13's libasan) does not take its allocator's locks around
// fork(). A child that allocates can then block forever on a lock that one of the parent's
// other threads held at the fork: fork-default hung in about half of the GCC 13 ASan runs, and
// every hung child sat in the sanitizer's own allocator lock (SizeClassAllocator64 ->
// Mutex::Lock, from operator new), never in the engine. Clang's runtime takes those locks
// around fork (clang 19: 20 of 20 runs clean), so the test runs under clang's ASan
// (CXX=clang++ tests/disk_engine/build.sh).
#if defined(__SANITIZE_ADDRESS__) && !defined(__clang__)
#define EXL3_TEST_GCC_ASAN 1
#else
#define EXL3_TEST_GCC_ASAN 0
#endif

using namespace exl3_disk;

namespace
{

std::atomic<int> g_fail { 0 };      // CHECK runs on the concurrency tests' threads too
std::atomic<int> g_checks { 0 };
bool g_quick = false;
std::string g_dir;
uint64_t g_seed = 12345;

#define CHECK(cond, ...)                                                                          \
    do                                                                                            \
    {                                                                                             \
        ++g_checks;                                                                               \
        if (!(cond))                                                                              \
        {                                                                                         \
            ++g_fail;                                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);                  \
            std::fprintf(stderr, __VA_ARGS__);                                                    \
            std::fprintf(stderr, "\n");                                                           \
        }                                                                                         \
    } while (0)

int64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

const int64_t kWait = 60ll * 1000000000ll;     // a hang is a failure, not a stuck test

// Wait with a bound; a timeout aborts the run (releasing would block on the stuck read)
int wait_or_die(Engine& e, TicketId id, const char* what)
{
    int r = e.wait(id, kWait);
    if (r == -ETIMEDOUT)
    {
        std::fprintf(stderr, "FAIL: %s: ticket %llu did not complete in 60 s\n", what,
                     (unsigned long long) id);
        std::_Exit(3);
    }
    return r;
}

struct File
{
    std::string path;
    int fd = -1;
    int64_t size = 0;
};

File make_file(const std::string& name, int64_t size, uint64_t seed)
{
    File f;
    f.path = g_dir + "/" + name;
    f.size = size;
    int fd = ::open(f.path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0)
    {
        std::perror(f.path.c_str());
        std::exit(2);
    }
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> buf(1 << 17);
    int64_t done = 0;
    while (done < size)
    {
        for (auto& w : buf) w = rng();
        int64_t n = std::min<int64_t>(size - done, (int64_t) (buf.size() * 8));
        ssize_t w = ::pwrite(fd, buf.data(), (size_t) n, done);
        if (w != n)
        {
            std::perror("pwrite");
            std::exit(2);
        }
        done += n;
    }
    ::fsync(fd);
    f.fd = fd;
    return f;
}

void ref_read(int fd, int64_t off, int64_t len, uint8_t* dst)
{
    int64_t got = 0;
    while (got < len)
    {
        ssize_t r = ::pread(fd, dst + got, (size_t) (len - got), off + got);
        if (r <= 0)
        {
            std::fprintf(stderr, "reference pread failed at %lld\n", (long long) (off + got));
            std::exit(2);
        }
        got += r;
    }
}

// Destination with guard bytes on both sides
struct Guarded
{
    static constexpr size_t kGuard = 4096;
    std::vector<uint8_t> mem;
    uint8_t* p = nullptr;
    size_t n = 0;
    uint8_t* base = nullptr;
    explicit Guarded(size_t bytes, size_t align = 4096)
    {
        n = bytes;
        void* q = nullptr;
        if (posix_memalign(&q, align, bytes + 2 * kGuard) != 0) std::abort();
        base = static_cast<uint8_t*>(q);
        std::memset(base, 0xA5, bytes + 2 * kGuard);
        p = base + kGuard;
    }
    ~Guarded() { std::free(base); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    bool guards_ok() const
    {
        for (size_t i = 0; i < kGuard; ++i)
            if (base[i] != 0xA5 || p[n + i] != 0xA5) return false;
        return true;
    }
};

Config cfg_of(const Overrides& ov)
{
    return config_from_env(ov);
}

int count_fds()
{
    int n = 0;
    DIR* d = opendir("/proc/self/fd");
    if (!d) return -1;
    while (dirent* e = readdir(d)) if (e->d_name[0] != '.') ++n;
    closedir(d);
    return n;
}

// Descriptors of this process whose target contains `needle` (e.g. a path, "(deleted)")
int count_fds_to(const std::string& needle)
{
    int n = 0;
    DIR* d = opendir("/proc/self/fd");
    if (!d) return -1;
    while (dirent* e = readdir(d))
    {
        if (e->d_name[0] == '.') continue;
        char link[600], target[600];
        std::snprintf(link, sizeof link, "/proc/self/fd/%s", e->d_name);
        ssize_t k = ::readlink(link, target, sizeof target - 1);
        if (k <= 0) continue;
        target[k] = '\0';
        if (std::strstr(target, needle.c_str())) ++n;
    }
    closedir(d);
    return n;
}

// Threads of this process named exactly `name` (comm), as tids
std::vector<int> threads_named(const char* name)
{
    std::vector<int> v;
    DIR* d = opendir("/proc/self/task");
    if (!d) return v;
    while (dirent* e = readdir(d))
    {
        if (e->d_name[0] == '.') continue;
        std::string p = std::string("/proc/self/task/") + e->d_name + "/comm";
        FILE* f = std::fopen(p.c_str(), "r");
        if (!f) continue;
        char buf[64] = {};
        if (std::fgets(buf, sizeof buf, f))
        {
            size_t l = std::strlen(buf);
            if (l && buf[l - 1] == '\n') buf[l - 1] = '\0';
            if (std::strcmp(buf, name) == 0) v.push_back(std::atoi(e->d_name));
        }
        std::fclose(f);
    }
    closedir(d);
    return v;
}

// A thread named `name` that is not in `before` (threads name themselves when they start)
int new_thread_named(const char* name, const std::vector<int>& before)
{
    for (int i = 0; i < 1000; ++i)
    {
        for (int t : threads_named(name))
            if (std::find(before.begin(), before.end(), t) == before.end()) return t;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 0;
}

// utime + stime of a thread of this process, in clock ticks (-1 if it is gone)
long thread_ticks(int tid)
{
    std::string p = "/proc/self/task/" + std::to_string(tid) + "/stat";
    FILE* f = std::fopen(p.c_str(), "r");
    if (!f) return -1;
    char buf[1024] = {};
    size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    const char* q = std::strrchr(buf, ')');      // comm may hold spaces
    if (!q) return -1;
    long ut = 0, st = 0;
    // after ")": state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
    if (std::sscanf(q + 1, " %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %ld %ld", &ut, &st) != 2)
        return -1;
    return ut + st;
}

std::string task_status_field(int tid, const char* field)
{
    std::string p = "/proc/self/task/" + std::to_string(tid) + "/status";
    FILE* f = std::fopen(p.c_str(), "r");
    if (!f) return "";
    char line[512];
    std::string r;
    size_t fl = std::strlen(field);
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, field, fl) == 0 && line[fl] == ':')
        {
            const char* v = line + fl + 1;
            while (*v == ' ' || *v == '\t') ++v;
            r = v;
            while (!r.empty() && (r.back() == '\n' || r.back() == ' ')) r.pop_back();
            break;
        }
    std::fclose(f);
    return r;
}

// The extent slot geometry the engine must use for a file: G = max(4096, O_DIRECT offset
// alignment), the alignment from EXL3_DISK_ALIGN or, like the engine, statx
int64_t geom_of(const Engine& e, int fd)
{
    int64_t a = e.config().align;
    if (!a)
    {
        a = 4096;
#ifdef STATX_DIOALIGN
        struct statx sx;
        std::memset(&sx, 0, sizeof sx);
        if (::statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 &&
            (sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align &&
            !(sx.stx_dio_offset_align & (sx.stx_dio_offset_align - 1)) &&
            sx.stx_dio_offset_align <= 65536 && sx.stx_dio_mem_align <= 65536)
            a = sx.stx_dio_offset_align;
#else
        (void) fd;
#endif
    }
    return std::max<int64_t>(4096, a);
}

// ---- row gathers -----------------------------------------------------------------------------

struct RowCase
{
    int64_t row_bytes;
    int64_t base;
    std::vector<int64_t> uids;
    int64_t uid_base;
};

std::vector<int64_t> make_ids(std::mt19937_64& rng, int64_t n, int64_t max_row, int kind)
{
    std::vector<int64_t> v;
    if (max_row <= 0 || n <= 0) return v;
    std::uniform_int_distribution<int64_t> d(0, max_row - 1);
    for (int64_t i = 0; i < n; ++i) v.push_back(d(rng));
    switch (kind)
    {
        case 0: // sorted unique (the n-gram contract)
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            break;
        case 1: // unsorted with duplicates
            for (int64_t i = 0; i + 1 < (int64_t) v.size(); i += 5) v[(size_t) i + 1] = v[(size_t) i];
            break;
        case 2: // consecutive runs
        {
            std::vector<int64_t> r;
            for (int64_t x : v)
            {
                int64_t len = 1 + (x % 7);
                for (int64_t k = 0; k < len && x + k < max_row; ++k) r.push_back(x + k);
                if ((int64_t) r.size() >= n) break;
            }
            std::sort(r.begin(), r.end());
            r.erase(std::unique(r.begin(), r.end()), r.end());
            v = r;
            break;
        }
        case 3: // the last rows of the file (EOF)
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            v.push_back(max_row - 1);
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            break;
    }
    return v;
}

void check_rows(Engine& e, const std::vector<File*>& files, std::mt19937_64& rng, int ntables,
                int kind, int64_t row_bytes, int cls, bool async_flag, const char* what)
{
    std::vector<RowTable> tabs;
    std::vector<std::unique_ptr<Guarded>> outs;
    std::vector<int64_t> bases;
    std::vector<File*> tf;
    int64_t max_rows = INT64_MAX;
    for (int t = 0; t < ntables; ++t)
    {
        File* f = files[(size_t) (rng() % files.size())];
        int64_t base = (int64_t) (rng() % 4099);
        if (base + row_bytes > f->size) base = 0;
        int64_t rows = (f->size - base) / row_bytes;
        if (kind == 3) base = f->size - rows * row_bytes;   // last row ends exactly at EOF
        max_rows = std::min(max_rows, rows);
        bases.push_back(base);
        tf.push_back(f);
    }
    if (max_rows <= 0) return;
    int64_t n = 1 + (int64_t) (rng() % (g_quick ? 200 : 1200));
    if (row_bytes > 100000) n = std::min<int64_t>(n, 20);
    std::vector<int64_t> ids = make_ids(rng, n, max_rows, kind);
    int64_t uid_base = (int64_t) (rng() % 1000) - 500;
    for (auto& x : ids) x += uid_base;
    for (int t = 0; t < ntables; ++t)
    {
        outs.emplace_back(new Guarded((size_t) ((int64_t) ids.size() * row_bytes), 64));
        tabs.push_back({ tf[(size_t) t]->fd, bases[(size_t) t], row_bytes, outs.back()->p,
                         (int64_t) outs.back()->n });
    }
    Options o;
    o.cls = cls;
    uint32_t flag alignas(4) = 0;
    if (async_flag)
    {
        o.flag = &flag;
        o.flag_value = 0xC0FFEEu;
    }
    if (!async_flag && rng() % 2)
    {
        // synchronous path (the pool reads page-cache hits inline first); sometimes cold
        if (rng() % 3 == 0)
            for (File* f : tf) (void) ::posix_fadvise(f->fd, 0, 0, POSIX_FADV_DONTNEED);
        int r = e.gather_rows(ids.data(), (int64_t) ids.size(), uid_base, tabs.data(), ntables, o);
        CHECK(r == 0, "%s: sync gather failed %d", what, r);
    }
    else
    {
    TicketId id = e.submit_rows(ids.data(), (int64_t) ids.size(), uid_base, tabs.data(), ntables, o);
    if (async_flag)
    {
        // nobody waits: the engine must still finish and set the flag
        int64_t t0 = now_ns();
        while (__atomic_load_n(&flag, __ATOMIC_ACQUIRE) != 0xC0FFEEu && now_ns() - t0 < kWait)
            std::this_thread::yield();
        CHECK(__atomic_load_n(&flag, __ATOMIC_ACQUIRE) == 0xC0FFEEu, "%s: flag not set", what);
    }
    int r = wait_or_die(e, id, what);
    CHECK(r == 0, "%s: gather failed %d", what, r);
    CHECK(e.done(id), "%s: not done after wait", what);
    e.release(id);
    }
    std::vector<uint8_t> ref((size_t) row_bytes);
    for (int t = 0; t < ntables; ++t)
    {
        CHECK(outs[(size_t) t]->guards_ok(), "%s: guard bytes of table %d overwritten", what, t);
        bool ok = true;
        for (size_t i = 0; i < ids.size() && ok; ++i)
        {
            ref_read(tabs[(size_t) t].fd, bases[(size_t) t] + (ids[i] - uid_base) * row_bytes,
                     row_bytes, ref.data());
            if (std::memcmp(ref.data(), outs[(size_t) t]->p + (int64_t) i * row_bytes,
                            (size_t) row_bytes) != 0)
            {
                ok = false;
                CHECK(false, "%s: table %d row %zu (id %lld, %lld bytes, kind %d) differs", what,
                      t, i, (long long) ids[i], (long long) row_bytes, kind);
            }
        }
        CHECK(ok, "%s: table %d", what, t);
    }
}

void test_rows(Engine& e, std::vector<File*>& files, const char* what)
{
    std::mt19937_64 rng(g_seed);
    const int64_t sizes[] = { 1, 7, 8, 256, 264, 1000, 4096, 5000, 65536, 70001, 200000 };
    int iters = g_quick ? 1 : 3;
    for (int it = 0; it < iters; ++it)
        for (int64_t rb : sizes)
            for (int kind = 0; kind < 4; ++kind)
                check_rows(e, files, rng, 1 + (int) (rng() % 3), kind, rb, (int) (rng() % 4),
                           (rng() % 4) == 0, what);
    // empty gather
    uint8_t dummy[8];
    RowTable t { files[0]->fd, 0, 8, dummy, 8 };
    Options o;
    TicketId id = e.submit_rows(nullptr, 0, 0, &t, 1, o);
    CHECK(wait_or_die(e, id, what) == 0, "%s: empty gather", what);
    e.release(id);
}

// ---- extents ---------------------------------------------------------------------------------

void check_extents(Engine& e, File& f, std::mt19937_64& rng, int n, int cls, const char* what,
                   bool eof)
{
    struct X
    {
        int64_t off, len;
        ExtentGeom g;
        std::unique_ptr<Guarded> slot;
    };
    std::vector<X> xs;
    std::vector<ExtentReq> reqs;
    for (int i = 0; i < n; ++i)
    {
        X x;
        const int64_t lens[] = { 1, 511, 512, 4095, 4096, 4097, 65536, 1310719, 1310720, 1310721,
                                 3000001, 13315596 };
        x.len = lens[rng() % (sizeof lens / sizeof lens[0])];
        if (x.len > f.size) x.len = f.size / 2 + 1;
        x.off = (int64_t) (rng() % (uint64_t) (f.size - x.len + 1));
        if (i == 0 && eof) x.off = f.size - x.len;             // ends exactly at EOF
        if (i == 1 && !eof) x.off = 0;
        x.g = e.extent_geometry(f.fd, x.off, x.len);
        const int64_t G = geom_of(e, f.fd);
        CHECK(x.g.payload_offset == (x.off & (G - 1)), "%s: payload offset %lld for %lld (G %lld)",
              what, (long long) x.g.payload_offset, (long long) x.off, (long long) G);
        CHECK(x.g.span_bytes % G == 0 && x.g.span_bytes >= x.g.payload_offset + x.len &&
              x.g.span_bytes - (x.g.payload_offset + x.len) < G,
              "%s: span %lld for payload %lld + %lld (G %lld)", what, (long long) x.g.span_bytes,
              (long long) x.g.payload_offset, (long long) x.len, (long long) G);
        x.slot.reset(new Guarded((size_t) x.g.span_bytes, 4096));
        xs.push_back(std::move(x));
    }
    for (auto& x : xs) reqs.push_back({ f.fd, x.off, x.len, x.slot->p, (int64_t) x.slot->n });
    std::vector<int64_t> pay(xs.size(), -1);
    Options o;
    o.cls = cls;
    int r;
    if (rng() % 2)
    {
        if (rng() % 2) (void) ::posix_fadvise(f.fd, 0, 0, POSIX_FADV_DONTNEED);
        r = e.read_extents(reqs.data(), (int64_t) reqs.size(), o, pay.data());
    }
    else
    {
        TicketId id = e.submit_extents(reqs.data(), (int64_t) reqs.size(), o, pay.data());
        r = wait_or_die(e, id, what);
        e.release(id);
    }
    CHECK(r == 0, "%s: extent read failed %d", what, r);
    for (size_t i = 0; i < xs.size(); ++i)
    {
        X& x = xs[i];
        CHECK(pay[i] == x.g.payload_offset, "%s: payload offset", what);
        CHECK(x.slot->guards_ok(), "%s: extent %zu wrote outside its span", what, i);
        std::vector<uint8_t> ref((size_t) x.len);
        ref_read(f.fd, x.off, x.len, ref.data());
        CHECK(std::memcmp(ref.data(), x.slot->p + pay[i], (size_t) x.len) == 0,
              "%s: extent %zu (off %lld len %lld) differs", what, i, (long long) x.off,
              (long long) x.len);
    }
}

void test_extents(Engine& e, File& big, File& small, const char* what)
{
    std::mt19937_64 rng(g_seed + 1);
    int iters = g_quick ? 2 : 6;
    for (int it = 0; it < iters; ++it)
    {
        check_extents(e, big, rng, 1 + (int) (rng() % 8), 1 + (int) (rng() % 3), what, it & 1);
        check_extents(e, small, rng, 1 + (int) (rng() % 4), 1, what, !(it & 1));
    }
}

// ---- concurrency -----------------------------------------------------------------------------

void test_concurrency(Engine& e, std::vector<File*>& files, File& big, const char* what)
{
    int nt = 8;
    std::vector<std::thread> th;
    std::atomic<int> started { 0 };
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t]
        {
            std::mt19937_64 rng(g_seed * 31 + (uint64_t) t);
            started.fetch_add(1);
            while (started.load() < nt) std::this_thread::yield();
            int iters = g_quick ? 6 : 24;
            for (int i = 0; i < iters; ++i)
            {
                if (rng() % 3 == 0) check_extents(e, big, rng, 1 + (int) (rng() % 3),
                                                  (int) (rng() % 4), what, false);
                else check_rows(e, files, rng, 1 + (int) (rng() % 2), (int) (rng() % 3),
                                (rng() % 2) ? 256 : 8, (int) (rng() % 4), false, what);
            }
        });
    for (auto& x : th) x.join();
}

// Many async tickets, waited out of order
void test_async(Engine& e, File& big, const char* what)
{
    std::mt19937_64 rng(g_seed + 7);
    int n = g_quick ? 16 : 64;
    struct A
    {
        TicketId id;
        int64_t off, len;
        std::unique_ptr<Guarded> slot;
        int64_t pay;
    };
    std::vector<A> as;
    for (int i = 0; i < n; ++i)
    {
        A a;
        a.len = 1 + (int64_t) (rng() % 3000000);
        a.off = (int64_t) (rng() % (uint64_t) (big.size - a.len));
        ExtentGeom g = e.extent_geometry(big.fd, a.off, a.len);
        a.slot.reset(new Guarded((size_t) g.span_bytes));
        ExtentReq r { big.fd, a.off, a.len, a.slot->p, (int64_t) a.slot->n };
        Options o;
        o.cls = (int) (rng() % 4);
        o.deadline_ns = o.cls == kPrefetch ? now_ns() + (int64_t) (rng() % 1000000) : 0;
        a.id = e.submit_extents(&r, 1, o, &a.pay);
        as.push_back(std::move(a));
    }
    std::shuffle(as.begin(), as.end(), rng);
    for (auto& a : as)
    {
        CHECK(wait_or_die(e, a.id, what) == 0, "%s: async ticket", what);
        TicketTimes tt = e.times(a.id);
        CHECK(tt.t_done >= tt.t_first_issue && tt.t_first_issue >= tt.t_submit,
              "%s: ticket times out of order", what);
        e.release(a.id);
        std::vector<uint8_t> ref((size_t) a.len);
        ref_read(big.fd, a.off, a.len, ref.data());
        CHECK(std::memcmp(ref.data(), a.slot->p + a.pay, (size_t) a.len) == 0, "%s: async data",
              what);
        CHECK(a.slot->guards_ok(), "%s: async guards", what);
    }
}

// Threads that submit and then exit (or give up waiting) before their reads complete. An
// io_uring read belongs to the thread that submitted it and fails when that thread is gone, so
// the engine must never let such a thread own one. The file is dropped from the page cache
// first, so the buffered reads really wait for the device
void test_thread_exit(Engine& e, File& big, const char* what)
{
    for (int round = 0; round < (g_quick ? 2 : 6); ++round)
    {
        (void) ::posix_fadvise(big.fd, 0, 0, POSIX_FADV_DONTNEED);
        const int nt = 16, rows = 200;
        std::vector<std::vector<int64_t>> ids((size_t) nt);
        std::vector<std::vector<uint8_t>> outs((size_t) nt);
        std::vector<TicketId> tickets((size_t) nt, 0);
        std::vector<std::thread> th;
        for (int k = 0; k < nt; ++k)
        {
            std::mt19937_64 rng(g_seed * 131 + (uint64_t) (round * nt + k));
            ids[(size_t) k] = make_ids(rng, rows, (big.size - 256) / 256, 0);
            outs[(size_t) k].assign(ids[(size_t) k].size() * 256, 0);
        }
        for (int k = 0; k < nt; ++k)
            th.emplace_back([&, k]
            {
                RowTable t { big.fd, 0, 256, outs[(size_t) k].data(), (int64_t) outs[(size_t) k].size() };
                Options o;
                o.cls = k % 4;
                tickets[(size_t) k] = e.submit_rows(ids[(size_t) k].data(),
                                                    (int64_t) ids[(size_t) k].size(), 0, &t, 1, o);
                if (k % 3 == 0) (void) e.wait(tickets[(size_t) k], 0);   // gives up at once
                // the thread ends here, reads possibly still in flight
            });
        for (auto& x : th) x.join();
        std::vector<uint8_t> ref(256);
        for (int k = 0; k < nt; ++k)
        {
            int r = wait_or_die(e, tickets[(size_t) k], what);
            CHECK(r == 0, "%s: round %d thread %d: %d", what, round, k, r);
            e.release(tickets[(size_t) k]);
            bool ok = true;
            for (size_t i = 0; i < ids[(size_t) k].size() && ok; ++i)
            {
                ref_read(big.fd, ids[(size_t) k][i] * 256, 256, ref.data());
                ok = std::memcmp(ref.data(), outs[(size_t) k].data() + i * 256, 256) == 0;
            }
            CHECK(ok, "%s: round %d thread %d: data differs", what, round, k);
        }
    }
}

// ---- errors ------------------------------------------------------------------------------------

template <class F> int expect_error(F&& f)
{
    try
    {
        f();
    }
    catch (const Error& err)
    {
        return err.code();
    }
    return 0;
}

void test_errors(Engine& e, File& big, const char* what)
{
    uint8_t out[1024];
    Options o;
    int64_t ids[2] = { 0, 5 };
    // a descriptor that is closed
    int fd = ::open(big.path.c_str(), O_RDONLY | O_CLOEXEC);
    ::close(fd);
    RowTable t1 { fd, 0, 8, out, sizeof out };
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &t1, 1, o); }) == EBADF, "%s: closed fd", what);
    // rows beyond the end of the file
    RowTable t2 { big.fd, big.size - 8, 8, out, sizeof out };
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &t2, 1, o); }) == ERANGE, "%s: rows past EOF",
          what);
    // id below uid_base
    RowTable t3 { big.fd, 0, 8, out, sizeof out };
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 1, &t3, 1, o); }) == ERANGE, "%s: id < base", what);
    // output too small, zero row size, negative base
    RowTable t4 { big.fd, 0, 8, out, 15 };
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &t4, 1, o); }) == EINVAL, "%s: small out", what);
    RowTable t5 { big.fd, 0, 0, out, sizeof out };
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &t5, 1, o); }) == EINVAL, "%s: row 0", what);
    RowTable t6 { big.fd, -1, 8, out, sizeof out };
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &t6, 1, o); }) == EINVAL, "%s: base < 0", what);
    // overflowing offsets
    int64_t huge[1] = { INT64_MAX / 2 };
    CHECK(expect_error([&] { e.submit_rows(huge, 1, 0, &t3, 1, o); }) == ERANGE, "%s: overflow", what);
    // overlapping outputs of two tables
    RowTable two[2] = { { big.fd, 0, 8, out, 16 }, { big.fd, 0, 8, out + 8, 16 } };
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, two, 2, o); }) == EINVAL, "%s: overlap", what);
    // bad class, too many tables
    Options bad;
    bad.cls = 4;
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &t3, 1, bad); }) == EINVAL, "%s: class", what);
    CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &t3, 0, o); }) == EINVAL, "%s: 0 tables", what);
    // not a regular file
    int pfd[2];
    if (::pipe(pfd) == 0)
    {
        RowTable tp { pfd[0], 0, 8, out, sizeof out };
        CHECK(expect_error([&] { e.submit_rows(ids, 2, 0, &tp, 1, o); }) == EINVAL, "%s: pipe", what);
        ::close(pfd[0]);
        ::close(pfd[1]);
    }
    // extents: misaligned slot, small slot, past EOF, overlapping slots
    Guarded g(1 << 16);
    ExtentReq x1 { big.fd, 100, 1000, g.p + 1, 8192 };
    CHECK(expect_error([&] { e.submit_extents(&x1, 1, o, nullptr); }) == EINVAL, "%s: misaligned", what);
    ExtentReq x2 { big.fd, 4000, 1000, g.p, 4096 };
    CHECK(expect_error([&] { e.submit_extents(&x2, 1, o, nullptr); }) == EINVAL, "%s: small slot", what);
    ExtentReq x3 { big.fd, big.size - 10, 11, g.p, 8192 };
    CHECK(expect_error([&] { e.submit_extents(&x3, 1, o, nullptr); }) == ERANGE, "%s: past EOF", what);
    ExtentReq x4[2] = { { big.fd, 0, 8192, g.p, 8192 }, { big.fd, 0, 8192, g.p + 4096, 8192 } };
    CHECK(expect_error([&] { e.submit_extents(x4, 2, o, nullptr); }) == EINVAL, "%s: slot overlap", what);
    // unknown tickets
    CHECK(expect_error([&] { e.wait(987654321, 0); }) == ENOENT, "%s: unknown ticket", what);
    e.release(987654321);   // no-op
    // nothing was left queued by the failed submissions
    Stats s = e.stats(false);
    CHECK(s.queued_ops_now == 0, "%s: queued ops after failed submissions", what);
}

// Caller descriptors can be closed right after submitting, and a reused number is a new file
void test_fd_lifetime(Engine& e, File& big, File& small, const char* what)
{
    std::mt19937_64 rng(g_seed + 11);
    int fd = ::open(big.path.c_str(), O_RDONLY | O_CLOEXEC);
    Guarded g(8 << 20);
    int64_t off = 12345, len = 5000000;
    ExtentReq x { fd, off, len, g.p, (int64_t) g.n };
    int64_t pay = 0;
    Options o;
    TicketId id = e.submit_extents(&x, 1, o, &pay);
    ::close(fd);
    CHECK(wait_or_die(e, id, what) == 0, "%s: read after the caller closed its fd", what);
    e.release(id);
    std::vector<uint8_t> ref((size_t) len);
    ref_read(big.fd, off, len, ref.data());
    CHECK(std::memcmp(ref.data(), g.p + pay, (size_t) len) == 0, "%s: data after close", what);

    // the same descriptor number, another file
    int a = ::open(big.path.c_str(), O_RDONLY | O_CLOEXEC);
    uint8_t out[64], ref2[64];
    int64_t ids[1] = { 3 };
    RowTable t { a, 0, 64, out, 64 };
    CHECK(e.gather_rows(ids, 1, 0, &t, 1, o) == 0, "%s: gather a", what);
    ::close(a);
    int b = ::open(small.path.c_str(), O_RDONLY | O_CLOEXEC);
    RowTable t2 { b, 0, 64, out, 64 };
    CHECK(e.gather_rows(ids, 1, 0, &t2, 1, o) == 0, "%s: gather b", what);
    ref_read(small.fd, 3 * 64, 64, ref2);
    CHECK(std::memcmp(out, ref2, 64) == 0, "%s: reused descriptor number read the old file (a=%d b=%d)",
          what, a, b);
    ::close(b);
}

// ---- scheduler ---------------------------------------------------------------------------------

// Class 2 bulk and a class 0 hold batch: nothing of class 2 may be issued while the batch runs,
// and after arm_hold nothing until the arm expires or a hold batch finished
void test_hold(const Overrides& base, File& big, const char* what)
{
    Overrides ov = base;
    ov["trace"] = "100000";
    ov["window_prefetch"] = "inf/0";
    ov["hold_arm_ms"] = "200";
    ov["chunk"] = "256K";
    Engine e(cfg_of(ov));
    Guarded bulk((40 << 20) + (1 << 16));     // + the largest slot geometry (G <= 64 KiB)
    ExtentReq xb { big.fd, 4096, 40 << 20, bulk.p, (int64_t) bulk.n };
    if (xb.offset + xb.length > big.size) xb.length = big.size - xb.offset - 4096;
    Options ob;
    ob.cls = kPrefetch;
    TicketId tb = e.submit_extents(&xb, 1, ob, nullptr);
    std::vector<int64_t> ids;
    for (int i = 0; i < 96; ++i) ids.push_back((int64_t) i * 997);
    std::vector<uint8_t> out(ids.size() * 256);
    RowTable t { big.fd, 17, 256, out.data(), (int64_t) out.size() };
    Options oh;
    oh.cls = kEngram;
    oh.hold = true;
    TicketId th = e.submit_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, oh);
    CHECK(wait_or_die(e, th, what) == 0, "%s: hold batch", what);
    TicketTimes hh = e.times(th);
    CHECK(wait_or_die(e, tb, what) == 0, "%s: bulk", what);
    std::vector<TraceRec> tr = e.trace(true);
    int bad = 0, c2 = 0;
    for (const TraceRec& r : tr)
        if (r.cls == kPrefetch)
        {
            ++c2;
            if (r.t_issue > hh.t_submit && r.t_issue < hh.t_done) ++bad;
        }
    CHECK(bad == 0, "%s: %d class-2 reads issued while a hold batch was out", what, bad);
    CHECK(c2 > 10, "%s: bulk ran %d ops", what, c2);
    e.release(th);
    e.release(tb);

    // arm_hold: class 2 waits for the arm to expire (no hold ticket follows)
    e.arm_hold();
    int64_t t_arm = now_ns();
    ExtentReq xs { big.fd, 0, 1 << 20, bulk.p, (int64_t) bulk.n };
    TicketId t2 = e.submit_extents(&xs, 1, ob, nullptr);
    CHECK(wait_or_die(e, t2, what) == 0, "%s: armed bulk", what);
    TicketTimes tt = e.times(t2);
    CHECK(tt.t_first_issue - t_arm >= 150000000ll,
          "%s: class 2 issued %.1f ms after arm_hold (arm is 200 ms)", what,
          (double) (tt.t_first_issue - t_arm) / 1e6);
    e.release(t2);

    // arm_hold then a hold batch: the arm clears when the batch finishes
    e.arm_hold();
    TicketId t3 = e.submit_extents(&xs, 1, ob, nullptr);
    TicketId t4 = e.submit_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, oh);
    CHECK(wait_or_die(e, t4, what) == 0, "%s: batch after arm", what);
    CHECK(wait_or_die(e, t3, what) == 0, "%s: bulk after arm", what);
    TicketTimes t3t = e.times(t3), t4t = e.times(t4);
    CHECK(t3t.t_first_issue >= t4t.t_done, "%s: class 2 issued before the hold batch finished", what);
    CHECK(t3t.t_first_issue - t4t.t_done < 150000000ll, "%s: arm not cleared by the hold batch", what);
    e.release(t3);
    e.release(t4);
}

// Windows and the reserve bound what is in flight per class
void test_windows(const Overrides& base, File& big, const char* what)
{
    Overrides ov = base;
    ov["qd"] = "16";
    ov["reserve0"] = "12";
    ov["window_expert"] = "256K";
    ov["chunk"] = "256K";
    Engine e(cfg_of(ov));
    Guarded g1(24 << 20), g2(24 << 20);
    ExtentReq x1 { big.fd, 0, 20 << 20, g1.p, (int64_t) g1.n };
    ExtentReq x2 { big.fd, 1 << 20, 20 << 20, g2.p, (int64_t) g2.n };
    Options o1;
    o1.cls = kExpert;
    Options o3;
    o3.cls = kRefill;
    TicketId a = e.submit_extents(&x1, 1, o1, nullptr);
    TicketId b = e.submit_extents(&x2, 1, o3, nullptr);
    CHECK(wait_or_die(e, a, what) == 0 && wait_or_die(e, b, what) == 0, "%s: windowed reads", what);
    Stats s = e.stats(false);
    CHECK(s.cls[kExpert].max_inflight_bytes <= 256 << 10, "%s: class 1 had %lld bytes in flight",
          what, (long long) s.cls[kExpert].max_inflight_bytes);
    CHECK(s.cls[kExpert].max_inflight + s.cls[kRefill].max_inflight <= 16,
          "%s: classes 1+3 exceeded qd", what);
    CHECK(s.cls[kExpert].max_inflight <= 4 && s.cls[kRefill].max_inflight <= 4,
          "%s: reserve0 violated (%lld / %lld in flight, qd 16 - reserve 12)", what,
          (long long) s.cls[kExpert].max_inflight, (long long) s.cls[kRefill].max_inflight);
    CHECK(s.cls[kExpert].held > 0, "%s: the window never held class 1", what);
    e.release(a);
    e.release(b);
}

// Queued work: a ticket that is not admitted (class 3 under an armed hold: lo window 0) times
// out, is promoted, completes; another is cancelled; refill ages into prefetch
void test_queue_ops(const Overrides& base, File& big, const char* what)
{
    {
        Overrides ov = base;
        ov["window_refill"] = "8M/0";
        ov["refill_age_ms"] = "0";
        ov["hold_arm_ms"] = "10000";
        Engine e(cfg_of(ov));
        e.arm_hold();                       // classes 2-3 held for 10 s (no hold ticket follows)
        Guarded g(4 << 20);
        ExtentReq x { big.fd, 777, 3 << 20, g.p, (int64_t) g.n };
        Options o;
        o.cls = kRefill;
        int64_t pay = 0;
        TicketId id = e.submit_extents(&x, 1, o, &pay);
        CHECK(e.wait(id, 50000000) == -ETIMEDOUT, "%s: window 0 admitted a read", what);
        CHECK(!e.done(id), "%s: done while blocked", what);
        e.promote(id, kExpert);
        CHECK(wait_or_die(e, id, what) == 0, "%s: promoted ticket", what);
        e.release(id);
        std::vector<uint8_t> ref((size_t) x.length);
        ref_read(big.fd, x.offset, x.length, ref.data());
        CHECK(std::memcmp(ref.data(), g.p + pay, (size_t) x.length) == 0, "%s: promoted data", what);

        TicketId c = e.submit_extents(&x, 1, o, nullptr);
        e.cancel(c);
        CHECK(wait_or_die(e, c, what) == -ECANCELED, "%s: cancelled ticket", what);
        Stats s = e.stats(false);
        CHECK(s.cls[kRefill].ops_cancelled > 0, "%s: cancel not counted", what);
        e.release(c);
        // release of a still-blocked ticket cancels it
        TicketId d = e.submit_extents(&x, 1, o, nullptr);
        e.release(d);
    }
    {
        Overrides ov = base;
        ov["window_refill"] = "8M/0";
        ov["window_prefetch"] = "inf/inf";
        ov["refill_age_ms"] = "40";
        ov["hold_arm_ms"] = "10000";
        Engine e(cfg_of(ov));
        e.arm_hold();                       // class 3 held; once aged, class 2 (lo inf) is not
        Guarded g(4 << 20);
        ExtentReq x { big.fd, 0, 2 << 20, g.p, (int64_t) g.n };
        Options o;
        o.cls = kRefill;
        int64_t t0 = now_ns();
        TicketId id = e.submit_extents(&x, 1, o, nullptr);
        CHECK(wait_or_die(e, id, what) == 0, "%s: aged ticket", what);
        TicketTimes tt = e.times(id);
        CHECK(tt.t_first_issue - t0 >= 35000000ll, "%s: refill issued before aging (%.1f ms)",
              what, (double) (tt.t_first_issue - t0) / 1e6);
        Stats s = e.stats(false);
        CHECK(s.cls[kRefill].aged == 1, "%s: aged count %llu", what,
              (unsigned long long) s.cls[kRefill].aged);
        e.release(id);
    }
}

// Earliest deadline first within class 2 (pool backend with one slow slot)
void test_edf(File& big, const char* what)
{
    Overrides ov;
    ov["backend"] = "pread";
    ov["threads"] = "1";
    ov["qd"] = "1";
    ov["reserve0"] = "0";
    ov["fault_delay_us"] = "20000";
    ov["trace"] = "1000";
    Engine e(cfg_of(ov));
    Guarded g(1 << 20);
    uint8_t out[64 * 8];
    int64_t id0[1] = { 0 };
    RowTable blocker { big.fd, 0, 64, out, 64 };
    Options ob;
    ob.cls = kPrefetch;
    TicketId b = e.submit_rows(id0, 1, 0, &blocker, 1, ob);     // occupies the one slot
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    int64_t now = now_ns();
    const int64_t dl[5] = { 50, 10, 40, 20, 30 };
    TicketId ids[5];
    for (int i = 0; i < 5; ++i)
    {
        RowTable t { big.fd, 0, 64, out + 64 * (i + 1), 64 };
        Options o;
        o.cls = kPrefetch;
        o.deadline_ns = now + dl[i] * 1000000ll;
        int64_t one[1] = { i + 1 };
        ids[i] = e.submit_rows(one, 1, 0, &t, 1, o);
    }
    for (int i = 0; i < 5; ++i) CHECK(wait_or_die(e, ids[i], what) == 0, "%s: edf ticket", what);
    CHECK(wait_or_die(e, b, what) == 0, "%s: blocker", what);
    std::vector<std::pair<int64_t, int64_t>> order;
    for (int i = 0; i < 5; ++i) order.push_back({ e.times(ids[i]).t_first_issue, dl[i] });
    std::sort(order.begin(), order.end());
    for (int i = 0; i < 5; ++i)
        CHECK(order[(size_t) i].second == (i + 1) * 10, "%s: issue %d had deadline %lld", what, i,
              (long long) order[(size_t) i].second);
    for (int i = 0; i < 5; ++i) e.release(ids[i]);
    e.release(b);
}

// A file truncated while reads are queued: the reads fail with EIO, nothing crashes
void test_truncate(const Overrides& base, const char* what)
{
    File f = make_file("trunc.bin", 8 << 20, 99);
    Overrides ov = base;
    ov["window_refill"] = "8M/0";
    ov["refill_age_ms"] = "0";
    ov["hold_arm_ms"] = "10000";
    Engine e(cfg_of(ov));
    e.arm_hold();                           // class 3 stays queued until promoted
    Guarded g(8 << 20);
    ExtentReq x { f.fd, 4096, 6 << 20, g.p, (int64_t) g.n };
    Options o;
    o.cls = kRefill;
    TicketId id = e.submit_extents(&x, 1, o, nullptr);
    CHECK(::ftruncate(f.fd, 4096) == 0, "%s: truncate", what);
    e.promote(id, kExpert);
    int r = wait_or_die(e, id, what);
    CHECK(r == -EIO, "%s: truncated file gave %d, expected -EIO", what, r);
    e.release(id);
    ::close(f.fd);
    ::unlink(f.path.c_str());
}

// Fault injection: EIO fails tickets; short reads and EINTR are resumed byte-exact
void test_faults(const Overrides& base, std::vector<File*>& files, File& big, const char* what)
{
    {
        Overrides ov = base;
        ov["fault_short"] = "3";
        ov["fault_eintr"] = "5";
        Engine e(cfg_of(ov));
        std::string w = std::string(what) + " short+eintr";
        std::mt19937_64 rng(g_seed + 5);
        for (int i = 0; i < (g_quick ? 4 : 12); ++i)
        {
            check_rows(e, files, rng, 2, (int) (rng() % 4), (rng() % 2) ? 256 : 5000, 0, false,
                       w.c_str());
            check_extents(e, big, rng, 2, 1, w.c_str(), i & 1);
        }
        Stats s = e.stats(false);
        CHECK(s.cls[0].op_errors == 0 && s.cls[1].op_errors == 0, "%s: resumed reads failed", what);
    }
    {
        Overrides ov = base;
        ov["fault_eio"] = "4";
        Engine e(cfg_of(ov));
        int fails = 0;
        for (int i = 0; i < 20; ++i)
        {
            std::vector<int64_t> ids;
            for (int k = 0; k < 16; ++k) ids.push_back(k * 1000 + i);
            std::vector<uint8_t> out(ids.size() * 256);
            RowTable t { big.fd, 0, 256, out.data(), (int64_t) out.size() };
            Options o;
            int r = e.gather_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, o);
            if (r == -EIO) ++fails;
            else CHECK(r == 0, "%s: unexpected result %d", what, r);
        }
        CHECK(fails > 0, "%s: injected EIO never surfaced", what);
        Stats s = e.stats(false);
        CHECK(s.cls[0].op_errors > 0, "%s: errors not counted", what);
    }
}

// Engine shutdown with reads queued and in flight; tickets never released
void test_shutdown(const Overrides& base, File& big, const char* what)
{
    for (int round = 0; round < 3; ++round)
    {
        std::vector<std::unique_ptr<Guarded>> slots;
        {
            Overrides ov = base;
            ov["window_expert"] = "1M";
            Engine e(cfg_of(ov));
            std::mt19937_64 rng(g_seed + (uint64_t) round);
            for (int i = 0; i < 24; ++i)
            {
                int64_t len = 4 << 20;
                int64_t off = (int64_t) (rng() % (uint64_t) (big.size - len));
                slots.emplace_back(new Guarded(8 << 20));
                ExtentReq x { big.fd, off, len, slots.back()->p, (int64_t) slots.back()->n };
                Options o;
                o.cls = (int) (rng() % 4);
                (void) e.submit_extents(&x, 1, o, nullptr);
            }
            if (round == 1) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            // leaves scope: cancels the queued reads, waits for those in flight
        }
        for (auto& s : slots) CHECK(s->guards_ok(), "%s: shutdown round %d wrote outside", what, round);
        slots.clear();    // freed after the engine: ASan flags any late write
    }
}

// Registered buffers (io_uring READ_FIXED) and their lifetime rules
void test_registered(const Overrides& base, File& big, const char* what)
{
    Engine e(cfg_of(base));
    Guarded region(32 << 20);
    int idx = e.register_buffer(region.p, region.n);
    if (idx < 0)
    {
        std::printf("   (%s: no registered buffers in this backend)\n", what);
        return;
    }
    std::mt19937_64 rng(g_seed + 3);
    for (int i = 0; i < 8; ++i)
    {
        int64_t len = 1 + (int64_t) (rng() % (6 << 20));
        int64_t off = (int64_t) (rng() % (uint64_t) (big.size - len));
        size_t slot_off = (size_t) (rng() % 3) * (8 << 20);
        ExtentReq x { big.fd, off, len, region.p + slot_off, 8 << 20 };
        int64_t pay = 0;
        Options o;
        o.cls = kExpert;
        TicketId id = e.submit_extents(&x, 1, o, &pay);
        if (i == 0)
            CHECK(expect_error([&] { e.unregister_buffer(idx); }) == EBUSY || e.done(id),
                  "%s: unregister with reads in flight", what);
        CHECK(wait_or_die(e, id, what) == 0, "%s: fixed read", what);
        e.release(id);
        std::vector<uint8_t> ref((size_t) len);
        ref_read(big.fd, off, len, ref.data());
        CHECK(std::memcmp(ref.data(), region.p + slot_off + pay, (size_t) len) == 0,
              "%s: READ_FIXED data", what);
    }
    // rows whose whole output lies in the registered region
    {
        std::vector<int64_t> ids;
        for (int i = 0; i < 300; ++i) ids.push_back((int64_t) i * 131);
        uint8_t* out = region.p + (24 << 20);
        RowTable t { big.fd, 77, 256, out, (int64_t) ids.size() * 256 };
        Options o;
        CHECK(e.gather_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, o) == 0,
              "%s: rows into a registered buffer", what);
        std::vector<uint8_t> ref(256);
        bool ok = true;
        for (size_t i = 0; i < ids.size() && ok; ++i)
        {
            ref_read(big.fd, 77 + ids[i] * 256, 256, ref.data());
            ok = std::memcmp(ref.data(), out + i * 256, 256) == 0;
        }
        CHECK(ok, "%s: rows into a registered buffer differ", what);
    }
    CHECK(region.guards_ok(), "%s: registered region guards", what);
    e.unregister_buffer(idx);
    CHECK(expect_error([&] { e.unregister_buffer(idx); }) == EINVAL, "%s: double unregister", what);
}

// An engine is refused in a forked child
void test_fork(const Overrides& base, File& big, const char* what)
{
    Engine e(cfg_of(base));
    uint8_t out[64];
    int64_t ids[1] = { 1 };
    RowTable t { big.fd, 0, 64, out, 64 };
    Options o;
    CHECK(e.gather_rows(ids, 1, 0, &t, 1, o) == 0, "%s: parent gather", what);
    pid_t pid = ::fork();
    if (pid == 0)
    {
        int code = 1;
        try
        {
            (void) e.submit_rows(ids, 1, 0, &t, 1, o);
        }
        catch (const Error& err)
        {
            code = err.code() == ECHILD ? 0 : 2;
        }
        std::_Exit(code);   // never runs the engine's destructor in the child
    }
    int st = 0;
    ::waitpid(pid, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "%s: forked child used the engine (status %d)",
          what, st);
    CHECK(e.gather_rows(ids, 1, 0, &t, 1, o) == 0, "%s: parent after fork", what);
}


// ---- lifetimes and races -------------------------------------------------------------------------

// wait() on one thread while another releases the ticket; two releases at once; several waiters.
// Every wait returns the ticket's result, or ENOENT when the release finished first; nothing is
// read after it was freed (ASan), nothing races (TSan)
void test_release_race(const Overrides& base, File& big, const char* what)
{
    Overrides ov = base;
    if (ov["backend"] != "io_uring" || ov.count("fault_no_uring")) ov["fault_delay_us"] = "3000";
    Engine e(cfg_of(ov));
    Guarded g(9 << 20);
    std::mt19937_64 rng(g_seed + 21);
    for (int round = 0; round < (g_quick ? 6 : 20); ++round)
    {
        (void) ::posix_fadvise(big.fd, 0, 0, POSIX_FADV_DONTNEED);
        int64_t len = 8 << 20;
        int64_t off = (int64_t) (rng() % (uint64_t) (big.size - len));
        ExtentReq x { big.fd, off, len, g.p, (int64_t) g.n };
        Options o;
        o.cls = kExpert;
        TicketId id = e.submit_extents(&x, 1, o, nullptr);
        int kind = round % 3;               // 0: wait + release, 1: two releases + wait, 2: 4 waiters
        std::atomic<int> go { 0 };
        std::atomic<int> bad { 0 };
        std::vector<std::thread> th;
        int nwait = kind == 2 ? 4 : 1;
        for (int k = 0; k < nwait; ++k)
            th.emplace_back([&]
            {
                while (!go.load()) std::this_thread::yield();
                try
                {
                    int r = e.wait(id, kWait);
                    if (r != 0 && r != -ECANCELED) bad.fetch_add(1);
                }
                catch (const Error& err)
                {
                    if (err.code() != ENOENT) bad.fetch_add(1);
                }
            });
        int nrel = kind == 1 ? 2 : 1;
        for (int k = 0; k < nrel; ++k)
            th.emplace_back([&]
            {
                while (!go.load()) std::this_thread::yield();
                e.release(id);
            });
        go.store(1);
        for (auto& t : th) t.join();
        CHECK(bad.load() == 0, "%s: round %d: a waiter saw a wrong result", what, round);
        CHECK(expect_error([&] { e.wait(id, 0); }) == ENOENT, "%s: ticket still known", what);
        CHECK(g.guards_ok(), "%s: guards", what);
    }
    Stats s = e.stats(false);
    CHECK(s.tickets_live == 0 && s.inflight_now == 0, "%s: %lld tickets live, %lld in flight",
          what, (long long) s.tickets_live, (long long) s.inflight_now);
}

// A hold ticket of class 2 or 3 would be held by its own rule: refused
void test_hold_class(const Overrides& base, File& big, const char* what)
{
    Engine e(cfg_of(base));
    uint8_t out[64];
    int64_t ids[1] = { 3 };
    RowTable t { big.fd, 0, 64, out, 64 };
    for (int cls = 0; cls < kClasses; ++cls)
    {
        Options o;
        o.cls = cls;
        o.hold = true;
        if (cls >= kPrefetch)
        {
            CHECK(expect_error([&] { e.submit_rows(ids, 1, 0, &t, 1, o); }) == EINVAL,
                  "%s: hold with class %d accepted", what, cls);
            Guarded g(8192);
            ExtentReq x { big.fd, 0, 100, g.p, 8192 };
            CHECK(expect_error([&] { e.submit_extents(&x, 1, o, nullptr); }) == EINVAL,
                  "%s: extent hold with class %d accepted", what, cls);
        }
        else CHECK(e.gather_rows(ids, 1, 0, &t, 1, o) == 0, "%s: hold class %d", what, cls);
    }
    // the class-3 queue still moves after the refused submissions
    Options o3;
    o3.cls = kRefill;
    CHECK(e.gather_rows(ids, 1, 0, &t, 1, o3) == 0, "%s: class 3 after refusals", what);
}

// Offsets, lengths and ids at the edges of int64_t: clean errors, no signed overflow (UBSan)
void test_overflow(Engine& e, File& big, const char* what)
{
    CHECK(expect_error([&] { e.extent_geometry(big.fd, 0, INT64_MAX); }) == EINVAL,
          "%s: geometry of INT64_MAX bytes", what);
    CHECK(expect_error([&] { e.extent_geometry(big.fd, INT64_MAX - 10, 5); }) == EINVAL,
          "%s: geometry near INT64_MAX", what);
    ExtentGeom g = e.extent_geometry(big.fd, 12345, (int64_t) 1 << 40);
    CHECK(g.span_bytes >= ((int64_t) 1 << 40) + (12345 & 4095), "%s: geometry of 1 TiB", what);
    uint8_t out[2 * 64];
    RowTable t { big.fd, 0, 64, out, sizeof out };
    Options o;
    int64_t top[2] = { INT64_MAX - 1, INT64_MAX };
    std::string msg;
    int code = 0;
    try
    {
        (void) e.submit_rows(top, 2, -1, &t, 1, o);
    }
    catch (const Error& err)
    {
        code = err.code();
        msg = err.what();
    }
    CHECK(code == ERANGE && msg.find(std::to_string(INT64_MAX)) != std::string::npos,
          "%s: rows at INT64_MAX: %d %s", what, code, msg.c_str());
    int64_t low[1] = { INT64_MIN };
    CHECK(expect_error([&] { e.submit_rows(low, 1, 1, &t, 1, o); }) == ERANGE,
          "%s: id below uid_base overflow", what);
    Guarded gg(8192);
    ExtentReq x { big.fd, INT64_MAX - 100, 50, gg.p, 8192 };
    CHECK(expect_error([&] { e.submit_extents(&x, 1, o, nullptr); }) == ERANGE,
          "%s: extent near INT64_MAX", what);
    Stats s = e.stats(false);
    CHECK(s.queued_ops_now == 0 && s.tickets_live == 0, "%s: leftovers", what);
}

// EXL3_DISK_KEEPALIVE_MS: reads while the engine is in use, none while its own reads are
// queued, none after the idle window until the next submission, and never keeps a forgotten
// file open
void test_keepalive(const Overrides& base, const char* what)
{
    Overrides ov = base;
    ov["keepalive_ms"] = "2";
    ov["keepalive_idle_s"] = "1";
    ov["hold_arm_ms"] = "10000";
    Engine e(cfg_of(ov));
    File a = make_file("keepalive_a.bin", 1 << 20, 51);
    uint8_t out[64];
    int64_t ids[1] = { 7 };
    Options o;
    RowTable ta { a.fd, 0, 64, out, 64 };
    auto reads = [&] { return e.stats(false).keepalive_reads; };
    auto wait_until = [&](const std::function<bool()>& f)
    {
        int64_t t0 = now_ns();
        while (!f() && now_ns() - t0 < 3000000000ll)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return f();
    };
    auto sleep_ms = [](int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); };
    CHECK(e.config().keepalive_ms == 2 && e.describe().find("keepalive 2 ms") != std::string::npos,
          "%s: config %s", what, e.describe().c_str());
    sleep_ms(20);
    CHECK(reads() == 0, "%s: keep-alive read before any submission", what);
    CHECK(e.gather_rows(ids, 1, 0, &ta, 1, o) == 0, "%s: gather", what);
    CHECK(wait_until([&] { return reads() >= 3; }), "%s: no keep-alive reads (%llu)", what,
          (unsigned long long) reads());

    // reads of its own queued (a class-3 ticket held back by an armed hold): no keep-alive
    e.arm_hold();
    Guarded g(1 << 16);
    ExtentReq x { a.fd, 8192, 4096, g.p, (int64_t) g.n };
    Options o3;
    o3.cls = kRefill;
    TicketId id = e.submit_extents(&x, 1, o3, nullptr);
    sleep_ms(5);
    uint64_t k0 = reads();
    sleep_ms(30);
    CHECK(reads() == k0, "%s: keep-alive read while reads were queued", what);
    e.release(id);

    // dormant once nothing was submitted for keepalive_idle_s, until the next submission
    sleep_ms(1300);
    uint64_t k1 = reads();
    sleep_ms(60);
    CHECK(reads() == k1, "%s: keep-alive reads after the idle window", what);
    CHECK(e.gather_rows(ids, 1, 0, &ta, 1, o) == 0, "%s: gather again", what);
    CHECK(wait_until([&] { return reads() > k1; }), "%s: keep-alive not resumed", what);

    // forgotten (at once, or once the keep-alive read holding it finished): closed, and no
    // keep-alive reads without a file
    int f = e.forget(a.fd);
    CHECK(f == 1 || f == -1, "%s: forget %d", what, f);
    CHECK(wait_until([&] { return count_fds_to("keepalive_a.bin") == 1; }),
          "%s: forgotten file kept open (%d descriptors)", what, count_fds_to("keepalive_a.bin"));
    sleep_ms(5);
    uint64_t k2 = reads();
    sleep_ms(30);
    CHECK(reads() == k2, "%s: keep-alive read without a file", what);
    CHECK(e.stats(true).keepalive_reads == k2 && e.stats(false).keepalive_reads == 0,
          "%s: stats reset", what);
    ::close(a.fd);
    ::unlink(a.path.c_str());
}

// forget(): an engine descriptor goes away with the file; a busy file goes once idle; an
// unlinked file is closed on its own
void test_forget(const Overrides& base, const char* what)
{
    Overrides ov = base;
    ov["hold_arm_ms"] = "10000";
    Engine e(cfg_of(ov));
    File a = make_file("forget_a.bin", 1 << 20, 41);
    File b = make_file("forget_b.bin", 1 << 20, 42);
    uint8_t out[64];
    int64_t ids[1] = { 5 };
    Options o;
    RowTable ta { a.fd, 0, 64, out, 64 };
    CHECK(e.gather_rows(ids, 1, 0, &ta, 1, o) == 0, "%s: gather a", what);
    int64_t open0 = e.stats(false).files_open;
    CHECK(count_fds_to("forget_a.bin") >= 2, "%s: engine has not opened a", what);
    CHECK(e.forget(a.fd) == 1, "%s: forget idle", what);
    CHECK(e.stats(false).files_open == open0 - 1, "%s: still open after forget", what);
    CHECK(count_fds_to("forget_a.bin") == 1, "%s: engine descriptor left", what);
    CHECK(e.forget(a.fd) == 0, "%s: forget twice", what);
    CHECK(e.gather_rows(ids, 1, 0, &ta, 1, o) == 0, "%s: gather after forget", what);
    CHECK(e.forget_path(a.path) == 1, "%s: forget by path", what);
    CHECK(e.forget_path(g_dir + "/no_such_file") == 0, "%s: forget missing path", what);

    // busy: a queued class-3 ticket (armed hold) keeps it; closed once released and idle
    e.arm_hold();
    Guarded g(1 << 20);
    ExtentReq x { a.fd, 0, 4096, g.p, (int64_t) g.n };
    Options o3;
    o3.cls = kRefill;
    TicketId id = e.submit_extents(&x, 1, o3, nullptr);
    CHECK(e.forget(a.fd) == -1, "%s: forget busy", what);
    CHECK(count_fds_to("forget_a.bin") >= 2, "%s: busy file closed", what);
    e.release(id);
    RowTable tb { b.fd, 0, 64, out, 64 };
    CHECK(e.gather_rows(ids, 1, 0, &tb, 1, o) == 0, "%s: gather b", what);
    CHECK(count_fds_to("forget_a.bin") == 1, "%s: doomed file not closed once idle", what);

    // unlinked: closed by the periodic sweep, and its blocks are released
    CHECK(e.gather_rows(ids, 1, 0, &ta, 1, o) == 0, "%s: gather a again", what);
    ::close(a.fd);
    ::unlink(a.path.c_str());
    CHECK(count_fds_to("forget_a.bin (deleted)") >= 1, "%s: engine let go too early", what);
    for (int i = 0; i < 200; ++i) (void) e.gather_rows(ids, 1, 0, &tb, 1, o);
    CHECK(count_fds_to("forget_a.bin") == 0, "%s: unlinked file still held by the engine", what);
    ::close(b.fd);
    ::unlink(b.path.c_str());
}

// Submissions racing a fatal ring error: the losers get an error, nothing leaks, nothing is
// used after it was freed (the unwinding path of enqueue)
void test_broken_submit(const Overrides& base, File& big, const char* what)
{
    Overrides ov = base;
    ov["fault_enter_fatal"] = "2";
    Engine e(cfg_of(ov));
    const int nt = 12;
    std::atomic<int> errors { 0 }, bad { 0 };
    std::vector<std::thread> th;
    for (int k = 0; k < nt; ++k)
        th.emplace_back([&, k]
        {
            std::vector<int64_t> ids;
            for (int i = 0; i < 400; ++i) ids.push_back((int64_t) i * 7 + k);   // not adjacent
            std::vector<uint8_t> out(ids.size() * 256);
            RowTable t { big.fd, 0, 256, out.data(), (int64_t) out.size() };
            Options o;
            o.cls = k % 2 ? kPrefetch : kEngram;
            for (int it = 0; it < 200; ++it)
            {
                try
                {
                    TicketId id = e.submit_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, o);
                    int r = wait_or_die(e, id, what);
                    if (r != 0 && r != -EBADFD) bad.fetch_add(1);
                    e.release(id);
                }
                catch (const Error& err)
                {
                    if (err.code() != EBADFD) bad.fetch_add(1);
                    errors.fetch_add(1);
                    break;
                }
            }
        });
    for (auto& t : th) t.join();
    CHECK(bad.load() == 0, "%s: unexpected results", what);
    CHECK(errors.load() == nt, "%s: %d of %d threads saw the broken engine", what, errors.load(), nt);
    Stats s = e.stats(false);
    CHECK(s.tickets_live == 0 && s.queued_ops_now == 0 && s.inflight_now == 0,
          "%s: leftovers after the fatal error", what);
}

// After a fatal ring error: no thread of the engine spins (reaper timer, poller), and reads
// prepared on the IOPOLL ring in the same pass are failed too
void test_fatal_quiet(const Overrides& base, File& big, const char* what)
{
    long hz = ::sysconf(_SC_CLK_TCK);
    // (a) armed hold + held class-2 ticket, then a synchronous gather hits the fatal error
    {
        Overrides ov = base;
        ov["fault_enter_fatal"] = "1";
        ov["hold_arm_ms"] = "200";
        std::vector<int> before = threads_named("exl3-disk-reap");
        Engine e(cfg_of(ov));
        int reaper = new_thread_named("exl3-disk-reap", before);
        e.arm_hold();
        Guarded g(1 << 20);
        ExtentReq x { big.fd, 0, 1 << 19, g.p, (int64_t) g.n };
        Options o2;
        o2.cls = kPrefetch;
        TicketId held = e.submit_extents(&x, 1, o2, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));   // the reaper saw it
        uint8_t out[64];
        int64_t ids[1] = { 9 };
        RowTable t { big.fd, 0, 64, out, 64 };
        Options o;
        int r = e.gather_rows(ids, 1, 0, &t, 1, o);
        CHECK(r == -EBADFD, "%s: sync gather on a failing ring gave %d", what, r);
        CHECK(e.wait(held, kWait) == -EBADFD, "%s: held ticket not failed", what);
        e.release(held);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // past the armed hold
        long t0 = reaper ? thread_ticks(reaper) : -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        long t1 = reaper ? thread_ticks(reaper) : -1;
        CHECK(reaper && t0 >= 0 && t1 - t0 <= hz / 10,
              "%s: reaper used %ld ticks in 0.5 s after the fatal error (hz %ld)", what, t1 - t0, hz);
    }
    // (b) IOPOLL ring (forced) with direct rows, buffered extents on the main ring: both held
    //     by the arm, admitted in one reaper pass whose first io_uring_enter fails
    {
        Overrides ov = base;
        ov["fault_enter_fatal"] = "1";
        ov["hold_arm_ms"] = "100";
        ov["direct"] = "rows";
        ov["iopoll"] = "1";
        ov["fault_force_iopoll"] = "1";
        ov["register"] = "none";
        std::vector<int> before = threads_named("exl3-disk-poll");
        Engine e(cfg_of(ov));
        int poller = new_thread_named("exl3-disk-poll", before);
        CHECK(poller != 0, "%s: no IOPOLL poller thread", what);
        e.arm_hold();
        std::vector<int64_t> ids = { 3, 100, 2000, 40000 };
        std::vector<uint8_t> out(ids.size() * 256);
        RowTable t { big.fd, 11, 256, out.data(), (int64_t) out.size() };
        Options o2;
        o2.cls = kPrefetch;
        TicketId rows = e.submit_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, o2);
        Guarded g(1 << 20);
        ExtentReq x { big.fd, 4096, 1 << 19, g.p, (int64_t) g.n };
        TicketId ext = e.submit_extents(&x, 1, o2, nullptr);
        int r1 = e.wait(rows, 3000000000ll), r2 = e.wait(ext, 3000000000ll);
        CHECK(r1 == -EBADFD && r2 == -EBADFD,
              "%s: after a fatal enter: rows %d, extent %d (expected %d each)", what, r1, r2,
              -EBADFD);
        if (r1 == -ETIMEDOUT || r2 == -ETIMEDOUT)
        {
            std::fprintf(stderr, "FAIL: %s: stranded reads would hang the shutdown\n", what);
            std::_Exit(3);
        }
        e.release(rows);
        e.release(ext);
        Stats s = e.stats(false);
        CHECK(s.inflight_now == 0, "%s: %lld still in flight", what, (long long) s.inflight_now);
        if (poller)
        {
            long t0 = thread_ticks(poller);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            long t1 = thread_ticks(poller);
            CHECK(t0 >= 0 && t1 - t0 <= hz / 10, "%s: poller used %ld ticks in 0.5 s", what, t1 - t0);
        }
    }
}

// EXL3_DISK_AFFINITY reaches the io_uring workers of every submitting thread, not only of the
// thread that created the engine
void test_iowq_affinity(const Overrides& base, File& big, const char* what)
{
    int ncpu = (int) std::thread::hardware_concurrency();
    if (ncpu < 4)
    {
        std::printf("   (%s: fewer than 4 CPUs)\n", what);
        return;
    }
    Overrides ov = base;
    ov["affinity"] = "1-2";
    ov["uring_async"] = "1";                // every buffered read goes to io-wq
    ov["direct"] = "none";
    std::vector<int> before = threads_named("exl3-disk-reap");
    Engine e(cfg_of(ov));
    int reaper = new_thread_named("exl3-disk-reap", before);
    std::string bad;
    int seen = 0;
    auto check_workers = [&](int owner)
    {
        std::string name = "iou-wrk-" + std::to_string(owner);
        for (int tid : threads_named(name.c_str()))
        {
            ++seen;
            std::string cpus = task_status_field(tid, "Cpus_allowed_list");
            if (cpus != "1-2") bad += " " + name + "/" + std::to_string(tid) + ":" + cpus;
        }
    };
    std::thread caller([&]
    {
        (void) ::posix_fadvise(big.fd, 0, 0, POSIX_FADV_DONTNEED);
        std::vector<int64_t> ids;
        for (int i = 0; i < 64; ++i) ids.push_back((int64_t) i * 509);
        std::vector<uint8_t> out(ids.size() * 256);
        RowTable t { big.fd, 0, 256, out.data(), (int64_t) out.size() };
        Options o;
        for (int it = 0; it < 4; ++it)
            CHECK(e.gather_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, o) == 0, "%s: gather", what);
        // a submitting task's workers are named after it
        check_workers((int) ::gettid());
        // an asynchronous ticket: the reaper submits it
        Options o3;
        o3.cls = kRefill;
        (void) ::posix_fadvise(big.fd, 0, 0, POSIX_FADV_DONTNEED);
        TicketId id = e.submit_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, o3);
        CHECK(wait_or_die(e, id, what) == 0, "%s: async gather", what);
        e.release(id);
        if (reaper) check_workers(reaper);
    });
    caller.join();
    CHECK(bad.empty(), "%s: caller's io_uring workers not pinned:%s", what, bad.c_str());
    if (!seen) std::printf("   (%s: no io_uring worker of the caller seen)\n", what);
}

// Forked children: no Engine entry point locks what the parent's threads held at fork time,
// and the default engine's mutex is released in the child
void test_fork_busy(const Overrides& base, File& big, const char* what)
{
    Engine e(cfg_of(base));
    std::atomic<bool> stop { false };
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
        th.emplace_back([&, k]
        {
            std::vector<int64_t> ids;
            for (int i = 0; i < 48; ++i) ids.push_back((int64_t) i * 97 + k);
            std::vector<uint8_t> out(ids.size() * 256);
            RowTable t { big.fd, 0, 256, out.data(), (int64_t) out.size() };
            Options o;
            while (!stop.load()) (void) e.gather_rows(ids.data(), (int64_t) ids.size(), 0, &t, 1, o);
        });
    uint8_t out[64];
    int64_t ids[1] = { 1 };
    RowTable t { big.fd, 0, 64, out, 64 };
    Options o3;
    o3.cls = kRefill;
    TicketId id = e.submit_rows(ids, 1, 0, &t, 1, o3);
    int hung = 0, wrong = 0;
    for (int i = 0; i < (g_quick ? 40 : 150); ++i)
    {
        pid_t pid = ::fork();
        if (pid == 0)
        {
            ::alarm(5);
            int ok = 0;
            auto one = [&](const std::function<void()>& f)
            {
                try { f(); }
                catch (const Error& err) { if (err.code() == ECHILD) ++ok; }
            };
            one([&] { (void) e.done(id); });
            one([&] { (void) e.times(id); });
            one([&] { e.cancel(id); });
            one([&] { e.promote(id, 0); });
            one([&] { e.arm_hold(); });
            one([&] { (void) e.stats(false); });
            one([&] { (void) e.trace(false); });
            one([&] { (void) e.wait(id, 0); });
            e.release(id);                  // a no-op in a child
            std::_Exit(ok == 8 ? 0 : 2);
        }
        int st = 0;
        ::waitpid(pid, &st, 0);
        if (WIFSIGNALED(st)) ++hung;
        else if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) ++wrong;
    }
    stop.store(true);
    for (auto& x : th) x.join();
    CHECK(hung == 0 && wrong == 0, "%s: forked children: %d hung, %d wrong", what, hung, wrong);
    CHECK(wait_or_die(e, id, what) == 0, "%s: parent ticket", what);
    e.release(id);
}

// The process-wide engine in forked children while other threads use it
void test_fork_default(File& big)
{
    const char* what = "fork-default";
#if EXL3_TEST_TSAN
    // TSan kills a child of a multi-threaded process that starts threads (die_after_fork)
    std::printf("   (%s: skipped under TSan)\n", what);
    (void) big;
#elif EXL3_TEST_GCC_ASAN
    // the child builds a fresh engine (allocations) while four threads allocate in the parent:
    // GCC's ASan runtime can deadlock the child (see EXL3_TEST_GCC_ASAN at the top)
    std::printf("   (%s: skipped under GCC's AddressSanitizer)\n", what);
    (void) big;
#else
    configure_default({ { "backend", "pread" } });
    std::atomic<bool> stop { false };
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
        th.emplace_back([&, k]
        {
            std::vector<int64_t> ids;
            for (int i = 0; i < 16; ++i) ids.push_back((int64_t) i * 31 + k);
            std::vector<uint8_t> out(ids.size() * 264);
            while (!stop.load())
                (void) ngram_gather(big.fd, 0, 264, ids.data(), (int64_t) ids.size(), 0, out.data(),
                                    (int64_t) out.size());
        });
    int hung = 0, wrong = 0;
    for (int i = 0; i < (g_quick ? 40 : 120); ++i)
    {
        pid_t pid = ::fork();
        if (pid == 0)
        {
            ::alarm(10);
            int64_t ids[2] = { 4, 5 };
            uint8_t out[2 * 264], ref[2 * 264];
            int r = ngram_gather(big.fd, 0, 264, ids, 2, 0, out, sizeof out);
            ref_read(big.fd, 4 * 264, 2 * 264, ref);
            std::_Exit(r == 0 && std::memcmp(out, ref, sizeof out) == 0 ? 0 : 2);
        }
        int st = 0;
        ::waitpid(pid, &st, 0);
        if (WIFSIGNALED(st)) ++hung;
        else if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) ++wrong;
    }
    stop.store(true);
    for (auto& x : th) x.join();
    shutdown_default();
    CHECK(hung == 0 && wrong == 0, "%s: children: %d hung, %d wrong", what, hung, wrong);
#endif
}

// ---- knobs -------------------------------------------------------------------------------------

void test_config()
{
    const char* what = "config";
    auto err = [](const Overrides& ov) -> std::string
    {
        try
        {
            (void) config_from_env(ov);
        }
        catch (const Error& e)
        {
            return e.what();
        }
        return "";
    };
    CHECK(err({ { "backend", "uring" } }).find("EXL3_DISK_BACKEND=uring") != std::string::npos,
          "%s: bad backend message: %s", what, err({ { "backend", "uring" } }).c_str());
    CHECK(!err({ { "qd", "0" } }).empty(), "%s: qd 0", what);
    CHECK(!err({ { "qd", "12x" } }).empty(), "%s: qd 12x", what);
    CHECK(!err({ { "qd", "8" }, { "reserve0", "8" } }).empty(), "%s: reserve0 >= qd", what);
    CHECK(!err({ { "window_prefetch", "8M/16M" } }).empty(), "%s: lo > hi", what);
    CHECK(!err({ { "window_expert", "8M/0" } }).empty(), "%s: expert lo", what);
    // the expert window is a power of two in [256K, 1G], or inf
    for (const char* v : { "3M", "12M", "1000K", "128K", "64K", "2G", "4G" })
        CHECK(!err({ { "window_expert", v } }).empty(), "%s: window_expert=%s accepted", what, v);
    for (const char* v : { "256K", "512K", "1M", "8M", "64M", "1G", "inf" })
        CHECK(err({ { "window_expert", v } }).empty(), "%s: window_expert=%s refused", what, v);
    CHECK(config_from_env({}).window_hi[kExpert] == 8 << 20 &&
          config_from_env({ { "window_expert", "64M" } }).window_hi[kExpert] == 64 << 20 &&
          config_from_env({ { "window_expert", "inf" } }).window_hi[kExpert] < 0,
          "%s: window_expert values", what);
    // a zero hi window would never admit its class
    for (const char* k : { "window_expert", "window_prefetch", "window_refill" })
        for (const char* v : { "0", "0K", "0/0" })
            CHECK(!err({ { k, v } }).empty(), "%s: %s=%s accepted", what, k, v);
    CHECK(err({ { "window_prefetch", "8M/0" } }).empty(), "%s: lo 0 refused", what);
    CHECK(err({ { "fault_enter_fatal", "3" }, { "fault_force_iopoll", "1" } }).empty(),
          "%s: test knobs", what);
    CHECK(!err({ { "keepalive_ms", "1001" } }).empty(), "%s: keepalive_ms range", what);
    CHECK(!err({ { "keepalive_ms", "5ms" } }).empty(), "%s: keepalive_ms unit", what);
    CHECK(!err({ { "keepalive_idle_s", "0" } }).empty(), "%s: keepalive_idle_s 0", what);
    CHECK(config_from_env({ { "keepalive_ms", "auto" } }).keepalive_ms == kAutoKeepaliveMs &&
          config_from_env({}).keepalive_ms == auto_keepalive_ms() &&
          config_from_env({ { "keepalive_ms", "0" } }).keepalive_ms == 0 &&
          config_from_env({ { "keepalive_ms", "5" } }).keepalive_ms == 5 &&
          config_from_env({}).keepalive_idle_s == 30, "%s: keepalive values", what);
    CHECK(!err({ { "chunk", "100K" } }).empty(), "%s: chunk not 64K multiple", what);
    CHECK(!err({ { "align", "1000" } }).empty(), "%s: align not pow2", what);
    CHECK(!err({ { "affinity", "3-1" } }).empty(), "%s: cpu range", what);
    CHECK(!err({ { "affinity", "0,,1" } }).empty(), "%s: cpu list", what);
    CHECK(!err({ { "direct", "sometimes" } }).empty(), "%s: direct", what);
    CHECK(!err({ { "sqpoll", "2" } }).empty(), "%s: sqpoll bool", what);
    CHECK(!err({ { "no_such_knob", "1" } }).empty(), "%s: unknown override", what);

    Config c = config_from_env({ { "backend", "io_uring" }, { "window_prefetch", "32M/1M" },
                                 { "affinity", "0-2,5" }, { "qd", "64" } });
    CHECK(c.backend == Backend::IoUring && c.backend_named, "%s: backend", what);
    CHECK(c.window_hi[kPrefetch] == 32 << 20 && c.window_lo[kPrefetch] == 1 << 20, "%s: window", what);
    CHECK(c.affinity == std::vector<int>({ 0, 1, 2, 5 }), "%s: affinity", what);
    CHECK(c.reserve0 == 48 && c.engram_qd == 64, "%s: derived depths", what);
    CHECK(!c.direct_rows && c.direct_extents, "%s: io_uring direct auto", what);
    Config p = config_from_env({ { "backend", "odirect" } });
    CHECK(p.direct_rows && p.direct_extents, "%s: odirect direct auto", what);
    // auto and original: disk_auto.h's backend, with its own direct defaults; the n-gram route
    // is disk_auto.h's for auto, the original pool for original, the engine when named
    for (const char* v : { "auto", "original", "" })
    {
        Config a = config_from_env({ { "backend", v } });
        bool route = std::string(v) == "original" ? false : kAutoNgramEngine;
        CHECK(a.backend == kAutoBackend && a.backend == auto_backend() && !a.backend_named &&
              a.ngram_engine == route && a.direct_rows == (kAutoBackend == Backend::ODirect) &&
              a.direct_extents == (kAutoBackend != Backend::Pread), "%s: backend '%s'", what, v);
    }
    CHECK(auto_ngram_engine() == kAutoNgramEngine, "%s: auto route", what);
    CHECK(c.ngram_engine && p.ngram_engine, "%s: named backend routes the engine", what);
    CHECK(err({ { "backend", "legacy" } }).find("original") != std::string::npos,
          "%s: backend values message", what);
    Config n = config_from_env({ { "backend", "pread" }, { "sqpoll", "1" } });
    CHECK(!n.notes.empty(), "%s: inapplicable knob not noted", what);
    Config w = config_from_env({ { "window_refill", "inf" } });
    CHECK(w.window_hi[kRefill] == -1 && w.window_lo[kRefill] == 0, "%s: inf window", what);
    CHECK(!c.describe().empty(), "%s: describe", what);
    {
        // io_uring refused (seccomp, io_uring_disabled=2): the pool takes over, and says so
        Engine e(config_from_env({ { "backend", "io_uring" }, { "fault_no_uring", "1" } }));
        CHECK(e.config().backend == Backend::ODirect && e.fallbacks().size() == 1 &&
              e.describe().find("odirect pool") != std::string::npos,
              "%s: io_uring fallback: %s", what, e.describe().c_str());
    }

    // histogram buckets: monotone, and each value falls in its bucket
    const int64_t probes[] = { 0, 1, 7, 8, 9, 15, 16, 1000, 123456789, (int64_t) 1 << 40, INT64_MAX };
    for (int64_t v : probes)
    {
        int b = hist_bucket_of(v);
        CHECK(hist_bucket_lower(b) <= v && (b + 1 >= kHistBuckets || hist_bucket_lower(b + 1) > v),
              "%s: histogram bucket of %lld", what, (long long) v);
    }
    for (int i = 1; i < kHistBuckets; ++i)
        CHECK(hist_bucket_lower(i) > hist_bucket_lower(i - 1), "%s: bucket %d", what, i);

    // the ngram route follows EXL3_DISK_BACKEND only
    ::unsetenv("EXL3_DISK_BACKEND");
    CHECK(ngram_engine_from_env() == kAutoNgramEngine, "%s: unset route", what);
    ::setenv("EXL3_DISK_BACKEND", "auto", 1);
    CHECK(ngram_engine_from_env() == kAutoNgramEngine, "%s: auto route", what);
    ::setenv("EXL3_DISK_BACKEND", "Original", 1);
    CHECK(!ngram_engine_from_env(), "%s: original route", what);
    CHECK(windows_backend_refusal().empty(), "%s: windows original", what);
    ::setenv("EXL3_DISK_BACKEND", "io_uring", 1);
    CHECK(ngram_engine_from_env(), "%s: named route", what);
    ::setenv("EXL3_DISK_BACKEND", "bogus", 1);
    CHECK(expect_error([] { (void) ngram_engine_from_env(); }) == EINVAL, "%s: bad route", what);
    CHECK(windows_backend_refusal().find("bogus") != std::string::npos, "%s: windows msg", what);
    ::setenv("EXL3_DISK_BACKEND", "odirect", 1);
    CHECK(windows_backend_refusal().empty(), "%s: windows odirect", what);
    ::setenv("EXL3_DISK_BACKEND", "pread", 1);
    CHECK(!windows_backend_refusal().empty(), "%s: windows pread", what);
    ::unsetenv("EXL3_DISK_BACKEND");
}

// The process-wide engine and the n-gram entry point
void test_default(File& big)
{
    const char* what = "default engine";
    std::shared_ptr<Engine> e = configure_default({ { "backend", "io_uring" } });
    CHECK(ngram_route_engine(), "%s: route after configure", what);
    std::vector<int64_t> ids = { 1, 2, 3, 10, 11, 500 };
    std::vector<uint8_t> out(ids.size() * 264), ref(264);
    CHECK(ngram_gather(big.fd, 99, 264, ids.data(), (int64_t) ids.size(), 0, out.data(),
                       (int64_t) out.size()) == 0, "%s: gather", what);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        ref_read(big.fd, 99 + ids[i] * 264, 264, ref.data());
        CHECK(std::memcmp(ref.data(), out.data() + i * 264, 264) == 0, "%s: row %zu", what, i);
    }
    CHECK(expect_error([&] { ngram_gather(big.fd, big.size, 264, ids.data(), 1, 0, out.data(),
                                          (int64_t) out.size()); }) == ERANGE, "%s: range", what);
    CHECK(set_thread_class(kPrefetch) == kEngram && thread_class() == kPrefetch, "%s: class", what);
    set_thread_class(kEngram);
    std::shared_ptr<Engine> p = configure_default({ { "backend", "original" } });
    CHECK(!ngram_route_engine(), "%s: route after original", what);
    CHECK(default_engine() == p, "%s: default engine", what);
    p = configure_default({ { "backend", "auto" } });
    CHECK(ngram_route_engine() == kAutoNgramEngine, "%s: route after auto", what);
    CHECK(default_engine() == p, "%s: default engine after auto", what);
    shutdown_default();
    e.reset();
}

// ---- driver ------------------------------------------------------------------------------------

struct Backend_
{
    const char* name;
    Overrides ov;
};

}  // namespace

int main(int argc, char** argv)
{
    std::string only;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) g_dir = argv[++i];
        else if (a == "--quick") g_quick = true;
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--seed" && i + 1 < argc) g_seed = std::strtoull(argv[++i], nullptr, 10);
        else
        {
            std::fprintf(stderr, "usage: %s --dir DIR [--quick] [--only NAME] [--seed N]\n", argv[0]);
            return 2;
        }
    }
    if (g_dir.empty())
    {
        std::fprintf(stderr, "--dir is required (a scratch directory on the file system to test)\n");
        return 2;
    }
    ::mkdir(g_dir.c_str(), 0700);
    for (const char* v : { "EXL3_DISK_BACKEND", "EXL3_DISK_DIRECT", "EXL3_DISK_QD" }) ::unsetenv(v);

    int fds0 = count_fds();
    int64_t big_size = (g_quick ? 24ll : 48ll) * (1 << 20) + 12345;
    File big = make_file("big.bin", big_size, g_seed);
    File small = make_file("small.bin", (1 << 20) + 777, g_seed + 1);
    std::vector<File*> files = { &big, &small };

    std::vector<Backend_> backends = {
        { "pread", { { "backend", "pread" } } },
        { "pread-1thread", { { "backend", "pread" }, { "threads", "1" }, { "qd", "2" } } },
        { "pread+direct-extents", { { "backend", "pread" }, { "direct", "extents" } } },
        { "odirect", { { "backend", "odirect" } } },
        { "odirect-align4k", { { "backend", "odirect" }, { "align", "4096" } } },
        { "io_uring", { { "backend", "io_uring" } } },
        { "io_uring-direct", { { "backend", "io_uring" }, { "direct", "all" } } },
        { "io_uring-buffered", { { "backend", "io_uring" }, { "direct", "none" } } },
        { "io_uring-noreg", { { "backend", "io_uring" }, { "direct", "all" }, { "register", "none" } } },
        { "io_uring-qd4", { { "backend", "io_uring" }, { "direct", "all" }, { "qd", "4" }, { "reserve0", "1" } } },
        { "io_uring-async", { { "backend", "io_uring" }, { "uring_async", "1" } } },
        { "io_uring-sqpoll", { { "backend", "io_uring" }, { "direct", "all" }, { "sqpoll", "1" } } },
        { "io_uring-iopoll", { { "backend", "io_uring" }, { "direct", "all" }, { "iopoll", "1" } } },
        { "io_uring-refused", { { "backend", "io_uring" }, { "fault_no_uring", "1" } } },
        { "odirect-iopoll", { { "backend", "odirect" }, { "iopoll", "1" } } },
        // IOPOLL ring really used (the device's io_poll check is skipped; a disk without poll
        // queues still completes polled reads), with and without registered buffers
        { "io_uring-iopoll-forced", { { "backend", "io_uring" }, { "direct", "all" }, { "iopoll", "1" },
                                      { "fault_force_iopoll", "1" } } },
        { "io_uring-iopoll-forced-noreg", { { "backend", "io_uring" }, { "direct", "all" }, { "iopoll", "1" },
                                            { "fault_force_iopoll", "1" }, { "register", "none" } } },
        { "odirect-iopoll-forced", { { "backend", "odirect" }, { "iopoll", "1" }, { "fault_force_iopoll", "1" } } },
        // slot geometry above 4 KiB and small chunks
        { "odirect-align8k", { { "backend", "odirect" }, { "align", "8192" } } },
        { "odirect-align64k", { { "backend", "odirect" }, { "align", "65536" } } },
        { "io_uring-direct-align16k", { { "backend", "io_uring" }, { "direct", "all" }, { "align", "16384" } } },
        { "io_uring-chunk64k", { { "backend", "io_uring" }, { "direct", "all" }, { "chunk", "64K" } } },
        { "pread-chunk64k", { { "backend", "pread" }, { "direct", "extents" }, { "chunk", "64K" } } },
    };

    auto run = [&](const std::string& name, const std::function<void()>& f)
    {
        if (!only.empty() && name.find(only) == std::string::npos) return;
        int f0 = g_fail.load();
        int64_t t0 = now_ns();
        f();
        std::printf("%s %-44s %7.1f ms\n", g_fail.load() == f0 ? "PASS" : "FAIL", name.c_str(),
                    (double) (now_ns() - t0) / 1e6);
        std::fflush(stdout);
    };

    run("config", [] { test_config(); });
    for (auto& b : backends)
    {
        std::string n = b.name;
        run(n + "/rows", [&] { Engine e(cfg_of(b.ov)); test_rows(e, files, (n + "/rows").c_str()); });
        run(n + "/extents", [&] { Engine e(cfg_of(b.ov)); test_extents(e, big, small, (n + "/extents").c_str()); });
        run(n + "/concurrency", [&] { Engine e(cfg_of(b.ov)); test_concurrency(e, files, big, (n + "/concurrency").c_str()); });
        run(n + "/async", [&] { Engine e(cfg_of(b.ov)); test_async(e, big, (n + "/async").c_str()); });
        run(n + "/thread-exit", [&] { Engine e(cfg_of(b.ov)); test_thread_exit(e, big, (n + "/thread-exit").c_str()); });
        run(n + "/errors", [&] { Engine e(cfg_of(b.ov)); test_errors(e, big, (n + "/errors").c_str()); });
        run(n + "/fd-lifetime", [&] { Engine e(cfg_of(b.ov)); test_fd_lifetime(e, big, small, (n + "/fd-lifetime").c_str()); });
        run(n + "/hold", [&] { test_hold(b.ov, big, (n + "/hold").c_str()); });
        run(n + "/windows", [&] { test_windows(b.ov, big, (n + "/windows").c_str()); });
        run(n + "/queue", [&] { test_queue_ops(b.ov, big, (n + "/queue").c_str()); });
        run(n + "/truncate", [&] { test_truncate(b.ov, (n + "/truncate").c_str()); });
        run(n + "/faults", [&] { test_faults(b.ov, files, big, (n + "/faults").c_str()); });
        run(n + "/shutdown", [&] { test_shutdown(b.ov, big, (n + "/shutdown").c_str()); });
        run(n + "/registered", [&] { test_registered(b.ov, big, (n + "/registered").c_str()); });
        run(n + "/fork", [&] { test_fork(b.ov, big, (n + "/fork").c_str()); });
        run(n + "/fork-busy", [&] { test_fork_busy(b.ov, big, (n + "/fork-busy").c_str()); });
        run(n + "/release-race", [&] { test_release_race(b.ov, big, (n + "/release-race").c_str()); });
        run(n + "/hold-class", [&] { test_hold_class(b.ov, big, (n + "/hold-class").c_str()); });
        run(n + "/overflow", [&] { Engine e(cfg_of(b.ov)); test_overflow(e, big, (n + "/overflow").c_str()); });
        run(n + "/forget", [&] { test_forget(b.ov, (n + "/forget").c_str()); });
        if (n == "pread" || n == "odirect" || n == "io_uring" || n == "io_uring-direct" ||
            n == "io_uring-sqpoll" || n == "io_uring-refused")
            run(n + "/keepalive", [&] { test_keepalive(b.ov, (n + "/keepalive").c_str()); });
        bool uring = b.ov.at("backend") == "io_uring" && !b.ov.count("fault_no_uring");
        if (uring && !b.ov.count("sqpoll"))
        {
            run(n + "/broken-submit", [&] { test_broken_submit(b.ov, big, (n + "/broken-submit").c_str()); });
            run(n + "/fatal-quiet", [&] { test_fatal_quiet(b.ov, big, (n + "/fatal-quiet").c_str()); });
            run(n + "/iowq-affinity", [&] { test_iowq_affinity(b.ov, big, (n + "/iowq-affinity").c_str()); });
        }
    }
    run("edf", [&] { test_edf(big, "edf"); });
    run("default-engine", [&] { test_default(big); });
    run("fork-default", [&] { test_fork_default(big); });

    ::close(big.fd);
    ::close(small.fd);
    ::unlink(big.path.c_str());
    ::unlink(small.path.c_str());
    int fds1 = count_fds();
    // the default engine is never destroyed at exit, but shutdown_default() released it
    CHECK(fds1 == fds0, "descriptor leak: %d open before, %d after", fds0, fds1);
    std::printf("%s: %d checks, %d failures\n", g_fail.load() ? "FAILED" : "ALL PASSED",
                g_checks.load(), g_fail.load());
    return g_fail.load() ? 1 : 0;
}
