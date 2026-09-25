// Native benchmark driver for the disk engine. Nothing but the engine call is timed, and no
// Python runs in the process: tests/disk_engine/bench_suite.py starts one driver per engine
// configuration, passes the files and cases, and collects the JSON lines this writes (one
// object per case). A human-readable line per case goes to stdout.
//
//   disk_engine_bench --probe [--file PATH]...
//   disk_engine_bench --out FILE [--set knob=value]... [--label TEXT] [--seed N]
//                     [--layer TABLE[+TABLE...]]...        TABLE = PATH:OFFSET:ROW_BYTES:ROWS
//                     [--extent-region PATH:OFFSET:LENGTH]... [--extent-list FILE]
//                     [--extent-bytes N] [--max-read-gb G] [--preserve-cache 0|1]
//                     [--verify N] [--idle-sample-ms MS] --case SPEC...
//
// A layer is the set of tables one engine call reads for the same ids (a DeepSeek-V4.1 engram
// layer: a 256-byte row table and an 8-byte scale table). Row cases alternate between layers.
// Extents come from --extent-list (lines "PATH OFFSET LENGTH", real expert extents) or are drawn
// at random offsets inside --extent-region(s), --extent-bytes long.
//
// SPEC = KIND[:key=value,...]
//   rows     u=48 cache=cold iters=200 cls=0 hold=auto gap_us=0
//                                                             one call per batch of u unique ids;
//                                                             gap_us: the device idles that long
//                                                             before each call (decode's spacing)
//   extents  k=4 cache=cold iters=20 cls=1 reg=1              k extents in one call, all at once
//   mixed    bulk=extents|rows|none bulk_cls=1 depth=4 bulk_u=8192 u=48 cache=cold iters=200
//            period_us=2000 arm_us=500                        decode row batches (class 0, hold,
//                                                             armed arm_us ahead) every period
//                                                             while one thread runs bulk reads
//   null     iters=1000                                       an empty call: the fixed cost
//   wake     gaps=1/5/10/20/50/100/200/500 iters=6            one cold row of the first layer
//                                                             after each idle gap (ms): what a
//                                                             device that sleeps when idle costs
//
// cache = cold | warm | F (0 <= F <= 1). Before every call the pages the call reads are put in
// that state and checked: cold drops exactly those pages (posix_fadvise DONTNEED, rounded out to
// whole pages; never a global drop_caches), warm reads them, F makes that fraction of them
// resident and drops the rest. Residency is checked with cachestat(2) (Linux 6.5+), else
// mincore(2) on a mapping that is never touched. With --preserve-cache 1 (the default for files
// this driver did not create), every call's pages are put back afterwards: pages that were
// resident before the call are read again, the others dropped, so a run leaves the page cache of
// the files as it found it. O_DIRECT reads never use the page cache, whatever the cache state.
//
// Measured per case: call latency (p50/p90/p99/p99.9/max, nearest rank; p99.9 means something
// from ~1000 calls), throughput, device ops per second and their service time (the engine's
// histograms, 12.5 % buckets), bytes the process read from the device (/proc/self/io read_bytes)
// against payload bytes, process CPU time during the calls (all threads, io_uring workers and
// the SQPOLL thread included), and the host's other load around the case. --max-read-gb stops
// the run (the current case is marked truncated, later ones skipped) once the process has read
// that much from the device, preparation and restore included.

#include "disk/disk_engine.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#define EXL3_BENCH_HAVE_URING_H 1
#else
#define EXL3_BENCH_HAVE_URING_H 0
#endif

using namespace exl3_disk;

namespace
{

// ---- small utilities ---------------------------------------------------------------------------

int64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

[[noreturn]] void die(const std::string& msg)
{
    std::fprintf(stderr, "disk_engine_bench: %s\n", msg.c_str());
    std::exit(2);
}

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> v;
    std::string x;
    std::stringstream ss(s);
    while (std::getline(ss, x, sep)) if (!x.empty()) v.push_back(x);
    return v;
}

// Strict integer parse: the whole string, no overflow
bool parse_i64(const std::string& s, int64_t* out)
{
    if (s.empty() || s.size() > 20) return false;
    size_t i = (s[0] == '-') ? 1 : 0;
    if (i == s.size()) return false;
    for (size_t k = i; k < s.size(); ++k) if (s[k] < '0' || s[k] > '9') return false;
    errno = 0;
    char* end = nullptr;
    long long r = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0') return false;
    *out = (int64_t) r;
    return true;
}

int64_t need_i64(const std::string& s, int64_t lo, int64_t hi, const std::string& what)
{
    int64_t v = 0;
    if (!parse_i64(s, &v) || v < lo || v > hi)
        die(what + ": '" + s + "' is not an integer in [" + std::to_string(lo) + ", " +
            std::to_string(hi) + "]");
    return v;
}

double need_f64(const std::string& s, double lo, double hi, const std::string& what)
{
    errno = 0;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (s.empty() || errno != 0 || !end || *end != '\0' || !(v >= lo && v <= hi))
        die(what + ": '" + s + "' is not a number in [" + std::to_string(lo) + ", " +
            std::to_string(hi) + "]");
    return v;
}

int64_t round_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

// ---- JSON output -------------------------------------------------------------------------------

std::string jstr(const std::string& s)
{
    std::string r = "\"";
    for (char c : s)
    {
        unsigned char ch = static_cast<unsigned char>(c);
        switch (ch)
        {
            case '"': r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n"; break;
            case '\r': r += "\\r"; break;
            case '\t': r += "\\t"; break;
            default:
                if (ch < 0x20)
                {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", (unsigned) ch);
                    r += b;
                }
                else r += c;
        }
    }
    return r + "\"";
}

std::string jnum(double v)
{
    if (!std::isfinite(v)) return "null";
    char b[40];
    std::snprintf(b, sizeof b, "%.6g", v);
    return b;
}

std::string jint(int64_t v) { return std::to_string(v); }
std::string jbool(bool v) { return v ? "true" : "false"; }

class JObj
{
public:
    JObj& add(const std::string& k, const std::string& raw)
    {
        s_ += (s_.empty() ? "{" : ", ") + jstr(k) + ": " + raw;
        return *this;
    }
    std::string str() const { return s_.empty() ? "{}" : s_ + "}"; }
private:
    std::string s_;
};

std::string jarr(const std::vector<std::string>& raws)
{
    std::string s = "[";
    for (size_t i = 0; i < raws.size(); ++i) s += (i ? ", " : "") + raws[i];
    return s + "]";
}

// ---- statistics ----------------------------------------------------------------------------------

// Nearest rank: the smallest value with at least q of the samples at or below it
double rank(const std::vector<int64_t>& sorted, double q)
{
    if (sorted.empty()) return NAN;
    double r = std::ceil(q * (double) sorted.size());
    size_t i = (size_t) std::max(1.0, std::min((double) sorted.size(), r)) - 1;
    return (double) sorted[i];
}

std::string lat_json(std::vector<int64_t> v)
{
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (int64_t x : v) sum += (double) x;
    JObj o;
    o.add("n", jint((int64_t) v.size()));
    if (v.empty()) return o.str();
    o.add("min", jnum((double) v.front() / 1e3)).add("p50", jnum(rank(v, 0.5) / 1e3))
     .add("p90", jnum(rank(v, 0.9) / 1e3)).add("p99", jnum(rank(v, 0.99) / 1e3))
     .add("p99.9", jnum(rank(v, 0.999) / 1e3)).add("max", jnum((double) v.back() / 1e3))
     .add("mean", jnum(sum / (double) v.size() / 1e3));
    return o.str();
}

// Percentiles of an engine histogram (ns buckets, 8 per octave): the geometric middle of the
// bucket holding the rank, so within ~6 % of the true value
std::string hist_json(const std::vector<uint64_t>& h)
{
    uint64_t n = 0;
    for (uint64_t c : h) n += c;
    JObj o;
    o.add("n", jint((int64_t) n));
    if (!n) return o.str();
    auto at = [&](double q) -> double
    {
        uint64_t want = (uint64_t) std::max(1.0, std::ceil(q * (double) n)), acc = 0;
        for (size_t i = 0; i < h.size(); ++i)
        {
            acc += h[i];
            if (acc >= want)
            {
                double lo = (double) hist_bucket_lower((int) i);
                double hi = i + 1 < h.size() ? (double) hist_bucket_lower((int) i + 1) : lo;
                return std::sqrt(std::max(lo, 1.0) * std::max(hi, 1.0)) / 1e3;
            }
        }
        return NAN;
    };
    o.add("p50", jnum(at(0.5))).add("p90", jnum(at(0.9))).add("p99", jnum(at(0.99)))
     .add("p99.9", jnum(at(0.999)));
    return o.str();
}

// ---- process and host counters -----------------------------------------------------------------

double cpu_s(int who)
{
    struct rusage ru;
    if (::getrusage(who, &ru) != 0) return NAN;
    return (double) ru.ru_utime.tv_sec + (double) ru.ru_utime.tv_usec / 1e6 +
           (double) ru.ru_stime.tv_sec + (double) ru.ru_stime.tv_usec / 1e6;
}

// Bytes this process (every thread, io_uring workers included) caused to be read from storage;
// -1 when the kernel has no task I/O accounting
int64_t proc_read_bytes()
{
    FILE* f = std::fopen("/proc/self/io", "re");
    if (!f) return -1;
    char line[256];
    int64_t v = -1;
    while (std::fgets(line, sizeof line, f))
    {
        long long x = 0;
        if (std::sscanf(line, "read_bytes: %lld", &x) == 1) { v = (int64_t) x; break; }
    }
    std::fclose(f);
    return v;
}

// Busy and total jiffies over all CPUs (/proc/stat first line)
bool host_jiffies(double* busy, double* total)
{
    FILE* f = std::fopen("/proc/stat", "re");
    if (!f) return false;
    unsigned long long v[10] = {};
    int n = std::fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1],
                        &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]);
    std::fclose(f);
    if (n < 8) return false;
    // user nice system idle iowait irq softirq steal (guest time is inside user)
    double idle = (double) v[3] + (double) v[4];
    double all = 0;
    for (int i = 0; i < 8; ++i) all += (double) v[i];
    *busy = all - idle;
    *total = all;
    return true;
}

double clk_tck()
{
    long t = ::sysconf(_SC_CLK_TCK);
    return t > 0 ? (double) t : 100.0;
}

std::string loadavg()
{
    FILE* f = std::fopen("/proc/loadavg", "re");
    if (!f) return "null";
    double a = 0, b = 0, c = 0;
    int n = std::fscanf(f, "%lf %lf %lf", &a, &b, &c);
    std::fclose(f);
    if (n != 3) return "null";
    return jarr({ jnum(a), jnum(b), jnum(c) });
}

// Cores busy on the whole host over ms milliseconds (this process is idle meanwhile)
double host_busy_cores(int ms)
{
    double b0 = 0, t0 = 0, b1 = 0, t1 = 0;
    if (ms <= 0 || !host_jiffies(&b0, &t0)) return NAN;
    int64_t w0 = now_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    if (!host_jiffies(&b1, &t1)) return NAN;
    double secs = (double) (now_ns() - w0) / 1e9;
    return (b1 - b0) / clk_tck() / secs;
}

// ---- files and page-cache state -------------------------------------------------------------------

struct CsRange { uint64_t off, len; };
struct CsStat { uint64_t nr_cache, nr_dirty, nr_writeback, nr_evicted, nr_recently_evicted; };

#if defined(SYS_cachestat)
constexpr long kSysCachestat = SYS_cachestat;
#elif defined(__x86_64__) || defined(__aarch64__)
constexpr long kSysCachestat = 451;
#else
constexpr long kSysCachestat = -1;
#endif

int64_t g_page = 4096;

struct File
{
    std::string path;
    int fd = -1;
    int64_t size = 0;
    void* map = nullptr;            // mincore fallback only: mapped, never touched
};

std::vector<File> g_files;          // filled while the arguments are parsed, before any thread
std::mutex g_files_mx;              // file_index, and the mincore fallback's mapping
std::atomic<int> g_residency { 0 }; // 0 unknown, 1 cachestat, 2 mincore, 3 unavailable

int file_index(const std::string& path)
{
    std::lock_guard<std::mutex> lk(g_files_mx);
    for (size_t i = 0; i < g_files.size(); ++i) if (g_files[i].path == path) return (int) i;
    File f;
    f.path = path;
    f.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f.fd < 0) die(path + ": " + std::strerror(errno));
    struct stat st;
    if (::fstat(f.fd, &st) != 0) die(path + ": fstat: " + std::strerror(errno));
    if (!S_ISREG(st.st_mode)) die(path + ": not a regular file");
    f.size = (int64_t) st.st_size;
    // Our own descriptor: no readahead around the pages we warm or check. The engine opens its
    // own descriptors and applies its own advice to them.
    (void) ::posix_fadvise(f.fd, 0, 0, POSIX_FADV_RANDOM);
    g_files.push_back(f);
    return (int) g_files.size() - 1;
}

// Resident pages of [p0, p0 + n) (pages of g_page bytes), or -1 when it cannot be known
int64_t resident_count(int fi, int64_t p0, int64_t n)
{
    File& f = g_files[(size_t) fi];
    if (n <= 0) return 0;
    if (g_residency == 0 || g_residency == 1)
    {
        if (kSysCachestat >= 0)
        {
            CsRange r { (uint64_t) (p0 * g_page), (uint64_t) (n * g_page) };
            CsStat cs {};
            long rc = ::syscall(kSysCachestat, (unsigned) f.fd, &r, &cs, 0u);
            if (rc == 0)
            {
                g_residency = 1;
                return (int64_t) cs.nr_cache;
            }
            if (errno != ENOSYS && errno != EPERM && errno != EOPNOTSUPP)
                die(f.path + ": cachestat: " + std::strerror(errno));
        }
        g_residency = 2;
    }
    if (g_residency == 2)
    {
        // mincore reports page-cache residency of a file mapping without faulting anything in
        // (for files the process could open for writing; others read as not resident)
        void* map = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_files_mx);
            if (!f.map && f.size > 0)
            {
                void* m = ::mmap(nullptr, (size_t) f.size, PROT_READ, MAP_SHARED, f.fd, 0);
                if (m == MAP_FAILED)
                {
                    g_residency = 3;
                    return -1;
                }
                f.map = m;
            }
            map = f.map;
        }
        int64_t end = std::min(p0 + n, (f.size + g_page - 1) / g_page);
        if (end <= p0 || !map) return 0;
        std::vector<unsigned char> v((size_t) (end - p0));
        if (::mincore(static_cast<char*>(map) + p0 * g_page, (size_t) ((end - p0) * g_page),
                      v.data()) != 0)
        {
            g_residency = 3;
            return -1;
        }
        int64_t c = 0;
        for (unsigned char x : v) c += (x & 1);
        return c;
    }
    return -1;
}

const char* residency_method()
{
    switch (g_residency)
    {
        case 1: return "cachestat";
        case 2: return "mincore";
        case 3: return "unavailable";
        default: return "unused";
    }
}

struct PRange
{
    int f;
    int64_t p0, n;          // pages
};

// Sort and merge overlapping or adjacent ranges of the same file
void merge(std::vector<PRange>& v)
{
    std::sort(v.begin(), v.end(), [](const PRange& a, const PRange& b)
              { return a.f != b.f ? a.f < b.f : a.p0 < b.p0; });
    std::vector<PRange> o;
    for (const PRange& r : v)
    {
        if (!o.empty() && o.back().f == r.f && r.p0 <= o.back().p0 + o.back().n)
            o.back().n = std::max(o.back().n, r.p0 + r.n - o.back().p0);
        else o.push_back(r);
    }
    v.swap(o);
}

// Pages of the byte range [off, off + len) of file f (rounded out to whole pages)
PRange pages_of(int f, int64_t off, int64_t len)
{
    int64_t a = off / g_page, b = (off + len + g_page - 1) / g_page;
    return { f, a, b - a };
}

// Resident runs of [p0, p0 + n): bisect on counts, so a range that is all in or all out costs
// one query
bool resident_runs(int fi, int64_t p0, int64_t n, std::vector<PRange>& out)
{
    int64_t c = resident_count(fi, p0, n);
    if (c < 0) return false;
    if (c == 0) return true;
    if (c >= n)
    {
        out.push_back({ fi, p0, n });
        return true;
    }
    int64_t h = n / 2;
    return resident_runs(fi, p0, h, out) && resident_runs(fi, p0 + h, n - h, out);
}

int64_t total_pages(const std::vector<PRange>& v)
{
    int64_t t = 0;
    for (const PRange& r : v) t += r.n;
    return t;
}

int64_t resident_total(const std::vector<PRange>& v)
{
    int64_t t = 0;
    for (const PRange& r : v)
    {
        int64_t c = resident_count(r.f, r.p0, r.n);
        if (c < 0) return -1;
        t += c;
    }
    return t;
}

void drop(const std::vector<PRange>& v)
{
    for (const PRange& r : v)
    {
        int rc = ::posix_fadvise(g_files[(size_t) r.f].fd, (off_t) (r.p0 * g_page),
                                 (off_t) (r.n * g_page), POSIX_FADV_DONTNEED);
        if (rc != 0) die(g_files[(size_t) r.f].path + ": posix_fadvise: " + std::strerror(rc));
    }
}

// Read the pages through our descriptor (page cache); returns bytes read
int64_t warm(const std::vector<PRange>& v)
{
    thread_local std::vector<uint8_t> buf(1 << 20);
    int64_t got = 0;
    for (const PRange& r : v)
    {
        const File& f = g_files[(size_t) r.f];
        int64_t off = r.p0 * g_page, end = std::min(f.size, (r.p0 + r.n) * g_page);
        while (off < end)
        {
            size_t want = (size_t) std::min<int64_t>((int64_t) buf.size(), end - off);
            ssize_t n = ::pread(f.fd, buf.data(), want, (off_t) off);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) die(f.path + ": pread: " + std::strerror(errno));
            if (n == 0) break;
            off += n;
            got += n;
        }
    }
    return got;
}

// Pages of `all` not covered by the sorted runs `in` (both merged, same order)
std::vector<PRange> minus(const std::vector<PRange>& all, const std::vector<PRange>& in)
{
    std::vector<PRange> out;
    size_t j = 0;
    for (const PRange& r : all)
    {
        int64_t cur = r.p0, end = r.p0 + r.n;
        while (j < in.size() && (in[j].f < r.f || (in[j].f == r.f && in[j].p0 + in[j].n <= cur))) ++j;
        size_t k = j;
        while (k < in.size() && in[k].f == r.f && in[k].p0 < end)
        {
            if (in[k].p0 > cur) out.push_back({ r.f, cur, in[k].p0 - cur });
            cur = std::max(cur, in[k].p0 + in[k].n);
            ++k;
        }
        if (cur < end) out.push_back({ r.f, cur, end - cur });
    }
    return out;
}

// ---- cache preparation per call -------------------------------------------------------------------

struct CacheMode
{
    bool cold = true;
    double frac = 0.0;       // warm fraction (cold: 0, warm: 1)
    std::string name = "cold";
};

CacheMode parse_cache(const std::string& s)
{
    CacheMode m;
    m.name = s;
    if (s == "cold") { m.cold = true; m.frac = 0.0; }
    else if (s == "warm") { m.cold = false; m.frac = 1.0; }
    else
    {
        m.frac = need_f64(s, 0.0, 1.0, "cache");
        m.cold = m.frac == 0.0;
    }
    return m;
}

struct PrepTally
{
    int64_t calls = 0;
    int64_t pages = 0;              // pages the calls touch
    int64_t resident = 0;           // of those, resident when the call started
    int64_t want_resident = 0;
    bool known = true;
    int64_t restored_pages = 0;     // resident before, read back afterwards
    int64_t prep_read_bytes = 0;    // warm-up and restore reads
};

struct Prepared
{
    std::vector<PRange> ranges;     // merged pages of the call
    std::vector<PRange> before;     // resident runs before preparation (preserve only)
    bool preserve = false;
};

// Put the call's pages in the requested state; returns what to restore afterwards
Prepared prepare(std::vector<PRange> ranges, const CacheMode& m, bool preserve, std::mt19937_64& rng,
                 PrepTally& t)
{
    Prepared p;
    merge(ranges);
    p.ranges = ranges;
    p.preserve = preserve;
    if (preserve)
        for (const PRange& r : p.ranges)
            if (!resident_runs(r.f, r.p0, r.n, p.before)) { p.before.clear(); break; }
    drop(p.ranges);
    int64_t n = total_pages(p.ranges), want = 0;
    if (m.frac >= 1.0)
    {
        t.prep_read_bytes += warm(p.ranges);
        want = n;
    }
    else if (m.frac > 0.0)
    {
        // exactly round(frac * pages) pages, chosen uniformly
        std::vector<PRange> pick;
        want = (int64_t) std::llround(m.frac * (double) n);
        std::vector<int64_t> idx((size_t) n);
        for (int64_t i = 0; i < n; ++i) idx[(size_t) i] = i;
        for (int64_t i = 0; i < want; ++i)
        {
            std::uniform_int_distribution<int64_t> d(i, n - 1);
            std::swap(idx[(size_t) i], idx[(size_t) d(rng)]);
        }
        idx.resize((size_t) want);
        std::sort(idx.begin(), idx.end());
        size_t ri = 0;
        int64_t base = 0;
        for (int64_t k : idx)
        {
            while (k >= base + p.ranges[ri].n) { base += p.ranges[ri].n; ++ri; }
            pick.push_back({ p.ranges[ri].f, p.ranges[ri].p0 + (k - base), 1 });
        }
        merge(pick);
        t.prep_read_bytes += warm(pick);
    }
    int64_t res = resident_total(p.ranges);
    ++t.calls;
    t.pages += n;
    t.want_resident += want;
    if (res < 0) t.known = false;
    else t.resident += res;
    return p;
}

void restore(const Prepared& p, PrepTally& t)
{
    if (!p.preserve) return;
    drop(minus(p.ranges, p.before));
    std::vector<PRange> back;
    for (const PRange& r : p.before)
    {
        int64_t c = resident_count(r.f, r.p0, r.n);
        if (c >= 0 && c < r.n) back.push_back(r);
    }
    t.restored_pages += total_pages(back);
    t.prep_read_bytes += warm(back);
}

std::string prep_json(const PrepTally& t, const CacheMode& m)
{
    JObj o;
    o.add("mode", jstr(m.name)).add("method", jstr(residency_method()))
     .add("pages_per_call", jnum(t.calls ? (double) t.pages / (double) t.calls : NAN))
     .add("target_resident_fraction", jnum(t.pages ? (double) t.want_resident / (double) t.pages : NAN))
     .add("resident_fraction", t.known && t.pages ? jnum((double) t.resident / (double) t.pages)
                                                  : std::string("null"))
     .add("off_target_pages", t.known ? jint((int64_t) std::llabs(t.resident - t.want_resident))
                                      : std::string("null"))
     .add("restored_pages", jint(t.restored_pages))
     .add("prep_read_bytes", jint(t.prep_read_bytes));
    return o.str();
}

// ---- the benchmark ------------------------------------------------------------------------------

struct Table
{
    int f;
    int64_t off, row_bytes, rows;
};

struct ExtentSrc
{
    int f;
    int64_t off, len;
};

struct Budget
{
    int64_t limit = -1;             // bytes, -1 unlimited
    int64_t start = 0;
    bool proc = false;
    std::atomic<int64_t> est { 0 }; // fallback when /proc/self/io is missing

    void init(double gb)
    {
        limit = gb > 0 ? (int64_t) (gb * 1e9) : -1;
        int64_t v = proc_read_bytes();
        proc = v >= 0;
        start = proc ? v : 0;
    }
    int64_t used() const
    {
        if (proc)
        {
            int64_t v = proc_read_bytes();
            if (v >= 0) return v - start;
        }
        return est.load();
    }
    bool exhausted() const { return limit >= 0 && used() >= limit; }
};

struct Args
{
    std::string out, label;
    Overrides ov;
    uint64_t seed = 1;
    std::vector<std::vector<Table>> layers;
    std::vector<ExtentSrc> regions, extents;
    int64_t extent_bytes = 13315596;
    double max_read_gb = 0;
    int preserve = -1;              // -1: auto (on)
    int verify = 2;
    int idle_ms = 200;
    std::vector<std::string> cases;
    bool probe = false;
    std::vector<std::string> probe_files;
};

Args g_args;
Budget g_budget;
FILE* g_out = nullptr;

struct Kv
{
    std::string kind;
    std::map<std::string, std::string> kv;
    std::map<std::string, bool> used;

    std::string get(const std::string& k, const std::string& def)
    {
        used[k] = true;
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    void check_all_used(const std::string& spec) const
    {
        for (const auto& p : kv)
            if (!used.count(p.first)) die("case '" + spec + "': unknown key '" + p.first + "'");
    }
};

Kv parse_spec(const std::string& spec)
{
    Kv k;
    size_t c = spec.find(':');
    k.kind = spec.substr(0, c);
    if (c == std::string::npos) return k;
    for (const std::string& item : split(spec.substr(c + 1), ','))
    {
        size_t eq = item.find('=');
        if (eq == std::string::npos || eq == 0) die("case '" + spec + "': expected key=value, got '" + item + "'");
        k.kv[item.substr(0, eq)] = item.substr(eq + 1);
    }
    return k;
}

// Unique sorted ids in [0, rows), exactly u of them
std::vector<int64_t> pick_ids(std::mt19937_64& rng, int64_t u, int64_t rows)
{
    if (u > rows) die("rows: u=" + std::to_string(u) + " exceeds the table's " + std::to_string(rows) + " rows");
    std::uniform_int_distribution<int64_t> d(0, rows - 1);
    std::vector<int64_t> ids;
    ids.reserve((size_t) u);
    while ((int64_t) ids.size() < u)
    {
        while ((int64_t) ids.size() < u) ids.push_back(d(rng));
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    }
    return ids;
}

int64_t layer_rows(const std::vector<Table>& L)
{
    int64_t r = INT64_MAX;
    for (const Table& t : L) r = std::min(r, t.rows);
    return r;
}

// Row batch of one layer: destinations, the engine's tables, the pages it reads
struct RowBatch
{
    std::vector<int64_t> ids;
    std::vector<std::vector<uint8_t>> outs;
    std::vector<RowTable> tabs;
    std::vector<PRange> pages;

    void build(const std::vector<Table>& L, std::vector<int64_t> id)
    {
        ids = std::move(id);
        outs.resize(L.size());
        tabs.clear();
        pages.clear();
        for (size_t t = 0; t < L.size(); ++t)
        {
            const Table& T = L[t];
            outs[t].assign((size_t) ((int64_t) ids.size() * T.row_bytes), 0);
            tabs.push_back({ g_files[(size_t) T.f].fd, T.off, T.row_bytes, outs[t].data(),
                             (int64_t) outs[t].size() });
            for (int64_t id2 : ids) pages.push_back(pages_of(T.f, T.off + id2 * T.row_bytes, T.row_bytes));
        }
    }
};

// Compare sampled rows with a pread of the same bytes; returns mismatches
int verify_rows(const std::vector<Table>& L, const RowBatch& b, int per_table, std::mt19937_64& rng)
{
    int bad = 0;
    std::vector<uint8_t> ref;
    for (size_t t = 0; t < L.size() && per_table > 0; ++t)
    {
        const Table& T = L[t];
        ref.resize((size_t) T.row_bytes);
        std::uniform_int_distribution<size_t> d(0, b.ids.size() - 1);
        for (int s = 0; s < per_table && !b.ids.empty(); ++s)
        {
            size_t i = s == 0 ? 0 : (s == 1 ? b.ids.size() - 1 : d(rng));
            int64_t off = T.off + b.ids[i] * T.row_bytes;
            ssize_t n;
            do n = ::pread(g_files[(size_t) T.f].fd, ref.data(), ref.size(), (off_t) off);
            while (n < 0 && errno == EINTR);
            if (n != (ssize_t) ref.size() ||
                std::memcmp(ref.data(), b.outs[t].data() + (int64_t) i * T.row_bytes, ref.size()) != 0)
                ++bad;
        }
    }
    return bad;
}

// 4 KiB windows at the start, middle and end of each payload
int verify_extent(const ExtentReq& x, int64_t payload_off)
{
    int bad = 0;
    int64_t w = std::min<int64_t>(4096, x.length);
    std::vector<uint8_t> ref((size_t) w);
    for (int64_t at : { (int64_t) 0, (x.length - w) / 2, x.length - w })
    {
        ssize_t n;
        do n = ::pread(x.fd, ref.data(), (size_t) w, (off_t) (x.offset + at));
        while (n < 0 && errno == EINTR);
        if (n != (ssize_t) w || std::memcmp(ref.data(), x.slot + payload_off + at, (size_t) w) != 0) ++bad;
    }
    return bad;
}

struct Aligned
{
    uint8_t* p = nullptr;
    size_t n = 0;
    explicit Aligned(size_t bytes) : n(bytes)
    {
        void* m = nullptr;
        if (::posix_memalign(&m, 4096, bytes) != 0) die("out of memory");
        p = static_cast<uint8_t*>(m);
        std::memset(p, 0, bytes);
    }
    ~Aligned() { std::free(p); }
    Aligned(const Aligned&) = delete;
    Aligned& operator=(const Aligned&) = delete;
};

// Slot size that holds any extent of `len` bytes whatever the alignment (G <= 64 KiB)
int64_t slot_for(int64_t len) { return round_up(len + 2 * 65536, 4096); }

struct ExtentPicker
{
    std::vector<ExtentSrc> list, regions;
    int64_t len;

    ExtentSrc pick(std::mt19937_64& rng) const
    {
        if (!list.empty())
        {
            std::uniform_int_distribution<size_t> d(0, list.size() - 1);
            return list[d(rng)];
        }
        // a region, weighted by the room it has for an extent
        int64_t room = 0;
        for (const ExtentSrc& r : regions) room += std::max<int64_t>(0, r.len - len + 1);
        if (room <= 0) die("extents: no region holds a " + std::to_string(len) + "-byte extent");
        std::uniform_int_distribution<int64_t> d(0, room - 1);
        int64_t x = d(rng);
        for (const ExtentSrc& r : regions)
        {
            int64_t k = std::max<int64_t>(0, r.len - len + 1);
            if (x < k) return { r.f, r.off + x, len };
            x -= k;
        }
        return { regions.back().f, regions.back().off, len };
    }
    bool empty() const { return list.empty() && regions.empty(); }
};

ExtentPicker g_picker;

struct CaseOut
{
    JObj o;
    std::string line;
};

// Counters around the timed calls
struct CallMeter
{
    std::vector<int64_t> lat;
    double cpu = 0;                 // process CPU seconds inside the calls
    int64_t dev = 0;                // bytes read from the device inside the calls
    bool dev_known = true;
    double t_io = 0, t_cpu = 0;
    int64_t io0 = 0;
    int64_t t0 = 0;

    void begin()
    {
        io0 = proc_read_bytes();
        t_cpu = cpu_s(RUSAGE_SELF);
        t0 = now_ns();
    }
    void end()
    {
        int64_t t1 = now_ns();
        double c1 = cpu_s(RUSAGE_SELF);
        int64_t io1 = proc_read_bytes();
        lat.push_back(t1 - t0);
        cpu += c1 - t_cpu;
        if (io0 < 0 || io1 < 0) dev_known = false;
        else dev += io1 - io0;
    }
    double wall_s() const
    {
        double s = 0;
        for (int64_t x : lat) s += (double) x;
        return s / 1e9;
    }
};

double warmup(Engine& e);

struct Host
{
    double foreign_before = NAN;
    double wake_us = NAN;
    std::string load_before;
    double b0 = 0, tot0 = 0, self0 = 0;
    int64_t w0 = 0;
    bool ok = false;

    void begin(Engine& e)
    {
        foreign_before = host_busy_cores(g_args.idle_ms);
        wake_us = warmup(e);                // after the idle sample: the device may have slept
        load_before = loadavg();
        ok = host_jiffies(&b0, &tot0);
        self0 = cpu_s(RUSAGE_SELF);
        w0 = now_ns();
    }
    // host busy cores over the case, and the part of it that was not this process
    void finish(JObj& o)
    {
        double b1 = 0, tot1 = 0;
        double secs = (double) (now_ns() - w0) / 1e9;
        double self = cpu_s(RUSAGE_SELF) - self0;
        JObj h;
        h.add("foreign_cores_before", jnum(foreign_before)).add("first_read_after_idle_us", jnum(wake_us))
         .add("idle_ms", jint(g_args.idle_ms)).add("loadavg_before", load_before)
         .add("loadavg_after", loadavg()).add("case_wall_s", jnum(secs));
        if (ok && host_jiffies(&b1, &tot1) && secs > 0)
        {
            double busy = (b1 - b0) / clk_tck();
            h.add("busy_cores", jnum(busy / secs)).add("self_cores", jnum(self / secs))
             .add("foreign_cores", jnum(std::max(0.0, busy - self) / secs));
        }
        o.add("host", h.str());
    }
};

std::string stats_json(const Stats& s, double wall, int64_t* ops_total)
{
    JObj o;
    std::vector<std::string> cls;
    int64_t ops = 0;
    for (int c = 0; c < kClasses; ++c)
    {
        const ClassStats& x = s.cls[c];
        ops += (int64_t) x.ops;
        if (!x.ops && !x.tickets && !x.held) continue;
        JObj k;
        k.add("class", jint(c)).add("tickets", jint((int64_t) x.tickets))
         .add("ops", jint((int64_t) x.ops)).add("op_errors", jint((int64_t) x.op_errors))
         .add("bytes", jint((int64_t) x.bytes)).add("held", jint((int64_t) x.held))
         .add("max_inflight", jint(x.max_inflight))
         .add("max_inflight_bytes", jint(x.max_inflight_bytes))
         .add("op_service_us", hist_json(x.hist_service))
         .add("op_queue_us", hist_json(x.hist_queue));
        cls.push_back(k.str());
    }
    o.add("classes", jarr(cls)).add("inline_reaped", jint((int64_t) s.inline_reaped))
     .add("reaper_reaped", jint((int64_t) s.reaper_reaped)).add("enters", jint((int64_t) s.enters))
     .add("resubmits", jint((int64_t) s.resubmits)).add("stray_cqes", jint((int64_t) s.stray_cqes))
     .add("keepalive_reads", jint((int64_t) s.keepalive_reads))
     .add("ops_per_s", jnum(wall > 0 ? (double) ops / wall : NAN));
    *ops_total = ops;
    return o.str();
}

void emit(Engine& e, const std::string& spec, const std::string& kind, JObj& o,
          const std::string& human)
{
    JObj top;
    top.add("schema", "1").add("tool", jstr("disk_engine_bench")).add("label", jstr(g_args.label))
       .add("case", jstr(spec)).add("kind", jstr(kind));
    std::vector<std::string> ov;
    JObj knobs;
    for (const auto& p : g_args.ov) knobs.add(p.first, jstr(p.second));
    const Config& c = e.config();
    JObj cfg;
    cfg.add("backend", jstr(backend_name(c.backend))).add("knobs", knobs.str())
       .add("direct_rows", jbool(c.direct_rows)).add("direct_extents", jbool(c.direct_extents))
       .add("qd", jint(c.qd)).add("threads", jint(c.threads)).add("reserve0", jint(c.reserve0))
       .add("describe", jstr(e.describe()));
    std::vector<std::string> fb;
    for (const std::string& f : e.fallbacks()) fb.push_back(jstr(f));
    cfg.add("fallbacks", jarr(fb));
    top.add("config", cfg.str());
    std::string body = o.str();
    std::string head = top.str();
    // splice: {top..., body...}
    std::string line = head.substr(0, head.size() - 1) + (body.size() > 2 ? ", " + body.substr(1) : "}");
    if (g_out)
    {
        std::fprintf(g_out, "%s\n", line.c_str());
        std::fflush(g_out);
    }
    std::printf("%-9s %-44s %s\n", backend_name(c.backend), spec.c_str(), human.c_str());
    std::fflush(stdout);
}

std::string human_lat(const std::vector<int64_t>& v0)
{
    std::vector<int64_t> v = v0;
    std::sort(v.begin(), v.end());
    char b[160];
    std::snprintf(b, sizeof b, "n %5zu  p50 %9.1f p90 %9.1f p99 %9.1f p99.9 %9.1f us", v.size(),
                  rank(v, 0.5) / 1e3, rank(v, 0.9) / 1e3, rank(v, 0.99) / 1e3, rank(v, 0.999) / 1e3);
    return b;
}

void run_rows(Engine& e, const std::string& spec, Kv& k, std::mt19937_64& rng)
{
    if (g_args.layers.empty()) die("rows: no --layer given");
    int64_t u = need_i64(k.get("u", "48"), 1, 1 << 24, "rows u");
    CacheMode cm = parse_cache(k.get("cache", "cold"));
    int64_t iters = need_i64(k.get("iters", "200"), 1, 10000000, "rows iters");
    int cls = (int) need_i64(k.get("cls", "0"), 0, kClasses - 1, "rows cls");
    std::string hs = k.get("hold", "auto");
    bool hold = hs == "auto" ? (cls <= 1 && u < 256) : need_i64(hs, 0, 1, "rows hold") == 1;
    std::string ls = k.get("layer", "all");
    int64_t gap_us = need_i64(k.get("gap_us", "0"), 0, 10000000, "rows gap_us");
    k.check_all_used(spec);
    std::vector<size_t> use;
    if (ls == "all") for (size_t i = 0; i < g_args.layers.size(); ++i) use.push_back(i);
    else use.push_back((size_t) need_i64(ls, 0, (int64_t) g_args.layers.size() - 1, "rows layer"));

    bool preserve = g_args.preserve != 0;
    Host host;
    host.begin(e);
    e.stats(true);
    CallMeter m;
    PrepTally pt;
    RowBatch b;
    int64_t bad = 0, verified = 0, units = 0;
    int err = 0;
    bool truncated = false;
    for (int64_t it = 0; it < iters; ++it)
    {
        if (g_budget.exhausted()) { truncated = true; break; }
        const std::vector<Table>& L = g_args.layers[use[(size_t) it % use.size()]];
        b.build(L, pick_ids(rng, u, layer_rows(L)));
        Prepared p = prepare(b.pages, cm, preserve, rng, pt);
        if (gap_us) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        Options o;
        o.cls = cls;
        o.hold = hold;
        m.begin();
        int r = e.gather_rows(b.ids.data(), (int64_t) b.ids.size(), 0, b.tabs.data(),
                              (int) b.tabs.size(), o);
        m.end();
        if (r != 0) { err = r; break; }
        units += (int64_t) b.ids.size();
        if (g_args.verify > 0)
        {
            bad += verify_rows(L, b, g_args.verify, rng);
            verified += (int64_t) L.size() * std::min<int64_t>(g_args.verify, (int64_t) b.ids.size());
        }
        restore(p, pt);
        g_budget.est.fetch_add(total_pages(p.ranges) * g_page);
    }
    Stats s = e.stats(false);
    double wall = m.wall_s();
    int64_t ops = 0;
    JObj o;
    o.add("params", JObj().add("u", jint(u)).add("tables", jint((int64_t) g_args.layers[use[0]].size()))
                          .add("layers", jint((int64_t) use.size())).add("cls", jint(cls))
                          .add("hold", jbool(hold)).add("gap_us", jint(gap_us)).str())
     .add("cache", prep_json(pt, cm))
     .add("calls", jint((int64_t) m.lat.size())).add("errors", jint(err ? 1 : 0))
     .add("error", err ? jstr(std::strerror(-err)) : std::string("null"))
     .add("truncated", jbool(truncated)).add("verify_failures", jint(bad))
     .add("verified_rows", jint(verified))
     .add("lat_us", lat_json(m.lat)).add("wall_s", jnum(wall))
     .add("rows_per_s", jnum(wall > 0 ? (double) units / wall : NAN))
     .add("engine", stats_json(s, wall, &ops)).add("ops", jint(ops))
     .add("iops", jnum(wall > 0 ? (double) ops / wall : NAN))
     .add("payload_bytes", jint((int64_t) (s.cls[cls].bytes)))
     .add("device_read_bytes", m.dev_known ? jint(m.dev) : std::string("null"))
     .add("cpu", JObj().add("call_s", jnum(m.cpu)).add("cores", jnum(wall > 0 ? m.cpu / wall : NAN))
                       .add("us_per_op", jnum(ops ? m.cpu * 1e6 / (double) ops : NAN))
                       .add("us_per_call", jnum(m.lat.empty() ? NAN : m.cpu * 1e6 / (double) m.lat.size())).str());
    host.finish(o);
    char tail[96];
    std::snprintf(tail, sizeof tail, " | %.4g rows/s %.4g IOPS%s%s", wall > 0 ? (double) units / wall : 0.0,
                  wall > 0 ? (double) ops / wall : 0.0, err ? " ERROR" : "", bad ? " VERIFY-FAIL" : "");
    emit(e, spec, "rows", o, human_lat(m.lat) + tail);
}

void run_extents(Engine& e, const std::string& spec, Kv& k, std::mt19937_64& rng)
{
    if (g_picker.empty()) die("extents: no --extent-list or --extent-region given");
    int64_t kk = need_i64(k.get("k", "4"), 1, 256, "extents k");
    CacheMode cm = parse_cache(k.get("cache", "cold"));
    int64_t iters = need_i64(k.get("iters", "20"), 1, 1000000, "extents iters");
    int cls = (int) need_i64(k.get("cls", "1"), 0, kClasses - 1, "extents cls");
    bool reg = need_i64(k.get("reg", "1"), 0, 1, "extents reg") == 1;
    k.check_all_used(spec);
    int64_t maxlen = g_picker.list.empty() ? g_picker.len : 0;
    for (const ExtentSrc& x : g_picker.list) maxlen = std::max(maxlen, x.len);
    int64_t slot = slot_for(maxlen);
    Aligned mem((size_t) (slot * kk));
    int reg_idx = reg ? e.register_buffer(mem.p, mem.n) : -1;

    bool preserve = g_args.preserve != 0;
    Host host;
    host.begin(e);
    e.stats(true);
    CallMeter m;
    PrepTally pt;
    int64_t bad = 0, bytes = 0;
    int err = 0;
    bool truncated = false;
    std::vector<ExtentReq> ex((size_t) kk);
    std::vector<int64_t> pay((size_t) kk);
    for (int64_t it = 0; it < iters; ++it)
    {
        if (g_budget.exhausted()) { truncated = true; break; }
        std::vector<PRange> pages;
        int64_t call_bytes = 0;
        for (int64_t i = 0; i < kk; ++i)
        {
            ExtentSrc x = g_picker.pick(rng);
            ex[(size_t) i] = { g_files[(size_t) x.f].fd, x.off, x.len, mem.p + i * slot, slot };
            pages.push_back(pages_of(x.f, x.off, x.len));
            call_bytes += x.len;
        }
        Prepared p = prepare(pages, cm, preserve, rng, pt);
        Options o;
        o.cls = cls;
        m.begin();
        int r = e.read_extents(ex.data(), kk, o, pay.data());
        m.end();
        if (r != 0) { err = r; break; }
        bytes += call_bytes;
        if (g_args.verify > 0)
            for (int64_t i = 0; i < kk; ++i) bad += verify_extent(ex[(size_t) i], pay[(size_t) i]);
        restore(p, pt);
        g_budget.est.fetch_add(call_bytes);
    }
    if (reg_idx >= 0) e.unregister_buffer(reg_idx);
    Stats s = e.stats(false);
    double wall = m.wall_s();
    int64_t ops = 0;
    JObj o;
    o.add("params", JObj().add("k", jint(kk)).add("extent_bytes", jint(maxlen)).add("cls", jint(cls))
                          .add("registered", jbool(reg_idx >= 0))
                          .add("source", jstr(g_picker.list.empty() ? "random offsets in regions"
                                                                     : "extent list")).str())
     .add("cache", prep_json(pt, cm))
     .add("calls", jint((int64_t) m.lat.size())).add("errors", jint(err ? 1 : 0))
     .add("error", err ? jstr(std::strerror(-err)) : std::string("null"))
     .add("truncated", jbool(truncated)).add("verify_failures", jint(bad))
     .add("lat_us", lat_json(m.lat)).add("wall_s", jnum(wall))
     .add("gb_per_s", jnum(wall > 0 ? (double) bytes / wall / 1e9 : NAN))
     .add("engine", stats_json(s, wall, &ops)).add("ops", jint(ops))
     .add("iops", jnum(wall > 0 ? (double) ops / wall : NAN))
     .add("payload_bytes", jint(bytes))
     .add("device_read_bytes", m.dev_known ? jint(m.dev) : std::string("null"))
     .add("cpu", JObj().add("call_s", jnum(m.cpu)).add("cores", jnum(wall > 0 ? m.cpu / wall : NAN))
                       .add("us_per_op", jnum(ops ? m.cpu * 1e6 / (double) ops : NAN))
                       .add("us_per_gb", jnum(bytes ? m.cpu * 1e6 / ((double) bytes / 1e9) : NAN)).str());
    host.finish(o);
    char tail[96];
    std::snprintf(tail, sizeof tail, " | %.3f GB/s%s%s", wall > 0 ? (double) bytes / wall / 1e9 : 0.0,
                  err ? " ERROR" : "", bad ? " VERIFY-FAIL" : "");
    emit(e, spec, "extents", o, human_lat(m.lat) + tail);
}

void run_mixed(Engine& e, const std::string& spec, Kv& k, std::mt19937_64& rng)
{
    if (g_args.layers.empty()) die("mixed: no --layer given");
    std::string bulk = k.get("bulk", "extents");
    if (bulk != "extents" && bulk != "rows" && bulk != "none") die("mixed: bulk must be extents, rows or none");
    int bulk_cls = (int) need_i64(k.get("bulk_cls", bulk == "rows" ? "2" : "1"), 1, kClasses - 1, "mixed bulk_cls");
    int64_t depth = need_i64(k.get("depth", "4"), 1, 64, "mixed depth");
    int64_t bulk_u = need_i64(k.get("bulk_u", "8192"), 1, 1 << 22, "mixed bulk_u");
    int64_t u = need_i64(k.get("u", "48"), 1, 4096, "mixed u");
    CacheMode cm = parse_cache(k.get("cache", "cold"));
    int64_t iters = need_i64(k.get("iters", "200"), 1, 1000000, "mixed iters");
    int64_t period = need_i64(k.get("period_us", "2000"), 100, 10000000, "mixed period_us");
    int64_t arm = need_i64(k.get("arm_us", "500"), 0, period, "mixed arm_us");
    k.check_all_used(spec);
    if (bulk == "extents" && g_picker.empty()) die("mixed: bulk=extents needs extents");

    bool preserve = g_args.preserve != 0;
    int64_t maxlen = g_picker.list.empty() ? g_picker.len : 0;
    for (const ExtentSrc& x : g_picker.list) maxlen = std::max(maxlen, x.len);
    int64_t slot = slot_for(std::max<int64_t>(maxlen, 4096));
    std::unique_ptr<Aligned> mem;
    int reg_idx = -1;
    if (bulk == "extents")
    {
        mem.reset(new Aligned((size_t) (slot * depth)));
        reg_idx = e.register_buffer(mem->p, mem->n);
    }

    Host host;
    host.begin(e);
    e.stats(true);
    std::atomic<bool> stop { false };
    std::atomic<int64_t> bulk_bytes { 0 }, bulk_rows { 0 };
    std::atomic<int> bulk_err { 0 };
    std::atomic<int64_t> bulk_t0 { 0 }, bulk_t1 { 0 };
    PrepTally bulk_pt;                  // bulk thread only until joined
    std::thread th;
    uint64_t bulk_seed = rng();
    if (bulk == "extents")
        th = std::thread([&, bulk_seed]
        {
            std::mt19937_64 r2(bulk_seed);
            struct Live { TicketId id; int64_t slot_i; Prepared p; int64_t len; };
            std::deque<Live> q;
            std::vector<int64_t> free_slots;
            for (int64_t i = depth - 1; i >= 0; --i) free_slots.push_back(i);
            bulk_t0.store(now_ns());
            try
            {
                while (!stop.load())
                {
                    while (!free_slots.empty() && !g_budget.exhausted())
                    {
                        int64_t si = free_slots.back();
                        free_slots.pop_back();
                        ExtentSrc x = g_picker.pick(r2);
                        ExtentReq req { g_files[(size_t) x.f].fd, x.off, x.len, mem->p + si * slot, slot };
                        CacheMode cold;
                        Prepared p = prepare({ pages_of(x.f, x.off, x.len) }, cold, preserve, r2, bulk_pt);
                        Options o;
                        o.cls = bulk_cls;
                        q.push_back({ e.submit_extents(&req, 1, o, nullptr), si, p, x.len });
                    }
                    if (q.empty()) break;           // budget exhausted
                    int r = e.wait(q.front().id, -1);
                    e.release(q.front().id);
                    if (r != 0 && !bulk_err.load()) bulk_err.store(r);
                    restore(q.front().p, bulk_pt);
                    bulk_bytes.fetch_add(q.front().len);
                    g_budget.est.fetch_add(q.front().len);
                    free_slots.push_back(q.front().slot_i);
                    q.pop_front();
                }
            }
            catch (const std::exception& ex)
            {
                std::fprintf(stderr, "bulk: %s\n", ex.what());
                bulk_err.store(-EIO);
            }
            for (Live& l : q)
            {
                e.release(l.id);            // waits for its reads in flight
                restore(l.p, bulk_pt);
            }
            bulk_t1.store(now_ns());
        });
    else if (bulk == "rows")
        th = std::thread([&, bulk_seed]
        {
            std::mt19937_64 r2(bulk_seed);
            RowBatch bb;
            size_t li = 0;
            bulk_t0.store(now_ns());
            try
            {
                while (!stop.load() && !g_budget.exhausted())
                {
                    const std::vector<Table>& L = g_args.layers[li++ % g_args.layers.size()];
                    bb.build(L, pick_ids(r2, bulk_u, layer_rows(L)));
                    Prepared p = prepare(bb.pages, cm, preserve, r2, bulk_pt);
                    Options o;
                    o.cls = bulk_cls;
                    int r = e.gather_rows(bb.ids.data(), (int64_t) bb.ids.size(), 0, bb.tabs.data(),
                                          (int) bb.tabs.size(), o);
                    if (r != 0 && !bulk_err.load()) bulk_err.store(r);
                    restore(p, bulk_pt);
                    int64_t by = 0;
                    for (const Table& t : L) by += t.row_bytes * (int64_t) bb.ids.size();
                    bulk_bytes.fetch_add(by);
                    bulk_rows.fetch_add((int64_t) bb.ids.size());
                    g_budget.est.fetch_add(total_pages(p.ranges) * g_page);
                }
            }
            catch (const std::exception& ex)
            {
                std::fprintf(stderr, "bulk: %s\n", ex.what());
                bulk_err.store(-EIO);
            }
            bulk_t1.store(now_ns());
        });
    if (bulk != "none") std::this_thread::sleep_for(std::chrono::milliseconds(20));

    CallMeter m;
    PrepTally pt;
    RowBatch b;
    int64_t bad = 0;
    int err = 0;
    bool truncated = false;
    double cpu0 = cpu_s(RUSAGE_SELF);
    int64_t w0 = now_ns(), io0 = proc_read_bytes();
    for (int64_t it = 0; it < iters; ++it)
    {
        int64_t next = w0 + it * period * 1000;
        if (g_budget.exhausted()) { truncated = true; break; }
        const std::vector<Table>& L = g_args.layers[(size_t) it % g_args.layers.size()];
        b.build(L, pick_ids(rng, u, layer_rows(L)));
        Prepared p = prepare(b.pages, cm, preserve, rng, pt);
        int64_t now = now_ns();
        if (next > now) std::this_thread::sleep_for(std::chrono::nanoseconds(next - now));
        e.arm_hold();                                   // sampling starts
        if (arm) std::this_thread::sleep_for(std::chrono::microseconds(arm));
        Options o;
        o.cls = kEngram;
        o.hold = true;
        m.begin();
        int r = e.gather_rows(b.ids.data(), (int64_t) b.ids.size(), 0, b.tabs.data(),
                              (int) b.tabs.size(), o);
        m.end();
        if (r != 0) { err = r; break; }
        if (g_args.verify > 0) bad += verify_rows(L, b, g_args.verify, rng);
        restore(p, pt);
        g_budget.est.fetch_add(total_pages(p.ranges) * g_page);
    }
    stop.store(true);
    if (th.joinable()) th.join();
    double secs = (double) (now_ns() - w0) / 1e9;
    double cpu = cpu_s(RUSAGE_SELF) - cpu0;
    int64_t io1 = proc_read_bytes();
    if (reg_idx >= 0) e.unregister_buffer(reg_idx);
    Stats s = e.stats(false);
    double bulk_secs = (double) (bulk_t1.load() - bulk_t0.load()) / 1e9;
    int64_t ops = 0;
    JObj o;
    o.add("params", JObj().add("bulk", jstr(bulk)).add("bulk_cls", jint(bulk_cls)).add("depth", jint(depth))
                          .add("bulk_u", jint(bulk_u)).add("u", jint(u)).add("period_us", jint(period))
                          .add("arm_us", jint(arm)).add("extent_bytes", jint(maxlen))
                          .add("registered", jbool(reg_idx >= 0)).str())
     .add("cache", prep_json(pt, cm))
     .add("calls", jint((int64_t) m.lat.size())).add("errors", jint((err || bulk_err.load()) ? 1 : 0))
     .add("error", err ? jstr(std::strerror(-err)) : bulk_err.load() ? jstr(std::string("bulk: ") + std::strerror(-bulk_err.load()))
                                                                     : std::string("null"))
     .add("truncated", jbool(truncated)).add("verify_failures", jint(bad))
     .add("lat_us", lat_json(m.lat)).add("wall_s", jnum(m.wall_s()))
     .add("bulk_gb_per_s", jnum(bulk != "none" && bulk_secs > 0 ? (double) bulk_bytes.load() / bulk_secs / 1e9 : NAN))
     .add("bulk_bytes", jint(bulk_bytes.load()))
     .add("bulk_rows_per_s", jnum(bulk == "rows" && bulk_secs > 0 ? (double) bulk_rows.load() / bulk_secs : NAN))
     .add("engine", stats_json(s, secs, &ops)).add("ops", jint(ops))
     .add("iops", jnum(secs > 0 ? (double) ops / secs : NAN))
     .add("device_read_bytes", io0 >= 0 && io1 >= 0 ? jint(io1 - io0) : std::string("null"))
     .add("cpu", JObj().add("case_s", jnum(cpu)).add("cores", jnum(secs > 0 ? cpu / secs : NAN))
                       .add("us_per_op", jnum(ops ? cpu * 1e6 / (double) ops : NAN)).str());
    host.finish(o);
    char tail[128];
    std::snprintf(tail, sizeof tail, " | bulk %.3f GB/s%s%s",
                  bulk != "none" && bulk_secs > 0 ? (double) bulk_bytes.load() / bulk_secs / 1e9 : 0.0,
                  (err || bulk_err.load()) ? " ERROR" : "", bad ? " VERIFY-FAIL" : "");
    emit(e, spec, "mixed", o, human_lat(m.lat) + tail);
}

// First-read latency against the idle time before it: one cold row (every table of the first
// layer, one call) after each gap, the gaps interleaved so drift spreads over all of them
void run_wake(Engine& e, const std::string& spec, Kv& k, std::mt19937_64& rng)
{
    if (g_args.layers.empty()) die("wake: no --layer given");
    std::vector<int64_t> gaps;
    for (const std::string& x : split(k.get("gaps", "1/5/10/20/50/100/200/500"), ';'))
        for (const std::string& y : split(x, '/'))
            gaps.push_back(need_i64(y, 0, 60000, "wake gaps"));
    int64_t iters = need_i64(k.get("iters", "6"), 1, 10000, "wake iters");
    k.check_all_used(spec);
    if (gaps.empty()) die("wake: no gaps");
    const std::vector<Table>& L = g_args.layers[0];
    bool preserve = g_args.preserve != 0;
    Host host;
    host.begin(e);
    e.stats(true);
    CacheMode cold;
    PrepTally pt;
    RowBatch b;
    std::vector<std::vector<int64_t>> lat(gaps.size());
    std::vector<int64_t> all;
    int err = 0;
    bool truncated = false;
    for (int64_t it = 0; it < iters && !err && !truncated; ++it)
    {
        for (size_t gi = 0; gi < gaps.size(); ++gi)
        {
            if (g_budget.exhausted()) { truncated = true; break; }
            b.build(L, pick_ids(rng, 1, layer_rows(L)));
            Prepared p = prepare(b.pages, cold, preserve, rng, pt);
            std::this_thread::sleep_for(std::chrono::milliseconds(gaps[gi]));
            Options o;
            int64_t t0 = now_ns();
            int r = e.gather_rows(b.ids.data(), 1, 0, b.tabs.data(), (int) b.tabs.size(), o);
            int64_t dt = now_ns() - t0;
            if (r != 0) { err = r; break; }
            lat[gi].push_back(dt);
            all.push_back(dt);
            restore(p, pt);
        }
    }
    Stats s = e.stats(false);
    std::vector<std::string> curve;
    std::string human;
    for (size_t gi = 0; gi < gaps.size(); ++gi)
    {
        std::vector<int64_t> v = lat[gi];
        std::sort(v.begin(), v.end());
        curve.push_back(JObj().add("gap_ms", jint(gaps[gi])).add("n", jint((int64_t) v.size()))
                              .add("p50", jnum(rank(v, 0.5) / 1e3)).add("p90", jnum(rank(v, 0.9) / 1e3))
                              .add("max", v.empty() ? std::string("null") : jnum((double) v.back() / 1e3)).str());
        char b2[64];
        std::snprintf(b2, sizeof b2, " %lldms:%.0f", (long long) gaps[gi], rank(v, 0.5) / 1e3);
        human += b2;
    }
    JObj o;
    o.add("params", JObj().add("tables", jint((int64_t) L.size())).str())
     .add("cache", prep_json(pt, cold)).add("calls", jint((int64_t) all.size()))
     .add("errors", jint(err ? 1 : 0)).add("error", err ? jstr(std::strerror(-err)) : std::string("null"))
     .add("truncated", jbool(truncated)).add("verify_failures", "0")
     .add("curve", jarr(curve)).add("lat_us", lat_json(all))
     .add("keepalive_reads", jint((int64_t) s.keepalive_reads));
    host.finish(o);
    emit(e, spec, "wake", o, "p50 us by gap:" + human);
}

void run_null(Engine& e, const std::string& spec, Kv& k)
{
    int64_t iters = need_i64(k.get("iters", "1000"), 1, 100000000, "null iters");
    k.check_all_used(spec);
    if (g_args.layers.empty()) die("null: no --layer given");
    std::vector<uint8_t> out(1);
    std::vector<RowTable> tabs;
    for (const Table& t : g_args.layers[0])
        tabs.push_back({ g_files[(size_t) t.f].fd, t.off, t.row_bytes, out.data(), 0 });
    Host host;
    host.begin(e);
    e.stats(true);
    CallMeter m;
    int err = 0;
    for (int64_t it = 0; it < iters; ++it)
    {
        Options o;
        m.begin();
        int r = e.gather_rows(nullptr, 0, 0, tabs.data(), (int) tabs.size(), o);
        m.end();
        if (r != 0) { err = r; break; }
    }
    JObj o;
    o.add("calls", jint((int64_t) m.lat.size())).add("errors", jint(err ? 1 : 0))
     .add("error", err ? jstr(std::strerror(-err)) : std::string("null"))
     .add("lat_us", lat_json(m.lat)).add("wall_s", jnum(m.wall_s()))
     .add("cpu", JObj().add("call_s", jnum(m.cpu))
                       .add("us_per_call", jnum(m.lat.empty() ? NAN : m.cpu * 1e6 / (double) m.lat.size())).str());
    host.finish(o);
    emit(e, spec, "null", o, human_lat(m.lat));
}

// The engine opens a file (and its O_DIRECT descriptor, and its fixed-file slot) on first use,
// and a device that sat idle may take milliseconds to serve its first read (power states, a
// host's idle I/O thread). Read one cold row of every table and one small cold extent of every
// extent file, outside the timed calls, so no case pays either; the pages touched are put back
// as they were. Returns the slowest of these reads (us): the cost of the first read after idle.
double warmup(Engine& e)
{
    PrepTally t;
    int64_t worst = 0;
    std::mt19937_64 rng(7);
    CacheMode cold;
    for (const std::vector<Table>& L : g_args.layers)
    {
        RowBatch b;
        b.build(L, { 0 });
        Prepared p = prepare(b.pages, cold, true, rng, t);
        Options o;
        int64_t t0 = now_ns();
        int r = e.gather_rows(b.ids.data(), 1, 0, b.tabs.data(), (int) b.tabs.size(), o);
        worst = std::max(worst, now_ns() - t0);
        if (r != 0) die(std::string("warm-up row read failed: ") + std::strerror(-r));
        restore(p, t);
    }
    std::vector<int> seen;
    std::vector<ExtentSrc> src = g_picker.list;
    src.insert(src.end(), g_picker.regions.begin(), g_picker.regions.end());
    Aligned mem((size_t) slot_for(4096));
    for (const ExtentSrc& x : src)
    {
        if (std::find(seen.begin(), seen.end(), x.f) != seen.end()) continue;
        seen.push_back(x.f);
        ExtentReq req { g_files[(size_t) x.f].fd, x.off, std::min<int64_t>(4096, x.len), mem.p,
                        (int64_t) mem.n };
        Prepared p = prepare({ pages_of(x.f, req.offset, req.length) }, cold, true, rng, t);
        Options o;
        o.cls = kExpert;
        int64_t t0 = now_ns();
        int r = e.read_extents(&req, 1, o, nullptr);
        worst = std::max(worst, now_ns() - t0);
        if (r != 0) die(std::string("warm-up extent read failed: ") + std::strerror(-r));
        restore(p, t);
    }
    e.stats(true);
    return (double) worst / 1e3;
}

// ---- probe ----------------------------------------------------------------------------------------

std::string probe_json()
{
    JObj o;
    struct utsname u;
    if (::uname(&u) == 0) o.add("kernel", jstr(u.release)).add("machine", jstr(u.machine));
    o.add("page_size", jint(g_page)).add("online_cpus", jint((int64_t) ::sysconf(_SC_NPROCESSORS_ONLN)));

    // io_uring: can this process create a ring (seccomp, io_uring_disabled, old kernel)?
#if EXL3_BENCH_HAVE_URING_H && defined(__NR_io_uring_setup)
    {
        struct io_uring_params p;
        std::memset(&p, 0, sizeof p);
        long fd = ::syscall(__NR_io_uring_setup, 4u, &p);
        if (fd >= 0)
        {
            o.add("io_uring", jstr("ok")).add("io_uring_features", jint((int64_t) p.features));
            ::close((int) fd);
        }
        else o.add("io_uring", jstr(std::string("refused: ") + std::strerror(errno)));
    }
#else
    o.add("io_uring", jstr("not compiled in"));
#endif
    std::vector<std::string> files;
    for (const std::string& path : g_args.probe_files)
    {
        int fi = file_index(path);
        const File& f = g_files[(size_t) fi];
        JObj x;
        x.add("path", jstr(path)).add("size", jint(f.size));
        int64_t c = resident_count(fi, 0, std::min<int64_t>(16, (f.size + g_page - 1) / g_page));
        x.add("residency_method", jstr(residency_method())).add("residency_probe", c >= 0 ? jint(c) : "null");
#ifdef STATX_DIOALIGN
        struct statx sx;
        std::memset(&sx, 0, sizeof sx);
        if (::statx(f.fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 && (sx.stx_mask & STATX_DIOALIGN))
            x.add("dio_mem_align", jint(sx.stx_dio_mem_align)).add("dio_offset_align", jint(sx.stx_dio_offset_align));
        else x.add("dio_align", jstr("not reported"));
#else
        x.add("dio_align", jstr("statx DIOALIGN not compiled in"));
#endif
        int d = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        x.add("o_direct", jstr(d >= 0 ? "ok" : std::string("refused: ") + std::strerror(errno)));
        if (d >= 0) ::close(d);
        files.push_back(x.str());
    }
    o.add("files", jarr(files));
    // what each backend resolves to with default knobs (fallbacks show here)
    std::vector<std::string> be;
    for (const char* b : { "pread", "odirect", "io_uring" })
    {
        Overrides ov = g_args.ov;
        ov["backend"] = b;
        try
        {
            Engine e(config_from_env(ov));
            JObj x;
            x.add("backend", jstr(b)).add("describe", jstr(e.describe()));
            be.push_back(x.str());
        }
        catch (const std::exception& ex)
        {
            be.push_back(JObj().add("backend", jstr(b)).add("error", jstr(ex.what())).str());
        }
    }
    o.add("engines", jarr(be));
    o.add("auto_backend", jstr(backend_name(auto_backend())))
     .add("auto_ngram_route", jstr(auto_ngram_engine() ? "engine" : "original"));
    return o.str();
}

// ---- arguments -----------------------------------------------------------------------------------

void usage()
{
    std::fprintf(stderr, "see the header of tests/disk_engine/disk_engine_bench.cpp for the arguments\n");
    std::exit(2);
}

Table parse_table(const std::string& s)
{
    // PATH:OFFSET:ROW_BYTES:ROWS (the path may contain ':'; the last three fields are numbers)
    size_t c3 = s.rfind(':');
    size_t c2 = c3 == std::string::npos || c3 == 0 ? std::string::npos : s.rfind(':', c3 - 1);
    size_t c1 = c2 == std::string::npos || c2 == 0 ? std::string::npos : s.rfind(':', c2 - 1);
    if (c1 == std::string::npos) die("--layer: expected PATH:OFFSET:ROW_BYTES:ROWS, got '" + s + "'");
    Table t;
    t.f = file_index(s.substr(0, c1));
    t.off = need_i64(s.substr(c1 + 1, c2 - c1 - 1), 0, INT64_MAX / 2, "--layer offset");
    t.row_bytes = need_i64(s.substr(c2 + 1, c3 - c2 - 1), 1, 1 << 20, "--layer row bytes");
    t.rows = need_i64(s.substr(c3 + 1), 1, INT64_MAX / 2, "--layer rows");
    const File& f = g_files[(size_t) t.f];
    if (t.rows > (f.size - t.off) / t.row_bytes)
        die("--layer '" + s + "': the table runs past the end of the file (" + std::to_string(f.size) + " bytes)");
    return t;
}

ExtentSrc parse_region(const std::string& s)
{
    size_t c2 = s.rfind(':');
    size_t c1 = c2 == std::string::npos || c2 == 0 ? std::string::npos : s.rfind(':', c2 - 1);
    if (c1 == std::string::npos) die("--extent-region: expected PATH:OFFSET:LENGTH, got '" + s + "'");
    ExtentSrc r;
    r.f = file_index(s.substr(0, c1));
    r.off = need_i64(s.substr(c1 + 1, c2 - c1 - 1), 0, INT64_MAX / 2, "--extent-region offset");
    r.len = need_i64(s.substr(c2 + 1), 1, INT64_MAX / 2, "--extent-region length");
    if (r.off + r.len > g_files[(size_t) r.f].size) die("--extent-region '" + s + "' runs past the end of the file");
    return r;
}

void read_extent_list(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "re");
    if (!f) die(path + ": " + std::strerror(errno));
    char line[4096];
    int n = 0;
    while (std::fgets(line, sizeof line, f))
    {
        ++n;
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.empty() || s[0] == '#') continue;
        // PATH OFFSET LENGTH, split from the right (paths may hold spaces)
        size_t b = s.rfind(' ');
        size_t a = b == std::string::npos || b == 0 ? std::string::npos : s.rfind(' ', b - 1);
        if (a == std::string::npos) die(path + ":" + std::to_string(n) + ": expected PATH OFFSET LENGTH");
        ExtentSrc x;
        x.f = file_index(s.substr(0, a));
        x.off = need_i64(s.substr(a + 1, b - a - 1), 0, INT64_MAX / 2, "extent offset");
        x.len = need_i64(s.substr(b + 1), 1, 1ll << 34, "extent length");
        if (x.off + x.len > g_files[(size_t) x.f].size) die(path + ":" + std::to_string(n) + ": extent past the end of the file");
        g_picker.list.push_back(x);
    }
    std::fclose(f);
    if (g_picker.list.empty()) die(path + ": no extents");
}

}  // namespace

int main(int argc, char** argv)
{
    long ps = ::sysconf(_SC_PAGESIZE);
    if (ps > 0) g_page = ps;
    Args& a = g_args;
    std::vector<std::string> layer_specs, region_specs;
    std::string extent_list;
    for (int i = 1; i < argc; ++i)
    {
        std::string k = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (k == "--probe") a.probe = true;
        else if (k == "--file") a.probe_files.push_back(val());
        else if (k == "--out") a.out = val();
        else if (k == "--label") a.label = val();
        else if (k == "--seed") a.seed = (uint64_t) need_i64(val(), 0, INT64_MAX, "--seed");
        else if (k == "--layer") layer_specs.push_back(val());
        else if (k == "--extent-region") region_specs.push_back(val());
        else if (k == "--extent-list") extent_list = val();
        else if (k == "--extent-bytes") a.extent_bytes = need_i64(val(), 4096, 1ll << 32, "--extent-bytes");
        else if (k == "--max-read-gb") a.max_read_gb = need_f64(val(), 0, 1e6, "--max-read-gb");
        else if (k == "--preserve-cache") a.preserve = (int) need_i64(val(), 0, 1, "--preserve-cache");
        else if (k == "--verify") a.verify = (int) need_i64(val(), 0, 64, "--verify");
        else if (k == "--idle-sample-ms") a.idle_ms = (int) need_i64(val(), 0, 60000, "--idle-sample-ms");
        else if (k == "--case") a.cases.push_back(val());
        else if (k == "--set")
        {
            std::string kv = val();
            size_t eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) usage();
            a.ov[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        else usage();
    }
    try
    {
        if (a.probe)
        {
            std::printf("%s\n", probe_json().c_str());
            return 0;
        }
        for (const std::string& s : layer_specs)
        {
            std::vector<Table> L;
            for (const std::string& t : split(s, '+')) L.push_back(parse_table(t));
            if (L.empty()) die("--layer: empty");
            a.layers.push_back(L);
        }
        for (const std::string& s : region_specs) g_picker.regions.push_back(parse_region(s));
        if (!extent_list.empty()) read_extent_list(extent_list);
        g_picker.len = a.extent_bytes;
        if (a.cases.empty()) die("no --case given");
        if (!a.out.empty())
        {
            g_out = std::fopen(a.out.c_str(), "ae");
            if (!g_out) die(a.out + ": " + std::strerror(errno));
        }
        // --preserve-cache promises to leave the page cache as found: refuse to run without a
        // way to see residency
        for (size_t i = 0; i < g_files.size(); ++i)
        {
            // mincore sees the page cache only for files this process could write: check that
            // a page just read shows as resident, or treat residency as unknown
            if (resident_count((int) i, 0, 1) >= 0 && g_residency == 2 && g_files[i].size > 0)
            {
                (void) warm({ { (int) i, 0, 1 } });
                if (resident_count((int) i, 0, 1) < 1) g_residency = 3;
            }
        }
        if (a.preserve != 0)
            for (size_t i = 0; i < g_files.size(); ++i)
                if (resident_count((int) i, 0, 1) < 0)
                    die("--preserve-cache needs cachestat(2) or mincore(2) to see page-cache "
                        "residency, and neither works here; pass --preserve-cache 0 only for "
                        "files nothing else uses");
        g_budget.init(a.max_read_gb);
        Engine e(config_from_env(a.ov));
        std::printf("-- %s\n", e.describe().c_str());
        warmup(e);
        std::mt19937_64 rng(a.seed);
        int rc = 0;
        for (const std::string& spec : a.cases)
        {
            Kv k = parse_spec(spec);
            if (g_budget.exhausted())
            {
                JObj o;
                o.add("skipped", jstr("read budget exhausted")).add("calls", "0").add("errors", "0");
                emit(e, spec, k.kind, o, "skipped: read budget exhausted");
                continue;
            }
            if (k.kind == "rows") run_rows(e, spec, k, rng);
            else if (k.kind == "extents") run_extents(e, spec, k, rng);
            else if (k.kind == "mixed") run_mixed(e, spec, k, rng);
            else if (k.kind == "null") run_null(e, spec, k);
            else if (k.kind == "wake") run_wake(e, spec, k, rng);
            else die("unknown case kind '" + k.kind + "' (rows, extents, mixed, null, wake)");
        }
        std::printf("-- read from the device: %.3f GB (%s)\n", (double) g_budget.used() / 1e9,
                    g_budget.proc ? "/proc/self/io" : "estimate");
        if (g_out) std::fclose(g_out);
        for (File& f : g_files)
        {
            if (f.map) ::munmap(f.map, (size_t) f.size);
            ::close(f.fd);
        }
        return rc;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "disk_engine_bench: %s\n", ex.what());
        return 1;
    }
}
