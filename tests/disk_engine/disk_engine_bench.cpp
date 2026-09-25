// Native benchmark for the disk engine: latency distributions and throughput per backend and
// access pattern, with no Python in the loop (tests/disk_engine/bench_engine.py measures the
// same patterns through the extension and reports the Python share).
//
//   disk_engine_bench --file PATH | --scratch DIR [--scratch-mb N]
//                     [--backends pread,odirect,io_uring] [--direct auto|none|rows|extents|all]
//                     [--pattern rows|extents|mixed|all] [--batch 48,96,8192] [--tables 2]
//                     [--iters N] [--extent-bytes 13315596] [--inflight 1,2,4,8]
//                     [--mixed-bulk extents,rows] [--cold] [--max-gb G] [--set knob=value ...]
//
// rows     one call per batch: U unique random rows of every table (--tables 2: a 256-byte and
//          an 8-byte table 1 GiB apart, like one DeepSeek-V4.1 engram layer), latency from
//          submit to completion
// extents  N contiguous extents per call, all in flight at once, unaligned offsets
// mixed    decode row batches (class 0, hold, armed 0.5 ms ahead; --tables 2: both engram
//          tables) every 2 ms while a second thread runs bulk: `extents` streams extents 4 deep
//          (class 1 demand or class 2 prefetch; buffered with --direct none), `rows` makes
//          synchronous 8192-row class-2 gathers of both tables (a prefetch worker staging
//          prefill rows, warm unless --cold); row latency with and without it (with --cold the
//          rows are dropped from the page cache first)
//
// --cold drops the file's pages before every call (posix_fadvise DONTNEED): only for a file this
// run created (--scratch), never for model shards. --max-gb bounds the bytes read per case.

#include "disk/disk_engine.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

using namespace exl3_disk;

namespace
{

int64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, sep)) if (!x.empty()) v.push_back(x);
    return v;
}

struct Pct
{
    double p50, p90, p99, p999, mean, max;
};

Pct pct(std::vector<int64_t> v)
{
    Pct p {};
    if (v.empty()) return p;
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return (double) v[(size_t) std::min<double>((double) v.size() - 1, std::floor(q * (double) v.size()))] / 1e3; };
    p.p50 = at(0.5);
    p.p90 = at(0.9);
    p.p99 = at(0.99);
    p.p999 = at(0.999);
    double s = 0;
    for (int64_t x : v) s += (double) x;
    p.mean = s / (double) v.size() / 1e3;
    p.max = (double) v.back() / 1e3;
    return p;
}

struct Args
{
    std::string file, scratch;
    int64_t scratch_mb = 1024;
    std::vector<std::string> backends = { "pread", "odirect", "io_uring" };
    std::string direct = "auto";
    std::string pattern = "all";
    std::vector<int64_t> batches = { 48, 96, 8192 };
    int tables = 2;
    int iters = 200;
    int64_t extent_bytes = 13315596;
    std::vector<int64_t> inflight = { 1, 2, 4, 8 };
    std::vector<std::string> mixed_bulk = { "extents" };
    bool cold = false;
    double max_gb = 4.0;
    Overrides extra;
};

void usage()
{
    std::fprintf(stderr, "see the header of disk_engine_bench.cpp for the arguments\n");
    std::exit(2);
}

void emit(const char* backend, const char* pattern, const std::string& shape, int64_t calls,
          double wall_s, double units, const char* unit, const Pct& p)
{
    std::printf("%-10s %-8s %-28s calls %6lld  %10.1f %-6s  lat us p50 %8.1f p90 %8.1f p99 %8.1f "
                "p99.9 %8.1f max %8.1f\n", backend, pattern, shape.c_str(), (long long) calls,
                wall_s > 0 ? units / wall_s : 0.0, unit, p.p50, p.p90, p.p99, p.p999, p.max);
    std::fflush(stdout);
}

struct Ctx
{
    Args a;
    int fd = -1;
    int64_t size = 0;
    bool owned = false;
};

void drop(Ctx& c)
{
    if (c.a.cold && c.owned) (void) ::posix_fadvise(c.fd, 0, 0, POSIX_FADV_DONTNEED);
}

void bench_rows(Ctx& c, Engine& e, const char* backend)
{
    // table 0: 256-byte rows from offset 664; table 1: 8-byte rows 1 GiB further (or halfway)
    int64_t t1_base = std::min<int64_t>(1ll << 30, c.size / 2);
    int64_t rows = std::min((c.size - 664) / 256, (c.size - t1_base) / 8);
    rows = std::min(rows, (t1_base - 664) / 256);
    if (rows < 100000) rows = (c.size - 664) / 256 / 2;
    std::mt19937_64 rng(1);
    for (int64_t B : c.a.batches)
    {
        int64_t bytes_per = B * (256 + (c.a.tables > 1 ? 8 : 0)) * 16;   // ~pages touched, cold
        int iters = (int) std::max<int64_t>(1, std::min<int64_t>(c.a.iters,
                        (int64_t) (c.a.max_gb * 1e9) / std::max<int64_t>(1, bytes_per)));
        if (B >= 4096) iters = std::min(iters, std::max(3, c.a.iters / 20));
        std::vector<uint8_t> w((size_t) B * 256), s((size_t) B * 8);
        std::vector<int64_t> lat;
        std::uniform_int_distribution<int64_t> d(0, rows - 1);
        int64_t t_all = 0;
        for (int it = 0; it < iters; ++it)
        {
            std::vector<int64_t> ids((size_t) B);
            for (auto& x : ids) x = d(rng);
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            RowTable tb[2] = { { c.fd, 664, 256, w.data(), (int64_t) w.size() },
                               { c.fd, t1_base, 8, s.data(), (int64_t) s.size() } };
            drop(c);
            Options o;
            o.cls = kEngram;
            int64_t t0 = now_ns();
            int r = e.gather_rows(ids.data(), (int64_t) ids.size(), 0, tb, c.a.tables, o);
            int64_t dt = now_ns() - t0;
            if (r != 0)
            {
                std::fprintf(stderr, "rows: error %d\n", r);
                std::exit(1);
            }
            lat.push_back(dt);
            t_all += dt;
        }
        emit(backend, "rows", "U=" + std::to_string(B) + " tables=" + std::to_string(c.a.tables),
             (int64_t) lat.size(), (double) t_all / 1e9, (double) (B * (int64_t) lat.size()),
             "rows/s", pct(lat));
    }
}

void bench_extents(Ctx& c, Engine& e, const char* backend)
{
    int64_t L = c.a.extent_bytes;
    int64_t slot = ((4095 + L) / 4096 + 1) * 4096;
    std::mt19937_64 rng(2);
    for (int64_t k : c.a.inflight)
    {
        void* mem = nullptr;
        if (posix_memalign(&mem, 4096, (size_t) (slot * k)) != 0) std::abort();
        std::memset(mem, 0, (size_t) (slot * k));
        int iters = (int) std::max<int64_t>(1, std::min<int64_t>(c.a.iters,
                        (int64_t) (c.a.max_gb * 1e9) / (L * k)));
        std::vector<int64_t> lat;
        int64_t t_all = 0;
        std::uniform_int_distribution<int64_t> d(0, c.size - L - 1);
        for (int it = 0; it < iters; ++it)
        {
            std::vector<ExtentReq> ex;
            for (int64_t i = 0; i < k; ++i)
                ex.push_back({ c.fd, d(rng), L, static_cast<uint8_t*>(mem) + i * slot, slot });
            drop(c);
            Options o;
            o.cls = kExpert;
            int64_t t0 = now_ns();
            int r = e.read_extents(ex.data(), k, o, nullptr);
            int64_t dt = now_ns() - t0;
            if (r != 0)
            {
                std::fprintf(stderr, "extents: error %d\n", r);
                std::exit(1);
            }
            lat.push_back(dt);
            t_all += dt;
        }
        emit(backend, "extents", std::to_string(k) + " x " + std::to_string(L) + " B",
             (int64_t) lat.size(), (double) t_all / 1e9,
             (double) (L * k * (int64_t) lat.size()) / 1e9, "GB/s", pct(lat));
        std::free(mem);
    }
}

// Row batches every 2 ms (class 0, hold) with and without bulk in another thread: an extent
// stream (bulk_rows false) or synchronous 8192-row class-2 gathers (bulk_rows true)
void bench_mixed(Ctx& c, Engine& e, const char* backend, int bulk_cls, bool bulk_rows)
{
    int64_t L = c.a.extent_bytes;
    int64_t slot = ((4095 + L) / 4096 + 1) * 4096;
    const int depth = 4;
    void* mem = nullptr;
    if (posix_memalign(&mem, 4096, (size_t) (slot * depth)) != 0) std::abort();
    std::memset(mem, 0, (size_t) (slot * depth));
    int64_t t1_base = std::min<int64_t>(1ll << 30, c.size / 2);
    int64_t rows = std::min((t1_base - 664) / 256, (c.size - t1_base) / 8);
    const int tables = std::max(1, std::min(2, c.a.tables));
    for (int with_bulk = 0; with_bulk < 2; ++with_bulk)
    {
        std::atomic<bool> stop { false };
        std::atomic<int64_t> bulk_bytes { 0 };
        std::thread bulk;
        int64_t tb0 = now_ns();
        if (with_bulk && bulk_rows)
            bulk = std::thread([&]
            {
                std::mt19937_64 rng(5);
                std::uniform_int_distribution<int64_t> d(0, rows - 1);
                std::vector<uint8_t> w((size_t) 8192 * 256), s8((size_t) 8192 * 8);
                while (!stop.load())
                {
                    std::vector<int64_t> ids(8192);
                    for (auto& x : ids) x = d(rng);
                    std::sort(ids.begin(), ids.end());
                    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                    RowTable tb[2] = { { c.fd, 664, 256, w.data(), (int64_t) w.size() },
                                       { c.fd, t1_base, 8, s8.data(), (int64_t) s8.size() } };
                    Options o;
                    o.cls = kPrefetch;
                    int r = e.gather_rows(ids.data(), (int64_t) ids.size(), 0, tb, 2, o);
                    if (r != 0) std::fprintf(stderr, "bulk rows error %d\n", r);
                    bulk_bytes.fetch_add((int64_t) ids.size() * 264);
                }
            });
        else if (with_bulk)
            bulk = std::thread([&]
            {
                std::mt19937_64 rng(3);
                std::uniform_int_distribution<int64_t> d(0, c.size - L - 1);
                std::vector<TicketId> q;
                int i = 0;
                while (!stop.load())
                {
                    while ((int) q.size() < depth)
                    {
                        ExtentReq x { c.fd, d(rng), L, static_cast<uint8_t*>(mem) + (i++ % depth) * slot, slot };
                        Options o;
                        o.cls = bulk_cls;
                        q.push_back(e.submit_extents(&x, 1, o, nullptr));
                    }
                    int r = e.wait(q.front(), -1);
                    if (r != 0) std::fprintf(stderr, "bulk error %d\n", r);
                    e.release(q.front());
                    q.erase(q.begin());
                    bulk_bytes.fetch_add(L);
                }
                for (TicketId id : q) e.release(id);
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(with_bulk ? 20 : 0));
        std::mt19937_64 rng(4 + (uint64_t) with_bulk * 1000 + (uint64_t) bulk_cls * 77 +
                            (bulk_rows ? 333u : 0u));                                      // fresh rows
        std::uniform_int_distribution<int64_t> d(0, rows - 1);
        std::vector<int64_t> lat;
        std::vector<uint8_t> w(48 * 256), s8(48 * 8);
        int iters = std::min(c.a.iters, 400);
        for (int it = 0; it < iters; ++it)
        {
            std::vector<int64_t> ids(48);
            for (auto& x : ids) x = d(rng);
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            RowTable tb[2] = { { c.fd, 664, 256, w.data(), (int64_t) w.size() },
                               { c.fd, t1_base, 8, s8.data(), (int64_t) s8.size() } };
            Options o;
            o.cls = kEngram;
            o.hold = true;
            drop(c);
            e.arm_hold();
            std::this_thread::sleep_for(std::chrono::microseconds(500));   // sampling time
            int64_t t0 = now_ns();
            (void) e.gather_rows(ids.data(), (int64_t) ids.size(), 0, tb, tables, o);
            lat.push_back(now_ns() - t0);
            std::this_thread::sleep_for(std::chrono::microseconds(1500));
        }
        stop.store(true);
        if (bulk.joinable()) bulk.join();
        double secs = (double) (now_ns() - tb0) / 1e9;
        std::string shape = std::string(with_bulk ? "rows 48 + " : "rows 48 alone") +
                            (with_bulk ? (bulk_rows ? "8192-row gathers (cls 2)"
                                                    : bulk_cls == kExpert ? "extents (cls 1)"
                                                                          : "extents (cls 2)")
                                       : "");
        emit(backend, "mixed", shape, (int64_t) lat.size(), secs,
             with_bulk ? (double) bulk_bytes.load() / 1e9 : 0.0, with_bulk ? "GB/s" : "-",
             pct(lat));
    }
    std::free(mem);
}

}  // namespace

int main(int argc, char** argv)
{
    Ctx c;
    Args& a = c.a;
    for (int i = 1; i < argc; ++i)
    {
        std::string k = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        auto ints = [&]() { std::vector<int64_t> v; for (auto& s : split(val(), ',')) v.push_back(std::atoll(s.c_str())); return v; };
        if (k == "--file") a.file = val();
        else if (k == "--scratch") a.scratch = val();
        else if (k == "--scratch-mb") a.scratch_mb = std::atoll(val().c_str());
        else if (k == "--backends") a.backends = split(val(), ',');
        else if (k == "--direct") a.direct = val();
        else if (k == "--pattern") a.pattern = val();
        else if (k == "--batch") a.batches = ints();
        else if (k == "--tables") a.tables = std::atoi(val().c_str());
        else if (k == "--iters") a.iters = std::atoi(val().c_str());
        else if (k == "--extent-bytes") a.extent_bytes = std::atoll(val().c_str());
        else if (k == "--inflight") a.inflight = ints();
        else if (k == "--mixed-bulk") a.mixed_bulk = split(val(), ',');
        else if (k == "--cold") a.cold = true;
        else if (k == "--max-gb") a.max_gb = std::atof(val().c_str());
        else if (k == "--set")
        {
            std::string kv = val();
            size_t eq = kv.find('=');
            if (eq == std::string::npos) usage();
            a.extra[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        else usage();
    }
    if (a.file.empty() == a.scratch.empty()) usage();
    std::string path = a.file;
    if (!a.scratch.empty())
    {
        path = a.scratch + "/bench_scratch.bin";
        int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0)
        {
            std::perror(path.c_str());
            return 2;
        }
        std::vector<uint64_t> buf(1 << 17);
        std::mt19937_64 rng(9);
        for (int64_t done = 0; done < (a.scratch_mb << 20); done += (int64_t) buf.size() * 8)
        {
            for (auto& w : buf) w = rng();
            if (::pwrite(fd, buf.data(), buf.size() * 8, done) != (ssize_t) (buf.size() * 8))
            {
                std::perror("pwrite");
                return 2;
            }
        }
        ::fsync(fd);
        ::close(fd);
        c.owned = true;
    }
    c.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (c.fd < 0)
    {
        std::perror(path.c_str());
        return 2;
    }
    struct stat st;
    ::fstat(c.fd, &st);
    c.size = st.st_size;
    if (a.cold && !c.owned)
    {
        std::fprintf(stderr, "--cold is only allowed on a --scratch file this run created\n");
        return 2;
    }
    std::printf("file %s, %.2f GiB%s\n", path.c_str(), (double) c.size / (1 << 30),
                a.cold ? ", cold (pages dropped before every call)" : "");
    for (const std::string& b : a.backends)
    {
        Overrides ov = a.extra;
        ov["backend"] = b;
        ov["direct"] = a.direct;
        Engine e(config_from_env(ov));
        std::printf("-- %s\n", e.describe().c_str());
        if (a.pattern == "rows" || a.pattern == "all") bench_rows(c, e, b.c_str());
        if (a.pattern == "extents" || a.pattern == "all") bench_extents(c, e, b.c_str());
        if (a.pattern == "mixed" || a.pattern == "all")
        {
            for (const std::string& mb : a.mixed_bulk)
            {
                if (mb == "rows") bench_mixed(c, e, b.c_str(), kPrefetch, true);
                else
                {
                    bench_mixed(c, e, b.c_str(), kExpert, false);
                    bench_mixed(c, e, b.c_str(), kPrefetch, false);
                }
            }
        }
    }
    ::close(c.fd);
    if (c.owned) ::unlink(path.c_str());
    return 0;
}
