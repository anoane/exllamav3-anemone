// Standalone test driver of the expert tier's host side (exllamav3_ext/tier/tier_host.cpp) with the
// disk engine: synthetic "shards" written here (experts at unaligned offsets, some split into three
// pieces across two files), a HostCopier standing in for the GPU, and after every call a check that
// every pool slot, RAM tier slot and transient staging slot holds exactly the bytes of the expert the
// policy says it holds. Random traces over every policy x demote x admit, both engine backends,
// deferred returns with rescues, layer mode with the read-ahead, asynchronous refills, engine reads
// failing (pread fallback) and a shard cut short (the error names the expert). Built and run by
// tests/expert_tier/build.sh (release, ASan+UBSan, TSan). No torch, no CUDA.
//
//   tier_host_test --dir DIR [--quick] [--seed N]

#include "tier/tier_host.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace exl3_tier;

namespace
{

std::atomic<int> g_fail { 0 };
int g_checks = 0;
std::string g_dir;
bool g_quick = false;
uint64_t g_seed = 99;

#define CHECK(cond, ...)                                                                  \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_fail;                                                                     \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                    \
            std::fprintf(stderr, __VA_ARGS__);                                            \
            std::fprintf(stderr, "\n");                                                   \
        }                                                                                 \
    } while (0)

constexpr int64_t kTB = 12288;          // trellis bytes per projection
constexpr int64_t kAux = 211;           // aux bytes before each trellis (odd)

// A synthetic checkpoint: L x E experts over two files; expert k's projection p trellis at a known
// offset; experts whose index % 7 == 3 are read as three pieces
struct Ckpt
{
    int64_t L, E;
    std::vector<std::string> files;
    std::vector<std::vector<uint8_t>> data;
    std::vector<ExpertExtent> ext;
    std::vector<std::vector<uint8_t>> expected;     // per key: gate | up | down trellis
    Geometry geo;

    Ckpt(int64_t L_, int64_t E_, const std::string& tag, uint64_t seed) : L(L_), E(E_)
    {
        std::mt19937_64 rng(seed);
        files = { g_dir + "/" + tag + "_a.bin", g_dir + "/" + tag + "_b.bin" };
        data.resize(2);
        for (auto& d : data) d.resize(13);
        ext.resize((size_t) (L * E));
        expected.resize((size_t) (L * E));
        for (int64_t k = 0; k < L * E; ++k)
        {
            int f = k < L * E / 2 ? 0 : 1;
            std::vector<uint8_t>& d = data[(size_t) f];
            ExpertExtent& x = ext[(size_t) k];
            x.name = "expert." + std::to_string(k);
            bool split = k % 7 == 3;
            int64_t start = (int64_t) d.size();
            expected[(size_t) k].resize((size_t) (3 * kTB));
            for (int p = 0; p < 3; ++p)
            {
                for (int64_t i = 0; i < kAux + p; ++i) d.push_back((uint8_t) rng());
                int64_t at = (int64_t) d.size();
                for (int64_t i = 0; i < kTB; ++i)
                {
                    uint8_t b = (uint8_t) rng();
                    d.push_back(b);
                    expected[(size_t) k][(size_t) (p * kTB + i)] = b;
                }
                if (split)
                {
                    x.file[p] = f;
                    x.offset[p] = at;
                    x.length[p] = kTB;
                    x.tpiece[p] = p;
                    x.trel[p] = 0;
                }
                else
                {
                    x.tpiece[p] = 0;
                    x.trel[p] = at - start;
                }
                if (split && p == 1) for (int i = 0; i < 5; ++i) d.push_back(0x11);     // foreign bytes
            }
            if (split) x.npieces = 3;
            else
            {
                x.npieces = 1;
                x.file[0] = f;
                x.offset[0] = start;
                x.length[0] = (int64_t) d.size() - start;
            }
            for (int i = 0; i < 13; ++i) d.push_back(0x5a);
        }
        for (size_t f = 0; f < 2; ++f)
        {
            int fd = ::open(files[f].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (fd < 0 || ::write(fd, data[f].data(), data[f].size()) != (ssize_t) data[f].size()) std::abort();
            ::close(fd);
        }
        geo.projections = 3;
        for (int p = 0; p < 3; ++p) geo.proj_bytes[p] = kTB;
        int64_t one = ((3 * kTB + 3 * kAux + 3) / 4096 + 2) * 4096;
        int64_t three = 3 * ((kTB + 4095) / 4096 * 4096 + 4096);
        geo.ram_slot_bytes = std::max(one, three);
    }
    ~Ckpt()
    {
        for (auto& f : files) ::unlink(f.c_str());
    }
};

std::vector<int32_t> round_robin(int64_t L, int64_t E)
{
    std::vector<int32_t> o;
    for (int64_t e = 0; e < E; ++e)
        for (int64_t l = 0; l < L; ++l) o.push_back((int32_t) (l * E + e));
    return o;
}

// Every expert the policy places in the pool, a retiring slot or the RAM tier holds its bytes
void verify(TierHost& h, HostCopier& cp, const Ckpt& ck, const char* what)
{
    const size_t n = (size_t) (3 * kTB);
    for (int32_t s = 0; s < h.core.vram.S; ++s)
    {
        uint8_t st = h.core.vram.state[(size_t) s];
        if (st != kValid && st != kRetiring) continue;
        int32_t k = h.core.vram.key[(size_t) s];
        CHECK(std::memcmp(cp.pool(s), ck.expected[(size_t) k].data(), n) == 0, "%s: pool slot %d key %d", what, s, k);
    }
    std::vector<uint8_t> buf(n);
    for (int64_t r = 0; r < h.core.ram.R; ++r)
    {
        int32_t k = h.core.ram.key[(size_t) r];
        if (k < 0) continue;
        h.ram_trellis((int32_t) r, buf.data());
        CHECK(std::memcmp(buf.data(), ck.expected[(size_t) k].data(), n) == 0, "%s: RAM slot %lld key %d", what,
              (long long) r, k);
    }
}

struct Run
{
    std::unique_ptr<HostCopier> cp;
    std::unique_ptr<TierHost> h;
};

Run make(const Ckpt& ck, const PolicyConfig& pc, int S, int64_t R, bool defer, HostConfig hc = HostConfig())
{
    Run r;
    hc.chunk_bytes = 5 * ck.geo.ram_slot_bytes;         // several chunks, several registrations
    hc.staging_slots = (int) ck.E;
    r.cp.reset(new HostCopier(S, ck.E, 3 * kTB));
    r.h.reset(new TierHost(pc, ck.L, ck.E, S, R, 0, ck.geo, ck.files, ck.ext, hc, r.cp.get(), defer));
    return r;
}

void trace(const Ckpt& ck, const PolicyConfig& pc, int S, int64_t R, bool defer, int calls, uint64_t seed,
           const char* what, HostConfig hc = HostConfig())
{
    Run run = make(ck, pc, S, R, defer, hc);
    TierHost& h = *run.h;
    h.cold_fill(round_robin(ck.L, ck.E));
    verify(h, *run.cp, ck, what);
    std::mt19937_64 rng(seed);
    std::vector<double> w((size_t) ck.E);
    for (int64_t e = 0; e < ck.E; ++e) w[(size_t) e] = 1.0 / std::pow((double) (e + 1), 0.9);
    std::discrete_distribution<int32_t> pick(w.begin(), w.end());
    std::vector<int32_t> ids;
    int half = 0;
    for (int i = 0; i < calls; ++i)
    {
        int32_t lc = (int32_t) (i % ck.L);
        if (lc == 0) h.tick(1);
        ids.clear();
        uint64_t r = rng() % 100;
        char where[128];
        std::snprintf(where, sizeof where, "%s call %d", what, i);
        try
        {
            if (r < 4)
            {
                for (int j = 0; j < 40; ++j) ids.push_back((int32_t) (rng() % (uint64_t) ck.E));
                half ^= 1;
                if (rng() % 2) h.prefetch_layer(lc, 0);
                h.layer_call(lc, ids.data(), (int64_t) ids.size(), half);
                for (const Action& a : h.last_actions())
                    if (a.op == kActStageRam || a.op == kActStageSsd)
                    {
                        int32_t idx = a.op == kActStageRam ? a.c : a.b;
                        CHECK(std::memcmp(run.cp->staging(half, idx), ck.expected[(size_t) a.a].data(), (size_t) (3 * kTB)) == 0,
                              "%s: staging %d key %d", where, idx, a.a);
                    }
            }
            else
            {
                int rows = 1 << (rng() % 3);
                for (int j = 0; j < 6 * rows; ++j) ids.push_back(pick(rng));
                int mode = r < 8 ? kRouted : kDecode;
                Record rec = h.call(lc, ids.data(), (int64_t) ids.size(), mode);
                for (const Entry& x : rec.entries)
                {
                    const uint8_t* got = x.kind == kTransient ? run.cp->staging(0, x.dst) : run.cp->pool(x.dst);
                    CHECK(std::memcmp(got, ck.expected[(size_t) x.key].data(), (size_t) (3 * kTB)) == 0,
                          "%s: entry key %d kind %d", where, x.key, (int) x.kind);
                }
            }
            if (defer && rng() % 5 == 0) h.complete();
            if (!hc.deterministic && rng() % 3 == 0) h.poll();
            if (i % 5 == 0)
            {
                if (!hc.deterministic) h.drain();
                h.core.check(!defer);
                verify(h, *run.cp, ck, where);
            }
        }
        catch (const std::exception& e)
        {
            CHECK(false, "%s: %s", where, e.what());
            return;
        }
    }
    h.complete();
    h.drain();
    h.core.check(true);
    verify(h, *run.cp, ck, what);
    CHECK(h.error().empty(), "%s: error %s", what, h.error().c_str());
}

void configure(const exl3_disk::Overrides& ov) { exl3_disk::configure_default(ov); }

void test_traces()
{
    Ckpt ck(3, 24, "traces", g_seed);
    std::mt19937_64 rng(g_seed + 1);
    const int calls = g_quick ? 60 : 200;
    for (const char* backend : { "io_uring", "pread" })
    {
        configure({ { "backend", backend } });
        for (int pol = 0; pol < 3; ++pol)
            for (int dem = 0; dem < 4; ++dem)
                for (int adm = 0; adm < 3; ++adm)
                {
                    PolicyConfig pc;
                    pc.policy = pol;
                    pc.demote = dem;
                    pc.admit = adm;
                    pc.evict = (int) (rng() % 2);
                    pc.ram_evict = (int) (rng() % 2);
                    pc.ram_admit = (int) (rng() % 2);
                    pc.spare = 3;
                    pc.seed = rng();
                    pc.adapt_every = 64;
                    pc.halflife = 32;
                    const int S = 16;
                    const int64_t R = pol == kInclusive ? 24 : 20;
                    char what[96];
                    std::snprintf(what, sizeof what, "%s policy %d demote %d admit %d", backend, pol, dem, adm);
                    trace(ck, pc, S, R, false, calls, rng(), what);
                }
    }
    configure({ { "backend", "io_uring" } });
    {
        PolicyConfig pc;
        pc.spare = 4;
        pc.disk_off = true;
        trace(ck, pc, 20, ck.L * ck.E - 16, false, calls, 7, "disk off");
        pc = PolicyConfig();
        pc.spare = 0;
        pc.demote = kDemoteOff;
        trace(ck, pc, 12, 16, false, calls, 8, "spare 0");
        pc = PolicyConfig();
        pc.spare = 3;
        pc.demote = kDemoteAll;
        pc.policy = kExclusive;
        trace(ck, pc, 14, 20, true, calls, 9, "deferred returns");
        pc = PolicyConfig();
        pc.spare = 2;
        HostConfig hc;
        hc.deterministic = false;
        hc.slab_slots = 3;
        trace(ck, pc, 12, 20, false, calls, 10, "asynchronous refills", hc);
    }
}

void test_faults()
{
    Ckpt ck(2, 16, "faults", g_seed + 5);
    configure({ { "backend", "io_uring" }, { "fault_eio", "1" } });
    {
        PolicyConfig pc;
        pc.spare = 2;
        Run run = make(ck, pc, 8, 10, false);
        run.h->cold_fill(round_robin(ck.L, ck.E));
        std::vector<int32_t> ids = { 15, 14, 13, 12, 11, 10 };
        run.h->call(1, ids.data(), 6, kDecode);
        verify(*run.h, *run.cp, ck, "fault_eio");
        CHECK(run.h->stats.pread_fallbacks == run.h->stats.ssd_reads && run.h->stats.ssd_reads > 18,
              "fault_eio: %lld fallbacks for %lld reads", (long long) run.h->stats.pread_fallbacks,
              (long long) run.h->stats.ssd_reads);
    }
    configure({ { "backend", "io_uring" } });
    {
        PolicyConfig pc;
        pc.spare = 2;
        pc.policy = kExclusive;
        pc.demote = kDemoteOff;
        Run run = make(ck, pc, 6, 0, false);
        run.h->cold_fill(round_robin(ck.L, ck.E));
        const ExpertExtent& x = ck.ext[(size_t) (ck.L * ck.E - 1)];
        CHECK(::truncate(ck.files[(size_t) x.file[0]].c_str(), x.offset[0] + 10) == 0, "truncate");
        int32_t id = (int32_t) (ck.E - 1);
        std::string msg;
        try
        {
            run.h->call((int32_t) (ck.L - 1), &id, 1, kDecode);
        }
        catch (const TierError& e)
        {
            msg = e.what();
        }
        std::string want = "reading layer " + std::to_string(ck.L - 1) + " expert " + std::to_string(ck.E - 1) +
                           " (" + x.name + ") from " + ck.files[(size_t) x.file[0]];
        CHECK(msg.find(want) != std::string::npos && msg.find("errno") != std::string::npos, "error message: '%s'",
              msg.c_str());
        bool again = false;
        try
        {
            int32_t z = 0;
            run.h->call(0, &z, 1, kDecode);
        }
        catch (const TierError&)
        {
            again = true;
        }
        CHECK(again, "a failed tier stays failed");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) g_dir = argv[++i];
        else if (a == "--quick") g_quick = true;
        else if (a == "--seed" && i + 1 < argc) g_seed = std::strtoull(argv[++i], nullptr, 10);
        else
        {
            std::fprintf(stderr, "usage: %s --dir DIR [--quick] [--seed N]\n", argv[0]);
            return 2;
        }
    }
    if (g_dir.empty())
    {
        std::fprintf(stderr, "--dir is required (a scratch directory; O_DIRECT needs a real file system)\n");
        return 2;
    }
    ::mkdir(g_dir.c_str(), 0700);
    int f0 = g_fail.load();
    test_traces();
    std::printf("%s traces\n", g_fail.load() == f0 ? "PASS" : "FAIL");
    f0 = g_fail.load();
    test_faults();
    std::printf("%s faults\n", g_fail.load() == f0 ? "PASS" : "FAIL");
    exl3_disk::shutdown_default();
    if (g_fail.load())
    {
        std::printf("FAILED: %d of %d checks\n", g_fail.load(), g_checks);
        return 1;
    }
    std::printf("ALL PASSED: %d checks, 0 failures\n", g_checks);
    return 0;
}
