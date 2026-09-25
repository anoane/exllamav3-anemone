// Standalone test driver of the expert tier's native policy core (exllamav3_ext/tier/tier_policy.cpp):
// the micro-scenarios of doc/expert_tiers_impl_plan.md 3.7 with their exact records, counters and
// final states compiled in, then a soak of random calls (10^6 by default) with every invariant
// checked and a mirror directory applying the records. No torch, no CUDA: g++ and libc only.
// Built and run by tests/expert_tier/build.sh (release, ASan+UBSan, TSan).
//
//   tier_policy_test [--quick] [--calls N] [--seed N]

#include "tier/tier_policy.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace exl3_tier;

namespace
{

std::atomic<int> g_fail { 0 };
int g_checks = 0;

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

PolicyConfig base(int spare)
{
    PolicyConfig c;
    c.spare = spare;
    c.admit = kAdmitAlways;
    c.evict = kLru;
    c.policy = kExclusive;
    c.demote = kDemoteOff;
    c.halflife = 1u << 30;
    return c;
}

// "a0 a1v0 h2 t3": admit 0; admit 1 with victim 0; hit 2; transient 3
std::string shortrec(const Record& r)
{
    std::string s;
    for (const Entry& x : r.entries)
    {
        if (!s.empty()) s += " ";
        s += x.kind == kHit ? "h" : x.kind == kAdmitted ? "a" : "t";
        s += std::to_string(x.key);
        if (x.vkey >= 0) s += "v" + std::to_string(x.vkey);
    }
    return s;
}

std::vector<std::string> run(TierCore& c, const std::vector<std::vector<int32_t>>& calls, int mode = kDecode)
{
    std::vector<std::string> out;
    for (const auto& ids : calls)
    {
        Record r = c.lookup(0, ids.data(), (int64_t) ids.size(), mode);
        std::vector<Action> acts;
        c.host(r, acts);
        out.push_back(shortrec(r));
    }
    return out;
}

void expect_recs(const std::vector<std::string>& got, const std::vector<std::string>& want, const char* what)
{
    CHECK(got.size() == want.size(), "%s: %zu records, want %zu", what, got.size(), want.size());
    for (size_t i = 0; i < got.size() && i < want.size(); ++i)
        CHECK(got[i] == want[i], "%s: call %zu: '%s', want '%s'", what, i + 1, got[i].c_str(), want[i].c_str());
}

std::map<int32_t, int32_t> valid(const TierCore& c)
{
    std::map<int32_t, int32_t> m;
    for (int s = 0; s < c.vram.S; ++s)
        if (c.vram.state[(size_t) s] == kValid) m[s] = c.vram.key[(size_t) s];
    return m;
}

// key -> (dup, slot)
std::map<int32_t, std::pair<int, int32_t>> ramview(const TierCore& c)
{
    std::map<int32_t, std::pair<int, int32_t>> m;
    for (size_t k = 0; k < c.ram.of.size(); ++k)
    {
        int32_t s = c.ram.of[k];
        if (s >= 0) m[(int32_t) k] = { (int) c.ram.dup[(size_t) s], s };
    }
    return m;
}

void test_micro()
{
    {
        TierCore c(base(2), 1, 8, 5, 0, 0, false);
        expect_recs(run(c, { { 0, 1 }, { 2, 3 }, { 0, 2 }, { 1, 3 }, { 4, 5 } }),
                    { "a0 a1", "a2 a3v0", "a0v1 h2", "a1v0 h3", "a4v2 a5v1" }, "S1");
        CHECK((valid(c) == std::map<int32_t, int32_t> { { 0, 4 }, { 3, 3 }, { 4, 5 } }), "S1 final VRAM");
        CHECK(c.vram.stamp[0] == 5 && c.vram.stamp[3] == 4 && c.vram.stamp[4] == 5, "S1 stamps");
        CHECK(c.c[kHits] == 2 && c.c[kAdmits] == 8 && c.c[kRetires] == 5 && c.c[kDrops] == 5 && c.c[kSsd] == 8 &&
              c.c[kD2H] == 0, "S1 counters");
        c.check(true);
    }
    {
        PolicyConfig cfg = base(2);
        cfg.admit = kAdmitHeat;
        TierCore c(cfg, 1, 8, 4, 0, 0, false);
        expect_recs(run(c, { { 0, 1 }, { 0, 2 }, { 3, 0 }, { 2, 4 }, { 2, 5 }, { 2, 4 } }),
                    { "a0 a1", "h0 a2v1", "a3v2 h0", "t2 t4", "a2v0 a5v3", "h2 a4v5" }, "S2");
        CHECK(c.c[kHits] == 3 && c.c[kAdmits] == 7 && c.c[kTransients] == 2 && c.c[kRetires] == 5 && c.c[kSsd] == 9,
              "S2 counters");
        CHECK((valid(c) == std::map<int32_t, int32_t> { { 0, 4 }, { 2, 2 } }), "S2 final VRAM");
    }
    // S3: (policy, demote, victim heat) -> RAM, D2H, duplicates reclaimed
    struct S3 { int pol, dem, h; std::map<int32_t, std::pair<int, int32_t>> ram; int d2h, reclaim; };
    const std::map<int32_t, std::pair<int, int32_t>> r3 = { { 3, { 0, 1 } } }, r03 = { { 0, { 0, 0 } }, { 3, { 0, 1 } } },
        r23 = { { 2, { 1, 0 } }, { 3, { 0, 1 } } };
    const S3 s3[] = {
        { kExclusive, kDemoteOff, 5, r3, 0, 0 }, { kExclusive, kDemoteOff, 1, r3, 0, 0 },
        { kExclusive, kDemoteSwap, 5, r03, 1, 0 }, { kExclusive, kDemoteSwap, 1, r3, 0, 0 },
        { kExclusive, kDemoteHeat, 5, r03, 1, 0 }, { kExclusive, kDemoteHeat, 1, r03, 1, 0 },
        { kExclusive, kDemoteAll, 5, r03, 1, 0 }, { kExclusive, kDemoteAll, 1, r03, 1, 0 },
        { kLazyExclusive, kDemoteOff, 5, r23, 0, 0 }, { kLazyExclusive, kDemoteOff, 1, r23, 0, 0 },
        { kLazyExclusive, kDemoteSwap, 5, r03, 1, 0 }, { kLazyExclusive, kDemoteSwap, 1, r23, 0, 0 },
        { kLazyExclusive, kDemoteHeat, 5, r03, 1, 1 }, { kLazyExclusive, kDemoteHeat, 1, r03, 1, 1 },
        { kLazyExclusive, kDemoteAll, 5, r03, 1, 1 }, { kLazyExclusive, kDemoteAll, 1, r03, 1, 1 },
    };
    for (const S3& t : s3)
    {
        PolicyConfig cfg = base(1);
        cfg.policy = t.pol;
        cfg.demote = t.dem;
        TierCore c(cfg, 1, 8, 3, 2, 0, false);
        c.vram.place(0, 0, 1);
        c.vram.place(1, 1, 2);
        c.seq = 2;
        c.ram.add(2, false, 0, -1, false);
        c.ram.add(3, false, 0, -1, false);
        const int u[4] = { t.h, 1, 4, 2 };
        for (int k = 0; k < 4; ++k) c.heat.set(k, (uint64_t) u[k] * kQ16);
        auto got = run(c, { { 2 } });
        char what[64];
        std::snprintf(what, sizeof what, "S3 policy %d demote %d h %d", t.pol, t.dem, t.h);
        expect_recs(got, { "a2v0" }, what);
        CHECK(ramview(c) == t.ram, "%s: RAM", what);
        CHECK(c.c[kD2H] == t.d2h && c.c[kDrops] == 1 - t.d2h && c.c[kDupReclaim] == t.reclaim, "%s: counters", what);
        CHECK(c.c[kAdmits] == 1 && c.c[kRetires] == 1 && c.c[kH2DRam] == 1 && c.c[kSsd] == 0, "%s: counters", what);
        CHECK((bool) c.ram.compact[0] == (t.d2h == 1), "%s: compact layout", what);
        c.check(true);
    }
    {
        PolicyConfig cfg = base(2);
        cfg.policy = kLazyExclusive;
        cfg.demote = kDemoteSwap;
        cfg.disk_off = true;
        TierCore c(cfg, 1, 6, 4, 4, 0, false);
        c.vram.place(0, 0, 1);
        c.vram.place(1, 1, 1);
        c.seq = 1;
        for (int32_t e : { 2, 3, 4, 5 }) c.ram.add(e, false, 0, -1, false);
        auto got = run(c, { { 2, 3 }, { 4, 0 }, { 5, 1 }, { 2, 4 }, { 0, 1 }, { 3, 5 } });
        expect_recs(got, { "a2v0 a3v1", "a4v2 a0v3", "a5v4 a1v0", "a2v5 a4v1", "a0v2 a1v4", "a3v0 a5v1" }, "S4");
        CHECK(c.c[kAdmits] == 12 && c.c[kRetires] == 12 && c.c[kD2H] == 12 && c.c[kH2DRam] == 12 && c.c[kSsd] == 0,
              "S4 counters");
        CHECK((valid(c) == std::map<int32_t, int32_t> { { 0, 3 }, { 1, 5 } }), "S4 final VRAM");
        CHECK((ramview(c) == std::map<int32_t, std::pair<int, int32_t>> {
                  { 0, { 0, 0 } }, { 1, { 0, 2 } }, { 2, { 0, 1 } }, { 4, { 0, 3 } } }), "S4 final RAM");
        c.check(true);
        c.check_disk_off();
    }
    {
        PolicyConfig cfg = base(2);
        cfg.policy = kInclusive;
        TierCore c(cfg, 1, 8, 4, 5, 0, false);
        expect_recs(run(c, { { 0, 1 }, { 2, 0 }, { 3, 4 }, { 1, 2 } }), { "a0 a1", "a2v1 h0", "a3v0 a4v2", "a1v3 a2v4" }, "S5");
        CHECK(c.c[kHits] == 1 && c.c[kAdmits] == 7 && c.c[kRetires] == 5 && c.c[kDrops] == 5 && c.c[kH2DRam] == 2 &&
              c.c[kSsd] == 5, "S5 counters");
        CHECK((ramview(c) == std::map<int32_t, std::pair<int, int32_t>> {
                  { 0, { 0, 0 } }, { 1, { 1, 1 } }, { 2, { 1, 2 } }, { 3, { 0, 3 } }, { 4, { 0, 4 } } }), "S5 final RAM");
        c.check(true);
    }
    for (int p = 0; p <= 1; ++p)
    {
        PolicyConfig cfg = base(2);
        cfg.admit = kAdmitAdaptive;
        cfg.p_pinned = true;
        cfg.p = p;
        TierCore c(cfg, 1, 8, 4, 0, 0, false);
        auto got = run(c, { { 0, 1 }, { 2, 3 }, { 4, 0 } });
        if (p == 0)
        {
            expect_recs(got, { "a0 a1", "t2 t3", "t4 h0" }, "S6 p=0");
            CHECK(c.c[kAdmits] == 2 && c.c[kTransients] == 3 && c.c[kSsd] == 5, "S6 p=0 counters");
        }
        else
        {
            expect_recs(got, { "a0 a1", "a2v0 a3v1", "a4v2 a0v3" }, "S6 p=1");
            CHECK(c.c[kAdmits] == 6 && c.c[kRetires] == 4 && c.c[kSsd] == 6, "S6 p=1 counters");
        }
    }
    {
        PolicyConfig cfg = base(1);
        cfg.halflife = 2;
        TierCore c(cfg, 1, 8, 4, 0, 0, false);
        int32_t a[] = { 0, 1 }, b[] = { 0 }, d[] = { 1, 1, 2 };
        std::vector<Action> acts;
        c.host(c.lookup(0, a, 2, kDecode), acts);
        CHECK(c.heat.read(0) == 65536 && c.heat.read(1) == 65536, "S7 tok 0");
        c.tick(1);
        c.host(c.lookup(0, b, 1, kDecode), acts);
        CHECK(c.heat.read(0) == 131072 && c.heat.read(1) == 65536, "S7 tok 1");
        c.tick(1);
        CHECK(c.heat.read(0) == 65536 && c.heat.read(1) == 32768, "S7 tok 2");
        c.tick(3);
        c.host(c.lookup(0, d, 3, kRouted), acts);
        CHECK(c.heat.read(0) == 32768 && c.heat.read(1) == 24576 && c.heat.read(2) == 4096, "S7 tok 5");
        c.tick(35);
        CHECK(c.heat.read(0) == 0 && c.heat.read(1) == 0 && c.heat.read(2) == 0, "S7 tok 40");
        c.heat.set(5, 0xFFFFFFFFull - 10);
        c.heat.add(5, 100);
        CHECK(c.heat.read(5) == 0xFFFFFFFFu, "S7 saturation");
    }
    {
        PolicyConfig cfg = base(2);
        cfg.evict = kLfu;
        cfg.policy = kLazyExclusive;
        cfg.demote = kDemoteHeat;
        TierCore c(cfg, 1, 8, 4, 1, 0, false);
        expect_recs(run(c, { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 3, 4 } }), { "a0 a1", "h0 a2v1", "h0 a3v2", "h3 a4v0" }, "S8");
        CHECK(c.c[kHits] == 3 && c.c[kAdmits] == 5 && c.c[kRetires] == 3 && c.c[kDrops] == 1 && c.c[kD2H] == 2 &&
              c.c[kSsd] == 5 && c.c[kRamEvict] == 1 && c.c[kDupReclaim] == 3, "S8 counters");
        CHECK((ramview(c) == std::map<int32_t, std::pair<int, int32_t>> { { 0, { 0, 0 } } }), "S8 final RAM");
        c.check(true);
    }
    {
        TierCore c(base(1), 1, 8, 3, 0, 0, false);
        expect_recs(run(c, { { 0, 1 }, { 2, 3 } }), { "a0 a1", "a2v0 t3" }, "S9");
        CHECK(c.c[kAdmits] == 3 && c.c[kTransients] == 1 && c.c[kStarved] == 1, "S9 counters");
    }
    {
        AdaptiveP a(4, 12, 0.05, false, 0.0);
        const double want[] = { 0.1875, 0.1875, 0.234375, 0.25, 0.1275, 0.065025, 0.05 };
        const int64_t lives[][2] = { { 1, 3 }, { 1, 1 }, { 3, 1 }, { 9, 1 }, { 1, 99 }, { 1, 99 }, { 1, 99 } };
        CHECK(a.p == 0.25, "adaptive start");
        for (int i = 0; i < 7; ++i) CHECK(a.update(lives[i][0], lives[i][1]) == want[i], "adaptive step %d", i);
        CHECK(p_q32(0.0) == 0 && p_q32(1.0) == (1ull << 32), "p_q32");
    }
}

// Random calls on a set of configurations, every invariant checked, a mirror applying the records
void test_soak(int64_t calls, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    const int policies[] = { kLazyExclusive, kExclusive, kInclusive };
    int64_t done = 0;
    int round = 0;
    while (done < calls)
    {
        PolicyConfig cfg;
        cfg.policy = policies[rng() % 3];
        cfg.demote = (int) (rng() % 4);
        cfg.admit = (int) (rng() % 3);
        cfg.evict = (int) (rng() % 2);
        cfg.ram_evict = (int) (rng() % 2);
        cfg.ram_admit = (int) (rng() % 2);
        cfg.seed = rng();
        cfg.halflife = 1u + (uint32_t) (rng() % 256);
        cfg.adapt_every = 64;
        cfg.sample = 1 + (int) (rng() % 16);
        cfg.refill_inflight = 1 + (int) (rng() % 3);
        const int64_t L = 1 + (int64_t) (rng() % 4), E = 16 << (rng() % 3);
        const int S = 8 + (int) (rng() % 120);
        cfg.spare = (cfg.demote == kDemoteOff || cfg.policy == kInclusive) ? (int) (rng() % 9) : 1 + (int) (rng() % 8);
        cfg.spare = std::min(cfg.spare, S - 1);
        int64_t R = (int64_t) (rng() % 256);
        if (cfg.policy == kInclusive) R = std::max<int64_t>(R, S + cfg.spare + 1);
        else if (rng() % 4 == 0 && cfg.spare > 0 && R >= L * E - (S - cfg.spare)) cfg.disk_off = true;
        const bool defer = rng() % 4 == 0;
        TierCore c(cfg, L, E, S, R, 0, defer);
        VramDirectory mirror(L * E, S, cfg.spare, 0);
        std::vector<int32_t> order;
        for (int32_t e = 0; e < E; ++e)
            for (int32_t l = 0; l < L; ++l) order.push_back((int32_t) (l * E + e));
        std::vector<int32_t> vk;
        c.cold_fill(order, &vk, nullptr);
        for (size_t i = 0; i < vk.size(); ++i) mirror.place(vk[i], (int32_t) i, 0);
        std::vector<double> zipf((size_t) E);
        for (int64_t e = 0; e < E; ++e) zipf[(size_t) e] = 1.0 / std::pow((double) (e + 1), 0.8);
        std::discrete_distribution<int32_t> pick(zipf.begin(), zipf.end());
        const int64_t n = std::min<int64_t>(calls - done, 20000);
        std::vector<int32_t> ids;
        std::vector<Action> acts;
        try
        {
            for (int64_t i = 0; i < n; ++i)
            {
                const int32_t lc = (int32_t) (i % L);
                if (lc == 0) c.tick(1);
                ids.clear();
                const uint64_t r = rng() % 100;
                acts.clear();
                if (r < 92)
                {
                    const int rows = 1 << (rng() % 4);
                    for (int j = 0; j < rows * 6; ++j) ids.push_back((pick(rng) * 7 + lc) % (int32_t) E);
                    const int mode = r < 88 ? kDecode : kRouted;
                    Record rec = c.lookup(lc, ids.data(), (int64_t) ids.size(), mode);
                    c.host(rec, acts);
                    mirror.apply(rec);
                }
                else
                {
                    for (int j = 0; j < 64; ++j) ids.push_back((int32_t) (rng() % (uint64_t) E));
                    c.layer_call(lc, ids.data(), (int64_t) ids.size(), acts);
                }
                if (defer && rng() % 3 == 0) c.complete(acts);
                for (const Action& a : acts)
                    if (a.op == kActReturn) mirror.return_slot(a.a);
                if (i % 97 == 0)
                {
                    c.check(!defer);
                    if (cfg.disk_off) c.check_disk_off();
                    CHECK(mirror.state == c.vram.state && mirror.key == c.vram.key && mirror.stamp == c.vram.stamp &&
                          mirror.slot_of == c.vram.slot_of && mirror.free == c.vram.free, "soak round %d: mirror differs", round);
                }
            }
            acts.clear();
            c.complete(acts);
            c.check(true);
            CHECK(c.c[kD2H] + c.c[kDrops] == c.c[kRetires], "soak round %d: victims", round);
            CHECK(c.c[kHits] + c.c[kAdmits] + c.c[kTransients] == (int64_t) c.vram.access, "soak round %d: accesses", round);
            if (cfg.disk_off) CHECK(c.c[kSsd] + c.c[kPrefillSsd] == 0, "soak round %d: SSD reads with disk off", round);
        }
        catch (const TierError& e)
        {
            CHECK(false, "soak round %d: %s", round, e.what());
        }
        done += n;
        ++round;
    }
    std::printf("   soak: %lld calls over %d configurations\n", (long long) done, round);
}

}  // namespace

int main(int argc, char** argv)
{
    int64_t calls = 1000000;
    uint64_t seed = 12345;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--quick") calls = 100000;
        else if (a == "--calls" && i + 1 < argc) calls = std::atoll(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else
        {
            std::fprintf(stderr, "usage: %s [--quick] [--calls N] [--seed N]\n", argv[0]);
            return 2;
        }
    }
    test_micro();
    std::printf("%s micro-scenarios\n", g_fail.load() ? "FAIL" : "PASS");
    int f0 = g_fail.load();
    test_soak(calls, seed);
    std::printf("%s soak\n", g_fail.load() == f0 ? "PASS" : "FAIL");
    if (g_fail.load())
    {
        std::printf("FAILED: %d of %d checks\n", g_fail.load(), g_checks);
        return 1;
    }
    std::printf("ALL PASSED: %d checks, 0 failures\n", g_checks);
    return 0;
}
