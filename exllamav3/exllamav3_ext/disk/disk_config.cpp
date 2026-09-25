// EXL3_DISK_* knobs: parsing and validation. Portable (also compiled on Windows, where only
// windows_backend_refusal() is used).

#include "disk_engine.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <thread>

namespace exl3_disk
{

// The backend `auto` resolves to. It must name the fastest backend measured on the test host
// (doc/env_vars.md, "Disk I/O engine"); until that benchmark has run it stays pread, which is
// the original gather's technique, and ngram_gather_cpu keeps its original pool unless a
// backend is named explicitly.
static constexpr Backend kAutoBackend = Backend::Pread;

const char* backend_name(Backend b)
{
    switch (b)
    {
        case Backend::Pread: return "pread";
        case Backend::ODirect: return "odirect";
        case Backend::IoUring: return "io_uring";
    }
    return "?";
}

namespace
{

// Every knob, as its override key; the environment variable is EXL3_DISK_ + upper case
const char* const kKnobs[] =
{
    "backend", "direct", "threads", "qd", "reserve0", "engram_qd", "window_expert",
    "window_prefetch", "window_refill", "chunk", "align", "spin_us", "hold_arm_ms",
    "refill_age_ms", "sqpoll", "sqpoll_idle_ms", "sqpoll_cpu", "iopoll", "register",
    "uring_async", "iowq_workers", "affinity", "fadvise", "extent_dontneed", "verbose", "trace",
    "fault_eio", "fault_short", "fault_eintr", "fault_delay_us", "fault_no_uring",
    "fault_enter_fatal", "fault_force_iopoll",
};

std::string env_name(const std::string& key)
{
    std::string s = "EXL3_DISK_";
    for (char ch : key) s += (char) ((ch >= 'a' && ch <= 'z') ? ch - 'a' + 'A' : ch);
    return s;
}

std::string lower(const std::string& s)
{
    std::string r = s;
    for (char& ch : r) if (ch >= 'A' && ch <= 'Z') ch = (char) (ch - 'A' + 'a');
    return r;
}

std::string trim(const std::string& s)
{
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

class Knobs
{
public:
    explicit Knobs(const Overrides& ov) : ov_(ov)
    {
        std::set<std::string> known(std::begin(kKnobs), std::end(kKnobs));
        for (const auto& kv : ov_)
            if (!known.count(kv.first))
                throw Error(EINVAL, "disk engine: unknown knob override '" + kv.first + "'");
    }

    // Value of the knob, or "" when unset (an empty variable counts as unset)
    std::string get(const std::string& key) const
    {
        auto it = ov_.find(key);
        if (it != ov_.end()) return trim(it->second);
        const char* v = std::getenv(env_name(key).c_str());
        return v ? trim(std::string(v)) : std::string();
    }

    [[noreturn]] void bad(const std::string& key, const std::string& val,
                          const std::string& expect) const
    {
        throw Error(EINVAL, env_name(key) + "=" + val + ": expected " + expect);
    }

    bool is_auto(const std::string& key) const
    {
        std::string v = lower(get(key));
        return v.empty() || v == "auto";
    }

    int64_t integer(const std::string& key, int64_t lo, int64_t hi, int64_t def) const
    {
        std::string v = get(key);
        if (v.empty() || lower(v) == "auto") return def;
        int64_t r = 0;
        if (!parse_int(v, &r) || r < lo || r > hi)
            bad(key, v, "an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
        return r;
    }

    bool boolean(const std::string& key, bool def) const
    {
        std::string v = lower(get(key));
        if (v.empty() || v == "auto") return def;
        if (v == "1" || v == "on" || v == "true" || v == "yes") return true;
        if (v == "0" || v == "off" || v == "false" || v == "no") return false;
        bad(key, get(key), "0 or 1");
    }

    // Byte size with an optional binary suffix (K, M, G, KiB, ...); "inf", "unlimited" and -1
    // mean unlimited (-1)
    int64_t size(const std::string& key, const std::string& text, bool allow_inf) const
    {
        std::string v = lower(trim(text));
        if (allow_inf && (v == "inf" || v == "unlimited" || v == "-1")) return -1;
        size_t i = 0;
        while (i < v.size() && v[i] >= '0' && v[i] <= '9') ++i;
        if (i == 0 || i > 18) bad(key, get(key), size_expect(allow_inf));
        int64_t n = 0;
        if (!parse_int(v.substr(0, i), &n)) bad(key, get(key), size_expect(allow_inf));
        std::string suf = trim(v.substr(i));
        int64_t mul = 1;
        if (suf.empty() || suf == "b") mul = 1;
        else if (suf == "k" || suf == "kb" || suf == "kib") mul = 1ll << 10;
        else if (suf == "m" || suf == "mb" || suf == "mib") mul = 1ll << 20;
        else if (suf == "g" || suf == "gb" || suf == "gib") mul = 1ll << 30;
        else bad(key, get(key), size_expect(allow_inf));
        if (n > std::numeric_limits<int64_t>::max() / mul) bad(key, get(key), "a smaller size");
        return n * mul;
    }

    static std::string size_expect(bool allow_inf)
    {
        return std::string("a byte size such as 1280K, 8M or 0") + (allow_inf ? ", or inf" : "");
    }

    static bool parse_int(const std::string& s, int64_t* out)
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

private:
    const Overrides& ov_;
};

// "0-3,8,10-11" -> sorted unique CPU ids
std::vector<int> parse_cpu_list(const Knobs& k, const std::string& key)
{
    std::string v = k.get(key);
    std::vector<int> cpus;
    if (v.empty()) return cpus;
    const char* expect = "a CPU list such as 0-3,8 (ids below 1024)";
    std::stringstream ss(v);
    std::string item;
    std::set<int> seen;
    while (std::getline(ss, item, ','))
    {
        item = trim(item);
        if (item.empty()) k.bad(key, v, expect);
        size_t dash = item.find('-');
        int64_t a = 0, b = 0;
        if (dash == std::string::npos)
        {
            if (!Knobs::parse_int(item, &a)) k.bad(key, v, expect);
            b = a;
        }
        else
        {
            if (!Knobs::parse_int(trim(item.substr(0, dash)), &a) ||
                !Knobs::parse_int(trim(item.substr(dash + 1)), &b)) k.bad(key, v, expect);
        }
        if (a < 0 || b < a || b >= 1024) k.bad(key, v, expect);
        for (int64_t c = a; c <= b; ++c) seen.insert((int) c);
    }
    cpus.assign(seen.begin(), seen.end());
    return cpus;
}

// hi must be positive or unlimited: a class whose window is 0 outside the hold rule would
// never be admitted. lo may be 0 (nothing new while the hold rule is active, which ends)
void parse_window(const Knobs& k, const std::string& key, bool allow_lo, int64_t def_hi,
                  int64_t def_lo, int64_t* hi, int64_t* lo)
{
    std::string v = k.get(key);
    *hi = def_hi;
    *lo = def_lo;
    if (v.empty() || lower(v) == "auto") return;
    size_t slash = v.find('/');
    const char* positive = allow_lo
        ? "a positive byte size or inf, optionally /lo (hi 0 would never admit the class)"
        : "a positive byte size or inf (0 would never admit the class)";
    if (slash == std::string::npos)
    {
        *hi = k.size(key, v, true);
        if (*hi == 0) k.bad(key, v, positive);
        *lo = allow_lo ? std::min<int64_t>(def_lo, *hi < 0 ? def_lo : *hi) : *hi;
        return;
    }
    if (!allow_lo)
        k.bad(key, v, "one byte size (expert demand reads are never held, so there is no lo "
                      "window)");
    *hi = k.size(key, v.substr(0, slash), true);
    *lo = k.size(key, v.substr(slash + 1), true);
    if (*hi == 0) k.bad(key, v, positive);
    if (*hi >= 0 && (*lo < 0 || *lo > *hi))
        k.bad(key, v, "hi/lo with lo <= hi (lo applies while a decode row batch is held)");
}

}  // namespace

Config config_from_env(const Overrides& ov)
{
    Knobs k(ov);
    Config c;

    // Backend
    std::string b = lower(k.get("backend"));
    if (b.empty() || b == "auto")
    {
        c.backend = kAutoBackend;
        c.backend_named = false;
    }
    else if (b == "pread") { c.backend = Backend::Pread; c.backend_named = true; }
    else if (b == "odirect") { c.backend = Backend::ODirect; c.backend_named = true; }
    else if (b == "io_uring") { c.backend = Backend::IoUring; c.backend_named = true; }
    else k.bad("backend", k.get("backend"), "auto, pread, odirect or io_uring");
    bool uring = c.backend == Backend::IoUring;

    // Page cache per traffic kind
    std::string d = lower(k.get("direct"));
    if (d.empty() || d == "auto")
    {
        c.direct_rows = c.backend == Backend::ODirect;
        c.direct_extents = c.backend != Backend::Pread;
    }
    else if (d == "none" || d == "0") { c.direct_rows = false; c.direct_extents = false; }
    else if (d == "rows") { c.direct_rows = true; c.direct_extents = false; }
    else if (d == "extents") { c.direct_rows = false; c.direct_extents = true; }
    else if (d == "all" || d == "1") { c.direct_rows = true; c.direct_extents = true; }
    else k.bad("direct", k.get("direct"), "auto, none, rows, extents or all");

    // Depths
    unsigned hw = std::thread::hardware_concurrency();
    int def_threads = (int) std::min(32u, std::max(4u, hw));
    c.threads = (int) k.integer("threads", 1, 256, def_threads);
    c.qd = (int) k.integer("qd", 1, 4096, uring ? 128 : c.threads + 8);
    c.reserve0 = (int) k.integer("reserve0", 0, c.qd - 1, (int64_t) c.qd * 3 / 4);
    c.engram_qd = (int) k.integer("engram_qd", 1, c.qd, c.qd);

    // Windows (bytes in flight per class); class 0 has none
    c.window_hi[kEngram] = c.window_lo[kEngram] = -1;
    parse_window(k, "window_expert", false, 32ll << 20, 32ll << 20,
                 &c.window_hi[kExpert], &c.window_lo[kExpert]);
    parse_window(k, "window_prefetch", true, 8ll << 20, 0,
                 &c.window_hi[kPrefetch], &c.window_lo[kPrefetch]);
    parse_window(k, "window_refill", true, 8ll << 20, 0,
                 &c.window_hi[kRefill], &c.window_lo[kRefill]);

    // Geometry
    if (k.is_auto("chunk")) c.chunk_bytes = 0;
    else
    {
        c.chunk_bytes = k.size("chunk", k.get("chunk"), false);
        if (c.chunk_bytes < (64 << 10) || c.chunk_bytes > (64 << 20) || c.chunk_bytes % 65536)
            k.bad("chunk", k.get("chunk"), "auto, or a multiple of 64K in [64K, 64M]");
    }
    if (k.is_auto("align")) c.align = 0;
    else
    {
        int64_t a = k.size("align", k.get("align"), false);
        if (a < 512 || a > 65536 || (a & (a - 1)))
            k.bad("align", k.get("align"), "auto, or a power of two in [512, 65536]");
        c.align = (uint32_t) a;
    }

    // Timing
    c.spin_us = (int) k.integer("spin_us", 0, 100000, 50);
    c.hold_arm_ms = (int) k.integer("hold_arm_ms", 0, 10000, 50);
    c.refill_age_ms = (int) k.integer("refill_age_ms", 0, 3600000, 250);

    // io_uring
    c.sqpoll = k.boolean("sqpoll", false);
    c.sqpoll_idle_ms = (int) k.integer("sqpoll_idle_ms", 1, 60000, 10);
    c.sqpoll_cpu = (int) k.integer("sqpoll_cpu", -1, 1023, -1);
    c.iopoll = k.boolean("iopoll", false);
    std::string r = lower(k.get("register"));
    if (r.empty() || r == "auto" || r == "all") { c.reg_files = uring; c.reg_buffers = uring; }
    else if (r == "none" || r == "0") { c.reg_files = false; c.reg_buffers = false; }
    else if (r == "files") { c.reg_files = uring; c.reg_buffers = false; }
    else if (r == "buffers") { c.reg_files = false; c.reg_buffers = uring; }
    else k.bad("register", k.get("register"), "auto, none, files, buffers or all");
    c.uring_async = (int) k.integer("uring_async", 0, 1 << 30, 0);
    c.iowq_workers = (int) k.integer("iowq_workers", 0, 4096, 0);

    // Threads and page cache
    c.affinity = parse_cpu_list(k, "affinity");
    std::string f = lower(k.get("fadvise"));
    if (f.empty() || f == "auto") c.fadvise = Fadvise::Auto;
    else if (f == "none") c.fadvise = Fadvise::None;
    else if (f == "random") c.fadvise = Fadvise::Random;
    else if (f == "normal") c.fadvise = Fadvise::Normal;
    else if (f == "sequential") c.fadvise = Fadvise::Sequential;
    else k.bad("fadvise", k.get("fadvise"), "auto, none, random, normal or sequential");
    c.extent_dontneed = k.boolean("extent_dontneed", true);

    c.verbose = k.boolean("verbose", false);
    c.trace = (int) k.integer("trace", 0, 1 << 24, 0);
    c.fault_eio = (int) k.integer("fault_eio", 0, 1 << 30, 0);
    c.fault_short = (int) k.integer("fault_short", 0, 1 << 30, 0);
    c.fault_eintr = (int) k.integer("fault_eintr", 0, 1 << 30, 0);
    c.fault_delay_us = (int) k.integer("fault_delay_us", 0, 1000000, 0);
    c.fault_no_uring = k.boolean("fault_no_uring", false);
    c.fault_enter_fatal = (int) k.integer("fault_enter_fatal", 0, 1 << 30, 0);
    c.fault_force_iopoll = k.boolean("fault_force_iopoll", false);

    // Knobs that do not apply to the chosen backend: noted, not fatal
    auto note_if = [&](bool set, const std::string& key, const std::string& why)
    {
        if (set) c.notes.push_back(env_name(key) + " ignored: " + why);
    };
    std::string only_uring = std::string("applies to the io_uring backend (backend is ") +
                             backend_name(c.backend) + ")";
    note_if(!uring && c.sqpoll, "sqpoll", only_uring);
    note_if(!uring && !k.get("register").empty() && !k.is_auto("register"), "register", only_uring);
    note_if(!uring && c.uring_async, "uring_async", only_uring);
    note_if(!uring && c.iowq_workers, "iowq_workers", only_uring);
    note_if(uring && !k.get("threads").empty(), "threads",
            "applies to the pread and odirect pools (io_uring: EXL3_DISK_IOWQ_WORKERS)");
    note_if(!c.sqpoll && (!k.get("sqpoll_cpu").empty() || !k.get("sqpoll_idle_ms").empty()),
            "sqpoll_cpu / sqpoll_idle_ms", "EXL3_DISK_SQPOLL is off");
    note_if(c.iopoll && !c.direct_rows && !c.direct_extents, "iopoll",
            "no reads bypass the page cache (EXL3_DISK_DIRECT), and only O_DIRECT reads are polled");
    return c;
}

bool backend_named_in_env()
{
    Overrides none;
    Knobs k(none);
    std::string b = lower(k.get("backend"));
    if (b.empty() || b == "auto") return false;
    if (b == "pread" || b == "odirect" || b == "io_uring") return true;
    k.bad("backend", k.get("backend"), "auto, pread, odirect or io_uring");
}

std::string Config::describe() const
{
    auto sz = [](int64_t v) -> std::string
    {
        if (v < 0) return "inf";
        if (v == 0) return "0";
        if (v % (1 << 20) == 0) return std::to_string(v >> 20) + "M";
        if (v % (1 << 10) == 0) return std::to_string(v >> 10) + "K";
        return std::to_string(v);
    };
    std::ostringstream s;
    s << "backend " << backend_name(backend) << (backend_named ? "" : " (auto)")
      << ", direct " << (direct_rows ? (direct_extents ? "all" : "rows")
                                     : (direct_extents ? "extents" : "none"))
      << ", qd " << qd << ", reserve0 " << reserve0 << ", engram_qd " << engram_qd;
    if (backend != Backend::IoUring) s << ", threads " << threads;
    s << ", windows expert " << sz(window_hi[kExpert])
      << " prefetch " << sz(window_hi[kPrefetch]) << "/" << sz(window_lo[kPrefetch])
      << " refill " << sz(window_hi[kRefill]) << "/" << sz(window_lo[kRefill])
      << ", chunk " << (chunk_bytes ? sz(chunk_bytes) : std::string("auto"))
      << ", align " << (align ? std::to_string(align) : std::string("auto"));
    if (backend == Backend::IoUring)
    {
        s << ", register " << (reg_files ? (reg_buffers ? "files+buffers" : "files")
                                         : (reg_buffers ? "buffers" : "none"));
        s << ", sqpoll " << (sqpoll ? "on" : "off");
        if (sqpoll) s << " (idle " << sqpoll_idle_ms << " ms, cpu " << sqpoll_cpu << ")";
        if (uring_async) s << ", uring_async " << uring_async;
        if (iowq_workers) s << ", iowq_workers " << iowq_workers;
    }
    s << ", iopoll " << (iopoll ? "on" : "off");
    if (!affinity.empty()) s << ", affinity " << affinity.size() << " cpus";
    return s.str();
}

std::string windows_backend_refusal()
{
    std::string v;
    const char* e = std::getenv("EXL3_DISK_BACKEND");
    if (e) v = lower(trim(std::string(e)));
    if (v.empty() || v == "auto" || v == "odirect") return std::string();
    if (v == "pread")
        return "EXL3_DISK_BACKEND=pread is not available on Windows: the n-gram gather reads "
               "through an unbuffered overlapped handle, which is what odirect means there. "
               "Unset EXL3_DISK_BACKEND or set it to odirect.";
    if (v == "io_uring")
        return "EXL3_DISK_BACKEND=io_uring is Linux-only. On Windows unset EXL3_DISK_BACKEND "
               "(the overlapped unbuffered ReadFile gather runs) or set it to odirect.";
    return "EXL3_DISK_BACKEND=" + std::string(e) + ": expected auto, pread, odirect or io_uring";
}

int64_t hist_bucket_lower(int i)
{
    if (i < 8) return i < 0 ? 0 : i;
    int e = (i - 8) / 8 + 3;
    int m = (i - 8) % 8;
    return (int64_t) (8 + m) << (e - 3);
}

int hist_bucket_of(int64_t ns)
{
    if (ns < 8) return ns < 0 ? 0 : (int) ns;
#if defined(__GNUC__) || defined(__clang__)
    int e = 63 - __builtin_clzll((unsigned long long) ns);
#else
    int e = 0;
    for (uint64_t v = (uint64_t) ns; v >>= 1; ) ++e;
#endif
    int m = (int) ((ns >> (e - 3)) & 7);
    int i = 8 + (e - 3) * 8 + m;
    return i < kHistBuckets ? i : kHistBuckets - 1;
}

}  // namespace exl3_disk
