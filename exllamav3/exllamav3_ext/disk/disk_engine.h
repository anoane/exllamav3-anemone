#pragma once

// Disk I/O engine: positioned reads of rows (n-gram / engram tables) and extents (experts kept
// on disk) behind one scheduler with request classes, byte windows and a hold rule, over three
// backends (thread pool buffered, thread pool O_DIRECT, io_uring). Design: doc/disk_engine.md.
// Knobs: doc/env_vars.md, "Disk I/O engine".
//
// This header is torch-free and portable. The engine itself is Linux-only; elsewhere every
// Engine constructor throws (Windows keeps ngram_gather_win.cpp).

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace exl3_disk
{

constexpr int kClasses = 4;
enum Class : int
{
    kEngram = 0,      // rows the forward needs now
    kExpert = 1,      // a missed expert needed at its layer
    kPrefetch = 2,    // look-ahead rows, expert prefetch (earliest deadline first)
    kRefill = 3,      // background fills; ages into kPrefetch
};

enum class Backend : int { Pread = 0, ODirect = 1, IoUring = 2 };
const char* backend_name(Backend b);

// Errors carry an errno-style code (positive) and a message naming what was wrong.
class Error : public std::runtime_error
{
public:
    Error(int code, const std::string& msg) : std::runtime_error(msg), code_(code) {}
    int code() const { return code_; }
private:
    int code_;
};

// Knob overrides: name without the EXL3_DISK_ prefix, lower case (e.g. "backend", "qd"), mapped
// to the value string. Looked up before the environment.
using Overrides = std::map<std::string, std::string>;

enum class Fadvise : int { Auto = 0, None = 1, Random = 2, Normal = 3, Sequential = 4 };

struct Config
{
    Backend backend = Backend::Pread;
    bool backend_named = false;         // EXL3_DISK_BACKEND named a backend (not unset / auto)
    bool direct_rows = false;           // rows bypass the page cache
    bool direct_extents = false;        // extents bypass the page cache
    int threads = 0;                    // pool workers
    int qd = 0;                         // requests in flight, all classes
    int reserve0 = 0;                   // slots only class 0 may use
    int engram_qd = 0;                  // class-0 requests in flight
    int64_t window_hi[kClasses] = {};   // bytes in flight per class, -1 = unlimited
    int64_t window_lo[kClasses] = {};   // under the hold rule (classes 2, 3)
    int64_t chunk_bytes = 0;            // bulk chunk, 0 = per device (max_sectors_kb)
    uint32_t align = 0;                 // O_DIRECT offset alignment override, 0 = statx
    int spin_us = 0;                    // waiter spin before sleeping
    int hold_arm_ms = 0;                // expiry of arm_hold() without a hold ticket
    int refill_age_ms = 0;              // class 3 -> 2 after this long queued, 0 = never
    bool sqpoll = false;
    int sqpoll_idle_ms = 0;
    int sqpoll_cpu = -1;
    bool iopoll = false;
    bool reg_files = false;
    bool reg_buffers = false;
    int uring_async = 0;                // tickets with at least this many buffered ops use io-wq
    int iowq_workers = 0;               // bound on io-wq workers, 0 = kernel default
    std::vector<int> affinity;          // CPUs for engine threads, empty = inherit
    Fadvise fadvise = Fadvise::Auto;
    bool extent_dontneed = true;        // drop buffered extent pages after the read
    bool verbose = false;
    int trace = 0;                      // trace ring entries, 0 = off
    // Fault injection (tests): one read attempt in N (a reproducible pseudo-random draw) fails
    // with EIO / returns short / fails with EINTR; every read sleeps delay_us first (pools)
    int fault_eio = 0;
    int fault_short = 0;
    int fault_eintr = 0;
    int fault_delay_us = 0;
    bool fault_no_uring = false;        // io_uring_setup "fails": the thread-pool fallback runs
    int fault_enter_fatal = 0;          // the Nth submitting io_uring_enter fails with EBADFD
    bool fault_force_iopoll = false;    // every device counts as able to poll (IOPOLL ring runs)
    std::vector<std::string> notes;     // knobs that do not apply to the chosen backend

    std::string describe() const;
};

// Parse and validate every EXL3_DISK_* knob (overrides first). Throws Error(EINVAL) naming the
// knob and the accepted values.
Config config_from_env(const Overrides& ov = Overrides());

// EXL3_DISK_BACKEND names a backend (not unset / auto); throws Error on an invalid value
bool backend_named_in_env();

// Windows: message for an EXL3_DISK_BACKEND the overlapped ReadFile gather cannot honour, or ""
std::string windows_backend_refusal();

struct RowTable
{
    int fd;                             // caller's descriptor (the engine reopens the file)
    int64_t base_offset;                // file offset of row 0
    int64_t row_bytes;
    uint8_t* out;                       // row i lands at out + i * row_bytes
    int64_t out_bytes;                  // capacity of out
};

struct ExtentReq
{
    int fd;
    int64_t offset;                     // payload file offset
    int64_t length;                     // payload bytes
    uint8_t* slot;                      // 4 KiB aligned; payload lands at slot + payload_offset
    int64_t slot_bytes;                 // capacity, at least ExtentGeom::span_bytes
};

struct ExtentGeom
{
    int64_t payload_offset;             // offset mod G, G = O_DIRECT alignment rounded up to 4 KiB
    int64_t span_bytes;                 // roundup(payload_offset + length, G)
};

struct Options
{
    int cls = kEngram;
    bool hold = false;                  // classes 2-3 use their lo window until this completes;
                                        // only for classes 0 and 1 (EINVAL otherwise)
    int64_t deadline_ns = 0;            // class 2 order (CLOCK_MONOTONIC), 0 = after deadlines
    uint32_t* flag = nullptr;           // set to flag_value (release) when the ticket completes
    uint32_t flag_value = 0;
};

constexpr int kHistBuckets = 8 + 60 * 8;     // log-linear, 8 per octave: <= 12.5 % wide
int64_t hist_bucket_lower(int i);            // lower bound (ns) of bucket i
int hist_bucket_of(int64_t ns);

struct ClassStats
{
    uint64_t tickets = 0;               // tickets completed
    uint64_t ops = 0;                   // ops finished (read or failed)
    uint64_t op_errors = 0;
    uint64_t ops_cancelled = 0;
    uint64_t bytes = 0;                 // payload bytes delivered
    uint64_t held = 0;                  // admissions refused by a cap, window or the hold
    uint64_t aged = 0;                  // class 3 tickets aged into class 2
    int64_t max_inflight = 0;           // requests
    int64_t max_inflight_bytes = 0;
    std::vector<uint64_t> hist_queue;   // submit -> issue, per op (ns)
    std::vector<uint64_t> hist_service; // issue -> finish, per op (ns)
    std::vector<uint64_t> hist_ticket;  // submit -> complete, per ticket (ns)
};

struct Stats
{
    ClassStats cls[kClasses];
    uint64_t inline_reaped = 0;         // io_uring completions drained by a waiting caller
    uint64_t reaper_reaped = 0;         // ... by the reaper / poller threads
    uint64_t enters = 0;                // io_uring_enter calls that submitted
    uint64_t resubmits = 0;             // short reads, EAGAIN, EINTR resumed
    uint64_t stray_cqes = 0;            // completions with an unknown tag (must stay 0)
    int64_t inflight_now = 0;
    int64_t queued_ops_now = 0;
    int64_t tickets_live = 0;
    int64_t files_open = 0;
};

struct TraceRec
{
    uint64_t ticket;
    int32_t cls;                        // class the op was issued in
    int32_t res;                        // payload bytes or -errno
    int64_t dev_off;
    int64_t dev_len;
    int64_t t_submit, t_issue, t_done;
};

// CLOCK_MONOTONIC ns. t_submit: the ticket entered the scheduler; t_first_issue: its first read
// started (before t_submit for reads a synchronous caller completed inline); t_done: 0 until done
struct TicketTimes
{
    int64_t t_submit = 0, t_first_issue = 0, t_done = 0;
    int64_t ops = 0;
};

using TicketId = uint64_t;

namespace detail { struct Core; }

class Engine
{
public:
    explicit Engine(const Config& cfg);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Asynchronous rows: for every table, out row i = file bytes [base + (uids[i] - uid_base) *
    // row_bytes, + row_bytes). uids need not be sorted or unique (consecutive ids coalesce).
    // Validates everything before queueing anything; throws Error on bad arguments. The engine's
    // threads issue the reads (io_uring: the reaper), so the caller may go on or even exit;
    // its buffers must stay valid until the ticket is released.
    TicketId submit_rows(const int64_t* uids, int64_t n, int64_t uid_base,
                         const RowTable* tables, int ntables, const Options& o);

    // Extents; payload_offsets (optional, n entries) receives each payload's offset in its slot
    TicketId submit_extents(const ExtentReq* ex, int64_t n, const Options& o,
                            int64_t* payload_offsets);

    ExtentGeom extent_geometry(int fd, int64_t offset, int64_t length);

    // Wait until the ticket completed: 0, the first error (-errno, -ECANCELED when cancelled),
    // or -ETIMEDOUT when timeout_ns (>= 0) passed first. The ticket stays valid either way.
    // Any number of threads may wait for one ticket, also while another releases it.
    int wait(TicketId id, int64_t timeout_ns = -1);
    bool done(TicketId id);
    TicketTimes times(TicketId id);
    // Cancel queued ops, wait for ops in flight and for every thread still inside wait() on
    // this ticket, forget the ticket. No read lands after this. Concurrent releases of one
    // ticket all return after it completed; releasing an unknown ticket does nothing.
    void release(TicketId id);
    void cancel(TicketId id);
    void promote(TicketId id, int cls);
    void arm_hold();

    // Synchronous: submit, wait, release; 0 or -errno (throws on bad arguments). The calling
    // thread submits its own reads (io_uring: no hand-off to the reaper). t_submitted, when
    // given, receives the time the submission returned (CLOCK_MONOTONIC ns)
    int gather_rows(const int64_t* uids, int64_t n, int64_t uid_base,
                    const RowTable* tables, int ntables, const Options& o,
                    int64_t* t_submitted = nullptr);
    int read_extents(const ExtentReq* ex, int64_t n, const Options& o, int64_t* payload_offsets,
                     int64_t* t_submitted = nullptr);

    // io_uring registered buffers for extent slots (READ_FIXED). Returns an index, or -1 when
    // the backend has no registration (not an error)
    int register_buffer(void* p, size_t n);
    void unregister_buffer(int idx);

    // Close the engine's own descriptors of a file (the caller's descriptor, or a path), so an
    // unloaded or deleted model file no longer stays open. Returns 1 when they were closed, 0
    // when the engine had no such file open, -1 when tickets still read it: it is then closed
    // as soon as it is idle. A later read of the file simply reopens it. Files whose last name
    // was removed are also closed on their own once idle (checked every few submissions).
    int forget(int fd);
    int forget_path(const std::string& path);

    Stats stats(bool reset);
    std::vector<TraceRec> trace(bool clear);
    const Config& config() const;
    std::string describe() const;       // backend, fallbacks, registration state
    std::vector<std::string> fallbacks() const;

private:
    std::unique_ptr<detail::Core> core_;
};

// Process-wide engine used by the extension. Created on first use from the environment; never
// destroyed at exit (like the original gather pool). configure_default replaces it (tests,
// benchmarks); a replaced engine lives on until its last ticket is released.
std::shared_ptr<Engine> default_engine();
std::shared_ptr<Engine> default_engine_if_created();   // nullptr before first use; never creates
std::shared_ptr<Engine> configure_default(const Overrides& ov);
void shutdown_default();

// ngram_gather_cpu runs on the engine only when EXL3_DISK_BACKEND names a backend (or
// configure_default did); otherwise its original pool runs verbatim
bool ngram_route_engine();

// Class used by ngram_gather_cpu calls from this thread (default kEngram)
int thread_class();
int set_thread_class(int cls);

// ngram_gather_cpu on the default engine: 0, or -errno when a read failed. Throws Error for bad
// knobs or arguments (rows past the end of the file included)
int ngram_gather(int fd, int64_t base_offset, int64_t row_bytes, const int64_t* uids, int64_t n,
                 int64_t uid_base, uint8_t* out, int64_t out_bytes);

}  // namespace exl3_disk
