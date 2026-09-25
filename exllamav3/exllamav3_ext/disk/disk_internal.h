#pragma once

// Engine internals shared by the scheduler (disk_engine.cpp) and the io_uring executor
// (disk_uring.cpp). Linux only.
//
// Locking (see doc/disk_engine.md, "Threading"):
//   Core::mx        queues, tickets, in-flight accounting, stats, executor slot tables
//   FileTable::mx   the file table; never taken while Core::mx is held (and vice versa)
//   uring sq_mx     the submission queue, from preparing SQEs to io_uring_enter; taken before
//                   Core::mx (the only nesting), so a thread submits exactly what it prepared
//   uring cq_mx     draining the completion queue; never held with any other lock
//
// io_uring request ownership: a read is owned by the thread whose io_uring_enter submitted
// it, and a buffered read whose owner exited before it completed fails (EFAULT). So a caller
// thread only ever submits reads of the ticket it is waiting for (without a timeout); every
// other read is submitted by the reaper, which lives as long as the ring.

#if defined(__linux__)

#include "disk_engine.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/types.h>

namespace exl3_disk
{
namespace detail
{

int64_t now_ns();                       // CLOCK_MONOTONIC

constexpr int kMaxRetries = 32;         // EINTR / EAGAIN resubmissions per op
constexpr int kMaxShort = 64;           // short-read resumptions per op
constexpr uint32_t kSpanOpMax = 65536;  // pool: ops up to this size are taken in spans
constexpr int kSpanMax = 256;           // pool: ops per span
constexpr int64_t kMaxOpBytes = 1ll << 30;
constexpr int kMaxTables = 64;
constexpr int kMaxUserBuffers = 64;     // io_uring buffer table (index 0 = bounce arena)

// One file, as the engine holds it: its own descriptors, never the caller's
struct FileEnt
{
    dev_t dev = 0;
    ino_t ino = 0;
    std::string name;                   // for messages
    bool is_blk = false;
    int64_t blk_size = 0;               // block devices: size at open
    int fd_buf = -1;                    // buffered descriptor
    int fd_dir = -1;                    // O_DIRECT descriptor: -1 not opened yet, -2 refused
    int fixed_buf = -1;                 // io_uring fixed-file slots, -1 none
    int fixed_dir = -1;
    uint32_t mem_align = 4096;          // O_DIRECT buffer alignment
    uint32_t off_align = 4096;          // O_DIRECT offset / length alignment
    uint32_t geom_align = 4096;         // extent slot geometry: max(4096, off_align)
    int64_t chunk = 0;                  // bulk chunk for this file
    bool poll_ok = false;               // device polls (IOPOLL); poll_why otherwise
    std::string poll_why;
    std::atomic<bool> poll_warned { false };
    std::atomic<bool> direct_warned { false };
    std::atomic<bool> bounce_warned { false };
    int advice = -1;                    // posix_fadvise applied to fd_buf, -1 none yet
    std::atomic<int64_t> refs { 0 };    // unfinished tickets referencing this file
    uint64_t last_use = 0;
};

// What a submission needs from a file, copied under FileTable::mx
struct FileRef
{
    FileEnt* ent = nullptr;
    int64_t size = 0;                   // bytes readable (st_size, or the block device size)
    int fd_buf = -1, fd_dir = -1;       // fd_dir < 0: direct unavailable
    int fixed_buf = -1, fixed_dir = -1;
    uint32_t mem_align = 4096, off_align = 4096, geom_align = 4096;
    int64_t chunk = 0;
    bool poll_ok = false;
};

class Executor;

class FileTable
{
public:
    // Look up (or open) the file behind a caller's descriptor. want_direct opens the O_DIRECT
    // descriptor on first use (refused -> FileRef::fd_dir < 0 and one message per file).
    // advice: posix_fadvise to apply once to the buffered descriptor (-1 none). Increments
    // refs; the caller must drop it with unref(). Throws Error.
    FileRef acquire(int fd, bool want_direct, int advice, const Config& cfg, Executor* ex);
    static void unref(FileEnt* e) { e->refs.fetch_sub(1, std::memory_order_acq_rel); }
    void close_all(Executor* ex);
    int64_t size();

private:
    void evict_locked(Executor* ex);
    void close_ent(FileEnt* e, Executor* ex);

    struct Key
    {
        dev_t dev;
        ino_t ino;
        bool operator==(const Key& o) const { return dev == o.dev && ino == o.ino; }
    };
    struct KeyHash
    {
        size_t operator()(const Key& k) const
        {
            return std::hash<uint64_t>()((uint64_t) k.dev * 0x9E3779B97F4A7C15ull ^ (uint64_t) k.ino);
        }
    };
    std::mutex mx;
    std::unordered_map<Key, std::unique_ptr<FileEnt>, KeyHash> map;
    uint64_t clock = 0;
};

struct Ticket;

// One device read. Owned by its ticket; between admission and finish it belongs to exactly one
// executor thread (pool) or ring slot (io_uring).
struct Op
{
    Ticket* t = nullptr;
    FileEnt* f = nullptr;
    uint8_t* dst = nullptr;             // where the device read lands (bounce ops: set at issue)
    uint8_t* copy_dst = nullptr;        // bounce ops: payload destination
    int64_t dev_off = 0;
    uint32_t dev_len = 0;
    uint32_t need = 0;                  // bytes that must arrive from dev_off
    uint32_t pay_off = 0;               // bounce ops: payload offset in the read
    uint32_t pay_len = 0;               // payload bytes delivered (stats)
    uint32_t got = 0;                   // bytes read so far
    int32_t res = 0;
    int fd = -1;                        // descriptor to read (buffered or O_DIRECT)
    int fixed = -1;                     // io_uring fixed-file slot, -1 none
    int fd_advise = -1;                 // buffered descriptor for FADV_DONTNEED (dontneed ops)
    int32_t buf_index = -1;             // io_uring registered buffer holding dst, -1 none
    uint32_t align = 1;                 // O_DIRECT: resumption offsets must stay aligned
    uint16_t retries = 0;               // EINTR / EAGAIN resubmissions (cap kMaxRetries)
    uint16_t shorts = 0;                // short-read resumptions (cap kMaxShort)
    uint8_t direct = 0;
    uint8_t bounce = 0;
    uint8_t dontneed = 0;
    uint8_t async = 0;                  // io_uring: force io-wq (IOSQE_ASYNC)
    uint8_t poll = 0;                   // io_uring: goes to the IOPOLL ring
    uint8_t cls_issued = 0;
    int64_t t_issue = 0, t_done = 0;
};

struct Ticket
{
    uint64_t id = 0;
    int cls = 0;                        // current class (promote / aging change it)
    int cls_submit = 0;                 // class at submission (ticket stats)
    bool hold = false;
    int64_t deadline = 0;
    uint32_t* flag = nullptr;
    uint32_t flag_value = 0;
    std::vector<Op> ops;
    size_t next_issue = 0;              // ops [0, next_issue) were admitted or cancelled
    size_t n_inflight = 0;
    size_t n_finished = 0;
    int err = 0;                        // first error, -errno
    std::vector<FileEnt*> files;        // unique files, one ref each
    std::vector<int> buffers;           // io_uring registered buffers, one ref each
    int64_t t_submit = 0, t_first_issue = 0, t_done = 0, t_queued = 0;
    std::atomic<bool> complete { false };
    std::condition_variable cv;
    // class queue membership (intrusive)
    int q_cls = -1;
    Ticket* q_prev = nullptr;
    Ticket* q_next = nullptr;
};

struct TicketQueue
{
    Ticket* head = nullptr;
    Ticket* tail = nullptr;
};

// A pool admission: consecutive small ops of one ticket, run back to back on one thread and
// accounted as one request in flight
struct Span
{
    Ticket* t = nullptr;
    size_t first = 0;
    size_t count = 0;
    int cls = 0;
    int64_t bytes = 0;                  // accounted bytes in flight (largest op)
};

struct Counters
{
    uint64_t tickets = 0, ops = 0, op_errors = 0, ops_cancelled = 0, bytes = 0, held = 0, aged = 0;
    int64_t max_inflight = 0, max_inflight_bytes = 0;
    uint64_t hq[kHistBuckets] = {}, hs[kHistBuckets] = {}, ht[kHistBuckets] = {};
};

struct Core
{
    explicit Core(const Config& c);

    Config cfg;
    pid_t pid = 0;
    std::vector<std::string> fallbacks;

    std::mutex mx;
    TicketQueue q[kClasses];
    int64_t queued_ops = 0;             // unissued ops in the queues
    int inflight_total = 0;             // requests in flight (pool: spans)
    int inflight[kClasses] = {};
    int64_t inflight_bytes[kClasses] = {};
    int cap[kClasses] = {};
    int span_div = 0;                   // pool: spans of remaining / span_div ops; 0 = single ops
    int hold_pending = 0;               // unfinished hold tickets
    int64_t hold_armed_until = 0;
    uint64_t next_id = 1;
    std::unordered_map<uint64_t, std::unique_ptr<Ticket>> tickets;
    bool stopping = false;
    std::atomic<int> broken { 0 };      // fatal executor error (-errno), 0 = healthy
    std::condition_variable cv_drained;
    Counters cc[kClasses];
    uint64_t inline_reaped = 0, reaper_reaped = 0, resubmits = 0, stray_cqes = 0;
    std::atomic<uint64_t> enters { 0 };        // io_uring_enter calls that submitted (no lock)
    std::vector<TraceRec> trace;
    size_t trace_pos = 0;
    bool trace_wrapped = false;
    std::atomic<uint64_t> fault_ctr { 0 };

    FileTable files;
    std::unique_ptr<Executor> ex;

    // Scheduler (Core::mx held)
    bool hold_active(int64_t now) const;
    void age_refill(int64_t now);
    int64_t next_timer(int64_t now) const;
    bool gate(int c, int64_t bytes, bool hold) const;
    Op* admit_one(Ticket* only, int64_t now);           // only: that ticket's ops alone
    bool admit_span(Ticket* only, Span& sp, int64_t now);
    void op_done(Op& op, int res, int64_t now);
    void op_done_inline(Ticket* t, size_t n, int64_t now);  // ops [0, n) finished before queueing
    void unit_done(int c, int64_t bytes);
    void finish_op(Op& op, int res, int64_t now);           // io_uring: one op, one request
    void finish_span(const Span& sp, int64_t now);          // pool: a span, one request
    void complete_ticket(Ticket* t, int64_t now);
    void cancel_queued(Ticket* t, int err, int64_t now);
    void fail_all_queued(int err, int64_t now);
    void q_insert(Ticket* t, int c, int64_t now);
    void q_remove(Ticket* t);
    Ticket* find(uint64_t id);

    // Fault injection: returns the errno to fake for this attempt (0 none)
    int fault_errno();
    bool fault_short();
};

// Mechanism behind the scheduler
class Executor
{
public:
    virtual ~Executor() = default;
    virtual const char* name() const = 0;
    // Core::mx held: tickets were queued or a rule changed (pool: wake workers)
    virtual void kick(std::unique_lock<std::mutex>&) {}
    // No locks held, before a ticket its caller will wait for is queued: complete what can be
    // completed without blocking (pool: page-cache hits with RWF_NOWAIT). Moves the finished ops
    // to the front of t->ops and returns how many there are
    virtual size_t prepass(Ticket*) { return 0; }
    // No locks held, after a ticket was queued (t) or a rule changed (t == nullptr).
    // will_wait: this thread waits for t right away, without a timeout
    virtual void submitted(Ticket* t, bool will_wait) = 0;
    // Core::mx held on entry and exit: wait for t (helping where the backend can). Returns
    // false when deadline_ns (>= 0) passed first
    virtual bool wait(Ticket* t, std::unique_lock<std::mutex>& lk, int64_t deadline_ns) = 0;
    // No locks held, nothing in flight or queued: stop the threads
    virtual void stop() = 0;
    // FileTable::mx held
    virtual void file_opened(FileEnt*) {}
    virtual void file_closing(FileEnt*) {}
    // Registered buffers (io_uring)
    virtual int register_buffer(void*, size_t) { return -1; }
    virtual void unregister_buffer(int) {}
    virtual int find_buffer(const uint8_t*, size_t) { return -1; }   // Core::mx held
    virtual void buffer_ref(int, int) {}                              // Core::mx held
    virtual bool direct_bounce_ok() const { return true; }
    virtual std::string describe() const = 0;
};

std::unique_ptr<Executor> make_pool(Core& c);
// Returns nullptr (and why) when io_uring is unavailable
std::unique_ptr<Executor> make_uring(Core& c, std::string* why);

// Bounce buffer geometry for O_DIRECT rows
constexpr size_t kBounceBytes = 64 * 1024;

void set_thread_affinity(const std::vector<int>& cpus, const char* who);
void set_thread_name(const char* name);

}  // namespace detail
}  // namespace exl3_disk

#endif  // __linux__
