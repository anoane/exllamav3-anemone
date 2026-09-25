# Disk I/O engine

A native engine for positioned reads from model files: small scattered rows (n-gram / engram
tables) and large contiguous extents (routed experts kept on disk). One engine per process
serves every caller, so reads of different kinds share one admission point with priorities.
Code: `exllamav3/exllamav3_ext/disk/`. Knobs: `doc/env_vars.md`, section "Disk I/O engine".

## Backends

Three backends sit behind one interface, chosen by `EXL3_DISK_BACKEND`:

| Backend | Mechanism | Page cache |
|---|---|---|
| `pread` | a thread pool; the calling thread helps with its own reads | buffered |
| `odirect` | the same pool | `O_DIRECT`; a read covers the aligned superset of the bytes it needs |
| `io_uring` | one ring driven with raw syscalls (no liburing); fixed files, registered buffers, optional `SQPOLL`, optional `IOPOLL` ring | rows buffered, extents `O_DIRECT` |

`EXL3_DISK_DIRECT` overrides the page-cache column per traffic kind (rows, extents). `auto`
resolves to the fastest backend measured on the test host; until the maintenance-window
benchmark has run it resolves to `pread`, and `ngram_gather_cpu` keeps running its original
thread pool verbatim unless a backend is named explicitly. Windows keeps the overlapped
`ReadFile` gather (`ngram_gather_win.cpp`); `odirect` is accepted there because that path is
already unbuffered, `pread` and `io_uring` are refused with a message, and the new entry points
raise. Model code asks the extension which route is active (`disk_ngram_route()`), never the
environment, so on Windows the DeepSeek-V4.1 engram keeps its `ngram_gather_cpu` calls whatever
`EXL3_DISK_BACKEND` says.

## Requests, tickets and classes

A call describes its reads (rows of one or more tables for one sorted id list, or extents) and
gets one **ticket**. The native side turns the description into **ops**: consecutive rows are
coalesced into runs, runs and extents are cut into chunks (`EXL3_DISK_CHUNK`, default the
device's `max_sectors_kb`), and `O_DIRECT` ops get their aligned span. Python makes one call per
table set; no per-row or per-op work happens in Python.

Every ticket carries a **class**, which decides admission order:

| Class | Use | Byte window |
|---|---|---|
| 0 engram demand | rows the forward needs now; every `ngram_gather_cpu` call by default | none; its own request cap `EXL3_DISK_ENGRAM_QD` |
| 1 expert demand | a missed expert needed at its layer | `EXL3_DISK_WINDOW_EXPERT` |
| 2 prefetch | look-ahead rows and expert prefetch; earliest deadline first | `EXL3_DISK_WINDOW_PREFETCH` (hi / lo) |
| 3 refill | background RAM-tier fills; ages into class 2 after `EXL3_DISK_REFILL_AGE_MS` | `EXL3_DISK_WINDOW_REFILL` (hi / lo) |

The device has no priorities of its own (scheduler `none`, virtio), so priority is only what
the engine declines to submit yet. Admission runs in class order on every submission and
every completion, and an op is admitted when all of these hold:

- total ops in flight `<` `EXL3_DISK_QD`;
- for classes 1-3, total in flight `<` `EXL3_DISK_QD - EXL3_DISK_RESERVE0` (slots kept for rows);
- the class's own request cap;
- the class's bytes in flight plus the op's bytes `<=` its window (an op larger than the window
  runs alone, so it cannot starve). A window is positive or unlimited: `0` is refused, except
  as a `lo` window (below).

**Hold.** While a ticket submitted with `hold` is unfinished, or after `arm_hold()` until the
next hold ticket finishes (or `EXL3_DISK_HOLD_ARM_MS` passes), classes 2 and 3 use their `lo`
window (default 0: nothing new is submitted). Class 1 is never held. This keeps bulk reads
from queueing in front of a decode row batch; ops already in flight are never preempted. Only
class 0 and 1 tickets may carry `hold` (`EINVAL` otherwise): a class 2 or 3 hold ticket would be
held by its own rule and never finish.

`promote(ticket, cls)` moves a ticket's unissued ops to a higher class; `cancel(ticket)` drops
them (ops in flight finish into their buffers). A ticket completes when every op finished or
was cancelled; its result is 0 or the first error (`-errno`). The DeepSeek-V4.1 engram's
prefill look-ahead is a class-2 ticket that its forward promotes to class 0 when it starts
waiting for the rows: speculative until needed, then at full depth ahead of bulk.

## Threading

- **Synchronous calls take the synchronous path**, from C++ (`gather_rows`, `read_extents`)
  and from Python (`disk_gather_rows` / `disk_read_extents` with `wait=True`) alike. The calling
  thread builds the ops, takes the engine mutex, enqueues the ticket and runs admission for its
  own ticket. With `io_uring`, a class-0 caller prepares the SQEs and calls `io_uring_enter`
  itself when the submission lock is free: a decode batch leaves in one system call with no
  thread hand-off, and page-cache hits complete inside that call. If another thread is
  submitting, the caller does not wait for it (that thread may be inside a long
  `io_uring_enter`): it hands the ticket to the reaper, which admits class 0 first. Callers of
  classes 1-3 always hand their tickets to the reaper, so the only thread that ever submits
  bulk is the reaper. With the pools the caller first reads what the page cache holds with
  `RWF_NOWAIT` (decode-sized tickets; it stops once misses dominate, and the kernel has already
  started reading those), then wakes a few workers, each of which wakes the next while work
  remains, and reads its own queued ops alongside them.
- **Request ownership (`io_uring`).** A read belongs to the thread whose `io_uring_enter`
  submitted it, and a buffered read whose owner thread exits before it completes fails with
  `EFAULT` (its completion runs without the process's memory). So a caller thread submits only
  reads of the class-0 ticket it waits for without a timeout, and it holds the submission lock
  from preparing its SQEs to `io_uring_enter`, so it never submits SQEs another thread
  prepared. Everything else (asynchronous tickets, bulk, reads admitted when capacity frees,
  resumed short reads) is submitted by the reaper, which lives as long as the ring.
- **What an `io_uring_enter` costs the submitter.** A buffered read that hits the page cache
  is copied inside `io_uring_enter`, under the submission lock. Buffered reads of classes 1-3
  larger than 64 KiB are therefore flagged `IOSQE_ASYNC` and copied by the kernel's workers,
  so the reaper never holds the lock for long; row-sized reads stay inline (at most
  `QD - RESERVE0` of them per pass).
- **Completion.** `io_uring`: the reaper sleeps in `ppoll` on the ring (readable when
  completions are posted) and on an eventfd (the wake-up for new work), drains completions,
  then admits and submits what can start. A waiting caller spins for `EXL3_DISK_SPIN_US`,
  draining completions itself (those of its own reads are posted by its own task work), then
  sleeps on its ticket. The bounce copy of an `O_DIRECT` row and the `FADV_DONTNEED` of a
  buffered extent run after the engine mutex is released; the read counts as in flight until
  they are done. With `IOPOLL`, a poller thread polls the second ring only while it has reads
  in flight. Pools: `EXL3_DISK_THREADS` workers pull ops in class order; small ops are taken in
  spans, so a large gather costs a few hundred lock round trips, not one per row.
- **Fatal ring errors.** If `io_uring_enter` refuses submissions for good, the reads no ring
  consumed (on both rings) and everything queued fail with that error, later submissions are
  refused with it, and no engine thread keeps running on a timer (a stale hold or aging
  deadline no longer wakes the reaper).
- **Locks.** The engine mutex guards queues, counters, tickets and the ring slot tables.
  `io_uring` adds a submission lock (from preparing SQEs to `io_uring_enter`; taken before the
  engine mutex, the only nesting; a caller only ever tries it) and a completion-queue lock
  (around the CQ head, never held with another lock). The file table has its own mutex, never
  held with the engine mutex. A ticket counts the threads inside `wait()` / `release()` for it
  (under the engine mutex), and is destroyed only by a `release()`, after it completed and every
  such thread has left.
- **Time-based rules** (hold expiry, refill aging) are re-evaluated by idle workers and by the
  reaper, which sleep with a timeout while such a rule is pending.

## Memory ownership

- **Destinations belong to the caller** and must stay valid until the ticket completes. The
  Python ticket object keeps its tensors alive and waits for in-flight ops before it lets go of
  them; releasing a ticket always cancels its queued ops and waits for the ones in flight, so
  no read can land after a release returns. Any number of threads may wait for a ticket while
  another releases it: every release returns after the ticket completed, the ticket is freed
  after the last waiter left, and a wait that starts after the release finished reports an
  unknown (released) ticket. Rows land at `out + i * row_bytes`; an extent's
  payload lands at `slot + payload_offset`, where `payload_offset = offset mod G` and `G` is
  the file's `O_DIRECT` alignment rounded up to 4 KiB. The slot must be 4 KiB aligned and hold
  `span = roundup(payload_offset + length, G)` bytes, whatever the backend, so one slot geometry
  serves every backend; bytes of the slot outside the payload are scratch.
- **Bounce buffers belong to the engine.** `O_DIRECT` rows are read into aligned bounce slots
  (one per pool thread, or one per ring slot for `io_uring`, registered as fixed buffer 0) and the
  payload is copied to the destination.
- **Files.** The engine never reads through a caller's descriptor. It keys files by
  `(st_dev, st_ino)`, reopens each once through `/proc/self/fd` (buffered, and `O_DIRECT` on
  first direct use), applies its `posix_fadvise` policy to its own descriptors, and closes them
  at shutdown. A caller may close its descriptor at any time after submitting. Since the
  engine's descriptors (and, with `io_uring`, its fixed-file slots) would keep a deleted file's
  blocks allocated, it closes a file when asked (`forget(fd)`, `forget_path(path)`,
  `ext.disk_engine_forget(path)`; the DeepSeek-V4.1 engram calls it when it unloads; a file that
  is still being read is closed once idle), and on its own once an idle file's last name was
  removed (checked every 64 submissions). A forgotten file that is read again is simply
  reopened. Offsets are bounds-checked against the file size at submission, and every size
  computation is overflow-checked (no read may end past `INT64_MAX - 1 MiB`, so the alignment
  arithmetic cannot overflow).
- **Flags.** A ticket may name a 32-bit word (for example in pinned, mapped memory) that the
  engine sets to a value with a release store when the ticket completes, whatever the outcome,
  so a GPU stream waiting on it (`exl3_moe_flag_wait`) never hangs; the host learns an error
  from `wait()`.
- **Shutdown** cancels queued ops, waits for every op in flight, then joins the threads and
  closes the ring and the descriptors. It never returns while a read may still land. The
  process-wide default engine is never destroyed at exit, like the original gather pool. A
  forked child never touches its parent's engine: every entry point refuses with `ECHILD`
  before it takes a lock the parent's threads may have held at `fork()` (`release()` does
  nothing), and the mutexes of the process-wide engine and of the registered-buffer list are
  held across `fork()`, so a child can create an engine of its own.
- **Registered buffers** (Python) keep their tensor referenced while their engine exists;
  once the engine is gone (replaced by `disk_engine_configure`, shut down, or released by its
  last ticket) the next configure, shutdown or (un)register call drops them.

## Entry points

C++ (`disk/disk_engine.h`): `Engine::submit_rows`, `submit_extents`, `gather_rows`,
`read_extents`, `wait`, `done`, `release`, `cancel`, `promote`, `arm_hold`, `stats`,
`extent_geometry`, `register_buffer`, `forget`, `forget_path`. Python (`exllamav3_ext`):
`disk_gather_rows`, `disk_read_extents`, `disk_extent_geometry`, `DiskTicket`, `disk_arm_hold`,
`disk_stats`, `disk_engine_info`, `disk_engine_configure`, `disk_engine_forget`,
`disk_set_thread_class`, `disk_register_buffer`, `disk_ngram_route`. `ngram_gather_cpu` keeps
its signature; with a backend named it runs on the engine.

## Tests

- `tests/disk_engine/`: the standalone C++ driver (byte-exact checks against a reference `pread`,
  concurrency, priorities, cancellation, shutdown with reads in flight, fault injection) and its
  release, ASan/UBSan and TSan builds.
- `tests/test_disk_engine_.py`: the Python bindings and `ngram_gather_cpu` against the original
  gather, tickets waited for and released from different threads, registered buffers and
  forgotten files.
- `tests/test_dsv41_engram_disk_.py`: the DeepSeek-V4.1 engram's staging (inline, and the
  look-ahead promoted by its consumer, and retired) against the original gather, and its route
  taken from the extension.
- `tests/disk_engine/bench_engine.py`: throughput, latency percentiles and the Python share of
  each call, per backend and access pattern.
