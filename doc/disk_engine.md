# Disk I/O engine

A native engine for positioned reads from model files: small scattered rows (n-gram and engram
tables) and large contiguous extents (routed experts kept on disk). One engine per process serves
every caller, so reads of different kinds share one admission point with priorities, and three
I/O backends (`pread`, `odirect`, `io_uring`) sit behind the same interface. Code:
`exllamav3/exllamav3_ext/disk/`. Tests and benchmarks: `tests/disk_engine/`.

Contents: [quick start](#quick-start), [backends](#backends), [how `auto` is
decided](#how-auto-is-decided), [requests, classes and windows](#requests-tickets-and-classes),
[threading](#threading), [memory ownership](#memory-ownership), [knob reference](#knob-reference),
[virtual machines and bare metal](#virtual-machines-and-bare-metal),
[benchmarking your own machine](#benchmarking-your-own-machine), [measured on the test
host](#measured-on-the-test-host), [entry points](#entry-points), [tests](#tests).

## Quick start

- **Nothing to do by default.** `EXL3_DISK_BACKEND` unset means `auto`, which resolves to the
  choice recorded in `exllamav3_ext/disk/disk_auto.h` (see [below](#how-auto-is-decided)). Until
  the benchmark has decided on the test host, that is the behaviour from before the engine
  existed: `ngram_gather_cpu` runs its original thread pool, and the engine's own entry points
  use the buffered thread pool.
- **Pick a backend:** `EXL3_DISK_BACKEND=io_uring` (or `pread`, `odirect`). `ngram_gather_cpu`
  and the DeepSeek-V4.1 engram then read through the engine with that backend.
- **Keep the original gather whatever `auto` says:** `EXL3_DISK_BACKEND=original`.
- **See what is active:** `EXL3_DISK_VERBOSE=1` prints the resolved configuration once; from
  Python, `exllamav3_ext.disk_engine_info()` returns it (`backend`, `ngram_route`,
  `auto_backend`, `auto_ngram_route`, `describe`, `fallbacks`).
- **Measure your machine:** [benchmarking your own machine](#benchmarking-your-own-machine).

Every `EXL3_DISK_*` variable is read once, when the engine is first created (the first read that
needs it), and cached for the process; `EXL3_DISK_BACKEND` alone decides the route of
`ngram_gather_cpu`, also read once. A bad value raises at that first use, naming the variable
and the accepted values. A variable that does not apply to the chosen backend is reported once
on stderr and ignored. `exllamav3_ext.disk_engine_configure({...})` replaces the engine with
another configuration inside one process (tests, benchmarks); its keys are the variable names
without `EXL3_DISK_`, in lower case.

## Backends

| `EXL3_DISK_BACKEND` | Mechanism | Page cache | `ngram_gather_cpu` |
|---|---|---|---|
| `original` | not an engine backend: the original gather pool (one `pread` per coalesced run on up to 24 threads; Windows: overlapped `ReadFile`) | buffered (Windows: unbuffered) | original pool |
| `pread` | engine thread pool; a synchronous caller reads what the page cache holds itself, then helps with its own reads | buffered | engine |
| `odirect` | the same pool, `O_DIRECT`; a read covers the aligned superset of the bytes it needs | bypassed | engine |
| `io_uring` | one ring driven with raw system calls (no liburing); fixed files, registered buffers, optional `SQPOLL`, optional `IOPOLL` ring | rows buffered, extents `O_DIRECT` | engine |
| `auto` (default) | what `disk_auto.h` records | per that backend | per `disk_auto.h` |

- **`pread`**: `EXL3_DISK_THREADS` workers read through the page cache. A synchronous caller
  first reads, on its own thread, whatever the page cache already holds (`preadv2` with
  `RWF_NOWAIT`, for calls of up to 1024 reads; it gives up after a few misses, which the kernel
  has by then started reading), then reads its own queued reads alongside the workers. Works
  everywhere Linux runs, containers included. Best when rows are mostly warm and the process
  has CPUs to spare.
- **`odirect`**: the same pool with `O_DIRECT` reads. Rows are read as the aligned sector range
  around them into per-thread bounce buffers and copied out; extents land directly in their
  slots. No page-cache pollution, and no page-cache warmth either: a row read twice is read from
  the device twice. Best for experts on a machine whose RAM is committed elsewhere.
- **`io_uring`**: one ring per engine. A synchronous class-0 caller (decode rows) prepares its
  own reads and submits them with one `io_uring_enter`; page-cache hits complete inside that call
  on the caller's thread. When another thread is submitting at that moment it hands its reads to
  the reaper instead of waiting. The reaper thread submits everything else (asynchronous tickets,
  all bulk, reads freed by the windows) and completes; buffered bulk reads above 64 KiB go to the
  kernel's workers (`IOSQE_ASYNC`), so the reaper never spends long inside `io_uring_enter`. Rows
  are read through the page cache and extents with `O_DIRECT` unless `EXL3_DISK_DIRECT` says
  otherwise. Needs Linux 5.11 or later; where `io_uring_setup` is refused (a container's seccomp
  profile, `kernel.io_uring_disabled=2`, an old kernel) the engine falls back to the thread pool
  (`odirect` pool when extents are direct) and says so once. Best for cold rows: one system call
  submits a whole batch at full queue depth.

**Windows** keeps the overlapped `ReadFile` gather (`ngram_gather_win.cpp`) for
`ngram_gather_cpu`: `auto`, `original` and `odirect` are accepted there (that path already reads
unbuffered, which is what `odirect` means), `pread` and `io_uring` are refused with a message,
and the engine's own entry points raise. Model code asks the extension which route is active
(`disk_ngram_route()`), never the environment, so on Windows the DeepSeek-V4.1 engram keeps its
`ngram_gather_cpu` calls whatever `EXL3_DISK_BACKEND` says.

## How `auto` is decided

`auto` is three constants in `exllamav3/exllamav3_ext/disk/disk_auto.h`:

- `kAutoBackend`: the engine backend `auto` resolves to (for the extension's `disk_*` entry
  points, and for `ngram_gather_cpu` when the route below is the engine);
- `kAutoNgramEngine`: whether `ngram_gather_cpu` (PLE n-gram tables, the DeepSeek-V4.1 engram)
  uses the engine (`true`) or keeps its original pool (`false`);
- `kAutoKeepaliveMs`: the keep-alive period `EXL3_DISK_KEEPALIVE_MS` takes when unset (`0` off).

The file is written whole by `bench_suite.py decide --apply` from a benchmark on the test host
(the maintenance-window run, `tests/disk_engine/maintenance_window.sh --apply`), with the
evidence in its comment, and committed with the report. Nobody edits it by hand without evidence.
Until that run it records the original behaviour (`Pread`, `false`, `0`).

The rule, applied to the `decision` runs (every backend at its default knobs, `--repeats` runs
with the backend order rotated, on the model's own engram tables):

1. **Validity.** A run counts only if it read without errors, every sampled row and extent
   matched a plain `pread` of the same bytes, it was not cut short by the read budget, the pages
   it read were in the state asked for (within 2 %), and the host was quiet before it (at most
   `--max-foreign` other cores busy, default 1). Otherwise `decide` refuses (`--force` overrides
   and says so). A backend that fell back on the host (e.g. `io_uring` refused) is not a
   candidate.
2. **Decode-critical metric.** The p50 latency of one DeepSeek-V4.1 decode gather: 48 unique
   ids, both tables of one engram layer in one engine call, class 0 with the hold rule, with 75 %
   of the pages it reads resident (`--decode-cache`). 0.75 approximates the steady state of a
   page cache holding ~26 GiB per engram layer on natural text, where replaying real prompts
   removed about three quarters of the row reads; cold and warm are reported alongside, and the
   report says whether they would pick the same backend. Median over the runs.
3. **Ties.** Candidates within the tolerance of the best (5 %, or the largest run-to-run spread
   among them if larger) stay; the next step decides among them: p99 of the same case, then the
   decode p50 next to a class-1 expert stream (the latency the expert tier will impose), then
   extent throughput with 4 in flight (higher is better), then CPU time per device read. Still
   tied: the current `auto` backend stays (no change without evidence).
4. **N-gram route.** Through Python, exactly as the engram calls it (`bench_engine.py`), the
   winner's decode gather must beat the original two `ngram_gather_cpu` calls by the tolerance,
   and its cold 8192-row prefill gather must not be slower than the original's by more than the
   tolerance. Then `kAutoNgramEngine = true`; otherwise `false`. Without Python runs the route
   stays as it is.
5. **Keep-alive.** For the winner, decode rows after 20 ms of idle before each call (the spacing
   between tokens), from the `keepalive` runs: if a 5 ms keep-alive makes their p50 faster by the
   tolerance, `kAutoKeepaliveMs = 5`; otherwise `0`.

`decide` also reports, without applying anything, what the `window_expert` sweep says about
`EXL3_DISK_WINDOW_EXPERT` for each backend swept: decode latency next to a class-1 extent stream
and the stream's rate at each window, and a window worth considering when it keeps the bulk rate
within the tolerance of the default's while cutting that latency by at least the tolerance. The
default window is a knob default, not part of `auto`; changing it is a decision for the maintainer.

`decide` writes `decision.md` / `decision.json` (every value and step), and with `--apply ROOT`
rewrites `ROOT/exllamav3/exllamav3_ext/disk/disk_auto.h` and leaves the diff in
`disk_auto.patch`. On another machine the same command tells you what `auto` would be there;
to use it without rebuilding, set `EXL3_DISK_BACKEND` to the winner.

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

The device has no priorities of its own (scheduler `none`, virtio), so priority is only what the
engine declines to submit yet. Admission runs in class order on every submission and every
completion, and an op is admitted when all of these hold:

- total ops in flight `<` `EXL3_DISK_QD`;
- for classes 1-3, total in flight `<` `EXL3_DISK_QD - EXL3_DISK_RESERVE0` (slots kept for rows);
- the class's own request cap;
- the class's bytes in flight plus the op's bytes `<=` its window (an op larger than the window
  runs alone, so it cannot starve). A window is positive or unlimited: `0` is refused, except as
  a `lo` window (below).

**Hold.** While a ticket submitted with `hold` is unfinished, or after `arm_hold()` until the next
hold ticket finishes (or `EXL3_DISK_HOLD_ARM_MS` passes), classes 2 and 3 use their `lo` window
(default 0: nothing new is submitted). Class 1 is never held. This keeps bulk reads from queueing
in front of a decode row batch; ops already in flight are never preempted. Only class 0 and 1
tickets may carry `hold` (`EINVAL` otherwise): a class 2 or 3 hold ticket would be held by its own
rule and never finish.

`promote(ticket, cls)` moves a ticket's unissued ops to a higher class; `cancel(ticket)` drops
them (ops in flight finish into their buffers). A ticket completes when every op finished or was
cancelled; its result is 0 or the first error (`-errno`). The DeepSeek-V4.1 engram's prefill
look-ahead is a class-2 ticket that its forward promotes to class 0 when it starts waiting for
the rows: speculative until needed, then at full depth ahead of bulk.

## Threading

- **Synchronous calls take the synchronous path**, from C++ (`gather_rows`, `read_extents`) and
  from Python (`disk_gather_rows` / `disk_read_extents` with `wait=True`) alike. The calling
  thread builds the ops, takes the engine mutex, enqueues the ticket and runs admission for its
  own ticket. With `io_uring`, a class-0 caller prepares the SQEs and calls `io_uring_enter`
  itself when the submission lock is free: a decode batch leaves in one system call with no
  thread hand-off, and page-cache hits complete inside that call. If another thread is
  submitting, the caller does not wait for it (that thread may be inside a long
  `io_uring_enter`): it hands the ticket to the reaper, which admits class 0 first. Callers of
  classes 1-3 always hand their tickets to the reaper, so the only thread that ever submits bulk
  is the reaper. With the pools the caller first reads what the page cache holds with
  `RWF_NOWAIT` (decode-sized tickets; it stops once misses dominate, and the kernel has already
  started reading those), then wakes a few workers, each of which wakes the next while work
  remains, and reads its own queued ops alongside them.
- **Request ownership (`io_uring`).** A read belongs to the thread whose `io_uring_enter`
  submitted it, and a buffered read whose owner thread exits before it completes fails with
  `EFAULT` (its completion runs without the process's memory). So a caller thread submits only
  reads of the class-0 ticket it waits for without a timeout, and it holds the submission lock
  from preparing its SQEs to `io_uring_enter`, so it never submits SQEs another thread prepared.
  Everything else (asynchronous tickets, bulk, reads admitted when capacity frees, resumed short
  reads) is submitted by the reaper, which lives as long as the ring.
- **What an `io_uring_enter` costs the submitter.** A buffered read that hits the page cache is
  copied inside `io_uring_enter`, under the submission lock. Buffered reads of classes 1-3 larger
  than 64 KiB are therefore flagged `IOSQE_ASYNC` and copied by the kernel's workers, so the
  reaper never holds the lock for long; row-sized reads stay inline (at most `QD - RESERVE0` of
  them per pass).
- **Completion.** `io_uring`: the reaper sleeps in `ppoll` on the ring (readable when completions
  are posted) and on an eventfd (the wake-up for new work), drains completions, then admits and
  submits what can start. A waiting caller spins for `EXL3_DISK_SPIN_US`, draining completions
  itself (those of its own reads are posted by its own task work), then sleeps on its ticket. The
  bounce copy of an `O_DIRECT` row and the `FADV_DONTNEED` of a buffered extent run after the
  engine mutex is released; the read counts as in flight until they are done. With `IOPOLL`, a
  poller thread polls the second ring only while it has reads in flight. Pools:
  `EXL3_DISK_THREADS` workers pull ops in class order; small ops are taken in spans, so a large
  gather costs a few hundred lock round trips, not one per row.
- **Fatal ring errors.** If `io_uring_enter` refuses submissions for good, the reads no ring
  consumed (on both rings) and everything queued fail with that error, later submissions are
  refused with it, and no engine thread keeps running on a timer.
- **Locks.** The engine mutex guards queues, counters, tickets and the ring slot tables.
  `io_uring` adds a submission lock (from preparing SQEs to `io_uring_enter`; taken before the
  engine mutex, the only nesting; a caller only ever tries it) and a completion-queue lock (around
  the CQ head, never held with another lock). The file table has its own mutex, never held with
  the engine mutex. A ticket counts the threads inside `wait()` / `release()` for it (under the
  engine mutex), and is destroyed only by a `release()`, after it completed and every such thread
  has left.
- **Time-based rules** (hold expiry, refill aging) are re-evaluated by idle workers and by the
  reaper, which sleep with a timeout while such a rule is pending.

## Memory ownership

- **Destinations belong to the caller** and must stay valid until the ticket completes. The
  Python ticket object keeps its tensors alive and waits for in-flight ops before it lets go of
  them; releasing a ticket always cancels its queued ops and waits for the ones in flight, so no
  read can land after a release returns. Any number of threads may wait for a ticket while another
  releases it: every release returns after the ticket completed, the ticket is freed after the
  last waiter left, and a wait that starts after the release finished reports an unknown
  (released) ticket. Rows land at `out + i * row_bytes`; an extent's payload lands at
  `slot + payload_offset`, where `payload_offset = offset mod G` and `G` is the file's `O_DIRECT`
  alignment rounded up to 4 KiB. The slot must be 4 KiB aligned and hold
  `span = roundup(payload_offset + length, G)` bytes, whatever the backend, so one slot geometry
  serves every backend; bytes of the slot outside the payload are scratch.
- **Bounce buffers belong to the engine.** `O_DIRECT` rows are read into aligned bounce slots
  (one per pool thread, or one per ring slot for `io_uring`, registered as fixed buffer 0; 64 KiB
  each, allocated whatever `EXL3_DISK_DIRECT` says, since a request may ask for direct rows) and
  the payload is copied to the destination.
- **Files.** The engine never reads through a caller's descriptor. It keys files by
  `(st_dev, st_ino)`, reopens each once through `/proc/self/fd` (buffered, and `O_DIRECT` on first
  direct use), applies its `posix_fadvise` policy to its own descriptors, and closes them at
  shutdown. A caller may close its descriptor at any time after submitting. Since the engine's
  descriptors (and, with `io_uring`, its fixed-file slots) would keep a deleted file's blocks
  allocated, it closes a file when asked (`forget(fd)`, `forget_path(path)`,
  `ext.disk_engine_forget(path)`; the DeepSeek-V4.1 engram calls it when it unloads; a file that is
  still being read is closed once idle), and on its own once an idle file's last name was removed
  (checked every 64 submissions). A forgotten file that is read again is simply reopened. Offsets
  are bounds-checked against the file size at submission, and every size computation is
  overflow-checked (no read may end past `INT64_MAX - 1 MiB`, so the alignment arithmetic cannot
  overflow).
- **Flags.** A ticket may name a 32-bit word (for example in pinned, mapped memory) that the
  engine sets to a value with a release store when the ticket completes, whatever the outcome, so
  a GPU stream waiting on it (`exl3_moe_flag_wait`) never hangs; the host learns an error from
  `wait()`.
- **Shutdown** cancels queued ops, waits for every op in flight, then joins the threads and closes
  the ring and the descriptors. It never returns while a read may still land. The process-wide
  default engine is never destroyed at exit, like the original gather pool. A forked child never
  touches its parent's engine: every entry point refuses with `ECHILD` before it takes a lock the
  parent's threads may have held at `fork()` (`release()` does nothing), and the mutexes of the
  process-wide engine and of the registered-buffer list are held across `fork()`, so a child can
  create an engine of its own.
- **Registered buffers** (Python) keep their tensor referenced while their engine exists; once the
  engine is gone (replaced by `disk_engine_configure`, shut down, or released by its last ticket)
  the next configure, shutdown or (un)register call drops them.

## Knob reference

Every variable below is also listed in `doc/env_vars.md` ("Disk I/O engine"). Sizes take `K`,
`M` or `G` (binary: `8M` = 8 MiB); booleans take `0`/`1`, `on`/`off`, `true`/`false`, `yes`/`no`.
"Test host" is the Proxmox guest described [below](#virtual-machines-and-bare-metal).

### `EXL3_DISK_BACKEND` (default: `auto`)

`auto`, `original`, `pread`, `odirect` or `io_uring` (see [backends](#backends)). `auto`
resolves through `disk_auto.h` ([how `auto` is decided](#how-auto-is-decided)); on the test host
it is `io_uring` with the n-gram gather through the engine and a 5 ms keep-alive, decided from the
2026-09-25 maintenance-window run. `original` keeps
`ngram_gather_cpu` on its original pool while the engine's own entry points use `auto`'s backend:
the switch back to the pre-engine behaviour, whatever `auto` says. Naming a backend sends
`ngram_gather_cpu` and the DeepSeek-V4.1 engram (one call per layer for both tables) through the
engine. When to change it: to try a backend on your machine before (or instead of) re-deciding
`auto`, or `original` to rule the engine out when chasing a problem.

### `EXL3_DISK_DIRECT` (default: `auto`)

Which reads bypass the page cache: `none`, `rows`, `extents` or `all` (`0`/`1` mean `none`/`all`).
`auto` follows the backend: `pread` none, `odirect` all, `io_uring` extents. Rows through the page
cache keep a warm tier for free (on the test host a warm engram layer-chunk costs 25 ms against
1.04 s cold); extents through it would evict those rows and duplicate the RAM tier, so buffered
extent reads drop their pages afterwards (`EXL3_DISK_EXTENT_DONTNEED`). When a file system refuses
`O_DIRECT` (tmpfs on older kernels, some network file systems), that file is read through the
page cache and a message names it. Change it to `all` with `io_uring` when RAM for the page cache
is scarce (every row read then costs a device read), or to `none` to measure the page cache's
share.

A request may set its own mode, which overrides this variable for its reads only: `Options::direct`
(`-1` this variable's choice for its kind of read, `0` buffered, `1` `O_DIRECT`), the `direct=`
argument of `ext.disk_gather_rows`, and, for `ngram_gather_cpu` (whose signature is the original
one), the calling thread's mode (`ext.disk_set_thread_direct(-1|0|1)`, which returns the previous
one). A placement's `disk io=direct|buffered` sets it for the expert tier's reads and the n-gram /
engram rows of that model, so two models in one process may read differently
([expert_tiers.md](expert_tiers.md)). A file opened buffered is reopened `O_DIRECT` on the first
direct request, as for this variable.

### `EXL3_DISK_QD` (default: `128` for `io_uring`, threads + 8 for the pools)

Requests in flight at once, all classes together; `1`-`4096`. More than the device's own queue
(`/sys/block/<dev>/queue/nr_requests`, 256 on the test host; `device/queue_depth` 128) only
queues inside the kernel, where the engine can no longer order it. For the pools the thread count
caps it anyway (a caller helping with its own reads adds one). On the test host random 4 KiB
reads went from 150 K/s at depth 24 to 210 K/s at 96; on bare-metal NVMe, depth 128-256 per
device is typical, and a PCIe 5 drive needs that depth to reach its IOPS. Lower it when bulk
reads must not flood a shared device; raise it (with `EXL3_DISK_THREADS` for the pools) on
fast NVMe. The benchmark's `qd` sweep measures 32, 64 and 256 against the default.

### `EXL3_DISK_RESERVE0` (default: 3/4 of `EXL3_DISK_QD`)

Slots only class 0 (rows the forward waits for) may use: classes 1-3 together never have more
than `EXL3_DISK_QD - EXL3_DISK_RESERVE0` requests in flight, so a decode row batch always finds
room. `0` to `QD - 1`. With the pool defaults (24 threads, qd 32, reserve 24) the bulk classes get
8 slots, 8 chunks in flight: on the test host that still read 13.3 MB extents at ~11 GB/s with
`odirect`, but a device that needs more depth wants a larger `EXL3_DISK_THREADS` and `QD`. With
`io_uring` (qd 128) bulk gets 32 slots of `EXL3_DISK_CHUNK` each.

### `EXL3_DISK_ENGRAM_QD` (default: `EXL3_DISK_QD`)

Requests in flight for class 0 alone, `1` to `QD`. Random 4 KiB reads stop gaining past ~96 in
flight on the test host; beyond that depth only adds latency to everything else. Lower it when a
large prefill gather (8192 rows and more) should leave depth to expert reads; the `engram_qd`
sweep measures 32 and 64.

### `EXL3_DISK_WINDOW_EXPERT` (default: `8M`), `EXL3_DISK_WINDOW_PREFETCH` (default: `8M/0`), `EXL3_DISK_WINDOW_REFILL` (default: `8M/0`)

Bytes in flight per class: 1 (expert demand), 2 (prefetch, earliest deadline first) and 3
(background refill); `inf` means unlimited. For classes 2 and 3, `hi/lo`: `lo` applies while the
hold rule is active, `hi` otherwise; a single value sets `hi` and keeps `lo` at 0. Class 1 is
never held and takes one value: a power of two from `256K` to `1G` (`1M`, `2M`, `4M`, `8M`, ...),
or `inf` for measurements; anything else is refused with the accepted range in the message. The
lower bound is one bulk chunk at its smallest; the upper bound leaves room for devices much faster
than today's. For the other classes `hi` must be positive or `inf` (`0` would never admit the
class and is refused); `lo` may be 0. A read larger than its window is admitted when nothing else of
its class is in flight, so a window never starves a class.

Why windows: the device serves roughly in order, so a row read queued behind B bytes of bulk
waits about B / bandwidth (~0.8 ms behind 8 MiB at 10.5 GB/s, ~3 ms behind 32 MiB); two 13 MB
extents in flight already reach full bandwidth on the test host. So the class-1 window trades
decode-row latency against expert-miss latency: 32 MiB lets two extents stream at full rate, and
a demand stream then costs a concurrent decode batch ~3 ms at p50 on the test host; 8 MiB bounds
the batch's wait to ~0.8 ms, and whether it costs bulk rate depends on how much depth the device
needs. On the test host's smoke run it cost none: with four extents queued, `8M` kept the stream
at 10.0 GB/s (9.7 GB/s at 32M) and cut the decode batch next to it from 2.9 ms to 0.70 ms at p50
(`io_uring`, one run of 15 calls on a busy host, so a hint, not a result). The maintenance-window
run on the model's shards (2026-09-25, `io_uring`, four extents in flight, a cold 48-row decode
batch every 2 ms) confirmed it, so the default is `8M`: decode p50 / p99 750 / 1116 us and a
bulk rate of 8.94 GB/s at 8M, against 1419 / 3198 us and 8.39 GB/s at 32M and 1717 / 3662 us and
8.25 GB/s at 64M. The window run sweeps the alternatives (32M, 64M), and `decide` reports what the
sweep says (as advice: the default window is not part of `auto`). The prefetch window with `lo` 0 keeps look-ahead reads entirely out of the
way of a held decode batch (raising it to `32M/0` made decode next to a prefetch stream 6x slower
in the same smoke run). Chunks of `EXL3_DISK_CHUNK` make the windows fine-grained. On a faster
device (bare-metal PCIe 5 NVMe at ~14 GB/s) the same window costs less time, so larger windows
become affordable.

### `EXL3_DISK_HOLD_ARM_MS` (default: `50`)

The hold rule keeps classes 2 and 3 at their `lo` window while a row batch submitted with `hold`
is unfinished (the DeepSeek-V4.1 engram at decode), and from `disk_arm_hold()` until the next
hold batch finishes, so the device drains before the rows arrive. This is the safety expiry of an
arm that no hold batch follows. `0` disables arming. Only class 0 and 1 submissions may ask for
`hold` (a class 2 or 3 one would be held by its own rule; it is refused with `EINVAL`).

### `EXL3_DISK_REFILL_AGE_MS` (default: `250`)

A class-3 ticket queued this long moves to class 2 (after the deadlined prefetches), so refills
cannot starve behind a steady prefetch stream. `0` disables aging.

### `EXL3_DISK_CHUNK` (default: `auto`)

Bulk reads (extents, long row runs) are split into chunks of this size: `auto` uses the device's
`queue/max_sectors_kb` (the block layer splits there anyway; 1280 KiB on the test host), clamped
to 256 KiB - 4 MiB; otherwise a multiple of 64 KiB in [64K, 64M]. Smaller chunks let the windows
act at a finer grain at the cost of more requests; the kernel sees the same requests either way
below `max_sectors_kb`. The `chunk` sweep measures 512K and 4M.

### `EXL3_DISK_ALIGN` (default: `auto`)

`O_DIRECT` offset and length alignment. `auto` asks the file system (`statx` `STATX_DIOALIGN`,
512 bytes on the test host's ext4), then the device's logical block size, then assumes 4096. A
power of two in [512, 65536] overrides it upwards (below the file system's own alignment is
refused). A 256-byte row read with 512-byte alignment reads 512 or 1024 bytes; with 4096, a
whole page. Some devices serve 4 KiB-aligned reads faster than 512-byte ones (4Kn media behind a
512e interface); the `align` sweep measures 4096 against `auto`. Extent slots always use a
4 KiB-aligned geometry whatever this says: the payload of an extent at `offset` lands at
`slot + offset % G`, `G` = max(4096, alignment), and the slot must hold
`roundup(offset % G + length, G)` bytes (`disk_extent_geometry`).

### `EXL3_DISK_THREADS` (default: `min(32, max(4, CPUs))`)

Workers of the `pread` and `odirect` pools, `1`-`256` (24 on the test host, like the original
gather). A pool thread keeps one read in flight, so the pools reach queue depth only with many
threads: raise it (48-96) for cold row batches or extents on a fast device. Ignored by
`io_uring` (see `EXL3_DISK_IOWQ_WORKERS`). The `threads` sweep measures 8, 48 and 96.

### `EXL3_DISK_SPIN_US` (default: `50`)

A waiting caller spins this long (draining completions itself with `io_uring`) before it sleeps.
A sleeping waiter costs one wake-up (13 us p50, 35 us p99 on the test host) when its reads
complete. `0`-`100000`. `0` saves CPU on hosts where every core is busy (the spin competes with
the CPU MoE workers); larger values help only when reads complete within the spin (warm rows,
very fast devices). The `spin_us` sweep measures 0 and 200.

### `EXL3_DISK_SQPOLL` (default: `0`), `EXL3_DISK_SQPOLL_IDLE_MS` (default: `10`), `EXL3_DISK_SQPOLL_CPU` (default: `-1`)

`io_uring` only. A kernel thread polls the submission queue, so submitting costs no system call.
It burns a CPU while busy and sleeps after `IDLE_MS` of no work (the next submission wakes it with
one call). `SQPOLL_CPU` pins that thread to one CPU (`-1`: anywhere). Worth trying on bare metal
with CPUs to spare and very high request rates (hundreds of thousands of reads per second); in a
guest whose vCPUs are shared with the CPU MoE workers it mostly steals a core. Refused at ring
creation (old kernels, missing privileges): a message, and the engine continues without it. The
benchmark reports the process's CPU time with the SQPOLL thread included.

### `EXL3_DISK_IOPOLL` (default: `0`)

Polled completions for `O_DIRECT` reads: `io_uring` puts them on a second ring created with
`IORING_SETUP_IOPOLL` and a poller thread spins on the device while they are in flight; the pools
read with `preadv2(RWF_HIPRI)`. Only an NVMe device with poll queues can do this
(`nvme.poll_queues=N` on the kernel command line, visible as `queue/io_poll = 1`). The engine
checks each file's device and refuses the flag for files that cannot poll, once per device, with
the reason; virtio and SCSI disks (the test host's `sda`) have no poll queues, so it is refused
there and the reads use interrupts as usual. Saves a few microseconds per read (the interrupt and
wake-up) at the cost of a busy CPU while reads are in flight; relevant for small `O_DIRECT` rows
on bare metal or NVMe passthrough. Registered buffers are registered on the polled ring too.
(`EXL3_DISK_FAULT_FORCE_IOPOLL` skips the device check, for tests only.)

### `EXL3_DISK_REGISTER` (default: `auto`)

`io_uring` only: `none`, `files`, `buffers` or `all` (`auto` = `all`). Fixed files skip the
per-read descriptor lookup; registered buffers (the `O_DIRECT` bounce arena, and caller buffers
registered with `disk_register_buffer`, e.g. RAM-tier arena chunks of up to 1 GiB) make reads into
them `READ_FIXED`, which skips pinning the pages per read. Registration failures fall back to
plain reads with a message (a locked-memory limit, `RLIMIT_MEMLOCK`, is the usual cause). The
table holds up to 1,023 caller buffers (index 0 is the bounce arena), kept sorted by address, so
finding the buffer of a read's destination is a binary search however many are registered (an
expert RAM tier of 126 GiB registers 126 chunks); a registration beyond the table returns -1 (not
an error: that memory is read with plain reads). A
registered buffer stays referenced until `disk_unregister_buffer`, which refuses while reads
still target it, or until its engine is gone. The `register` sweep measures each setting.

Not every read into a registered buffer is `READ_FIXED`. The kernel describes a fixed read as the
buffer's page vectors from the destination's page to the buffer's end, and a direct I/O bio keeps
that count in 16 bits: with 65,537 pages left (or 131,073, 196,609, ...) it reads as a one-page
bio, which skips the block layer's segment split, and an `O_DIRECT` read of many pages then
reaches the disk driver with more segments than the driver allocated for. On the test host
(kernel 6.8.0-139, a virtio-scsi disk) that is a kernel BUG in `scsi_alloc_sgtables`, and the
machine locks up: it happened once, in an expert-tier run reading into its 1 GiB RAM-tier chunks.
Kernels of that generation also count a destination inside the buffer's first page, past its
start, as that page's length in vectors. So the engine uses `READ_FIXED` only when fewer than
65,536 pages remain from the destination's page to the buffer's end and the destination is not
inside the first page (unless it is the buffer's start); any other read into a registered buffer
is a plain `READ` into the same memory, which the kernel maps page by page
(`exl3_disk::fixed_read_ok`; `disk_stats()["fixed_plain"]` counts them). In a 1 GiB buffer that
is roughly its first three quarters; since `REGISTER=none` measured within noise, the cost is
none that the sweeps can see.

### `EXL3_DISK_URING_ASYNC` (default: `0`)

`io_uring` only: tickets with at least this many page-cache reads go to the kernel's worker
threads (`IOSQE_ASYNC`) instead of being tried inline. Inline, a warm 8192-row gather is copied by
one thread (5-7 ms on the test host, against 2.3 ms across the `pread` pool); through the workers
it spreads over CPUs at a wake-up per read. `0` never. Independently of this knob, buffered reads
of classes 1-3 larger than 64 KiB always go to the workers. The `uring_async` sweep measures 64
on warm prefill-sized gathers: on the test host's smoke run it made a warm 8192-row gather 3.4x
slower (15.5 ms against 4.5 ms), the per-read hand-offs costing more than the single-threaded copy
they spread; leave it at `0` unless your own sweep says otherwise.

### `EXL3_DISK_IOWQ_WORKERS` (default: `0`)

`io_uring` only: upper bound on the kernel's bounded worker threads for reads it must hand off
(`IORING_REGISTER_IOWQ_MAX_WORKERS`); `0` keeps the kernel default (which scales with the ring
size and the CPU count). Bound it when `EXL3_DISK_URING_ASYNC` or buffered bulk would
otherwise spread over every CPU.

### `EXL3_DISK_AFFINITY` (default: unset)

CPU list (`0-3,8`) for the engine's threads: pool workers, the reaper, the poller, and the
kernel's `io_uring` workers. Keeps I/O off the CPUs the CPU MoE workers use; on a NUMA machine,
pick CPUs near the NVMe device (`/sys/class/nvme/nvme0/device/numa_node`). The kernel keeps
`io_uring` workers per submitting thread and applies the mask to workers it creates afterwards,
so every thread that submits (the reaper, and each caller thread that submits its own decode
rows) sets it before its first submission (on kernels before 5.18: right after it, so the
workers of that first submission may run elsewhere until they exit). With `EXL3_DISK_SQPOLL` the
mask applies to the polling thread's workers; pin the polling thread itself with
`EXL3_DISK_SQPOLL_CPU`.

### `EXL3_DISK_FADVISE` (default: `auto`)

Page-cache advice the engine applies to its own buffered descriptor of each file (the caller's
descriptor is never touched): `auto` (`RANDOM` for row tables: a scattered 256-byte row then reads
exactly its page, no readahead; nothing for extents), `none`, `random`, `normal` or `sequential`
(applied to rows and extents alike). `normal` makes the kernel read ahead around every row
(128 KiB by default): more device bytes per row, sometimes a free hit for the next token's
neighbour; the benchmark's read-amplification column shows the cost.

### `EXL3_DISK_EXTENT_DONTNEED` (default: `1`)

After an extent is read through the page cache (the `pread` backend, `EXL3_DISK_DIRECT=none`, or
a file that refused `O_DIRECT`), drop its pages (`FADV_DONTNEED`), so expert bytes do not evict
engram rows. The invalidation runs outside the engine's scheduling lock. `0` keeps them (a
second read of the same extent is then a page-cache copy).

### `EXL3_DISK_KEEPALIVE_MS` (default: `auto`), `EXL3_DISK_KEEPALIVE_IDLE_S` (default: `30`)

Some devices drop into a low-power state when they sit idle, and the next read pays for the
wake-up. On the test host (below) a 4 KiB read after 10-100 ms of idle took ~0.65-0.8 ms instead
of ~0.12 ms, and after 150 ms or more ~13.8 ms; decode leaves the disk idle for about that long
between the engram reads of consecutive layers and tokens. With `KEEPALIVE_MS` = N (`1`-`1000`),
while the engine is in use a thread reads one aligned block (512 B - 64 KiB, the file's `O_DIRECT`
alignment, from the start of the most recently read file) whenever the engine has submitted and
completed nothing for N ms, so the device never idles long enough to fall asleep. On the test host
a read every 2-8 ms brought reads after 20-400 ms of idle back to 0.12-0.2 ms, and in the smoke
run ([below](#measured-on-the-test-host)) a 48-row decode call after 20 ms of idle took 0.40 ms
at p50 with a 5 ms keep-alive against 1.0 ms without (`io_uring`; after 200 ms: 0.58 ms against
14 ms). It stops `KEEPALIVE_IDLE_S` (`1`-`86400`) seconds after the engine's last submission,
so an idle server lets the device sleep, and starts again with the next read (which pays the
wake-up once).

The keep-alive reads bypass the scheduler (one small read per period cannot crowd anything), skip
while the engine has reads in flight or queued, use `O_DIRECT` (a cached block would not reach the
device; files that refuse `O_DIRECT` get none), hold a reference to their file only for the read,
and never keep a forgotten or deleted file open. Their count is `keepalive_reads` in
`disk_stats()`. Cost: at 5 ms, 200 reads of 4 KiB per second (0.8 MB/s) and a few microseconds of
CPU per read while the model is in use, and the device's idle power saving while it is. `auto`
takes `kAutoKeepaliveMs` from `disk_auto.h` (decided by the benchmark, `0` until then); `0` turns
it off. The better fix, where you control the host, is the device's power management (next
section); the keep-alive is for hosts where you do not.

### `EXL3_DISK_VERBOSE` (default: `0`)

Print the resolved configuration once when the engine is created. Fallbacks and refusals are
printed whatever this says.

### `EXL3_DISK_TRACE` (default: `0`)

Keep the last N reads (ticket, class, result, offset, length, submit / issue / done times) for
`disk_trace()`; `0` off. For tests and scheduling analysis.

### Test-only knobs

`EXL3_DISK_FAULT_EIO`, `EXL3_DISK_FAULT_SHORT`, `EXL3_DISK_FAULT_EINTR` (one read attempt in N,
a reproducible pseudo-random draw, fails with `EIO`, returns short, or fails with `EINTR`),
`EXL3_DISK_FAULT_DELAY_US` (every pool read sleeps first), `EXL3_DISK_FAULT_NO_URING`
(`io_uring` setup is treated as refused), `EXL3_DISK_FAULT_ENTER_FATAL` (the Nth submitting
`io_uring_enter` fails for good with `EBADFD`), `EXL3_DISK_FAULT_FORCE_IOPOLL` (every device
counts as able to poll). All default to `0`. Short reads and `EINTR` must be resumed byte-exact,
`EIO` must fail the ticket and nothing else, a refused `io_uring` must leave a working thread
pool behind, and a ring that fails must fail every queued read, refuse new ones and leave no
thread spinning.

## Virtual machines and bare metal

What the engine can reach depends on what the machine gives it. The test host is a Proxmox
guest. Its storage path, confirmed on the Proxmox host:

**Samsung 9100 PRO (PCIe 5 NVMe) -> LVM-thin volume -> QEMU `host_device` block backend
(`aio=io_uring`, `cache=none`) -> `virtio-scsi-pci` controller, no iothread, no passthrough.**

The guest sees `/dev/sda`, a "QEMU HARDDISK" with 24 blk-mq hardware queues, scheduler `none`,
`nr_requests` 256, 512-byte logical blocks and `io_poll` 0: no NVMe device, so no `IOPOLL`, no GPU
Direct Storage, no SPDK. Without an iothread every request of that disk is handled on QEMU's main
loop, one host thread shared with the rest of the device emulation, which bounds the IOPS the
guest can reach and adds jitter under load. Every number in this document was measured through
that path.

**Settings on the Proxmox host** (outside this code; each needs a VM restart, `qm shutdown`
rather than `qm stop`):

1. **Controller and iothread**: `scsihw: virtio-scsi-single` with `iothread=1` on the disk. Each
   disk then gets its own controller and its own I/O thread instead of the main loop. Guest device
   names do not change. This is the standard, low-risk first step.
2. **`aio`**: with an iothread, A/B `aio=io_uring` (the Proxmox default on recent versions) against
   `aio=native`; both need `cache=none` (or `directsync`) to bypass the host page cache. Avoid
   `aio=threads` for this workload.
3. **`cache=none`**: the host opens the volume with `O_DIRECT`, so guest reads reach the drive and
   benchmark numbers mean what they say. With `writeback`, repeated reads can come from host RAM
   and look faster than the drive (and the host's RAM holds a second copy of the model).
4. **virtio-blk** (`virtio0:` with `iothread=1`) instead of SCSI: fewer layers per request; the
   guest device becomes `/dev/vda` (mount by UUID first).
5. **NVMe passthrough** (`hostpci0: <bdf>`): the guest drives the NVMe itself, with native queues,
   `nvme.poll_queues` for `EXL3_DISK_IOPOLL`, and none of the virtio hops. The controller then
   belongs to that guest alone (other guests on the same drive must move).

6. **Device power management.** The test host's first read after idle (measured with the
   benchmark's `wake` case, 4 KiB `O_DIRECT` reads):

   | Idle before the read | 1-5 ms | 10-100 ms | 150 ms and more |
   |---|---|---|---|
   | p50 | 0.12-0.15 ms | 0.65-0.8 ms | 13.8 ms |

   Two steps like these are what NVMe autonomous power state transitions (APST) and PCIe link
   power management (ASPM L1 substates) look like; from inside the guest the cause cannot be seen.
   On the Proxmox host, `nvme id-ctrl /dev/nvmeN` lists the drive's power states and their exit
   latencies, `nvme get-feature /dev/nvmeN -f 0x0c -H` shows the APST table in use, and `lspci -vv`
   shows ASPM on the drive's link. The host kernel parameter `nvme_core.default_ps_max_latency_us`
   caps which states APST may use (`0` disables APST; a value below the deepest state's exit
   latency keeps only the shallow ones), and `pcie_aspm.policy=performance` (or `pcie_aspm=off`)
   keeps the link awake; both need a host reboot and cost idle power. With NVMe passthrough the
   guest's own kernel sets these. Where the host cannot be changed, `EXL3_DISK_KEEPALIVE_MS` keeps
   the device awake from inside the guest while the model is in use.

**Inside the guest, or on bare metal:** leave the scheduler at `none` for NVMe and virtio; do not
change `read_ahead_kb` (the engine advises its own descriptors); give the engine CPUs that the CPU
MoE workers do not use (`EXL3_DISK_AFFINITY`); on bare metal with NVMe, boot with
`nvme.poll_queues=N` if you want to try `EXL3_DISK_IOPOLL`.

**Starting points** (to measure with the benchmark, not measured results):

| Knob | Proxmox guest, virtio, no iothread (test host) | Proxmox guest with iothread | NVMe passthrough or bare metal |
|---|---|---|---|
| `EXL3_DISK_BACKEND` | `auto` (decided here) | re-run `decide` | re-run `decide`; `io_uring` likely |
| `EXL3_DISK_QD` | 128 | 128 | 128-256 |
| `EXL3_DISK_THREADS` (pools) | 24 | 24-48 | 48-96 |
| `EXL3_DISK_SQPOLL` | off | off | try on, with `SQPOLL_CPU` on a spare core |
| `EXL3_DISK_IOPOLL` | refused (no poll queues) | refused | try on, with `nvme.poll_queues` |
| `EXL3_DISK_REGISTER` | `all` | `all` | `all` |
| `EXL3_DISK_WINDOW_EXPERT` | 8M (measured best: see the window run) | measure 8M-32M | measure 8M-64M |
| `EXL3_DISK_AFFINITY` | CPUs outside the CPU MoE set | same | CPUs on the drive's NUMA node |
| `EXL3_DISK_KEEPALIVE_MS` | `auto` (decided here; 5 if the host's power states stay) | same | 0 if APST / ASPM are off, else measure |

## Benchmarking your own machine

Everything below runs without a GPU (`CUDA_VISIBLE_DEVICES=` empty) and needs g++ (C++17);
the Python-share runs also need torch, numpy and a CUDA toolkit to build the test extension (or
the full `exllamav3_ext`).

**Tools** (`tests/disk_engine/`):

- `disk_engine_bench` (built by `build.sh`): the native driver. One process per engine
  configuration runs a list of cases and writes one JSON object per case. Nothing but the engine
  call is timed.
- `bench_engine.py`: the same patterns through the extension, as the model calls it, with the
  Python share of every call and the original gather for comparison.
- `bench_suite.py`: runs both over presets (`smoke`, `window`), enforces a read budget, records
  the environment, writes `report.md`, compares two machines (`compare`) and applies the `auto`
  rule (`decide`).
- `maintenance_window.sh`: the full run on a model's own files, with its preconditions, and
  `--apply` to write the decision into `disk_auto.h`.

**Cache state without touching anyone else's.** A cold case drops exactly the pages it is about
to read (`posix_fadvise(DONTNEED)` on those ranges, rounded out to whole pages), a warm case reads
them first, a fraction `F` makes that share of them resident; `cachestat(2)` (Linux 6.5+) checks
the state before every call, and the report shows the resident share achieved. No global
`drop_caches`, ever. On files the benchmark did not create, every call's pages are put back
afterwards (`--preserve-cache`, the default there): pages that were resident are read again, the
others dropped, so the run leaves the page cache as it found it.

**A quick run on a scratch file** (up to ~10 GB of reads of a 1 GiB file, a minute or two; put
the scratch file on the disk you care about):

```sh
tests/disk_engine/build.sh --build-dir /tmp/de-build --modes release --no-run
python3 tests/disk_engine/bench_suite.py plan --preset smoke
python3 tests/disk_engine/bench_suite.py run --preset smoke \
    --scratch /path/on/the/model/disk --scratch-mb 1024 --budget-gb 10 \
    --driver /tmp/de-build/disk_engine_bench --out results-smoke --python off \
    --env-label "bare metal, 9100 PRO on CPU lanes"
```

(`--python mini` adds the Python share; it builds the test extension with torch, which needs
`CUDA_HOME` and `TORCH_CUDA_ARCH_LIST`.)

**The full run on a model's files** (tens of GB of reads; stop anything that uses the disk first):

```sh
tests/disk_engine/maintenance_window.sh --model /models/DeepSeek-V4.1-Flash-EXL3 \
    --service my-inference.service --env-label "bare metal, 9100 PRO" --out results-full
python3 tests/disk_engine/bench_suite.py decide results-full --src .
```

`decide` prints what `auto` would be on your machine; `--apply .` writes it into your tree's
`disk_auto.h` (or just set `EXL3_DISK_BACKEND` to the winner). **Compare with the test host:**

```sh
python3 tests/disk_engine/bench_suite.py compare results-testhost results-full --md compare.md
```

**Reading the report.** Latency is per call (one engine call: one layer's rows, or k extents),
nearest-rank percentiles; p99.9 needs about 1000 calls to mean anything. *IOPS* = device reads
per second of call time; *Op p50 / p99* = one device read's service time from the engine's own
histogram (12.5 % buckets), the number to compare with `fio`'s completion latency; *Read amp.* =
bytes the process read from the device (`/proc/self/io`) over payload bytes (4096 / 264 ~ 15.5
for buffered rows, less with `O_DIRECT` and 512-byte alignment); *CPU cores* = process CPU time
during the calls (all threads, the kernel's `io_uring` workers and the SQPOLL thread included)
over call time; *Other cores* = CPUs busy before the case, measured with the process idle (a run
with other load is flagged invalid for the decision); *Py us* / *Py %* = the Python side of the
same call and its share of the call. Mixed cases pair each bulk kind with a run without bulk
(`vs alone`). Every case starts after 200 ms of idle (the other-cores sample) and one untimed
wake-up read; the report gives that read's latency (a device that sleeps when idle shows there),
and the `keepalive` block measures decode reads after 20 ms and 200 ms of idle with and without
the keep-alive, and the first-read latency after 1-500 ms of idle.

**Adding a case.** The native driver takes `--case KIND:key=value,...` (`rows`, `extents`,
`mixed`, `null`, `wake`; the header of `disk_engine_bench.cpp` lists every key) and `--set knob=value`
for any `EXL3_DISK_*` knob, so a one-off question needs no code:

```sh
/tmp/de-build/disk_engine_bench --out r.jsonl --set backend=io_uring --set qd=64 \
    --layer /models/m/model-00047-of-00048.safetensors:664:256:384006168+/models/m/model-00047-of-00048.safetensors:98305579672:8:384006168 \
    --case rows:u=48,cache=cold,iters=1000 --case rows:u=48,cache=0.75,iters=1000
```

## Measured on the test host

Smoke numbers, not the decision: a 1 GiB scratch file on the test host (above) while the model
server was running on the same disk and other jobs shared the host, so absolute values carry that
load. The maintenance-window run on the model's own files, with the service stopped, decides
`auto` and replaces this section.

`bench_suite.py run --preset smoke` on 2026-09-25: 9.3 GB read in 70 s, every block of the suite
run, 104 native cases and 40 Python cases. p50 / p99 per call unless marked; one call reads one
engram layer (both tables, 2 x u reads) or k extents. The `decision` rows are the median of 2 runs;
everything else is one run of 2-60 calls, so treat single-run differences under ~10 % as noise.

| | `pread` | `odirect` | `io_uring` |
|---|---|---|---|
| 48 rows, cold | 666 / 1233 us | 610 / 1338 us | 364 / 611 us |
| 48 rows, 75 % resident (the decode-critical case) | 423 / 605 us | 759 / 1038 us | **233 / 337 us** |
| 48 rows, warm | 29.6 / 46.7 us | 721 / 893 us | 34.7 / 80.0 us |
| 96 rows, cold | 1174 / 1542 us | 1034 / 1333 us | 694 / 976 us |
| 8192 rows, cold | 60.2 / 66.2 ms | 74.8 / 85.0 ms | 47.8 / 49.7 ms |
| 8192 rows, warm | 2.55 / 2.64 ms | 75.5 / 78.2 ms | 4.53 / 5.49 ms |
| extents, 1 in flight (13.3 MB each) | 4.9 GB/s | 9.2 GB/s | 8.8 GB/s |
| extents, 4 in flight | 5.4 GB/s | 11.4 GB/s | 10.3 GB/s |
| 48 cold rows every 2 ms, alone | 681 / 798 us | 582 / 956 us | 370 / 570 us |
| ... next to a class-1 (demand) extent stream | 1348 / 2884 us, 3.6 GB/s bulk | 1500 / 1670 us, 10.0 GB/s | 2938 / 3308 us, 9.7 GB/s |
| ... next to a class-2 (prefetch) extent stream | 1335 / 5818 us, 3.3 GB/s | 898 / 1083 us, 9.0 GB/s | 463 / 544 us, 9.2 GB/s |
| ... next to 8192-row class-2 gathers | 909 / 1038 us | 1064 / 1209 us | 445 / 729 us |
| 48 rows (75 %) after 20 ms idle, keep-alive off / 5 ms | 970 / 565 us (p50) | 1302 / 965 us | 1001 / 396 us |
| 48 rows (75 %) after 200 ms idle, keep-alive off / 5 ms | 13.9 / 0.63 ms (p50) | 14.5 / 1.05 ms | 14.0 / 0.58 ms |
| Through Python, 48 rows 75 %: wall p50, Python share | 430 us, 6.6 us (1.5 %) | 779 us, 8.6 us (1.1 %) | 262 us, 6.6 us (2.5 %) |
| Through Python, 48 rows warm | 41.8 us, 4.9 us (11.7 %) | 733 us, 7.1 us (1.1 %) | 50.8 us, 6.4 us (13.3 %) |
| Through Python, 8192 rows cold | 70.0 ms, 15.2 us (0.02 %) | 82.3 ms, 11.4 us (0.01 %) | 45.8 ms, 10.8 us (0.02 %) |

- **The original gather** (two `ngram_gather_cpu` calls through its pool, as the engram calls it
  today), p50: 549 us at 75 % resident, 851 us cold, 292 us warm, 76.2 ms for 8192 cold rows.
  `io_uring` through Python did the same work in 262 us, 45.8 ms.
- **Python's share** of a call is 5-9 us for 48 rows and 11-15 us for 8192 (0.7 us for an empty
  call): pybind dispatch and argument conversion going in (3.5-5 us for 48 rows), the return with
  the GIL taken again (1-2.5 us). That is 1-2.5 % of any 48-row call that reaches the device,
  0.02 % of a prefill gather, 0.1 % of an extent call, and 12-13 % of a fully warm 48-row call
  (42-51 us); per decoded token (two engram layers) about 13 us, 0.03 % of a ~42 ms token.
  Keyword arguments instead of positional ones add 2-3 us. Nothing in the hot path runs in
  Python.
- **Idle wake-up.** One cold 4 KiB read after the disk sat idle: 0.14-0.15 ms after 1 ms, 0.55-0.65
  ms after 10 ms, 0.77-0.82 ms after 50 ms, and 9-14 ms after 200 ms (every case in the run
  started with a 200 ms idle sample: its first read took 13.8 ms at the median). With a 5 ms
  keep-alive: 0.12-0.26 ms at every gap. Decode leaves the disk idle for 13-30 ms between engram
  reads, which is where the keep-alive rows above sit.
- **Sweeps** (one run each; `io_uring` unless named): `EXL3_DISK_WINDOW_EXPERT=8M` cut decode latency
  next to a class-1 stream from 2.9 ms to 0.70 ms at the same bulk rate (10.0 GB/s);
  `EXL3_DISK_DIRECT=all` made cold 48-row calls 26 % faster (270 us: 512-byte `O_DIRECT` reads
  instead of whole pages) at the cost of the page cache's warm tier; `EXL3_DISK_QD=32` made them
  1.67x slower; `EXL3_DISK_URING_ASYNC=64` made a warm 8192-row gather 3.4x slower (15.5 ms);
  `EXL3_DISK_WINDOW_PREFETCH=32M/0` made decode next to a prefetch stream 6x slower (2.8 ms);
  `SQPOLL`, `REGISTER=none`, `SPIN_US=0`, `CHUNK=512K` and `ALIGN=4096` (`odirect`) stayed within
  noise; 8 `pread` threads made cold rows 1.3x slower.
- **The decision rule on these numbers** picks `io_uring`, the engine route for the n-gram gather,
  and a 5 ms keep-alive, and advises an 8 MiB class-1 window; it refused to apply them (and was
  right to) because the host was busy before three `pread` cases (up to 20.8 other cores). The
  decision belongs to the maintenance-window run.

## Entry points

C++ (`disk/disk_engine.h`): `Engine::submit_rows`, `submit_extents`, `gather_rows`,
`read_extents`, `wait`, `done`, `release`, `cancel`, `promote`, `arm_hold`, `stats`,
`extent_geometry`, `register_buffer`, `find_registered`, `forget`, `forget_path`; `auto_backend()`,
`auto_ngram_engine()`, `auto_keepalive_ms()`, `ngram_engine_from_env()`. Python (`exllamav3_ext`): `disk_gather_rows`,
`disk_read_extents`, `disk_extent_geometry`, `DiskTicket`, `disk_arm_hold`, `disk_stats`,
`disk_engine_info`, `disk_engine_configure`, `disk_engine_forget`, `disk_set_thread_class`,
`disk_register_buffer`, `disk_ngram_route`. `ngram_gather_cpu` keeps its signature; it runs on the
engine when a backend is named, or when `auto` routes it there.

## Tests

- `tests/disk_engine/build.sh`: the standalone C++ test driver (byte-exact checks against a
  reference `pread`, concurrency, priorities, cancellation, shutdown with reads in flight, fault
  injection, the `auto` / `original` resolution, the keep-alive) in release, ASan/UBSan and TSan
  builds, and the benchmark driver in the same three builds.
- `tests/test_disk_engine_.py`: the Python bindings and `ngram_gather_cpu` against the original
  gather, the `auto` choice and `original`, tickets waited for and released from different
  threads, registered buffers and forgotten files.
- `tests/test_dsv41_engram_disk_.py`: the DeepSeek-V4.1 engram's staging (inline, and the
  look-ahead promoted by its consumer, and retired) against the original gather, for every
  backend, `auto` and `original`, and its route taken from the extension.
- `tests/disk_engine/bench_suite.py`, `disk_engine_bench.cpp`, `bench_engine.py`,
  `maintenance_window.sh`: the benchmarks above.
