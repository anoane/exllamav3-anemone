#pragma once

// What EXL3_DISK_BACKEND=auto (or unset) resolves to.
//
// This file is written whole by `python tests/disk_engine/bench_suite.py decide --apply`
// from a benchmark run on the test host (doc/disk_engine.md, "How auto is decided"). Change it
// by hand only together with the evidence below.
//
//   kAutoBackend      the engine backend for the extension's disk_* entry points
//   kAutoNgramEngine  whether ngram_gather_cpu (PLE n-gram tables, the DeepSeek-V4.1 engram)
//                     uses the engine (true) or keeps its original thread pool (false)
//   kAutoKeepaliveMs  EXL3_DISK_KEEPALIVE_MS=auto (unset): the keep-alive read period, 0 off
//
// Evidence: decided 2026-09-25 from disk (Samsung 9100 PRO -> LVM-thin -> QEMU host_device
// (aio=io_uring, cache=none) -> virtio-scsi-pci, no iothread, no passthrough). Decode rows
// (u=48, two tables, 75% of its pages resident), p50 us: pread 457, odirect 761, io_uring 267.
// N-gram route: io_uring through Python 275 us vs original 577 us at decode (u=48, cache 0.75);
// 8192 cold rows 70.6 ms vs 104.0 ms: engine. Keep-alive: io_uring, decode rows after 20 ms
// idle: 1055 us without, 414 us with a 5 ms keep-alive (after 200 ms idle: 14034 vs 487 us):
// on.

#include "disk_engine.h"

namespace exl3_disk
{

constexpr Backend kAutoBackend = Backend::IoUring;
constexpr bool kAutoNgramEngine = true;
constexpr int kAutoKeepaliveMs = 5;

}  // namespace exl3_disk
