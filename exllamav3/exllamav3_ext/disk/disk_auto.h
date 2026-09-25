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
// Evidence: none yet. The maintenance-window benchmark has not run on the test host, so auto
// keeps the behaviour from before the engine existed: ngram_gather_cpu runs its original pool,
// the engine's own entry points use the buffered thread pool, and no keep-alive reads run.

#include "disk_engine.h"

namespace exl3_disk
{

constexpr Backend kAutoBackend = Backend::Pread;
constexpr bool kAutoNgramEngine = false;
constexpr int kAutoKeepaliveMs = 0;

}  // namespace exl3_disk
