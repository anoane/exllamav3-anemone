#pragma once

// Python side of the disk engine (bindings in disk_bc.h). One call per table set or extent
// list; the GIL is released while the engine works. See doc/disk_engine.md.

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "disk_engine.h"

namespace py = pybind11;

// An asynchronous submission. Keeps its destination tensors (and flag tensor) alive; releasing
// it (explicitly or when it is garbage collected) cancels what is still queued and waits for
// the reads in flight, so no read lands after the tensors are dropped.
class DiskTicket
{
public:
    DiskTicket(std::shared_ptr<exl3_disk::Engine> eng, uint64_t id, std::vector<at::Tensor> keep);
    ~DiskTicket();
    DiskTicket(const DiskTicket&) = delete;
    DiskTicket& operator=(const DiskTicket&) = delete;

    void wait(py::object timeout);      // raises RuntimeError on a failed read, TimeoutError
    bool done();
    void cancel();
    void promote(int64_t cls);
    py::dict times();
    int64_t error();                    // 0 or -errno once done
    void release();
    uint64_t id() const { return id_; }

private:
    void check() const;
    std::shared_ptr<exl3_disk::Engine> eng_;
    uint64_t id_;
    std::vector<at::Tensor> keep_;
    bool released_ = false;
};

std::unique_ptr<DiskTicket> disk_gather_rows
(
    const at::Tensor& uids,             // (U) int64 CPU contiguous; need not be sorted
    int64_t uid_base,
    const std::vector<int64_t>& fds,    // per table
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& row_bytes,
    const std::vector<at::Tensor>& outs,
    int64_t cls,                        // -1: this thread's class (disk_set_thread_class)
    bool hold,
    bool wait,
    const c10::optional<at::Tensor>& flag,
    int64_t flag_index,
    int64_t flag_value,
    int64_t deadline_ns
);

std::unique_ptr<DiskTicket> disk_read_extents
(
    const at::Tensor& fds,              // (N) int64
    const at::Tensor& offsets,          // (N) int64, payload file offsets
    const at::Tensor& lengths,          // (N) int64
    const at::Tensor& dst,              // CPU contiguous; slot i starts at dst + dst_offsets[i]
    const at::Tensor& dst_offsets,      // (N) int64 byte offsets, 4 KiB aligned slots
    int64_t slot_bytes,
    const c10::optional<at::Tensor>& payload_offsets,   // (N) int64 out, or None
    int64_t cls,
    bool hold,
    bool wait,
    const c10::optional<at::Tensor>& flag,
    int64_t flag_index,
    int64_t flag_value,
    int64_t deadline_ns
);

std::tuple<int64_t, int64_t> disk_extent_geometry(int64_t fd, int64_t offset, int64_t length);
void disk_arm_hold();
py::dict disk_stats(bool reset);
py::dict disk_engine_info();
py::dict disk_engine_configure(const py::dict& overrides);
void disk_engine_shutdown();
int64_t disk_set_thread_class(int64_t cls);
std::string disk_ngram_route();
int64_t disk_register_buffer(const at::Tensor& t);
void disk_unregister_buffer(int64_t idx);
void disk_profile(bool enable);
std::tuple<int64_t, int64_t, int64_t, int64_t> disk_profile_last();
at::Tensor disk_trace(bool clear);
