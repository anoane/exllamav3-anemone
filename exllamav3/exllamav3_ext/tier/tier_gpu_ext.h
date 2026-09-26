#pragma once

// Python side of the VRAM expert cache (tier_gpu.h); bindings in tier_gpu_bc.h, used by
// exllamav3/model/expert_tier.py. doc/expert_tiers.md, "The VRAM cache".

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace exl3_tier { class TierGpu; }

class TierGpuHandle
{
public:
    TierGpuHandle
    (
        const py::dict& policy,         // PolicyConfig fields by name
        int64_t layers,
        int64_t experts,
        int64_t pool_slots,
        int64_t ram_slots,
        int64_t pins,
        const py::object& geometry,     // proj_bytes [..], ram_slot_bytes; None with no extents (policy only)
        const std::vector<std::string>& files,
        const py::list& extents,        // as TierHost's; empty: no bytes move (the kernel's tests)
        const py::dict& host,           // TierHost settings (chunk_bytes, hugepage, slab_slots, ...)
        const py::dict& gpu             // device, pool, staging, staging_per_half, tables, proj_off,
                                        // deterministic, spin_us, affinity, rec_ring_bytes, trace
    );
    ~TierGpuHandle();

    void cold_fill(const std::vector<int32_t>& order, const std::vector<int32_t>& pins);
    void release_slab();
    void start();
    void stop();
    void lookup(int64_t lc, const at::Tensor& ids, int64_t mode, int64_t seq, int64_t tokens);
    // layer mode (tier_gpu.h): the caller's current stream is the compute stream
    void layer_begin();
    void layer_plan(int64_t lc, int64_t half);
    void layer_run(int64_t lc, const at::Tensor& ids, int64_t seq, int64_t tokens);
    void layer_after(int64_t lc);
    void predict(int64_t lc, const at::Tensor& ids, int64_t seq, int64_t tokens);
    std::vector<int64_t> layer_addrs(int64_t lc, int64_t e);
    int64_t step(bool wait);
    void drain();
    void verify();
    py::list records();
    py::dict state();
    py::dict counters();
    py::dict stats();
    py::dict device_state();
    std::string error();
    at::Tensor ram_trellis(int64_t r);
    // seeding (tests): the mirror, then upload() puts it on the device
    void place(int64_t key, int64_t slot, int64_t stamp);
    void ram_add(int64_t key, bool dup, int64_t slot, int64_t stamp);
    void heat_set(int64_t key, int64_t value);
    void set_seq(int64_t seq);
    void upload();

private:
    std::unique_ptr<exl3_tier::TierGpu> gpu_;
    int device_ = 0;
};
