#pragma once

// Python side of the expert tier's native code (bindings in tier_bc.h): the policy core's replay
// (tests compare it with exllamav3/model/expert_tier_policy.py) and the RAM tier host
// (tier_host.h). doc/expert_tiers.md.

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace exl3_tier { class TierHost; class HostCopier; }

// Run a script of operations on a fresh policy core; returns the records, the actions of every
// operation, the counters and the final state (see tests/test_expert_tier_native_.py)
py::dict tier_policy_replay
(
    const py::dict& config,             // PolicyConfig fields by name (expert_tier_policy.PolicyConfig)
    int64_t layers,
    int64_t experts,
    int64_t slots,
    int64_t ram_slots,
    int64_t pins,
    bool defer_returns,
    const py::list& script
);

// The RAM tier host over the checkpoint's shards, with a host buffer standing in for the GPU's pool
// and staging (tier_host.h: TierHost + HostCopier). The GIL is released while it reads and copies
class TierHostHandle
{
public:
    TierHostHandle
    (
        const py::dict& policy,         // PolicyConfig fields by name
        int64_t layers,
        int64_t experts,
        int64_t pool_slots,
        int64_t ram_slots,
        int64_t pins,
        const py::dict& geometry,       // projections, proj_bytes [..], ram_slot_bytes
        const std::vector<std::string>& files,
        const py::list& extents,        // (name, [(file, offset, length)..], [(piece, offset in piece)..]) per key
        const py::dict& host,           // chunk_bytes, hugepage, slab_slots, fill_inflight, staging_slots, deterministic
        bool defer_returns
    );
    ~TierHostHandle();

    void cold_fill(const std::vector<int32_t>& order);
    py::list call(int64_t lc, const std::vector<int32_t>& ids, int64_t mode);
    void process(const py::tuple& record);
    void layer(int64_t lc, const std::vector<int32_t>& ids, int64_t half);
    void prefetch(int64_t lc, int64_t deadline_ns);
    void promote(int64_t lc);
    void tick(int64_t tokens);
    void complete();
    void drain();
    void poll();
    at::Tensor vram_slot(int64_t slot);
    at::Tensor staging(int64_t half, int64_t i);
    at::Tensor ram_trellis(int64_t r);
    at::Tensor ram_slot(int64_t r);
    int64_t ram_offset(int64_t r, int64_t p);
    py::list last_actions();
    py::dict state();
    py::dict counters();
    py::dict stats();
    py::dict arena();
    std::string error();

private:
    std::unique_ptr<exl3_tier::HostCopier> copier_;
    std::unique_ptr<exl3_tier::TierHost> host_;
};
