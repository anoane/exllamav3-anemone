#pragma once

// Python side of the expert tier's native code (bindings in tier_bc.h): the policy core's replay
// (tests compare it with exllamav3/model/expert_tier_policy.py) and the RAM tier host
// (tier_host.h). doc/expert_tiers.md.

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

namespace py = pybind11;

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
