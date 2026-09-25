#pragma once

// Conversions between the expert tier's native types and Python objects, shared by the policy /
// RAM tier bindings (tier_ext.cpp) and the GPU runtime's (tier_gpu_ext.cpp)

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>

#include <initializer_list>
#include <vector>

#include "tier_host.h"
#include "tier_policy.h"

namespace py = pybind11;

namespace tier_py
{

template <typename T>
T value(const py::dict& d, const char* key, T dflt)
{
    if (!d.contains(key) || d[key].is_none()) return dflt;
    return d[key].cast<T>();
}

int choice(const py::dict& d, const char* key, std::initializer_list<const char*> names, int dflt);
exl3_tier::PolicyConfig config_of(const py::dict& d);
exl3_tier::HostConfig host_config_of(const py::dict& d);
std::vector<exl3_tier::ExpertExtent> extents_of(const py::list& l);
exl3_tier::Record record_of(const py::tuple& t);
py::tuple action_tuple(const exl3_tier::Action& a);
py::tuple entry_tuple(const exl3_tier::Entry& x);
py::dict core_state(const exl3_tier::TierCore& core);
py::dict core_counters(const exl3_tier::TierCore& core);
at::Tensor bytes_tensor(const uint8_t* p, int64_t n);

}  // namespace tier_py
