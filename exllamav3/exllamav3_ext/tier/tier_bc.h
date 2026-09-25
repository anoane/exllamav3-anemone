// Expert tier bindings; included inside PYBIND11_MODULE (see tier_ext.h, doc/expert_tiers.md)

m.def("tier_policy_replay", &tier_policy_replay, "tier_policy_replay",
      py::arg("config"), py::arg("layers"), py::arg("experts"), py::arg("slots"), py::arg("ram_slots"),
      py::arg("pins"), py::arg("defer_returns"), py::arg("script"));
