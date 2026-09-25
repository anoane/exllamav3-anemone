// Expert tier bindings; included inside PYBIND11_MODULE (see tier_ext.h, doc/expert_tiers.md)

m.def("tier_policy_replay", &tier_policy_replay, "tier_policy_replay",
      py::arg("config"), py::arg("layers"), py::arg("experts"), py::arg("slots"), py::arg("ram_slots"),
      py::arg("pins"), py::arg("defer_returns"), py::arg("script"));

py::class_<TierHostHandle>(m, "TierHost")
    .def(py::init<const py::dict&, int64_t, int64_t, int64_t, int64_t, int64_t, const py::dict&,
                  const std::vector<std::string>&, const py::list&, const py::dict&, bool>(),
         py::arg("policy"), py::arg("layers"), py::arg("experts"), py::arg("pool_slots"), py::arg("ram_slots"),
         py::arg("pins"), py::arg("geometry"), py::arg("files"), py::arg("extents"), py::arg("host"),
         py::arg("defer_returns") = false)
    .def("cold_fill", &TierHostHandle::cold_fill, py::arg("order"))
    .def("call", &TierHostHandle::call, py::arg("lc"), py::arg("ids"), py::arg("mode") = 0)
    .def("process", &TierHostHandle::process, py::arg("record"))
    .def("layer", &TierHostHandle::layer, py::arg("lc"), py::arg("ids"), py::arg("half") = 0)
    .def("prefetch", &TierHostHandle::prefetch, py::arg("lc"), py::arg("deadline_ns") = 0)
    .def("promote", &TierHostHandle::promote, py::arg("lc"))
    .def("tick", &TierHostHandle::tick, py::arg("tokens") = 1)
    .def("complete", &TierHostHandle::complete)
    .def("drain", &TierHostHandle::drain)
    .def("poll", &TierHostHandle::poll)
    .def("vram_slot", &TierHostHandle::vram_slot, py::arg("slot"))
    .def("staging", &TierHostHandle::staging, py::arg("half"), py::arg("i"))
    .def("ram_trellis", &TierHostHandle::ram_trellis, py::arg("r"))
    .def("ram_slot", &TierHostHandle::ram_slot, py::arg("r"))
    .def("ram_offset", &TierHostHandle::ram_offset, py::arg("r"), py::arg("p"))
    .def("last_actions", &TierHostHandle::last_actions)
    .def("state", &TierHostHandle::state)
    .def("counters", &TierHostHandle::counters)
    .def("stats", &TierHostHandle::stats)
    .def("arena", &TierHostHandle::arena)
    .def("error", &TierHostHandle::error);
