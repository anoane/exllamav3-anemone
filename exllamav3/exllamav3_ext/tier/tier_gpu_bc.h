// VRAM expert cache bindings; included inside PYBIND11_MODULE (see tier_gpu_ext.h)

py::class_<TierGpuHandle>(m, "TierGpu")
    .def(py::init<const py::dict&, int64_t, int64_t, int64_t, int64_t, int64_t, const py::object&,
                  const std::vector<std::string>&, const py::list&, const py::dict&, const py::dict&>(),
         py::arg("policy"), py::arg("layers"), py::arg("experts"), py::arg("pool_slots"), py::arg("ram_slots"),
         py::arg("pins"), py::arg("geometry"), py::arg("files"), py::arg("extents"), py::arg("host"), py::arg("gpu"))
    .def("cold_fill", &TierGpuHandle::cold_fill, py::arg("order"), py::arg("pins"))
    .def("release_slab", &TierGpuHandle::release_slab)
    .def("start", &TierGpuHandle::start)
    .def("stop", &TierGpuHandle::stop)
    .def("lookup", &TierGpuHandle::lookup, py::arg("lc"), py::arg("ids"), py::arg("mode"), py::arg("seq"),
         py::arg("tokens"))
    .def("step", &TierGpuHandle::step, py::arg("wait") = true)
    .def("drain", &TierGpuHandle::drain)
    .def("verify", &TierGpuHandle::verify)
    .def("records", &TierGpuHandle::records)
    .def("state", &TierGpuHandle::state)
    .def("counters", &TierGpuHandle::counters)
    .def("stats", &TierGpuHandle::stats)
    .def("device_state", &TierGpuHandle::device_state)
    .def("error", &TierGpuHandle::error)
    .def("ram_trellis", &TierGpuHandle::ram_trellis, py::arg("r"))
    .def("place", &TierGpuHandle::place, py::arg("key"), py::arg("slot"), py::arg("stamp"))
    .def("ram_add", &TierGpuHandle::ram_add, py::arg("key"), py::arg("dup"), py::arg("slot"), py::arg("stamp"))
    .def("heat_set", &TierGpuHandle::heat_set, py::arg("key"), py::arg("value"))
    .def("set_seq", &TierGpuHandle::set_seq, py::arg("seq"))
    .def("upload", &TierGpuHandle::upload);
