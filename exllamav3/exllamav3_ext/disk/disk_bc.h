// Disk I/O engine bindings; included inside PYBIND11_MODULE (see disk_ext.h, doc/disk_engine.md)

py::class_<DiskTicket>(m, "DiskTicket")
    .def("wait", &DiskTicket::wait, py::arg("timeout") = py::none())
    .def("done", &DiskTicket::done)
    .def("cancel", &DiskTicket::cancel)
    .def("promote", &DiskTicket::promote, py::arg("cls"))
    .def("times", &DiskTicket::times)
    .def("error", &DiskTicket::error)
    .def("release", &DiskTicket::release)
    .def_property_readonly("id", &DiskTicket::id);

m.def("disk_gather_rows", &disk_gather_rows, "disk_gather_rows",
      py::arg("uids"), py::arg("uid_base"), py::arg("fds"), py::arg("offsets"),
      py::arg("row_bytes"), py::arg("outs"), py::arg("cls") = -1, py::arg("hold") = false,
      py::arg("wait") = true, py::arg("flag") = py::none(), py::arg("flag_index") = 0,
      py::arg("flag_value") = 1, py::arg("deadline_ns") = 0);
m.def("disk_read_extents", &disk_read_extents, "disk_read_extents",
      py::arg("fds"), py::arg("offsets"), py::arg("lengths"), py::arg("dst"),
      py::arg("dst_offsets"), py::arg("slot_bytes"), py::arg("payload_offsets") = py::none(),
      py::arg("cls") = 1, py::arg("hold") = false, py::arg("wait") = false,
      py::arg("flag") = py::none(), py::arg("flag_index") = 0, py::arg("flag_value") = 1,
      py::arg("deadline_ns") = 0);
m.def("disk_extent_geometry", &disk_extent_geometry, "disk_extent_geometry",
      py::arg("fd"), py::arg("offset"), py::arg("length"));
m.def("disk_arm_hold", &disk_arm_hold, "disk_arm_hold");
m.def("disk_stats", &disk_stats, "disk_stats", py::arg("reset") = false);
m.def("disk_engine_info", &disk_engine_info, "disk_engine_info");
m.def("disk_engine_configure", &disk_engine_configure, "disk_engine_configure",
      py::arg("overrides"));
m.def("disk_engine_shutdown", &disk_engine_shutdown, "disk_engine_shutdown");
m.def("disk_set_thread_class", &disk_set_thread_class, "disk_set_thread_class", py::arg("cls"));
m.def("disk_ngram_route", &disk_ngram_route, "disk_ngram_route");
m.def("disk_register_buffer", &disk_register_buffer, "disk_register_buffer", py::arg("t"));
m.def("disk_unregister_buffer", &disk_unregister_buffer, "disk_unregister_buffer",
      py::arg("idx"));
m.def("disk_profile", &disk_profile, "disk_profile", py::arg("enable"));
m.def("disk_profile_last", &disk_profile_last, "disk_profile_last");
m.def("disk_trace", &disk_trace, "disk_trace", py::arg("clear") = false);
