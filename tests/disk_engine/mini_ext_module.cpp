// Test-only extension module: the disk engine bindings and ngram_gather_cpu, built without the
// rest of exllamav3_ext (tests/disk_engine/mini_ext.py). The bindings are the extension's own
// (disk/disk_bc.h), so the Python-facing behaviour and cost are the same.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngram.cuh"
#include "disk/disk_ext.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ngram_gather_cpu", &ngram_gather_cpu, "ngram_gather_cpu");
    #include "disk/disk_bc.h"
}
