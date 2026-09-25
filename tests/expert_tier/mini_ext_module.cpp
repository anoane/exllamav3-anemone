// Test-only extension module: the expert tier's bindings and the disk engine's, built without the
// rest of exllamav3_ext (tests/expert_tier/mini_ext.py). The bindings are the extension's own.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "disk/disk_ext.h"
#include "tier/tier_ext.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    #include "disk/disk_bc.h"
    #include "tier/tier_bc.h"
}
