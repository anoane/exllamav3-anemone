"""
Build (or load) a small torch extension holding the disk engine bindings and ngram_gather_cpu,
for testing the Python side where the full exllamav3_ext is not built. The sources are the
extension's own; only the module definition differs (mini_ext_module.cpp).

    from tests.disk_engine.mini_ext import load_ext
    ext = load_ext()                       # EXL3_DISK_MINI_BUILD_DIR sets the build directory
"""

import os

_ext = None


def load_ext(build_dir: str | None = None, verbose: bool = False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "..", "exllamav3", "exllamav3_ext"))
    sources = [os.path.join(src, "disk", f)
               for f in ("disk_config.cpp", "disk_engine.cpp", "disk_uring.cpp", "disk_ext.cpp")]
    sources += [os.path.join(src, "ngram.cu"), os.path.join(here, "mini_ext_module.cpp")]
    build_dir = build_dir or os.environ.get("EXL3_DISK_MINI_BUILD_DIR")
    if build_dir:
        os.makedirs(build_dir, exist_ok = True)
    _ext = load(
        name = "exl3_disk_mini",
        sources = sources,
        extra_include_paths = [src],
        extra_cflags = ["-O3", "-Wall", "-Wextra"],
        extra_cuda_cflags = ["-O3", "-lineinfo"],
        build_directory = build_dir,
        verbose = verbose,
    )
    return _ext
