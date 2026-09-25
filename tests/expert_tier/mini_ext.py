"""
Build (or load) a small torch extension holding the expert tier's native code and the disk engine
bindings, for testing where the full exllamav3_ext is not built. The sources are the extension's
own (exllamav3_ext/tier, exllamav3_ext/disk); only the module definition differs
(mini_ext_module.cpp). C++ only: no CUDA compiler needed.

    from tests.expert_tier.mini_ext import load_ext
    ext = load_ext()                       # EXL3_TIER_MINI_BUILD_DIR sets the build directory
"""

import glob
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
    sources += sorted(glob.glob(os.path.join(src, "tier", "*.cpp")))
    sources += [os.path.join(here, "mini_ext_module.cpp")]
    build_dir = build_dir or os.environ.get("EXL3_TIER_MINI_BUILD_DIR")
    if build_dir:
        os.makedirs(build_dir, exist_ok = True)
    _ext = load(
        name = "exl3_tier_mini",
        sources = sources,
        extra_include_paths = [src],
        extra_cflags = ["-O2", "-Wall", "-Wextra"],
        build_directory = build_dir,
        verbose = verbose,
    )
    return _ext


def load_any():
    """The mini extension with EXL3_TIER_TEST_MINI=1, else the full one"""
    if os.environ.get("EXL3_TIER_TEST_MINI") == "1":
        return load_ext()
    from exllamav3.ext import exllamav3_ext
    return exllamav3_ext
