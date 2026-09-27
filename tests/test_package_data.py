"""
Every license file inside the exllamav3 package must be matched by [tool.setuptools.package-data], or it is
left out of the sdist and wheels. Regression: exllamav3/vendor/fla/LICENSE (the MIT notice the vendored
flash-linear-attention sources point to) was never packaged. Only the "exllamav3" key's patterns are resolved
(recursive glob relative to the package directory, like setuptools' build_py), so no build is needed.
"""
import glob, os
import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "exllamav3")


def packaged_data_files():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        package_data = tomllib.load(f)["tool"]["setuptools"]["package-data"]
    assert set(package_data) == {"exllamav3"}, "extend this test for other package-data keys"
    files = set()
    for pattern in package_data["exllamav3"]:
        for p in glob.glob(pattern, root_dir = PKG, recursive = True):
            p = os.path.normpath(os.path.join(PKG, p))
            if os.path.isfile(p):
                files.add(p)
    return files


def test_license_files_are_packaged():
    licenses = [
        os.path.normpath(os.path.join(d, f))
        for d, _, fs in os.walk(PKG) for f in fs
        if f.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE"))
    ]
    assert os.path.join(PKG, "vendor", "fla", "LICENSE") in licenses
    missing = sorted(os.path.relpath(p, ROOT) for p in set(licenses) - packaged_data_files())
    assert not missing, f"license files not in package-data: {missing}"
