"""
check_host_memory (the guard before the CPU MoE arena chunks and the --ngram_ram table) must honour
the process's memory cgroup (v2), not only the host-wide MemAvailable. Regression: under a container
memory limit or a systemd MemoryMax scope the guard passed on MemAvailable, and the allocation then
pushed the cgroup past memory.max, where the OOM killer sent SIGKILL instead of the guard's error.
Runs against fake /proc and /sys/fs/cgroup trees; no GPU, no root, no real cgroup needed.
"""
import builtins, importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "exl3_util_memory", Path(__file__).resolve().parents[1] / "exllamav3/util/memory.py")
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)

MiB, GiB = 1 << 20, 1 << 30


def _tree(tmp_path, self_cgroup = "0::/a/b\n", **levels):
    """Fake /proc/self/cgroup and a cgroup root with levels root, a and a_b (= a/b), each given as
    (memory.max, memory.high, memory.current, active_file, inactive_file); returns the paths to pass
    to cgroup_memory_available"""
    (tmp_path / "self_cgroup").write_text(self_cgroup)
    for name, rel in (("root", ""), ("a", "a"), ("a_b", "a/b")):
        d = tmp_path / "cg" / rel
        d.mkdir(parents = True, exist_ok = True)
        limit, high, current, active_file, inactive_file = levels.get(name, (None, "max", 0, 0, 0))
        if limit is not None:
            (d / "memory.max").write_text(f"{limit}\n")
            (d / "memory.high").write_text(f"{high}\n")
        (d / "memory.current").write_text(f"{current}\n")
        (d / "memory.stat").write_text(f"anon {current}\ninactive_file {inactive_file}\nactive_file {active_file}\n")
    return str(tmp_path / "self_cgroup"), str(tmp_path / "cg")


@pytest.mark.parametrize("self_cgroup, levels, expected", [
    ("0::/a/b\n", dict(a = (8 * GiB, "max", 7 * GiB, 0, 0), a_b = (4 * GiB, "max", GiB, 0, 0)), GiB),
    ("0::/a/b\n", dict(a_b = ("max", 3 * GiB, GiB, 0, 0)), 2 * GiB),
    ("0::/a/b\n", dict(a_b = (4 * GiB, 3 * GiB, GiB, 0, 0)), 2 * GiB),
    ("0::/a/b\n", dict(a_b = (4 * GiB, "max", 3 * GiB, 256 * MiB, 768 * MiB)), 2 * GiB),
    ("0::/a/b\n", dict(a_b = (GiB, "max", 2 * GiB, 0, 0)), 0),
    ("0::/\n", dict(root = (2 * GiB, "max", 512 * MiB, 0, 0)), 1536 * MiB),
    ("0::/a/b\n", dict(), None),
    ("0::/a/b\n", dict(a_b = ("garbage", "max", 0, 0, 0)), None),
    ("12:memory:/a/b\n4:cpu,cpuacct:/a/b\n", dict(a_b = (GiB, "max", 0, 0, 0)), None),
    ("", dict(a_b = (GiB, "max", 0, 0, 0)), None),
    ("0::a/b\n", dict(a_b = (GiB, "max", 0, 0, 0)), None),
    ("0::/../../x\n", dict(a_b = (GiB, "max", 0, 0, 0)), None),
], ids = ["tightest_level", "max_is_unlimited", "min_max_high", "file_lru_is_room", "clamp_at_zero",
          "namespace_root", "no_limit", "garbage", "v1_only", "empty", "relative", "outside_namespace"])
def test_cgroup_memory_available(tmp_path, self_cgroup, levels, expected):
    proc, root = _tree(tmp_path, self_cgroup, **levels)
    assert memory.cgroup_memory_available(proc, root) == expected
    assert memory.cgroup_memory_available(str(tmp_path / "missing"), root) is None
    assert memory.cgroup_memory_available(proc, str(tmp_path / "missing")) is None


@pytest.mark.parametrize("mem_available, a_b, match", [
    (64 * GiB, (4 * GiB, "max", GiB, 0, 0), r"only 3072 MiB is available under the memory cgroup limit and"),
    (None, (4 * GiB, "max", GiB, 0, 0), r"only 3072 MiB is available under the memory cgroup limit and"),
    (3 * GiB, (17 * GiB, "max", GiB, 0, 0), r"only 3072 MiB is available and"),
    (3 * GiB, None, r"only 3072 MiB is available and"),
], ids = ["cgroup_binds", "meminfo_unknown", "host_binds", "no_cgroup_limit"])
def test_guard(monkeypatch, tmp_path, mem_available, a_b, match):
    # 3 GiB left under the binding limit: a 2 GiB chunk plus the default 2 GiB reserve must be refused
    # with the guard's error, not handed to the OOM killer
    _tree(tmp_path, **({"a_b": a_b} if a_b else {}))
    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        file = str(file)
        if file == "/proc/self/cgroup":
            file = str(tmp_path / "self_cgroup")
        elif file.startswith("/sys/fs/cgroup"):
            file = str(tmp_path / "cg") + file[len("/sys/fs/cgroup"):]
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(memory, "open", fake_open, raising = False)
    monkeypatch.setattr(memory, "host_memory_available", lambda: mem_available)
    monkeypatch.delenv("EXL3_HOST_MEM_RESERVE_MB", raising = False)
    memory.check_host_memory(512 * MiB, "test chunk")
    with pytest.raises(RuntimeError, match = match) as e:
        memory.check_host_memory(2 * GiB, "test chunk")
    assert ("raise the memory cgroup limit" in str(e.value)) == ("cgroup" in match)
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "0")
    memory.check_host_memory(16 * GiB, "test chunk")
