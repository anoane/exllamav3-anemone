"""
Host-memory guard (check_host_memory) on Windows: there is no /proc/meminfo, and psutil is an
optional extra, so without it host_memory_available() falls back to GlobalMemoryStatusEx's
available physical RAM (what psutil reports there) instead of returning None and skipping the
check. The Windows branch is selected by os.name, so it runs on any platform with a fake kernel32.
"""
import builtins, ctypes, os, sys
from types import SimpleNamespace
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MiB = 1 << 20


class _FakeKernel32:
    """kernel32 stand-in: GlobalMemoryStatusEx fills the caller's MEMORYSTATUSEX, or fails"""
    def __init__(self, avail_phys, avail_commit, ok = True):
        self.avail_phys, self.avail_commit, self.ok = avail_phys, avail_commit, ok
        self.calls = 0

    def GlobalMemoryStatusEx(self, ref):
        self.calls += 1
        if not self.ok:
            return 0
        status = ref._obj
        assert status.dwLength == ctypes.sizeof(status)          # the API rejects anything else
        status.ullAvailPhys = self.avail_phys
        status.ullAvailPageFile = self.avail_commit
        return 1


def _no_meminfo(monkeypatch):
    real_open = builtins.open
    def fake_open(file, *a, **k):
        if file == "/proc/meminfo":
            raise FileNotFoundError(2, "No such file or directory", file)
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", fake_open)


def _host(monkeypatch, os_name, k32, psutil_available = None):
    """No /proc/meminfo, psutil absent (or reporting `psutil_available`), os.name = `os_name`"""
    from exllamav3.util import memory
    _no_meminfo(monkeypatch)
    if psutil_available is None:
        monkeypatch.setitem(sys.modules, "psutil", None)
    else:
        fake = SimpleNamespace(virtual_memory = lambda: SimpleNamespace(available = psutil_available))
        monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr("ctypes.WinDLL", lambda *a, **k: k32, raising = False)
    monkeypatch.setattr("ctypes.get_last_error", lambda: 8, raising = False)
    monkeypatch.setattr("ctypes.WinError", lambda code: OSError(code, "injected"), raising = False)
    monkeypatch.setattr(os, "name", os_name)
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "2048")
    return memory


def test_windows_without_psutil_reports_available_physical_ram(monkeypatch):
    k32 = _FakeKernel32(avail_phys = 6144 * MiB, avail_commit = 20480 * MiB)
    memory = _host(monkeypatch, "nt", k32)
    assert memory.host_memory_available() == 6144 * MiB
    assert k32.calls == 1


def test_windows_without_psutil_guard_raises_past_the_reserve(monkeypatch):
    k32 = _FakeKernel32(avail_phys = 6144 * MiB, avail_commit = 20480 * MiB)
    memory = _host(monkeypatch, "nt", k32)
    memory.check_host_memory(4096 * MiB, "test allocation")           # 4096 + 2048 reserve fits
    with pytest.raises(RuntimeError, match = r"test allocation needs 4097 MiB.*only 6144 MiB"):
        memory.check_host_memory(4097 * MiB, "test allocation")
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "0")
    memory.check_host_memory(1 << 50, "test allocation")              # 0 still disables it


def test_windows_with_psutil_keeps_the_psutil_figure(monkeypatch):
    k32 = _FakeKernel32(avail_phys = 6144 * MiB, avail_commit = 20480 * MiB)
    memory = _host(monkeypatch, "nt", k32, psutil_available = 3072 * MiB)
    assert memory.host_memory_available() == 3072 * MiB
    assert k32.calls == 0


def test_windows_query_failure_leaves_the_guard_inactive(monkeypatch):
    k32 = _FakeKernel32(0, 0, ok = False)
    memory = _host(monkeypatch, "nt", k32)
    assert memory.host_memory_available() is None
    memory.check_host_memory(1 << 50, "test allocation")
    assert k32.calls == 2                                               # queried, failed, skipped


def test_other_platforms_without_meminfo_or_psutil_stay_unknown(monkeypatch):
    k32 = _FakeKernel32(avail_phys = 6144 * MiB, avail_commit = 20480 * MiB)
    memory = _host(monkeypatch, "posix", k32)
    assert memory.host_memory_available() is None
    memory.check_host_memory(1 << 50, "test allocation")
    assert k32.calls == 0


@pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason = "Linux /proc/meminfo")
def test_linux_meminfo_is_still_read_first(monkeypatch):
    from exllamav3.util import memory
    k32 = _FakeKernel32(avail_phys = 6144 * MiB, avail_commit = 20480 * MiB)
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr("ctypes.WinDLL", lambda *a, **k: k32, raising = False)
    monkeypatch.setattr(os, "name", "nt")
    avail = memory.host_memory_available()
    assert avail is not None and avail > 0 and k32.calls == 0
