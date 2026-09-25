"""
Sizes with units, the host memory this process may still take, and a ledger of the host memory a
load is about to hold, checked against it before anything is allocated. Torch-free: the loader,
the placement parser and the torch-free tests all use it.

Sizes (the RAM budgets --expert_ram, --draft_expert_ram, --ngram_ram, EXL3_EXPERT_RAM,
EXL3_NGRAM_RAM and the placement's ram / cuda:<n> rules, doc/expert_tiers.md):

    48GiB   48 * 2^30 bytes (also 48Gi, 48gib: units are case-insensitive)
    48GB    48 * 10^9 bytes
    48      48 GiB: a bare number is GiB, as -gs / use_per_device read theirs
    1.5     1.5 GiB, printed 1536MiB (the largest exact unit)
    48G     refused: ambiguous between 48GiB and 48GB

Host memory: MemAvailable from /proc/meminfo, lowered to what the process's memory cgroup (v2)
still allows when a cgroup on its path has a limit (systemd-run -p MemoryMax=..., a container).
EXL3_HOST_MEM_RESERVE_MB (default 2048) is kept free on top of every check; 0 disables the checks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import os
import re

KiB, MiB, GiB, TiB = 1 << 10, 1 << 20, 1 << 30, 1 << 40

# ---------------------------------------------------------------------------------------- sizes

_UNIT = {
    "": GiB,                # a bare number is GiB, as -gs / -ccs / use_per_device read theirs
    "b": 1,
    "kib": KiB, "ki": KiB, "kb": 10 ** 3,
    "mib": MiB, "mi": MiB, "mb": 10 ** 6,
    "gib": GiB, "gi": GiB, "gb": 10 ** 9,
    "tib": TiB, "ti": TiB, "tb": 10 ** 12,
}
# bare K / M / G / T: 2^10.. or 10^3..? Refused, never guessed
_AMBIGUOUS = {"k": 10, "m": 20, "g": 30, "t": 40}
# Printing: the largest unit that divides the size exactly, binary first at equal magnitude
_CANON_UNIT = (("TiB", TiB), ("TB", 10 ** 12), ("GiB", GiB), ("GB", 10 ** 9),
               ("MiB", MiB), ("MB", 10 ** 6), ("KiB", KiB), ("KB", 10 ** 3), ("B", 1))
# ASCII digits only: \d would also take other scripts' digits
_SIZE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([A-Za-z]*)$")
SIZE_HELP = "0, auto, all, or a number with B, KiB, MiB, GiB, TiB, KB, MB, GB or TB (a bare number is GiB)"


def fmt_size(n: int) -> str:
    """Exact, in the largest unit that divides n (the canonical form: parse_size reads it back)"""
    if n == 0:
        return "0"
    for name, unit in _CANON_UNIT:
        if n % unit == 0:
            return f"{n // unit}{name}"
    return f"{n}B"


def human(n: int) -> str:
    """Rounded, for messages: '52.4 GiB', '177 MiB'"""
    return f"{n / GiB:.1f} GiB" if abs(n) >= GiB else f"{n / MiB:.0f} MiB"


@dataclass(frozen = True, eq = False)
class Size:
    """A RAM or VRAM budget: a byte count, 'auto' (what is free, less what other budgets and the
    page cache keep) or 'all' (whatever the layers need)"""
    kind: str = "bytes"     # "bytes", "auto" or "all"
    nbytes: int = 0

    @property
    def is_bytes(self) -> bool:
        return self.kind == "bytes"

    @property
    def is_auto(self) -> bool:
        return self.kind == "auto"

    @property
    def is_all(self) -> bool:
        return self.kind == "all"

    @property
    def is_zero(self) -> bool:
        return self.kind == "bytes" and self.nbytes == 0

    def __str__(self):
        return self.kind if self.kind != "bytes" else fmt_size(self.nbytes)

    def __repr__(self):
        return f"Size({str(self)!r})"

    # Structural: a module loaded by path (the torch-free tests) and the same module imported
    # through the package are two classes; their sizes still compare
    def __eq__(self, other):
        if not (hasattr(other, "kind") and hasattr(other, "nbytes")):
            return NotImplemented
        return (self.kind, self.nbytes) == (other.kind, other.nbytes)

    def __hash__(self):
        return hash((self.kind, self.nbytes))


AUTO, ALL, ZERO = Size("auto"), Size("all"), Size("bytes", 0)


def parse_size(text, what: str, allow = ("auto", "all")) -> Size:
    """
    A size as written in a flag, variable or placement: see the module docstring. `what` names
    the setting in messages ('experts' gives "experts=48G: ..."); `allow` lists the words besides
    a size that it accepts. Raises ValueError
    """
    if isinstance(text, Size):
        if not text.is_bytes and text.kind not in allow:
            raise ValueError(f"{what}={text}: {text.kind} is not allowed here (use {_allowed(allow)})")
        return text
    t = str(text).strip()
    low = t.lower()
    if low in ("auto", "all"):
        if low not in allow:
            raise ValueError(f"{what}={t}: {low} is not allowed here (use {_allowed(allow)})")
        return Size(low)
    m = _SIZE.match(t)
    unit = m.group(2).lower() if m else None
    if m and unit in _AMBIGUOUS:
        u, e = unit.upper(), _AMBIGUOUS[unit]
        raise ValueError(f"{what}={t}: {t!r} is ambiguous: write {m.group(1)}{u}iB (2^{e} bytes) or "
                         f"{m.group(1)}{u}B (10^{e * 3 // 10} bytes)")
    if m is None or unit not in _UNIT:
        raise ValueError(f"{what}={t}: not a size (use {SIZE_HELP})")
    return Size("bytes", int(round(Fraction(m.group(1)) * _UNIT[unit])))


def _allowed(allow) -> str:
    return " or ".join(list(allow) + ["a size"]) if allow else "a size"


# ---------------------------------------------------------------------------------------- memory

@dataclass(frozen = True)
class HostMemory:
    """What the host can still give this process. None: unknown on this platform"""
    available: int | None               # min(MemAvailable, the memory cgroup's room)
    mem_available: int | None = None    # MemAvailable
    mem_total: int | None = None
    swap_total: int | None = None
    cgroup_room: int | None = None      # tightest room of the cgroups on this process's path
    cgroup_path: str | None = None      # the cgroup that sets cgroup_room

    @property
    def cgroup_limits(self) -> bool:
        """The memory cgroup, not MemAvailable, is what limits this process"""
        return self.cgroup_room is not None and self.mem_available is not None and \
            self.cgroup_room < self.mem_available


def _meminfo(proc_root: str) -> dict:
    out = {}
    try:
        with open(os.path.join(proc_root, "meminfo")) as f:
            for line in f:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts:
                    out[k.strip()] = int(parts[0]) * (1024 if parts[1:] == ["kB"] else 1)
    except OSError:
        pass
    return out


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cgroup_room(proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> tuple[int | None, str | None]:
    """
    (bytes, cgroup) the tightest memory cgroup (v2) on this process's path still allows, or
    (None, None) without a limit. Per level: min(memory.max, memory.high) - memory.current + the
    level's file LRU (page cache it can drop: the loader reads whole checkpoints through it, and
    the kernel reclaims it before refusing a charge); shmem and pinned pages are not on the file
    LRU and stay counted
    """
    rel = None
    text = _read(os.path.join(proc_root, "self", "cgroup"))
    for line in (text or "").splitlines():
        hid, _, rest = line.partition(":")
        ctrl, _, path = rest.partition(":")
        if hid == "0" and ctrl == "":
            rel = path
            break
    if rel is None:
        return None, None
    best = (None, None)
    path = rel.strip("/")
    while True:
        d = os.path.join(cgroup_root, path) if path else cgroup_root
        limits = []
        for name in ("memory.max", "memory.high"):
            v = _read(os.path.join(d, name))
            if v is not None and v != "max" and v.isdigit():
                limits.append(int(v))
        current = _read(os.path.join(d, "memory.current"))
        if limits and current is not None and current.isdigit():
            stat = {}
            for line in (_read(os.path.join(d, "memory.stat")) or "").splitlines():
                k, _, v = line.partition(" ")
                if v.strip().isdigit():
                    stat[k] = int(v)
            reclaimable = stat.get("active_file", 0) + stat.get("inactive_file", 0)
            room = max(0, min(limits) - int(current) + reclaimable)
            if best[0] is None or room < best[0]:
                best = (room, "/" + path)
        if not path:
            break
        path = path.rpartition("/")[0]
    return best


def host_memory(proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> HostMemory:
    """The host memory this process may still take (see HostMemory)"""
    info = _meminfo(proc_root)
    avail = info.get("MemAvailable")
    total = info.get("MemTotal")
    swap = info.get("SwapTotal")
    if avail is None:
        try:
            import psutil
            vm = psutil.virtual_memory()
            avail, total = int(vm.available), int(vm.total)
            swap = int(psutil.swap_memory().total)
        except Exception:
            return HostMemory(None)
    room, cg = cgroup_room(proc_root, cgroup_root)
    return HostMemory(min(avail, room) if room is not None else avail, avail, total, swap, room, cg)


def reserve_bytes(environ = None) -> int:
    """EXL3_HOST_MEM_RESERVE_MB (default 2048) in bytes: kept free on top of every host-memory
    check; 0 (or less) disables the checks"""
    environ = os.environ if environ is None else environ
    v = str(environ.get("EXL3_HOST_MEM_RESERVE_MB", "2048")).strip()
    if not re.fullmatch(r"-?[0-9]+", v):
        raise ValueError(f"EXL3_HOST_MEM_RESERVE_MB={v!r} must be a whole number of MiB (0 disables the check)")
    return max(0, int(v)) * MiB


# ---------------------------------------------------------------------------------------- ledger

@dataclass
class LedgerItem:
    label: str              # as the message prints it: "ram experts", "ngram", ...
    nbytes: int
    pinned: bool = True     # page-locked (never reclaimable) or plain anonymous memory


@dataclass
class HostLedger:
    """
    The host memory one load is about to hold, item by item, checked at once against what the host
    can give before any of it is allocated:

        ledger = HostLedger()
        ledger.add("ram experts", 96 * GiB)
        ledger.add("ngram", 6 * GiB)
        ledger.check("placement")      # RuntimeError naming every item when they do not fit
    """
    memory: HostMemory | None = None    # None: read the host when checking
    reserve: int | None = None          # None: EXL3_HOST_MEM_RESERVE_MB
    items: list = field(default_factory = list)

    def add(self, label: str, nbytes: int, pinned: bool = True):
        if nbytes > 0:
            self.items.append(LedgerItem(label, int(nbytes), pinned))

    @property
    def total(self) -> int:
        return sum(i.nbytes for i in self.items)

    def check(self, what: str, advice: str = "lower the budgets or use auto"):
        """Raise RuntimeError when the items plus the reserve exceed the available memory. `what`
        prefixes the message ("placement", "-mcl 11", ...)"""
        reserve = reserve_bytes() if self.reserve is None else self.reserve
        if reserve <= 0 or not self.items:
            return
        mem = self.memory if self.memory is not None else host_memory()
        if mem.available is None:
            return
        total = self.total
        if total + reserve <= mem.available:
            return
        parts = " + ".join(f"{i.label}={human(i.nbytes)}" for i in self.items)
        cg = ""
        if mem.cgroup_limits:
            cg = (f" (the memory cgroup {mem.cgroup_path} allows {human(mem.cgroup_room)} more; "
                  f"MemAvailable is {human(mem.mem_available)})")
        swap = " (no swap: pinned and anonymous memory cannot be reclaimed)" if mem.swap_total == 0 else ""
        raise RuntimeError(f"{what}: {parts} need{'s' if len(self.items) == 1 else ''} {human(total)} of host "
                           f"memory, but {human(mem.available)} is available{cg} and {human(reserve)} is kept "
                           f"free (EXL3_HOST_MEM_RESERVE_MB){swap}; {advice}")
