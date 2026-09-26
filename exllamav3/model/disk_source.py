"""
Where a component reads its routed experts and n-gram tables at runtime, as the placement's disk
rule says (doc/expert_tiers.md, "disk: where experts and n-gram rows are read"):

    disk experts=<dir>      the expert tier reads every expert (cold fill, misses, read-ahead,
                            refills) from a copy of the checkpoint's shards in <dir>
    disk ngram=<dir>        n-gram / engram rows come from a copy of the table's shards in <dir>
    disk io=direct|buffered the page-cache mode of those reads; auto: the disk engine's own
                            (EXL3_DISK_DIRECT: io_uring reads extents O_DIRECT and rows buffered)

A copy holds the same shard files under the same names. It is checked before it is used: every
shard a read would touch must exist there with the same size and the same safetensors header,
byte for byte, so the offsets the loader took from the model's own shards hold in the copy. A
mismatch is refused with the file named; nothing is read from a copy that differs.

Torch-free.
"""

from __future__ import annotations
import os
import struct
import threading

_checked = {}                   # (model path, size, mtime, copy path, size, mtime) -> None when verified
_lock = threading.Lock()


def _header(path: str) -> bytes:
    """The safetensors header of a file: its 8-byte length and the JSON that follows"""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        n = struct.unpack("<Q", head)[0]
        if n > 1 << 30:
            raise ValueError(f"{path}: a safetensors header of {n} bytes is implausible")
        body = f.read(n)
        if len(body) != n:
            raise ValueError(f"{path}: the safetensors header is cut short")
        return head + body


def copy_path(path: str, directory: str, what: str) -> str:
    """The copy of the model's shard `path` in `directory` (same file name), after checking that it
    is the same file as far as reads go: same size, same header. `what` names the rule (experts or
    ngram) for the messages"""
    alt = os.path.join(directory, os.path.basename(path))
    if not os.path.isdir(directory):
        raise ValueError(f"placement: disk {what}={directory}: no such directory")
    if not os.path.isfile(alt):
        raise ValueError(f"placement: disk {what}={directory}: {os.path.basename(path)} is not there (the copy must "
                         f"hold every shard the reads touch, under the model's file names)")
    if os.path.realpath(path) == os.path.realpath(alt):
        return alt
    # checked once per version of the two files (a copy rewritten since is checked again)
    st_a, st_b = os.stat(path), os.stat(alt)
    key = (os.path.realpath(path), st_a.st_size, st_a.st_mtime_ns, os.path.realpath(alt), st_b.st_size, st_b.st_mtime_ns)
    with _lock:
        if key in _checked:
            return alt
    sa, sb = st_a.st_size, st_b.st_size
    if sa != sb:
        raise ValueError(f"placement: disk {what}={directory}: {alt} is {sb} bytes, the model's {path} is {sa}; "
                         f"the copy must be the same file")
    if _header(path) != _header(alt):
        raise ValueError(f"placement: disk {what}={directory}: the safetensors header of {alt} differs from the "
                         f"model's {path}; the copy must be the same file")
    with _lock:
        _checked[key] = None
    return alt


def source_dir(placement, what: str) -> str | None:
    """The directory `disk experts=` / `disk ngram=` names, None for the model's own shards (model)
    or no placement; `off` is not a source (the caller never reads then)"""
    if placement is None:
        return None
    disk = placement.disk_store()
    v = disk.experts if what == "experts" else disk.ngram
    return None if v in ("model", "off") else v


def expert_files(placement, files: list) -> list:
    """The shard files the expert tier reads: the model's own, or their checked copies in the directory
    disk experts= names"""
    d = source_dir(placement, "experts")
    return [copy_path(f, d, "experts") for f in files] if d else list(files)


def io_direct(placement) -> int:
    """The page-cache mode of the component's expert and n-gram reads: -1 the disk engine's choice
    (disk io=auto, or no placement), 1 O_DIRECT (io=direct), 0 buffered (io=buffered)"""
    if placement is None:
        return -1
    return {"auto": -1, "direct": 1, "buffered": 0}[placement.disk_store().io]


def check_route(placement, route: str, what: str):
    """disk io= needs the disk engine: the original n-gram gather pool reads buffered only"""
    if io_direct(placement) >= 0 and route != "engine":
        raise ValueError(f"placement: disk io={placement.disk_store().io} sets the page-cache mode of the disk "
                         f"engine's reads, but {what} are read by the original gather pool (EXL3_DISK_BACKEND="
                         f"original); unset EXL3_DISK_BACKEND or name a backend, or drop io=")
