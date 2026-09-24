# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Tech2wild
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
Reference reader for the file-backed engram embedding tables.

The engram tables are far too large to hold in VRAM or host RAM -- about
189 GiB across two layers for DeepSeek-V4.1-Flash:

    layers.L.engram.embed.weight   [num_embeddings, head_dim]  fp8_e4m3
    layers.L.engram.embed.scale    [num_embeddings, head_dim/32] fp8_e8m0

so they stay on disk and are read by row. A forward pass touches only
n_hash_cols rows per token per engram layer (24 here), which is a few KiB,
making random reads the right access pattern rather than streaming.

Rows are block-scaled fp8: every 32 consecutive values share one e8m0
(exponent-only) scale, hence head_dim/32 scale bytes per row.

The engine reads the rows through the extension's ngram_gather_cpu. This
numpy reader is the independent reference the tests compare against, and it
has two paths of its own that must agree bitwise: gather (pread) and
gather_mmap.

gather reads with ``pread``, not ``mmap``, and marks the fd
``POSIX_FADV_RANDOM``:

  * A row is 264 bytes (256 fp8 + 8 scale). Under mmap the kernel faults a
    whole page per row and, with default readahead, pulls up to 128 KiB for
    it -- hundreds of times the bytes actually wanted.
  * Those pages land in the page cache. Against a table larger than host RAM
    that evicts the pages the rest of the process lives in. Buffered pread
    still populates the page cache; FADV_RANDOM limits unnecessary read-ahead,
    but does not provide direct I/O or guarantee a cold-storage latency
    measurement.

A pool of threads issues the row reads, because a single-threaded random
read pattern spends all its time in disk latency rather than bandwidth.
This threaded pread reader (EngramTable._fd, EngramTable._pread_rows) is
adapted from Tech2wild's engram-on-disk reader (MIT); see NOTICE and
LICENSE-TECH2WILD.
"""

from __future__ import annotations

import mmap
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import load_package_file

# the checkpoint-header checks the engine itself uses, loaded without the compiled extension
_header = load_package_file("exllamav3/architecture/dsv41/checkpoint_header.py", "_dsv41_checkpoint_header")
read_header, validate_engram_layout = _header.read_header, _header.validate_engram_layout


def e8m0_to_f32(codes: np.ndarray) -> np.ndarray:
    """Decode every E8M0 encoding: 0 is 2^-127 and 255 is NaN."""
    codes = np.asarray(codes, dtype = np.uint8)
    bits = codes.astype(np.uint32) << 23
    # Setting the high mantissa bit forms the subnormal at code 0 and a
    # quiet NaN at code 255. The interior codes are ordinary FP32 exponents.
    bits |= ((codes == 0) | (codes == 255)).astype(np.uint32) << 22
    return bits.view(np.float32)


def _scan_shards(directory: str) -> dict:
    """tensor name -> (path, dtype, shape, absolute byte start, byte end)."""
    index = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(directory, name)
        hdr, base, _ = read_header(path)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            off = v.get("data_offsets")
            if not off:
                continue
            index[k] = (path, v["dtype"], tuple(v["shape"]), base + off[0], base + off[1])
    return index


class EngramTable:
    """One layer's engram table, read by row from disk."""

    def __init__(self, directory: str, layer_id: int, head_dim: int, threads: int = 32, chunk: int = 16):
        """threads: pread workers; chunk: rows per worker task."""
        idx = _scan_shards(directory)
        wk = f"layers.{layer_id}.engram.embed.weight"
        sk = f"layers.{layer_id}.engram.embed.scale"
        if wk not in idx or sk not in idx:
            raise KeyError(f"engram table for layer {layer_id} not found in {directory}")
        self.layer_id = layer_id
        self.head_dim = head_dim
        self._w = idx[wk]
        self._s = idx[sk]
        self.num_embeddings = validate_engram_layout(
            {"dtype": self._w[1], "shape": list(self._w[2])},
            {"dtype": self._s[1], "shape": list(self._s[2])}, head_dim)
        self.scale_groups = self._s[2][1]
        self.group_size = head_dim // self.scale_groups
        self._maps: dict[str, mmap.mmap] = {}
        self._fds: dict[str, int] = {}
        self.threads = int(threads)
        self.chunk = int(chunk)
        if self.threads < 1 or self.chunk < 1:
            raise ValueError("engram disk thread/chunk counts must be positive")
        self._pool = ThreadPoolExecutor(max_workers = self.threads)

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            try:
                # no readahead: a 264-byte row must not drag in 128 KiB
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except (AttributeError, OSError):
                pass
            self._fds[path] = fd
        return fd

    def _map(self, path: str) -> mmap.mmap:
        m = self._maps.get(path)
        if m is None:
            fh = open(path, "rb")
            m = mmap.mmap(fh.fileno(), 0, prot = mmap.PROT_READ)
            self._maps[path] = m
        return m

    def _pread_rows(self, path: str, base: int, rows, row_bytes: int) -> np.ndarray:
        """[N, row_bytes] uint8, one pread per row, issued from a thread pool."""
        n = len(rows)
        buf = np.empty((n, row_bytes), dtype = np.uint8)
        if n == 0:
            return buf
        fd = self._fd(path)
        view = memoryview(buf).cast("B")

        def work(lo: int, hi: int):
            for i in range(lo, hi):
                off = base + int(rows[i]) * row_bytes
                dst = view[i * row_bytes:(i + 1) * row_bytes]
                got = 0
                while got < row_bytes:
                    k = os.preadv(fd, [dst[got:]], off + got)
                    if k <= 0:
                        raise OSError(
                            f"engram table: short read at row {rows[i]} of {path}")
                    got += k

        if n <= self.chunk:
            work(0, n)
        else:
            futs = [self._pool.submit(work, lo, min(lo + self.chunk, n))
                    for lo in range(0, n, self.chunk)]
            for f in futs:
                f.result()
        return buf

    def gather(self, rows: np.ndarray) -> np.ndarray:
        """
        Read rows and dequantise to float32.

        :param rows: [N] int64 row indices
        :return:     [N, head_dim] float32
        """
        rows = np.asarray(rows, dtype = np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.num_embeddings):
            raise IndexError("engram row index out of range")
        wpath, _, _, wstart, _ = self._w
        spath, _, _, sstart, _ = self._s
        raw = self._pread_rows(wpath, wstart, rows, self.head_dim)
        exp = self._pread_rows(spath, sstart, rows, self.scale_groups)
        return self._dequant(raw, exp)

    def _dequant(self, raw: np.ndarray, exp: np.ndarray) -> np.ndarray:
        """[N, head_dim] uint8 fp8 rows x [N, groups] e8m0 -> [N, head_dim] f32."""
        vals = _fp8_e4m3_to_f32(raw.reshape(-1)).reshape(
            -1, self.scale_groups, self.group_size)
        scale = e8m0_to_f32(exp)
        return (vals * scale[:, :, None]).reshape(-1, self.head_dim)

    def gather_mmap(self, rows: np.ndarray) -> np.ndarray:
        """
        Same result through mmap, kept as the reference the pread path is
        checked against. Not the path to serve from -- see the module
        docstring on readahead and page-cache pressure.
        """
        rows = np.asarray(rows, dtype = np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.num_embeddings):
            raise IndexError("engram row index out of range")
        wpath, _, _, wstart, _ = self._w
        spath, _, _, sstart, _ = self._s
        wm, sm = self._map(wpath), self._map(spath)
        raw = np.empty((rows.size, self.head_dim), dtype = np.uint8)
        exp = np.empty((rows.size, self.scale_groups), dtype = np.uint8)
        for i, r in enumerate(rows):
            wo = wstart + int(r) * self.head_dim
            so = sstart + int(r) * self.scale_groups
            raw[i] = np.frombuffer(wm[wo:wo + self.head_dim], dtype = np.uint8)
            exp[i] = np.frombuffer(sm[so:so + self.scale_groups], dtype = np.uint8)
        return self._dequant(raw, exp)

    def close(self):
        # A failed future may leave other reads queued. Drain them while their
        # descriptors still refer to these files, never to a newly reused fd.
        self._pool.shutdown(wait = True)
        for m in self._maps.values():
            m.close()
        self._maps.clear()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def _fp8_e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    """Decode fp8 e4m3fn (1-4-3, no inf, 0xFF/0x7F = NaN) to float32, as
    torch.float8_e4m3fn does on all 256 byte values."""
    b = b.astype(np.uint16)
    sign = (b >> 7) & 0x1
    exp = (b >> 3) & 0xF
    man = b & 0x7
    val = np.where(
        exp == 0,
        np.ldexp(man.astype(np.float32) / 8.0, -6),          # subnormal
        np.ldexp(1.0 + man.astype(np.float32) / 8.0, exp.astype(np.int32) - 7),
    )
    val = np.where(sign == 1, -val, val).astype(np.float32)
    return np.where((exp == 15) & (man == 7), np.float32(np.nan), val)
