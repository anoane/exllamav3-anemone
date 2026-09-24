"""
Malformed checkpoint metadata is rejected before any payload allocation/read: the V4.1
safetensors header checks (architecture/dsv41/checkpoint_header.py) and the reference engram
table reader built on them (tests/dsv41_ref/engram_tables.py), on synthetic shards. CPU only;
the compiled extension is not imported.

    python -m pytest tests/test_dsv41_checkpoint_header_.py
"""

import json
import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file
from dsv41_ref import engram_tables as tables

h = load_package_file("exllamav3/architecture/dsv41/checkpoint_header.py", "_dsv41_checkpoint_header")
WK, SK = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"


def metadata():
    return {"__metadata__": {"format": "pt"},
            WK: {"dtype": "F8_E4M3", "shape": [3, 32], "data_offsets": [0, 96]},
            SK: {"dtype": "F8_E8M0", "shape": [3, 1], "data_offsets": [96, 99]}}


def write(path, header, payload = b"\x38" * 96 + b"\x7f" * 3):
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def handles(path, header):
    _, base, _ = h.read_header(path)
    result = []
    for key in (WK, SK):
        desc = header[key]
        result.append(SimpleNamespace(filename = str(path), key = key, shape = desc["shape"],
            abs_offset = base + desc["data_offsets"][0], num_rows = 3,
            row_bytes = (desc["data_offsets"][1] - desc["data_offsets"][0]) // 3))
    return result


def test_normal_metadata_and_payload_are_preserved(tmp_path):
    path = tmp_path / "table.safetensors"
    write(path, metadata())
    actual, _, size = h.read_header(path)
    assert actual == metadata() and size == path.stat().st_size
    assert h.validate_engram_handles(*handles(path, actual), 32, 3) == 3
    table = tables.EngramTable(tmp_path, 1, 32)
    try:
        expected = np.ones((2, 32), dtype = np.float32)
        assert np.array_equal(table.gather([0, 2]), expected)
        assert np.array_equal(table.gather_mmap([0, 2]), expected)
        for gather in (table.gather, table.gather_mmap):
            with pytest.raises(IndexError):
                gather([-1])
            with pytest.raises(IndexError):
                gather([3])
    finally:
        table.close()


@pytest.mark.parametrize("mutation", ["negative_offset", "overlap", "short_file", "bad_shape",
                                     "nan", "unknown_dtype", "invalid_metadata", "bool_offset"])
def test_invalid_descriptors_fail_every_header_reader(tmp_path, mutation):
    header = metadata()
    payload = b"\x38" * 96 + b"\x7f" * 3
    if mutation == "negative_offset":
        header[WK]["data_offsets"] = [-1, 95]
    elif mutation == "overlap":
        header[SK]["data_offsets"] = [95, 98]
    elif mutation == "short_file":
        payload = payload[:-1]
    elif mutation == "bad_shape":
        header[WK]["shape"] = [3, 33]
    elif mutation == "nan":
        header[WK]["shape"] = [float("nan"), 32]
    elif mutation == "unknown_dtype":
        header[WK]["dtype"] = "UNKNOWN"
    elif mutation == "invalid_metadata":
        header["__metadata__"]["format"] = 7
    else:
        header[WK]["data_offsets"][0] = False
    path = tmp_path / "table.safetensors"
    write(path, header, payload)
    for call in (lambda: h.read_header(path), lambda: tables.EngramTable(tmp_path, 1, 32)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("raw", [b"", b"1234567", struct.pack("<Q", (1 << 64) - 1),
                                 struct.pack("<Q", h.MAX_HEADER_SIZE + 1),
                                 struct.pack("<Q", 2) + b"[]",
                                 struct.pack("<Q", 13) + b'{"x":0,"x":1}'])
def test_header_length_and_json_guards(tmp_path, raw):
    path = tmp_path / "table.safetensors"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        h.read_header(path)


@pytest.mark.parametrize("mutation", ["weight_bf16", "scale_u8", "scale_shape", "row_count", "handle_offset"])
def test_engram_requires_exact_encoding_not_equal_row_bytes(tmp_path, mutation):
    header = metadata()
    if mutation == "weight_bf16":
        header[WK].update(dtype = "BF16", shape = [3, 16])   # still 32 bytes per row
    elif mutation == "scale_u8":
        header[SK]["dtype"] = "U8"
    elif mutation == "scale_shape":
        header[SK]["shape"] = [1, 3]
    path = tmp_path / "table.safetensors"
    write(path, header)
    got = handles(path, header)
    if mutation == "handle_offset":
        got[0].abs_offset += 1
    with pytest.raises(ValueError):
        h.validate_engram_handles(*got, 32, 4 if mutation == "row_count" else 3)
    if mutation not in ("handle_offset", "row_count"):
        with pytest.raises(ValueError):
            tables.EngramTable(tmp_path, 1, 32)


def test_disk_worker_counts_are_positive(tmp_path):
    write(tmp_path / "table.safetensors", metadata())
    for counts in ({"threads": 0}, {"chunk": 0}):
        with pytest.raises(ValueError):
            tables.EngramTable(tmp_path, 1, 32, **counts)


def test_reference_reader_drains_workers_before_closing_files(monkeypatch):
    order = []
    table = object.__new__(tables.EngramTable)
    table._pool = SimpleNamespace(shutdown = lambda **kwargs: order.append(("workers", kwargs)))
    table._maps = {"table": SimpleNamespace(close = lambda: order.append(("map", None)))}
    table._fds = {"table": 123}
    monkeypatch.setattr(tables.os, "close", lambda fd: order.append(("fd", fd)))
    table.close()
    assert order == [("workers", {"wait": True}), ("map", None), ("fd", 123)]
    assert not table._maps and not table._fds
