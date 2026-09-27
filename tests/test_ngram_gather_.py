"""
ngram_gather_cpu reads the unique n-gram table rows straight from disk. On Windows the handle is
unbuffered, so each coalesced run is read as a 4K-aligned span into a 64 KiB bounce slot
(ngram_gather_win.cpp): runs are capped so the aligned span always fits the slot, and on Windows a
row larger than the per-read cap is refused. Model-sized rows (82, 320, 640 bytes) plus wider rows
start at unaligned offsets, so runs hit the cap at every sector phase; the gather must return the
file bytes on both the decode (<= 64 runs) and the pooled (prefill) path.
"""
import os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.loader.safetensors import DiskTensorHandle

WIN = os.name == "nt"
BOUNCE = 64 * 1024


def make_table(tmp_path, pad, num_rows, row_bytes):
    g = torch.Generator().manual_seed(row_bytes + pad)
    data = torch.randint(0, 256, (num_rows, row_bytes), dtype = torch.uint8, generator = g)
    filename = str(tmp_path / f"table_{row_bytes}_{pad}.bin")
    with open(filename, "wb") as f:
        f.write(b"\xa5" * pad)
        f.write(data.numpy().tobytes())
    return DiskTensorHandle("table", filename, pad, [num_rows, row_bytes], torch.uint8), data


def gather(handle, uids):
    out = torch.empty((uids.numel(), handle.row_bytes), dtype = torch.uint8)
    ext.ngram_gather_cpu(handle._ensure_open(), handle.abs_offset, handle.row_bytes, uids, 0, out)
    return out


@pytest.mark.parametrize("pooled", [False, True], ids = ["decode", "pooled"])
@pytest.mark.parametrize("pad", [0, 1000, 4091, 4095])
@pytest.mark.parametrize("row_bytes", [82, 320, 640, 4389, 7000, 40000, 61441])
def test_gather_long_runs_at_unaligned_offsets(tmp_path, row_bytes, pad, pooled):
    # one run longer than a slot, then (pooled) > 64 single-row runs
    block = 2 * (BOUNCE // row_bytes + 2)
    scattered = 70 if pooled else 0
    handle, data = make_table(tmp_path, pad, block + 2 * scattered, row_bytes)
    uids = torch.cat([torch.arange(block), block + 1 + 2 * torch.arange(scattered)])
    try:
        out = gather(handle, uids)
    finally:
        handle.close()
    assert torch.equal(out, data[uids])


@pytest.mark.skipif(not WIN, reason = "the bounce slot bound applies to the unbuffered Windows reads")
@pytest.mark.parametrize("row_bytes", [61442, 70000])
def test_gather_refuses_rows_larger_than_bounce_slot(tmp_path, row_bytes):
    handle, _ = make_table(tmp_path, 4091, 2, row_bytes)
    try:
        with pytest.raises(RuntimeError, match = "bounce slot"):
            gather(handle, torch.arange(2))
    finally:
        handle.close()
