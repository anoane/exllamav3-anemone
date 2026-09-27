"""
ngram_gather_cpu (Linux pread path, ngram.cu): run-coalesced row gathers must return the file bytes on
both the single-read and the pooled path, and a failed read must say why: a range running past the end
of the file (truncated table, out-of-range row id) reports "unexpected end of file" with its offset, and
a hard read error reports the errno. Regression: every failure used to surface as a bare "short read".
The EINTR retry needs an interrupt-honouring FUSE mount and is not exercised here.
CPU only; the Windows port (overlapped ReadFile on a HANDLE) is a separate implementation.
"""
import os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(os.name == "nt", reason = "Linux pread implementation")

from exllamav3.ext import exllamav3_ext as ext

ROW_BYTES = 256
ROWS = 256
BASE = 512      # table data offset inside the file


@pytest.fixture
def table(tmp_path):
    data = bytes((i * 7 + 3) & 0xff for i in range(BASE + ROWS * ROW_BYTES))
    path = tmp_path / "table.bin"
    path.write_bytes(data)
    fd = os.open(path, os.O_RDONLY)
    yield fd, data, path
    os.close(fd)


def gather(fd, uids, uid_base = 0):
    uids = torch.tensor(uids, dtype = torch.long)
    out = torch.zeros((len(uids), ROW_BYTES // 2), dtype = torch.half)
    ext.ngram_gather_cpu(fd, BASE, ROW_BYTES, uids, uid_base, out)
    return out.view(torch.uint8).view(-1, ROW_BYTES)


def expect(data, uids, uid_base = 0):
    rows = [data[BASE + (u - uid_base) * ROW_BYTES:][:ROW_BYTES] for u in uids]
    return torch.frombuffer(bytearray(b"".join(rows)), dtype = torch.uint8).view(-1, ROW_BYTES)


def test_gather_matches_file(table):
    fd, data, _ = table
    # one run, a few pooled runs, and more than 64 runs (grouped into span tasks)
    for uids in ([5, 6, 7, 8], [0, 2, 3, 9, 10, 11, 40, 63], list(range(0, ROWS, 2))):
        assert torch.equal(gather(fd, uids), expect(data, uids)), uids
    # shard-local row index
    uids = [100, 101, 130]
    assert torch.equal(gather(fd, uids, uid_base = 100), expect(data, uids, uid_base = 100))


@pytest.mark.parametrize("uids", [[ROWS - 1, ROWS], [0, 3, ROWS - 1, ROWS]], ids = ["single_run", "pooled"])
def test_gather_past_eof(table, uids):
    # The run [ROWS - 1, ROWS] reads one row, then hits the end of the file
    fd, _, _ = table
    run_offset = BASE + (ROWS - 1) * ROW_BYTES
    with pytest.raises(RuntimeError, match = f"unexpected end of file reading {2 * ROW_BYTES} bytes "
                                             f"at offset {run_offset}"):
        gather(fd, uids)


def test_gather_truncated_file(table):
    # Table file cut short after it was opened: rows before the cut still read, the rest hit EOF
    fd, data, path = table
    os.truncate(path, BASE + 10 * ROW_BYTES + 100)
    assert torch.equal(gather(fd, [2, 9]), expect(data, [2, 9]))
    with pytest.raises(RuntimeError, match = "unexpected end of file"):
        gather(fd, [2, 9, 10])


def test_gather_read_error():
    # pread on a pipe fails with ESPIPE: a hard error, reported with its errno
    r, w = os.pipe()
    try:
        with pytest.raises(RuntimeError, match = r"error reading file: .* \(errno=\d+\)"):
            gather(r, [1, 2])
        with pytest.raises(RuntimeError, match = r"error reading file: .* \(errno=\d+\)"):
            gather(r, [1, 5, 9])
    finally:
        os.close(r)
        os.close(w)
