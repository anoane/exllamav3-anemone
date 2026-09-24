"""A stream-mode layer (experts=stream, -mcm stream_only) never hands a batch to the CPU worker:
every batch of cpu_offload_issue and cpu_split_submit, one-token decode included, takes the
streamed GPU path. A hybrid layer still issues decode-size batches to the worker.
Run: python -m pytest -q tests/test_moe_stream_decode_.py  or  python tests/test_moe_stream_decode_.py"""
import torch
from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU


class _Host:
    stream_min_rows = 32

    def __init__(self, mode):
        self.mode, self.calls = mode, []

    def gpu_only(self, layer_idx):
        return self.mode == "stream"

    def submit_issue(self, layer_idx, y, sel, w):
        self.calls.append("issue")
        return "pending"

    def submit_prefill(self, layer_idx, y, sel, w):
        self.calls.append("prefill")
        return torch.zeros(y.shape, dtype = torch.float)


class _Layer(BlockSparseMLP_CPU):
    def __init__(self, mode):
        self.cpu_host, self.cpu_layer_idx = _Host(mode), 0


def _issue(mode, rows):
    layer = _Layer(mode)
    y = torch.zeros((rows, 8), dtype = torch.half)
    sel = torch.zeros((rows, 2), dtype = torch.long)
    w = torch.zeros((rows, 2), dtype = torch.half)
    layer.cpu_offload_issue((rows, 8), y, sel, w, {})
    return layer.cpu_host.calls


def test_stream_mode_decode_takes_the_streamed_path():
    for rows in (1, 4, 31, 32, 4096):
        assert _issue("stream", rows) == ["prefill"], f"stream mode, {rows} rows: sent to the CPU worker"


def test_hybrid_mode_decode_is_issued_to_the_worker():
    assert _issue("hybrid", 1) == ["issue"]
    assert _issue("hybrid", 31) == ["issue"]
    assert _issue("hybrid", 32) == ["prefill"]


if __name__ == "__main__":
    test_stream_mode_decode_takes_the_streamed_path()
    test_hybrid_mode_decode_is_issued_to_the_worker()
    print("  -- stream decode: OK")
