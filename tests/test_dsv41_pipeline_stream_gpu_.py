"""Two-GPU regression for pipelined prefill on a non-default caller stream.

The toy layers exercise the real pipeline driver, not model math. The destination GPU's
default stream is delayed, so a driver whose worker thread silently enqueues the second
stage there (CUDA current streams are thread-local) is caught: cache reads and a simulated
decode queued on the caller's stream right after the prefill must see every prefill write
without a device-wide synchronization. A negative control runs the same driver with the
second stage forced onto the default stream and must fail. The timing depends on the GPUs
being otherwise idle; no checkpoint and no compiled extension are needed.

    python tests/test_dsv41_pipeline_stream_gpu_.py --source 0 --destination 1
"""

import argparse
import os
import sys
import types
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_pipeline, skip


class State:
    def __init__(self):
        self.position = self.window_beg = self.wshift = self.last_history = 0
        self.cache = types.SimpleNamespace(initialized = True)

    def post_advance(self):
        self.window_beg += self.wshift
        self.wshift = 0


def check(pipeline, source, destination, default_stream_worker = False):
    """Three prefills on fresh caller streams; returns the number whose reads were wrong."""
    if source == destination:
        raise ValueError("this regression requires two distinct CUDA devices")
    modules = [types.SimpleNamespace(device = device, modules = []) for device in (source, destination)]
    model = types.SimpleNamespace(
        fwd_modules = [(module, 0, i) for i, module in enumerate(modules)],
        modules = modules, first_block_idx = 0,
        loaded_tp = False, _get_prefetch_layers = [], config = types.SimpleNamespace(moe_cpu_hosts = {}),
        _dsv41_loaded_max_chunk = 4, last_kv_module_idx_instance = (1, 0),
        prepare_inputs = lambda ids, params: ids.float(),
    )
    cache_rows = torch.zeros(12, device = destination)
    expected = torch.tensor([1] * 4 + [5] * 4 + [9] * 4, device = "cpu", dtype = torch.float32)
    expected[-1] += 100

    def run(model, modules, x, params):
        if modules[0][2] == 1:
            start = params["recurrent_states"][0].position
            cache_rows[start:start + x.shape[1]].fill_(start + 1)
        return x

    current_stream = torch.cuda.current_stream
    if default_stream_worker:
        # What a driver that does not carry the caller's stream into its worker thread does
        current_stream = lambda device = None: torch.cuda.default_stream(device)

    wrong = 0
    with patch.object(pipeline, "PIPELINE", True), patch.object(pipeline, "SUB_CHUNK", 4), \
            patch.object(pipeline, "_run", run):
        # Warm both contexts and create the source side stream before timing control
        pipeline._plan(model)
        torch.cuda.synchronize(source)
        torch.cuda.synchronize(destination)
        for _ in range(3):
            cache_rows.zero_()
            torch.cuda.synchronize(destination)
            caller = torch.cuda.Stream(device = destination)
            with torch.cuda.device(destination), torch.cuda.stream(torch.cuda.default_stream(destination)):
                torch.cuda._sleep(500_000_000)
            state = State()
            params = dict(cache = state.cache, recurrent_states = [state],
                          cache_seqlens = torch.tensor([0], dtype = torch.int32),
                          block_table = torch.tensor([[0]], dtype = torch.int32))
            with torch.cuda.stream(caller):
                with patch.object(torch.cuda, "current_stream", current_stream):
                    pipeline.prefill_pipelined(model, torch.ones(1, 12, dtype = torch.long), params)
                # This models a decode/cache consumer queued immediately after prefill
                cache_rows[-1].add_(100)
                snapshot = cache_rows.clone()
            caller.synchronize()
            got = snapshot.cpu()
            # Finish the delayed default stream even on a failing negative control
            torch.cuda.synchronize(destination)
            assert state.position == 12
            if not torch.equal(got, expected):
                wrong += 1
    return wrong


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type = int, default = 0)
    parser.add_argument("--destination", type = int, default = 1)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        skip("test_dsv41_pipeline_stream_gpu_", "needs two CUDA devices")
        sys.exit(0)
    pipeline = load_pipeline()
    src, dst = torch.device("cuda", args.source), torch.device("cuda", args.destination)
    with torch.inference_mode():
        negative = check(pipeline, src, dst, default_stream_worker = True)
        assert negative, "negative control: a worker on the default stream was not caught"
        print(f"  OK  negative control: {negative}/3 prefills with the worker on the default stream read stale rows")
        wrong = check(pipeline, src, dst)
        assert wrong == 0, f"{wrong}/3 prefills on a caller stream read stale rows"
    print("PASS pipeline: three caller streams preserve prefill-to-decode cache ordering")
