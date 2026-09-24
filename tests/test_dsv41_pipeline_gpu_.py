"""Two-GPU allocator lifetime stress for the pipelined prefill's split crossing.

A tensor produced on the first GPU's default stream and read by the second stage's side
stream must not be handed back to the caching allocator while that read is still queued.
The pipeline records such tensors on the side stream (_record_source_tensors); this test
delays the read and then reallocates, in fp32 and bf16, with direct copies.

A deliberately unprotected negative control must see corrupted reads, so that a setup where
something else happens to synchronize cannot pass without exercising the hazard. Host-bounced
copies (tensor.cpu().to(dst), as util/device_copy does without peer access) block the worker
until the delayed read has finished, so they carry no reuse hazard: their trials only check
that the bounced path delivers intact data, and cannot fail from early reuse. The test runs
on PyTorch's default allocator settings, not the expandable segments exllamav3 turns on at
import (it does not import the package); record_stream behaves the same in both. The timing
depends on the GPUs being otherwise idle. This is a scheduling/lifetime test, not a model
quality test; it needs no checkpoint and no compiled extension.

    python tests/test_dsv41_pipeline_gpu_.py
"""
import gc
import json
import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_pipeline, skip

_record_source_tensors = load_pipeline()._record_source_tensors


def trial(dtype, protect, bounce = False):
    for device in (0, 1):
        torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    src, dst = torch.device("cuda:0"), torch.device("cuda:1")
    producer, consumer = torch.cuda.default_stream(src), torch.cuda.Stream(device = src)
    with torch.cuda.device(src), torch.cuda.stream(producer):
        value = torch.full((1 << 20,), 7, dtype = dtype, device = src)
        pointer = value.data_ptr()
        ready = torch.cuda.Event()
        ready.record(producer)
    box = {}

    def worker(tensor):
        try:
            with torch.inference_mode(), torch.cuda.stream(consumer), torch.cuda.device(dst):
                consumer.wait_event(ready)
                if protect:
                    _record_source_tensors({"x": tensor, "metadata": [tensor]}, consumer)
                with torch.cuda.device(src):
                    torch.cuda._sleep(100_000_000)
                box["out"] = tensor.cpu().to(dst) if bounce else tensor.to(dst)
        except BaseException as error:
            box["error"] = error

    thread = threading.Thread(target = worker, args = (value,))
    thread.start()
    thread.join()
    del value, thread
    if "error" in box:
        raise box["error"]
    with torch.cuda.device(src), torch.cuda.stream(producer):
        churn = [torch.full((1 << 20,), 9, dtype = dtype, device = src) for _ in range(8)]
    reused = any(t.data_ptr() == pointer for t in churn)
    for device in (0, 1):
        torch.cuda.synchronize(device)
    wrong = int((box["out"] != 7).sum().item())
    return dict(dtype = str(dtype), protected = protect, bounce = bounce, reused = reused, wrong = wrong)


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        skip("test_dsv41_pipeline_gpu_", "needs two CUDA devices")
        sys.exit(0)
    for dtype in (torch.float32, torch.bfloat16):
        negatives = [trial(dtype, False) for _ in range(4)]
        print(json.dumps(negatives), flush = True)
        assert any(r["wrong"] for r in negatives), "negative control did not exercise early allocation reuse"
        positives = [trial(dtype, True) for _ in range(4)]
        print(json.dumps(positives), flush = True)
        assert all(r["wrong"] == 0 for r in positives), "pipeline crossing corrupted protected storage"
        # host-synchronous, so no hazard to protect against: a delivery check only
        bounced = [trial(dtype, True, bounce = True) for _ in range(2)]
        print(json.dumps(bounced), flush = True)
        assert all(r["wrong"] == 0 for r in bounced), "host-bounced copy delivered wrong data"
    print("ALL OK: source storage survives delayed direct copies in fp32 and bf16 (and bounced "
          "copies deliver intact data)")
