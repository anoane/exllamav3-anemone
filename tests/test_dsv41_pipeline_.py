"""CPU-only contract checks of the DeepSeek-V4.1 pipelined prefill driver.

Loads exllamav3/architecture/dsv41/pipeline.py by path (no compiled extension) and replaces
the CUDA stream, event and device objects with stand-ins, so the eligibility rules, the
schedule of the two stages, the job-state snapshots, the failure poisoning, the event waits,
the allocator ownership records, the threads of prefetches and CPU-MoE passes and the stream
ownership run without a GPU. The model is a two-module stand-in (one module per device).
CrossingTests also cover EXL3_DSV41_XDEV_BF16 in the block that receives the streams at the
split (DSV41Block.prepare_for_device, compiled from the source and run against stand-ins).

    python tests/test_dsv41_pipeline_.py
    pytest tests/test_dsv41_pipeline_.py
"""
import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file, load_pipeline

P = load_pipeline()
spec = importlib.util.spec_from_file_location("stable_helpers", Path(__file__).with_name("test_stable_arithmetic_.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class Event:
    created = []                    # every Event in creation order (reset per test)

    def __init__(self):
        Event.created.append(self)
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream


class Stream:
    def __init__(self, device):
        self.device = torch.device(device)
        self.waits = []
        self.stream_waits = []          # streams this one was made to wait for (wait_stream)

    def wait_event(self, event):
        self.waits.append(event)

    def wait_stream(self, stream):
        self.stream_waits.append(stream)


class State:
    def __init__(self):
        self.position = self.window_beg = self.wshift = self.last_history = 0
        self.cache = types.SimpleNamespace(initialized = True)

    def post_advance(self):
        self.window_beg += self.wshift
        self.wshift = 0


def model():
    first = types.SimpleNamespace(device = "cuda:0", modules = [])
    second = types.SimpleNamespace(device = "cuda:1", modules = [])
    return types.SimpleNamespace(
        fwd_modules = [(first, 0, 0), (second, 0, 1)], loaded_tp = False,
        modules = [first, second], first_block_idx = 0,
        _get_prefetch_layers = [], config = types.SimpleNamespace(moe_cpu_hosts = {}),
        _dsv41_loaded_max_chunk = 4, last_kv_module_idx_instance = (1, 0),
        prepare_inputs = lambda ids, p: ids.float(),
    )


def params(state):
    return dict(cache = state.cache, recurrent_states = [state],
                cache_seqlens = torch.tensor([0], dtype = torch.int32),
                block_table = torch.tensor([[0]], dtype = torch.int32))


def stage_of(mods):
    return 1 if mods[0][2] == 0 else 2


class PipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cuda_initialized = torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        if not cls.cuda_initialized:
            assert not torch.cuda.is_initialized(), "CPU contract tests initialized CUDA"

    def setUp(self):
        # The path is opt-in and the CUDA objects must never be touched: enable it with a
        # 4-token sub-chunk and replace every CUDA entry point the driver uses
        Event.created = []
        for target, name, value in (
            (P, "PIPELINE", True),
            (P, "SUB_CHUNK", 4),
            (torch.cuda, "Stream", Stream),
            (torch.cuda, "Event", Event),
            (torch.cuda, "device", lambda device: contextlib.nullcontext()),
            (torch.cuda, "stream", lambda stream: contextlib.nullcontext()),
            (torch.cuda, "current_stream", lambda device: Stream(device)),
        ):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_chunk_parsed_only_when_enabled(self):
        # A malformed sub-chunk must not break the default path, and must fail when enabled
        pkg = "_dsv41_pipeline_pkg.architecture.dsv41"
        for i, (env, ok) in enumerate((
            ({"EXL3_DSV41_PIPELINE": "0", "EXL3_DSV41_PIPELINE_CHUNK": "invalid"}, True),
            ({"EXL3_DSV41_PIPELINE": "1", "EXL3_DSV41_PIPELINE_CHUNK": "invalid"}, False),
            ({"EXL3_DSV41_PIPELINE": "1", "EXL3_DSV41_PIPELINE_CHUNK": "0"}, False),
            ({"EXL3_DSV41_PIPELINE": "1", "EXL3_DSV41_PIPELINE_CHUNK": "2048"}, True),
        )):
            with self.subTest(env = env), patch.dict(os.environ, env):
                load = lambda: load_package_file("exllamav3/architecture/dsv41/pipeline.py",
                                                 f"{pkg}.pipeline_env{i}", pkg)
                if ok:
                    m = load()
                    self.assertEqual(m.SUB_CHUNK, None if env["EXL3_DSV41_PIPELINE"] == "0" else 2048)
                else:
                    self.assertRaises(ValueError, load)
        for value in (0, -1, "invalid", None):
            self.assertRaises(ValueError, P._positive_chunk, value)
        self.assertEqual(P._positive_chunk("4"), 4)

    def test_preconditions(self):
        m, state = model(), State()
        ids, p = torch.ones(1, 12, dtype = torch.long), params(state)
        self.assertTrue(P.eligible(m, ids, p))
        # Longer than one sub-chunk is enough: the last piece may be shorter
        self.assertTrue(P.eligible(m, ids[:, :5], p))
        self.assertFalse(P.eligible(m, ids[:, :4], p))
        self.assertFalse(P.eligible(m, torch.ones(2, 12, dtype = torch.long), p))
        for key in P._PLAIN_KEYS:
            # More than one element detects accidental bool(tensor) evaluation
            self.assertFalse(P.eligible(m, ids, dict(p, **{key: torch.ones(2)})), key)
            self.assertFalse(P.eligible(m, ids, dict(p, **{key: [1]})), key)
            # Unset, the way callers leave it: None, False, 0 or an empty container
            for absent in (None, False, 0, [], {}):
                self.assertTrue(P.eligible(m, ids, dict(p, **{key: absent})), (key, absent))
        # The generator's prefill params for a text-only job
        generator = dict(p, attn_mode = "flash_attn", indexed_embeddings = [], inv_freq = None,
                         mm_span_prefix = 0)
        self.assertTrue(P.eligible(m, ids, generator))
        self.assertFalse(P.eligible(m, ids, dict(generator, indexed_embeddings = [object()])))
        self.assertFalse(P.eligible(m, ids, dict(p, recurrent_history = True)))
        self.assertFalse(P.eligible(m, ids, dict(p, recurrent_states = None)))
        self.assertFalse(P.eligible(m, ids, {k: v for k, v in p.items() if k != "block_table"}))
        with patch.object(P, "PIPELINE", False):
            self.assertFalse(P.eligible(m, ids, p))
        m.loaded_tp = True
        self.assertFalse(P.eligible(m, ids, p))
        m = model()
        m._dsv41_loaded_max_chunk = 2
        self.assertRaises(ValueError, P.eligible, m, ids, p)
        m._dsv41_loaded_max_chunk = None
        self.assertFalse(P.eligible(m, ids, p))
        # CPU-MoE layers in the first half
        m = model()
        m.fwd_modules[0][0].modules = [types.SimpleNamespace(cpu_host = object())]
        self.assertFalse(P.eligible(m, ids, p))
        # Repeated layer instances
        m = model()
        m.fwd_modules.append(m.fwd_modules[1])
        self.assertFalse(P.eligible(m, ids, p))
        # One device, three devices, a module of the second half back on the first device
        for devices in (("cuda:0", "cuda:0"), ("cuda:0", "cuda:1", "cuda:2"), ("cuda:0", "cuda:1", "cuda:0")):
            m = model()
            m.fwd_modules = [(types.SimpleNamespace(device = d, modules = []), 0, i)
                             for i, d in enumerate(devices)]
            m.last_kv_module_idx_instance = (len(devices) - 1, 0)
            self.assertFalse(P.eligible(m, ids, p), devices)
        # The split must come before the last cache-writing module: with the split after it,
        # the prefill never reaches the second device
        m = model()
        m.fwd_modules.insert(1, (types.SimpleNamespace(device = "cuda:0", modules = []), 0, 1))
        m.fwd_modules[2] = (m.fwd_modules[2][0], 0, 2)
        m.last_kv_module_idx_instance = (1, 0)
        self.assertFalse(P.eligible(m, ids, p))
        m.last_kv_module_idx_instance = (2, 0)
        self.assertTrue(P.eligible(m, ids, p))
        m.last_kv_module_idx_instance = None
        self.assertFalse(P.eligible(m, ids, p))
        # No decoder layer before the split: an unplaced autosplit can put layer 0 on the second
        # GPU and leave only the embedding (CPU) and the stream expansion on the first. Stage 1
        # would own no sliding-window ring, so the ring-shift check could only fail
        m = model()
        embed, expand = (types.SimpleNamespace(device = d, modules = []) for d in ("cpu", "cuda:0"))
        blocks = [types.SimpleNamespace(device = "cuda:1", modules = []) for _ in range(2)]
        m.modules, m.first_block_idx = [embed, expand] + blocks, 2
        m.fwd_modules = [(mod, 0, i) for i, mod in enumerate(m.modules)]
        m.last_kv_module_idx_instance = (3, 0)
        self.assertFalse(P.eligible(m, ids, p))
        self.assertEqual(P._plan(m).reason, "no decoder layer before the split")
        blocks[0].device = "cuda:0"
        self.assertTrue(P.eligible(m, ids, p))
        self.assertIsNone(P._plan(m).reason)
        # The cached plan follows the loaded modules
        m = model()
        original = P._plan(m)
        m.fwd_modules[1][0].device = "cuda:2"
        self.assertIsNot(P._plan(m), original)
        self.assertEqual(P._plan(m).dev2, torch.device("cuda:2"))

    def test_stage_view(self):
        state = State()
        state.position, state.window_beg = 100, 64
        view = P._StageView(state, 4, 2)
        view.position, view.window_beg, view.wshift, view.last_history = 5, 3, 1, 2
        self.assertEqual((state.position, state.window_beg, state.wshift, state.last_history), (100, 64, 0, 0))
        self.assertIs(view.cache, state.cache)

    def test_schedule_and_failure(self):
        calls = []

        def run(m, mods, x, p):
            state = p["recurrent_states"][0]
            calls.append((stage_of(mods), state.position, state.window_beg, x.shape[1]))
            state.wshift = 2
            return x

        for length, spans in ((12, [(0, 0, 4), (4, 2, 4), (8, 4, 4)]), (9, [(0, 0, 4), (4, 2, 4), (8, 4, 1)])):
            with self.subTest(length = length):
                calls.clear()
                m, state = model(), State()
                with patch.object(P, "_run", run):
                    P.prefill_pipelined(m, torch.ones(1, length, dtype = torch.long), params(state))
                self.assertEqual((state.position, state.window_beg), (length, 2 * len(spans)))
                for stage in (1, 2):
                    self.assertEqual([c[1:] for c in calls if c[0] == stage], spans)

        ids = torch.ones(1, 12, dtype = torch.long)
        for failed_stage in (1, 2):
            with self.subTest(failed_stage = failed_stage):
                m, state = model(), State()

                def fail(m, mods, x, p):
                    if stage_of(mods) == failed_stage:
                        raise RuntimeError("injected failure")
                    return run(m, mods, x, p)

                with patch.object(P, "_run", fail):
                    self.assertRaises(RuntimeError, P.prefill_pipelined, m, ids, params(state))
                # Only the job state is refused: the Cache stays usable for other jobs and for a
                # new state (the generator reaps the failed job and frees its pages)
                self.assertTrue(state.cache.initialized)
                self.assertRaises(RuntimeError, P.check_state, params(state))
                self.assertRaises(RuntimeError, P.eligible, m, ids, params(state))
                fresh_state = State()
                fresh_state.cache = state.cache
                P.check_state(params(fresh_state))
                self.assertTrue(P.eligible(m, ids, params(fresh_state)))

        # A failure in the last sub-chunk's second half only, found after the loop
        m, state = model(), State()

        def fail_last(m, mods, x, p):
            if stage_of(mods) == 2 and p["recurrent_states"][0].position == 8:
                raise RuntimeError("injected failure")
            return run(m, mods, x, p)

        with patch.object(P, "_run", fail_last):
            self.assertRaisesRegex(RuntimeError, "injected", P.prefill_pipelined, m, ids, params(state))
        self.assertRaises(RuntimeError, P.check_state, params(state))
        self.assertTrue(state.cache.initialized)

        # The two halves must shift the sliding-window ring alike
        m, state = model(), State()

        def uneven(m, mods, x, p):
            p["recurrent_states"][0].wshift = stage_of(mods)
            return x

        with patch.object(P, "_run", uneven):
            self.assertRaisesRegex(RuntimeError, "shifted the SWA ring by 2, the first by 1",
                                   P.prefill_pipelined, m, ids, params(state))
        self.assertRaises(RuntimeError, P.check_state, params(state))

    def test_position_mismatch_is_a_caller_error(self):
        # Found before any layer runs: a ValueError, and neither the state nor the Cache is marked
        m, state = model(), State()
        state.position = 8
        calls = []
        with patch.object(P, "_run", lambda m, mods, x, p: calls.append(mods) or x):
            self.assertRaisesRegex(ValueError, "job state at 8, cache_seqlens at 0",
                                   P.prefill_pipelined, m, torch.ones(1, 12, dtype = torch.long), params(state))
        self.assertEqual(calls, [])
        self.assertTrue(state.cache.initialized)
        P.check_state(params(state))

    def test_driver_ordering_and_threads(self):
        # What makes the overlap safe, observed through the stand-ins: stage 2 of each sub-chunk
        # waits (on the side stream) for the event its own stage 1 recorded, hands that chunk's
        # tensors to the allocator on the side stream, and runs in the worker; stage 1 runs
        # under the first GPU's device context on the calling thread; each half prefetches its
        # own engram layers in its own thread, and the CPU-MoE hosts begin one pass per stage 2
        main = threading.current_thread().name
        device = threading.local()

        @contextlib.contextmanager
        def device_context(d):
            previous = getattr(device, "current", None)
            device.current = torch.device(d)
            try:
                yield
            finally:
                device.current = previous

        log = []

        class Prefetch:
            def __init__(self, dev):
                self.device = dev

            def prefetch(self, x, p):
                log.append(("prefetch", self.device, threading.current_thread().name,
                            p["recurrent_states"][0].position))

        class Host:
            def begin_pass(self):
                log.append(("pass", None, threading.current_thread().name, None))

        m = model()
        m._get_prefetch_layers = [Prefetch("cuda:0"), Prefetch("cuda:1")]
        m.config.moe_cpu_hosts = {0: Host()}
        plan = P._plan(m)
        recorded, event_at = [], {}

        def run(m, mods, x, p):
            position = p["recurrent_states"][0].position
            if stage_of(mods) == 1:
                self.assertEqual(threading.current_thread().name, main)
                self.assertEqual(device.current, plan.dev1)
                event_at[position] = len(Event.created)       # the event stage 1 records next
            else:
                self.assertNotEqual(threading.current_thread().name, main)
                # A new thread does not inherit inference mode: stage 2 must enter it itself
                self.assertTrue(torch.is_inference_mode_enabled())
                event = Event.created[event_at[position]]
                self.assertEqual(event.recorded_on.device, plan.dev1)
                self.assertIs(plan.side1.waits[-1], event)
                (ids, x_seen, p_seen), stream = recorded[-1]
                self.assertIs(stream, plan.side1)
                self.assertIs(x_seen, x)
                self.assertIs(p_seen, p)
                self.assertEqual(ids.shape[1], x.shape[1])
            log.append(("run", stage_of(mods), threading.current_thread().name, position))
            return x

        with patch.object(torch.cuda, "device", device_context), patch.object(P, "_run", run), \
                patch.object(P, "_record_source_tensors", lambda value, stream: recorded.append((value, stream))):
            P.prefill_pipelined(m, torch.ones(1, 10, dtype = torch.long), params(State()))
        self.assertEqual(len(recorded), 3)
        self.assertEqual(len(plan.side1.waits), 3)
        pf1 = [(t, pos) for kind, d, t, pos in log if kind == "prefetch" and d == "cuda:0"]
        pf2 = [(t, pos) for kind, d, t, pos in log if kind == "prefetch" and d == "cuda:1"]
        self.assertEqual(pf1, [(main, 0), (main, 4), (main, 8)])
        self.assertEqual([pos for _, pos in pf2], [0, 4, 8])
        self.assertTrue(all(t != main for t, _ in pf2))
        passes = [t for kind, _, t, _ in log if kind == "pass"]
        self.assertEqual(len(passes), 3)
        self.assertTrue(all(t != main for t in passes))

    def test_record_ownership(self):
        source = torch.empty(4, device = "meta")
        other = torch.empty(4)
        stream = types.SimpleNamespace(device = torch.device("meta"))
        nested = {"x": source, "twice": [source, (other,)]}
        nested["cycle"] = nested
        calls = []
        with patch.object(torch.Tensor, "record_stream", lambda t, s: calls.append((id(t), s))):
            P._record_source_tensors(nested, stream)
        self.assertEqual(calls, [(id(source), stream)])

    def test_caller_destination_stream(self):
        # A worker begins with default CUDA streams, not its caller's current ones. Model that
        # thread-local property without initializing CUDA, and reuse the same plan with two
        # caller streams to catch a cached-stream implementation
        current = threading.local()
        defaults = {torch.device(f"cuda:{i}"): Stream(f"cuda:{i}") for i in (0, 1)}

        def get_stream(device):
            return getattr(current, "streams", {}).get(torch.device(device), defaults[torch.device(device)])

        @contextlib.contextmanager
        def stream_context(stream):
            if not hasattr(current, "streams"):
                current.streams = {}
            previous = current.streams.get(stream.device)
            current.streams[stream.device] = stream
            try:
                yield
            finally:
                if previous is None:
                    current.streams.pop(stream.device)
                else:
                    current.streams[stream.device] = previous

        m = model()
        plan = P._plan(m)
        seen = []

        def run(m, mods, x, p):
            if stage_of(mods) == 2:
                self.assertIs(get_stream(plan.dev1), plan.side1)
                seen.append(get_stream(plan.dev2))
            return x

        with patch.object(torch.cuda, "current_stream", get_stream), \
                patch.object(torch.cuda, "stream", stream_context), patch.object(P, "_run", run):
            for _ in range(2):
                destination = Stream("cuda:1")
                seen.clear()
                with stream_context(destination):
                    P.prefill_pipelined(m, torch.ones(1, 12, dtype = torch.long), params(State()))
                    self.assertIs(get_stream(plan.dev2), destination)
                self.assertEqual(len(seen), 3)
                self.assertTrue(all(stream is destination for stream in seen))

            # Stage 2 reads persistent first-GPU Cache rows on the side stream, which allocator
            # records do not cover: when the call returns, after a failure too, the caller's
            # current first-GPU stream (here the default one) must wait for the side stream
            first = defaults[plan.dev1]
            first.stream_waits.clear()
            P.prefill_pipelined(m, torch.ones(1, 12, dtype = torch.long), params(State()))
            self.assertEqual(first.stream_waits, [plan.side1])

            def fail(m, mods, x, p):
                if stage_of(mods) == 2:
                    raise RuntimeError("injected failure")
                return x

            first.stream_waits.clear()
            with patch.object(P, "_run", fail):
                self.assertRaises(RuntimeError, P.prefill_pipelined, m,
                                  torch.ones(1, 12, dtype = torch.long), params(State()))
            self.assertEqual(first.stream_waits, [plan.side1])


class CrossingTests(unittest.TestCase):
    """EXL3_DSV41_XDEV_BF16: what crosses the split, in the block and in the pipelined prefill"""

    class X:
        """A tensor on a stated device (the check never touches CUDA)"""
        def __init__(self, t, device):
            self.t, self.device, self.dtype = t, torch.device(device), t.dtype

        def to(self, dtype):
            return CrossingTests.X(self.t.to(dtype), self.device)

        def float(self):
            return self.to(torch.float)

    def block(self, switch):
        moved, fallback = [], []
        ns = dict(torch = torch, dsv41_placement = types.SimpleNamespace(XDEV_BF16 = switch),
                  to_device = lambda x, device: moved.append((x.dtype, str(device))) or self.X(x.t, device),
                  super = lambda: types.SimpleNamespace(
                      prepare_for_device = lambda x, p: fallback.append(x.dtype) or x))
        method = helpers.method("exllamav3/modules/dsv41_block.py", "DSV41Block", "prepare_for_device", ns)
        return (lambda x: method(types.SimpleNamespace(device = torch.device("cuda:1")), x, {})), moved, fallback

    def test_block_rounds_and_widens_only_under_the_switch(self):
        t = torch.tensor([1.0 + 2 ** -12, -3.0e38, 1.0e-30, 0.1])
        prepare, moved, fallback = self.block(True)
        y = prepare(self.X(t, "cuda:0"))
        self.assertEqual(moved, [(torch.bfloat16, "cuda:1")])
        self.assertEqual((y.dtype, str(y.device)), (torch.float, "cuda:1"))
        self.assertTrue(torch.equal(y.t, t.to(torch.bfloat16).float()))
        self.assertFalse(torch.equal(y.t, t))
        # a BF16 input (rounded by the pipelined prefill's first stage) is only widened
        moved.clear()
        y = prepare(self.X(t.to(torch.bfloat16), "cuda:0"))
        self.assertEqual(moved, [(torch.bfloat16, "cuda:1")])
        self.assertTrue(torch.equal(y.t, t.to(torch.bfloat16).float()))
        # not a GPU-to-GPU crossing of the FP32 streams: the plain move
        moved.clear()
        for x in (self.X(t, "cuda:1"), self.X(t, "cpu"), self.X(t.half(), "cuda:0")):
            self.assertIs(prepare(x), x)
        self.assertEqual(moved, [])
        self.assertEqual(fallback, [torch.float, torch.float, torch.half])
        # without the switch nothing is rounded or widened, BF16 input included
        prepare, moved, fallback = self.block(False)
        for x in (self.X(t, "cuda:0"), self.X(t.to(torch.bfloat16), "cuda:0")):
            self.assertIs(prepare(x), x)
        self.assertEqual(moved, [])
        self.assertEqual(fallback, [torch.float, torch.bfloat16])

    def test_pipeline_rounds_at_the_end_of_stage_1_only_under_the_switch(self):
        for target, name, value in (
            (P, "PIPELINE", True), (P, "SUB_CHUNK", 4),
            (torch.cuda, "Stream", Stream), (torch.cuda, "Event", Event),
            (torch.cuda, "device", lambda device: contextlib.nullcontext()),
            (torch.cuda, "stream", lambda stream: contextlib.nullcontext()),
            (torch.cuda, "current_stream", lambda device: Stream(device)),
        ):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        for switch in (False, True):
            with self.subTest(switch = switch):
                seen = []

                def run(m, mods, x, p):
                    if stage_of(mods) == 1:
                        return x + 1 / 3
                    seen.append(x)
                    return x

                with patch.object(P.placement, "XDEV_BF16", switch), patch.object(P, "_run", run):
                    P.prefill_pipelined(model(), torch.ones(1, 12, dtype = torch.long), params(State()))
                self.assertEqual(len(seen), 3)
                expect = torch.full((1, 4), 1 + 1 / 3)
                for x in seen:
                    if switch:
                        self.assertEqual(x.dtype, torch.bfloat16)
                        self.assertTrue(torch.equal(x, expect.to(torch.bfloat16)))
                    else:
                        self.assertEqual(x.dtype, torch.float)
                        self.assertTrue(torch.equal(x, expect))


if __name__ == "__main__":
    unittest.main()
