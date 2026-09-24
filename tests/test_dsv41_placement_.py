"""
CPU-only checks for where a layer-split load of DeepSeek-V4.1 may change device
(architecture/dsv41/placement.py): the crossings of every cut and the free cuts,
the load guard's refusal of a cut kv group while a Cache is attached -- through
exllamav3's REAL autosplit loop (Model_LSMixin._load_autosplit) with CUDA stubbed
out of it -- and the post-load check.

No GPU memory is allocated and no kernel runs: devices are torch.device
objects, module loads are replaced by bookkeeping that raises
OutOfMemoryError when a simulated budget runs out, model_ls's CUDA calls are
stubbed, and the CPU worker is never spawned. (Importing exllamav3 still
initializes the CUDA runtime, as it does for every test that imports it.)

    PYTHONPATH=$PWD python3 tests/test_dsv41_placement_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

Part A needs only the checkpoint's config.json; part B also needs a working
exllamav3 import (the built extension; nothing is compiled).
"""
import contextlib, importlib.util, io, json, os, sys, types

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip


def load_by_path(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raises(exc, fn, *a, match = None, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        if match is not None:
            assert match in str(e), f"{exc.__name__} raised, but {match!r} not in: {e}"
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


# ---------------------------------------------------------------------------
# A. pure logic (stdlib; placement.py and cache_plan.py loaded by path)

def part_a(path):
    P = load_by_path("_placement", "exllamav3/architecture/dsv41/placement.py")
    CP = load_by_path("_cache_plan", "exllamav3/architecture/dsv41/cache_plan.py")
    topo = P.Topology.from_config_dict(json.load(open(os.path.join(path, "config.json"))))
    n = topo.num_hidden_layers

    # 1. what each cut splits, on the real config
    free = P.free_cuts(topo)
    assert free == [1, 2, 8, 14, 20], free
    pools, topk, cand = P.crossings(topo, 12)
    assert pools == {(8, 12), (8, 13)} and topk == {(8, 12), (8, 13)} and cand == set(), (pools, topk, cand)
    pools, topk, cand = P.crossings(topo, 28)
    assert pools == {(20, i) for i in range(28, 40)}, pools
    assert topk == set(), topk                                     # 28 is an index source
    assert cand == {(20, 28), (20, 32), (20, 36)}, cand
    pools, topk, cand = P.crossings(topo, 29)
    assert topk == {(28, 29), (28, 30), (28, 31)} and cand == {(20, 32), (20, 36)}, (topk, cand)
    for c in free:
        assert P.crossings(topo, c) == (set(), set(), set()), c
    # on V4.1-Flash every other cut separates a kv group from its pool
    assert all(P.crossings(topo, c)[0] for c in range(1, n) if c not in free)
    for bad in (0, n):
        raises(ValueError, P.crossings, topo, bad, match = f"1..{n - 1}")
    assert P.format_layer_ranges({12, 13, 14, 20, 22, 23}) == "12-14,20,22-23"
    assert P.format_cuts(free) == "layer 1, 2, 8, 14 or 20"
    assert P.format_cuts([14]) == "layer 14" and P.format_cuts([]) == "no layer"
    print(f"  OK  crossings: free cuts {free}; a cut at 12 splits pool 8 and its top-k from layers "
          f"12-13, at 28 pool 20 and the candidates of 28/32/36, at 29 also 28's top-k")

    # 2. the message a refused cut gets: what is split, the free cuts, both ways to move it
    m12 = P.cut_message(topo, 12, "cuda:0", "cuda:1")
    for want in ("DeepSeek-V4.1: the layer split put layers.12 on cuda:1, but layers 8-13 share the "
                 "compressed-KV pool of layers.8, which is on cuda:0",
                 "at layer 1, 2, 8, 14 or 20 on this model",
                 "(-gs / use_per_device / reserve_per_device): less room on cuda:0, so that it holds "
                 "layers up to 7 only, or room for layers 12-13 as well, so that the change falls at "
                 "layer 14",
                 'doc/env_vars.md, DeepSeek-V4.1, "Loading across GPUs"'):
        assert want in m12, (want, m12)
    m24 = P.cut_message(topo, 24, "cuda:1", "cuda:2")
    assert "layers 20-39 share the compressed-KV pool of layers.20, which is on cuda:1" in m24, m24
    assert ("less room on cuda:1, so that it holds layers up to 19 only, or room for layers 24-39 as "
            "well, so that no change of device is left") in m24, m24
    # the other two kinds of crossing, on topologies that have them alone: candidate blocks
    # (a candidate source that is no kv source) and a top-k selection (a kv source that is no
    # index source, which DeepseekV41Config refuses but a hand-built Topology allows)
    t_cand = P.Topology(16, [0, 0] + [2] * 14, [2, 10], [2, 6, 10], 6)
    assert P.crossings(t_cand, 10) == (set(), set(), {(6, 10)}) and P.free_cuts(t_cand) == [1, 2]
    mc = P.cut_message(t_cand, 10, "cuda:0", "cuda:1")
    assert ("layers 10 mask their index scores to the candidate blocks of layers.6, which is on "
            "cuda:0") in mc and "so that it holds layers up to 1 only" in mc, mc
    t_topk = P.Topology(12, [0, 0] + [2] * 10, [2, 6], [2, 4])
    assert P.crossings(t_topk, 6) == (set(), {(4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11)}, set())
    mt = P.cut_message(t_topk, 6, "cuda:0", "cuda:1")
    assert "layers 6-11 reuse the top-k selection of layers.4, which is on cuda:0" in mt, mt
    print("  OK  refusal message: the split pool (or selection, or candidate blocks), the free cuts, "
          "less room below or more room above")

    # 3. cache plan: a ring on every layer, a pool on the kv sources only
    cp = CP.CachePlan(topo.compress_ratios, topo.kv_source_layer_ids)
    assert all(cp.owns_ring(i) for i in range(n))
    assert [i for i in range(n) if cp.owns_pool(i)] == [2, 8, 14, 20] and cp.allocation_count() == 4
    assert cp.owner == [topo.kv_source_for(i) for i in range(n)]
    assert not cp.owns_pool(0) and cp.owner[0] is None           # sliding: ring only
    print(f"  OK  cache plan: {cp.describe()}")
    return P, topo


# ---------------------------------------------------------------------------
# B. DSV41MoE and the real autosplit loop (exllamav3 import, no CUDA calls)

GIB = 1 << 30
RESIDENT = 5_222_576_916      # header scan of DeepSeek-V4.1-Flash 3.0 bpw: one fully resident layer
CPU_KEEP = 126_492_948        # a CPU-offloaded layer's GPU part, suh/svh included


def fake_ip(mcl = 0, mcs = 0):
    return types.SimpleNamespace(moe_cpu_offload = mcl, moe_cpu_split = mcs,
                                 draft_moe_cpu_offload = 0, moe_cpu_offload_assigned = {},
                                 moe_cpu_component = "text", vision_pinned = False)


def build(path, out = None):
    from exllamav3 import Config, Model
    with contextlib.redirect_stdout(out if out is not None else io.StringIO()):
        cfg = Config.from_directory(path)
        model = Model.from_config(cfg)
    return cfg, model


@contextlib.contextmanager
def patched(obj, name, value):
    had = name in vars(obj)
    old = vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def claim_everything(mlps):
    """load_cpu_offload stand-in: claims without a worker, like the real one sets cpu_offload."""
    for m in mlps:
        def fake(device, _m = m, **kw):
            _m.cpu_offload = True
            return True
        m.load_cpu_offload = fake


def part_b(path, P):
    import torch
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.dsv41_moe import DSV41MoE
    from exllamav3.cache.dsv41 import DSV41State
    cuda0, cuda1 = torch.device("cuda", 0), torch.device("cuda", 1)

    # 3. -mcl / -mcs are upstream's: DSV41MoE does not override the claims, so the first N
    # eligible layers go to the CPU worker and -mcs splits every layer
    cfg, model = build(path)
    blocks = model._blocks()
    mlps = [b.mlp for b in blocks]
    assert all(type(m) is DSV41MoE for m in mlps) and model.recurrent_state_cls is DSV41State
    assert all(m.load_guard is model.load_guard for m in mlps)
    assert model.load_guard.free == [1, 2, 8, 14, 20], model.load_guard.free
    for name in ("cpu_maybe_offload_load", "cpu_maybe_split_load"):
        assert getattr(DSV41MoE, name) is getattr(BlockSparseMLP, name), name
    cfg.infer_params = fake_ip(mcl = 11)
    claim_everything(mlps)
    got = [i for i, m in enumerate(mlps) if m.cpu_maybe_offload_load(cuda1)]
    assert got == list(range(0, 11)), got
    assert cfg.infer_params.moe_cpu_offload_assigned == {"text": 11}
    print(f"  OK  -mcl 11 claims layers {got[0]}-{got[-1]}, as for every MoE model")

    # 4. the guard itself: a change of device off the free cuts is refused only with a Cache,
    # before the block's MoE loads anything
    g = model.load_guard
    calls = []
    with patched(BlockSparseMLP, "load", lambda self, device, **kw: calls.append((self.layer_idx, device))):
        mlps[5].load(cuda0)                               # guard inactive: no checks at all
        mlps[6].load(cuda1)
        for cache in (False, True):
            with patched(model, "_cache_attached", lambda c = cache: c):
                g.begin()
                try:
                    for i in range(12):
                        mlps[i].load(cuda0)
                    if cache:
                        raises(RuntimeError, mlps[12].load, cuda1,
                               match = "the layer split put layers.12 on cuda:1, but layers 8-13 share "
                                       "the compressed-KV pool of layers.8, which is on cuda:0")
                    else:
                        mlps[12].load(cuda1)
                    # a free cut takes a Cache: 14 after 13 on the first device
                    g.devices[13] = cuda0
                    mlps[14].load(cuda1)
                finally:
                    g.end()
    want = [(5, cuda0), (6, cuda1)] + [(i, cuda0) for i in range(12)] + [(12, cuda1), (14, cuda1)] + \
           [(i, cuda0) for i in range(12)] + [(14, cuda1)]
    assert calls == want, calls
    assert not g.active
    print("  OK  load guard: layers.12 after layers.11 on another device refused with a Cache and "
          "nothing of its MoE loaded; accepted without one; the free cut 14 accepted with one")

    # 5. the real autosplit loop
    run_autosplit_cases(path, P)

    # 6. tensor parallel is refused before anything loads; every other load goes to upstream
    raises(NotImplementedError, lambda: next(model.load_gen(use_per_device = [63, 91], tensor_p = True)),
           match = "DeepSeek-V4.1: tensor-parallel loading is not implemented; load it as a layer "
                   "split (-gs / use_per_device)")
    raises(NotImplementedError, mlps[3].make_tp_allocation, {}, match = "tensor-parallel")
    assert model.caps["supports_tp"] is False
    assert not model.load_guard.active
    print("  OK  tensor-parallel refused")
    # a pipelined-prefill sub-chunk the load's max_chunk_size does not cover
    from exllamav3.architecture.dsv41 import pipeline
    with patched(pipeline, "PIPELINE", True), patched(pipeline, "SUB_CHUNK", 8192):
        raises(ValueError, lambda: next(model.load_gen(use_per_device = [63, 91], max_chunk_size = 4096)),
               match = "EXL3_DSV41_PIPELINE_CHUNK=8192 exceeds this load's max_chunk_size=4096")
    # the switch turned on in-process without a sub-chunk (the variable was off at import)
    with patched(pipeline, "PIPELINE", True), patched(pipeline, "SUB_CHUNK", None):
        raises(ValueError, lambda: next(model.load_gen(use_per_device = [63, 91], max_chunk_size = 4096)),
               match = "SUB_CHUNK is not set")
    assert not model.load_guard.active and model._dsv41_loaded_max_chunk is None
    print("  OK  pipeline sub-chunk above the load's max_chunk_size, or unset with the switch on, "
          "refused before anything loads")

    # 7. every compressed layer reads its own kv source's pool, and only kv sources own one
    attn = [b.attn for b in blocks]
    for i, a in enumerate(attn):
        assert bool(a.caps.get("kv_cache")) == cfg.is_kv_source(i), i
        if a.compress_ratio:
            assert a.kv_source_layer == cfg.kv_source_for(i), i
    print("  OK  pools only on the kv sources, every compressed layer reads its own source's")

    # 8. the post-load check
    run_post_load_check(path)


# -- the real Model_LSMixin._load_autosplit, CUDA stubbed out ---------------

class _TorchNoCuda(types.ModuleType):
    """model_ls's view of torch: everything real except that tensors are made on the host."""
    def __init__(self, torch):
        super().__init__("torch")
        self._t = torch
    def __getattr__(self, k):
        return getattr(self._t, k)
    def zeros(self, *shape, dtype = None, device = None, **kw):
        return self._t.zeros(*shape, dtype = dtype)
    def device(self, *args, **kw):
        # a bare index names a CUDA device here, also on a host whose torch build has another
        # (or no) accelerator, where torch.device(0) would name that one instead
        if len(args) == 1 and isinstance(args[0], int) and not kw:
            return self._t.device("cuda", args[0])
        return self._t.device(*args, **kw)


class FakeBlock:
    """Stands in for a DSV41Block in the autosplit loop: loading it runs the REAL DSV41MoE.load
    (guard, claims) and books the layer's header-scan size against its device's budget."""
    def __init__(self, sim, mlp):
        self.sim, self.mlp = sim, mlp
        self.key = f"layers.{mlp.layer_idx}"
        self.caps = {}
        self.device = None
        self.modules = []
        self.size = 0
    def __iter__(self):
        yield self
    def can_defer_load(self):
        return False
    def load(self, device, **kw):
        self.device = device
        self.mlp.load(device, **kw)
    def unload(self):
        if self.size:
            self.sim.used[self.device] -= self.size
            self.size = 0
        if self.mlp.cpu_offload:
            self.mlp.cpu_offload = False
            asn = self.mlp.config.infer_params.moe_cpu_offload_assigned
            asn["text"] -= 1
        self.device = None


def _sim_blocksparse_load(sim):
    import torch
    def load(self, device, **kw):
        # the order BlockSparseMLP.load uses: claim first, then the GPU load
        claimed = self.cpu_maybe_offload_load(device)
        size = CPU_KEEP if claimed else RESIDENT
        blk = sim.blocks[self.layer_idx]
        if sim.used.get(device, 0) + size > sim.budget[device]:
            if claimed:
                self.cpu_offload = False
                self.config.infer_params.moe_cpu_offload_assigned["text"] -= 1
            raise torch.cuda.OutOfMemoryError(f"sim: {device} budget exhausted at {blk.key}")
        sim.used[device] = sim.used.get(device, 0) + size
        blk.size = size
    return load


def autosplit(path, budgets_gib, mcl = 0, cache_attached = False):
    """Run model_ls._load_autosplit over 40 FakeBlocks wrapping the real DSV41MoE objects.
    Returns (device index per layer, claimed CPU layers) or raises what the loop raised."""
    import torch
    import exllamav3.model.model_ls as model_ls
    from exllamav3.model.model_ls import Model_LSMixin
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    cfg, model = build(path)
    cfg.infer_params = fake_ip(mcl = mcl)
    mlps = [b.mlp for b in model._blocks()]
    claim_everything(mlps)
    for m in mlps:
        m.cpu_offload = False

    sim = types.SimpleNamespace(used = {}, budget = {torch.device("cuda", i): int(b * GIB)
                                                     for i, b in enumerate(budgets_gib)})
    sim.blocks = [FakeBlock(sim, m) for m in mlps]

    class FakeModel(Model_LSMixin):
        def __init__(self):
            self.caps = {}
        def get_layer_instances(self, layer_idx):
            return [(layer_idx, 0)]
        def __iter__(self):
            return iter(sim.blocks)
    fake_cfg = types.SimpleNamespace(
        stc = types.SimpleNamespace(begin_deferred_load = lambda **kw: None,
                                    end_deferred_load = lambda: None,
                                    abort_deferred_load = lambda: None, close = lambda: None),
        infer_params = types.SimpleNamespace(vision_pinned = False))
    if cache_attached:
        model._cache_attached = lambda: True

    patches = [(model_ls, "torch", _TorchNoCuda(torch)),
               (model_ls, "set_memory_fraction_use", lambda use, i: use),
               (model_ls, "set_memory_fraction_reserve", lambda r, i: r),
               (model_ls, "unset_memory_fraction", lambda devs: None),
               (model_ls, "free_mem", lambda: None)]
    olds = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, v in patches:
        setattr(o, n, v)
    model.load_guard.begin()
    try:
        with patched(BlockSparseMLP, "load", _sim_blocksparse_load(sim)):
            for _ in FakeModel()._load_autosplit(
                    False, None, [int(b * GIB) for b in budgets_gib], list(range(len(budgets_gib))),
                    2048, 32, 1, None, False, fake_cfg, sim.blocks, False, 1, {}, True):
                pass
    finally:
        model.load_guard.end()
        for o, n, v in olds:
            setattr(o, n, v)
    devs = [b.device.index for b in sim.blocks]
    claimed = [i for i, m in enumerate(mlps) if m.cpu_offload]
    return devs, claimed


def cuts_of(devs):
    return [i for i in range(1, len(devs)) if devs[i] != devs[i - 1]]


def run_autosplit_cases(path, P):
    # budgets that land on a free cut load with a Cache: 70 GiB holds 14 resident layers
    # (14 x 4.864 = 68.1 GiB), so the change falls at 14
    for cache in (False, True):
        devs, _ = autosplit(path, [70, 200], cache_attached = cache)
        assert devs == [0] * 14 + [1] * 26, devs
    # 63 GiB holds 12: the change falls at 12, inside kv group 8-13. With a Cache that fails at
    # once, naming the group, the free cuts and both ways to move the change ...
    e = raises(RuntimeError, autosplit, path, [63, 200], cache_attached = True,
               match = "the layer split put layers.12 on cuda:1, but layers 8-13 share the "
                       "compressed-KV pool of layers.8, which is on cuda:0")
    assert "at layer 1, 2, 8, 14 or 20 on this model" in str(e), str(e)
    assert ("less room on cuda:0, so that it holds layers up to 7 only, or room for layers 12-13 "
            "as well, so that the change falls at layer 14") in str(e), str(e)
    # ... and without a Cache it loads (the stateless path copies the cut group's pool)
    devs, _ = autosplit(path, [63, 200])
    assert cuts_of(devs) == [12], devs
    # three devices, both changes on free cuts: 8 layers (38.9 GiB) on the first, 12 (58.4 GiB)
    # on the second, the rest on the third
    devs, _ = autosplit(path, [39, 58.5, 200], cache_attached = True)
    assert devs == [0] * 8 + [1] * 12 + [2] * 20, devs
    # ... and the second change moved off a free cut fails there, at the first layer past it
    raises(RuntimeError, autosplit, path, [39, 50, 200], cache_attached = True,
           match = "the layer split put layers.18 on cuda:2, but layers 14-19 share the "
                   "compressed-KV pool of layers.14, which is on cuda:1")
    # -mcl 11: upstream -- the FIRST 11 layers keep only their GPU part and the change falls where
    # the budget runs out, at 23 on [63, 91]; with a Cache that is inside group 20-39, which
    # needs no change of device at all past 20
    devs, claimed = autosplit(path, [63, 91], mcl = 11)
    assert claimed == list(range(11)) and cuts_of(devs) == [23], (claimed, cuts_of(devs))
    raises(RuntimeError, autosplit, path, [63, 91], mcl = 11, cache_attached = True,
           match = "less room on cuda:0, so that it holds layers up to 19 only, or room for layers "
                   "23-39 as well, so that no change of device is left")
    # a first device that holds nothing is no change of device between layers
    devs, _ = autosplit(path, [4, 200], cache_attached = True)
    assert devs == [1] * 40, devs
    print("  OK  real autosplit loop: -gs 70,200 changes device at the free cut 14 and loads with a "
          "Cache; -gs 63,200 cuts group 8-13 at 12: refused with a Cache, naming the fix, loaded "
          "without one; three devices at 8 and 20 load, 8 and 18 do not; -mcl 11 on 63,91 cuts at "
          "23 (upstream), refused with a Cache")


def run_post_load_check(path):
    import torch
    cfg, model = build(path)
    cuda0, cuda1, cuda2 = (torch.device("cuda", i) for i in range(3))
    def place(model_, *cuts):
        devs = [cuda0, cuda1, cuda2]
        for m in model_.modules:
            m.device = cuda0
        model_.modules[0].device = torch.device("cpu")
        for i, b in enumerate(model_._blocks()):
            b.device = devs[sum(1 for c in cuts if i >= c)]
        for m in model_.modules[model_.first_block_idx + cfg.num_hidden_layers:]:
            m.device = devs[len(cuts)]
    # a cut kv group without a Cache: one note, naming the layers and the free cuts
    place(model, 12)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        model._check_split()
    line = out.getvalue().strip()
    assert line == ("!! DSV41: layers 12-13 read a compressed-KV pool on another device; without a "
                    "Cache each forward copies those pools whole, and a Cache would need every "
                    "change of device at layer 1, 2, 8, 14 or 20"), line
    print(f"  OK  post-load note:{line[2:]}")
    # ... and with a Cache attached an error (the guard refuses this while loading; this is the
    # safety net for a load that went around it)
    with patched(model, "_cache_attached", lambda: True):
        raises(RuntimeError, model._check_split,
               match = "DeepSeek-V4.1 load check failed: layers 12-13 read a compressed-KV pool on "
                       "another device, which the cached path cannot do; reload with every change "
                       "of device at layer 1, 2, 8, 14 or 20")
    # changes of device at free cuts: silent, with or without a Cache
    for cuts in ((14,), (8, 20), ()):
        place(model, *cuts)
        for cache in (False, True):
            out = io.StringIO()
            with patched(model, "_cache_attached", lambda c = cache: c), contextlib.redirect_stdout(out):
                model._check_split()
            assert out.getvalue() == "", (cuts, out.getvalue())
    print("  OK  post-load check: an error with a Cache, silent at free cuts")


def main(path):
    P, _ = part_a(path)
    part_b(path, P)
    print("  OK  test_dsv41_placement_")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 placement", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
