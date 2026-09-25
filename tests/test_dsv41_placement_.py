"""
CPU-only checks for where a layer-split load of DeepSeek-V4.1 may change device
(architecture/dsv41/placement.py): the crossings of every cut and the free cuts,
the explicit placement (EXL3_PLACEMENT) as V4.1 reads it when the model is built
(its changes of device, the one allowed split and that split's pool routes and
replica), the load guard's refusal of a cut kv group while a Cache is attached --
through exllamav3's REAL autosplit loop (Model_LSMixin._load_autosplit) with CUDA
stubbed out of it -- the checks at load and the post-load check.

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

HINT = ("An explicit placement (EXL3_PLACEMENT / --placement) can instead change device inside one kv "
        "group; the layers past it then read a replica of its pool (doc/placement.md, \"DeepSeek-V4.1\")")


def load_by_path(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod         # dataclasses resolve their module's annotations through it
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
# A. pure logic (stdlib; placement.py, cache_plan.py and model/placement.py loaded by path)

def part_a(path):
    P = load_by_path("_placement", "exllamav3/architecture/dsv41/placement.py")
    CP = load_by_path("_cache_plan", "exllamav3/architecture/dsv41/cache_plan.py")
    G = load_by_path("_generic_placement", "exllamav3/model/placement.py")
    topo = P.Topology.from_config_dict(json.load(open(os.path.join(path, "config.json"))))
    n = topo.num_hidden_layers
    default = {i: (topo.kv_source_for(i), None, False)
               for i in range(n) if topo.kv_source_for(i) is not None}
    assert len(default) == 38, len(default)

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

    # 2. the message a refused cut gets: what is split, the free cuts, both ways to move it,
    # and the explicit placement's split as the alternative
    m12 = P.cut_message(topo, 12, "cuda:0", "cuda:1")
    for want in ("DeepSeek-V4.1: the layer split put layers.12 on cuda:1, but layers 8-13 share the "
                 "compressed-KV pool of layers.8, which is on cuda:0",
                 "at layer 1, 2, 8, 14 or 20 on this model",
                 "(-gs / use_per_device / reserve_per_device): less room on cuda:0, so that it holds "
                 "layers up to 7 only, or room for layers 12-13 as well, so that the change falls at "
                 "layer 14",
                 'doc/env_vars.md, DeepSeek-V4.1, "Loading across GPUs"'):
        assert want in m12, (want, m12)
    assert m12.endswith(". " + HINT), m12
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
    assert P.split_what(topo, 12) == "layers 8-13 share the compressed-KV pool of layers.8"
    print("  OK  refusal message: the split pool (or selection, or candidate blocks), the free cuts, "
          "less room below or more room above, the explicit placement's split")

    # 3. the routes of a split
    r12, t12, c12 = P.compute_routes(topo, 12)
    assert r12[12] == (12, 8, False), r12[12]                     # replica of 8, no index K
    assert r12[13] == (12, None, False), r12[13]                  # 13 reads 12's replica
    assert {i: r for i, r in r12.items() if i not in (12, 13)} == \
        {i: r for i, r in default.items() if i not in (12, 13)}
    assert t12 == {(8, 12), (8, 13)} and c12 == set(), (t12, c12)
    r28, t28, c28 = P.compute_routes(topo, 28)
    assert r28[28] == (28, 20, True), r28[28]                     # 32/36 borrow 20's index K
    assert all(r28[i] == (28, None, False) for i in range(29, 40))
    assert all(r28[i] == (20, None, False) for i in range(20, 28))
    assert c28 == {(20, 28), (20, 32), (20, 36)} and t28 == set(), (c28, t28)
    # split 29: layer 29 is not an index source, so 28's top-k crosses too
    r29, t29, c29 = P.compute_routes(topo, 29)
    assert r29[29] == (29, 20, True) and t29 == {(28, 29), (28, 30), (28, 31)}, (r29[29], t29)
    assert c29 == {(20, 32), (20, 36)}, c29
    # past the last index source borrowing 20's K, the replica carries none
    assert P.compute_routes(topo, 37)[0][37] == (37, 20, False)
    # a split at a free cut routes nothing across, as no split does
    for s in free:
        assert P.compute_routes(topo, s) == (default, set(), set()), s
    assert P.compute_routes(topo, None) == (default, set(), set()) and P.default_routes(topo) == default
    print("  OK  routes: a split at 12 gives 12 a replica of 8 (13 reads it, 8's top-k crosses); at 28 "
          "a replica of 20 with index K and 20's candidates crossing; at 29 also 28's top-k")

    # 4. the explicit placement as V4.1 reads it: every change of device a free cut, but one
    assert P.DSV41Placement.from_generic(None, topo) is None
    spec = "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1"
    g12 = P.DSV41Placement.from_generic(G.parse(spec), topo)
    assert g12.cuts == (12,) and g12.split_at == 12 and g12.spec == spec, g12
    assert (g12.routes, g12.topk_cross, g12.cand_cross) == (r12, t12, c12)
    assert g12.replicas == {12: 8}
    assert g12.crossings() == "replicas 12<-8 | top-k cross 8->12-13 | candidates cross none", g12.crossings()
    assert g12.needs() == "a pool replica for a change of device at layer 12", g12.needs()
    g22 = P.DSV41Placement.from_generic(G.parse("0-21=cuda:0; 22-39=cuda:1"), topo)
    assert g22.crossings() == "replicas 22<-20+idxK | top-k cross 20->22-23 | candidates cross 20->24,28,32,36"
    # reversed devices: still the split at 12
    assert P.DSV41Placement.from_generic(G.parse("0-11=cuda:1; *=cuda:0"), topo).split_at == 12
    # changes of device at free cuts only: no split, the default routes, any number of devices,
    # also returning to a device
    for text, cuts in (("0-13=cuda:0; 14-39=cuda:1", (14,)), ("0-7=cuda:0; 8-19=cuda:1; 20-39=cuda:2", (8, 20)),
                       ("0-7=cuda:0; 8-13=cuda:1; 14-39=cuda:0", (8, 14)),
                       ("*=cuda:0; 12-22=cuda:0 experts=stream", ())):
        pl = P.DSV41Placement.from_generic(G.parse(text), topo)
        assert pl.cuts == cuts and pl.split_at is None and pl.routes == default, (text, pl)
        assert pl.needs() == "no pool replica" and not pl.replicas
    # one split among free cuts, on three devices
    three = P.DSV41Placement.from_generic(G.parse("0-11=cuda:0; 12-19=cuda:1; 20-39=cuda:2"), topo)
    assert three.cuts == (12, 20) and three.split_at == 12 and three.routes == r12, three
    # two changes of device where something is shared: refused, naming both and the free cuts
    two = "0-5=cuda:0; 6-11=cuda:1; 12-39=cuda:0"
    e = raises(ValueError, P.DSV41Placement.from_generic, G.parse(two), topo)
    assert str(e) == (
        f"DeepSeek-V4.1: the explicit placement '{two}' changes device in 2 places where a pool, a top-k "
        f"selection or candidate blocks are shared across the change: at layer 6 (layers 2-7 share the "
        f"compressed-KV pool of layers.2) and at layer 12 (layers 8-13 share the compressed-KV pool of "
        f"layers.8). At most one change of device may fall there (the layers past it then read a replica "
        f"of the pool); every other one must fall at layer 1, 2, 8, 14 or 20 on this model "
        f"(doc/placement.md, \"DeepSeek-V4.1\")"), str(e)
    raises(ValueError, P.DSV41Placement.from_generic, G.parse("0-9=cuda:0; 10-11=cuda:1; 12-39=cuda:0"),
           topo, match = "changes device in 2 places")
    raises(ValueError, P.DSV41Placement.from_generic, G.parse("0-40=cuda:0"), topo, match = "does not have: 40")
    raises(ValueError, P.DSV41Placement.from_generic, G.parse("0-38=cuda:0"), topo, match = "not placed")
    print(f"  OK  explicit placement: {g12.describe()}; free cuts alone give no replica (also on three "
          f"devices and returning to a device); one split among free cuts accepted; two refused")

    # 5. cache plan: a ring on every layer, a pool on the kv sources, plus the replica owner of a
    # split; routes that would read the wrong pool are refused
    cp0 = CP.CachePlan(topo.compress_ratios, topo.kv_source_layer_ids)
    assert all(cp0.owns_ring(i) for i in range(n))
    assert [i for i in range(n) if cp0.owns_pool(i)] == [2, 8, 14, 20] and cp0.allocation_count() == 4
    assert cp0.owner == [topo.kv_source_for(i) for i in range(n)] and cp0.routes == default
    assert not cp0.owns_pool(0) and cp0.owner[0] is None           # sliding: ring only
    cp = CP.CachePlan(topo.compress_ratios, topo.kv_source_layer_ids, routes = r12)
    assert [i for i in range(n) if cp.owns_pool(i)] == [2, 8, 12, 14, 20]
    assert cp.allocation_count() == 5 and cp.replica_of(12) == 8 and cp.borrows_from(13) == 12
    raises(ValueError, cp0.set_routes, {**default, 13: (12, None, False)}, match = "neither")
    # routes that would read the wrong pool: another group's replica (21 -> 12, which copies
    # 8), a kv source routed away from its own pool (8), a member below the replica owner (9)
    for i, r, why in ((21, (12, None, False), "neither"), (8, (12, None, False), "kv source"),
                      (9, (12, None, False), "neither"), (13, (12, None, True), "index K")):
        raises(ValueError, cp.set_routes, {**r12, i: r}, match = why)
    for s in range(1, n):
        CP.CachePlan(topo.compress_ratios, topo.kv_source_layer_ids, routes = P.compute_routes(topo, s)[0])
    print(f"  OK  cache plan: {cp0.describe()}; with the split at 12: {cp.describe()}; wrong-pool routes "
          f"refused, every split's routes accepted")
    return P, topo


# ---------------------------------------------------------------------------
# B. DSV41MoE and the real autosplit loop (exllamav3 import, no CUDA calls)

GIB = 1 << 30
RESIDENT = 5_222_576_916      # header scan of DeepSeek-V4.1-Flash 3.0 bpw: one fully resident layer
CPU_KEEP = 126_492_948        # a CPU-offloaded layer's GPU part, suh/svh included


def fake_ip(mcl = 0, mcs = 0, placement = None):
    return types.SimpleNamespace(moe_cpu_offload = mcl, moe_cpu_split = mcs,
                                 draft_moe_cpu_offload = 0, moe_cpu_offload_assigned = {},
                                 moe_cpu_component = "text", vision_pinned = False,
                                 moe_cpu_mode = "compute", placement = placement)


def build(path, out = None, generic = None):
    """Config and model; `generic` (an explicit placement, text or parsed) is set before
    Model.from_config, where V4.1 reads it"""
    from exllamav3 import Config, Model
    with contextlib.redirect_stdout(out if out is not None else io.StringIO()):
        cfg = Config.from_directory(path)
        cfg.infer_params.placement = generic
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
    assert model.placement is None and model.load_guard.split_at is None
    for name in ("cpu_maybe_offload_load", "cpu_maybe_split_load"):
        assert getattr(DSV41MoE, name) is getattr(BlockSparseMLP, name), name
    cfg.infer_params = fake_ip(mcl = 11)
    claim_everything(mlps)
    got = [i for i, m in enumerate(mlps) if m.cpu_maybe_offload_load(cuda1)]
    assert got == list(range(0, 11)), got
    assert cfg.infer_params.moe_cpu_offload_assigned == {"text": 11}
    print(f"  OK  -mcl 11 claims layers {got[0]}-{got[-1]}, as for every MoE model")

    # 4. the guard itself: a change of device off the free cuts is refused only with a Cache,
    # before the block's MoE loads anything; the split of the explicit placement the model was
    # built with is let through
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
                        e = raises(RuntimeError, mlps[12].load, cuda1,
                                   match = "the layer split put layers.12 on cuda:1, but layers 8-13 share "
                                           "the compressed-KV pool of layers.8, which is on cuda:0")
                        assert str(e).endswith(HINT), str(e)
                    else:
                        mlps[12].load(cuda1)
                    # a free cut takes a Cache: 14 after 13 on the first device
                    g.devices[13] = cuda0
                    mlps[14].load(cuda1)
                finally:
                    g.end()
        with patched(model, "_cache_attached", lambda: True), patched(g, "split_at", 12):
            g.begin()
            try:
                mlps[11].load(cuda0)
                mlps[12].load(cuda1)                     # the placement's split: its replica exists
            finally:
                g.end()
    want = [(5, cuda0), (6, cuda1)] + [(i, cuda0) for i in range(12)] + [(12, cuda1), (14, cuda1)] + \
           [(i, cuda0) for i in range(12)] + [(14, cuda1), (11, cuda0), (12, cuda1)]
    assert calls == want, calls
    assert not g.active
    print("  OK  load guard: layers.12 after layers.11 on another device refused with a Cache and "
          "nothing of its MoE loaded; accepted without one, at the free cut 14, and as the split of "
          "the explicit placement the model was built with")

    # 5. the real autosplit loop, without and with an explicit placement
    run_autosplit_cases(path, P)
    run_placement_cases(path)

    # 6. tensor parallel is refused before anything loads; every other load goes to upstream
    raises(NotImplementedError, lambda: next(model.load_gen(use_per_device = [63, 91], tensor_p = True)),
           match = "DeepSeek-V4.1: tensor-parallel loading is not implemented; load it as a layer "
                   "split (-gs / use_per_device), or with an explicit placement (EXL3_PLACEMENT / "
                   "--placement)")
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

    # 7. without a placement every compressed layer reads its own kv source's pool, and only kv
    # sources own one; with a split the routes reach the attention before any Cache exists
    attn = [b.attn for b in blocks]
    for i, a in enumerate(attn):
        assert bool(a.caps.get("kv_cache")) == cfg.is_kv_source(i), i
        if a.compress_ratio:
            assert a.kv_source_layer == cfg.kv_source_for(i) == a.pool_owner_layer, i
    from exllamav3.modules.dsv41 import DSV41Attention
    seen = {}
    real = DSV41Attention.set_pool_route
    def spy(self, owner_layer, replica_of = None, with_index_k = False):
        seen[self.layer_idx] = (owner_layer, replica_of, with_index_k)
        return real(self, owner_layer, replica_of = replica_of, with_index_k = with_index_k)
    DSV41Attention.set_pool_route = spy
    try:
        _, mr = build(path, generic = "0-11=cuda:0; 12-39=cuda:1")
    finally:
        DSV41Attention.set_pool_route = real
    assert seen == mr.placement.routes == mr.pool_routes and len(seen) == 38
    assert mr.load_guard.split_at == 12 and mr.cache_plan.owns_pool(12) and mr.cache_plan.borrows_from(13) == 12
    ar = [b.attn for b in mr._blocks()]
    assert ar[12].replica_of == 8 and ar[12].caps["kv_cache"] and ar[13].pool_owner_layer == 12
    print(f"  OK  pools only on the kv sources without a placement; with the split at 12, set_pool_route "
          f"called for all {len(seen)} compressed layers before any Cache: 12 -> {seen[12]}, 13 -> {seen[13]}")

    # 8. the checks at load: the placement set now must need the routes the model was built with
    run_load_checks(path)

    # 9. the post-load check
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
        self.layer_idx = mlp.layer_idx
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


def autosplit(path, budgets_gib, mcl = 0, cache_attached = False, generic = None):
    """Run model_ls._load_autosplit over 40 FakeBlocks wrapping the real DSV41MoE objects of a
    model built with the explicit placement `generic` (a string) if given. Returns (device index
    per layer, claimed CPU layers) or raises what the loop raised."""
    import torch
    import exllamav3.model.model_ls as model_ls
    from exllamav3.model.model_ls import Model_LSMixin
    from exllamav3.model.placement import parse
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    pl = parse(generic)
    cfg, model = build(path, generic = pl)
    cfg.infer_params = fake_ip(mcl = mcl, placement = pl)
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
                    2048, 32, 1, None, False, fake_cfg, sim.blocks, False, 1, {}, True,
                    placement = pl):
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
    # once, naming the group, the free cuts, both ways to move the change and the placement ...
    e = raises(RuntimeError, autosplit, path, [63, 200], cache_attached = True,
               match = "the layer split put layers.12 on cuda:1, but layers 8-13 share the "
                       "compressed-KV pool of layers.8, which is on cuda:0")
    assert "at layer 1, 2, 8, 14 or 20 on this model" in str(e), str(e)
    assert ("less room on cuda:0, so that it holds layers up to 7 only, or room for layers 12-13 "
            "as well, so that the change falls at layer 14") in str(e), str(e)
    assert str(e).endswith(HINT), str(e)
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


def run_placement_cases(path):
    """The explicit placement on V4.1 through the real autosplit loop"""
    # every layer lands where the placement says, whatever the budgets would do (the budget alone
    # changes device at 14), and the experts=cpu layers are claimed by the generic mixin path;
    # with a Cache attached the split at 12 loads, its replica route wired when the model was built
    spec = "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1"
    for cache in (False, True):
        devs, claimed = autosplit(path, [70, 200], generic = spec, cache_attached = cache)
        assert devs == [0] * 12 + [1] * 28 and claimed == list(range(12, 23)), (devs, claimed)
    # reversed devices, with a Cache attached: layer 12 reads its replica of 8 on its own device
    devs, claimed = autosplit(path, [200, 70], generic = "0-11=cuda:1; *=cuda:0", cache_attached = True)
    assert devs == [1] * 12 + [0] * 28 and claimed == [], (devs, claimed)
    # free cuts only, with a Cache, on three devices and returning to the first
    devs, _ = autosplit(path, [200, 200, 200], generic = "0-7=cuda:0; 8-19=cuda:1; 20-39=cuda:2",
                        cache_attached = True)
    assert devs == [0] * 8 + [1] * 12 + [2] * 20, devs
    devs, _ = autosplit(path, [200, 200], generic = "0-7=cuda:0; 8-13=cuda:1; 14-39=cuda:0",
                        cache_attached = True)
    assert devs == [0] * 8 + [1] * 6 + [0] * 26, devs
    # the split plus a free cut, on three devices, with a Cache
    devs, _ = autosplit(path, [200, 200, 200], generic = "0-11=cuda:0; 12-19=cuda:1; 20-39=cuda:2",
                        cache_attached = True)
    assert cuts_of(devs) == [12, 20], devs
    # a layer that does not fit its device is an error with the loader's reason, not a move
    e = raises(RuntimeError, autosplit, path, [60, 200], generic = "0-13=cuda:0; *=cuda:1",
               match = "placement: layers.12 does not fit on cuda:0")
    assert "sim: cuda:0 budget exhausted at layers.12" in str(e), str(e)
    print("  OK  explicit placement through the real autosplit loop: the split at 12 with the experts of "
          "12-22 claimed (the budget alone would change device at 14), with and without a Cache; "
          "reversed devices; free cuts on three devices and back to the first; the split plus a free "
          "cut; a layer that does not fit is a RuntimeError with the loader's reason")


def run_load_checks(path):
    """DeepseekV41Model.load_gen: the placement set at load must need the pool routes the model was
    built with; its devices, expert storage and free cuts may change"""
    import functools
    from exllamav3.model.model import Model
    cfg, model = build(path, generic = "0-11=cuda:1; 12-39=cuda:0")
    assert model.placement.split_at == 12 and model.load_guard.split_at == 12
    for changed, needs in ((None, "no pool replica"), ("0-13=cuda:1; 14-39=cuda:0", "no pool replica"),
                           ("*=cuda:1", "no pool replica"),
                           ("0-21=cuda:1; 22-39=cuda:0", "a pool replica for a change of device at layer 22")):
        cfg.infer_params.placement = changed
        e = raises(ValueError, lambda: next(model.load_gen(use_per_device = [90, 90])))
        now = "None" if changed is None else repr(changed)
        assert str(e) == (
            f"DeepSeek-V4.1: the model was built for a pool replica for a change of device at layer 12, "
            f"but config.infer_params.placement is now {now}, which needs {needs}. The cross-device pool "
            f"routes are fixed when the model object is built (the Cache is sized from them): set the "
            f"placement before Model.from_config, or build a new model to change it"), str(e)
    # a placement set only after the model was built that needs a replica: refused the same way
    _, m0 = build(path)
    m0.config.infer_params.placement = "0-11=cuda:0; 12-39=cuda:1"
    raises(ValueError, lambda: next(m0.load_gen(use_per_device = [90, 90])),
           match = "the model was built for no pool replica, but config.infer_params.placement is now "
                   "'0-11=cuda:0; 12-39=cuda:1', which needs a pool replica for a change of device at layer 12")
    # a placement with two changes of device where something is shared is refused at load too
    m0.config.infer_params.placement = "0-5=cuda:0; 6-11=cuda:1; 12-39=cuda:0"
    raises(ValueError, lambda: next(m0.load_gen(use_per_device = [90, 90])), match = "changes device in 2 places")
    assert not model.load_guard.active and not m0.load_guard.active
    seen = []
    real = Model.load_gen
    @functools.wraps(real)
    def fake(self, *a, **kw):
        seen.append((self.placement.spec if self.placement is not None else None, self.load_guard.split_at))
        yield from ()
    # the same split, written differently, with other expert storage or across other devices;
    # and, on a model built without one, a placement whose changes of device are free cuts, or none
    with patched(Model, "load_gen", fake):
        for same in ("0-11=cuda:1; *=cuda:0", "0-11=cuda:1; 12-22=cuda:0 experts=stream; *=cuda:0",
                     "0-11=cuda:0; *=cuda:1", "0-11=cuda:0; 12-19=cuda:1; 20-39=cuda:2"):
            cfg.infer_params.placement = same
            for _ in model.load_gen(use_per_device = [90, 90, 90]):
                pass
        for free in ("0-13=cuda:0; 14-39=cuda:1", "*=cuda:0", None):
            m0.config.infer_params.placement = free
            for _ in m0.load_gen(use_per_device = [90, 90]):
                pass
    assert seen == [("0-11=cuda:1; *=cuda:0", 12), ("0-11=cuda:1; 12-22=cuda:0 experts=stream; *=cuda:0", 12),
                    ("0-11=cuda:0; *=cuda:1", 12), ("0-11=cuda:0; 12-19=cuda:1; 20-39=cuda:2", 12),
                    ("0-13=cuda:0; 14-39=cuda:1", None), ("*=cuda:0", None), (None, None)], seen
    print("  OK  load_gen refuses a placement set, cleared or moved to another split after the model "
          "was built, or one it would need a replica for; accepts the same split with other devices, "
          "expert storage or free cuts, and free cuts on a model built without a placement")


def run_post_load_check(path):
    import torch
    cuda0, cuda1, cuda2 = (torch.device("cuda", i) for i in range(3))
    def place(model_, *cuts):
        devs = [cuda0, cuda1, cuda2]
        for m in model_.modules:
            m.device = cuda0
        model_.modules[0].device = torch.device("cpu")
        n = model_.config.num_hidden_layers
        for i, b in enumerate(model_._blocks()):
            b.device = devs[sum(1 for c in cuts if i >= c)]
        for m in model_.modules[model_.first_block_idx + n:]:
            m.device = devs[len(cuts)]
    cfg, model = build(path)
    # a cut kv group without a Cache: one note, naming the layers, the free cuts and the placement
    place(model, 12)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        model._check_split()
    line = out.getvalue().strip()
    assert line == ("!! DSV41: layers 12-13 read a compressed-KV pool on another device; without a "
                    "Cache each forward copies those pools whole, and a Cache would need every "
                    "change of device at layer 1, 2, 8, 14 or 20. " + HINT), line
    print(f"  OK  post-load note:{line[2:]}")
    # ... and with a Cache attached an error (the guard refuses this while loading; this is the
    # safety net for a load that went around it)
    with patched(model, "_cache_attached", lambda: True):
        e = raises(RuntimeError, model._check_split,
                   match = "DeepSeek-V4.1 load check failed: layers 12-13 read a compressed-KV pool on "
                           "another device, which the cached path cannot do; reload with every change "
                           "of device at layer 1, 2, 8, 14 or 20. " + HINT)
    # changes of device at free cuts: silent, with or without a Cache
    for cuts in ((14,), (8, 20), ()):
        place(model, *cuts)
        for cache in (False, True):
            out = io.StringIO()
            with patched(model, "_cache_attached", lambda c = cache: c), contextlib.redirect_stdout(out):
                model._check_split()
            assert out.getvalue() == "", (cuts, out.getvalue())
    # with an explicit placement: one line, where the modules landed and what crosses; the layers
    # past the split read the replica on their own device, so nothing is refused with a Cache
    _, ms = build(path, generic = "0-11=cuda:0; 12-39=cuda:1")
    place(ms, 12)
    out = io.StringIO()
    with patched(ms, "_cache_attached", lambda: True), contextlib.redirect_stdout(out):
        ms._check_split()
    line = out.getvalue().strip()
    assert line.startswith("-- DSV41 placement [0-11=cuda:0; 12-39=cuda:1]:"), line
    # device names may follow "cuda:N" when the runtime is up; check around them
    runs = line.split(": ", 1)[1].split(" | ")
    assert runs[0] == "cpu embed", runs
    assert runs[1].startswith("cuda:0") and runs[1].endswith(" hc_expand,layers.0-11"), runs
    assert runs[2].startswith("cuda:1") and runs[2].endswith(" layers.12-39,hc_head,norm,head"), runs
    assert line.endswith(" | replicas 12<-8 | top-k cross 8->12-13 | candidates cross none"), line
    print(f"  OK  post-load summary:{line[2:]}")
    # ... but a block off the placement's devices (a load that went around the loader) still
    # reads a pool on another device
    place(ms, 13)
    with patched(ms, "_cache_attached", lambda: True), contextlib.redirect_stdout(io.StringIO()):
        raises(RuntimeError, ms._check_split, match = "layers 13 read a compressed-KV pool on another device")
    print("  OK  post-load check: a note without a Cache, an error with one, silent at free cuts; with "
          "an explicit placement one summary line and no error past its split")


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
