"""
The RAM tier host (exllamav3_ext/tier/tier_host.cpp, model/expert_tier_host.py) over a synthetic
checkpoint written here, CPU only (a host buffer stands in for the GPU's pool and staging):

  - the arena: chunks of whole slots, 4 KiB aligned, each registered with the disk engine;
  - the cold fill: every pool slot and every RAM tier slot holds its expert's bytes, read in place
    from the shards (pread, odirect and io_uring backends);
  - routing traces (scripted micro-scenarios and random Zipf traces over every policy x demote x
    admit x evict, disk on and off, spare 0): after every call the native host's records, actions
    and state equal the Python reference's exactly, and every VALID or retiring pool slot, every
    RAM tier slot (extent or compact layout) and every transient's staging slot holds exactly the
    checkpoint bytes of the expert the policy says it holds;
  - layer-mode prefill with the class-2 read-ahead into the slab, refills (class 3), the mirror
    path (records decided elsewhere), deferred returns with rescues;
  - failures: every engine read failing (EXL3_DISK_FAULT_EIO) is served by the plain pread
    fallback; a shard cut short fails the call with a message naming layer, expert, file and errno;
  - the real DeepSeek-V4.1 3.0 bpw shards when DSV41_MODEL_DIR is set (two layers, 13 MB experts).

Needs the extension: EXL3_TIER_TEST_MINI=1 builds the small one of tests/expert_tier (C++ only),
otherwise exllamav3_ext is used. EXL3_DISK_TEST_DIR puts the scratch files on a chosen file system
(O_DIRECT needs a real one).

    EXL3_TIER_TEST_MINI=1 python -m pytest tests/test_expert_tier_ram_.py
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "expert_tier"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


P = load("_exl3_torch_free_expert_tier_policy", ROOT / "exllamav3/model/expert_tier_policy.py")
X = load("_expert_extents_for_ram", ROOT / "exllamav3/model/expert_extents.py")
H = load("_expert_tier_host_subject", ROOT / "exllamav3/model/expert_tier_host.py")
EXT = None


def ext():
    global EXT
    if EXT is None:
        try:
            from mini_ext import load_any
            EXT = load_any()
        except Exception as e:
            raise AssertionError(f"the tier's native code is needed (EXL3_TIER_TEST_MINI=1 builds a small extension): {e}")
    return EXT


# ---------------------------------------------------------------------------------------- checkpoint

def write_shard(path, tensors):
    header, off = {}, 0
    for key, dtype, shape, data in tensors:
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for t in tensors:
            f.write(t[3])


class FakeSTC:
    def __init__(self, paths):
        self.file_headers, self.tensor_file_map = {}, {}
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h["_header_offset"] = 8 + n
            self.file_headers[p] = h
            for k in h:
                if k not in ("__metadata__", "_header_offset"):
                    self.tensor_file_map[k] = p

    def find_stc(self, key):
        return self


def pattern(seed, n):
    r = random.Random(seed)
    return r.randbytes(n)


class Ckpt:
    """L layers x E experts, gate/up/down trellis of TB bytes each, with odd-sized aux tensors and
    13-byte pads; expert (L-1, 3) has a foreign tensor inside its span, expert (0, 5)'s down
    projection lives in an override file: both are read as three pieces"""

    def __init__(self, d, L = 3, E = 16, TB = 12288):
        self.L, self.E, self.TB = L, E, TB
        self.files = [os.path.join(d, "model-00001.safetensors"), os.path.join(d, "model-00002.safetensors"),
                      os.path.join(d, "override.safetensors")]
        shards = [[], []]
        self.keys, self.expected = [], {}
        over = []
        for l in range(L):
            t = shards[0 if l < (L + 1) // 2 else 1]
            for e in range(E):
                t.append((f"pad.{l}.{e}", "U8", [13], b"\x5a" * 13))
                pre = f"layers.{l}.ffn.experts.{e}"
                trel = []
                for j, p in enumerate(("w1", "w3", "w2")):
                    seed = (l * 1000 + e) * 10 + j
                    t.append((f"{pre}.{p}.suh", "F16", [66], pattern(seed + 1_000_000, 132)))
                    t.append((f"{pre}.{p}.svh", "F16", [34], pattern(seed + 2_000_000, 68)))
                    t.append((f"{pre}.{p}.mul1", "I32", [], pattern(seed + 3_000_000, 4)))
                    data = pattern(seed, TB)
                    t.append((f"{pre}.{p}.trellis", "I16", [TB // 2], data))
                    trel.append(data)
                    if l == L - 1 and e == 3 and p == "w3":
                        t.append((f"layers.{l}.foreign", "U8", [7], b"\x11" * 7))
                    if l == 0 and e == 5 and p == "w2":
                        data = pattern(seed + 7, TB)             # the override file's copy wins
                        over.append((f"{pre}.{p}.trellis", "I16", [TB // 2], data))
                        trel[2] = data
                # slot order gate (w1), up (w3), down (w2)
                self.expected[l * E + e] = trel[0] + trel[1] + trel[2]
                self.keys.append((f"{pre}.w1", f"{pre}.w3", f"{pre}.w2"))
        write_shard(self.files[0], shards[0])
        write_shard(self.files[1], shards[1])
        write_shard(self.files[2], [("pad", "U8", [3], b"xyz")] + over)
        self.index = X.build_extent_index(FakeSTC(self.files), self.keys)


def cfg(**kw):
    base = {"admit": "always", "evict": "lru", "policy": "lazy-exclusive", "demote": "swap", "halflife": 64}
    base.update(kw)
    return P.PolicyConfig(**base)


def tbytes(t):
    return bytes(t.numpy())


class Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix = "tier_ram_", dir = os.environ.get("EXL3_DISK_TEST_DIR"))
        cls.ck = Ckpt(cls.tmp.name)
        ext().disk_engine_configure({"backend": "io_uring"})

    @classmethod
    def tearDownClass(cls):
        ext().disk_engine_shutdown()
        cls.tmp.cleanup()

    def host(self, policy, S, R, **kw):
        ck = kw.pop("ck", self.ck)
        return H.TierHost(ck.index, ck.L, ck.E, policy, S, R, ext = ext(), **kw)

    def verify(self, h, ck = None, where = ""):
        """Every pool slot that holds an expert (VALID or retiring) and every RAM tier slot holds
        exactly that expert's checkpoint bytes"""
        ck = ck or self.ck
        st = h.state()
        for s, (state, key) in enumerate(zip(st["vram_state"], st["vram_key"])):
            if state in (P.VALID, P.RETIRING):
                self.assertEqual(tbytes(h.vram_slot(s)), ck.expected[key], f"{where}: pool slot {s} key {key}")
        for r, key in enumerate(st["ram_key"]):
            if key >= 0:
                self.assertEqual(tbytes(h.ram_trellis(r)), ck.expected[key], f"{where}: RAM slot {r} key {key}")


class Arena(Base):

    def test_layout_and_registration(self):
        g = self.ck.index.geometry
        self.assertEqual(g.ram_slot_bytes % 4096, 0)
        for backend, registered in (("io_uring", True), ("pread", False)):
            ext().disk_engine_configure({"backend": backend})
            h = self.host(cfg(spare = 2), 8, 10, chunk_bytes = 3 * g.ram_slot_bytes + 100, slab_slots = 5)
            a = h.arena()
            ram = a["ram"]
            self.assertEqual((ram["slots"], ram["chunk_slots"], ram["slot_bytes"]), (10, 3, g.ram_slot_bytes))
            self.assertEqual(ram["bytes"], [3 * g.ram_slot_bytes] * 3 + [g.ram_slot_bytes])
            self.assertTrue(all(b % 4096 == 0 for b in ram["base"]))
            self.assertEqual(a["slab"]["bytes"], [3 * g.ram_slot_bytes, 2 * g.ram_slot_bytes])
            if registered:
                self.assertTrue(all(i >= 1 for i in ram["registered"] + a["slab"]["registered"]), a)
                self.assertEqual(len(set(ram["registered"] + a["slab"]["registered"])), 6)
            else:
                self.assertEqual(ram["registered"] + a["slab"]["registered"], [-1] * 6)
            self.assertEqual((a["slab"]["slots"], a["vram_slot_bytes"], a["staging_slots"]),
                             (5, 3 * self.ck.TB, self.ck.E))
            del h
        ext().disk_engine_configure({"backend": "io_uring"})


class ColdFill(Base):

    def test_every_backend(self):
        for conf in ({"backend": "pread"}, {"backend": "odirect"}, {"backend": "io_uring"},
                     {"backend": "io_uring", "direct": "none"}):
            ext().disk_engine_configure(conf)
            h = self.host(cfg(spare = 2), 12, 20, fill_inflight = 3)
            h.cold_fill()
            order = P.round_robin_order(self.ck.L, self.ck.E)
            st = h.state()
            self.assertEqual([st["vram_key"][s] for s in range(10)], order[:10], conf)
            self.assertEqual(sorted(k for k in st["ram_key"] if k >= 0), sorted(order[10:30]))
            self.verify(h, where = str(conf))
            s = h.stats()
            self.assertEqual((s["cold_ram"], s["cold_vram"], s["ssd_reads"], s["pread_fallbacks"]), (20, 10, 30, 0), conf)
            # the whole extent landed in place: the raw slot holds the file's bytes at the payload offset
            raw = {f: open(f, "rb").read() for f in self.ck.files}
            for r, key in enumerate(st["ram_key"]):
                x = self.ck.index.extents[key]
                if not x.contiguous:
                    continue
                slot = tbytes(h.h.ram_slot(r))
                off = h.h.ram_offset(r, 0) - x.trellis[0][1]           # payload offset of the extent
                self.assertEqual(slot[off : off + x.length], raw[x.path][x.offset : x.offset + x.length])
            del h
        ext().disk_engine_configure({"backend": "io_uring"})

    def test_inclusive_pool_from_ram(self):
        """Under inclusive the pool's cold fill copies from the RAM tier's copy (no second read)"""
        h = self.host(cfg(spare = 2, policy = "inclusive", demote = "off"), 8, 20)
        h.cold_fill()
        s = h.stats()
        self.assertEqual((s["cold_ram"], s["cold_vram"], s["ssd_reads"]), (20, 6, 20))
        self.verify(h)


# ---------------------------------------------------------------------------------------- traces

def zipf_trace(rng, L, E, n, routed = 0.05, layer = 0.03):
    w = [1.0 / (i + 1) ** 0.9 for i in range(E)]
    perm = [rng.sample(range(E), E) for _ in range(L)]
    ops = []
    for i in range(n):
        lc = i % L
        if lc == 0:
            ops.append(("tick", 1))
        r = rng.random()
        if r < layer:
            ops.append(("layer", lc, [rng.randrange(E) for _ in range(rng.randrange(16, 64))]))
        elif r < layer + routed:
            ops.append(("call", lc, [rng.randrange(E) for _ in range(rng.randrange(9, 40))], P.ROUTED))
        else:
            rows = rng.choice((1, 1, 2, 4))
            ops.append(("call", lc, [perm[lc][e] for e in rng.choices(range(E), w, k = 6 * rows)], P.DECODE))
    return ops


class Traces(Base):
    """The native host against the reference core, call by call, with every byte checked"""

    def run_trace(self, policy, S, R, ops, defer = False, slab = 6, check_every = 1, what = "", order = None):
        ck = self.ck
        h = self.host(policy, S, R, slab_slots = slab, defer_returns = defer)
        py = P.TierCore(policy, ck.L, ck.E, S, R, 0, defer)
        order = order or P.round_robin_order(ck.L, ck.E)
        h.cold_fill(order)
        py.cold_fill(order)
        self.verify(h, where = f"{what} cold fill")
        half = 0
        for i, op in enumerate(ops):
            w = f"{what} op {i} {op[:2]}"
            if op[0] == "tick":
                h.tick(op[1])
                py.tick(op[1])
                continue
            if op[0] == "complete":
                h.complete()
                acts = py.complete()
                self.assertEqual(h.last_actions(), [tuple(a) for a in acts], w)
                continue
            if op[0] == "layer":
                half ^= 1
                h.layer(op[1], op[2], half)
                acts = py.layer_call(op[1], op[2])
                self.assertEqual(h.last_actions(), [tuple(a) for a in acts], w)
                for a in acts:
                    if a[0] in ("stage_ram", "stage_ssd"):
                        self.assertEqual(tbytes(h.staging(half, a[-1])), ck.expected[a[1]], f"{w}: staging {a}")
            else:
                ents = h.call(op[1], op[2], op[3])
                rec, acts = py.call(op[1], op[2], op[3])
                self.assertEqual([tuple(e) for e in ents], [x.astuple() for x in rec.entries], w)
                self.assertEqual(h.last_actions(), [tuple(a) for a in acts], w)
                for x in rec.entries:
                    if x.kind == P.TRANSIENT:
                        self.assertEqual(tbytes(h.staging(0, x.dst)), ck.expected[x.key], f"{w}: transient {x.key}")
                    else:
                        self.assertEqual(tbytes(h.vram_slot(x.dst)), ck.expected[x.key], f"{w}: slot {x.dst}")
            if i % check_every == 0:
                st, want = h.state(), None
                self.assertEqual(st["vram_key"], py.vram.key, w)
                self.assertEqual(st["ram_key"], py.ram.key, w)
                self.assertEqual(st["ram_compact"], [int(c) for c in py.ram.compact], w)
                self.verify(h, where = w)
                py.check(deterministic = not defer)
        h.drain()
        self.verify(h, where = f"{what} end")
        c = h.counters()
        self.assertEqual(c, dict(py.c), what)
        return h, py

    def test_micro_scenarios(self):
        calls = lambda lst: [("call", 0, ids, P.DECODE) for ids in lst]
        # S1, S2, S5, S6, S8, S9 over the synthetic checkpoint (layer 0; keys = experts)
        self.run_trace(cfg(spare = 2, policy = "exclusive", demote = "off"), 5, 0,
                       calls([[0, 1], [2, 3], [0, 2], [1, 3], [4, 5]]), what = "S1")
        self.run_trace(cfg(spare = 2, policy = "exclusive", demote = "off", admit = "heat"), 4, 0,
                       calls([[0, 1], [0, 2], [3, 0], [2, 4], [2, 5], [2, 4]]), what = "S2")
        self.run_trace(cfg(spare = 2, policy = "inclusive", demote = "off"), 4, 8,
                       calls([[0, 1], [2, 0], [3, 4], [1, 2]]), what = "S5")
        for p in (0.0, 1.0):
            self.run_trace(cfg(spare = 2, policy = "exclusive", demote = "off", admit = "adaptive", p = p), 4, 0,
                           calls([[0, 1], [2, 3], [4, 0]]), what = f"S6 p={p}")
        self.run_trace(cfg(spare = 2, evict = "lfu", policy = "lazy-exclusive", demote = "heat"), 4, 1,
                       calls([[0, 1], [0, 2], [0, 3], [3, 4]]), what = "S8")
        self.run_trace(cfg(spare = 1, policy = "exclusive", demote = "off"), 3, 0, calls([[0, 1], [2, 3]]), what = "S9")

    def test_every_policy(self):
        rng = random.Random(77)
        K = self.ck.L * self.ck.E
        n = int(os.environ.get("EXL3_TIER_RAM_TRACE_CALLS", "150"))
        for pol in P.POLICIES:
            for dem in P.DEMOTES:
                for adm in P.ADMITS:
                    S, spare = 14, 3
                    R = 20 if pol != "inclusive" else S + spare + 6
                    policy = cfg(policy = pol, demote = dem, admit = adm, evict = rng.choice(P.EVICTS),
                                 ram_evict = rng.choice(P.EVICTS), ram_admit = rng.choice(P.RAM_ADMITS), spare = spare,
                                 seed = rng.randrange(1 << 32), adapt_every = 64)
                    with self.subTest(policy = pol, demote = dem, admit = adm):
                        self.run_trace(policy, S, R, zipf_trace(rng, self.ck.L, self.ck.E, n),
                                       what = f"{pol}/{dem}/{adm}")
        # disk experts=off: VRAM (S - spare) + RAM hold every expert; nothing is read after the load
        policy = cfg(spare = 4, disk_off = True, demote = "swap")
        h, py = self.run_trace(policy, 20, K - 16, zipf_trace(rng, self.ck.L, self.ck.E, n, layer = 0.05),
                               what = "disk off")
        self.assertEqual(h.stats()["ssd_reads"], 16 + K - 16)            # the cold fill only
        # spare=0 (demote off): admissions replace their victims in place
        self.run_trace(cfg(spare = 0, demote = "off"), 10, 12, zipf_trace(rng, self.ck.L, self.ck.E, n), what = "spare=0")
        # spare=0 with demotion: an in-place admission would have to copy its victim back into the
        # RAM slot its own copy reads from; refused, as the placement grammar refuses it
        with self.assertRaisesRegex(RuntimeError, "spare=0 needs ram demote=off or policy=inclusive"):
            self.host(cfg(spare = 0, demote = "all", policy = "exclusive"), 10, 12)

    def test_deferred_returns_rescue(self):
        """Production-like: demoted slots return late; a miss on a retiring expert is copied from its
        retiring slot (device to device), and the bytes are right"""
        policy = cfg(spare = 2, policy = "exclusive", demote = "all")
        K = self.ck.L * self.ck.E
        # experts 0, 1, 2 of layer 0 in the pool (slots 0-2), the rest in RAM
        ops = [("call", 0, [3], P.DECODE),          # 3 admitted, victim 0 demoted: slot 0 retiring
               ("call", 0, [0, 4], P.DECODE),       # 0 copied from its retiring slot; 4 starved
               ("complete",), ("call", 0, [5, 6, 0], P.DECODE), ("complete",)]
        h, py = self.run_trace(policy, 5, 40, ops, defer = True, what = "rescue", order = list(range(K)))
        self.assertEqual(h.counters()["rescued"], 1)
        self.assertEqual(h.stats()["d2d"], 1)
        rng = random.Random(5)
        ops = []
        for op in zipf_trace(rng, self.ck.L, self.ck.E, 300):
            ops.append(op)
            if rng.random() < 0.15:
                ops.append(("complete",))
        h, py = self.run_trace(cfg(spare = 3, demote = "heat"), 12, 16, ops + [("complete",)], defer = True,
                               what = "deferred random")
        self.assertEqual(h.stats()["d2d"], h.counters()["rescued"])


class LayerMode(Base):

    def test_prefetch_then_layer(self):
        ck = self.ck
        policy = cfg(spare = 2)
        h = self.host(policy, 8, 6, slab_slots = 5)
        py = P.TierCore(policy, ck.L, ck.E, 8, 6)
        order = P.round_robin_order(ck.L, ck.E)
        h.cold_fill(order)
        py.cold_fill(order)
        lc = 1
        ssd_only = [k for k in range(lc * ck.E, (lc + 1) * ck.E) if py.vram.slot_of[k] < 0 and py.ram.of[k] < 0]
        h.prefetch(lc)
        s = h.stats()
        self.assertEqual((s["prefetch_issued"], s["prefetch_starved"]), (5, len(ssd_only) - 5))
        h.promote(lc)
        ids = [3, 3, 7, 9, 11, 2]
        h.layer(lc, ids, 1)
        acts = py.layer_call(lc, ids)
        self.assertEqual(h.last_actions(), [tuple(a) for a in acts])
        for a in acts:
            if a[0] in ("stage_ram", "stage_ssd"):
                self.assertEqual(tbytes(h.staging(1, a[-1])), ck.expected[a[1]], a)
        s = h.stats()
        self.assertEqual(s["prefetch_used"], 5)
        self.assertEqual(s["staged"], len([a for a in acts if a[0].startswith("stage")]))
        self.assertEqual(s["ssd_reads"], 6 + 6 + len(ssd_only) + s["refill_reads"])
        self.assertEqual(s["refill_reads"], h.counters()["refills"])
        self.verify(h)

    def test_prefetch_serves_a_decode_miss(self):
        """A read-ahead serves a decode miss of the same expert before its layer call: the admission's
        copy comes from the slab, and the RAM slot the policy gives the expert is filled by a
        background read; at the layer call the expert is staged from RAM, and the rest of the
        read-ahead from the slab"""
        policy = cfg(spare = 2)
        h = self.host(policy, 8, 4, slab_slots = 16)
        h.cold_fill()                                   # layer 2: 32, 33 in the pool, 34 in RAM
        h.prefetch(2)
        issued = h.stats()["prefetch_issued"]
        self.assertEqual(issued, 13)
        h.call(2, [2], P.DECODE)                        # 34 promoted from RAM: its copy becomes a duplicate
        h.call(2, [15], P.DECODE)                       # 47 admitted, its RAM slot over that duplicate
        self.assertTrue(any(a[:2] == ("ssd", 47) and a[2] >= 0 and a[3] == P.ADMIT for a in h.last_actions()),
                        h.last_actions())
        s = h.stats()
        self.assertEqual((s["slab_to_vram"], s["prefetch_used"]), (1, 1))
        h.layer(2, [1], 0)
        s = h.stats()
        self.assertEqual(s["prefetch_used"] + s["prefetch_dropped"], issued)
        self.assertEqual(s["prefetch_dropped"], 0)
        self.verify(h)

    def test_split_layer_call(self):
        """The GPU runtime runs a layer call in two halves: the staging before the layer runs (stage),
        then the call's record (heat, refills) once its routing is known. Together they give the CPU
        replay's layer call: the same actions in the same order, the same staging bytes, the same state"""
        ck = self.ck
        rng = random.Random(41)
        for what, policy, S, R in (("ram", cfg(spare = 2), 12, 14), ("disk off", cfg(spare = 2, disk_off = True), 12,
                                                                        ck.L * ck.E - 10),
                                   ("inclusive", cfg(spare = 2, policy = "inclusive", demote = "off"), 10, 20)):
            h1, h2 = self.host(policy, S, R, slab_slots = 20), self.host(policy, S, R, slab_slots = 20)
            h1.cold_fill()
            h2.cold_fill()
            half = 0
            for i, op in enumerate(zipf_trace(rng, ck.L, ck.E, 120, layer = 0.2)):
                if op[0] == "tick":
                    h1.tick(op[1])
                    h2.tick(op[1])
                    continue
                if op[0] == "call":
                    self.assertEqual(h1.call(op[1], op[2], op[3]), h2.call(op[1], op[2], op[3]))
                    continue
                half ^= 1
                lc, ids = op[1], op[2]
                h1.layer(lc, ids, half)
                acts1 = h1.last_actions()
                keys = h2.stage(lc, half)
                acts2 = h2.last_actions()
                self.assertEqual(keys, [a[1] for a in acts2], f"{what} op {i}")
                seq = h1.state()["seq"]
                order = []
                for e in ids:
                    if e not in order:
                        order.append(e)
                rec = (seq, lc, P.LAYER, [(lc * ck.E + e, P.TRANSIENT, -1, ids.count(e), 0, -1, -1, 0, 0, 0)
                                          for e in order])
                h2.process(rec)
                self.assertEqual(acts2 + h2.last_actions(), acts1, f"{what} op {i}")
                for a in acts2:
                    self.assertEqual(tbytes(h2.staging(half, a[-1])), ck.expected[a[1]], f"{what} op {i}: {a}")
                st1, st2 = h1.state(), h2.state()
                for k in ("vram_key", "ram_key", "heat_v", "heat_ep", "seq"):
                    self.assertEqual(st1[k], st2[k], f"{what} op {i}: {k}")
            h1.drain()
            h2.drain()
            self.assertEqual(h1.counters(), h2.counters(), what)
            self.verify(h2, where = what)

    def test_router_read_ahead(self):
        """prefetch=router's prediction records read the experts neither VRAM nor RAM holds into the
        slab; a decode miss on one of them is served from there, its RAM slot filled in the background"""
        ck = self.ck
        policy = cfg(spare = 2)
        h = self.host(policy, 8, 4, slab_slots = 16)
        h.cold_fill()
        st = h.state()
        lc = 2
        disk_only = [e for e in range(ck.E) if st["slot_of"][lc * ck.E + e] < 0 and st["ram_of"][lc * ck.E + e] < 0]
        pred = disk_only[:3] + [e for e in range(ck.E) if e not in disk_only][:2]
        rec = (0, lc, 3, [(lc * ck.E + e, P.TRANSIENT, -1, 1, 0, -1, -1, 0, 0, 0) for e in pred])
        h.read_ahead(rec)
        s = h.stats()
        self.assertEqual(s["predicted_reads"], 3)
        h.read_ahead(rec)                                   # already there: nothing more is read
        self.assertEqual(h.stats()["predicted_reads"], 3)
        reads = h.stats()["ssd_reads"]
        h.call(lc, disk_only[:1] + disk_only[:1], P.DECODE)
        s = h.stats()
        self.assertEqual(s["prefetch_used"], 1)
        self.assertEqual(s["ssd_reads"] - reads, s["slab_to_vram"])       # only a RAM slot's background fill
        self.verify(h)

    def test_refills(self):
        """Routed prefill calls read SSD-only experts past the RAM tier; the warm ones are refilled
        (class 3), at most refill_inflight per call; with deterministic=False they land on poll() and
        count against refill_inflight while they read, so the decisions may differ from the
        reference's (never the bytes)"""
        ck = self.ck
        for deterministic in (True, False):
            policy = cfg(spare = 2, refill_inflight = 2)
            h = self.host(policy, 8, 10, deterministic = deterministic)
            py = P.TierCore(policy, ck.L, ck.E, 8, 10)
            order = P.round_robin_order(ck.L, ck.E)
            h.cold_fill(order)
            py.cold_fill(order)
            for ids in ([12, 13, 14, 15, 12, 13], [9, 10, 11, 12], [15, 15, 15, 14, 14, 14]):
                h.call(0, ids, P.ROUTED)
                rec, acts = py.call(0, ids, P.ROUTED)
                if deterministic:
                    self.assertEqual(h.last_actions(), [tuple(a) for a in acts])
            h.drain()
            c = h.counters()
            if deterministic:
                self.assertEqual(c, dict(py.c))
            self.assertGreater(c["refills"], 0)
            self.assertEqual(h.stats()["refill_reads"], c["refills"])
            self.verify(h)


    def test_refills_in_flight_bound(self):
        """Asynchronous refills count against EXL3_MOE_TIER_REFILL_INFLIGHT until they land"""
        policy = cfg(spare = 2, refill_inflight = 2)
        h = self.host(policy, 8, 10, deterministic = False)
        h.cold_fill()
        h.call(0, [12, 13, 14, 15], P.ROUTED)
        first = h.counters()["refills"]
        self.assertEqual(first, 2)
        # without poll() or drain(), the two refills may still be reading: the next call may add at
        # most as many as have landed (poll() retires the landed ones first)
        h.call(0, [9, 10, 11], P.ROUTED)
        self.assertLessEqual(h.counters()["refills"], 4)
        h.drain()
        self.verify(h)


class Mirror(Base):

    def test_process_records_decided_elsewhere(self):
        """The host fed the reference directory's records (as the lookup kernel will feed it) ends in
        the same state, with the same bytes, as the host that decides itself"""
        ck = self.ck
        rng = random.Random(9)
        for policy in (cfg(spare = 2), cfg(spare = 0, demote = "off"), cfg(spare = 3, policy = "exclusive", demote = "all",
                                                                            admit = "adaptive", adapt_every = 32)):
            S, R = 12, 16
            ref = P.TierCore(policy, ck.L, ck.E, S, R)
            h = self.host(policy, S, R)
            order = P.round_robin_order(ck.L, ck.E)
            ref.cold_fill(order)
            h.cold_fill(order)
            for op in zipf_trace(rng, ck.L, ck.E, 200, layer = 0):
                if op[0] == "tick":
                    ref.tick(1)
                    h.tick(1)
                    continue
                rec = ref.lookup(op[1], op[2], op[3])
                acts = ref.host(rec)
                h.process((rec.seq, rec.lc, rec.mode, [x.astuple() for x in rec.entries]))
                self.assertEqual(h.last_actions(), [tuple(a) for a in acts])
            st = h.state()
            self.assertEqual((st["vram_key"], st["vram_stamp"], st["ram_key"], st["heat_v"]),
                             (ref.vram.key, ref.vram.stamp, ref.ram.key, ref.heat.v))
            c = dict(ref.c)
            got = h.counters()
            c["starved"] = got["starved"] = 0                 # the kernel counts those
            self.assertEqual(got, c)
            self.verify(h)


class Placement(Base):

    def test_from_placement(self):
        """The placement's words and the tuning reach the host; the cache layers' experts are indexed"""
        from types import SimpleNamespace as NS
        T = load("_etc_for_ram", ROOT / "exllamav3/model/expert_tier_config.py")
        PL = load("_placement_for_ram", ROOT / "exllamav3/model/placement.py")
        pl = PL.parse("*=cuda:0 experts=cache; cuda:0 spare=3 evict=lfu admit=heat; "
                      "ram experts=1GiB policy=exclusive demote=heat")
        ck = self.ck

        def module(l):
            lin = lambda p: [NS(key = f"layers.{l}.ffn.experts.{e}.{p}") for e in range(ck.E)]
            return NS(num_experts = ck.E, gated = True, gates = lin("w1"), ups = lin("w3"), downs = lin("w2"),
                      key = f"layers.{l}.ffn", placement_plan = lambda: ("cache", "gpu", 0))

        tc = T.TierConfig.from_env({"EXL3_MOE_TIER_SAMPLE": "4", "EXL3_MOE_TIER_DISK_SLAB": "3"})
        h = H.TierHost.from_placement(FakeSTC(ck.files), [module(l) for l in range(ck.L)], pl, "cuda:0", tc, 10, 12,
                                      ext = ext())
        self.assertEqual((h.policy.policy, h.policy.demote, h.policy.admit, h.policy.evict, h.policy.spare,
                          h.policy.sample), ("exclusive", "heat", "heat", "lfu", 3, 4))
        self.assertEqual(h.arena()["slab"]["slots"], 3)
        h.cold_fill()
        h.call(1, [3, 4, 5, 6, 7, 8])
        self.verify(h)
        bad = module(0)
        bad.placement_plan = lambda: ("cache", "gpu", 4)
        with self.assertRaisesRegex(ValueError, "hot pins are placed by the GPU runtime"):
            H.TierHost.from_placement(FakeSTC(ck.files), [bad], pl, "cuda:0", tc, 10, 12, ext = ext())

    def test_disk_rule(self):
        """disk experts=<dir>: every read from the checked copy (the model's shards unreadable meanwhile
        proves it); a copy that differs is refused naming the file. disk io=direct / buffered: the
        page-cache mode of the reads, the same bytes either way"""
        from types import SimpleNamespace as NS
        import shutil
        T = load("_etc_for_ram", ROOT / "exllamav3/model/expert_tier_config.py")
        PL = load("_placement_for_ram", ROOT / "exllamav3/model/placement.py")
        ck = self.ck

        def module(l):
            lin = lambda p: [NS(key = f"layers.{l}.ffn.experts.{e}.{p}") for e in range(ck.E)]
            return NS(num_experts = ck.E, gated = True, gates = lin("w1"), ups = lin("w3"), downs = lin("w2"),
                      key = f"layers.{l}.ffn", placement_plan = lambda: ("cache", "gpu", 0))

        tc = T.TierConfig()
        mods = [module(l) for l in range(ck.L)]
        with tempfile.TemporaryDirectory(dir = os.environ.get("EXL3_DISK_TEST_DIR")) as d:
            for f in ck.files:
                shutil.copy(f, d)
            pl = PL.parse(f'*=cuda:0 experts=cache; ram experts=1GiB; disk experts="{d}"')
            h = H.TierHost.from_placement(FakeSTC(ck.files), mods, pl, "cuda:0", tc, 10, 12, ext = ext())
            self.assertEqual(sorted(h.files), sorted(os.path.join(d, os.path.basename(f)) for f in ck.files))
            saved = [f + ".away" for f in ck.files]
            for f, g in zip(ck.files, saved):
                os.rename(f, g)
            try:
                h.cold_fill()
                for ids in ([3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14], [0, 1, 2, 15, 3, 4]):
                    h.call(2, ids)
                self.verify(h)
                self.assertGreater(h.stats()["ssd_reads"], 22)
            finally:
                for f, g in zip(ck.files, saved):
                    os.rename(g, f)
            del h
            # a copy with another header is refused before anything is read
            with open(os.path.join(d, os.path.basename(ck.files[1])), "r+b") as f:
                f.seek(9)
                f.write(b"#")
            with self.assertRaisesRegex(ValueError, "the safetensors header of .* differs from the model's"):
                H.TierHost.from_placement(FakeSTC(ck.files), mods, pl, "cuda:0", tc, 10, 12, ext = ext())
        for io, direct in (("direct", 1), ("buffered", 0), ("auto", -1)):
            pl = PL.parse(f"*=cuda:0 experts=cache; ram experts=1GiB; disk io={io}")
            h = H.TierHost.from_placement(FakeSTC(ck.files), mods, pl, "cuda:0", tc, 10, 12, ext = ext())
            h.cold_fill()
            h.call(1, [3, 4, 5, 6, 7, 8])
            self.verify(h, where = f"io={io}")
            del h


class Faults(Base):

    def test_engine_reads_fail_pread_serves(self):
        ext().disk_engine_configure({"backend": "io_uring", "fault_eio": "1"})
        try:
            rng = random.Random(3)
            h = self.host(cfg(spare = 2), 10, 12)
            h.cold_fill()
            for op in zipf_trace(rng, self.ck.L, self.ck.E, 60, layer = 0):
                if op[0] == "tick":
                    continue
                h.call(op[1], op[2], op[3])
            self.verify(h)
            s = h.stats()
            self.assertGreater(s["ssd_reads"], 22)
            self.assertEqual(s["pread_fallbacks"], s["ssd_reads"])
            self.assertEqual(h.error(), "")
        finally:
            ext().disk_engine_configure({"backend": "io_uring"})

    def test_read_error_names_the_expert(self):
        d = tempfile.mkdtemp(prefix = "tier_trunc_", dir = os.environ.get("EXL3_DISK_TEST_DIR"))
        try:
            ck = Ckpt(d)
            h = self.host(cfg(spare = 2, policy = "exclusive", demote = "off"), 6, 0, ck = ck)
            h.cold_fill()
            # the second shard (layers 2..) loses its tail: its last expert can no longer be read
            victim = ck.L * ck.E - 1
            x = ck.index.extents[victim]
            os.truncate(x.path, x.offset + 10)
            with self.assertRaises(RuntimeError) as cm:
                h.call(ck.L - 1, [ck.E - 1], P.DECODE)
            msg = str(cm.exception)
            self.assertIn(f"reading layer {ck.L - 1} expert {ck.E - 1} ({x.name}) from {x.path} at offset {x.offset}", msg)
            self.assertIn("errno", msg)
            self.assertEqual(h.error(), msg)
            with self.assertRaises(RuntimeError):
                h.call(0, [0], P.DECODE)                 # the tier stays failed
        finally:
            shutil.rmtree(d)


class RealCheckpoint(unittest.TestCase):
    """Two DeepSeek-V4.1 3.0 bpw layers (13 MB experts) from DSV41_MODEL_DIR, read in place"""

    @unittest.skipUnless(os.environ.get("DSV41_MODEL_DIR"), "DSV41_MODEL_DIR is not set")
    def test_v41(self):
        import time
        d = os.environ["DSV41_MODEL_DIR"]
        paths = sorted(str(p) for p in Path(d).glob("*.safetensors"))
        layers = (12, 25)
        keys = [tuple(f"layers.{l}.ffn.experts.{e}.{p}" for p in ("w1", "w3", "w2")) for l in layers for e in range(384)]
        index = X.build_extent_index(FakeSTC(paths), keys)
        ext().disk_engine_configure({"backend": "io_uring"})
        try:
            policy = cfg(spare = 4, halflife = 256, admit = "adaptive", adapt_every = 64)
            h = H.TierHost(index, 2, 384, policy, 16, 32, slab_slots = 8, staging_slots = 48, ext = ext())
            t0 = time.monotonic()
            h.cold_fill()
            dt = time.monotonic() - t0
            s = h.stats()
            self.assertEqual((s["cold_vram"], s["cold_ram"], s["pread_fallbacks"]), (12, 32, 0))
            self.assertEqual(h.arena()["ram"]["chunk_slots"], 80)           # 80 slots of 13,320,192 B per GiB
            print(f"\n  V4.1 cold fill: {s['cold_bytes'] / 1e6:.0f} MB in {dt:.2f} s ({s['cold_bytes'] / dt / 1e9:.2f} GB/s)")
            cache = {}

            def want(key):
                if key not in cache:
                    x = index.extents[key]
                    with open(x.path, "rb", buffering = 0) as f:
                        blob = os.pread(f.fileno(), x.length, x.offset)
                    cache[key] = b"".join(blob[t[1] : t[1] + t[2]] for t in x.trellis)
                return cache[key]

            def check(where):
                st = h.state()
                for sl, (state, key) in enumerate(zip(st["vram_state"], st["vram_key"])):
                    if state == P.VALID:
                        self.assertEqual(tbytes(h.vram_slot(sl)), want(key), f"{where}: pool slot {sl}")
                for r, key in enumerate(st["ram_key"]):
                    if key >= 0:
                        self.assertEqual(tbytes(h.ram_trellis(r)), want(key), f"{where}: RAM slot {r}")

            check("cold fill")
            rng = random.Random(41)
            hot = [rng.sample(range(384), 24) for _ in range(2)]
            t0 = time.monotonic()
            for i in range(40):
                lc = i % 2
                if lc == 0:
                    h.tick()
                ids = [rng.choice(hot[lc]) if rng.random() < 0.7 else rng.randrange(384) for _ in range(6)]
                for x in h.call(lc, ids, P.DECODE):
                    if x[1] == P.TRANSIENT:
                        self.assertEqual(tbytes(h.staging(0, x[2])), want(x[0]), f"call {i} transient")
            dt = time.monotonic() - t0
            check("after 40 decode calls")
            s = h.stats()
            print(f"  40 decode calls: {s['ssd_reads'] - 44} SSD reads, {s['h2d']} promotions, {s['d2h']} demotions, "
                  f"{dt * 1000 / 40:.1f} ms per call (CPU copies)")
        finally:
            ext().disk_engine_shutdown()


if __name__ == "__main__":
    unittest.main()
