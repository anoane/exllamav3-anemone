"""
The expert extent index (exllamav3/model/expert_extents.py): where each routed expert lies in the
checkpoint and where its trellis tensors lie inside one read of it.

  - synthetic shards written here: odd offsets (a 13-byte tensor before each expert), an expert
    with a foreign tensor inside its span, an expert whose down projection sits in an override
    file, a gateless layer: exact offsets, sub-offsets, contiguity and slot geometry;
  - every extent read through the disk engine (pread, odirect, io_uring) into 4 KiB-aligned slots
    gives the trellis bytes exactly (needs torch and the extension; EXL3_TIER_TEST_MINI=1 or
    EXL3_DISK_TEST_MINI=1 builds a small one, tests/expert_tier or tests/disk_engine: one of them
    per pytest process, as pybind11 registers the disk engine's types once);
  - the real DeepSeek-V4.1 3.0 bpw headers when DSV41_MODEL_DIR is set: 40 x 384 decoder experts,
    each one contiguous piece of 13,315,596 bytes, VRAM slot 13,271,040, RAM slot 13,320,192.

    python -m pytest tests/test_expert_extents_.py
"""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


X = load("_expert_extents_subject", "exllamav3/model/expert_extents.py")


# ---------------------------------------------------------------------------------------- shards

def write_shard(path, tensors):
    """Write a safetensors file: tensors = [(key, dtype, shape, bytes)] in data order. Returns the
    absolute data offset of every key"""
    header, off = {}, 0
    for key, dtype, shape, data in tensors:
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for _, _, _, data in tensors:
            f.write(data)
    base = 8 + len(hj)
    return {k: base + v["data_offsets"][0] for k, v in header.items()}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h["_header_offset"] = 8 + n
    return h


class FakeSTC:
    """file_headers / tensor_file_map / find_stc over files, the later file winning a key"""

    def __init__(self, paths):
        self.file_headers, self.tensor_file_map = {}, {}
        for p in paths:
            h = read_header(p)
            self.file_headers[p] = h
            for k in h:
                if k not in ("__metadata__", "_header_offset"):
                    self.tensor_file_map[k] = p

    def find_stc(self, key):
        return self


def pattern(seed, n):
    """Deterministic, position-dependent bytes"""
    return bytes(((i * 131 + seed * 7 + (i >> 8)) & 0xFF) for i in range(n))


TREL = {"w1": 1536, "w3": 1536, "w2": 1536}


def expert_tensors(prefix, seed, projs = ("w1", "w2", "w3"), trellis = TREL):
    """The tensors of one expert in convert.py's order: per projection suh, svh, mul1, trellis"""
    out = []
    for j, p in enumerate(projs):
        key = f"{prefix}.{p}"
        out.append((key + ".suh", "F16", [64], pattern(seed * 10 + j, 128)))
        out.append((key + ".svh", "F16", [32], pattern(seed * 10 + j + 3, 64)))
        out.append((key + ".mul1", "I32", [], pattern(seed * 10 + j + 5, 4)))
        out.append((key + ".trellis", "I16", [trellis[p] // 2], pattern(seed * 10 + j + 7, trellis[p])))
    return out


def odd(key):
    return (key, "U8", [13], b"\x5a" * 13)


class Synthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix = "expert_extents_", dir = os.environ.get("EXL3_DISK_TEST_DIR"))
        d = cls.tmp.name
        cls.main = os.path.join(d, "model-00001.safetensors")
        cls.over = os.path.join(d, "override.safetensors")
        t = []
        for e in range(4):
            t.append(odd(f"layers.1.pad.{e}"))
            t.extend(expert_tensors(f"layers.1.ffn.experts.{e}", e))
            if e == 2:
                # a foreign tensor inside expert 2's span: between its w2 and w3
                i = next(i for i, x in enumerate(t) if x[0] == "layers.1.ffn.experts.2.w3.suh")
                t.insert(i, ("layers.1.foreign", "U8", [7], b"\x11" * 7))
        cls.off_main = write_shard(cls.main, t)
        # expert 3's down projection (w2) also in an override file, which wins
        cls.off_over = write_shard(cls.over, [odd("pad")] + [x for x in expert_tensors("layers.1.ffn.experts.3", 3)
                                                             if ".w2." in x[0]])
        cls.stc = FakeSTC([cls.main, cls.over])
        cls.keys = [(f"layers.1.ffn.experts.{e}.w1", f"layers.1.ffn.experts.{e}.w3", f"layers.1.ffn.experts.{e}.w2")
                    for e in range(4)]
        cls.index = X.build_extent_index(cls.stc, cls.keys)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_contiguous_experts(self):
        for e in (0, 1):
            x = self.index.extents[e]
            pre = f"layers.1.ffn.experts.{e}"
            self.assertTrue(x.contiguous)
            begin = self.off_main[pre + ".w1.suh"]
            end = self.off_main[pre + ".w3.trellis"] + TREL["w3"]
            self.assertEqual((x.path, x.offset, x.length), (self.main, begin, end - begin))
            self.assertEqual(x.sub_offsets(), tuple(self.off_main[pre + f".{p}.trellis"] - begin
                                                    for p in ("w1", "w3", "w2")))
            self.assertEqual(x.trellis_bytes, (1536, 1536, 1536))
            self.assertEqual(x.name, pre + ".w1")
        # the 13-byte pads leave the trellis tensors at scattered, unaligned offsets
        at = {(x.offset + o) % 16 for x in self.index.extents[:2] for o in x.sub_offsets()}
        self.assertGreater(len(at), 1)

    def test_foreign_tensor_splits(self):
        x = self.index.extents[2]
        pre = "layers.1.ffn.experts.2"
        self.assertFalse(x.contiguous)
        self.assertEqual([(p.path, p.offset, p.length) for p in x.pieces],
                         [(self.main, self.off_main[pre + f".{p}.trellis"], 1536) for p in ("w1", "w3", "w2")])
        self.assertEqual(x.trellis, ((0, 0, 1536), (1, 0, 1536), (2, 0, 1536)))
        with self.assertRaisesRegex(ValueError, "3 pieces"):
            x.sub_offsets()

    def test_override_file_splits(self):
        x = self.index.extents[3]
        pre = "layers.1.ffn.experts.3"
        self.assertFalse(x.contiguous)
        self.assertEqual([(p.path, p.offset) for p in x.pieces],
                         [(self.main, self.off_main[pre + ".w1.trellis"]), (self.main, self.off_main[pre + ".w3.trellis"]),
                          (self.over, self.off_over[pre + ".w2.trellis"])])

    def test_geometry(self):
        g = self.index.geometry
        self.assertEqual(g.vram_slot_bytes, 3 * 1536)
        self.assertEqual(g.proj_off, (0, 1536, 3072))
        self.assertEqual(g.proj_bytes, (1536, 1536, 1536))
        span = self.index.extents[0].length
        # the widest need: three pieces of 1536 bytes, each rounded to a page plus a page
        self.assertEqual(g.ram_slot_bytes, max(3 * (4096 + 4096), -(-span // 4096) * 4096 + 4096))
        self.assertEqual(self.index.files, [self.main, self.over])
        self.assertEqual(len(self.index), 4)

    def test_gateless(self):
        d = self.tmp.name
        p = os.path.join(d, "gateless.safetensors")
        off = write_shard(p, [odd("x")] + expert_tensors("m.experts.0", 9, projs = ("up", "down"),
                                                         trellis = {"up": 1024, "down": 2048}))
        idx = X.build_extent_index(FakeSTC([p]), [("m.experts.0.up", "m.experts.0.down")])
        x = idx.extents[0]
        self.assertTrue(x.contiguous)
        self.assertEqual(x.sub_offsets(), (off["m.experts.0.up.trellis"] - x.offset,
                                           off["m.experts.0.down.trellis"] - x.offset))
        self.assertEqual(idx.geometry.proj_off, (0, 1024))
        self.assertEqual(idx.geometry.vram_slot_bytes, 3072)

    def test_refusals(self):
        d = self.tmp.name
        p = os.path.join(d, "mixed.safetensors")
        write_shard(p, expert_tensors("a", 1) + expert_tensors("b", 2, trellis = {"w1": 1536, "w3": 1536, "w2": 512}))
        stc = FakeSTC([p])
        with self.assertRaisesRegex(ValueError, r"b\.w1 has trellis tensors of \(1536, 1536, 512\) bytes"):
            X.build_extent_index(stc, [("a.w1", "a.w3", "a.w2"), ("b.w1", "b.w3", "b.w2")])
        with self.assertRaisesRegex(ValueError, r"c\.w1\.trellis is not in the checkpoint"):
            X.build_extent_index(stc, [("c.w1", "c.w3", "c.w2")])

    def test_read_through_the_engine(self):
        """Every extent, read by the disk engine into a 4 KiB-aligned slot, holds the trellis bytes"""
        try:
            import numpy as np
            import torch
            # one small extension per process (pybind11 registers the disk engine's types once):
            # the expert tier's (with the disk bindings) or the disk engine's, else the full one
            if os.environ.get("EXL3_TIER_TEST_MINI") == "1":
                sys.path.insert(0, str(ROOT / "tests" / "expert_tier"))
                from mini_ext import load_ext
                ext = load_ext()
            elif os.environ.get("EXL3_DISK_TEST_MINI") == "1":
                sys.path.insert(0, str(ROOT / "tests" / "disk_engine"))
                from mini_ext import load_ext
                ext = load_ext()
            else:
                from exllamav3.ext import exllamav3_ext as ext
        except Exception as e:
            self.fail(f"the disk engine is needed for this test (torch and the extension, or EXL3_TIER_TEST_MINI=1 "
                      f"or EXL3_DISK_TEST_MINI=1): {e}")
        raw = {p: open(p, "rb").read() for p in (self.main, self.over)}
        slot = self.index.geometry.ram_slot_bytes
        try:
            for cfg in ({"backend": "pread"}, {"backend": "odirect"}, {"backend": "io_uring"}):
                ext.disk_engine_configure(cfg)
                for x in self.index.extents:
                    buf = torch.zeros(slot + 4096, dtype = torch.uint8)
                    pad = (-buf.data_ptr()) % 4096
                    dst = buf[pad : pad + slot]
                    base, pos = 0, []
                    fds = []
                    for p in x.pieces:
                        fds.append(os.open(p.path, os.O_RDONLY))
                        pos.append(base)
                        base += -(-p.length // 4096) * 4096 + 4096
                    po = torch.zeros(len(x.pieces), dtype = torch.long)
                    ext.disk_read_extents(torch.tensor(fds), torch.tensor([p.offset for p in x.pieces]),
                                          torch.tensor([p.length for p in x.pieces]), dst, torch.tensor(pos),
                                          max(-(-p.length // 4096) * 4096 + 4096 for p in x.pieces),
                                          payload_offsets = po, wait = True)
                    for fd in fds:
                        os.close(fd)
                    got = dst.numpy()
                    for j, (pi, rel, nb) in enumerate(x.trellis):
                        piece = x.pieces[pi]
                        at = pos[pi] + int(po[pi]) + rel
                        want = raw[piece.path][piece.offset + rel : piece.offset + rel + nb]
                        self.assertEqual(got[at : at + nb].tobytes(), want, f"{cfg} {x.name} projection {j}")
        finally:
            ext.disk_engine_shutdown()


class RealCheckpoint(unittest.TestCase):
    """The DeepSeek-V4.1-Flash 3.0 bpw headers (DSV41_MODEL_DIR), CPU only, no tensor data read"""

    @unittest.skipUnless(os.environ.get("DSV41_MODEL_DIR"), "DSV41_MODEL_DIR is not set")
    def test_v41(self):
        d = os.environ["DSV41_MODEL_DIR"]
        paths = sorted(str(p) for p in Path(d).glob("*.safetensors"))
        stc = FakeSTC(paths)
        keys = [tuple(f"layers.{l}.ffn.experts.{e}.{p}" for p in ("w1", "w3", "w2")) for l in range(40) for e in range(384)]
        idx = X.build_extent_index(stc, keys)
        self.assertEqual(len(idx), 15360)
        self.assertTrue(all(x.contiguous and x.length == 13_315_596 for x in idx.extents))
        g = idx.geometry
        self.assertEqual((g.vram_slot_bytes, g.ram_slot_bytes), (13_271_040, 13_320_192))
        self.assertEqual(g.proj_bytes, (4_423_680,) * 3)
        self.assertEqual(g.chunk_slots(), 80)
        # the trellis tensors do not sit on sector boundaries: a read covers the aligned superset
        unaligned = sum(1 for x in idx.extents for o in x.sub_offsets() if (x.offset + o) % 512)
        self.assertGreater(unaligned, 3 * 15360 // 2)
        # the MTP head's experts are another component: the decoder's index has none of them
        mtp = [k for k in stc.tensor_file_map if k.startswith("mtp.") and ".experts." in k and k.endswith(".trellis")]
        self.assertTrue(mtp)
        self.assertFalse(any(x.name.startswith("mtp.") for x in idx.extents))


if __name__ == "__main__":
    unittest.main()
