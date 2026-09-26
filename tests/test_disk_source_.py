"""
CPU-only tests of where a component reads its experts and n-gram rows (exllamav3/model/disk_source.py;
doc/expert_tiers.md, "disk"): a copy in another directory is used only when every shard read there
has the model's size and safetensors header, byte for byte; disk io= maps to the disk engine's
page-cache modes and needs the engine; the placement's words reach these helpers as parsed.

    python tests/test_disk_source_.py

disk_source.py and placement.py are torch-free and loaded by path.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
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


D = load("_disk_source_subject", "exllamav3/model/disk_source.py")
P = load("_placement_disk_source", "exllamav3/model/placement.py")


def shard(path, tensors):
    """A safetensors file: tensors = [(key, bytes)] (U8)"""
    header, off = {}, 0
    for k, b in tensors:
        header[k] = {"dtype": "U8", "shape": [len(b)], "data_offsets": [off, off + len(b)]}
        off += len(b)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for _, b in tensors:
            f.write(b)


class CopyTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix = "exl3-disk-source-")
        self.model = os.path.join(self.tmp, "model")
        self.copy = os.path.join(self.tmp, "copy")
        os.makedirs(self.model)
        os.makedirs(self.copy)
        self.a = os.path.join(self.model, "model-00001.safetensors")
        shard(self.a, [("x", b"\1" * 100), ("y", b"\2" * 50)])
        D._checked.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_identical_copy(self):
        shutil.copy(self.a, self.copy)
        self.assertEqual(D.copy_path(self.a, self.copy, "experts"), os.path.join(self.copy, "model-00001.safetensors"))
        # checked once, then remembered
        self.assertEqual(len(D._checked), 1)
        self.assertEqual(D.copy_path(self.a, self.copy, "experts"), os.path.join(self.copy, "model-00001.safetensors"))
        # the model's own directory is its own copy
        self.assertEqual(D.copy_path(self.a, self.model, "ngram"), self.a)

    def test_refusals(self):
        b = os.path.join(self.copy, "model-00001.safetensors")
        with self.assertRaisesRegex(ValueError, r"^placement: disk experts=/nonexistent: no such directory$"):
            D.copy_path(self.a, "/nonexistent", "experts")
        with self.assertRaisesRegex(ValueError, r"^placement: disk ngram=.*: model-00001.safetensors is not there \(the "
                                                r"copy must hold every shard the reads touch, under the model's file names\)$"):
            D.copy_path(self.a, self.copy, "ngram")
        shard(b, [("x", b"\1" * 100), ("y", b"\2" * 51)])
        with self.assertRaisesRegex(ValueError, r"is \d+ bytes, the model's .* is \d+; the copy must be the same file$"):
            D.copy_path(self.a, self.copy, "experts")
        # same size, another header (a tensor renamed)
        shard(b, [("z", b"\1" * 100), ("y", b"\2" * 50)])
        with self.assertRaisesRegex(ValueError, r"the safetensors header of .* differs from the model's"):
            D.copy_path(self.a, self.copy, "experts")
        # same header, other payload bytes: accepted (the reads trust the copy's data; only the offsets are checked)
        shard(b, [("x", b"\3" * 100), ("y", b"\2" * 50)])
        self.assertEqual(D.copy_path(self.a, self.copy, "experts"), b)
        with open(os.path.join(self.copy, "bad.safetensors"), "wb") as f:
            f.write(b"\0\0")
        with open(os.path.join(self.model, "bad.safetensors"), "wb") as f:
            f.write(b"\0\0")
        with self.assertRaisesRegex(ValueError, "too short to be a safetensors file"):
            D.copy_path(os.path.join(self.model, "bad.safetensors"), self.copy, "experts")

    def test_placement_words(self):
        p = P.parse("*=cuda:0 experts=cache; disk experts=/nvme/v41 ngram=\"/mnt/engram disk\" io=direct")
        self.assertEqual((D.source_dir(p, "experts"), D.source_dir(p, "ngram"), D.io_direct(p)),
                         ("/nvme/v41", "/mnt/engram disk", 1))
        p = P.parse("*=cuda:0 experts=cache; disk io=buffered")
        self.assertEqual((D.source_dir(p, "experts"), D.source_dir(p, "ngram"), D.io_direct(p)), (None, None, 0))
        p = P.parse("*=cuda:0 experts=cache; ram experts=all; disk experts=off")
        self.assertEqual((D.source_dir(p, "experts"), D.io_direct(p)), (None, -1))
        self.assertEqual((D.source_dir(None, "ngram"), D.io_direct(None)), (None, -1))
        # io= needs the disk engine
        p = P.parse("*=cuda:0; disk io=direct")
        D.check_route(p, "engine", "rows")
        with self.assertRaisesRegex(ValueError, r"^placement: disk io=direct sets the page-cache mode of the disk engine's "
                                                r"reads, but rows are read by the original gather pool"):
            D.check_route(p, "original", "rows")
        D.check_route(P.parse("*=cuda:0; disk ngram=/x"), "original", "rows")


if __name__ == "__main__":
    unittest.main(verbosity = 2)
