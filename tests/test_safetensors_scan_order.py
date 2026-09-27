"""
SafetensorsCollection directory scan: when two shards in one directory carry the same key (e.g. stale
shards left in an output dir by an earlier conversion with a different shard count), the file whose
name sorts last wins, regardless of the order the filesystem enumerates the directory in. Files and
directories added after the base scan still override it. CPU only.
"""
import os, sys
import torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exllamav3.loader.safetensors as st_mod
from exllamav3.loader.safetensors import SafetensorsCollection

KEY = "model.layers.0.mlp.weight"
FILE_A = "model-00001-of-00004.safetensors"
FILE_B = "model-00001-of-00005.safetensors"


def _write_shards(d):
    save_file({KEY: torch.full((4,), 1.0)}, os.path.join(d, FILE_A))
    save_file({KEY: torch.full((4,), 2.0)}, os.path.join(d, FILE_B))


def _collection(d, monkeypatch, reverse):
    real_glob = st_mod.glob.glob
    def fake_glob(pattern):
        files = sorted(real_glob(pattern), reverse = reverse)
        assert len(files) == 2
        return files
    with monkeypatch.context() as m:
        m.setattr(st_mod.glob, "glob", fake_glob)
        stc = SafetensorsCollection(str(d), load_method = "python")
    return stc


def test_duplicate_key_winner_independent_of_enumeration_order(tmp_path, monkeypatch, capsys):
    _write_shards(tmp_path)
    for reverse in (False, True):
        stc = _collection(tmp_path, monkeypatch, reverse)
        assert os.path.basename(stc.tensor_file_map[KEY]) == FILE_B
        assert [os.path.basename(f) for f in stc.tensor_files] == [FILE_A, FILE_B]
        t = stc.get_tensor(KEY, device = "cpu")
        assert torch.equal(t, torch.full((4,), 2.0))
        stc.close()
        # the override warning is unchanged
        out = capsys.readouterr().out
        assert f" !! Overriding {KEY} from {os.path.join(str(tmp_path), FILE_A)} " \
               f"with {os.path.join(str(tmp_path), FILE_B)}" in out
        assert " !! Replaced 1 tensors" in out


def test_single_file_and_override_directory_unchanged(tmp_path):
    # an explicitly added file or directory still overrides the base scan, whatever its name
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    base.mkdir()
    extra.mkdir()
    save_file({KEY: torch.full((4,), 2.0)}, str(base / FILE_B))
    save_file({KEY: torch.full((4,), 3.0)}, str(extra / FILE_A))
    stc = SafetensorsCollection(str(base), load_method = "python")
    stc.add_tensor_files(str(extra / FILE_A), warn_if_override = False)
    assert torch.equal(stc.get_tensor(KEY, device = "cpu"), torch.full((4,), 3.0))
    stc.close()
    stc = SafetensorsCollection(str(base), load_method = "python")
    stc.add_tensor_files(str(extra))
    assert torch.equal(stc.get_tensor(KEY, device = "cpu"), torch.full((4,), 3.0))
    stc.close()
