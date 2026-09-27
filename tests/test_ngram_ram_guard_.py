"""
NGramEmbedding.load with the table held in RAM (--ngram_ram) must run the host-memory guard for
every table layout, sized to the exact bytes it allocates. Regression: only the sharded arm was
checked, so a table stored as one tensor (the single .trellis / .weight layout, or a table that
fits in one shard) was read into RAM unchecked. CPU only, tiny tables.
"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from safetensors.torch import save_file
from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules import NGramEmbedding
from exllamav3.modules.quant.exl3_lib.ngram_codec import ROW_DIM, words_per_row
from exllamav3.util import memory

PARENT = "model.layers.1.ple.ple_embedding"
KEY = PARENT + ".ngram_embedding"
ROWS = 64
RESERVE_MB = 1


def _tables(suffix, dtype, cols, shards):
    if shards == 0:
        return {f"{KEY}.{suffix}": torch.zeros((ROWS, cols), dtype = dtype)}
    rows = -(-ROWS // shards)
    return {f"{KEY}.shard_{i}.{suffix}": torch.zeros((min(rows, ROWS - i * rows), cols), dtype = dtype)
            for i in range(shards)}


def _layout(quantized, dtype, shards):
    if quantized:
        aux = {f"{KEY}.head_offsets": torch.zeros(1, dtype = torch.long),
               f"{KEY}.head_vocab_sizes": torch.full((1,), ROWS, dtype = torch.long),
               f"{KEY}.layer_multipliers": torch.ones(2, dtype = torch.long)}
        return {**aux, **_tables("trellis", torch.int16, words_per_row(2), shards)}
    aux = {f"{PARENT}.ngram_heads_offsets": torch.zeros(1, dtype = torch.long),
           f"{PARENT}.ngram_heads_vocab_sizes": torch.full((1,), ROWS, dtype = torch.long),
           f"{PARENT}.layer_multipliers": torch.ones(2, dtype = torch.long)}
    return {**aux, **_tables("weight", dtype, ROW_DIM, shards)}


def _module(path):
    stc = SafetensorsCollection(str(path))
    return NGramEmbedding(config = SimpleNamespace(stc = stc), key = KEY, ngram_size = 2, heads_per_ngram = 1,
                          ple_embed_dim = ROW_DIM, eos_token_id = 0, stream_from_disk = False), stc


def _set_available(monkeypatch, nbytes):
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", str(RESERVE_MB))
    monkeypatch.setattr(memory, "host_memory_available", lambda: nbytes + (RESERVE_MB << 20))


# shards = 0: single-tensor layout (<key>.trellis of older files, unquantized <key>.weight)
@pytest.mark.parametrize("quantized, dtype, shards", [
    (True, torch.int16, 0),
    (True, torch.int16, 1),
    (True, torch.int16, 3),
    (False, torch.half, 0),
    (False, torch.bfloat16, 1),
    (False, torch.float, 2),
])
def test_ram_table_is_guarded(tmp_path, monkeypatch, quantized, dtype, shards):
    tensors = _layout(quantized, dtype, shards)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    nbytes = sum(t.nbytes for k, t in tensors.items() if k.endswith((".trellis", ".weight")))

    # one byte short of table + reserve: refused before the table is read
    _set_available(monkeypatch, nbytes - 1)
    mod, stc = _module(tmp_path)
    with pytest.raises(RuntimeError, match = "--ngram_ram"):
        mod.load(torch.device("cpu"))
    stc.close()

    # exactly enough: loads, and the guarded size is what the table occupies in RAM
    _set_available(monkeypatch, nbytes)
    mod, stc = _module(tmp_path)
    mod.load(torch.device("cpu"))
    assert mod.mode == ("trellis_ram" if quantized else "fp16_ram")
    assert len(mod.tables) == 1 and mod.tables[0].dtype == dtype and mod.tables[0].nbytes == nbytes
    mod.unload()
    stc.close()


def test_ram_guard_during_conversion_reload(tmp_path, monkeypatch):
    # convert_model.py reloads each quantized module with stc.new_tensors set; sizing the guard
    # must not go through the loader calls that refuse to run in that state
    tensors = _layout(True, torch.int16, 2)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    nbytes = sum(t.nbytes for k, t in tensors.items() if k.endswith(".trellis"))
    _set_available(monkeypatch, nbytes)
    mod, stc = _module(tmp_path)
    stc.set_new_tensors({"model.layers.0.mlp.down_proj.trellis": torch.zeros(1, dtype = torch.int16)})
    mod.load(torch.device("cpu"))
    assert mod.tables[0].nbytes == nbytes
    stc.set_new_tensors(None)
    mod.unload()
    stc.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
