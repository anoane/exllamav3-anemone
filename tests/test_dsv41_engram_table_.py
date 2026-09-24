"""Read real engram rows off disk with the reference table reader (tests/dsv41_ref/engram_tables.py):
the pread path must equal the mmap path, and the checkpoint's E8M0 scale codes must stay inside the
window where the engine's fp16 dequantisation (engram_torch.dequant_rows) is exact. Every engram
layer's table is checked. CPU only, no exllamav3 import.

    python tests/test_dsv41_engram_table_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip
from dsv41_ref import engram_tables as et


def check_decoders():
    # All finite E8M0 codes, including the subnormal scale at code 0.
    e = np.arange(0, 255, dtype = np.uint8)
    shifted = et.e8m0_to_f32(e)
    arith = np.exp2(e.astype(np.int32) - 127).astype(np.float32)
    assert np.array_equal(shifted, arith), "e8m0 shift decode disagrees with exp2"
    assert et.e8m0_to_f32(np.array([0], dtype = np.uint8))[0] == np.float32(2.0 ** -127)
    assert np.isnan(et.e8m0_to_f32(np.array([255], dtype = np.uint8))[0])

    # the fp8 decoder: 0x7F / 0xFF are NaN in e4m3fn (no inf), 0x7E / 0xFE are +-448,
    # 0x01 the smallest subnormal 2^-9
    dec = et._fp8_e4m3_to_f32(np.array([0x7F, 0xFF, 0x7E, 0xFE, 0x01, 0x00, 0x80], dtype = np.uint8))
    assert np.isnan(dec[:2]).all() and dec[2] == 448.0 and dec[3] == -448.0, dec
    assert dec[4] == 2.0 ** -9 and dec[5] == 0.0 and dec[6] == 0.0, dec


def check_table(path, t, index):
    layer = int(t["engram_layer_ids"][index])
    hd = int(t["engram_head_dim"])
    tbl = et.EngramTable(path, layer, hd)
    try:
        assert tbl.num_embeddings == int(t["engram_num_embeddings"][index]), \
            f"layer {layer}: {tbl.num_embeddings} rows, config says {t['engram_num_embeddings'][index]}"
        assert tbl.group_size == 32, f"layer {layer}: scale group {tbl.group_size}, expected 32"

        rng = np.random.default_rng(index)
        rows = rng.integers(0, tbl.num_embeddings, size = 256, dtype = np.int64)
        # include the extremes: a short read at the last row is the classic bug
        rows[0], rows[1] = 0, tbl.num_embeddings - 1

        # Not timed against each other on purpose: whichever runs first pays the
        # cold read and warms the page cache for the other, so any number here
        # measures ordering, not the readers.
        got = tbl.gather(rows)
        ref = tbl.gather_mmap(rows)

        assert got.shape == (rows.size, hd), got.shape
        assert np.array_equal(got, ref), \
            f"layer {layer}: pread != mmap, max |diff| {np.abs(got - ref).max():.3e}"
        assert np.isfinite(got).all(), f"layer {layer}: non-finite values decoded from the table"
        assert (got != 0).any(), f"layer {layer}: every gathered row decoded to zero"

        # The checkpoint's finite scales stay within the FP16-exact range. Sampling wide:
        sp, _, _, sst, _ = tbl._s
        codes = np.unique(tbl._pread_rows(sp, sst, rng.integers(
            0, tbl.num_embeddings, size = 4096, dtype = np.int64), tbl.scale_groups))
        assert codes.min() >= 1 and codes.max() <= 254, \
            f"layer {layer}: scale codes {codes.min()}..{codes.max()} reach the inexact ends"
        # the module dequantises straight to fp16, exact for codes 112..134 (a multiple of
        # 2^-24 and at most 65504 once scaled); both tables use 116..123 on a full scan
        assert codes.min() >= 112 and codes.max() <= 134, \
            f"layer {layer}: scale codes {codes.min()}..{codes.max()} leave the fp16-exact window 112..134"
    finally:
        tbl.close()
    print(f"  OK  engram table L{layer}: {tbl.num_embeddings:,} rows x {hd}; "
          f"{rows.size} rows gathered, pread == mmap bitwise; scale codes "
          f"{codes.min()}-{codes.max()}, inside the range where the e8m0 "
          f"shift decode and the fp16 dequant are exact")


def main(path):
    t = json.load(open(os.path.join(path, "config.json")))
    t = t.get("text_config", t)
    check_decoders()
    for index in range(len(t["engram_layer_ids"])):
        check_table(path, t, index)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("engram table", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
