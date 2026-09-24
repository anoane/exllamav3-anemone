"""CPU-only check of the V4.1 compressor (architecture/dsv41/compressor.py) and the cached
path's carry ring (modules/dsv41_cached.py): chunked pooling must equal one shot, the carry ring
must equal the stateful chunked path bitwise, and both must equal DeepSeek's own Compressor
(inference/model.py, imported as-is from DSV41_DEEPSEEK_REF with its GPU-kernel modules stubbed)
over a prefill followed by token-by-token decode -- the reference's only two regimes --
including the index key the reference derives from the pre-RoPE latent. No exllamav3 import
(no extension, no GPU).

    python tests/test_dsv41_compressor_.py      or      python -m pytest tests/test_dsv41_compressor_.py

Without DSV41_DEEPSEEK_REF the comparison with DeepSeek's Compressor is skipped."""
import os, sys, types
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import load_package_file, skip
from dsv41_ref import deepseek


def _load_cached_helpers():
    """Load the cached helpers and their real CPU-safe copy dependency without CUDA imports."""
    P = "dsv41cstub"
    for n in (P, f"{P}.modules", f"{P}.util", f"{P}.architecture", f"{P}.architecture.dsv41"):
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    c = types.ModuleType(f"{P}.constants"); c.PAGE_SIZE = 256; sys.modules[c.__name__] = c
    load_package_file("exllamav3/util/device_copy.py", f"{P}.util.device_copy")
    load_package_file("exllamav3/architecture/dsv41/compressor.py", f"{P}.architecture.dsv41.compressor")
    return load_package_file("exllamav3/modules/dsv41_cached.py", f"{P}.modules.dsv41_cached", f"{P}.modules")


def check_carry_ring(cp):
    """The cached path's position-indexed carry ring (CompressCarry) against the stateful
    chunked path (compress_chunk + CompressorState) and the one-shot reference, bitwise, over a
    prefill that ends mid-group followed by single-token decode steps. Needs no reference code."""
    dc = _load_cached_helpers()
    torch.manual_seed(2)
    hd, eps, total, P = 32, 1e-20, 61, 23
    for ratio in (2, 1):
        kv = torch.randn(total, hd)
        sc = torch.randn(total, hd) if ratio > 1 else None
        w = 1.0 + 0.1 * torch.randn(hd)
        spans = [(0, P)] + [(p, p + 1) for p in range(P, total)]
        st = cp.CompressorState(hd, ratio) if ratio > 1 else None
        carry = torch.zeros(256 + ratio, 2 * hd) if ratio > 1 else None
        got1, got2 = [], []
        for a, b in spans:
            lat, g = cp.compress_chunk(kv[a:b], None if sc is None else sc[a:b], a, ratio, w, eps, st)
            got1.append(lat)
            lat, first = dc.CompressCarry.step(carry, kv[a:b], None if sc is None else sc[a:b], a, ratio, w, eps)
            assert first == a // ratio, (ratio, a, first)
            got2.append(lat)
        got1, got2 = torch.cat(got1), torch.cat(got2)
        ref = cp.compress_reference(kv, sc, ratio, w, eps) if ratio > 1 else cp.compress_chunk(kv, None, 0, 1, w, eps)[0]
        assert got2.shape == (total // ratio, hd), got2.shape
        assert torch.equal(got1, got2), f"ratio {ratio}: CompressCarry != compress_chunk"
        assert torch.equal(got2, ref), f"ratio {ratio}: chunked carry ring != one shot"
    print(f"  OK  carry ring: CompressCarry == compress_chunk == one shot, bitwise, at rates 2 and 1 "
          f"(prefill {P} + {total - P} decode steps)")


def check_vs_deepseek(cp):
    """DeepSeek's Compressor.forward: prefill at start_pos 0 (a trailing odd row waits in
    kv_state/score_state), then one token per call, pooling when (start_pos + 1) % ratio == 0.
    The port's chunked path (compress_chunk + CompressorState) and the cached path's
    position-indexed carry ring (CompressCarry) must produce the same latents. Tolerance 1e-5
    relative: the reference pools with torch.softmax, the port writes the softmax out (so any
    chunking is bitwise; see group_pool)."""
    ds = deepseek.load_model()
    dc = _load_cached_helpers()
    torch.manual_seed(1)
    dim, hd = 64, 32
    for ratio in (2, 1):
        args = ds.ModelArgs(max_batch_size = 1, max_seq_len = 256, dim = dim, head_dim = hd,
                            n_layers = 2, n_mtp_layers = 0, compress_ratios = (0, ratio),
                            kv_source_layers = (1,), index_source_layers = (1,), norm_eps = 1e-20,
                            index_head_dim = 16, rope_head_dim = 8, q_lora_rank = 16, index_n_heads = 2)
        comp = ds.Compressor(args, 1).float()
        for prm in comp.parameters():
            prm.data.normal_(0, 0.3)
        with torch.no_grad():
            comp.norm.weight.data = 1.0 + 0.1 * torch.randn(hd)
        total, P = 60, 23                                  # odd prefill: decode opens mid-group
        x = torch.randn(1, total, dim)
        ref = []
        with torch.no_grad():
            out = comp(x[:, :P], 0)
            if out is not None: ref.append(out[0])
            for p in range(P, total):
                out = comp(x[:, p:p + 1], p)
                if out is not None: ref.append(out[0])
        ref = torch.cat(ref, dim = 0)
        assert ref.shape[0] == total // ratio, ref.shape

        kv = torch.nn.functional.linear(x[0], comp.wkv.weight)
        sc = torch.nn.functional.linear(x[0], comp.wgate.weight) if ratio > 1 else None
        w, eps = comp.norm.weight.data, args.norm_eps
        # port, stateful chunked path
        st = cp.CompressorState(hd, ratio)
        got1 = []
        for a, b in [(0, P)] + [(p, p + 1) for p in range(P, total)]:
            lat, g = cp.compress_chunk(kv[a:b], None if sc is None else sc[a:b], a, ratio, w, eps,
                                       st if ratio > 1 else None)
            got1.append(lat)
        got1 = torch.cat(got1)
        # port, cached-path carry ring
        carry = torch.zeros(256 + ratio, 2 * hd) if ratio > 1 else None
        got2 = []
        for a, b in [(0, P)] + [(p, p + 1) for p in range(P, total)]:
            lat, first = dc.CompressCarry.step(carry, kv[a:b], None if sc is None else sc[a:b], a, ratio, w, eps)
            assert first == a // ratio
            got2.append(lat)
        got2 = torch.cat(got2)
        es = []
        for name, got in (("compress_chunk", got1), ("CompressCarry", got2)):
            e = ((got - ref).norm() / ref.norm()).item()
            assert got.shape == ref.shape and e < 1e-5, f"ratio {ratio}: {name} vs DeepSeek rel {e:.2e}"
            es.append(e)
        assert torch.equal(got1, got2), f"ratio {ratio}: the two port paths disagree"

        # index key from the pre-RoPE latent: k_norm(wk(latent)), the reference Indexer's
        # arithmetic, against the port's indexer_k_from_latent
        ix = ds.Indexer(args, 1).float()
        for prm in ix.parameters():
            prm.data.normal_(0, 0.3)
        with torch.no_grad():
            k_ref = ix.k_norm(ix.wk(ref.unsqueeze(0)))[0]
        k = cp.indexer_k_from_latent(ref, ix.wk.weight, ix.k_norm.weight, eps)
        ek = ((k - k_ref).norm() / k_ref.norm()).item()
        assert ek < 1e-5, f"ratio {ratio}: index K vs DeepSeek rel {ek:.2e}"
        print(f"  OK  ratio {ratio}: prefill {P} + {total - P} decode steps, {ref.shape[0]} latents == "
              f"DeepSeek Compressor (rel {max(es):.1e}; compress_chunk and CompressCarry bitwise equal), "
              f"index K == DeepSeek Indexer (rel {ek:.1e})")

def _compressor():
    return load_package_file("exllamav3/architecture/dsv41/compressor.py", "_dsv41_compressor")


def check_port(cp):
    """The port alone: rate 1's gate cancels, chunked rate-2 pooling equals one shot, the
    trailing partial group, and the indexer key from the latent."""
    torch.manual_seed(0)
    hd, seq, eps = 512, 97, 1e-20
    kv = torch.randn(seq, hd, dtype = torch.float32)
    score = torch.randn(seq, hd, dtype = torch.float32)
    w = torch.randn(hd, dtype = torch.float32) * 0.1 + 1.0

    # ratio 1: gate cancels, so a gated pool over groups of one must equal the
    # ungated path the checkpoint actually uses (no wgate tensor exists there)
    r1, g0 = cp.compress_chunk(kv, None, 0, 1, w, eps)
    assert r1.shape == (seq, hd) and g0 == 0, r1.shape
    gated1 = cp.group_pool(kv.view(-1, 1, hd), score.view(-1, 1, hd))
    assert torch.equal(cp.rms_norm(gated1, w, eps), r1), "ratio-1 gate did not cancel"

    # ratio 2: one shot
    ref = cp.compress_reference(kv, score, 2, w, eps)
    assert ref.shape == (seq // 2, hd), ref.shape

    # ratio 2: chunked with state, including chunks that split a group
    for chunks in ([32, 32, 33], [1, 96], [7, 5, 3, 82], [96, 1], [48, 49]):
        st = cp.CompressorState(hd, 2)
        out, pos, first = [], 0, None
        for n in chunks:
            lat, g = cp.compress_chunk(kv[pos:pos + n], score[pos:pos + n], pos, 2, w, eps, st)
            if lat.shape[0]:
                if first is None: first = g
                else: assert g == first + sum(o.shape[0] for o in out), \
                    f"group index gap at chunk starting {pos}"
                out.append(lat)
            pos += n
        got = torch.cat(out, dim = 0)
        assert got.shape == ref.shape, f"{chunks}: {got.shape} != {ref.shape}"
        assert torch.equal(got, ref), \
            f"{chunks}: chunked != one shot, max err {(got - ref).abs().max():.3e}"
        assert st.pending == seq % 2, f"{chunks}: {st.pending} rows left open"

    # the trailing partial group is dropped without a state, kept with one
    lat, _ = cp.compress_chunk(kv, score, 0, 2, w, eps, None)
    assert lat.shape[0] == seq // 2, "stateless path emitted a partial group"

    # indexer K: k_norm(wk(latent)), the V4.1 replacement for a second compressor
    ihd = 128
    wk = torch.randn(ihd, hd, dtype = torch.float32) * 0.02
    kn = torch.randn(ihd, dtype = torch.float32) * 0.1 + 1.0
    k = cp.indexer_k_from_latent(ref, wk, kn, eps)
    assert k.shape == (seq // 2, ihd), k.shape
    assert torch.isfinite(k).all(), "non-finite index keys"
    rms = k.pow(2).mean(dim = -1).sqrt()
    assert (rms > 0).all(), "index key collapsed to zero"

    print(f"  OK  compressor: rate 1 gate cancels; rate 2 chunked == one shot over "
          f"5 chunkings (seq {seq}, {ref.shape[0]} groups); indexer K {tuple(k.shape)}")


def test_compressor():
    """pytest entry point: the port checks and the carry ring, no reference code."""
    cp = _compressor()
    check_port(cp)
    check_carry_ring(cp)


def test_compressor_vs_deepseek():
    """pytest entry point: against DeepSeek's Compressor, skipped without DSV41_DEEPSEEK_REF."""
    if deepseek.ref_dir() is None:
        import pytest
        pytest.skip("compressor vs DeepSeek Compressor: DSV41_DEEPSEEK_REF not set")
    check_vs_deepseek(_compressor())


def main():
    test_compressor()
    if deepseek.ref_dir() is None:
        skip("compressor vs DeepSeek Compressor", "DSV41_DEEPSEEK_REF not set")
    else:
        check_vs_deepseek(_compressor())


if __name__ == "__main__":
    main()
