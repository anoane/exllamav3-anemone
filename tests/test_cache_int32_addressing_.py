"""
The Triton attention kernels address paged cache planes with int32 element offsets (row * width + column), so a
plane can hold at most 2^31 elements. Regression: DSA pools, DSpark drafter caches and MLA caches were constructed at
any size, and past the limit the offsets wrapped and the kernels silently read (and, for MLA, wrote) outside the tensor
(DeepSeek-V4 CSA pool_c, 448 wide, from max_num_tokens = 19,174,144; DSpark kv and MLA latent, 512 wide, from
4,194,560). Construction only, allocates nothing.
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from types import SimpleNamespace
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.dsa import CacheLayer_dsa, CacheLayer_dspark
from exllamav3.cache.mla import CacheLayer_MLA_fp16, CacheLayer_MLA_quant


def dsv4_attn(layer_type):
    return SimpleNamespace(
        compress_rate = 4 if layer_type == "csa" else 128,
        head_dim = 512,
        rope_head_dim = 64,
        index_head_dim = 128,
        layer_type = layer_type,
    )


@pytest.mark.parametrize("layer_type, k_bits, limit", [
    # fp16 pool_c, 448 wide: 2^31 // 448 = 4,793,490 rows, 64 entries per page at 4:1
    ("csa", 0, 19_173_888),
    # packed pool_c is at most 14 * 8 = 112 words wide, the 128-wide indexer key plane binds
    ("csa", 4, 67_108_864),
    ("csa", 8, 67_108_864),
    # 2 entries per page at 128:1, pool_c binds
    ("hca", 0, 613_566_720),
    ("hca", 8, 2_454_266_880),
])
def test_dsa_pool_limit(layer_type, k_bits, limit):
    attn = dsv4_attn(layer_type)
    CacheLayer_dsa(None, attn, 0, limit, k_bits = k_bits)
    with pytest.raises(ValueError, match = "int32"):
        CacheLayer_dsa(None, attn, 0, limit + PAGE_SIZE, k_bits = k_bits)


def test_dspark_cache_limit():
    # fp16 kv, 512 wide: exactly 2^31 elements at 4,194,304 tokens
    attn = SimpleNamespace(head_dim = 512)
    CacheLayer_dspark(None, attn, 0, 4_194_304)
    with pytest.raises(ValueError, match = "int32"):
        CacheLayer_dspark(None, attn, 0, 4_194_304 + PAGE_SIZE)


@pytest.mark.parametrize("k_bits, indexer, limit", [
    # fp16 latent, 512 wide: exactly 2^31 elements at 4,194,304 tokens
    (0, False, 4_194_304),
    (0, True, 4_194_304),
    # packed latent, 16 groups * k_bits words
    (4, False, 33_554_432),
    (8, False, 16_777_216),
    # 128-wide indexer key plane binds over the 64-word Q4 latent
    (4, True, 16_777_216),
])
def test_mla_cache_limit(k_bits, indexer, limit):
    attn = SimpleNamespace(kv_lora_rank = 512, qk_rope_head_dim = 64)
    if indexer:
        # DSA-on-MLA indexer key plane and pooled key plane
        attn.idx_plane_dim = 128
        attn.index_kpool = 4
        attn.index_head_dim = 128

    def make(max_num_tokens):
        if k_bits:
            return CacheLayer_MLA_quant(None, attn, 0, max_num_tokens, k_bits = k_bits)
        return CacheLayer_MLA_fp16(None, attn, 0, max_num_tokens)

    make(limit)
    with pytest.raises(ValueError, match = "int32"):
        make(limit + PAGE_SIZE)
