"""Build the real V4.1 graph from a checkpoint and check it against the shards.

    PYTHONPATH=$PWD python3 tests/test_dsv41_build_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)

Needs a working exllamav3 import (the compiled extension). Direct script invocation puts
``tests/`` on sys.path[0]; the working directory alone does not ensure package discovery.

No GPU work and no weight loading beyond one small projection: module
construction is lazy. Check graph shape/key claims plus the CPU head-collapse
and carried-pre safe-copy contract, and that the converter refuses the
architecture before any work; this is not a full-model numerical test.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip

def main(path):
    from exllamav3 import Config, Model

    check_copy_contract()
    check_forward_rows()
    config = Config.from_directory(path)
    assert config.arch_string == "DeepseekV41ForCausalLM", config.arch_string
    model = Model.from_config(config)

    c = config
    n = c.num_hidden_layers
    blocks = model.modules[model.first_block_idx : model.first_block_idx + n]
    assert len(blocks) == n, f"{len(blocks)} blocks for {n} layers"
    assert len(model.modules) == n + 5, \
        f"{len(model.modules)} modules, expected {n} blocks + embed/expand/hc_head/norm/head"

    # every tensor the graph claims must exist in the checkpoint, and the
    # topology-dependent ones must appear on exactly the right layers
    stc = c.stc
    _SFX = ("weight", "suh", "svh", "trellis", "mul1", "scale")
    def present(k):
        return stc.has_tensor(k) or any(stc.has_tensor(f"{k}.{x}") for x in _SFX)

    missing = []
    for i, b in enumerate(blocks):
        attn = b.attn
        assert attn.layer_idx == i
        assert attn.compress_ratio == c.compress_ratios[i]
        missing += [k for k in attn.expected_keys() if not present(k)]
        assert (attn.compressor is not None) == c.is_kv_source(i), f"layer {i} compressor"
        assert (attn.indexer is not None) == c.is_index_source(i), f"layer {i} indexer"
        if attn.indexer is not None:
            assert (attn.indexer.wk is not None) == c.index_owns_k(i), f"layer {i} wk"
        eg = getattr(b, "engram", None)
        assert (eg is not None) == c.has_engram(i), f"layer {i} engram"
        if eg is not None:
            assert eg.wkv.out_features_unpadded == c.hidden_size * (c.hc_mult + 1), \
                f"layer {i} engram wkv out {eg.wkv.out_features_unpadded}"
            # the table stays on disk: placement must not see it. weights_numel is what the
            # autosplit sizes: wkv and the two gate vectors, nothing of the table
            dev_numel = eg.wkv.weights_numel() + 2 * c.hc_mult * c.hidden_size
            assert eg.weights_numel() == dev_numel, \
                f"layer {i}: weights_numel {eg.weights_numel()} counts more than the device weights"
            assert eg.table_numel() > 50 * eg.weights_numel(), \
                f"layer {i}: table {eg.table_numel()} is not the dominant term"
    assert not missing, f"graph claims {len(missing)} absent tensors, e.g. {missing[:5]}"

    # A converted checkpoint that keeps the original engram shards carries engram.wkv twice:
    # exl3 (suh/svh/trellis/mul1) and the original fp8 pair (weight/scale). They do not collide
    # by name, so nothing warns if the wrong one wins -- the projection would just quietly use
    # stale weights. Load one and look.
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    for i, eg in model.engram_modules.items():
        if not stc.has_tensor(f"{eg.key}.wkv.trellis"):
            continue
        both = stc.has_tensor(f"{eg.key}.wkv.weight")
        eg.wkv.load(torch.device("cpu"))
        assert isinstance(eg.wkv.inner, LinearEXL3), \
            f"layer {i}: engram wkv loaded as {type(eg.wkv.inner).__name__}, not exl3"
        got = set(eg.wkv.get_tensors())
        assert f"{eg.key}.wkv.weight" not in got, \
            f"layer {i}: engram wkv took the stale fp8 copy"
        eg.wkv.unload()
        print(f"  OK  layer {i}: engram wkv loads as exl3" +
              (" (the checkpoint also carries the original fp8 pair)" if both else ""))

    ng = sum(1 for i in range(n) if c.is_kv_source(i))
    ni = sum(1 for i in range(n) if c.is_index_source(i))
    eg_dev = sum(m.weights_numel() for m in model.engram_modules.values())
    eg_tab = sum(m.table_numel() for m in model.engram_modules.values())
    print(f"  OK  built {len(model.modules)} modules: {n} blocks, {ng} compressors, "
          f"{ni} indexers, {len(model.engram_modules)} engram "
          f"({eg_dev / 2**20:.0f} MiB on device, {eg_tab / 2**30:.0f} GiB on disk)")

    check_assembly(model)
    check_conversion_refused(path)
    check_keymap(model, path)
    check_per_forward(model)
    check_chat_prompt(model, path)


def check_conversion_refused(path):
    """convert.py stops at the model it builds, before any work, with V4.1's own reason."""
    import contextlib, io
    from exllamav3.architecture.deepseek_v41 import QUANTIZE_REFUSAL
    from exllamav3.conversion.convert_model import get_base_model
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            get_base_model({"in_dir": path})
    except NotImplementedError as e:
        assert str(e) == QUANTIZE_REFUSAL, str(e)
    else:
        raise AssertionError("the converter accepted a DeepSeek-V4.1 checkpoint")
    print(f"  OK  conversion refused before any work: {QUANTIZE_REFUSAL[:60]}...")


def check_copy_contract():
    """Execute the shared copy/head paths without constructing any CUDA tensors.

    This must run in the support-only tree, before optional transport code exists:
    importability alone does not establish that a helper is bound when called.
    """
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    from exllamav3.modules import dsv41_block as block

    assert not torch.cuda.is_initialized(), "copy contract must remain CPU-only"
    streams = torch.arange(24, dtype = torch.float).reshape(1, 3, 4, 2)
    pre = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype = torch.float).expand(1, 3, 4)
    params = {"dsv41_hc_pre": pre}
    head = block.DSV41HeadCollapse(config = None, key = "cpu_head_contract", hc_mult = 4)
    expected = (pre.unsqueeze(-1) * streams).sum(dim = 2)
    # This real CPU call fails with NameError if the support tree forgot the import.
    assert torch.equal(head.forward(streams, params), expected)
    assert block._carried(params, streams) is pre
    assert block._carried({}, streams) is None

    foreign_pre = SimpleNamespace(shape = pre.shape, device = "synthetic-remote-device")
    calls = []
    def copied(value, device):
        assert value is foreign_pre and device == streams.device
        calls.append((value, device))
        return pre
    # Do not create a missing attribute: both paths must already import the helper.
    with patch.object(block, "to_device", copied, create = False):
        remote_params = {"dsv41_hc_pre": foreign_pre}
        assert block._carried(remote_params, streams) is pre
        assert torch.equal(head.forward(streams, remote_params), expected)
    assert len(calls) == 2
    assert not torch.cuda.is_initialized()
    print("  OK  CPU copy contract: actual head collapse, same-device/empty carried pre, "
          "and both cross-device helper paths")


def check_forward_rows():
    """util.tensor.forward_rows (EXL3_EXACT_ROWS): one call per row, in the shape a one-row call
    passes, each result copied out before the next call."""
    import torch
    from exllamav3.util.tensor import forward_rows
    buf = torch.empty(3, dtype = torch.double)
    for shape in ((4, 5), (1, 4, 5)):
        x = torch.arange(20, dtype = torch.float).reshape(shape)
        seen = []
        def fn(row):
            seen.append(tuple(row.shape))
            # the same buffer on every call, as the MoE decode kernels return theirs
            buf.copy_(torch.stack([row.sum(), row.min(), row.max()]))
            return buf.view(*([1] * (row.dim() - 1)), 3)
        y = forward_rows(fn, x)
        assert seen == [(1,) * (len(shape) - 1) + (5,)] * 4, seen
        assert y.shape == shape[:-1] + (3,) and y.dtype == torch.double, (y.shape, y.dtype)
        rows = x.reshape(4, 5).double()
        want = torch.stack([rows.sum(-1), rows.amin(-1), rows.amax(-1)], dim = -1)
        assert torch.equal(y.reshape(4, 3), want), y
    # a strided input reaches fn row by row too
    x = torch.arange(40, dtype = torch.float).reshape(4, 10)[:, ::2]
    assert torch.equal(forward_rows(lambda row: row * 2, x), x * 2)
    print("  OK  forward_rows: (1, D) and (1, 1, D) rows, the input's leading shape, the results' dtype, "
          "distinct rows from a reused result buffer")


# per-forward keys prepare_inputs must set fresh on every call
PER_FORWARD = ("dsv41_pools", "dsv41_hc_pre", "dsv41_pool_mirrors", "dsv41_topk", "dsv41_candidates",
               "dsv41_dev_memo", "dsv41_rope", "dsv41_engram_lookback_fwd")


def check_per_forward(model):
    """A params dict reused across forwards: every per-forward key comes out fresh, and an
    explicit engram lookback reaches only the forward it was given to."""
    import torch
    from exllamav3.modules.dsv41_engram import _EngramCtx
    from exllamav3.architecture.dsv41.engram_torch import UNK
    assert set(PER_FORWARD) <= set(model.PER_FORWARD_KEYS), set(PER_FORWARD) - set(model.PER_FORWARD_KEYS)
    stale = object()
    params = {"attn_mode": "flash_attn_nc"}
    for k in model.PER_FORWARD_KEYS:
        params[k] = stale
    ids = torch.randint(0, 1000, (1, 6))
    lb = torch.tensor([[11, 12, 13]])
    params["dsv41_engram_lookback"] = lb
    model.prepare_inputs(ids, params)
    left = [k for k in model.PER_FORWARD_KEYS if params.get(k, stale) is stale]
    assert not left, f"stale per-forward keys after prepare_inputs: {left}"
    assert all(params[k] == {} for k in model.PER_FORWARD_KEYS
               if k not in ("dsv41_hc_pre", "dsv41_engram_lookback_fwd")), "a per-forward dict is not empty"
    assert params["dsv41_hc_pre"] is None and params["input_ids"] is ids
    assert params["dsv41_engram_lookback_fwd"] is lb and "dsv41_engram_lookback" not in params

    # the engram reads the lookback in its forward, and not in the next one
    eg = next(iter(model.engram_modules.values()))
    eg.engram_ctx = _EngramCtx.get(model.config, eg.layout)
    try:
        h1, _ = eg.history(ids, params)
        assert torch.equal(h1[:, :3], lb), h1[:, :3]
        model.prepare_inputs(ids, params)                # the next forward, same dict
        assert params["dsv41_engram_lookback_fwd"] is None
        h2, _ = eg.history(ids, params)
        assert (h2[:, :3] == UNK).all(), f"the second forward hashed with the first one's lookback: {h2[:, :3]}"
    finally:
        eg.engram_ctx = None
    print(f"  OK  per-forward state: {len(model.PER_FORWARD_KEYS)} keys fresh after prepare_inputs on a "
          f"reused dict; an explicit engram lookback reaches only its own forward")


def check_chat_prompt(model, path):
    """default_chat_prompt is what the checkpoint's chat template renders for one user turn."""
    try:
        import jinja2
    except ImportError:
        print("  --  default_chat_prompt: jinja2 not installed, template comparison skipped")
        return
    src = open(os.path.join(path, "chat_template.jinja")).read()
    env = jinja2.Environment()
    def raise_exception(msg):
        raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    t = env.from_string(src)
    for system in (None, "Answer briefly."):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": "What is 2 + 2?"}]
        want = t.render(messages = msgs, add_generation_prompt = True)
        got = model.default_chat_prompt("What is 2 + 2?", system)
        assert got == want, f"default_chat_prompt {got!r} != template {want!r}"
    print("  OK  default_chat_prompt == chat_template.jinja rendered for one turn, with and without "
          "a system prompt")


def check_assembly(model):
    """Block/MoE classes, the job state class and module_plan vs the graph."""
    from exllamav3.cache.dsv41 import DSV41State
    from exllamav3.modules.dsv41_block import DSV41Block, DSV41HeadCollapse
    from exllamav3.modules.dsv41_moe import DSV41MoE
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    c = model.config
    n = c.num_hidden_layers
    blocks = model.modules[model.first_block_idx : model.first_block_idx + n]
    assert all(type(b) is DSV41Block for b in blocks), {type(b).__name__ for b in blocks}
    assert all(type(b.mlp) is DSV41MoE and isinstance(b.mlp, BlockSparseMLP) for b in blocks)
    assert [b.mlp.layer_idx for b in blocks] == list(range(n))
    assert type(model.modules[model.first_block_idx + n]) is DSV41HeadCollapse
    # without it a Cache has no job state class (V4 sets the same at deepseek_v4.py)
    assert model.recurrent_state_cls is DSV41State, model.recurrent_state_cls
    assert model.caps.get("supports_tp") is False
    assert model.caps.get("can_quantize") is False and model.caps.get("quantize_refusal")

    plan = model.module_plan()
    kinds = [p["kind"] for p in plan]
    assert kinds == ["Embedding", "ExpandStreams"] + ["DSV41Block"] * n + \
        ["DSV41HeadCollapse", "RMSNorm", "Linear"], kinds
    blk = [p for p in plan if p["kind"] == "DSV41Block"]
    assert all(p["mlp_kind"] == "DSV41MoE" and p["mlp"] == f"layers.{i}.ffn"
               for i, p in enumerate(blk))
    assert all(p["owns_ring"] for p in blk)
    # pools: the kv sources, plus a replica owner only where an explicit placement's split wires
    # one (none without a placement); every compressed layer reads its route owner's pool
    pools = [i for i, p in enumerate(blk) if p["allocates_cache"]]
    wired = {i for i, r in model.pool_routes.items() if r[1] is not None}
    assert wired == (set(model.placement.replicas) if model.placement else set()), wired
    assert pools == sorted(set(c.kv_source_layer_ids) | wired), (pools, c.kv_source_layer_ids, wired)
    assert all(p["cache_owner"] == (p["pool_route"][0] if p["pool_route"] else None) for p in blk)
    if model.placement is None:
        assert all(p["cache_owner"] == c.kv_source_for(i) for i, p in enumerate(blk))
    assert all(p["cache_owner"] is None for p in blk if p["layer_type"] == "sliding")

    print(f"  OK  assembly: {n} DSV41Block + DSV41MoE, DSV41HeadCollapse, recurrent_state_cls "
          f"DSV41State, pools on {pools}")


def check_keymap(model, path):
    """tools/dsv41_keymap.py claims every tensor the checkpoint carries for the text model, and
    every stem it claims has an owner in the real graph"""
    import importlib.util
    from exllamav3.modules.dsv41_block import DSV41HyperConnection
    spec = importlib.util.spec_from_file_location(
        "_dsv41_keymap", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "tools", "dsv41_keymap.py"))
    km = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(km)
    c = model.config
    n = c.num_hidden_layers
    blocks = model.modules[model.first_block_idx : model.first_block_idx + n]
    r = km.check(path)
    assert not r["missing"] and not r["unclaimed"], (r["missing"][:5], r["unclaimed"][:5])
    owners = {m.key for m in model}
    for m in model:
        if isinstance(m, DSV41HyperConnection):
            owners |= {f"{m.key}_{t}" for t in ("fn", "base", "scale")}
    for b in blocks:
        owners.add(f"{b.attn.key}.attn_sink")
    for eg in model.engram_modules.values():
        owners |= {f"{eg.key}.{t}" for t in ("wkv", "q_weight", "k_weight", "embed")}
    orphan = sorted(set(km.expected(dict(
        num_hidden_layers = n, o_groups = c.o_groups, compress_ratios = c.compress_ratios,
        kv_source_layer_ids = c.kv_source_layer_ids,
        index_source_layer_ids = c.index_source_layer_ids,
        n_routed_experts = c.n_routed_experts, engram_layer_ids = c.engram_layer_ids))) - owners)
    assert not orphan, f"keymap claims {len(orphan)} stems no module owns, e.g. {orphan[:5]}"
    print(f"  OK  keymap claims all {r['expected']:,} text stems, each owned by a graph module "
          f"({len(r['skipped'])} mtp/vision/aligner stems not loaded)")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 graph build", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
