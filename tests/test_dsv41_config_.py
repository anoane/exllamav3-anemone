"""
CPU-only checks for the DeepSeek-V4.1 config topology and engram layout.

Runs without a GPU and without the compiled extension: the config module is
loaded directly so the package __init__ (which JIT-builds the CUDA ext) is
not triggered. Needs only the checkpoint's config.json.

    python tests/test_dsv41_config_.py [checkpoint-dir]    (default: $DSV41_MODEL_DIR)
"""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv41_ref import model_dir, skip

no_default = object()


def _read_dict(d, typ, keys, default=no_default):
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        cur, ok = d, True
        for part in k.split("->"):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    if default is no_default:
        raise KeyError(keys)
    return default


class _StubModel:
    def __init__(self, config, **kw):
        self.config = config


class _StubConfig:
    def __init__(self, directory, components, **kw):
        with open(os.path.join(directory, "config.json")) as f:
            self.config_dict = json.load(f)

    def read_cfg(self, *a):
        return _read_dict(self.config_dict, *a)


def load_config_module():
    """Import architecture/deepseek_v41.py without the compiled extension."""
    for name, attrs in (
        ("exllamav3", {}),
        ("exllamav3.util", {}),
        ("exllamav3.util.file", {"no_default": no_default}),
        ("exllamav3.model", {}),
        ("exllamav3.model.config", {"Config": _StubConfig}),
        ("exllamav3.model.model", {"Model": _StubModel}),
    ):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules.setdefault(name, m)

    pkg = types.ModuleType("exllamav3.architecture")
    pkg.__path__ = [os.path.join(_HERE, "exllamav3", "architecture")]
    sys.modules["exllamav3.architecture"] = pkg

    spec = importlib.util.spec_from_file_location(
        "exllamav3.architecture.deepseek_v41",
        os.path.join(_HERE, "exllamav3", "architecture", "deepseek_v41.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(directory):
    mod = load_config_module()
    cfg = mod.DeepseekV41Config(directory)

    # compress_ratios are rates, not kind selectors
    assert set(cfg.compress_ratios) <= {0, 1, 2}
    assert len(cfg.layer_types) == cfg.num_hidden_layers

    # a compressed layer always resolves to a source at or below it
    for i, r in enumerate(cfg.compress_ratios):
        if r:
            assert cfg.kv_source_for(i) <= i
            assert cfg.index_source_for(i) <= i
        else:
            assert cfg.kv_source_for(i) is None

    # Every compressed layer occurs exactly once under its resolved owner.
    members = [m for _, ms in cfg.kv_groups() for m in ms]
    assert members == [i for i, ratio in enumerate(cfg.compress_ratios) if ratio]
    for source, group in cfg.kv_groups():
        assert source == group[0] and all(cfg.kv_source_for(i) == source for i in group)

    # an index source owns K only if it is also a kv source
    for i in cfg.index_source_layer_ids:
        assert cfg.index_owns_k(i) == (i in cfg.kv_source_layer_ids)

    # engram bucket layout must reproduce the checkpoint's own row counts
    layout = cfg.engram_layout()
    if layout is not None:
        for i, lid in enumerate(layout.layer_ids):
            total = sum(p for per in layout.primes[i] for p in per)
            assert total == layout.num_embeddings[i], (
                f"engram layer {lid}: primes sum to {total}, checkpoint "
                f"declares {layout.num_embeddings[i]}"
            )
        flat = [p for layer in layout.primes for per in layer for p in per]
        assert len(flat) == len(set(flat)), "engram bucket primes must be disjoint"
        assert (layout.hash_multipliers % 2 == 1).all(), "multipliers must be odd"
        assert all(np.all(np.diff(r) > 0) for r in layout.offsets)

    print(f"  OK  {cfg.arch_string}: {cfg.num_hidden_layers} layers, "
          f"groups {[(s, f'{m[0]}-{m[-1]}') for s, m in cfg.kv_groups()]}, "
          f"engram {cfg.engram_layer_ids or 'none'}")

    check_refusals(mod, directory)

    check_placement_topology(cfg, directory)


def check_refusals(mod, directory):
    """
    A config that parses but describes another model would build a wrong one without an
    error. Each mutation of the real config below must be refused, naming its reason; the
    unmutated config parses, with DeepSeek's defaults where keys are absent.
    """
    with open(os.path.join(directory, "config.json")) as f:
        base = json.load(f)

    def parse(mutate):
        d = copy.deepcopy(base)
        mutate(d.get("text_config", d))
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(d, f)
            return mod.DeepseekV41Config(tmp)

    def set_(k, v):
        def m(t): t[k] = v
        return m

    def del_(*ks):
        def m(t):
            for k in ks:
                t.pop(k, None)
        return m

    def ratio(i, r):
        def m(t): t["compress_ratios"][i] = r
        return m

    t0 = base.get("text_config", base)
    n = t0["num_hidden_layers"]
    refused = {
        "compress_ratios shorter than the model": (set_("compress_ratios", t0["compress_ratios"][:n - 1]), "entries for"),
        "compress_ratios missing": (del_("compress_ratios"), "compress_ratios is missing"),
        "more than one kv head": (set_("num_key_value_heads", 2),
                                  "sets num_key_value_heads = 2, but this implementation supports only 1"),
        "a kv source at rate 0": (ratio(2, 0), "sliding-window layer"),
        "a layer at another rate than its kv source": (ratio(3, 1), "reads kv source"),
        "a kv source that is not an index source": (set_("index_source_layer_ids", [2, 14, 20, 24, 28, 32, 36]),
                                                    "is not an index source"),
        "index_topk missing": (del_("index_topk"), "index_topk"),
        "index_n_heads 0": (set_("index_n_heads", 0), "index_n_heads"),
        "index_head_dim missing": (del_("index_head_dim"), "index_head_dim"),
        "engram_pad_token_id missing": (del_("engram_pad_token_id"), "engram_pad_token_id"),
        "candidate sizes without a source": (del_("candidate_source_layer_id"), "no source"),
        "candidates on a different compression grid": (set_("candidate_source_layer_id", 14),
                                                       "compression grid"),
        "another scoring function": (set_("scoring_func", "softmax"), "scoring_func"),
        "another top-k method": (set_("topk_method", "greedy"), "topk_method"),
        "unnormalised routing weights": (set_("norm_topk_prob", False), "norm_topk_prob"),
    }
    for what, (mutate, why) in refused.items():
        try:
            parse(mutate)
        except ValueError as e:
            assert why in str(e), f"{what}: refused for another reason: {e}"
        else:
            raise AssertionError(f"accepted a config with {what}")

    cfg = parse(del_("rms_norm_eps", "scoring_func", "topk_method", "norm_topk_prob", "hc_mult",
                     "hc_sinkhorn_iters", "o_groups", "n_shared_experts"))
    assert cfg.rms_norm_eps == 1e-20, cfg.rms_norm_eps
    got = (cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.o_groups, cfg.n_shared_experts)
    assert got == (4, 20, 8, 1), got
    # entries past num_hidden_layers describe the draft blocks, which the text model ignores
    trimmed = parse(set_("compress_ratios", t0["compress_ratios"][:n]))
    assert trimmed.compress_ratios == cfg.compress_ratios
    assert not hasattr(cfg, "engram_table_dir")
    print(f"  OK  config refusals: {len(refused)} mutations of the real config refused with their "
          f"reason; absent keys take DeepSeek's defaults (rms_norm_eps 1e-20, hc_mult 4, "
          f"hc_sinkhorn_iters 20, o_groups 8, n_shared_experts 1); draft entries of compress_ratios "
          f"ignored")


def check_placement_topology(cfg, directory):
    """
    placement.py's Topology reads config.json without the package; it must answer
    every topology query exactly as the config class does, and give the same free
    cuts and crossings.
    """
    spec = importlib.util.spec_from_file_location(
        "_placement", os.path.join(_HERE, "exllamav3", "architecture", "dsv41", "placement.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    with open(os.path.join(directory, "config.json")) as f:
        topo = P.Topology.from_config_dict(json.load(f))
    n = cfg.num_hidden_layers
    assert topo.num_hidden_layers == n and topo.compress_ratios == cfg.compress_ratios
    for q in ("kv_source_for", "index_source_for", "is_kv_source", "is_index_source",
              "index_owns_k", "uses_candidates"):
        a = [getattr(cfg, q)(i) for i in range(n)]
        b = [getattr(topo, q)(i) for i in range(n)]
        assert a == b, f"{q}: config {a} vs topology {b}"
    assert topo.kv_groups() == cfg.kv_groups()
    # the free cuts and every cut's crossings are the same through either
    free = P.free_cuts(cfg)
    assert free == P.free_cuts(topo), (free, P.free_cuts(topo))
    for c in range(1, n):
        assert P.crossings(cfg, c) == P.crossings(topo, c), c
    print(f"  OK  placement topology: config class and header-only Topology agree on all "
          f"{n} layers; a change of device is free only at {P.format_cuts(free)}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else model_dir()
    if not d:
        skip("V4.1 config", "no checkpoint (pass a directory or set DSV41_MODEL_DIR)")
        sys.exit(0)
    main(d)
