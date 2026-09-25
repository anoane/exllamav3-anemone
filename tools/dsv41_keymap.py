"""
Check the module graph V4.1 needs against a real checkpoint's tensor names.

CPU-only, no GPU, no model load, stdlib only. Both directions, every layer:

  missing    a stem the graph will ask for that the checkpoint lacks -- the
             multi-minute load would die on it
  unclaimed  a stem the checkpoint carries that no module claims -- weights
             silently left out of the forward

Stems are tensor names with the storage suffix (.trellis/.suh/.svh/.mul1/
.weight/.scale/...) stripped. Prefixes the text-only port deliberately does
not load (the MTP draft layers mtp.*, vision.*, aligner.*, image_*) are counted
and reported, not failed. Exits 1 on any missing or unclaimed stem.

    python tools/dsv41_keymap.py <checkpoint-dir>

See doc/dsv41_tools.md.
"""
import glob, importlib.util, json, os, re, sys

_spec = importlib.util.spec_from_file_location("dsv41_checkpoint_header", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "architecture", "dsv41",
    "checkpoint_header.py"))
_header = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_header)
read_header = _header.read_header

_SUFFIX = re.compile(r"\.(trellis|suh|svh|mul1|mcg|weight|bias|scale|weight_scale_inv)$")
NOT_LOADED = ("mtp.", "vision.", "aligner.", "image_start", "image_end", "image_newline")


def ckpt_stems(path):
    stems = set(); total = 0
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        hdr, _, _ = read_header(f)
        for k in hdr:
            if k == "__metadata__": continue
            total += 1
            stems.add(_SUFFIX.sub("", k))
    return stems, total


def expected(cfg):
    """Every stem the V4.1 text graph claims, from the config topology."""
    n = cfg["num_hidden_layers"]
    ratios = [int(r) for r in cfg["compress_ratios"][:n]]
    kv = set(cfg["kv_source_layer_ids"]); ix = set(cfg["index_source_layer_ids"])
    e = ["embed", "norm", "head"]
    for i in range(n):
        k = f"layers.{i}"
        e += [f"{k}.attn_norm", f"{k}.ffn_norm",
              f"{k}.attn.wq_a", f"{k}.attn.wq_b", f"{k}.attn.wkv",
              f"{k}.attn.wo_b", f"{k}.attn.q_norm", f"{k}.attn.kv_norm",
              f"{k}.attn.attn_sink"]
        e += [f"{k}.attn.wo_a.slice.{g}" for g in range(cfg["o_groups"])]
        e += [f"{k}.hc_{s}_{t}" for s in ("attn", "ffn") for t in ("fn", "base", "scale")]
        e += [f"{k}.ffn.gate"]
        e += [f"{k}.ffn.shared_experts.w{w}" for w in (1, 2, 3)]
        e += [f"{k}.ffn.experts.{x}.w{w}" for x in range(cfg["n_routed_experts"]) for w in (1, 2, 3)]
        if i in kv:
            e += [f"{k}.attn.compressor.wkv", f"{k}.attn.compressor.norm"]
            if ratios[i] == 2:
                e += [f"{k}.attn.compressor.wgate"]
        if i in ix:
            e += [f"{k}.attn.indexer.wq_b", f"{k}.attn.indexer.weights_proj"]
            if i in kv:
                e += [f"{k}.attn.indexer.wk", f"{k}.attn.indexer.k_norm"]
        if i in cfg["engram_layer_ids"]:
            # embed = the file-backed table (FP8 weight + E8M0 scale), gathered from the shards
            # by row
            e += [f"{k}.engram.k_weight", f"{k}.engram.q_weight", f"{k}.engram.wkv",
                  f"{k}.engram.embed"]
    return e


def check(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    t = c.get("text_config", c)
    cfg = dict(num_hidden_layers = t["num_hidden_layers"], o_groups = t.get("o_groups", 8),
               compress_ratios = t["compress_ratios"],
               kv_source_layer_ids = t.get("kv_source_layer_ids", []),
               index_source_layer_ids = t.get("index_source_layer_ids", []),
               n_routed_experts = t["n_routed_experts"],
               engram_layer_ids = t.get("engram_layer_ids", []))
    stems, total = ckpt_stems(path)
    exp = expected(cfg)
    claimed = set(exp)
    missing = [k for k in exp if k not in stems]
    skipped = sorted(s for s in stems if s.startswith(NOT_LOADED))
    unclaimed = sorted(s for s in stems if s not in claimed and not s.startswith(NOT_LOADED))
    return dict(total = total, stems = len(stems), expected = len(exp), missing = missing,
                unclaimed = unclaimed, skipped = skipped)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tools/dsv41_keymap.py <checkpoint-dir>")
        sys.exit(2)
    r = check(sys.argv[1])
    print(f"  checkpoint: {r['total']:,} tensors, {r['stems']:,} distinct stems")
    print(f"  graph claims {r['expected']:,} stems | MISSING {len(r['missing'])} | "
          f"UNCLAIMED {len(r['unclaimed'])}")
    for k in r["missing"][:12]: print("     -", k)
    for k in r["unclaimed"][:12]: print("     +", k)
    by = {}
    for s in r["skipped"]:
        p = s.split(".")[0]
        by[p] = by.get(p, 0) + 1
    print(f"  not loaded by the text-only port: " +
          (", ".join(f"{p} ({n} stems)" for p, n in sorted(by.items())) or "none"))
    sys.exit(1 if r["missing"] or r["unclaimed"] else 0)
