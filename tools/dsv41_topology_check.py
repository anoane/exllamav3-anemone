"""
Validate the V4.1 source topology against a checkpoint's tensor presence.

The config declares kv_source_layer_ids / index_source_layer_ids /
compress_ratios; the checkpoint proves them, because compressors, indexers
and engrams are only materialised on the layers that own them. CPU-only,
stdlib only. Exits 1 on any disagreement.

    python tools/dsv41_topology_check.py <checkpoint-dir>

See doc/dsv41_tools.md.
"""
import glob, importlib.util, json, os, re, sys

_spec = importlib.util.spec_from_file_location("dsv41_checkpoint_header", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3", "architecture", "dsv41",
    "checkpoint_header.py"))
_header = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_header)
read_header = _header.read_header

def layer_stems(path):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        hdr, _, _ = read_header(f)
        for k in hdr:
            m = re.match(r"layers\.(\d+)\.(.+)", k)
            if not m or ".experts." in m.group(2):
                continue
            rest = re.sub(r"\.(trellis|suh|svh|mul1|mcg|weight|bias|scale|weight_scale_inv)$",
                          "", m.group(2))
            out.setdefault(int(m.group(1)), set()).add(rest)
    return out

def main(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    t = c.get("text_config", c)
    n = t["num_hidden_layers"]
    ratios = [int(r) for r in t["compress_ratios"][:n]]
    kv = set(t["kv_source_layer_ids"]); ix = set(t["index_source_layer_ids"])
    eg = set(t["engram_layer_ids"])
    st = layer_stems(path)
    fails = []
    def check(cond, msg):
        if not cond: fails.append(msg)
    for i in range(n):
        s = st.get(i, set())
        has_comp = any(x.startswith("attn.compressor.") for x in s)
        has_wgate = "attn.compressor.wgate" in s
        has_wk = "attn.indexer.wk" in s
        has_idx = "attn.indexer.wq_b" in s
        has_eg = any(x.startswith("engram.") for x in s)
        # compressors live exactly on kv sources
        check(has_comp == (i in kv), f"layer {i}: compressor present={has_comp} but kv_source={i in kv}")
        # wgate only where the compression rate is 2 (ratio 1 is a plain full-length cache)
        if has_comp:
            check("attn.compressor.wkv" in s and "attn.compressor.norm" in s,
                  f"layer {i}: compressor requires both wkv and norm")
            check(has_wgate == (ratios[i] == 2),
                  f"layer {i}: wgate={has_wgate} but compress_ratio={ratios[i]}")
        # indexers live on index sources; only those that are ALSO kv sources own K
        check(has_idx == (i in ix), f"layer {i}: indexer present={has_idx} but index_source={i in ix}")
        check(("attn.indexer.weights_proj" in s) == (i in ix),
              f"layer {i}: indexer.weights_proj presence does not match index_source={i in ix}")
        check(("attn.indexer.k_norm" in s) == (i in ix and i in kv),
              f"layer {i}: indexer.k_norm presence does not match K ownership")
        if has_idx:
            check(has_wk == (i in kv), f"layer {i}: indexer.wk={has_wk} but owns_k={i in kv}")
        check(has_eg == (i in eg), f"layer {i}: engram present={has_eg} but engram_layer={i in eg}")
    print(f"  layers checked      : {n}")
    print(f"  kv sources          : {sorted(kv)}")
    print(f"  index sources       : {sorted(ix)}   (own K: {sorted(kv & ix)}, borrow: {sorted(ix - kv)})")
    print(f"  ratio-2 layers      : {[i for i in range(n) if ratios[i]==2]}")
    print(f"  ratio-1 layers      : {[i for i in range(n) if ratios[i]==1]}")
    print(f"  engram layers       : {sorted(eg)}")
    print(f"\n  FAILURES: {len(fails)}")
    for f in fails[:10]: print("    -", f)
    return 1 if fails else 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tools/dsv41_topology_check.py <checkpoint-dir>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
