"""
The "silu_ref" activation (DeepSeek's reference swiglu, exllamav3_ext/silu_ref.cuh) through every
Python dispatch that selects an activation kernel, checked on the production source without the
compiled extension: the statements are compiled from the engine files and run against stand-ins.
Also that the native sources agree on its id, 5, everywhere an id selects it, and that the fused
routed-expert kernel implements it in instances of its own (a template parameter, so the default
instances compile without it), which the launcher selects for that id. Native arithmetic is
covered by test_silu_ref_gpu_.py.

    python -m pytest tests/test_silu_ref_wiring_.py
"""

import ast
import importlib.util
from pathlib import Path
import re
import types
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stable_helpers", Path(__file__).with_name("test_stable_arithmetic_.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

BSM = "exllamav3/modules/block_sparse_mlp.py"
MIXIN = "exllamav3/modules/block_sparse_mlp_cpu.py"
MLP = "exllamav3/modules/mlp.py"
HOST = "exllamav3/model/moe_cpu_host.py"
RECON = "exllamav3/modules/moe_batch_recon.py"


def ext():
    return types.SimpleNamespace(**{n: Mock(name = n) for n in (
        "silu_mul", "silu_ref_mul", "gelu_mul", "silu_oai_mul", "relu2_mul", "relu_mul")})


def run(nodes, ns):
    module = ast.Module(body = ast.parse("from __future__ import annotations").body + list(nodes), type_ignores = [])
    exec(compile(module, "<wiring>", "exec"), ns)
    return ns


def test_routed_experts_select_the_kernel_and_the_fused_kernel_id():
    init = helpers.function(BSM, "BlockSparseMLP", "__init__")
    gated = next(n for n in init.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.gated")
    e = ext()
    for name, idx, call in (("silu", 0, e.silu_mul), ("silu_ref", 5, e.silu_ref_mul), ("gelu", 1, e.gelu_mul),
                            ("swiglu_oai", 3, e.silu_oai_mul), ("relu2", 2, e.relu2_mul)):
        obj = types.SimpleNamespace(gated = True)
        run([gated], dict(ext = e, self = obj, activation_fn = name))
        assert (obj.activation_fn_idx, obj.activation_fn_call) == (idx, call), name
    with pytest.raises(ValueError):
        run([gated], dict(ext = e, self = types.SimpleNamespace(gated = True), activation_fn = "silu_reference"))


def test_shared_experts_select_the_kernel():
    init = helpers.function(MLP, "GatedMLP", "__init__")
    match = next(n for n in init.body if isinstance(n, ast.Match))
    e = ext()
    for name, call in (("silu", e.silu_mul), ("silu_ref", e.silu_ref_mul), ("gelu", e.gelu_mul)):
        obj = types.SimpleNamespace()
        run([match], dict(ext = e, self = obj, activation_fn = name))
        assert obj.activation_fn_call is call, name


@pytest.mark.parametrize("path, cls", [(BSM, "BC_BlockSparseMLP"), (MLP, "BC_GatedMLP")])
def test_bound_classes_get_the_flag_only_for_silu_ref(path, cls):
    tree = ast.parse((ROOT / path).read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == cls)
    flag = next(k.value for k in call.keywords if k.arg == "act_silu_ref")
    value = lambda node, name: eval(compile(ast.Expression(node), "<flag>", "eval"),
                                    dict(self = types.SimpleNamespace(activation_fn = name)))
    for name in ("silu", "silu_ref", "gelu", "relu2"):
        assert value(flag, name) is (name == "silu_ref"), name
    # the other activation flags, which the native side requires to be off, are off for silu_ref
    others = [a for a in list(call.args) + [k.value for k in call.keywords if k.arg != "act_silu_ref"]
              if isinstance(a, ast.Compare) and ast.unparse(a.left) == "self.activation_fn"]
    assert len(others) == (4 if cls == "BC_BlockSparseMLP" else 3), [ast.unparse(o) for o in others]
    assert not any(value(o, "silu_ref") for o in others)


def test_fused_and_graph_paths_accept_silu_ref():
    load = helpers.function(BSM, "BlockSparseMLP", "load_local")
    linear = types.SimpleNamespace(inner = types.SimpleNamespace(bias = None), trim_padded_out = False,
                                   out_features = 128, out_features_unpadded = 128)
    for name, expected in (("silu_ref", True), ("silu", True), ("swiglu_oai", False)):
        obj = types.SimpleNamespace(is_quantized = True, activation_fn = name, gated = True,
                                    gates = [linear], ups = [linear], downs = [linear],
                                    config = types.SimpleNamespace(infer_params = types.SimpleNamespace(no_reconstruct = False)))
        node = next(n for n in load.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "self.support_quant_paths")
        run([node], dict(self = obj))
        assert obj.support_quant_paths is expected, name
        node = next(n for n in load.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "self.support_bc_bszn")
        run([node], dict(self = obj, _uniform_bias = lambda ls: True))
        assert obj.support_bc_bszn is True, name


def test_shared_expert_fused_decode_accepts_silu_ref():
    class GatedMLP:
        pass
    linear = types.SimpleNamespace(quant_type = "exl3", in_features = 128, out_features = 128)
    fn = run([helpers.function(BSM, "BlockSparseMLP", "shared_coop_ok")],
             dict(_moe_shared_coop = True, GatedMLP = GatedMLP))["shared_coop_ok"]
    for name, expected in (("silu_ref", True), ("silu", True), ("swiglu_oai", False)):
        se = GatedMLP()
        se.gates, se.ups, se.downs, se.activation_fn = [linear], [linear], [linear], name
        obj = types.SimpleNamespace(shared_experts = se, shared_experts_post_norm = None, latent_in = None,
                                    alt_residual_channel = False, hidden_size = 128, config = None)
        assert fn(obj) is expected, name


def test_batched_reconstruct_tier_maps_the_worker_id():
    tree = ast.parse((ROOT / RECON).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id.startswith("_ACT_") for t in n.targets)]
    e = ext()
    ns = run(nodes, dict(ext = e))
    assert ns["_ACT_IDX"] == {0: "silu", 1: "gelu", 2: "relu2", 3: "swiglu_oai", 5: "silu_ref"}
    assert ns["_ACT_CALLS_GATED"]["silu_ref"] is e.silu_ref_mul


def test_ram_held_experts_accept_silu_ref():
    # the one eligibility test of the CPU-offload claims (without an entry for this activation a
    # placed layer could not hold its experts in RAM) and the worker's activation ids
    fn = run([helpers.function(MIXIN, "BlockSparseMLP_CPU", "_cpu_eligible")],
             dict(torch = types.SimpleNamespace(device = lambda d: types.SimpleNamespace(type = "cuda"))))["_cpu_eligible"]
    for name in ("silu", "silu_ref", "gelu", "swiglu_oai"):
        obj = types.SimpleNamespace(num_local_experts = None, num_experts = 8, gated = True, activation_fn = name)
        assert fn(obj, "cuda:0") is True, name
    maps = [n for n in ast.walk(ast.parse((ROOT / MIXIN).read_text())) if isinstance(n, ast.Dict)
            and any(isinstance(k, ast.Constant) and k.value == "silu_ref" for k in n.keys)]
    assert len(maps) == 2
    for node in maps:
        assert ast.literal_eval(node) == {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3, "silu_ref": 5}


def test_streamed_experts_use_the_native_kernel_and_the_fused_tier():
    e = ext()
    act = run([helpers.function(HOST, "MoeCpuHost", "_act")], dict(ext = e, torch = torch))["_act"]
    g, u = torch.ones(2, 4, dtype = torch.half), torch.ones(2, 4, dtype = torch.half)
    a = act(None, {"activation": 5, "act_limit": 10}, g, u)
    assert a.dtype == torch.half and a.shape == u.shape
    e.silu_ref_mul.assert_called_once()
    args = e.silu_ref_mul.call_args.args
    assert args[0] is g and args[1] is u and args[2] is a and args[3] == 10.0
    fused_t = run([helpers.function(HOST, "MoeCpuHost", "_stream_fused_t")],
                  dict(TUNING = types.SimpleNamespace(stream_fused_t = 256, fused_prefill = False),
                       FUSED_COUNT_LIMIT = 1 << 20))["_stream_fused_t"]
    assert fused_t(None, {"activation": 5, "hi": 128, "ho": 128}, {}, 128) == 256
    assert fused_t(None, {"activation": 5, "hi": 128, "ho": 128}, {"bias_g": object()}, 128) == 0
    assert fused_t(None, {"activation": 3, "hi": 128, "ho": 128}, {}, 128) == 0


def test_native_sources_agree_on_the_id():
    def read(p):
        return (ROOT / "exllamav3/exllamav3_ext" / p).read_text()
    assert re.search(r"#define ACT_SILU_REF 5\b", read("activation.cu"))
    assert re.search(r"#define MOE_ACT_SILU_REF 5\b", read("quant/exl3_moe_common.cuh"))
    assert re.search(r"#define MOE_COOP_ACT_SILU_REF 5\b", read("quant/exl3_moe_coop.cuh"))
    cpu = read("cpu/moe_mul1.cpp")
    assert "case 5:" in cpu and "activation == 5" in cpu
    # every activation id in use is distinct
    ids = [int(v) for v in re.findall(r"#define ACT_[A-Z0-9_]+ (\d+)", read("activation.cu"))]
    assert len(ids) == len(set(ids)) and 5 in ids
    assert 'm.def("silu_ref_mul"' in read("bindings.cpp")



def test_fused_kernel_instances_of_their_own():
    quant = ROOT / "exllamav3/exllamav3_ext/quant"
    # The Hadamard/activation step takes silu_ref as a template parameter: its own instances apply
    # DeepSeek's reference swiglu in place of the switch, the limits and the gate product
    guad = (quant / "hadamard_inner.cuh").read_text().split("void had_hf_r_128_guad_inner", 1)
    assert guad[0].rstrip().endswith("template <bool silu_ref = false>\ninline __device__")
    body = guad[1].split("\n}\n", 1)[0]
    assert "if constexpr (silu_ref)" in body and "else switch (act_function)" in body
    assert "if (!silu_ref && act_limit != 0.0f)" in body and "if constexpr (!silu_ref)" in body
    assert "ACT_SILU_REF" not in body
    kernel = (quant / "exl3_moe_kernel.cuh").read_text()
    assert "had_hf_r_128_guad_inner<silu_ref>" in kernel
    assert "bool silu_ref = false>" in kernel.split("void exl3_moe_kernel", 1)[0]
    # A row-striped and a whole-K reference twin of every default instance, and launcher tables for
    # both with the default tables' layout
    getter = re.compile(r"fp_exl3_moe_kernel (exl3_moe_kernel_\w+)\(\) \{ return exl3_moe_kernel<([^>]*)>; \}")
    defined = {}
    for path in (quant / "comp_units").glob("exl3_moe_inst_*.cu"):
        for name, args in getter.findall(path.read_text()):
            defined[name] = ([a.strip() for a in args.split(",")], path.name)
    plain = {n: a for n, (a, _) in defined.items() if re.fullmatch(r"exl3_moe_kernel_k\d_n(128|256)_cb[12](_m32|_m64)?", n)}
    assert len(plain) == 54
    for name, args in plain.items():
        full = args + ["MOE_TILESIZE_M"] if len(args) == 3 else args
        assert defined[name + "_sr"][0] == full + ["true", "false", "true"], name
        assert defined[name + "_sr_wk"][0] == full + ["true", "true", "true"], name
        assert defined[name + "_sr_wk"][1].endswith("_sr_wk.cu"), name
    launcher = (quant / "exl3_moe.cu").read_text()
    def entries(table):
        return re.findall(r"(exl3_moe_kernel_\w+)\(\)", launcher.split(f"fp_exl3_moe_kernel {table}[] =", 1)[1].split("};", 1)[0])
    for table in ("exl3_moe_kernel_instances", "exl3_moe_kernel_instances_m32", "exl3_moe_kernel_instances_m64"):
        for suffix in ("_sr", "_sr_wk"):
            assert [e + suffix for e in entries(table)] == entries(table + suffix), (table, suffix)
    assert "const bool silu_ref = act_function == MOE_ACT_SILU_REF;" in launcher
    assert "silu_ref ? (whole_k ? moe_family_silu_ref_whole_k : moe_family_silu_ref) :" in launcher



def test_fused_decode_kernel_instances_of_their_own():
    # The fused decode kernel (exl3_moe_coop) takes it as a template parameter of its gate/up
    # kernel, whose instances of their own the launcher picks for the id; act_gate, which every
    # other instance runs, has no case for it
    quant = ROOT / "exllamav3/exllamav3_ext/quant"
    kernel = (quant / "exl3_moe_coop_kernel.cuh").read_text()
    act_gate = kernel.split("float act_gate(int act, bool gated", 1)[1].split("\n}\n", 1)[0]
    assert "SILU_REF" not in act_gate
    assert "template <int bits, int cb, bool WIDE, bool HALF = false, bool SILU_REF = false>\n" \
           "__global__ __launch_bounds__(THREADS)\nvoid exl3_moe_coop_a_kernel" in kernel
    assert kernel.count("act_gate_t<SILU_REF>(p.act") == 4
    units = {p.name: p.read_text() for p in (quant / "comp_units").glob("exl3_moe_coop_inst_*.cu")}
    for k in range(1, 9):
        assert units[f"exl3_moe_coop_inst_k{k}_sr.cu"].strip().endswith(f"EXL3_MOE_COOP_INST_DEF_SR({k})")
    for k in range(1, 4):
        assert units[f"exl3_moe_coop_inst_h{k}_sr.cu"].strip().endswith(f"EXL3_MOE_COOP_INST_DEF_H_SR({k})")
    header = (quant / "comp_units/exl3_moe_coop_instances.cuh").read_text()
    for macro in ("EXL3_MOE_COOP_INST_DEF_SR(K)", "EXL3_MOE_COOP_INST_DEF_H_SR(K)"):
        body = header.split("#define " + macro, 1)[1].split("\n\n", 1)[0]
        instances = re.findall(r"exl3_moe_coop_a_kernel<([^>]*)>", body)
        assert instances and all(i.replace(" ", "").endswith(",true") for i in instances), macro
    launcher = (quant / "exl3_moe_coop.cu").read_text()
    assert "moe_coop_kernel_a(K_gu, cb, p.Hi, wide_a, p.act == MOE_COOP_ACT_SILU_REF)" in launcher

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
