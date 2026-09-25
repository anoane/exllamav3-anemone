"""
CPU-only negative controls for the V4.1 validation harness (tools/dsv41_refcompare.py,
tools/dsv41_validate.py): its gates (the calibrated limits, the noise-shifted limits and the
isolation metadata), its plan, its report and the engine settings it applies.
No model, capture, GPU or compiled extension required.
"""
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = load("safety_refcompare", "tools/dsv41_refcompare.py")
v = load("safety_validate", "tools/dsv41_validate.py")


def case(n=20000):
    return rc.Case(n=n, rep=0, prompt_ids=np.ones(n, dtype=np.int64),
        ref_ids=np.tile(np.arange(1, 6), (n-1, 1)),
        ref_lp=np.tile(np.array([-0.5, -2., -3., -4., -5.]), (n-1, 1)),
        ref_actual_lp=np.full(n-1, -0.5))


@pytest.mark.parametrize("mutation", ["single", "balanced", "rankings", "missing_reference", "rank_nan"])
def test_localized_corruption_never_sets_own_tolerance(mutation):
    c = case()
    ids, lp, act = rc.as_ours(c)
    assert rc.gates(rc.compare(c, ids, lp, act))[0]
    if mutation == "single":
        act[0] -= 10
    elif mutation == "balanced":
        act[:126] += np.tile([-10., 10.], 63)
    elif mutation == "rankings":
        ids[:127] += 50
    elif mutation == "missing_reference":
        c.ref_actual_lp[10:100] = np.nan
    else:
        lp[0, 0] = np.nan
    assert not rc.gates(rc.compare(c, ids, lp, act))[0]


def test_partial_duplicate_and_contended_reference_fail():
    c = case(300)
    data = rc.as_ours(c)
    assert not rc.gates(rc.compare(c, *(x[:20] for x in data), positions=np.arange(1, 21)))[0]
    with pytest.raises(ValueError, match="duplicates"):
        rc.compare(c, *(x[:20] for x in data), positions=np.ones(20, dtype=int))
    for key in ("contended", "probe_errors"):
        c.meta = {key: 1}
        assert not rc.gates(rc.compare(c, *data))[0]


def test_zero_noise_floor_does_not_allow_three_flips_or_ranking_corruption():
    c = case(300)
    original = rc.as_ours(c)
    floor = rc.compare_self(original, original, n=c.n)
    changed = tuple(x.copy() for x in original)
    changed[0][:3, 0] = 99
    assert not rc.self_gates(rc.compare_self(original, changed, n=c.n), floor=floor)[0]
    changed = tuple(x.copy() for x in original)
    changed[0][:, 1:] += 50
    assert not rc.self_gates(rc.compare_self(original, changed, n=c.n), floor=floor)[0]


def test_self_comparison_rejects_duplicate_or_incomplete_rankings():
    c = case(300)
    original = rc.as_ours(c)
    changed = tuple(x.copy() for x in original)
    changed[0][:, 1] = changed[0][:, 0]
    assert not rc.self_gates(rc.compare_self(changed, changed, n=c.n))[0]
    for ids, lp, actual in ((original[0][:, :4], original[1][:, :4], original[2]),
                            (original[0].astype(float), original[1], original[2]),
                            (original[0], original[1][:-1], original[2])):
        with pytest.raises(ValueError, match="top-5"):
            rc.compare_self((ids, lp, actual), original, n=c.n)


@pytest.mark.parametrize("chunk", [0, -1, -256])
def test_schedules_fail_before_nonprogressing_loop(chunk):
    with pytest.raises(ValueError):
        rc.decode_schedule(300, chunk)
    with pytest.raises(ValueError):
        rc.uniform_schedule(300, chunk)


def test_requested_generator_and_prefill_errors_fail_overall():
    rec = {"runs": {"cached": {"ok": True}}, "self": []}
    assert v.case_ok(rec)
    assert not v.case_ok({**rec, "generator_ok": False})
    assert not v.case_ok({**rec, "generator_error": "failed"})
    assert not v.case_ok({**rec, "prefill_check": {"ok": False}})
    assert not v.case_ok({"runs": {}, "self": []})


def test_sweep_requires_successful_complete_baseline_and_exercised_ablation():
    c = case(300)
    base = rc.compare(c, *rc.as_ours(c))
    data = rc.as_ours(c)
    data[2][:] -= 1
    ablated = rc.compare(c, *data)
    def wrap(report, ok):
        report = dict(report, nonfinite=0, unfilled=0)
        return {"ok": ok, "cases": [{"case": c.name, "runs": {"cached": {"ok": ok, "report": report}}}]}
    good, bad = wrap(base, True), wrap(ablated, False)
    assert v.sweep_summary(rc, ["engram"], {None: good, "engram": bad})[2]
    assert not v.sweep_summary(rc, ["no_candidates"], {None: good, "no_candidates": good})[2]
    assert not v.sweep_summary(rc, ["engram"], {None: bad, "engram": bad})[2]
    partial = copy.deepcopy(bad)
    partial["cases"] = []
    assert not v.sweep_summary(rc, ["engram"], {None: good, "engram": partial})[2]


@pytest.mark.parametrize("mutation", ["nonfinite", "unfilled", "invalid_ranked_rows", "coverage",
    "positions", "reference_contended", "reference_probe_errors", "compared", "missing_actual",
    "region", "missing_flag", "case", "rep"])
def test_ablation_detection_requires_intact_measurements(mutation):
    c = case(300)
    base = dict(rc.compare(c, *rc.as_ours(c)), nonfinite=0, unfilled=0)
    data = rc.as_ours(c)
    data[2][:] -= 1
    changed = dict(rc.compare(c, *data), nonfinite=0, unfilled=0)
    assert rc.ablation_check(changed, "engram", base)["ok"]
    if mutation == "coverage":
        changed[mutation] = 0.5
    elif mutation == "positions":
        changed[mutation] -= 1
    elif mutation in ("compared", "missing_actual"):
        changed["global"][mutation] += -1 if mutation == "compared" else 1
    elif mutation == "region":
        changed["regions"] = changed["regions"][1:]
    elif mutation == "missing_flag":
        del changed["nonfinite"]
    elif mutation == "case":
        changed["case"] = "different_case"
    elif mutation == "rep":
        changed["rep"] += 1
    else:
        changed[mutation] = 1
    def wrap(report, ok):
        return {"ok": ok, "cases": [{"case": c.name, "runs": {"cached": {"ok": ok, "report": report}}}]}
    summary, _, ok = v.sweep_summary(rc, ["engram"],
        {None: wrap(base, True), "engram": wrap(changed, False)})
    assert not ok and summary["tokens"]["engram"][0]["status"] == "invalid"
    # A stale outer success flag must not rescue an invalid baseline either.
    assert not v.sweep_summary(rc, ["engram"],
        {None: wrap(changed, True), "engram": wrap(changed, False)})[2]


@pytest.mark.parametrize("argv,n,expected", [
    (["--mode", "cached", "--chunk", "2048", "--logit-rows", "32"], 5000, 2048),
    (["--mode", "cached", "--chunk", "2048"], 300, 300),
    (["--mode", "nc", "--greedy", "0"], 300, 300),
    (["--mode", "nc", "--greedy", "16"], 300, 315),
    (["--mode", "self", "--chunks", "64,128", "--nc-max", "128"], 300, 128),
])
def test_output_budget_covers_actual_forwards_not_reduction_rows(argv, n, expected):
    args = v.build_parser(rc).parse_args(argv)
    plan = v.make_plan(args, rc, [case(n)])
    assert not plan["errors"]
    assert plan["max_output_size"] == expected
    assert plan["max_chunk_size"] >= expected


def test_prefill_tail_is_included_in_output_budget(monkeypatch):
    monkeypatch.setenv("EXL3_DSV41_PIPELINE", "1")
    monkeypatch.setenv("EXL3_DSV41_PIPELINE_CHUNK", "1024")
    args = v.build_parser(rc).parse_args(["--mode", "cached", "--chunk", "64", "--prefill-check",
                                        "--prefill-chunk", "2048", "--prefill-tail", "512"])
    plan = v.make_plan(args, rc, [case(6000)])
    assert plan["max_output_size"] == 512
    assert plan["max_chunk_size"] >= 1024


def test_explicit_load_chunk_cannot_hide_larger_logit_forward():
    args = v.build_parser(rc).parse_args(["--mode", "cached", "--chunk", "2048", "--max-chunk-size", "32"])
    plan = v.make_plan(args, rc, [case(5000)])
    assert any("logit-producing forward (2048 rows)" in error for error in plan["errors"])


def test_loader_receives_planned_output_budget(tmp_path, monkeypatch):
    import types
    args = v.build_parser(rc).parse_args(["--mode", "cached", "--chunk", "2048"])
    plan = v.make_plan(args, rc, [case(5000)])
    observed = {}
    def load_model(**kwargs):
        observed.update(kwargs)
        raise RuntimeError("load inspected without CUDA")
    fake = types.SimpleNamespace(
        Config=types.SimpleNamespace(from_directory=lambda path: types.SimpleNamespace()),
        Model=types.SimpleNamespace(from_config=lambda config: types.SimpleNamespace(load=load_model)),
        Cache=lambda *args, **kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "exllamav3", fake)
    monkeypatch.setattr(v, "start_ram_watchdog", lambda *args: None)
    with pytest.raises(RuntimeError, match="load inspected"):
        v.run_gpu(args, rc, [], [], plan, str(tmp_path / "report.json"))
    assert observed["max_output_size"] == 2048
    assert observed["max_chunk_size"] >= observed["max_output_size"]


def test_environment_report_is_an_allowlist():
    assert "EXL3_SECRET_TOKEN" not in v.REPORT_ENV
    assert "EXL3_NO_FUSED_RECONSTRUCT" in v.REPORT_ENV
    assert "EXL3_DSV41_PIPELINE" in v.REPORT_ENV
    assert "EXL3_MOE_PINNED_ARENA" in v.REPORT_ENV
    assert "EXL3_MOE_CPU_WSLOTS" in v.REPORT_ENV


def test_environment_report_names_the_engine_settings():
    # the series' settings (every opt-in included) and the main dispatch and precision switches of
    # V4.1's default path (int8 GEMV activations are on by default for its mul1 decode GEMVs)
    for name in ("EXL3_STABLE_ARITHMETIC", "EXL3_HGEMM_FIXED_ROWS", "EXL3_MOE_FUSED_PREFILL",
                 "EXL3_INT8_GEMV", "EXL3_INT8_GEMV_MAX_K", "EXL3_GEMV", "EXL3_MGEMM_K_THRESHOLD",
                 "EXL3_MGEMM_N_THRESHOLD", "EXL3_MOE_FUSED_ROWS", "EXL3_MOE_FUSED_ROWS_WIDE",
                 "EXL3_MOE_BATCH_RECON",
                 "EXL3_MOE_FUSED_DET", "EXL3_MOE_RECON_DET", "EXL3_MOE_RECON_FOLDED", "EXL3_HGEMM_F16ACC",
                 "EXL3_PLACEMENT", "EXL3_MOE_CPU_MODE", "EXL3_MOE_CPU_OFFLOAD", "EXL3_MOE_CPU_SPLIT",
                 "EXL3_HGEMM_FP32_REDUCTION", "EXL3_DSA_DEBUG_BOUNDS",
                 "EXL3_FP32_LOGITS", "EXL3_DSV41_NUMERICS", "EXL3_DSV41_ENGRAM_PREFETCH",
                 "EXL3_DSV41_PIPELINE_CHUNK", "EXL3_DSV41_XDEV_BF16", "EXL3_DSV41_ROUTER_BIAS",
                 "EXL3_DSV41_FUSED_COMPRESS", "EXL3_DSV41_ACTIVATION", "EXL3_DSV41_ABLATE"):
        assert name in v.REPORT_ENV, name
    # names the engine no longer reads
    for name in ("EXL3_MOE_EXPERT_POLICY", "EXL3_DSV41_SKIP_ENGRAM", "EXL3_DSV41_PLACEMENT"):
        assert name not in v.REPORT_ENV, name
    assert len(set(v.REPORT_ENV)) == len(v.REPORT_ENV)
    # every recorded EXL3_ name is read somewhere in the engine (a typo would record nothing): it
    # must appear as a whole quoted literal, so neither a longer name that contains it
    # (EXL3_MOE_CPU_SPLIT_LAYERS for EXL3_MOE_CPU_SPLIT) nor a mention in prose counts. An older
    # checkout (without the arithmetic policy module) lacks some of them, which is harmless
    if (ROOT / "exllamav3/model/math_policy.py").is_file():
        sources = "".join(p.read_text(errors = "ignore") for p in (ROOT / "exllamav3").rglob("*")
                          if p.suffix in (".py", ".cpp", ".cu", ".cuh", ".h"))
        unread = [name for name in v.REPORT_ENV if name.startswith("EXL3_")
                  and not re.search(r"[\"']" + re.escape(name) + r"[\"']", sources)]
        assert not unread, unread
        # the check itself: a prefix of a quoted name is not read
        assert not re.search(r"[\"']" + re.escape("EXL3_MOE_CPU_SPLIT_LAYER") + r"[\"']", sources)


def test_ablations_match_the_engine_registry():
    registry = load("safety_ablation", "exllamav3/modules/dsv41_ablation.py")
    tokens = set()
    for a in rc.ABLATIONS.values():
        assert set(a.env) == {registry.ENV_ABLATE}, a
        tokens |= set(a.env[registry.ENV_ABLATE].split(","))
    assert tokens == set(registry.ABLATE_TOKENS) == set(rc.ABLATIONS)


def test_engine_settings_of_the_command_line():
    parser = v.build_parser(rc)
    args = parser.parse_args([])
    assert args.moe_cpu_mode is None and args.dsv41_numerics is None and not args.fp32_logits
    for value in ("compute", "stream_only"):
        assert parser.parse_args(["--moe-cpu-mode", value]).moe_cpu_mode == value
    for value in ("offload_only", "storage_only", "typo"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--moe-cpu-mode", value])
    assert "no CPU expert arithmetic" in " ".join(parser.format_help().split())
    for removed in ("--stable-arithmetic", "--moe-expert-policy"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed])
    # --placement goes to EXL3_PLACEMENT, before the import (an empty one clears the caller's);
    # nothing else is set through the environment
    env = v.run_env(parser.parse_args(["--placement", "0-11=cuda:0; 12-39=cuda:1"]), rc)
    assert env == {"EXL3_PLACEMENT": "0-11=cuda:0; 12-39=cuda:1"}
    env = v.run_env(parser.parse_args(["--placement", "", "--dsv41-numerics", "precise",
                                       "--fp32-logits"]), rc)
    assert env == {"EXL3_PLACEMENT": ""}
    env = v.run_env(parser.parse_args(["--ablate", "engram", "--ablate", "v4mix"]), rc)
    assert env == {"EXL3_DSV41_ABLATE": "engram,v4mix"}


def test_engine_settings_are_checked_before_loading(tmp_path):
    parser = v.build_parser(rc)
    def errors(*argv):
        return v.make_plan(parser.parse_args(list(argv)), rc, [case(300)])["errors"]
    assert not errors("--dsv41-numerics", "deepseek:index")
    assert not errors("--dsv41-numerics", "vllm:window,compressed")
    assert any("--dsv41-numerics" in e for e in errors("--dsv41-numerics", "fp8"))
    assert any("--dsv41-numerics" in e for e in errors("--dsv41-numerics", "precise:index"))
    assert any("mutually exclusive" in e for e in errors("--mcl", "2", "--mcs", "8"))
    if (ROOT / "exllamav3/model/placement.py").is_file():
        assert not errors("--placement", "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1")
        assert any("--placement" in e for e in errors("--placement", "0-11=cuda:0 experts=hybrid"))
    else:
        assert any("explicit placement" in e for e in errors("--placement", "0-11=cuda:0; *=cuda:1"))
    # DeepSeek-V4.1's rule for a placement's changes of device is checked against the model's
    # topology when the model is given (kv group 1-3 here: a change of device at 1 is free, one
    # at 2 or 3 is the split, both at once are refused)
    if (ROOT / "exllamav3/model/placement.py").is_file():
        (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 4, "compress_ratios": [0, 2, 2, 2],
                                                          "kv_source_layer_ids": [1], "index_source_layer_ids": [1]}))
        model = ("--model", str(tmp_path))
        assert not errors("--placement", "0=cuda:0; 1-3=cuda:1", *model)
        assert not errors("--placement", "0-1=cuda:0; 2-3=cuda:1", *model)
        assert any("changes device in 2 places" in e for e in errors("--placement", "0-1=cuda:0; 2=cuda:1; 3=cuda:0",
                                                                     *model))
        assert any("does not have: 4" in e for e in errors("--placement", "0-4=cuda:0", *model))
        # split=<k>, the removed V4.1 split's grammar, is refused with the placement that means it
        found = errors("--placement", "split=2", *model)
        assert any("split=<k> is not a placement rule" in e and "'0-1=cuda:0; 2-3=cuda:1'" in e for e in found), found


def test_settings_are_applied_to_the_config_as_model_init_applies_them():
    import types
    parser = v.build_parser(rc)
    class Config:
        def __init__(self):
            self.infer_params = types.SimpleNamespace(moe_cpu_offload = 0, moe_cpu_split = 0, moe_cpu_threads = None,
                                                      moe_cpu_mode = "compute", fp32_logits = False)
            self.numerics = "deepseek:index"
        @property
        def dsv41_numerics(self):
            return self.numerics
        @dsv41_numerics.setter
        def dsv41_numerics(self, value):
            if value not in ("precise", "deepseek:index"):
                raise ValueError(value)
            self.numerics = value
    cfg = Config()
    v.apply_settings(cfg, parser.parse_args([]))
    assert cfg.dsv41_numerics == "deepseek:index" and cfg.infer_params.fp32_logits is False
    v.apply_settings(cfg, parser.parse_args(["--mcs", "96", "--mct", "8", "--moe-cpu-mode", "stream_only",
                                             "--fp32-logits", "--dsv41-numerics", "precise"]))
    ip = cfg.infer_params
    assert (ip.moe_cpu_split, ip.moe_cpu_threads, ip.moe_cpu_mode, ip.fp32_logits) == (96, 8, "stream_only", True)
    assert cfg.dsv41_numerics == "precise"
    # a tree without the setting refuses it before the model is built
    del cfg.infer_params.fp32_logits
    with pytest.raises(ValueError, match = "fp32_logits"):
        v.apply_settings(cfg, parser.parse_args(["--fp32-logits"]))
    with pytest.raises(ValueError, match = "dsv41_numerics"):
        v.apply_settings(types.SimpleNamespace(infer_params = ip), parser.parse_args(["--dsv41-numerics", "precise"]))


def test_report_writer_does_not_follow_predictable_temporary_symlink(tmp_path):
    # A regression guard: _write uses an unpredictable mkstemp name, which cannot be planted in
    # advance. A writer that went back to the predictable '<report>.tmp' opened for writing would
    # follow this link, overwrite the victim and fail here.
    victim = tmp_path / "victim"
    victim.write_text("preserve")
    output = tmp_path / "report.json"
    (tmp_path / "report.json.tmp").symlink_to(victim)
    v._write(str(output), {"ok": True})
    assert victim.read_text() == "preserve"
    assert json.loads(output.read_text()) == {"ok": True}
    assert not list(tmp_path.glob(".dsv41-report-*"))


def test_new_run_invalidates_stale_success_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text(json.dumps({"ok": True, "session": "old"}))
    args = v.build_parser(rc).parse_args([])
    def failed_env(*args):
        raise RuntimeError("simulated pre-load failure")
    monkeypatch.setattr(v, "run_env", failed_env)
    with pytest.raises(RuntimeError, match="pre-load"):
        v.run_gpu(args, rc, [], [], {}, str(output))
    saved = json.loads(output.read_text())
    assert not saved["ok"] and saved["status"] == "running" and saved["session"] != "old"


@pytest.mark.parametrize("flag", ["--tie-eps", "--min-avail-gib"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_nonfinite_plan_parameters_fail(flag, value):
    args = v.build_parser(rc).parse_args([flag, value])
    with pytest.raises(ValueError):
        v.make_plan(args, rc, [case(300)])


def test_prefill_geometry_validated_before_model_load(monkeypatch):
    monkeypatch.setenv("EXL3_DSV41_PIPELINE", "1")
    monkeypatch.setenv("EXL3_DSV41_PIPELINE_CHUNK", "1024")
    args = v.build_parser(rc).parse_args(["--mode", "cached", "--prefill-check",
                                        "--prefill-chunk", "2048", "--prefill-tail", "32"])
    if not (ROOT / "exllamav3/architecture/dsv41/pipeline.py").is_file():
        assert any("optional" in e for e in v.make_plan(args, rc, [case(2080)])["errors"])
        return
    assert not v.make_plan(args, rc, [case(2080)])["errors"]
    args.prefill_tail = 1
    assert v.make_plan(args, rc, [case(2080)])["errors"]
    args.prefill_tail = 32
    args.max_chunk_size = 512
    assert v.make_plan(args, rc, [case(2080)])["errors"]


def _changed_report(n=300):
    c = case(n)
    base = dict(rc.compare(c, *rc.as_ours(c)), nonfinite=0, unfilled=0)
    data = rc.as_ours(c)
    data[2][:] -= 1
    return c, base, dict(rc.compare(c, *data), nonfinite=0, unfilled=0)


def test_engram_nocarry_is_not_applicable_without_a_chunk_start():
    c, base, changed = _changed_report()
    # without the run's chunk starts the check cannot know, and scores the run as any other
    assert rc.ablation_check(base, "engram_nocarry", base)["status"] == "blind"
    # a single forward (nc, or a cached run with n <= chunk) and a start at the last input
    # position (its prediction is past the prompt) leave nothing for the ablation to act on
    for starts in ([], rc.chunk_starts(rc.uniform_schedule(c.n, 2048)), [c.n - 1]):
        chk = rc.ablation_check(base, "engram_nocarry", base, chunk_starts=starts)
        assert chk["status"] == "n/a" and chk["ok"] is None, (starts, chk)
    starts = rc.chunk_starts(rc.uniform_schedule(c.n, 128))
    assert rc.ablation_check(changed, "engram_nocarry", base, chunk_starts=starts)["status"] == "detected"
    assert rc.ablation_check(base, "engram_nocarry", base, chunk_starts=starts)["status"] == "blind"
    # the other ablations do not depend on chunk starts
    assert rc.ablation_check(changed, "engram", base, chunk_starts=[])["status"] == "detected"


def test_sweep_takes_chunk_starts_from_the_recorded_plan():
    c, base, changed = _changed_report()
    def wrap(report, ok, runs):
        plan = {"cases": [{"case": c.name, "runs": [
            {"name": name, **({"schedule": schedule} if schedule else {})} for name, schedule in runs]}]}
        return {"ok": ok, "plan": plan, "cases": [{"case": c.name, "runs": {
            name: {"ok": ok, "report": report} for name, _ in runs}}]}
    single = [("cached2048", [c.n])]
    summary, lines, ok = v.sweep_summary(rc, ["engram_nocarry"],
        {None: wrap(base, True, single), "engram_nocarry": wrap(base, False, single)})
    assert not ok and summary["tokens"]["engram_nocarry"][0]["status"] == "n/a"
    assert any("never exercised: no run has a chunk start" in line for line in lines), lines
    chunked = [("nc", None), ("cached128", rc.uniform_schedule(c.n, 128))]
    summary, lines, ok = v.sweep_summary(rc, ["engram_nocarry"],
        {None: wrap(base, True, chunked), "engram_nocarry": wrap(changed, False, chunked)})
    statuses = {e["run"]: e["status"] for e in summary["tokens"]["engram_nocarry"]}
    assert ok and statuses == {"nc": "n/a", "cached128": "detected"}, (statuses, lines)


def test_sweep_tokens_are_those_the_run_can_exercise(capsys):
    parser = v.build_parser(rc)
    tokens, errors = v.sweep_tokens(parser.parse_args(["--mode", "cached"]), rc)
    assert tokens == list(rc.ABLATIONS) and not errors
    for argv in (["--mode", "nc"], ["--mode", "cached", "--engram-lookback", "harness"]):
        tokens, errors = v.sweep_tokens(parser.parse_args(argv), rc)
        assert not errors and "engram_nocarry" not in tokens and "engram" in tokens, (argv, tokens)
        assert "not sweeping engram_nocarry" in capsys.readouterr().out
        # named explicitly, it is refused before any child starts
        tokens, errors = v.sweep_tokens(parser.parse_args(argv + ["--sweep-tokens", "engram,engram_nocarry"]), rc)
        assert any("engram_nocarry" in e for e in errors), (argv, errors)
        assert v.run_sweep(parser.parse_args(argv + ["--sweep-tokens", "engram_nocarry"]), rc, [], "unused") == 2
    assert v.sweep_tokens(parser.parse_args(["--sweep-tokens", "bogus"]), rc)[1]
    assert v.sweep_tokens(parser.parse_args(["--sweep-tokens", ","]), rc)[1]
    # the same rule for a single run
    plan = v.make_plan(parser.parse_args(["--mode", "cached", "--engram-lookback", "harness",
                                          "--ablate", "engram_nocarry"]), rc, [case(300)])
    assert any("--engram-lookback state" in e for e in plan["errors"]), plan["errors"]
    assert not v.make_plan(parser.parse_args(["--mode", "cached", "--ablate", "engram_nocarry"]), rc,
                           [case(300)])["errors"]


def test_abbreviated_options_are_refused():
    # --ablate-sw would reach every sweep child (the parent strips only exact spellings), and each
    # child would start a sweep of its own
    parser = v.build_parser(rc)
    for argv in (["--ablate-sw"], ["--ablate-s"], ["--sweep-tok", "engram"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    child = v._strip_args(["--mode", "cached", "--ablate-sweep", "--sweep-tokens", "engram", "--out", "x"],
                          {"--ablate-sweep", "--dry-run"}, {"--sweep-tokens", "--out", "--ablate"})
    assert not parser.parse_args(child).ablate_sweep


def test_pipeline_switch_is_read_as_the_engine_reads_it(monkeypatch):
    if not (ROOT / "exllamav3/architecture/dsv41/pipeline.py").is_file():
        pytest.skip("no pipelined prefill in this tree")
    monkeypatch.setenv("EXL3_DSV41_PIPELINE_CHUNK", "1024")
    args = v.build_parser(rc).parse_args(["--mode", "cached", "--prefill-check", "--prefill-chunk", "2048"])
    # pipeline.py: any value but '0' is on, the empty string included
    for value, refused in (("", False), ("1", False), ("yes", False), ("0", True)):
        monkeypatch.setenv("EXL3_DSV41_PIPELINE", value)
        errors = v.make_plan(args, rc, [case(2080)])["errors"]
        assert refused == any("EXL3_DSV41_PIPELINE=1" in e for e in errors), (value, errors)
    monkeypatch.delenv("EXL3_DSV41_PIPELINE")
    assert any("EXL3_DSV41_PIPELINE=1" in e for e in v.make_plan(args, rc, [case(2080)])["errors"])


def test_placement_conflicts_are_checked_before_loading():
    if not (ROOT / "exllamav3/model/placement.py").is_file():
        pytest.skip("no explicit placement in this tree")
    parser = v.build_parser(rc)
    rules = "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1"
    def errors(argv, **environ):
        return v.engine_setting_errors(parser.parse_args(argv), environ)
    assert not errors(["--placement", rules])
    assert not errors(["--placement", rules, "--moe-cpu-mode", "compute"])
    assert not errors(["--placement", rules, "--mcl", "0"], EXL3_MOE_CPU_OFFLOAD="4")
    # an empty --placement is no placement and clears the caller's; the words the engine refuses
    # are refused here too
    assert not errors(["--placement", "", "--mcs", "8"], EXL3_PLACEMENT=rules)
    assert any("no placement is written as an empty value" in e for e in errors(["--placement", "none"]))
    # what the loader refuses next to an explicit placement (model/placement.conflicts), from the
    # options and from the caller's environment alike
    for argv, environ, what in (
            (["--mcl", "2"], {}, "EXL3_MOE_CPU_OFFLOAD"),
            ([], {"EXL3_MOE_CPU_OFFLOAD": "2"}, "EXL3_MOE_CPU_OFFLOAD"),
            ([], {"EXL3_MOE_CPU_SPLIT": "8"}, "EXL3_MOE_CPU_SPLIT"),
            (["--moe-cpu-mode", "stream_only"], {}, "EXL3_MOE_CPU_MODE"),
            ([], {"EXL3_MOE_CPU_MODE": "stream_only"}, "EXL3_MOE_CPU_MODE"),
            ([], {"EXL3_MOE_CPU_SPLIT_LAYERS": "2"}, "EXL3_MOE_CPU_SPLIT_LAYERS")):
        found = errors(["--placement", rules] + argv, **environ)
        assert any(what in e for e in found), (argv, environ, found)
    # the removed V4.1 split: its variable set in the caller's environment is refused (the engine
    # would ignore it), whatever its value, and so is its split=<k> grammar as --placement
    for value in ("split=12", "none"):
        found = errors(["--placement", rules], EXL3_DSV41_PLACEMENT=value)
        assert any(f"EXL3_DSV41_PLACEMENT={value!r} is set, but the engine does not read it" in e
                   for e in found), found
    found = errors(["--placement", "split=12"])
    assert any("--placement 'split=12': split=<k> is not a placement rule" in e for e in found), found
    # the caller's explicit placement stays in force without --placement
    assert any("EXL3_MOE_CPU_SPLIT" in e for e in errors(["--mcs", "8"], EXL3_PLACEMENT=rules))
    assert any("EXL3_PLACEMENT" in e for e in errors([], EXL3_PLACEMENT="0-11=cuda:0 experts=hybrid"))
    # -mcl and -mcs are exclusive whichever channel sets them
    assert any("mutually exclusive" in e for e in errors(["--mcs", "8"], EXL3_MOE_CPU_OFFLOAD="2"))
    assert any("not an integer" in e for e in errors([], EXL3_MOE_CPU_SPLIT="many"))


def test_documented_scorer_loading_works(monkeypatch):
    # the snippet doc/dsv41_tools.md gives for loading the scorer by path, run as written
    doc = (ROOT / "doc" / "dsv41_tools.md").read_text()
    section = doc[doc.index("### `tools/dsv41_refcompare.py`"):]
    block = section[section.index("```python") + len("```python"):]
    lines = block[:block.index("```")].strip().splitlines()
    loading = "\n".join(lines[:next(i for i, l in enumerate(lines) if "exec_module" in l) + 1])
    monkeypatch.chdir(ROOT)
    monkeypatch.delitem(sys.modules, "dsv41_refcompare", raising=False)
    scope = {}
    exec(loading, scope)
    assert scope["rc"].GATES == rc.GATES


# ---------------------------------------------------------------------------------
# Noise-shifted limits and the isolation metadata (doc/dsv41_tools.md, "Gates")
# ---------------------------------------------------------------------------------

ISOLATED = {"contended": False, "probe_errors": 0, "isolation_metadata_missing": []}


def captured(n=300, rep=0, meta=ISOLATED, shift=None):
    """A captured case; shift: {positions: delta} added to the reference's actual-token logprobs."""
    c = case(n)
    c.rep, c.meta = rep, dict(meta)
    for positions, delta in (shift or {}).items():
        rows = np.asarray(positions) - 1
        c.ref_actual_lp[rows] += delta
        c.ref_lp[rows, 0] += delta
    return c


def ours(c, shift=None):
    ids, lp, act = rc.as_ours(captured(c.n))
    for positions, delta in (shift or {}).items():
        act[np.asarray(positions) - 1] += delta
    return ids, lp, act


# 14 of the 127 positions of region [1,128): a region p90 of |delta| (its region |dNLL| stays below
# 0.05 up to |delta| 0.45), a global p90 of 0
SPREAD = tuple(range(1, 127, 9))


def scored(reference, candidate, others):
    report = dict(rc.compare(reference, *candidate), nonfinite=0, unfilled=0)
    return rc.attach_noise(report, *rc.reference_noise([reference] + others, reference))


def test_a_check_within_the_reference_noise_is_reported_and_passes():
    ref, other = captured(), captured(rep=1, shift={SPREAD: -0.3})
    noise, excluded = rc.reference_noise([ref, other], ref)
    assert [r["label"] for r in noise] == ["vLLM rep1 vs rep0 n=300"] and not excluded
    # region [1,128) p90: calibrated 0.4; vLLM against itself 0.3 there, so 0.3 + (0.4 - 0.216)
    within = scored(ref, ours(ref, {SPREAD: -0.45}), [other])
    ok, msgs = rc.gates(within)
    assert ok and rc.verdict(within) == "within-noise", msgs
    shifted = [m for m in msgs if m.startswith("FAIL within reference noise  region [1,128) |dlp| p90")]
    assert len(shifted) == 1 and "noise-shifted limit 0.4840" in shifted[0] and "vLLM rep1 vs rep0" in shifted[0]
    assert not [m for m in msgs if m.startswith("FAIL  ")], msgs
    # the same run without the reference noise: the calibrated limits alone, which it fails
    bare = dict(rc.compare(ref, *ours(ref, {SPREAD: -0.45})), nonfinite=0, unfilled=0)
    assert not rc.gates(bare)[0] and rc.verdict(bare) == "fail"
    # beyond the noise-shifted limit: a FAIL of the run
    beyond = scored(ref, ours(ref, {SPREAD: -0.6}), [other])
    ok, msgs = rc.gates(beyond)
    assert not ok and rc.verdict(beyond) == "fail"
    assert any(m.startswith("FAIL  region [1,128) |dlp| p90 0.6000 <= 0.4; beyond the noise-shifted limit 0.4840")
               for m in msgs), msgs
    # the limits come from the capture only: the candidate cannot widen its own
    assert rc.noise_limits(within) == rc.noise_limits(beyond)


def test_noise_below_the_calibration_noise_never_tightens_a_limit():
    ref, other = captured(), captured(rep=1, shift={SPREAD: -0.1})
    assert rc.gates(scored(ref, ours(ref, {SPREAD: -0.39}), [other]))[0]
    ok, msgs = rc.gates(scored(ref, ours(ref, {SPREAD: -0.45}), [other]))
    assert not ok and any("p90 0.4500 <= 0.4; no wider here" in m for m in msgs), msgs


def test_reference_noise_counts_only_comparisons_of_the_same_positions():
    ref = captured()
    # a shorter capture of the same prompt covers region [1,128) whole only from n=128 on
    for m, covered in ((100, False), (200, True)):
        short = captured(m, shift={tuple(p for p in SPREAD if p < m): -0.3})
        report = scored(ref, ours(ref, {SPREAD: -0.45}), [short])
        rn = report["reference_noise"]
        assert rn["sources"] == [f"vLLM n={m} rep0 vs n=300 rep0 positions 1-{m - 1}"]
        assert rn["global"] == {}                       # never the whole case
        regions = {(x["lo"], x["hi"]): x["noise"] for x in rn["regions"]}
        assert bool(regions[(1, 128)]) == covered and regions[(128, 512)] == {}, regions
        assert rc.verdict(report) == ("within-noise" if covered else "fail")
    # a longer capture that starts with the same prompt covers every position
    longer = captured(600, shift={SPREAD: -0.3})
    report = scored(ref, ours(ref, {SPREAD: -0.45}), [longer])
    assert report["reference_noise"]["sources"] == ["vLLM n=600 rep0 prefix vs n=300 rep0"]
    assert report["reference_noise"]["global"] and rc.verdict(report) == "within-noise"
    # another prompt is no measurement of this one
    other_prompt = captured(rep=1, shift={SPREAD: -0.3})
    other_prompt.prompt_ids = other_prompt.prompt_ids + 1
    assert rc.reference_noise([ref, other_prompt], ref) == ([], [])


def test_dlp_max_is_judged_position_by_position():
    # vLLM moves position 1 by 3 nats between two captures of the same prompt
    ref, other = captured(), captured(rep=1, shift={(1,): -3.0})
    at_noisy = scored(ref, ours(ref, {(1,): -6.0}), [other])
    ok, msgs = rc.gates(at_noisy)
    assert ok and rc.verdict(at_noisy) == "within-noise", msgs
    assert any(m.startswith("FAIL within reference noise  |dlp| max 6.0000 <= 5.0") and "position 1 6.0000 <= 7.2560"
               in m for m in msgs), msgs
    # the same difference anywhere else is not excused by position 1's noise
    elsewhere = scored(ref, ours(ref, {(50,): -6.0}), [other])
    ok, msgs = rc.gates(elsewhere)
    assert not ok and any(m.startswith("FAIL  |dlp| max 6.0000 <= 5.0") for m in msgs), msgs
    # nor at position 1 beyond its own noise plus the headroom
    assert not rc.gates(scored(ref, ours(ref, {(1,): -7.5}), [other]))[0]
    # more positions beyond the limit than the report lists: none of them is excused
    many = tuple(range(1, 60))
    report = scored(ref, ours(ref, {many: -6.0}), [captured(rep=1, shift={many: -3.0})])
    ok, msgs = rc.gates(report)
    assert not ok and any("more positions exceed 5.0 than the report lists" in m for m in msgs), msgs


def test_contended_and_bare_captures_are_not_reference_noise():
    ref = captured()
    contended = captured(rep=1, meta={**ISOLATED, "contended": True}, shift={SPREAD: -0.3})
    probed = captured(rep=2, meta={**ISOLATED, "probe_errors": 2}, shift={SPREAD: -0.3})
    bare = captured(rep=3, meta={}, shift={SPREAD: -0.3})
    noise, excluded = rc.reference_noise([ref, contended, probed, bare], ref)
    assert [r["label"] for r in noise] == ["vLLM rep3 vs rep0 n=300"]
    assert excluded == ["n300r1: recorded as contended", "n300r2: recorded with 2 probe errors"]
    noise, excluded = rc.reference_noise([ref, contended, probed, bare], ref, strict=True)
    assert noise == [] and excluded[2] == "n300r3: no isolation metadata (contended, probe_errors; --strict)"
    report = rc.attach_noise(dict(rc.compare(ref, *ours(ref, {SPREAD: -0.45})), nonfinite=0, unfilled=0),
                             noise, excluded)
    ok, msgs = rc.gates(report)
    assert not ok and "info  not used as reference noise: n300r1: recorded as contended" in msgs, msgs


def test_strict_fails_a_reference_without_isolation_metadata(capsys):
    bare = captured(meta={})
    report = dict(rc.compare(bare, *rc.as_ours(bare)), nonfinite=0, unfilled=0)
    ok, msgs = rc.gates(report)
    assert ok and rc.verdict(report) == "pass"
    assert msgs[0].startswith("WARNING reference isolation NOT RECORDED (no contended, probe_errors)")
    ok, msgs = rc.gates(report, strict=True)
    assert not ok and rc.verdict(report, strict=True) == "fail"
    assert msgs[0] == ("FAIL  reference isolation not recorded (no contended, probe_errors): nothing shows "
                       "that vLLM served this prompt alone (--strict)")
    full = captured()
    assert rc.gates(dict(rc.compare(full, *rc.as_ours(full)), nonfinite=0, unfilled=0), strict=True)[0]
    # the harness refuses a bare gated case before anything loads, and says so loudly without --strict
    parser = v.build_parser(rc)
    assert not parser.parse_args([]).strict
    errors = v.make_plan(parser.parse_args(["--strict"]), rc, [bare])["errors"]
    assert any(e.startswith("--strict: the gated reference cases n300r0 carry no isolation metadata") for e in errors)
    assert not v.make_plan(parser.parse_args(["--strict"]), rc, [full])["errors"]
    assert not v.make_plan(parser.parse_args([]), rc, [bare])["errors"]
    banner = v.isolation_banner(rc, [bare, full], strict=False)
    assert banner[0] == "!!! WARNING: 1 of the capture's 2 cases carry no isolation metadata: n300r0"
    assert any("--strict fails them instead" in line for line in banner)
    assert any("--strict: such a case fails" in line for line in v.isolation_banner(rc, [bare], strict=True))
    assert v.isolation_banner(rc, [full], strict=True) == []


def test_within_noise_neither_detects_nor_leaks():
    ref, other = captured(1500), captured(1500, rep=1, shift={SPREAD: -0.3})
    base = scored(ref, ours(ref), [other])
    assert rc.verdict(base) == "pass"
    # an ablation whose only effect is within the reference noise is not detected ...
    within = scored(ref, ours(ref, {SPREAD: -0.45}), [other])
    assert rc.verdict(within) == "within-noise"
    assert rc.ablation_check(within, "engram", base)["status"] == "blind"
    # ... while the same effect judged on the calibrated limits alone would have counted
    plain = lambda r: {k: x for k, x in r.items() if k != "reference_noise"}
    assert rc.ablation_check(plain(within), "engram", plain(base))["status"] == "detected"
    assert rc.ablation_check(scored(ref, ours(ref, {SPREAD: -0.6}), [other]), "engram", base)["status"] == "detected"
    # an isolated ablation (dense_consumers, from 512) does not leak through noise-level changes below it
    late = tuple(range(600, 1400))
    leaky = scored(ref, ours(ref, {SPREAD: -0.45, late: -0.3}), [other])
    assert rc.ablation_check(leaky, "dense_consumers", base)["status"] == "detected"
    leaky = scored(ref, ours(ref, {SPREAD: -0.6, late: -0.3}), [other])
    assert rc.ablation_check(leaky, "dense_consumers", base)["status"] == "leak"


def test_sweep_baseline_within_the_reference_noise_passes():
    ref, other = captured(), captured(rep=1, shift={SPREAD: -0.3})
    def wrap(report):
        ok = rc.gates(report)[0]
        return {"ok": ok, "cases": [{"case": ref.name, "runs": {"cached": {"ok": ok, "report": report}}}]}
    base = scored(ref, ours(ref, {SPREAD: -0.45}), [other])
    ablated = scored(ref, ours(ref, {SPREAD: -0.45, (200,): -8.0}), [other])
    assert rc.verdict(base) == "within-noise"
    summary, lines, ok = v.sweep_summary(rc, ["rope_consecutive"], {None: wrap(base), "rope_consecutive": wrap(ablated)})
    assert ok and summary["tokens"]["rope_consecutive"][0]["status"] == "detected", lines
    # --strict applies to the baseline the sweep re-checks
    bare = captured(meta={})
    base = scored(bare, ours(bare), [])
    assert v.sweep_summary(rc, ["rope_consecutive"], {None: wrap(base), "rope_consecutive": wrap(
        scored(bare, ours(bare, {(200,): -8.0}), []))})[2]
    assert not v.sweep_summary(rc, ["rope_consecutive"], {None: wrap(base), "rope_consecutive": wrap(
        scored(bare, ours(bare, {(200,): -8.0}), []))}, strict=True)[2]
