#!/usr/bin/env python3
"""
Capture vLLM reference logprobs for the DeepSeek-V4.1-Flash port.

Run it against a vLLM OpenAI-compatible server that serves DeepSeek-V4.1-Flash
(--url), while nothing else uses that server: a shared batch changes vLLM's
numerics, and a long prompt holds the server for a while (n=20000 took about
76 s of exclusive service time on the capture the gates were calibrated on).

The harness shifts each calibrated limit by the reference's own noise on the
gated case's positions, measured on the other captures of the same file: other
reps of the prompt and the prompt at other lengths (tools/dsv41_refcompare.py,
reference_noise). The calibration capture has one rep at n=1500 and one at 20000,
so nothing in it measures vLLM against itself on positions >= 1500 of the n=20000
prompt, and the calibrated limits alone apply there. Several reps at a length give
that measurement. Every case also records whether vLLM served it alone
('contended', 'probe_errors'), which the harness's --strict requires.

  * idle-gated: before every request, poll /metrics until
    vllm:num_requests_running == 0 and vllm:num_requests_waiting == 0; while the
    request runs, a poller records whether anyone else's request shared the
    batch (a shared batch changes vLLM's numerics), and such a capture is
    flagged 'contended';
  * one request at a time;
  * token-id prompts (no BOS, as the existing file), max_tokens 1, temperature 0,
    prompt_logprobs and logprobs 20, generated token returned as an id;
  * prompts default to prefixes of the longest prompt in the existing capture
    (--ref), so every capture shares its prefix with it;
  * writes a NEW file in the same schema (top = 20): it refuses to overwrite an
    existing file and never touches --ref.

See doc/dsv41_tools.md.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

REF_PATH = os.environ.get("DSV41_VLLM_REF")


def http(base: str, path: str, body: dict | None = None, timeout: float = 3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data,
                                 {"content-type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        raw = r.read()
    return raw.decode() if path == "/metrics" else json.loads(raw)


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text -> {metric name: value summed over label sets}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line[:line.index("{")], line[line.rindex("}") + 1:].split()
        else:
            name, *rest = line.split()
        if not rest:
            continue
        try:
            v = float(rest[0])              # an optional timestamp follows the value
        except ValueError:
            continue
        out[name.strip()] = out.get(name.strip(), 0.0) + v
    return out


def load_state(base: str) -> tuple[float, float]:
    m = parse_metrics(http(base, "/metrics", timeout = 30))
    if "vllm:num_requests_running" not in m or "vllm:num_requests_waiting" not in m:
        raise RuntimeError("/metrics has no vllm:num_requests_running / vllm:num_requests_waiting")
    return m["vllm:num_requests_running"], m["vllm:num_requests_waiting"]


def wait_idle(base: str, timeout: float, poll: float = 2.0, log = print) -> float:
    """Block until nothing runs and nothing waits; returns the seconds waited."""
    t0 = time.monotonic()
    said = False
    while True:
        run, wait = load_state(base)
        if run == 0 and wait == 0:
            return time.monotonic() - t0
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"service not idle after {timeout:.0f}s (running {run:g}, waiting {wait:g})")
        if not said:
            log(f"  waiting for idle: running {run:g}, waiting {wait:g}")
            said = True
        time.sleep(poll)


class ContentionProbe:
    """Polls /metrics while our request runs; anything beyond our own request is contention."""

    def __init__(self, base: str, period: float = 0.5):
        self.base, self.period = base, period
        self.max_running = 0.0
        self.max_waiting = 0.0
        self.samples = 0
        self.errors = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target = self._loop, daemon = True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                r, w = load_state(self.base)
                self.max_running = max(self.max_running, r)
                self.max_waiting = max(self.max_waiting, w)
                self.samples += 1
            except Exception:
                self.errors += 1
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout = 35)
        if self._t.is_alive():
            self.errors += 1

    @property
    def contended(self) -> bool:
        return self.max_running > 1 or self.max_waiting > 0


def request_body(model: str, ids: list[int], top: int, token_ids: bool) -> dict:
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": top, "logprobs": top}
    if token_ids:
        body["return_tokens_as_token_ids"] = True
        body["return_token_ids"] = True
    return body


def to_case(resp: dict, n: int, rep: int, ids: list[int], seconds: float) -> dict:
    """A /v1/completions response -> one case of the capture schema (tools/dsv41_refcompare.py load_ref)."""
    c = resp["choices"][0]
    pl = [None if p is None else {str(k): (v["logprob"] if isinstance(v, dict) else v) for k, v in p.items()}
          for p in c["prompt_logprobs"]]
    ranks = [None if p is None else next((v.get("rank") for k, v in p.items()
                                          if str(k) == str(ids[i]) and isinstance(v, dict)), None)
             for i, p in enumerate(c["prompt_logprobs"])]
    lp = c.get("logprobs") or {}
    if len(pl) != n:
        raise ValueError(f"n={n}: service returned {len(pl)} prompt_logprobs entries")
    for i, row in enumerate(pl[1:], 1):
        if not row or str(ids[i]) not in row:
            raise ValueError(f"missing actual-token reference logprob at position {i}")
        for token, value in row.items():
            if int(token) < 0 or not math.isfinite(float(value)):
                raise ValueError(f"invalid ID/logprob at position {i}")
    return {"n": n, "rep": rep, "prompt_ids": ids, "prompt_logprobs": pl, "actual_ranks": ranks,
            "gen_text": c.get("text", ""), "gen_tokens": lp.get("tokens") or [],
            "gen_token_ids": c.get("token_ids"), "gen_top_logprobs": lp.get("top_logprobs") or [],
            "seconds": round(seconds, 2)}


def source_ids(ref: str, lengths: list[int]) -> list[int]:
    with open(ref) as f:
        d = json.load(f)
    longest = max(d["cases"], key = lambda c: c["n"])["prompt_ids"]
    need = max(lengths)
    if need > len(longest):
        raise SystemExit(f"longest prompt in {ref} has {len(longest)} ids; {need} requested "
                         f"(pass --ids-file with a longer token-id list)")
    return list(longest)


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default = "http://127.0.0.1:8000",
                    help = "base URL of the vLLM server (default: vLLM's own default, http://127.0.0.1:8000)")
    ap.add_argument("--model", default = None, help = "served model name (default: the first of /v1/models)")
    ap.add_argument("--lengths", default = "1500,20000")
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--top", type = int, default = 20)
    ap.add_argument("--out", required = True, help = "NEW file to write (refuses to overwrite)")
    ap.add_argument("--ref", default = REF_PATH,
                    help = "prompt ids are prefixes of this capture's longest prompt (default: DSV41_VLLM_REF)")
    ap.add_argument("--ids-file", default = None, help = "JSON list of token ids to take prefixes from instead")
    ap.add_argument("--idle-timeout", type = float, default = 1800)
    ap.add_argument("--no-token-ids", action = "store_true",
                    help = "do not ask for return_tokens_as_token_ids / return_token_ids")
    ap.add_argument("--dry-run", action = "store_true", help = "print the plan, touch no network")
    args = ap.parse_args(argv)

    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]
    if not lengths or min(lengths) < 2 or args.reps < 1 or args.top < 5 \
            or not math.isfinite(args.idle_timeout) or args.idle_timeout <= 0:
        ap.error("lengths >= 2, reps >= 1, top >= 5 and a positive idle timeout are required")
    out = os.path.abspath(args.out)
    if os.path.exists(out) or os.path.exists(out + ".partial"):
        print(f"refusing: {out} (or its .partial) exists; captures always go to a new file")
        return 2
    if os.path.realpath(out) in {os.path.realpath(p) for p in (REF_PATH, args.ref) if p}:
        print(f"refusing: never overwrite {args.ref or REF_PATH}")
        return 2
    if args.ids_file:
        with open(args.ids_file) as f:
            ids = [int(v) for v in json.load(f)]
        if max(lengths) > len(ids):
            print(f"--ids-file has {len(ids)} ids; {max(lengths)} requested")
            return 2
    elif args.ref:
        ids = source_ids(args.ref, lengths)
    else:
        print("no prompt source: pass --ids-file, or --ref (or DSV41_VLLM_REF) with an existing capture")
        return 2

    plan = [(n, rep) for n in lengths for rep in range(args.reps)]
    print(f"plan: {len(plan)} requests, one at a time, idle-gated on {args.url}/metrics -> {out}")
    for n, rep in plan:
        print(f"  n={n:6d} rep={rep}  prompt_logprobs {args.top}  max_tokens 1")
    if args.dry_run:
        return 0

    model = args.model or http(args.url, "/v1/models", timeout = 30)["data"][0]["id"]
    token_ids = not args.no_token_ids
    doc = {"meta": {"engine": "vLLM reference capture", "captured": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "top": args.top, "gen_tokens": 1, "tool": "tools/dsv41_capture_ref.py",
                    "served_model_id_sha256": hashlib.sha256(model.encode()).hexdigest(),
                    "server_build_identity": "not exposed by completion API; record separately",
                    "prompt_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()},
           "cases": []}
    stage_dir = tempfile.mkdtemp(prefix=".dsv41-capture-", dir=os.path.dirname(out))
    partial = os.path.join(stage_dir, "capture.partial")
    print(f"partial capture retained on failure: {partial}")
    # Keep one descriptor in a private directory throughout capture. Never reopen a
    # predictable public .partial path that another writer could replace with a symlink.
    with open(partial, "x") as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(doc, f)
        f.flush()
        for n, rep in plan:
            waited = wait_idle(args.url, args.idle_timeout)
            prompt = ids[:n]
            t0 = time.monotonic()
            with ContentionProbe(args.url) as probe:
                try:
                    resp = http(args.url, "/v1/completions", request_body(model, prompt, args.top, token_ids))
                except urllib.error.HTTPError as e:
                    if e.code == 400 and token_ids:
                        print("  service rejected the token-id options; retrying without them")
                        token_ids = False
                        resp = http(args.url, "/v1/completions", request_body(model, prompt, args.top, False))
                    else:
                        raise
            secs = time.monotonic() - t0
            case = to_case(resp, n, rep, prompt, secs)
            case["idle_wait_s"] = round(waited, 1)
            case["contended"] = probe.contended
            case["probe_max_running"] = probe.max_running
            case["probe_samples"] = probe.samples
            case["probe_errors"] = probe.errors + (probe.samples == 0)
            doc["cases"].append(case)
            f.seek(0)
            f.truncate()
            json.dump(doc, f)
            f.flush()
            os.fsync(f.fileno())
            print(f"  n={n:6d} rep={rep}  {secs:6.1f}s  gen={case['gen_text']!r}"
                  f"{'  CONTENDED (another request shared the batch)' if probe.contended else ''}", flush = True)
    os.link(partial, out)       # atomically fails if out was created after the preflight
    os.unlink(partial)         # same completed inode remains at out
    os.rmdir(stage_dir)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
