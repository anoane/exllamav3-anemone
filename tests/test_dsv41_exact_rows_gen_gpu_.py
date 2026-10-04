"""
GPU, EXL3_EXACT_ROWS=1: DeepSeek-V4.1 generation with an n-gram draft returns the tokens, and the
logits, generation without a draft returns.

Every case runs twice through the Generator, on one loaded model: without a draft, and with
n-gram drafting (ngram_match_min, num_draft_tokens = 7). Each run gets a new Generator, hence a
new page table over the Cache (no prompt-cache reuse) and a new checkpoint cache, and the same
max_chunk_size, so both runs prefill the prompt with the same forwards; the test checks that too.

Cases:
  repeat    a repetitive prompt (the n-gram draft is accepted), greedy, tokens and logits
  prose     a prose prompt (short matches that diverge are rejected), greedy, tokens and logits
  plan      a prompt of under 120 tokens generating past position 1100: the generation crosses the
            positions where a one-token step changes its attention plan (128, 257, 512, 1025 on
            DeepSeek-V4.1-Flash), at which a draft window is cut
  requeue   a small max_rq_tokens and a recurrent checkpoint on every page: the job requeues, at
            the same token in both runs, and the requeued job prefills the same forwards (none,
            when the requeue is on a checkpoint); no verify forward reaches a checkpoint position
  budget    max_new_tokens unset: both runs generate until the Cache is full, the same number of
            tokens, across recurrent checkpoints
  sampled   a seeded sampler with temperature
  banned    a job with banned strings: it must not draft at all

In the drafted run every trunk forward is recorded: a verify forward has at most 8 rows and
carries params["exact_rows"] (no row of it changes the attention plan), and accepted + rejected
draft tokens add up to the drafted ones. Over all cases some draft tokens must be accepted and
some rejected.

Tokens per second of both runs are printed, over the generation only (the job's time_generate:
first forward to last token, without the prompt). Before the cases, one undrafted and one drafted
generation run without being compared or timed (the plan case's, which passes the positions where
the selection starts): compilation and the first use of a launch bucket, which the launch
autotuner may time with another configuration, then fall in neither run of a case. --no-warmup
skips them.

An extension that reports the single launch (exact_rows_caps() & 32) serves the rows of an EXL3
linear and of the grouped output projection with ONE launch, under the launch record of a one-row
call, where that call is the cooperative FP16 kernel: the grouped projection always, a mul1 linear
with EXL3_INT8_GEMV=0. The tokens and logits do not show which route ran, so each case also reads
the extension's count of such calls (exact_rows_one_launches) around its drafted run: a run with
verify forwards must have made some, and the count per verify forward is printed (with the int8
path off, about the EXL3 linears and grouped projections of a forward; with it on, about the
grouped projections). Run it under both settings, each with one autotune file:

    EXL3_EXACT_ROWS=1 python tests/test_dsv41_exact_rows_gen_gpu_.py [checkpoint-dir] [options]
    EXL3_EXACT_ROWS=1 EXL3_INT8_GEMV=0 python tests/test_dsv41_exact_rows_gen_gpu_.py [checkpoint-dir] [options]

(checkpoint-dir defaults to $DSV41_MODEL_DIR; the placement comes from EXL3_PLACEMENT and
CUDA_VISIBLE_DEVICES.) Run it once with EXL3_MOE_TIER_VERIFY=1 too: under the mode the MoE looks
the expert cache up once for the rows of a verify forward (up to 8 x top-k experts in one decode
lookup) where the extension has the row-exact entry points, and once per row, back to back, where
it does not. Loads the whole model: run it alone on the host. Exit status 0 passed, 1 failed,
2 not run.
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPEAT = "The quick brown fox jumps over the lazy dog. " * 8
PROSE = ("The lighthouse at the end of the northern pier had been dark for eleven years when the harbour board "
         "finally agreed to sell it. Nobody in the town expected a buyer, and nobody expected the buyer to be")
DRAFT_TOKENS = 7


class Forwards:
    """Every trunk forward and prefill of a run: (position of row 0, rows, params["exact_rows"])."""

    def __init__(self, model):
        self.model, self.gen, self.pre = model, [], []

    def _wrap(self, call, log):
        def wrapped(input_ids, params = None):
            states = (params or {}).get("recurrent_states")
            position = states[0].position if states else None
            out = call(input_ids, params)
            log.append((position, int(input_ids.shape[-1]), bool((params or {}).get("exact_rows"))))
            return out
        return wrapped

    def __enter__(self):
        m = self.model
        m.forward, m.prefill = self._wrap(m.forward, self.gen), self._wrap(m.prefill, self.pre)
        return self

    def __exit__(self, *exc):
        del self.model.forward, self.model.prefill
        return False


def run(torch, model, cache, tok, ids, *, draft, ngram_min, chunk, max_new_tokens, sampled = False, logits = False,
        generator = None, job = None) -> dict:
    from exllamav3 import ArgmaxSampler, DefaultSampler, Generator, Job
    kw = dict(generator or {})
    if draft:
        kw.update(ngram_match_min = ngram_min, num_draft_tokens = DRAFT_TOKENS)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1, max_chunk_size = chunk, **kw)
    j = Job(input_ids = ids, max_new_tokens = max_new_tokens, return_logits = logits,
            sampler = DefaultSampler() if sampled else ArgmaxSampler(), seed = 20261004 if sampled else None,
            **(job or {}))
    out = {"tokens": [], "requeue_at": [], "reason": None, "seconds": None}
    digest = hashlib.sha256()
    with Forwards(model) as fw:
        gen.enqueue(j)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "error":
                    raise r["error"]
                if r.get("token_ids") is not None:
                    out["tokens"] += r["token_ids"].flatten().tolist()
                if r.get("logits") is not None:
                    lg = r["logits"].detach().to("cpu").contiguous().reshape(-1)
                    digest.update(lg.view({2: torch.int16, 4: torch.int32}[lg.element_size()]).numpy().tobytes())
                if r.get("requeue"):
                    out["requeue_at"].append(len(out["tokens"]))
                if r.get("eos"):
                    out["reason"] = r.get("eos_reason")
                    # generation only, summed over the requeues of the job
                    out["seconds"] = r.get("time_generate")
    out.update(logits = digest.hexdigest() if logits else None, forwards = fw.gen, prefills = fw.pre,
               accepted = j.accepted_draft_tokens, rejected = j.rejected_draft_tokens,
               drafted = sum(rows - 1 for _, rows, _ in fw.gen))
    return out


def check(model, name, prompt_len, plain, drafted, expect_draft = True, expect_requeue = False,
          page_checkpoints = False) -> list:
    """The failures of one case (empty when it passed). page_checkpoints: the run stashes a recurrent
    checkpoint at the end of every page."""
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.model.math_policy import EXACT_ROWS_MAX
    bad = []
    if plain["tokens"] != drafted["tokens"]:
        first = next((i for i, (a, b) in enumerate(zip(plain["tokens"], drafted["tokens"])) if a != b),
                     min(len(plain["tokens"]), len(drafted["tokens"])))
        bad.append(f"tokens differ from token {first} (position {prompt_len + first}): {len(plain['tokens'])} "
                   f"undrafted, {len(drafted['tokens'])} drafted")
    for key, what in (("logits", "logits"), ("reason", "eos reason"), ("requeue_at", "requeue points"),
                      ("prefills", "prefill forwards")):
        if plain[key] != drafted[key]:
            bad.append(f"{what} differ: {plain[key]!r} undrafted, {drafted[key]!r} drafted")
    if any(rows != 1 for _, rows, _ in plain["forwards"]):
        bad.append("the undrafted run made a forward of several rows")
    for position, rows, flagged in drafted["forwards"]:
        if rows > 1 and not (flagged and rows <= EXACT_ROWS_MAX and model.exact_rows_span(position, rows) == rows):
            bad.append(f"a verify forward of {rows} rows at position {position} "
                       f"{'carried' if flagged else 'did not carry'} params['exact_rows'] "
                       f"(the attention plan holds for {model.exact_rows_span(position, rows)} rows)")
            break
    if page_checkpoints:
        crossing = [(p, rows) for p, rows, _ in drafted["forwards"] if (p + rows - 1) // PAGE_SIZE != p // PAGE_SIZE]
        if crossing:
            bad.append(f"verify forwards reach a checkpoint position (position, rows): {crossing[:4]}")
    if drafted["accepted"] + drafted["rejected"] != drafted["drafted"]:
        bad.append(f"{drafted['accepted']} accepted + {drafted['rejected']} rejected draft tokens, "
                   f"{drafted['drafted']} drafted")
    if not expect_draft and drafted["drafted"]:
        bad.append(f"{drafted['drafted']} tokens were drafted for a job that must not draft")
    if expect_requeue and not plain["requeue_at"]:
        bad.append("the undrafted run did not requeue: the case does not test what it is for")
    n = len(plain["tokens"])
    rate = lambda r: len(r["tokens"]) / max(r["seconds"] or 0.0, 1e-9)
    windows = sorted({rows for _, rows, _ in drafted["forwards"] if rows > 1})
    print(f"  {'FAIL' if bad else 'OK  '} {name}: prompt {prompt_len}, {n} tokens, eos {plain['reason']}, requeues at "
          f"{plain['requeue_at']}; drafted {drafted['drafted']} (accepted {drafted['accepted']}, rejected "
          f"{drafted['rejected']}) in forwards of {windows or 'one'} rows; {rate(plain):.1f} tok/s undrafted, "
          f"{rate(drafted):.1f} tok/s drafted ({rate(drafted) / max(rate(plain), 1e-9):.2f}x)", flush = True)
    for b in bad:
        print(f"       {b}", flush = True)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description = "EXL3_EXACT_ROWS: drafted generation against undrafted generation")
    ap.add_argument("model", nargs = "?", default = os.environ.get("DSV41_MODEL_DIR"))
    ap.add_argument("--cache-tokens", type = int, default = 4096, help = "Cache size (the budget case fills it)")
    ap.add_argument("--chunk", type = int, default = 2048, help = "max_chunk_size of the load and of every Generator")
    ap.add_argument("--ngram-min", type = int, default = 2, help = "ngram_match_min of the drafted runs")
    ap.add_argument("--cases", default = None, help = "comma-separated subset of the cases")
    ap.add_argument("--no-warmup", action = "store_true", help = "skip the two generations that run before the cases")
    args = ap.parse_args()
    if not args.model:
        print("  --  exact rows, generator: no checkpoint (pass a directory or set DSV41_MODEL_DIR), skipped")
        return 2
    from exllamav3.model.math_policy import EXACT_ROWS
    if not EXACT_ROWS:
        print("  --  exact rows, generator: needs EXL3_EXACT_ROWS=1, skipped")
        return 2
    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer
    if not torch.cuda.is_available():
        print("  --  exact rows, generator: needs CUDA, skipped")
        return 2

    # The single launch for the rows of a linear or of the grouped projection (exact_rows.h): the
    # count of calls served that way, or None for an extension without it
    from exllamav3.ext import exllamav3_ext as ext
    caps = int(ext.exact_rows_caps()) if hasattr(ext, "exact_rows_caps") else 0
    one_launches = (lambda: int(ext.exact_rows_one_launches())) if caps & 32 else None
    print(f"  --  exact rows, generator: row-exact entry points of the extension: {caps or 'none'}; "
          f"EXL3_INT8_GEMV={os.environ.get('EXL3_INT8_GEMV', 'unset (2)')}; the rows of a linear or of the grouped "
          f"projection as one launch: {'counted' if one_launches else 'not in this extension'}", flush = True)

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    # max_history as model_init sets it for a draft of this length
    cache = Cache(model, max_num_tokens = args.cache_tokens, max_batch_size = 1, max_history = DRAFT_TOKENS)
    model.load(progressbar = False, max_chunk_size = args.chunk)
    tok = Tokenizer.from_config(config)
    assert model.caps.get("exact_rows"), "not a model EXL3_EXACT_ROWS covers"
    encode = lambda text: tok.encode(text, add_bos = True)
    repeat, prose = encode(REPEAT), encode(PROSE)

    cases = (
        ("repeat", repeat, dict(max_new_tokens = 160, logits = True), {}),
        ("prose", prose, dict(max_new_tokens = 160, logits = True), {}),
        ("plan", repeat, dict(max_new_tokens = 1150), {}),
        ("requeue", repeat, dict(max_new_tokens = 700, generator = dict(recurrent_checkpoint_interval = 256),
                                 job = dict(max_rq_tokens = 300)),
         dict(expect_requeue = True, page_checkpoints = True)),
        ("budget", repeat, dict(max_new_tokens = None), {}),
        ("sampled", prose, dict(max_new_tokens = 160, sampled = True, logits = True), {}),
        ("banned", repeat, dict(max_new_tokens = 96, job = dict(banned_strings = ["qzxqzx"])), dict(expect_draft = False)),
    )
    wanted = None if args.cases is None else {c.strip() for c in args.cases.split(",") if c.strip()}
    unknown = (wanted or set()) - {name for name, _, _, _ in cases}
    if unknown:
        print(f"unknown case(s) {sorted(unknown)}; the cases are {[name for name, _, _, _ in cases]}")
        return 2

    failures, accepted, rejected = [], 0, 0
    try:
        if not args.no_warmup:
            # results discarded: compilation and the first use of every launch bucket happen here
            warm = dict(next(kw for name, _, kw, _ in cases if name == "plan"), ngram_min = args.ngram_min,
                        chunk = args.chunk)
            for draft in (False, True):
                run(torch, model, cache, tok, repeat, draft = draft, **warm)
            print("  --  warm-up: one undrafted and one drafted generation, not compared", flush = True)
        for name, ids, kw, expect in cases:
            if wanted is not None and name not in wanted:
                continue
            common = dict(ngram_min = args.ngram_min, chunk = args.chunk, **kw)
            plain = run(torch, model, cache, tok, ids, draft = False, **common)
            before = one_launches() if one_launches else 0
            drafted = run(torch, model, cache, tok, ids, draft = True, **common)
            served = one_launches() - before if one_launches else 0
            prompt_len = ids.shape[-1]
            bad = check(model, name, prompt_len, plain, drafted, **expect)
            verifies = sum(1 for _, rows, flagged in drafted["forwards"] if rows > 1 and flagged)
            if one_launches:
                print(f"       one-launch calls of the drafted run: {served} in {verifies} verify forwards"
                      + (f" ({served / verifies:.1f} per forward)" if verifies else ""), flush = True)
                if verifies and not served:
                    bad.append(f"{verifies} verify forwards and no call served with one launch: the extension "
                               f"reports the single launch (exact_rows_caps() & 32) and did not make it")
                    print(f"       {bad[-1]}", flush = True)
            if name == "plan" and (prompt_len >= 120 or prompt_len + len(plain["tokens"]) <= 1100):
                bad.append(f"plan: prompt {prompt_len} tokens, generation to position "
                           f"{prompt_len + len(plain['tokens'])}: it must start below 120 and pass 1100")
                print(f"       {bad[-1]}", flush = True)
            if name == "budget" and len(plain["tokens"]) != args.cache_tokens - prompt_len - 1:
                # informative: the default budget of a job is what the Cache still holds
                print(f"       note: {len(plain['tokens'])} tokens for a Cache of {args.cache_tokens} and a prompt "
                      f"of {prompt_len}", flush = True)
            failures += [f"{name}: {b}" for b in bad]
            accepted += drafted["accepted"]
            rejected += drafted["rejected"]
    finally:
        model.unload()
    if wanted is None and not (accepted and rejected):
        failures.append(f"over all cases {accepted} draft tokens were accepted and {rejected} rejected: both must "
                        f"occur for the comparison to cover them (try another --ngram-min)")
    if failures:
        print(f"FAIL exact rows, generator: {len(failures)} failure(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"PASS exact rows, generator: drafted generation equals undrafted generation in every case "
          f"({accepted} draft tokens accepted, {rejected} rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
