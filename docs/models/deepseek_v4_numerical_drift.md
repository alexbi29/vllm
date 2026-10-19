# Measuring numerical drift: decode versus prefill

Use this when a branch, kernel, quantization format, scheduler or CUDA-graph
change alters reasoning length or output tokens. Compare the probability of
**the same token under the same token history**, before judging free-running
answers. A deterministic answer can still come from incorrect arithmetic.

This guide adapts the method and published measurements in
[vLLM #59709's reproduction comment](https://github.com/vllm-project/vllm/pull/59709#issuecomment-6133054205).
The [upstream PR](https://github.com/vllm-project/vllm/pull/59709) fixes C128
compressor updates being omitted from captured graphs. The source comment,
including its original script, is archived in
`/home/ubuntu/src/vllm/bench/experiments/ds4-drift-method-20261015/upstream-measurement-comment.json` (local archive).
The measurements below are upstream results, not a new local drift campaign.

## The method

1. Generate a continuation while collecting each sampled token's decode logprob.
2. Append those exact token IDs to the original prompt IDs.
3. Submit the combined sequence as a prompt with `prompt_logprobs=0`.
   This scores the observed continuation through prefill, rather than asking
   the model to generate a second continuation.
4. Align by token position and compare decode and prefill logprobs.
5. Examine whether error increases after a cache boundary, graph transition,
   context length or concurrency change.

Both passes use the same causal prefix at every scored position. They exercise
different execution paths, which makes this useful for finding decode/cache
bugs that barely affect a short-answer accuracy benchmark. It is a consistency
check; neither path is automatically a numerical reference.

### What upstream measured

The author used a GB300 node, TP2/EP, `FULL_AND_PIECEWISE` graphs, base
`56e3b081a` versus the same source plus the fix. Each arm sampled 64 prompts,
up to 384 output tokens, temperature 1, seed equal to prompt index, with 16 client
threads. The original script then rescored each arm's own sampled continuations.

| Reported metric | Old main | Fixed |
| --- | ---: | ---: |
| Scored tokens | 23522 | 22813 |
| Mean absolute logprob difference, all tokens | 0.1270 | 0.0701 |
| Generated positions 0–127 | 0.0845 | 0.0799 |
| Generated positions 128+ | 0.1485 | 0.0649 |
| Tokens with absolute difference greater than 1 | 535 /2.27% | 112 /0.49% |

These are descriptive results from different sampled token sets, not a paired
old/new token comparison or a significance test. They are not SM120 acceptance
thresholds. The residual error after the fix is also a reason not to demand
zero drift between different prefill/decode kernels.

## Exact alignment

Let `p` contain `L` prompt tokens and `y[t]` be generated token`t`, zero-based.

```text
decode[t] = log P(y[t] | p, y[:t])
prefill[t] = scored_prompt_logprobs[L + t][str(y[t])]["logprob"]
signed_delta[t] = decode[t] - prefill[t]
abs_delta[t] = abs(signed_delta[t])
```

The `prompt_logprobs` entry at position`i` scores the token at`i`, predicted
from the preceding position. The first prompt entry is normally unscored;
do not remove that entry and shift the remaining indices.

Record both the generated index`t` and absolute predicted position`L+t`.
Its conditioning input position is`L+t-1`. Binning by generated positions128+
reproduces the upstream report, but does not locate an actual C128 boundary
unless prompt length is accounted for.

Our frozen 292-token prompt first showed affected accepted target logits at
generated index 92: predicted absolute position 384, after input 383. Visible
greedy output did not split until index 121. See the
`/home/ubuntu/src/vllm/bench/experiments/ds4-async-causal-20261014/report.md` (local archive).
Logprob drift can therefore precede token divergence.

Also label generated token 0 separately: its initial generation probability
normally comes from the prompt's prefill, rather than a subsequent decode step.
Keep it for reproduction, but do not describe every generated row as decode-only.

## Freeze the controls first

- Establish unchanged-source A/A repeatability before making exact A/B scores
  a release gate, especially at long context. Repeat the same literal IDs,
  seed and sampling configuration with genuinely cold caches. If A/A differs,
  preserve that finding and localize it; do not silently loosen the A/B gate
  or attribute all observed variation to the patch.
- Record actual process PID, source `PYTHONPATH`, source head plus dirty-file
  hashes, native environment/build, GPU model and driver. A version string or
  YAML path alone does not establish what the live worker loaded.
- Pin checkpoint, tokenizer and draft revisions; preserve literal prompt IDs,
  rendered chat template, reasoning effort and sampling configuration.
- Record weights/KV dtypes, backend, TP/EP, graph mode and captured shapes,
  scheduler, batch budget, slots, context, physical KV pool and CPU offload.
- Record effective environment defaults, not just explicitly set variables.
- Generate at temperature 1 with neutral penalties/filtering for the original
  experiment. Use `raw_logprobs` consistently: prompt scores bypass sampling
  processors, while processed decode scores can include temperature/top-k/top-p.
  Do not compare logits from one arm with logprobs from another.
- Prevent scoring from reusing decode-produced KV. Use an isolated worker with
  prefix caching disabled, or distinct request `cache_salt` values as below.
  Check returned cached-token counts when available; missing metadata is not0.
- Start at C1, then test the intended concurrency. Concurrency changes batching
  and kernel shapes. The upstream16-thread workload is not automatically a
  suitable default for our 12-slot profile.

Do not reconstruct token IDs by decoding output text and retokenizing it.
That can alter whitespace, special tokens and Unicode. For chat/reasoning
models, render/tokenize once and retain the exact IDs, including the thinking
prefix. Raw completions avoid parser-dependent removal of reasoning text.

For two free-running continuations, their score distributions have the same
history only through the **first differing token position**, inclusive. Scores
at that position still condition on identical preceding IDs. Afterward,
teacher-force one frozen continuation into both arms before comparing scores.
A maximum logprob difference over already divergent histories is not a measure
of branch-induced numerical drift. Selected-token scores are comparable only
when the selected token itself matches; otherwise compare common token IDs in
the distributions. A top20 return is a partial distribution, not a full-vocab
equivalence check.

For an intended exact storage/layout change, use an additional direct operator
check: retain actual model activations, routing, scale words and weights, run
both layouts on those same operands, and compare every valid output byte.
Exclude padding, cover sparse/empty expert routes and poisoned CUDA-graph
replay, and keep instrumentation out of performance and production workers.
This isolates the changed operation from other model nondeterminism. It proves
only the captured cases; whole-model quality and long-context release checks
remain separate. Our `/home/ubuntu/src/vllm/bench/experiments/ds4-csf-align64-20261015/WORKING-STATE.md` (local archive)
records a failed original long-score gate, unchanged-source64K variance, and
356 byte-exact real-operand products rather than erasing the failed gate.

## Reference workflow for one prompt

This is a raw-API numerical diagnostic, like our existing frozen-token probes;
there is currently no `vbench.py` subcommand for it. Continue using
`bench/vbench.py` for quality, throughput and long-context benchmarks.

Point this example at a separately staged worker. The endpoint must support
`logprobs`, `prompt_logprobs`, token-ID returns and `cache_salt`. Use the existing
environment through `uv`; do not modify the production launch to run the example.
The snippet is a reusable reference, not a record of a run executed for this doc.

```bash
DRIFT_BASE=http://127.0.0.1:18010 \
DRIFT_MODEL=deepseek-v4-flash \
VIRTUAL_ENV=/home/ubuntu/vllm-env-integration-20261010 \
uv run --no-project --active python - <<'PY'
import json
import math
import os
from pathlib import Path
import uuid

import requests

base = os.environ["DRIFT_BASE"].rstrip("/")
model = os.environ["DRIFT_MODEL"]
run_id = uuid.uuid4().hex
out = Path("/tmp") / ("decode-prefill-" + run_id)
out.mkdir()

def call(route, payload, artifact):
    response = requests.post(base + route, json=payload, timeout=1800)
    (out / (artifact + ".response.txt")).write_text(response.text)
    response.raise_for_status()
    return response.json()

# Replace this with frozen rendered chat IDs when testing reasoning settings.
p = call("/tokenize", {
    "model": model, "prompt": "Explain how TCP congestion control works.\n\n",
    "add_special_tokens": True,
}, "tokenize")["tokens"]
generated = call("/v1/completions", {
    "model": model, "prompt": p, "add_special_tokens": False,
    "max_tokens": 384, "temperature": 1, "top_p": 1, "top_k": -1,
    "min_p": 0, "repetition_penalty": 1,
    "presence_penalty": 0, "frequency_penalty": 0, "seed": 0,
    "logprobs": 0, "return_token_ids": True,
    "return_tokens_as_token_ids": True, "skip_special_tokens": False,
    "cache_salt": run_id + "-generate",
}, "decode")
choice = generated["choices"][0]
assert choice.get("prompt_token_ids") in (None, p), "Server changed prompt IDs"
y = choice["token_ids"]
decode = choice["logprobs"]["token_logprobs"]
assert p and y and len(y) == len(decode), "Empty or misaligned decode response"

scored = call("/v1/completions", {
    "model": model, "prompt": p + y, "add_special_tokens": False,
    "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
    "return_token_ids": True, "cache_salt": run_id + "-score",
}, "prefill")
scores = scored["choices"][0]["prompt_logprobs"]
assert scored["choices"][0].get("prompt_token_ids") in (None, p + y)
assert len(scores) == len(p) + len(y), "Misaligned prompt scores"
cached = (scored.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")
assert cached in (None, 0), "Scoring reused cached KV"

rows = []
for t, tid in enumerate(y):
    entry = scores[len(p) + t]
    assert entry is not None and str(tid) in entry, (t, tid, "Missing score")
    d, f = decode[t], entry[str(tid)]["logprob"]
    assert d is not None and f is not None, (t, "Null logprob")
    assert math.isfinite(d) and math.isfinite(f), (t, "Non-finite logprob")
    rows.append({"generated_index": t, "absolute_prediction_position": len(p) + t,
                 "token_id": tid, "decode": d, "prefill": f,
                 "signed_delta": d - f, "abs_delta": abs(d - f)})

def mean(values):
    return sum(values) / len(values) if values else None

summary = {
    "tokens": len(rows), "finish_reason": choice["finish_reason"],
    "mean_abs_all": mean([r["abs_delta"] for r in rows]),
    "mean_abs_generated_0_127": mean([r["abs_delta"] for r in rows[:128]]),
    "mean_abs_generated_128_plus": mean([r["abs_delta"] for r in rows[128:]]),
    "fraction_abs_gt_1": sum(r["abs_delta"] > 1 for r in rows) / len(rows),
    "scoring_cached_prompt_tokens": cached,
}
(out / "aligned.json").write_text(json.dumps({
    "prompt_ids": p, "generated_ids": y, "rows": rows, "summary": summary,
}, indent=2))
print(json.dumps(summary, indent=2))
print("Raw responses and alignment:", out)
PY
```

The extra token generated by the scoring request is irrelevant. Only its
**prompt** scores enter the comparison. `logprobs=0`/`prompt_logprobs=0` request
the observed token's score without an expensive full-vocabulary return.
If cached-token metadata is absent, retain that uncertainty and verify the
worker's cache behavior before treating the scoring pass as a fresh prefill.

The rescoring prompt plus its extra generated token must fit the server's
context limit. Keep `len(p) + len(y) + 1` within that limit; do not silently
truncate a long-context case to obtain scores.

For an upstream-sized cohort, use 64 frozen prompts/variants, seeds 0–63 and
384-token budgets. Save each pair independently and cap client concurrency.
Retain early EOS, caps and failures; report actual scored-token counts and
unscorable requests. Never use `zip` to silently discard missing score rows.

## Interpret and localize the difference

Report mean/median/p95/p99/max absolute drift, signed drift, the fraction above
a declared threshold, and a position-binned curve. A logprob difference of 1 is
a factor of `e` in probability, not a one-percentage-point change. Report both
token-weighted summaries and per-prompt summaries so long completions do not
hide the behavior of short ones.

1. Repeat the same profile to establish its own numerical/batching variability.
2. Compare eager and captured execution using identical inputs and shapes.
3. Toggle one feature at a time: graph mode, scheduling, speculative path,
   backend or scale compression. Avoid conflating async, batch size and CSF.
4. Locate the first differing **logit/logprob**, not just the first different
   sampled token. Inspect input IDs, positions, padding and effective metadata.
5. Follow the first affected layer into intermediate values and logical cache
   slots. Ignore stale/recycled physical cache positions that are no longer
   inputs to the model.
6. Reproduce the mechanism in a small regression and show old-fails/new-passes.
   For C128 this is
   `tests/kernels/test_compressor_kv_cache.py::test_c128_skip_is_not_baked_into_cuda_graph`:
   eager no-boundary execution may skip; replay of a captured graph must retain
   the compressor operation. Checking a graph-mode enum is insufficient when
   the graph manager calls the model with mode NONE during actual CUDA capture.

For cross-build comparison, freeze a single prompt-plus-continuation corpus
and score those same IDs on both builds. Independent temperature 1 runs, even
with equal seeds, can follow different histories. Comparing their later token
positions does not isolate numerical drift. A prefill comparison also does not
by itself test both builds' decode paths: that requires forced-history decode
replay, or comparison restricted to a genuinely shared prefix.

If estimating uncertainty, use per-prompt paired summaries on a shared corpus
and repeat across recorded boots/blocks. Tokens within one prompt are correlated;
shared serving batches also create dependence. Do not treat thousands of token
rows as independent samples or turn a single-arm mean into a significance claim.
Calibrate an equivalence margin against matched controls; no detected difference
does not prove equivalence.

## Evidence and release gates

Keep source/native/config receipts, prompt and continuation IDs, seeds, actual
cache-hit metadata, both raw HTTP responses, aligned per-token scores, first
difference, position bins, errors and exact commands. Preserve raw benchmark
output alongside any derived tables. Our existing tracing examples are
`/home/ubuntu/src/vllm/bench/experiments/ds4-async-causal-20261014/trace_hooks.py` (local archive),
`/home/ubuntu/src/vllm/bench/experiments/ds4-async-causal-20261014/cache_hooks.py` (local archive)
and `/home/ubuntu/src/vllm/bench/experiments/ds4-async-causal-20261014/eager_hooks.py` (local archive).
They are investigation artifacts tied to their recorded checkout, not a generic
drop-in interface for every future version.

A drift diagnostic complements final-answer quality, all-attempt token cost,
long-context and performance checks. It does not replace them. In particular,
short-answer accuracy can stay flat while cache math is wrong, and agreement
between two paths can mean they share the same defect.

For the C128 fix actually activated on epyc, see the
`/home/ubuntu/src/vllm/bench/experiments/ds4-main-c128-production-20261015/report.md` (local archive)
and `/home/ubuntu/src/vllm/bench/experiments/ds4-main-c128-production-20261015/final-live-verification.json` (local archive).
Those gates included the graph regression, a 2048-token corrected reference
match, quick hard8x1 and uncached 1.03M recall. They did not execute the upstream
64-prompt decode-versus-prefill experiment.
