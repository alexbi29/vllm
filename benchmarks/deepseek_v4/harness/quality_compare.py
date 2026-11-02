# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline all-attempt quality cost comparisons with explicit inference units."""

import argparse
import hashlib
import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ORDER = [
    "code-1",
    "euler-totient",
    "fence-post",
    "mod-arith",
    "code-circular",
    "code-levenshtein",
    "code-gcdrec",
    "train-accel",
]


def load_run(label, path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    artifact = json.loads(raw)
    data = artifact.get("data", artifact)
    request = data.get("request_args") or {}
    template = request.get("chat_template_kwargs") or {}
    enabled = template.get("thinking", False) or template.get("enable_thinking", False)
    effort = template.get(
        "reasoning_effort",
        request.get("reasoning_effort", "high" if enabled else "none"),
    )
    if request.get("reasoning_effort", effort) != effort:
        raise ValueError(f"{label}: inconsistent reasoning directives")
    questions = {}
    for q in data["results"]:
        if q["id"] in questions or not q["attempts"]:
            raise ValueError(f"{label}: duplicate or empty question {q['id']}")
        tokens = np.asarray([a["tokens"] for a in q["attempts"]], dtype=float)
        if not np.all(np.isfinite(tokens)) or np.any(tokens < 0):
            raise ValueError(f"{label}: missing/invalid token cost")
        attempts = q["attempts"]
        questions[q["id"]] = dict(
            tokens=tokens,
            mean=float(tokens.mean()),
            n=len(tokens),
            seeds=[a.get("seed") for a in attempts],
            blocks=[a.get("experiment_block", 0) for a in attempts],
            total_tokens=float(tokens.sum()),
            usable=sum(
                bool(a.get("correct"))
                and a.get("finish") == "stop"
                and bool((a.get("response_content") or "").strip())
                for a in attempts
            ),
            caps=sum(a.get("finish") == "length" for a in attempts),
            errors=sum(a.get("finish") in ("error", "timeout") for a in attempts),
        )
    if not questions:
        raise ValueError(f"{label}: no questions")
    return dict(
        label=label,
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        tag=data.get("tag"),
        date=data.get("date"),
        model_root=data.get("model_root"),
        concurrency=data.get("concurrency"),
        effort=effort,
        temperature=data.get("temperature"),
        top_p=request.get("top_p", 1),
        max_tokens=data.get("max_tokens"),
        request_args=request,
        # Repeated or incomplete recorded seeds cannot establish independent draws.
        independent_draws=request.get("seed") is None
        and (data.get("temperature") or 0) > 0
        and all(
            all(s is None for s in q["seeds"])
            or (
                all(type(s) is int for s in q["seeds"])
                and len(set(q["seeds"])) == q["n"]
            )
            for q in questions.values()
        ),
        questions=questions,
        mean=float(np.mean([q["mean"] for q in questions.values()])),
        attempts=sum(q["n"] for q in questions.values()),
        total_tokens=sum(q["total_tokens"] for q in questions.values()),
    )


def compatible(a, b):
    for key in ("concurrency", "effort", "temperature", "top_p", "max_tokens"):
        if a[key] != b[key]:
            raise ValueError(f"{a['label']}/{b['label']}: mismatched {key}")
    if set(a["questions"]) != set(b["questions"]):
        raise ValueError("Question sets differ; refusing silent intersection")


def holm(pvalues):
    """Holm step-down family-wise correction, preserving input order."""
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        result[index] = min(1.0, running)
    return result.tolist()


def tails(null, observed, exact):
    tolerance = 100 * np.finfo(float).eps * max(1.0, abs(observed))
    offset = 0 if exact else 1
    lower = (np.count_nonzero(null <= observed + tolerance) + offset) / (
        len(null) + offset
    )
    higher = (np.count_nonzero(null >= observed - tolerance) + offset) / (
        len(null) + offset
    )
    return dict(
        p_candidate_lower=float(lower),
        p_candidate_higher=float(higher),
        p_two_sided=float(min(1, 2 * min(lower, higher))),
        exact=exact,
        permutations=len(null),
    )


def stratified_permutation(pairs, resamples, rng):
    """Swap labels only within questions; never pool heterogeneous questions."""
    observed = float(np.mean([b.mean() - a.mean() for a, b in pairs]))
    partitions = [math.comb(len(a) + len(b), len(a)) for a, b in pairs]
    if math.prod(partitions) <= min(resamples, 1_000_000):
        distributions = []
        for a, b in pairs:
            pool = np.concatenate([a, b])
            distributions.append(
                [
                    (pool.sum() - pool[list(indices)].sum()) / len(b)
                    - pool[list(indices)].mean()
                    for indices in itertools.combinations(range(len(pool)), len(a))
                ]
            )
        null = np.asarray([np.mean(d) for d in itertools.product(*distributions)])
        return tails(null, observed, True)
    null = np.zeros(resamples)
    for a, b in pairs:
        pool = np.concatenate([a, b])
        for start in range(0, resamples, 512):
            count = min(512, resamples - start)
            shuffled = rng.permuted(np.broadcast_to(pool, (count, len(pool))), axis=1)
            null[start : start + count] += (
                shuffled[:, len(a) :].mean(axis=1) - shuffled[:, : len(a)].mean(axis=1)
            ) / len(pairs)
    return tails(null, observed, False)


def bootstrap_ci(samples, statistic, paired, confidence, resamples, rng):
    """BCa intervals; explicitly label degenerate empirical fallback."""
    with warnings.catch_warnings(record=True):
        result = stats.bootstrap(
            tuple(samples),
            statistic,
            paired=paired,
            vectorized=True,
            confidence_level=confidence,
            n_resamples=resamples,
            batch=256,
            method="BCa",
            rng=rng,
            axis=-1,
        )
    low = np.atleast_1d(result.confidence_interval.low)
    high = np.atleast_1d(result.confidence_interval.high)
    intervals = []
    for i, (lo, hi) in enumerate(zip(low, high)):
        method = "BCa"
        if not np.isfinite(lo) or not np.isfinite(hi):
            distribution = result.bootstrap_distribution
            if distribution.ndim > 1:
                distribution = distribution[i]
            lo, hi = np.quantile(
                distribution, [(1 - confidence) / 2, (1 + confidence) / 2]
            )
            method = "percentile (BCa degenerate; empirical variability only)"
        if not np.isfinite(lo) or not np.isfinite(hi):
            intervals.append(
                dict(
                    low=None,
                    high=None,
                    confidence=confidence,
                    method="undefined: relative statistic has zero-cost bootstrap "
                    "denominator",
                )
            )
        else:
            intervals.append(
                dict(
                    low=float(lo), high=float(hi), confidence=confidence, method=method
                )
            )
    return intervals


def contrast_statistic(*arrays, axis=-1):
    half = len(arrays) // 2
    a = np.mean([np.mean(x, axis=axis) for x in arrays[:half]], axis=0)
    b = np.mean([np.mean(x, axis=axis) for x in arrays[half:]], axis=0)
    # Outputs: absolute delta, candidate/baseline percentage delta.
    return np.stack([b - a, 100 * (b / a - 1)])


def compare(a, b, unit="auto", confidence=0.95, resamples=19999, seed=0):
    compatible(a, b)
    independent = a["independent_draws"] and b["independent_draws"]
    if unit == "auto":
        unit = "attempts" if independent else "questions"
        if all(
            a["questions"][q]["seeds"]
            and all(
                type(s) is int
                for s in a["questions"][q]["seeds"] + b["questions"][q]["seeds"]
            )
            and len(set(a["questions"][q]["seeds"])) == a["questions"][q]["n"]
            and set(a["questions"][q]["seeds"]) == set(b["questions"][q]["seeds"])
            for q in a["questions"]
        ):
            unit = "paired-attempts"
    if unit == "paired-attempts":
        from harness.paired_quality import compare_paired

        result = compare_paired(a, b, confidence, resamples, seed)
        result.update(unit=unit, unit_count=result["pair_count"], per_question=[])
        return result
    if unit == "attempts" and not independent:
        raise ValueError(
            "Attempt inference requires unseeded stochastic draws; "
            "repeated fixed seeds are not independent"
        )
    names = sorted(a["questions"])
    pairs = [(a["questions"][q]["tokens"], b["questions"][q]["tokens"]) for q in names]
    rng = np.random.default_rng(seed)
    details = []
    if a["mean"] <= 0:
        raise ValueError("A positive baseline cost is required for relative intervals")
    if unit == "questions":
        ma = np.asarray([x.mean() for x, _ in pairs])
        mb = np.asarray([y.mean() for _, y in pairs])
        if len(names) < 2:
            raise ValueError(
                "Question-level inference needs at least two question units"
            )
        ci = bootstrap_ci(
            [ma, mb], contrast_statistic, True, confidence, resamples, rng
        )
        # Exact paired label-swap over8 question blocks =256 assignments.
        if len(names) <= 20:
            null = np.asarray(
                [
                    np.mean(np.asarray(signs) * (mb - ma))
                    for signs in itertools.product((-1, 1), repeat=len(names))
                ]
            )
            p = tails(null, float((mb - ma).mean()), True)
        else:
            null = (rng.choice([-1, 1], (resamples, len(names))) * (mb - ma)).mean(
                axis=1
            )
            p = tails(null, float((mb - ma).mean()), False)
        scope = (
            "Question-mix uncertainty over paired question means; "
            "assumes independent exchangeable "
            "question units and under H0 within-question method-label exchangeability. "
            "Not within-question sampling uncertainty on this fixed "
            "suite. No per-question p-values."
        )
    else:
        samples = [x for x, _ in pairs] + [y for _, y in pairs]
        if any(len(s) < 2 for s in samples):
            raise ValueError(
                "Attempt inference needs at least two draws per arm/question"
            )
        ci = bootstrap_ci(
            samples, contrast_statistic, False, confidence, resamples, rng
        )
        p = stratified_permutation(pairs, resamples, rng)
        scope = (
            "Fixed-suite within-run sampling uncertainty; assumes "
            "independent stochastic requests "
            "within each arm/question and common within-question "
            "distributions under H0. "
            "No arbitrary pairing by attempt index; not between-boot "
            "or causal uncertainty."
        )
        for q, (x, y) in zip(names, pairs):
            qci = bootstrap_ci(
                [x, y], contrast_statistic, False, confidence, resamples, rng
            )
            details.append(
                dict(
                    question=q,
                    baseline_mean=float(x.mean()),
                    candidate_mean=float(y.mean()),
                    delta=float(y.mean() - x.mean()),
                    delta_ci=qci[0],
                    percent_ci=qci[1],
                    **stratified_permutation([(x, y)], resamples, rng),
                )
            )
    return dict(
        baseline=a["label"],
        candidate=b["label"],
        concurrency=a["concurrency"],
        unit=unit,
        unit_count=len(names) if unit == "questions" else [len(x) for x, _ in pairs],
        scope=scope,
        delta=b["mean"] - a["mean"],
        percent_change=100 * (b["mean"] / a["mean"] - 1),
        delta_ci=ci[0],
        percent_ci=ci[1],
        per_question=details,
        **p,
    )


def run_mean_ci(run, unit, confidence, resamples, rng):
    samples = [q["tokens"] for q in run["questions"].values()]
    if unit == "questions":
        samples = [np.asarray([s.mean() for s in samples])]
    elif unit == "paired-attempts":
        samples, weights = [], []
        for q in run["questions"].values():
            blocks = np.asarray(q["blocks"])
            for block in sorted(set(q["blocks"])):
                sample = q["tokens"][blocks == block]
                samples.append(sample)
                weights.append(len(sample) / q["n"] / len(run["questions"]))
        return bootstrap_ci(
            samples,
            lambda *xs, axis=-1: sum(
                w * np.mean(x, axis=axis) for w, x in zip(weights, xs)
            ),
            False,
            confidence,
            resamples,
            rng,
        )[0]
    return bootstrap_ci(
        samples,
        lambda *xs, axis=-1: np.mean([np.mean(x, axis=axis) for x in xs], axis=0),
        False,
        confidence,
        resamples,
        rng,
    )[0]


def serializable_run(run):
    return {
        **run,
        "questions": {
            name: {**q, "tokens": q["tokens"].tolist()}
            for name, q in run["questions"].items()
        },
    }


def render(result):
    runs = result["runs"]
    lines = [
        "# All-attempt token-cost comparison",
        "",
        "All completion tokens, including reasoning, caps and "
        "incorrect answers. Questions receive equal weight. "
        "No generation, grader, GPU or service calls. CIs are "
        "pointwise; Holm-adjusted two-sided p-values "
        "control all reported contrasts in this invocation. A "
        "p-value is not the probability a model is better.",
        "",
        "| Run | All tok/q | 95% CI | CI scope | Usable | Caps | Errors |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    level = 100 * result["confidence"]
    lines[4] = lines[4].replace("95%", f"{level:g}%")
    for r in runs:
        ci = r["mean_ci"]
        lines.append(
            f"| {r['label']} | {r['mean']:.1f} | [{ci['low']:.1f}, {ci['high']:.1f}] | "
            f"{r['ci_unit']} | "
            f"{sum(q['usable'] for q in r['questions'].values())}/{r['attempts']} | "
            f"{sum(q['caps'] for q in r['questions'].values())} | "
            f"{sum(q['errors'] for q in r['questions'].values())} |"
        )
    names = sorted(
        runs[0]["questions"],
        key=lambda q: (ORDER.index(q) if q in ORDER else len(ORDER), q),
    )
    lines += [
        "",
        "All-attempt tok/q by question:",
        "",
        "| Question | " + " | ".join(r["label"] for r in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for q in names:
        lines.append(
            "| "
            + q
            + " | "
            + " | ".join(f"{r['questions'][q]['mean']:.1f}" for r in runs)
            + " |"
        )
    lines += [
        "",
        "Candidate minus baseline: negative means cheaper. "
        "Attempt/question label-swap tests require "
        "distributional exchangeability. Matched-seed "
        "paired-attempt tests use a centered studentized "
        "bootstrap of the weak mean-cost null and are "
        "asymptotic. Serving-profile differences are confounded; "
        ""
        "p-values do not isolate checkpoint/backend causation.",
        "",
        "| Baseline → candidate | Unit | Δ tok/q | CI for Δ | "
        "Change | p(two-sided) | p(Holm) |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for c in result["comparisons"]:
        ci = c["delta_ci"]
        lines.append(
            f"| {c['baseline']} → {c['candidate']} | {c['unit']} | {c['delta']:.1f} | "
            f"[{ci['low']:.1f}, {ci['high']:.1f}] | {c['percent_change']:+.1f}% | "
            f"{format_p(c['p_two_sided'])} | {format_p(c.get('p_two_sided_holm'))} |"
        )
    lines += ["", "Inference assumptions:", ""]
    for unit in sorted({c["unit"] for c in result["comparisons"]}):
        scope = next(c["scope"] for c in result["comparisons"] if c["unit"] == unit)
        lines.append(f"- **{unit}:** {scope}")
    lines += [
        "",
        "A shared fixed seed switches auto inference to question "
        "units; repeated runs with "
        "that seed are not counted as independent sampled seeds. "
        "Only the selected questions "
        "provide question units. Question-mix CIs "
        "cannot be read as uncertainty for these exact fixed "
        "questions. Per-question CIs/p-values "
        "are in JSON only for attempt-level comparisons; "
        "per-question uncertainty is unavailable "
        "for the fixed-seed comparisons. All intervals are "
        "conditional on their stated assumptions.",
        "",
        "Holm correction covers this declared family only, not "
        "earlier exploratory selection or "
        "sequential peeking across the campaign. These "
        "conditional intervals do not estimate "
        "variation across unseen boots. A cheaper failing model is not better; "
        "usable-answer counts and caps are reported separately.",
        "",
        "Token costs use recorded completion usage. An "
        "error/timeout can have unreported "
        "server work; zero recorded usage does not establish zero GPU cost.",
        "",
        "[SciPy bootstrap "
        "method](https://docs.scipy.org/doc/scipy/reference/gene"
        "rated/scipy.stats.bootstrap.html), "
        "[permutation "
        "definitions](https://docs.scipy.org/doc/scipy/reference"
        "/generated/scipy.stats.permutation_test.html).",
    ]
    return "\n".join(lines) + "\n"


def format_p(value):
    return "unavailable" if value is None else f"{value:.4g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument(
        "--unit",
        choices=["auto", "attempts", "questions", "paired-attempts"],
        default="auto",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=19999)
    parser.add_argument(
        "--seed", type=int, default=20261013, help="Offline statistical RNG only"
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSON and adjacent Markdown report"
    )
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or args.resamples < 999:
        parser.error("Use confidence in(0,1) and at least999 resamples")
    try:
        runs = [load_run(*spec.split("=", 1)) for spec in args.run]
        if len(runs) < 2 or len({r["label"] for r in runs}) != len(runs):
            raise ValueError("Need at least two uniquely labeled runs")
        if any(set(r["questions"]) != set(runs[0]["questions"]) for r in runs):
            raise ValueError("Question sets differ")
        comparisons = []
        for index, (a, b) in enumerate(itertools.combinations(runs, 2)):
            if a["concurrency"] != b["concurrency"]:
                continue
            comparisons.append(
                compare(
                    a, b, args.unit, args.confidence, args.resamples, args.seed + index
                )
            )
        if not comparisons:
            raise ValueError("No equal-concurrency pairs to compare")
        tests = [
            item
            for c in comparisons
            for item in [c, *c["per_question"]]
            if item["p_two_sided"] is not None
        ]
        for item, p in zip(tests, holm([item["p_two_sided"] for item in tests])):
            item["p_two_sided_holm"] = p
            item["significant_after_holm"] = p < 1 - args.confidence
        for index, run in enumerate(runs):
            run["ci_unit"] = (
                args.unit
                if args.unit != "auto"
                else ("attempts" if run["independent_draws"] else "questions")
            )
            participating = [
                c
                for c in comparisons
                if run["label"] in (c["baseline"], c["candidate"])
            ]
            if (
                args.unit == "auto"
                and participating
                and all(c["unit"] == "paired-attempts" for c in participating)
            ):
                run["ci_unit"] = "paired-attempts"
            if run["ci_unit"] == "attempts" and not run["independent_draws"]:
                raise ValueError("Fixed-seed run cannot use attempt inference")
            run["mean_ci"] = run_mean_ci(
                run,
                run["ci_unit"],
                args.confidence,
                args.resamples,
                np.random.default_rng(args.seed + 1000 + index),
            )
        result = dict(
            metric="equal-question mean all-attempt completion tok/q",
            confidence=args.confidence,
            resamples=args.resamples,
            statistical_rng_seed=args.seed,
            multiple_testing=dict(
                method="Holm two-sided",
                family_size=len(tests),
                includes="All overall and available per-question contrasts in "
                "this invocation",
            ),
            runs=[serializable_run(r) for r in runs],
            comparisons=comparisons,
        )
        text = render(result)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            args.output.with_suffix(".md").write_text(text)
        print(text, end="")
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
