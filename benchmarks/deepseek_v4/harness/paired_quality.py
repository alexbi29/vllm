# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-suite inference from recorded, unique matched seeds and fixed blocks.

BCa intervals and studentized centered bootstrap test the weak mean-cost null.
Resampling respects each question/block and preserves each seed's two outcomes.
"""

import numpy as np
from harness.quality_compare import bootstrap_ci, compatible, tails


def align_seed_pairs(a, b):
    compatible(a, b)
    if (a["temperature"] or 0) <= 0 or (b["temperature"] or 0) <= 0:
        raise ValueError(
            "Matched-seed inference requires stochastic positive-temperature requests"
        )
    strata = []
    names = sorted(a["questions"])
    for name in names:
        qa, qb = a["questions"][name], b["questions"][name]
        sa, sb = qa.get("seeds"), qb.get("seeds")
        if (
            not sa
            or not sb
            or any(type(s) is not int for s in sa + sb)
            or len(set(sa)) != len(sa)
            or len(set(sb)) != len(sb)
            or set(sa) != set(sb)
        ):
            raise ValueError(
                f"{name}: need identical unique recorded seeds in both arms"
            )
        ba, bb = qa.get("blocks", [0] * len(sa)), qb.get("blocks", [0] * len(sb))
        ma = {s: (t, block) for s, t, block in zip(sa, qa["tokens"], ba)}
        mb = {s: (t, block) for s, t, block in zip(sb, qb["tokens"], bb)}
        for s in sa:
            if ma[s][1] != mb[s][1]:
                raise ValueError("Matched seeds have different experiment blocks")
        for block in sorted(set(ba)):
            seeds = sorted(s for s in sa if ma[s][1] == block)
            if len(seeds) < 2:
                raise ValueError("Need at least two seed pairs in every question/block")
            strata.append(
                dict(
                    question=name,
                    block=block,
                    seeds=seeds,
                    pair=np.asarray(
                        [[ma[s][0] for s in seeds], [mb[s][0] for s in seeds]]
                    ),
                    weight=len(seeds) / len(sa) / len(names),
                )
            )
    return strata


def paired_mean_test(strata, resamples, rng, candidate_multiplier=1.0):
    differences = [candidate_multiplier * s["pair"][1] - s["pair"][0] for s in strata]
    weights = np.asarray([s["weight"] for s in strata])
    delta = float(sum(w * d.mean() for w, d in zip(weights, differences)))
    variance = sum(w * w * d.var(ddof=1) / len(d) for w, d in zip(weights, differences))
    if variance <= 0:
        return dict(
            p_candidate_higher=None,
            p_candidate_lower=None,
            p_two_sided=None,
            reason="No observed paired variance; cannot studentize",
            exact=False,
        )
    observed_t = delta / np.sqrt(variance)
    null = np.empty(resamples)
    degenerate = 0
    for start in range(0, resamples, 512):
        count = min(512, resamples - start)
        means = np.zeros(count)
        variances = np.zeros(count)
        for w, d in zip(weights, differences):
            samples = d[rng.integers(len(d), size=(count, len(d)))]
            means += w * samples.mean(axis=1)
            variances += w * w * samples.var(axis=1, ddof=1) / len(d)
        centered = means - delta
        valid = variances > 0
        values = np.zeros(count)
        values[valid] = centered[valid] / np.sqrt(variances[valid])
        values[~valid & (centered > 0)] = np.inf
        values[~valid & (centered < 0)] = -np.inf
        degenerate += int(np.count_nonzero(~valid))
        null[start : start + count] = values
    if degenerate > 0.01 * resamples:
        return dict(
            p_candidate_higher=None,
            p_candidate_lower=None,
            p_two_sided=None,
            reason="More than1percent bootstrap draws have zero estimated "
            "variance; inference unavailable",
            exact=False,
            degenerate_bootstraps=degenerate,
        )
    return dict(
        **tails(null, float(observed_t), False),
        observed_t=float(observed_t),
        paired_standard_error=float(np.sqrt(variance)),
        degenerate_bootstraps=degenerate,
        method="Studentized centered stratified paired bootstrap; "
        "asymptotic weak mean null",
    )


def compare_paired(a, b, confidence=0.95, resamples=49999, seed=20261013):
    strata = align_seed_pairs(a, b)
    weights = np.asarray([s["weight"] for s in strata])

    def statistic(*samples, axis=-1):
        # Each sample has two rows; resampling its last axis keeps seed outcomes paired.
        means = sum(w * x.mean(axis=axis) for w, x in zip(weights, samples))
        return np.stack([means[1] - means[0], 100 * (means[1] / means[0] - 1)])

    rng = np.random.default_rng(seed)
    ci = bootstrap_ci(
        [s["pair"] for s in strata], statistic, False, confidence, resamples, rng
    )
    per_block = []
    for block in sorted({s["block"] for s in strata}):
        selected = [s for s in strata if s["block"] == block]
        per_block.append(
            dict(
                block=block,
                baseline_mean=float(np.mean([s["pair"][0].mean() for s in selected])),
                candidate_mean=float(np.mean([s["pair"][1].mean() for s in selected])),
            )
        )
    return dict(
        baseline=a["label"],
        candidate=b["label"],
        concurrency=a["concurrency"],
        baseline_mean=a["mean"],
        candidate_mean=b["mean"],
        delta=b["mean"] - a["mean"],
        percent_change=100 * (b["mean"] / a["mean"] - 1),
        delta_ci=ci[0],
        percent_ci=ci[1],
        pair_count=sum(len(s["seeds"]) for s in strata),
        stratum_count=len(strata),
        per_block=per_block,
        scope=f"Mean cost over {len(a['questions'])} fixed questions and "
        f"{len(per_block)} recorded block labels, conditional on these runs. "
        "Assumes independent random seed pairs within question/block, not independent "
        "models within a seed. Shared serving-batch dependence is not modeled. "
        "Block labels do not establish counterbalancing or "
        "independent boot replication. "
        "Does not estimate a population of boots or unseen questions.",
        **paired_mean_test(strata, resamples, rng),
    )
