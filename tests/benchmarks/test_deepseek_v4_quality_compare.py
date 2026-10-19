# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synthetic offline checks for inferential units, cost accounting and null tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "deepseek_v4")
)
from harness.paired_quality import align_seed_pairs, compare_paired
from harness.quality_compare import (
    bootstrap_ci,
    compare,
    holm,
    load_run,
    stratified_permutation,
    tails,
)


def fixture(values, seed=None):
    return dict(
        concurrency=2,
        temperature=1,
        max_tokens=20000,
        request_args=dict(
            reasoning_effort="high", **({"seed": seed} if seed is not None else {})
        ),
        results=[
            dict(
                id=name,
                attempts=[
                    dict(
                        tokens=t, correct=True, finish="stop", response_content="answer"
                    )
                    for t in tokens
                ],
            )
            for name, tokens in values.items()
        ],
    )


class QualityCompareTests(unittest.TestCase):
    def load(self, values, label="run", seed=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "result.json"
        path.write_text(json.dumps(fixture(values, seed)))
        return load_run(label, path)

    def test_all_attempts_and_equal_question_weights(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        data = fixture({"easy": [10], "hard": [100, 1000, 1000]})
        data["results"][1]["attempts"][1].update(
            correct=False, finish="length", response_content=""
        )
        path = Path(directory.name) / "result.json"
        path.write_text(json.dumps(data))
        run = load_run("cost", path)
        self.assertEqual(run["mean"], 355)  # not pooled2110/4; capped1000included
        self.assertEqual(run["total_tokens"], 2110)
        self.assertEqual(run["questions"]["hard"]["caps"], 1)

    def test_exact_permutation_agrees_with_scipy(self):
        a = np.array([1.0, 2.0])
        b = np.array([8.0, 10.0])
        ours = stratified_permutation([(a, b)], 999, np.random.default_rng(1))
        reference = stats.permutation_test(
            (a, b), lambda x, y: y.mean() - x.mean(), n_resamples=np.inf
        )
        self.assertTrue(ours["exact"])
        self.assertEqual(ours["permutations"], 6)
        self.assertAlmostEqual(ours["p_two_sided"], reference.pvalue)

    def test_never_pool_different_questions(self):
        pairs = [
            (np.array([0.0, 1.0]), np.array([2.0, 3.0])),
            (np.array([100.0, 101.0]), np.array([102.0, 103.0])),
        ]
        result = stratified_permutation(pairs, 999, np.random.default_rng(1))
        self.assertEqual(result["permutations"], 36)  # 6×6, not choose8,4=70
        self.assertAlmostEqual(result["p_candidate_higher"], 1 / 36)

    def test_monte_carlo_p_is_not_zero(self):
        result = tails(np.array([1.0, 2.0, 3.0]), 100.0, False)
        self.assertEqual(result["p_candidate_higher"], 0.25)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_fixed_seed_refuses_attempt_inference(self):
        a = self.load({"q1": [10.0, 11.0], "q2": [100.0, 101.0]}, "A")
        b = self.load({"q1": [12.0, 13.0], "q2": [102.0, 103.0]}, "B", seed=1)
        with self.assertRaisesRegex(ValueError, "fixed seeds"):
            compare(a, b, unit="attempts", resamples=999)
        result = compare(a, b, resamples=999)
        self.assertEqual(result["unit"], "questions")
        self.assertEqual(result["unit_count"], 2)
        self.assertEqual(result["per_question"], [])

    def test_paired_question_exact_direction_and_interval(self):
        a = self.load({f"q{i}": [10.0 + i, 20.0 + i] for i in range(8)}, "A", seed=1)
        b = self.load({f"q{i}": [20.0 + i, 30.0 + i] for i in range(8)}, "B", seed=1)
        result = compare(a, b, resamples=999)
        self.assertEqual(result["permutations"], 256)
        self.assertEqual(result["p_candidate_higher"], 1 / 256)
        self.assertEqual(result["p_two_sided"], 2 / 256)
        self.assertAlmostEqual(result["delta_ci"]["low"], 10)
        self.assertAlmostEqual(result["delta_ci"]["high"], 10)

    def test_mismatched_question_sets_fail(self):
        a = self.load({"q1": [1.0, 2.0]}, "A")
        b = self.load({"q2": [1.0, 2.0]}, "B")
        with self.assertRaisesRegex(ValueError, "Question sets"):
            compare(a, b, resamples=999)

    def test_constant_cost_empirical_interval(self):
        result = bootstrap_ci(
            [np.array([7.0, 7.0, 7.0])],
            lambda x, axis=-1: np.mean(x, axis=axis),
            False,
            0.95,
            999,
            np.random.default_rng(1),
        )[0]
        self.assertEqual((result["low"], result["high"]), (7.0, 7.0))
        self.assertIn("degenerate", result["method"])

    def test_no_arbitrary_trial_pairing_and_seed_reproducibility(self):
        a = self.load({"q1": [1.0, 2.0, 8.0], "q2": [40.0, 60.0, 90.0]}, "A")
        b = self.load({"q1": [2.0, 4.0, 10.0], "q2": [50.0, 70.0, 100.0]}, "B")
        first = compare(a, b, resamples=999, seed=123)
        second = compare(a, b, resamples=999, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["unit"], "attempts")
        self.assertEqual(len(first["per_question"]), 2)
        # Each group assigns6tokens to3: choose6,3 squared, not paired2^6.
        self.assertEqual(first["permutations"], 400)

    def test_invalid_tokens_are_not_imputed_as_zero(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        data = fixture({"q1": [1.0, 2.0]})
        data["results"][0]["attempts"][0]["tokens"] = None
        path = Path(directory.name) / "result.json"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "token cost"):
            load_run("missing", path)


if __name__ == "__main__":
    unittest.main()


class PairedQualityTests(unittest.TestCase):
    def make_run(self, label, costs, seeds, blocks):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "data.json"
        path.write_text(
            json.dumps(
                dict(
                    concurrency=2,
                    temperature=1,
                    max_tokens=20000,
                    request_args=dict(reasoning_effort="high"),
                    results=[
                        dict(
                            id=name,
                            attempts=[
                                dict(
                                    tokens=t,
                                    seed=s,
                                    experiment_block=b,
                                    correct=True,
                                    finish="stop",
                                    response_content="4",
                                )
                                for t, s, b in zip(values, seeds[name], blocks[name])
                            ],
                        )
                        for name, values in costs.items()
                    ],
                )
            )
        )
        return load_run(label, path)

    def test_seed_alignment_and_covariance(self):
        a = self.make_run(
            "A", {"q": [100, 200, 300, 400]}, {"q": [1, 2, 3, 4]}, {"q": [0, 0, 1, 1]}
        )
        b = self.make_run(
            "B", {"q": [410, 310, 210, 110]}, {"q": [4, 3, 2, 1]}, {"q": [1, 1, 0, 0]}
        )
        result = compare_paired(a, b, resamples=999)
        self.assertEqual(result["pair_count"], 4)
        self.assertEqual(result["stratum_count"], 2)
        self.assertAlmostEqual(result["delta_ci"]["low"], 10)
        self.assertAlmostEqual(result["delta_ci"]["high"], 10)
        self.assertIsNone(result["p_two_sided"])

    def test_duplicate_or_mismatched_seed_refused(self):
        a = self.make_run("A", {"q": [1, 2]}, {"q": [1, 2]}, {"q": [0, 0]})
        b = self.make_run("B", {"q": [1, 2]}, {"q": [1, 1]}, {"q": [0, 0]})
        with self.assertRaisesRegex(ValueError, "unique"):
            align_seed_pairs(a, b)

    def test_block_mismatch_refused(self):
        a = self.make_run(
            "A", {"q": [1, 2, 3, 4]}, {"q": [1, 2, 3, 4]}, {"q": [0, 0, 1, 1]}
        )
        b = self.make_run(
            "B", {"q": [1, 2, 3, 4]}, {"q": [1, 2, 3, 4]}, {"q": [0, 1, 0, 1]}
        )
        with self.assertRaisesRegex(ValueError, "different experiment blocks"):
            align_seed_pairs(a, b)

    def test_strong_positive_mean_direction(self):
        seeds = {"q": list(range(16))}
        blocks = {"q": [0] * 8 + [1] * 8}
        a = self.make_run("A", {"q": list(range(100, 260, 10))}, seeds, blocks)
        b = self.make_run("B", {"q": list(range(1100, 1292, 12))}, seeds, blocks)
        result = compare_paired(a, b, resamples=999)
        self.assertLessEqual(result["p_candidate_higher"], 0.01)
        self.assertGreater(result["delta_ci"]["low"], 0)
