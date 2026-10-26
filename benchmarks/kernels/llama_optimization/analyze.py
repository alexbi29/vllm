# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize saved Llama operator and serving measurements without a GPU."""

import json
import math
import statistics
from pathlib import Path


def main():
    data = json.loads(Path(__file__).with_name("results.json").read_text())
    cells = data["attention"]["retained_screen"]["cells"]
    print("FlashInfer/page64 versus FlashAttention2/page16")
    for name, cell in cells.items():
        rates = []
        for side in ("control", "flashinfer64"):
            rate = cell[side]["rate"]
            values = rate["values"]
            assert len(values) == rate["n"] == 3
            assert all(math.isfinite(v) and v > 0 for v in values)
            median = statistics.median(values)
            assert median == rate["median"]
            assert min(values) == rate["minimum"]
            assert max(values) == rate["maximum"]
            rates.append(median)
        change = 100 * (rates[1] / rates[0] - 1)
        assert math.isclose(change, cell["rate_change_percent"], abs_tol=1e-10)
        print(f"  {name}: {rates[0]:.3f} -> {rates[1]:.3f}, {change:+.2f}%")

    for mode, quality in data["quality"].items():
        for run in quality["runs"]:
            assert len(run["questions"]) == 8 and run["attempts"] == 120
            means = []
            for question in run["questions"].values():
                assert len(question["tokens"]) == question["n"] == 15
                means.append(statistics.mean(question["tokens"]))
            assert math.isclose(statistics.mean(means), run["mean"])
        cost = quality["comparisons"][0]
        print(
            f"Quality {mode}: cost {cost['percent_change']:+.2f}%, "
            f"95% CI [{cost['percent_ci']['low']:+.2f}, "
            f"{cost['percent_ci']['high']:+.2f}]"
        )
    assert data["quality"]["on"]["comparisons"][0]["percent_ci"]["high"] > 10

    head = data["screened_head"]
    for family in ("synthetic", "real"):
        assert len(head[family]["cells"]) == 18
        for cell in head[family]["cells"]:
            assert len(cell["repetitions"]) == 3
            assert cell["numerics"]["true_top_k_excluded"] == 0
            assert cell["numerics"]["invalid_scores"] == 0
            ratio = cell["control_median_ms"] / cell["candidate_median_ms"]
            assert math.isclose(ratio, cell["head_speedup"])
    greedy = [c for c in head["real"]["cells"] if c["top_k"] == 1]
    assert any(
        c["projected_whole_decode_gain_percent"] < 1.5
        for c in greedy
        if c["batch"] == 1
    )
    print("Screened head: 36 completed cells; real-input speed gate rejected")

    gemm = data["GEMM"]
    rows = gemm["corrected"]["cells"]
    assert len(rows) == 12 and gemm["corrected"]["exit_code"] == 0
    assert all(len(r["methods"]) == 6 for r in rows)
    for row in rows:
        assert row["methods"]["cutlass_control"]["bit_exact_to_native"]
        for result in row["methods"].values():
            assert result["status"] == "complete"
            assert result["numerical_screen_pass"]
            assert len(result["windows_ms"]) == 3
            assert result["median_ms"] == statistics.median(result["windows_ms"])
    share = gemm["profile_share_for_projection"]
    for projection in gemm["projections"]:
        method = projection["method"]
        gains = []
        for saved in projection["batches"]:
            batch = [r for r in rows if r["M"] == saved["M"]]
            control = sum(r["methods"]["cutlass_control"]["median_ms"] for r in batch)
            candidate = sum(r["methods"][method]["median_ms"] for r in batch)
            gain = 100 * (1 / (1 - share + share * candidate / control) - 1)
            assert math.isclose(
                gain, saved["projected_decode_gain_percent"], abs_tol=1e-10
            )
            gains.append(gain)
        gate = gemm["advancement_gate"]
        clears = (
            method != "cutlass_control"
            and gains[0] >= gate["projected_C1_gain_min_percent"]
            and all(g >= gate["projected_C2_C16_gain_min_percent"] for g in gains[1:])
        )
        assert clears == projection["clears_advancement_gate"]
        assert not clears
        print(f"  {method}: projected C1/C2/C16 {gains}")
    print("72 GEMM cases pass numerics; no alternative clears the +3% C1 gate")


if __name__ == "__main__":
    main()
