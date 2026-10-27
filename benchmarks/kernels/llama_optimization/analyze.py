# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize saved Llama operator and serving measurements without a GPU."""

import hashlib
import json
import math
import statistics
from pathlib import Path


def analyze_compression():
    path = Path(__file__).with_name("compressible_memory_results.json")
    data = json.loads(path.read_text())
    gate = data["frozen_protocol"]["performance_margin"]
    probe = data["allocation_probe"]
    assert probe["stage1_pass"] and len(probe["attempts"]) == 8
    for attempt in probe["attempts"]:
        assert attempt["mapped_readwrite"]
        assert attempt["granted_compression"] == attempt["requested_compression"]
    print("Transparent CUDA memory compression (operator screen, not model TPS)")
    for run in data["timing_attempts"]:
        assert run["exit_code"] == 0 and len(run["cells"]) == 18
        for cell in run["cells"]:
            assert cell["eager_value_equal"] and cell["graph_replay_value_equal"]
            assert cell["weight_bytes"] == cell["N"] * cell["K"]
            for name, windows in cell["arms"].items():
                assert len(windows) == run["windows"] == 6
                for window in windows:
                    samples = window["samples_ms"]
                    assert len(samples) == run["iterations"] == 20
                    assert all(math.isfinite(v) and v > 0 for v in samples)
                    assert window["median_ms"] == statistics.median(samples)
                assert cell["median_ms"][name] == statistics.median(
                    window["median_ms"] for window in windows
                )
                allocation = cell["allocations"][name]
                assert allocation["granted_compression"] == int(name == "generic")
            assert math.isclose(
                cell["speedup"],
                cell["median_ms"]["none"] / cell["median_ms"]["generic"],
            )
        gains = {}
        for m in (1, 8, 32, 128):
            cells = [
                c for c in run["cells"] if c["pattern"] == "actual" and c["M"] == m
            ]
            assert len(cells) == 4
            ordinary = sum(c["median_ms"]["none"] for c in cells)
            compressed = sum(c["median_ms"]["generic"] for c in cells)
            gains[m] = 100 * (ordinary / compressed - 1)
            saved = run["summed_gemm_proxy"][str(m)]
            assert math.isclose(gains[m], saved["rate_change_percent"], abs_tol=1e-10)
        limit = gate["per_m8_cell_and_m32_m128_aggregate_regression_max_percent"]
        per_cell_pass = all(
            cell["speedup"] >= 1 - limit / 100
            for cell in run["cells"]
            if cell["pattern"] == "actual" and cell["M"] == 8
        )
        passes = (
            gains[8] >= gate["m8_summed_gemm_gain_min_percent"]
            and gains[32] >= -limit
            and gains[128] >= -limit
            and per_cell_pass
        )
        assert passes == run["operator_gate_passed"] and not passes
        print(f"  {run['id']}: summed GEMM rate changes {gains}; held")
    correctness = data["final_byte_correctness"]
    assert correctness["exit_code"] == 0 and correctness["correctness_only"]
    assert len(correctness["cells"]) == 18
    assert correctness["build"]["head"] == data["timing_attempts"][-1]["build"]["head"]
    assert not correctness["build"]["diff"]
    assert (
        correctness["build"]["native_sha256"]
        == data["timing_attempts"][-1]["build"]["native_sha256"]
    )
    for timed, checked in zip(
        data["timing_attempts"][-1]["cells"], correctness["cells"]
    ):
        for key in (
            "shape",
            "pattern",
            "M",
            "weight_sha256",
            "input_sha256",
            "weight_scale_sha256",
        ):
            assert timed[key] == checked[key]
        assert checked["eager_bit_exact"] and checked["graph_replay_bit_exact"]
    for name, digest in correctness["build"]["benchmark_sha256"].items():
        assert hashlib.sha256(path.with_name(name).read_bytes()).hexdigest() == digest
    assert data["capacity_setup_failure"]["exit_code"] == 1
    assert not data["capacity_setup_failure"]["rows"]
    capacity = data["capacity"]
    assert capacity["exit_code"] == 0 and len(capacity["rows"]) == 12
    for row in capacity["rows"]:
        assert row["bytes_exact"] and row["cuda_release_delta"] == 0
    for pair in data["capacity_savings"]:
        rows = {
            row["compressed"]: row
            for row in capacity["rows"]
            if (row["pattern"], row["repeat"]) == (pair["pattern"], pair["repeat"])
        }
        assert set(rows) == {False, True}
        for counter in ("cuda", "nvml"):
            saving = rows[False][f"{counter}_charge"] - rows[True][f"{counter}_charge"]
            assert saving == pair[f"{counter}_bytes_saved"] == 0
    print("  18 eager/graph raw-bit comparisons pass; 0 reusable VRAM saved")


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
    analyze_compression()


if __name__ == "__main__":
    main()
