# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare identical FP8 GEMMs with ordinary and compressible VMM weights."""

import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
import urllib.request
import warnings
from pathlib import Path

from cuda_vmm import VMMBuffer

SHAPES = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "down": ("mlp.down_proj",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="dense-fp8-llama")
    parser.add_argument("--service-model")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--cublaslt-m-range", default="64:256")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists(), "Retain previous attempts"

    import ctypes as c

    import numpy as np
    import pynvml as nv
    import torch
    from compression_benchmark_kernels import copy_bytes, evict_cache
    from safetensors import safe_open

    import vllm
    from vllm import _custom_ops as ops
    from vllm.triton_utils import triton

    nv.nvmlInit()
    board = nv.nvmlDeviceGetHandleByIndex(0)
    started = time.monotonic()
    state = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "label": args.label,
        "cells": [],
        "windows": args.windows,
        "iterations": args.iterations,
        "cublaslt_m_range": args.cublaslt_m_range,
        "physical_dram_counters": "not collected; logical IO rates only",
        "correctness_only": args.correctness_only,
        "output_comparison": "raw BF16 storage bytes, including signed zero",
    }

    def save():
        temp = args.output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)

    def meter():
        return {
            "board_joules": nv.nvmlDeviceGetTotalEnergyConsumption(board) / 1000,
            "gpu_free_bytes": nv.nvmlDeviceGetMemoryInfo(board).free,
            "sm_mhz": nv.nvmlDeviceGetClockInfo(board, nv.NVML_CLOCK_SM),
            "temperature_c": nv.nvmlDeviceGetTemperature(
                board, nv.NVML_TEMPERATURE_GPU
            ),
            "loadavg": Path("/proc/loadavg").read_text().strip(),
        }

    def guard():
        assert time.monotonic() - started < 900, "Operator budget exhausted"
        assert nv.nvmlDeviceGetPowerManagementLimit(board) == 450000
        assert nv.nvmlDeviceGetMemoryInfo(board).free >= 1024**3
        available = next(
            int(line.split()[1]) << 10
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        assert available >= 16 * 1024**3
        if args.service_model:
            url = f"{args.url}/upstream/{args.service_model}/metrics"
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode()
            for line in text.splitlines():
                if line.startswith(
                    ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
                ):
                    assert float(line.rsplit(" ", 1)[1]) == 0, line

    guard()
    assert nv.nvmlDeviceGetMemoryInfo(board).free >= 2 * 1024**3
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261026)
    torch.cuda.init()
    device = torch.cuda.get_device_properties(0)
    driver = c.CDLL("libcuda.so.1")
    l2 = c.c_int()
    assert driver.cuDeviceGetAttribute(c.byref(l2), 38, 0) == 0
    state.update(
        device=device.name,
        capability=torch.cuda.get_device_capability(),
        torch_version=torch.__version__,
        l2_bytes=l2.value,
        power_w=450,
        before=meter(),
    )
    source = Path(vllm.__file__).resolve().parent.parent
    library_paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/vllm/" in line and line.split()[-1].endswith(".so")
    }
    state["build"] = {
        "source": str(source),
        "head": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "diff": subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ),
        "vllm_version": vllm.__version__,
        "driver_version": nv.nvmlSystemGetDriverVersion(),
        "native_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in sorted(library_paths)
        },
        "benchmark_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "benchmark_compressible_memory.py",
                "cuda_vmm.py",
                "compression_benchmark_kernels.py",
            )
        },
    }
    use_cupti = importlib.util.find_spec("cupti") is not None
    state["timer"] = (
        "FlashInfer CUPTI graph/cold-L2"
        if use_cupti
        else ("CUDA graph external-event nodes; high-entropy L2 flush outside timing")
    )
    if use_cupti:
        from flashinfer.testing import bench_gpu_time_with_cupti

    flush = torch.empty(
        max(2 * l2.value, 128 * 1024**2) // 4, device="cuda", dtype=torch.int32
    )

    def clear_cache():
        evict_cache[(triton.cdiv(flush.numel(), 1024),)](
            flush, flush.numel(), BLOCK=1024
        )

    lo, hi = map(int, args.cublaslt_m_range.split(":"))
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    rng = np.random.default_rng(20261026)

    def load_weight(parts):
        weights, scales = [], []
        for part in parts:
            name = f"model.layers.0.{part}"
            with safe_open(
                str(args.checkpoint / index[name + ".weight"]),
                framework="pt",
                device="cpu",
            ) as file:
                value = file.get_tensor(name + ".weight")
                if value.dtype != torch.float8_e4m3fn or value.ndim != 2:
                    raise TypeError("Expected FP8 E4M3 matrix weights")
                scale = file.get_tensor(name + ".weight_scale")
                if scale.dtype != torch.float32 or scale.numel() != value.shape[0]:
                    raise TypeError("Expected one FP32 weight scale per output channel")
                weights.append(value.view(torch.uint8))
                scales.append(scale.reshape(-1, 1))
        return torch.cat(weights).contiguous(), torch.cat(scales).contiguous()

    try:
        for shape, parts in SHAPES.items():
            raw, scales = load_weight(parts)
            n, k = raw.shape
            patterns = (
                ("actual", "zero", "high_entropy") if shape == "down" else ("actual",)
            )
            for pattern in patterns:
                guard()
                source = raw
                if pattern == "zero":
                    source = torch.zeros_like(raw)
                elif pattern == "high_entropy":
                    values = rng.integers(0, 254, size=raw.shape, dtype=np.uint8)
                    values += values >= 127  # Exclude the two E4M3 NaN encodings.
                    source = torch.from_numpy(values)
                digest = hashlib.sha256(source.numpy().tobytes()).hexdigest()
                with (
                    VMMBuffer(tuple(raw.shape), False) as ordinary,
                    VMMBuffer(tuple(raw.shape), True) as compressed,
                ):
                    buffers = {"none": ordinary, "generic": compressed}
                    byte_views = {
                        name: torch.as_tensor(buffer, device="cuda")
                        for name, buffer in buffers.items()
                    }
                    for name, view in byte_views.items():
                        assert view.data_ptr() == buffers[name].pointer.value
                    byte_views["none"].copy_(source)
                    count = source.numel()
                    for name in ("none", "generic"):
                        copy_bytes[(triton.cdiv(count, 1024),)](
                            byte_views["none"], byte_views[name], count, BLOCK=1024
                        )
                    torch.accelerator.synchronize()
                    for view in byte_views.values():
                        assert torch.equal(view.cpu(), source)
                    weights = {
                        name: value.view(torch.float8_e4m3fn).t()
                        for name, value in byte_views.items()
                    }
                    scale_b = scales.to(device="cuda")
                    inputs = torch.randn(128, k, dtype=torch.bfloat16, device="cuda")
                    for m in (1, 8, 32, 128) if pattern == "actual" else (8,):
                        guard()
                        a, scale_a = ops.scaled_fp8_quant(
                            inputs[:m].contiguous(), use_per_token_if_dynamic=True
                        )

                        def operation(
                            weight, m=m, a=a, scale_a=scale_a, scale_b=scale_b
                        ):
                            if lo < m <= hi:
                                return torch._scaled_mm(
                                    a,
                                    weight,
                                    scale_a=scale_a,
                                    scale_b=scale_b.view(1, -1),
                                    out_dtype=torch.bfloat16,
                                )
                            return ops.cutlass_scaled_mm(
                                a, weight, scale_a, scale_b, torch.bfloat16
                            )

                        reference = operation(weights["none"])
                        candidate = operation(weights["generic"])
                        assert torch.equal(
                            reference.view(torch.uint8), candidate.view(torch.uint8)
                        )
                        assert bool(reference.isfinite().all())
                        graphs, outputs, events = {}, {}, {}
                        for name, weight in weights.items():
                            for _ in range(3):
                                operation(weight)
                            torch.accelerator.synchronize()
                            start_event = torch.cuda.Event(
                                enable_timing=True, external=True
                            )
                            end_event = torch.cuda.Event(
                                enable_timing=True, external=True
                            )
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                start_event.record()
                                output = operation(weight)
                                end_event.record()
                            graph.replay()
                            torch.accelerator.synchronize()
                            assert torch.equal(
                                reference.view(torch.uint8), output.view(torch.uint8)
                            )
                            assert start_event.elapsed_time(end_event) > 0
                            graphs[name], outputs[name] = graph, output
                            events[name] = start_event, end_event
                        row = dict(
                            shape=shape,
                            pattern=pattern,
                            M=m,
                            N=n,
                            K=k,
                            weight_sha256=digest,
                            weight_bytes=count,
                            input_sha256=hashlib.sha256(
                                a.view(torch.uint8).cpu().numpy().tobytes()
                            ).hexdigest(),
                            weight_scale_sha256=hashlib.sha256(
                                scales.numpy().tobytes()
                            ).hexdigest(),
                            allocations={
                                name: b.metadata for name, b in buffers.items()
                            },
                            eager_bit_exact=True,
                            graph_replay_bit_exact=True,
                            before=meter(),
                            arms={name: [] for name in weights},
                        )
                        if args.correctness_only:
                            state["cells"].append(row)
                            save()
                            print(
                                json.dumps(
                                    {key: row[key] for key in ("shape", "pattern", "M")}
                                ),
                                flush=True,
                            )
                            del (
                                graphs,
                                outputs,
                                events,
                                graph,
                                output,
                                reference,
                                candidate,
                            )
                            continue
                        for window in range(args.windows):
                            order = (
                                ("none", "generic")
                                if window % 2 == 0
                                else ("generic", "none")
                            )
                            for name in order:
                                guard()
                                before_window = meter()
                                if use_cupti:
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings(
                                            "error", message="(?i).*falling back.*"
                                        )
                                        samples = bench_gpu_time_with_cupti(
                                            lambda op=operation, w=weights[name]: op(w),
                                            use_cuda_graph=True,
                                            cold_l2_cache=True,
                                            dry_run_iters=3,
                                            repeat_iters=args.iterations,
                                        )
                                else:
                                    samples = []
                                    for _ in range(args.iterations):
                                        clear_cache()
                                        graphs[name].replay()
                                        events[name][1].synchronize()
                                        samples.append(
                                            events[name][0].elapsed_time(
                                                events[name][1]
                                            )
                                        )
                                row["arms"][name].append(
                                    dict(
                                        samples_ms=list(map(float, samples)),
                                        median_ms=statistics.median(samples),
                                        gross_board_joules=meter()["board_joules"]
                                        - before_window["board_joules"],
                                        energy_scope=(
                                            "Includes cache flush and host/event idle; "
                                            "not steady GEMM or output-token energy"
                                        ),
                                    )
                                )
                        row["after"] = meter()
                        row["median_ms"] = {
                            name: statistics.median(w["median_ms"] for w in windows)
                            for name, windows in row["arms"].items()
                        }
                        row["speedup"] = (
                            row["median_ms"]["none"] / row["median_ms"]["generic"]
                        )
                        logical_bytes = count + m * k + 2 * m * n + 4 * n + 4 * m
                        row["logical_gbps"] = {
                            name: logical_bytes / value / 1e6
                            for name, value in row["median_ms"].items()
                        }
                        state["cells"].append(row)
                        save()
                        print(
                            json.dumps(
                                {
                                    key: row[key]
                                    for key in (
                                        "shape",
                                        "pattern",
                                        "M",
                                        "median_ms",
                                        "speedup",
                                    )
                                }
                            ),
                            flush=True,
                        )
                        del graphs, outputs, events, graph, output, reference, candidate
                    del weights, byte_views, inputs, a, scale_a, scale_b, weight, view
                    gc.collect()
                    torch.accelerator.synchronize()
                torch.accelerator.empty_cache()
            del raw, scales, source
            gc.collect()
        state.update(exit_code=0, after=meter())
    except BaseException as error:
        state.update(exit_code=1, error=repr(error))
        raise
    finally:
        state.update(
            elapsed_seconds=time.monotonic() - started,
            finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        save()


if __name__ == "__main__":
    main()
