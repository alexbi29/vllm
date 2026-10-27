# Llama optimization measurements

Saved results for one dense `LlamaForCausalLM` FP8 workload on SM120.
The [optimization report](../../../docs/configuration/llama_optimization.md)
describes the serving recipe, workload, performance, quality and uncertainty.

`results.json` preserves every timing repetition used in the report:

- `profile`: five CUDA profile breakdowns and original trace hashes.
- `attention`: control, retained FlashInfer/page64, rejected FlashInfer/page16,
  numerical diagnostics and all repeated rates/board J per token.
- `quality`: per-question token costs, all attempt token counts/seeds,
  raw/usable/capped/error counts, bootstrap intervals and permutation results.
- `screened_head`: all 18 synthetic and 18 captured-state operator cases,
  timing windows, candidate counts, selected IDs and numerical diagnostics.
- `GEMM`: all 72 corrected operator cases, their three windows each,
  accuracy diagnostics and conditional decode projections. The invalid first
  attempt is separate and excluded from the analysis.
- `compressible_memory_results.json`: actual VMM compression grants, two
  complete 18-cell FP8 timing screens, raw-bit correctness on matching fixtures,
  and 12 allocator-capacity measurements. The capacity setup failure is separate.

The source build that produced these results is recorded in `build`.
These are saved measurements, not results from building this PR branch.
Checkpoint weights, captured tensors, generated text and private serving
configuration are not distributed with this record.

Recompute the rate ratios, GEMM projections and numerical/count checks on CPU:

```bash
uv run --no-project python benchmarks/kernels/llama_optimization/analyze.py
```

The analyzer does not load vLLM, Torch, a model or a GPU. It uses saved
observations; it does not rerun inference or bootstrap sampling.
For ordinary experiments, use the existing `vllm bench` and
`benchmarks/kernels/benchmark_fp8_gemm.py` tools, preserving the workload's
actual precision and scale scheme.

## Explicit memory-compression operator probe

`benchmark_compressible_memory.py` fills the allocation-property gap in the
ordinary GEMM benchmark. It calls the existing FP8 GEMMs and isolates only VMM
compression. Its helpers initialize exact bytes, condition L2 and own benchmark
allocations; they are not a serving allocator or a new GEMM backend.

Use an installed matching vLLM/Torch CUDA environment on a single physical
SM120 GPU at 450 W, with at least 2 GiB free and no active serving requests:

```bash
MAX_JOBS=2 NVCC_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
    uv run --no-project --active python \
    benchmarks/kernels/llama_optimization/benchmark_compressible_memory.py \
    --checkpoint "$MODEL_PATH" --output /tmp/llama-compression.json \
    --service-model "$MODEL_ID" --url http://127.0.0.1:8000 \
    --cublaslt-m-range 64:256
```

The checkpoint must have standard Llama layer names, FP8 E4M3 matrix weights and
FP32 per-output-channel scales. Use disk-backed compilation caches and bounded
CPU memory. The probe stops on active serving requests, insufficient headroom
or wrong bytes/math. Existing output files are never overwritten.
`--correctness-only` verifies eager/graph raw BF16 bits without timing windows.

The saved timing uses explicit graph event nodes because CUPTI Python support
was absent. Allocation, initialization, warmup and a twice-L2 high-entropy flush
are outside each measured GEMM. Six windows of 20 samples alternate arm order.
Gross board energy includes cache conditioning and host/event idle; it is not
steady GEMM or output-token energy. No physical DRAM counters were collected.

All real-data speed gates failed, despite a 3.13× zero-data positive control.
The separate CUDA/NVML capacity measurements found 0 reusable VRAM saving.
This is a held experiment: no serving feature, default or precision changed.
