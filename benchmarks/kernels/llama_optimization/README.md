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
For new experiments, use the existing `vllm bench` and
`benchmarks/kernels/benchmark_fp8_gemm.py` tools, preserving the workload's
actual precision and scale scheme. This record adds no competing GPU harness.
