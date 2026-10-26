# Llama optimization on SM120

For the dense FP8 Llama workload measured here, FlashInfer with 64-token pages
improved 64K prefill by **9.49%** and lowered GPU-board J/input token by **8.32%**.
Decode changes were small. Screened output-head and alternative FP8 GEMM
operators did not clear their advancement gates, so no new kernel was retained.

This is a record of one workload, not a default for every Llama checkpoint.
The [saved results](../../benchmarks/kernels/llama_optimization/README.md)
include every timing repetition, per-question quality costs and rejected cases.

## Workload and retained recipe

- RTX PRO 6000 Blackwell Workstation Edition, 96 GB, SM120, fixed/enforced 450 W.
- Dense `LlamaForCausalLM`: 52 layers, hidden 7168, SwiGLU intermediate 22016,
  56 query heads / 8 KV heads, head dimension 128, vocabulary 166912.
- Per-channel FP8 weights, dynamic per-token FP8 activations, BF16 residuals/KV;
  the target output head remains BF16. No speculative decoding.
- TP1, 131072 context, 16 slots, 8192 batched tokens, .96 GPU budget,
  async scheduling and prefix caching.
- Measured source build: `834d22133e9a64d04db4f6f818bd91467786557a`, based on
  upstream `4bceda5dbf265e5d20f171c9c339ffe6997b2fd0`; Torch 2.14,
  FlashInfer 0.7.2. This PR records that build's results without changing it.

The only retained serving change from FlashAttention2 with 16-token pages was:

```bash
vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.96 \
    --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --enable-prefix-caching --async-scheduling \
    --attention-backend FLASHINFER --block-size 64
```

The checkpoint supplies the compressed-tensors FP8 quantization configuration.
Keep checkpoint-specific tokenizer/template and parser settings consistent
between arms; the command shows the performance-relevant options.

## Profile and measured performance

Separate CUDA profiles, outside throughput/energy windows:

| Workload | FP8 GEMM | Attention | BF16 output head | Fused quant/norm/activation |
| --- | ---: | ---: | ---: | ---: |
| 8K prefill | 78.35% | 16.52% | 0.11% | 4.24% |
| Short C1 decode | 85.85% | 2.99% | 6.58% | 3.34% |
| Short C2 decode | 85.99% | 2.40% | 6.46% | 3.37% |
| 64K prefill | 37.81% | 59.70% | 0.05% | 2.06% |
| 64K C1 decode | 62.61% | 29.20% | 4.78% | 2.52% |

Short C1 decode was GPU busy for about 97.3% of the traced span. These are
kernel-time shares, not DRAM bandwidth counters. GEMMs dominate short decode;
attention dominates long prefill.

Three saved repeats per cell, medians [minimum, maximum], one sequential boot
per arm. C2/C16 rates are aggregate output tokens/s. Prefill rates use actual
8069 and 64375 input tokens per request. Context decode excludes TTFT and the
first chunk. Every repeat remains in the saved results.

| Workload | Control tokens/s | FlashInfer/page64 tokens/s | Change | Board J/token control → candidate |
| --- | ---: | ---: | ---: | ---: |
| C1 decode | 43.621 [43.589,43.706] | 44.049 [43.809,44.087] | +0.98% | 10.30119 → 10.21602 |
| C2 decode | 87.149 [87.068,87.157] | 86.713 [86.097,88.219] | −0.50% | 5.16239 → 5.13892 |
| C16 decode | 670.516 [669.951,670.572] | 683.989 [683.465,684.587] | +2.01% | .66961 → .65861 |
| 8K prefill | 5592.128 [5587.725,5597.219] | 5682.291 [5591.092,5687.791] | +1.61% | .08020 → .07947 |
| 64K prefill | 2727.554 [2707.701,2735.715] | 2986.318 [2927.321,2990.089] | +9.49% | .16435 → .15067 |
| 8K context decode | 41.800 [41.770,41.800] | 42.410 [42.390,42.410] | +1.46% | Unmeasured |
| 64K context decode | 31.890 [31.890,31.920] | 32.140 [32.070,33.030] | +0.78% | Unmeasured |

Raw decode replays fixed prompt manifests, temperature 0, seed 0, ignore-EOS,
512 output tokens/request, and 4/4/16 requests at C1/C2/C16. Prefill replays
literal token IDs, 8/2 requests per window, one output token each, two excluded
warmups and unique cache salts. NVML cumulative board-energy counters bracket
contiguous windows; client/inter-request idle is included, warmup excluded.
Board energy excludes CPU/DRAM. Prefill energy includes its one output token.

These observations do not estimate between-boot variation or establish a
deployment-wide speedup. FlashInfer with default 16-token pages was separately
rejected: 8K prefill regressed 3.98%, exceeding its frozen 3% limit. Its results
are preserved separately from the 64-token-page recipe.

## Quality and uncertainty

Complete unchanged controls were reused. Each arm/mode has 8 questions × 15
seeded attempts, C15, temperature 1, top-p .95, with the original 5K/10K/20K
question caps. All-attempt token cost includes reasoning, wrong answers and
caps, averaged equally across questions. Usable means correct, nonempty and
normal stop. There were zero request errors in each arm/mode.

| Reasoning | Raw correct control → candidate | Usable | Caps | All-attempt tok/question | Cost change, 95% BCa interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| On | 120 → 120 | 120 → 120 | 0 → 0 | 1791.56 → 1813.65 | +1.23% [−14.18%,+19.53%] |
| Off | 111 → 111 | 106 → 108 | 8 → 4 | 1286.37 → 758.34 | −41.05% [−67.66%,+11.00%] |

The primary on-mode +10% token-cost upper margin remains **inconclusive**:
the observed upper bound is +19.53%. Operational retention was explicitly
accepted with this uncertainty; the statistical decision remains unchanged.
Off mode passed its declared finite compatibility screen, not a statistical
cost-equivalence gate. Analysis used 49999 unpaired within-question resamples,
95% BCa intervals and Holm correction for overall/question contrasts. Matching
seeds do not imply paired trajectories. Shared batching and one boot per arm
limit independence assumptions. Degenerate all-success bootstrap intervals
do not prove zero failure risk; pooled score sensitivity is ±3.10 pp.

Per-question raw/usable/cap counts, all-attempt and correct-only token costs,
all attempt token counts and statistical outputs are retained in `results.json`.
Quality board energy lacks a matched control; no quality-energy gain is claimed.

Arithmetic is not bit-exact: one short continuation diverged at token 10,
while 8K/64K 64-token continuations and nine matched-history next-token IDs
agreed. Common top-20 logprob differences reached 1.25. Numerical diagnostics
are separate from quality gates.

All 15 serving and 10 tool cases passed, with strict JSON, reasoning modes,
cancellation recovery and exact recall at 130891 actual input tokens.
The accepted worker had 281408 BF16 logical KV tokens, fixed 450 W, verified
source/native/configuration, normal memory limits and no OOMs. Capacity is
boot-specific, not an attributed gain.

## Decode candidates ranked and executed

The upstream scan checked 17 selected PRs. The existing SM120 small-M FP8
optimization, [#34424](https://github.com/vllm-project/vllm/pull/34424), was
already included. Hardware family alone does not make block-scale FP8,
Hopper-only or unmatched-shape kernels applicable to this W8A8 workload.

| Execution rank | Candidate | Result |
| --- | --- | --- |
| 1 | [Screened BF16 head #60566](https://github.com/vllm-project/vllm/pull/60566) | Real-state performance rejects the port; no deployment. |
| 2 | Original same-precision FP8 GEMM plan | All 72 operator cases pass numerics; no alternative clears the speed gate. |
| 3 | [FP8 target head #61815](https://github.com/vllm-project/vllm/pull/61815) | Separate precision-changing candidate; not executed. |
| 4 | [RoPE/KV fusion #52363](https://github.com/vllm-project/vllm/pull/52363) | Requires FlashAttention; deferred under retained FlashInfer. |
| 5 | [Manual norm/quant fusion #45364](https://github.com/vllm-project/vllm/pull/45364) | Existing trace already contains compiler fusion; low incremental upside. |
| 6 | Other hardware/scale/shape-specific kernels | Excluded from this workload. |

### Screened head

Pinned source: `229abbf75d8ad63fcb0edc0e64c764ec5a5fe377`. The actual BF16 head
has shape [166912,7168]; candidate buffers added 1162.81 MiB in the probe.
After 18 mixed synthetic cases, 35 hidden-state/logit calls were captured from
20 ordinary serving requests, intentionally capped at 64 tokens each. These
1280 generated tokens form activation fixtures, not a quality evaluation.

Three real fixtures × M=1/2/16 × k=1/20 produce 18 further cases. Early C1
uses calls 0/1/2/4, late C1 uses 4/8/16/31, interleaved across four prompts;
call 4 overlaps. The C16 fixture is the last captured 16-row call, number 31.
M1/M2 use the first 1/2 rows of each fixture. Three counterbalanced 64-replay
CUDA-graph windows use each fixed input, not independent request samples.

Greedy head speedups vary from 1.813× on early C1 to **0.642× on late C1**;
the actual C16 M16 fixture reaches only **0.211×**. Top-k20 retains most
vocabulary rows and is much slower. Ordinary top-p-only requests use the
upstream full-head fallback; adding small top-k to obtain speed would change
the sampling contract.

All 36 synthetic/real cells retained every reference top-k row, with zero
argmax mismatches on these inputs. Kept logits change; this is not bit-exact.
Upstream tests gave **14 passed, 7 failed**: dispatch tests expect
`Sampler.uses_logits_processors`, absent in the tested build. Its
`needs_logits_processing` also includes normal sampling transforms, so a
simple rename would change eligibility semantics. No sampler port was made.

### Same-precision FP8 GEMMs

Exact serialized layer 0 weights/channel scales, with no requantization:
QKV 9216×7168, O 7168×7168, gate/up 44032×7168, down 7168×22016 (N×K).
Inputs reuse captured final hidden states; down inputs derive through actual
gate/up weights and SiLU-mul. These are operator fixtures, not layer 0 states.
Four physical matrix copies exceed the GPU's 128 MiB L2. Each of three
windows replays a four-matmul graph 64 times, in forward/reverse/forward order.

| Method | Projected C1 change | Projected C2 change | Projected C16 change |
| --- | ---: | ---: | ---: |
| Native CUTLASS control | 0% | 0% | 0% |
| Torch rowwise | −19.61% | −14.67% | −13.87% |
| Torch FP32 + compiled scale epilogue | −2.59% | −2.64% | −2.49% |
| Stock Triton | +0.47% | +0.32% | +0.42% |
| Triton M16/N64/K128 | −0.85% | −0.89% | −0.88% |
| Triton M16/N128/K128 | −1.19% | −1.32% | −1.35% |

All 72 cases pass rtol=.01/atol=.01 against FP64 accumulation of the exact FP8
operands, scales and BF16 rounding. Native graph output matches its eager
control in all 12 cases. The Torch FP32 path uses persistent unit-scale tensors
outside capture, following [#61890](https://github.com/vllm-project/vllm/pull/61890).
No production source patch was installed.

The frozen advancement gate requires projected C1 ≥+3%, C2/C16 ≥0%, and all
shapes/numerical checks passing. **No alternative clears it.** Projections sum
52 identical layers of the four measured matrices, then apply the original
85.85% short-C1 GEMM share. Reusing that share for M2/M16 is a screening
assumption, not measured whole-model throughput. No new decode J/token gain
is inferred from short operator windows.

The first GEMM attempt read graph outputs before the first replay, invalidating
its numerical checks; it also hit the private compiler specialization limit.
It remains separate in the data. The corrected attempt replays/synchronizes
before checking output and raises the private limit from 8 to 16 for nine
unique (M,N) pairs. Precision, counts, timing windows and margins are unchanged.
Only corrected observations are used above. Both decode candidates failed
their preceding performance gate, so no additional full-model quality campaign
was entered and the accepted FlashInfer recipe remained in service.
