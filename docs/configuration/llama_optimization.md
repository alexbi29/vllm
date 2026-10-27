# Llama optimization on SM120

For the dense FP8 Llama workload measured here, FlashInfer with 64-token pages
improved 64K prefill by **9.49%** and lowered GPU-board J/input token by **8.32%**.
Decode changes were small. Screened output-head and alternative FP8 GEMM
operators did not clear their advancement gates, so no new kernel was retained.
An additional transparent CUDA memory-compression screen also stayed out:
actual FP8 GEMMs gained no material speed and returned no reusable VRAM for KV.

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

## Transparent CUDA weight-memory compression

This independent operator screen uses the same FP8 bytes, per-channel scales,
dynamic per-token input quantization and BF16 outputs in both arms. It changes
only the VMM allocation hint, `CU_MEM_ALLOCATION_COMP_NONE` versus
`CU_MEM_ALLOCATION_COMP_GENERIC`. Every candidate allocation was checked with
`cuMemGetAllocationPropertiesFromHandle`; a device support attribute alone did
not count as enabled compression. Serving integration would be opt-in and
default-off; no serving allocator or default was changed.

The measured source is `28c118b2ca0671df7fe8df1bcedc56255f3ef925`, with its
existing native build and Torch 2.14, on the same SM120 card at 450 W. This is
separate from the older serving measurements above. The existing GEMM dispatch
uses CUTLASS at M=1/8/32 and cuBLASLt at M=128 (`64 < M <= 256`).

Four serialized layer-0 matrices use the QKV/O/gate-up/down shapes listed above.
There are six counterbalanced windows of 20 samples per arm/cell. CUDA graph
external timing-event nodes surround only the GEMM; a high-entropy buffer of
twice the 128 MiB L2 is written outside the timed nodes before each replay.
CUPTI Python support and Nsight Compute were unavailable, so physical DRAM
bytes and compression ratios were not measured. Reported logical GB/s is not
a physical DRAM counter.

Final timed run; medians in microseconds:

| Operand/cell | Ordinary | Compressible | Rate change |
| --- | ---: | ---: | ---: |
| Actual FP8, sum of four GEMMs at M=1 | 419.840 | 420.184 | −0.08% |
| Actual FP8, sum at M=8 | 422.240 | 425.040 | −0.66% |
| Actual FP8, sum at M=32 | 429.696 | 429.864 | −0.04% |
| Actual FP8, sum at M=128 | 465.256 | 466.376 | −0.24% |
| Zero weights, down/M=8 | 108.872 | 34.816 | 3.13× speedup |
| High-entropy weights, down/M=8 | 109.056 | 108.544 | +0.47% |

The six M=8 window medians have these minimum/maximum ranges (microseconds):

| Actual FP8 matrix | Ordinary range | Compressible range |
| --- | ---: | ---: |
| QKV | 51.200–52.000 | 51.200–51.200 |
| O | 40.960–42.784 | 40.960–42.800 |
| Gate/up | 221.184–221.920 | 221.184–223.024 |
| Down | 108.544–108.544 | 108.544–110.592 |

The zero control demonstrates a working compression path. It does not establish
a benefit for real FP8 operands. The initial timed run also failed the gate:
its M=8 aggregate rate changed −0.32%. Both complete 18-cell runs and every
sample remain in `compressible_memory_results.json`. The aggregate sums describe
four operator cases, not measured model TPS. Neither run clears the frozen +3%
M=8 advancement margin. A subsequent correctness-only run compares raw BF16
storage bits, including signed zero, on the final timed run's matching operand
hashes. All 18 eager and graph cases pass; timing was not repeated for that check.

A separately declared capacity screen addresses KV headroom. In three paired
repetitions each, the real FP8 matrix and the all-zero control both consumed
**152 MiB with either allocation type**, measured independently by CUDA free
memory and NVML. Every release returned to baseline. **Reusable VRAM saving was
zero**, so this path did not improve KV capacity. The initial capacity setup
failed before collecting samples because no CUDA context was current; it is
retained separately from the corrected 12 measurements.

The benchmark is explicitly invoked, and the production allocation path remains
unchanged. No model quality/TPS/prefill campaign was entered after the operator
and capacity gates failed. No new production energy or KV-capacity gain is
claimed. CPU entropy-code compression of a file is a different mechanism from
the hardware patterns recognized by transparent compute-data compression.

## Draft length and prefill batching

This later screen uses the same source `28c118b2ca`, native assets and 450 W as
the compression probe, with the current DFlash2 recipe. Target FP8 weights,
BF16 target head/KV, FlashInfer/page64, probabilistic draft sampling, FP8 draft
head, 16 slots and 131072 context stay fixed. The FP8 GEMM dispatch remains
cuBLASLt for `64 < M <= 256` and native CUTLASS outside that band.

The saved current control uses seven draft tokens, 8192 maximum batched tokens
and a .97 memory budget. This differs from the historical no-speculation recipe
above; the two baselines are not pooled. Each candidate reuses the same complete
16-prompt realistic bank, seeds, sampling and 1024-token cap. C1/C4 have one pass,
C16 has two. Natural stops are respected and output lengths/caps are retained.

| Configuration | C1 on tok/s | C4 on | C16 on | C1 off |
| --- | ---: | ---: | ---: | ---: |
| Saved 7-token / 8192-batch control | 67.2 | 237.2 | 626.9 | 72.9 |
| 3-token / 8192-batch | 64.8 (−3.7%) | 224.8 (−5.2%) | 680.1 (+8.5%) | 69.8 (−4.3%) |
| 7-token / 4096-batch | 67.3 (+0.1%) | 235.3 (−0.8%) | 615.8 (−1.8%) | 72.9 (−0.1%) |

Three tokens reduces mean accepted length at C1 from about 1.72 to 1.65.
Board J/output rises 3.8%/5.4% at C1/C4 and falls 7.7% at C16. It fails the
frozen general-regression screen. The user canceled the five-token test after
seeing the C1/C4 slowdown; no five-token performance or quality data was
collected. The cancellation exits125 without changing the configuration.

For batch4096, board J/output changes −0.1%/+0.8%/+1.8% at C1/C4/C16.
Exact saved 8,069/64,375-token inputs use two excluded warmups and three complete
windows per size, with the original request counts and isolated cache salts.

| Prefill configuration | 8K input tok/s | 64K input tok/s | Board J/input change, 8K / 64K |
| --- | ---: | ---: | ---: |
| Saved control | 5,700 | 3,002 | reference |
| 3-token / 8192-batch | 5,664 (−0.6%) | 3,019 (+0.6%) | +0.8% / −0.7% |
| 7-token / 4096-batch | 5,865 (+2.9%) | 3,046 (+1.5%) | −2.7% / −1.4% |

Capacity requires distinguishing cold compilation from a normal cached load:

| Startup observation | BF16 KV tokens |
| --- | ---: |
| Original 7-token / 8192, cached | 259,520 |
| 3-token / 8192, first compilation, unmeasured setup | 255,616 |
| 3-token / 8192, cached measured boot | 260,736 |
| 7-token / 4096, first compilation, measured boot | 255,680 |
| 7-token / 4096, single cached capacity-only reload | 263,552 |
| Original 7-token / 8192, final restored boot | 259,520 |

The cached batch4096 observation clears the 262,144-token target: 2.011 times
the context limit, 4,032 tokens (+1.6%) above the preceding cached control.
The independent follow-up was declared before that reload; no throughput or
quality suite was repeated. Its positive capacity result does not overwrite
the initial screen's cold-capacity failure or qualify a deployment.

Both measured arms pass all 15 serving probes, including near-maximum-context
recall, 10 tool cases, reasoning modes and strict JSON. No OOM/fatal error was
found in their complete retained logs. No candidate hard8 run followed the
failed initial advancement screens; saved control per-question costs and usable
rates remain in the data, with no new quality-equivalence claim.

The initial unmeasured attempt rejected legitimately unused longer-query
FlashInfer libraries. A final audit also rejected an additional generated
Triton helper after restoration; independent verification confirms all 17
original native-library hashes and records that helper separately. Both audit
failures and successful public restoration receipts remain in the evidence.
The original 7-token/8192 recipe is restored, with no serving/default change.
Any future integration remains opt-in/default-off. All numeric rows, prefill
windows, caps, startup observations and decisions are in
`draft_batch_results.json`. One boot per candidate versus a saved control gives
descriptive comparisons, not a passed statistical performance margin.

## Subsequent explicit adoption and maximum context

The user subsequently adopted batch4096 and requested the largest context
that fits available VRAM. Only the selected serving stanza's batch/context
arguments change; source, environment, native assets, precision, FlashInfer,
seven draft tokens, .97 and 450 W remain the same.

Both target and draft declare a native 524,288-token positional limit. The
existing `--max-model-len auto` fits context to the KV memory budget without
changing RoPE scaling. A normal-cache load resolves **263,488 total input/output
tokens**, with **263,552 logical KV tokens**. This supports approximately one
maximum-length request; 16 shorter sequences remain configured. The earlier
2× capacity comparison refers to the former 131,072-token context.

All **15/15 serving cases** pass, including exact recall at **263,308 input
tokens**, cancellation recovery, reasoning modes and strict JSON. All **10/10
tool cases** pass. A request exceeding the resolved limit returns HTTP 400.
The complete current-boot log has no OOM/fatal error, all 17 original native
hashes match, and the final public answer is 187. The accepted recipe remains
running; backups and a standalone rollback are saved operationally.

The initial cold auto-fit boot resolves 255,616 tokens. Its audit mistakenly
reads proxy-generated model metadata, then restores the original recipe before
the serving suite. The corrected audit reads the existing engine metadata
route, reuses compiled artifacts and verifies the accepted limit. This attempt
remains separate from successful acceptance.

No new hard8 or throughput campaign was run for adoption. Explicit user
authorization is recorded separately from the earlier failed/inconclusive
screens; no statistical decision is relabeled. Auto-fit can select a different
limit on another boot, so the effective engine metadata and current boot are
the evidence for its actual context. This deployment selects existing opt-in
settings; the PR introduces no global serving default or allocation feature.
