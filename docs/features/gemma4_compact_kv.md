# Gemma 4 compact KV and assistant tuning

Gemma 4 26B-A4B and 31B can use compact BF16 global KV with compatible
512-dimensional global heads, shared K/V projections and proportional 25% RoPE:

```bash
vllm serve /path/to/Gemma-4-26B-A4B-NVFP4 \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --hf-overrides '{"gemma4_compact_kv":true,"gemma4_compact_no_v_proj":true}'
```

Each global head stores 128 rotated K elements and 512 normalized V elements,
640 BF16 values in total. The reader reconstructs unrotated K from V and the
target's K-normalization weight. Global cache payload falls37.5% compared with
separate 512-element K and V. Sliding-window cache and model weights keep their
existing formats. The page planner, active prefill reservations and graph
allocations determine the resulting logical capacity.

Compact KV requires BF16 activations and explicit BF16 KV with quantized model
weights. LoRA, target cross-layer KV sharing and incompatible head/rotation
geometry are rejected. Compact FP8 storage is a separate experimental option;
the recipe above retains BF16 cache precision.

## NVFP4 checkpoint reload

NVFP4 gate/up scale compensation is derived again when raw checkpoint weights
are loaded. An in-place reload from unequal to equal branch scales clears the
previous derived ratio; it must not keep the earlier checkpoint's correction.
This reset adds no work to the forward path and leaves fresh-load arithmetic
unchanged.

## Query-only assistant execution

The Gemma assistant reads target KV. On the first draft step, only the last
accepted row of each request contributes to the draft. Query-only execution
compacts those rows, preserves request mappings and sampling positions, and
reuses one-query decode graphs for every step. When no drafts are requested,
the assistant forward can be skipped entirely.

Enable `VLLM_GEMMA4_Q_ONLY_DRAFTING=1` on Model Runner V2. This is default-off
and currently scoped to TP/PP/DP/PCP=1 without external multimodal draft
embeddings. The target's existing verification policy is retained. Batch-shape
and attention-route changes can affect floating-point rounding and acceptance;
validate output-token cost and usable-answer rates on your workload.
See [Gemma assistant configuration](speculative_decoding/mtp.md).

## Measured SM120 example

The November 1 screen used NVIDIA `Gemma-4-26B-A4B-NVFP4` revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6` and the 26B assistant revision
`6e5aaaf4c42b98394530b8fda2e95cadd65c151c`, one RTX PRO6000 Blackwell 96GB,
TP1, 450 W, MTP3 with an FP8 draft-head copy, 32 slots, async batch 8192, .96 GPU memory
utilization and 262144 context. BF16 compact KV/no-V and the audited
`VLLM_GEMMA4_DENSE_CUDNN=1` shared MLP were held fixed.

Measurements use integration 22e973ca0f versus its query-only port fac53cdb72,
with the same Torch 2.14/Triton 3.8/FlashInfer 0.7.2.post1/native stack. They are
integration-port measurements; the older maintenance-PR base has separate
operator and CPU validation. Two control boots and one query-only boot have three complete windows per
workload. The table retains each median and its full range. Energy change
uses the mean of the two control-boot median J/token values.

|Work|Control A tok/s|Control A2 tok/s|Query-only tok/s|Board J/token change|
|---|---:|---:|---:|---:|
|C1 decode|498.5 (497.5–502.9)|500.2 (499.0–500.9)|518.5 (518.2–519.7)|-3.23%|
|C16 aggregate decode|4644.1 (4123.9–4713.9)|4653.9 (4560.6–4673.1)|4685.1 (4134.2–4715.4)|-1.35%|
|4K prefill|33317.4 (32467.9–34316.1)|34030.9 (33395.7–34276.7)|34487.2 (34115.9–35242.1)|-0.25%|
|64K prefill|9725.8 (9684.4–9934.7)|9872.6 (9806.8–9967.4)|11272.2 (11217.9–11344.0)|-13.04%|

Decode uses 2048 output tokens/request, temperature 0 and EOS ignored. Prefill
uses exact 4096/65536 input IDs with unique cache salts, one output token and
eight/two requests per window. Warmup is outside the cumulative NVML energy
bookends. Energy covers the GPU board. Shared CPU load and boot effects limit
causal interpretation; the earlier control cohort and every sample are retained.

The 64K compiled profile confirms 40 full-prompt global-attention calls instead
of 48. The five target global layers retain their existing kernel; the four
assistant layers' full-prompt work disappears. Three 2048-token greedy counting
outputs match the control. Candidate A/A probes at 31/8224/65568 prompt tokens
repeat token IDs and selected/top20 scores exactly. Reasoning-on serving checks
include image input, prefix branches, cancellation and near 259K recall.

The fixed reasoning-on efficiency pilot has 120 attempts/arm:114 → 116 usable,
six → four caps and zero request errors. All-attempt output cost changes +1.88%,
with a 95% within-question BCa interval of [−5.62%, +10.35%]. Its declared finite
screening margins clear; equal general quality and token efficiency are not
established. Quality remains separate from fixed-work speed measurements.

## Combined capacity profile

A separately declared follow-up kept query-only drafting enabled and reduced
`max_num_batched_tokens` to 4096. Its 120-attempt reasoning-on pilot also had 116
usable answers, four caps and zero errors. Relative to query-only/8192,
all-attempt cost changed +0.45%, 95% BCa [−6.62%, +7.89%]; the usable-rate interval
was [−4.17, +4.17] percentage points. Both finite pilots cleared their declared
cost/usable/cap margins. No token-cost improvement or general quality equality
is established.

The final combined profile measured 517.3/4665.5 aggregate tok/s at C1/C16 and
33788.2/10975.6 input tok/s at 4K/64K. Relative to the mean of both original
control-boot medians, these are +3.6%/+0.4%/+0.3%/+12.0%; 64K board J/input token
falls 10.8%. The three-window C16 range includes 4112.1–4667.1 tok/s; retain this
variation instead of selecting only the faster repetitions.

Normal-cache startup reports 3,781,455 logical BF16 KV tokens versus 2,847,089
for the last control boot (+32.8%), with 262144 maximum per-request context.
The combined profile uses 70.83GiB of KV versus 70.24GiB for that control.
Most of the logical-capacity gain comes from smaller prefill reservations;
this is not a 32.8% reduction in allocated VRAM. The unchanged 8192 query-only
normal-cache profile reports 2,850,305 tokens, approximately flat. Cold-load
capacity and boot-dependent memory variation are retained separately.

## Capacity and energy tradeoffs

`max_num_batched_tokens` affects the hybrid cache's prefill reservations.
An 8192 → 4096 screen increased reported logical capacity from about 2.85M to 3.73M
tokens at the same context, memory utilization and BF16 representation.
This changes how many tokens the planner can admit across requests. Repeat
capacity checks after a normal-cache startup and validate the combined serving
profile. Batch 2048 failed with the enabled multimodal item budget 2496; preserve
that modality requirement when choosing a batch limit.

A separate interleaved 450/400/350 W screen retained the same model and knobs.
At 400 W, decode medians stayed within 1% while board J/output token fell about
10–11%; prefill slowed 7–8%. At 350 W, C1 energy fell about 21%, while 64K prefill
slowed 15%. 450 W was restored for the balanced performance recipe. Choose a
power limit for the workload and measure energy over the same completed work.
