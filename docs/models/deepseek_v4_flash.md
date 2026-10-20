# DeepSeek V4 Flash: retained SM120 work and validation

This fork's PR #24 collects the retained DeepSeek-V4-Flash work, including
resident MXFP4 scale compression, numerical fixes, memory accounting and native
small-batch expert dispatch. It uses ordinary DeepSeek weights; Docker, B12X
and a converted checkpoint are not required.

## Change inventory

| Retained change | Original serving commit | Location in this PR |
| --- | --- | --- |
| Canonical sparse-attention token order on DS4/SM120; canonical Marlin expert regions | `c8d6be705e` | `sparse_attn_indexer.py`, `marlin_moe.py` and tail/graph regressions |
| Release prefill-chunk logits before allocating the next chunk | `1998a727f7` | `sparse_attn_indexer.py` and allocation-lifetime regression |
| Avoid the target MTP residual buffer unused by DSpark | `2c3522a930` | `deepseek_v4/nvidia/model.py` and target-construction tests |
| Support one-argument TORCH_LIBRARY alignment bindings | `b30f963bf5` | `utils/deep_gemm.py` and binding-contract tests |
| Quantize live routed FP4 activation rows, leaving padding alone | `00a0baaf0c` | `fp8_utils.py`, permutation/MoE callers and graph tests |
| Account for the real allocating FP8 Q insertion operation when profiling | `5f51f08dc8` | `deepseek_v4/attention.py` and allocation-contract tests |
| C128 capture-state correction | upstream `877f6df8891`, serving backport `4508a03af5` | Already an ancestor of the pinned PR base; not duplicated |
| Opt-in lossless resident compression, lifecycle guards and review fixes | `7539238b64`, `2bdbc7a310` | Original PR implementation and regressions |
| Native-order 32-row CSF reconstruction | `176f1cb2de` / serving `0fd6170f20` | Codec/decoder and CI tests |
| Supported small-batch 64-row native layouts | `ecfa49960d` | Alignment selection and boundary tests |
| Capability-gated 32-row FP8-by-FP4 grouped layouts | `c19d895e98`, DeepGEMM `a0c6de48` | Python dispatch, native patch, build wiring and fallback tests |
| Share expert selection and reconstruct both scale tensors in one launch | net `c19d895e98..0ea15f152a` | `mxfp4_csf_pair.py`, MoE caller and changing-route/padding tests |
| Statistical cost comparison and numerical-drift methodology | October 13–18 studies | `benchmarks/deepseek_v4/harness/` and [drift guide](deepseek_v4_numerical_drift.md) |

The original two prototype commits included an in-place Q experiment. This PR
includes their final **pair-only net diff**: allocating Q remains and there is no
dependency on the experimental Q extension. Unrelated Gemma/Qwen/integration
changes are outside this PR. Upstream DSpark/model support already in the base
is inherited rather than copied from the older special branch.

## Build and configuration

Normal CMake builds apply the checked-in
`cmake/patches/deepgemm-sm120-fp8-fp4-m32.patch`. Reapplication is idempotent;
incompatible or partially patched sources fail before modification. The patch
adds the native capability query and enables M32 only for grouped-contiguous
FP8 activations by packed-FP4 weights at runtime alignment32. It changes no
device arithmetic. Python also requires the native capability before selecting
M32; older prebuilt libraries retain the supported64-row fallback.

The patch targets the pinned DeepGEMM
`727f847082d2e7946f70dc02b94949db206310d3` (2.8.1), whose stable-ABI op
registration declares the capability query with `m.def` and binds it with
`m.impl(TORCH_BOX(...))`. Earlier revisions of the patch targeted
`1e1842a833699298f7afc02eefb2ed168fff6938`, reproducing the four changed files
of the qualified private native commit
`a0c6de48f042160409fdb4ede3be9caeda49a7ae`; the device heuristic is unchanged.
The existing
`DEEPGEMM_SRC_DIR` override remains available; use a separate source checkout
because the build applies the patch to that checkout.

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --tensor-parallel-size 2 --enable-expert-parallel --moe-backend deep_gemm \
  --kernel-config '{"mxfp4_scale_compression":true}'
```

Compression is opt-in. Paired reconstruction defaults on within that path;
`DS4_CSF_PAIR_FUSION=0` plus a cold restart restores separate reconstruction.
Compressed target layers share expanded native-layout scale scratch. Draft
layers are excluded and incompressible payloads retain their original storage.
EPLB and overlapping microbatches are unsupported. Weight reload/update/reset,
state export, level-2 sleep and weight discard fail closed while named scale
Parameters alias mutable scratch. CPU weight offload requires UVA; native CPU
KV offload is a separate supported facility.

## Qualified integration profile

The saved October18 release used two RTX PRO6000 Blackwell96GB GPUs at300W each,
TP2/EP,16 slots, synchronous8192-token batches, GMU0.963, DSpark5 with greedy
drafts/standard rejection, FP8 KV/block256,512MiB indexer workspace,32GiB CPU KV
offload and1,048,576 maximum context. Target/tokenizer/draft use the original
0731 revision above. Automatic logical GPU KV is **1,648,445**.

Source `0ea15f152a5408dabea0c3dc55aae677cfcf8967` ran on the October10 integration
core `0.31.1rc1.dev297+ga7558f6b53.cu134`, CUDA13.4, with private DeepGEMM
`a0c6de48`. This is a main-based integration, separate from the older Jasl SM120
branch. The clean PR uses newer upstream base `a2fb4827324b`; these saved
full-model results **do not qualify a fresh native build of the clean PR base**.
PR-port CPU/interpreter/lint checks are recorded separately in the PR description.

At matched16-slot settings, the earlier uncompressed-main/CSF comparison measured
1,141,047 versus1,647,038 logical KV tokens (+44.3%). Subsequent decoder changes
reached the current1,648,445; those measurements are distinct from that matched
compression comparison.

### Decode and prefill

| Metric, tok/s | Accepted native32, October16 | Pair-only, October18 |
| --- | ---: | ---: |
| C1 decode | 309.46 | 315.12 |
| C2 aggregate | 583.27 | 591.00 |
| C4 aggregate | 1029.27 | 1018.46 |
| C8 aggregate | 1724.54 | 1774.70 |
| C16 aggregate | 2822.61 | 2965.17 |
| 8K prefill | 10118.65 | 10013.40 |
| 64K prefill | 9629.60 | 9532.80 |
| 128K prefill | 8709.95 | 8637.30 |

Speed uses T0/thinking off, three decode repeats and two measured prefill runs
after excluded warmups. These historical full-sweep contrasts are descriptive,
not independent-boot or source-causal significance.

A separately frozen warm C2 confirmation processed76,800 fixed output tokens
per arm:583.27→592.05 aggregate tok/s (+1.504%), with all128 output-text hashes
matching. Board cumulative-energy counters measured969.87→952.79J/1K output
tokens (−1.762%), one boot per arm. Sustained C1 energy was effectively flat.
Neither observation establishes a population energy/heat improvement. The exact
work reduction remains useful: compactions2→1 and preparation launches4→2 per MoE.

### Full high hard8×100/C10

Two cold candidate loads,400 attempts each, use the saved-main100 distinct
seed/question pairs and historical grader. Every completion token counts,
including reasoning, capped answers and completed wrong answers.

| Arm | All-attempt tok/q | Usable/800 | Caps | Completed wrong | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved main | 2221.669 | 764 | 34 | 2 | 0 |
| Accepted native32 | 2290.425 | 758 | 41 | 1 | 0 |
| Pair-only | 2281.964 | 767 | 32 | 1 | 0 |

Pair-only minus main cost:+2.714%,95%BCaCI[−5.906%,+12.242%],two-sided
studentized centered paired-bootstrap p=.5558. Usable difference:+.375pp,
paired95%CI[−1.25,+2.0]pp, exact McNemar p=.76599. No secondary contrast survives
Holm correction. Against accepted native32, cost−.369%,paired p=.944.
Non-significance is not equivalence or noninferiority. Main had12 slots versus
candidate16 atC10; historical timing, batching, boot and shared-batch dependence
remain limitations. Results condition on eight fixed questions and recorded blocks.

| Question | Main tok/q | Native32 tok/q | Pair-only tok/q |
| --- | ---: | ---: | ---: |
| code-1 | 61.07 | 61.80 | 62.97 |
| mod-arith | 177.04 | 170.50 | 175.77 |
| train-accel | 925.73 | 925.70 | 900.40 |
| euler-totient | 223.41 | 204.60 | 210.65 |
| fence-post | 107.96 | 103.90 | 112.17 |
| code-circular | 9897.43 | 10172.40 | 10509.42 |
| code-levenshtein | 4476.87 | 4601.00 | 4431.91 |
| code-gcdrec | 1903.84 | 2083.50 | 1852.42 |

### Numerical and context validation

The qualified integration passed78 changing-route graph byte/padding cases
across both GPUs,8 native MoE cases,2048 greedy IDs/selected/top20 scores and
same-history prefixes0/64/256/1024 on both qualification loads and the public
worker. Uncached1,030,000-token high/T0 recall completed correctly in284.3s.
Allocator retry warnings were retained; the request completed normally.
Earlier unchanged-source64K score variability remains unresolved. These finite
checks do not establish universal logprob parity.

## Reusable comparison tools

The saved comparison utilities and their regression tests are included:

```bash
PYTHONPATH=benchmarks/deepseek_v4 \
  uv run python -m harness.quality_compare --help
uv run python -m pytest tests/benchmarks/test_deepseek_v4_quality_compare.py -q
```

Inputs retain the original per-question attempts, seeds, experiment blocks,
reasoning configuration, completion costs and outcomes. The tool refuses
mismatched controls and silent question intersections, counts all attempts,
supports matched-seed inference, confidence intervals and Holm correction, and
withholds inference when its required units are missing. A directional p-value
is a null-tail probability, not the probability that a branch is better.

## Rejected experiments and remaining work

In-place Q passed exact checks but did not reduce the global peak or increase
KV; its code and experimental extension are excluded. Static dense dimensions
saved only0.35% pooled kernel time; forced64 variants were14–17% slower and
changed BF16 bytes. Shared-expert fusion, broad/draft Marlin substitution,
decoder/FC1 overlap, compact graph coverage and additional draft-policy screens
produced no retained winner. Generic shared scratch remains held for stream
lifetime races. Async4096 is excluded from the qualified profile; older
pre-C128-fix async comparisons are confounded and must not be pooled with
corrected results. GMU increases are not a substitute for reducing memory use.

Potential future work is native dense-FP8 warp/split-K specialization, a bounded
small-message all-reduce geometry screen, in-GEMM scale decoding, and measured
peak-allocation lifetime reduction. None is enabled by this PR. Preserve the
current KV floor and use cheap operator/score controls for iteration, then full
high800, speed/context and public checks for a final promotion candidate.

The complete raw local archive remains under
`/home/ubuntu/src/vllm/bench/experiments/`, notably `ds4-main-perf-20261012`,
`ds4-async-causal-20261014`, `ds4-c128-fixed-csf-hard8x100-c10-20261015`,
`ds4-csf-align64-20261015`, `ds4-spec-gemm-20261016` and
`ds4-decode-execution-20261018`. Source/native receipts, frozen tools/seeds,
incremental response journals, failures and all-attempt analyses are retained
there. PR24 is the code/review record; the archive contains the original large
traces, tensors and model outputs.
