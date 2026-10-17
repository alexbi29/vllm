# GLM-5.3-Flash memory compression on SM120

This fork maintenance change packages the configuration tested on two 96 GB
RTX PRO 6000 Blackwell GPUs. It combines optional resident NVFP4 scale
compression, optional compact NoPE FP8 cache rows, and repeatable GLM pool
selection. The target checkpoint is NVIDIA GLM-5.3-Flash-NVFP4 revision
`da920bb0b9f4a06727223a349e55468e38352348`.

## Results and scope

| Measurement | Original layout | Compact layout |
| --- | ---: | ---: |
| Model allocation per GPU, compressed scales | 84.33 GiB | 84.33 GiB |
| KV allocation per GPU | 3.43 GiB | 3.43 GiB |
| Logical hybrid KV capacity | 277,267 | 345,324 (+24.5457%) |
| Sparse MLA row | 656 bytes | 528 bytes |

Resident scale compression saves approximately 4.04 GiB net per GPU across
42 MoE layers, after accounting for one shared 216 MiB reconstruction pair.
It preserves the packed weights and every E4M3 scale byte. A per-expert
four-bit palette stores exact sparse byte exceptions; active experts are
reconstructed into native FlashInfer layout before the unchanged GEMM.
Incompressible scales retain their original representation.

The KV change removes 128 unused padding bytes. All 512 FP8 values and four
FP32 group scales remain unchanged. KDA recurrent state stays FP32. The
hybrid capacity comes from the server's explicit `kv_cache_size_tokens`
metric and scoped startup log, not `num_gpu_blocks * block_size`.

Both layouts passed 13 cases each: repeated and varied 8192/64512-input
pairs with 64 output tokens, a 512-token continuation, and four prefix
probes. Output IDs and top-five scores match exactly between layouts and
across repeats; all scores are finite. Operator checks also cover native
GEMM output bytes, changing graph routes, all finite BF16 cache maxima,
strided pages and native/public FlashInfer prefill/decode parity.

Full hard8×15/C2 is running separately. No throughput gain or completed
general-quality result is claimed here. See
[validation summary](glm53_sm120_compression_validation.json).

Qualified serving configuration: TP2/DCP1, no EP, synchronous scheduling,
eager execution, text only, MTP off, BF16 activations, FP8 KV, two request
slots, 2048 batched tokens, 65536 request context, and 0.95 memory budget.
Capacity is not a validated single-request context limit. Full-model
graphs, MTP, external KV connectors and other hardware are not qualified.

## Sparse-selector repair

Native selectors emit pools in atomic arrival order and can choose different
IDs at an exact cutoff tie. At long rows, a native 4096-entry intermediate
bucket can also overflow: the first traced failure had 11989 candidates in
that bucket and discarded higher scores.

The GLM path validates the supplied cutoff against all valid logits, repairs
an invalid cutoff using four exact FP32 radix passes, and fills cutoff ties
by ascending pool ID. Pool IDs are emitted in logical order before token
expansion. The scan has no intermediate candidate limit or host readback.
The row length stays dynamic to bound compilation by power-of-two scan widths.
Both qualification arms use this repair; the cache comparison does not compare
against the original non-repeatable selector.

## FlashInfer companion overlay

Compact mode requires the companion patch against the packaged FlashInfer
0.7.0.post1 source. It includes the public validator, planner byte stride and
a distinct `sparse_mla_sm120_glm53_compact` JIT module. Stock 656-byte and
private 528-byte ABIs are checked explicitly at startup.

Create a private copy of the installed `flashinfer` package, preserving its
`data/include` tree. From that copied package directory, run:

```bash
git apply --check /path/to/vllm/tools/patches/flashinfer-glm53-compact-kv-0.7.0.post1.patch
git apply /path/to/vllm/tools/patches/flashinfer-glm53-compact-kv-0.7.0.post1.patch
```

The accompanying manifest records original and patched file hashes. Put the
directory containing that private `flashinfer` copy before vLLM on
`PYTHONPATH`; do not replace the shared installation. Use a private
`FLASHINFER_WORKSPACE_BASE` for native builds.

The tested planner constants are included as
`tools/patches/flashinfer-glm53-sm120-cpb.json`. They apply to the named
RTX PRO 6000 Blackwell devices, not arbitrary GPUs. Copy that file to a private
directory as `sparse_mla_sm120_cpb.json` and set `FLASHINFER_AUTOTUNE_DIR` to
that directory. Both qualification arms used identical planner constants.
Autotuning during model startup was disabled because its extra allocations
exceeded memory headroom after KV allocation.

Example launch after building this vLLM base and preparing the overlay:

```bash
vllm serve nvidia/GLM-5.3-Flash-NVFP4 \
  --tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype fp8 \
  --max-model-len 65536 --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --enforce-eager --no-async-scheduling \
  --moe-backend flashinfer_cutlass --language-model-only \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --kernel-config '{"nvfp4_scale_compression":true,"enable_flashinfer_autotune":false,"glm53_compact_kv":true}'
```

Original-layout control: remove `glm53_compact_kv` and use installed FlashInfer
instead of the overlay, retaining the selector repair and weight-scale codec.
Resident compression rejects EP, EPLB, ubatching, asynchronous scheduling,
sleep and state export; its scratch is shared across serialized target layers.
Draft/MTP layers are excluded.

## Existing upstream work

- [vLLM #58305](https://github.com/vllm-project/vllm/pull/58305) implements
  general compact NoPE cache support. This fork snapshot isolates the tested
  528-byte ABI behind a flag and a private FlashInfer overlay, without changing
  shared native binaries.
- [vLLM #55122](https://github.com/vllm-project/vllm/pull/55122) works on the
  native persistent selector. This snapshot keeps existing binaries and repairs
  GLM selection in Triton, with an actual long-context overflow reproducer.
- [vLLM #60646](https://github.com/vllm-project/vllm/pull/60646) decodes CSF
  scales at load. It does not keep scales compressed in GPU memory; the resident
  representation here addresses that separate memory cost.

This is a fork review package, not a competing upstream submission. Human
semantic review and the remaining quality result are required before promotion.
