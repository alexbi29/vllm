# Quantized KV Cache

## Experimental Gemma4 compact global storage

Gemma4 models with shared global K/V projections can opt into a compact
global cache. It stores 128 rotated key coordinates and 512 value coordinates
per head; the other keys are reconstructed from values and the learned key
normalization. BF16 reconstruction changes rounding from the dense cache.
The optional redundant global V projection can also be omitted:

```bash
vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --hf-overrides '{"gemma4_compact_kv":true,"gemma4_compact_no_v_proj":true}'
```

On SM120, adding `"gemma4_compact_fp8_storage":true` to those overrides stores
the compact global payload in unit-scale E4M3 FP8. Local attention remains
BF16; keep `--kv-cache-dtype bfloat16`. This is lossy cache quantization,
independent of the model's weight quantization. Values are saturated to the
finite FP8 range when stored, then converted to BF16 for attention. Global
512-token pages and local 80-token pages preserve the payload saving without
padding local pages. Whole-model capacity also depends on the local caches,
speculative draft and scheduler reservations; halving the global payload
does not imply doubling the automatic token budget. Qualify accuracy and
long-context behavior for the checkpoint and workload before enabling it.

These compact modes require 512-dimensional global heads, proportional 25%
RoPE, 256-dimensional local heads with a 4:1 KV-head ratio, BF16 activations,
and no LoRA or target KV-sharing layers. BF16, FP8 and NVFP4 model weights
are supported independently of the cache representation. Gemma MTP readers
inherit the target compact layout and precision.

## FP8 KV Cache Overview

Efficient memory usage is crucial for working with large language models. Quantizing the KV (Key-Value) cache to FP8 format can significantly reduce its memory footprint. This optimization enables you to store more tokens in memory, leading to improved throughput and support for longer context windows.

> **Note:** When using the Flash Attention 3 backend with FP8 KV cache, attention operations are also performed in the quantized (FP8) domain. In this configuration, queries are quantized to FP8 in addition to keys and values.

### Supported FP8 KV-Cache Quantization Schemes

vLLM supports two main quantization strategies for the FP8 KV-cache:

- **Per-tensor quantization:**  
  A single scale is applied for each Q, K, and V tensor individually. (`q/k/v_scale = [1]`)
- **Per-attention-head quantization:**  
  Each scale corresponds to an attention head: `q_scale = [num_heads]`, `k/v_scale = [num_kv_heads]`.

> **Note:**  
> Per-attention-head quantization is currently available **only with the Flash Attention backend** and requires the calibration pathway provided by **llm-compressor**.

### Scale Calibration Approaches

You can configure how the quantization scales are computed in vLLM using three different approaches:

1. **No calibration (default scales):**  
   All quantization scales are set to `1.0`.  
   _Configure with:_  
   ```python
   kv_cache_dtype="fp8"
   ```

2. **[Recommended] Calibration with a dataset (via llm-compressor):**  
   Scales are estimated using a curated calibration dataset for maximum accuracy.  
   This requires the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.  
   _See example below!_

#### Additional `kv_cache_dtype` Options

- `kv_cache_dtype="auto"`: Use the model's default data type
- `kv_cache_dtype="fp8_e4m3"`: Supported on CUDA 11.8+ and ROCm (AMD GPUs)
- `kv_cache_dtype="fp8_e5m2"`: Supported on CUDA 11.8+

### Skipping Specific Layers from KV-Cache Quantization

Some attention layer types (e.g. sliding-window) are more sensitive to KV-cache quantization. The `--kv-cache-dtype-skip-layers` flag leaves the specified layers at the model's native dtype while keeping the rest of the layers under the chosen quantized dtype. The flag accepts either layer indices or layer-type names:

```bash
# Skip every sliding-window attention layer.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window

# Skip specific layer indices.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers 0 1 23
```

Programmatic usage:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["sliding_window"],
)
```

---

## Examples

### 1. No Calibration (`kv_cache_dtype="fp8"`)

All quantization scales are set to 1.0.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 2. **[Recommended] Calibration Using a Dataset (with `llm-compressor`)**

For the highest-quality quantization, we recommend calibrating against a dataset using `llm-compressor`. This enables advanced strategies such as per-attention-head quantization.

#### Install the required package

```bash
pip install llmcompressor
```

#### Example: Quantize Llama Attention & KV Cache to FP8

```python
"""
Quantize Llama attention + KV cache to FP8 (choose either 'tensor' or 'attn_head' strategy)
using llm-compressor one-shot calibration.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"       # or "attn_head"
NUM_CALIB_SAMPLES = 512   # Good starting value
MAX_SEQ_LEN = 2048

# -----------------------------
# Helpers
# -----------------------------
def process_and_tokenize(example, tokenizer: AutoTokenizer):
    """Convert chat messages to tokens."""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],  # Quantize queries: q_scale
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,           # Quantize KV cache: k/v_scale
    )

# -----------------------------
# Main
# -----------------------------
def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    recipe = build_recipe(STRATEGY)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
```

For more detailed and up-to-date examples, see the [`llm-compressor` official examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache).
