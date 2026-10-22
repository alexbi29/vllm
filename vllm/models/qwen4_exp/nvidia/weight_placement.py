# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in post-load placement of Qwen4Exp BF16 weights.

``move_embedding_to_host`` keeps the input embedding table in pinned host
memory and reads it through UVA: a lookup touches only the batch rows, so the
table's device memory goes to the KV cache at a small PCIe cost.

``quantize_dense_fp8`` requantizes matching BF16 projections to 128x128-block
FP8 weights with dynamic 1x128 activation quantization. This is lossy.
"""

import regex as re
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


def move_embedding_to_host(embed: VocabParallelEmbedding) -> int:
    """Move `embed`'s weight to pinned host memory; return the bytes moved."""
    param = embed.weight
    if not param.is_cuda or hasattr(embed, "_host_weight"):
        return 0
    host = torch.empty(param.shape, dtype=param.dtype, pin_memory=True)
    host.copy_(param.data)
    # The UVA view does not own the pinned storage.
    embed._host_weight = host
    param.data = get_accelerator_view_from_cpu_tensor(host)
    torch.accelerator.empty_cache()
    nbytes = host.numel() * host.element_size()
    logger.info("Qwen4Exp embed_tokens in pinned host memory: %.2f GiB", nbytes / 2**30)
    return nbytes


def _fp8_block_kernel(name: str):
    if not name:
        return None
    from vllm.model_executor.kernels import linear as kernels

    for candidates in kernels._POSSIBLE_FP8_BLOCK_KERNELS.values():
        for kernel in candidates:
            if kernel.__name__ == name:
                return kernel
    raise ValueError(f"Unknown FP8 block-scaled linear kernel {name!r}")


def quantize_dense_fp8(
    model: nn.Module, pattern: str, kernel_name: str = ""
) -> list[str]:
    """Requantize unquantized BF16 linears whose names match `pattern`.

    Only bias-free 2D weights with both dimensions divisible by 128 qualify.
    Returns the converted module names.
    """
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
    )
    from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
        Qwen4ExpLowLatencyLinearMethod,
    )

    regex = re.compile(pattern)
    force_kernel = _fp8_block_kernel(kernel_name)
    converted, nbytes = [], 0
    for name, module in model.named_modules():
        if not (
            isinstance(module, LinearBase)
            and type(module.quant_method)
            in (UnquantizedLinearMethod, Qwen4ExpLowLatencyLinearMethod)
            and regex.search(name)
            and getattr(module, "bias", None) is None
        ):
            continue
        weight = module.weight
        if (
            weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or weight.shape[0] % 128
            or weight.shape[1] % 128
        ):
            continue
        method = Fp8PerBlockOnlineLinearMethod()
        # The method reads its output dtype from the default dtype, which is
        # only the model dtype while the model itself is being built.
        method.out_dtype = weight.dtype
        module.weight_block_size = method.weight_block_size
        method.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=method.activation_quant_key,
            weight_quant_key=method.weight_quant_key,
            weight_shape=tuple(weight.shape),
            input_dtype=method.input_dtype,
            out_dtype=method.out_dtype,
            force_kernel=force_kernel,
            module_name=type(method).__name__,
        )
        nbytes += weight.numel() * weight.element_size()
        method.process_weights_after_loading(module)
        module.quant_method = method
        converted.append(name)
    torch.accelerator.empty_cache()
    logger.info(
        "Qwen4Exp FP8 dense: %d projections, %.2f GiB of BF16 weights",
        len(converted),
        nbytes / 2**30,
    )
    return converted
