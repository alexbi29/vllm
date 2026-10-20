# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM NoPE FP8 cache writer without the unused positional-key tail."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _write_compact(
    Values,
    Cache,
    Slots,
    INPUT_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    slot = tl.load(Slots + token)
    if slot >= 0:
        lane = tl.arange(0, 128)
        values = tl.load(Values + token * INPUT_STRIDE + group * 128 + lane).to(
            tl.float32
        )
        maximum = tl.max(tl.abs(values), 0)
        # BF16 maxima have seven fractional bits. Division by 448 crosses a
        # power-of-two boundary exactly when the significand exceeds 1.75.
        # This reproduces exp2(ceil(log2(max(amax/448,1e-4)))) without libm
        # approximation, including its minimum scale of 2**-13.
        bits = maximum.to(tl.uint32, bitcast=True)
        exponent = ((bits >> 23) & 255).to(tl.int32) - 135
        exponent += ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
        exponent = tl.maximum(exponent, -13)
        scale_bits = (exponent + 127).to(tl.uint32) << 23
        scale = scale_bits.to(tl.float32, bitcast=True)
        inverse = ((127 - exponent).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        quantized = (values * inverse).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        row = (slot // BLOCK_SIZE) * BLOCK_STRIDE + (slot % BLOCK_SIZE) * 528
        tl.store(Cache + row + group * 128 + lane, quantized)
        scale_cache = Cache.to(tl.pointer_type(tl.float32))
        tl.store(scale_cache + row // 4 + 128 + group, scale)


def write_compact_nope(values: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor):
    if values.dtype != torch.bfloat16 or values.ndim != 2 or values.shape[1] != 512:
        raise ValueError("Compact GLM cache requires BF16 [tokens,512] values")
    if values.stride(1) != 1:
        raise ValueError("Compact GLM values must have contiguous columns")
    if cache.dtype != torch.uint8 or cache.ndim != 3 or cache.shape[2] != 528:
        raise ValueError("Compact GLM cache requires uint8 [blocks,block_size,528]")
    if cache.stride(2) != 1 or cache.stride(1) != 528:
        raise ValueError("Compact GLM cache rows must be densely packed")
    if slots.dtype != torch.int64 or slots.ndim != 1 or not slots.is_contiguous():
        raise ValueError("Compact GLM slots must be contiguous int64")
    if slots.numel() > values.shape[0]:
        raise ValueError("More cache slots than input values")
    if values.device != cache.device or slots.device != cache.device:
        raise ValueError("Compact GLM values, slots and cache must share a device")
    if not slots.numel():
        return
    _write_compact[(slots.numel(), 4)](
        values,
        cache,
        slots,
        values.stride(0),
        cache.shape[1],
        cache.stride(0),
        num_warps=4,
    )
