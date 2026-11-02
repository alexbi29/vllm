# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw compact KV metadata, retaining the existing 640-word cache geometry."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gemma4_compact import Gemma4CompactRawImpl


@triton.jit
def _store_u32_as_u16(pointer, value, mask):
    bits = value.to(tl.uint32, bitcast=True)
    target = pointer.to(tl.pointer_type(tl.uint16))
    tl.store(target, (bits & 65535).to(tl.uint16), mask)
    tl.store(target + 1, (bits >> 16).to(tl.uint16), mask)


@triton.jit
def load_raw_metadata(key_cache, row_offsets, mask):
    words = key_cache.to(tl.pointer_type(tl.uint16))
    lo = tl.load(words + row_offsets + 124, mask, other=0).to(tl.uint32)
    hi = tl.load(words + row_offsets + 125, mask, other=0).to(tl.uint32)
    inverse_rms = (lo | (hi << 16)).to(tl.float32, bitcast=True)
    lo = tl.load(words + row_offsets + 126, mask, other=0).to(tl.uint32)
    hi = tl.load(words + row_offsets + 127, mask, other=0).to(tl.uint32)
    positions = (lo | (hi << 16)).to(tl.int32, bitcast=True)
    return inverse_rms, positions


@triton.jit(do_not_specialize=["num_tokens", "rope_rows"])
def _write_raw_cache(
    raw,
    positions,
    cache,
    slots,
    rope,
    weight,
    num_tokens,
    raw_stride_t: tl.int64,
    raw_stride_h: tl.int64,
    cache_stride_b: tl.int64,
    cache_stride_h: tl.int64,
    cache_stride_t: tl.int64,
    position_stride: tl.int64,
    rope_stride: tl.int64,
    rope_rows,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    token = row // NUM_HEADS
    head = row % NUM_HEADS
    slot = tl.load(slots + token, token < num_tokens, other=-1).to(tl.int64)
    valid = (token < num_tokens) & (slot >= 0)
    dim = tl.arange(0, 512)
    x = tl.load(
        raw
        + token[:, None] * raw_stride_t
        + head[:, None] * raw_stride_h
        + dim[None, :],
        valid[:, None],
        other=0.0,
    )
    xf = x.to(tl.float32)
    inverse = tl.rsqrt(tl.sum(xf * xf, 1) / 512.0 + EPSILON)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int32)
    gamma = tl.load(weight).to(tl.float32)
    rd = tl.arange(0, 128)
    kd = tl.where(rd < 64, rd, rd + 192)
    partner = (kd + 256) % 512
    a = tl.gather(xf, tl.broadcast_to(kd[None, :], (ROWS, 128)), 1)
    b = tl.gather(xf, tl.broadcast_to(partner[None, :], (ROWS, 128)), 1)
    # Match the reader: positions outside the RoPE table contribute no rotation.
    in_table = valid & (position >= 0) & (position < rope_rows)
    rope_base = position.to(tl.int64) * rope_stride
    c = tl.load(
        rope + rope_base[:, None] + (kd % 256)[None, :], in_table[:, None], other=0.0
    ).to(tl.float32)
    s = tl.load(
        rope + rope_base[:, None] + 256 + (kd % 256)[None, :],
        in_table[:, None],
        other=0.0,
    ).to(tl.float32)
    rotated = (a * c + tl.where(kd[None, :] < 256, -b * s, b * s)) * (inverse * gamma)[
        :, None
    ]
    base = (
        slot // BLOCK_SIZE * cache_stride_b
        + head * cache_stride_h
        + slot % BLOCK_SIZE * cache_stride_t
    )
    tl.store(
        cache + base[:, None] + rd[None, :],
        rotated,
        valid[:, None] & (rd[None, :] < 124),
    )
    _store_u32_as_u16(cache + base + 124, inverse, valid)
    _store_u32_as_u16(cache + base + 126, position, valid)
    tl.store(cache + base[:, None] + 128 + dim[None, :], x, valid[:, None])


def write_raw_compact_cache(raw, positions, cache, slots, rope, weight, epsilon):
    if (
        raw.dtype != torch.bfloat16
        or raw.ndim != 3
        or raw.shape[-1] != 512
        or raw.stride(-1) != 1
        or cache.dtype != torch.bfloat16
        or cache.shape[-1] != 640
        or cache.shape[1] != raw.shape[1]
        or cache.stride(-1) != 1
        or raw.shape[0] < slots.numel()
        or positions.ndim != 1
        or positions.numel() < slots.numel()
        or positions.dtype not in (torch.int32, torch.int64)
        or raw.device != cache.device
        or positions.device != raw.device
        or slots.device != raw.device
        or rope.device != raw.device
        or weight.device != raw.device
        or rope.ndim != 2
        or rope.shape[1] != 512
        or rope.stride(1) != 1
        or weight.dtype != torch.bfloat16
        or weight.numel() not in (1, 512)
    ):
        raise ValueError("Invalid raw compact KV writer inputs")
    count = slots.numel()
    if not count:
        return
    _write_raw_cache[(triton.cdiv(count * raw.shape[1], 4),)](
        raw,
        positions,
        cache,
        slots,
        rope,
        weight,
        count,
        raw.stride(0),
        raw.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        positions.stride(0),
        rope.stride(0),
        rope.shape[0],
        raw.shape[1],
        cache.shape[2],
        epsilon,
        ROWS=4,
        num_warps=4,
    )


def raw_compact_cache_update(
    raw: torch.Tensor, positions: torch.Tensor, layer_name: LayerNameType
) -> torch.Tensor:
    from vllm.model_executor.layers.attention.attention import get_attention_context

    layer_name = _resolve_layer_name(layer_name)
    _, layer, cache, slots = get_attention_context(layer_name)
    if slots is not None:
        impl = cast("Gemma4CompactRawImpl", layer.impl)
        write_raw_compact_cache(
            raw,
            positions,
            cache,
            slots,
            impl.compact_raw_rope,
            impl.compact_k_norm,
            impl.compact_raw_epsilon,
        )
    return raw.new_empty(0)


def _raw_compact_cache_update_fake(
    raw: torch.Tensor, positions: torch.Tensor, layer_name: LayerNameType
) -> torch.Tensor:
    return raw.new_empty(0)


direct_register_custom_op(
    op_name="gemma4_raw_compact_cache_update",
    op_func=raw_compact_cache_update,
    fake_impl=_raw_compact_cache_update_fake,
)
