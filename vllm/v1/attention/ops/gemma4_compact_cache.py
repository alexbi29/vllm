# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Gemma4 global cache: K[0:64,256:320] followed by V[0:512].

Layout follows leDissolution/gewell's compact global cache. Unrotated keys
are reconstructed as BF16(V * KNorm); this changes rounding relative to
independently normalizing K and V, and requires model quality validation.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _write_compact_cache(
    key,
    value,
    cache,
    slots,
    key_token_stride: tl.int64,
    key_head_stride: tl.int64,
    value_token_stride: tl.int64,
    value_head_stride: tl.int64,
    cache_block_stride: tl.int64,
    cache_head_stride: tl.int64,
    cache_token_stride: tl.int64,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots + token).to(tl.int64)
    if slot < 0:
        return
    base = (
        slot // BLOCK_SIZE * cache_block_stride
        + head * cache_head_stride
        + slot % BLOCK_SIZE * cache_token_stride
    )
    kr = tl.arange(0, 128)
    kd = tl.where(kr < 64, kr, kr + 192)
    k = tl.load(key + token * key_token_stride + head * key_head_stride + kd)
    vd = tl.arange(0, 512)
    v = tl.load(value + token * value_token_stride + head * value_head_stride + vd)
    tl.store(cache + base + kr, k)
    tl.store(cache + base + 128 + vd, v)


def write_compact_cache(key, value, cache, slots):
    """Write paged BF16 cache; negative slots are graph-padding tokens."""
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError("Gemma4 compact KV requires BF16 keys and values")
    if cache.dtype != torch.bfloat16 or cache.shape[-1] != 640:
        raise ValueError("Gemma4 compact KV requires a BF16 640-element row")
    if key.shape[-1] != 512 or value.shape != key.shape:
        raise ValueError("Gemma4 compact KV requires matching 512-element heads")
    if key.stride(-1) != 1 or value.stride(-1) != 1 or cache.stride(-1) != 1:
        raise ValueError("Gemma4 compact KV requires contiguous head dimensions")
    _write_compact_cache[(slots.numel(), key.shape[1])](
        key,
        value,
        cache,
        slots,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.shape[2],
    )
