# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Untimed initialization and cache conditioning for the VMM benchmark."""

from vllm.triton_utils import tl, triton


@triton.jit
def copy_bytes(source, destination, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offsets, offsets < size, other=0)
    tl.store(destination + offsets, value, offsets < size)


@triton.jit
def evict_cache(destination, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = offsets.to(tl.uint32) * 0x9E3779B9
    value = (value ^ (value >> 16)) * 0x85EBCA6B
    tl.store(destination + offsets, value, offsets < size)
