# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired reconstruction preserving DeepSeek's existing one-bit codec."""

from vllm.model_executor.layers.quantization.utils.mxfp4_csf import (
    _compact_active_experts,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _decode_one(
    Bases,
    Selectors,
    Exceptions,
    Offsets,
    Output,
    ExpertIds,
    ROWS: tl.constexpr,
    WORDS: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    TILE_OFFSET: tl.constexpr,
):
    expert = tl.load(ExpertIds + tl.program_id(0))
    tile = tl.program_id(1) - TILE_OFFSET
    if expert >= 0:
        row = (tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS))[None, :]
        word = tl.arange(0, BLOCK_K)[:, None]
        base = tl.load(Bases + expert * ROWS + row, row < ROWS, 0).to(tl.uint32)
        bits = tl.load(
            Selectors + (expert * ROWS + row) * tl.cdiv(WORDS, 2) + word // 2,
            (row < ROWS) & (word < WORDS),
            0,
        ).to(tl.uint32)
        nibble = (bits >> ((word % 2) * 4)) & 15
        packed = base * 0x01010101
        packed += (nibble & 1) | ((nibble & 2) << 7)
        packed += ((nibble & 4) << 14) | ((nibble & 8) << 21)
        index = (expert * WORDS + word) * ALIGNED_ROWS + row
        tl.store(Output + index, packed, (row < ROWS) & (word < WORDS))
        tl.debug_barrier()
        partition = expert * TILES + tile
        first = tl.load(Offsets + partition)
        last = tl.load(Offsets + partition + 1)
        lane = tl.arange(0, 128)
        byte_output = Output.to(tl.pointer_type(tl.uint8))
        for start in range(first, last, 128):
            record = tl.load(Exceptions + start + lane, start + lane < last, 0)
            position = record & 0xFFFFFF
            r = position // (WORDS * 4)
            col = position % (WORDS * 4)
            byte_index = ((expert * WORDS + col // 4) * ALIGNED_ROWS + r) * 4
            byte_index += col % 4
            tl.store(byte_output + byte_index, record >> 24, start + lane < last)


@triton.jit
def _decode_pair(
    B1,
    S1,
    X1,
    F1,
    O1,
    B2,
    S2,
    X2,
    F2,
    O2,
    Ids,
    R1: tl.constexpr,
    W1: tl.constexpr,
    A1: tl.constexpr,
    R2: tl.constexpr,
    W2: tl.constexpr,
    A2: tl.constexpr,
    TILE: tl.constexpr,
):
    first: tl.constexpr = tl.cdiv(R1, TILE)
    if tl.program_id(1) < first:
        _decode_one(
            B1,
            S1,
            X1,
            F1,
            O1,
            Ids,
            R1,
            W1,
            A1,
            first,
            triton.next_power_of_2(W1),
            TILE,
            0,
        )
    else:
        _decode_one(
            B2,
            S2,
            X2,
            F2,
            O2,
            Ids,
            R2,
            W2,
            A2,
            tl.cdiv(R2, TILE),
            triton.next_power_of_2(W2),
            TILE,
            first,
        )


def decode_pair(items, outputs, counts, ids, slots):
    if not slots:
        return
    first, second = items
    _compact_active_experts[(1,)](
        counts, ids, first.shape[0], triton.next_power_of_2(first.shape[0])
    )
    _, r1, w1 = first.shape
    _, r2, w2 = second.shape
    _decode_pair[(slots, triton.cdiv(r1, 32) + triton.cdiv(r2, 32))](
        first.bases,
        first.selectors,
        first.exceptions,
        first.offsets,
        outputs[0],
        second.bases,
        second.selectors,
        second.exceptions,
        second.offsets,
        outputs[1],
        ids,
        r1,
        w1,
        first.stride[2],
        r2,
        w2,
        second.stride[2],
        32,
        num_warps=4,
    )
