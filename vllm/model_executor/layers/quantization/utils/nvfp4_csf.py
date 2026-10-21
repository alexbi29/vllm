# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless four-bit scale deltas with exact sparse byte exceptions."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton


def swizzle_reference(values):
    experts, rows, columns = values.shape
    padded_rows = triton.cdiv(rows, 128) * 128
    padded_columns = triton.cdiv(columns, 4) * 4
    padded = np.zeros((experts, padded_rows, padded_columns), dtype=np.uint8)
    padded[:, :rows, :columns] = values
    return (
        padded.reshape(experts, padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .transpose(0, 1, 4, 3, 2, 5)
        .copy()
        .reshape(padded.shape)
    )


@dataclass
class PaletteScales:
    """Fixed four-bit deltas in native order, with exact sparse byte exceptions."""

    bases: np.ndarray
    selectors: np.ndarray
    exceptions: np.ndarray
    offsets: np.ndarray
    shape: tuple[int, int, int]
    tile: int = 4096

    @classmethod
    def encode(cls, values, tile=4096):
        return cls.from_native(swizzle_reference(values), tile)

    @classmethod
    def from_native(cls, values, tile=4096):
        if values.dtype != np.uint8 or values.ndim != 3:
            raise ValueError("Expected native uint8 [experts, rows, columns]")
        experts, rows, columns = values.shape
        size = rows * columns
        if size >= 1 << 24 or size % 2 or tile % 4:
            raise ValueError("Palette requires even sizes <2^24 and word-aligned tiles")
        tiles = triton.cdiv(size, tile)
        flat = np.ascontiguousarray(values).reshape(experts, size)
        bases = np.empty(experts, dtype=np.uint8)
        selectors = np.empty((experts, size // 2), dtype=np.uint8)
        records = []
        counts = np.zeros(experts * tiles, dtype=np.int64)
        for e, v in enumerate(flat):
            hist = np.bincount(v, minlength=256)
            prefix = np.concatenate(([0], np.cumsum(hist)))
            base = int((prefix[16:] - prefix[:-16]).argmax())
            bases[e] = base
            delta = (v.astype(np.int16) - base) & 15
            selectors[e] = delta[::2] | (delta[1::2] << 4)
            positions = np.flatnonzero((v < base) | (v.astype(np.int16) > base + 15))
            records.append(
                positions.astype(np.uint32) | (v[positions].astype(np.uint32) << 24)
            )
            counts[e * tiles : (e + 1) * tiles] = np.bincount(
                positions // tile, minlength=tiles
            )
        offsets = np.concatenate(([0], np.cumsum(counts)))
        if offsets[-1] >= 1 << 31:
            raise ValueError("Palette exception offsets exceed int32")
        return cls(
            bases,
            selectors,
            np.concatenate(records),
            offsets.astype(np.int32),
            tuple(values.shape),
            tile,
        )

    @property
    def nbytes(self):
        return sum(
            a.nbytes
            for a in (self.bases, self.selectors, self.exceptions, self.offsets)
        )

    def decode_cpu(self):
        experts, rows, columns = self.shape
        size, tiles = rows * columns, triton.cdiv(rows * columns, self.tile)
        out = np.empty((experts, size), dtype=np.uint8)
        out[:, ::2] = self.bases[:, None] + (self.selectors & 15)
        out[:, 1::2] = self.bases[:, None] + (self.selectors >> 4)
        for e in range(experts):
            records = self.exceptions[
                self.offsets[e * tiles] : self.offsets[(e + 1) * tiles]
            ]
            out[e, records & 0xFFFFFF] = records >> 24
        return out.reshape(self.shape)

    def to_gpu(self, device):
        bases, selectors, exceptions, offsets = (
            torch.from_numpy(a).to(device)
            for a in (self.bases, self.selectors, self.exceptions, self.offsets)
        )
        return GpuPaletteScales(
            bases, selectors, exceptions, offsets, self.shape, self.tile
        )


@triton.jit
def _decode_palette(
    Bases,
    Selectors,
    Exceptions,
    Offsets,
    Output,
    Ids,
    EXPERTS: tl.constexpr,
    SIZE: tl.constexpr,
    TILE: tl.constexpr,
    TILE_OFFSET: tl.constexpr = 0,
):
    expert = tl.load(Ids + tl.program_id(0))
    tile = tl.program_id(1) - TILE_OFFSET
    if (expert >= 0) & (expert < EXPERTS):
        positions = tile * TILE + tl.arange(0, TILE)
        base = tl.load(Bases + expert).to(tl.uint32)
        packed = tl.load(
            Selectors + expert * (SIZE // 2) + positions // 2, positions < SIZE, 0
        ).to(tl.uint32)
        value = base + ((packed >> ((positions % 2) * 4)) & 15)
        tl.store(Output + expert * SIZE + positions, value, positions < SIZE)
        tl.debug_barrier()
        partition = expert * tl.cdiv(SIZE, TILE) + tile
        first = tl.load(Offsets + partition)
        last = tl.load(Offsets + partition + 1)
        lane = tl.arange(0, 128)
        for index in range(first, last, 128):
            record = tl.load(Exceptions + index + lane, index + lane < last, 0)
            tl.store(
                Output + expert * SIZE + (record & 0xFFFFFF),
                record >> 24,
                index + lane < last,
            )


@dataclass
class GpuPaletteScales:
    bases: torch.Tensor
    selectors: torch.Tensor
    exceptions: torch.Tensor
    offsets: torch.Tensor
    shape: tuple[int, int, int]
    tile: int

    def decode(self, output, expert_ids, *, warps=4):
        if tuple(output.shape) != self.shape or output.dtype != torch.uint8:
            raise ValueError("Native output shape/dtype mismatch")
        if not output.is_contiguous() or not expert_ids.is_contiguous():
            raise ValueError("Native output and IDs must be contiguous")
        if output.device != self.selectors.device or expert_ids.device != output.device:
            raise ValueError("Payload, output and IDs must share the codec GPU")
        if expert_ids.dtype != torch.int32:
            raise ValueError("Expert IDs must be int32")
        if not expert_ids.numel():
            return
        experts, rows, columns = self.shape
        size = rows * columns
        _decode_palette[(expert_ids.numel(), triton.cdiv(size, self.tile))](
            self.bases,
            self.selectors,
            self.exceptions,
            self.offsets,
            output,
            expert_ids,
            experts,
            size,
            self.tile,
            num_warps=warps,
        )


@triton.jit
def _decode_palette_pair(
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
    EXPERTS: tl.constexpr,
    SIZE1: tl.constexpr,
    SIZE2: tl.constexpr,
    TILE: tl.constexpr,
):
    first: tl.constexpr = tl.cdiv(SIZE1, TILE)
    if tl.program_id(1) < first:
        _decode_palette(B1, S1, X1, F1, O1, Ids, EXPERTS, SIZE1, TILE)
    else:
        _decode_palette(B2, S2, X2, F2, O2, Ids, EXPERTS, SIZE2, TILE, first)


def decode_palette_pair(codecs, outputs, expert_ids, *, warps=4):
    first, second = codecs
    if first.shape[0] != second.shape[0] or first.tile != second.tile:
        raise ValueError("Palette pair must share experts and decoder tile")
    for codec, output in zip(codecs, outputs):
        if tuple(output.shape) != codec.shape or output.dtype != torch.uint8:
            raise ValueError("Native output shape/dtype mismatch")
        if not output.is_contiguous() or output.device != codec.selectors.device:
            raise ValueError("Output must be contiguous on the codec GPU")
    if expert_ids.device != outputs[0].device or expert_ids.dtype != torch.int32:
        raise ValueError("Expert IDs must be int32 on the codec GPU")
    if not expert_ids.is_contiguous():
        raise ValueError("Expert IDs must be contiguous")
    if not expert_ids.numel():
        return
    size1, size2 = first.shape[1] * first.shape[2], second.shape[1] * second.shape[2]
    _decode_palette_pair[
        (
            expert_ids.numel(),
            triton.cdiv(size1, first.tile) + triton.cdiv(size2, first.tile),
        )
    ](
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
        expert_ids,
        first.shape[0],
        size1,
        size2,
        first.tile,
        num_warps=warps,
    )
