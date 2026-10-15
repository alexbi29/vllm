# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Row-base/selector encoding adapted from Local Inference Lab's Apache-2.0
# b12x/_lib/quant/mxfp4_csf_inline.py at 52640cb15d4a. Modified to retain
# DeepGEMM's native exponent bytes and layout, without B12X dependencies.

"""Lossless resident compression of native DeepGEMM MXFP4 scale words."""

from dataclasses import dataclass

import numpy as np
import regex as re
import torch

from vllm.triton_utils import tl, triton

_ROWS = 16


def should_compress_mxfp4_scales(layer: torch.nn.Module) -> bool:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or not config.kernel_config.mxfp4_scale_compression:
        return False
    from vllm.compilation import backends

    if backends.model_tag != "backbone":
        return False
    # Speculative layers retain their own scales and backend selection.
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
    return not (
        match and int(match[1]) >= config.model_config.hf_text_config.num_hidden_layers
    )


def reject_compressed_scale_export(module, prefix, keep_vars):
    raise RuntimeError("State export is unsupported with MXFP4 scale compression")


@triton.jit
def _compact_active_experts(Counts, Ids, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.arange(0, BLOCK)
    active = tl.load(Counts + expert, expert < EXPERTS, 0) > 0
    position = tl.cumsum(active.to(tl.int32)) - 1
    tl.store(Ids + expert, -1, expert < EXPERTS)
    tl.debug_barrier()
    tl.store(Ids + position, expert, active)


@triton.jit
def _decode_scales(
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
    ACTIVE_ONLY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    slot = tl.program_id(0)
    expert = tl.load(ExpertIds + slot) if ACTIVE_ONLY else slot
    tile = tl.program_id(1)
    if expert >= 0:
        row = tile * 16 + tl.arange(0, 16)
        word = tl.arange(0, BLOCK_K)
        base = tl.load(Bases + expert * ROWS + row, row < ROWS, 0).to(tl.uint32)
        # One selector byte covers two native four-scale words.
        bits = tl.load(
            Selectors
            + (expert * ROWS + row[:, None]) * tl.cdiv(WORDS, 2)
            + word[None, :] // 2,
            (row[:, None] < ROWS) & (word[None, :] < WORDS),
            0,
        ).to(tl.uint32)
        nibble = (bits >> ((word[None, :] % 2) * 4)) & 15
        packed = base[:, None] * 0x01010101
        packed += (nibble & 1) | ((nibble & 2) << 7)
        packed += ((nibble & 4) << 14) | ((nibble & 8) << 21)
        index = (expert * WORDS + word[None, :]) * ALIGNED_ROWS + row[:, None]
        tl.store(
            Output + index, packed, (row[:, None] < ROWS) & (word[None, :] < WORDS)
        )
        # Each CTA owns these rows, including every exception byte. Palette
        # stores must finish before unique byte stores patch their words.
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


@dataclass
class CompressedMxfp4Scales:
    """Native exponent bytes, with exception positions local to each expert."""

    bases: torch.Tensor
    selectors: torch.Tensor
    exceptions: torch.Tensor
    offsets: torch.Tensor
    shape: tuple[int, int, int]
    stride: tuple[int, int, int]

    @classmethod
    def from_native(cls, scales: torch.Tensor) -> "CompressedMxfp4Scales":
        if scales.dtype != torch.int32 or scales.ndim != 3:
            raise ValueError(
                "CSF requires native int32 [experts, rows, packedK] scales"
            )
        e, n, k = scales.shape
        stride = scales.stride()
        if stride[1] != 1 or stride[2] < n or stride[0] != k * stride[2]:
            raise ValueError("CSF requires DeepGEMM's MN-major scale layout")
        if n * k * 4 > 1 << 24:
            raise ValueError("CSF exception positions exceed 24 bits per expert")
        # Packing is load-time CPU work. Integer views preserve native bytes,
        # including 0 and 255; no float conversion, clamp or requantization.
        # Singleton dimensions can retain non-unit strides after contiguous().
        values = (
            scales.detach()
            .cpu()
            .clone(memory_format=torch.contiguous_format)
            .view(torch.uint8)
            .numpy()
        )
        values = values.reshape(e * n, k * 4)
        bases = np.empty(e * n, dtype=np.uint8)
        selectors = np.empty((e * n, (k + 1) // 2), dtype=np.uint8)
        records = []
        tile_counts = np.zeros(e * ((n + _ROWS - 1) // _ROWS), dtype=np.int64)
        for first in range(0, e * n, 8192):
            rows = values[first : first + 8192]
            indices = np.arange(len(rows), dtype=np.int64)[:, None] * 256 + rows
            histogram = np.bincount(indices.ravel(), minlength=len(rows) * 256)
            histogram = histogram.reshape(len(rows), 256)
            base = (histogram[:, :255] + histogram[:, 1:]).argmax(axis=1)
            bases[first : first + len(rows)] = base
            bits = rows == base[:, None] + 1
            selectors[first : first + len(rows)] = np.packbits(
                bits, axis=1, bitorder="little"
            )
            r, c = np.nonzero((rows != base[:, None]) & ~bits)
            global_row = first + r
            expert = global_row // n
            position = (global_row % n) * (k * 4) + c
            records.append(
                position.astype(np.uint32) | (rows[r, c].astype(np.uint32) << 24)
            )
            tile = expert * ((n + _ROWS - 1) // _ROWS) + (global_row % n) // _ROWS
            tile_counts += np.bincount(tile, minlength=len(tile_counts))
        offsets = np.concatenate(([0], np.cumsum(tile_counts)))
        if offsets[-1] > np.iinfo(np.int32).max:
            raise ValueError("CSF exception stream exceeds int32 offsets")
        device = scales.device

        def upload(array):
            return torch.from_numpy(array).to(device)

        return cls(
            upload(bases.reshape(e, n)),
            upload(selectors.reshape(e, n, (k + 1) // 2)),
            upload(np.concatenate(records)),
            upload(offsets.astype(np.int32)),
            tuple(scales.shape),
            tuple(stride),
        )

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.bases, self.selectors, self.exceptions, self.offsets)
        )

    def make_scratch(self) -> torch.Tensor:
        e, n, k = self.shape
        # The physical allocation includes trailing TMA alignment padding.
        storage = torch.zeros(
            (e, k, self.stride[2]), dtype=torch.int32, device=self.bases.device
        )
        return storage.as_strided(self.shape, self.stride)

    def decode(
        self,
        output: torch.Tensor,
        expert_counts: torch.Tensor | None = None,
        max_active_experts: int | None = None,
    ) -> None:
        if output.shape != self.shape or output.stride() != self.stride:
            raise ValueError("CSF scratch shape/stride mismatch")
        if output.dtype != torch.int32 or output.device != self.bases.device:
            raise ValueError("CSF scratch dtype/device mismatch")
        if expert_counts is not None and (
            expert_counts.numel() != self.shape[0]
            or expert_counts.device != output.device
            or expert_counts.ndim != 1
            or not expert_counts.is_contiguous()
            or expert_counts.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "CSF active counts require contiguous local integer counts"
            )
        e, n, k = self.shape
        expert_ids = None
        slots = e
        if expert_counts is not None:
            expert_ids = torch.empty(e, dtype=torch.int32, device=output.device)
            _compact_active_experts[(1,)](
                expert_counts, expert_ids, e, triton.next_power_of_2(e)
            )
            if max_active_experts is not None:
                slots = min(e, max_active_experts)
        if slots == 0:
            return
        _decode_scales[(slots, triton.cdiv(n, _ROWS))](
            self.bases,
            self.selectors,
            self.exceptions,
            self.offsets,
            output,
            expert_ids,
            n,
            k,
            self.stride[2],
            triton.cdiv(n, _ROWS),
            expert_counts is not None,
            triton.next_power_of_2(k),
            num_warps=4,
        )
