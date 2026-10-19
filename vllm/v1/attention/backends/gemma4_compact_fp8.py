# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in unit-scale FP8 global storage; local attention keeps BF16 KV."""

from dataclasses import replace
from math import lcm

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.gemma4_compact import (
    Gemma4CompactBackend,
    Gemma4CompactImpl,
)
from vllm.v1.attention.ops.gemma4_compact_cache import write_compact_cache


class Gemma4CompactFp8Backend(Gemma4CompactBackend):
    @staticmethod
    def get_name():
        return "GEMMA4_COMPACT_FP8"

    @staticmethod
    def get_impl_cls():
        return Gemma4CompactFp8Impl

    @classmethod
    def customize_spec(cls, spec):
        spec = super().customize_spec(spec)
        return replace(
            spec,
            block_size=lcm(spec.block_size, 512),
            dtype=torch.float8_e4m3fn,
            state_content_bytes=640,
        )


class Gemma4CompactFp8Impl(Gemma4CompactImpl):
    def __init__(self, *args, **kwargs):
        if not current_platform.is_device_capability_family(120):
            raise ValueError("Gemma4 compact FP8 storage is currently SM120-only")
        super().__init__(*args, **kwargs)
        self.compact_fp8_storage = True

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        write_compact_cache(key, value, kv_cache, slot_mapping, fp8_storage=True)
