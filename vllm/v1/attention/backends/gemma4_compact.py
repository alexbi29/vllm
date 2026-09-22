# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in BF16 compact global KV for Gemma4 26B/31B; no dense cache expansion."""

from copy import copy
from dataclasses import replace
from math import lcm
from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.gemma4_compact_cache import write_compact_cache
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVQuantMode

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import Attention


def share_compact_kv_with_draft(
    draft: "Attention", target: "Attention", target_layer_name: str
) -> None:
    """Give either model runner's MTP reader the target's cache representation."""
    if not hasattr(target.impl, "compact_k_norm"):
        return
    if (
        draft.head_size != target.head_size
        or draft.num_kv_heads != target.num_kv_heads
        or draft.kv_cache_dtype != target.kv_cache_dtype
    ):
        raise ValueError("Gemma4 MTP compact KV geometry mismatch")
    draft.attn_backend = target.attn_backend
    draft.backend = target.backend
    draft_scale = draft.impl.scale
    draft_impl = copy(cast("Gemma4CompactImpl", target.impl))
    draft_impl.num_heads = draft.num_heads
    draft_impl.num_queries_per_kv = draft.num_heads // draft.num_kv_heads
    draft_impl.scale = draft_scale
    draft_impl.kv_sharing_target_layer_name = target_layer_name
    draft.impl = draft_impl


def validate_compact_kv_dtype(
    cache_dtype: str, activation_dtype: torch.dtype, quant_config=None
) -> None:
    """Reject quantized KV, including checkpoint-selected defaults, before loading."""
    checkpoint_kv = (
        getattr(quant_config, "kv_cache_scheme", None)
        or getattr(quant_config, "kv_cache_quant_method", None)
        or getattr(quant_config, "kv_cache_quant_algo", None)
    )
    if cache_dtype not in ("auto", "bfloat16") or (
        cache_dtype == "auto" and checkpoint_kv not in (None, "", "none", "NONE")
    ):
        raise ValueError(
            "gemma4_compact_kv does not support FP8 or other quantized KV caches "
            f"(kv_cache_dtype={cache_dtype!r}, checkpoint KV={checkpoint_kv!r}). "
            "FP8/NVFP4 model weights require BF16 activations and "
            "--kv-cache-dtype bfloat16 with compact KV. Disable "
            "gemma4_compact_kv to use FP8 KV; no automatic fallback is performed."
        )
    if activation_dtype != torch.bfloat16:
        raise ValueError(
            "gemma4_compact_kv requires BF16 activations even with FP8/NVFP4 "
            f"weights; got {activation_dtype}. Use --dtype bfloat16."
        )


class Gemma4CompactMetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Gemma has 256-dimensional local heads and 512-dimensional global
        # heads. The generic builder uses the model-wide (local) head size.
        shape = (*self.softmax_segm_output.shape[:-1], kv_cache_spec.head_size)
        self.softmax_segm_output = torch.empty(
            shape, dtype=torch.float32, device=device
        )


class Gemma4CompactBackend(TritonAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name():
        return "GEMMA4_COMPACT"

    @staticmethod
    def get_impl_cls():
        return Gemma4CompactImpl

    @staticmethod
    def get_builder_cls():
        return Gemma4CompactMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size):
        return head_size == 512

    @classmethod
    def supports_sliding_window(cls):
        return False

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if (
            not isinstance(spec, FullAttentionSpec)
            or spec.head_size != 512
            or spec.head_size_v != 512
            or spec.dtype != torch.bfloat16
            or spec.kv_quant_mode != KVQuantMode.NONE
        ):
            raise ValueError("Compact Gemma4 KV requires BF16 global 512-d attention")
        # Gemma26B/31B local layers have four times as many KV heads and
        # 256-d K/V. Their bytes/token : compact global bytes/token is 16:5.
        # Global blocks of 256 and local blocks of 80 share a page size
        # without padding away the compression. The generic hybrid planner
        # grows the local blocks from 16 to 80 during page unification.
        return replace(
            spec, block_size=lcm(spec.block_size, 256), state_content_bytes=640 * 2
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        validate_compact_kv_dtype(cache_dtype_str, torch.bfloat16)
        if head_size != 512:
            raise ValueError("Compact Gemma4 KV only supports BF16 512-d heads")
        if block_size % 16:
            raise ValueError("Block size must be a multiple of 16")
        return num_blocks, num_kv_heads, block_size, 640


class Gemma4CompactImpl(TritonAttentionImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        *args,
        compact_k_norm,
        **kwargs,
    ):
        validate_compact_kv_dtype(kv_cache_dtype, compact_k_norm.dtype)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            *args,
            **kwargs,
        )
        if (
            self.head_size != 512
            or self.kv_cache_dtype not in ("auto", "bfloat16")
            or self.sliding_window != (-1, -1)
            or compact_k_norm.dtype != torch.bfloat16
            or compact_k_norm.shape != (512,)
        ):
            raise ValueError("Compact Gemma4 KV requires BF16 global 512-d attention")
        self.compact_k_norm = compact_k_norm
        self.use_td = False

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        write_compact_cache(key, value, kv_cache, slot_mapping)

    def fused_rope_kvcache_supported(self):
        return False
