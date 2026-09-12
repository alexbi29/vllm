# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from tests.v1.attention.utils import create_vllm_config
from vllm.config import MultiModalConfig, SpeculativeConfig
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@pytest.fixture
def model_path(tmp_path):
    LlamaConfig(architectures=["LlamaForCausalLM"]).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("adaptive", [False, True])
def test_sparse_swa_adaptive_varlen_graph_support(adaptive):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=adaptive)
    )
    support = DeepseekSparseSWAMetadataBuilder.get_cudagraph_support(config, None)
    assert support == (
        AttentionCGSupport.ALWAYS if adaptive else AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize(
    "vision_layers,mm_kwargs,expected_width",
    [
        (0, None, 128),
        (32, None, 512),
        (32, {}, 512),
        (32, {"language_model_only": True}, 128),
        (32, {"limit_per_prompt": {"image": 0}}, 128),
        (32, {"limit_per_prompt": {"image": 1}}, 512),
    ],
)
def test_sparse_swa_prefill_width_respects_image_limits(
    model_path, vision_layers, mm_kwargs, expected_width
):
    """Disabled images must not force an unsupported packed-prefill width."""
    config = create_vllm_config(
        model_name=model_path,
        block_size=256,
        hf_config_override={
            "sliding_window": 128,
            "compress_ratios": [1, 4, 128],
            "vision_n_layers": vision_layers,
            "vision_max_n_token": 384,
        },
    )
    config.model_config.multimodal_config = (
        MultiModalConfig(**mm_kwargs) if mm_kwargs is not None else None
    )
    builder = DeepseekSparseSWAMetadataBuilder(
        kv_cache_spec=MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            tokens_per_state=4,
        ),
        layer_names=["dummy"],
        vllm_config=config,
        device=torch.device("cpu"),
    )
    assert builder.prefill_index_width == expected_width
    assert builder.prefill_swa_indices.shape[-1] == expected_width
    assert builder.max_image_tokens == expected_width - 128
    assert hasattr(builder, "left_visible") == (expected_width > 128)


def test_sparse_swa_opts_out_of_reorder_batch_vote(model_path):
    vllm_config = create_vllm_config(
        model_name=model_path,
        block_size=256,
        hf_config_override={
            "sliding_window": 128,
            "compress_ratios": [1, 4, 128],
        },
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=2,
    )
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )

    builder = DeepseekSparseSWAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    assert builder.decode_threshold == 3
    # sparse_swa deliberately opts OUT of the runner's reorder-batch vote
    # (upstream #47327): the runner reduces thresholds with min_none_high, so
    # publishing a real threshold here would drag flashmla_sparse's 128-1024
    # dense-MHA routing threshold down for the whole model. indexer.py uses the
    # same None opt-out. decode_threshold above stays the local spec value.
    assert builder.reorder_batch_threshold is None
