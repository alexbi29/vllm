# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig

from vllm.config import ModelConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.models.qwen4_exp.nvidia.weight_placement import (
    move_embedding_to_host,
    quantize_dense_fp8,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda_only
def test_host_embedding_lookups_are_exact(default_vllm_config, dist_init):
    embed = VocabParallelEmbedding(1000, 256).to("cuda", torch.bfloat16)
    embed.weight.data.normal_()
    ids = torch.randint(0, 1000, (4096,), device="cuda")
    ref = embed(ids).clone()
    before = torch.accelerator.memory_allocated()
    table_bytes = embed.weight.numel() * embed.weight.element_size()

    moved = move_embedding_to_host(embed)

    assert moved == table_bytes
    assert embed._host_weight.is_pinned()
    assert embed.weight.is_cuda
    assert torch.accelerator.memory_allocated() <= before - moved
    assert torch.equal(embed(ids), ref)
    # CUDA graphs keep reading the same host-mapped rows.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = embed(ids)
    graph.replay()
    assert torch.equal(out, ref)
    assert move_embedding_to_host(embed) == 0


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Module()
        self.a.proj = ReplicatedLinear(
            512, 1024, bias=False, params_dtype=torch.bfloat16
        )
        self.b = nn.Module()
        self.b.proj = ReplicatedLinear(
            512, 1024, bias=False, params_dtype=torch.bfloat16
        )


@cuda_only
def test_fp8_dense_converts_only_matching_projections(
    default_vllm_config, dist_init, monkeypatch
):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    model = _Model().to("cuda")
    for linear in (model.a.proj, model.b.proj):
        linear.weight.data.normal_(std=0.02)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref_a, _ = model.a.proj(x)
    ref_b, _ = model.b.proj(x)

    converted = quantize_dense_fp8(model, r"^a\.proj$")

    assert converted == ["a.proj"]
    assert model.a.proj.weight.dtype == torch.float8_e4m3fn
    assert model.a.proj.weight.untyped_storage().nbytes() == (
        model.a.proj.weight.numel() * model.a.proj.weight.element_size()
    )
    assert type(model.b.proj.quant_method) is UnquantizedLinearMethod
    out_a, _ = model.a.proj(x)
    out_b, _ = model.b.proj(x)
    assert torch.equal(out_b, ref_b)
    rel = ((out_a.float() - ref_a.float()).norm() / ref_a.float().norm()).item()
    assert 0 < rel < 0.06


@cuda_only
def test_fp8_dense_rejects_unknown_kernel(default_vllm_config, dist_init):
    with pytest.raises(ValueError, match="Unknown FP8"):
        quantize_dense_fp8(_Model().to("cuda"), r"proj", "NoSuchKernel")


@pytest.mark.skipif(
    not CUTLASS_BLOCK_FP8_SUPPORTED,
    reason="requires a CUDA build with block-FP8 kernel selection",
)
def test_fp8_dense_releases_bf16_storage_on_cpu(default_vllm_config, tmp_path):
    """Post-load placement shrinks storage without changing bytes or loaders.

    Kernel selection uses the CUDA build, but conversion and parameter storage
    are CPU tensors; no CUDA context or matrix multiplication is required.
    """
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        vocab_size=256,
    ).save_pretrained(tmp_path)
    default_vllm_config.model_config = ModelConfig(
        model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
    )
    cuda_initialized = torch.cuda.is_initialized()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp_path / 'dist'}",
        backend="gloo",
    )
    initialize_model_parallel(1, 1, backend="gloo")
    try:
        model = nn.Sequential(
            ReplicatedLinear(
                128, 256, bias=False, params_dtype=torch.bfloat16, disable_tp=True
            )
        )
        checkpoint = (
            torch.arange(128 * 256, dtype=torch.float32)
            .remainder(8)
            .reshape(256, 128)
            .div(8)
            .to(torch.bfloat16)
        )
        layer = model[0]
        layer.weight.weight_loader(layer.weight, checkpoint)
        parameter = layer.weight
        loader = parameter.weight_loader
        bf16_bytes = parameter.untyped_storage().nbytes()
        # Every block has max=7/8, so its exact E4M3 scale is (7/8)/448=1/512.
        expected_weight = (checkpoint * 512).to(torch.float8_e4m3fn)
        expected_scale = torch.full((2, 1), 1 / 512, dtype=torch.float32)

        with torch.compiler.set_stance("force_eager"):
            converted = quantize_dense_fp8(
                model, r"^0$", "CutlassFp8BlockScaledMMKernel"
            )

        assert converted == ["0"]
        assert layer.weight is parameter
        assert layer.weight.weight_loader == loader
        assert layer.weight.device.type == "cpu"
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight.untyped_storage().nbytes() == bf16_bytes // 2
        assert torch.equal(
            layer.weight.view(torch.uint8), expected_weight.view(torch.uint8)
        )
        assert torch.equal(layer.weight_scale_inv, expected_scale)
        assert torch.cuda.is_initialized() == cuda_initialized
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
