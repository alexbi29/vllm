# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.models.qwen4_exp.nvidia.weight_placement import (
    move_embedding_to_host,
    quantize_dense_fp8,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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
    assert type(model.b.proj.quant_method) is UnquantizedLinearMethod
    out_a, _ = model.a.proj(x)
    out_b, _ = model.b.proj(x)
    assert torch.equal(out_b, ref_b)
    rel = ((out_a.float() - ref_a.float()).norm() / ref_a.float().norm()).item()
    assert 0 < rel < 0.06


def test_fp8_dense_rejects_unknown_kernel(default_vllm_config, dist_init):
    with pytest.raises(ValueError, match="Unknown FP8"):
        quantize_dense_fp8(_Model().to("cuda"), r"proj", "NoSuchKernel")
