# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import hc_down_silu
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="HC kernels require CUDA and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6
LORA_RANK = 320
DOWN_N = LORA_RANK + HC + 12  # merged down+inject weight, 16-row padded

requires_sm90 = pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="fused HC down+SiLU requires SM90+",
)


def test_grouped_gemma_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)

    grouped = x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected = grouped * torch.rsqrt(variance + EPS)
    expected = expected.flatten(-2) * (1.0 + weight.float())
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_gate_mix() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_gate_mix(x, gate, HC)
    expected = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN_SIZE)))
        * x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


@pytest.mark.skipif(not current_platform.is_device_capability(120), reason="SM120 plan")
@pytest.mark.parametrize("rows", [6912, 8191, 8192])
@pytest.mark.parametrize("strided", [False, True])
def test_hc_up_mix_keeps_bf16_gate_rounding(rows: int, strided: bool) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.hc_up_mix import hc_up_mix

    torch.manual_seed(42)
    padding = 16 if strided else 0
    a = torch.randn(rows, LORA_RANK + padding, device="cuda").bfloat16() * 0.1
    x = torch.randn(rows, HYPER_HIDDEN_SIZE + padding, device="cuda").bfloat16()
    a = a[:, padding // 2 : padding // 2 + LORA_RANK]
    x = x[:, padding // 2 : padding // 2 + HYPER_HIDDEN_SIZE]
    w = torch.randn(HYPER_HIDDEN_SIZE, LORA_RANK, device="cuda").bfloat16() * 0.05
    actual = hc_up_mix(a, w, x)
    gate = torch.nn.functional.linear(a, w)
    expected = hc_gate_mix(x, gate, HC)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.015, rtol=0.02)
    delta = actual.float() - expected.float()
    nrmse = (delta.square().mean() / expected.float().square().mean()).sqrt()
    assert nrmse.item() < 0.005


@pytest.mark.skipif(not current_platform.is_device_capability(120), reason="SM120 plan")
def test_hc_up_mix_graph_reads_changed_inputs() -> None:
    from vllm.models.qwen4_exp.nvidia.ops.hc_up_mix import hc_up_mix

    torch.manual_seed(43)
    a = torch.randn(6912, LORA_RANK, device="cuda").bfloat16() * 0.1
    x = torch.randn(6912, HYPER_HIDDEN_SIZE, device="cuda").bfloat16()
    w = torch.randn(HYPER_HIDDEN_SIZE, LORA_RANK, device="cuda").bfloat16() * 0.05
    hc_up_mix(a, w, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = hc_up_mix(a, w, x)
    for factor in (0.5, -1.5):
        a.mul_(factor)
        x.neg_()
        actual.fill_(float("nan"))
        graph.replay()
        expected = hc_gate_mix(x, torch.nn.functional.linear(a, w), HC)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=0.015, rtol=0.02
        )


def test_hc_combine() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, injection, HC)
    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)

    torch.testing.assert_close(actual, expected.flatten(-2).to(torch.bfloat16))


def test_hc_combine_unit_injection() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, None, HC)
    expected = residual.unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.unsqueeze(-2)

    assert torch.equal(actual, expected.flatten(-2))


def test_hc_combine_norm() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        residual, block_output, injection, weight, EPS, HC
    )

    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)
    expected = expected.flatten(-2).to(residual.dtype)
    grouped = expected.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected_norm = grouped * torch.rsqrt(variance + EPS)
    expected_norm = expected_norm.flatten(-2) * (1.0 + weight.float())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm.to(torch.bfloat16))


@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
def test_hc_combine_norm_unit_injection(num_tokens: int) -> None:
    torch.manual_seed(0)
    embedding = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    hidden = torch.randn(
        num_tokens, HC, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        hidden.flatten(1), embedding, None, weight, EPS, HC
    )

    expected = (hidden + embedding.unsqueeze(1)).flatten(1)
    expected_norm = grouped_gemma_rmsnorm(expected, weight, EPS, HC)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm)


@requires_sm90
@pytest.mark.parametrize("num_tokens", [1, 3, 5, 17, 48])
def test_hc_down_silu_fused(num_tokens: int) -> None:
    # Compare computed columns with the unfused ll_bf16 + hc_silu reference.
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    torch.manual_seed(0)
    x = torch.randn(num_tokens, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(DOWN_N, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    lora, injection = hc_down_silu(x, weight, LORA_RANK, HC)

    down = ll_bf16_gemm(x, weight).to(torch.bfloat16)
    torch.testing.assert_close(
        lora, hc_silu(down[:, :LORA_RANK], HC), rtol=0.01, atol=0.01
    )
    torch.testing.assert_close(
        injection, down[:, LORA_RANK : LORA_RANK + HC], rtol=0.01, atol=0.01
    )
