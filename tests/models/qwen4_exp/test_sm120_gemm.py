# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.sm120_gemm import (
    SM120_PROJECTION_PLANS,
    sm120_projection_gemm,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(120), reason="SM120 BF16 plans"
)


@pytest.mark.parametrize("n,k,m", list(SM120_PROJECTION_PLANS))
@pytest.mark.parametrize("strided", [False, True])
def test_projection_plan_matches_bf16_linear(n, k, m, strided):
    torch.manual_seed(43)
    padding = 16 if strided else 0
    a = torch.randn(m, k + padding, device="cuda").bfloat16() * 0.1
    a = a[:, padding // 2 : padding // 2 + k]
    w = torch.randn(n, k, device="cuda").bfloat16() * 0.1
    actual = sm120_projection_gemm(a, w)
    expected = torch.nn.functional.linear(a, w)
    assert actual is not None
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.015, rtol=0.02)
    delta = actual.float() - expected.float()
    nrmse = (delta.square().mean() / expected.float().square().mean()).sqrt()
    assert nrmse.item() < 0.005


@pytest.mark.parametrize("enabled", [False, True])
def test_projection_dispatch_graph_reads_changed_input(enabled):
    torch.manual_seed(44)
    a = torch.randn(16, 6144, device="cuda").bfloat16() * 0.1
    w = torch.randn(2560, 6144, device="cuda").bfloat16() * 0.1
    op = torch.ops.vllm.qwen4_exp_low_latency_gemm
    op(a, w, enabled)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(a, w, enabled)
    for factor in (0.5, -1.5):
        a.mul_(factor)
        actual.fill_(float("nan"))
        graph.replay()
        expected = torch.nn.functional.linear(a, w)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=0.015, rtol=0.02
        )
