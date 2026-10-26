# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 FP8 GEMM M-band dispatch (VLLM_SM120_FP8_CUBLASLT_M_RANGE).

Run `pytest tests/kernels/quantization/test_sm120_banded_fp8_scaled_mm.py`.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.is_device_capability_family(
    120
):
    pytest.skip("SM120-only dispatch", allow_module_level=True)

# Registers torch.ops.vllm.sm120_banded_fp8_scaled_mm.
import vllm.model_executor.kernels.linear.scaled_mm.cutlass  # noqa: E402, F401


@pytest.mark.parametrize("m", [1, 16, 64, 65, 128, 200, 256, 257, 1024])
@pytest.mark.parametrize("n,k", [(9216, 7168), (7168, 22016)])
def test_banded_matches_cutlass(m: int, n: int, k: int):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    a, scale_a = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    w_q, scale_b = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)
    b = w_q.t()  # [K, N] column-major, as the linear layer stores it
    expected = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16)
    actual = torch.ops.vllm.sm120_banded_fp8_scaled_mm(a, b, scale_a, scale_b, 64, 256)
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    if not 64 < m <= 256:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_banded_under_cuda_graph():
    torch.manual_seed(0)
    k, n = 7168, 7168
    w_q, scale_b = ops.scaled_fp8_quant(
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5,
        use_per_token_if_dynamic=True,
    )
    b = w_q.t()
    for m in (32, 128):
        a, scale_a = ops.scaled_fp8_quant(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            use_per_token_if_dynamic=True,
        )
        eager = torch.ops.vllm.sm120_banded_fp8_scaled_mm(
            a, b, scale_a, scale_b, 64, 256
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = torch.ops.vllm.sm120_banded_fp8_scaled_mm(
                a, b, scale_a, scale_b, 64, 256
            )
        graph.replay()
        torch.testing.assert_close(out, eager, rtol=0, atol=0)
