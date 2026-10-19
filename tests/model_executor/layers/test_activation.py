# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.platforms import CpuArchEnum, current_platform


@pytest.mark.skipif(
    not current_platform.is_cpu()
    or current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC,
    reason="Native CPU activations require a non-PowerPC CPU platform",
)
@pytest.mark.parametrize(
    "activation_factory,reference",
    [
        (SiluAndMul, nn.SiLU()),
        (GeluAndMul, nn.GELU()),
        (partial(GeluAndMul, approximate="tanh"), nn.GELU(approximate="tanh")),
    ],
    ids=["silu", "gelu", "gelu_tanh"],
)
def test_cpu_act_and_mul_without_compiled_ops(
    default_vllm_config, monkeypatch, activation_factory, reference
):
    """CPU activations must construct and run without compiled wheel operators."""
    default_vllm_config.compilation_config.custom_ops = ["all"]
    monkeypatch.setattr(torch.ops, "_C", object())
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], device="cpu")

    layer = activation_factory()

    torch.testing.assert_close(layer(x), reference(x[:, :4]) * x[:, 4:])


@pytest.mark.parametrize(
    "requested,expected",
    [("GELU", "Geglu"), ("GELU_TANH", "GegluTanh")],
)
def test_flashinfer_preserves_gated_gelu_approximation(requested, expected):
    """The external activation ABI must retain the checkpoint's GELU variant."""
    flashinfer = pytest.importorskip("flashinfer.fused_moe.core")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_type,
    )

    activation = MoEActivation[requested]
    if not hasattr(flashinfer.ActivationType, expected):
        with pytest.raises(ValueError, match="GELU.*tanh"):
            activation_to_flashinfer_type(activation)
    else:
        assert activation_to_flashinfer_type(activation) == getattr(
            flashinfer.ActivationType, expected
        )
