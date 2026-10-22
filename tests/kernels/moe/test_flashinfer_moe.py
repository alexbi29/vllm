# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
    is_valid_flashinfer_cutlass_fused_moe,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    100
):
    pytest.skip(
        "Requires flashinfer_cutlass_fused_moe and nvfp4 support",
        allow_module_level=True,
    )

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.skipif(
    not current_platform.is_device_capability(120), reason="SM120 tactic plans"
)
@pytest.mark.parametrize("m", [4, 8, 16, 64, 2048, 6912, 8000, 8191, 8192])
@torch.inference_mode()
def test_qwen4_exp_resident_tactics_graph_replays_new_routes(
    m, monkeypatch, workspace_init
):
    """Tactics must track new routes and exactly repeat unchanged graph inputs."""
    from types import SimpleNamespace

    if pytest.importorskip("flashinfer").__version__ != "0.7.2":
        pytest.skip("The opt-in tactic plans require FlashInfer 0.7.2")

    from vllm.compilation import backends
    from vllm.model_executor.layers.quantization.utils import nvfp4_resident

    set_random_seed(20261021)
    config = VllmConfig()
    with set_current_vllm_config(config):
        w1, w2, quant = make_test_quant_config(
            512, 640, 2560, torch.bfloat16, "nvfp4", make_gate=True
        )
        moe = FusedMoEConfig(
            num_experts=512,
            experts_per_token=10,
            hidden_dim=2560,
            intermediate_size=640,
            num_local_experts=512,
            num_logical_experts=512,
            activation=MoEActivation.SILU,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=m,
        )
        experts = FlashInferExperts(moe_config=moe, quant_config=quant)
        kernel = FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe,
                quant_config=quant,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )
        original = (experts.w1_scale, experts.w2_scale)
        layer = torch.nn.Module()
        layer.layer_name = "model.layers.0.mlp.experts"
        for name, value in zip(
            ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
            (w1, w2, *original),
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )
        compression_config = SimpleNamespace(
            kernel_config=SimpleNamespace(
                nvfp4_scale_compression=True, qwen4_exp_sm120_moe_tactics=True
            ),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    model_type="qwen4_exp_text", num_hidden_layers=48
                )
            ),
        )
        monkeypatch.setattr(
            nvfp4_resident,
            "get_current_vllm_config_or_none",
            lambda: compression_config,
        )
        monkeypatch.setattr(backends, "model_tag", "backbone")
        resident = nvfp4_resident.prepare_resident_nvfp4_scales(experts, layer)
        assert resident is not None and m in resident.moe_tactics
        compressed = (experts.w1_scale, experts.w2_scale)
        experts._nvfp4_resident = resident
        a = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16) * 0.1
        scores = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        weights, ids, _ = fused_topk(a, scores, 10, renormalize=True)

        def run():
            return kernel.apply(
                hidden_states=a,
                w1=w1,
                w2=w2,
                topk_weights=weights,
                topk_ids=ids,
                activation=MoEActivation.SILU,
                global_num_experts=512,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for _ in range(3):
            a.copy_(torch.randn_like(a) * 0.1)
            scores.normal_()
            next_weights, next_ids, _ = fused_topk(a, scores, 10, renormalize=True)
            weights.copy_(next_weights)
            ids.copy_(next_ids)
            experts._nvfp4_resident = None
            quant._w1.scale, quant._w2.scale = original
            reference = run().clone()
            quant._w1.scale, quant._w2.scale = compressed
            experts._nvfp4_resident = resident
            for scratch in resident.scratch:
                scratch.fill_(0xA5)
            graph.replay()
            torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
            repeated = actual.clone()
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(actual, repeated, atol=0, rtol=0)


@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUOAI_UNINTERLEAVE],
)
def test_flashinfer_swigluoai_params_are_forwarded(activation, monkeypatch):
    from flashinfer.fused_moe.core import ActivationType

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=activation,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
    )
    quant_config = FusedMoEQuantConfig.make(
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = FlashInferExperts(moe_config=moe_config, quant_config=quant_config)

    call_args = {}

    def fake_flashinfer_cutlass_fused_moe(**kwargs):
        call_args.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.experts."
        "flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe",
        fake_flashinfer_cutlass_fused_moe,
    )
    experts.apply(
        output=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        hidden_states=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        w1=torch.empty((2, 256, 128), device="cuda", dtype=torch.bfloat16),
        w2=torch.empty((2, 128, 128), device="cuda", dtype=torch.bfloat16),
        topk_weights=torch.ones((1, 1), device="cuda", dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        activation=activation,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert experts._supports_activation(activation)
    assert call_args["activation_type"] == ActivationType.Swiglu
    for name, value in (
        ("swiglu_alpha", 1.702),
        ("swiglu_beta", 1.0),
        ("swiglu_limit", 7.0),
    ):
        torch.testing.assert_close(
            call_args[name],
            torch.full((2,), value, device="cuda", dtype=torch.float32),
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16
        is_gated_act = activation.is_gated

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated_act,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        flashinfer_experts = FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )

        flashinfer_output = flashinfer_experts.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        # Reference check:
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None
        assert quant_config.g1_alphas is not None
        assert quant_config.g2_alphas is not None
        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                quant_config.w1_scale[idx],
                (1 / quant_config.g1_alphas[idx]),
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                quant_config.w2_scale[idx],
                (1 / quant_config.g2_alphas[idx]),
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=activation
        )

        torch.testing.assert_close(
            torch_output, flashinfer_output, atol=1e-1, rtol=1e-1
        )

        if (
            activation == MoEActivation.GELU_TANH
            and current_platform.is_device_capability_family(120)
        ):
            from flashinfer import autotune

            def run():
                return flashinfer_experts.apply(
                    hidden_states=a,
                    w1=w1_q,
                    w2=w2_q,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=activation,
                    global_num_experts=e,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                )

            with autotune(tune_mode=True):
                run()
            expected = run().clone()
            for _ in range(8):
                torch.testing.assert_close(run(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)


def test_flashinfer_nvfp4_geglu_tanh_reference(workspace_init):
    """Exercise Gemma's activation through the real quantized expert path."""
    test_flashinfer_fp4_moe_no_graph(
        m=3,
        n=704,
        k=2816,
        e=128,
        topk=8,
        dtype=torch.bfloat16,
        activation=MoEActivation.GELU_TANH,
        workspace_init=workspace_init,
    )


def test_flashinfer_nvfp4_geglu_tanh_prefill_repeatability(workspace_init):
    """Check prefill math and SM120's tuned ordered reduction."""
    test_flashinfer_fp4_moe_no_graph(
        m=29,
        n=704,
        k=2816,
        e=128,
        topk=8,
        dtype=torch.bfloat16,
        activation=MoEActivation.GELU_TANH,
        workspace_init=workspace_init,
    )
