# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-test DeepGEMM FP8 and FP4 kernels (no DeepEP).
Compare DeepGEMM path against the Triton fallback inside vLLM's fused_experts.
"""

import importlib
import math

import pytest
import torch

# vLLM fused-expert reference (Triton fallback + DeepGEMM option)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_moe_permute,
    ep_gather,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmFP4Experts,
    _fp8_workspace_shape,
)
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_supported,
    per_block_cast_to_fp8,
)

BLOCK_SIZE = [128, 128]


@pytest.mark.parametrize("num_tokens", [6, 12, 72, 128])
@pytest.mark.parametrize("clamp", [None, 7.0])
def test_indexed_fp4_activation_replays_live_permutation(num_tokens, clamp):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        silu_mul_quant_fp8_packed_triton,
    )

    torch.manual_seed(917)
    x = torch.randn(num_tokens, 128, device="cuda", dtype=torch.bfloat16)
    aq, scales = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    ids = torch.zeros((num_tokens, 6), device="cuda", dtype=torch.int64)
    expert_map = torch.tensor([0, -1, 1, -1, 2, -1, 3, -1], device="cuda")
    padded_rows = min(num_tokens * 6, 4) * 128 + num_tokens * 6
    activations = (
        torch.randn(padded_rows, 4096, device="cuda", dtype=torch.bfloat16) * 4
    )

    def invoke():
        _, _, _, inverse, _ = deepgemm_moe_permute(aq, scales, ids, 4, expert_map, None)
        return silu_mul_quant_fp8_packed_triton(
            activations, clamp_limit=clamp, row_indices=inverse.reshape(-1)
        ) + (inverse,)

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_s, inverse = invoke()
    reference_q, reference_s = silu_mul_quant_fp8_packed_triton(
        activations, clamp_limit=clamp
    )
    for mode in range(3):
        ids.copy_(torch.arange(num_tokens * 6, device="cuda").reshape_as(ids) % 8)
        if mode == 1:
            ids.fill_(7)
            ids[0, 0] = 6
        elif mode == 2:
            ids.fill_(7)
        graph.replay()
        torch.accelerator.synchronize()
        expected_local = expert_map[ids] >= 0
        assert torch.equal(inverse >= 0, expected_local)
        live = inverse[expected_local].long()
        assert live.unique().numel() == live.numel()
        assert torch.equal(
            actual_q.view(torch.uint8)[live], reference_q.view(torch.uint8)[live]
        )
        assert torch.equal(actual_s[live], reference_s[live])


@pytest.mark.cpu_test
def test_registered_alignment_operator_preserves_small_expert_batches(monkeypatch):
    """Torch-library arity errors must not silently restore 128-row padding."""
    import vllm.utils.deep_gemm as dg
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        compute_aligned_M_and_alignment,
    )

    received = []
    library = torch.library.Library("test_dg_alignment", "FRAGMENT")
    library.define("alignment(int? expected_m=None) -> int")

    def alignment(expected_m):
        received.append(expected_m)
        return 32 if expected_m <= 8 else 128

    library.impl("alignment", alignment, "CompositeExplicitAutograd")
    monkeypatch.setattr(dg, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        dg,
        "_get_theoretical_mk_alignment_for_contiguous_layout_impl",
        torch.ops.test_dg_alignment.alignment,
    )
    assert compute_aligned_M_and_alignment(6, 6, 128, 128, None) == (1152, 32)
    assert received == [1]


@pytest.mark.cpu_test
@pytest.mark.parametrize("arguments", [1, 2])
def test_alignment_python_binding_compatibility(monkeypatch, arguments):
    import vllm.utils.deep_gemm as dg

    received = []

    def legacy(expected_m):
        received.append((expected_m,))
        return 32

    def grouped(expected_m, groups):
        received.append((expected_m, groups))
        return 32

    monkeypatch.setattr(dg, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        dg,
        "_get_theoretical_mk_alignment_for_contiguous_layout_impl",
        legacy if arguments == 1 else grouped,
    )
    assert dg.get_theoretical_mk_alignment_for_contiguous_layout(36, 128) == 32
    assert received == ([(1,)] if arguments == 1 else [(36, 128)])


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "expected_m,groups,sm12x,expected",
    [
        (None, None, True, 128),
        (1, None, True, 64),
        (64, None, True, 64),
        (65, None, True, 128),
        (512, 8, True, 64),
        (513, 8, True, 128),
        (512, 8, False, 128),
    ],
)
def test_sm12x_legacy_alignment_steps_down_at_per_expert_boundary(
    monkeypatch, expected_m, groups, sm12x, expected
):
    """A legacy library must not force small SM12x expert batches to 128 rows."""
    import vllm.utils.deep_gemm as dg

    monkeypatch.setattr(dg, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        dg,
        "_get_theoretical_mk_alignment_for_contiguous_layout_impl",
        lambda expected_m: 128,
    )
    monkeypatch.setattr(
        dg.current_platform, "is_device_capability_family", lambda family: sm12x
    )
    assert (
        dg.get_theoretical_mk_alignment_for_contiguous_layout(expected_m, groups)
        == expected
    )
    for invalid_groups in (0, -1):
        with pytest.raises(ValueError, match="num_groups must be positive"):
            dg.get_theoretical_mk_alignment_for_contiguous_layout(
                expected_m, invalid_groups
            )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "per_expert,enabled,capability,sm120,expected",
    [
        (1, True, True, True, 32),
        (32, True, True, True, 32),
        (33, True, True, True, 64),
        (1, False, True, True, 64),
        (1, True, False, True, 64),
        (1, True, None, True, 64),
        (1, True, True, False, 128),
    ],
)
def test_sm120_fp4_m32_alignment_requires_matching_native_support(
    monkeypatch, per_expert, enabled, capability, sm120, expected
):
    """M32 must never reach an older native selector or an unrelated caller."""
    from types import SimpleNamespace

    import vllm.utils.deep_gemm as dg

    module = SimpleNamespace()
    if capability is not None:
        module.supports_sm120_grouped_fp8_fp4_m32 = lambda: capability
    monkeypatch.setattr(dg, "_lazy_init", lambda: None)
    monkeypatch.setattr(dg, "_import_deep_gemm", lambda: module)
    monkeypatch.setattr(
        dg, "_get_theoretical_mk_alignment_for_contiguous_layout_impl", lambda m: 128
    )
    monkeypatch.setattr(
        dg.current_platform, "is_device_capability_family", lambda family: sm120
    )
    assert (
        dg.get_theoretical_mk_alignment_for_contiguous_layout(
            per_expert * 128, 128, allow_sm120_fp4_m32=enabled
        )
        == expected
    )


def test_ep_gather_uses_64_bit_row_offsets():
    hidden_size = 4096
    source_row = 1 << 18
    input_tensor = torch.empty(
        (source_row + 1, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    input_tensor[source_row].fill_(1)
    output = torch.empty((1, hidden_size), device="cuda", dtype=torch.bfloat16)

    ep_gather(
        input_tensor=input_tensor,
        recv_topk_ids=torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        recv_topk_weight=torch.ones((1, 1), device="cuda", dtype=torch.float32),
        input_index=torch.full((1, 1), source_row, device="cuda", dtype=torch.int32),
        expert_map=None,
        output_tensor=output,
    )

    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)


def test_ep_gather_uses_64_bit_output_row_offsets():
    hidden_size = 6144
    output_row = (1 << 31) // hidden_size + 1
    num_rows = output_row + 1
    output = torch.empty(
        (num_rows, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    ep_gather(
        input_tensor=torch.ones((1, hidden_size), device="cuda", dtype=torch.bfloat16),
        recv_topk_ids=torch.zeros((num_rows, 1), device="cuda", dtype=torch.int64),
        recv_topk_weight=torch.ones((num_rows, 1), device="cuda", dtype=torch.float32),
        input_index=torch.zeros((num_rows, 1), device="cuda", dtype=torch.int32),
        expert_map=None,
        output_tensor=output,
    )

    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)


@pytest.mark.parametrize("workspace_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_columns", [2048, 6144, 6145])
def test_fp8_workspace_shape(workspace_dtype, num_columns):
    num_rows = 17
    shape = _fp8_workspace_shape(num_rows, num_columns, workspace_dtype)

    allocated_bytes = math.prod(shape) * workspace_dtype.itemsize
    required_bytes = num_rows * num_columns * torch.float8_e4m3fn.itemsize

    assert allocated_bytes >= required_bytes
    assert allocated_bytes - required_bytes < num_rows * workspace_dtype.itemsize


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_moe_permute_initializes_padding_scales(workspace_init):
    hidden_states = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    activations, scales = per_token_group_quant_fp8(
        hidden_states,
        group_size=128,
        use_ue8m0=True,
    )
    topk_ids = torch.tensor([[0], [1]], device="cuda", dtype=torch.int64)

    _, permuted_scales, expert_ids, _, _ = deepgemm_moe_permute(
        aq=activations,
        aq_scale=scales,
        topk_ids=topk_ids,
        local_num_experts=2,
        expert_map=None,
        expert_tokens_meta=None,
    )

    padding = expert_ids < 0
    assert padding.any()
    torch.testing.assert_close(
        permuted_scales[padding],
        torch.zeros_like(permuted_scales[padding]),
        rtol=0,
        atol=0,
    )


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
):
    """Generate (w1, w2) expert weights and their per-block scale tensors
    in FP8 block-quantized format.

      w1 shape: (E, 2N, K)
      w2 shape: (E, K, N)
    """
    dtype = torch.bfloat16
    fp8_max, fp8_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )

    # bf16 reference weights
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) / 10
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) / 10
    w1_bf16.clamp_(fp8_min, fp8_max)
    w2_bf16.clamp_(fp8_min, fp8_max)

    block_n, block_k = block_size
    n_tiles_w1 = math.ceil((2 * n) / block_n)
    k_tiles_w1 = math.ceil(k / block_k)
    n_tiles_w2 = math.ceil(k / block_n)
    k_tiles_w2 = math.ceil(n / block_k)

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
    w1_s = torch.empty(e, n_tiles_w1, k_tiles_w1, device="cuda", dtype=torch.float32)
    w2_s = torch.empty(e, n_tiles_w2, k_tiles_w2, device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(
            w1_bf16[i], block_size=block_size, use_ue8m0=True
        )
        w2[i], w2_s[i] = per_block_cast_to_fp8(
            w2_bf16[i], block_size=block_size, use_ue8m0=True
        )

    return w1, w2, w1_s, w2_s


@pytest.mark.parametrize("num_tokens", [16, 129, 512])
@pytest.mark.parametrize("with_metadata", [False, True])
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_block_fp8_padding_on_graph_replay(num_tokens, with_metadata, monkeypatch):
    """Both FP8 GEMMs consume only live rows as routing changes on replay."""
    from unittest.mock import Mock

    import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as impl
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    if not (
        impl.current_platform.is_device_capability_family(90)
        or (
            impl.current_platform.is_device_capability_family(100)
            and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
        )
    ):
        pytest.skip("Prefix-sum FP8 dispatch requires SM90 or SM10.x")

    torch.manual_seed(0)
    e, n, k, topk = 8, 1024, 512, 2
    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16) / 10
    aq, scales = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    w1, w2, s1, s2 = make_block_quant_fp8_weights(e, n, k, BLOCK_SIZE)
    quant = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=BLOCK_SIZE)
    experts = impl.DeepGemmExperts(make_dummy_moe_config(), quant)
    ids = torch.zeros(num_tokens, topk, device="cuda", dtype=torch.int64)
    weights = torch.full((num_tokens, topk), 0.5, device="cuda")
    counts = torch.zeros(e, device="cuda", dtype=torch.int32)
    metadata = mk.ExpertTokensMetadata(counts, None) if with_metadata else None
    shapes = experts.workspace_shapes(
        num_tokens, 2 * n, k, topk, e, e, metadata, MoEActivation.SILU
    )

    def route(mode):
        ids[:, 0] = torch.arange(num_tokens, device="cuda") % e
        ids[:, 1] = (ids[:, 0] + 1) % e
        if mode == 1:
            ids.fill_(-1)
            ids[:1, 0] = e - 1
        elif mode == 2:
            ids.fill_(-1)
        counts.copy_(torch.bincount(ids[ids >= 0], minlength=e).int())

    route(0)
    variants = []
    for dense in (True, False):
        ws1, ws2, output = [
            torch.empty(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes
        ]

        def invoke(output=output, ws1=ws1, ws2=ws2):
            experts.apply(
                output,
                aq,
                w1,
                w2,
                weights,
                ids,
                MoEActivation.SILU,
                e,
                None,
                scales,
                None,
                ws1,
                ws2,
                metadata,
                False,
            )

        with monkeypatch.context() as mp:
            if dense:
                platform = Mock(wraps=impl.current_platform)
                platform.is_device_capability_family.return_value = False
                mp.setattr(impl, "current_platform", platform)
            gemm = impl.m_grouped_fp8_gemm_nt_contiguous
            calls: list[int] = []

            def check_layout(*args, dense=dense, calls=calls, gemm=gemm, **kwargs):
                assert kwargs.get("use_psum_layout", False) == (not dense)
                calls.append(args[3].numel())
                return gemm(*args, **kwargs)

            mp.setattr(impl, "m_grouped_fp8_gemm_nt_contiguous", check_layout)
            invoke()
            assert len(calls) == 2
            if not dense:
                assert calls == [e, e]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                invoke()
        variants.append((graph, output, ws1, ws2))

    for mode in (0, 1, 2, 0):
        route(mode)
        for graph, _, ws1, ws2 in variants:
            ws1.fill_(float("nan"))
            ws2.fill_(float("nan"))
            graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(variants[0][1], variants[1][1], rtol=0, atol=0)
        assert torch.isfinite(variants[1][1]).all()


@pytest.mark.parametrize(("n", "k"), [(2048, 512), (512, 1024)])
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_prefix_sum_leaves_unused_capacity_untouched(n, k):
    """Both projection shapes skip unused tiles as counts change on replay."""
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import (
        m_grouped_fp8_gemm_nt_contiguous,
        mk_alignment_scope,
    )

    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        pytest.skip("Requires SM90 or SM10.x")

    capacity, experts, alignment = 512, 4, 128
    x = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16) / 10
    aq, scales = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    w, _, ws, _ = make_block_quant_fp8_weights(experts, n // 2, k, BLOCK_SIZE)
    ends = torch.tensor([0, 1, 128, 129], device="cuda", dtype=torch.int32)
    output = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)

    def invoke():
        with mk_alignment_scope(alignment):
            m_grouped_fp8_gemm_nt_contiguous(
                (aq, scales), (w, ws), output, ends, use_psum_layout=True
            )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for bounds, allocated_end in (([0, 1, 128, 129], 256), ([0] * 4, 0)):
        ends.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
        output.fill_(91)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.all(output[allocated_end:] == 91)
        if allocated_end:
            assert torch.all(output[[0, 128]] != 91)


def run_single_case(m, n, k, topk, num_experts, block_size):
    """Run one (M,N,K) configuration on a single GPU and assert DeepGEMM ==
    Triton baseline within tolerance.
    """
    tokens_bf16 = (
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        .clamp_min_(-1)
        .clamp_max_(1)
    )
    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, block_size[1])

    # expert weight tensors
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(num_experts, n, k, block_size)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=a1_scale,
        block_shape=block_size,
    )
    moe_config = make_dummy_moe_config()

    deep_gemm_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=TritonOrDeepGemmExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # triton reference
    out_triton = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=quant_config,
    )

    # DeepGemm
    out_deepgemm = deep_gemm_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 0.001, f"Diff exceeded 1%: {diff}"


# Note: N <= 512 will disable the deepgemm path due to performance issues.
MNKs = [
    (1024, 768, 128),
    (2048, 768, 512),
    (512, 1024, 1024),
    (4096, 4096, 1024),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize(("m", "n", "k"), MNKs)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_vs_triton(m, n, k, topk, num_experts, monkeypatch, workspace_init):
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmExperts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmExperts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmExperts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmExperts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
            block_size=BLOCK_SIZE,
        )

        # ensure that the DeepGEMM path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )


# ---------------------------------------------------------------------------
# FP4 weight tests (DeepGEMM m_grouped_fp8_fp4_gemm_nt_contiguous)
# ---------------------------------------------------------------------------


def make_mxfp4_weights(
    e: int,
    n: int,
    k: int,
):
    """Generate (w1, w2) expert weights in MXFP4 packed format with float32 scales,
    plus BF16 reference weights for validation.

      w1 shape: (E, 2N, K//2) uint8    — packed FP4
      w2 shape: (E, K, N//2)  uint8    — packed FP4
      w1_s shape: (E, 2N, K//32) float32  — per-row block-32 scales
      w2_s shape: (E, K, N//32)  float32  — per-row block-32 scales
      w1_bf16: (E, 2N, K)   — original BF16 for reference
      w2_bf16: (E, K, N)    — original BF16 for reference
    """
    from vllm.utils.deep_gemm import _import_deep_gemm

    per_token_cast_to_fp4 = importlib.import_module(
        _import_deep_gemm().__name__ + ".utils.math"
    ).per_token_cast_to_fp4

    dtype = torch.bfloat16
    gran_k = 32  # MXFP4 block size

    # bf16 reference weights — scale by 1/sqrt(dim) for numerical stability
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) * (k**-0.5)
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) * (n**-0.5)

    # Quantize per-expert to FP4
    w1 = torch.empty(e, 2 * n, k // 2, device="cuda", dtype=torch.uint8)
    w2 = torch.empty(e, k, n // 2, device="cuda", dtype=torch.uint8)
    w1_s = torch.empty(
        e, 2 * n, math.ceil(k / gran_k), device="cuda", dtype=torch.float32
    )
    w2_s = torch.empty(e, k, math.ceil(n / gran_k), device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_token_cast_to_fp4(
            w1_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )
        w2[i], w2_s[i] = per_token_cast_to_fp4(
            w2_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )

    return w1, w2, w1_s, w2_s, w1_bf16, w2_bf16


def _bf16_moe_reference(x, w1, w2, topk_weights, topk_ids):
    """BF16 token-loop MoE reference for correctness testing."""
    import torch.nn.functional as F

    num_tokens, hidden_size = x.shape
    intermediate = w1.shape[1] // 2
    top_k = topk_ids.shape[1]

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=x.device)
    for t in range(num_tokens):
        for kk in range(top_k):
            e = topk_ids[t, kk].item()
            w = topk_weights[t, kk].item()
            fc1 = x[t : t + 1].float() @ w1[e].float().T
            linear = fc1[:, :intermediate]
            gate = fc1[:, intermediate:]
            act = F.silu(gate) * linear
            fc2 = act @ w2[e].float().T
            output[t] += w * fc2[0]
    return output.to(torch.bfloat16)


def run_single_fp4_case(m, n, k, topk, num_experts):
    """Run one (M,N,K) configuration with FP4 weights on DeepGEMM and assert
    DeepGEMM FP4 == BF16 reference within tolerance.
    """
    tokens_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)

    # FP4 expert weight tensors + BF16 originals for reference
    w1, w2, w1_s, w2_s, w1_bf16, w2_bf16 = make_mxfp4_weights(num_experts, n, k)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
    )
    from vllm.platforms import current_platform

    _fp8_dtype = current_platform.fp8_dtype()
    _block_shape = GroupShape(128, 128)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _a2=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_s, None, None, None),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
    )
    moe_config = make_dummy_moe_config()

    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )

    deep_gemm_fp4_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=DeepGemmFP4Experts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # DeepGEMM FP4 path
    out_deepgemm_fp4 = deep_gemm_fp4_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )

    # BF16 reference using the same original weights
    out_ref = _bf16_moe_reference(tokens_bf16, w1_bf16, w2_bf16, topk_weights, topk_ids)

    # FP4 vs BF16 reference: quantization error from FP4 weights + FP8 activations
    diff = calc_diff(out_deepgemm_fp4, out_ref)
    assert diff < 0.05, f"FP4 diff exceeded 5%: {diff}"


# DeepSeek V4 dims: H=4096, I=2048, so N=2*I=4096, K=H=4096.
# FP4 quantization with block_k=32 needs large K for good accuracy.
FP4_MNKs = [
    (128, 4096, 4096),  # DeepSeek V4 shape
    (256, 2048, 2048),  # Half-size variant
    (128, 384, 3584),  # Kimi-K3 TP8 latent-MoE shape
]

FP4_TOPKS = [2]
FP4_NUM_EXPERTS = [8]


@pytest.mark.parametrize(("m", "n", "k"), FP4_MNKs)
@pytest.mark.parametrize("topk", FP4_TOPKS)
@pytest.mark.parametrize("num_experts", FP4_NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_fp4_vs_triton(
    m, n, k, topk, num_experts, monkeypatch, workspace_init
):
    pytest.importorskip("deep_gemm.utils.math")
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmFP4Experts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmFP4Experts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmFP4Experts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmFP4Experts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_fp4_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
        )

        # ensure that the DeepGEMM FP4 path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM FP4 path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )


@pytest.mark.skipif(
    not DeepGemmFP4Experts._supports_current_device(),
    reason="Requires Blackwell DeepGEMM FP4 kernels",
)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.SITU])
@pytest.mark.parametrize("m,e", [(6, 4), (96, 16), (128, 16), (129, 16)])
def test_resident_csf_matches_native_moe_across_local_routes(
    activation, m, e, monkeypatch, workspace_init
):
    """Shared scratch must preserve native output after routing and layer changes."""
    from types import SimpleNamespace

    from torch.multiprocessing.reductions import StorageWeakRef

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _pack_deepgemm_mxfp4_scales,
    )
    from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform

    torch.manual_seed(917)
    n, k = 512, 1024
    w1, w2, s1, s2, *_ = make_mxfp4_weights(e, n, k)
    s1, s2 = _pack_deepgemm_mxfp4_scales(w1, w2, s1, s2)
    config = VllmConfig()
    config.kernel_config.mxfp4_scale_compression = True
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=2)
    )
    moe = make_dummy_moe_config(activation=activation)
    moe.activation_situ_beta = 4.0
    moe.activation_situ_linear_beta = 25.0

    def build(a, b):
        quant = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(current_platform.fp8_dtype(), GroupShape(128, 128)),
            _a2=FusedMoEQuantDesc(current_platform.fp8_dtype(), GroupShape(128, 128)),
            _w1=FusedMoEQuantDesc("mxfp4", scale=a),
            _w2=FusedMoEQuantDesc("mxfp4", scale=b),
        )
        experts = DeepGemmFP4Experts(moe, quant)
        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe, quant, allow_new_interface=True, use_monolithic=False
            ),
            experts,
        )
        return experts, kernel

    native, reference = build(s1, s2)
    # Preserve native strides when making independent load-time parameters.
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.0.mlp.experts"
    for name, original in (("w13_weight_scale", s1), ("w2_weight_scale", s2)):
        copy = torch.empty_strided(
            original.shape,
            original.stride(),
            dtype=original.dtype,
            device=original.device,
        )
        copy.copy_(original)
        layer.register_parameter(name, torch.nn.Parameter(copy, requires_grad=False))
    packed, candidate = build(layer.w13_weight_scale, layer.w2_weight_scale)
    # replace_parameter keeps the registered Parameter and rebinds its data, so
    # check that the original scale storage itself is released.
    old = StorageWeakRef(layer.w13_weight_scale.untyped_storage())
    with set_current_vllm_config(config):
        packed.process_weights_after_loading(layer)
    assert packed._csf_scales is not None
    assert old.expired()
    assert layer.w13_weight_scale.data_ptr() == packed.w1_scale.data_ptr()
    with pytest.raises(RuntimeError, match="State export"):
        layer.state_dict()

    # EP map: even global expert IDs belong to this rank, odd ones are remote.
    expert_map = torch.arange(e, device="cuda").repeat_interleave(2)
    expert_map[1::2] = -1
    tokens = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)
    weights = torch.full((m, 2), 0.5, device="cuda")
    original_apply = packed.apply
    provide_metadata = False

    def apply_with_metadata(**kwargs):
        if provide_metadata:
            counts = count_expert_num_tokens(kwargs["topk_ids"], e, expert_map)
            kwargs["expert_tokens_meta"] = mk.ExpertTokensMetadata(counts, None)
        return original_apply(**kwargs)

    monkeypatch.setattr(packed, "apply", apply_with_metadata)
    for provide_metadata in (False, True):
        for ids in ([0, 2], [1, 3], [4, 6], [0, 7]):
            topk_ids = torch.tensor(ids, device="cuda").expand(m, 2).contiguous()
            args = dict(
                hidden_states=tokens,
                w1=w1,
                w2=w2,
                topk_weights=weights,
                topk_ids=topk_ids,
                global_num_experts=2 * e,
                activation=activation,
                expert_map=expert_map,
                apply_router_weight_on_input=False,
            )
            expected = reference.apply(**args).clone()
            packed.w1_scale.fill_(0)
            packed.w2_scale.fill_(0)
            actual = candidate.apply(**args)
            assert torch.equal(actual, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = candidate.apply(**args)
    for ids in ([0, 2], [1, 3], [4, 6], [0, 7]):
        args["topk_ids"].copy_(torch.tensor(ids, device="cuda").expand(m, 2))
        expected = reference.apply(**args).clone()
        packed.w1_scale.fill_(0)
        packed.w2_scale.fill_(0)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(replay_output, expected)
