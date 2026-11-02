# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.mxfp4_csf import (
        CompressedMxfp4Scales,
    )

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    count_expert_num_tokens,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton as fused_silu_mul_fp8_quant_packed,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_supported,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    mk_alignment_scope,
)
from vllm.utils.import_utils import has_deep_gemm

logger = init_logger(__name__)


def _fp8_workspace_shape(
    num_rows: int, num_columns: int, workspace_dtype: torch.dtype
) -> tuple[int, int]:
    """Size an FP8 byte buffer stored in a workspace of another dtype."""
    bytes_per_workspace_element = workspace_dtype.itemsize
    fp8_columns_per_workspace_element = (
        bytes_per_workspace_element // torch.float8_e4m3fn.itemsize
    )
    assert bytes_per_workspace_element % torch.float8_e4m3fn.itemsize == 0
    return (
        num_rows,
        -(-num_columns // fp8_columns_per_workspace_element),
    )


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align = get_mk_alignment_for_contiguous_layout()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()

    align = get_mk_alignment_for_contiguous_layout()[0]

    if not _valid_deep_gemm_shape(M, N, K):
        logger.debug_once(
            "DeepGemm disabled due to unaligned problem size. "
            "M: %s, N: %s, K: %s. M should >= %s "
            "and N and K must be multiples of %s. "
            "This is not an error and we will fall back to triton.",
            M,
            N,
            K,
            align,
            align,
        )
        return False
    elif N <= 512:
        logger.debug_once(
            "DeepGemm disabled for N <= 512. M: %s, N: %s, K: %s. "
            "This means we will fallback to triton "
            "for this specific shape for further speed up.",
            M,
            N,
            K,
        )
        return False

    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm disabled: weights or activations not contiguous. "
            "hidden_states.is_contiguous(): %s, w1.is_contiguous(): %s, "
            "w2.is_contiguous(): %s",
            hidden_states.is_contiguous(),
            w1.is_contiguous(),
            w2.is_contiguous(),
        )
        return False

    return True


class DeepGemmExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # MXFP8: FP8 e4m3 values + UE8M0 1x32 block scales (Blackwell). Reuses
        # the same grouped GEMM (aliased to fp8_fp4) with recipe (1, 32).
        self.mxfp8 = quant_config.block_shape == [1, 32]
        if self.mxfp8:
            assert quant_config.quant_dtype == "mxfp8"
        else:
            assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()
            assert quant_config.quant_dtype == torch.float8_e4m3fn
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        # Gated-activation params: silu == swigluoai with alpha=1, beta=0.
        # FP8 (silu) configs leave these None, reproducing plain silu.
        self.gemm1_alpha = (
            quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        if (weight_key, activation_key) == (kFp8Static128BlockSym, kFp8Dynamic128Sym):
            return True
        # MXFP8 1x32 uses the fp8_fp4 grouped GEMM with recipe (1, 32) — only
        # available on Blackwell (SM100).
        if (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic):
            return current_platform.is_device_capability_family(100)
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # silu/swigluoai go through the fused alpha/beta kernel; swiglustep
        # uses the unfused activation path. The fused kernel reads packed w13
        # (gate = first half, up = second half), so it implements the
        # *uninterleaved* SwiGLU-OAI variant.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # NOTE(rob): discovered an IMA with this combination. Needs investigation.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.block_shape is not None
        # Use the contiguous-layout M alignment (matches apply()); block_shape[0]
        # is the quant block (1 for MXFP8) and would under-size the workspace.
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace1 is allocated in the activation dtype by the workspace
        # manager, but is only ever viewed and used as FP8 in apply(). Size it
        # by bytes instead of reserving one BF16/FP16 element per FP8 element.
        workspace1 = _fp8_workspace_shape(
            M_sum,
            max(activation_out_dim, K),
            self.workspace_dtype(self.moe_config.in_dtype),
        )
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        activation: MoEActivation,
        expert_ends: torch.Tensor | None = None,
        expert_alignment: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.block_shape is not None
        block_k = self.block_shape[1]
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        # silu and swigluoai are both expressible by the fused gated kernel via
        # (alpha, beta): silu uses alpha=1, beta=0; swigluoai uses config values.
        # The fused kernel reads packed w13, hence SWIGLUOAI_UNINTERLEAVE.
        fused_gated = activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

        # 1. DeepGemm UE8M0: fused gate+mul+clamp+quant+pack
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if fused_gated:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                    clamp_limit=self.gemm1_clamp_limit,
                    alpha=self.gemm1_alpha,
                    beta=self.gemm1_beta,
                    expert_ends=expert_ends,
                    expert_alignment=expert_alignment,
                )
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale

        # 2. Hopper / non‑E8M0: prefer the fused gate+mul+quant kernel
        if fused_gated:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
                group_size=block_k,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
                expert_ends=expert_ends,
                expert_alignment=expert_alignment,
            )

        # 3. fallback path for non-SiLU activations in non‑UE8M0 cases.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, block_k, column_major_scales=True, out_q=output
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        _, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        # Both GEMMs and quantization must agree on the live expert ranges.
        use_psum_layout = (
            not self.mxfp8
            and (
                current_platform.is_device_capability_family(90)
                or (
                    current_platform.is_device_capability_family(100)
                    and DeepGemmQuantScaleFMT.from_oracle()
                    == DeepGemmQuantScaleFMT.UE8M0
                )
            )
            and activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE)
        )
        a1q, a1q_scale, grouped_layout, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            # MXFP8 uses a 32-element activation-scale group (block_shape[1]);
            # FP8-block keeps the default (128) alignment.
            block_size=self.block_shape[1] if self.mxfp8 else None,
            use_psum_layout=use_psum_layout,
        )
        assert a1q.size(0) == M_sum

        # MXFP8 (1x32) drives the fp8_fp4-aliased grouped GEMM with recipe
        # (1, 32); the FP8 block path keeps the default (128) recipe.
        gemm_kwargs: dict = (
            {"recipe_a": (1, self.block_shape[1]), "recipe_b": (1, self.block_shape[1])}
            if self.mxfp8
            else {}
        )

        if use_psum_layout:
            gemm_kwargs["use_psum_layout"] = True

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # otherwise the scheduler can pick the wrong expert id from m_indices
        # under cudagraph replay.
        with mk_alignment_scope(align_used):
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                grouped_layout,
                **gemm_kwargs,
            )

            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N),
                output=quant_out,
                activation=activation,
                expert_ends=grouped_layout if use_psum_layout else None,
                expert_alignment=align_used if use_psum_layout else 0,
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                grouped_layout,
                **gemm_kwargs,
            )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class DeepGemmFP4Experts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation for FP4 weights.

    Uses m_grouped_fp8_fp4_gemm_nt_contiguous with FP8 activations and
    MXFP4 (FP4 E2M1 packed as uint8) weights. Requires Blackwell-family
    GPUs (SM100 datacenter or SM120 consumer).
    """

    # FP8 activation block size (hardcoded since mxfp4_w4a8 quant config
    # does not set a block_shape on the activation descriptor).
    _ACT_BLOCK_K = 128
    # FP4 weight block size
    _WEIGHT_BLOCK_K = 32

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.weight_quant_dtype == "mxfp4"
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self._csf_scales: tuple[CompressedMxfp4Scales, ...] | None = None
        self._csf_pair_fusion = os.getenv("DS4_CSF_PAIR_FUSION", "1") == "1"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.config import get_current_vllm_config_or_none
        from vllm.model_executor.layers.quantization.utils.mxfp4_csf import (
            CompressedMxfp4Scales,
            reject_compressed_scale_export,
            should_compress_mxfp4_scales,
        )

        if not should_compress_mxfp4_scales(layer):
            return
        config = get_current_vllm_config_or_none()
        assert config is not None
        if config.parallel_config.enable_eplb or config.parallel_config.use_ubatching:
            raise ValueError("MXFP4 scale compression does not support EPLB/ubatching")
        from vllm.model_executor.utils import replace_parameter
        from vllm.v1.worker.workspace import current_workspace_manager

        assert self.w1_scale is not None and self.w2_scale is not None
        compressed = tuple(
            CompressedMxfp4Scales.from_native(s) for s in (self.w1_scale, self.w2_scale)
        )
        if (
            sum(c.nbytes for c in compressed)
            >= (self.w1_scale.numel() + self.w2_scale.numel()) * 4
        ):
            logger.warning_once("Keeping incompressible MXFP4 scales uncompressed.")
            return
        key = (
            CompressedMxfp4Scales,
            tuple((c.shape, c.stride, c.bases.device) for c in compressed),
        )
        scratch = current_workspace_manager().get_persistent_resource(
            key, lambda: tuple(c.make_scratch() for c in compressed)
        )
        for c, out in zip(compressed, scratch):
            c.decode(out)
        self.quant_config._w1.scale, self.quant_config._w2.scale = scratch
        for name, out in (
            ("w13_weight_scale", scratch[0]),
            ("w2_weight_scale", scratch[1]),
        ):
            param = getattr(layer, name)
            if isinstance(param, torch.nn.Parameter):
                # Alias the shared scratch. replace_parameter would copy into the
                # same-shaped original storage and keep every layer's scales.
                param.data = out
            else:
                replace_parameter(layer, name, out)
        self._csf_scales = compressed
        layer.register_state_dict_pre_hook(reject_compressed_scale_export)
        logger.info_once(
            "Lossless native MXFP4 scale compression enabled; "
            "active experts decode into shared DeepGEMM scratch."
        )
        if self._csf_pair_fusion:
            logger.info_once("Paired CSF scale preparation enabled.")

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms import current_platform

        return is_deep_gemm_supported() and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # SILU has fused gate+mul+quant kernels; SWIGLUSTEP/SITU take the
        # general path (activation applied via self.activation, which forwards
        # the situ betas, then FP8 requant).
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SITU,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M,
            topk,
            local_num_experts,
            block_m,
            expert_tokens_meta,
            allow_sm120_fp4_m32=self._csf_scales is not None and 0 < M <= 128,
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace1 holds the permuted and requantized FP8 activations. The
        # workspace manager allocates it in the model activation dtype, so
        # account for the dtype sizes rather than overallocating BF16/FP16.
        workspace1 = _fp8_workspace_shape(
            M_sum,
            max(activation_out_dim, K),
            self.workspace_dtype(self.moe_config.in_dtype),
        )
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        activation: MoEActivation,
        expert_ends: torch.Tensor | None = None,
        expert_alignment: int = 0,
        row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_k = self._ACT_BLOCK_K
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        if activation == MoEActivation.SILU:
            # Fused gate+mul+quant kernels for the common SILU case.
            if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                    clamp_limit=self.gemm1_clamp_limit,
                    expert_ends=expert_ends,
                    expert_alignment=expert_alignment,
                    row_indices=row_indices,
                )
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
            )

        # General gated activations (SWIGLUSTEP, SITU): apply the activation
        # (self.activation forwards the situ betas from moe_config) then
        # FP8-requant into the layout DeepGEMM expects for this scale format.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            return per_token_group_quant_fp8_packed_for_deepgemm(
                act_out, block_k, use_ue8m0=True, out_q=output
            )
        use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        return per_token_group_quant_fp8(
            act_out,
            block_k,
            column_major_scales=True,
            out_q=output,
            use_ue8m0=use_ue8m0,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        _, N, _ = w1.size()
        # K comes from activations (full hidden dim), not from w1 which is
        # packed FP4 (E, N, K//2).
        K = a1q.size(1)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        if self._csf_scales is not None:
            counts = (
                expert_tokens_meta.expert_num_tokens
                if expert_tokens_meta is not None
                else None
            )
            if counts is None:
                counts = count_expert_num_tokens(
                    topk_ids, local_num_experts, expert_map
                )
                expert_tokens_meta = (
                    mk.ExpertTokensMetadata(counts, None)
                    if expert_tokens_meta is None
                    else replace(expert_tokens_meta, expert_num_tokens=counts)
                )
            if self._csf_pair_fusion:
                from vllm.model_executor.layers.quantization.utils import mxfp4_csf_pair

                ids = torch.empty(
                    local_num_experts, device=counts.device, dtype=torch.int32
                )
                mxfp4_csf_pair.decode_pair(
                    self._csf_scales,
                    (self.w1_scale, self.w2_scale),
                    counts,
                    ids,
                    min(local_num_experts, topk_ids.numel()),
                )
            else:
                for compressed, scratch in zip(
                    self._csf_scales, (self.w1_scale, self.w2_scale)
                ):
                    compressed.decode(scratch, counts, topk_ids.numel())

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
            allow_sm120_fp4_m32=(
                self._csf_scales is not None and 0 < topk_ids.size(0) <= 128
            ),
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        use_psum_layout = current_platform.is_device_capability_family(100)
        a1q, a1q_scale, grouped_layout, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            use_psum_layout=use_psum_layout,
            allow_sm120_fp4_m32=(
                self._csf_scales is not None and 0 < topk_ids.size(0) <= 128
            ),
        )
        assert a1q.size(0) == M_sum

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # see DeepGemmExperts.apply for rationale.
        with mk_alignment_scope(align_used):
            # FC1: FP8 activations x FP4 weights
            # DeepGEMM 2.4.2 requires FP4-packed weights as int8 (kPackedFP4).
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1.view(torch.int8), self.w1_scale),
                mm1_out,
                grouped_layout,
                use_psum_layout=use_psum_layout,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

            # SiLU+mul and FP8 requant use the same live expert ranges as GEMM.
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N),
                output=quant_out,
                activation=activation,
                expert_ends=grouped_layout if use_psum_layout else None,
                expert_alignment=align_used if use_psum_layout else 0,
                row_indices=inv_perm.reshape(-1) if topk_ids.size(0) <= 128 else None,
            )

            # FC2: FP8 activations x FP4 weights
            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2.view(torch.int8), self.w2_scale),
                mm2_out,
                grouped_layout,
                use_psum_layout=use_psum_layout,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
