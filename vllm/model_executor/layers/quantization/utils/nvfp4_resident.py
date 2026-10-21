# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless resident Qwen4Exp NVFP4 scales for SM120 FlashInfer CUTLASS."""

from dataclasses import dataclass
from functools import cache

import regex as re
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.nvfp4_csf import (
    GpuPaletteScales,
    PaletteScales,
    decode_palette_pair,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_SM120_MOE_TACTICS = {
    4: (12, 44),
    8: (12, 44),
    16: (12, 47),
    64: (12, 45),
    2048: (13, 47),
    8192: (10, 25),
}


@cache
def _validated_sm120_moe_tactics(device_index: int) -> dict[int, tuple[int, int]]:
    import flashinfer
    from flashinfer.fused_moe.core import ActivationType, get_cutlass_fused_moe_module
    from flashinfer.utils import device_support_pdl

    if flashinfer.__version__ != "0.7.2":
        raise RuntimeError("Qwen4Exp SM120 MoE tactics require FlashInfer 0.7.2")
    device = torch.device("cuda", device_index)
    with torch.accelerator.device_index(device_index):
        runner = (
            get_cutlass_fused_moe_module("120")
            .MoERunner(
                x_dtype=torch.uint8,
                weight_dtype=torch.long,
                output_dtype=torch.bfloat16,
                top_k=10,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                cluster_size=1,
                cluster_rank=0,
                enable_alltoall=False,
                use_deepseek_fp8_block_scale=False,
                use_w4_group_scaling=False,
                use_mxfp8_act_scaling=False,
                min_latency_mode=False,
                enable_pdl=device_support_pdl(device),
                activation_type=ActivationType.Swiglu,
                use_packed_weights=False,
                use_fused_finalize=True,
                use_wfp4afp8_humming=False,
            )
            .fused_moe_runner
        )
        counts = (runner.get_gemm1_tactic_count(), runner.get_gemm2_tactic_count())
        if counts != (16, 32) or any(
            runner.get_tactic_occupancy(tactic) <= 0
            for pair in _SM120_MOE_TACTICS.values()
            for tactic in pair
        ):
            raise RuntimeError("FlashInfer CUTLASS tactics differ from the SM120 plans")
    logger.info_once(
        "Qwen4Exp SM120 offline MoE tactics enabled: %s", str(_SM120_MOE_TACTICS)
    )
    return _SM120_MOE_TACTICS


@triton.jit(do_not_specialize=["NUM", "CAPACITY"])
def _select_active(
    TopK,
    Output,
    NUM,
    EXPERTS: tl.constexpr,
    CAPACITY,
    ROUTE_BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
):
    lane = tl.arange(0, ROUTE_BLOCK)
    ids = tl.load(TopK + lane, lane < NUM, -1)
    valid = (lane < NUM) & (ids >= 0) & (ids < EXPERTS)
    counts = tl.histogram(ids.to(tl.int32), EXPERT_BLOCK, mask=valid)
    expert = tl.arange(0, EXPERT_BLOCK)
    active = (counts > 0) & (expert < EXPERTS)
    positions = tl.cumsum(active.to(tl.int32)) - 1
    tl.store(Output + expert, -1, expert < CAPACITY)
    tl.debug_barrier()
    tl.store(Output + positions, expert, active & (positions < CAPACITY))


def _reject_export(module, prefix, keep_vars):
    raise RuntimeError(
        "State export is unsupported with resident NVFP4 scale compression"
    )


@dataclass
class ResidentNvfp4Scales:
    codecs: tuple[GpuPaletteScales, GpuPaletteScales]
    scratch: tuple[torch.Tensor, torch.Tensor]
    all_ids: torch.Tensor
    selected_ids: torch.Tensor
    moe_tactics: dict[int, tuple[int, int]] | None = None

    def decode(self, topk_ids):
        if not topk_ids.is_contiguous():
            raise ValueError("Resident NVFP4 routing IDs must be contiguous")
        experts = self.codecs[0].shape[0]
        count = topk_ids.numel()
        if count == 0:
            return
        # Large prefills generally visit every expert; avoid route preparation.
        if count >= 8 * experts:
            ids = self.all_ids
        else:
            capacity = min(count, experts)
            ids = self.selected_ids[:capacity]
            _select_active[(1,)](
                topk_ids,
                ids,
                count,
                experts,
                capacity,
                triton.next_power_of_2(count),
                triton.next_power_of_2(experts),
                num_warps=4,
            )
        decode_palette_pair(self.codecs, self.scratch, ids)


def prepare_resident_nvfp4_scales(experts, layer):
    config = get_current_vllm_config_or_none()
    if config is None or not config.kernel_config.nvfp4_scale_compression:
        return None
    if config.model_config.hf_text_config.model_type != "qwen4_exp_text":
        return None
    from vllm.compilation import backends

    if backends.model_tag != "backbone":
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
    if match and int(match[1]) >= config.model_config.hf_text_config.num_hidden_layers:
        return None
    if experts.weight_quant_dtype != "nvfp4":
        return None
    original = (experts.w1_scale, experts.w2_scale)
    if any(
        s is None
        or s.dtype != torch.float8_e4m3fn
        or s.ndim != 3
        or not s.is_contiguous()
        for s in original
    ):
        raise ValueError(
            "Resident NVFP4 compression requires contiguous native FP8 scales"
        )
    encoded = tuple(
        PaletteScales.from_native(s.detach().cpu().view(torch.uint8).numpy())
        for s in original
    )
    raw_bytes = sum(s.numel() for s in original)
    compressed_bytes = sum(c.nbytes for c in encoded)
    if compressed_bytes >= raw_bytes:
        logger.warning_once("Keeping incompressible NVFP4 scales uncompressed")
        return None
    device = original[0].device
    shapes = tuple(tuple(s.shape) for s in original)
    key = (ResidentNvfp4Scales, shapes, device)

    def allocate():
        scratch = tuple(
            torch.empty(shape, dtype=torch.uint8, device=device) for shape in shapes
        )
        all_ids = torch.arange(shapes[0][0], dtype=torch.int32, device=device)
        selected = torch.empty_like(all_ids)
        return scratch, all_ids, selected

    scratch, all_ids, selected = current_workspace_manager().get_persistent_resource(
        key, allocate
    )
    codecs = tuple(c.to_gpu(device) for c in encoded)
    resident = ResidentNvfp4Scales(codecs, scratch, all_ids, selected)
    if getattr(config.kernel_config, "qwen4_exp_sm120_moe_tactics", False):
        if (
            tuple(layer.w13_weight.shape) != (512, 1280, 1280)
            or tuple(layer.w2_weight.shape) != (512, 2560, 320)
            or experts.out_dtype != torch.bfloat16
            or experts.moe_config.experts_per_token != 10
        ):
            raise ValueError(
                "SM120 MoE tactics require BF16 NVFP4 E512/top10/H2560/I640"
            )
        resident.moe_tactics = _validated_sm120_moe_tactics(device.index)
    decode_palette_pair(codecs, scratch, all_ids)
    experts.quant_config._w1.scale = scratch[0].view(torch.float8_e4m3fn)
    experts.quant_config._w2.scale = scratch[1].view(torch.float8_e4m3fn)
    for name, scale in zip(("w13_weight_scale", "w2_weight_scale"), scratch):
        param = getattr(layer, name)
        param.data = scale.view(torch.float8_e4m3fn)
    layer.register_state_dict_pre_hook(_reject_export)
    logger.info(
        "Resident NVFP4 scales: %.2f -> %.2f MiB; shared scratch %.2f MiB",
        raw_bytes / 2**20,
        compressed_bytes / 2**20,
        raw_bytes / 2**20,
    )
    return resident


def validate_nvfp4_scale_compression(config):
    from vllm.platforms import current_platform

    model = config.model_config
    parallel = config.parallel_config
    if (
        model is None
        or model.hf_text_config.model_type != "qwen4_exp_text"
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability(120)
        or config.kernel_config.moe_backend not in ("auto", "flashinfer_cutlass")
    ):
        raise ValueError(
            "NVFP4 scale compression requires Qwen4Exp on SM120 with FlashInfer CUTLASS"
        )
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_expert_parallel
        or parallel.enable_eplb
        or parallel.use_ubatching
        or config.scheduler_config.async_scheduling
        or model.enable_sleep_mode
        or config.lora_config is not None
        or config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
    ):
        raise ValueError(
            "NVFP4 scale compression requires synchronous TP1 without "
            "EP/EPLB, ubatching, sleep, LoRA or CPU weight offload"
        )
