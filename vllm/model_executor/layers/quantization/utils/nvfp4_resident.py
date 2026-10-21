# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless resident Qwen4Exp NVFP4 scales for SM120 FlashInfer CUTLASS."""

from dataclasses import dataclass

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


@triton.jit
def _select_active(
    TopK,
    Output,
    NUM: tl.constexpr,
    EXPERTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.arange(0, BLOCK)
    active = tl.full((BLOCK,), False, tl.int1)
    lane = tl.arange(0, 16)
    for start in range(0, NUM, 16):
        ids = tl.load(TopK + start + lane, start + lane < NUM, -1)
        found = tl.sum((ids[:, None] == expert[None, :]).to(tl.int32), axis=0) > 0
        active = active | found
    active = active & (expert < EXPERTS)
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

    def decode(self, topk_ids):
        if not topk_ids.is_contiguous():
            raise ValueError("Resident NVFP4 routing IDs must be contiguous")
        experts = self.codecs[0].shape[0]
        count = topk_ids.numel()
        if count == 0:
            return
        # Large prefills generally visit every expert; avoid route preparation.
        if count >= experts:
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
