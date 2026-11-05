# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NOTICE: uses the W12 format and kernel ported from SpecPilot
# (https://github.com/StargazerX0/SpecPilot, commit 78c42af, Apache-2.0); see
# vllm/model_executor/layers/w12_linear.py and csrc/w12_gemm_sm120.cu.
"""Opt-in lossless W12 storage for Gemma4's BF16 linears on SM120.

``VLLM_GEMMA4_W12=1`` re-encodes the target model's BF16 attention
projections, shared-MLP projections and the tied embedding / LM head into
the W12 format after loading, verifies every weight bit-exactly and frees the
BF16 originals. Weights stream 12.03 bits instead of 16, so small-batch
(decode / speculative verify) projections read about 25% fewer bytes. Larger
batches decode the weight into one shared BF16 scratch and use torch.mm.
Outputs are not bitwise equal to cuBLAS because accumulation order differs.
"""

from __future__ import annotations

import dataclasses
import gc
from collections import Counter

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers import w12_linear as w12
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)

# Largest row count that runs the W12 kernel (in 64-row launches); above it
# the weight is decoded into the shared scratch for torch.mm. Each value is the
# largest measured M up to which the kernel was not slower than decode +
# torch.mm in two runs (CUDA-graph replays, DRAM-resident weights, RTX PRO 6000
# Blackwell at 450 W, Gemma4-26B weights). Keyed by (N, K, activation). The LM
# head has no whole-weight scratch and always uses the kernel.
KERNEL_MAX_M: dict[tuple[int, int, int], int] = {
    (8192, 2816, w12.ACT_NONE): 112,  # sliding qkv
    (9216, 2816, w12.ACT_NONE): 112,  # global q|k (compact KV, V omitted)
    (2816, 4096, w12.ACT_NONE): 64,  # sliding o_proj
    (2816, 8192, w12.ACT_NONE): 64,  # global o_proj
    (4224, 2816, w12.ACT_GELU_TANH): 64,  # shared-MLP gate_up + GELU-tanh
    (4224, 2816, w12.ACT_NONE): 80,  # shared-MLP gate_up, unfused
    (2816, 2112, w12.ACT_NONE): 48,  # shared-MLP down
}
DEFAULT_KERNEL_MAX_M = 64
# Shared-MLP gate_up runs act(gate) * up in the GEMM epilogue.
FUSE_GELU_TANH = True
PDL = False

_STATE: dict = {
    "installed": Counter(),
    "skipped": Counter(),
    "bf16_bytes": 0,
    "w12_bytes": 0,
    "workspace_bytes": 0,
}


def gemma4_w12_state() -> dict:
    return {
        "installed": dict(_STATE["installed"]),
        "skipped": dict(_STATE["skipped"]),
        "bf16_bytes_freed": _STATE["bf16_bytes"],
        "w12_bytes_resident": _STATE["w12_bytes"],
        "workspace_bytes": _STATE["workspace_bytes"],
    }


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def supported_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in (
        (12, 0),
        (12, 1),
    )


def check_supported(config) -> None:
    """Raise if the explicit opt-in cannot be honoured (never silently off)."""
    if config is None:
        raise RuntimeError("VLLM_GEMMA4_W12 needs a vLLM config")
    if config.lora_config is not None:
        raise ValueError("VLLM_GEMMA4_W12 does not support LoRA")
    if not getattr(config, "use_v2_model_runner", True):
        # The V1 proposer reads the target's embed_tokens.weight.
        raise ValueError("VLLM_GEMMA4_W12 requires the V2 model runner")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("VLLM_GEMMA4_W12 requires tensor/pipeline parallel size 1")
    if not w12.kernels_available():
        raise RuntimeError(
            "VLLM_GEMMA4_W12 needs the SM120 W12 kernels (_C::w12_gemm); this "
            "build has none"
        )


class W12LinearMethod(UnquantizedLinearMethod):
    """Replaces an installed BF16 linear's method; the BF16 weight is gone."""

    def __init__(self, n: int, act: int, plan: w12.W12Plan, kernel_max_m: int):
        super().__init__()
        self.n = n
        self.act = act
        self.plan = plan
        self.kernel_max_m = kernel_max_m

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise RuntimeError("W12 weights cannot be reloaded in place")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise RuntimeError("W12 linears are bias-free")
        return torch.ops.vllm.w12_linear(
            x,
            layer.w12_planes,
            layer.w12_scale,
            layer.w12_esc_rp,
            layer.w12_esc_col,
            layer.w12_esc_val,
            layer.w12_esc_idx,
            layer.w12_esc_bf16,
            self.n,
            self.act,
            self.plan.num_ctas,
            self.plan.max_segments,
            self.kernel_max_m,
            PDL,
        )


class W12EmbeddingMethod(UnquantizedEmbeddingMethod):
    """Tied embedding + LM head from one W12 copy (rows gathered exactly)."""

    def __init__(self, n: int, k: int, plan: w12.W12Plan):
        super().__init__()
        self.n = n
        self.k = k
        self.plan = plan

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise RuntimeError("W12 weights cannot be reloaded in place")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise RuntimeError("W12 LM head is bias-free")
        return torch.ops.vllm.w12_linear(
            x,
            layer.w12_planes,
            layer.w12_scale,
            layer.w12_esc_rp,
            layer.w12_esc_col,
            layer.w12_esc_val,
            layer.w12_esc_idx,
            layer.w12_esc_bf16,
            self.n,
            w12.ACT_NONE,
            self.plan.num_ctas,
            self.plan.max_segments,
            -1,
            PDL,
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.w12_embedding(
            input_,
            layer.w12_planes,
            layer.w12_scale,
            layer.w12_esc_rp,
            layer.w12_esc_col,
            layer.w12_esc_bf16,
            self.n,
            self.k,
        )


def _attach(module: torch.nn.Module, packed: w12.W12Packed) -> None:
    for name, tensor in packed.tensors().items():
        module.register_buffer(f"w12_{name}", tensor, persistent=False)


def _sample_input(rows: int, k: int, device, salt: int) -> torch.Tensor:
    values = torch.arange(rows * k, device=device, dtype=torch.float32)
    return ((values * (0.37 + 0.01 * salt)).sin() * 0.5).reshape(rows, k).bfloat16()


def _act_and_mul_ref(gate_up: torch.Tensor) -> torch.Tensor:
    half = gate_up.shape[-1] // 2
    out = torch.empty(
        (*gate_up.shape[:-1], half), dtype=gate_up.dtype, device=gate_up.device
    )
    torch.ops._C.gelu_tanh_and_mul(out, gate_up)
    return out


def _verify(
    name: str,
    weight: torch.Tensor,
    packed: w12.W12Packed,
    plan: w12.W12Plan,
    kernel_max_m: int,
    scratch: torch.Tensor | None,
) -> None:
    """Fail closed unless the device decode is exact and the GEMM is sane."""
    t = packed.tensors()
    n, k = packed.n, packed.k
    if scratch is not None:
        wd = scratch[: n * k].view(n, k)
        w12._decode_into(
            wd, t["planes"], t["scale"], t["esc_idx"], t["esc_bf16"], n, k, packed.act
        )
        if not torch.equal(wd.view(torch.int16), weight.view(torch.int16)):
            raise RuntimeError(f"W12 device decode of {name} is not bit-exact")
    # Kernel at the smallest and largest tile; decode fallback above kmax.
    checks = [(3, -1), (64, -1)]
    if kernel_max_m >= 0:
        checks.append((kernel_max_m + 1, kernel_max_m))
    for salt, (rows, kmax) in enumerate(checks):
        x = _sample_input(rows, k, weight.device, salt)
        y = w12.w12_linear_impl(
            x,
            t["planes"],
            t["scale"],
            t["esc_rp"],
            t["esc_col"],
            t["esc_val"],
            t["esc_idx"],
            t["esc_bf16"],
            n,
            packed.act,
            plan.num_ctas,
            plan.max_segments,
            kmax,
            PDL,
        )
        ref32 = torch.cat(
            [x.float() @ w.float().t() for w in weight.split(32768)], dim=1
        )
        cublas = torch.nn.functional.linear(x, weight)
        if packed.act:
            ref32 = _act_and_mul_ref(ref32.bfloat16()).float()
            cublas = _act_and_mul_ref(cublas)
        err = (y.float() - ref32).abs().max()
        base = (cublas.float() - ref32).abs().max()
        # A gross-error screen (layout or escape bugs give O(1) errors); the
        # fused epilogue rounds gate/up to bf16 before the activation.
        bound = 2 * base + ref32.abs().max() * (2.0**-6 if packed.act else 2.0**-8)
        if not bool(torch.isfinite(y).all()) or bool(err > bound):
            raise RuntimeError(
                f"W12 GEMM check failed for {name} rows={rows}: "
                f"err={float(err):.3e} bound={float(bound):.3e}"
            )


def _encode_linear(
    name: str, linear: torch.nn.Module, act: int, num_sms: int, device: torch.device
) -> tuple[w12.W12Packed, w12.W12Plan] | None:
    weight = linear.weight
    if (
        type(linear.quant_method) is not UnquantizedLinearMethod
        or getattr(linear, "bias", None) is not None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or weight.device != device
        or weight.shape[1] % w12.KC
    ):
        _STATE["skipped"][f"unsupported:{name.rsplit('.', 1)[-1]}"] += 1
        return None
    # Exactness is checked on the device decode in _verify (what the kernel
    # reads), not with the slower pure-torch decode.
    packed = w12.encode(weight.data, act=act, check=False)
    if packed.n_inexact:
        _STATE["skipped"][f"inexact:{tuple(weight.shape)}"] += 1
        return None
    return packed, w12.W12Plan.for_shape(packed.n, packed.k, num_sms)


def _gemma4_linears(model: torch.nn.Module):
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            if hasattr(attn, "qkv_proj"):
                yield f"layers.{i}.self_attn.qkv_proj", attn.qkv_proj, None
            yield f"layers.{i}.self_attn.o_proj", attn.o_proj, None
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            yield f"layers.{i}.mlp.gate_up_proj", mlp.gate_up_proj, mlp
            yield f"layers.{i}.mlp.down_proj", mlp.down_proj, None


def _moved(packed: w12.W12Packed, device) -> w12.W12Packed:
    moved = {k: v.to(device) for k, v in packed.tensors().items()}
    return dataclasses.replace(packed, top=packed.top.to(device), **moved)


def install_gemma4_w12(model: torch.nn.Module, config=None) -> None:
    """Encode Gemma4ForCausalLM's BF16 linears and tied head into W12."""
    if config is None:
        config = get_current_vllm_config_or_none()
    check_supported(config)
    device = model.model.embed_tokens.weight.device
    if not supported_device(device):
        raise ValueError("VLLM_GEMMA4_W12 requires an SM120/SM121 CUDA device")
    num_sms = _num_sms(device)

    candidates = [
        item
        for item in _gemma4_linears(model)
        if not getattr(item[1], "_gemma4_w12", False)
        and getattr(item[1], "weight", None) is not None
    ]
    # The shared workspace is sized from the shapes and allocated first, so it
    # is never carved out of an encoding transient. The decode scratch also
    # serves the per-weight device-decode check.
    shapes = [tuple(item[1].weight.shape) for item in candidates]
    head_weight = getattr(model.model.embed_tokens, "weight", None)
    if head_weight is not None:
        shapes.append(tuple(head_weight.shape))
    ws = w12.get_workspace(device)
    for n, k in shapes:
        if k % w12.KC == 0:
            w12.reserve_workspace(device, w12.W12Plan.for_shape(n, k, num_sms))
    ws.reserve(0, 0, max((n * k for n, k in shapes[: len(candidates)]), default=0))

    # Encode and verify one weight at a time, staging the result on the host.
    staged = []
    for name, linear, mlp in candidates:
        fuse = mlp is not None and FUSE_GELU_TANH and _fusable_mlp(mlp)
        act = w12.ACT_GELU_TANH if fuse else w12.ACT_NONE
        result = _encode_linear(name, linear, act, num_sms, device)
        if result is None:
            continue
        packed, plan = result
        kernel_max_m = KERNEL_MAX_M.get(
            (packed.n, packed.k, packed.act), DEFAULT_KERNEL_MAX_M
        )
        _verify(name, linear.weight.data, packed, plan, kernel_max_m, ws.scratch)
        _STATE["bf16_bytes"] += linear.weight.numel() * linear.weight.element_size()
        _STATE["w12_bytes"] += packed.resident_bytes()
        method = W12LinearMethod(packed.n, packed.act, plan, kernel_max_m)
        staged.append(((linear,), method, _moved(packed, "cpu"), mlp))
        del linear.weight, packed
    head = _encode_tied_head(model, device, num_sms)
    if head is not None:
        staged.append(head)

    # Compaction: release the encoding transients and the freed BF16 blocks
    # before allocating the resident W12 tensors, so that no partially used
    # cache segment holds memory the KV-cache profiler counts as consumed.
    gc.collect()
    if device.type == "cuda":
        torch.accelerator.empty_cache()
    for modules, method, packed, mlp in staged:
        resident = _moved(packed, device)
        for module in modules:
            _attach(module, resident)
            module.quant_method = method
            module._gemma4_w12 = True
        if mlp is not None and packed.act:
            mlp.w12_fused_act = True
        label = "tied_head" if len(modules) > 1 else packed.act
        _STATE["installed"][str((packed.n, packed.k, label))] += 1
    _STATE["workspace_bytes"] = ws.nbytes()
    logger.info("Gemma4 W12 installed: %s", gemma4_w12_state())


def _fusable_mlp(mlp: torch.nn.Module) -> bool:
    from vllm.model_executor.layers.activation import GeluAndMul

    act = getattr(mlp, "act_fn", None)
    return isinstance(act, GeluAndMul) and act.approximate == "tanh"


def _encode_tied_head(model: torch.nn.Module, device, num_sms: int):
    embed = model.model.embed_tokens
    head = getattr(model, "lm_head", None)
    if getattr(embed, "_gemma4_w12", False):
        return None
    param = getattr(embed, "weight", None)
    if param is None or head is None or getattr(head, "weight", None) is not param:
        _STATE["skipped"]["lm_head:untied"] += 1
        return None
    if (
        type(embed.quant_method) is not UnquantizedEmbeddingMethod
        or type(head.quant_method)
        not in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        or getattr(head, "bias", None) is not None
        or param.dtype != torch.bfloat16
    ):
        _STATE["skipped"]["lm_head:quantized_or_biased"] += 1
        return None
    weight = param.data
    n, k = weight.shape
    if n != embed.num_embeddings_per_partition or k % w12.KC:
        _STATE["skipped"]["lm_head:shape"] += 1
        return None
    packed = w12.encode(weight, check=False)
    if packed.n_inexact:
        _STATE["skipped"]["lm_head:inexact"] += 1
        return None
    plan = w12.W12Plan.for_shape(n, k, num_sms)
    full = torch.empty_like(weight)  # one-time exactness check of the table
    _verify("lm_head", weight, packed, plan, -1, full)
    del full
    t = packed.tensors()
    ids = torch.cat(
        [
            torch.tensor([0, 1, n - 1], device=device),
            torch.div(packed.esc_idx[:64], k, rounding_mode="floor"),
            (torch.arange(64, device=device) * (n // 64) + 7) % n,
        ]
    )
    rows = w12.w12_embedding_impl(
        ids, t["planes"], t["scale"], t["esc_rp"], t["esc_col"], t["esc_bf16"], n, k
    )
    if not torch.equal(rows.view(torch.int16), weight[ids].view(torch.int16)):
        raise RuntimeError("W12 embedding gather is not bit-exact")
    _STATE["bf16_bytes"] += weight.numel() * weight.element_size()
    _STATE["w12_bytes"] += packed.resident_bytes()
    staged = (
        (embed, head),
        W12EmbeddingMethod(n, k, plan),
        _moved(packed, "cpu"),
        None,
    )
    del embed.weight, head.weight, weight, param, packed, t, rows
    return staged
