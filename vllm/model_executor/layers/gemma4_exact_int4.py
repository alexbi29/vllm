# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in exact INT4 storage for Gemma4 QAT tensors kept in BF16.

Gemma4 QAT checkpoints (`google/gemma-4-31B-it-qat-w4a16-ct` and its
`...-qat-q4_0-unquantized-assistant` drafter) ship the tied embedding / LM
head, and every drafter weight, in BF16. Every group of 32 consecutive values
along the input dimension of those tensors is `bf16(fp32(q) * d)` for integer
`q` in [-8, 7] and one FP32 scale `d` per group: the QAT grid survived the
export, but the checkpoint stores it at 16 bits per weight.

`VLLM_GEMMA4_EXACT_INT4=1` recovers that representation at load time,
verifies on the device that it reproduces every weight bit for bit, and keeps
the tensor as packed nibbles plus FP32 group scales (5 bits per weight instead
of 16). The tensor is left in BF16, and the reason logged, whenever any group
does not fit. A Triton GEMM rebuilds the exact BF16 weights in registers and
accumulates in FP32, so outputs equal a BF16 GEMM up to summation order; the
embedding gather is bit-exact. Decode-sized LM-head GEMMs read 3.2x fewer
bytes.
"""

from __future__ import annotations

import time
from collections import Counter

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# K-permuted layout: within each 256-wide K block, int32 word j (0..31) holds
# in nibble i (0..7) the code u = q + 8 of the weight at k = 256*kb + 32*i + j.
# Nibble plane i of a block is therefore one contiguous 32-wide K slice, which
# is exactly one scale group.
KB = 256
GROUP = 32
# 2^23 + 8: (u | 0x4B000000) reinterpreted as fp32 is 2^23 + u exactly.
_MAGIC_BIAS = 8388616.0

_STATE: dict = {"installed": Counter(), "skipped": Counter(), "bytes_saved": 0}


def gemma4_exact_int4_state() -> dict:
    return {
        "installed": dict(_STATE["installed"]),
        "skipped": dict(_STATE["skipped"]),
        "bytes_saved": _STATE["bytes_saved"],
    }


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------


@triton.jit
def _dequant_plane(wq, s_row, kb, i: tl.constexpr, n_mask):
    s = tl.load(s_row + kb * 8 + i, mask=n_mask, other=0.0)
    u = (wq >> (4 * i)) & 0xF
    q = (u | 0x4B000000).to(tl.float32, bitcast=True) - 8388616.0
    # fp32 product, then RNE to bf16: the exact checkpoint value.
    return (q * s[:, None]).to(tl.bfloat16)


@triton.jit
def _exact_int4_gemm_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = x @ W^T with W rebuilt exactly from nibbles and FP32 group scales."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x_row = x_ptr + offs_m[:, None] * stride_xm + offs_j[None, :]
    w_row = w_ptr + offs_n[:, None] * stride_wn + offs_j[None, :]
    s_row = s_ptr + offs_n * stride_sn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // 256):
        wq = tl.load(w_row + kb * 32, mask=n_mask[:, None], other=0)
        for i in tl.static_range(8):
            x = tl.load(x_row + kb * 256 + i * 32, mask=m_mask[:, None], other=0.0)
            w = _dequant_plane(wq, s_row, kb, i, n_mask)
            acc = tl.dot(x, tl.trans(w), acc)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _exact_int4_rows_kernel(
    ids_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    num_rows,
    K,
    stride_wn,
    stride_sn,
    stride_on,
    BLOCK_R: tl.constexpr,
):
    """o[r, :] = exact BF16 row ids[r] (embedding gather / full decode)."""
    pid = tl.program_id(0)
    kb = tl.program_id(1)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = offs_r < num_rows
    rows = tl.load(ids_ptr + offs_r, mask=r_mask, other=0).to(tl.int64)
    offs_j = tl.arange(0, 32)
    wq = tl.load(
        w_ptr + rows[:, None] * stride_wn + kb * 32 + offs_j[None, :],
        mask=r_mask[:, None],
        other=0,
    )
    s_row = s_ptr + rows * stride_sn
    o_row = o_ptr + offs_r.to(tl.int64)[:, None] * stride_on + kb * 256
    for i in tl.static_range(8):
        w = _dequant_plane(wq, s_row, kb, i, r_mask)
        tl.store(o_row + i * 32 + offs_j[None, :], w, mask=r_mask[:, None])


# (BLOCK_M, BLOCK_N, num_warps, num_stages) by row count; measured on RTX PRO
# 6000 Blackwell (SM120) for the 262144 x 5376 Gemma4-31B head and the
# 262144 x 1024 assistant head.
def _gemm_config(m: int) -> tuple[int, int, int, int]:
    if m <= 16:
        return 16, 128, 4, 3
    if m <= 32:
        return 32, 128, 4, 2
    return 64, 128, 4, 2


def exact_int4_linear_impl(
    x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, n: int
) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    y = torch.empty(m, n, dtype=x.dtype, device=x.device)
    if m == 0:
        return y.reshape(*x.shape[:-1], n)
    bm, bn, warps, stages = _gemm_config(m)
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))
    _exact_int4_gemm_kernel[grid](
        x2,
        packed,
        scales,
        y,
        m,
        n,
        k,
        x2.stride(0),
        packed.stride(0),
        scales.stride(0),
        y.stride(0),
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=warps,
        num_stages=stages,
    )
    return y.reshape(*x.shape[:-1], n)


def _exact_int4_linear_fake(
    x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, n: int
) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], n)


def exact_int4_rows_impl(
    ids: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, k: int
) -> torch.Tensor:
    flat = ids.reshape(-1)
    out = torch.empty(flat.numel(), k, dtype=torch.bfloat16, device=ids.device)
    if flat.numel():
        block_r = 16
        grid = (triton.cdiv(flat.numel(), block_r), k // KB)
        _exact_int4_rows_kernel[grid](
            flat,
            packed,
            scales,
            out,
            flat.numel(),
            k,
            packed.stride(0),
            scales.stride(0),
            out.stride(0),
            BLOCK_R=block_r,
            num_warps=4,
        )
    return out.reshape(*ids.shape, k)


def _exact_int4_rows_fake(
    ids: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, k: int
) -> torch.Tensor:
    return torch.empty(*ids.shape, k, dtype=torch.bfloat16, device=ids.device)


direct_register_custom_op(
    op_name="gemma4_exact_int4_linear",
    op_func=exact_int4_linear_impl,
    fake_impl=_exact_int4_linear_fake,
)
direct_register_custom_op(
    op_name="gemma4_exact_int4_rows",
    op_func=exact_int4_rows_impl,
    fake_impl=_exact_int4_rows_fake,
)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def _bf16_preimage(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed fp64 interval of reals that round (RNE) to each bf16 value of x."""
    bits = x.view(torch.int16).to(torch.int32)
    nxt = (bits + 1).to(torch.int16).view(torch.bfloat16).to(torch.float64)
    prv = (bits - 1).to(torch.int16).view(torch.bfloat16).to(torch.float64)
    xv = x.to(torch.float64)
    # bits +/- 1 steps the magnitude, so the order flips for negative values.
    a = (xv + prv) / 2
    b = (xv + nxt) / 2
    return torch.minimum(a, b), torch.maximum(a, b)


def _fit_groups(
    groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """groups: bf16 [G, 32]. Returns (ok [G], q int8 [G, 32], d fp32 [G])."""
    num = groups.shape[0]
    ok = torch.zeros(num, dtype=torch.bool, device=groups.device)
    q_out = torch.zeros(groups.shape, dtype=torch.int8, device=groups.device)
    d_out = torch.zeros(num, dtype=torch.float32, device=groups.device)
    todo = torch.arange(num, device=groups.device)
    # The largest-magnitude value sits on level k for some k in 1..8; the
    # largest scale that fits is the grid's (a smaller one would also fit q*2).
    # Each level only revisits the groups that are still unresolved.
    for level in (8, 7, 6, 5, 4, 3, 2, 1):
        if todo.numel() == 0:
            break
        g = groups[todo]
        x = g.to(torch.float64)
        nonzero = x != 0
        amax = x.abs().amax(1, keepdim=True)
        lo, hi = _bf16_preimage(g)
        d0 = torch.where(amax == 0, torch.ones_like(amax), amax / level)
        # + 0.0 canonicalizes -0.0: a stored code of 0 decodes to +0.0, so a
        # -0.0 weight must fail the bit comparison below.
        q = (x / d0).round() + 0.0
        in_range = ((q >= -8) & (q <= 7) & ((q != 0) | ~nonzero)).all(1)
        qs = torch.where(q == 0, torch.ones_like(q), q)
        a, b = lo / qs, hi / qs
        dlo = torch.where(nonzero, torch.minimum(a, b), -torch.inf).amax(1)
        dhi = torch.where(nonzero, torch.maximum(a, b), torch.inf).amin(1)
        d = ((dlo + dhi) / 2).to(torch.float32)
        d = torch.where(torch.isfinite(d), d, torch.ones_like(d))
        recon = (q.to(torch.float32) * d[:, None]).to(torch.bfloat16)
        exact = (recon.view(torch.int16) == g.view(torch.int16)).all(1)
        fit = in_range & (dlo <= dhi) & exact
        hit = todo[fit]
        ok[hit] = True
        q_out[hit] = q[fit].to(torch.int8)
        d_out[hit] = d[fit]
        todo = todo[~fit]
    return ok, q_out, d_out


def _pack_rows(q: torch.Tensor) -> torch.Tensor:
    """Q int8 [R, K] -> int32 [R, K // 8] in the K-permuted layout."""
    r, k = q.shape
    u = (q.to(torch.int32) + 8).reshape(r, k // KB, 8, 32)
    out = u[:, :, 0, :].clone()
    for i in range(1, 8):
        out |= u[:, :, i, :] << (4 * i)
    return out.reshape(r, k // 8)


@torch.no_grad()
def fit_exact_int4(
    weight: torch.Tensor, chunk_rows: int = 4096
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(packed, scales) reproducing BF16 `weight` [N, K] bit-exactly, or None.

    Device-agnostic torch; every group is checked bit for bit in the same
    fp32-multiply / RNE-to-bf16 arithmetic as the kernels.
    """
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or weight.shape[1] % KB:
        return None
    n, k = weight.shape
    packed = torch.empty(n, k // 8, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // GROUP, dtype=torch.float32, device=weight.device)
    for r0 in range(0, n, chunk_rows):
        rows = weight[r0 : r0 + chunk_rows]
        ok, q, d = _fit_groups(rows.reshape(-1, GROUP))
        if not bool(ok.all()):
            return None
        packed[r0 : r0 + chunk_rows] = _pack_rows(q.reshape(rows.shape))
        scales[r0 : r0 + chunk_rows] = d.reshape(rows.shape[0], k // GROUP)
    return packed, scales


def decode_reference(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pure-torch decode of the K-permuted layout (tests / CPU reference)."""
    n, words = packed.shape
    k = words * 8
    w = packed.reshape(n, k // KB, 32)
    planes = [((w >> (4 * i)) & 0xF) - 8 for i in range(8)]  # [n, kb, 32] each
    q = torch.stack(planes, dim=2).reshape(n, k).to(torch.float32)
    s = scales.repeat_interleave(GROUP, dim=1)
    return (q * s).to(torch.bfloat16)


@torch.no_grad()
def encode_exact_int4(
    weight: torch.Tensor, chunk_rows: int = 4096
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """fit_exact_int4 plus a bit-exact check through the device decode kernel."""
    if not weight.is_cuda:
        return None
    encoded = fit_exact_int4(weight, chunk_rows)
    if encoded is None:
        return None
    packed, scales = encoded
    n, k = weight.shape
    ids = torch.arange(n, device=weight.device)
    for r0 in range(0, n, chunk_rows):
        rows = exact_int4_rows_impl(ids[r0 : r0 + chunk_rows], packed, scales, k)
        if not torch.equal(
            rows.view(torch.int16), weight[r0 : r0 + chunk_rows].view(torch.int16)
        ):
            raise RuntimeError("exact INT4 decode kernel is not bit-exact")
    return packed, scales


def _verify_gemm(weight, packed, scales) -> None:
    """Check the GEMM against an FP64 reference on a subset of output rows.

    The output is rounded to BF16, so the expected relative error is that
    rounding alone (~1.7e-3); anything past 2.5e-3 is a kernel defect.
    """
    n, k = weight.shape
    rows = torch.unique(
        torch.cat(
            [
                torch.arange(min(n, 2048), device=weight.device),
                torch.arange(max(0, n - 2048), n, device=weight.device),
                torch.randint(0, n, (4096,), device=weight.device),
            ]
        )
    )
    ref_w = weight[rows].to(torch.float64)
    for m in (1, 5, 33, 80):
        x = torch.randn(m, k, device=weight.device, dtype=torch.bfloat16)
        got = exact_int4_linear_impl(x, packed, scales, n)[:, rows].to(torch.float64)
        ref = x.to(torch.float64) @ ref_w.t()
        err = (got - ref).norm() / ref.norm().clamp_min(1e-30)
        if not bool(err < 2.5e-3):
            raise RuntimeError(
                f"exact INT4 GEMM disagrees with the FP64 reference "
                f"(m={m}, rel={err:.2e})"
            )


# --------------------------------------------------------------------------
# Layer methods and installation
# --------------------------------------------------------------------------


class ExactInt4EmbeddingMethod(UnquantizedEmbeddingMethod):
    """Embedding table and/or LM head from one exact INT4 copy."""

    def __init__(self, n: int, k: int):
        super().__init__()
        self.n = n
        self.k = k

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise RuntimeError("exact INT4 weights cannot be reloaded in place")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise RuntimeError("exact INT4 LM head is bias-free")
        return torch.ops.vllm.gemma4_exact_int4_linear(
            x, layer.exact_int4_packed, layer.exact_int4_scales, self.n
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.gemma4_exact_int4_rows(
            input_, layer.exact_int4_packed, layer.exact_int4_scales, self.k
        )


def check_supported(config) -> None:
    """Raise if the explicit opt-in cannot be honoured (never silently off)."""
    if config is None:
        raise RuntimeError("VLLM_GEMMA4_EXACT_INT4 needs a vLLM config")
    lora = config.lora_config
    targets = set(getattr(lora, "target_modules", None) or ()) if lora else set()
    if targets & {"embed_tokens", "lm_head"}:
        # Gemma4 declares no LoRA embedding modules, so adapters on the
        # linears never wrap the table; an explicit embedding/head target
        # would need the BF16 weight.
        raise ValueError(
            "VLLM_GEMMA4_EXACT_INT4 does not support LoRA on embed_tokens/lm_head"
        )
    if not getattr(config, "use_v2_model_runner", True):
        # The V1 proposer reads the target's embed_tokens.weight.
        raise ValueError("VLLM_GEMMA4_EXACT_INT4 requires the V2 model runner")


def _convert_table(name: str, modules: tuple[torch.nn.Module, ...]) -> bool:
    """Replace the shared BF16 weight of `modules` (embedding and/or head)."""
    first = modules[0]
    param = getattr(first, "weight", None)
    if param is None or any(getattr(m, "weight", None) is not param for m in modules):
        _STATE["skipped"][f"{name}:untied"] += 1
        return False
    if any(
        type(m.quant_method)
        not in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        for m in modules
    ) or any(getattr(m, "bias", None) is not None for m in modules):
        _STATE["skipped"][f"{name}:quantized_or_biased"] += 1
        return False
    weight = param.data
    encoded = encode_exact_int4(weight)
    if encoded is None:
        _STATE["skipped"][f"{name}:not_exact_int4"] += 1
        logger.info(
            "Gemma4 exact INT4: %s %s is not on an int4 x fp32-scale grid; "
            "kept in BF16",
            name,
            tuple(weight.shape),
        )
        return False
    packed, scales = encoded
    _verify_gemm(weight, packed, scales)
    n, k = weight.shape
    method = ExactInt4EmbeddingMethod(n, k)
    saved = weight.numel() * weight.element_size() - (
        packed.numel() * packed.element_size() + scales.numel() * scales.element_size()
    )
    for m in modules:
        del m.weight
    del param, weight
    for m in modules:
        m.register_buffer("exact_int4_packed", packed, persistent=False)
        m.register_buffer("exact_int4_scales", scales, persistent=False)
        m.quant_method = method
        m._gemma4_exact_int4 = True
    _STATE["installed"][f"{name}:{n}x{k}"] += 1
    _STATE["bytes_saved"] += saved
    return True


def install_gemma4_exact_int4_target(model: torch.nn.Module, config) -> None:
    """Gemma4ForCausalLM: the tied embedding / LM head."""
    check_supported(config)
    start = time.perf_counter()
    embed = model.model.embed_tokens
    head = getattr(model, "lm_head", None)
    if getattr(embed, "_gemma4_exact_int4", False):
        return
    modules = (embed,) if head is None or head is embed else (embed, head)
    _convert_table("target_embed_head", modules)
    torch.accelerator.empty_cache()
    logger.info(
        "Gemma4 exact INT4 (target, %.1f s): %s",
        time.perf_counter() - start,
        gemma4_exact_int4_state(),
    )


def install_gemma4_exact_int4_draft(draft_model: torch.nn.Module, config) -> None:
    """Gemma4 MTP assistant: its own draft-dim LM head (after embedding sharing)."""
    check_supported(config)
    head = draft_model.lm_head
    if getattr(head, "_gemma4_exact_int4", False):
        return
    if getattr(draft_model, "masked_embedding", None) is not None:
        _STATE["skipped"]["draft_head:centroid_masked"] += 1
        return
    # The assistant's own linears are also on the grid, but at decode sizes they
    # are launch-latency bound (8-16 us) and INT4 does not make them faster.
    _convert_table("draft_head", (head,))
    torch.accelerator.empty_cache()
    logger.info("Gemma4 exact INT4 (draft): %s", gemma4_exact_int4_state())
