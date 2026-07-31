# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@cache
def _needs_dense_kv_fallback() -> bool:
    """True on SM120, where FA4's forward has no paged-KV support.

    ``flash_attn_varlen_func``'s ``arch // 10 == 12`` branch builds
    ``FlashAttentionForwardSm120`` with no paging parameters and asserts
    ``page_table is None`` ("Paged KV not supported on SM 12.0 in this PR",
    from Dao-AILab #2329). The same assert exists in ``third_party.tml_fa4``,
    so both bias paths are blocked. Until the kernel learns paging, gather the
    paged cache into a dense varlen buffer and call the non-paged kernel.
    """
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 12


# Extra token-rows gathered past the logical dense-buffer end; must cover the
# SM120 kernel's cp.async prefetch depth (4 tiles of 128 is generous).
_TAIL_SLACK_TOKENS = 512


import triton
import triton.language as tl


@triton.jit
def _gather_kv_triton(
    src,                # packed cache rows: (num_slots, 2 * HD) contiguous
    phys,               # (n_padded,) int64: physical slot per output token
    k_out,              # (n_padded, H, D) contiguous
    v_out,              # (n_padded, H, D) contiguous
    HD: tl.constexpr,   # num_kv_heads * head_dim (elements per K or V half)
    D: tl.constexpr,    # head_dim
):
    """One program per token: read the packed (head, K|V, D) cache row once,
    coalesced, and scatter the K/V halves to their contiguous dense buffers."""
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(phys + t)
    offs = tl.arange(0, HD)
    h = offs // D
    d = offs - h * D
    row = src + slot * (2 * HD)
    k = tl.load(row + h * (2 * D) + d)
    v = tl.load(row + h * (2 * D) + D + d)
    tl.store(k_out + t * HD + offs, k)
    tl.store(v_out + t * HD + offs, v)


def _pow2_pages(pages: int, table_width: int) -> int:
    """Bucket a page count to a power of two, capped at the table width.

    Bucketing keeps the gather at a handful of distinct buffer shapes instead
    of a new one every step -- kinder to the caching allocator and to
    cudagraph shape bucketing.
    """
    pages = 1 << max(0, max(1, pages) - 1).bit_length()
    return min(pages, table_width)


def _packed_kv_base(
    key_cache: torch.Tensor, value_cache: torch.Tensor
) -> torch.Tensor | None:
    """Return a flat ``(num_slots, num_kv_heads * 2 * head_dim)`` view of the
    packed K/V storage, or ``None`` if the caches are not packed.

    ``FlashAttentionBackend`` packs K and V into the content dim (logical
    ``(B, H, N, 2*D)``); under the NHD layout the physical memory is
    token-major ``(B, N, H, 2*D)`` contiguous, and ``_split_kv_cache`` hands us
    the two halves as strided views over that one buffer. Detecting that lets
    the fallback gather K and V for a token with a **single coalesced row
    copy** instead of two strided ``index_select``s -- the strided form fell
    into ATen's generic scatter-gather kernel at ~347 us/call (71% of decode
    GPU time); the packed form is a plain contiguous gather.
    """
    num_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    if (
        key_cache.stride(-1) == 1
        and key_cache.stride(2) == 2 * head_dim
        and key_cache.stride(1) == num_kv_heads * 2 * head_dim
        and key_cache.stride(0) == block_size * num_kv_heads * 2 * head_dim
        and value_cache.stride() == key_cache.stride()
        and value_cache.storage_offset() - key_cache.storage_offset() == head_dim
    ):
        row = num_kv_heads * 2 * head_dim
        return torch.as_strided(
            key_cache, (num_blocks * block_size, row), (row, 1)
        )
    return None


def _gather_dense_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seq_len: int | None,
    window_left: int,
    max_seqlen_q: int,
) -> dict[str, Any]:
    """Materialize the paged K/V into padded-ragged dense buffers for the
    non-paged FA4 kernel (SM120 fallback).

    Returns the ``k``/``v``/``cu_seqlens_k``/``seqused_k``/``max_seqlen_k``
    kwargs. Every request occupies a fixed ``max_k`` stride in the dense
    buffer; ``SeqlenInfo.create`` takes the offset from ``cu_seqlens`` but the
    length from ``seqused`` (seqlen_info.py), so the padding is never read.

    Two trims keep the copy small:

    - **Global layers** (``window_left < 0``): front-trim the page table to
      ``max_seq_len`` pages. ``max_seq_len`` is a **host-side int** from
      ``FlashAttentionMetadata`` -- no device read, because a sync is illegal
      under the warmup's ``FakeTensorMode`` and aborts CUDA graph capture.
    - **Local layers** (``window_left >= 0``): gather only the last
      ``window_left + max_seqlen_q`` tokens, page-aligned per request, and
      shift ``seqused_k`` accordingly. The rel-bias score_mod and the window
      mask both depend only on q/k *distances*, which the shift preserves.
      Tokens between the page-aligned start and the oldest row's window floor
      are below every row's window, masked by the kernel, never read.

    **Every shape here is derived from tensor shapes or host ints, never from
    tensor values.** The CuTeDSL warmup traces this under ``FakeTensorMode``
    to discover compile keys, so a ``.item()`` or boolean-mask index would
    raise ``DataDependentOutputException``.
    """
    num_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    batch, table_width = block_table.shape
    device = key_cache.device

    if window_left >= 0:
        # Local layer: page-aligned tail window per request.
        w_tokens = window_left + max_seqlen_q
        n_pages = _pow2_pages(
            (w_tokens + block_size - 1) // block_size + 1, table_width
        )
        start_page = (
            (cache_seqlens - w_tokens).clamp_min_(0) // block_size
        )  # (batch,) int32
        col = start_page.to(torch.int64).unsqueeze(1) + torch.arange(
            n_pages, device=device, dtype=torch.int64
        )
        col.clamp_(0, table_width - 1)
        pages = block_table.gather(1, col)  # (batch, n_pages)
        seqused_k = cache_seqlens - start_page * block_size
    else:
        # Global layer: front-trim to the longest sequence in the batch.
        if max_seq_len is not None and max_seq_len > 0:
            n_pages = _pow2_pages(
                (max_seq_len + block_size - 1) // block_size, table_width
            )
        else:
            n_pages = table_width
        pages = block_table[:, :n_pages]
        seqused_k = cache_seqlens

    max_k = n_pages * block_size
    # Padded cudagraph batch rows carry STALE seq_lens (only real rows are
    # rewritten each step). A stale value above max_k would make the kernel
    # read past that row's dense-buffer stride -- for the last row, past the
    # buffer itself (illegal access at decode conc>1). Clamp: padded rows
    # have zero-length q, so their (garbage) attention output is never read.
    seqused_k = torch.clamp(seqused_k, min=0, max=max_k)
    # Physical slot of every (request, logical position) pair:
    #   slot = page * block_size + pos % block_size
    base = pages.to(torch.int64) * block_size                # (batch, n_pages)
    off = torch.arange(block_size, device=device, dtype=torch.int64)
    phys = (base.unsqueeze(2) + off.view(1, 1, -1)).reshape(batch * max_k)
    # Page-table entries outside a request's live range are arbitrary; clamp
    # so the gather stays in bounds. Those rows are masked/beyond seqused_k
    # and never read by the kernel.
    phys = phys.clamp_(0, num_blocks * block_size - 1)
    # Tail slack: the SM120 kernel's cp.async pipeline prefetches K/V tiles
    # past the last valid tile with imperfect predication (verified via CUDA
    # coredump: Warp MMU Fault on an LDGSTS into the K/V smem stage). The
    # over-read lands inside the next request's rows except for the LAST
    # request, where it runs off the allocation -- so gather a few extra
    # (valid, unread) rows and hand the kernel a view of the logical size.
    n_tokens = batch * max_k
    phys = torch.cat([phys, phys.new_zeros(_TAIL_SLACK_TOKENS)])

    packed = _packed_kv_base(key_cache, value_cache)
    if packed is not None and not isinstance(
        phys, torch._subclasses.fake_tensor.FakeTensor
    ):
        # Triton gather: one program per token reads the full packed
        # (H, K|V, D) cache row ONCE, fully coalesced, and writes the K and V
        # halves to their contiguous dense buffers. Alternatives measured on
        # this path: index_select on the strided _split_kv_cache view fell
        # into ATen's generic scatter-gather at ~347 us/call (71% of decode
        # GPU time); a row-interleaved index_select pair read 256B strided of
        # every 2KB row at ~96 us/call (42%). The kernel must also get
        # CONTIGUOUS k/v: handing it (H*2D, 2D, 1)-strided views faults
        # (CUDA 700) inside the vLLM server process despite the identical
        # call passing standalone -- root cause never isolated.
        n_padded = n_tokens + _TAIL_SLACK_TOKENS
        dense_k = torch.empty(
            n_padded, num_kv_heads, head_dim,
            dtype=key_cache.dtype, device=device,
        )
        dense_v = torch.empty_like(dense_k)
        _gather_kv_triton[(n_padded,)](
            packed, phys, dense_k, dense_v,
            HD=num_kv_heads * head_dim, D=head_dim,
        )
        dense_k = dense_k[:n_tokens]
        dense_v = dense_v[:n_tokens]
    else:
        # FakeTensorMode warmup trace (or an unexpected cache layout): use
        # plain ATen ops so tracing sees the same output shapes/strides.
        flat_k = key_cache.reshape(num_blocks * block_size, num_kv_heads, head_dim)
        flat_v = value_cache.reshape(num_blocks * block_size, num_kv_heads, head_dim)
        dense_k = flat_k.index_select(0, phys)[:n_tokens]
        dense_v = flat_v.index_select(0, phys)[:n_tokens]

    cu_seqlens_k = (
        torch.arange(batch + 1, device=device, dtype=torch.int32) * max_k
    )
    return {
        "k": dense_k,
        "v": dense_v,
        "cu_seqlens_k": cu_seqlens_k,
        # True per-request lengths; the padded stride above is never read.
        "seqused_k": seqused_k,
        "max_seqlen_k": max_k,
    }


from collections import deque

_VALIDATE_RING: deque = deque(maxlen=90)


def _validate_call(
    q: torch.Tensor,
    kv_kwargs: dict[str, Any],
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    rel_logits: torch.Tensor,
) -> None:
    """Debug-only (INKLING_FA4_VALIDATE=1): check every input invariant of the
    dense-path FA4 call on the host and keep a ring of call summaries. Slow
    (syncs); meant to run under CUDA_LAUNCH_BLOCKING while chasing kernel
    faults."""
    if isinstance(q, torch._subclasses.fake_tensor.FakeTensor):
        return
    cu_q = cu_seqlens_q.cpu().tolist()
    su = kv_kwargs["seqused_k"].cpu().tolist()
    cu_k = kv_kwargs["cu_seqlens_k"].cpu().tolist()
    max_k = kv_kwargs["max_seqlen_k"]
    k = kv_kwargs["k"]
    summary = {
        "q": tuple(q.shape), "nt": cu_q[-1], "cu_q": cu_q,
        "seqused": su, "max_k": max_k, "k_rows": k.shape[0],
        "max_q": max_seqlen_q, "rel": tuple(rel_logits.shape),
    }
    _VALIDATE_RING.append(summary)
    errs = []
    if cu_q[0] != 0 or any(b < a for a, b in zip(cu_q, cu_q[1:])):
        errs.append("cu_seqlens_q not monotone from 0")
    if cu_q[-1] != q.shape[0]:
        errs.append(f"cu_q[-1]={cu_q[-1]} != q rows {q.shape[0]}")
    if any(b - a > max_seqlen_q for a, b in zip(cu_q, cu_q[1:])):
        errs.append("per-request q len exceeds max_seqlen_q")
    if any(s < 0 or s > max_k for s in su):
        errs.append(f"seqused out of [0, {max_k}]: {su}")
    if cu_k != [i * max_k for i in range(len(su) + 1)]:
        errs.append("cu_seqlens_k not the expected uniform stride")
    if k.shape[0] < len(su) * max_k:
        errs.append(f"k rows {k.shape[0]} < batch*max_k {len(su) * max_k}")
    if rel_logits.shape[0] != q.shape[0]:
        errs.append("rel_logits rows != q rows")
    if not (k.is_contiguous() and kv_kwargs["v"].is_contiguous()):
        errs.append("k/v not contiguous")
    if errs:
        raise RuntimeError(f"FA4 input invariant violated: {errs}; {summary}")


def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    """Round the FA4 scheduling bound up to a power of two."""
    return 1 << max(0, max_seqlen_q - 1).bit_length()


@cache
def _use_sheared_bias() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (10, 11)


@cache
def _get_score_mod(rel_extent: int) -> Callable:
    """Return the score modification that adds Inkling relative bias."""
    import cutlass.cute as cute
    from cutlass.cute import Float32

    from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        rel_dist_0 = rel_dist[0]
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        # `apply_score_mod` runs on every tile coordinate before masking, so
        # the q-row can be out of range on ragged/degenerate tiles. The kv
        # axis is clamped above; clamp the q axis symmetrically or the gather
        # walks off `rel_logits` (vllm-project/vllm#49049: deterministic
        # illegal address on sm_121a, silent wrong-reads elsewhere).
        n_rows = rel_logits.shape[0]
        q_row = global_q_idx[0]
        q_row = q_row if q_row >= 0 else 0
        q_row = q_row if q_row < n_rows else (n_rows - 1)

        rel_bias = rel_logits[q_row, h_idx[0], rel_idx]
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
        return scores + rel_bias

    return score_mod_rel_bias


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Return the split-KV cap for Inkling relative attention."""
    capability = current_platform.get_device_capability()
    # SM90 has no split-KV in FA4, and the SM120 forward asserts
    # `not is_split_kv` outright (cute/interface.py, `arch // 10 == 12`).
    # Only SM100/110 take the tml-fa4 sheared path that supports splitting.
    if capability is not None and capability.major in (9, 12):
        return 1
    if is_local:
        return 1

    q_rows = max_query_len * (num_heads // num_kv_heads)
    q_tiles = (q_rows + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    # Shearing makes split/combine overhead more visible. Multi-tile causal
    # prefill saturates around 64 CTAs. Batch-1 decode at very long context is
    # memory-bound and uses a TP-specific cap measured through 1M KV tokens.
    target_ctas = (
        256 if q_tiles == 1 and batch_size == 1 else (128 if q_tiles == 1 else 64)
    )
    max_splits = 128
    if q_tiles == 1 and batch_size == 1:
        if num_kv_heads == 8:
            max_splits = 16
        elif num_kv_heads == 4 or max_kv_len <= 8192:
            max_splits = 32
        elif max_kv_len <= 65536:
            max_splits = 64
        else:
            max_splits = 128
    return max(
        1,
        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


def inkling_fa4_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 32,
    out: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Paged varlen FA4 over the bound K/V cache with the Inkling relative bias.

    ``q`` is ``(num_tokens, num_heads, head_dim)``; ``key_cache`` / ``value_cache``
    are the paged caches ``(num_blocks, block_size, num_kv_heads, head_dim)``;
    ``block_table`` is the per-request page table and ``cache_seqlens`` the
    per-request KV lengths (``seqused_k``). ``rel_logits`` is
    ``(num_tokens, num_heads, rel_extent)``.

    Hopper uses standard FA4's score-mod gather. Blackwell uses tml-fa4's
    sheared relative-bias layout.
    """
    # cute uses (None, None) to mean "no window".
    cute_window = (None, None) if window_size == (-1, -1) else window_size

    rel_logits = rel_logits.contiguous()
    if _use_sheared_bias():
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func

        bias_kwargs: dict[str, Any] = {"rel_bias": rel_logits}
    else:
        from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }

    if _use_sheared_bias() or not _needs_dense_kv_fallback():
        kv_kwargs: dict[str, Any] = {
            "k": key_cache,
            "v": value_cache,
            "seqused_k": cache_seqlens,
            "page_table": block_table,
        }
    else:
        # SM120: no paged-KV in the FA4 forward. Gather to dense varlen and
        # call the non-paged kernel.
        logger.warning_once(
            "SM120 has no paged-KV support in the FA4 forward; Inkling "
            "attention is gathering the paged KV cache into a dense buffer "
            "on every call."
        )
        kv_kwargs = _gather_dense_kv(
            key_cache,
            value_cache,
            block_table,
            cache_seqlens,
            max_seq_len,
            window_size[0],
            max_seqlen_q,
        )

    import os

    if os.environ.get("INKLING_FA4_VALIDATE"):
        _validate_call(q, kv_kwargs, cu_seqlens_q, max_seqlen_q, rel_logits)

    # Pin the TVM-FFI environment stream to torch's current stream for the
    # launch: the CuTeDSL kernels are compiled with
    # `make_fake_stream(use_tvm_ffi_env_stream=True)` and nothing in vLLM
    # sets that env stream otherwise.
    import tvm_ffi

    try:
        with tvm_ffi.use_torch_stream():
            ret = flash_attn_varlen_func(
                q=q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=cute_window,
                num_splits=num_splits,
                return_lse=False,
                out=out,
                **kv_kwargs,
                **bias_kwargs,
            )
    except Exception:
        if _VALIDATE_RING:
            logger.error(
                "FA4 call failed; last %d call summaries (newest last):\n%s",
                len(_VALIDATE_RING),
                "\n".join(str(e) for e in _VALIDATE_RING),
            )
        raise
    if isinstance(ret, tuple):
        return ret[0]
    return ret
