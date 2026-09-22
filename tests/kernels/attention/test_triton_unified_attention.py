# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

DEVICE_TYPE = current_platform.device_type

NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]

DTYPES = [torch.bfloat16]
QDTYPES = [None, current_platform.fp8_dtype()]
FP8_DTYPE = current_platform.fp8_dtype()

# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]

# 0: use 2D kernel for decode
# 8: use 3D kernel for decode
SEQ_THRESHOLD_3D_VALUES = [0, 8]


@pytest.mark.parametrize("query_len,segments", [(1, 1), (1, 4), (5, 1)])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("block_size", [16, 256])
def test_gemma4_compact_kv(
    query_len, segments, kv_heads, layout, block_size, monkeypatch
):
    """Compact paged attention agrees with explicit BF16 K reconstruction.

    TRITON_INTERPRET=1 runs the same kernels on CPU without a GPU allocation.
    Covers TP1/TP2 head counts, page indirection, a partial page, masked
    cache writes, chunked prefill and segmented decode.
    """
    import os

    from vllm.v1.attention.ops.gemma4_compact_cache import write_compact_cache

    device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else DEVICE_TYPE
    if device == "cpu":
        # Triton 3.7.1's interpreter feeds BF16 uint16 storage directly to
        # numpy.matmul. Decode BF16 operands for the CPU-only test runner.
        import numpy as np
        from triton.runtime.interpreter import TensorHandle, interpreter_builder

        from vllm.triton_utils import tl

        original_dot = interpreter_builder.create_dot
        original_cast = interpreter_builder.cast_impl

        def bf16_cast(src, dst):
            if src.dtype.scalar == tl.float32 and dst.scalar == tl.bfloat16:
                # Its default FP32->BF16 cast also truncates instead of RNE.
                data = torch.from_numpy(src.data).to(torch.bfloat16)
                return TensorHandle(data.view(torch.uint16).numpy(), tl.bfloat16)
            return original_cast(src, dst)

        def bf16_dot(a, b, d, precision, max_imprecise):
            def decode(x):
                if x.dtype == tl.bfloat16:
                    data = (x.data.astype(np.uint32) << 16).view(np.float32)
                    return TensorHandle(data, tl.float32)
                return x

            return original_dot(decode(a), decode(b), d, precision, max_imprecise)

        monkeypatch.setattr(interpreter_builder, "create_dot", bf16_dot)
        monkeypatch.setattr(interpreter_builder, "cast_impl", bf16_cast)
    torch.manual_seed(17)
    dtype = torch.bfloat16
    heads, length = kv_heads * 8, 37
    key = torch.randn(length + 1, kv_heads, 512, dtype=dtype, device=device)
    value = torch.randn_like(key)
    norm = torch.randn(512, dtype=dtype, device=device)
    query = torch.randn(query_len, heads, 512, dtype=dtype, device=device) * 0.1
    cache = torch.full((4, kv_heads, block_size, 640), -7, dtype=dtype, device=device)
    if layout == "NHD":
        cache = cache.transpose(1, 2).contiguous().transpose(1, 2)
    block_table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device=device)
    positions = torch.arange(length, device=device)
    slots = (
        block_table[0, positions // block_size] * block_size + positions % block_size
    )
    slots = torch.cat((slots, torch.tensor([-1], device=device))).long()
    write_compact_cache(key, value, cache, slots)
    # Padding writes must not touch an unused page.
    torch.testing.assert_close(cache[1], torch.full_like(cache[1], -7))
    packed = cache[block_table[0, :3].long()].transpose(1, 2).reshape(-1, kv_heads, 640)
    packed = packed[:length]
    torch.testing.assert_close(packed[..., :64], key[:length, ..., :64])
    torch.testing.assert_close(packed[..., 64:128], key[:length, ..., 256:320])
    torch.testing.assert_close(packed[..., 128:], value[:length])

    reconstructed = (value[:length].float() * norm.float()).to(dtype)
    reconstructed[..., :64] = key[:length, ..., :64]
    reconstructed[..., 256:320] = key[:length, ..., 256:320]
    dense_k = reconstructed.repeat_interleave(8, dim=1).float()
    dense_v = value[:length].repeat_interleave(8, dim=1).float()
    logits = torch.einsum("qhd,khd->hqk", query.float(), dense_k)
    mask = (
        positions[None, :]
        > (length - query_len + torch.arange(query_len, device=device))[:, None]
    )
    logits.masked_fill_(mask, -float("inf"))
    expected = torch.einsum(
        "hqk,khd->qhd", logits.softmax(-1).to(dtype).float(), dense_v
    ).to(dtype)
    output = torch.empty_like(query)
    k, v = cache.transpose(1, 2).split([128, 512], dim=-1)
    unified_attention(
        q=query,
        k=k,
        v=v,
        out=output,
        cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        max_seqlen_q=query_len,
        seqused_k=torch.tensor([length], dtype=torch.int32, device=device),
        max_seqlen_k=length,
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=8 if segments > 1 else None,
        num_par_softmax_segments=segments,
        softmax_segm_output=torch.empty((8, heads, segments, 512), device=device),
        softmax_segm_max=torch.empty((8, heads, segments), device=device),
        softmax_segm_expsum=torch.empty((8, heads, segments), device=device),
        compact_k_norm=norm,
    )
    torch.testing.assert_close(output, expected, atol=0.016, rtol=0.02)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    q_dtype: torch.dtype | None,
    seq_threshold_3D: int,
) -> None:
    torch.set_default_device(DEVICE_TYPE)

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    kv_quant_mode = KVQuantMode.NONE
    if q_dtype is not None:
        # Use non-1 scales so FP8 Q/K/V descale handling is tested explicitly.
        q_scale = torch.tensor(0.75, dtype=torch.float32)
        k_scale = torch.tensor(0.5, dtype=torch.float32)
        v_scale = torch.tensor(0.25, dtype=torch.float32)
        q_descale = q_scale
        scale_shape = (num_seqs, num_kv_heads)
        k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
        v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)
        maybe_quantized_query = (query / q_scale).to(q_dtype)
        maybe_quantized_key_cache = (key_cache / k_scale).to(q_dtype)
        maybe_quantized_value_cache = (value_cache / v_scale).to(q_dtype)
        kv_quant_mode = KVQuantMode.FP8_PER_TENSOR

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=kv_quant_mode,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_bf16_query_fp8_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test bf16 Q with FP8 per-tensor KV cache (dequant via _cast_kv_tile)."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (-1, -1)
    scale = head_size**-0.5

    dtype = torch.bfloat16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    fp8_key_cache = (key_cache / k_scale).to(FP8_DTYPE)
    fp8_value_cache = (value_cache / v_scale).to(FP8_DTYPE)

    scale_shape = (num_seqs, num_kv_heads)
    k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
    v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=fp8_key_cache,
        v=fp8_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1)] * 533,
        [(533, 533)] * 533,
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_fp16_input_fp8_output(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test with fp16 input and fp8 output using output_scale."""
    torch.set_default_device(DEVICE_TYPE)

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    dtype = torch.float16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty(sum(query_lens), num_query_heads, head_size, dtype=FP8_DTYPE)

    output_scale = torch.tensor(0.5, dtype=torch.float32)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        output_scale=output_scale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    output_fp16 = output.to(torch.float32) * output_scale.item()
    output_fp16 = output_fp16.to(torch.float16)

    atol, rtol = 2e-1, 2e-1
    (
        torch.testing.assert_close(output_fp16, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output_fp16 - ref_output))}",
    )


# USE_TD path covers two head-size regimes:
# - pow2 (HEAD_SIZE == HEAD_SIZE_PADDED): full TD path including Q/O.
# - non-pow2 (96, HEAD_SIZE_PADDED=128): gates USE_TD_QO off — Q load
#   and output store fall back to pointer path, KV tile TD load remains.
# The non-pow2 case mirrors real models like Phi-3-mini (head_size=96).
HEAD_SIZES_USE_TD = [128, 256, 96]


def _run_use_td_case(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    sliding_window: int | None,
    soft_cap: float | None,
    seq_threshold_3D: int,
    dtype: torch.dtype = torch.bfloat16,
    num_blocks: int = 2048,
) -> None:
    """Shared driver for the USE_TD test cases.

    Runs ``unified_attention(..., use_td=True)`` and compares against the
    reference paged-attention implementation that the sibling non-TD
    tests use.
    """
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        use_td=True,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    torch.testing.assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES_USE_TD)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_use_td(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Exercise the USE_TD (tensor-descriptor) Q/K/V load/store path.

    Covers both 2D and 3D kernels via ``seq_threshold_3D``. Two routes
    to the USE_TD_QO=False fallback (pointer path for Q/O with TD still
    active for KV tile loads):

    - non-pow2 ``num_queries_per_kv`` via ``NUM_HEADS`` entry ``(5, 1)``,
    - non-pow2 ``head_size`` via ``HEAD_SIZES_USE_TD`` entry ``96``.
    """
    _run_use_td_case(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        seq_threshold_3D=seq_threshold_3D,
        num_blocks=num_blocks,
    )


# Prefill-heavy shape: long query drives the prefill kernel path where
# ``_get_tile_size`` returns 32, which exceeds block_size=16 and must be
# clamped by the fix in 'clamp TILE_SIZE to block_size when USE_TD'.
# Only the prefill launch exercises the clamp, so parameterize only over
# the (num_heads, seq_threshold_3D=0) combinations needed to cover it.
@pytest.mark.parametrize("num_heads", [(4, 4), (5, 1)])
@torch.inference_mode()
def test_triton_unified_attn_use_td_tile_clamp(
    num_heads: tuple[int, int],
) -> None:
    """Regression guard: ``USE_TD`` needs ``BLOCK_SIZE % TILE_SIZE == 0``.

    With ``block_size=16`` and ``head_size=128`` (non-Gemma3),
    ``_get_tile_size`` returns 32 for prefill, which violates the
    ``USE_TD`` constraint unless clamped to ``block_size``.  Without
    the clamp the triton kernel ``static_assert`` fires at compile time.
    """
    _run_use_td_case(
        seq_lens=[(256, 256), (128, 128)],
        num_heads=num_heads,
        head_size=128,
        block_size=16,
        sliding_window=None,
        soft_cap=None,
        seq_threshold_3D=0,
    )
