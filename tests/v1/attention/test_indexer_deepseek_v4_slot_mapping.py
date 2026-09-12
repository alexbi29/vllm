# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.indexer as indexer_module
from tests.v1.attention.utils import create_vllm_config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    CompressedSlotMappingKernel,
)
from vllm.v1.attention.backends.mla.indexer import (
    BuildPrefillChunkMetadataKernel,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    ConvertReqIndexToGlobalIndexKernel,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressed_decode_lengths_keep_capture_address():
    """Capture uses four single-token rows; replay uses one four-token row."""
    config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B",
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=3,
        enable_adaptive_verification=True,
    )
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["dummy"],
        vllm_config=config,
        device=torch.device("cuda"),
        block_table_width=4,
    )

    def build(starts, seq_lens):
        cpu_starts = torch.tensor(starts, dtype=torch.int32)
        metadata = CommonAttentionMetadata(
            query_start_loc=cpu_starts.cuda(),
            query_start_loc_cpu=cpu_starts,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
            num_reqs=4,
            num_actual_tokens=4,
            max_query_len=int(cpu_starts.diff().max()),
            max_seq_len=128,
            block_table_tensor=torch.ones((4, 4), dtype=torch.int32, device="cuda"),
            slot_mapping=torch.arange(4, dtype=torch.int64, device="cuda"),
        )
        return builder.build(0, metadata).decode.seq_lens

    captured = build([0, 1, 2, 3, 4], [128, 128, 128, 128])
    replay = build([0, 4, 4, 4, 4], [128, 0, 0, 0])
    assert captured.data_ptr() == replay.data_ptr()
    assert captured.flatten().tolist() == [31, 31, 31, 32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_adaptive_flatten_replay_uses_device_query_lengths():
    """Reallocate a fixed token budget without updating the CPU boundaries."""
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=True)
    )
    builder.supports_varlen = False
    builder.decode_seq_lens_buffer = torch.zeros(8, dtype=torch.int32, device="cuda")
    builder.expanded_block_table_buffer = torch.zeros(
        (8, 2), dtype=torch.int32, device="cuda"
    )
    builder.decode_lens_buffer = torch.zeros(8, dtype=torch.int32, device="cuda")
    builder.arange_buffer = torch.arange(8, dtype=torch.int32, device="cuda")
    lens = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    seq = torch.tensor([13, 23], dtype=torch.int32, device="cuda")
    blocks = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32, device="cuda")
    cpu_lens = torch.tensor([3, 3], dtype=torch.int32)

    def prepare():
        return builder._prepare_decode_tensors(
            seq, blocks, lens, cpu_lens, starts, 2, 6, False, 4, 3
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepare()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_seq, out_blocks, _, batch_size, padded = prepare()
    lens.copy_(torch.tensor([1, 5], device="cuda"))
    starts.copy_(torch.tensor([0, 1], device="cuda"))
    seq.copy_(torch.tensor([11, 25], device="cuda"))
    graph.replay()
    assert batch_size == 6 and not padded
    assert out_seq.tolist() == [11, 21, 22, 23, 24, 25]
    assert out_blocks.tolist() == [[5, 6]] + [[7, 8]] * 5


@pytest.mark.parametrize(
    ("is_prefilling", "expected_treat_short_extends_as_decodes"),
    [
        (torch.tensor([False, False]), True),
        (torch.tensor([False, True]), False),
    ],
)
def test_indexer_builder_keeps_short_prefill_continuations_as_prefills(
    monkeypatch,
    is_prefilling,
    expected_treat_short_extends_as_decodes,
):
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.decode_threshold = 1
    builder.reorder_batch_threshold = 1
    builder.use_flattening = False
    builder.supports_varlen = False
    # PCP is off on this path, so treat_short_extends_as_decodes reduces to
    # ``not has_prefilling_rows`` (the DSv4 ubatch-continuation guard).
    builder.use_pcp = False

    captured = {}

    def fake_split_decodes_and_prefills(
        common_attn_metadata,
        *,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        captured["treat_short_extends_as_decodes"] = treat_short_extends_as_decodes
        raise RuntimeError("stop after split_decodes_and_prefills")

    monkeypatch.setattr(
        indexer_module,
        "split_decodes_and_prefills",
        fake_split_decodes_and_prefills,
    )
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor([128, 129], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=129,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(2, dtype=torch.int64),
        is_prefilling=is_prefilling,
        seq_lens_cpu_upper_bound=torch.tensor([128, 129], dtype=torch.int32),
    )

    with pytest.raises(RuntimeError, match="stop after"):
        builder.build(common_prefix_len=0, common_attn_metadata=metadata)

    assert (
        captured["treat_short_extends_as_decodes"]
        is expected_treat_short_extends_as_decodes
    )


def test_indexer_warmup_normalizes_zero_compress_ratios():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 0, 4, 128, 0], index_kpool=32)
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = BuildPrefillChunkMetadataKernel().get_warmup_keys(config)

    assert {key.compress_ratio for key in keys} == {1, 4, 32, 128}
    assert {(key.query_slice_start, key.query_slice_stop) for key in keys} == {
        (query_slice_start, query_slice_stop)
        for query_slice_start in (1, 2, 16)
        for query_slice_stop in (1, 2, 16)
    }


def test_compressed_slot_mapping_warmup_includes_index_kpool():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_kpool=32)),
    )

    keys = CompressedSlotMappingKernel().get_warmup_keys(config)
    assert {(key.compress_ratio, key.block_size) for key in keys} == {(32, 2)}


def test_index_conversion_warmup_uses_physical_block_stride():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_config=SimpleNamespace(index_topk=2048),
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = ConvertReqIndexToGlobalIndexKernel().get_warmup_keys(
        config,
        block_stride_rows=4096,
    )
    assert {key.block_stride_rows for key in keys} == {4096}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_num_states():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # num_states = block_size // tokens_per_state = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.num_states  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)
