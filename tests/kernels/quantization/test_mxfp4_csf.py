# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact native-layout reconstruction and shared-scratch graph safety."""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp4_csf import (
    CompressedMxfp4Scales,
)


def native_scales(values, device="cpu", padding=0):
    e, n, cols = values.shape
    assert cols % 4 == 0
    storage = torch.zeros((e, cols // 4, n + padding), dtype=torch.int32, device=device)
    out = storage.transpose(1, 2)[:, :n]
    out.copy_(values.contiguous().view(torch.int32).to(device))
    return out


@pytest.mark.cpu_test
@pytest.mark.parametrize("words,padding", [(1, 0), (1, 13), (3, 13)])
@pytest.mark.parametrize("rows", [19, 35])
def test_codec_preserves_outlier_bytes_and_expert_local_positions(words, padding, rows):
    """Every original byte is retained, including four exceptions in one word."""
    values = torch.full((3, rows, words * 4), 127, dtype=torch.uint8)
    values[:, :, 1::2] = 128
    values[0, 0, :4] = torch.tensor([0, 255, 247, 249], dtype=torch.uint8)
    values[1, -1, -4:] = torch.tensor([255, 0, 42, 200], dtype=torch.uint8)
    original = native_scales(values, padding=padding)
    csf = CompressedMxfp4Scales.from_native(original)
    assert csf.stride == original.stride()
    selectors = np.unpackbits(csf.selectors.numpy(), axis=-1, bitorder="little")
    restored = csf.bases.numpy()[..., None] + selectors[..., : words * 4]
    restored = restored.astype(np.uint8)
    offsets = csf.offsets.numpy()
    tiles = (len(offsets) - 1) // 3
    for expert in range(3):
        records = csf.exceptions.numpy()[
            offsets[expert * tiles] : offsets[(expert + 1) * tiles]
        ]
        restored[expert].flat[records & 0xFFFFFF] = records >> 24
    np.testing.assert_array_equal(restored, values.numpy())


@pytest.mark.cpu_test
def test_codec_rejects_non_native_layout_and_position_overflow():
    with pytest.raises(ValueError, match="MN-major"):
        CompressedMxfp4Scales.from_native(torch.zeros((2, 19, 3), dtype=torch.int32))
    # A strided view tests the bound without allocating the enormous plane.
    n = (1 << 22) + 1
    huge = torch.empty_strided((1, n, 1), (n, 1, n), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="24 bits"):
        CompressedMxfp4Scales.from_native(huge)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "counts",
    [
        torch.ones(6, dtype=torch.int32)[::2],
        torch.ones(3),
        torch.ones(1, 3, dtype=torch.int32),
    ],
)
def test_decoder_rejects_counts_that_would_select_incorrect_experts(counts):
    csf = CompressedMxfp4Scales.from_native(
        native_scales(torch.full((3, 19, 12), 127, dtype=torch.uint8))
    )
    with pytest.raises(ValueError, match="contiguous local integer counts"):
        csf.decode(csf.make_scratch(), counts)


@pytest.mark.cpu_test
@pytest.mark.parametrize("nested", [False, True])
def test_compression_excludes_draft_tags_and_shifted_mtp_layers(nested):
    from types import SimpleNamespace

    from transformers import PreTrainedConfig, Qwen3VLMoeConfig

    from vllm.compilation.backends import set_model_tag
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.utils.mxfp4_csf import (
        should_compress_mxfp4_scales,
    )
    from vllm.transformers_utils.config import get_hf_text_config

    config = VllmConfig()
    config.kernel_config.mxfp4_scale_compression = True
    hf_config = (
        Qwen3VLMoeConfig(text_config={"num_hidden_layers": 2})
        if nested
        else PreTrainedConfig(num_hidden_layers=2)
    )
    config.model_config = SimpleNamespace(
        hf_config=hf_config, hf_text_config=get_hf_text_config(hf_config)
    )
    layer = SimpleNamespace(layer_name="model.layers.0.mlp.experts")
    with set_current_vllm_config(config):
        assert should_compress_mxfp4_scales(layer)
        for tag in ("eagle_head", "draft_model", "medusa_head"):
            with set_model_tag(tag):
                assert not should_compress_mxfp4_scales(layer)
        layer.layer_name = "model.layers.2.mlp.experts"
        assert not should_compress_mxfp4_scales(layer)


@pytest.mark.cpu_test
@pytest.mark.parametrize("backend", ["auto", "uva"])
@pytest.mark.parametrize("uva_available,disable_uva", [(False, False), (True, True)])
def test_compression_rejects_non_uva_weight_offload_before_model_load(
    monkeypatch, backend, uva_available, disable_uva
):
    from vllm.config import KernelConfig, OffloadConfig, VllmConfig
    from vllm.config.offload import UVAOffloadConfig

    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_uva_available", lambda: uva_available
    )
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", str(int(disable_uva)))
    offload = OffloadConfig(
        offload_backend=backend, uva=UVAOffloadConfig(cpu_offload_gb=1)
    )
    VllmConfig(offload_config=offload)  # The default uncompressed path still works.
    with pytest.raises(ValueError, match="CPU weight offload without UVA"):
        VllmConfig(
            kernel_config=KernelConfig(mxfp4_scale_compression=True),
            offload_config=offload,
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("offload_mode", ["none", "uva", "prefetch"])
def test_compression_allows_other_offload_modes(monkeypatch, offload_mode):
    from vllm.config import KernelConfig, OffloadConfig, VllmConfig
    from vllm.config.offload import PrefetchOffloadConfig, UVAOffloadConfig

    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_uva_available", lambda: offload_mode == "uva"
    )
    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "0")
    offload = OffloadConfig(
        uva=UVAOffloadConfig(cpu_offload_gb=int(offload_mode == "uva")),
        prefetch=PrefetchOffloadConfig(
            offload_group_size=2 if offload_mode == "prefetch" else 0
        ),
    )
    VllmConfig(
        kernel_config=KernelConfig(mxfp4_scale_compression=True),
        offload_config=offload,
    )


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "rows,words,padding",
    [(19, 1, 0), (19, 1, 13), (19, 3, 13), (35, 3, 13), (256, 32, 0)],
)
def test_gpu_decoder_is_byte_exact_with_native_padding(rows, words, padding):
    values = torch.randint(0, 256, (3, rows, words * 4), dtype=torch.uint8)
    original = native_scales(values, "cuda", padding)
    csf = CompressedMxfp4Scales.from_native(original)
    scratch = csf.make_scratch()
    scratch.fill_(123456789)
    csf.decode(scratch)
    assert torch.equal(scratch, original)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="GPU required")
def test_active_routes_and_alternating_graphs_restore_poisoned_shared_scratch():
    """Replay restores active experts after another layer overwrites scratch."""
    a = native_scales(torch.randint(0, 256, (8, 35, 64), dtype=torch.uint8), "cuda")
    b = native_scales(torch.randint(0, 256, (8, 35, 64), dtype=torch.uint8), "cuda")
    ca, cb = (CompressedMxfp4Scales.from_native(x) for x in (a, b))
    scratch = ca.make_scratch()
    counts = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.int32, device="cuda")
    ca.decode(scratch, counts, max_active_experts=2)
    cb.decode(scratch, counts, max_active_experts=2)
    torch.accelerator.synchronize()
    ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga):
        ca.decode(scratch, counts, max_active_experts=2)
    with torch.cuda.graph(gb):
        cb.decode(scratch, counts, max_active_experts=2)
    for mask in ([0, 0, 0, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0, 0, 0], [0] * 8):
        counts.copy_(torch.tensor(mask, device="cuda", dtype=torch.int32))
        active = torch.tensor(mask, device="cuda", dtype=torch.bool)
        for graph, original in ((ga, a), (gb, b), (ga, a)):
            scratch.fill_(-1234567)
            graph.replay()
            assert torch.equal(scratch[active], original[active])
            assert torch.all(scratch[~active] == -1234567)
