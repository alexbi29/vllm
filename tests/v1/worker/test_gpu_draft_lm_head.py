# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import json
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ModelConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.online.lm_head import (
    quantized_lm_head_copy,
    refresh_quantized_lm_heads,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker

# Not a multiple of the vocab padding, so the head has padded rows.
VOCAB, HIDDEN = 4000, 512
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Marlin needs CUDA"
)
# NVFP4 needs a CUDA W4A16 kernel; ROCm only runs the fp8 cases.
requires_nvfp4 = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="nvfp4 needs a CUDA W4A16 kernel"
)
NVFP4 = pytest.param("nvfp4", marks=requires_nvfp4)


@pytest.fixture(autouse=True)
def head_model_config(default_vllm_config, tmp_path):
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": VOCAB,
        "max_position_embeddings": 1024,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    default_vllm_config.model_config = ModelConfig(
        model=str(tmp_path),
        dtype="bfloat16",
        max_model_len=1024,
        skip_tokenizer_init=True,
    )


def _target_head(seed: int = 0) -> ParallelLMHead:
    head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(seed)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
    return head.to("cuda", torch.bfloat16)


def _tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(module.named_parameters()) | dict(module.named_buffers())


@requires_cuda
@pytest.mark.parametrize(
    ("quantization", "max_rel_err", "min_top1"),
    [("fp8", 0.05, 0.9), pytest.param("nvfp4", 0.15, 0.75, marks=requires_nvfp4)],
)
def test_tracks_bf16_head_through_logits_processor(
    default_vllm_config, dist_init, monkeypatch, quantization, max_rel_err, min_top1
):
    head = _target_head()
    target_weight = head.weight.detach().clone()
    draft_head = quantized_lm_head_copy(head, quantization)

    assert isinstance(draft_head, ParallelLMHead)
    # The source head keeps its bf16 weight, and no bf16 copy is kept around.
    assert torch.equal(head.weight, target_weight)
    assert draft_head.weight.dtype != torch.bfloat16

    logits_processor = LogitsProcessor(VOCAB)
    hidden = torch.randn(2048, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = logits_processor(head, hidden).float()
    out = logits_processor(draft_head, hidden).float()
    assert out.shape == ref.shape == (2048, VOCAB)
    rel_err = ((out - ref).norm() / ref.norm()).item()
    top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert rel_err < max_rel_err
    assert top1 > min_top1
    if quantization == "fp8":
        from vllm.model_executor.layers.quantization.online.fp8 import (
            _fp8_channel_scale,
            _fp8_quant_per_channel,
        )

        scales = _fp8_channel_scale(head.weight.abs().amax(-1, keepdim=True))
        baseline = _fp8_quant_per_channel(head.weight, scales).float() * scales
        reference = head.weight[:VOCAB].float().T
        basis = torch.eye(HIDDEN, device="cuda", dtype=torch.bfloat16)
        reconstructed = logits_processor(draft_head, basis).float()
        before = (baseline[:VOCAB].T - reference).square().sum()
        after = (reconstructed - reference).square().sum()
        assert after < before * 0.99


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_fitted_head_scales_preserve_or_reduce_each_rows_fp64_error(dtype, fp8_dtype):
    from vllm.model_executor.layers.quantization.online.lm_head_fp8 import (
        fit_fp8_head_scales,
    )

    generator = torch.Generator().manual_seed(29)
    weight = torch.randn(37, 1024, generator=generator, dtype=dtype)
    weight[0].zero_()
    weight[1].fill_(0.125)
    weight[2] *= 1e-5
    weight[3] *= 100
    original = weight.clone()
    limit = torch.finfo(fp8_dtype).max
    scale = (weight.float().abs().amax(-1, keepdim=True) / limit).clamp_min(
        1 / (limit * 512)
    )
    quantized = (weight.float() / scale).clamp(-limit, limit).to(fp8_dtype)
    before = (quantized.double() * scale.double() - weight.double()).square().sum(-1)
    pointers = quantized.data_ptr(), scale.data_ptr()
    fitted, fitted_scale = fit_fp8_head_scales(weight, quantized, scale)
    after = (fitted.double() * fitted_scale.double() - weight.double()).square().sum(-1)
    assert torch.all(after <= before)
    assert after.sum() < before.sum()
    assert (fitted.data_ptr(), fitted_scale.data_ptr()) == pointers
    assert torch.equal(weight, original)


@requires_cuda
@pytest.mark.parametrize("quantization", ["fp8", NVFP4])
def test_refresh_follows_target_update(
    default_vllm_config, dist_init, monkeypatch, quantization
):
    """After the target's weights change, the copy is re-derived in place: it
    equals a fresh quantization of the new target, keeps its storage, and
    frees the temporary copy without waiting for the cycle collector."""
    head = _target_head()
    drafter = torch.nn.Module()
    drafter.lm_head = quantized_lm_head_copy(head, quantization)
    ptrs = {k: t.data_ptr() for k, t in _tensors(drafter.lm_head).items()}

    new_weight = _target_head(seed=1).weight.detach()
    head.weight.data.copy_(new_weight)
    gc.disable()
    try:
        allocated = torch.accelerator.memory_allocated()
        refresh_quantized_lm_heads(drafter)
        assert torch.accelerator.memory_allocated() == allocated
    finally:
        gc.enable()

    expected = _tensors(quantized_lm_head_copy(head, quantization))
    for name, tensor in _tensors(drafter.lm_head).items():
        assert tensor.data_ptr() == ptrs[name]
        if name != "workspace":
            assert torch.equal(tensor, expected[name]), name
    assert torch.equal(head.weight, new_weight)


def _draft_update_worker(vllm_config, drafter: torch.nn.Module, **engine) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = vllm_config
    worker.weight_transfer_engine = SimpleNamespace(
        supports_draft_weight_update=True,
        reset_weight_update_target=lambda: None,
        **engine,
    )
    worker.model_runner = SimpleNamespace(get_draft_model=lambda: drafter)
    worker._weight_update_active = False
    worker._incomplete_weight_updates = set()
    worker._kv_cache_reclaimable = False
    worker._weight_update_kv_cache_loan = ExitStack()
    worker._set_draft_weight_update_target = lambda: None  # type: ignore[method-assign]
    return worker


@requires_cuda
def test_draft_weight_update_refreshes_copy(
    default_vllm_config, dist_init, monkeypatch
):
    """A draft weight update can write a target weight the drafter shares, such
    as an embedding tied to the lm_head; the copy must follow it."""
    head = _target_head()
    drafter = torch.nn.Module()
    drafter.lm_head = quantized_lm_head_copy(head, "fp8")
    worker = _draft_update_worker(
        default_vllm_config,
        drafter,
        start_weight_update=lambda: None,
        finish_weight_update=lambda: None,
    )

    worker.start_draft_weight_update()
    head.weight.data.copy_(_target_head(seed=1).weight)
    worker.finish_weight_update()

    expected = _tensors(quantized_lm_head_copy(head, "fp8"))
    for name, tensor in _tensors(drafter.lm_head).items():
        assert torch.equal(tensor, expected[name]), name


@requires_cuda
@pytest.mark.parametrize("quantization", ["fp8", NVFP4])
def test_draft_weight_update_loads_lm_head(
    default_vllm_config, dist_init, monkeypatch, quantization
):
    """A draft update carrying lm_head.weight loads the shared target head, as
    without quantization, and the copy is re-derived from it in place."""
    head = _target_head()
    record_metadata_for_reloading(head)
    drafter = torch.nn.Module()
    draft_head = quantized_lm_head_copy(head, quantization)
    drafter.lm_head = draft_head
    ptrs = {k: t.data_ptr() for k, t in _tensors(draft_head).items()}
    new_weight = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def load(update_info):
        param = drafter.get_parameter("lm_head.weight")
        getattr(param, "weight_loader", default_weight_loader)(param, new_weight)

    worker = _draft_update_worker(
        default_vllm_config,
        drafter,
        start_weight_update=lambda: initialize_layerwise_reload(drafter),
        update_weights=load,
        finish_weight_update=lambda: finalize_layerwise_reload(drafter, None),
    )
    worker.start_draft_weight_update()
    worker.update_weights({})
    worker.finish_weight_update()

    assert drafter.lm_head is draft_head
    assert torch.equal(head.weight[:VOCAB], new_weight)
    expected = _tensors(quantized_lm_head_copy(head, quantization))
    for name, tensor in _tensors(draft_head).items():
        assert tensor.data_ptr() == ptrs[name]
        if name != "workspace":
            assert torch.equal(tensor, expected[name]), name
    hidden = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
    assert LogitsProcessor(VOCAB)(draft_head, hidden).isfinite().all()


@requires_cuda
@requires_nvfp4
def test_layerwise_reload_of_drafter_keeps_copy(
    default_vllm_config, dist_init, monkeypatch
):
    """A drafter weight update reloads the drafter layerwise; the derived copy
    must come back unchanged, not be re-quantized."""
    head = _target_head()
    drafter = torch.nn.Module()
    record_metadata_for_reloading(drafter)
    drafter.lm_head = quantized_lm_head_copy(head, "nvfp4")
    before = {k: t.clone() for k, t in _tensors(drafter.lm_head).items()}

    def fail(*args, **kwargs):
        raise AssertionError("the copy was re-quantized")

    monkeypatch.setattr(
        drafter.lm_head.quant_method, "process_weights_after_loading", fail
    )
    initialize_layerwise_reload(drafter)
    finalize_layerwise_reload(drafter, None)

    after = _tensors(drafter.lm_head)
    assert after.keys() == before.keys()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor), name


@requires_cuda
def test_token_ranges_head_scores_only_its_ranges(
    default_vllm_config, dist_init, monkeypatch
):
    """Inside the ranges the logits equal the target's; outside they are -inf,
    and the copy shares (not copies) the target weight, so in-place target
    updates are seen without re-deriving anything."""
    from vllm.model_executor.layers.quantization.online.lm_head import (
        subset_lm_head_copy,
    )

    head = _target_head()
    ranges = [(0, 1000), (3900, VOCAB)]
    draft_head = subset_lm_head_copy(head, ranges)
    assert draft_head.weight.data_ptr() == head.weight.data_ptr()

    logits_processor = LogitsProcessor(VOCAB)
    hidden = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
    inside = torch.zeros(VOCAB, dtype=torch.bool, device="cuda")
    for a, b in ranges:
        inside[a:b] = True
    for _ in range(2):
        ref = logits_processor(head, hidden)
        out = logits_processor(draft_head, hidden)
        assert out.shape == ref.shape == (64, VOCAB)
        torch.testing.assert_close(out[:, inside], ref[:, inside], rtol=0, atol=0)
        assert torch.isneginf(out[:, ~inside]).all()
        head.weight.data.copy_(_target_head(seed=1).weight)


@pytest.mark.parametrize(
    "ranges", [[], [[5, 5]], [[-1, 4]], [[0, 4, 8]]], ids=["empty", "eq", "neg", "3"]
)
def test_token_ranges_rejects_invalid(ranges):
    from vllm.config.speculative import normalize_token_ranges

    with pytest.raises(ValueError):
        normalize_token_ranges(ranges)
    assert normalize_token_ranges([[0, 8], [10, 12]]) == [(0, 8), (10, 12)]


@requires_cuda
@pytest.mark.parametrize("quantization", ["fp8", NVFP4])
def test_token_ranges_quantize_only_their_rows(
    default_vllm_config, dist_init, monkeypatch, quantization
):
    """Token ranges and draft-head quantization compose: only the selected rows
    are quantized, their logits match a quantized full head, the rest stay -inf,
    and a target update re-derives the compact copy in place."""
    from vllm.model_executor.layers.quantization.online.lm_head import (
        subset_lm_head_copy,
    )

    head = _target_head()
    ranges = [(0, 1000), (3900, VOCAB)]
    rows = sum(b - a for a, b in ranges)
    draft_head = subset_lm_head_copy(head, ranges, quantization)
    compact = draft_head.compact
    assert compact.weight.dtype != torch.bfloat16
    assert compact.output_size_per_partition == rows
    assert draft_head.weight.data_ptr() == head.weight.data_ptr()

    logits_processor = LogitsProcessor(VOCAB)
    hidden = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
    inside = torch.zeros(VOCAB, dtype=torch.bool, device="cuda")
    for a, b in ranges:
        inside[a:b] = True
    out = logits_processor(draft_head, hidden).float()
    ref = logits_processor(head, hidden).float()
    assert torch.isneginf(out[:, ~inside]).all()
    rel_err = (out[:, inside] - ref[:, inside]).norm() / ref[:, inside].norm()
    assert rel_err < (0.05 if quantization == "fp8" else 0.15)
    if quantization == "fp8":
        # Per-row scales: the rows quantize exactly as in a full quantized head.
        full = logits_processor(quantized_lm_head_copy(head, "fp8"), hidden).float()
        torch.testing.assert_close(out[:, inside], full[:, inside])

    drafter = torch.nn.Module()
    drafter.lm_head = draft_head
    ptrs = {k: t.data_ptr() for k, t in _tensors(compact).items()}
    head.weight.data.copy_(_target_head(seed=1).weight)
    refresh_quantized_lm_heads(drafter)
    fresh = _tensors(subset_lm_head_copy(head, ranges, quantization).compact)
    for name, tensor in _tensors(compact).items():
        assert tensor.data_ptr() == ptrs[name]
        if name != "workspace":
            assert torch.equal(tensor, fresh[name]), name


def test_token_ranges_accept_draft_head_quantization():
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.eagle.config import EagleSpeculatorConfig

    spec = SimpleNamespace(
        draft_lm_head_quantization="fp8",
        draft_lm_head_token_ranges=[[0, 8], [10, 12]],
        draft_model_config=None,
    )
    EagleSpeculatorConfig.apply_options(spec)
    assert spec.draft_lm_head_token_ranges == [(0, 8), (10, 12)]
