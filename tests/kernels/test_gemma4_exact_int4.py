# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in exact INT4 storage of Gemma4 QAT tables (VLLM_GEMMA4_EXACT_INT4).

The fit, the K-permuted packing and the installation bookkeeping run on CPU;
the Triton GEMM / row-gather kernels run when CUDA is available.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import gemma4_exact_int4 as xi4
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)


def qat_table(n: int, k: int, seed: int = 0, device="cpu", qmax: int = 7):
    """BF16 table on an int4 x fp32-scale grid, like a Gemma4 QAT export."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-8, qmax + 1, (n, k), generator=g, dtype=torch.int8)
    d = torch.rand(n, k // 32, generator=g) * 2e-3 + 1e-5
    w = (q.float().reshape(n, k // 32, 32) * d[:, :, None]).reshape(n, k)
    return w.to(torch.bfloat16).to(device)


def bits_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("shape", [(64, 256), (37, 512), (8, 5376)])
def test_fit_reproduces_grid_bit_exactly(shape):
    w = qat_table(*shape)
    packed, scales = xi4.fit_exact_int4(w, chunk_rows=16)
    assert packed.dtype == torch.int32 and packed.shape == (shape[0], shape[1] // 8)
    assert scales.dtype == torch.float32 and scales.shape == (shape[0], shape[1] // 32)
    assert bits_eq(xi4.decode_reference(packed, scales), w)


def test_fit_handles_every_top_level_and_zero_groups():
    # Groups whose largest magnitude sits on levels 1..8, plus all-zero groups.
    rows = []
    for top in range(1, 9):
        q = torch.randint(-top, top, (1, 256), dtype=torch.int8)
        q[0, 0] = -top if top == 8 else top
        rows.append(q)
    q = torch.cat(rows)
    d = torch.rand(q.shape[0], 8) * 1e-2 + 1e-4
    w = (q.float().reshape(-1, 8, 32) * d[:, :, None]).reshape(q.shape).bfloat16()
    w[3, :32] = 0
    packed, scales = xi4.fit_exact_int4(w)
    assert bits_eq(xi4.decode_reference(packed, scales), w)


def test_pack_layout_plane_is_one_scale_group():
    n, k = 2, 512
    q = torch.zeros(n, k, dtype=torch.int8)
    # Weight at k = 256*kb + 32*i + j must land in word 32*kb + j, nibble i.
    q[0, 256 * 1 + 32 * 5 + 7] = 3
    words = xi4._pack_rows(q)
    nibble = (words[0, 32 * 1 + 7] >> (4 * 5)) & 0xF
    assert int(nibble) == 3 + 8
    # Every other nibble holds the code of 0 (= 8).
    other = words.clone()
    other[0, 32 * 1 + 7] &= ~(0xF << (4 * 5))
    other[0, 32 * 1 + 7] |= 8 << (4 * 5)
    assert torch.all(other == 0x88888888 - (1 << 32))


@pytest.mark.parametrize(
    "mutate",
    ["random", "perturb", "neg_zero", "inf", "nan", "fp16", "k_not_256"],
)
def test_fit_rejects_tensors_off_the_grid(mutate):
    w = qat_table(16, 512)
    if mutate == "random":
        w = torch.randn(16, 512).bfloat16()
    elif mutate == "perturb":
        w[3, 100] = w[3, 100] + torch.tensor(1e-3).bfloat16()
    elif mutate == "neg_zero":
        w[1, 33] = -0.0
        w[1, 32:64] = 0
        w[1, 33] = -0.0
    elif mutate == "inf":
        w[0, 0] = float("inf")
    elif mutate == "nan":
        w[0, 0] = float("nan")
    elif mutate == "fp16":
        w = w.half()
    elif mutate == "k_not_256":
        w = w[:, :480]
    assert xi4.fit_exact_int4(w) is None


def test_negative_zero_is_rejected_not_silently_flipped():
    w = torch.zeros(4, 256, dtype=torch.bfloat16)
    assert xi4.fit_exact_int4(w) is not None
    w[2, 17] = -0.0
    assert xi4.fit_exact_int4(w) is None


def _config(lora_targets=None, v2=True):
    lora = (
        None if lora_targets is None else SimpleNamespace(target_modules=lora_targets)
    )
    return SimpleNamespace(lora_config=lora, use_v2_model_runner=v2)


def test_check_supported():
    xi4.check_supported(_config())
    xi4.check_supported(_config(lora_targets=["q_proj", "down_proj"]))
    xi4.check_supported(_config(lora_targets=[]))
    with pytest.raises(ValueError, match="embed_tokens/lm_head"):
        xi4.check_supported(_config(lora_targets=["embed_tokens"]))
    with pytest.raises(ValueError, match="V2 model runner"):
        xi4.check_supported(_config(v2=False))
    with pytest.raises(RuntimeError, match="vLLM config"):
        xi4.check_supported(None)


def _tied_model(weight):
    embed = torch.nn.Module()
    embed.weight = torch.nn.Parameter(weight, requires_grad=False)
    embed.quant_method = UnquantizedEmbeddingMethod()
    head = torch.nn.Module()
    head.weight = embed.weight
    head.quant_method = UnquantizedEmbeddingMethod()
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = embed
    model.lm_head = head
    return model


@pytest.fixture
def cpu_encode(monkeypatch):
    """Run the installation on CPU: fit + reference decode check, no kernels."""

    def encode(weight, chunk_rows=4096):
        out = xi4.fit_exact_int4(weight, chunk_rows)
        if out is not None:
            assert bits_eq(xi4.decode_reference(*out), weight)
        return out

    monkeypatch.setattr(xi4, "encode_exact_int4", encode)
    monkeypatch.setattr(xi4, "_verify_gemm", lambda *a: None)
    monkeypatch.setattr(xi4.torch.accelerator, "empty_cache", lambda: None)
    for key in ("installed", "skipped"):
        xi4._STATE[key].clear()
    xi4._STATE["bytes_saved"] = 0


def test_install_target_replaces_tied_table(cpu_encode):
    w = qat_table(64, 512)
    model = _tied_model(w.clone())
    xi4.install_gemma4_exact_int4_target(model, _config())
    embed, head = model.model.embed_tokens, model.lm_head
    for m in (embed, head):
        assert isinstance(m.quant_method, xi4.ExactInt4EmbeddingMethod)
        assert not hasattr(m, "weight")
        assert m._gemma4_exact_int4
    assert embed.exact_int4_packed is head.exact_int4_packed
    assert bits_eq(
        xi4.decode_reference(embed.exact_int4_packed, embed.exact_int4_scales), w
    )
    state = xi4.gemma4_exact_int4_state()
    assert state["installed"] == {"target_embed_head:64x512": 1}
    assert state["bytes_saved"] == 64 * 512 * 2 - (64 * 64 * 4 + 64 * 16 * 4)
    # Idempotent: a second call leaves the converted table alone.
    xi4.install_gemma4_exact_int4_target(model, _config())
    assert xi4.gemma4_exact_int4_state()["installed"] == state["installed"]
    with pytest.raises(RuntimeError, match="reloaded"):
        embed.quant_method.process_weights_after_loading(embed)


def test_install_keeps_bf16_tables_off_the_grid(cpu_encode):
    w = torch.randn(64, 512).bfloat16()
    model = _tied_model(w.clone())
    xi4.install_gemma4_exact_int4_target(model, _config())
    assert type(model.model.embed_tokens.quant_method) is UnquantizedEmbeddingMethod
    assert bits_eq(model.model.embed_tokens.weight, w)
    assert xi4.gemma4_exact_int4_state()["skipped"] == {
        "target_embed_head:not_exact_int4": 1
    }


def test_install_skips_untied_and_quantized(cpu_encode):
    model = _tied_model(qat_table(64, 512))
    model.lm_head.weight = torch.nn.Parameter(qat_table(64, 512, seed=1))
    xi4.install_gemma4_exact_int4_target(model, _config())
    assert xi4.gemma4_exact_int4_state()["skipped"] == {"target_embed_head:untied": 1}

    model = _tied_model(qat_table(64, 512))
    model.lm_head.quant_method = object()
    xi4.install_gemma4_exact_int4_target(model, _config())
    assert (
        xi4.gemma4_exact_int4_state()["skipped"][
            "target_embed_head:quantized_or_biased"
        ]
        == 1
    )


def test_install_draft_head_only(cpu_encode):
    head_w = qat_table(64, 256, seed=3)
    draft = torch.nn.Module()
    draft.lm_head = torch.nn.Module()
    draft.lm_head.weight = torch.nn.Parameter(head_w.clone(), requires_grad=False)
    draft.lm_head.quant_method = UnquantizedEmbeddingMethod()
    draft.masked_embedding = None
    xi4.install_gemma4_exact_int4_draft(draft, _config())
    assert isinstance(draft.lm_head.quant_method, xi4.ExactInt4EmbeddingMethod)
    assert xi4.gemma4_exact_int4_state()["installed"] == {"draft_head:64x256": 1}

    masked = torch.nn.Module()
    masked.lm_head = draft.lm_head
    masked.lm_head._gemma4_exact_int4 = False
    masked.masked_embedding = object()
    xi4.install_gemma4_exact_int4_draft(masked, _config())
    assert xi4.gemma4_exact_int4_state()["skipped"] == {"draft_head:centroid_masked": 1}


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("n", [4096, 4133])
def test_kernels_against_fp64_reference(n):
    k = 1024
    w = qat_table(n, k, seed=n, device="cuda")
    packed, scales = xi4.encode_exact_int4(w)  # includes the gather bit check
    ids = torch.tensor([[0, 5], [n - 1, 17]], device="cuda")
    assert bits_eq(
        torch.ops.vllm.gemma4_exact_int4_rows(ids, packed, scales, k), w[ids]
    )
    xi4._verify_gemm(w, packed, scales)
    for m in (1, 7, 16, 17, 33, 64, 100):
        x = torch.randn(2, m, k, dtype=torch.bfloat16, device="cuda")
        y = torch.ops.vllm.gemma4_exact_int4_linear(x, packed, scales, n)
        assert y.shape == (2, m, n) and y.dtype == torch.bfloat16
        ref = x.double() @ w.double().t()
        err = (y.double() - ref).norm() / ref.norm()
        assert err < 2.5e-3, (m, err)
    empty = torch.ops.vllm.gemma4_exact_int4_linear(
        torch.empty(0, k, dtype=torch.bfloat16, device="cuda"), packed, scales, n
    )
    assert empty.shape == (0, n)


@cuda
def test_encode_returns_none_for_cpu_tensor():
    assert xi4.encode_exact_int4(qat_table(8, 256)) is None
