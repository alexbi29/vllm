# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of the opt-in Gemma4 W12 installation (VLLM_GEMMA4_W12)."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.test_w12_linear import FakeOps, _act_and_mul_ref, bits_eq, gaussian
from vllm.model_executor.layers import gemma4_w12
from vllm.model_executor.layers import w12_linear as w12
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)


class _Linear(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()


def _tiny_gemma(hidden=128, inter=128, vocab=256):
    from vllm.model_executor.layers.activation import GeluAndMul

    def lin(n, k, seed):
        return _Linear(gaussian(n, k, seed))

    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = lin(3 * hidden, hidden, 1)
    layer.self_attn.o_proj = lin(hidden, hidden, 2)
    layer.mlp = torch.nn.Module()
    layer.mlp.gate_up_proj = lin(2 * inter, hidden, 3)
    layer.mlp.down_proj = lin(hidden, inter, 4)
    layer.mlp.act_fn = GeluAndMul.__new__(GeluAndMul)
    torch.nn.Module.__init__(layer.mlp.act_fn)
    layer.mlp.act_fn.approximate = "tanh"
    layer.mlp.w12_fused_act = False
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    embed = torch.nn.Module()
    embed.weight = torch.nn.Parameter(gaussian(vocab, hidden, 5), requires_grad=False)
    embed.num_embeddings_per_partition = vocab
    embed.quant_method = UnquantizedEmbeddingMethod()
    head = torch.nn.Module()
    head.weight = embed.weight
    head.quant_method = UnquantizedEmbeddingMethod()
    model.model.embed_tokens = embed
    model.lm_head = head
    return model


@pytest.fixture
def fake_ops(monkeypatch):
    ops = FakeOps()
    monkeypatch.setattr(w12, "_ops", lambda: ops)
    monkeypatch.setattr(w12, "act_and_mul", _act_and_mul_ref)
    w12._WORKSPACES.clear()
    yield ops
    w12._WORKSPACES.clear()


def _gelu_tanh_and_mul_ref(gate_up):
    half = gate_up.shape[1] // 2
    act = torch.nn.functional.gelu(gate_up[:, :half].float(), approximate="tanh")
    return act.bfloat16().float().mul(gate_up[:, half:].float()).bfloat16()


@pytest.fixture
def cpu_install(monkeypatch, fake_ops):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        lora_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    monkeypatch.setattr(gemma4_w12, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(w12, "kernels_available", lambda: True)
    monkeypatch.setattr(gemma4_w12, "supported_device", lambda device: True)
    monkeypatch.setattr(gemma4_w12, "_num_sms", lambda device: 188)
    monkeypatch.setattr(gemma4_w12, "_act_and_mul_ref", _gelu_tanh_and_mul_ref)
    return config


def test_install_encodes_frees_bf16_and_fuses_the_activation(cpu_install, fake_ops):
    model = _tiny_gemma()
    originals = {
        name: mod.weight.clone()
        for name, mod in model.named_modules()
        if isinstance(getattr(mod, "weight", None), torch.Tensor)
    }
    gemma4_w12.install_gemma4_w12(model)
    attn = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp
    for mod in (attn.qkv_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj):
        assert not hasattr(mod, "weight")
        assert isinstance(mod.quant_method, gemma4_w12.W12LinearMethod)
        assert mod.w12_planes.dtype == torch.uint8
    assert mlp.w12_fused_act and mlp.gate_up_proj.quant_method.act == w12.ACT_GELU_TANH
    embed, head = model.model.embed_tokens, model.lm_head
    assert not hasattr(embed, "weight") and not hasattr(head, "weight")
    assert embed.quant_method is head.quant_method
    assert embed.w12_planes.data_ptr() == head.w12_planes.data_ptr()
    # the tied table is still exact through the embedding path
    ids = torch.tensor([0, 3, 255])
    rows = w12.w12_embedding_impl(
        ids, embed.w12_planes, embed.w12_scale, embed.w12_esc_rp,
        embed.w12_esc_col, embed.w12_esc_bf16, 256, 128,
    )  # fmt: skip
    assert bits_eq(rows, originals["model.embed_tokens"][ids])
    state = gemma4_w12.gemma4_w12_state()
    assert state["bf16_bytes_freed"] > state["w12_bytes_resident"]
    # reloading in place is refused rather than silently reading stale data
    with pytest.raises(RuntimeError, match="reloaded"):
        attn.o_proj.quant_method.process_weights_after_loading(attn.o_proj)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_v2_model_runner", False),
        ("lora_config", object()),
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
    ],
)
def test_unsupported_configurations_fail_closed(cpu_install, field, value):
    if field in ("lora_config", "use_v2_model_runner"):
        setattr(cpu_install, field, value)
    else:
        setattr(cpu_install.parallel_config, field, value)
    with pytest.raises(ValueError):
        gemma4_w12.install_gemma4_w12(_tiny_gemma())


def test_missing_kernels_fail_closed(cpu_install, monkeypatch):
    monkeypatch.setattr(w12, "kernels_available", lambda: False)
    with pytest.raises(RuntimeError, match="_C::w12_gemm"):
        gemma4_w12.install_gemma4_w12(_tiny_gemma())


def test_biased_or_quantized_linears_are_left_alone(cpu_install):
    model = _tiny_gemma()
    o_proj = model.model.layers[0].self_attn.o_proj
    o_proj.bias = torch.nn.Parameter(torch.zeros(128), requires_grad=False)
    gemma4_w12.install_gemma4_w12(model)
    assert (
        hasattr(o_proj, "weight")
        and type(o_proj.quant_method) is UnquantizedLinearMethod
    )
