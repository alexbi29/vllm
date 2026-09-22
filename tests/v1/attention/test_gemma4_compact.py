# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for compact-cache allocation and cross-model readers."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.gemma4_compact import Gemma4CompactBackend
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode


def _weight_config(name):
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8Config,
        ModelOptNvFp4Config,
    )

    if name == "fp8":
        return Fp8Config(is_checkpoint_fp8_serialized=True)
    if name == "modelopt":
        return ModelOptFp8Config("FP8", True, None, [])
    if name == "modelopt_fp4":
        return ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    return compressed_tensors.CompressedTensorsConfig({}, [], "float-quantized")


@pytest.mark.parametrize("name", ["fp8", "modelopt", "modelopt_fp4"])
def test_compact_accepts_quantized_weights_with_bf16_cache(name):
    from vllm.model_executor.models.gemma4 import _validate_compact_weight_quantization
    from vllm.v1.attention.backends.gemma4_compact import validate_compact_kv_dtype

    quant = _weight_config(name)
    _validate_compact_weight_quantization(quant, SimpleNamespace())
    validate_compact_kv_dtype("auto", torch.bfloat16, quant)
    validate_compact_kv_dtype("bfloat16", torch.bfloat16, quant)


@pytest.mark.parametrize(
    "scheme_name",
    [
        "CompressedTensorsW8A8Fp8",
        "CompressedTensorsW8A16Fp8",
        "CompressedTensorsW4A4Fp4",
    ],
)
def test_compact_accepts_compressed_tensors_float_schemes(scheme_name):
    from vllm.model_executor.layers.quantization.compressed_tensors import schemes
    from vllm.model_executor.models.gemma4 import _validate_compact_weight_quantization

    # Kernel construction needs CUDA; selection depends on the resolved scheme
    # type, not its kernel instance or weight tensors.
    scheme = object.__new__(getattr(schemes, scheme_name))
    projection = SimpleNamespace(quant_method=None, scheme=scheme)
    _validate_compact_weight_quantization(
        _weight_config("compressed-tensors"), projection
    )


def test_compact_rejects_other_compressed_tensors_schemes():
    from vllm.model_executor.models.gemma4 import _validate_compact_weight_quantization

    with pytest.raises(ValueError, match="supports BF16, FP8, and NVFP4"):
        _validate_compact_weight_quantization(
            _weight_config("compressed-tensors"),
            SimpleNamespace(quant_method=None, scheme=object()),
        )


def test_compact_allows_unquantized_attention_in_compressed_tensors_model():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.gemma4 import _validate_compact_weight_quantization

    _validate_compact_weight_quantization(
        _weight_config("compressed-tensors"),
        SimpleNamespace(quant_method=UnquantizedLinearMethod()),
    )


def test_compact_rejects_non_bf16_model_dtype():
    from vllm.v1.attention.backends.gemma4_compact import validate_compact_kv_dtype

    with pytest.raises(ValueError, match="--dtype bfloat16"):
        validate_compact_kv_dtype("bfloat16", torch.float16, _weight_config("fp8"))


@pytest.mark.parametrize(
    "cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2", "fp8_per_token_head"]
)
def test_compact_backend_rejects_fp8_before_backend_initialization(cache_dtype):
    from vllm.v1.attention.backends.gemma4_compact import Gemma4CompactImpl

    with pytest.raises(ValueError, match="does not support FP8"):
        Gemma4CompactImpl(
            8,
            512,
            1.0,
            1,
            None,
            None,
            cache_dtype,
            compact_k_norm=torch.ones(512, dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="does not support FP8"):
        Gemma4CompactBackend.get_kv_cache_shape(1, 256, 1, 512, cache_dtype)


@pytest.mark.parametrize(
    "cache_dtype",
    [
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_per_token_head",
        "nvfp4",
    ],
)
def test_compact_rejects_quantized_cache_before_projection_creation(
    cache_dtype, monkeypatch
):
    from vllm.config import CacheConfig
    from vllm.model_executor.models import gemma4

    monkeypatch.setattr(
        gemma4,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(gemma4_compact_kv=True)
            )
        ),
    )
    with pytest.raises(ValueError, match="--kv-cache-dtype bfloat16"):
        # No distributed environment or model weights: rejection must precede
        # both TP setup and projection allocation, even with skip-layer flags.
        gemma4.Gemma4Attention(
            config=SimpleNamespace(),
            hidden_size=512,
            num_heads=1,
            num_kv_heads=1,
            head_dim=512,
            max_position_embeddings=1024,
            cache_config=CacheConfig(
                cache_dtype=cache_dtype, kv_cache_dtype_skip_layers=["5"]
            ),
            quant_config=_weight_config("modelopt_fp4"),
        )


@pytest.mark.parametrize(
    "name,attribute,value",
    [
        ("compressed-tensors", "kv_cache_scheme", {"type": "float", "num_bits": 8}),
        ("modelopt", "kv_cache_quant_method", "FP8"),
        ("modelopt_fp4", "kv_cache_quant_algo", "FP8"),
    ],
)
def test_compact_auto_rejects_checkpoint_fp8_cache_but_honors_bf16(
    name, attribute, value
):
    from vllm.v1.attention.backends.gemma4_compact import validate_compact_kv_dtype

    quant = _weight_config(name)
    setattr(quant, attribute, value)
    with pytest.raises(
        ValueError, match="checkpoint KV=.*FP8|checkpoint KV=.*num_bits"
    ):
        validate_compact_kv_dtype("auto", torch.bfloat16, quant)
    validate_compact_kv_dtype("bfloat16", torch.bfloat16, quant)


@pytest.mark.parametrize("compact", [False, True])
def test_compact_forward_uses_one_shared_projection_result(compact):
    from vllm.model_executor.models.gemma4 import Gemma4Attention

    qkv = torch.cat(
        [torch.full((2, 512), x, dtype=torch.bfloat16) for x in (1, 2, 9)], -1
    )
    seen: dict[str, torch.Tensor] = {}

    def attention(q, k, v):
        seen.update(k=k, v=v)
        return q

    layer = SimpleNamespace(
        qkv_proj=lambda _: (qkv, None),
        q_size=512,
        kv_size=512,
        use_compact_kv=compact,
        num_heads=1,
        num_kv_heads=1,
        head_dim=512,
        q_norm=lambda q: q,
        k_norm=lambda k: k * 2,
        v_norm=lambda v: v,
        is_kv_shared_layer=False,
        rotary_emb=lambda positions, q, k: (q, k),
        attn=attention,
        o_proj=lambda x: (x, None),
    )
    Gemma4Attention.forward(layer, torch.arange(2), torch.empty(2, 512))
    torch.testing.assert_close(seen["k"], torch.full((2, 512), 4, dtype=torch.bfloat16))
    torch.testing.assert_close(
        seen["v"], torch.full((2, 512), 2 if compact else 9, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize("weight_dtype", [torch.float8_e4m3fn, torch.uint8])
def test_gemma4_loader_duplicates_shared_quantized_weights_and_scales(
    weight_dtype, monkeypatch
):
    from vllm.model_executor.models import gemma4

    prefix = "model.language_model.layers.1.self_attn.k_proj."
    weights = [(prefix + "weight", torch.ones(2, 4).to(weight_dtype))]
    weights += [
        (prefix + field, torch.tensor([0.25]))
        for field in (
            "weight_scale",
            "weight_scale_2",
            "input_scale",
        )
    ]
    local_name = "model.language_model.layers.0.self_attn.k_proj.weight"
    weights.append((local_name, torch.ones(2, 4).to(weight_dtype)))
    monkeypatch.setattr(
        gemma4,
        "AutoWeightsLoader",
        lambda *a, **k: SimpleNamespace(load_weights=lambda iterator: dict(iterator)),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            attention_k_eq_v=True,
            layer_types=["sliding_attention", "full_attention"],
            tie_word_embeddings=False,
        )
    )
    loaded = gemma4.Gemma4ForCausalLM.load_weights(model, weights)
    for name, weight in weights[:-1]:
        k_name = name.replace("model.language_model.", "model.")
        v_name = k_name.replace("k_proj", "v_proj")
        torch.testing.assert_close(
            loaded[k_name].float(), weight.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            loaded[v_name].float(), weight.float(), rtol=0, atol=0
        )
        assert loaded[v_name].dtype == weight.dtype
    assert "model.layers.0.self_attn.v_proj.weight" not in loaded


def test_compact_metadata_uses_global_head_width():
    """Segmented decode scratch must fit global heads, not local head_dim=256."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.attention.backends.gemma4_compact import Gemma4CompactMetadataBuilder

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 16,
            get_num_kv_heads=lambda _: 8,
            get_head_size=lambda: 256,
            rswa_window=None,
        ),
        parallel_config=None,
    )
    spec = FullAttentionSpec(
        block_size=256, num_kv_heads=2, head_size=512, dtype=torch.bfloat16
    )
    builder = Gemma4CompactMetadataBuilder(spec, [], config, torch.device("cpu"))
    assert builder.softmax_segm_output.shape[-1] == 512


@pytest.mark.parametrize(
    "dtype,mode,head_size",
    [
        (torch.float16, KVQuantMode.NONE, 512),
        (torch.bfloat16, KVQuantMode.FP8_PER_TENSOR, 512),
        (torch.bfloat16, KVQuantMode.NONE, 256),
    ],
)
def test_compact_cache_rejects_incompatible_storage(dtype, mode, head_size):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        kv_quant_mode=mode,
    )
    with pytest.raises(ValueError, match="BF16 global 512-d"):
        Gemma4CompactBackend.customize_spec(spec)


def test_gemma4_mtp_reads_target_compact_norm(monkeypatch):
    """A draft's independent dense backend must not misinterpret compact KV."""
    from vllm.v1.spec_decode import gemma4

    norm = torch.randn(512, dtype=torch.bfloat16)
    target_impl = SimpleNamespace(
        compact_k_norm=norm, num_heads=16, num_queries_per_kv=8, scale=1.0
    )
    target = SimpleNamespace(
        impl=target_impl,
        head_size=512,
        num_kv_heads=2,
        kv_cache_dtype="auto",
        attn_backend=Gemma4CompactBackend,
        backend="compact-enum",
    )
    draft = SimpleNamespace(
        impl=SimpleNamespace(scale=0.75),
        head_size=512,
        num_heads=8,
        num_kv_heads=2,
        kv_cache_dtype="auto",
        attn_backend="dense",
    )
    text = SimpleNamespace(layer_types=["full_attention"], num_kv_shared_layers=0)
    hf = SimpleNamespace(get_text_config=lambda: text, gemma4_compact_kv=True)
    proposer = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        ),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf)),
        model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=draft))]
            )
        ),
    )
    target_name = "model.layers.0.self_attn.attn"
    monkeypatch.setattr(
        gemma4, "get_layers_from_vllm_config", lambda *args: {target_name: target}
    )
    gemma4.Gemma4Proposer._setup_gemma4_kv_sharing(proposer, {target_name})
    assert draft.attn_backend is Gemma4CompactBackend
    assert draft.impl.compact_k_norm is norm
    assert draft.impl.num_heads == 8
    assert draft.impl.num_queries_per_kv == 4
    assert draft.impl.scale == 0.75
    assert draft.kv_sharing_target_layer_name == target_name
    assert target_impl.num_heads == 16
    assert target_impl.scale == 1.0
