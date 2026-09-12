# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every DSpark model class the V2 speculator can load must expose the
sampling hooks it calls. The greedy branch of DSparkSpeculator._sample_logits
calls model.map_draft_to_target on every profile_run (draft_logits is None
during profiling), so a missing hook is a cold-boot blocker, not an edge case
-- upstream #49969 added the hooks to amd/, xpu/, kimi_k3 and qwen3_dspark
but not the nvidia deepseek_v4 class (reported by alexbi29 in
vllm-project/vllm#41834)."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import dspark as nvidia_dspark
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


@pytest.mark.parametrize("adaptive,num_tokens,expected", [(False, 5, 5), (True, 3, 3)])
def test_runtime_block_matches_adaptive_draft_batch(adaptive, num_tokens, expected):
    spec = SimpleNamespace(
        enable_adaptive_verification=adaptive,
        num_speculative_tokens=num_tokens,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dspark_block_size=5)
        ),
    )
    block = nvidia_dspark._dspark_runtime_block_size(
        SimpleNamespace(speculative_config=spec)
    )
    assert block == expected
    # Two requests must remain separate groups of num_tokens, not groups of 5.
    assert torch.arange(2 * expected).view(2, block).shape == (2, expected)


def test_nvidia_dspark_exposes_v2_speculator_hooks():
    cls = nvidia_dspark.DSparkDeepseekV4ForCausalLM
    for hook in ("map_draft_to_target", "compute_draft_logits"):
        assert hasattr(cls, hook), (
            f"{cls.__name__} lacks {hook}; the V2 DSpark speculator dies in "
            "profile_run on the first cold boot"
        )


def test_dspark_layer_forwards_hash_boundary_to_moe(monkeypatch):
    captured = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.n_local_heads = 1

    class FakeMoE(torch.nn.Module):
        def __init__(
            self,
            vllm_config,
            prefix="",
            use_sequence_parallel=False,
            *,
            num_hash_layers,
        ):
            super().__init__()
            captured.update(
                prefix=prefix,
                use_sequence_parallel=use_sequence_parallel,
                num_hash_layers=num_hash_layers,
            )

    class FakeNorm(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(
        nvidia_dspark,
        "_select_dsv4_attn_cls",
        lambda _vllm_config: FakeAttention,
    )
    monkeypatch.setattr(nvidia_dspark, "DeepseekV4MoE", FakeMoE)
    monkeypatch.setattr(nvidia_dspark, "RMSNorm", FakeNorm)
    monkeypatch.setattr(
        nvidia_dspark,
        "triton_sparse_mla_head_block_size",
        lambda: 1,
    )

    config = SimpleNamespace(
        hidden_size=2,
        rms_norm_eps=1e-6,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        hc_eps=1e-6,
        num_hidden_layers=4,
        num_hash_layers=2,
        sliding_window=2,
        head_dim=2,
    )
    spec = SimpleNamespace(
        enable_adaptive_verification=False,
        num_speculative_tokens=5,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dspark_block_size=5)
        ),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        speculative_config=spec,
    )

    nvidia_dspark.DeepSeekV4DSparkLayer(
        vllm_config,
        dspark_layer_idx=1,
        prefix="draft",
    )

    assert captured == {
        "prefix": "draft.layers.5.ffn",
        "use_sequence_parallel": False,
        "num_hash_layers": 2,
    }


@pytest.mark.parametrize("adaptive", [False, True])
def test_adaptive_logits_normalize_without_mutating_confidence_input(adaptive):
    hidden = torch.tensor([[2.0, 4.0]])
    model = SimpleNamespace(
        enable_adaptive_verification=adaptive,
        norm=lambda x: x / 2,
        head=None,
        logits_processor=lambda head, x: x,
    )
    logits = nvidia_dspark.DSparkDeepseekV4ForCausalLM.compute_logits(model, hidden)
    torch.testing.assert_close(logits, hidden / 2 if adaptive else hidden)
    torch.testing.assert_close(hidden, torch.tensor([[2.0, 4.0]]))


def test_map_draft_to_target_is_identity_for_full_vocab():
    src = inspect.getsource(
        nvidia_dspark.DSparkDeepseekV4ForCausalLM.map_draft_to_target
    )
    assert "return draft_ids" in src


def test_sequential_sampling_keeps_logits_in_reduced_draft_vocabulary():
    """Markov bias is draft-sized; target-sized logits cannot be added to it."""
    draft_logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
    d2t = torch.tensor([7, 11, 19])
    model = SimpleNamespace(
        compute_draft_logits=lambda hidden: draft_logits,
        compute_logits=lambda hidden: torch.zeros(2, 20),
        markov_embed=lambda previous: previous,
        markov_bias=lambda embedding: torch.zeros(1, 3),
    )
    speculator = SimpleNamespace(
        _draft_topk=None,
        num_speculative_steps=2,
        sample_indices=torch.arange(2),
        sample_idx_mapping=torch.zeros(2, dtype=torch.long),
        sample_pos=torch.tensor([5, 6]),
        input_buffers=SimpleNamespace(input_ids=torch.tensor([4, 0])),
        _anchor_idx=torch.tensor([0]),
        model=model,
        enable_adaptive_verification=False,
        _sample_logits=lambda logits, *args: d2t[logits.argmax(dim=-1)],
        draft_tokens=torch.empty(1, 2, dtype=torch.long),
    )
    DSparkSpeculator._sample_sequential(speculator, 1, torch.zeros(2, 4))
    assert speculator.draft_tokens.tolist() == [[11, 7]]
