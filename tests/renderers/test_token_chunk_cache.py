# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Renderer token chunk cache: piecewise encoding at added tokens."""

import random

import pytest
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import TokenizersBackend

from vllm.renderers import base

SPECIALS = ["<|turn>", "<turn|>", "<|tool_call>"]
WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "naïve",
    "漢字",
    "tool_call",
    "{json}",
    "\t",
    "tabs\nnew",
    "lines",
]


def _corpus(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(40)) for _ in range(n)]


def _tokenizer(pre_tokenizer=None, special_kwargs=None) -> TokenizersBackend:
    tk = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    # Gemma4-like: spaces become a marker in the normalizer.
    tk.normalizer = normalizers.Replace(" ", "▁")
    tk.pre_tokenizer = pre_tokenizer or pre_tokenizers.Split(
        Regex(" "), behavior="merged_with_previous"
    )
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>"])
    tk.train_from_iterator(_corpus(0, 200), trainer)
    tk.add_special_tokens([AddedToken(s, **(special_kwargs or {})) for s in SPECIALS])
    return TokenizersBackend(tokenizer_object=tk)


def _chat(seed: int, turns: int) -> str:
    rng = random.Random(seed)
    text = []
    for t in range(turns):
        body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 120)))
        lead = rng.choice(["", " ", "\n", "  "])
        text.append(f"<|turn>{'user' if t % 2 else 'model'}\n{lead}{body}<turn|>\n")
        if rng.random() < 0.3:
            text.append(f"<|tool_call>{rng.choice(WORDS)}{{x}}<turn|>")
    return "".join(text)


@pytest.fixture(autouse=True)
def _small_threshold(monkeypatch):
    monkeypatch.setattr(base, "_PREFIX_TOKEN_CACHE_MIN_CHARS", 100)
    monkeypatch.setattr(base, "_TOKEN_CHUNK_MIN_CHARS", 8)


def _full(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def test_chunk_cache_matches_full_encode_and_reuses_pieces():
    tok = _tokenizer()
    cache = base._TokenChunkCache(max_chars=10**7)
    history = ""
    for turn in range(30):
        history += _chat(turn, 2)
        assert cache.encode(tok, history) == _full(tok, history)
    assert cache.hit_chars > cache.miss_chars


@pytest.mark.parametrize(
    "pre_tokenizer, special_kwargs",
    [
        # A marker added only at the start of the input is not piece-local.
        (pre_tokenizers.Metaspace(prepend_scheme="first"), None),
        # Whitespace-stripping added tokens change their neighbours.
        (None, {"lstrip": True}),
        # Normalized added tokens are matched after normalization.
        (None, {"normalized": True}),
    ],
)
def test_chunk_cache_refuses_unsafe_tokenizers(pre_tokenizer, special_kwargs):
    tok = _tokenizer(pre_tokenizer, special_kwargs)
    cache = base._TokenChunkCache(max_chars=10**7)
    assert cache.encode(tok, _chat(1, 20)) is None


def test_chunk_cache_skips_short_prompts():
    tok = _tokenizer()
    cache = base._TokenChunkCache(max_chars=10**7)
    assert cache.encode(tok, "<|turn>user\nhi<turn|>") is None


def test_chunk_cache_evicts_by_cached_characters():
    tok = _tokenizer()
    cache = base._TokenChunkCache(max_chars=2000)
    for seed in range(40):
        text = _chat(100 + seed, 10)
        assert cache.encode(tok, text) == _full(tok, text)
        assert cache._chars <= 2000


def test_chunk_cache_disables_itself_on_mismatch(monkeypatch):
    monkeypatch.setattr(base, "_TOKEN_CHUNK_VERIFY_EVERY", 1)
    tok = _tokenizer()
    cache = base._TokenChunkCache(max_chars=10**7)
    text = _chat(7, 20)
    cache.encode(tok, text)
    for key in cache._entries:
        cache._entries[key] = [0]
    assert cache.encode(tok, text) == _full(tok, text)
    assert cache.encode(tok, text) is None
