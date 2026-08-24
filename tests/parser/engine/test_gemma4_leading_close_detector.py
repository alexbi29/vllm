# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 reasoning-leak repair for post-tool-response continuations.

After a tool response the chat template PRE-OPENS the thought channel
(prompt ends with ``<|channel>thought\\n``). A malformed continuation can
emit ``<channel|>`` at token 0 — closing an *empty* thought — and then write
its chain-of-thought, which the parser (correctly, on that stream) labels
content, and a voice driver speaks verbatim. The parser absorbs those
leading close tags so the stream stays REASONING (CoT never reaches the
content channel), and logs the signature at ERROR level.
"""

from unittest.mock import patch

from tests.parser.engine import conftest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.gemma4 import Gemma4Parser, CHANNEL_END

CHANNEL_START_ID = 50  # <|channel>
CHANNEL_END_ID = 51  # <channel|>
VOCAB = {
    "<|channel>": CHANNEL_START_ID,
    "<channel|>": CHANNEL_END_ID,
    "<|tool_call>": 48,
    "<|tool_response>": 50,
    "<|turn>": 53,
}

MONOLOGUE = (
    "I should probably ask for a different factor Since both were wrong"
    "According to the instructions, when the number is on file only one is "
    "needed."
)


def _make_parser() -> tuple[Gemma4Parser, object]:
    tok = conftest.make_mock_tokenizer(VOCAB)
    return Gemma4Parser(tok), tok


def _open_channel_prompt() -> list[int]:
    # prompt tail inside an open <|channel> block (post-tool continuation)
    return [CHANNEL_START_ID, 3000, 3001]  # <|channel> thought \n ... (open)


def test_adjust_initial_state_arms_detector():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())
    assert parser._guarded_leading_close is True
    assert parser._engine.state == ParserState.REASONING


def test_fresh_turn_does_not_arm_detector():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt([40, 41, 42])  # no open channel
    assert parser._guarded_leading_close is False


def test_leading_close_absorbed_when_only_token():
    """A lone spurious close is dropped; the stream stays armed (still
    watching for more leading closes)."""
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed(CHANNEL_END, [CHANNEL_END_ID])
        assert mock_err.call_count == 1
        assert text == ""
        assert not ids
        assert parser._guarded_leading_close is True


def test_leading_close_stripped_from_text_and_ids():
    """Close tag stripped from both channels; the rest flows; guard disarms."""
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    text_in = CHANNEL_END + MONOLOGUE
    ids_in = [CHANNEL_END_ID, 6000, 6001, 6002]
    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed(text_in, ids_in)
        assert mock_err.call_count == 1
        assert text == MONOLOGUE
        assert list(ids) == [6000, 6001, 6002]
        assert parser._guarded_leading_close is False


def test_multiple_leading_closes_all_absorbed():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    text_in = CHANNEL_END + CHANNEL_END + "answer"
    ids_in = [CHANNEL_END_ID, CHANNEL_END_ID, 7000]
    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed(text_in, ids_in)
        assert mock_err.call_count == 1
        assert text == "answer"
        assert list(ids) == [7000]


def test_real_reasoning_first_is_untouched():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed(MONOLOGUE, [6000, 6001])
        assert mock_err.call_count == 0
        assert text == MONOLOGUE
        assert list(ids) == [6000, 6001]
        assert parser._guarded_leading_close is False


def test_empty_delta_keeps_detector_armed():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed("", [])
        assert mock_err.call_count == 0
        assert parser._guarded_leading_close is True  # still watching


def test_parse_delta_leak_is_reclassified_to_reasoning():
    """End-to-end: a leaked monologue after an empty close stays reasoning;
    content is empty; the error fires exactly once."""
    parser, tok = _make_parser()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    prompt_ids = _open_channel_prompt()

    seq = [CHANNEL_END_ID, 6000, 6001, 6002, 6003]
    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        results = []
        prompt = prompt_ids
        for i, tid in enumerate(seq):
            text = tok.decode([tid])
            r = parser.parse_delta(
                text,
                [tid],
                request,
                prompt_token_ids=prompt,
                finished=(i == len(seq) - 1),
            )
            prompt = None
            results.append(r)

    assert mock_err.call_count == 1
    reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
    content = "".join(r.content for r in results if r and r.content)
    # the CoT is captured on the reasoning channel; nothing reaches content
    assert content == ""
    # mock tokenizer renders unknown ids as <6000>...; reasoning is non-empty
    assert reasoning != ""
