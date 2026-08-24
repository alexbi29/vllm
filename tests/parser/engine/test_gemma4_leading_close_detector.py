# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 reasoning-leak detector for post-tool-response continuations.

After a tool response the chat template PRE-OPENS the thought channel
(prompt ends with ``<|channel>thought\\n``). A malformed continuation can
emit ``<channel|>`` at token 0 — closing an *empty* thought — and then write
its chain-of-thought, which the parser (correctly, on that stream) labels
content. The detector logs that signature at ERROR level and must NOT modify
the stream.
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


def test_leading_close_logs_error_and_passes_stream_through():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed(CHANNEL_END, [CHANNEL_END_ID])
        assert mock_err.call_count == 1
        # stream untouched
        assert text == CHANNEL_END
        assert list(ids) == [CHANNEL_END_ID]


def test_leading_close_with_text_logs_and_passes_both_through():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    text_in = CHANNEL_END + MONOLOGUE
    ids_in = [CHANNEL_END_ID, 6000, 6001, 6002]
    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed(text_in, ids_in)
        assert mock_err.call_count == 1
        assert text == text_in
        assert list(ids) == ids_in
        assert parser._guarded_leading_close is False  # disarmed after output


def test_real_reasoning_first_does_not_log():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed(MONOLOGUE, [6000, 6001])
        assert mock_err.call_count == 0
        assert text == MONOLOGUE
        assert list(ids) == [6000, 6001]


def test_empty_delta_keeps_detector_armed():
    parser, _ = _make_parser()
    parser.adjust_initial_state_from_prompt(_open_channel_prompt())

    with patch("vllm.parser.gemma4.logger.error") as mock_err:
        text, ids = parser._preprocess_feed("", [])
        assert mock_err.call_count == 0
        assert parser._guarded_leading_close is True  # still watching


def test_parse_delta_leak_path_is_log_only():
    """End-to-end: the leaked monologue still streams as content (unchanged),
    and the error fires exactly once for the leading close."""
    parser, tok = _make_parser()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    # prompt tail in an open channel -> detector armed via parse_delta path
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
    # the stream is not modified: the monologue after the close is content
    # (the mock tokenizer renders unknown ids as <id>, so just assert the
    # trailing tokens streamed as content and none as reasoning)
    assert reasoning == ""
    assert content != ""
    assert CHANNEL_END not in content
