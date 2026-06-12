# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


class _TerminatingMatcher:
    def __init__(self) -> None:
        self.accepted_tokens: list[int] = []
        self.terminated = False
        self.rollback_count = 0

    def accept_token(self, token: int) -> bool:
        assert not self.terminated, f"accepted token after termination: {token}"
        self.accepted_tokens.append(token)
        if token == 1:
            self.terminated = True
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def rollback(self, num_tokens: int) -> None:
        self.rollback_count += num_tokens
        del self.accepted_tokens[-num_tokens:]
        self.terminated = False


def test_accept_tokens_stops_after_grammar_termination():
    matcher = _TerminatingMatcher()
    grammar = XgrammarGrammar(vocab_size=10, matcher=matcher, ctx=None)  # type: ignore[arg-type]

    assert grammar.accept_tokens("req", [1, 198])

    assert matcher.accepted_tokens == [1]
    assert grammar.num_processed_tokens == 1
    assert grammar.is_terminated()


def test_validate_tokens_stops_after_grammar_termination_and_rolls_back():
    matcher = _TerminatingMatcher()
    grammar = XgrammarGrammar(vocab_size=10, matcher=matcher, ctx=None)  # type: ignore[arg-type]

    assert grammar.validate_tokens([1, 198]) == [1]

    assert matcher.accepted_tokens == []
    assert matcher.rollback_count == 1
    assert not grammar.is_terminated()
