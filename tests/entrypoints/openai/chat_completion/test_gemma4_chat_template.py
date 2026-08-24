"""Regression tests for the deployed Gemma 4 tool chat template."""

from pathlib import Path

import pytest
from jinja2 import Environment


TEMPLATE = (
    Path(__file__).parents[4] / "examples" / "tool_chat_template_gemma4.jinja"
)
USER = {"role": "user", "content": "question"}
TOOL_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "call-1",
        "function": {"name": "faq_search", "arguments": {"query": "q"}},
    }],
}
TOOL_RESULT = {
    "role": "tool",
    "tool_call_id": "call-1",
    "content": "answer",
}
REASONED = {
    "role": "assistant",
    "content": None,
    "reasoning": "injected hidden reasoning",
}


@pytest.mark.parametrize(
    ("messages", "suffix"),
    [
        ([USER], "<|turn>model\n<|channel>thought\n<channel|>"),
        ([USER, TOOL_CALL, TOOL_RESULT], "<|channel>thought\n<channel|>"),
        # Injection: supplied reasoning in no-think mode is rendered, so no
        # empty thought-close is appended after the turn opener.
        ([USER, REASONED], "<|turn>model\n"),
    ],
    ids=["ordinary", "after-tool", "injected-reasoning"],
)
def test_no_think_generation_boundary(messages, suffix):
    rendered = Environment().from_string(TEMPLATE.read_text()).render(
        messages=messages,
        tools=None,
        bos_token="<BOS>",
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=False,
        raise_exception=lambda message: pytest.fail(message),
    )
    assert rendered.endswith(suffix)
