"""
Reasoning parser for LFM2/LFM2.5 models.

Model outputs: <think>\n...reasoning...\n</think>\n...answer...
"""
from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

_START = "<think>"
_END = "</think>"


class Lfm2ReasoningParser(ReasoningParser):
    """Reasoning parser for LFM2/LFM2.5 models."""

    @property
    def reasoning_start_str(self):
        return _START

    @property
    def reasoning_end_str(self):
        return _END

    def extract_reasoning(self, model_output, request):
        start = model_output.find(_START)
        if start == -1:
            return None, model_output

        start += len(_START)
        while start < len(model_output) and model_output[start] in " \n\r\t":
            start += 1

        end = model_output.find(_END, start)
        if end == -1:
            reasoning = model_output[start:].strip()
            return (reasoning or None), None

        reasoning = model_output[start:end].strip()
        content_start = end + len(_END)
        while (content_start < len(model_output)
               and model_output[content_start] in " \n\r\t"):
            content_start += 1
        content = model_output[content_start:].strip() or None
        return (reasoning or None), content

    def extract_reasoning_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids,
    ):
        prev_end = _END in previous_text
        curr_end = _END in current_text

        if curr_end and prev_end:
            return DeltaMessage(content=delta_text)

        if curr_end and not prev_end:
            idx = current_text.find(_END, len(previous_text))
            if idx < 0:
                idx = current_text.find(_END)
            r = current_text[len(previous_text):idx]
            c = current_text[idx + len(_END):].lstrip() or None
            return DeltaMessage(reasoning=r or None, content=c)

        prev_start = _START in previous_text
        curr_start = _START in current_text

        if not curr_start and not prev_start:
            return DeltaMessage(content=delta_text)

        if curr_start and not prev_start:
            idx = current_text.find(_START, len(previous_text))
            if idx < 0:
                idx = current_text.find(_START)
            cb = current_text[len(previous_text):idx]
            r = current_text[idx + len(_START):].lstrip()
            return DeltaMessage(reasoning=r or None, content=cb or None)

        return DeltaMessage(reasoning=delta_text)
