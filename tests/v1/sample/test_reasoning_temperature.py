# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the reasoning/answer temperature split on SamplingParams.

These lock in the contract the InputBatch split gate relies on: an unset
``reasoning_temperature`` must mean "mirror temperature" (no split, no
overhead), while a set, differing value opts the request into phase-dependent
sampling.
"""

from unittest.mock import Mock

import pytest

from vllm.sampling_params import _MAX_TEMP, SamplingParams, SamplingType


def test_default_is_unset():
    """An unspecified reasoning_temperature stays None (the no-split sentinel),
    so an ordinary greedy request is not pushed onto the split path."""
    params = SamplingParams(temperature=0.0)
    assert params.reasoning_temperature is None
    # Plain greedy request remains greedy and gets the greedy top_p/k reset.
    assert params.sampling_type == SamplingType.GREEDY
    assert params.top_p == 1.0
    assert params.top_k == 0
    assert params.min_p == 0.0


def test_from_optional_passes_none_through():
    """from_optional must not coerce an unset value to a concrete float;
    coercing to 1.0 was what made the split fire for nearly every request."""
    params = SamplingParams.from_optional(temperature=0.7)
    assert params.reasoning_temperature is None


@pytest.mark.parametrize("reasoning_temperature", [0.0, 0.7, 1.5])
def test_split_is_set_when_specified(reasoning_temperature):
    params = SamplingParams.from_optional(
        temperature=0.0, reasoning_temperature=reasoning_temperature
    )
    assert params.reasoning_temperature == reasoning_temperature


def test_greedy_reset_skipped_when_reasoning_is_random():
    """temperature=0 with a stochastic reasoning_temperature must NOT collapse
    top_p/top_k/min_p, since the reasoning phase still samples randomly."""
    params = SamplingParams(
        temperature=0.0,
        reasoning_temperature=0.7,
        top_p=0.5,
        top_k=20,
        min_p=0.1,
    )
    assert params.top_p == 0.5
    assert params.top_k == 20
    assert params.min_p == 0.1


def test_sampling_type_uses_reasoning_temperature():
    params = SamplingParams(
        temperature=0.0,
        reasoning_temperature=0.7,
        seed=123,
    )
    assert params.sampling_type == SamplingType.RANDOM_SEED


def test_spec_decode_disables_reasoning_temperature_split():
    params = SamplingParams(
        temperature=0.0,
        reasoning_temperature=0.7,
        top_p=0.5,
        top_k=20,
        min_p=0.1,
    )
    assert params.sampling_type == SamplingType.RANDOM

    params._validate_spec_decode(Mock())

    assert params.reasoning_temperature is None
    assert params.sampling_type == SamplingType.GREEDY
    assert params.top_p == 1.0
    assert params.top_k == 0
    assert params.min_p == 0.0


def test_greedy_reset_applies_when_both_greedy():
    """temperature=0 and reasoning_temperature=0 is fully greedy and should
    reset the nucleus parameters just like a plain greedy request."""
    params = SamplingParams(
        temperature=0.0,
        reasoning_temperature=0.0,
        top_p=0.5,
        top_k=20,
        min_p=0.1,
    )
    assert params.top_p == 1.0
    assert params.top_k == 0
    assert params.min_p == 0.0


def test_negative_reasoning_temperature_rejected():
    with pytest.raises(ValueError):
        SamplingParams(reasoning_temperature=-0.1)


def test_tiny_positive_reasoning_temperature_clamped():
    params = SamplingParams(reasoning_temperature=_MAX_TEMP / 2)
    assert params.reasoning_temperature == _MAX_TEMP


def test_unset_reasoning_temperature_not_clamped():
    """None must survive __post_init__ untouched (no clamp, no validation
    error)."""
    params = SamplingParams(temperature=0.0)
    assert params.reasoning_temperature is None
