# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.kv_offload.cpu import gpu_worker


def test_registration_error_is_drained_before_retry(monkeypatch):
    zero = Mock(side_effect=[RuntimeError("latched registration error"), None])
    empty = Mock(return_value=SimpleNamespace(zero_=zero))
    monkeypatch.setattr(gpu_worker.torch, "empty", empty)
    gpu_worker._drain_latched_cuda_error()
    assert zero.call_count == 2
    empty.assert_called_with(1, device="cuda")


def test_persistent_registration_error_is_reported(monkeypatch):
    zero = Mock(side_effect=RuntimeError("unrecoverable"))
    monkeypatch.setattr(
        gpu_worker.torch, "empty", Mock(return_value=SimpleNamespace(zero_=zero))
    )
    warning = Mock()
    monkeypatch.setattr(gpu_worker.logger, "warning", warning)
    gpu_worker._drain_latched_cuda_error()
    assert zero.call_count == 2
    warning.assert_called_once()
