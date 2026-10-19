# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM120 BF16 Gemma4 shared-MLP projections with zero-scratch cuDNN."""

import functools
from collections import Counter

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Audited on NVIDIA Gemma4-26B checkpoint weights, M=2/3/4/6/8.
# Attention/logits projections and other rows retain Torch's implementation.
_TACTIC = (10, ((12, 18),))
_FLASHINFER_VERSION = "0.7.0.post1"
_SHAPES = {(4224, 2816), (2816, 2112)}
_ROWS = {2, 3, 4, 6, 8}
_DISPATCH: Counter[str] = Counter()
_FALLBACK: Counter[str] = Counter()
_PREWARM = False
_INSTALLED = 0
_PREWARMED = 0
_PLAN_RECEIPTS: dict[str, dict] = {}
_AUDITED_GRAPHS: dict[str, object] = {}


def gemma4_dense_cudnn_state() -> dict:
    return {
        "scope": (
            "Python calls during prewarm/eager/capture only; "
            "CUDA graph replay does not increment these counters"
        ),
        "installed_linears": _INSTALLED,
        "prewarmed_linears": _PREWARMED,
        "retained_workspace_bytes": 0,
        "rows": sorted(_ROWS),
        "requested_tactic": _TACTIC,
        "dispatch": dict(_DISPATCH),
        "fallback": dict(_FALLBACK),
        "plan_receipts": dict(_PLAN_RECEIPTS),
    }


def _check_flashinfer_compatibility(version: str, gemm_base) -> None:
    required = (
        "_cudnn_gemm_bf16_runner",
        "_get_bf16_3d_shape_stride",
        "_get_cudnn_handle",
    )
    if version != _FLASHINFER_VERSION or any(
        not callable(getattr(gemm_base, name, None)) for name in required
    ):
        raise RuntimeError(
            "VLLM_GEMMA4_DENSE_CUDNN requires FlashInfer "
            f"{_FLASHINFER_VERSION} and its audited cuDNN override-shape APIs"
        )
    uids = getattr(gemm_base, "UIDs", None)
    if any(getattr(uids, name, None) is None for name in ("A_UID", "B_UID", "O_UID")):
        raise RuntimeError("FlashInfer cuDNN tensor UIDs are unavailable")


@functools.cache
def _gemm_base():
    import flashinfer
    from flashinfer.gemm import gemm_base

    # Public mm_bf16 may allocate scratch or select a heuristic after an error.
    # Retain the measured graph builder; dependency upgrades need qualification.
    _check_flashinfer_compatibility(flashinfer.__version__, gemm_base)
    return gemm_base


@functools.cache
def _supports_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (12, 0)


def _fallback_reason(x: torch.Tensor, weight: torch.Tensor) -> str:
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        return "device"
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return "dtype"
    if x.ndim != 2 or weight.ndim != 2:
        return "rank"
    if tuple(weight.shape) not in _SHAPES or x.shape[1] != weight.shape[1]:
        return "shape"
    if x.shape[0] not in _ROWS:
        return "rows"
    if not x.is_contiguous() or not weight.is_contiguous():
        return "layout"
    if x.data_ptr() % 16 or weight.data_ptr() % 16:
        return "alignment"
    if not _supports_device(x.device):
        return "capability"
    return ""


def _resolve_exact_plan(graph, expected_tactic) -> tuple[int, tuple]:
    required = (
        "get_execution_plan_count",
        "get_engine_and_knobs_at_index",
        "get_workspace_size_plan_at_index",
        "execute_plan_at_index",
    )
    if any(not callable(getattr(graph, name, None)) for name in required):
        raise RuntimeError("cuDNN graph lacks the audited indexed execution APIs")
    expected = (
        int(expected_tactic[0]),
        tuple(sorted((int(k), int(v)) for k, v in expected_tactic[1])),
    )
    observed: list[tuple[int, object]] = []
    for index in range(graph.get_execution_plan_count()):
        try:
            engine, knobs = graph.get_engine_and_knobs_at_index(index)
        except (AttributeError, RuntimeError) as exc:
            observed.append((index, repr(exc)))
            continue
        actual = (
            int(engine),
            tuple(sorted((int(k), int(v)) for k, v in knobs.items())),
        )
        if actual == expected:
            return index, actual
        observed.append((index, actual))
    raise RuntimeError(
        f"Requested cuDNN plan {expected} is absent; observed={observed}"
    )


def _execute_exact_plan(
    graph,
    variant_pack,
    workspace,
    *,
    handle,
    uids,
    shapes,
    strides,
    receipt_key: str,
    capturing: bool = False,
    expected_tactic=_TACTIC,
) -> dict:
    # Do not call the dependency's tactic resolver: it warns and returns -1
    # on missing engines, allowing its executor to run a heuristic instead.
    plan_index, actual = _resolve_exact_plan(graph, expected_tactic)
    if plan_index < 0:
        raise RuntimeError("Exact cuDNN plan must have a nonnegative index")
    required = int(
        graph.get_workspace_size_plan_at_index(
            plan_index, handle, uids, shapes, strides
        )
    )
    supplied = workspace.numel() * workspace.element_size()
    if required != 0 or supplied != 0:
        raise RuntimeError(
            "Gemma4 dense cuDNN requires zero workspace: "
            f"required={required}, supplied={supplied}"
        )
    receipt = {
        "plan_index": plan_index,
        "execution_plan_count": graph.get_execution_plan_count(),
        "engine_id": actual[0],
        "knobs": actual[1],
        "required_workspace_bytes": required,
        "supplied_workspace_bytes": supplied,
        "override_uids": uids,
        "override_shapes": shapes,
        "override_strides": strides,
    }
    previous = _PLAN_RECEIPTS.get(receipt_key)
    if previous is None:
        if capturing:
            raise RuntimeError("cuDNN plan was not audited before CUDA capture")
        _PLAN_RECEIPTS[receipt_key] = receipt
        logger.info("Dense cuDNN observed plan %s: %s", receipt_key, receipt)
    elif previous != receipt:
        raise RuntimeError("cuDNN plan metadata changed after its warmup audit")
    # Execute the verified index directly. This path cannot select a heuristic
    # or allocate substitute scratch in the dependency's workspace helper.
    graph.execute_plan_at_index(
        variant_pack,
        workspace,
        plan_index,
        handle=handle,
        override_uids=uids,
        override_shapes=shapes,
        override_strides=strides,
    )
    return receipt


@functools.cache
def _runner_and_workspace(device_index: int):
    gemm_base = _gemm_base()

    runner = gemm_base._cudnn_gemm_bf16_runner(True, True)
    if not runner._use_override_shape:
        raise RuntimeError("Gemma4 dense cuDNN requires override-shape support")
    if not callable(getattr(runner, "_get_override_graph", None)):
        raise RuntimeError(
            "FlashInfer cuDNN override-shape graph builder is unavailable"
        )
    # The selected engine reports zero workspace for all screened shapes.
    # Bypass mm_bf16's unconditional 40MiB generic workspace allocation.
    workspace = torch.empty(0, device=f"cuda:{device_index}", dtype=torch.uint8)
    return runner, workspace


def _cudnn_bf16_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0] if x.ndim else 0
    reason = _fallback_reason(x, weight)
    if reason:
        _FALLBACK[f"{reason}/rows={rows}/weight={tuple(weight.shape)}"] += 1
        return torch.nn.functional.linear(x, weight)
    # Capture and cuDNN handles must refer to the input's device, even when
    # another device happens to be current in the calling host thread.
    with torch.accelerator.device_index(x.device.index):
        return _cudnn_bf16_cuda(x, weight)


def _cudnn_bf16_cuda(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0]
    capturing = torch.cuda.is_current_stream_capturing()
    key = (
        f"rows={rows}/weight={tuple(weight.shape)}/"
        f"capture={capturing}/prewarm={_PREWARM}"
    )
    gemm_base = _gemm_base()

    receipt_key = f"device={x.device.index}/rows={rows}/weight={tuple(weight.shape)}"
    graph = _AUDITED_GRAPHS.get(receipt_key)
    if graph is None and capturing:
        raise RuntimeError("cuDNN plan was not audited before CUDA capture")
    out = x.new_empty((x.shape[0], weight.shape[0]))
    runner, workspace = _runner_and_workspace(x.device.index)
    if graph is None:
        graph = runner._get_override_graph(x, weight.t(), None, out, tactic=_TACTIC)
    b = weight.t()
    uids = [
        gemm_base.UIDs.A_UID.value,
        gemm_base.UIDs.B_UID.value,
        gemm_base.UIDs.O_UID.value,
    ]
    descriptors = [gemm_base._get_bf16_3d_shape_stride(value) for value in (x, b, out)]
    shapes = [list(item[0]) for item in descriptors]
    strides = [list(item[1]) for item in descriptors]
    handle = gemm_base._get_cudnn_handle(x.device, torch.cuda.current_stream(x.device))
    _execute_exact_plan(
        graph,
        dict(zip(uids, (x, b, out))),
        workspace,
        handle=handle,
        uids=uids,
        shapes=shapes,
        strides=strides,
        receipt_key=receipt_key,
        capturing=capturing,
    )
    _AUDITED_GRAPHS[receipt_key] = graph
    _DISPATCH[key] += 1
    return out


def _cudnn_bf16_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="gemma4_dense_cudnn",
    op_func=_cudnn_bf16_impl,
    fake_impl=_cudnn_bf16_fake,
)


def install_gemma4_mlp_cudnn(linear: torch.nn.Module) -> bool:
    """Install only the audited unquantized Gemma4 shared-MLP projections."""
    global _INSTALLED
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(linear, "_gemma4_dense_cudnn", False):
        return True
    config = get_current_vllm_config_or_none()
    if config is None or config.lora_config is not None:
        return False
    if config.parallel_config.tensor_parallel_size != 1:
        return False
    # FlashInfer shares one stream-rebound cuDNN handle per device.
    if config.parallel_config.use_ubatching:
        return False
    if getattr(linear, "bias", None) is not None:
        return False
    method = linear.quant_method
    if type(method) is not UnquantizedLinearMethod:
        return False
    weight = linear.weight
    if not weight.is_cuda or weight.dtype != torch.bfloat16:
        return False
    if tuple(weight.shape) not in _SHAPES or not _supports_device(weight.device):
        return False
    if not weight.is_contiguous() or weight.data_ptr() % 16:
        return False
    original = method._gemm_impl
    original_process = method.process_weights_after_loading

    def process(layer):
        global _PREWARM, _PREWARMED
        original_process(layer)
        weight = layer.weight
        if weight.is_cuda and weight.dtype == torch.bfloat16:
            if not weight.is_contiguous():
                raise RuntimeError(
                    "Gemma4 dense cuDNN requires contiguous loaded weights"
                )
            if weight.data_ptr() % 16:
                raise RuntimeError(
                    "Gemma4 dense cuDNN requires 16B-aligned loaded weights"
                )
            # Plan construction and zero-workspace validation finish before
            # profiling/capture. Plans depend on shape/layout, not weight ptr.
            _PREWARM = True
            try:
                for rows in sorted(_ROWS):
                    values = torch.arange(
                        rows * weight.shape[1],
                        device=weight.device,
                        dtype=torch.float32,
                    )
                    x = (values.sin() * 0.3).reshape(rows, -1).to(weight.dtype)
                    actual = _cudnn_bf16_impl(x, weight)
                    expected = torch.nn.functional.linear(x, weight)
                    if not torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    ):
                        raise RuntimeError(
                            "Gemma4 dense cuDNN prewarm differs from Torch: "
                            f"weight={tuple(weight.shape)}, rows={rows}"
                        )
            finally:
                _PREWARM = False
            _PREWARMED += 1

    def gemm(layer, x, weight, bias=None):
        if bias is None:
            return torch.ops.vllm.gemma4_dense_cudnn(x, weight)
        return original(layer, x, weight, bias)

    method._gemm_impl = gemm
    # Keep the existing unquantized method and its original load processing.
    method.process_weights_after_loading = process  # type: ignore[method-assign]
    linear._gemma4_dense_cudnn = True
    _INSTALLED += 1
    return True
