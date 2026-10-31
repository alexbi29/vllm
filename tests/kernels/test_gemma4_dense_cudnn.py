# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regressions for audited-plan execution and narrow Gemma4 dispatch."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import gemma4_dense_cudnn as dense
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


@dataclass
class TensorMetadata:
    shape: tuple = (3, 2816)
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda:0")
    contiguous: bool = True
    pointer: int = 4096

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer


class Graph:
    def __init__(self, plans=None, workspace=0):
        self.plans = plans if plans is not None else [(11, {12: 18}), (10, {12: 18})]
        self.workspace = workspace
        self.queries = []
        self.executions = []

    def get_execution_plan_count(self):
        return len(self.plans)

    def get_engine_and_knobs_at_index(self, index):
        return self.plans[index]

    def get_workspace_size_plan_at_index(self, index, *args):
        self.queries.append(index)
        return self.workspace

    def execute_plan_at_index(self, pack, workspace, index, **kwargs):
        self.executions.append((index, pack, kwargs))

    def execute(self, *args, **kwargs):
        raise AssertionError("Heuristic execution is forbidden")


@pytest.fixture(autouse=True)
def clear_audits():
    cuda_initialized_before = torch.cuda.is_initialized()
    dense._PLAN_RECEIPTS.clear()
    dense._AUDITED_GRAPHS.clear()
    yield
    assert torch.cuda.is_initialized() == cuda_initialized_before, (
        "These tests must not change CUDA initialization state"
    )


def execute(graph, *, workspace=None, capturing=False):
    return dense._execute_exact_plan(
        graph,
        {1: "input", 2: "weight", 3: "output"},
        torch.empty(0, dtype=torch.uint8) if workspace is None else workspace,
        handle="handle",
        uids=[1, 2, 3],
        shapes=[[1, 3, 8]] * 3,
        strides=[[24, 8, 1]] * 3,
        receipt_key="fixture",
        capturing=capturing,
    )


def test_matching_plan_executes_observed_index_with_zero_scratch():
    graph = Graph()
    receipt = execute(graph)
    assert receipt["plan_index"] == 1
    assert receipt["engine_id"] == 10
    assert receipt["knobs"] == ((12, 18),)
    assert receipt["required_workspace_bytes"] == 0
    assert receipt["supplied_workspace_bytes"] == 0
    assert graph.executions[0][0] == 1
    assert execute(graph, capturing=True) == receipt


@pytest.mark.parametrize("plans", [[], [(11, {12: 18})], [(10, {12: 19})]])
def test_missing_exact_engine_or_knobs_never_executes(plans):
    graph = Graph(plans=plans)
    with pytest.raises(RuntimeError, match="absent"):
        execute(graph)
    assert graph.queries == graph.executions == []


@pytest.mark.parametrize("required,supplied", [(64, 0), (-1, 0), (0, 64)])
def test_nonzero_or_invalid_workspace_never_executes(required, supplied):
    graph = Graph(workspace=required)
    with pytest.raises(RuntimeError, match="zero workspace"):
        execute(graph, workspace=torch.empty(supplied, dtype=torch.uint8))
    assert graph.executions == []


def test_capture_without_audit_and_changed_plan_both_fail_closed():
    graph = Graph()
    with pytest.raises(RuntimeError, match="not audited"):
        execute(graph, capturing=True)
    assert graph.executions == []
    execute(graph)
    graph.plans = [(10, {12: 18})]
    with pytest.raises(RuntimeError, match="metadata changed"):
        execute(graph, capturing=True)
    assert len(graph.executions) == 1


def test_unknown_indexed_graph_api_is_rejected(monkeypatch):
    graph = Graph()
    monkeypatch.setattr(graph, "execute_plan_at_index", None)
    with pytest.raises(RuntimeError, match="indexed execution APIs"):
        execute(graph)
    assert graph.queries == []


@pytest.mark.parametrize(
    "version,missing",
    [
        ("0.8.0", None),
        ("0.7.1", None),
        ("0.7.0.post1", "_get_cudnn_handle"),
        ("0.7.2", "_get_cudnn_handle"),
        ("0.7.2.post1", "_get_cudnn_handle"),
    ],
)
def test_unqualified_flashinfer_dependency_is_rejected(version, missing):
    api = SimpleNamespace(
        _cudnn_gemm_bf16_runner=lambda: None,
        _get_bf16_3d_shape_stride=lambda: None,
        _get_cudnn_handle=lambda: None,
    )
    if missing:
        setattr(api, missing, None)
    with pytest.raises(RuntimeError, match="audited cuDNN"):
        dense._check_flashinfer_compatibility(version, api)


@pytest.mark.parametrize(
    "target,changes,reason",
    [
        ("x", {"device": torch.device("cpu")}, "device"),
        ("weight", {"device": torch.device("cuda:1")}, "device"),
        ("x", {"dtype": torch.float16}, "dtype"),
        ("weight", {"shape": (1, 4224, 2816)}, "rank"),
        ("weight", {"shape": (4225, 2816)}, "shape"),
        ("x", {"shape": (3, 2815)}, "shape"),
        ("x", {"shape": (1, 2816)}, "rows"),
        ("x", {"contiguous": False}, "layout"),
        ("x", {"pointer": 4098}, "alignment"),
        ("weight", {"pointer": 4098}, "alignment"),
    ],
)
def test_unqualified_operands_retain_torch_fallback(
    monkeypatch, target, changes, reason
):
    monkeypatch.setattr(dense, "_supports_device", lambda device: True)
    x, weight = TensorMetadata(), TensorMetadata(shape=(4224, 2816))
    for name, value in changes.items():
        setattr(x if target == "x" else weight, name, value)
    assert dense._fallback_reason(x, weight) == reason


def test_supported_rows_and_actual_device_capability_are_required(monkeypatch):
    x, weight = TensorMetadata(), TensorMetadata(shape=(4224, 2816))
    monkeypatch.setattr(dense, "_supports_device", lambda device: True)
    for rows in (2, 3, 4, 6, 8):
        x.shape = (rows, 2816)
        assert dense._fallback_reason(x, weight) == ""
    monkeypatch.setattr(dense, "_supports_device", lambda device: False)
    assert dense._fallback_reason(x, weight) == "capability"


def test_cpu_fallback_preserves_torch_result_without_loading_cudnn(monkeypatch):
    monkeypatch.setattr(dense, "_gemm_base", lambda: pytest.fail("cuDNN must not load"))
    x, weight = torch.randn(2, 8), torch.randn(4, 8)
    assert torch.equal(
        dense._cudnn_bf16_impl(x, weight), torch.nn.functional.linear(x, weight)
    )


@pytest.mark.parametrize(
    "exclusion", ["lora", "tp", "ubatching", "bias", "quantized", "dtype", "alignment"]
)
def test_installation_excludes_unqualified_linear_methods(monkeypatch, exclusion):
    config = SimpleNamespace(
        lora_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
    )
    method = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
    original = lambda *args: "original"
    method._gemm_impl = original
    linear = SimpleNamespace(
        quant_method=method, bias=None, weight=TensorMetadata(shape=(4224, 2816))
    )
    if exclusion == "lora":
        config.lora_config = object()
    elif exclusion == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif exclusion == "ubatching":
        config.parallel_config.use_ubatching = True
    elif exclusion == "bias":
        linear.bias = object()
    elif exclusion == "quantized":
        linear.quant_method = object()
    elif exclusion == "alignment":
        linear.weight.pointer += 2
    else:
        linear.weight.dtype = torch.float16
    monkeypatch.setattr(dense, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(dense, "_supports_device", lambda device: True)
    assert not dense.install_gemma4_mlp_cudnn(linear)
    assert method._gemm_impl is original


def test_capture_cannot_build_a_plan_on_first_use(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(dense, "_gemm_base", lambda: None)
    with pytest.raises(RuntimeError, match="not audited"):
        dense._cudnn_bf16_cuda(TensorMetadata(), TensorMetadata(shape=(4224, 2816)))


def test_capture_reuses_audited_graph_with_current_weight_variant_pack(monkeypatch):
    graph = Graph()
    dense._AUDITED_GRAPHS["device=None/rows=3/weight=(4, 8)"] = graph
    runner = SimpleNamespace(
        _get_override_graph=lambda *a, **k: pytest.fail("No plan build")
    )
    workspace = torch.empty(0, dtype=torch.uint8)
    api = SimpleNamespace(
        UIDs=SimpleNamespace(
            **{
                name: SimpleNamespace(value=i)
                for i, name in enumerate(("A_UID", "B_UID", "O_UID"), 1)
            }
        ),
        _get_bf16_3d_shape_stride=lambda value: (
            [1, *value.shape],
            [value.numel(), *value.stride()],
        ),
        _get_cudnn_handle=lambda *args: "handle",
    )
    monkeypatch.setattr(dense, "_gemm_base", lambda: api)
    monkeypatch.setattr(
        dense, "_runner_and_workspace", lambda device: (runner, workspace)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "stream")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x, first, second = torch.empty(3, 8), torch.empty(4, 8), torch.empty(4, 8)
    dense._cudnn_bf16_cuda(x, first)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    dense._cudnn_bf16_cuda(x, second)
    assert len(graph.executions) == 2
    assert graph.executions[-1][1][2].data_ptr() == second.data_ptr()


def test_install_is_per_instance_idempotent_and_preserves_bias_fallback(monkeypatch):
    config = SimpleNamespace(
        lora_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
    )
    method = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
    method._gemm_impl = lambda *args: "original bias path"
    linear = SimpleNamespace(
        quant_method=method, bias=None, weight=TensorMetadata(shape=(4224, 2816))
    )
    monkeypatch.setattr(dense, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(dense, "_supports_device", lambda device: True)
    before = dense._INSTALLED
    assert dense.install_gemma4_mlp_cudnn(linear)
    assert dense.install_gemma4_mlp_cudnn(linear)
    assert before + 1 == dense._INSTALLED
    assert type(method) is UnquantizedLinearMethod
    assert method._gemm_impl(linear, None, None, object()) == "original bias path"


def test_dynamic_export_keeps_row_dispatch_inside_the_custom_op():
    class Projection(torch.nn.Module):
        def forward(self, x, weight):
            return torch.ops.vllm.gemma4_dense_cudnn(x, weight)

    program = torch.export.export(
        Projection(),
        (torch.empty(3, 2816), torch.empty(4224, 2816)),
        dynamic_shapes=({0: torch.export.Dim("rows", min=1, max=16)}, None),
    )
    assert (
        sum(
            node.target == torch.ops.vllm.gemma4_dense_cudnn.default
            for node in program.graph.nodes
        )
        == 1
    )
