# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NOTICE: tests a port of SpecPilot's W12 kernel
# (https://github.com/StargazerX0/SpecPilot, commit 78c42af, Apache-2.0).
"""SM120 W12 GEMM, decode, scatter and embedding kernels against references."""

import math

import pytest
import torch

from vllm.model_executor.layers import w12_linear as w12


def _available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in ((12, 0), (12, 1))
        and w12.kernels_available()
    )


pytestmark = pytest.mark.skipif(
    not _available(), reason="needs an SM120/SM121 GPU and the W12 kernels"
)

# Gemma4-26B BF16 linears; the LM-head stand-in keeps T > #SMs (owner slots).
SHAPES = {
    "qkv_sliding": (8192, 2816),
    "o_sliding": (2816, 4096),
    "qkv_global": (9216, 2816),
    "o_global": (2816, 8192),
    "gate_up": (4224, 2816),
    "down": (2816, 2112),
    "head_like": (32768, 2816),
}
ROWS = [1, 2, 3, 4, 7, 8, 9, 16, 17, 24, 31, 32, 40, 47, 48, 63, 64]


def _weight(n, k, seed, special=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    row_scale = torch.exp(torch.randn(n, 1, generator=g, device="cuda") * 0.5)
    w = torch.randn(n, k, generator=g, device="cuda") * 0.02 * row_scale
    w = w.bfloat16()
    if special:
        w[0] = 0.0  # all-zero row: E = 127, every weight escapes
        w[1, ::5] = 0.0
        w[2, 7] = 1e-30  # below the window
        w[3, 9] = 300.0  # above the window
        w[4, ::3] = torch.ldexp(
            torch.ones_like(w[4, ::3], dtype=torch.float32),
            torch.arange(w[4, ::3].numel(), device="cuda") % 80 - 40,
        ).bfloat16()  # 80 binades in one row
        sub = torch.arange(1, 128, device="cuda", dtype=torch.int16)
        w[5, : sub.numel()] = sub.view(torch.bfloat16)  # subnormals
    return w


def _x(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    return torch.randn(m, k, generator=g, device="cuda").bfloat16()


def _device(packed):
    t = packed.tensors()
    return {name: v.cuda() for name, v in t.items()}


def _run(x, packed, plan, kernel_max_m=-1, pdl=False):
    t = packed.tensors()
    return torch.ops.vllm.w12_linear(
        x,
        t["planes"],
        t["scale"],
        t["esc_rp"],
        t["esc_col"],
        t["esc_val"],
        t["esc_idx"],
        t["esc_bf16"],
        packed.n,
        packed.act,
        plan.num_ctas,
        plan.max_segments,
        kernel_max_m,
        pdl,
    )


def _prepare(n, k, seed=0, act=w12.ACT_NONE, special=True):
    w = _weight(n, k, seed, special)
    packed = w12.encode(w, act=act)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    plan = w12.W12Plan.for_shape(n, k, sms)
    w12.reserve_workspace(w.device, plan, n * k)
    return w, packed, plan


def _check_close(y, x, w, act=w12.ACT_NONE):
    ref = x.double() @ w.double().t()
    cublas = torch.nn.functional.linear(x, w).double()
    if act:
        half = ref.shape[1] // 2
        g, u = ref[:, :half], ref[:, half:]
        inner = math.sqrt(2.0 / math.pi) * (g + 0.044715 * g**3)
        ref = 0.5 * g * (1 + torch.tanh(inner)) * u
        out = torch.empty(x.shape[0], half, dtype=x.dtype, device=x.device)
        torch.ops._C.gelu_tanh_and_mul(out, cublas.bfloat16())
        cublas = out.double()
    err = (y.double() - ref).abs().max()
    base = (cublas - ref).abs().max()
    assert torch.isfinite(y).all()
    assert err <= 2 * base + ref.abs().max() * 2.0**-10, (float(err), float(base))


@pytest.mark.parametrize("name", list(SHAPES))
def test_gemm_matches_reference(name):
    n, k = SHAPES[name]
    w, packed, plan = _prepare(n, k, seed=len(name))
    for m in ROWS:
        x = _x(m, k, m)
        y = _run(x, packed, plan)
        _check_close(y, x, w)
        # bitwise deterministic for a fixed CTA count
        assert torch.equal(y, _run(x, packed, plan))


@pytest.mark.parametrize("m", [65, 100, 128, 129, 300])
def test_large_rows_kernel_chunks_and_decode_fallback(m):
    n, k = SHAPES["o_sliding"]
    w, packed, plan = _prepare(n, k, seed=3)
    x = _x(m, k, m)
    chunked = _run(x, packed, plan, kernel_max_m=-1)
    _check_close(chunked, x, w)
    fallback = _run(x, packed, plan, kernel_max_m=64)
    # decode + torch.mm on the exactly decoded weight is cuBLAS bitwise
    assert torch.equal(fallback, torch.nn.functional.linear(x, w))


def test_fused_gelu_tanh_gate_up():
    n, k = SHAPES["gate_up"]
    w, packed, plan = _prepare(n, k, seed=5, act=w12.ACT_GELU_TANH)
    for m in (1, 4, 8, 16, 33, 64):
        x = _x(m, k, m)
        y = _run(x, packed, plan)
        assert y.shape == (m, n // 2)
        _check_close(y, x, w, act=w12.ACT_GELU_TANH)
    # above the crossover: decode (de-interleaving) + torch.mm + vLLM's kernel
    x = _x(80, k, 80)
    y = _run(x, packed, plan, kernel_max_m=64)
    out = torch.empty(80, n // 2, dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_tanh_and_mul(out, torch.nn.functional.linear(x, w))
    assert torch.equal(y, out)


@pytest.mark.parametrize("act", [w12.ACT_NONE, w12.ACT_GELU_TANH])
def test_device_decode_is_bit_exact(act):
    n, k = SHAPES["gate_up"]
    w, packed, _ = _prepare(n, k, seed=7, act=act)
    t = packed.tensors()
    out = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
    w12._decode_into(
        out, t["planes"], t["scale"], t["esc_idx"], t["esc_bf16"], n, k, act
    )
    assert torch.equal(out.view(torch.int16), w.view(torch.int16))


def test_all_bf16_bit_patterns_decode_and_gather():
    bits = torch.arange(65536, dtype=torch.int32, device="cuda")
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    w = bits.view(torch.bfloat16).view(512, 128)
    packed = w12.encode(w)
    t = packed.tensors()
    out = torch.empty(512, 128, dtype=torch.bfloat16, device="cuda")
    w12._decode_into(
        out, t["planes"], t["scale"], t["esc_idx"], t["esc_bf16"], 512, 128, 0
    )
    assert torch.equal(out.view(torch.int16), w.view(torch.int16))
    ids = torch.arange(511, -1, -1, device="cuda")
    rows = torch.ops.vllm.w12_embedding(
        ids, t["planes"], t["scale"], t["esc_rp"], t["esc_col"], t["esc_bf16"], 512, 128
    )
    assert torch.equal(rows.view(torch.int16), w[ids].view(torch.int16))


def test_nonfinite_weights_propagate_like_cublas():
    n, k = 256, 2816
    w = _weight(n, k, 9, special=False)
    w[10, 3] = float("inf")
    w[11, 4] = float("nan")
    w[12, 5] = float("-inf")
    packed = w12.encode(w)
    plan = w12.W12Plan.for_shape(
        n, k, torch.cuda.get_device_properties(0).multi_processor_count
    )
    w12.reserve_workspace(w.device, plan)
    x = _x(5, k, 1)
    y = _run(x, packed, plan)
    ref = torch.nn.functional.linear(x, w)
    for row in (10, 11, 12):
        assert torch.equal(torch.isnan(y[:, row]), torch.isnan(ref[:, row]))
        assert torch.equal(torch.isinf(y[:, row]), torch.isinf(ref[:, row]))
    finite = [r for r in range(n) if r not in (10, 11, 12)]
    _check_close(y[:, finite], x, w[finite])


def test_embedding_gather_matches_rows():
    n, k = SHAPES["head_like"]
    w, packed, _ = _prepare(n, k, seed=11)
    t = packed.tensors()
    ids = torch.cat(
        [
            torch.tensor([0, 1, 2, 3, 4, 5, n - 1], device="cuda"),
            torch.randint(0, n, (497,), device="cuda"),
        ]
    ).view(-1, 7)
    rows = torch.ops.vllm.w12_embedding(
        ids, t["planes"], t["scale"], t["esc_rp"], t["esc_col"], t["esc_bf16"], n, k
    )
    assert rows.shape == (*ids.shape, k)
    assert torch.equal(rows.view(torch.int16), w[ids].view(torch.int16))


def test_cuda_graph_replay_matches_eager():
    n, k = SHAPES["qkv_sliding"]
    w, packed, plan = _prepare(n, k, seed=13)
    static_x = {m: _x(m, k, m) for m in (4, 16, 64, 96)}
    eager = {m: _run(x, packed, plan, kernel_max_m=64) for m, x in static_x.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = {
            m: _run(x, packed, plan, kernel_max_m=64) for m, x in static_x.items()
        }
    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize()
        for m in static_x:
            assert torch.equal(captured[m], eager[m])
    # new inputs written into the static buffers are picked up on replay
    for m, x in static_x.items():
        x.copy_(_x(m, k, m + 500))
    graph.replay()
    for m, x in static_x.items():
        assert torch.equal(captured[m], _run(x, packed, plan, kernel_max_m=64))


def test_pdl_launch_matches_plain_launch():
    n, k = SHAPES["down"]
    w, packed, plan = _prepare(n, k, seed=17)
    x = _x(8, k, 8)
    assert torch.equal(_run(x, packed, plan, pdl=True), _run(x, packed, plan))
