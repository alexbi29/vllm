# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NOTICE: tests a port of SpecPilot's W12 format
# (https://github.com/StargazerX0/SpecPilot, commit 78c42af, Apache-2.0).
"""CPU-only tests of the W12 encoder, record layout, kernel algebra, stream-K
plan and op routing (with a pure-torch fake of the native kernels that asserts
their contract)."""

import math
import zlib

import pytest
import torch

from vllm.model_executor.layers import w12_linear as w12

GEMMA_SHAPES = {
    "qkv_sliding": (8192, 2816),
    "o_sliding": (2816, 4096),
    "qkv_global": (9216, 2816),
    "o_global": (2816, 8192),
    "gate_up": (4224, 2816),
    "down": (2816, 2112),
    "lm_head": (262144, 2816),
}


@pytest.fixture(autouse=True)
def keep_cuda_uninitialized():
    before = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == before


def bits_eq(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def gaussian(n, k, seed, std=0.02):
    g = torch.Generator().manual_seed(seed)
    row_scale = torch.exp(torch.randn(n, 1, generator=g) * 0.5)
    return (torch.randn(n, k, generator=g) * std * row_scale).bfloat16()


def from_bits(bits):
    bits = bits.to(torch.int32) & 0xFFFF
    signed = torch.where(bits >= 0x8000, bits - 0x10000, bits)
    return signed.to(torch.int16).view(torch.bfloat16)


def roundtrip(w, act=w12.ACT_NONE):
    p = w12.encode(w, act=act, check=False)
    assert bits_eq(w12.decode(p), w)
    assert torch.isfinite(p.scale).all()
    assert bool((p.top >= 15).all()) and bool((p.top <= 254).all())
    assert p.planes.shape == (p.tiles, (p.k // 64) * w12.WREC)
    return p


# ----------------------------------------------------------------- format


@pytest.mark.parametrize("name", list(GEMMA_SHAPES))
def test_roundtrip_gemma_shapes(name):
    _, k = GEMMA_SHAPES[name]
    p = roundtrip(gaussian(256, k, seed=zlib.crc32(name.encode()) % 1000))
    assert p.n_inexact == 0
    assert 12.0 < p.stats()["bits_per_weight_stream"] < 12.1


def test_roundtrip_padded_rows_and_interleaved_gate_up():
    assert roundtrip(gaussian(200, 2816, seed=3)).tiles == 2
    p = roundtrip(gaussian(4224, 2816, seed=11), act=w12.ACT_GELU_TANH)
    assert p.n_out == 2112 and p.tiles == 33


def test_all_65536_bf16_patterns_roundtrip():
    w = from_bits(torch.arange(65536)).view(512, 128)
    p = roundtrip(w)
    assert p.breakdown["zero"] == 2
    assert p.breakdown["subnormal"] == 2 * 127
    assert p.breakdown["nonfinite"] == 2 * 128
    g = torch.Generator().manual_seed(0)
    roundtrip(w.view(-1)[torch.randperm(65536, generator=g)].view(256, 256))


def test_degenerate_rows_use_a_finite_scale():
    k = 2816
    g = torch.Generator().manual_seed(7)
    rows = [
        torch.zeros(k),
        -torch.zeros(k),
        torch.full((k,), float("inf")),
        torch.full((k,), float("nan")),
        torch.ldexp(torch.ones(k), torch.randint(-126, 127, (k,), generator=g)),
    ]
    w = torch.cat([torch.stack(rows), gaussian(123, k, seed=8).float()]).bfloat16()
    p = roundtrip(w)
    for r in (0, 1, 2, 3):  # no in-window weight: E = 127, all escape
        assert int(p.top[r]) == 127
        assert int(p.esc_rp[r + 1] - p.esc_rp[r]) == k


def test_nonfinite_weights_do_not_move_the_window():
    w = gaussian(128, 2816, seed=9)
    w[:, ::7] = float("inf")
    w[:, 3::11] = float("nan")
    p1 = roundtrip(w)
    finite = torch.isfinite(w.float())
    p0 = roundtrip(torch.where(finite, w.float(), torch.zeros(())).bfloat16())
    assert torch.equal(p0.top, p1.top)
    assert p1.breakdown["nonfinite"] == int((~finite).sum())


def test_window_rule_17_binades_and_tie_to_highest_top():
    e = torch.arange(100, 117).repeat_interleave(64)
    p = roundtrip(from_bits((e << 7) | 0x15).view(1, -1).repeat(128, 1))
    assert int(p.top[0]) == 116 and p.n_escapes == 128 * 64
    two = torch.cat([torch.full((64,), 100), torch.full((64,), 130)]) << 7
    p = roundtrip(from_bits(two).view(1, 128).repeat(128, 1))
    assert int(p.top[0]) == 145 and p.breakdown["below_window"] == 128 * 64


def test_inexact_escapes_are_flagged():
    w = torch.full((128, 256), 2.0**100).bfloat16()
    w[0, 0] = 2.0**-120  # val underflows fp32 against E ~ 227
    assert roundtrip(w).n_inexact >= 1


# ----------------------------------------------------------------- layout


def _byte_perm(x, y, s):
    b = [(x >> (8 * i)) & 0xFF for i in range(4)] + [
        (y >> (8 * i)) & 0xFF for i in range(4)
    ]
    return sum(b[(s >> (4 * i)) & 7] << (8 * i) for i in range(4))


def _decode_lane(q0, q1, bb):
    """Bit-level emulation of the kernel's decode_n for one lane."""
    q = list(q0) + list(q1)
    a = [[0] * 4 for _ in range(4)]
    for j in range(4):
        for c8 in range(2):
            word = q[2 * j + c8]
            b = (bb[j] >> 4) & 0x0F0F0F0F if c8 else bb[j] & 0x0F0F0F0F
            lo = (word & 0xF0F0F0F0) | b
            hi = (((word << 4) & 0x80808080) | (word & 0x07070707)) | 0x38383838
            a[j][2 * c8] = _byte_perm(lo, hi, 0x5140)
            a[j][2 * c8 + 1] = _byte_perm(lo, hi, 0x7362)
    return a


def _u32(buf, off):
    v = buf[off : off + 16].tolist()
    return [
        v[4 * i] | (v[4 * i + 1] << 8) | (v[4 * i + 2] << 16) | (v[4 * i + 3] << 24)
        for i in range(4)
    ]


def test_record_layout_matches_mma_a_fragments():
    w = torch.randn(128, 128, generator=torch.Generator().manual_seed(1)) * 0.05
    w = w.bfloat16()
    w[3, 5] = 0.0
    w[17, 70] = 1e-30
    p = w12.encode(w)
    u = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    c = e - (p.top[:, None] - 15)
    esc = (e == 0) | (e == 255) | (c < 0) | (c > 15)
    expect = torch.where(
        esc, torch.full_like(u, 0x3800), (u & 0x807F) | ((112 + c) << 7)
    )
    for kc in range(2):
        rec = p.planes[0, kc * w12.WREC : (kc + 1) * w12.WREC]
        for warp in range(8):
            for lane in range(0, 32, 3):
                a = _decode_lane(
                    _u32(rec, warp * 1024 + lane * 16),
                    _u32(rec, warp * 1024 + 512 + lane * 16),
                    _u32(rec, w12.QREG + warp * 512 + lane * 16),
                )
                g, t = lane // 4, lane % 4
                for j in range(4):
                    for r in range(4):
                        row = warp * 16 + g + 8 * (r & 1)
                        col = kc * 64 + j * 16 + 8 * (r >> 1) + 2 * t
                        assert a[j][r] & 0xFFFF == int(expect[row, col])
                        assert a[j][r] >> 16 == int(expect[row, col + 1])


# ----------------------------------------------------------- kernel algebra


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


@pytest.mark.parametrize("name", ["qkv_sliding", "o_global", "down"])
def test_kernel_algebra_is_lossless(name):
    _, k = GEMMA_SHAPES[name]
    w = gaussian(256, k, seed=21)
    w[0, :100] = 0.0
    w[1, 7] = 1e-20
    w[2, 9] = 300.0
    p = w12.encode(w)
    x = torch.randn(16, k, generator=torch.Generator().manual_seed(2)).bfloat16()
    ref = x.double() @ w.double().T
    assert _rel(w12.reference_gemm(x, p), ref) < 1e-12


@pytest.mark.parametrize("act", [w12.ACT_SILU, w12.ACT_GELU_TANH])
def test_act_epilogue_reference(act):
    w = gaussian(256, 512, seed=5)
    p = w12.encode(w, act=act)
    x = torch.randn(3, 512).bfloat16()
    gu = x.double() @ w.double().T
    g, u = gu[:, :128], gu[:, 128:]
    if act == w12.ACT_SILU:
        ref = g * torch.sigmoid(g) * u
    else:
        inner = math.sqrt(2 / math.pi) * (g + 0.044715 * g**3)
        ref = 0.5 * g * (1 + torch.tanh(inner)) * u
    assert _rel(w12.reference_gemm(x, p), ref) < 1e-12


# ----------------------------------------------------------- stream-K plan


@pytest.mark.parametrize("sms", [188, 170, 84, 7, 1])
@pytest.mark.parametrize("name", list(GEMMA_SHAPES))
def test_streamk_plan_invariants(name, sms):
    n, k = GEMMA_SHAPES[name]
    tiles, chunks = (n + 127) // 128, k // 64
    units = tiles * chunks
    plan = w12.W12Plan.for_shape(n, k, sms)
    ctas = plan.num_ctas
    assert 1 <= ctas <= units and plan.slots == min(tiles, ctas)
    owned_split = {}
    for p in range(ctas):
        u0, u1 = p * units // ctas, (p + 1) * units // ctas
        assert u1 > u0
    for t in range(tiles):
        first = w12.owner(t * chunks, ctas, units)
        last = w12.owner(t * chunks + chunks - 1, ctas, units)
        assert last - first + 1 <= plan.max_segments
        if last > first and tiles > ctas:  # owner-indexed slot: one split tile
            assert first not in owned_split
            owned_split[first] = t
    assert plan.workspace_floats() == plan.slots * plan.max_segments * 128 * 64


def test_smem_budget_fits_sm120():
    for nt, (ns, ncb) in w12.CONFIGS.items():
        assert (w12.WREC + nt * 128) % 1024 == 0
        assert w12.smem_bytes(nt) + 8 * (2 * ns + 2 * ncb) <= w12.SM120_MAX_DYN_SMEM
    assert [w12.pick_nt(m) for m in (1, 8, 9, 33, 64)] == [8, 8, 16, 40, 64]
    with pytest.raises(ValueError):
        w12.pick_nt(65)


# --------------------------------------------- routing with a fake backend


def _packed_from(planes, scale, rp, col, val, n, k, act, idx=None, esc=None):
    top = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.int32)
    empty_l = torch.empty(0, dtype=torch.int64)
    empty_b = torch.empty(0, dtype=torch.bfloat16)
    return w12.W12Packed(
        n=n, k=k, act=act, planes=planes, top=top, scale=scale, esc_rp=rp,
        esc_col=col, esc_val=val,
        esc_idx=empty_l if idx is None else idx,
        esc_bf16=empty_b if esc is None else esc,
        n_inexact=0, max_row_escapes=0,
    )  # fmt: skip


class FakeOps:
    """Pure-torch stand-in for torch.ops._C.w12_* that checks the contract."""

    def __init__(self):
        self.calls = []

    def w12_gemm(self, x, planes, scale, rp, col, val, out, ws, cnt, ctas, segs, n,
                 act, esc_seg, pdl):  # fmt: skip
        m, k = x.shape
        assert 1 <= m <= w12.MAX_M and x.stride(1) == 1 and x.stride(0) % 8 == 0
        assert out.shape == (m, n // 2 if act else n) and out.stride(1) == 1
        tiles = planes.shape[0]
        assert ws.numel() >= min(tiles, ctas) * segs * 128 * 64
        assert cnt.numel() >= min(tiles, ctas) and esc_seg == (act == 0)
        self.calls.append(("gemm", m, act))
        p = _packed_from(planes, scale, rp, col, val, n, k, act)
        out.copy_(w12.reference_gemm(x, p, torch.float32).bfloat16())

    def w12_decode(self, planes, scale, out, n, k, interleave):
        self.calls.append(("decode", n, k))
        empty_i = torch.zeros(planes.shape[0] * 128 + 1, dtype=torch.int32)
        p = _packed_from(
            planes, scale, empty_i, empty_i[:0], torch.empty(0), n, k, int(interleave)
        )
        out.reshape(-1)[: n * k].view(n, k).copy_(w12.decode(p))

    def w12_scatter(self, out, idx, val):
        self.calls.append(("scatter", idx.numel()))
        out[idx] = val

    def w12_embedding(self, out, ids, planes, scale, rp, col, esc, n):
        self.calls.append(("embedding", ids.numel()))
        k = out.shape[1]
        counts = (rp[1:] - rp[:-1]).long()
        rows = torch.repeat_interleave(torch.arange(rp.numel() - 1), counts)
        idx = rows * k + col.long()
        p = _packed_from(planes, scale, rp, col, torch.empty(0), n, k, 0, idx, esc)
        out.copy_(w12.decode(p)[ids])


def _act_and_mul_ref(out, gate_up, act):
    half = gate_up.shape[1] // 2
    gelu = torch.nn.functional.gelu(gate_up[:, :half].float(), approximate="tanh")
    out.copy_(gelu.bfloat16().float().mul(gate_up[:, half:].float()))


@pytest.fixture
def fake_ops(monkeypatch):
    ops = FakeOps()
    monkeypatch.setattr(w12, "_ops", lambda: ops)
    monkeypatch.setattr(w12, "act_and_mul", _act_and_mul_ref)
    w12._WORKSPACES.clear()
    yield ops
    w12._WORKSPACES.clear()


def _linear(x, p, plan, kmax):
    t = p.tensors()
    return w12.w12_linear_impl(
        x, t["planes"], t["scale"], t["esc_rp"], t["esc_col"], t["esc_val"],
        t["esc_idx"], t["esc_bf16"], p.n, p.act, plan.num_ctas, plan.max_segments,
        kmax, False,
    )  # fmt: skip


@pytest.mark.parametrize("m,kmax,expect", [
    (5, 64, [("gemm", 5, 0)]),
    (64, 64, [("gemm", 64, 0)]),
    (130, -1, [("gemm", 64, 0), ("gemm", 64, 0), ("gemm", 2, 0)]),
    (65, 64, [("decode", 256, 256), ("scatter", None)]),
])  # fmt: skip
def test_row_routing_chunks_and_fallback(fake_ops, m, kmax, expect):
    w = gaussian(256, 256, seed=m)
    w[0, 0] = 0.0
    p = w12.encode(w)
    plan = w12.W12Plan.for_shape(256, 256, 188)
    w12.reserve_workspace(torch.device("cpu"), plan, 256 * 256)
    x = torch.randn(m, 256).bfloat16()
    y = _linear(x, p, plan, kmax)
    calls = [(c[0], None) if c[0] == "scatter" else c for c in fake_ops.calls]
    assert calls == expect
    assert y.shape == (m, 256)
    if expect[0][0] == "decode":  # exact decode + torch.mm
        assert torch.equal(y, torch.mm(x, w.t()))
    else:
        assert _rel(y.double(), x.double() @ w.double().t()) < 1e-2


def test_fallback_needs_a_reserved_scratch(fake_ops):
    p = w12.encode(gaussian(128, 128, seed=1))
    plan = w12.W12Plan.for_shape(128, 128, 188)
    w12.reserve_workspace(torch.device("cpu"), plan)
    with pytest.raises(RuntimeError, match="scratch"):
        _linear(torch.randn(70, 128).bfloat16(), p, plan, 64)


def test_leading_dims_and_empty_input(fake_ops):
    p = w12.encode(gaussian(128, 128, seed=2))
    plan = w12.W12Plan.for_shape(128, 128, 188)
    w12.reserve_workspace(torch.device("cpu"), plan)
    assert _linear(torch.randn(2, 3, 128).bfloat16(), p, plan, 64).shape == (2, 3, 128)
    assert _linear(torch.randn(0, 128).bfloat16(), p, plan, 64).shape == (0, 128)
    assert fake_ops.calls == [("gemm", 6, 0)]


def test_embedding_gather_is_exact(fake_ops):
    w = gaussian(256, 128, seed=4)
    w[7, 3] = 0.0
    w[9, 1] = float("nan")
    p = w12.encode(w)
    t = p.tensors()
    ids = torch.tensor([[7, 9], [0, 255]])
    rows = w12.w12_embedding_impl(
        ids, t["planes"], t["scale"], t["esc_rp"], t["esc_col"], t["esc_bf16"], 256, 128
    )
    assert rows.shape == (2, 2, 128) and bits_eq(rows, w[ids])
