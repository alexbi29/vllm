# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NOTICE: the W12 record layout, window rule, plane split, stream-K plan and
# M routing are ported from SpecPilot (https://github.com/StargazerX0/SpecPilot,
# commit 78c42af, Apache-2.0), file
# settings/pro6000_c1_qwen38-27b_dflash2/files/python/sglang/kernels/ops/gemm/
# specpilot_d2_w12.py (`_frag`, `_encode_rows`, `W12Weight`, `CONFIGS`, the
# CTA count and segment helpers). Changes: inf/NaN always escape and never move
# the window; the window top is clamped to [15, 254]; rows without in-window
# weights use E = 127; weights with fp32-inexact escapes are rejected; split-K
# slots are min(T, P); vLLM workspace and custom-op plumbing.
"""W12: lossless 12-bit coding of BF16 weights and its SM120 GEMM wrapper.

A bf16 weight ``w = s * 2^(e-127) * 1.m`` of kernel row ``n`` is coded against
a per-row window top ``E_n``: the top of the 16-binade window that covers the
most normal, finite weights of the row (ties go to the highest top; ``E_n`` is
clamped to [15, 254] so that ``2^(E_n-127)`` is a normal fp32; a row without
any in-window weight uses 127). The code ``c = e - (E_n - 15)`` is in [0, 15].
Each weight is stored as a Q byte ``[c0 m6 m5 m4 | s c3 c2 c1]`` plus a B
nibble ``[m3..m0]``. The kernel rebuilds ``w~ = s * 2^(c-15) * 1.m`` in
registers and scales the fp32 accumulator by the exact power of two
``2^(E_n-127)``. Zeros, subnormals, inf/NaN and out-of-window weights are
escapes: their slot decodes as the placeholder ``2^-15`` and a per-row CSR
list ``(rp, col, val = w * 2^(127-E_n))`` corrects the accumulator exactly.
The true bf16 escape values are also kept, so decoding is bit-exact for every
bf16 pattern.

Records hold 128 rows x 64 k = 12 KB (an 8 KB Q region and a 4 KB B region)
in mma.sync m16n8k16 A-fragment lane order; ``planes`` is uint8
``[T, (K / 64) * 12288]`` with ``T = ceil(N / 128)``. With a fused
act-and-mul epilogue the gate_up rows are interleaved ``[64 gate | 64 up]``
per tile.

The GEMM runs one persistent stream-K launch per 64 rows of ``x``. Larger
inputs either repeat it per 64-row chunk or decode the weight into a shared
bf16 scratch and run ``torch.mm``; the caller chooses the crossover per shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from vllm.utils.torch_utils import direct_register_custom_op

RB = 128  # weight rows per tile
KC = 64  # k per record
QREG = 8192  # bytes of the Q region of a record
WREC = 12288  # bytes per record
PLACEHOLDER = 2.0**-15  # decoded w~ of an escape slot
ACT_NONE, ACT_SILU, ACT_GELU_TANH = 0, 1, 2
MAX_M = 64  # rows per kernel launch
# token tile NT -> (ring depth, correction buffers); mirrors the C++ dispatch
CONFIGS = {
    8: (3, 2),
    16: (3, 2),
    24: (3, 2),
    32: (2, 2),
    40: (2, 2),
    48: (2, 2),
    64: (2, 1),
}
SM120_MAX_DYN_SMEM = 101376
_E_MIN, _E_MAX = 15, 254


# ----------------------------------------------------------------- layout


def _frag(p: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """[R, K] -> [R/16, K/64, 32 lanes, 32 bytes] in A-fragment order.

    row = rg*16 + h*8 + g, col = kc*64 + j*16 + c8*8 + t*2 + b,
    lane = g*4 + t, byte = j*8 + c8*4 + h*2 + b.
    """
    return (
        p.view(rows // 16, 2, 8, k // 64, 4, 2, 4, 2)
        .permute(0, 3, 2, 6, 4, 5, 1, 7)
        .reshape(rows // 16, k // 64, 32, 32)
    )


def _unfrag(f: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    return (
        f.reshape(rows // 16, k // 64, 8, 4, 4, 2, 2, 2)
        .permute(0, 6, 2, 1, 4, 5, 3, 7)
        .reshape(rows, k)
    )


def interleave_perm(n: int, device=None) -> torch.Tensor:
    """Act-interleaved weights: kernel row r -> original row (gate | up)."""
    r = torch.arange(n, device=device)
    t, wr = r // RB, r % RB
    return torch.where(wr < 64, t * 64 + wr, n // 2 + t * 64 + wr - 64)


def _window_top(e: torch.Tensor) -> torch.Tensor:
    """Per-row window top E (int32 [R]) from biased exponents e [R, K]."""
    rows = e.shape[0]
    valid = ((e > 0) & (e < 255)).to(torch.int32)
    hist = torch.zeros(rows, 256, dtype=torch.int32, device=e.device)
    hist.scatter_add_(1, e.to(torch.int64), valid)
    cum = torch.cumsum(hist, dim=1)
    tops = torch.arange(_E_MIN, _E_MAX + 1, device=e.device)
    low = tops - 16
    below = torch.where(
        low >= 0, cum[:, low.clamp(min=0)], torch.zeros_like(cum[:, tops])
    )
    cover = cum[:, tops] - below
    best = cover.max(dim=1, keepdim=True).values
    # ties -> highest top (argmax returns the first maximum of the flip)
    tie = (cover == best).to(torch.int32).flip(1)
    top = tops[(tops.numel() - 1) - tie.argmax(dim=1)].to(torch.int32)
    return torch.where(best[:, 0] == 0, torch.full_like(top, 127), top)


def _encode_rows(wc: torch.Tensor):
    """bf16 [R, K] (R % 128 == 0, kernel row order) -> records, E, escapes."""
    rows, k = wc.shape
    u = wc.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    top = _window_top(e)
    c = e - (top[:, None] - 15)
    esc = (e == 0) | (e == 255) | (c < 0) | (c > 15)
    zero = torch.zeros_like(u)
    c = torch.where(esc, zero, c)
    s = torch.where(esc, zero, u >> 15)
    m = torch.where(esc, zero, u & 0x7F)
    low = ((c & 1) << 7) | m  # [c0 m6..m0]
    nb = (s << 3) | (c >> 1)  # [s c3 c2 c1]
    del u, e, c, s, m
    qf = _frag(((low & 0xF0) | nb).to(torch.uint8), rows, k)
    qb = qf.view(rows // 16, k // 64, 32, 2, 16).transpose(2, 3)
    qb = qb.reshape(rows // 16, k // 64, 1024)
    bf = _frag((low & 0x0F).to(torch.uint8), rows, k)
    bf = bf.view(rows // 16, k // 64, 32, 4, 2, 4)
    bb = (bf[..., 0, :] | (bf[..., 1, :] << 4)).reshape(rows // 16, k // 64, 512)
    tiles = rows // RB
    qb = qb.view(tiles, RB // 16, k // 64, 1024).transpose(1, 2)
    qb = qb.reshape(tiles, k // 64, 8 * 1024)
    bb = bb.view(tiles, RB // 16, k // 64, 512).transpose(1, 2)
    bb = bb.reshape(tiles, k // 64, 8 * 512)
    blocks = torch.cat([qb, bb], dim=2).reshape(tiles, (k // 64) * WREC)
    esc_rows, esc_cols = esc.nonzero(as_tuple=True)  # row-major order
    return blocks, top, esc_rows, esc_cols


# ----------------------------------------------------------------- packing


@dataclass
class W12Packed:
    """A W12-coded weight. Kernel row order; escapes in row-major CSR order."""

    n: int  # original output rows
    k: int
    act: int  # ACT_NONE, or rows interleaved [64 gate | 64 up] per tile
    planes: torch.Tensor  # uint8 [T, S * 12288]
    top: torch.Tensor  # int32 [T * 128] window tops (padded rows included)
    scale: torch.Tensor  # fp32 [T * 128] = 2^(E - 127)
    esc_rp: torch.Tensor  # int32 [T * 128 + 1] CSR row pointer
    esc_col: torch.Tensor  # int32 [n_esc]
    esc_val: torch.Tensor  # fp32 [n_esc] = w * 2^(127 - E_row)
    esc_idx: torch.Tensor  # int64 [n_esc] flat index into the ORIGINAL [N, K]
    esc_bf16: torch.Tensor  # bf16 [n_esc] exact escape values
    n_inexact: int  # finite escapes whose val is not exact in fp32
    max_row_escapes: int
    breakdown: dict = field(default_factory=dict)

    @property
    def tiles(self) -> int:
        return self.planes.shape[0]

    @property
    def n_out(self) -> int:
        return self.n // 2 if self.act else self.n

    @property
    def n_escapes(self) -> int:
        return int(self.esc_col.numel())

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "planes": self.planes,
            "scale": self.scale,
            "esc_rp": self.esc_rp,
            "esc_col": self.esc_col,
            "esc_val": self.esc_val,
            "esc_idx": self.esc_idx,
            "esc_bf16": self.esc_bf16,
        }

    def stream_bytes(self) -> int:
        """Bytes one GEMM call reads (records, scales and the kernel CSR)."""
        names = ("planes", "scale", "esc_rp", "esc_col", "esc_val")
        t = self.tensors()
        return sum(t[k].numel() * t[k].element_size() for k in names)

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors().values())

    def stats(self) -> dict:
        weights = self.n * self.k
        return {
            "n": self.n,
            "k": self.k,
            "act": self.act,
            "bits_per_weight_stream": 8.0 * self.stream_bytes() / weights,
            "bits_per_weight_resident": 8.0 * self.resident_bytes() / weights,
            "escapes": self.n_escapes,
            "escape_fraction": self.n_escapes / weights,
            "max_row_escapes": self.max_row_escapes,
            "inexact": self.n_inexact,
            "breakdown": dict(self.breakdown),
        }


def encode(
    w: torch.Tensor, act: int = ACT_NONE, chunk_rows: int = 2048, check: bool = True
) -> W12Packed:
    """Bit-exact W12 coding of a bf16 [N, K] weight (K % 64 == 0) on w's device.

    ``check`` asserts that the pure-torch decode reproduces ``w`` bitwise.
    """
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise ValueError(f"W12 needs a 2-D bf16 weight, got {w.dtype} {w.shape}")
    n, k = w.shape
    if k % KC:
        raise ValueError(f"W12 needs K % 64 == 0, got K={k}")
    if chunk_rows % RB:
        raise ValueError("chunk_rows must be a multiple of 128")
    if act and n % (2 * 64):
        raise ValueError(f"fused act needs N = 2I with I % 64 == 0, got N={n}")
    w = w.contiguous()
    dev = w.device
    if act:
        half = n // 2
        wi = w.view(2, half // 64, 64, k).transpose(0, 1).reshape(n, k)
    else:
        wi = w
    tiles = (n + RB - 1) // RB
    planes = torch.empty((tiles, (k // KC) * WREC), dtype=torch.uint8, device=dev)
    top_all = torch.empty(tiles * RB, dtype=torch.int32, device=dev)
    rows_l, cols_l = [], []
    for r0 in range(0, tiles * RB, chunk_rows):
        r1 = min(tiles * RB, r0 + chunk_rows)
        wc = wi[r0 : min(r1, n)]
        if wc.shape[0] < r1 - r0:  # padding rows are 1.0 and never escape
            pad = torch.ones(r1 - r0 - wc.shape[0], k, dtype=w.dtype, device=dev)
            wc = torch.cat([wc, pad])
        blocks, top, rows, cols = _encode_rows(wc.contiguous())
        planes[r0 // RB : r1 // RB] = blocks
        top_all[r0:r1] = top
        rows_l.append(rows + r0)
        cols_l.append(cols)
    rows = torch.cat(rows_l)
    cols = torch.cat(cols_l)
    if not bool((rows < n).all()):
        raise RuntimeError("W12 padding rows must not escape")
    scale = torch.pow(2.0, (top_all - 127).double()).float()
    orig = interleave_perm(n, dev)[rows] if act else rows
    wv = w[orig, cols]
    w64 = wv.double()
    val64 = w64 * torch.pow(2.0, (127 - top_all[rows]).double())
    val = val64.float()
    finite = torch.isfinite(w64)
    n_inexact = int(((val.double() != val64) & finite).sum()) if val.numel() else 0
    counts = torch.bincount(rows, minlength=tiles * RB)
    rp = torch.zeros(tiles * RB + 1, dtype=torch.int32, device=dev)
    rp[1:] = torch.cumsum(counts, 0).to(torch.int32)
    ub = wv.view(torch.int16).to(torch.int32) & 0xFFFF
    ee = (ub >> 7) & 0xFF
    is_zero = (ub & 0x7FFF) == 0
    cc = ee - (top_all[rows] - 15)
    normal = (ee > 0) & (ee < 255)
    breakdown = {
        "zero": int(is_zero.sum()),
        "subnormal": int(((ee == 0) & ~is_zero).sum()),
        "nonfinite": int((ee == 255).sum()),
        "below_window": int((normal & (cc < 0)).sum()),
        "above_window": int((normal & (cc > 15)).sum()),
    }
    packed = W12Packed(
        n=n,
        k=k,
        act=int(act),
        planes=planes,
        top=top_all,
        scale=scale,
        esc_rp=rp,
        esc_col=cols.to(torch.int32).contiguous(),
        esc_val=val.contiguous(),
        esc_idx=(orig * k + cols).to(torch.int64).contiguous(),
        esc_bf16=wv.contiguous(),
        n_inexact=n_inexact,
        max_row_escapes=int(counts.max()) if rows.numel() else 0,
        breakdown=breakdown,
    )
    if check:
        d = decode(packed)
        if not torch.equal(d.view(torch.int16), w.view(torch.int16)):
            bad = int((d.view(torch.int16) != w.view(torch.int16)).sum())
            raise RuntimeError(
                f"W12 roundtrip not bit-exact: {bad} mismatches for {tuple(w.shape)}"
            )
    return packed


# --------------------------------------------------------- reference decode


def _record_bytes(p: W12Packed):
    """-> (Q bytes, B nibbles) uint8 [T*128, K] in kernel row order."""
    tiles, s, k = p.tiles, p.k // KC, p.k
    rows = tiles * RB
    rec = p.planes.view(tiles, s, WREC)
    q = rec[:, :, :QREG].reshape(tiles, s, 8, 2, 32, 16).transpose(3, 4)
    q = q.reshape(tiles, s, 8, 32, 32).permute(0, 2, 1, 3, 4)
    q = q.reshape(rows // 16, s, 32, 32)
    b = rec[:, :, QREG:].reshape(tiles, s, 8, 32, 4, 4).permute(0, 2, 1, 3, 4, 5)
    b = b.reshape(rows // 16, s, 32, 4, 4)
    nib = torch.stack([b & 0x0F, b >> 4], dim=-2).reshape(rows // 16, s, 32, 32)
    return _unfrag(q, rows, k), _unfrag(nib, rows, k)


def _wtilde_bits(p: W12Packed) -> torch.Tensor:
    """int32 [T*128, K]: bf16 bits of w~ as the kernel's decode builds them."""
    qb, nb = _record_bytes(p)
    qb = qb.to(torch.int32)
    lo = (qb & 0xF0) | nb.to(torch.int32)
    hi = ((qb << 4) & 0x80) | (qb & 0x07) | 0x38
    return (hi << 8) | lo


def _bits_to_bf16(bits: torch.Tensor) -> torch.Tensor:
    bits = bits & 0xFFFF
    signed = torch.where(bits >= 0x8000, bits - 0x10000, bits)
    return signed.to(torch.int16).view(torch.bfloat16)


def decode_wtilde(p: W12Packed) -> torch.Tensor:
    """fp32 [T*128, K] w~ (kernel row order): what the mma consumes."""
    return _bits_to_bf16(_wtilde_bits(p)).float()


def decode(p: W12Packed) -> torch.Tensor:
    """Pure-torch decode -> bf16 [N, K] in the original row order."""
    bits = _wtilde_bits(p)
    bits = (bits + ((p.top.to(torch.int32)[:, None] - 127) << 7)) & 0xFFFF
    w = _bits_to_bf16(bits)
    if p.act:
        out = torch.empty((p.n, p.k), dtype=torch.bfloat16, device=w.device)
        out[interleave_perm(p.n, w.device)] = w[: p.n]
    else:
        out = w[: p.n].clone()
    if p.n_escapes:
        out.view(-1)[p.esc_idx] = p.esc_bf16
    return out


def reference_gemm(
    x: torch.Tensor, p: W12Packed, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """The kernel algebra (w~ GEMM + CSR correction + row scale [+ act]).

    With dtype=float64 this equals x @ W^T up to fp64 rounding iff the weight
    has no inexact escapes.
    """
    xd = x.to(dtype)
    rows = p.tiles * RB
    acc = xd @ decode_wtilde(p).to(dtype).T
    if p.n_escapes:
        counts = (p.esc_rp[1:] - p.esc_rp[:-1]).long()
        er = torch.repeat_interleave(torch.arange(rows, device=x.device), counts)
        cols = p.esc_col.long()
        contrib = xd[:, cols] * p.esc_val.to(dtype)[None, :]
        contrib = contrib - xd[:, cols] * PLACEHOLDER
        acc.index_add_(1, er, contrib)
    y = acc * p.scale.to(dtype)[None, :]
    if not p.act:
        return y[:, : p.n]
    yo = torch.empty((x.shape[0], p.n), dtype=dtype, device=x.device)
    yo[:, interleave_perm(p.n, x.device)] = y[:, : p.n]
    g, u = yo[:, : p.n // 2], yo[:, p.n // 2 :]
    if p.act == ACT_SILU:
        return g * torch.sigmoid(g) * u
    inner = math.sqrt(2.0 / math.pi) * (g + 0.044715 * g * g * g)
    return 0.5 * g * (1.0 + torch.tanh(inner)) * u


# ---------------------------------------------------------- stream-K plan


def pick_nt(m: int) -> int:
    for nt in sorted(CONFIGS):
        if m <= nt:
            return nt
    raise ValueError(f"W12: M={m} > {MAX_M}")


def smem_bytes(nt: int) -> int:
    ns, ncb = CONFIGS[nt]
    return ns * (WREC + nt * 128) + ncb * RB * nt * 4 + 64 * nt * 4 + 1024


def default_num_ctas(tiles: int, num_sms: int) -> int:
    """A multiple of T when T <= #SMs (equal splits per tile), else #SMs."""
    return tiles * max(1, num_sms // tiles) if tiles <= num_sms else num_sms


def owner(u: int, ctas: int, units: int) -> int:
    """CTA holding unit u when CTA p holds [p*U/P, (p+1)*U/P)."""
    return ((u + 1) * ctas - 1) // units


def max_segments(tiles: int, chunks: int, ctas: int) -> int:
    units = tiles * chunks
    return max(
        owner(t * chunks + chunks - 1, ctas, units) - owner(t * chunks, ctas, units) + 1
        for t in range(tiles)
    )


@dataclass(frozen=True)
class W12Plan:
    num_ctas: int
    max_segments: int
    slots: int  # split-K partial / counter slots: min(T, P)

    @classmethod
    def for_shape(cls, n: int, k: int, num_sms: int) -> W12Plan:
        tiles = (n + RB - 1) // RB
        chunks = k // KC
        ctas = min(default_num_ctas(tiles, num_sms), tiles * chunks)
        return cls(ctas, max_segments(tiles, chunks, ctas), min(tiles, ctas))

    def workspace_floats(self) -> int:
        return self.slots * self.max_segments * RB * MAX_M


# ---------------------------------------------------------------- runtime


def _ops():
    return torch.ops._C


def kernels_available() -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key("_C::w12_gemm", "CUDA")
    except RuntimeError:
        return False


class W12Workspace:
    """Per-device buffers shared by every W12 weight (stream-ordered use)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.split_k = torch.empty(1, dtype=torch.float32, device=device)
        self.counters = torch.zeros(1, dtype=torch.int32, device=device)
        self.scratch = torch.empty(1, dtype=torch.bfloat16, device=device)

    def reserve(self, split_k: int, counters: int, scratch: int) -> None:
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("W12 workspace cannot grow during graph capture")
        if split_k > self.split_k.numel():
            self.split_k = torch.empty(split_k, dtype=torch.float32, device=self.device)
        if counters > self.counters.numel():
            self.counters = torch.zeros(counters, dtype=torch.int32, device=self.device)
        if scratch > self.scratch.numel():
            self.scratch = torch.empty(
                scratch, dtype=torch.bfloat16, device=self.device
            )

    def nbytes(self) -> int:
        bufs = (self.split_k, self.counters, self.scratch)
        return sum(t.numel() * t.element_size() for t in bufs)


_WORKSPACES: dict[torch.device, W12Workspace] = {}


def get_workspace(device: torch.device) -> W12Workspace:
    ws = _WORKSPACES.get(device)
    if ws is None:
        ws = _WORKSPACES[device] = W12Workspace(device)
    return ws


def reserve_workspace(
    device: torch.device, plan: W12Plan, scratch_elems: int = 0
) -> W12Workspace:
    ws = get_workspace(device)
    ws.reserve(plan.workspace_floats(), plan.slots, scratch_elems)
    return ws


def _aligned_rows(x2: torch.Tensor) -> torch.Tensor:
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        return x2.clone(memory_format=torch.contiguous_format)
    return x2


def _decode_into(
    out: torch.Tensor,
    planes: torch.Tensor,
    scale: torch.Tensor,
    esc_idx: torch.Tensor,
    esc_bf16: torch.Tensor,
    n: int,
    k: int,
    act: int,
) -> torch.Tensor:
    ops = _ops()
    ops.w12_decode(planes, scale, out, n, k, bool(act))
    ops.w12_scatter(out.view(-1), esc_idx, esc_bf16)
    return out


def w12_linear_impl(
    x: torch.Tensor,
    planes: torch.Tensor,
    scale: torch.Tensor,
    esc_rp: torch.Tensor,
    esc_col: torch.Tensor,
    esc_val: torch.Tensor,
    esc_idx: torch.Tensor,
    esc_bf16: torch.Tensor,
    n: int,
    act: int,
    num_ctas: int,
    max_segments: int,
    kernel_max_m: int,
    pdl: bool,
) -> torch.Tensor:
    """Y = x @ W^T (act != 0: act(gate) * up) from a W12 weight.

    Rows up to ``kernel_max_m`` run the W12 kernel in 64-row launches; larger
    inputs decode W into the shared scratch and use torch.mm. A negative
    ``kernel_max_m`` means "always the kernel" (no whole-weight decode).
    """
    k = x.shape[-1]
    n_out = n // 2 if act else n
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    out = torch.empty((m, n_out), dtype=x.dtype, device=x.device)
    if m == 0:
        return out.view(*lead, n_out)
    ws = get_workspace(x.device)
    if kernel_max_m < 0 or m <= kernel_max_m:
        x2 = _aligned_rows(x2)
        ops = _ops()
        for i in range(0, m, MAX_M):
            ops.w12_gemm(
                x2[i : i + MAX_M],
                planes,
                scale,
                esc_rp,
                esc_col,
                esc_val,
                out[i : i + MAX_M],
                ws.split_k,
                ws.counters,
                num_ctas,
                max_segments,
                n,
                act,
                act == ACT_NONE,
                pdl,
            )
        return out.view(*lead, n_out)
    if ws.scratch.numel() < n * k:
        raise RuntimeError(
            f"W12 decode scratch holds {ws.scratch.numel()} < {n * k} elements"
        )
    wd = ws.scratch[: n * k].view(n, k)
    _decode_into(wd, planes, scale, esc_idx, esc_bf16, n, k, act)
    if act == ACT_NONE:
        torch.mm(x2, wd.t(), out=out)
    else:
        act_and_mul(out, torch.mm(x2, wd.t()), act)
    return out.view(*lead, n_out)


def act_and_mul(out: torch.Tensor, gate_up: torch.Tensor, act: int) -> None:
    """VLLM's act-and-mul kernels (the fused epilogue matches their rounding)."""
    if act == ACT_GELU_TANH:
        torch.ops._C.gelu_tanh_and_mul(out, gate_up)
    elif act == ACT_SILU:
        torch.ops._C.silu_and_mul(out, gate_up)
    else:
        raise ValueError(f"unknown W12 activation {act}")


def _w12_linear_fake(
    x: torch.Tensor,
    planes: torch.Tensor,
    scale: torch.Tensor,
    esc_rp: torch.Tensor,
    esc_col: torch.Tensor,
    esc_val: torch.Tensor,
    esc_idx: torch.Tensor,
    esc_bf16: torch.Tensor,
    n: int,
    act: int,
    num_ctas: int,
    max_segments: int,
    kernel_max_m: int,
    pdl: bool,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n // 2 if act else n))


def w12_embedding_impl(
    ids: torch.Tensor,
    planes: torch.Tensor,
    scale: torch.Tensor,
    esc_rp: torch.Tensor,
    esc_col: torch.Tensor,
    esc_bf16: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    flat = ids.reshape(-1).to(torch.int64).contiguous()
    out = torch.empty((flat.numel(), k), dtype=torch.bfloat16, device=ids.device)
    if flat.numel():
        _ops().w12_embedding(out, flat, planes, scale, esc_rp, esc_col, esc_bf16, n)
    return out.view(*ids.shape, k)


def _w12_embedding_fake(
    ids: torch.Tensor,
    planes: torch.Tensor,
    scale: torch.Tensor,
    esc_rp: torch.Tensor,
    esc_col: torch.Tensor,
    esc_bf16: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    return torch.empty((*ids.shape, k), dtype=torch.bfloat16, device=ids.device)


direct_register_custom_op(
    op_name="w12_linear",
    op_func=w12_linear_impl,
    fake_impl=_w12_linear_fake,
)
direct_register_custom_op(
    op_name="w12_embedding",
    op_func=w12_embedding_impl,
    fake_impl=_w12_embedding_fake,
)
