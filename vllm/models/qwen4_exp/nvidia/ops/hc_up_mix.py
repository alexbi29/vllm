# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse HC up projection with BF16 gate rounding and four-stream mixing."""

from vllm.triton_utils import tl, triton


@triton.jit
def _hc_up_mix(
    A,
    W,
    X,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    WS: tl.constexpr,
    XS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    c0 = tl.full((BM, BN), 0, tl.float32)
    c1 = tl.full((BM, BN), 0, tl.float32)
    c2 = tl.full((BM, BN), 0, tl.float32)
    c3 = tl.full((BM, BN), 0, tl.float32)
    for i in range(tl.cdiv(K, BK)):
        kk = i * BK + rk
        a = tl.load(
            A + rm[:, None] * AS + kk[None, :], (rm[:, None] < M) & (kk[None, :] < K), 0
        )
        mask = (rn[None, :] < H) & (kk[:, None] < K)
        w0 = tl.load(W + rn[None, :] * WS + kk[:, None], mask, 0)
        w1 = tl.load(W + (rn[None, :] + H) * WS + kk[:, None], mask, 0)
        w2 = tl.load(W + (rn[None, :] + 2 * H) * WS + kk[:, None], mask, 0)
        w3 = tl.load(W + (rn[None, :] + 3 * H) * WS + kk[:, None], mask, 0)
        c0 = tl.dot(a, w0, c0)
        c1 = tl.dot(a, w1, c1)
        c2 = tl.dot(a, w2, c2)
        c3 = tl.dot(a, w3, c3)
    # Preserve the materialized BF16 gate boundary before sigmoid.
    g0 = c0.to(tl.bfloat16).to(tl.float32)
    g1 = c1.to(tl.bfloat16).to(tl.float32)
    g2 = c2.to(tl.bfloat16).to(tl.float32)
    g3 = c3.to(tl.bfloat16).to(tl.float32)
    mask = (rm[:, None] < M) & (rn[None, :] < H)
    x0 = tl.load(X + rm[:, None] * XS + rn[None, :], mask, 0).to(tl.float32)
    x1 = tl.load(X + rm[:, None] * XS + rn[None, :] + H, mask, 0).to(tl.float32)
    x2 = tl.load(X + rm[:, None] * XS + rn[None, :] + 2 * H, mask, 0).to(tl.float32)
    x3 = tl.load(X + rm[:, None] * XS + rn[None, :] + 3 * H, mask, 0).to(tl.float32)
    result = tl.sigmoid(g0) * x0
    result += tl.sigmoid(g1) * x1
    result += tl.sigmoid(g2) * x2
    result += tl.sigmoid(g3) * x3
    tl.store(Y + rm[:, None] * H + rn[None, :], result * 0.25, mask)


def hc_up_mix(a, w, x):
    m, h = a.shape[0], w.shape[0] // 4
    out = a.new_empty((m, h))
    _hc_up_mix[(triton.cdiv(m, 64), triton.cdiv(h, 128))](
        a,
        w,
        x,
        out,
        m,
        h,
        a.shape[1],
        a.stride(0),
        w.stride(0),
        x.stride(0),
        64,
        128,
        32,
        num_warps=8,
        num_stages=3,
    )
    return out
