# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 SM120 projection kernels with offline measured row plans."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gemm(
    A,
    W,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AMS: tl.constexpr,
    WNS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = tl.cdiv(M, BM)
    pn = tl.cdiv(N, BN)
    group = pid // (GROUP * pn)
    first_m = group * GROUP
    count_m = tl.minimum(pm - first_m, GROUP)
    im = first_m + (pid % count_m)
    jn = (pid % (GROUP * pn)) // count_m
    rm = (im * BM + tl.arange(0, BM)) % M
    rn = (jn * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * AMS + rk[None, :]
    wp = W + rn[None, :] * WNS + rk[:, None]
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        a = tl.load(ap, mask=rk[None, :] + block * BK < K, other=0)
        w = tl.load(wp, mask=rk[:, None] + block * BK < K, other=0)
        acc = tl.dot(a, w, acc)
        ap += BK
        wp += BK
    rows = im * BM + tl.arange(0, BM)
    cols = jn * BN + tl.arange(0, BN)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _split_gemm(
    A,
    W,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AS: tl.constexpr,
    WS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    rk = split * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(
            A + rm[:, None] * AS + rk[None, :], (rm[:, None] < M) & (rk[None, :] < K), 0
        )
        w = tl.load(
            W + rn[None, :] * WS + rk[:, None], (rn[None, :] < N) & (rk[:, None] < K), 0
        )
        acc = tl.dot(a, w, acc)
        rk += BK * SPLIT
    tl.store(
        P + split * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _reduce_split(P, Out, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for s in tl.static_range(SPLIT):
        total += tl.load(P + s * SIZE + i, i < SIZE, 0)
    tl.store(Out + i, total, i < SIZE)


def split_gemm(a, w, cfg):
    bm, bn, bk, sp, nw, ns = cfg
    m, k = a.shape
    n = w.shape[0]
    partial = torch.empty((sp, m, n), device=a.device, dtype=torch.float32)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    _split_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn), sp)](
        a,
        w,
        partial,
        m,
        n,
        k,
        a.stride(0),
        w.stride(0),
        bm,
        bn,
        bk,
        sp,
        num_warps=nw,
        num_stages=ns,
    )
    _reduce_split[(triton.cdiv(m * n, 512),)](partial, out, m * n, sp, 512)
    return out


# (N, K, M): backend and launch parameters; other rows use the linear fallback.
SM120_PROJECTION_PLANS = {
    (16384, 2560, 32): ("gemm", (32, 128, 64, 4, 3, 4)),
    (2560, 6144, 32): ("split", (32, 64, 64, 8, 4, 3)),
    (1280, 2560, 32): ("split", (32, 64, 64, 8, 4, 3)),
    (2560, 640, 32): ("gemm", (16, 64, 64, 4, 3, 4)),
    (2560, 640, 64): ("gemm", (16, 64, 64, 4, 3, 4)),
    (16384, 2560, 8): ("split", (16, 64, 32, 4, 4, 3)),
    (16384, 2560, 16): ("gemm", (16, 128, 32, 4, 3, 4)),
    (2560, 6144, 8): ("split", (16, 64, 32, 8, 4, 3)),
    (2560, 6144, 16): ("split", (16, 64, 32, 8, 4, 3)),
    (1280, 2560, 16): ("split", (32, 64, 64, 8, 4, 3)),
    (2560, 640, 16): ("split", (16, 64, 32, 8, 4, 3)),
}


def sm120_projection_gemm(a, w):
    if (
        a.ndim != 2
        or w.ndim != 2
        or a.stride(1) != 1
        or not w.is_contiguous()
        or a.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or not a.is_cuda
        or a.device != w.device
        or a.shape[1] != w.shape[1]
    ):
        return None
    plan = SM120_PROJECTION_PLANS.get((w.shape[0], w.shape[1], a.shape[0]))
    if plan is None:
        return None
    backend, cfg = plan
    if backend == "split":
        return split_gemm(a, w, cfg)
    bm, bn, bk, nw, ns, group = cfg
    out = a.new_empty((a.shape[0], w.shape[0]))
    _gemm[(triton.cdiv(a.shape[0], bm) * triton.cdiv(w.shape[0], bn),)](
        a,
        w,
        out,
        a.shape[0],
        w.shape[0],
        a.shape[1],
        a.stride(0),
        w.stride(0),
        bm,
        bn,
        bk,
        group,
        num_warps=nw,
        num_stages=ns,
    )
    return out
