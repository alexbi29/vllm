# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time FP8 head fitting with the original per-channel runtime layout."""

import torch


@torch.no_grad()
def fit_fp8_head_scales(
    weight: torch.Tensor, quantized: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine rows in place, retaining the baseline if FP64 weight SSE increases.

    This bounds weight reconstruction error, not downstream logit/probability
    error. No calibration data or additional forward operations are needed.
    """
    info = torch.finfo(quantized.dtype)
    chunk_rows = max(1, min(1024, 1024 * 1024 // weight.shape[-1]))
    for start in range(0, weight.shape[0], chunk_rows):
        rows = slice(start, start + chunk_rows)
        w = weight[rows].float()
        baseline_q, baseline_s = quantized[rows], scale[rows]
        best_q, best_s = baseline_q, baseline_s
        best_error = (best_q.float() * best_s - w).square().sum(-1, keepdim=True)
        for step in range(1, 16):
            candidate_s = baseline_s * 2 ** (step / 16)
            candidate_q = (
                (w / candidate_s).clamp(info.min, info.max).to(quantized.dtype)
            )
            error = (
                (candidate_q.float() * candidate_s - w).square().sum(-1, keepdim=True)
            )
            better = error < best_error
            best_q = torch.where(
                better, candidate_q.view(torch.uint8), best_q.view(torch.uint8)
            ).view(quantized.dtype)
            best_s = torch.where(better, candidate_s, best_s)
            best_error = torch.minimum(error, best_error)
        for _ in range(4):
            q = best_q.float()
            candidate_s = (
                (w * q).sum(-1, keepdim=True)
                / q.square().sum(-1, keepdim=True).clamp_min(1e-30)
            ).clamp_min(baseline_s)
            candidate_q = (
                (w / candidate_s).clamp(info.min, info.max).to(quantized.dtype)
            )
            error = (
                (candidate_q.float() * candidate_s - w).square().sum(-1, keepdim=True)
            )
            better = error < best_error
            best_q = torch.where(
                better, candidate_q.view(torch.uint8), best_q.view(torch.uint8)
            ).view(quantized.dtype)
            best_s = torch.where(better, candidate_s, best_s)
            best_error = torch.minimum(error, best_error)
        w64 = w.double()
        before = (
            (baseline_q.double() * baseline_s.double() - w64)
            .square()
            .sum(-1, keepdim=True)
        )
        after = (best_q.double() * best_s.double() - w64).square().sum(-1, keepdim=True)
        keep = after <= before
        quantized[rows] = torch.where(
            keep, best_q.view(torch.uint8), baseline_q.view(torch.uint8)
        ).view(quantized.dtype)
        scale[rows] = torch.where(keep, best_s, baseline_s)
    return quantized, scale
