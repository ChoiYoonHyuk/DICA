"""Fused preview and outward bounds for the explicit row projection reference.

The original policy_projection implementation remains available for comparison.
This backend uses its exact same projection function and immutable summary.
No device work runs merely by importing this file.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from policy_projection import (ProjectionSummary, build_projection_summary,
                               dense_projection, _gamma, _hidden, _project)


@triton.jit
def _up64(value):
    return libdevice.nextafter(value, tl.full((), float('inf'), tl.float64))


@triton.jit
def _down64(value):
    return libdevice.nextafter(value, tl.full((), -float('inf'), tl.float64))


@triton.jit
def _preview_and_intervals(Q, SCALE, DELTA, WMAX, TMAX, H, CAP, LOWER, UPPER,
        D: tl.constexpr, BLOCK: tl.constexpr, HAS_CAP: tl.constexpr,
        GAMMA32: tl.constexpr, DENOMINATOR64: tl.constexpr, TINY_GUARD: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    q = tl.load(Q + row * D + columns, mask=columns < D, other=0).to(tl.float32)
    h = tl.load(H + columns, mask=columns < D, other=0).to(tl.float32)
    scale = tl.load(SCALE + row)
    approximate_weight = q * scale
    center = tl.sum(approximate_weight * h, axis=0).to(tl.float64)

    # Reuse the admitted input vector. Its conversion to FP64 is exact. This
    # reduction may round, so its positive sum receives the same gamma bound
    # as the nonfused implementation before any row radius is constructed.
    norm_hat = tl.sum(tl.abs(h.to(tl.float64)), axis=0)
    norm1 = _up64(norm_hat / tl.full((), DENOMINATOR64, tl.float64))
    delta = tl.load(DELTA + row)
    wmax = tl.load(WMAX + row)
    tmax = tl.load(TMAX + row)
    residual_error = _up64(delta * norm1)
    magnitudes = _up64(wmax + tmax)
    arithmetic_error = _up64(_up64(magnitudes * norm1) * tl.full((), GAMMA32, tl.float64))
    radius = _up64(_up64(residual_error + arithmetic_error) + tl.full((), TINY_GUARD, tl.float64))
    rounded = _up64(_up64(_up64(tl.abs(center) + radius) * tl.full((), 2.0 ** -8, tl.float64))
                    + tl.full((), 2.0 ** -126, tl.float64))
    radius = _up64(radius + rounded)
    lower = _down64(center - radius)
    upper = _up64(center + radius)
    if HAS_CAP:
        # This conversion chain is monotone and fixes every BF16 value. The
        # interval already encloses the rounded BF16 reference; direct BF16
        # rounding of the unrounded FP32 accumulator is not assumed here.
        lower_bf16 = lower.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding='rtne')
        upper_bf16 = upper.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding='rtne')
        lower_bits = lower_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 65535
        upper_bits = upper_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 65535
        lower = tl.load(CAP + lower_bits).to(tl.float64)
        upper = tl.load(CAP + upper_bits).to(tl.float64)
    tl.store(LOWER + row, lower)
    tl.store(UPPER + row, upper)


def fused_intervals_no_host_admission(hidden, summary):
    """Low-level device-only bounds for an ALREADY ADMITTED fixed-layout h.

    The caller must establish the same finite, normal-or-zero, <=2^20 BF16
    preconditions and immutable weight contract as _hidden. In a recorded GPU
    pipeline, an independent device validity flag must gate emission. This
    function deliberately performs no host boolean conversion or tensor-value
    read. It does not authorize emission for an unchecked input.
    """
    h = hidden.reshape(-1)
    if (h.numel() != summary.dimension or h.dtype != torch.bfloat16
            or h.device != summary.weight.device or not h.is_contiguous()):
        raise ValueError('Fixed-layout hidden vector differs from the summary')
    lower = torch.empty(summary.vocabulary, dtype=torch.float64, device=h.device)
    upper = torch.empty_like(lower)
    gamma64 = _gamma(summary.block + 2, 53)
    denominator64 = math.nextafter(1.0 - gamma64, -math.inf)
    gamma32 = _gamma(2 * summary.block + 4, 24)
    denominator32 = math.nextafter(1.0 - gamma32, -math.inf)
    tiny_guard = math.nextafter(2 * (2 * summary.block + 4) * (2.0 ** -126) / denominator32, math.inf)
    _preview_and_intervals[(summary.vocabulary,)](
        summary.quantized, summary.scales, summary.residual_max, summary.weight_max,
        summary.approximate_max, h, summary.scales if summary.cap_table is None else summary.cap_table,
        lower, upper, D=summary.dimension, BLOCK=summary.block,
        HAS_CAP=summary.cap_table is not None, GAMMA32=gamma32,
        DENOMINATOR64=denominator64, TINY_GUARD=tiny_guard,
        num_warps=summary.num_warps, enable_fp_fusion=False)
    return lower, upper


def projection_intervals(hidden, summary):
    h = _hidden(hidden, summary)
    lower, upper = fused_intervals_no_host_admission(h, summary)
    return h, lower, upper


def screened_topk_projection(hidden, summary, *, k=50):
    if type(k) is not int or not 1 <= k <= summary.vocabulary:
        raise ValueError('k must be an integer in [1,vocabulary]')
    h, lower, upper = projection_intervals(hidden, summary)
    threshold = torch.topk(lower, k, largest=True, sorted=False).values.min()
    ids = torch.nonzero(upper >= threshold, as_tuple=False).reshape(-1).contiguous()
    if ids.numel() < k:
        raise ArithmeticError('Certified candidate set contains fewer than k rows')
    logits = _project(h, summary, ids)
    return ids, logits, dict(
        implementation='fused preview and FP64 outward intervals',
        selected_rows=int(ids.numel()), vocabulary=summary.vocabulary,
        original_weight_row_read_fraction=ids.numel() / summary.vocabulary,
        quantized_weight_bytes=summary.quantized.numel(),
        original_weight_bytes_read=ids.numel() * summary.dimension * 2,
        lower_threshold=float(threshold), fp32_error_gamma=_gamma(2 * summary.block + 4, 24),
        row_kernel_block=summary.block, row_kernel_warps=summary.num_warps,
        candidate_rule='upper >= kth largest lower, retaining all ties',
        full_vocabulary_summary_scan=True, reference=summary.reference,
        original_cublas_reference_preserved=False, setup_charged_per_query=False,
        setup_seconds=summary.setup_seconds, summary_storage_bytes=summary.storage_bytes)


__all__ = ['ProjectionSummary', 'build_projection_summary', 'dense_projection',
           'projection_intervals', 'fused_intervals_no_host_admission',
           'screened_topk_projection', '_project']
