"""Vectorized projection bounds with one proved FP64 safety inflation.

The positive radius expression has fewer than 32 FP64 operations on any path.
The explicit FP64 factor 1+2^-30 covers their rounding and the final product.
The original upward hidden norm, gamma, tiny allowance and outward endpoints
are retained. See the independently reviewed numerical contract notes.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from policy_projection import (ProjectionSummary, build_projection_summary,
    dense_projection, _gamma, _hidden, _project)
from policy_projection_fused import _up64, _down64
from policy_projection_split import _single_hidden_norm, _resources
from policy_projection_tiled import tiled_preview


@triton.jit
def _inflated_vector_intervals(CENTER, NORM, DELTA, WMAX, TMAX, CAP, LOWER, UPPER,
        VOCAB: tl.constexpr, HAS_CAP: tl.constexpr, GAMMA32: tl.constexpr,
        TINY_GUARD: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    valid = row < VOCAB
    center = tl.load(CENTER + row, mask=valid, other=0).to(tl.float64)
    norm1 = tl.load(NORM)
    delta = tl.load(DELTA + row, mask=valid, other=0)
    wmax = tl.load(WMAX + row, mask=valid, other=0)
    tmax = tl.load(TMAX + row, mask=valid, other=0)
    gamma = tl.full((), GAMMA32, tl.float64)
    tiny = tl.full((), TINY_GUARD, tl.float64)
    r0 = (delta + gamma * (wmax + tmax)) * norm1 + tiny
    round_allowance = (tl.abs(center) + r0) * tl.full((), 2.0 ** -8, tl.float64)
    radius = r0 + round_allowance + tl.full((), 2.0 ** -126, tl.float64)
    # The factor MUST be FP64: FP32 would round it to exactly one.
    radius = radius * tl.full((), 1.0 + 2.0 ** -30, tl.float64)
    lower = _down64(center - radius)
    upper = _up64(center + radius)
    if HAS_CAP:
        lower_bf16 = lower.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding='rtne')
        upper_bf16 = upper.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding='rtne')
        lower_bits = lower_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 65535
        upper_bits = upper_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 65535
        lower = tl.load(CAP + lower_bits).to(tl.float64)
        upper = tl.load(CAP + upper_bits).to(tl.float64)
    tl.store(LOWER + row, lower, mask=valid)
    tl.store(UPPER + row, upper, mask=valid)


def split_intervals_with_validity_no_host_admission(hidden, summary, *, resource_report=None):
    """Device-only bounds; the returned validity flag must gate emission."""
    h = hidden.reshape(-1)
    if (h.numel() != summary.dimension or h.dtype != torch.bfloat16
            or h.device != summary.weight.device or not h.is_contiguous()):
        raise ValueError('Fixed-layout hidden vector differs from the summary')
    preview_resources = {} if resource_report is not None else None
    center = tiled_preview(h, summary, resource_report=preview_resources)
    norm = torch.empty(1, dtype=torch.float64, device=h.device)
    valid = torch.empty(1, dtype=torch.int32, device=h.device)
    lower = torch.empty(summary.vocabulary, dtype=torch.float64, device=h.device)
    upper = torch.empty_like(lower)
    gamma64 = _gamma(summary.block + 2, 53)
    denominator64 = math.nextafter(1.0 - gamma64, -math.inf)
    gamma32 = _gamma(2 * summary.block + 4, 24)
    denominator32 = math.nextafter(1.0 - gamma32, -math.inf)
    tiny_guard = math.nextafter(2 * (2 * summary.block + 4) * (2.0 ** -126) / denominator32, math.inf)
    norm_kernel = _single_hidden_norm[(1,)](h, norm, valid,
        D=summary.dimension, BLOCK=summary.block, DENOMINATOR64=denominator64,
        num_warps=summary.num_warps, enable_fp_fusion=False)
    bounds = _inflated_vector_intervals[(triton.cdiv(summary.vocabulary, 256),)](
        center, norm, summary.residual_max, summary.weight_max, summary.approximate_max,
        summary.scales if summary.cap_table is None else summary.cap_table, lower, upper,
        VOCAB=summary.vocabulary, HAS_CAP=summary.cap_table is not None,
        GAMMA32=gamma32, TINY_GUARD=tiny_guard, ROWS=256,
        num_warps=4, enable_fp_fusion=False)
    if resource_report is not None:
        resource_report.update(preview=preview_resources, norm=_resources(norm_kernel), bounds=_resources(bounds))
    return lower, upper, valid


def fused_intervals_no_host_admission(hidden, summary):
    lo, hi, _ = split_intervals_with_validity_no_host_admission(hidden, summary)
    return lo, hi


def projection_intervals(hidden, summary):
    h = _hidden(hidden, summary)
    lo, hi, _ = split_intervals_with_validity_no_host_admission(h, summary)
    return h, lo, hi


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
        implementation='INT8 preview, one hidden norm, FP64 positive-expression inflation',
        fp64_radius_inflation=1.0 + 2.0 ** -30,
        preview_row_tile=int(getattr(summary, 'preview_row_tile', 1)),
        preview_num_warps=int(getattr(summary, 'preview_num_warps', summary.num_warps)),
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
           'split_intervals_with_validity_no_host_admission',
           'screened_topk_projection', '_project']
