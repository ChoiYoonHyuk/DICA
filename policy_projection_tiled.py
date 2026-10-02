"""Optional row tiling of the INT8 preview; exact original-row kernel unchanged.

Set summary.preview_row_tile in {1,2,4,8} and summary.preview_num_warps in
{4,8}. These settings alter only the bounded preview, never the reference dot.
The default uses the preceding one-row preview specialization.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from policy_projection import (ProjectionSummary, build_projection_summary,
    dense_projection, _gamma, _hidden, _project, _row_preview)
from policy_projection_split import _single_hidden_norm, _vector_intervals, _resources


@triton.jit
def _tiled_int8_preview(Q, SCALE, H, OUT, D: tl.constexpr, VOCAB: tl.constexpr,
                       BLOCK: tl.constexpr, ROW_TILE: tl.constexpr):
    rows = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    columns = tl.arange(0, BLOCK)
    quantized = tl.load(Q + rows.to(tl.int64)[:, None] * D + columns[None, :],
        mask=(rows[:, None] < VOCAB) & (columns[None, :] < D), other=0).to(tl.float32)
    scale = tl.load(SCALE + rows, mask=rows < VOCAB, other=0)
    h = tl.load(H + columns, mask=columns < D, other=0).to(tl.float32)
    approximate_weight = quantized * scale[:, None]
    products = approximate_weight * h[None, :]
    centers = tl.sum(products, axis=1)
    tl.store(OUT + rows, centers, mask=rows < VOCAB)


def tiled_preview(hidden, summary, *, rows=None, num_warps=None, resource_report=None):
    """Preview component for an already admitted hidden vector, without sync."""
    rows = int(getattr(summary, 'preview_row_tile', 1) if rows is None else rows)
    warps = int(getattr(summary, 'preview_num_warps', summary.num_warps)
                if num_warps is None else num_warps)
    if rows not in (1, 2, 4, 8) or warps not in (4, 8):
        raise ValueError('Preview rows must be 1,2,4,8 and warps must be 4 or 8')
    h = hidden.reshape(-1)
    center = torch.empty(summary.vocabulary, dtype=torch.float32, device=h.device)
    if rows == 1:
        kernel = _row_preview[(summary.vocabulary,)](
            summary.quantized, summary.scales, h, center,
            D=summary.dimension, BLOCK=summary.block, num_warps=warps,
            enable_fp_fusion=False)
    else:
        kernel = _tiled_int8_preview[(triton.cdiv(summary.vocabulary, rows),)](
            summary.quantized, summary.scales, h, center,
            D=summary.dimension, VOCAB=summary.vocabulary, BLOCK=summary.block,
            ROW_TILE=rows, num_warps=warps, enable_fp_fusion=False)
    if resource_report is not None:
        resource_report.update(_resources(kernel), row_tile=rows, preview_warps=warps)
    return center


def split_intervals_with_validity_no_host_admission(hidden, summary, *, resource_report=None):
    """Device-only intervals plus int32 validity flag; emission must check it."""
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
    bounds = _vector_intervals[(triton.cdiv(summary.vocabulary, 256),)](
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
        implementation='tiled INT8 preview, one hidden norm, vectorized FP64 bounds',
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
           'split_intervals_with_validity_no_host_admission', 'tiled_preview',
           'screened_topk_projection', '_project']
