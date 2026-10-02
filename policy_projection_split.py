"""Three device kernels: INT8 preview, one hidden norm, vectorized bounds.

The explicit exact-row reference and all numerical bounds are unchanged.
Kernel resource reporting is optional and performs no additional device work.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from policy_projection import (ProjectionSummary, build_projection_summary,
    dense_projection, _gamma, _hidden, _project, _row_preview)
from policy_projection_fused import _up64, _down64


@triton.jit
def _single_hidden_norm(H, NORM, VALID, D: tl.constexpr, BLOCK: tl.constexpr,
                        DENOMINATOR64: tl.constexpr):
    columns = tl.arange(0, BLOCK)
    raw = tl.load(H + columns, mask=columns < D, other=0)
    magnitude_bits = raw.to(tl.int16, bitcast=True).to(tl.int32) & 32767
    # Bit admission avoids interpreting a flushed subnormal as an admitted zero.
    good = (magnitude_bits == 0) | ((magnitude_bits >= 128) & (magnitude_bits <= 0x4980))
    all_good = tl.sum((~good).to(tl.int32), axis=0) == 0
    norm_hat = tl.sum(tl.abs(raw.to(tl.float64)), axis=0)
    norm = _up64(norm_hat / tl.full((), DENOMINATOR64, tl.float64))
    tl.store(NORM, norm)
    tl.store(VALID, all_good.to(tl.int32))


@triton.jit
def _vector_intervals(CENTER, NORM, DELTA, WMAX, TMAX, CAP, LOWER, UPPER,
        VOCAB: tl.constexpr, HAS_CAP: tl.constexpr, GAMMA32: tl.constexpr,
        TINY_GUARD: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    valid = row < VOCAB
    center = tl.load(CENTER + row, mask=valid, other=0).to(tl.float64)
    norm1 = tl.load(NORM)
    delta = tl.load(DELTA + row, mask=valid, other=0)
    wmax = tl.load(WMAX + row, mask=valid, other=0)
    tmax = tl.load(TMAX + row, mask=valid, other=0)
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
        lower_bf16 = lower.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding='rtne')
        upper_bf16 = upper.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding='rtne')
        lower_bits = lower_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 65535
        upper_bits = upper_bf16.to(tl.int16, bitcast=True).to(tl.int32) & 65535
        lower = tl.load(CAP + lower_bits).to(tl.float64)
        upper = tl.load(CAP + upper_bits).to(tl.float64)
    tl.store(LOWER + row, lower, mask=valid)
    tl.store(UPPER + row, upper, mask=valid)


def _resources(kernel):
    result = {}
    for name in ('n_regs', 'n_spills', 'num_warps', 'shared'):
        value = getattr(kernel, name, None)
        if value is None:
            value = getattr(getattr(kernel, 'metadata', None), name, None)
        if value is not None:
            result[name] = int(value)
    for name in ('hash', 'name'):
        value = getattr(kernel, name, None)
        if value is not None:
            result[name] = str(value)
    return result


def split_intervals_with_validity_no_host_admission(hidden, summary, *, resource_report=None):
    """Return lower, upper, and a device int32 validity flag without sync.

    Emission requires validity[0]==1 plus the support certificate. The caller
    also owns immutable summary/weight lifetime. Layout checks below are host
    metadata checks only; no query-dependent tensor value is copied to host.
    """
    h = hidden.reshape(-1)
    if (h.numel() != summary.dimension or h.dtype != torch.bfloat16
            or h.device != summary.weight.device or not h.is_contiguous()):
        raise ValueError('Fixed-layout hidden vector differs from the summary')
    center = torch.empty(summary.vocabulary, dtype=torch.float32, device=h.device)
    norm = torch.empty(1, dtype=torch.float64, device=h.device)
    valid = torch.empty(1, dtype=torch.int32, device=h.device)
    lower = torch.empty(summary.vocabulary, dtype=torch.float64, device=h.device)
    upper = torch.empty_like(lower)
    gamma64 = _gamma(summary.block + 2, 53)
    denominator64 = math.nextafter(1.0 - gamma64, -math.inf)
    gamma32 = _gamma(2 * summary.block + 4, 24)
    denominator32 = math.nextafter(1.0 - gamma32, -math.inf)
    tiny_guard = math.nextafter(2 * (2 * summary.block + 4) * (2.0 ** -126) / denominator32, math.inf)
    preview = _row_preview[(summary.vocabulary,)](
        summary.quantized, summary.scales, h, center,
        D=summary.dimension, BLOCK=summary.block, num_warps=summary.num_warps,
        enable_fp_fusion=False)
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
        resource_report.update(preview=_resources(preview), norm=_resources(norm_kernel), bounds=_resources(bounds))
    return lower, upper, valid


def fused_intervals_no_host_admission(hidden, summary):
    """Compatibility API for an independently admitted input or external gate."""
    lower, upper, _ = split_intervals_with_validity_no_host_admission(hidden, summary)
    return lower, upper


def projection_intervals(hidden, summary):
    h = _hidden(hidden, summary)
    lower, upper, _ = split_intervals_with_validity_no_host_admission(h, summary)
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
        implementation='one INT8 preview, one hidden norm, vectorized FP64 bounds',
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
