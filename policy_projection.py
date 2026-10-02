"""Certified vocabulary screening against an explicit per-row BF16 reference.

The reference is NOT the original cuBLAS projection. Dense and selected rows
execute the identical row-separable kernel, with identical reduction width,
dtype, warps and fusion setting. Static summaries are built from weights alone.
Every input validation, norm, preview, interval and selection belongs online.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _row_reference(W, H, IDS, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.load(IDS + tl.program_id(0))
    col = tl.arange(0, BLOCK)
    w = tl.load(W + row * D + col, mask=col < D, other=0).to(tl.float32)
    h = tl.load(H + col, mask=col < D, other=0).to(tl.float32)
    products = w * h
    total = tl.sum(products, axis=0)
    tl.store(OUT + tl.program_id(0), total.to(tl.bfloat16, fp_downcast_rounding='rtne'))


@triton.jit
def _row_preview(Q, SCALE, H, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    q = tl.load(Q + row * D + col, mask=col < D, other=0).to(tl.float32)
    scale = tl.load(SCALE + row)
    h = tl.load(H + col, mask=col < D, other=0).to(tl.float32)
    # q*scale is exactly representable and normal (or zero) under admission.
    approximate_weight = q * scale
    products = approximate_weight * h
    total = tl.sum(products, axis=0)
    tl.store(OUT + row, total)


def _sync(device):
    torch.cuda.synchronize(device)


def _version(t):
    try:
        return t._version
    except RuntimeError:
        return None


def _up(x):
    return torch.nextafter(x, torch.full_like(x, float('inf')))


def _gamma(operations, bits):
    u = 2.0 ** -bits
    if operations * u >= 1:
        raise ValueError('Invalid arithmetic error bound')
    return math.nextafter((operations * u) / (1.0 - operations * u), math.inf)


@dataclass
class ProjectionSummary:
    weight: torch.Tensor
    quantized: torch.Tensor
    scales: torch.Tensor
    residual_max: torch.Tensor
    weight_max: torch.Tensor
    approximate_max: torch.Tensor
    all_ids: torch.Tensor
    cap_table: torch.Tensor | None
    softcap: float | None
    dimension: int
    vocabulary: int
    block: int
    num_warps: int
    weight_version: int | None
    setup_seconds: float
    storage_bytes: int
    reference: str
    static_admission: str


def build_projection_summary(weight, softcap=None, *, chunk_rows=4096, num_warps=8):
    """Construct query-independent INT8 data and outward residual bounds.

    Call once for immutable weights and report setup_seconds and storage_bytes.
    Inference tensors without version counters require caller-owned immutability.
    This constructor performs no hidden-state or query-dependent preparation.
    """
    if (weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda
            or not weight.is_contiguous()):
        raise ValueError('Expected contiguous CUDA BF16 vocabulary weights')
    vocabulary, dimension = map(int, weight.shape)
    if not 1 <= dimension <= 8192 or vocabulary < 1:
        raise ValueError('Supported head dimensions are 1 through 8192')
    if weight.numel() >= (1 << 31):
        raise ValueError('The fixed preview address contract requires fewer than 2^31 weight elements')
    if chunk_rows < 1 or num_warps not in (4, 8):
        raise ValueError('Positive chunks and four or eight warps required')
    if softcap is not None and (not math.isfinite(float(softcap)) or softcap <= 0):
        raise ValueError('Softcap must be finite and positive')
    _sync(weight.device)
    started = time.perf_counter()
    q = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scales = torch.empty(vocabulary, dtype=torch.float32, device=weight.device)
    residual = torch.empty(vocabulary, dtype=torch.float64, device=weight.device)
    maximum = torch.empty_like(residual)
    approximate_max = torch.empty_like(residual)
    tiny = 2.0 ** -126
    with torch.inference_mode():
        for first in range(0, vocabulary, chunk_rows):
            last = min(vocabulary, first + chunk_rows)
            w = weight[first:last].float()
            absolute = w.abs()
            magnitude_bits = weight[first:last].contiguous().view(torch.int16).to(torch.int32) & 32767
            valid = (magnitude_bits == 0) | ((magnitude_bits >= 0x80) & (magnitude_bits <= 0x4980))
            if not bool(valid.all()):
                raise ValueError('Weight admission requires finite BF16 normal-or-zero values with magnitude <=2^20')
            maxima = absolute.amax(dim=1)
            # Only the chosen power of two matters for the proof. Rounding in
            # this heuristic affects interval quality, not enclosure validity.
            exponent = torch.ceil(torch.log2(torch.clamp(maxima / 127.0, min=2.0 ** -64)))
            exponent = exponent.to(torch.int32).clamp(-64, 32)
            scale = ((exponent + 127) << 23).contiguous().view(torch.float32)
            quantized = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
            q[first:last] = quantized
            scales[first:last] = scale
            # Binary64 represents both BF16 weights and s*q exactly. One
            # nextafter covers subtraction rounding before the max reduction.
            approximate = quantized.double() * scale.double()[:, None]
            delta = _up((weight[first:last].double() - approximate).abs())
            residual[first:last] = delta.amax(dim=1)
            maximum[first:last] = maxima.double()
            approximate_max[first:last] = approximate.abs().amax(dim=1)
        all_ids = torch.arange(vocabulary, dtype=torch.int64, device=weight.device)
        cap_table = None
        if softcap is not None:
            raw = torch.arange(65536, dtype=torch.int32, device=weight.device).to(torch.int16)
            values = raw.view(torch.bfloat16)
            finite = torch.isfinite(values)
            cap_table = (torch.tanh(values / float(softcap)) * float(softcap)).contiguous()
            ordered = torch.argsort(values[finite].float(), stable=True)
            image = cap_table[finite][ordered].float()
            if not bool(torch.isfinite(image).all() and (image[1:] >= image[:-1]).all()):
                raise ArithmeticError('Full finite-BF16 softcap table is not finite and monotone')
            # Both methods explicitly use this immutable table as their cap
            # reference. Original shape-dependent tanh parity is a separate test.
        tensors = (q, scales, residual, maximum, approximate_max, all_ids, cap_table)
        storage_bytes = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        _sync(weight.device)
    return ProjectionSummary(weight, q, scales, residual, maximum, approximate_max,
        all_ids, cap_table, None if softcap is None else float(softcap), dimension,
        vocabulary, triton.next_power_of_2(dimension), num_warps, _version(weight),
        time.perf_counter() - started, storage_bytes,
        'fixed per-row FP32 reduction followed by BF16 round-to-nearest-even; optional finite-BF16 cap table',
        'finite normal-or-zero BF16 weights and queries; absolute value <=2^20; immutable weight ownership')


def _hidden(h, summary):
    h = h.reshape(-1)
    if (h.numel() != summary.dimension or h.dtype != torch.bfloat16
            or h.device != summary.weight.device or not h.is_contiguous()):
        raise ValueError('Hidden vector must match contiguous native BF16 layout')
    if _version(summary.weight) != summary.weight_version:
        raise RuntimeError('Projection weights changed after summary construction')
    magnitude_bits = h.contiguous().view(torch.int16).to(torch.int32) & 32767
    valid = (magnitude_bits == 0) | ((magnitude_bits >= 0x80) & (magnitude_bits <= 0x4980))
    if not bool(valid.all()):
        raise ValueError('Hidden admission requires finite normal-or-zero BF16 values with magnitude <=2^20')
    return h


def _cap(logits, summary):
    if summary.cap_table is None:
        return logits
    bits = logits.contiguous().view(torch.int16).to(torch.int32) & 65535
    return summary.cap_table[bits]


def _project(h, summary, ids):
    out = torch.empty(ids.numel(), dtype=torch.bfloat16, device=h.device)
    _row_reference[(ids.numel(),)](summary.weight, h, ids, out,
        D=summary.dimension, BLOCK=summary.block, num_warps=summary.num_warps,
        enable_fp_fusion=False)
    return _cap(out, summary)


def dense_projection(hidden, summary):
    """All rows of the explicitly shared reference, with online admission."""
    return _project(_hidden(hidden, summary), summary, summary.all_ids)


def projection_intervals(hidden, summary):
    """Certified intervals for the shared BF16 projection; no BF16 W read."""
    h = _hidden(hidden, summary)
    center = torch.empty(summary.vocabulary, dtype=torch.float32, device=h.device)
    _row_preview[(summary.vocabulary,)](summary.quantized, summary.scales, h, center,
        D=summary.dimension, BLOCK=summary.block, num_warps=summary.num_warps,
        enable_fp_fusion=False)
    # Correct the reduction of nonnegative exact BF16-to-FP64 inputs upwards.
    gamma64 = _gamma(summary.block + 2, 53)
    denominator64 = math.nextafter(1.0 - gamma64, -math.inf)
    norm1 = _up(h.double().abs().sum() / denominator64)
    gamma32 = _gamma(2 * summary.block + 4, 24)
    residual_error = _up(summary.residual_max * norm1)
    magnitudes = _up(summary.weight_max + summary.approximate_max)
    arithmetic_error = _up(_up(magnitudes * norm1) * gamma32)
    # Inputs are normal or zero. Absolute guards cover product/reduction
    # underflow or FTZ in both FP32 sums, plus the final BF16 conversion.
    denominator32 = math.nextafter(1.0 - gamma32, -math.inf)
    tiny_guard = math.nextafter(2 * (2 * summary.block + 4) * (2.0 ** -126) / denominator32, math.inf)
    radius = _up(_up(residual_error + arithmetic_error) + tiny_guard)
    c = center.double()
    bf16_round = _up(_up(_up(c.abs() + radius) * (2.0 ** -8)) + 2.0 ** -126)
    radius = _up(radius + bf16_round)
    lower = torch.nextafter(c - radius, torch.full_like(c, -math.inf))
    upper = torch.nextafter(c + radius, torch.full_like(c, math.inf))
    if summary.cap_table is not None:
        # Conversion is monotone and fixes every BF16 reference value, then
        # the exhaustively checked table preserves the interval order.
        lower = _cap(lower.to(torch.bfloat16), summary).double()
        upper = _cap(upper.to(torch.bfloat16), summary).double()
    return h, lower, upper


def screened_topk_projection(hidden, summary, *, k=50):
    """Return all retained original row IDs, exact logits, and diagnostics.

    Any true top-k row has upper bound at least the kth largest lower bound.
    Inclusive comparison retains every boundary tie; token-index ordering is
    applied by the caller's unchanged stable top-k policy.
    """
    if type(k) is not int or not 1 <= k <= summary.vocabulary:
        raise ValueError('k must be an integer in [1,vocabulary]')
    h, lower, upper = projection_intervals(hidden, summary)
    threshold = torch.topk(lower, k, largest=True, sorted=False).values.min()
    ids = torch.nonzero(upper >= threshold, as_tuple=False).reshape(-1).contiguous()
    if ids.numel() < k:
        raise ArithmeticError('Certified candidate set contains fewer than k rows')
    logits = _project(h, summary, ids)
    return ids, logits, dict(
        selected_rows=int(ids.numel()), vocabulary=summary.vocabulary,
        original_weight_row_read_fraction=ids.numel() / summary.vocabulary,
        quantized_weight_bytes=summary.quantized.numel(),
        original_weight_bytes_read=ids.numel() * summary.dimension * 2,
        lower_threshold=float(threshold),
        fp32_error_gamma=_gamma(2 * summary.block + 4, 24),
        row_kernel_block=summary.block, row_kernel_warps=summary.num_warps,
        candidate_rule='upper >= kth largest lower, retaining all ties',
        full_vocabulary_summary_scan=True,
        reference=summary.reference,
        original_cublas_reference_preserved=False,
        setup_charged_per_query=False,
        setup_seconds=summary.setup_seconds, summary_storage_bytes=summary.storage_bytes)


__all__ = ['ProjectionSummary', 'build_projection_summary', 'dense_projection',
           'projection_intervals', 'screened_topk_projection']
