"""Exact coupling for corrected_runner.draw_categorical's finite RNG grid.

The reference samples U=k/2**53, builds torch.float64 cumsum on the PMF's device,
forms the Python-binary64 product U*total, then uses right=True searchsorted and
clamps to the last token. We count the ranks producing each token under exactly
that algorithm. This is a distribution over the finite uniform RNG grid, not an
ideal real-number softmax law. No statistical approximation or epsilon is used.

Legacy CUDA cumsum is nondeterministic: reference_grid_counts counts one stored
CDF realization. The operational canonical_reference_grid_counts and shared
canonical_draw_categorical explicitly use a deterministic CDF, with flags scoped
to that operation. Their exact equivalence does not claim preservation of the
legacy GPU CDF's schedule-dependent incidental bit law.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
import torch


def _grid_size(grid_bits: int) -> int:
    if isinstance(grid_bits, bool) or not isinstance(grid_bits, int) or not 1 <= grid_bits <= 53:
        raise ValueError('grid_bits must be an integer in [1,53]')
    return 1 << grid_bits


def _first_threshold_ge(boundary: float, total: float, grid_size: int) -> int:
    """Count integer k in [0,M) with binary64((k/M)*total) < boundary."""
    low, high = 0, grid_size
    while low < high:
        mid = (low + high) // 2
        # k and 2**b are represented exactly for b<=53. Division is an exact
        # dyadic and multiplication uses the same binary64 rounding as reference.
        threshold = (mid / grid_size) * total
        if threshold < boundary:
            low = mid + 1
        else:
            high = mid
    return low


def _searchsorted_tree_grid_counts(cdf_cpu: np.ndarray, total: float, size: int):
    """Count PyTorch 2.8 upper_bound's actual comparison tree.

    CUDA parallel cumsum can backtrack by one ULP even on nonnegative inputs.
    Sorting/clamping those boundaries would change the reference sampler. The
    CUDA upper_bound code uses mid=start+((end-start)>>1), branches right iff
    !(cdf[mid]>threshold), and stops when start==end. Each such comparison splits
    a contiguous uniform-rank interval, even if the CDF itself is nonsorted.
    """
    count = len(cdf_cpu)
    masses = np.zeros(count, dtype=np.int64)
    stack = [(0, count, 0, size)]
    cuts = {}
    visited = 0
    while stack:
        start, end, rank_low, rank_high = stack.pop()
        if rank_low >= rank_high:
            continue
        if start == end:
            masses[min(start, count-1)] += rank_high-rank_low
            continue
        visited += 1
        mid = start + ((end-start) >> 1)
        boundary = float(cdf_cpu[mid])
        cut = cuts.get(boundary)
        if cut is None:
            cut = _first_threshold_ge(boundary, total, size)
            cuts[boundary] = cut
        # Left branch: threshold < cdf[mid]. Right branch includes exact ties.
        if rank_low < min(rank_high, cut):
            stack.append((start, mid, rank_low, min(rank_high, cut)))
        if max(rank_low, cut) < rank_high:
            stack.append((mid+1, end, max(rank_low, cut), rank_high))
    return masses, visited, len(cuts)


def grid_counts_from_cdf(cdf: torch.Tensor, grid_bits: int = 53,
                         *, diagnostics: dict | None = None) -> torch.Tensor:
    """Count the exact native upper_bound tree for a supplied stored CDF.

    The CDF is not sorted, clipped, or monotonized. The nonsorted case matches
    PyTorch v2.8's inspected CPU/CUDA upper_bound implementation rather than an
    abstract searchsorted specification requiring sorted boundaries.
    """
    size = _grid_size(grid_bits)
    if not torch.is_tensor(cdf) or cdf.ndim != 1 or cdf.numel()==0 or cdf.dtype != torch.float64:
        raise TypeError('cdf must be a nonempty 1D torch.float64 tensor')
    cdf_cpu = cdf.detach().cpu().numpy()
    total = float(cdf_cpu[-1])
    if not np.isfinite(cdf_cpu).all() or total <= 0:
        raise ValueError('invalid reference categorical normalizer or CDF')
    backwards = cdf_cpu[1:] < cdf_cpu[:-1]
    if np.any(backwards):
        counts, nodes, unique_cuts = _searchsorted_tree_grid_counts(cdf_cpu, total, size)
        algorithm = 'pytorch_2_8_upper_bound_comparison_tree'
    else:
        counts = np.zeros(cdf.numel(), dtype=np.int64)
        previous = 0
        boundaries = cdf_cpu[:-1]
        unique_cuts = 0
        if len(boundaries):
            changes = np.flatnonzero(np.r_[True, boundaries[1:] != boundaries[:-1]])
            for index in changes:
                boundary_count = _first_threshold_ge(float(boundaries[index]), total, size)
                counts[index] = boundary_count - previous
                previous = boundary_count
            unique_cuts = len(changes)
        counts[-1] = size-previous
        nodes = 0
        algorithm = 'monotone_boundary_partition'
    if np.any(counts < 0) or int(counts.sum()) != size:
        raise ArithmeticError('inconsistent exact categorical grid partition')
    if diagnostics is not None:
        diagnostics.update(algorithm=algorithm, grid_bits=grid_bits, total=total,
            cdf_length=cdf.numel(), cdf_backward_steps=int(backwards.sum()),
            cdf_max_backward_step=float(np.max(np.maximum(cdf_cpu[:-1]-cdf_cpu[1:],0))) if len(cdf_cpu)>1 else 0.,
            comparison_tree_nodes=nodes, distinct_threshold_searches=unique_cuts,
            positive_integer_tokens=int((counts>0).sum()),
            source='https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/Bucketization.cu')
    return torch.from_numpy(counts).to(cdf.device)


def reference_grid_counts(p: torch.Tensor, grid_bits: int = 53,
                           *, diagnostics: dict | None = None) -> torch.Tensor:
    """Exact integer token masses of the ORIGINAL finite-CDF sampler.

    ``p`` must be a nonempty one-dimensional float64 tensor, the dtype returned
    by the original policy_prob. Its sum need not equal one. CDF computation is
    deliberately performed on p's original device before one CPU copy. Duplicate
    CDF entries, zero weights, rounded boundary ties, and last-index clamping are
    all part of the target. CUDA cumsum backtracking is preserved by enumerating
    the actual upper_bound comparison tree, never by modifying the CDF. Returned
    int64 masses sum exactly to 2**grid_bits.
    """
    size = _grid_size(grid_bits)
    if not torch.is_tensor(p) or p.ndim != 1 or p.numel() == 0 or p.dtype != torch.float64:
        raise TypeError('p must be a nonempty 1D torch.float64 tensor')
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError('p must contain finite nonnegative weights')
    cdf = p.double().cumsum(0)  # MUST match the reference device and operation.
    return grid_counts_from_cdf(cdf, grid_bits, diagnostics=diagnostics)


def canonical_cdf(p: torch.Tensor) -> torch.Tensor:
    """Shared deterministic CDF for the revised Dense and Progressive sampler.

    Legacy CUDA parallel cumsum is nondeterministic even for the same finite PMF.
    PyTorch 2.8's deterministic CDF implementation is used only for this operation;
    the prior deterministic-algorithm and warn-only settings are always restored.
    Native attention, suffix, and policy_prob execute under their original flags.

    This explicitly changes the legacy CUDA CDF's incidental rounding schedule.
    It preserves the native model logits and policy PMF, not the legacy sampler's
    schedule-dependent bit law. Because the PyTorch flag is process-global, this
    scoped helper requires serialized calls, as in the single-process harness.
    """
    if not torch.is_tensor(p) or p.ndim != 1 or p.numel()==0 or p.dtype != torch.float64:
        raise TypeError('p must be a nonempty 1D torch.float64 tensor')
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError('p must contain finite nonnegative weights')
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        return p.double().cumsum(0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def canonical_reference_grid_counts(p: torch.Tensor, grid_bits: int = 53,
                                    *, diagnostics: dict | None = None) -> torch.Tensor:
    """Exact finite-grid law of canonical_draw_categorical, same native PMF."""
    cdf = canonical_cdf(p)
    counts = grid_counts_from_cdf(cdf, grid_bits, diagnostics=diagnostics)
    if diagnostics is not None:
        diagnostics.update(cdf_contract='PyTorch deterministic cumsum, flag scoped to CDF only',
            original_native_logits_and_policy_pmf_unchanged=True,
            legacy_nondeterministic_cuda_cdf_bitlaw_preserved=False)
    return counts


def canonical_draw_categorical(p: torch.Tensor, rng) -> int:
    """Dense reference draw sharing the explicit canonical deterministic CDF."""
    cdf = canonical_cdf(p)
    total = float(cdf[-1])
    if not math.isfinite(total) or total <= 0:
        raise ValueError('invalid reference categorical normalizer')
    uniform = float(rng.random())
    if not 0 <= uniform < 1 or not (uniform*(1<<53)).is_integer():
        raise ValueError('RNG output must belong to the uniform 53-bit reference grid')
    index = torch.searchsorted(cdf, torch.tensor(uniform*total, dtype=p.dtype,
        device=p.device), right=True)
    return min(int(index.item()), p.numel()-1)


def _validate_counts(counts: torch.Tensor, total: int | None = None) -> int:
    if not torch.is_tensor(counts) or counts.ndim != 1 or counts.numel() == 0 or counts.dtype != torch.int64:
        raise TypeError('counts must be a nonempty 1D torch.int64 tensor')
    if bool((counts < 0).any()):
        raise ValueError('negative integer probability mass')
    # Each admissible total is <=2**53, comfortably inside signed int64.
    if bool((counts > (1 << 53)).any()):
        raise ValueError('integer mass exceeds the largest supported grid')
    approximate = float(counts.double().sum().item())
    if not math.isfinite(approximate) or approximate > (1 << 54):
        # Reject grossly invalid inputs before an int64 reduction could wrap.
        raise ValueError('integer mass total exceeds the supported grid')
    observed = int(counts.sum().item())
    if observed < 0 or observed > (1 << 53):
        raise ValueError('integer mass total exceeds the supported grid')
    if total is not None and observed != total:
        raise ValueError(f'integer mass total {observed} differs from grid {total}')
    return observed


def token_for_integer_rank(counts: torch.Tensor, rank: int) -> int:
    """Select a token by an integer rank, with no floating normalization."""
    total = _validate_counts(counts)
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 0 <= int(rank) < total:
        raise ValueError('rank is outside the integer mass total')
    cdf = counts.cumsum(0)
    return int(torch.searchsorted(cdf, torch.tensor(int(rank), dtype=torch.int64,
        device=counts.device), right=True).item())


@dataclass(frozen=True)
class GridDecision:
    grid_bits: int
    rank: int
    common_total: int
    early: bool
    token: int | None
    residual_rank: int | None

    @property
    def common_probability(self) -> float:
        return self.common_total / (1 << self.grid_bits)


def begin_common_grid(lower_counts: torch.Tensor, rng, grid_bits: int = 53,
                      *, rank: int | None = None) -> GridDecision:
    """Decide/emit using only the certified common integer masses.

    Exactly one uniform grid rank supplies both the emission decision and token.
    A deferred result retains its rank in the residual domain. ``rank`` is an
    explicit exhaustive-test hook; ordinary callers supply only rng.
    """
    size = _grid_size(grid_bits)
    common = _validate_counts(lower_counts)
    if common > size:
        raise ValueError('common mass exceeds the selected grid')
    if rank is None:
        uniform = float(rng.random())
        if not 0 <= uniform < 1 or not math.isfinite(uniform):
            raise ValueError('RNG must return a finite number in [0,1)')
        if not (uniform * (1 << 53)).is_integer():
            raise ValueError('RNG output must belong to the uniform 53-bit reference grid')
        rank = int(uniform * size)
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 0 <= int(rank) < size:
        raise ValueError('rank is outside the selected grid')
    rank = int(rank)
    early = rank < common
    token = token_for_integer_rank(lower_counts, rank) if early else None
    return GridDecision(grid_bits, rank, common, early, token,
                        None if early else rank-common)


def finish_common_grid(decision: GridDecision, lower_counts: torch.Tensor,
                       target_counts: torch.Tensor) -> int:
    """Complete a deferred draw using exact target-minus-common integer mass."""
    size = _grid_size(decision.grid_bits)
    common = _validate_counts(lower_counts)
    _validate_counts(target_counts, size)
    if lower_counts.shape != target_counts.shape or lower_counts.device != target_counts.device:
        raise ValueError('target and common masses must share shape and device')
    if common != decision.common_total:
        raise ValueError('common mass changed after the decision')
    residual = target_counts - lower_counts
    if bool((residual < 0).any()):
        raise ArithmeticError('common mass is not dominated by the actual target')
    if decision.early:
        if decision.token != token_for_integer_rank(lower_counts, decision.rank):
            raise ValueError('inconsistent early decision')
        return int(decision.token)
    if decision.residual_rank != decision.rank-common or not 0 <= decision.rank < size:
        raise ValueError('inconsistent residual decision')
    return token_for_integer_rank(residual, decision.residual_rank)


def sample_common_grid(lower_counts: torch.Tensor, rng,
                       completion_counts: Callable[[], torch.Tensor],
                       grid_bits: int = 53) -> GridDecision:
    """Convenience wrapper; invokes completion ONLY after an actual deferral."""
    decision = begin_common_grid(lower_counts, rng, grid_bits)
    if decision.early:
        return decision
    token = finish_common_grid(decision, lower_counts, completion_counts())
    return GridDecision(decision.grid_bits, decision.rank, decision.common_total,
                        False, token, decision.residual_rank)
