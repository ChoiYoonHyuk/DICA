"""Conditional output-head experiment with immutable CPU-pinned BF16 source.

Only query-independent INT8 summaries remain on the GPU between queries. Both
shared-reference methods copy their original BF16 operands from the CPU source
inside each query and invoke the exact same row-separable Triton kernel. This
module makes no full-decoder, embedding-placement, or GPU-resident speed claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from policy_projection import build_projection_summary, _row_reference, _cap, _gamma, _version
from policy_projection_split import _single_hidden_norm, _resources
from policy_projection_tiled import tiled_preview
from policy_projection_inflated import _inflated_vector_intervals
from exact_grid_sampler import canonical_cdf


@triton.jit(do_not_specialize=['N'])
def _dynamic_packed_keys(LOGITS, IDS, KEYS, N, V: tl.constexpr, B: tl.constexpr):
    position = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(LOGITS + position, mask=position < N, other=0)
    bits = value.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    bits = tl.where((bits & 32767) == 0, 0, bits)
    rank = tl.where((bits & 32768) != 0, 65535-bits, bits+32768).to(tl.int64)
    token = tl.load(IDS + position, mask=position < N, other=0)
    key = rank * (V+1) + (V-1-token)
    tl.store(KEYS + position, key, mask=position < N)


def dynamic_rank_positions(logits, ids, *, k, vocabulary, hashes):
    """Same stable BF16/token key, with candidate count excluded from JIT keys."""
    n = logits.numel()
    if (logits.dtype != torch.bfloat16 or ids.dtype != torch.int64
            or logits.ndim != 1 or ids.shape != logits.shape
            or not logits.is_contiguous() or not ids.is_contiguous() or n < k):
        raise ValueError('Invalid packed top-k operands')
    keys = torch.empty(n, dtype=torch.int64, device=logits.device)
    kernel = _dynamic_packed_keys[(triton.cdiv(n, 256),)](
        logits, ids, keys, n, V=vocabulary, B=256, num_warps=4)
    identity = str(kernel.hash)
    hashes.add(identity)
    if len(hashes) != 1:
        raise ArithmeticError('Candidate count changed the compiled key encoder')
    return torch.topk(keys, k, largest=True, sorted=True).indices


def audit_dynamic_keys(device, vocabulary):
    """Public finite-BF16 key parity and runtime-count compilation invariance."""
    from packed_topk import packed_keys
    bits = torch.arange(65536, dtype=torch.int32, device=device)
    bits = bits[((bits & 32767) < 32640) | (bits == 65408)]
    logits = bits.to(torch.int16).view(torch.bfloat16).contiguous()
    ids = torch.arange(logits.numel()-1, -1, -1, device=device, dtype=torch.int64)
    expected = packed_keys(logits, ids, vocab_size=vocabulary)
    hashes = set()
    counts = [1, 15, 16, 17, 255, 256, 257, 4096, 8192, logits.numel()]
    for n in counts:
        actual = torch.empty(n, dtype=torch.int64, device=device)
        kernel = _dynamic_packed_keys[(triton.cdiv(n, 256),)](
            logits, ids, actual, n, V=vocabulary, B=256, num_warps=4)
        hashes.add(str(kernel.hash))
        if not torch.equal(actual, expected[:n]):
            raise ArithmeticError('Runtime-count keys differ from the exact frozen key rule')
    if len(hashes) != 1:
        raise ArithmeticError('Runtime count changed key encoder specialization')
    return dict(key_bits_equal=True, finite_alphabet_patterns=65280,
        negative_infinity_padding=True, tested_counts=counts,
        compiled_hashes=sorted(hashes), candidate_count_not_specialized=True)


@dataclass
class QueryResult:
    token: int
    pmf: torch.Tensor
    cdf: torch.Tensor
    ids: torch.Tensor
    logits: torch.Tensor
    diagnostic: dict


class HostProjection:
    """Own a pinned source and summaries, never an original-weight CUDA tensor.

    The caller must release the model and every external CUDA weight reference
    after construction. Source immutability is an ownership contract, reinforced
    by a version check when supported and whole-source hashes in the harness.
    """

    def __init__(self, weight, *, softcap=None, k=50, temperature=.7):
        if not 1 <= k <= weight.shape[0] or temperature <= 0:
            raise ValueError('Invalid top-k policy')
        self.summary = build_projection_summary(weight, softcap=softcap, num_warps=4)
        self.summary.preview_row_tile = 1
        self.summary.preview_num_warps = 4
        self.device = self.summary.quantized.device
        self.k, self.temperature = int(k), float(temperature)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        # Construct outside inference mode so ownership additionally has a
        # functioning version counter even when the caller uses inference mode.
        with torch.inference_mode(False):
            source = torch.empty(weight.shape, dtype=torch.bfloat16, device='cpu', pin_memory=True)
            source.copy_(weight.detach(), non_blocking=False)
        torch.cuda.synchronize(self.device)
        self.pin_and_copy_seconds = time.perf_counter() - started
        self.source = source
        self.source_version = _version(source)
        self.summary.weight = source
        self.summary.weight_version = self.source_version
        self.reference_kernels = {}
        self.key_kernel_hashes = set()
        self.full_weight_bytes = source.numel() * source.element_size()
        self.summary_read_bytes = (self.summary.quantized.numel()
            + self.summary.vocabulary * (4 + 3 * 8)
            + (self.summary.vocabulary * 4 if self.summary.cap_table is not None else 0))
        self._check_source()

    def _check_source(self):
        if (self.source.device.type != 'cpu' or not self.source.is_pinned()
                or self.summary.weight is not self.source or self.source.dtype != torch.bfloat16
                or not self.source.is_contiguous() or _version(self.source) != self.source_version):
            raise RuntimeError('Immutable pinned original-weight ownership changed')

    def _layout(self, hidden):
        self._check_source()
        h = hidden.reshape(-1)
        if (h.dtype != torch.bfloat16 or h.device != self.device
                or h.numel() != self.summary.dimension or not h.is_contiguous()):
            raise ValueError('Hidden vector differs from fixed native BF16 layout')
        return h

    def _norm(self, h):
        s = self.summary
        norm = torch.empty(1, dtype=torch.float64, device=self.device)
        valid = torch.empty(1, dtype=torch.int32, device=self.device)
        denominator = math.nextafter(1.0 - _gamma(s.block + 2, 53), -math.inf)
        _single_hidden_norm[(1,)](h, norm, valid, D=s.dimension, BLOCK=s.block,
            DENOMINATOR64=denominator, num_warps=4, enable_fp_fusion=False)
        return norm, valid

    def _intervals(self, h):
        # Explicit device ownership: summary.weight is CPU and is never passed
        # to a helper which assumes that it defines the computation device.
        s = self.summary
        center = tiled_preview(h, s)
        norm, valid = self._norm(h)
        lo = torch.empty(s.vocabulary, dtype=torch.float64, device=self.device)
        hi = torch.empty_like(lo)
        gamma = _gamma(2 * s.block + 4, 24)
        denominator = math.nextafter(1.0 - gamma, -math.inf)
        tiny = math.nextafter(2 * (2 * s.block + 4) * 2.0**-126 / denominator, math.inf)
        _inflated_vector_intervals[(triton.cdiv(s.vocabulary, 256),)](
            center, norm, s.residual_max, s.weight_max, s.approximate_max,
            s.scales if s.cap_table is None else s.cap_table, lo, hi,
            VOCAB=s.vocabulary, HAS_CAP=s.cap_table is not None,
            GAMMA32=gamma, TINY_GUARD=tiny, ROWS=256, num_warps=4, enable_fp_fusion=False)
        if not bool(valid.item()):
            raise ValueError('Hidden admission requires BF16 normal-or-zero magnitude <=2^20')
        return lo, hi

    def _project_transferred(self, h, gpu_weight, local_ids, method):
        s = self.summary
        if (gpu_weight.dtype != torch.bfloat16 or gpu_weight.device != self.device
                or not gpu_weight.is_contiguous() or gpu_weight.shape[1] != s.dimension):
            raise ValueError('Transferred source rows changed the reference operand layout')
        out = torch.empty(local_ids.numel(), dtype=torch.bfloat16, device=self.device)
        kernel = _row_reference[(local_ids.numel(),)](gpu_weight, h, local_ids, out,
            D=s.dimension, BLOCK=s.block, num_warps=4, enable_fp_fusion=False)
        if method not in self.reference_kernels:
            self.reference_kernels[method] = _resources(kernel)
        hashes = [value.get('hash') for value in self.reference_kernels.values()]
        if any(value is None for value in hashes) or len(set(hashes)) != 1:
            raise ArithmeticError('Dense and packed rows do not share one compiled row kernel')
        return _cap(out, s)

    def _policy_draw(self, ids, logits, uniform):
        if not 0 <= uniform < 1 or not (uniform * (1 << 53)).is_integer():
            raise ValueError('Uniform must belong to the 53-bit categorical grid')
        keep = dynamic_rank_positions(logits, ids, k=self.k,
            vocabulary=self.summary.vocabulary, hashes=self.key_kernel_hashes)
        values = logits[keep].double() / self.temperature
        pmf = torch.zeros(self.summary.vocabulary, dtype=torch.float64, device=self.device)
        pmf[ids[keep]] = torch.softmax(values, dim=0)
        cdf = canonical_cdf(pmf)
        total = float(cdf[-1])
        if not math.isfinite(total) or total <= 0:
            raise ArithmeticError('Invalid categorical normalization')
        threshold = torch.tensor(uniform * total, dtype=torch.float64, device=self.device)
        token = min(int(torch.searchsorted(cdf, threshold, right=True).item()), pmf.numel() - 1)
        return pmf, cdf, token

    def query(self, method, hidden, uniform):
        """All temporary gathering, allocation, copying, policy and draw are online."""
        h = self._layout(hidden)
        s = self.summary
        if method == 'adaptive_host':
            lo, hi = self._intervals(h)
            threshold = torch.topk(lo, self.k, largest=True, sorted=False).values.min()
            ids = torch.nonzero(hi >= threshold, as_tuple=False).reshape(-1).contiguous()
            if ids.numel() < self.k:
                raise ArithmeticError('Certified candidate set has fewer than k rows')
            ids_cpu = ids.to(device='cpu', non_blocking=False)
            packed = torch.empty((ids.numel(), s.dimension), dtype=torch.bfloat16,
                device='cpu', pin_memory=True)
            torch.index_select(self.source, 0, ids_cpu, out=packed)
            gpu_weight = packed.to(device=self.device, non_blocking=True)
            local_ids = torch.arange(ids.numel(), dtype=torch.int64, device=self.device)
            logits = self._project_transferred(h, gpu_weight, local_ids, method)
            summary_bytes = self.summary_read_bytes
            ids_to_host_bytes = ids.numel() * 8
        elif method in ('row_dense_host', 'cublas_dense_host'):
            _, valid = self._norm(h)
            if not bool(valid.item()):
                raise ValueError('Hidden admission failed')
            ids = s.all_ids
            gpu_weight = self.source.to(device=self.device, non_blocking=True)
            if method == 'row_dense_host':
                logits = self._project_transferred(h, gpu_weight, ids, method)
            else:
                logits = F.linear(h.reshape(1, 1, -1), gpu_weight, bias=None).reshape(-1)
                if s.softcap is not None:
                    logits = torch.tanh(logits / s.softcap) * s.softcap
            summary_bytes = 0
            ids_to_host_bytes = 0
        else:
            raise ValueError(method)
        pmf, cdf, token = self._policy_draw(ids, logits, float(uniform))
        n = int(ids.numel())
        diagnostic = dict(method=method, selected_rows=n, vocabulary=s.vocabulary,
            original_weight_bytes_read_cpu=n*s.dimension*2,
            original_weight_bytes_host_to_device=n*s.dimension*2,
            candidate_ids_bytes_device_to_host=ids_to_host_bytes,
            summary_logical_operand_bytes=summary_bytes,
            partial_weight_and_interval_operand_bytes=summary_bytes+n*s.dimension*2,
            logical_byte_scope='weight payload and interval operands only; excludes policy/CDF, IDs, temporary-buffer traffic, and selected-logit cap lookups',
            original_weight_row_read_fraction=n/s.vocabulary,
            full_vocabulary_pmf_and_cdf=True, candidate_cap=None,
            original_weight_gpu_residency='temporary query-owned transfer only')
        # The draw synchronized this stream. No return value retains gpu_weight
        # or the pinned candidate staging allocation under inference mode.
        return QueryResult(token, pmf, cdf, ids, logits, diagnostic)

    def description(self):
        s = self.summary
        return dict(source='immutable CPU pinned original BF16 rows',
            source_shape=list(self.source.shape), source_pinned=self.source.is_pinned(),
            source_has_version_counter=self.source_version is not None,
            original_weight_cpu_bytes=self.full_weight_bytes,
            original_weight_live_gpu_bytes_between_queries=0,
            summary_setup_seconds=s.setup_seconds, pin_and_copy_seconds=self.pin_and_copy_seconds,
            summary_gpu_storage_bytes=s.storage_bytes,
            adaptive_summary_logical_operand_bytes=self.summary_read_bytes,
            summary_bytes_definition='INT8 weight, scale, three FP64 row bounds, and two cap-table endpoint loads; logical operands, not measured HBM traffic',
            reference=s.reference, row_kernel_block=s.block, row_kernel_warps=4,
            preview_row_tile=1, preview_warps=4, temperature=self.temperature, topk=self.k,
            softcap=s.softcap, static_admission=s.static_admission,
            reference_kernels=self.reference_kernels,
            candidate_key_kernel_hashes=sorted(self.key_kernel_hashes),
            candidate_count_runtime_only=True,
            conditional_output_head_only=True, kv_read_reduction_percent=0,
            original_cublas_reference_preserved=False,
            setup_charged_per_query=False)
