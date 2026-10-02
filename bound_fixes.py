"""Audited operator fixes for original_runner; no checkpoint/GPU required to import.

These retain the runner's numerical, rather than formally verified CUDA,
arithmetic-error model. They fix model semantics, not the IBP dependency problem.
"""
from __future__ import annotations

import math
import inspect
import sys
import textwrap
import types
import torch
from original_runner import Box, down, up, gamma


def reference_cast_box(box, dtype):
    """Preserve the caller module's Box class (original and generated runners)."""
    if dtype == torch.float64:
        return box.inflate(8*torch.finfo(torch.float64).eps*(1+box.mid.abs()+box.rad))
    lo, hi = box.lo.to(dtype).double(), box.hi.to(dtype).double()
    if not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all()):
        raise ArithmeticError("Interval overflow in reference cast; certificate invalid")
    return type(box)(lo, hi)


def exact_suffix_point_box(runner, attention_box, h, per_layer_input=None, operator_trace=None):
    """Resolve a known attention singleton with the actual deterministic suffix.

    Returns None for every non-singleton, preserving all original guards there.
    A dense-oracle diagnostic can call this, but its result does not demonstrate
    savings: a real frontier must itself establish the singleton without oracle
    source reads. This uses the same native suffix and shapes as dense completion.
    """
    if not torch.equal(attention_box.lo, attention_box.hi):
        return None
    with torch.inference_mode():
        if operator_trace is not None:
            native = runner.native_suffix_trace(attention_box.lo, h, per_layer_input)
            for name, value in native.items():
                operator_trace.append((name, type(attention_box).point(value)))
            logits = next(reversed(native.values()))
        else:
            logits = runner.suffix(attention_box.lo, h, per_layer_input)
    if not bool(torch.isfinite(logits).all()):
        raise ArithmeticError("Non-finite deterministic singleton suffix")
    return type(attention_box).point(logits.reshape(1, -1))


def rms_box_corrected(
    x: Box, weight: torch.Tensor, eps: float, dtype: torch.dtype,
    *, scale_before_cast: bool, arithmetic_dtype: torch.dtype | None = None,
) -> Box:
    """Coordinate-dependent RMS bound with explicit model-specific cast order.

    Qwen3: FP32 norm -> model dtype -> multiply weight -> model dtype.
    Gemma4: FP32 norm -> multiply FP32 weight -> model dtype.
    ``scale_before_cast=True`` selects Gemma4. The arithmetic dtype defaults to
    FP32, as both native modules first call ``hidden_states.float()``.
    """
    if eps <= 0 or not bool(torch.isfinite(x.lo).all() and torch.isfinite(x.hi).all()):
        raise ArithmeticError("RMSNorm requires positive epsilon and finite intervals")
    if bool(torch.any(x.lo > x.hi)):
        raise ArithmeticError("Invalid RMSNorm input interval")
    adtype = arithmetic_dtype or torch.float32
    # The input cast is monotonic; this also handles float64 diagnostic inputs.
    native_input = reference_cast_box(x, adtype)
    l, u = native_input.lo, native_input.hi
    d = l.shape[-1]
    sqlo = torch.where((l <= 0) & (u >= 0), 0.0, torch.minimum(l.square(), u.square()))
    sqhi = torch.maximum(l.square(), u.square())

    def excluding_sum(z):
        zero = torch.zeros_like(z[..., :1])
        left = torch.cat((zero, z[..., :-1].cumsum(-1)), -1)
        right = torch.cat((z[..., 1:].flip(-1).cumsum(-1).flip(-1), zero), -1)
        return left + right

    slo, shi = excluding_sum(sqlo), excluding_sum(sqhi)
    sum_error = gamma(2*d + 8) * sqhi.sum(-1, keepdim=True)
    slo = down(slo - sum_error).clamp_min(0)
    shi = up(shi + sum_error)

    def endpoint(t, others):
        return math.sqrt(d) * t / torch.sqrt(t.square() + others + d*eps)

    low = endpoint(l, torch.where(l >= 0, shi, slo))
    high = endpoint(u, torch.where(u >= 0, slo, shi))
    guard = gamma(d + 8, adtype) + 32*torch.finfo(adtype).eps
    allowance = guard*torch.maximum(low.abs(), high.abs()) + 32*torch.finfo(adtype).tiny
    normalized = type(x)(down(low - allowance), up(high + allowance))
    if scale_before_cast:
        # Native Gemma4's normalized tensor and product both live in FP32.
        normalized = reference_cast_box(normalized, adtype)
        scaled = reference_cast_box(normalized * weight.to(adtype).double(), adtype)
        return reference_cast_box(scaled, dtype)
    normalized = reference_cast_box(normalized, dtype)
    return reference_cast_box(normalized * weight.to(dtype).double(), dtype)


def attention_score_roundoff(
    scaled_query: torch.Tensor, key_lo: torch.Tensor, key_hi: torch.Tensor,
    arithmetic_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Conservative dot-product allowance including a query-scaling allowance.

    Use absolute products, never the signed dot-product magnitude. This is a
    building block for the existing kernel-error model, not a proof of SDPA.
    ``scaled_query`` and key endpoints have the same trailing dimension.
    """
    absolute_key = torch.maximum(key_lo.double().abs(), key_hi.double().abs())
    absolute_products = (scaled_query.double().abs() * absolute_key).sum(-1)
    # d+4 retains the original head-dot/scaling error count. Include fp64
    # summation error in calculating this nonnegative diagnostic itself.
    d = scaled_query.shape[-1]
    return up((gamma(d + 4, arithmetic_dtype) + gamma(d + 4))*absolute_products)


def install_gemma_norm_patch(module):
    """Patch a runner module's suffix method without modifying its source file.

    The temporary function substitution is safe only for a single-threaded
    benchmark process; production code should pass model semantics explicitly.
    Use the returned uninstaller after the benchmark. A direct source change
    in ``normbox`` is preferred when creating a revised runner.
    """
    original = module.LastLayerRunner._suffix_box_impl

    def patched(self, attention_box, h, tight, operator_trace=None, per_layer_input=None):
        if self.family != "gemma4":
            return original(self, attention_box, h, tight, operator_trace, per_layer_input)
        original_rms = module.rms_box_tight if tight else module.rms_box

        def gemma_rms(x, weight, eps, dtype):
            return rms_box_corrected(x, weight, eps, dtype, scale_before_cast=True)

        attr = "rms_box_tight" if tight else "rms_box"
        setattr(module, attr, gemma_rms)
        try:
            return original(self, attention_box, h, tight, operator_trace, per_layer_input)
        finally:
            setattr(module, attr, original_rms)

    module.LastLayerRunner._suffix_box_impl = patched

    def uninstall():
        module.LastLayerRunner._suffix_box_impl = original

    return uninstall


def install_runner_fixes(runner):
    """Install all three corrections on one existing runner without global swaps.

    Recompiles the known method source with checked transformations. Progressive
    gets a private globals dictionary whose frontier factory returns corrected
    subclasses. Other instances and the original module's classes are unchanged.
    Returns an uninstaller. Repeated installation on one instance is idempotent.
    """
    if getattr(runner, "_audit_bound_uninstall", None) is not None:
        return runner._audit_bound_uninstall
    from build_corrected_runner import patch_suffix_source, patch_score_source
    module = sys.modules[type(runner).__module__]
    saved_suffix = runner._suffix_box_impl
    saved_progressive = runner.progressive

    def compile_method(method, transform):
        source = transform(textwrap.dedent(inspect.getsource(method)))
        namespace = dict(method.__globals__)
        namespace.update(exact_suffix_point_box=exact_suffix_point_box,
                         rms_box_corrected=rms_box_corrected,
                         attention_score_roundoff=attention_score_roundoff)
        exec(compile(source, "<audit-corrected-method>", "exec"), namespace)
        return namespace[method.__name__]

    suffix = compile_method(type(runner)._suffix_box_impl, patch_suffix_source)
    legacy_enclosure = compile_method(module.LegacyFrontier.enclosure, patch_score_source)
    tight_enclosure = compile_method(module.Frontier.enclosure, patch_score_source)
    fixed_legacy = type("AuditLegacyFrontier", (module.LegacyFrontier,), {"enclosure": legacy_enclosure})
    fixed_tight = type("AuditFrontier", (module.Frontier,), {"enclosure": tight_enclosure})

    def factory(cfg):
        original_class = module.frontier_class(cfg)
        return fixed_legacy if original_class is module.LegacyFrontier else fixed_tight

    fn = type(runner).progressive
    globals_copy = dict(fn.__globals__)
    globals_copy["frontier_class"] = factory
    progressive = types.FunctionType(fn.__code__, globals_copy, fn.__name__, fn.__defaults__, fn.__closure__)
    progressive.__kwdefaults__ = fn.__kwdefaults__
    runner._suffix_box_impl = types.MethodType(suffix, runner)
    runner.progressive = types.MethodType(progressive, runner)
    runner.audit_bound_fixes = "rea-v3-audit-corrected"

    def uninstall():
        runner._suffix_box_impl = saved_suffix
        runner.progressive = saved_progressive
        del runner._audit_bound_uninstall
        del runner.audit_bound_fixes

    runner._audit_bound_uninstall = uninstall
    return uninstall
