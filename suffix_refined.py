"""Finite-image certificates for the actual deterministic native suffix.

The suffix begins by casting attention to the model dtype. A real input box can
therefore have a singleton (or small finite) native image even when lo != hi.
We enumerate that COMPLETE image, never random corners, and evaluate every
candidate with the actual suffix at its original batch/sequence shape.

No interval arithmetic error allowance is needed inside these evaluations:
the target is the deterministic finite-precision suffix itself. Soundness is
conditional on the input attention enclosure actually enclosing the native
attention result, and on suffix determinism. This module does not certify an
SDPA enclosure or make the old numerical CUDA allowance into a formal proof.

Enumeration is capped BEFORE executing the suffix. A large image returns None,
so callers can refine the source or fall back to their existing enclosure. It
must never be truncated to the first candidates and called a certificate.
"""
from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass
from typing import Callable, Optional


def bits_to_ordered(bits: int, width: int) -> int:
    """IEEE bit pattern -> monotone total-order index; preserves signed zero."""
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    if not 0 <= bits <= mask:
        raise ValueError("bit pattern outside the declared format")
    return (~bits & mask) if bits & sign else (bits ^ sign)


def ordered_to_bits(index: int, width: int) -> int:
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    if not 0 <= index <= mask:
        raise ValueError("ordered index outside the declared format")
    return (index ^ sign) if index & sign else (~index & mask)


def bfloat16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def float_to_bfloat16_bits(value: float) -> int:
    """Round a binary64 scalar to BF16 via binary32, as torch's usual cast.

    Only a test helper: runtime endpoints use the actual torch dtype cast.
    """
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    if raw & 0x7F800000 == 0x7F800000:
        return (raw >> 16) | (0x40 if raw & 0x7FFFFF else 0)
    return ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 0xFFFF


def capped_product(counts, cap: int) -> Optional[int]:
    if cap < 1:
        raise ValueError("max_images must be positive")
    result = 1
    for count in counts:
        if count < 1:
            raise ValueError("empty or reversed representable interval")
        if result > cap // count:
            return None
        result *= count
    return result


def bfloat16_closed_interval_bits(lower: float, upper: float, max_values: int = 65536):
    """All BF16 patterns inside a closed interval with BF16 endpoints.

    A useful source-side alternative to an attention enclosure: if an unread
    stored BF16 scalar is known from public metadata to lie in this interval,
    exhaustive native completion over this list needs no SDPA monotonicity or
    arithmetic-error theorem. Its source saving is only one scalar and its
    compute cost may be large; this function makes no efficiency claim.

    The endpoints MUST already be exactly representable BF16 numbers. To cover
    arbitrary real bounds one must round lower downward / upper upward first.
    """
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ArithmeticError("Need ordered finite BF16 endpoints")
    blo, bhi = float_to_bfloat16_bits(lower), float_to_bfloat16_bits(upper)
    if bfloat16_bits_to_float(blo) != lower or bfloat16_bits_to_float(bhi) != upper:
        raise ValueError("Endpoints must be exactly representable in BF16")
    lo = bits_to_ordered(blo, 16)
    hi = bits_to_ordered(bhi, 16)
    lo, hi = min(lo, hi), max(lo, hi)
    if capped_product([hi - lo + 1], max_values) is None:
        return None
    return [ordered_to_bits(index, 16) for index in range(lo, hi + 1)]


@dataclass
class DiscreteImageStats:
    uncertain_coordinates: int
    image_count: Optional[int]
    max_images: int
    suffix_evaluations: int = 0
    mode: str = "refine_required"


@dataclass
class DiscreteSuffixResult:
    box: object
    stats: DiscreteImageStats
    probability_lower: object = None
    probability_upper: object = None


def _format(torch, dtype):
    if dtype == torch.bfloat16:
        return 16, torch.int16
    if dtype == torch.float16:
        return 16, torch.int16
    if dtype == torch.float32:
        return 32, torch.int32
    if dtype == torch.float64:
        return 64, torch.int64
    raise TypeError(f"Unsupported native floating dtype {dtype}")


def quantized_image_spec(attention_box, dtype, max_images: int = 64):
    """Return complete native input candidates' specification, not evaluations.

    Returned tuple: (native lower endpoint, integer dtype, coordinates, ranges,
    stats). If stats.image_count is None, the product exceeds max_images.
    Endpoint casts retain every attainable native value by monotonicity.
    Signed zero is retained because it can affect arbitrary native operations.
    """
    import torch

    lo, hi = attention_box.lo, attention_box.hi
    if lo.shape != hi.shape or not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all()):
        raise ArithmeticError("Attention enclosure must have matching finite endpoints")
    if bool(torch.any(lo > hi)):
        raise ArithmeticError("Reversed attention interval")
    width, integer_dtype = _format(torch, dtype)
    qlo, qhi = lo.to(dtype).contiguous(), hi.to(dtype).contiguous()
    if not bool(torch.isfinite(qlo).all() and torch.isfinite(qhi).all()):
        raise ArithmeticError("Attention cast overflows; finite image unavailable")
    iblo = qlo.view(integer_dtype).reshape(-1).cpu().tolist()
    ibhi = qhi.view(integer_dtype).reshape(-1).cpu().tolist()
    mask = (1 << width) - 1
    coordinates, ranges, counts = [], [], []
    for coordinate, (blo, bhi) in enumerate(zip(iblo, ibhi)):
        lower = bits_to_ordered(blo & mask, width)
        upper = bits_to_ordered(bhi & mask, width)
        # Real arithmetic compares -0 and +0 equal. An explicitly reversed
        # zero encoding is the same real interval and requires both images.
        lower, upper = min(lower, upper), max(lower, upper)
        if lower != upper:
            coordinates.append(coordinate)
            ranges.append(range(lower, upper + 1))
            counts.append(upper - lower + 1)
    count = capped_product(counts, max_images)
    stats = DiscreteImageStats(len(coordinates), count, max_images)
    return qlo, integer_dtype, coordinates, ranges, stats


def exact_native_suffix_image(runner, attention_box, h, per_layer_input=None,
                             *, max_images: int = 64,
                             probability_fn: Optional[Callable] = None,
                             stats_callback: Optional[Callable] = None):
    """Enclose ALL native suffix outputs of a sufficiently small cast image.

    probability_fn(logits) optionally supplies a policy measure: normalized
    probabilities OR exact integer categorical-grid counts in [0,2**53]. The
    latter remain exact through the internal float64 conversion and minima;
    callers convert the returned probability_lower field back to int64 before
    sampling. Its coordinatewise minimum over all images is a tighter common
    measure than forming independent logit bounds and then normalizing.
    No candidate logits/PMFs are stored after their running min/max update.
    """
    import torch

    base, integer_dtype, coordinates, ranges, stats = quantized_image_spec(
        attention_box, runner.dtype, max_images)
    if stats.image_count is None:
        if stats_callback is not None:
            stats_callback(stats)
        return None
    width, _ = _format(torch, runner.dtype)
    modulus, sign = 1 << width, 1 << (width - 1)
    native_bits = base.view(integer_dtype).reshape(-1)
    logit_lo = logit_hi = probability_lo = probability_hi = None
    with torch.inference_mode():
        for indices in itertools.product(*ranges):
            # Do not batch candidate suffix calls: changed shapes could choose
            # a different matmul kernel and define a different native target.
            candidate_bits = native_bits.clone()
            for coordinate, index in zip(coordinates, indices):
                raw = ordered_to_bits(index, width)
                candidate_bits[coordinate] = raw - modulus if raw >= sign else raw
            candidate = candidate_bits.view(runner.dtype).reshape(base.shape)
            logits = runner.suffix(candidate, h, per_layer_input).reshape(1, -1).double()
            if not bool(torch.isfinite(logits).all()):
                raise ArithmeticError("Non-finite native suffix image")
            if logit_lo is None:
                logit_lo, logit_hi = logits.clone(), logits.clone()
            else:
                logit_lo = torch.minimum(logit_lo, logits)
                logit_hi = torch.maximum(logit_hi, logits)
            if probability_fn is not None:
                p = probability_fn(logits.reshape(-1)).double()
                if not bool(torch.isfinite(p).all()) or bool(torch.any(p < 0)):
                    raise ArithmeticError("Invalid policy probability image")
                if probability_lo is None:
                    probability_lo, probability_hi = p.clone(), p.clone()
                else:
                    probability_lo = torch.minimum(probability_lo, p)
                    probability_hi = torch.maximum(probability_hi, p)
            stats.suffix_evaluations += 1
    stats.mode = "native_cast_singleton" if stats.image_count == 1 else "native_finite_image"
    if stats_callback is not None:
        stats_callback(stats)
    return DiscreteSuffixResult(type(attention_box)(logit_lo, logit_hi), stats,
                                probability_lo, probability_hi)


def exact_native_suffix_joint_columns(runner, attention_box, metadata, h,
                                      per_layer_input=None, *, max_images=256,
                                      probability_fn=None, stats_callback=None):
    """Use the source sweep's COMPLETE joint head images for each V column.

    Metadata comes from native_source._sweep_native_domains with
    capture_joint_columns=True. Unlike a coordinate box, each alternative
    retains correlations between all GQA heads sharing an unknown V scalar.
    Different V columns can still vary independently. The source provider is
    responsible for exhaustive domains and the PV-column separation contract.
    probability_fn may return stacked policy probabilities or stacked integer
    grid counts <=2**53; the historically named probability_lower/upper fields
    enclose that supplied measure. No normalization is performed here.
    """
    import torch

    required = ("joint_column_images", "joint_group", "joint_repeats")
    if any(name not in metadata for name in required):
        raise ValueError("Complete joint-column source images are required")
    group, repeats = int(metadata["joint_group"]), int(metadata["joint_repeats"])
    width = attention_box.lo.shape[-1]
    base = attention_box.lo.to(runner.dtype).clone()
    flat = base.reshape(-1, width)
    if repeats < 1 or group < 0 or (group + 1) * repeats > flat.shape[0]:
        raise ValueError("Invalid joint GQA image dimensions")
    images, counts, uncertain = [], [], 0
    for col_string, possibilities in sorted(metadata["joint_column_images"].items(),
                                             key=lambda item: int(item[0])):
        col = int(col_string)
        values = torch.tensor(possibilities, dtype=runner.dtype, device=base.device)
        if not 0 <= col < width or values.ndim != 2 or values.shape[1] != repeats or values.shape[0] == 0:
            raise ValueError("Malformed joint native image")
        if not bool(torch.isfinite(values).all()):
            raise ArithmeticError("Non-finite joint native image")
        images.append((col, values))
        counts.append(values.shape[0])
        uncertain += int((values.min(0).values != values.max(0).values).sum())
    count = capped_product(counts, max_images)
    stats = DiscreteImageStats(uncertain, count, max_images)
    if count is None:
        if stats_callback is not None:
            stats_callback(stats)
        return None
    low = high = plo = phi = None
    with torch.inference_mode():
        for selection in itertools.product(*(range(c) for c in counts)):
            candidate = base.clone()
            candidate_heads = candidate.reshape(-1, width)
            for (col, values), index in zip(images, selection):
                candidate_heads[group * repeats:(group + 1) * repeats, col] = values[index]
            logits = runner.suffix(candidate, h, per_layer_input).reshape(1, -1).double()
            if not bool(torch.isfinite(logits).all()):
                raise ArithmeticError("Non-finite native suffix image")
            low = logits.clone() if low is None else torch.minimum(low, logits)
            high = logits.clone() if high is None else torch.maximum(high, logits)
            if probability_fn is not None:
                p = probability_fn(logits.reshape(-1)).double()
                if not bool(torch.isfinite(p).all()) or bool(torch.any(p < 0)):
                    raise ArithmeticError("Invalid policy probability image")
                plo = p.clone() if plo is None else torch.minimum(plo, p)
                phi = p.clone() if phi is None else torch.maximum(phi, p)
            stats.suffix_evaluations += 1
    stats.mode = "native_joint_column_image"
    if stats_callback is not None:
        stats_callback(stats)
    return DiscreteSuffixResult(type(attention_box)(low, high), stats, plo, phi)


def install_discrete_suffix(runner, *, max_images: int = 64, records=None):
    """Instance-local optional refinement before an existing suffix enclosure.

    This keeps the existing numerical enclosure as fallback; its limitations
    remain. Positive certificates produced through fallback are not promoted
    to finite-image guarantees. The records list exposes which path was used.
    """
    import types

    original = runner._suffix_box_impl
    if records is None:
        records = []

    def wrapped(self, attention_box, h, tight, operator_trace=None, per_layer_input=None):
        found = exact_native_suffix_image(
            self, attention_box, h, per_layer_input, max_images=max_images,
            stats_callback=records.append)
        if found is None:
            return original(attention_box, h, tight, operator_trace, per_layer_input)
        if operator_trace is not None:
            operator_trace.append((found.stats.mode, found.box))
        return found.box

    runner._suffix_box_impl = types.MethodType(wrapped, runner)

    def uninstall():
        runner._suffix_box_impl = original

    return uninstall, records


def load_suffix_only(model_path, audit_dir=None, device="cuda"):
    """Load the checkpoint's actual last block, final norm and tied LM head.

    Constructs the same Transformers module classes on meta, then assigns
    selected stored checkpoint tensors. Earlier decoder blocks are never
    allocated. Native suffix parity must be checked against a full-model
    capture before this loader is used for research conclusions.
    """
    from pathlib import Path
    import importlib
    import os
    import sys
    from types import SimpleNamespace
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    # Match corrected_runner.load_checkpoint's numerical backend contract.
    # In particular the default BF16 reduction flag changes native logits.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device).index or 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if audit_dir is None:
        parent = Path(__file__).resolve().parent.parent
        audit_dir = next((p for p in (parent / "rea_audit_20261002",
                                    parent / "20261002_rea_audit") if p.exists()), None)
    if audit_dir is None:
        raise FileNotFoundError("Pass audit_dir containing corrected_runner.py")
    if str(audit_dir) not in sys.path:
        sys.path.insert(0, str(audit_dir))
    r = importlib.import_module("corrected_runner")
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    text = getattr(config, "text_config", config)
    layer_index = text.num_hidden_layers - 1
    family = "qwen3" if text.model_type == "qwen3" else "gemma4" if text.model_type == "gemma4_text" else None
    if family is None:
        raise ValueError(f"Unsupported suffix architecture {text.model_type}")
    if family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
        layer_type, norm_type, prefix = Qwen3DecoderLayer, Qwen3RMSNorm, "model."
    else:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4RMSNorm
        layer_type, norm_type, prefix = Gemma4TextDecoderLayer, Gemma4RMSNorm, "model.language_model."
    with torch.device("meta"):
        block = layer_type(text, layer_index)
        norm = norm_type(text.hidden_size, eps=text.rms_norm_eps)
        head = torch.nn.Linear(text.hidden_size, text.vocab_size, bias=False)
    block_prefix = f"{prefix}layers.{layer_index}."
    wanted = {block_prefix + name for name in block.state_dict()}
    wanted.add(prefix + "norm.weight")
    wanted.add("lm_head.weight")
    wanted.add(prefix + "embed_tokens.weight")
    selected = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in wanted.intersection(handle.keys()):
                selected[name] = handle.get_tensor(name)
    block.load_state_dict({name[len(block_prefix):]: value for name, value in selected.items()
                           if name.startswith(block_prefix)}, strict=True, assign=True)
    norm.load_state_dict({"weight": selected[prefix + "norm.weight"]}, strict=True, assign=True)
    if "lm_head.weight" in selected:
        head_weight = selected["lm_head.weight"]
    elif bool(getattr(config, "tie_word_embeddings", getattr(text, "tie_word_embeddings", False))):
        head_weight = selected[prefix + "embed_tokens.weight"]
    else:
        raise KeyError("Neither explicit LM head nor declared tied embeddings found")
    head.load_state_dict({"weight": head_weight}, strict=True, assign=True)
    block, norm, head = block.to(device).eval(), norm.to(device).eval(), head.to(device).eval()
    runner = object.__new__(r.LastLayerRunner)
    runner.model = SimpleNamespace(lm_head=head)
    runner.text_model = SimpleNamespace(norm=norm, layers=[block])
    runner.block, runner.attention = block, block.self_attn
    runner.device, runner.dtype = head.weight.device, head.weight.dtype
    runner.family, runner.text_config, runner.last_layer = family, text, layer_index
    runner.activation_name = str(getattr(text, "hidden_act", getattr(text, "hidden_activation", "silu")))
    cap = getattr(text, "final_logit_softcapping", None)
    runner.final_logit_softcap = None if cap is None else float(cap)
    runner.cfg = r.Config(certificate_backend="tight")
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    runner.native_attention = sdpa_attention_forward
    runner.suffix_checkpoint_contract = "selected original checkpoint tensors; parity requires capture check"
    return runner


def _probe_main():
    import argparse
    import dataclasses
    import json
    from pathlib import Path
    import torch

    parser = argparse.ArgumentParser(description="Check suffix-only loader and finite native image")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--image-file")
    parser.add_argument("--audit-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()
    runner = load_suffix_only(args.model_path, args.audit_dir, args.device)
    ctx = torch.load(args.capture, map_location=args.device, weights_only=False)
    h, ple = ctx["h"], ctx.get("per_layer_input")
    reference = ctx["evaluation_only"]
    with torch.inference_mode():
        actual = runner.suffix(reference["attention"], h, ple)
    target = reference["logits"].reshape(-1)
    result = dict(model=args.model_path, capture=args.capture,
                  suffix_bitwise_equal=bool(torch.equal(actual, target)),
                  max_suffix_difference=float((actual.double() - target.double()).abs().max()),
                  dtype=str(runner.dtype))
    if not result["suffix_bitwise_equal"]:
        raise ArithmeticError(f"Suffix-only loader does not match captured native target: {result}")
    if args.image_file:
        payload = torch.load(args.image_file, map_location=args.device, weights_only=False)
        # This file must be produced without using the evaluation-only native
        # attention/logits. They are inspected only after the certificate.
        @dataclass
        class Box:
            lo: object
            hi: object
        import corrected_runner as r
        box = Box(payload["lo"], payload["hi"])
        policy = r.Policy("topk", 50, 0.7)
        found = exact_native_suffix_image(runner, box, h, ple, max_images=args.max_images,
                                          probability_fn=lambda logits: r.policy_prob(logits, policy))
        if found is None:
            result["finite_image"] = dataclasses.asdict(quantized_image_spec(box, runner.dtype, args.max_images)[-1])
        else:
            target_p = r.policy_prob(target, policy)
            result.update(finite_image=dataclasses.asdict(found.stats),
                          certificate_mass=float(found.probability_lower.sum()),
                          max_probability_excess=float((found.probability_lower - target_p).max()),
                          target_logits_enclosed=bool(((target.double() >= found.box.lo)
                                                        & (target.double() <= found.box.hi)).all()))
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    # dataclass is already imported above; the CLI imports dependencies lazily.
    _probe_main()
