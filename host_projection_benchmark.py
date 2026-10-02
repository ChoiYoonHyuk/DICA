"""Paired host-source conditional-head benchmark, with every state retained."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import weakref


def arguments():
    root = Path('/home/user/yhchoi')
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=root)
    p.add_argument('--runtime-root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--inputs', type=Path, default=root/'policy_gap_20261002/inputs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--family', choices=['qwen3', 'gemma4'], required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261101)
    return p.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    pending = path.with_suffix('.partial')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def main():
    args = arguments()
    if args.repeats < 1 or args.warmup < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError('Invalid repetition or prefix count')
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(args.runtime_root)]
    import numpy as np
    import torch
    from suffix_refined import load_suffix_only
    from exact_grid_sampler import grid_counts_from_cdf
    from host_projection import HostProjection, audit_dynamic_keys

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.inputs/(args.family+'_inputs.json')
    metadata = json.loads(metadata_path.read_text())
    hidden_path = args.inputs/(args.family+'_hidden.pt')
    if sha(hidden_path) != metadata['hidden_file_sha256']:
        raise ValueError('Hidden input file changed')
    hidden = torch.load(hidden_path, map_location='cuda', weights_only=True)
    records = metadata['records'][:args.limit] if args.limit is not None else metadata['records']
    if hidden.shape[0] != len(metadata['records']):
        raise ValueError('Hidden count and planned state count differ')
    if len({r['state']['state_id'] for r in records}) != len(records):
        raise ValueError('Duplicate state identity')

    def tensor_sha(t):
        array = t.detach().cpu().contiguous().view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(array).cast('B')).hexdigest()

    def sync():
        torch.cuda.synchronize()

    def timed_call(method, h, uniform):
        sync()
        start = time.perf_counter()
        result = head.query(method, h, uniform)
        sync()
        elapsed = 1000*(time.perf_counter()-start)
        return result, elapsed

    methods = ['adaptive_host', 'row_dense_host', 'cublas_dense_host']
    runner = load_suffix_only(metadata['model'], audit_dir=Path(__file__).resolve().parent)
    if runner.model.lm_head.bias is not None:
        raise ValueError('The native head control requires a bias-free projection')
    original_reference = weakref.ref(runner.model.lm_head.weight)
    with torch.inference_mode():
        sync()
        memory_before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        head = HostProjection(runner.model.lm_head.weight, softcap=runner.final_logit_softcap)
        sync()
        setup_total = time.perf_counter()-start
        start = time.perf_counter()
        restored = head.source.to(device=head.device, non_blocking=True)
        copy_bits_equal = torch.equal(restored.view(torch.int16), runner.model.lm_head.weight.view(torch.int16))
        del restored
        if not copy_bits_equal:
            raise ArithmeticError('Pinned CPU source changed original BF16 bits')
        # Delete the entire suffix model, including any tied aliases. A bound
        # native logits method is deliberately never retained.
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        sync()
        release_and_copy_audit_seconds = time.perf_counter()-start
        if original_reference() is not None:
            raise RuntimeError('The original GPU weight Parameter remains reachable')
        if head.summary.weight.device.type != 'cpu':
            raise RuntimeError('Summary retained the original GPU source')
        memory_after = torch.cuda.memory_allocated()
        start = time.perf_counter()
        source_hash_before = tensor_sha(head.source)
        source_hash_seconds = time.perf_counter()-start
        key_audit = audit_dynamic_keys(head.device, head.summary.vocabulary)
        # Public query-independent zero hidden vector compiles all methods.
        # This warmup never selects candidates using an evaluation prefix.
        synthetic = torch.zeros(head.summary.dimension, dtype=torch.bfloat16, device=head.device)
        sync()
        start = time.perf_counter()
        for _ in range(args.warmup):
            for method in methods:
                warmed = head.query(method, synthetic, .375)
                del warmed
        sync()
        warmup_seconds = time.perf_counter()-start
        del synthetic
        gc.collect()
        torch.cuda.empty_cache()
        sync()
        source_files = {str(p.resolve()): sha(p) for folder in (Path(__file__).parent, args.runtime_root)
                        for p in sorted(folder.glob('*.py'))}
        design = dict(family=args.family, planned_states=len(records), repeats=args.repeats,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            records=records, input_metadata_sha256=sha(metadata_path), hidden_file_sha256=sha(hidden_path),
            script_hashes=source_files,
            timing_scope='GPU hidden vector to CPU token, CPU original weight source for all three methods; allocations, candidate selection, CPU gathering, PCIe transfers, exact projection, full-vocabulary FP64 policy/CDF, and draw included',
            full_decoder_timing=False, gpu_resident_weight_speedup_claim=False,
            embedding_placement_claim=False, primary_before_dense=True,
            reference='Shared fixed row-separable projection and BF16 cap table; stable top-50 at temperature 0.7; deterministic full-vocabulary FP64 CDF and uniform 53-bit draw',
            cublas_control='Independent full-weight CPU transfer followed by original bias-free torch linear and BF16 softcap chain; its arithmetic law is compared, not assumed identical',
            weight_copy_bits_equal=copy_bits_equal, original_gpu_parameter_weakref_dead=original_reference() is None,
            summary_weight_device=str(head.summary.weight.device), source_sha256=source_hash_before,
            setup_total_seconds=setup_total, release_and_copy_audit_seconds=release_and_copy_audit_seconds,
            source_hash_seconds=source_hash_seconds, compilation_and_warmup_seconds=warmup_seconds,
            dynamic_key_audit=key_audit,
            gpu_allocated_bytes_before_summary=memory_before,
            gpu_allocated_bytes_after_model_release=memory_after,
            gpu_allocated_bytes_before_queries=torch.cuda.memory_allocated(),
            host=head.description(), torch_version=torch.__version__,
            device=torch.cuda.get_device_name(), outcomes=[])
        save(args.output/'design.json', design)
        for index, record in enumerate(records):
            identity = record['state']['state_id']
            row = dict(state=record['state'], status='running', hidden_sha256=record['hidden_sha256'],
                       primary_before_dense=True, timings=[])
            try:
                h = hidden[index]
                if tensor_sha(h) != record['hidden_sha256']:
                    raise ValueError('Per-state hidden hash differs')
                uniform_rng = np.random.default_rng(args.seed+index)
                uniform = float(uniform_rng.random())
                adaptive, elapsed = timed_call('adaptive_host', h, uniform)
                row['primary_ms'] = elapsed
                row['diagnostic'] = adaptive.diagnostic
                dense, dense_ms = timed_call('row_dense_host', h, uniform)
                native, native_ms = timed_call('cublas_dense_host', h, uniform)
                row['primary_controls_ms'] = dict(row_dense_host=dense_ms, cublas_dense_host=native_ms)
                selected_equal = torch.equal(adaptive.logits.view(torch.int16),
                                             dense.logits[adaptive.ids].view(torch.int16))
                pmf_equal = torch.equal(adaptive.pmf.view(torch.int64), dense.pmf.view(torch.int64))
                cdf_equal = torch.equal(adaptive.cdf.view(torch.int64), dense.cdf.view(torch.int64))
                dense_counts = grid_counts_from_cdf(dense.cdf)
                adaptive_counts = grid_counts_from_cdf(adaptive.cdf)
                integer_equal = torch.equal(dense_counts, adaptive_counts)
                if not all((selected_equal, pmf_equal, cdf_equal, integer_equal, adaptive.token == dense.token)):
                    raise ArithmeticError('CPU-source screening changed the shared row categorical reference')
                row['validation'] = dict(selected_projection_bits_equal=selected_equal,
                    pmf_bits_equal=pmf_equal, cdf_bits_equal=cdf_equal, integer_counts_equal=integer_equal,
                    coupled_primary_token_equal=adaptive.token == dense.token,
                    categorical_counts_sha256=tensor_sha(dense_counts),
                    pmf_sha256=tensor_sha(dense.pmf), cdf_sha256=tensor_sha(dense.cdf),
                    native_projection_bit_differences=int((native.logits.view(torch.int16)!=dense.logits.view(torch.int16)).sum()),
                    native_projection_max_absolute_difference=float((native.logits.double()-dense.logits.double()).abs().max()),
                    native_pmf_total_variation=float(.5*(native.pmf-dense.pmf).abs().sum()),
                    native_integer_total_variation=float((grid_counts_from_cdf(native.cdf)-dense_counts).abs().sum().double()/(2*(1<<53))))
                del adaptive, dense, native, dense_counts, adaptive_counts
                order_rng = random.Random(args.seed+100000+index)
                repeat_agreement = 0
                for repeat in range(args.repeats):
                    order = methods.copy()
                    order_rng.shuffle(order)
                    uniform = float(uniform_rng.random())
                    tokens = {}
                    for method in order:
                        result, elapsed = timed_call(method, h, uniform)
                        row['timings'].append(dict(method=method, repeat=repeat, ms=elapsed,
                            token=result.token, uniform_rank=int(uniform*(1<<53)),
                            selected_rows=result.diagnostic['selected_rows'],
                            original_weight_bytes_host_to_device=result.diagnostic['original_weight_bytes_host_to_device']))
                        tokens[method] = result.token
                        del result
                    if tokens['adaptive_host'] != tokens['row_dense_host']:
                        raise ArithmeticError('Paired timed draw differs')
                    repeat_agreement += 1
                row['validation']['paired_repeats_equal'] = repeat_agreement
                row['medians_ms'] = {method:float(np.median([x['ms'] for x in row['timings'] if x['method']==method])) for method in methods}
                row['status'] = 'ok'
            except Exception as error:
                row['status'] = 'failed'
                row['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                # Retain the failed prefix and proceed without changing the plan.
                gc.collect()
                torch.cuda.empty_cache()
            save(args.output/(identity+'.json'), row)
            design['outcomes'].append(dict(state_id=identity, status=row['status'], file=identity+'.json'))
            save(args.output/'design.json', design)
            print(json.dumps(dict(state=identity, status=row['status'],
                selected=row.get('diagnostic',{}).get('selected_rows'),
                ms=row.get('medians_ms'), error=row.get('error',{}).get('message'))), flush=True)
        hash_after = tensor_sha(head.source)
        design['source_sha256_after_queries'] = hash_after
        design['source_unchanged'] = hash_after == source_hash_before
        design['host'] = head.description()
        design['complete'] = len(design['outcomes']) == len(records)
        design['successful_states'] = sum(x['status']=='ok' for x in design['outcomes'])
        design['failed_states'] = sum(x['status']=='failed' for x in design['outcomes'])
        save(args.output/'design.json', design)
        if not design['source_unchanged']:
            raise ArithmeticError('Immutable source hash changed')
        if design['failed_states']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
