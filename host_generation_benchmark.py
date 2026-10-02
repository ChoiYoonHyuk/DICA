"""Whole-decoder continuation with a CPU source shared by embedding and head.

The backbone stays on CUDA. Every input-embedding row and output-projection row
comes from the same immutable CPU-pinned BF16 matrix. Dense and adaptive output
projections share the fixed row reference; cuBLAS is an independent control.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import time
import traceback
import textwrap
import types
import weakref


def parse_args():
    root=Path('/home/user/yhchoi')
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=root)
    p.add_argument('--runtime-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--plan',type=Path,default=root/'expanded_eval_20261002/longbench32/plan.json')
    p.add_argument('--family',choices=['qwen3','gemma4'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cap',type=int,default=4096)
    p.add_argument('--limit',type=int)
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeats',type=int,default=1)
    return p.parse_args()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    temp=Path(path).with_suffix('.partial')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


class Rank:
    def __init__(self,value):self.value=value
    def random(self):return self.value


def install_gemma_pad_transfer(backbone, callback):
    """Preserve decorators and change only the direct CPU pad-row transfer."""
    core=inspect.unwrap(type(backbone).forward)
    if core.__code__.co_freevars:raise ValueError('Unexpected Gemma forward closure')
    original=textwrap.dedent(inspect.getsource(core))
    tree=ast.parse(original)
    definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    definition.decorator_list=[]
    count=0
    for node in ast.walk(definition):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pad_embedding' for t in node.targets):
            expression=ast.unparse(node.value)
            expected='self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]'
            if expression!=expected:raise ValueError('Gemma pad-row expression changed: '+expression)
            node.value=ast.Call(func=ast.Name(id='_host_cpu_pad_row_transfer',ctx=ast.Load()),
                args=[node.value,ast.Name(id='inputs_embeds',ctx=ast.Load())],keywords=[])
            count+=1
    if count!=1:raise ValueError('Exactly one Gemma pad-row assignment is required')
    ast.fix_missing_locations(tree)
    transformed=ast.unparse(tree)
    namespace=dict(core.__globals__)
    exec(compile(tree,'<host_gemma_pad_transfer>','exec'),namespace)
    transformed_code=namespace[definition.name].__code__
    original_code=core.__code__
    if '_host_cpu_pad_row_transfer' in core.__globals__:raise RuntimeError('Pad transform already installed')
    core.__globals__['_host_cpu_pad_row_transfer']=callback
    core.__code__=transformed_code
    def restore():
        core.__code__=original_code
        del core.__globals__['_host_cpu_pad_row_transfer']
    return restore,dict(original_forward_sha256=hashlib.sha256(original.encode()).hexdigest(),
        transformed_forward_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
        modification='Only the existing unscaled CPU pad row is transferred to inputs_embeds.device; decorators and all remaining forward arithmetic are preserved',
        original_source=original,transformed_source=transformed)


def main():
    args=parse_args()
    if min(args.tokens,args.repeats,args.cap)<1:raise ValueError('Positive sizes required')
    sys.path[:0]=[str(Path(__file__).resolve().parent),str(args.runtime_root)]
    import numpy as np
    import torch
    import corrected_runner as r
    from host_projection import HostProjection,audit_dynamic_keys

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    plan=json.loads(args.plan.read_text())
    states=[s for s in plan['states'] if s['family']==args.family and s['context_cap']==args.cap]
    if args.limit is not None:states=states[:args.limit]
    if not states:raise ValueError('No planned documents')
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=r.Config(models=(states[0]['model'],),device='cuda:0',dtype='bfloat16',generate_tasks=False)
    model,tok,model_meta=r.load_checkpoint(states[0]['model'],cfg)
    runner=r.LastLayerRunner(model,tok,cfg)
    embedding=model.get_input_embeddings()
    weight=model.lm_head.weight
    if model.lm_head.bias is not None:raise ValueError('Bias-free output head required')
    if embedding.weight.data_ptr()!=weight.data_ptr() or embedding.weight.shape!=weight.shape:
        raise ValueError('The experiment requires tied input/output source weights')
    if not isinstance(embedding,torch.nn.Embedding) or embedding.max_norm is not None:
        raise ValueError('Only immutable direct lookup embeddings are admitted')
    if args.family=='gemma4' and (type(embedding).__name__!='Gemma4TextScaledWordEmbedding' or not hasattr(embedding,'embed_scale')):
        raise ValueError('Unexpected Gemma embedding arithmetic')
    if args.family=='qwen3' and type(embedding) is not torch.nn.Embedding:
        raise ValueError('Unexpected Qwen embedding arithmetic')
    original_weight_ref=weakref.ref(weight)
    original_pointer=weight.untyped_storage().data_ptr()
    original_embedding_source=inspect.getsource(type(embedding).forward)
    original_embedding_source_sha=hashlib.sha256(original_embedding_source.encode()).hexdigest()

    def sync():torch.cuda.synchronize()
    def tensor_sha(t):
        a=t.detach().cpu().contiguous().view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

    with torch.inference_mode():
        sync();memory_before=torch.cuda.memory_allocated();start=time.perf_counter()
        head=HostProjection(weight,softcap=runner.final_logit_softcap)
        sync();head_setup=time.perf_counter()-start
        # Independent native output table used exclusively after measured calls.
        # It permits comparison of every actual token ID after GPU W removal.
        start=time.perf_counter()
        native_embedding_table=torch.empty(weight.shape,dtype=torch.bfloat16,device='cpu')
        for first in range(0,head.summary.vocabulary,4096):
            last=min(first+4096,head.summary.vocabulary)
            public_ids=torch.arange(first,last,device=head.device,dtype=torch.int64).reshape(1,-1)
            original_output=embedding(public_ids)
            native_embedding_table[first:last].copy_(original_output.reshape(-1,head.summary.dimension))
        del public_ids,original_output
        sync();embedding_table_seconds=time.perf_counter()-start
        source_sha=tensor_sha(head.source)
        table_sha=tensor_sha(native_embedding_table)
        # Source CPU shape/dtype is genuine. Keep Gemma's existing GPU scale
        # buffer and apply exactly its original BF16 multiplication afterward.
        embedding.weight=torch.nn.Parameter(head.source,requires_grad=False)
        observed={'last_ids':None,'last_output':None,'calls':0,'rows':0,'bytes':0,
            'pad_rows':0,'pad_bytes':0,'pad_output':None,'pad_original':None}
        def host_embedding_forward(self,input_ids):
            head._check_source()
            if input_ids.device!=head.device or input_ids.dtype not in (torch.int32,torch.int64):
                raise ValueError('Embedding IDs changed the native device/dtype')
            cpu_ids=input_ids.reshape(-1).to(device='cpu',dtype=torch.int64,non_blocking=False)
            rows=torch.empty((cpu_ids.numel(),head.summary.dimension),dtype=torch.bfloat16,
                             device='cpu',pin_memory=True)
            torch.index_select(head.source,0,cpu_ids,out=rows)
            output=rows.to(head.device,non_blocking=True).reshape(*input_ids.shape,head.summary.dimension)
            if args.family=='gemma4':
                output=output*self.embed_scale.to(self.weight.dtype)
            observed['last_ids']=cpu_ids
            observed['last_output']=output
            observed['calls']+=1;observed['rows']+=cpu_ids.numel()
            observed['bytes']+=cpu_ids.numel()*head.summary.dimension*2
            return output
        embedding.forward=types.MethodType(host_embedding_forward,embedding)
        restore_pad=None;pad_transform=None
        if args.family=='gemma4':
            def transfer_pad(row,inputs_embeds):
                if row.device.type!='cpu' or row.dtype!=torch.bfloat16:raise ValueError('Pad source must be CPU BF16')
                output=row.to(inputs_embeds.device)
                observed['pad_rows']+=1;observed['pad_bytes']+=row.numel()*2
                observed['pad_original']=row;observed['pad_output']=output
                return output
            restore_pad,pad_transform=install_gemma_pad_transfer(runner.backbone,transfer_pad)
        model.lm_head=torch.nn.Identity()
        runner._final_logits=lambda hidden:hidden
        del weight
        gc.collect();torch.cuda.empty_cache();sync()
        if original_weight_ref() is not None:raise RuntimeError('Tied original GPU weight object remains reachable')
        aliases=[]
        for name,p in list(model.named_parameters(remove_duplicate=False))+list(model.named_buffers(remove_duplicate=False)):
            if p.is_cuda and p.untyped_storage().data_ptr()==original_pointer:aliases.append(name)
        if aliases:raise RuntimeError('GPU original source aliases remain: '+str(aliases))
        memory_after=torch.cuda.memory_allocated()
        checked_ids=set()
        def audit_embedding():
            ids=observed['last_ids'];actual=observed['last_output']
            if ids is None or actual is None:raise RuntimeError('No input embedding was observed')
            expected=native_embedding_table.index_select(0,ids)
            if not torch.equal(actual.detach().cpu().reshape_as(expected).view(torch.int16),expected.view(torch.int16)):
                raise ArithmeticError('Host-source input embedding differs from original native embedding bits')
            if observed['pad_output'] is not None and not torch.equal(observed['pad_output'].cpu().view(torch.int16),observed['pad_original'].view(torch.int16)):
                raise ArithmeticError('Direct pad-row transfer changed BF16 bits')
            checked_ids.update(map(int,ids.tolist()))
            return dict(rows=ids.numel(),bits_equal=True,ids_sha256=tensor_sha(ids))
        public=torch.tensor([[0,1,2,head.summary.vocabulary//2,head.summary.vocabulary-1]],device=head.device)
        embedding(public);public_audit=audit_embedding();del public
        key_audit=audit_dynamic_keys(head.device,head.summary.vocabulary)
        public_hidden=torch.zeros(head.summary.dimension,dtype=torch.bfloat16,device=head.device)
        sync();warm_start=time.perf_counter()
        for name in ('adaptive_host','row_dense_host','cublas_dense_host'):
            head.query(name,public_hidden,.375)
        sync();public_warmup_seconds=time.perf_counter()-warm_start
        del public_hidden
        current={'method':'adaptive_host'}
        def decode(self,ctx,source,method,policy,rng,**kwargs):
            hidden=self.dense_logits(ctx,source)
            result=head.query(current['method'],hidden,float(rng.random()))
            emission=r.Emission(result.token,source,probabilities=result.pmf,trace=[result.diagnostic])
            emission.host_cdf=result.cdf
            return emission
        runner.decode=types.MethodType(decode,runner)
        methods=['adaptive_host','row_dense_host','cublas_dense_host']
        policy=r.Policy('topk',50,.7);method=r.Method('dense')
        design=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            plan_sha256=sha(args.plan),model=model_meta,planned_documents=len(states),states=states,
            scripts={str(p.resolve()):sha(p) for folder in (Path(__file__).parent,args.runtime_root) for p in sorted(folder.glob('*.py'))},
            timing_scope='whole one-token decoder forward and KV append, CPU input-embedding gather and H2D transfer, CPU-source output projection, policy, full-vocabulary deterministic CDF and 53-bit draw',
            restoration_and_postoutput_validation_outside_timer=True,
            input_embedding_and_output_head_share_cpu_source=True,backbone_device='cuda',
            tied_gpu_parameter_weakref_dead=original_weight_ref() is None,remaining_original_gpu_aliases=aliases,
            gpu_allocated_bytes_before_summary=memory_before,gpu_allocated_bytes_after_placement=memory_after,
            source_sha256=source_sha,head_setup_seconds=head_setup,head=head.description(),
            native_embedding_audit_table_seconds=embedding_table_seconds,
            native_embedding_audit_table_sha256=table_sha,
            native_embedding_audit_table_cpu_bytes=native_embedding_table.numel()*2,
            native_embedding_audit_table_used_in_forward=False,
            original_embedding_forward_sha256=original_embedding_source_sha,
            pad_row_transform=pad_transform,dynamic_key_audit=key_audit,
            public_hidden_compilation_and_warmup_seconds=public_warmup_seconds,
            query_dependent_warmup=False,
            embedding_scale_preserved=hasattr(embedding,'embed_scale'),public_embedding_audit=public_audit,
            reference='shared fixed row projection and finite BF16 cap table; stable top-50, temperature 0.7, deterministic full-vocabulary FP64 CDF and uniform 53-bit grid',
            native_cublas_is_separate_control=True,gpu_resident_speedup_claim=False,
            trajectory='only the shared-reference adaptive continuation is committed',
            eos_terminates=True,task_score_evaluated=False,kv_read_reduction_percent=0,outcomes=[])
        save(args.output/'design.json',design)
        for document,state in enumerate(states):
            row=dict(state=state,status='running',steps=[])
            try:
                payload=json.loads((args.plan.parent/state['token_file']).read_text())
                ids=payload['input_ids']
                digest=hashlib.sha256(json.dumps(ids,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                if digest!=state['token_sha256'] or len(ids)!=state['N']:raise ValueError('Prefix changed')
                row['prefix_preparation']=runner.prepare(ids,method,17,args.tokens+1)
                row['prefix_embedding_audit']=audit_embedding()
                rng=np.random.default_rng(20261031+document);order_rng=random.Random(20261101+document)
                previous=ids[-1];tokens=[]
                for step in range(args.tokens):
                    length=int(runner.state.get_seq_length());snapshot=runner.cache_snapshot()
                    def restore():
                        if snapshot is None:runner.crop(length)
                        else:runner.restore_cache_snapshot(snapshot)
                    uniform=float(rng.random())
                    timings=[];outputs={};embedding_checks=[]
                    for repeat in range(args.repeats):
                        order=methods.copy();order_rng.shuffle(order)
                        for name in order:
                            current['method']=name
                            before_bytes=observed['bytes'];before_rows=observed['rows']
                            before_pad_bytes=observed['pad_bytes'];before_pad_rows=observed['pad_rows']
                            sync();started=time.perf_counter()
                            emitted=runner.step(previous,method,policy,Rank(uniform))
                            sync();elapsed=(time.perf_counter()-started)*1000
                            outputs[name]=emitted
                            timings.append(dict(method=name,repeat=repeat,ms=elapsed,token=int(emitted.token),
                                input_embedding_rows=observed['rows']-before_rows,
                                input_embedding_bytes_host_to_device=observed['bytes']-before_bytes,
                                direct_pad_rows=observed['pad_rows']-before_pad_rows,
                                direct_pad_bytes_host_to_device=observed['pad_bytes']-before_pad_bytes,
                                output_weight_bytes_host_to_device=emitted.trace[0]['original_weight_bytes_host_to_device']))
                            embedding_checks.append(audit_embedding());restore()
                    adaptive=outputs['adaptive_host'];dense=outputs['row_dense_host'];native=outputs['cublas_dense_host']
                    pmf_equal=torch.equal(adaptive.probabilities.view(torch.int64),dense.probabilities.view(torch.int64))
                    cdf_equal=torch.equal(adaptive.host_cdf.view(torch.int64),dense.host_cdf.view(torch.int64))
                    if not pmf_equal or not cdf_equal or adaptive.token!=dense.token:raise ArithmeticError('Whole-decoder shared categorical law differs')
                    item=dict(index=step,prefix_length=length+1,timings=timings,pmf_bits_equal=pmf_equal,
                        cdf_bits_equal=cdf_equal,coupled_token_equal=adaptive.token==dense.token,
                        first_real_query_in_paired_timing=True,uniform_rank=int(uniform*(1<<53)),embedding_audits=embedding_checks,
                        native_pmf_tv=float(.5*(native.probabilities-dense.probabilities).abs().sum()),
                        diagnostic=adaptive.trace[0],token=int(adaptive.token))
                    current['method']='adaptive_host'
                    committed=runner.step(previous,method,policy,Rank(uniform));audit_embedding()
                    if committed.token!=item['token'] or not torch.equal(committed.host_cdf,dense.host_cdf):
                        raise ArithmeticError('Committed continuation changed')
                    row['steps'].append(item);previous=int(committed.token);tokens.append(previous)
                    if previous in r.eos_ids(runner):break
                row.update(status='ok',tokens=tokens,actual_tokens=len(tokens),continuation=tok.decode(tokens,skip_special_tokens=True))
                row['medians_ms']={name:float(np.median([m['ms'] for step in row['steps'] for m in step['timings'] if m['method']==name])) for name in methods}
            except Exception as error:
                row.update(status='failed',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            save(args.output/(state['state_id']+'.json'),row)
            design['outcomes'].append(dict(state_id=state['state_id'],status=row['status']))
            save(args.output/'design.json',design)
            print(json.dumps(dict(state=state['state_id'],status=row['status'],tokens=row.get('actual_tokens'),
                ms=row.get('medians_ms'),error=row.get('error',{}).get('message'))),flush=True)
        design.update(complete=len(design['outcomes'])==len(states),
            successful_documents=sum(x['status']=='ok' for x in design['outcomes']),
            failed_documents=sum(x['status']=='failed' for x in design['outcomes']),
            embedding_unique_ids_verified=len(checked_ids),embedding_calls=observed['calls'],
            embedding_rows=observed['rows'],source_sha256_after_queries=tensor_sha(head.source),head=head.description())
        design['direct_pad_rows']=observed['pad_rows'];design['direct_pad_bytes_host_to_device']=observed['pad_bytes']
        design['source_unchanged']=design['source_sha256_after_queries']==source_sha
        save(args.output/'design.json',design)
        runner.close()
        if restore_pad is not None:restore_pad()
        if not design['source_unchanged']:raise ArithmeticError('Shared CPU source changed')
        if design['failed_documents']:raise SystemExit(2)


if __name__=='__main__':main()
