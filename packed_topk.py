"""Exact BF16 value/token-index ordering with fixed-shape integer top-k.

Finite BF16 logits and negative-infinity padding are supported. Negative and
positive zero have one numeric rank. Inputs with NaN are outside this contract.
Token IDs must be in [0,vocab_size); callers generating candidates enforce this
without a query-dependent host synchronization. Duplicate padded IDs are legal.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _encode_keys(LOGITS, TOKEN_IDS, KEYS, N:tl.constexpr, V:tl.constexpr, B:tl.constexpr):
    position = tl.program_id(0)*B + tl.arange(0,B)
    value = tl.load(LOGITS+position,mask=position<N,other=0)
    bits = value.to(tl.int16,bitcast=True).to(tl.int32) & 65535
    bits = tl.where((bits & 32767)==0,0,bits)
    rank = tl.where((bits & 32768)!=0,65535-bits,bits+32768).to(tl.int64)
    token = tl.load(TOKEN_IDS+position,mask=position<N,other=0)
    key = rank*(V+1)+(V-1-token)
    tl.store(KEYS+position,key,mask=position<N)


def packed_keys(logits, token_ids, *, vocab_size):
    """Return exact integer keys; no value-dependent host operation on CUDA."""
    if type(vocab_size) is not int or not 1<=vocab_size<(1<<31):
        raise ValueError('vocab_size must be a positive integer below 2^31')
    if (logits.dtype != torch.bfloat16 or token_ids.dtype != torch.int64
            or logits.ndim != 1 or token_ids.ndim != 1
            or logits.shape != token_ids.shape or not logits.numel()
            or logits.device != token_ids.device
            or not logits.is_contiguous() or not token_ids.is_contiguous()):
        raise ValueError('Contiguous same-device BF16 logits and int64 token IDs required')
    if logits.is_cuda:
        keys = torch.empty(logits.numel(),dtype=torch.int64,device=logits.device)
        _encode_keys[(triton.cdiv(logits.numel(),256),)](
            logits,token_ids,keys,N=logits.numel(),V=vocab_size,B=256,num_warps=4)
        return keys
    if logits.device.type != 'cpu':
        raise ValueError('Only CPU and CUDA are supported')
    if bool(torch.isnan(logits).any()) or bool(((token_ids<0)|(token_ids>=vocab_size)).any()):
        raise ValueError('Invalid numeric logits or token IDs')
    bits = logits.view(torch.int16).to(torch.int64)&65535
    bits = torch.where((bits&32767)==0,0,bits)
    rank = torch.where((bits&32768)!=0,65535-bits,bits+32768)
    return rank*(vocab_size+1)+(vocab_size-1-token_ids)


def rank_positions(logits, token_ids, k=50, *, vocab_size):
    """Return positions ordered by descending value, then ascending token ID.

    The caller gathers logits and token_ids with these positions. This preserves
    the stable BF16 top-k ordering, including signed zero and cutoff ties, while
    keeping every allocation and launch shape static for CUDA graph capture.
    Equal keys occur only for duplicate value/token pairs, such as padding;
    their positions are interchangeable because their gathered values and IDs
    are identical. The certified branch contains at least k genuine rows.
    """
    if type(k) is not int or not 1<=k<=logits.numel():
        raise ValueError('k must be an integer in [1,number of logits]')
    keys = packed_keys(logits,token_ids,vocab_size=vocab_size)
    return torch.topk(keys,k,largest=True,sorted=True).indices


__all__=['packed_keys','rank_positions']
