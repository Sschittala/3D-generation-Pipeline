"""Opt-in inference adapter for the six FlashAttention calls used by TRELLIS.

Uses PyTorch SDPA, never imports/compiles the external flash_attn package.
Only noncausal, zero-dropout calls are implemented. Not a general FlashAttention
replacement. Install before importing TRELLIS. Variable-length sequences are
bucketed by length and evaluated separately to preserve isolation.
"""
import importlib.machinery
import os
import sys
import types


def dense(q, k, v):
    import torch.nn.functional as F
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('Expected [batch, tokens, heads, channels]')
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError('Incompatible attention dimensions')
    if k.shape[1] == 0 and q.shape[1] != 0:
        raise ValueError('Cannot attend to an empty key sequence')
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                         dropout_p=0.0,is_causal=False).transpose(1,2)


def flash_attn_func(q, k, v):
    return dense(q,k,v)


def flash_attn_qkvpacked_func(qkv):
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError('Expected packed QKV [batch,tokens,3,heads,channels]')
    return dense(*qkv.unbind(dim=2))


def flash_attn_kvpacked_func(q, kv):
    if kv.ndim != 5 or kv.shape[2] != 2:
        raise ValueError('Expected packed KV [batch,tokens,2,heads,channels]')
    k,v=kv.unbind(dim=2)
    return dense(q,k,v)


def offsets(cumulative, total, maximum):
    values=cumulative.tolist()
    if not values or values[0]!=0 or values[-1]!=total or any(type(x) is not int for x in values):
        raise ValueError('Invalid cumulative sequence offsets')
    if any(b<a or b-a>maximum for a,b in zip(values,values[1:])):
        raise ValueError('Sequence length exceeds the declared maximum or offsets decrease')
    return values


def flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k):
    import torch
    if q.ndim!=3 or k.ndim!=3 or v.ndim!=3 or len(k)!=len(v):
        raise ValueError('Expected packed token tensors [tokens,heads,channels]')
    qs=offsets(cu_seqlens_q,len(q),max_seqlen_q)
    ks=offsets(cu_seqlens_k,len(k),max_seqlen_k)
    if len(qs)!=len(ks): raise ValueError('Query/key sequence counts differ')
    if len(q)==0: return q.new_empty((0,q.shape[1],v.shape[2]))
    buckets={};outputs=[None]*(len(qs)-1)
    for i in range(len(qs)-1):
        lengths=(qs[i+1]-qs[i],ks[i+1]-ks[i])
        if lengths[0]==0:
            outputs[i]=q.new_empty((0,q.shape[1],v.shape[2]));continue
        if lengths[1]==0: raise ValueError('Nonempty query with empty keys')
        buckets.setdefault(lengths,[]).append(i)
    # Cap buckets to bound temporary allocations for windowed attention.
    for indices in buckets.values():
        for start in range(0,len(indices),32):
            chunk=indices[start:start+32]
            qb=torch.stack([q[qs[i]:qs[i+1]] for i in chunk])
            kb=torch.stack([k[ks[i]:ks[i+1]] for i in chunk])
            vb=torch.stack([v[ks[i]:ks[i+1]] for i in chunk])
            result=dense(qb,kb,vb)
            for j,i in enumerate(chunk): outputs[i]=result[j]
    return torch.cat(outputs,dim=0)


def flash_attn_varlen_qkvpacked_func(qkv,cu_seqlens,max_seqlen):
    if qkv.ndim!=4 or qkv.shape[1]!=3: raise ValueError('Invalid variable-length packed QKV')
    q,k,v=qkv.unbind(dim=1)
    return flash_attn_varlen_func(q,k,v,cu_seqlens,cu_seqlens,max_seqlen,max_seqlen)


def flash_attn_varlen_kvpacked_func(q,kv,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k):
    if kv.ndim!=4 or kv.shape[1]!=2: raise ValueError('Invalid variable-length packed KV')
    k,v=kv.unbind(dim=1)
    return flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k)


def install():
    if any(name=='trellis' or name.startswith('trellis.') for name in sys.modules):
        raise RuntimeError('Install SDPA adapter before importing TRELLIS')
    if 'flash_attn' in sys.modules:
        raise RuntimeError('flash_attn already imported; use a fresh process')
    module=types.ModuleType('flash_attn')
    module.__spec__=importlib.machinery.ModuleSpec('flash_attn',loader=None)
    module.__version__='trellis-sdpa-adapter'
    for name,value in globals().copy().items():
        if name.startswith('flash_attn_') and callable(value): setattr(module,name,value)
    sys.modules['flash_attn']=module
    os.environ['ATTN_BACKEND']='sdpa'
    os.environ['SPARSE_ATTN_BACKEND']='flash_attn'
    print('ATTENTION OVERRIDE: dense=native PyTorch SDPA; sparse=TRELLIS SDPA adapter (not external FlashAttention)',flush=True)


def self_test(device='cpu'):
    import torch
    torch.manual_seed(7)
    def reference(q,k,v):
        a=q.transpose(1,2);b=k.transpose(1,2);c=v.transpose(1,2)
        return ((a @ b.transpose(-2,-1) / a.shape[-1]**.5).softmax(-1) @ c).transpose(1,2)
    q=torch.randn(2,5,2,8,device=device);k=torch.randn(2,7,2,8,device=device);v=torch.randn_like(k)
    torch.testing.assert_close(dense(q,k,v),reference(q,k,v),rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(flash_attn_kvpacked_func(q,torch.stack([k,v],dim=2)),reference(q,k,v),rtol=1e-4,atol=1e-5)
    packed=torch.randn(2,5,3,2,8,device=device);q0,k0,v0=packed.unbind(dim=2)
    torch.testing.assert_close(flash_attn_qkvpacked_func(packed),reference(q0,k0,v0),rtol=1e-4,atol=1e-5)
    # Different lengths and ordering exercise bucketing and sequence isolation.
    q=torch.randn(8,2,8,device=device);k=torch.randn(10,2,8,device=device);v=torch.randn_like(k)
    cq=torch.tensor([0,3,5,8],dtype=torch.int32,device=device);ck=torch.tensor([0,4,6,10],dtype=torch.int32,device=device)
    expected=torch.cat([reference(q[a:b][None],k[c:d][None],v[c:d][None])[0] for a,b,c,d in [(0,3,0,4),(3,5,4,6),(5,8,6,10)]])
    torch.testing.assert_close(flash_attn_varlen_func(q,k,v,cq,ck,3,4),expected,rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(flash_attn_varlen_kvpacked_func(q,torch.stack([k,v],dim=1),cq,ck,3,4),expected,rtol=1e-4,atol=1e-5)
    packed=torch.stack([q,q*.7,q*1.2],dim=1)
    expected=torch.cat([reference(q[a:b][None],(q*.7)[a:b][None],(q*1.2)[a:b][None])[0] for a,b in [(0,3),(3,5),(5,8)]])
    torch.testing.assert_close(flash_attn_varlen_qkvpacked_func(packed,cq,3),expected,rtol=1e-4,atol=1e-5)
    # A change in the first sequence must never leak into others.
    changed=packed.clone();changed[:3]*=5
    torch.testing.assert_close(flash_attn_varlen_qkvpacked_func(changed,cq,3)[3:],expected[3:],rtol=1e-4,atol=1e-5)
    try: flash_attn_varlen_func(q,k,v,cq,ck,1,4)
    except ValueError: pass
    else: raise AssertionError('Invalid sequence maximum accepted')
    print(f'PASS: six adapter calls, variable-length ordering/isolation, invalid offsets; device={device}')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    self_test(parser.parse_args().device)
