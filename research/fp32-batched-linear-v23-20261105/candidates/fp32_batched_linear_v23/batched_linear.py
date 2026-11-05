"""FP32 batch-shared GEMV; no atomics or reduced-precision weights."""

import torch
import triton
import triton.language as tl

@triton.jit
def vector_partial(X,W,T,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BM:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.program_id(1)*BK+tl.arange(0,BK)
    b=tl.program_id(2)*BM+tl.arange(0,BM)
    w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
    x=tl.load(X+b[:,None]*K+k[None,:],(b[:,None]<M)&(k[None,:]<K),0)
    z=tl.sum(w[None,:,:]*x[:,None,:],axis=2)
    tl.store(T+tl.program_id(1)*M*N+b[:,None]*N+n[None,:],z,(b[:,None]<M)&(n[None,:]<N))

@triton.jit
def merge(T,B,Y,M:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,BIAS:tl.constexpr):
    n=tl.program_id(0)*128+tl.arange(0,128)
    b=tl.program_id(1)
    s=tl.arange(0,BS)
    z=tl.load(T+s[:,None]*M*N+b*N+n[None,:],(s[:,None]<S)&(n[None,:]<N),0)
    y=tl.sum(z,axis=0)
    if BIAS:y+=tl.load(B+n,n<N,0)
    tl.store(Y+b*N+n,y,n<N)

@triton.jit
def direct(X,W,B,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BM:tl.constexpr,BIAS:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    b=tl.program_id(1)*BM+tl.arange(0,BM)
    z=tl.full((BM,BN,BK),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        kk=start*BK+k
        w=tl.load(W+n[:,None]*K+kk[None,:],(n[:,None]<N)&(kk[None,:]<K),0)
        x=tl.load(X+b[:,None]*K+kk[None,:],(b[:,None]<M)&(kk[None,:]<K),0)
        z=z+w[None,:,:]*x[:,None,:]
    y=tl.sum(z,axis=2)
    if BIAS:y+=tl.load(B+n,n<N,0)[None,:]
    tl.store(Y+b[:,None]*N+n[None,:],y,(b[:,None]<M)&(n[None,:]<N))


@torch.library.custom_op("progen_v23::batched_linear",mutates_args=(),device_types="cuda")
def batched_linear(x:torch.Tensor,w:torch.Tensor,bias:torch.Tensor | None,kind:str,bn:int,bk:int,bm:int)->torch.Tensor:
    if x.ndim!=3 or x.shape[1]!=1 or x.dtype!=torch.float32 or w.ndim!=2 or w.dtype!=torch.float32 or not x.is_contiguous() or not w.is_contiguous():raise ValueError("expected contiguous FP32 inputs")
    m=x.shape[0];n,k=w.shape
    if m not in (4,8) or x.shape[2]!=k or kind not in ("direct","vector") or bn not in (2,4,8) or bk not in (256,512,1024) or bm not in (4,8):raise ValueError("unsupported batched config")
    if x.device!=w.device or (bias is not None and (bias.dtype!=torch.float32 or bias.shape!=(n,) or not bias.is_contiguous() or bias.device!=x.device)):raise ValueError("invalid device or bias")
    y=torch.empty((m,1,n),dtype=torch.float32,device=x.device)
    if kind=="direct":
        direct[(triton.cdiv(n,bn),triton.cdiv(m,bm))](x,w,bias if bias is not None else y,y,m,n,k,bn,bk,bm,bias is not None,num_warps=4,enable_fp_fusion=False)
    else:
        s=triton.cdiv(k,bk);scratch=torch.empty((s,m,n),dtype=torch.float32,device=x.device)
        vector_partial[(triton.cdiv(n,bn),s,triton.cdiv(m,bm))](x,w,scratch,m,n,k,bn,bk,bm,num_warps=4,enable_fp_fusion=False)
        merge[(triton.cdiv(n,128),m)](scratch,bias if bias is not None else y,y,m,n,s,triton.next_power_of_2(s),bias is not None,num_warps=4,enable_fp_fusion=False)
    return y

@batched_linear.register_fake
def _fake(x,w,bias,kind,bn,bk,bm):
    return x.new_empty((*x.shape[:-1],w.shape[0]),dtype=torch.float32)

