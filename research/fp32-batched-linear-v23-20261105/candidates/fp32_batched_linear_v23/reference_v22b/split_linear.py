"""FP32 split-K GEMV. Inspired by split parallelism, independently implemented."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _partial(X,W,T,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    ni=tl.program_id(0)*BN+tl.arange(0,BN)
    ki=tl.program_id(1)*BK+tl.arange(0,BK)
    w=tl.load(W+ni[:,None]*K+ki[None,:],(ni[:,None]<N)&(ki[None,:]<K),0)
    for b in tl.static_range(0,M):
        x=tl.load(X+b*K+ki,ki<K,0)
        value=tl.sum(w*x[None,:],axis=1)
        tl.store(T+tl.program_id(1)*M*N+b*N+ni,value,ni<N)

@triton.jit
def _reduce(T,BIAS,Y,M:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,HAS_BIAS:tl.constexpr):
    n=tl.program_id(0)*128+tl.arange(0,128)
    s=tl.arange(0,BS)
    for b in tl.static_range(0,M):
        v=tl.load(T+s[:,None]*M*N+b*N+n[None,:],(s[:,None]<S)&(n[None,:]<N),0)
        y=tl.sum(v,axis=0)
        if HAS_BIAS:y+=tl.load(BIAS+n,n<N,0)
        tl.store(Y+b*N+n,y,n<N)

@torch.library.custom_op("progen_v22::linear",mutates_args=(),device_types="cuda")
def split_linear(x:torch.Tensor,w:torch.Tensor,bias:torch.Tensor | None,bn:int,bk:int)->torch.Tensor:
    if x.dtype!=torch.float32 or w.dtype!=torch.float32 or x.ndim!=3 or x.shape[1]!=1 or not x.is_contiguous() or not w.is_contiguous():raise ValueError("expected contiguous FP32 inputs")
    m=x.shape[0];n,k=w.shape
    if m not in (1,4,8) or x.shape[2]!=k or bn not in (4,8) or bk not in (256,512,1024):raise ValueError("unsupported shape or config")
    if bias is not None and (bias.dtype!=torch.float32 or bias.shape!=(n,) or not bias.is_contiguous()):raise ValueError("invalid bias")
    splits=triton.cdiv(k,bk)
    scratch=torch.empty((splits,m,n),dtype=torch.float32,device=x.device)
    out=torch.empty((m,1,n),dtype=torch.float32,device=x.device)
    _partial[(triton.cdiv(n,bn),splits)](x,w,scratch,m,n,k,bn,bk,num_warps=4,enable_fp_fusion=False)
    _reduce[(triton.cdiv(n,128),)](scratch,bias if bias is not None else out,out,m,n,splits,triton.next_power_of_2(splits),bias is not None,num_warps=4,enable_fp_fusion=False)
    return out

@split_linear.register_fake
def _fake(x,w,bias,bn,bk):
    return x.new_empty((*x.shape[:-1],w.shape[0]),dtype=torch.float32)

def linear(x,module,mode):
    if mode is None:return F.linear(x,module.weight,module.bias)
    return split_linear(x,module.weight,module.bias,mode["bn"],mode["bk"])


from plan_utils import validate_plan
