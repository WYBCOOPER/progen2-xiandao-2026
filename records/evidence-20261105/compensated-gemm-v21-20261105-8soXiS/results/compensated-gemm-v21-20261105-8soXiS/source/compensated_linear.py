"""Compensated FP16 dot products for FP32 decode linear layers."""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _comp_kernel(X,W0,W1,BIAS,Y,M,N:tl.constexpr,K:tl.constexpr,SCALE:tl.constexpr,HAS_BIAS:tl.constexpr):
    rows=tl.program_id(0)*16+tl.arange(0,16)
    cols=tl.program_id(1)*64+tl.arange(0,64)
    kk=tl.arange(0,32)
    c00=tl.full((16,64),0,tl.float32);c01=tl.full((16,64),0,tl.float32)
    c10=tl.full((16,64),0,tl.float32);c11=tl.full((16,64),0,tl.float32)
    for start in range(0,K,32):
        k=start+kk
        a=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),0)*32.0
        a0=a.to(tl.float16);a1=((a-a0.to(tl.float32))*2048.0).to(tl.float16)
        w0=tl.load(W0+cols[None,:]*K+k[:,None],(cols[None,:]<N)&(k[:,None]<K),0)
        w1=tl.load(W1+cols[None,:]*K+k[:,None],(cols[None,:]<N)&(k[:,None]<K),0)
        c00=tl.dot(a0,w0,c00);c01=tl.dot(a0,w1,c01);c10=tl.dot(a1,w0,c10);c11=tl.dot(a1,w1,c11)
    result=(c00+(c01+c10)/2048.0+c11/4194304.0)/(32.0*SCALE)
    if HAS_BIAS:
        result=result+tl.load(BIAS+cols,cols<N,0)[None,:]
    tl.store(Y+rows[:,None]*N+cols[None,:],result,(rows[:,None]<M)&(cols[None,:]<N))

def _launch(x,hi,lo,bias,scale):
    if x.dtype!=torch.float32 or x.ndim!=3 or x.shape[1]!=1 or not x.is_contiguous():
        raise ValueError("expected contiguous FP32 [B,1,K]")
    n,k=hi.shape
    if lo.shape!=hi.shape or hi.dtype!=torch.float16 or lo.dtype!=torch.float16 or x.shape[-1]!=k:
        raise ValueError("invalid packed weight")
    if (n,k) not in ((4608,1536),(1536,1536),(6144,1536),(1536,6144)) and not (1<=n<=64 and 1<=k<=6144):
        raise ValueError("unsupported linear dimensions")
    if bias is not None and (bias.shape!=(n,) or bias.dtype!=torch.float32):raise ValueError("invalid bias")
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.float32)
    _comp_kernel[(triton.cdiv(x.shape[0],16),triton.cdiv(n,64))](x,hi,lo,bias if bias is not None else out,out,x.shape[0],n,k,scale,bias is not None,num_warps=4,enable_fp_fusion=False)
    return out

@torch.library.custom_op("progen_v21::linear",mutates_args=(),device_types="cuda")
def comp_linear(x:torch.Tensor,hi:torch.Tensor,lo:torch.Tensor,bias:torch.Tensor | None,scale:float)->torch.Tensor:
    return _launch(x,hi,lo,bias,scale)

@comp_linear.register_fake
def _fake(x,hi,lo,bias,scale):
    return x.new_empty((*x.shape[:-1],hi.shape[0]),dtype=torch.float32)

def linear(x,module):
    return comp_linear(x,module._v21_hi,module._v21_lo,module.bias,module._v21_scale)

def pack_model_weights(model):
    for block in model.transformer.h:
        for module in (block.attn.qkv_proj,block.attn.out_proj,block.mlp.fc_in,block.mlp.fc_out):
            w=module.weight.detach()
            if w.dtype!=torch.float32 or not torch.isfinite(w).all():raise ValueError("weights must be finite FP32")
            maxabs=float(w.abs().max().item())
            scale=4096.0 if maxabs==0 else 2.0**min(12,math.floor(math.log2(16384.0/maxabs)))
            scaled=w*scale
            hi=scaled.to(torch.float16);lo=((scaled-hi.float())*2048.0).to(torch.float16)
            if not math.isfinite(scale) or scale<=0 or not torch.isfinite(hi).all() or not torch.isfinite(lo).all():raise ValueError("weight packing overflow")
            module.register_buffer("_v21_hi",hi.contiguous(),persistent=False)
            module.register_buffer("_v21_lo",lo.contiguous(),persistent=False)
            module._v21_scale=scale
