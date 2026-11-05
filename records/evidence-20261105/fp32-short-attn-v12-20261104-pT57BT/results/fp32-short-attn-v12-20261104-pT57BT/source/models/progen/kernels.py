"""Experimental FP32 decode RoPE/KV kernel; not yet validated on DCU."""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["POS", "S"])
def _rope_kv(QKV, KC, VC, SIN, COS, OUT, S, POS, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Batch is a launch-grid axis, never a symbolic constexpr.
    valid = i < 16 * 96
    d = i % 96
    h = (i // 96) % 16
    b = tl.program_id(1)
    group = h // 2
    local = h % 2
    base = b * 4608 + group * 576 + local * 96 + d
    pair = d ^ 1
    pairbase = b * 4608 + group * 576 + local * 96 + pair
    q = tl.load(QKV + base, valid, 0)
    v = tl.load(QKV + base + 192, valid, 0)
    k = tl.load(QKV + base + 384, valid, 0)
    qp = tl.load(QKV + pairbase, valid, 0)
    kp = tl.load(QKV + pairbase + 384, valid, 0)
    rotating = d < 48
    si = tl.load(SIN + POS * 24 + d // 2, valid & rotating, 0)
    co = tl.load(COS + POS * 24 + d // 2, valid & rotating, 1)
    sign = tl.where(d % 2 == 0, -1.0, 1.0)
    # Opaque FP32 multiply instructions preserve the separate product rounding.
    # The PyTorch 2.4 user-kernel wrapper cannot carry enable_fp_fusion=False.
    # Do not replace these with a fused multiply-add: the reference is mul+mul+add.
    qco = tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", constraints="=v,v,v",
                                   args=[q, co], dtype=tl.float32, is_pure=True, pack=1)
    qsi = tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", constraints="=v,v,v",
                                   args=[qp * sign, si], dtype=tl.float32, is_pure=True, pack=1)
    kco = tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", constraints="=v,v,v",
                                   args=[k, co], dtype=tl.float32, is_pure=True, pack=1)
    ksi = tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", constraints="=v,v,v",
                                   args=[kp * sign, si], dtype=tl.float32, is_pure=True, pack=1)
    qr = tl.where(rotating, qco + qsi, q)
    kr = tl.where(rotating, kco + ksi, k)
    cache = ((b * 16 + h) * S + POS) * 96 + d
    tl.store(OUT + b * 1536 + i, qr, valid)
    tl.store(KC + cache, kr, valid)
    tl.store(VC + cache, v, valid)

def rope_kv_decode(qkv, k_cache, v_cache, sin, cos, cur_len):
    if cur_len >= k_cache.shape[2] or cur_len >= sin.shape[0]:
        raise ValueError("decode position exceeds cache or RoPE table")
    output = torch.empty((qkv.shape[0], 16, 1, 96), dtype=torch.float32, device=qkv.device)
    # Only literal block geometry is constexpr; shape-derived scalars stay runtime.
    _rope_kv[(6, qkv.shape[0])](qkv, k_cache, v_cache, sin, cos, output, k_cache.shape[2], cur_len, 256)
    return output


@triton.jit(do_not_specialize=["SEQ"])
def _short_attention(Q,K,V,OUT,SEQ,H,QB,QH,QD,KB,KH,KS,KD,VB,VH,VS,VD,
                     BLOCK_S:tl.constexpr,BLOCK_D:tl.constexpr):
    bh=tl.program_id(0)
    b=bh//H
    h=bh%H
    s=tl.arange(0,BLOCK_S)
    d=tl.arange(0,BLOCK_D)
    q=tl.load(Q+b*QB+h*QH+d*QD,d<96,0)
    k=tl.load(K+b*KB+h*KH+s[:,None]*KS+d[None,:]*KD,
              (s[:,None]<SEQ)&(d[None,:]<96),0)
    scores=tl.sum(k*q[None,:],1)/9.797959327697754
    scores=tl.where(s<SEQ,scores,-float("inf"))
    maximum=tl.max(scores,0)
    weights=tl.exp(scores-maximum)
    denom=tl.sum(weights,0)
    # Normalize before P*V, matching the reference operation order more closely.
    probs=weights/denom
    v=tl.load(V+b*VB+h*VH+s[:,None]*VS+d[None,:]*VD,
              (s[:,None]<SEQ)&(d[None,:]<96),0)
    output=tl.sum(probs[:,None]*v,0)
    tl.store(OUT+bh*96+d,output,d<96)


def short_attention(query,key,value):
    if key.shape[-2]<1 or key.shape[-2]>128:
        raise ValueError("Short attention supports only 1..128 valid KV positions")
    b,h=query.shape[:2]
    out=torch.empty((b,h,1,96),dtype=torch.float32,device=query.device)
    _short_attention[(b*h,)](query,key,value,out,key.shape[-2],h,
        query.stride(0),query.stride(1),query.stride(3),
        key.stride(0),key.stride(1),key.stride(2),key.stride(3),
        value.stride(0),value.stride(1),value.stride(2),value.stride(3),
        128,128,num_warps=4)
    return out
