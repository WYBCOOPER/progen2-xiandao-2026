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


@triton.jit(do_not_specialize=["S"])
def _rope_device_pos(QKV, KC, VC, SIN, COS, OUT, S, POS_PTR, BLOCK: tl.constexpr):
    POS = tl.load(POS_PTR)
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

@triton.jit
def _graph_partials(Q, K, V, PART, STATS, POS_PTR,
                     H, QB, QH,
                     QD, KB, KH,
                     KS, KD, VB,
                     VH, VS, VD,
                     BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
                     PARTS: tl.constexpr):
    SEQ = tl.load(POS_PTR) + 1
    bh = tl.program_id(0)
    part = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.arange(0, BLOCK_D)
    s = part * BLOCK_S + tl.arange(0, BLOCK_S)
    q = tl.load(Q + b * QB + h * QH + d * QD, d < 96, 0)
    k = tl.load(K + b * KB + h * KH + s[:, None] * KS + d[None, :] * KD,
                (s[:, None] < SEQ) & (d[None, :] < 96), 0)
    scores = tl.sum(k * q[None, :], axis=1) / 9.797959327697754
    scores = tl.where(s < SEQ, scores, -float("inf"))
    maximum = tl.max(scores, axis=0)
    safe_maximum = tl.where(part * BLOCK_S < SEQ, maximum, 0.0)
    weights = tl.where(s < SEQ, tl.exp(scores - safe_maximum), 0.0)
    denominator = tl.sum(weights, axis=0)
    v = tl.load(V + b * VB + h * VH + s[:, None] * VS + d[None, :] * VD,
                (s[:, None] < SEQ) & (d[None, :] < 96), 0)
    partial = tl.sum(weights[:, None] * v, axis=0)
    tl.store(PART + (bh * PARTS + part) * BLOCK_D + d, partial, d < 96)
    tl.store(STATS + (bh * PARTS + part) * 2, maximum)
    tl.store(STATS + (bh * PARTS + part) * 2 + 1, denominator)

@triton.jit
def _graph_merge(PART, STATS, OUT, PARTS: tl.constexpr, BLOCK_D: tl.constexpr):
    bh = tl.program_id(0)
    p = tl.arange(0, PARTS)
    d = tl.arange(0, BLOCK_D)
    maxima = tl.load(STATS + (bh * PARTS + p) * 2)
    denominators = tl.load(STATS + (bh * PARTS + p) * 2 + 1)
    maximum = tl.max(maxima, axis=0)
    factors = tl.exp(maxima - maximum)
    denominator = tl.sum(denominators * factors, axis=0)
    partials = tl.load(PART + (bh * PARTS + p[:, None]) * BLOCK_D + d[None, :], d[None, :] < 96, 0)
    output = tl.sum(partials * factors[:, None], axis=0) / denominator
    tl.store(OUT + bh * 96 + d, output, d < 96)

def graph_rope(qkv,kc,vc,sin,cos,pos):
    out=torch.empty((qkv.shape[0],16,1,96),dtype=torch.float32,device=qkv.device)
    _rope_device_pos[(6,qkv.shape[0])](qkv,kc,vc,sin,cos,out,kc.shape[2],pos,256)
    return out


def graph_attention(q,k,v,pos):
    b,h=q.shape[:2]
    partial=torch.empty((b,h,16,128),dtype=torch.float32,device=q.device)
    stats=torch.empty((b,h,16,2),dtype=torch.float32,device=q.device)
    out=torch.empty((b,h,1,96),dtype=torch.float32,device=q.device)
    _graph_partials[(b*h,16)](q,k,v,partial,stats,pos,h,
        q.stride(0),q.stride(1),q.stride(3),
        k.stride(0),k.stride(1),k.stride(2),k.stride(3),
        v.stride(0),v.stride(1),v.stride(2),v.stride(3),64,128,16,num_warps=4)
    _graph_merge[(b*h,)](partial,stats,out,16,128,num_warps=4)
    return out


@triton.jit
def _record_token(TOKEN,OUTPUT,INDEX,CAPACITY):
    b=tl.program_id(0)
    i=tl.load(INDEX)
    token=tl.load(TOKEN+b)
    tl.store(OUTPUT+b*CAPACITY+i,token)


def record_token(token,output,index):
    _record_token[(token.shape[0],)](token,output,index,output.shape[1])


# Explicit mutation boundary: the inline AMD assembly is unchanged, but the
# compiler no longer needs to infer its writes. Graph replay has no Python
# custom-op dispatch in the timed decode loop.
@torch.library.custom_op("progen_v15::rope_attention",
                         mutates_args=("kc", "vc"), device_types="cuda")
def graph_rope_attention(qkv: torch.Tensor, kc: torch.Tensor,
                         vc: torch.Tensor, sin: torch.Tensor,
                         cos: torch.Tensor, pos: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    q=graph_rope(qkv,kc,vc,sin,cos,pos)
    y=graph_attention(q,kc,vc,pos)
    return q,y


@graph_rope_attention.register_fake
def _graph_rope_attention_fake(qkv,kc,vc,sin,cos,pos):
    shape=(qkv.shape[0],16,1,96)
    return (torch.empty(shape,dtype=torch.float32,device=qkv.device),
            torch.empty(shape,dtype=torch.float32,device=qkv.device))
