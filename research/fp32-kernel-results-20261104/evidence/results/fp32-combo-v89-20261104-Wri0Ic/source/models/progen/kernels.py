"""Experimental split-sequence FP32 decode attention; no tensor-core dot product.

Each head uses sixteen fixed 64-position partitions. SEQ is a runtime argument,
not a specialization key; empty partitions produce zero contribution. A second
kernel merges the stable softmax partials. Full-model exactness is a DCU gate.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["SEQ"])
def _decode_partials(Q, K, V, PART, STATS, SEQ,
                     H, QB, QH,
                     QD, KB, KH,
                     KS, KD, VB,
                     VH, VS, VD,
                     BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
                     PARTS: tl.constexpr):
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
def _decode_merge(PART, STATS, OUT, PARTS: tl.constexpr, BLOCK_D: tl.constexpr):
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


def decode_attention(query, key, value):
    if key.shape[-2] < 1 or key.shape[-2] > 1024:
        raise ValueError("FP32 decode kernel supports 1..1024 valid KV positions")
    # Shape-derived dimensions and strides must remain runtime kernel arguments.
    b, h = query.shape[0], query.shape[1]
    partials = torch.empty((b, h, 16, 128), dtype=torch.float32, device=query.device)
    stats = torch.empty((b, h, 16, 2), dtype=torch.float32, device=query.device)
    output = torch.empty((b, h, 1, 96), dtype=torch.float32, device=query.device)
    _decode_partials[(b * h, 16)](
        query, key, value, partials, stats, key.shape[-2], h,
        query.stride(0), query.stride(1), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        64, 128, 16, num_warps=4)
    _decode_merge[(b * h,)](partials, stats, output, 16, 128,
                           num_warps=4)
    return output


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
