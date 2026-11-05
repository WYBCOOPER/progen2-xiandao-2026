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
                     H: tl.constexpr, QB: tl.constexpr, QH: tl.constexpr,
                     QD: tl.constexpr, KB: tl.constexpr, KH: tl.constexpr,
                     KS: tl.constexpr, KD: tl.constexpr, VB: tl.constexpr,
                     VH: tl.constexpr, VS: tl.constexpr, VD: tl.constexpr,
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
