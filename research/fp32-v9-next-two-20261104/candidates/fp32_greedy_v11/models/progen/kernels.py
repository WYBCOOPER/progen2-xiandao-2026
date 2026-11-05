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


@triton.jit(do_not_specialize=["ROW_STRIDE", "COL_STRIDE"])
def _greedy_token(LOGITS, ALLOWED, OUT, ROW_STRIDE, COL_STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    valid = j < 20
    ids = tl.load(ALLOWED + j, valid, 0)
    scores = tl.load(LOGITS + row * ROW_STRIDE + ids * COL_STRIDE, valid, -float("inf"))
    # Explicit first-NaN and first-tie rules match the parent's ordered selection.
    nan = valid & (scores != scores)
    first_nan = tl.min(tl.where(nan, j, 32), 0)
    finite_or_inf = tl.where(nan, -float("inf"), scores)
    best_score = tl.max(finite_or_inf, 0)
    first_best = tl.min(tl.where(valid & (scores == best_score), j, 32), 0)
    winner = tl.where(first_nan < 20, first_nan, first_best)
    token = tl.load(ALLOWED + winner)
    tl.store(OUT + row, token)


def greedy_token(logits, allowed_ids):
    if logits.ndim != 2 or logits.dtype != torch.float32 or allowed_ids.numel() != 20:
        raise ValueError("Expected FP32 [B,V] logits and 20 allowed token IDs")
    if allowed_ids.dtype != torch.long or not allowed_ids.is_contiguous():
        raise ValueError("Allowed token IDs must be contiguous int64")
    output = torch.empty((logits.shape[0], 1), dtype=torch.long, device=logits.device)
    # Strides are runtime values, never symbolic constexpr metadata.
    _greedy_token[(logits.shape[0],)](logits, allowed_ids, output, logits.stride(0), logits.stride(1), 32)
    return output
