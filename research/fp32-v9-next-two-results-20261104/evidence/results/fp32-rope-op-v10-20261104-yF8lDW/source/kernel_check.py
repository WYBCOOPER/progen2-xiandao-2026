"""Standalone DCU kernel check; does not run the model."""
import torch
from models.progen.kernels import rope_kv_decode


def main():
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("A HIP/DCU torch device is required")
    torch.manual_seed(137)
    device = "cuda"
    for batch in (1, 4, 8):
        for pos in (0, 1, 31, 32, 63, 64, 511, 1023):
            qkv = torch.randn(batch, 1, 4608, device=device, dtype=torch.float32)
            sin = torch.randn(1024, 24, device=device, dtype=torch.float32)
            cos = torch.randn(1024, 24, device=device, dtype=torch.float32)
            kc = torch.randn(batch, 16, 1024, 96, device=device, dtype=torch.float32)
            vc = torch.randn_like(kc)
            original = [x.clone() for x in (qkv, sin, cos, kc, vc)]
            packed = qkv.reshape(batch, 1, 8, 576)
            q, v, k = (x.reshape(batch, 1, 16, 96) for x in packed.split(192, dim=-1))
            def rotate(x):
                first = x[..., :48]
                swapped = torch.stack((-first[..., 1::2], first[..., ::2]), dim=-1).flatten(-2)
                return torch.cat((first * cos[pos].repeat_interleave(2) + swapped * sin[pos].repeat_interleave(2), x[..., 48:]), dim=-1)
            expected_q = rotate(q).permute(0, 2, 1, 3)
            expected_k = rotate(k)[:, 0]
            expected_v = v[:, 0]
            actual_q = rope_kv_decode(qkv, kc, vc, sin, cos, pos)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
            torch.testing.assert_close(kc[:, :, pos], expected_k, rtol=0, atol=0)
            torch.testing.assert_close(vc[:, :, pos], expected_v, rtol=0, atol=0)
            for current, before in ((kc, original[3]), (vc, original[4])):
                assert torch.equal(current[:, :, :pos], before[:, :, :pos])
                assert torch.equal(current[:, :, pos+1:], before[:, :, pos+1:])
            for current, before in zip((qkv, sin, cos), original[:3]):
                assert torch.equal(current, before)
            print(f"PASS batch={batch} pos={pos}", flush=True)
    print("KERNEL_CHECK=PASS", flush=True)

if __name__ == "__main__":
    main()
