"""Tiny compiled-call compatibility probe; no model or benchmark inputs."""
import torch
from models.progen.kernels import decode_attention

def main():
    assert torch.cuda.is_available() and torch.version.hip, "DCU/HIP required"
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(418)
    compiled=torch.compile(decode_attention, mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=True)
    with torch.inference_mode():
        for batch in (1,2,8):
            q=torch.randn(batch,1,16,96,device="cuda").transpose(1,2)
            k=torch.randn(batch,16,1024,96,device="cuda")
            v=torch.randn_like(k)
            for length in (33,65):
                kk=k[:,:,:length];vv=v[:,:,:length]
                expected=((q@kk.transpose(-1,-2))/torch.sqrt(torch.tensor(96.,device="cuda"))).softmax(-1)@vv
                for _ in range(2):
                    actual=compiled(q,kk,vv)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
                print(f"COMPILED_CALL_PASS B={batch} S={length}",flush=True)
    print("COMPILED_KERNEL_CHECK=PASS",flush=True)

if __name__=="__main__":main()
