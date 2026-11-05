"""Check both kernels together in one compiled graph before model warmup."""
import torch
from models.progen.kernels import rope_kv_decode, decode_attention

def step(qkv,kc,vc,sin,cos,pos):
    q=rope_kv_decode(qkv,kc,vc,sin,cos,pos)
    return decode_attention(q,kc[:,:,:pos+1],vc[:,:,:pos+1])

def main():
    assert torch.cuda.is_available() and torch.version.hip, "DCU/HIP required"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(891)
    compiled=torch.compile(step,mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=True)
    with torch.inference_mode():
        for batch in (1,2,8):
            for pos in (32,64):
                qkv=torch.randn(batch,1,4608,device="cuda")
                sin=torch.randn(1024,24,device="cuda");cos=torch.randn_like(sin)
                q,v,k=(x.reshape(batch,1,16,96) for x in qkv.reshape(batch,1,8,576).split(192,-1))
                def rotate(x):
                    a=x[...,:48]
                    swapped=torch.stack((-a[...,1::2],a[...,::2]),-1).flatten(-2)
                    return torch.cat((a*cos[pos].repeat_interleave(2)+swapped*sin[pos].repeat_interleave(2),x[...,48:]),-1)
                kc=torch.randn(batch,16,1024,96,device="cuda");vc=torch.randn_like(kc)
                expected_k=kc.clone();expected_v=vc.clone()
                expected_k[:,:,pos:pos+1]=rotate(k).transpose(1,2)
                expected_v[:,:,pos:pos+1]=v.transpose(1,2)
                query=rotate(q).transpose(1,2)
                expected=((query@expected_k[:,:,:pos+1].transpose(-1,-2))/torch.sqrt(torch.tensor(96.,device="cuda"))).softmax(-1)@expected_v[:,:,:pos+1]
                for _ in range(2):
                    actual=compiled(qkv,kc,vc,sin,cos,pos)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
                    torch.testing.assert_close(kc,expected_k,rtol=0,atol=0)
                    torch.testing.assert_close(vc,expected_v,rtol=0,atol=0)
                print(f"COMPILED_CALL_PASS B={batch} POS={pos}",flush=True)
    print("COMPILED_KERNEL_CHECK=PASS",flush=True)

if __name__=="__main__":main()
