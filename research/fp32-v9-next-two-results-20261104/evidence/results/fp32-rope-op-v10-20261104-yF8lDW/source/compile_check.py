"""Compile RoPE/KV invocation and verify exact output and in-place writes."""
import torch
from models.progen.kernels import rope_kv_decode

def step(qkv, kc, vc, sin, cos, pos):
    q = rope_kv_decode(qkv, kc, vc, sin, cos, pos)
    # Materialize reads after the custom-op mutation so reordering is detectable.
    return q, kc[:, :, pos:pos+1].clone(), vc[:, :, pos:pos+1].clone()

def main():
    assert torch.cuda.is_available() and torch.version.hip, "DCU/HIP required"
    torch.manual_seed(419)
    compiled=torch.compile(step,mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=True)
    with torch.inference_mode():
        for batch in (1,2,8):
            for pos in (32,64):
                qkv=torch.randn(batch,1,4608,device="cuda")
                sin=torch.randn(1024,24,device="cuda");cos=torch.randn_like(sin)
                original_qkv=qkv.clone();original_sin=sin.clone();original_cos=cos.clone()
                q,v,k=(x.reshape(batch,1,16,96) for x in qkv.reshape(batch,1,8,576).split(192,-1))
                def rotate(x):
                    a=x[...,:48]
                    swapped=torch.stack((-a[...,1::2],a[...,::2]),-1).flatten(-2)
                    return torch.cat((a*cos[pos].repeat_interleave(2)+swapped*sin[pos].repeat_interleave(2),x[...,48:]),-1)
                expected=rotate(q).transpose(1,2)
                kc=torch.randn(batch,16,1024,96,device="cuda");vc=torch.randn_like(kc)
                expected_k=kc.clone();expected_v=vc.clone()
                expected_k[:,:,pos:pos+1]=rotate(k).transpose(1,2)
                expected_v[:,:,pos:pos+1]=v.transpose(1,2)
                for _ in range(2):
                    actual, read_k, read_v=compiled(qkv,kc,vc,sin,cos,pos)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    torch.testing.assert_close(read_k,expected_k[:,:,pos:pos+1],rtol=0,atol=0)
                    torch.testing.assert_close(read_v,expected_v[:,:,pos:pos+1],rtol=0,atol=0)
                    torch.testing.assert_close(kc,expected_k,rtol=0,atol=0)
                    torch.testing.assert_close(vc,expected_v,rtol=0,atol=0)
                    assert torch.equal(qkv,original_qkv) and torch.equal(sin,original_sin) and torch.equal(cos,original_cos)
                print(f"COMPILED_CALL_PASS B={batch} POS={pos}",flush=True)
    print("COMPILED_KERNEL_CHECK=PASS",flush=True)

if __name__=="__main__":main()
