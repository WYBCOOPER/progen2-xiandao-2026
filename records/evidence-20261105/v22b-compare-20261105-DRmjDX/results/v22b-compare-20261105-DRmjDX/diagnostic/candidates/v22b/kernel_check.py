"""Standalone HIP check of split stream kernel and merge."""
import torch
from models.progen.kernels import stream_rope_attention

def main():
    assert torch.cuda.is_available() and torch.version.hip
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(137)
    with torch.inference_mode():
        for b in (1,4,8):
            for pos in (0,63,64,127,128,255,256,511,1022):
                qkv=torch.randn(b,1,4608,device="cuda"); sin=torch.randn(1024,24,device="cuda"); cos=torch.randn_like(sin)
                kc=torch.randn(b,16,1024,96,device="cuda"); vc=torch.randn_like(kc)
                oldk=kc.clone();oldv=vc.clone()
                q,v,k=(x.reshape(b,1,16,96) for x in qkv.reshape(b,1,8,576).split(192,-1))
                def rotate(x):
                    a=x[...,:48]; swapped=torch.stack((-a[...,1::2],a[...,::2]),-1).flatten(-2)
                    return torch.cat((a*cos[pos].repeat_interleave(2)+swapped*sin[pos].repeat_interleave(2),x[...,48:]),-1).transpose(1,2)
                eq=rotate(q);oldk[:,:,pos:pos+1]=rotate(k);oldv[:,:,pos:pos+1]=v.transpose(1,2)
                expected=((eq@oldk[:,:,:pos+1].transpose(-1,-2))/96**0.5).softmax(-1)@oldv[:,:,:pos+1]
                actualq,actual=stream_rope_attention(qkv,kc,vc,sin,cos,torch.tensor(pos,device="cuda",dtype=torch.int32))
                torch.testing.assert_close(actualq,eq,rtol=0,atol=0);torch.testing.assert_close(kc,oldk,rtol=0,atol=0);torch.testing.assert_close(vc,oldv,rtol=0,atol=0)
                torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
    print("KERNEL_CHECK=PASS",flush=True)
if __name__=="__main__":main()
