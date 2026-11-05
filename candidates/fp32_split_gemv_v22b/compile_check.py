import torch
from models.progen.kernels import graph_rope_attention,record_token

def main():
    assert torch.cuda.is_available() and torch.version.hip,"DCU/HIP required"
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1313)
    with torch.inference_mode():
        for b in (1,4,8):
            qkv=torch.randn(b,1,4608,device="cuda")
            kc=torch.randn(b,16,1024,96,device="cuda");vc=torch.randn_like(kc)
            sin=torch.randn(1024,24,device="cuda");cos=torch.randn_like(sin)
            pos=torch.tensor(32,dtype=torch.int32,device="cuda")
            idx=torch.zeros((),dtype=torch.int32,device="cuda")
            token=torch.full((b,1),7,dtype=torch.long,device="cuda")
            output=torch.zeros((b,1024),dtype=torch.long,device="cuda")
            def step():
                q,y=graph_rope_attention(qkv,kc,vc,sin,cos,pos)
                record_token(token,output,idx)
                pos.add_(1);idx.add_(1)
                return q,y
            step=torch.compile(step,mode="max-autotune-no-cudagraphs",
                               fullgraph=True,dynamic=False)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):step()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):qout,yout=step()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            for position in (0,63,64,127,128,255,256,511,1022):
                qkv.normal_();token.fill_(9)
                original_qkv=qkv.clone();original_sin=sin.clone();original_cos=cos.clone()
                pos.fill_(position);idx.zero_()
                q,v,k=(x.reshape(b,1,16,96) for x in qkv.reshape(b,1,8,576).split(192,-1))
                def rotate(x,p):
                    a=x[...,:48]
                    swapped=torch.stack((-a[...,1::2],a[...,::2]),-1).flatten(-2)
                    return torch.cat((a*cos[p].repeat_interleave(2)+swapped*sin[p].repeat_interleave(2),x[...,48:]),-1).transpose(1,2)
                ek=kc.clone();ev=vc.clone()
                for offset in (0,1):
                    p=position+offset
                    eq=rotate(q,p)
                    ek[:,:,p:p+1]=rotate(k,p);ev[:,:,p:p+1]=v.transpose(1,2)
                    expected=((eq@ek[:,:,:p+1].transpose(-1,-2))/torch.sqrt(torch.tensor(96.,device="cuda"))).softmax(-1)@ev[:,:,:p+1]
                    g.replay();torch.cuda.synchronize()
                    torch.testing.assert_close(qout,eq,rtol=0,atol=0)
                    torch.testing.assert_close(kc,ek,rtol=0,atol=0)
                    torch.testing.assert_close(vc,ev,rtol=0,atol=0)
                    torch.testing.assert_close(yout,expected,rtol=2e-4,atol=2e-4)
                    assert pos.item()==p+1 and idx.item()==offset+1
                    assert (output[:,:offset+1]==9).all()
                    assert torch.equal(qkv,original_qkv) and torch.equal(sin,original_sin) and torch.equal(cos,original_cos)
                print(f"GRAPH_PROBE_PASS B={b} POS={position}",flush=True)
    print("COMPILED_KERNEL_CHECK=PASS",flush=True)

if __name__=="__main__":main()
