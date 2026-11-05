import json,torch
from models.progen.kernels import short_attention

def tensors(batch,length,scale,device):
    q=torch.randn(batch,1,16,96,device=device).transpose(1,2)*scale
    k=torch.randn(batch,16,1024,96,device=device)
    v=torch.randn_like(k)
    return q,k[:,:,:length],v[:,:,:length]

def reference(q,k,v):
    return ((q@k.transpose(-1,-2))/torch.sqrt(torch.tensor(96.,device=q.device))).softmax(-1)@v

def main():
    assert torch.cuda.is_available() and torch.version.hip,"DCU/HIP required"
    torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1212);records=[]
    with torch.inference_mode():
        for b in (1,4,8):
            for s in (1,2,31,32,33,63,64,65,127,128):
                for scale in (1.,8.):
                    args=tensors(b,s,scale,"cuda")
                    before=[x.clone() for x in args]
                    y=short_attention(*args);expected=reference(*args)
                    torch.testing.assert_close(y,expected,rtol=2e-4,atol=2e-4)
                    for x,old in zip(args,before):assert torch.equal(x,old)
                    records.append({"B":b,"S":s,"scale":scale,"max_abs_error":(y-expected).abs().max().item()})
    print("SHORT_ATTENTION_DETAILS="+json.dumps(records),flush=True)
    print("SHORT_ATTENTION_CHECK=PASS cases=60",flush=True)

if __name__=="__main__":main()
