"""Check the selected new kernels under compiler + graph replay, before model load."""
import json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from batched_linear import batched_linear
from plan_utils import validate_plan

def main(path):
    plan=validate_plan(json.loads(Path(path).read_bytes()))
    torch.manual_seed(2307)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    count=0
    with torch.inference_mode():
        for key,mode in plan.items():
            if mode is None or 'kind' not in mode:continue
            b,n,k,has_bias=map(int,key.split(','))
            x=torch.randn(b,1,k,device='cuda');w=torch.randn(n,k,device='cuda')*.03
            bias=torch.randn(n,device='cuda') if has_bias else None
            torch._dynamo.reset()  # Standalone preflight process; ends before model load.
            def step(a,weight,bs):
                return batched_linear(a,weight,bs,mode['kind'],mode['bn'],mode['bk'],mode['bm'])
            fn=torch.compile(step,fullgraph=True,dynamic=False)
            for _ in range(3):fn(x,w,bias)
            torch.cuda.synchronize()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=fn(x,w,bias)
            torch.cuda.current_stream().wait_stream(stream)
            original=w.clone()
            for _ in range(3):
                x.normal_();g.replay();torch.cuda.synchronize()
                ref=F.linear(x.double(),w.double(),None if bias is None else bias.double())
                torch.testing.assert_close(out.double(),ref,rtol=1e-4,atol=1e-5)
                assert torch.equal(w,original)
            count+=1;print('BATCH_GRAPH_PASS '+key,flush=True)
    if count==0:raise RuntimeError('No new selected kernel')
    print('BATCH_GRAPH_CHECK=PASS',flush=True)

if __name__=='__main__':main(sys.argv[1])
