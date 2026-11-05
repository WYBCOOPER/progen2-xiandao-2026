"""Input-independent HIP preflight; these timings are not contest scores."""
import json, pathlib, torch
from compensated_linear import comp_linear, pack_model_weights, linear
from types import SimpleNamespace

def main():
    assert torch.cuda.is_available() and torch.version.hip,'DCU/HIP required'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(2121)
    results=[]
    with torch.inference_mode():
        for n,k in ((4608,1536),(1536,1536),(6144,1536),(1536,6144)):
            module=torch.nn.Linear(k,n,bias=True,device='cuda',dtype=torch.float32)
            module.weight.normal_(0,.03);module.bias.normal_(0,.01)
            block=SimpleNamespace(attn=SimpleNamespace(qkv_proj=module,out_proj=module),mlp=SimpleNamespace(fc_in=module,fc_out=module))
            pack_model_weights(SimpleNamespace(transformer=SimpleNamespace(h=[block])))
            for b in (1,4,8):
                x=torch.randn(b,1,k,device='cuda')
                reference=torch.nn.functional.linear(x,module.weight,module.bias)
                got=linear(x,module)
                torch.cuda.synchronize()
                torch.testing.assert_close(got,reference,rtol=2e-4,atol=2e-4)
                assert torch.isfinite(got).all()
                def measure(fn):
                    for _ in range(3):fn()
                    torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):fn()
                    end.record();end.synchronize()
                    return start.elapsed_time(end)/10
                base_ms=measure(lambda:torch.nn.functional.linear(x,module.weight,module.bias))
                comp_ms=measure(lambda:linear(x,module))
                results.append(dict(B=b,N=n,K=k,max_abs_error=(got-reference).abs().max().item(),torch_linear_ms=base_ms,compensated_ms=comp_ms))
                print('GEMM_PROBE='+json.dumps(results[-1]),flush=True)
            # Real compiled custom-op and graph lifecycle, not only eager calls.
            x=torch.randn(4,1,k,device='cuda')
            compiled=torch.compile(lambda t:linear(t,module),backend='inductor',fullgraph=True,dynamic=False)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):compiled(x)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=compiled(x)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            for _ in range(2):
                x.normal_();expected=linear(x,module)
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(output,expected,rtol=0,atol=0)
    pathlib.Path(__file__).resolve().parents[1].joinpath('gemm_diagnostic.json').write_text(json.dumps({'scope':'isolated synthetic matrix check; timings are not formal model speedups','probes':results},indent=2)+'\n',encoding='utf8')
    print('COMPENSATED_GEMM_CHECK=PASS',flush=True)

if __name__=='__main__':main()
