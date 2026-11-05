"""Shape-only synthetic DCU benchmark; output path is argv[1]."""
import json,sys,statistics
import torch
import torch.nn.functional as F
from split_linear import split_linear

CONFIGS=((4,256),(4,512),(8,512),(8,1024))
SHAPES=((4608,1536,False),(1536,1536,False),(6144,1536,True),(1536,6144,True))

def measure(fn,repeats=100):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    for _ in range(3):graph.replay()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):graph.replay()
        end.record();end.synchronize();samples.append(start.elapsed_time(end)/repeats)
    return statistics.median(samples)

def main(path):
    assert torch.cuda.is_available() and torch.version.hip,"HIP/DCU required"
    torch.manual_seed(2201);torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
    records=[];selected={}
    with torch.inference_mode():
        for b in (1,4,8):
            for n,k,has_bias in SHAPES:
                x=torch.randn((b,1,k),device="cuda",dtype=torch.float32);w=torch.randn((n,k),device="cuda",dtype=torch.float32)*0.03
                bias=torch.randn((n,),device="cuda",dtype=torch.float32) if has_bias else None
                ref=F.linear(x.double(),w.double(),bias.double() if bias is not None else None)
                base=torch.compile(lambda a:F.linear(a,w,bias),mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=False)
                baseline=base(x);torch.cuda.synchronize()
                assert torch.allclose(baseline.double(),ref,rtol=1e-4,atol=1e-5),"baseline numerical check failed"
                baseline_times=[measure(lambda:base(x)) for _ in range(2)]
                print(f"GEMV_SHAPE B={b} N={n} K={k} BASE_MS={baseline_times}",flush=True)
                choices=[]
                for bn,bk in CONFIGS:
                    fn=lambda:split_linear(x,w,bias,bn,bk)
                    try:
                        output=fn();torch.cuda.synchronize()
                    except Exception as e:
                        choices.append(dict(bn=bn,bk=bk,correct=False,times_ms=[],selected_eligible=False,error=str(e)))
                        print(f"CONFIG_UNAVAILABLE bn={bn} bk={bk}: {e}",flush=True)
                        continue
                    error=(output.double()-ref).abs().max().item()
                    passed=torch.allclose(output.double(),ref,rtol=1e-4,atol=1e-5)
                    times=[measure(fn) for _ in range(2)] if passed else []
                    qualifies=passed and all(t<=base_t*0.9 for t,base_t in zip(times,baseline_times))
                    entry=dict(bn=bn,bk=bk,max_abs_error=error,correct=passed,times_ms=times,selected_eligible=qualifies)
                    choices.append(entry)
                eligible=[c for c in choices if c["selected_eligible"]]
                mode=None
                if eligible:
                    best=min(eligible,key=lambda c:statistics.mean(c["times_ms"]))
                    mode=dict(bn=best["bn"],bk=best["bk"])
                print(f"GEMV_CHOICE B={b} N={n} K={k} MODE={mode}",flush=True)
                key=f"{b},{n},{k},{int(has_bias)}";selected[key]=mode
                records.append(dict(batch=b,n=n,k=k,bias=has_bias,baseline_ms=baseline_times,configs=choices,mode=mode))
    from split_linear import validate_plan
    validate_plan(dict(shape_plan=selected,records=records))
    with open(path,"w",encoding="utf-8",newline="\n") as f:json.dump(dict(shape_plan=selected,records=records),f,indent=2)
    print("SPLIT_GEMV_CHECK=PASS",flush=True)
    print("SPLIT_GEMV_SELECTED",sum(v is not None for v in selected.values()),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: gemm_check.py shape_plan.json")
    main(sys.argv[1])
