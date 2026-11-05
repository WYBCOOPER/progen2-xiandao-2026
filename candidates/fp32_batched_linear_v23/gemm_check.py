"""Independent synthetic preflight; invoke once before model initialization."""
import json,sys,statistics,math
from pathlib import Path
import torch
import torch.nn.functional as F
from batched_linear import batched_linear
from plan_utils import validate_plan,SHAPES

CONFIGS=(("direct",8,256,4),("direct",4,256,8),("vector",4,512,8),("direct",2,1024,4))

def measure(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    for _ in range(3):g.replay()
    torch.cuda.synchronize();samples=[]
    for _ in range(5):
        a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(100):g.replay()
        z.record();z.synchronize();samples.append(a.elapsed_time(z)/100)
    return statistics.median(samples)

def main(path):
    assert torch.cuda.is_available() and torch.version.hip,"HIP/DCU required"
    torch.manual_seed(2301);torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
    frozen=json.loads(Path(__file__).with_name("frozen_b1_plan.json").read_text(encoding="utf8"))
    plan={k:v for k,v in frozen["shape_plan"].items() if k.startswith("1,")};records=[r for r in frozen["records"] if r["batch"]==1]
    def persist():
        Path(path).with_name("gemm_progress.json").write_text(json.dumps(dict(complete=False,shape_plan=plan,records=records),indent=2)+"\n",encoding="utf8")
    persist()
    with torch.inference_mode():
        for b in (4,8):
            for n,k,has_bias in SHAPES:
                key=f"{b},{n},{k},{has_bias}";choices=[]
                try:
                    x=torch.randn((b,1,k),device="cuda");w=torch.randn((n,k),device="cuda")*.03;bias=torch.randn(n,device="cuda") if has_bias else None
                    ref=F.linear(x.double(),w.double(),bias.double() if bias is not None else None)
                    torch._dynamo.reset()
                    base=torch.compile(lambda a:F.linear(a,w,bias),mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=False)
                    y=base(x);torch.cuda.synchronize()
                    if not torch.isfinite(y).all() or not torch.allclose(y.double(),ref,rtol=1e-4,atol=1e-5):raise RuntimeError("baseline numerical check failed")
                    baseline=[measure(lambda:base(x)) for _ in range(2)]
                    for kind,bn,bk,bm in CONFIGS:
                        entry=dict(kind=kind,bn=bn,bk=bk,bm=bm,correct=False,finite=False,times_ms=[],selected_eligible=False)
                        try:
                            fn=lambda:batched_linear(x,w,bias,kind,bn,bk,bm)
                            out=fn();torch.cuda.synchronize();entry["finite"]=bool(torch.isfinite(out).all().item())
                            err=(out.double()-ref).abs().max().item();entry["max_abs_error"]=err if math.isfinite(err) else None
                            entry["correct"]=entry["finite"] and bool(torch.allclose(out.double(),ref,rtol=1e-4,atol=1e-5))
                            if entry["correct"]:
                                entry["times_ms"]=[measure(fn) for _ in range(2)]
                                entry["selected_eligible"]=all(math.isfinite(a) and math.isfinite(z) and 0<a<=z*.9 for a,z in zip(entry["times_ms"],baseline))
                        except Exception as e:entry["error"]=str(e)
                        choices.append(entry)
                        print(f"BATCH_CONFIG {key} {entry}",flush=True)
                    eligible=[c for c in choices if c["selected_eligible"]]
                    best=min(eligible,key=lambda c:statistics.mean(c["times_ms"])) if eligible else None
                    mode={k:best[k] for k in ("kind","bn","bk","bm")} if best else None
                    plan[key]=mode;records.append(dict(batch=b,n=n,k=k,bias=bool(has_bias),baseline_ms=baseline,configs=choices,mode=mode));persist()
                except Exception as e:
                    records.append(dict(batch=b,n=n,k=k,bias=bool(has_bias),baseline_ms=[],configs=choices,mode=None,error=str(e)));plan[key]=None;persist();raise
    document=dict(shape_plan=plan,records=records);validate_plan(document)
    Path(path).write_text(json.dumps(document,indent=2)+"\n",encoding="utf8")
    count=sum(plan[k] is not None for k in plan if not k.startswith("1,"))
    print("BATCHED_GEMM_CHECK=PASS",flush=True);print("BATCHED_GEMM_SELECTED",count,flush=True)
    return 0 if count else 3

if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: gemm_check.py shape_plan.json")
    raise SystemExit(main(sys.argv[1]))
