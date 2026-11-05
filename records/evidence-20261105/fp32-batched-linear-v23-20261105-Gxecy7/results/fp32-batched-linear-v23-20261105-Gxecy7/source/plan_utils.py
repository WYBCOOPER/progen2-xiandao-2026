import json,math
from pathlib import Path

SHAPES=((4608,1536,0),(1536,1536,0),(6144,1536,1),(1536,6144,1))

def validate_plan(document):
    frozen=json.loads(Path(__file__).with_name("frozen_b1_plan.json").read_text(encoding="utf8"))
    from parent_plan_utils import validate_plan as validate_parent
    validate_parent(frozen)
    plan=document["shape_plan"];records=document["records"]
    expected={f"{b},{n},{k},{bias}" for b in (1,4,8) for n,k,bias in SHAPES}
    if set(plan)!=expected or len(records)!=12:raise ValueError("incomplete shape plan")
    original={f"{r['batch']},{r['n']},{r['k']},{int(r['bias'])}":r for r in frozen["records"] if r["batch"]==1}
    seen=set()
    for r in records:
        b=r["batch"];key=f"{b},{r['n']},{r['k']},{int(r['bias'])}"
        if key not in expected or key in seen or r["mode"]!=plan[key]:raise ValueError("inconsistent shape plan")
        seen.add(key);mode=plan[key]
        if b==1 and (mode!=frozen["shape_plan"][key] or r!=original[key]):raise ValueError("B1 frozen plan changed")
        if mode is None:continue
        if b==1:
            if set(mode)!={"bn","bk"} or mode["bn"] not in (4,8) or mode["bk"] not in (256,512,1024):raise ValueError("bad B1 config")
        elif set(mode)!={"kind","bn","bk","bm"} or mode["kind"] not in ("vector","direct") or mode["bn"] not in (2,4,8) or mode["bk"] not in (256,512,1024) or mode["bm"] not in (4,8):raise ValueError("bad batched config")
        choices=[c for c in r["configs"] if all(c.get(k)==v for k,v in mode.items())]
        if len(choices)!=1:raise ValueError("missing selected benchmark")
        c=choices[0];times=c["times_ms"];base=r["baseline_ms"]
        if c["correct"] is not True or len(times)!=2 or len(base)!=2 or not all(math.isfinite(a) and math.isfinite(z) and 0<a<=z*.9 for a,z in zip(times,base)):raise ValueError("selected kernel failed gate")
        if b!=1 and (c.get("finite") is not True or not math.isfinite(c.get("max_abs_error",float("nan")))):raise ValueError("invalid numerical result")
    return plan
