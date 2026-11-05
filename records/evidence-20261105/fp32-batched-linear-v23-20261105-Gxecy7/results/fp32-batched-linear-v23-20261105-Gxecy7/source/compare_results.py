"""Paired screening only; not a statistically established speedup."""
import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1])
ids=['load1_b1_l32_g256','load2_b8_l32_g256','load3_b4_l32_g512','load4_b4_l512_g64']
totals=[256,2048,2048,256]
def load(name):
    d=json.loads((root/name).read_bytes());rows=d.get('loads',[])
    assert d.get('all_public_references_exact') is True and len(rows)==4
    assert [r['load_id'] for r in rows]==ids and [r['total'] for r in rows]==totals
    assert all(r['same']==r['total'] and r['exact_match'] is True for r in rows)
    return rows
a=load('candidate_public.json');b=load('previous_public.json')
log=(root/'previous.log').read_text(encoding='utf8',errors='replace').splitlines()
assert 'FP32_CANDIDATE=fp32_split_gemv_v22b' in log
assert 'SPLIT_GEMV_PLAN_SHA256='+hashlib.sha256((root/'source/frozen_b1_plan.json').read_bytes()).hexdigest() in log
for prefix in ['COMPILE_EVENTS_DURING_GENERATE=','TRITON_CACHE_MISSES_DURING_GENERATE=','GRAPH_CAPTURES_DURING_GENERATE=']:
    nums=[int(s[len(prefix):]) for s in log if s.startswith(prefix)]
    assert len(nums)==8 and nums[1::2]==[0]*4,prefix
replays=[int(s.split('=',1)[1]) for s in log if s.startswith('GRAPH_REPLAYS_DURING_GENERATE=')]
assert replays==[255,255,255,255,511,511,63,63]
audit=json.loads((root/'compile_audit.json').read_bytes())
assert audit['experiment_checks_passed'] is True
rows=[]
for old,new in zip(b,a):
    # Fail explicitly if the official tester schema changes.
    oldtime=old['time_s'];newtime=new['time_s']
    import math
    assert all(math.isfinite(t) and t>0 for t in [oldtime,newtime])
    rows.append(dict(load_id=old['load_id'],v22b_seconds=oldtime,v23_seconds=newtime,elapsed_reduction_percent=100*(oldtime-newtime)/oldtime))
result=dict(scope='One paired screening run; repeat before claiming stable gains',both_public_exact=True,both_formal_compile_zero=True,loads=rows)
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
text=['# V23 versus V22b','',result['scope'],'','| workload | v22b seconds | v23 seconds | elapsed reduction |','|---|---:|---:|---:|']
text += [f"| {r['load_id']} | {r['v22b_seconds']:.6f} | {r['v23_seconds']:.6f} | {r['elapsed_reduction_percent']:.2f}% |" for r in rows]
text+=['','B1 retains the frozen V22b plan; do not attribute its timing variation to the new B4/B8 kernels.','Both versions matched all 4608 public tokens and had zero formal compile/cache-miss/capture events.']
(root/'comparison.md').write_text('\n'.join(text)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
