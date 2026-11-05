def validate_plan(document):
    expected={f"{b},{n},{k},{bias}" for b in (1,4,8) for n,k,bias in ((4608,1536,0),(1536,1536,0),(6144,1536,1),(1536,6144,1))}
    plan=document['shape_plan'];records=document['records']
    if set(plan)!=expected or len(records)!=12:raise ValueError('incomplete shape plan')
    seen=set()
    for r in records:
        key=f"{r['batch']},{r['n']},{r['k']},{int(r['bias'])}"
        if key not in expected or key in seen or r['mode']!=plan[key]:raise ValueError('inconsistent shape plan')
        seen.add(key)
        mode=plan[key]
        if mode is None:continue
        if set(mode)!={'bn','bk'} or mode['bn'] not in (4,8) or mode['bk'] not in (256,512,1024):raise ValueError('bad kernel config')
        choices=[c for c in r['configs'] if c['bn']==mode['bn'] and c['bk']==mode['bk']]
        if len(choices)!=1:raise ValueError('missing selected benchmark')
        c=choices[0]
        import math
        times=c['times_ms'];base=r['baseline_ms']
        if c['correct'] is not True or len(times)!=2 or len(base)!=2 or not all(math.isfinite(a) and math.isfinite(b) and 0<a<=b*.9 for a,b in zip(times,base)):raise ValueError('selected kernel did not pass speed/numerical gate')
    return plan
