import json, pathlib, sys, math
root=pathlib.Path(sys.argv[1])
log=(root/'candidate.log').read_text(encoding='utf8',errors='replace').splitlines()
def values(prefix):return [int(s[len(prefix):]) for s in log if s.startswith(prefix)]
events=values('COMPILE_EVENTS_DURING_GENERATE=')
misses=values('TRITON_CACHE_MISSES_DURING_GENERATE=')
result=json.loads((root/'candidate_public.json').read_bytes());loads=result.get('loads',[])
expected=['load1_b1_l32_g256','load2_b8_l32_g256','load3_b4_l32_g512','load4_b4_l512_g64']
exact=(result.get('all_public_references_exact') is True and len(loads)==4
       and [r.get('load_id') for r in loads]==expected and [r.get('total') for r in loads]==[256,2048,2048,256]
       and all(r.get('exact_match') is True and r.get('same')==r.get('total') for r in loads))
compile_ok=len(events)==8 and all(n==0 for n in events[1::2])
triton_ok=len(misses)==8 and all(n==0 for n in misses[1::2])
identity='FP32_CANDIDATE=fp32_rope_kv_v9' in log
kernel_ok='KERNEL_CHECK=PASS' in (root/'kernel_check.log').read_text(encoding='utf8',errors='replace').splitlines()
report={'candidate':'fp32_rope_kv_v9','all_public_references_exact':exact,'dynamo_events':events,'triton_cache_misses':misses,
        'formal_dynamo_zero':compile_ok,'formal_triton_cache_misses_zero':triton_ok,'identity':identity,'kernel_check_passed':kernel_ok,
        'measurement_scope':'Four official workload calls after official warmup; no edits to official tester',
        'compile_audit_scope':'Dynamo on_compile_start and Triton 3.0 JIT cache-miss hook; not a hardware trace of every native compiler call',
        'experiment_checks_passed':exact and compile_ok and triton_ok and identity and kernel_ok}
(root/'compile_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print('ACCURACY_CHECK='+('PASS' if exact else 'FAIL'))
print('COMPILE_AUDIT='+('PASS' if compile_ok and triton_ok else 'FAIL'))
if not report['experiment_checks_passed']:
    print('Keep this evidence; this run is not an accepted optimization.')
    raise SystemExit(2)
