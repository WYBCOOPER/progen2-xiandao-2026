"""Run sequential same-session official comparisons; never alter test tools."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,shutil,statistics,platform,importlib.metadata,math

ORDER=('v20','v22b','v22b','v20')
IDS=('load1_b1_l32_g256','load2_b8_l32_g256','load3_b4_l32_g512','load4_b4_l512_g64')
TOTALS=(256,2048,2048,256)
OFFICIAL=('public_test.py','run_public_test.sh','baseline/solution.py','check_environment.py','public_test_data/public_cases.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def fingerprints(root):return {n:sha((root/n).read_bytes()) for n in OFFICIAL}
def audit(folder,label,plan_sha):
    log=(folder/'candidate.log').read_text(encoding='utf8',errors='replace').splitlines()
    result=json.loads((folder/'candidate_public.json').read_bytes());loads=result.get('loads',[])
    exact=result.get('all_public_references_exact') is True and len(loads)==4 and [r.get('load_id') for r in loads]==list(IDS) and [r.get('total') for r in loads]==list(TOTALS) and all(r.get('exact_match') is True and r.get('same')==r.get('total') and isinstance(r.get('time_s'),(int,float)) and math.isfinite(r['time_s']) and r['time_s']>0 for r in loads)
    counters={}
    for key in ('COMPILE_EVENTS','TRITON_CACHE_MISSES','GRAPH_CAPTURES','GRAPH_REPLAYS'):
        prefix=key+'_DURING_GENERATE='
        counters[key]=[int(line[len(prefix):]) for line in log if line.startswith(prefix)]
    formal_zero=all(len(counters[k])==8 and counters[k][1::2]==[0]*4 for k in ('COMPILE_EVENTS','TRITON_CACHE_MISSES','GRAPH_CAPTURES'))
    replays=counters['GRAPH_REPLAYS']==[255,255,255,255,511,511,63,63]
    marker='FP32_CANDIDATE='+('fp32_batch_hybrid_v20' if label=='v20' else 'fp32_split_gemv_v22b')
    identity=marker in log
    plan_verified=label=='v20' or 'SPLIT_GEMV_PLAN_SHA256='+plan_sha in log
    report=dict(label=label,accuracy_exact=exact,formal_compile_miss_capture_zero=formal_zero,replays_valid=replays,identity=identity,plan_verified=plan_verified,counters=counters,passed=exact and formal_zero and replays and identity and plan_verified)
    dump(folder/'audit.json',report)
    return report,loads

def summarize(runs):
    complete=len(runs)==4 and all(r.get('passed') is True for r in runs)
    report=dict(order=list(ORDER),runs=runs,all_four_runs_passed=complete,decision='INCOMPLETE_OR_FAILED',scope='Same-session two runs per version; descriptive comparison, not a confidence interval or official contest score')
    if not complete:return report
    old=[r for r in runs if r['label']=='v20'];new=[r for r in runs if r['label']=='v22b']
    rows=[]
    for i,name in enumerate(IDS):
        a=[r['loads'][i]['time_s'] for r in old];b=[r['loads'][i]['time_s'] for r in new]
        pairs=[(1-b[j]/a[j])*100 for j in range(2)]
        rows.append(dict(load_id=name,v20_seconds=a,v22b_seconds=b,v20_median_s=statistics.median(a),v22b_median_s=statistics.median(b),median_time_reduction_percent=(1-statistics.median(b)/statistics.median(a))*100,paired_time_reduction_percent=pairs,new_split_gemv_enabled=i==0))
    b1=rows[0];signal=min(b1['paired_time_reduction_percent'])>=3 and b1['median_time_reduction_percent']>=5
    regression=any(r['median_time_reduction_percent'] < -5 for r in rows[1:])
    report.update(workloads=rows,b1_repeat_signal=signal,other_workload_regression_over_5pct=regression,decision='REPEATABLE_B1_GAIN' if signal and not regression else 'NO_CLEAR_GAIN_OR_REGRESSION',attribution='Only B1 uses new split-GEMV. B4/B8 retain original linears; differences there are not split-GEMV gains.')
    return report

def markdown(report):
    lines=['# v20 / v22b 同环境复测','',f"状态：{report['decision']}",'','顺序：v20 → v22b → v22b → v20。每版两次，均调用原始测试器。','']
    if report['all_four_runs_passed']:
        lines+=['四轮均为 4608 个输出完全一致，正式计时编译/缓存未命中/图捕获为 0。','','|任务|v20 两次（秒）|v22b 两次（秒）|中位耗时减少|','|---|---|---|---|']
        for r in report['workloads']:
            a=' / '.join(f'{t:.4f}' for t in r['v20_seconds']);b=' / '.join(f'{t:.4f}' for t in r['v22b_seconds'])
            lines.append(f"|{r['load_id']}|{a}|{b}|{r['median_time_reduction_percent']:.2f}%|")
        lines+=['','仅 B1 启用了新算子。其他任务的波动不能归为本次优化收益。','两次复测只是实际筛选信号，不代表统计置信区间或官方综合成绩。']
    else:
        lines+=['有运行失败、结果缺失或校验未通过，不作加速结论。']
        for r in report['runs']:lines.append(f"- {r['name']}: {'PASS' if r.get('passed') else 'FAIL'} {r.get('error','')}")
    return '\n'.join(lines)+'\n'

def main(root,out):
    root=Path(root).resolve();out=Path(out).resolve();here=Path(__file__).resolve().parent
    import torch
    previous=json.loads((here/'prior_runtime_versions.json').read_bytes())
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),hip=torch.version.hip)
    for n in ('triton','transformers','tokenizers','numpy','accelerate'):
        runtime[n]=importlib.metadata.version(n)
    assert torch.cuda.is_available() and torch.version.hip,'Requires DCU/HIP'
    runtime['device']=torch.cuda.get_device_name(0);runtime['device_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
    runtime['official_tools_sha256']=fingerprints(root);dump(out/'runtime_versions.json',runtime)
    assert all(runtime[k]==previous[k] for k in ('torch','hip','triton','transformers','tokenizers','numpy','accelerate','device')),'Environment differs from the verified shape-plan run'
    assert runtime['official_tools_sha256']==previous['official_tools_sha256'],'Official tools differ from the verified prior run'
    sys.path.insert(0,str(here/'candidates/v22b'))
    from plan_utils import validate_plan
    raw=(here/'shape_plan.json').read_bytes();plan=validate_plan(json.loads(raw))
    enabled={k:v for k,v in plan.items() if v is not None}
    assert enabled=={'1,1536,1536,0':{'bn':4,'bk':512},'1,1536,6144,1':{'bn':4,'bk':256}},'Unexpected plan; B1 comparison needs the verified plan'
    plan_sha=sha(raw);runs=[]
    for i,label in enumerate(ORDER,1):
        name=f'{i:02d}_{label}';folder=out/name;folder.mkdir()
        shutil.copytree(here/'candidates'/label,folder/'source',ignore=shutil.ignore_patterns('__pycache__'))
        dump(folder/'source_sha256.json',{p.relative_to(folder/'source').as_posix():sha(p.read_bytes()) for p in sorted((folder/'source').rglob('*')) if p.is_file()})
        env=os.environ.copy();env['RESULT_FILE']=str(folder/'candidate_public.json');env['PYTHONUNBUFFERED']='1'
        env.pop('SPLIT_GEMV_PLAN',None)
        if label=='v22b':
            (folder/'shape_plan.json').write_bytes(raw);env['SPLIT_GEMV_PLAN']=str(folder/'shape_plan.json')
        print(f'\nCOMPARISON_RUN={name} ({i}/4)',flush=True)
        command=['bash',str(root/'run_public_test.sh'),str(folder/'source')]
        dump(folder/'command.json',dict(argv=command,RESULT_FILE=env['RESULT_FILE'],SPLIT_GEMV_PLAN=env.get('SPLIT_GEMV_PLAN')))
        row=dict(name=name,label=label,passed=False)
        try:
            with open(folder/'candidate.log','w',encoding='utf8',newline='\n') as log:
                process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
                try:
                    for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                    code=process.wait()
                except BaseException:
                    process.terminate();process.wait();raise
            row['exit_code']=code;(folder/'exit_status.txt').write_text(str(code)+'\n')
            checks,loads=audit(folder,label,plan_sha)
            row.update(passed=code==0 and checks['passed'],audit=checks,loads=loads)
        except Exception as exc:
            row['error']=repr(exc);print('COMPARISON_RUN_ERROR='+repr(exc),flush=True)
        runs.append(row);report=summarize(runs);dump(out/'summary.json',report)
        (out/'report.md').write_text(markdown(report),encoding='utf8')
    assert fingerprints(root)==runtime['official_tools_sha256'],'Official tools changed during comparison'
    report=summarize(runs);dump(out/'summary.json',report);text=markdown(report);(out/'report.md').write_text(text,encoding='utf8');print('\n'+text,flush=True)
    print('COMPARISON_DECISION='+report['decision'],flush=True)
    return 0 if report['all_four_runs_passed'] else 2

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: compare.py CONTEST_ROOT RESULT_DIRECTORY')
    raise SystemExit(main(*sys.argv[1:]))
