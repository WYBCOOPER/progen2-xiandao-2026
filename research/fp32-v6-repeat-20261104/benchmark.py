"""Independent repeated diagnostic; official files are imported, never edited."""
import argparse,contextlib,hashlib,importlib.util,json,math,statistics,sys,time
from pathlib import Path

def module_scope():
    return {k:v for k,v in sys.modules.items() if k=='models' or k.startswith('models.')}

@contextlib.contextmanager
def activate(modules,paths):
    previous=module_scope();previous_paths=sys.path[:]
    for name in previous:sys.modules.pop(name,None)
    sys.modules.update(modules);sys.path[:]=paths
    try:yield
    finally:
        for name in module_scope():sys.modules.pop(name,None)
        sys.modules.update(previous);sys.path[:]=previous_paths

def load_isolated(public,directory,model_dir,tokenizer,device):
    with activate({},[str(directory)]+sys.path):
        obj=public.load_solution(directory,model_dir,tokenizer,device)
        modules=module_scope();paths=sys.path[:]
        assert modules,'No models package loaded'
        files={k:str(Path(v.__file__).resolve()) for k,v in modules.items() if getattr(v,'__file__',None)}
        assert all(Path(p).is_relative_to(directory.resolve()) for p in files.values()),files
    return obj,modules,paths,files

def summarize(records):
    summary=[]
    for case in dict.fromkeys(r['load_id'] for r in records):
        sets={label:[r for r in records if r['load_id']==case and r['variant']==label] for label in ['v23','v6']}
        row={'load_id':case}
        for label,items in sets.items():
            assert len(items)==3
            vals=[r['time_s'] for r in items]
            assert all(math.isfinite(t) and t>0 for t in vals)
            row[label]={'times_s':vals,'median_s':statistics.median(vals),'min_s':min(vals),'max_s':max(vals)}
        row['v6_elapsed_change_pct']=(row['v6']['median_s']/row['v23']['median_s']-1)*100
        summary.append(row)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('repeat_official_test',root/'public_test.py')
    public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
    public.configure_runtime()
    import torch
    from tokenizers import Tokenizer
    import importlib.metadata
    assert torch.cuda.is_available(),'DCU is not visible'
    device=torch.device('cuda:0')
    assets=root/'progen2';model_dir=assets/'checkpoints/progen2-medium';tokenizer_path=assets/'tokenizer.json'
    tok=Tokenizer.from_file(str(tokenizer_path));cases=json.loads((root/'public_test_data/public_cases.json').read_text(encoding='utf8'))['loads']
    assert len(cases)==4
    hashes=json.loads((here/'source_hashes.json').read_text(encoding='utf8'))
    versions={'torch':str(torch.__version__),'hip':torch.version.hip,'device':torch.cuda.get_device_name(device)}
    for name in ['triton','transformers','tokenizers','numpy','accelerate']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
    objects={};provenance={}
    for label in ['v23','v6']:
        directory=here/'candidates'/('repeat_'+label)
        for name,expected in hashes[label].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected,(label,name)
        objects[label]=load_isolated(public,directory,model_dir,tokenizer_path,str(device))
        provenance[label]=objects[label][3]
        print('LOADED='+label,flush=True)
    records=[];warmups=[]
    def save():
        (out/'measurements.json').write_text(json.dumps({'purpose':'independent_repeated_diagnostic_not_official_score','design':'two resident models, sequential calls; no simultaneous inference; startup excluded; AB/BA/AB for each load',
            'versions':versions,'source_hashes':hashes,'loaded_module_paths':provenance,'warmups':warmups,'records':records},indent=2)+'\n',encoding='utf8')
    save()
    for case in cases:
        for prefixes in [case['warmup_prefixes'],case['prefixes']]:
            public.validate_prefixes(tok,prefixes,case['batch_size'],case['prefix_tokens'])
        public.validate_outputs(case['reference_outputs'],case['batch_size'],case['new_tokens'])
        for label in ['v23','v6']:
            obj,modules,paths,_=objects[label]
            with activate(modules,paths):
                obj.reset();before=obj._compile_events[0]
                result=obj.generate(case['warmup_prefixes'],case['new_tokens']);torch.cuda.synchronize(device)
                public.validate_outputs(result,case['batch_size'],case['new_tokens'])
                warmups.append({'load_id':case['id'],'variant':label,'compile_events':obj._compile_events[0]-before})
            save()
        for round_id,order in enumerate([['v23','v6'],['v6','v23'],['v23','v6']],1):
            for label in order:
                obj,modules,paths,_=objects[label]
                with activate(modules,paths):
                    obj.reset();torch.cuda.synchronize(device);before=obj._compile_events[0]
                    started=time.perf_counter();result=obj.generate(case['prefixes'],case['new_tokens']);torch.cuda.synchronize(device)
                    elapsed=time.perf_counter()-started;events=obj._compile_events[0]-before
                result=public.validate_outputs(result,case['batch_size'],case['new_tokens'])
                exact=result==case['reference_outputs']
                records.append({'load_id':case['id'],'variant':label,'round':round_id,'time_s':elapsed,'exact_match':exact,'compile_events':events})
                save();print(f"[{case['id']}] round={round_id} {label} time={elapsed:.6f}s exact={exact} compile={events}",flush=True)
                if not exact or events!=0:raise RuntimeError('Accuracy or formal compile check failed; partial evidence saved')
    summary=summarize(records)
    sums={label:sum(r[label]['median_s'] for r in summary) for label in ['v23','v6']}
    (out/'summary.json').write_text(json.dumps({'purpose':'diagnostic_not_official_score','loads':summary,'sum_of_load_medians_s':sums,'v6_elapsed_change_pct':(sums['v6']/sums['v23']-1)*100,
            'all_exact_and_formal_compile_zero':True,'limitations':['3 repeats are initial screening, not proof of sub-percent improvement','two models reside on device; official single-candidate score not replaced']},indent=2)+'\n',encoding='utf8')
    print('REPEAT_DIAGNOSTIC=PASS',flush=True)

if __name__=='__main__':main()
