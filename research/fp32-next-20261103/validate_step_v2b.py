from pathlib import Path
from collections import namedtuple
import ast, hashlib, importlib.util, json, types
import torch

torch.set_num_threads(1)
root=Path('deliverables/fp32-next-20261103')
old_path=Path('deliverables/fp32-decode-mask-20261102/candidates/fp32_decode_mask_v1/solution.py')
new_path=root/'candidates/fp32_decode_step_v2b/solution.py'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solution

Old,New=load(old_path,'step_old'),load(new_path,'step_new')
Out=namedtuple('Out',['logits','past_key_values'])

class Tokenizer:
    def encode(self,s): return types.SimpleNamespace(ids=[ord(c)%7 for c in s])
    def id_to_token(self,i): return chr(65+i)

class StatefulModel(torch.nn.Module):
    def __init__(self,tie=False):
        super().__init__()
        self.tie=tie
    def forward(self,input_ids,past_key_values=None,**kwargs):
        state=torch.zeros((input_ids.shape[0],1),device=input_ids.device) if past_key_values is None else past_key_values
        state=state+input_ids.float().sum(dim=-1,keepdim=True)
        if self.tie:
            scores=torch.zeros((input_ids.shape[0],1,8),device=input_ids.device)
            scores[:,:,2]=10.
            scores[:,:,5]=10.
        else:
            choices=torch.tensor([5,2,1],device=input_ids.device)
            desired=choices[(state.long()%3).squeeze(-1)]
            scores=-(torch.arange(8,device=input_ids.device)[None,None,:]-desired[:,None,None]).abs().float()
        return Out(scores,state)

def setup(cls,new,device,tie):
    s=cls.__new__(cls)
    s.torch=torch
    s.device=torch.device(device)
    s.tokenizer=Tokenizer()
    s.allowed_ids=torch.tensor([5,2,1],device=device)
    s.model=StatefulModel(tie).to(device).eval()
    seen=[]
    if new:
        s._install_compile_monitor()
        compiled=torch.compile(s._generation_step,backend='eager',fullgraph=True)
        def spy(ids,past):
            seen.append(past is None)
            token,cache=compiled(ids,past)
            assert cache.dtype==torch.float32 and token.dtype==torch.long
            return token,cache
        s.compiled_step=spy
    return s,seen

results=[]
devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
for device in devices:
    for tie in [False,True]:
        # These are independent model/device fixtures, unlike sequential requests.
        # Isolate their compiler caches; never reset between requests below.
        torch._dynamo.reset()
        old,_=setup(Old,False,device,tie)
        new,seen=setup(New,True,device,tie)
        cases=[(['AAAA'],12),(['ABCD','EFGH'],5),(['A'*32]*8,4),(['B'*512]*4,3),(['BBBB'],7),(['AAAA'],12)]
        outputs=[]
        for prefixes,steps in cases:
            seen.clear()
            expected=old.generate(prefixes,steps)
            actual=new.generate(prefixes,steps)
            assert actual==expected
            assert len(seen)==steps and seen==[True]+[False]*(steps-1)
            if tie: assert all(set(s)=={'F'} for s in actual)
            outputs.append(actual)
            results.append({'device':device,'tie':tie,'batch':len(prefixes),'prefix_length':len(prefixes[0]),'steps':steps,'equal':True,'cache_starts_empty':True})
        assert outputs[0]==outputs[-1]
        if not tie: assert len(set(''.join(outputs[0])))>1

monitor_cases=[]
for device in devices:
    torch._dynamo.reset()
    monitored,_=setup(New,True,device,False)
    for batch,length,steps in [(1,32,5),(8,32,5),(4,32,7),(4,512,3)]:
        start=monitored._compile_events[0]
        monitored.generate(['A'*length]*batch,steps)
        warmed=monitored._compile_events[0]
        monitored.generate(['B'*length]*batch,steps)
        end=monitored._compile_events[0]
        assert end==warmed
        monitor_cases.append({'device':device,'batch':batch,'length':length,'warmup_events':warmed-start,'measured_events':end-warmed})

tree=ast.parse(new_path.read_bytes())
generate=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate')
cpu_calls=[n for n in ast.walk(generate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cpu']
assert len(cpu_calls)==1
for loop in (n for n in ast.walk(generate) if isinstance(n,ast.For)):
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['cpu','item','tolist'] for n in ast.walk(loop))
result={'candidate_source_sha256':hashlib.sha256(new_path.read_bytes()).hexdigest(),'torch':torch.__version__,'hip':torch.version.hip,'cases':results,'compile_monitor_cases':monitor_cases,'fullgraph_backend':'eager','fixture_isolation':'Dynamo reset between independent device/tie fixtures, never between requests.','source_validation':'Stateful test model with tensor cache; same generate methods as real candidates; not full ProGen/Inductor/DCU.','single_cpu_transfer_outside_generation_loop':True}
(root/'step_v2b_validation.json').write_bytes(json.dumps(result,indent=2).encode())
print('PASS',len(results),'CPU/CUDA stateful comparisons, A-B-A, cache reset, ties, FP32 and fullgraph eager capture')
