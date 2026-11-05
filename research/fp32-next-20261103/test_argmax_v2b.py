import torch,importlib.util,pathlib
p=pathlib.Path('deliverables/fp32-next-20261103/candidates/fp32_decode_step_v2b/solution.py');spec=importlib.util.spec_from_file_location('v2b',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(42)
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 rows=torch.randn(1000,20,device=device); special=torch.tensor([[float('nan'),2,float('nan')]+[0]*17,[2,float('nan'),float('nan')]+[0]*17,[float('-inf')]*20,[float('inf'),float('inf')]+[0]*18,[-0.,0.]+[-1.]*18,[4.,4.]+[0.]*18],device=device);rows=torch.cat((rows,special));expected=rows.argmax(-1)
 assert torch.equal(m._greedy_local_index(rows),expected)
 f=torch.compile(m._greedy_local_index,backend='eager',fullgraph=True);assert torch.equal(f(rows),expected)
 print(device,'PASS',len(rows),'random NaN infinities signed zero ties; fullgraph eager')
