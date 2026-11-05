import importlib.util,pathlib,torch,types
from collections import namedtuple
p=pathlib.Path('deliverables/fp32-decode-mask-20261102/candidates/fp32_decode_mask_v1/solution.py');q=pathlib.Path('deliverables/fp32-next-20261103/candidates/fp32_decode_step_v2/solution.py')
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.Solution
class Tok:
 def encode(self,s):return types.SimpleNamespace(ids=[ord(c)%7 for c in s])
 def id_to_token(self,i):return chr(65+i)
Out = namedtuple("Out", ["logits", "past_key_values"])
class Model(torch.nn.Module):
 def forward(self,input_ids,past_key_values=None,**kw):
  past=torch.zeros((input_ids.shape[0],1),dtype=torch.float32) if past_key_values is None else past_key_values
  state=past+input_ids.float().sum(dim=-1,keepdim=True)
  scores=torch.zeros((input_ids.shape[0],1,8),dtype=torch.float32,device=input_ids.device)
  # tie between IDs 2 and 5; order in allowed_ids selects 5 first
  scores[:,:,2]=state;scores[:,:,5]=state
  return Out(scores,state)
def setup(cls,new):
 s=cls.__new__(cls);s.torch=torch;s.device=torch.device('cpu');s.tokenizer=Tok();s.allowed_ids=torch.tensor([5,2,1]);s.model=Model()
 if new:s.compiled_step=torch.compile(s._generation_step,backend='eager',fullgraph=True)
 return s
a=setup(load(p,'old_solution'),False);b=setup(load(q,'new_solution'),True)
for inputs,n in [(['AAAA'],4),(['AB','CD'],3),(['A'*64],5),(['BBBB'],2),(['AAAA'],4)]:
 x=a.generate(inputs,n);y=b.generate(inputs,n);assert x==y,(x,y);assert all(set(v)<=set('F') for v in y);print(len(inputs),len(inputs[0]),n,x)
assert b.model.training and b.model(torch.tensor([[1]])).logits.dtype==torch.float32
print('PASS: fullgraph eager backend; output equality, ties, A-B-A and FP32')
