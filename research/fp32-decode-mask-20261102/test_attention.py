import ast, pathlib, torch
from torch import nn
base=pathlib.Path('deliverables/pr22-abbfeb5/candidates/pr22_abbfeb5/models/progen/modeling_progen.py'); new=pathlib.Path('deliverables/fp32-decode-mask-20261102/candidates/fp32_decode_mask_v1/models/progen/modeling_progen.py')
def load(p):
 tree=ast.parse(p.read_text(encoding='utf8')); cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ProGenAttention'); f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_attn'); mod=ast.Module(body=[f],type_ignores=[]); ns={'torch':torch,'nn':nn};exec(compile(mod,str(p),'exec'),ns);return ns['_attn']
a,b=load(base),load(new)
class Obj:pass
o=Obj();o.bias=torch.tril(torch.ones(32,32,dtype=torch.bool)).view(1,1,32,32);o.scale_attn=torch.tensor(2.);o.masked_bias=torch.tensor(-1e9);o.attn_dropout=nn.Dropout(0.)
torch.manual_seed(17);count=0
for batch in (1,3):
 for qlen,klen in ((1,1),(1,7),(1,16),(4,4),(3,8)):
  for masked in (False,True):
   q=torch.randn(batch,2,qlen,4); k=torch.randn(batch,2,klen,4);v=torch.randn(batch,2,klen,4);am=torch.zeros(batch,1,1,klen) if masked else None;hm=torch.tensor([1.,0.]).view(1,2,1,1) if masked else None
   if masked:am[...,0]=-10000.
   x=a(o,q,k,v,am,hm);y=b(o,q,k,v,am,hm);assert all(torch.equal(i,j) for i,j in zip(x,y));
   if masked:assert torch.all(y[1][:,1]==0) and (klen==1 or qlen>1 or torch.all(y[1][:,0,:,0]==0))
   if qlen>1 and klen==qlen: assert torch.all(y[1].triu(1)==0)
   count+=1
print('PASS',count,'bitwise output and weights; masks and prefill checked')
