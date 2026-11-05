import torch,ast,pathlib
p=pathlib.Path('deliverables/fp32-kernels-20261104/candidates/fp32_rope_kv_v9');[ast.parse(x.read_text(encoding='utf8')) for x in p.rglob('*.py')]
torch.manual_seed(9)
for b in (1,4,8):
 qkv=torch.randn(b,1,4608);x=qkv.reshape(b,1,8,576);q,v,k=x.split(192,dim=-1);q=q.reshape(b,1,16,96);k=k.reshape(b,1,16,96);v=v.reshape(b,1,16,96);sin=torch.randn(24);cos=torch.randn(24)
 def rot(t):
  x=t[...,:48];pair=torch.stack((-x[...,1::2],x[...,::2]),-1).flatten(-2);return torch.cat((x*cos.repeat_interleave(2)+pair*sin.repeat_interleave(2),t[...,48:]),-1)
 qr,kr=rot(q),rot(k)
 for h in range(16):
  for d in range(96):
   g=h//2;loc=h%2;idx=g*576+loc*96+d;pr=g*576+loc*96+(d^1);sg=-1 if d%2==0 else 1
   qq=qkv[:,0,idx];kk=qkv[:,0,idx+384];vv=qkv[:,0,idx+192]
   if d<48:qq=qq*cos[d//2]+qkv[:,0,pr]*sg*sin[d//2];kk=kk*cos[d//2]+qkv[:,0,pr+384]*sg*sin[d//2]
   assert torch.equal(qq,qr[:,0,h,d]) and torch.equal(kk,kr[:,0,h,d]) and torch.equal(vv,v[:,0,h,d])
 print('CPU indexing reference PASS batch',b)
