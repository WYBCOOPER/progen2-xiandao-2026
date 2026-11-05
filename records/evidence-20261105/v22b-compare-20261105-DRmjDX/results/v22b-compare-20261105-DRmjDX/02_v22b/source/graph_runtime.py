"""Input-independent decode graph; request KV is reset and filled afresh."""
import torch
from split_linear import linear
from models.progen.kernels import graph_rope_attention,record_token


class DecodeGraph:
    def __init__(self,model,allowed_ids,batch,capacity,greedy_index):
        self.model=model
        self.allowed_ids=allowed_ids
        self.greedy_index=greedy_index
        self.capacity=capacity
        device=allowed_ids.device
        self.token=torch.zeros((batch,1),dtype=torch.long,device=device)
        self.pos=torch.zeros((),dtype=torch.int32,device=device)
        self.index=torch.zeros((),dtype=torch.int32,device=device)
        self.output=torch.empty((batch,capacity),dtype=torch.long,device=device)
        self.kv=[(torch.zeros((batch,16,capacity,96),device=device,dtype=torch.float32),
                  torch.zeros((batch,16,capacity,96),device=device,dtype=torch.float32))
                 for _ in model.transformer.h]
        plan=model._v22_plan
        first=model.transformer.h[0]
        modules=(first.attn.qkv_proj,first.attn.out_proj,first.mlp.fc_in,first.mlp.fc_out)
        self.linear_modes=tuple(plan.get(f"{batch},{m.weight.shape[0]},{m.weight.shape[1]},{int(m.bias is not None)}") for m in modules)
        self.graph=None
        # Fixed shape per batch: retain fusion/GEMM tuning before graph capture.
        # Disable compiler-managed graphs because prepare() captures explicitly.
        self.compiled_decode=torch.compile(self.step,
            mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=False)

    def step(self):
        transformer=self.model.transformer
        hidden=transformer.drop(transformer.wte(self.token))
        for block,(kc,vc) in zip(transformer.h,self.kv):
            residual=hidden
            norm=block.ln_1(hidden)
            attn=block.attn
            qkv=linear(norm,attn.qkv_proj,self.linear_modes[0])
            _,y=graph_rope_attention(qkv,kc,vc,attn.rope_sin,attn.rope_cos,self.pos)
            y=attn._merge_heads(y,attn.num_attention_heads,attn.head_dim)
            y=attn.resid_dropout(linear(y,attn.out_proj,self.linear_modes[1]))
            mlp=block.mlp.dropout(linear(block.mlp.act(linear(norm,block.mlp.fc_in,self.linear_modes[2])),block.mlp.fc_out,self.linear_modes[3]))
            hidden=y+mlp+residual
        hidden=transformer.ln_f(hidden)
        logits=self.model.lm_head(hidden).to(torch.float32)[:,-1,:]
        restricted=logits.index_select(-1,self.allowed_ids)
        local=self.greedy_index(restricted)
        nxt=self.allowed_ids.index_select(0,local).unsqueeze(-1)
        record_token(nxt,self.output,self.index)
        self.token.copy_(nxt)
        self.pos.add_(1)
        self.index.add_(1)

    def prepare(self):
        # Dummy tensors only: no real request prefix or generated KV is captured.
        stream=torch.cuda.Stream(device=self.token.device)
        stream.wait_stream(torch.cuda.current_stream(self.token.device))
        with torch.cuda.stream(stream):
            self.token.fill_(0);self.pos.fill_(32);self.index.zero_()
            for _ in range(3):self.compiled_decode()
        torch.cuda.current_stream(self.token.device).wait_stream(stream)
        torch.cuda.synchronize(self.token.device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.compiled_decode()
        torch.cuda.current_stream(self.token.device).wait_stream(stream)
        torch.cuda.synchronize(self.token.device)

    def begin(self,first_token,prefill_kv,prefix_length):
        assert len(prefill_kv)==len(self.kv)
        # Reuse addresses, never reuse a previous request's computed KV.
        for (kc,vc),past in zip(self.kv,prefill_kv):
            kc.zero_();vc.zero_()
            kc[:,:,:prefix_length].copy_(past[0][:,:,:prefix_length])
            vc[:,:,:prefix_length].copy_(past[1][:,:,:prefix_length])
        self.token.copy_(first_token)
        self.pos.fill_(prefix_length)
        self.index.zero_()

    def replay(self):
        if self.graph is None:raise RuntimeError("Decode graph was not prepared")
        self.graph.replay()
