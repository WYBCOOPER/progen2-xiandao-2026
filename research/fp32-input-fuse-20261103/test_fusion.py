"""Actual block classes; eager equivalence and fullgraph capture, not Inductor/DCU."""
import ast, copy, json, math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

work = Path(__file__).resolve().parent
path = work / "candidates/fp32_input_fuse_v4/models/progen/modeling_progen.py"
names = {"rotate_every_two", "apply_rotary_pos_emb", "ProGenAttention", "ProGenMLP", "ProGenBlock"}
nodes = [n for n in ast.parse(path.read_bytes()).body
         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))
ns = {"torch": torch, "nn": nn, "ACT2FN": {"gelu_new": gelu_new}}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
config = SimpleNamespace(max_position_embeddings=32, hidden_size=16, num_attention_heads=8,
                         attn_pdrop=0., resid_pdrop=0., rotary_dim=2, n_ctx=32, n_embd=16,
                         n_inner=32, activation_function="gelu_new", layer_norm_epsilon=1e-5)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
records = []
def compare_outputs(a, b, length):
    torch.testing.assert_close(a[0], b[0], atol=1e-6, rtol=1e-5)
    assert a[1][2] == b[1][2] == length
    for i in range(2):
        torch.testing.assert_close(a[1][i][:, :, :length], b[1][i][:, :, :length], atol=1e-6, rtol=1e-5)
for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    for mlp_bias in [True, False]:
        for qkv_bias in [True, False]:
            torch.manual_seed(1)
            a = ns["ProGenBlock"](config).to(device).eval()
            if not mlp_bias:
                a.mlp.fc_in.bias = None
            if qkv_bias:
                a.attn.qkv_proj.bias = nn.Parameter(torch.randn(a.attn.qkv_proj.out_features, device=device))
            b = copy.deepcopy(a)
            before = {k: v.clone() for k, v in b.state_dict().items()}
            b.install_input_fusion()
            b.install_input_fusion()
            assert before.keys() == b.state_dict().keys()
            assert all(torch.equal(v, b.state_dict()[k]) for k, v in before.items())
            for projection in [b.attn.qkv_proj, b.mlp.fc_in]:
                assert projection.weight.untyped_storage().data_ptr() == b._input_fused_weight.untyped_storage().data_ptr()
                if projection.bias is not None:
                    assert projection.bias.untyped_storage().data_ptr() == b._input_fused_bias.untyped_storage().data_ptr()
            with torch.inference_mode():
                for length in [1, 4]:
                    x = torch.randn(2, length, 16, device=device)
                    old, new = a(x, use_cache=True), b(x, use_cache=True)
                    compare_outputs(old, new, length)
                    x = torch.randn(2, 1, 16, device=device)
                    compare_outputs(a(x, layer_past=old[1], use_cache=True),
                                    b(x, layer_past=new[1], use_cache=True), length + 1)
            records.append({"device": device, "mlp_bias": mlp_bias, "qkv_bias": qkv_bias, "passed": True})
            print(device, "mlp_bias", mlp_bias, "qkv_bias", qkv_bias, "PASS", flush=True)
    torch._dynamo.reset()
    a = ns["ProGenBlock"](config).to(device).eval()
    b = copy.deepcopy(a)
    b.install_input_fusion()
    compiled = torch.compile(b, backend="eager", fullgraph=True)
    with torch.inference_mode():
        x = torch.randn(2, 4, 16, device=device)
        old, new = a(x, use_cache=True), compiled(x, use_cache=True)
        compare_outputs(old, new, 4)
        x = torch.randn(2, 1, 16, device=device)
        compare_outputs(a(x, layer_past=old[1], use_cache=True),
                        compiled(x, layer_past=new[1], use_cache=True), 5)
    records.append({"device": device, "fullgraph_capture_backend": "eager", "passed": True})
    print(device, "FULLGRAPH_CAPTURE_PASS", flush=True)
(work / "block_validation.json").write_text(json.dumps({
    "torch": torch.__version__, "scope": __doc__, "cases": records
}, indent=2), encoding="utf-8")
