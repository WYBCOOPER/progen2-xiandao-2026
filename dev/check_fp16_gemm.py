"""快速诊断：DCU 上 fp16 矩阵乘到底错多少（不跑模型，10 秒出结果）。

用法（DCU 容器里）：
    python dev/check_fp16_gemm.py

看四组数字：
  * 500-600 量级 vs fp32？  -> 用最大绝对误差 / 参考标准差 判断
  * fp16 default 与 fp16 reduced=False 的差距 -> 判断低精度累加开关是否生效
  * F.linear(fp16 权重) 与 "权重舍入但 fp32 计算" 的差距 -> 判断误差是权重舍入还是 GEMM 实现
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
if not torch.cuda.is_available():
    raise SystemExit("需要 CUDA/HIP 设备")

B, K, N = 8, 1536, 4608
a = torch.randn(B, K, device=dev)
b = (torch.randn(K, N, device=dev) * 0.02)  # 接近 ProGen 权重量级

ref = a @ b


def report(tag, out):
    out = out.float()
    d = (out - ref).abs()
    print("%-42s max_abs=%.6f  max_abs/ref_std=%.6f" % (tag, d.max().item(), (d.max() / ref.std()).item()))


print("torch:", torch.__version__, "| hip:", torch.version.hip, "| device:", torch.cuda.get_device_name(0))
print("ref std:", ref.std().item())

# 1) 纯 fp16 矩阵乘（默认允许 reduced precision reduction）
report("1) fp16 x fp16 matmul (default)", a.half() @ b.half())

# 2) 关闭 reduced precision reduction
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
report("2) fp16 x fp16 matmul (reduced=False)", a.half() @ b.half())
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# 3) 只做权重舍入、用 fp32 计算（误差下界：量化本身的误差）
report("3) fp32 compute + fp16-rounded weights", F.linear(a, b.half().float()))

# 4) 我们的实际路径：fp16 输入 + fp16 权重
report("4) F.linear(fp16 input, fp16 weight)", F.linear(a.half(), b.half()))

# 5) 再跑一次 4)，确认是否确定性
report("5) repeat of 4)", F.linear(a.half(), b.half()))
