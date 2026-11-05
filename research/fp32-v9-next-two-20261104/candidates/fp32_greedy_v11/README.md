# v11：候选字母筛选融合

独立基于已通过 DCU 测试的 v9c，不包含 v10。模型、权重、RoPE/KV 核、注意力和 FP32 均保留。将 logits → 取 20 个允许字母 → 找最高分 → 转回 token ID 合为一个 Triton 核；平分选允许列表中第一个，NaN 同样取第一个，与父版保持一致。不是换模型或改生成规则。

目的：减少逐个比较和中间张量的处理。原编译器可能已经融合了其中一些运算，新核未必更快，收益必须实测。

运行：`bash candidates/fp32_greedy_v11/run_test.sh`。脚本自动检查核/编译调用（含 NaN、平分、非连续 logits），再执行官方四项测试。必须 4608/4608 一致、正式调用两类编译计数为零；失败保留完整证据，不隐藏异常。

Triton 3.0 的 max/min 接口来源：
https://github.com/triton-lang/triton/blob/v3.0.0/python/triton/language/standard.py

这是优化候选，尚未完成 DCU 验证。本地替代核检查不等同于真实 HIP/Triton 或完整模型测试。
