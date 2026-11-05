# FP32 decode 注意力 v8c：动态步长兼容修复

v8b 在小张量编译调用时失败：Inductor 的 constants 包含 `KB:16*s1`、`KH:s1` 等未定义符号。没有正式模型成绩。

v8c 将头数和 Q/K/V 步长改为运行时核参数；constexpr 仅保留字面分块参数 64、128、16。分块 softmax、归约、缓存与生成规则不改变。修复方向与已在 DCU 跑通的 v9c 一致，但 v8c 自身尚待 DCU 验证，不保证精度或加速。

运行 `bash candidates/fp32_decode_attn_v8c/run_test.sh`，自动检查源码、直接核、动态批量编译调用，再运行官方四项测试并打包。小张量误差容限不变；最终全部 4608 个 token 必须精确一致，四次正式调用的两类编译计数必须为零。失败停止并保存，不静默关闭编译。

不覆盖 v8b/v9c 或官方 baseline、测试器、模型权重。此版本不叠加 v9c：先确认独立方向是否有效，再考虑组合。

本地 CPU 模拟核体与语法检查不等于实际 Triton/Inductor/DCU 执行。
