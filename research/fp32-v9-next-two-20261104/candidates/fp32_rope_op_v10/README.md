# v10：明确 RoPE/KV 的修改范围

独立基于已通过 DCU 测试的 v9c。RoPE Triton 核、本来的启动参数、FP32 舍入、注意力和生成规则均保留；只用 torch.library.custom_op 声明 K/V 缓存会被修改，QKV 与位置表只读，并注册输出形状的 fake 实现。

目的：v9c 日志显示编译器无法分析内联乘法汇编，因此保守认为每个输入都可能修改。本版尝试去掉这部分多余处理；自定义算子也增加调度开销，是否有收益必须实测。

运行：`bash candidates/fp32_rope_op_v10/run_test.sh`。脚本先核验源码，检查直接/编译调用及缓存读写，然后执行官方四项测试。必须 4608/4608 一致、正式调用两类编译计数为零；失败也自动打包，不隐藏异常、不切换普通执行。

PyTorch 2.4.1 的 custom_op / mutates_args / register_fake 接口：
https://github.com/pytorch/pytorch/blob/v2.4.1/torch/_library/custom_ops.py
int 参数在该版本推断为 SymInt：
https://github.com/pytorch/pytorch/blob/v2.4.1/torch/_library/infer_schema.py

这是优化候选，尚未完成 DCU 验证。正式编译审查仍不可省略；本地替代核测试不等同于 HIP/Triton 或完整模型测试。
