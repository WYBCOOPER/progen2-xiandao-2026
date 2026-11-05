# fp32_decode_attn_v8b — fixed1

修复上一轮 Inductor launcher 不接受 enable_fp_fusion 参数的问题。v9b 用 AMD FP32 乘法指令保留乘法与加法分步舍入，严格相等检查不放宽。v8b 由平台默认编译选项编译，仍须通过小张量误差和完整序列精确一致检查。

新增 compile_check.py：先对随机小张量执行真实 torch.compile 包装调用，检查相同接口及原位写入，再进入模型。这里只是准备检查；新版本尚未在 DCU 运行。

依据：https://github.com/pytorch/pytorch/blob/v2.4.1/torch/_inductor/codegen/wrapper.py 及 runtime/triton_heuristics.py；AMD 乘法指令约束参考 PyTorch test/higher_order_ops/test_inline_asm_elementwise.py，Triton 3.0 inline_asm_elementwise API。

方向1：FP32 单字母 decode 注意力，自编 Triton 分块 softmax 与加权求和；不使用低精度或 tl.dot。

从 v23 独立分支，两个方向不互相叠加；官方模型权重、测试器、公开样例、baseline、submission 不改。FP32、不启用 TF32。多 token 预填充保留原逻辑；不支持的常规语义走原路径，核编译或执行失败会停止，不静默关闭编译。

执行 bash candidates/fp32_decode_attn_v8b/run_test.sh。脚本先核验文件与环境，再在正式测试外做随机小张量核检查，通过后调用原 run_public_test.sh 跑四项。每次独立目录，成功或失败均自动打包。核检查只用于发现越界/数值错误，不是推理成绩，也不向模型提供任何请求相关 KV。

验收：四项生成与参考 100% 一致，四次正式 generate 的 Dynamo 事件和 Triton JIT cache-miss 事件都为 0。核小张量允许误差不能替代整段生成精确一致。统计钩子不是所有原生编译活动的硬件追踪。按比赛规定的预热流程使用；若后续评测不允许当前预热方式，需调整预编译流程再提交。

本机无 Triton，尚未验证核编译、DTK/Inductor 捕获或 DCU 模型输出。已完成语法和本地参考算法检查；不声明加速收益。direction1 改变 softmax 归约顺序，可能导致序列不一致。direction2 的原位写缓存与动态位置也需要 DCU 验证。

实现参考：PyTorch v2.4.1 test/inductor/test_triton_kernels.py 的用户 Triton 调用/原位更新测试；Triton v3.0.0 python/triton/runtime/jit.py 的 do_not_specialize、cache_hook 接口。核算法源码为本次实验自编，Apache 原模型头保留。未修改 Triton 本身。
