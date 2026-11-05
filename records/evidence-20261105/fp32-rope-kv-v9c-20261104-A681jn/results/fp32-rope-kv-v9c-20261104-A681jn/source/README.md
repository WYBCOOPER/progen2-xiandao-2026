# FP32 RoPE/KV v9c：修复动态批量的编译元数据

v9b 截图显示 `triton_meta.constants` 中 `B: s2`，生成文件加载时报 `NameError: name 's2' is not defined`。这与上一轮 launcher 关键字错误不同，尚不能据此判断速度或模型精度。

v9c 删除核函数的 B 参数，以第二个 grid 轴表示批量；缓存长度 S 也改为运行时标量。唯一 constexpr 是字面整数 BLOCK=256，不把张量尺寸表达式写进编译常量。RoPE 算法、FP32 分步舍入与 KV 写入语义保持不变。

执行：`bash candidates/fp32_rope_kv_v9c/run_test.sh`。先做直接核检查，再用 dynamic=True 的小张量编译检查覆盖 B=1、2、8，最后执行官方四项测试。完整生成必须 4608/4608 一致，正式调用的 Dynamo 编译和 Triton cache-miss 计数必须为零。检查失败停止并保存证据，不自动改为普通执行。

此包只修复方向2，不重测或覆盖方向1的结果。官方 baseline、模型权重、公开数据和测试器不修改。每轮建立新结果目录并自动打包。

本机仅能检查 Python/Bash 语法、批量索引覆盖和验收逻辑；没有在当前 DCU 上执行 v9c，不宣称已经解决运行问题或取得加速。

原因依据：PyTorch v2.4.1 的 define_user_defined_triton_kernel 将 constexpr 参数放入 constants，再直接输出元数据表示；其运行时参数单独进入 signature。截图中 B 的符号被写进 constants，与该路径吻合。
https://github.com/pytorch/pytorch/blob/v2.4.1/torch/_inductor/codegen/wrapper.py
