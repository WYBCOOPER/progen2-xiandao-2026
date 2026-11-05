# PR24 最新更新检查

固定提交：1bda90f824b34bfe4a522bef14f6d446661839ad；对比已测旧版5b30ef2，本次仅改2个文件。已下载4份Python源码及实验说明，保留分支原样；没有改本地已测候选或GitHub代码。

| 文件 | 本次新增内容 |
|---|---|
| submission/solution.py | ALLOW_FP16_REDUCED_PRECISION=False，同时设置FP16/BF16的reduction开关；COMPILE_MODEL=True，可改False关闭compile |
| experiments/EXP-20261102-01.md | 补旧版低一致率数据、原因假设和T1—T4诊断顺序 |

FP16、scope=all、108个Linear替换仍沿用旧版；模型文件、KV和贪心生成循环没有在本提交新增改动。本版仍是default编译模式，没有叠加本地已测组合版的单步包装与矩阵乘调优。

判断边界：这是诊断版，尚无新DCU结果；关闭低精度reduction不会恢复已舍入的半精度权重/输入，也不保证输出一致。属性设置成功不等于确认DTK/编译后端实际采取了预期路径，需要实测；设置异常会只打印后继续运行，应核对启动信息。

报告中“1/256说明首token已错”无法由汇总匹配数推出；需要逐位置数据。即使首token不同，也不能排除半精度舍入改变argmax。因此目前不能锁定为累加精度或编译根因。

分支仍缺submission/models/__init__.py。仅按原分支复制时可能让官方资产中的同名models包遮蔽候选；制作DCU测试包需另补加载标记并记录为包装修复，不能把修复后快照冒充分支原字节。

下一步：先在独立候选目录测试默认T1（半精度reduction关闭、compile开启），先看一致率；必要时T2只切换COMPILE_MODEL=False。当前未合并此PR、未运行真实DCU测试。

参考：[增量差异](https://github.com/WYBCOOPER/progen2-xiandao-2026/compare/5b30ef255984c72d2d2c6e87485b0fb8958a7d03...1bda90f824b34bfe4a522bef14f6d446661839ad)；[PyTorch数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)。后者为通用说明，不代表定制DCU后端已核实。
