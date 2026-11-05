# 10 月 29 日三人逐步执行资料包

先打开 EXECUTION.md。你执行 U1—U7，队友 1 执行 B1—B6，队友 2 执行 C1—C7。

dev/local_test.py 需复制到各自完整官方题包的 dev/ 下再运行。reference/ 为只读资料，不含公共权重和完整运行环境，不覆盖进容器。

脚本已做无 GPU 检查，尚未执行真实 NVIDIA/DCU 推理。适配输出是本地开发记录，正式验收仍使用 DCU 官方脚本。

本次 PR：https://github.com/WYBCOOPER/progen2-xiandao-2026/pull/15
