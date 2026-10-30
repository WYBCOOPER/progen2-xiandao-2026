# 任务进度分区

更新：2026-10-30（北京时间）。[当天执行单](TODAY_20261030.md) · [逐日计划](SCHEDULE.md) · [优化路线](OPTIMIZATION_PLAN.md)。

每日 22:00 前交付，23:00 前结束审查。没有运行或没有收到复核意见的部分照实记录。当前没有关闭任何尚缺验收的任务。

## 当前真实进度

PR #15、#16、#17、#18 已合并。恢复后的 FP32 完成四项 100% 一致及初步 cProfile 调用诊断；loveviolet2 提交位置表预计算＋torch.compile，WYBCOOPER 执行两轮 DCU 公开测试，四项均 100% 一致。相对此前 FP32 的观察时间比值约 3.00–3.70 倍，仍缺同场交替对照、单项拆分及请求独立性验证。

主分支 submission/ 的四个现有源文件与实测快照相同，但少 models/__init__.py；本轮修复先补空文件，随后以确切版本重新测。此前测试证明的是 records/20261029-03/source/ 中保存的副本。旧日志不改写成修复后新结果。

## 有交付，继续验证或待复核

| Issue | 已有证据 / 实际参与 | 当前状态 | 下一步 |
|---|---|---|---|
| [#1 环境](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/1) | WYBCOOPER，10-28 环境与资产校验、10-29 恢复环境 | 材料已归档，待独立复核 | 核对当前设备/软件、权重指纹与复现实验 |
| [#4 FP32](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/4) | [恢复实验](../experiments/EXP-20261029-01.md)，四项 100% 一致 | 已实测，待独立复核 | 作为同场交替对照基准，保留代码配对 |
| [#5 可运行程序](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/5) | FP32 及组合副本已实测；PR #17 已合并 | 验证与集成待补齐 | 补包文件、打印实际模型模块路径、重测新版本 |
| [#6 验证工具](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/6) | dev/local_test.py 已在 PR #15 合并；两轮 DCU 证据已归档 | 工具已有，真实本地 GPU 与状态检查证据缺失 | 原版自比较、候选比较；补 DCU 状态检查工具及证据 |
| [#7 优化](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/7) | loveviolet2 实现；[两轮组合测试](../experiments/EXP-20261029-03.md)由 WYBCOOPER 执行 | 有初步加速证据，待拆分/稳定性/复核 | 原版、仅位置表、仅 compile、组合四配置 |
| [#8 FP16](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/8) | [直接转换实验](../experiments/EXP-20261028-02.md) | 实验结束，当前不采用；记录待复核 | 不盲目重开；主线优先保持 FP32 |
| [#9 方案选择](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/9) | 已有一个组合候选和两轮公开结果 | 可开始证据复核，最终决策待完成 | 标明保留/放弃及版本，11-03 决策 |

## 仍缺队员交付

| Issue | 缺少什么 | 下一步和期限 |
|---|---|---|
| [#2 生成流程](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/2) | 一页带文件/函数定位的个人说明，不能从提交代码推断已经掌握 | 初次学习成员按当天清单先完成；10-29 原目标已过，10-30 交实际状态 |
| [#3 测试规则](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/3) | 队员独立解释四负载、计时范围、两种 80% 门槛 | 用现有三组 JSON 做逐项核对，10-30 交证据 |
| [#10 冻结](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/10) | 最终版本完整测试 | 11-04 |
| [#11 报告](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/11) | 正式报告 | 11-03 初稿，11-05 20:00 定稿 |
| [#12 提交](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues/12) | ZIP 和人工发送记录 | 11-05 21:00 检查、22:00 前发送、23:00 前核查 |

## 三人现在同时做什么

- WYBCOOPER：对齐 DCU 待测副本，保存环境/指纹，先完整回归，再串行运行四配置及交替复测。
- PR #17 的代码提交者 loveviolet2：解释实际改动，准备四配置中的代码副本和差异说明；AI 协助范围由本人补充，不代填。
- 初次学习的另一位成员（账号待确认）：完成最小代码理解，核对 JSON/日志，准备并测试请求状态检查；本地 CUDA 无法运行时先交完整错误与静态核对。

这是按当前证据提出的下一步安排，未代替本人认领。A/B/C 角色对应仍待确认；不按提交数、AI 品牌或代码行数分配贡献。合并记录没有包含 PR #17/#18 的独立 APPROVE 或队友复核结论，不将合并视为验收通过。

## 时间记录

10-29 组合日志在 23 时，压缩包 10-30 00:16 归档，未达到原 22:00/23:00 目标。PR #15—#18 于 10-30 19:42—19:43 合并。操作日期、归档日期与合并日期分开记录。后续按 22:00/23:00 节点提交实际成果或卡点。
