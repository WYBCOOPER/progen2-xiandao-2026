# 先导杯三人协作空间

目标：在指定海光 DCU 上优化公共 ProGen2-medium 的生成推理，完整完成四项测试，保留一致性和耗时证据，按赛题要求提交源码、报告与记录。

本目录用于队内协作。最终参赛 ZIP 只打包规定的 submission/，协作材料不必全部放入参赛包。

## 从这里开始

- [10 月 30 日当前进度与三人执行单](docs/TODAY_20261030.md)
- [10 月 29 日逐步执行单（历史安排）](docs/EXECUTION_20261029_STEPS.md)

- [10 月 29 日具体任务与协作流程](docs/TODAY_20261029.md)
- [与已有规则的调整对照](docs/WORKFLOW_COMPARISON_20261029.md)

- [优化路线与验证计划](docs/OPTIMIZATION_PLAN.md)
- [逐日执行计划：每日 22:00 交付、23:00 审查](docs/SCHEDULE.md)
- [三人分工与每日协作](docs/TEAM.md)
- [任务进度分区](docs/BOARD.md)
- [实验记录规范](docs/EXPERIMENTS.md)
- [贡献确认规则](docs/CONTRIBUTIONS.md)
- [提交检查](docs/RELEASE.md)
- [A 的进度与贡献](members/A.md) · [B 的进度与贡献](members/B.md) · [C 的进度与贡献](members/C.md)

截至 2026-10-30，PR #15—#18 均已合并：FP32 恢复、初步调用诊断、位置表预计算与 torch.compile 候选及两轮 DCU 公开测试已归档。组合候选四负载两轮均 100% 一致；相对此前 FP32 补跑的时间比值约 3.00–3.70 倍，尚未进行同场交替对照，不能据此确认稳定收益或最终成绩。当前阶段是补齐版本一致性、请求独立性、单项拆分和稳定性验证。

已知代码提交账号为 [loveviolet2](members/loveviolet2.md)，DCU 实际执行与归档账号为 [WYBCOOPER](members/WYBCOOPER.md)。第三位成员的账号与三人 A/B/C 对应关系仍待本人确认；已有贡献不再写成“队友均未开始”。材料合并与独立复核分别记录。

- [恢复后的 FP32 与诊断](experiments/EXP-20261029-01.md)
- [位置表＋compile 两轮结果与限制](experiments/EXP-20261029-03.md)
- 主分支 submission/ 比实测快照少一个 models/__init__.py，需先补齐再验收；本轮更新包含该修复，不把旧成绩当作修复后新测试。

## 首日实际记录

- [执行与贡献记录（角色待确认）](members/WYBCOOPER.md)
- [FP32 基准](experiments/EXP-20261028-01.md) · [直接 FP16 实验](experiments/EXP-20261028-02.md)
- [实验依据、共享处理和文件指纹](records/README.md)

## 使用方式

1. 从任务表领取一个小任务，每个任务只设一位主负责人，另一位复核。
2. 开始前写清要交什么、何时完成、如何验证，再改为进行中。
3. 工作结束后附代码、文档或实验记录链接，状态改为待复核。
4. 复核人检查证据并留下结论后，任务才改为已完成。
5. 每人每天用五分钟更新个人页：今天交付、证据、卡点、下一步。卡住也照实记录。

## GitHub 任务入口

仓库：[WYBCOOPER/progen2-xiandao-2026](https://github.com/WYBCOOPER/progen2-xiandao-2026)，按队内要求公开。以下入口按任务标题中的 A、B、C 分组，不代表已绑定真实 GitHub 账号。

- [全部任务](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues)
- [A 的任务](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues?q=is%3Aissue%20%22%5BA%5D%22%20in%3Atitle)
- [B 的任务](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues?q=is%3Aissue%20%22%5BB%5D%22%20in%3Atitle)
- [C 的任务](https://github.com/WYBCOOPER/progen2-xiandao-2026/issues?q=is%3Aissue%20%22%5BC%5D%22%20in%3Atitle)

Issue 管理当前状态，任务表保存导航。尚未设置 GitHub Projects；本页与任务表提供文档分区。Issues 中的 A、B、C 是建议负责人，确认成员账号后再设置实际指派。Public 允许所有人查看，队员直接写入仍需仓库协作者权限；添加协作者由仓库管理员执行。

公开空间只放可公开的技术记录；不上传 DCU 密码、令牌、完整学号、未经确认可公开的原始题包或个人信息。最终报告中的完整报名资料在正式提交材料中填写，不必公开。

## 时间目标

全部采用北京时间，每天 22:00 前提交当天成果，23:00 前结束审查。按[逐日计划](docs/SCHEDULE.md)推进：

| 日期 | 交付重点 |
|---|---|
| 10-28 | 首日归档、优化路线、逐日计划与授权合并 |
| 10-29 | DCU 恢复与初步诊断；本地基准、流程定位和 CUDA 验证适配准备 |
| 10-30 | 对齐已合并代码与实测副本，验证请求独立性，开始基准/位置表/compile/组合拆分对照 |
| 10-31 | 补齐四配置对照与多轮稳定性，判断组合收益及重编译影响 |
| 11-01 | 候选方案的稳定性验证和取舍 |
| 11-02 | 有依据时研究 KV；否则补验证与修复 |
| 11-03 | 选定方案，报告初稿 |
| 11-04 | 代码冻结、最终完整测试和报告主体 |
| 11-05 | 20:00 报告定稿，21:00 检查 ZIP，22:00 前人工发送比赛邮件，23:00 前核查发送与附件 |

官方截止：2026 年 11 月 5 日 24:00，即 11 月 6 日 00:00。最后一小时用于异常处理。技术效果以实际测试为准，不承诺加速比例；延期时优先保留正确可运行版本与真实证据。仓库更新不代表作品邮件已发送。
