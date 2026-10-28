# 先导杯三人协作空间

目标：在指定海光 DCU 上优化公共 ProGen2-medium 的生成推理，完整完成四项测试，保留一致性和耗时证据，按赛题要求提交源码、报告与记录。

本目录用于队内协作。最终参赛 ZIP 只打包规定的 submission/，协作材料不必全部放入参赛包。

## 从这里开始

- [三人分工与每日协作](docs/TEAM.md)
- [任务进度分区](docs/BOARD.md)
- [实验记录规范](docs/EXPERIMENTS.md)
- [贡献确认规则](docs/CONTRIBUTIONS.md)
- [提交检查](docs/RELEASE.md)
- [A 的进度与贡献](members/A.md) · [B 的进度与贡献](members/B.md) · [C 的进度与贡献](members/C.md)

目前三位成员暂用 A、B、C，真实姓名、队长与 GitHub 账号待队内填写。A/B/C 仍为建议角色，尚未绑定真人。2026-10-28 已归档实际运行材料，T01、T04、T08 待复核；其余任务仍按任务表推进。

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

| 日期（北京时间） | 队内目标 |
|---|---|
| 10 月 28 日—29 日 | 明确规则、跑通 DCU 环境和 FP32 基准 |
| 10 月 30 日—31 日 | 实现自己的生成接口，完成四项测试 |
| 11 月 1 日—3 日 | 一次只改一处，进行有证据的优化对比 |
| 11 月 4 日 | 选定并冻结最终代码，完成完整测试 |
| 11 月 5 日 20:00 前 | 完成报告、交叉检查与打包，这是队内目标 |

官方截止：2026 年 11 月 5 日 24:00，即 11 月 6 日 00:00。以上阶段是建议安排，延误时优先保证可运行版本和真实记录。
