# 先导杯 ProGen2 DCU 优化

更新：2026-11-05。按四项公开任务同一轮总耗时选择，**当前已记录最快完整版本为 V23：5.532917 秒**，4608/4608 输出一致，正式调用的编译、Triton 缓存未命中和图捕获计数均为零。

| 版本/轮次 | 四项合计 | 证据状态 |
|---|---:|---|
| V23 batch-shared Linear | **5.532917 s** | 当前最低有效合计；一轮新旧对照 |
| V22b 同轮对照 | 5.683739 s | V23 合计减少 2.6536% |
| V22b 历史最低整轮 | 5.639842 s | 另有 ABBA 对照确认 B1 收益 |
| V20 历史最低整轮 | 5.778319 s | 历史回顾，非同轮收益依据 |

总时间是四项规定预热后 time_s 的算术和，不是官方总评分。没有拼接不同轮次的最快项。V23 的 B4 两项本轮减少 6.53%/3.69%，B8 慢 1.34%，B1 慢 2.44%；B1 实现未改。这一轮支持按合计选 V23，不代表稳定提升或隐藏测试已通过。

- [最新完整结论](experiments/EXP-20261105-all-results.md)
- [50 组整轮成绩排名](records/20261105-speed-ranking.json)
- [V23 同轮原始数据的脱敏副本](records/v23-verified-20261105/comparison.md)
- [最快候选源码](candidates/fp32_batched_linear_v23/README.md) · [V22b 保底版本](candidates/fp32_split_gemv_v22b/README.md)
- [所有本地实验成果](research/) · [各归档的成绩、日志和源码](records/evidence-20261105/archive_index.json)
- [历史可视化](visualizations/20261103/README.md) · [分工](docs/TEAM.md) · [状态](docs/BOARD.md)

## 运行

在已有官方模型权重、Tokenizer 和测试工具的 progen2_contest 目录放入候选目录，再运行：
```bash
bash candidates/fp32_batched_linear_v23/run_test.sh
```
脚本先在独立进程比较 B4/B8 合成矩阵，保留冻结的 B1 配置。新内核两轮均达到门槛后才进行完整模型与 V22b 对照；否则保留原实现并归档。测量时的形状配置保存在 measured_plan.json，复现实测版本需核对该配置和源码指纹，重新调优可能选出不同配置。

本次同步源码与全部可公开成果，没有替换 submission/ 或根目录 solution.py，也没有发送比赛作品。两者不代表当前推荐方案。最终作品仍需选定固定配置、从最终材料复现并完成报告。

环境：DCU BW/gfx936，Python3.10.16，Torch2.4.1+das.opt2.dtk2504，HIP6.1.25065，Triton3.0.0+das.opt4.dtk2504。完整依赖及官方工具指纹见 runtime_versions.json；实际镜像名称仍需填写。设计、代码和本地检查有 Codex/Terra 协助；DCU 测试由 WYBCOOPER 执行。
