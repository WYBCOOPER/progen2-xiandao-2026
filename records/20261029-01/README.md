# 2026-10-29 恢复验证与Python调用诊断

[运行报告](../../experiments/EXP-20261029-01.md)；[原件指纹和共享处理清单](manifest.json)。

- 正常完整对照：fp32_retry_public.shared.json + fp32_retry.shared.log + solution.py。
- 首轮缺日志：fp32_public.shared.json，独立保留，不与补跑日志配对。
- 诊断：diagnostic_only.shared.json + diagnostic.shared.log + 三份统计表；全部与正常成绩分开，不作加速证据。
- python_calls.prof完整原件保留在用户本地备份，公开集合只有其指纹及提取的重点统计。文本和JSON只处理账号目录，原始成绩数值不改。
- 代码来源是官方FP32基准，SHA256与首日核验一致；参见[随包许可证](../LICENSE.txt)。没有新优化，不代表队友已经复核，也不代表独立设备复现。
