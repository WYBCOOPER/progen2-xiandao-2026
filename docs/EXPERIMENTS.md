# 实验记录规范

每次实验建立 experiments/EXP-日期-编号.md，复制模板填写。记录正式实验所用代码提交号；如有未提交修改，注明差异，不把提交号当成完整版本依据。

保留目录建议为 records/日期-编号/，放完整日志、原始 JSON 和环境输出。索引中保存相对链接。不要覆盖有引用的证据，不手改原始结果。

## 统一比较条件

同一设备规格、相同卡数、同一套规定输入和测试工具；正式计时前按工具规定预热。四项均记录 time_s、same、total、consistency_rate，不只报平均加速比或最好一项。

报告加速比 = 相同负载 FP32 耗时 / 优化耗时。总体一致率是四项一致率的平均，各项等权。性能准入逐项判断，一致率至少 80%；总体一致性得分在总体一致率大于 80% 后才开始增加。

## 正式保存命令

以下在 DCU Linux 环境的官方题目包根目录执行，不在 Windows 本地直接执行：

```bash
mkdir -p results
set -o pipefail
python check_environment.py --full-hash
bash run_baseline.sh 2>&1 | tee results/baseline.log
bash run_public_test.sh submission 2>&1 | tee results/submission.log
```

再次执行会覆盖同名文件，先归档需要保留的实验。官方入口构建自定义扩展；生成计时期间不编译、不安装依赖、不下载。

最终提交的作品 JSON 与日志必须来自最终代码的一轮完整测试，并附同规格基准记录。开发记录可保存多轮，但不能挑选不同轮次的单项最好值拼成最终成绩。

异常没有生成 JSON 时保留本轮完整日志、说明缺失原因；不能伪造 JSON 或拿旧 JSON 替代。必要时在组内共享原始日志，上传 GitHub 前检查是否包含账号凭据等无关敏感内容；保留本地完整原始件，需清理共享副本时说明处理内容。
