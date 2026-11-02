# 测试可视化

20轮有成绩记录（含诊断及一致性未通过的实验），V2失败无成绩另列于 data.json。历史跨日耗时仅展示；橙色为cProfile诊断。组合版与V3监测对照仅各一轮，四项合计差异2.33%，不能说明稳定收益。编译监测为Dynamo事件，并非完整冷启动或所有外部编译的证明。

- [历史耗时](history_latency.png)
- [一致率](output_consistency.png)
- [同机对照](combo_vs_v3.png)

在本目录运行 `python plot_results.py` 或打开笔记本。
