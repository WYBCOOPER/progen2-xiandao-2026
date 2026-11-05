# 基于 v9c 的两个独立候选

v10 只改 RoPE/KV 的算子封装；v11 只融合字母筛选。两版互不包含，不改官方测试器和基准。先分别测，不把未验证的改动直接合并。

在 progen2_contest 中解压本包。运行 `bash candidates/fp32_rope_op_v10/run_test.sh` 测第一版，运行 `bash candidates/fp32_greedy_v11/run_test.sh` 测第二版。每次自动生成独立结果目录和压缩包。两项都没有预填加速结论。

`bash run_v9_next_two_20261104.sh` 可连续跑两版；某一版失败仍保存证据并继续另一版。批次包额外保存两份子压缩包、状态与上传包源码指纹，方便整体下载。原 v9c 结果保留，性能结论应结合相同 DCU 上的对照/重复测量。
