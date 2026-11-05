# FP32 单 token 掩码快路径 v1

基于已通过的 abbfeb5，只在单 token 查询时跳过全真因果掩码；保持 FP32、多 token 前缀掩码、额外掩码和贪心规则。
本地52组注意力函数对照通过（CPU及NVIDIA CUDA）；未做DCU或完整模型验证，尚无加速结论。

在平台 progen2_contest 目录上传解压，然后运行：
```bash
bash candidates/fp32_decode_mask_v1/run_test.sh
```
脚本先核对五份源码指纹，再新建 results/fp32-decode-mask-v1-日期-随机串，保存源码，并调用原 run_public_test.sh。成绩为 candidate_public.json，日志为 candidate.log。
官方基准和测试器不改。四项一致率须全部100%，再与同机前版比较耗时。编译器可能已消除冗余，因此收益也可能为零。
