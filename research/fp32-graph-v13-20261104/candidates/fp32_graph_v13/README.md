# v13：整段 decode 的 HIP Graph 重放

基于 v9c 的实验分支，不叠加 v10/v11/v12。prefill 保留 v9c 编译路径；decode 用固定地址缓冲区、设备端位置和 HIP Graph 重放。单步覆盖完整 27 层、字母选择和结果写入。仍逐个字母自回归，不提前使用真实输入结果。

RoPE 数值计算取自 v9c；注意力使用先前 v8c 的 FP32 分区/合并算法，改为从设备位置读取有效长度。这属于配套改动，并非只换编译开关；归约和图路径可能影响精度，必须由整模型逐 token 一致测试判定。

每个首次出现的 batch 在官方不计时预热中用独立 dummy 数据建立图。每个真实请求重新清零固定 KV 并复制本次 prefill 的有效 KV，不复用旧请求内容。正式调用禁止新建图、Dynamo 编译或 Triton cache miss；脚本检查这三个计数。预热图不是对真实测试答案的缓存。

图中使用 eager PyTorch 线性层/MLP，不保证延续 max-autotune 对 decode GEMM 的全部收益；目标是减少 CPU/驱动调度，收益待测。当前 DTK 是否支持完整模型捕获也需 DCU 验证；失败不隐藏、不降级。

运行 `bash candidates/fp32_graph_v13/run_test.sh`。自动校验源码、检查核与图重放，再运行官方四项。正确性必须 4608/4608 一致；加载/预检/预热不计入官方四项耗时。成功或失败都输出 RESULT_ARCHIVE，下载该包。

未改官方工具、权重或生成规则。PyTorch 2.4 图捕获接口依据：
https://github.com/pytorch/pytorch/blob/v2.4.1/docs/source/notes/cuda.rst#cuda-graphs
HIP 共用 torch.cuda 接口，需以 vendor DTK 运行结果确认：
https://github.com/pytorch/pytorch/blob/v2.4.1/docs/source/notes/hip.rst
