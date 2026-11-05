# V3监测对照

Only add the same per-generate Dynamo compile monitor to V3; preserve the model-only compilation, FP32 and original greedy computation.

模型文件与父版逐字节一致，保持FP32、接口、贪心规则；请求开始KV为空，不预存测试答案或跨请求复用输入相关KV。无依赖安装/升级，官方基准和测试工具不改。

上传包解压到比赛根目录progen2_contest，运行：
```bash
bash candidates/fp32_gemm_tune_v3_audit/run_test.sh
```
脚本核对指纹、创建新的独立结果目录、保存源码快照与运行脚本、记录实际依赖版本，然后调用原官方测试。每次运行自动生成新目录，不覆盖旧成绩。模型使用已有官方资产，不重复打包权重。

四项公开输出必须100%一致，且COMPILE_AUDIT=PASS。按官方四项各预热/正式一次的8次generate检查，正式4次Dynamo新编译事件都应为0；否则保留日志和JSON并返回失败状态，不把本次耗时作为有效优化结果。只允许规定预热覆盖编译，不额外加入测试输入预热。监测范围是TorchDynamo事件，不等于主办方最终认证。

本地各24组CPU/NVIDIA状态模型对照和8组预热/正式编译监测通过，backend=eager/fullgraph；没有本地真实ProGen、Inductor或DCU验证。DCU正确性、计时编译及性能待测，未预填加速结论。实际镜像名称仍需从平台记录。

结果目录中的source为本轮输入快照。复现时将其中文件完整放回比赛根目录candidates/fp32_gemm_tune_v3_audit/，再按上述命令运行；测试器、测试数据和官方模型资产沿用比赛原件。
