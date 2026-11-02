# fp32_gemm_tune_v3

仅将模型compile模式改为max-autotune-no-cudagraphs；FP32保持，未叠加v2。

基于已实测通过的mask_v1，两个候选相互独立，并非v3叠加v2。只修改solution.py，四份模型文件与mask_v1逐字节一致。没有降低精度或启用TF32，也不改变官方测试工具。
本地检查范围：Syntax and source-diff checks only; actual DTK backend availability, numerical behavior and performance pending DCU test.

运行：
```bash
bash candidates/fp32_gemm_tune_v3/run_test.sh
```
自动核验五份源码指纹、新建独立结果目录、保存源码、调用原官方脚本并保留日志。四项必须100%一致，再比较同机耗时。首次编译可能耗时；若不支持就保留错误日志，不静默回退或修改为其他配置。
