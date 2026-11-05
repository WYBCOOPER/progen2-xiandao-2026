# fp32_decode_step_v2

扩大单步编译边界：模型计算、候选筛选、argmax、ID映射共同编译；FP32/default/fullgraph=True。

基于已实测通过的mask_v1，两个候选相互独立，并非v3叠加v2。只修改solution.py，四份模型文件与mask_v1逐字节一致。没有降低精度或启用TF32，也不改变官方测试工具。
本地检查范围：Local stateful-model checks and eager-backend fullgraph capture passed; real ProGen/Inductor/DCU pending.

运行：
```bash
bash candidates/fp32_decode_step_v2/run_test.sh
```
自动核验五份源码指纹、新建独立结果目录、保存源码、调用原官方脚本并保留日志。四项必须100%一致，再比较同机耗时。首次编译可能耗时；若不支持就保留错误日志，不静默回退或修改为其他配置。
