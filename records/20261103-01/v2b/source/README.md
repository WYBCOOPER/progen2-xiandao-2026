# V2b：联合单步编译的归约绕行候选

保持FP32、统一模型和贪心规则，改用顺序比较选择最高分，尝试绕开V2在DTK/Triton AMD编译中的卡点。未修改第三方依赖，未安装/下载新软件。是否解决编译失败及是否提速，均待DCU。

解压到平台 progen2_contest 目录后运行：
```bash
bash candidates/fp32_decode_step_v2b/run_test.sh
```
启动脚本核对源码，创建独立结果目录，收集实际已安装版本，再调用原官方测试。首次调用可能触发编译，必须由规定预热覆盖；脚本会检查正式调用是否仍触发编译。不预存测试答案或复用上一请求的输入相关KV。

除candidate.log、candidate_public.json之外，新增runtime_versions.json和compile_audit.json。四项必须100%一致，且COMPILE_AUDIT必须PASS，才进一步比较耗时。审计按当前官方四项各预热/正式一次的8次generate检查，每个正式调用的Dynamo编译事件必须为0；FAIL时返回非零，保留原始成绩与日志，不作为有效性能证据。

编译监测针对TorchDynamo事件；本程序未使用其他自定义JIT。此项检查不等于最终赛事认证。平台镜像名称仍需从平台实际记录，不能由脚本猜填。最终报告须保留实际软件版本、运行命令、所有自编修改源码和必要构建说明；未修改第三方框架无需整套打包。

本地验证：CPU与NVIDIA各1006行候选分数（包括并列、NaN、无穷和有符号零）与argmax相同，24组状态模型生成对照及8组预热/正式编译事件监测对照通过。使用PyTorch2.11/eager后端，不代表真实ProGen/Inductor/DCU通过。
