# PR24 1bda90f：T1 精度诊断

在DCU比赛根目录progen2_contest上传并解压，然后运行：
```bash
unzip -n pr24-1bda90f-upload.zip
bash candidates/pr24_1bda90f/run_test.sh
```

本轮保持上游配置：FP16、scope=all、ALLOW_FP16_REDUCED_PRECISION=False、COMPILE_MODEL=True。四份Python源码逐字节对应固定提交1bda90f824b34bfe4a522bef14f6d446661839ad；只补空models/__init__.py以隔离同名包，不改计算逻辑。

脚本核对SHA清单，建立独立results/pr24-1bda90f-t1-日期-随机码目录，保存源码、依赖、官方测试日志与JSON，并在结束或失败时自动打包为相邻.tar.gz。它调用既有run_public_test.sh/public_test.py及官方资产，不安装依赖，不覆盖baseline或submission，不重复打包权重。

末尾RESULT_ARCHIVE给出需要下载的结果包。四项输出100%一致且设置开关的成功日志存在时显示T1_CHECK=PASS；不一致或设置失败会保留证据并返回非零状态。打印的属性设置成功不能证明定制后端实际采用预期路径。

这是诊断包，尚未在真实DCU运行；不预填修复或加速结论。本包没有Dynamo计时编译监测，不能单独用于确认正式计时无编译。先测T1，本包不会自动切到T2或其他精度。
