# PR24 T2：关闭编译的诊断对照

在DCU比赛根目录progen2_contest上传并解压，运行：
```bash
unzip -n pr24-1bda90f-t2-upload.zip
bash candidates/pr24_1bda90f_t2/run_test.sh
```

相对T1只将solution.py的COMPILE_MODEL从True改为False；FP16、scope=all、ALLOW_FP16_REDUCED_PRECISION=False保持不变。其余四份Python文件与T1逐字节一致，空models/__init__.py加载标记保留。上游固定提交1bda90f824b34bfe4a522bef14f6d446661839ad，manifest明确记录这个单行修改，不冒充上游原字节。

启动应看到torch.compile已关闭（COMPILE_MODEL=False）。脚本自动校验指纹、建立独立results/pr24-1bda90f-t2-日期-随机码目录，保存源码、依赖、官方四项日志/JSON及诊断摘要，结束或模型失败时自动打包，相邻.tar.gz路径由RESULT_ARCHIVE给出。官方基准、测试工具及既有submission保持原样。

T2_CHECK=PASS要求四项100%一致，且日志中同时有低精度reduction关闭、compile关闭的确认。FAIL时保留负结果，不预填修复或加速。本轮用于诊断编译是否影响一致性，不以关闭编译后的速度选最终性能方案；不同结果仍需核对源码、日志与输出再判断原因。

尚未在真实DCU运行。不安装依赖、不带模型权重、不覆盖T1结果，也不自动进入T3/T4。
