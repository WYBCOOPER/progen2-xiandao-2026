# 10 月 29 日三人执行计划：具体到文件与操作

日期：2026-10-29，全部为北京时间。每日 22:00 前交付，23:00 前完成当日审查。今晚目标是确认稳定 FP32 状态，交付代码流程说明、实验复核和初步调用诊断；不预设位置表或 KV 一定值得优化。

本包是待执行的清单和只读材料。本地核对不等于已在 DCU 运行；所有结果由实际执行后填写。

## 0. 先把工作分开

| 人 | 工作地点 | 今天操作哪些文件 | 最后交什么 |
|---|---|---|---|
| 你 | 唯一的 DCU 实例 | 检查 baseline/solution.py 与 submission/solution.py；运行环境检查、官方测试、独立调用诊断 | 配套代码、原始 JSON、完整日志、调用表、运行报告 |
| 队友 1 | 自己电脑，无需容器、模型权重或 DCU | 阅读本包 reference/baseline/solution.py、reference/public_test.py、reference/progen2/models/progen/modeling_progen.py | 填完 deliver/TEAMMATE1_FLOW.md 的 8 行定位表和解释 |
| 队友 2 | 自己电脑，无需容器、模型权重或 DCU | 阅读 reference/docs/ 规则和 reference/records/ 昨天的 JSON/日志；收到今天记录后补表 | 填完 deliver/TEAMMATE2_REVIEW.md 的对照表与核查结论 |

参考资料保留了官方代码和许可证，只用于阅读。不要把本包 reference/ 覆盖进容器，也不要将整个本包当作最终参赛 ZIP。另两人的任务可以在你恢复容器时立刻开始。

## 1. 你：恢复实例以后，按顺序执行

以下命令都在平台的 Linux 终端执行，不在本地 Windows 终端执行。每一小步完成再做下一步；同一个终端窗口保持打开。

### U1. 找到昨天的项目

按平台实际提供的“启动/恢复”按钮恢复实例并打开终端；若只能新建，使用昨天已验证的环境。先运行：

```bash
pwd
ls
cd ~/progen2_contest
ls baseline/solution.py submission/solution.py public_test.py check_environment.py
```

最后应列出这四个文件。找不到项目目录就停下，先核对平台文件管理中的保留目录和昨天的本地备份；不要直接重新覆盖解压。是否成功恢复以实际文件和检查结果为准。

### U2. 建一个新的结果目录，保留当前代码

```bash
mkdir -p results
RUN_DIR=$(mktemp -d results/day2-20261029-XXXXXX)
printf '%s\n' "$RUN_DIR"
cp submission/solution.py "$RUN_DIR/solution.py"
sha256sum baseline/solution.py submission/solution.py | tee "$RUN_DIR/code_sha256.txt"
```

RUN_DIR 是本次结果目录；mktemp 每次生成新编号，避免覆盖昨天或本日上一轮结果。记下屏幕上的实际目录。两行指纹应相同，因为昨天结束时已恢复 FP32。若不同，停下保留当前文件并核查差异；不直接覆盖未知改动。

### U3. 确认环境仍能用

```bash
set -o pipefail
python check_environment.py --full-hash 2>&1 | tee "$RUN_DIR/environment.log"
```

检查设备是否可见、是否有 HIP、版本与昨天是否一致、是否出现 ENVIRONMENT_CHECK_PASSED。任意报错或版本明显变化先记录并处理，不继续把它当同条件对照。

### U4. 运行恢复后的 submission 程序，保存四项真实结果

```bash
RESULT_FILE="$RUN_DIR/fp32_public.json" bash run_public_test.sh submission 2>&1 | tee "$RUN_DIR/fp32.log"
```

RESULT_FILE 是官方脚本已有的输出设置，不修改测试器。输出应有四项 load 的耗时和一致率。程序正常完成后确认：

```bash
ls -lh "$RUN_DIR"
```

必须有 fp32_public.json 和完整 fp32.log；逐项查看是否 100% 一致。仅看到 PUBLIC_INTERFACE_TEST_PASSED 不够。如果报错、JSON 缺失或一致率异常，保留日志，先查原因，不进入下一步。今天这份结果对应恢复后的 submission，不用昨天的 FP16 JSON 替代。

### U5. 另外跑一次 Python 调用诊断，保留独立输出

只有 U4 正常后执行：

```bash
python -m cProfile -o "$RUN_DIR/python_calls.prof" public_test.py --submission submission --output "$RUN_DIR/diagnostic_only.json" 2>&1 | tee "$RUN_DIR/diagnostic.log"
```

该命令不改源码，使用 Python 标准库，诊断范围包含加载、预热、四项生成及校验，并非只测 generate。它会增加开销，diagnostic_only.json 的耗时不得用于宣称加速或替代 U4 的正常结果。本日无新增扩展，U4 已完成官方入口检查，才直接调用测试器进行诊断。

先检查 diagnostic.log 是否四项完成，以及 diagnostic_only.json 是否确实生成；诊断遇到异常时，不能只依据命令退出码或 prof 文件存在判定成功。没有完整结果就记录“诊断失败/部分完成”及错误，停止继续解释耗时。正常完成后生成文本表，在容器原环境执行：

```bash
python -c "import pstats; pstats.Stats('$RUN_DIR/python_calls.prof').strip_dirs().sort_stats('cumtime').print_stats(30)" 2>&1 | tee "$RUN_DIR/python_top30.txt"
python -c "import pstats; pstats.Stats('$RUN_DIR/python_calls.prof').strip_dirs().sort_stats('tottime').print_stats(30)" 2>&1 | tee "$RUN_DIR/python_self30.txt"
python -c "import pstats; pstats.Stats('$RUN_DIR/python_calls.prof').strip_dirs().sort_stats('cumtime').print_stats('fixed_pos_embedding|synchronize|generate|cat')" 2>&1 | tee "$RUN_DIR/python_focus.txt"
```

把文本发给队友即可，不要求队友在不同 Python/操作系统上读取二进制 prof 文件。原始 prof 自己保留。

这一步能观察调用次数、Python/C 调用时间和等待线索。cumtime 包含子调用，不能把表中多行相加当占比；DCU 异步执行也会影响归因。不能从 cProfile 直接断言“KV 占 DCU 时间多少”或算出 Prefill/Decode 精确占比。若看到大量同步等待、加载占主导或信息不足，结论写“需进一步设备侧诊断”，下一轮再核对当前 HIP 环境支持的工具。今天只要求交付可靠的初步线索。

### U6. 打包本轮材料并下载

```bash
tar -czf "${RUN_DIR}.tar.gz" "$RUN_DIR"
ls -lh "${RUN_DIR}.tar.gz"
```

在平台文件管理找到刚显示的 results/ 下压缩包，下载到电脑；原始件保留。填写本包 deliver/YOUR_RUN_REPORT.md，记录实际目录、命令、时间、结果和卡点。

通过队内既有渠道把代码快照、JSON 和文本日志提供给两位队友。公开上传时使用共享副本，按昨天约定处理账号目录等无关个人信息并说明，保留原始件，成绩不手改。今天先提交记录，暂不改精度或实现缓存优化。

## 2. 队友 1：“阅读生成流程”具体就是完成这 8 项

用任意文本编辑器打开下列文件，搜索表中关键词，把找到的行号、代码及自己的解释填进 [流程交付模板](deliver/TEAMMATE1_FLOW.md)。无需运行模型。

| 步骤 | 打开本包中的文件 | 搜索什么 | 必须写出的解释 |
|---|---|---|---|
| B1 | reference/public_test.py | load_solution、solution_class | 测试器怎样找到 solution.py 和 Solution |
| B2 | reference/baseline/solution.py | __init__、from_pretrained、torch.float32 | 模型何时加载，精度在哪里指定；从测试器判断加载是否计时 |
| B3 | 同上 | tokenizer.encode、input_ids | 字符前缀怎样变成模型输入；batch 是什么 |
| B4 | 同上 | past_key_values = None、next_input = input_ids | 第一轮为什么把整个前缀交给模型 |
| B5 | 同上 | logits、allowed_ids、argmax | 怎样只从 20 个氨基酸选出下一个字母 |
| B6 | 同上 | past_key_values = outputs、next_input = next_token | 后续轮次传入什么，为什么保留历史缓存 |
| B7 | reference/progen2/models/progen/modeling_progen.py | fixed_pos_embedding、past_key、torch.cat | 分别找到位置表调用及 K/V 历史拼接；说明两者用途，不断言谁更慢 |
| B8 | reference/public_test.py | reset、started =、elapsed =、validate_outputs | 测试器何时预热、清状态、计时和检查输出 |

每行填写 1 段短代码（建议 2—6 行）、行号和 1—3 句自己的解释。最后画一条调用链：测试脚本 → 测试器 → Solution → 模型 → 返回字符串 → 结果记录。

收到你的 python_focus.txt 后，选其中 2 个函数对应回上述文件；找不到就记录“未定位”。代码阅读得到的是可能的重复工作；是否值得优化需要测量。今天不要求改代码，不把 AI 生成的解释未经核对直接当作本人已理解。

交付：填写后的 TEAMMATE1_FLOW.md；标注实际执行者、AI 协助范围及不懂的 1—3 个点。对应 Issue #2；诊断线索供 #6 使用。

## 3. 队友 2：“核对结果”具体就是完成这 6 项

打开 [结果复核模板](deliver/TEAMMATE2_REVIEW.md)，按顺序做：

1. 打开 reference/docs/BASELINE.md 第 5、6 节，填写 time_s、same、total、consistency_rate、exact_match 各是什么意思，哪些阶段计时、哪些不计时。
2. 打开 reference/docs/SUBMISSION.md 的报告评分部分及 reference/docs/赛题说明-文字摘录.txt，搜索“基础运行”“结果一致性”“性能准入”，写明每负载性能准入为至少 80%；总体一致率须大于 80% 才开始有一致性分，两者不同。原始 Word 文件也在同目录中，文字摘录不代替含公式和表格的原件。
3. 打开 reference/records/20261028-01/baseline_public.shared.json，把四个 loads 中的 load_id、time_s、same、total、consistency_rate 抄到模板；用 same/total 验证一致率。
4. 打开 reference/records/20261028-02/submission_public.shared.json，按相同 load_id 配对；计算 FP32 time_s / FP16 time_s，查看两份完整共享日志中的对应行。写清楚哪一方案本轮更快、哪两项不满足性能准入，不能用平均值掩盖单项门槛。
5. 收到今天 U4 的 fp32_public.json、fp32.log、solution.py、code_sha256.txt 后，新增“10-29 恢复后 FP32”四行，核对 JSON 与日志一致、代码指纹对应、四负载齐全。只审查材料就写材料复核，不写自己已在 DCU 复现。
6. 收到 U5 的文本后写 3 条诊断观察：出现什么函数、调用次数/时间是什么、它不能证明什么。没有收到则写缺失，不编结果。

文件名不决定精度：昨天名为 submission_public.shared.json 的记录是 FP16，而今天 fp32_public.json 必须由 U4 实际产生；以配套代码和日志判定。

交付：填写后的 TEAMMATE2_REVIEW.md，附具体核查意见、异常和未收到的材料。对应 Issue #3；对 #1/#4/#8 的已有材料做复核，今天新结果用于 #6。

## 4. 单实例交接与当天时间

| 时间节点 | 你 | 队友 1 | 队友 2 |
|---|---|---|---|
| 开始后立即 | 恢复实例，执行 U1—U3 | 下载本包，执行 B1—B8 | 读规则、对照昨天 JSON 与日志 |
| 20:30 前 | 确认能够测试；有阻塞尽早留日志 | 提交流程说明初稿，标出不懂位置 | 提交昨日对照及核查初稿 |
| 20:30—22:00 | 顺序执行 U4、U5，打包记录；只有你在实例中运行任务 | 将诊断函数对应到代码 | 接收 JSON/日志并填今天的表 |
| 22:00 前 | 提交运行报告和证据，未完成则交卡点 | 提交实际完成的说明 | 提交已收到材料的核查结论 |
| 22:00—23:00 | 解释实际操作、处理反馈 | 核对函数定位与诊断局限 | 核对版本、数值、日志及缺项 |

这些是截止节点，能提前就提前。若开始时已晚于某节点，按实际时间记录，不倒填。23:00 前收尾时明确通过、修改或待复核；等待队友回复不当作已通过。若无法完成 U4，不强行进行 U5；若 U5 失败，保存错误并交付已完成部分。

## 5. 放到 GitHub 哪儿

本包本次只在本地生成，尚未上传或合并。确认交付后用各自账号在现有任务下提交：

- 队友 1：docs/learning/20261029-generation-flow.md（来自流程模板），关联 #2。
- 队友 2：docs/reviews/20261029-results-review.md（来自复核模板），关联 #3、#6；对昨天材料的意见关联 #1/#4/#8。
- 你：experiments/EXP-20261029-01.md（来自运行报告），以及 records/20261029-01/ 下的代码、JSON 和完整日志。保留实际结果目录对应关系，不把诊断 JSON 当正常成绩。
- 技术状态采用“待运行 → 已运行待分析 → 待复核 → 已验收”。文档完成与有效加速分别记录。当天没有做位置表/KV 修改，就不把 #7 标成已实现。

谁先上传不影响另外两人阅读和写作。其他人没有协作者权限时，可先交付 Markdown 文件，由有权限的人代录并注明实际作者。

## 6. 今天怎样算完成

最低交付：环境与代码状态记录、恢复后 FP32 正常结果或完整失败日志、两位队友的实际文档；诊断失败有完整错误和下一步也如实交付。理想交付是在此基础上拿到 Python 调用线索，选出下一轮设备侧测量问题。

今天不承诺找出全部 DCU 瓶颈或实现加速。没有设备侧证据时，不把 Python 调用时间写成 DCU 算子百分比。下一步根据线索决定测量/优化方向，位置表和 KV 仍为候选。

依据：本包 reference/ 中的官方接口、测试器与脚本；Python 调用诊断方法见 https://docs.python.org/3.10/library/profile.html 。脚本接口和 RESULT_FILE 已对照题包源码核实；本文命令尚未在你的恢复实例中执行。
