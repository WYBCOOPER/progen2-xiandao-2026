# 10 月 29 日三人逐步执行单

日期均为 2026 年北京时间；今天 22:00 前交付、23:00 前结束审查。每一步失败就保存错误、交给指定协作者处理，不跳过失败假装下一步已通过。今天先完成共同的第一项闭环，优化收益以实测为准。

## 先回答 Issue #5 是什么

它的意思是：让 submission/solution.py 中的 FP32 生成程序能被官方测试器找到，完成加载、生成和四项检查。你昨天已恢复一份 FP32 代码，今天要在恢复实例、确认环境后实际测试它；队友同时准备本地原版和验证工具。复制文件本身不是加速成果，检查通过也不等于已完成优化。

你负责下面 U 步骤，队友 1 负责 B，队友 2 负责 C。A/B/C 在仓库仍是建议角色，不把用户自动绑定为 A，也不替队友确认账号或贡献。

## 两个目录不要混淆

| 目录 | 在哪里 | 包含什么 |
|---|---|---|
| 官方题包根目录 progen2_contest/ | 你的 DCU 和每个人的本地电脑 | baseline/、submission/、progen2/、public_test.py、公共权重 |
| GitHub 协作仓库 | 三个人都可读 | 计划、实验记录及本次新增的 dev/local_test.py；不包含约 3GB 权重 |

两位队友先解压同一份原始题包，再把本 PR 的 dev/local_test.py 单独复制到题包根目录/dev/local_test.py。不要把协作仓库误当成完整模型目录，不要用代码参考副本覆盖你的整个容器目录。官方 public_test.py、baseline/、模型权重和参考结果保留原样。

新增脚本由 Codex 协助起草，队友 2 负责理解、运行、核查和维护；本次已做本地语法、帮助参数和记录比较函数检查，尚未在 NVIDIA 或 DCU 执行真实推理。

## 同时开始，不要互相等待

| 你正在做 | 队友 1 同时做 | 队友 2 同时做 |
|---|---|---|
| U1—U3 恢复、备份与环境检查 | B1—B2 解压、读代码、说明候选位置 | C1—C3 读测试器、检查本地环境、试跑开发脚本 |
| U4—U5 正常测试与初步诊断 | B3—B4 本地基准与候选目录准备 | C4—C5 原版自比较、状态检查与数据复核 |
| 分析 DCU 测量并准备候选运行 | B5 获得共识后只改一处 | C6 审查改动；用自己的原版比较候选 |
| U6 在 DCU 单独验收候选 | 根据错误修复代码、解释实现 | 定位差异、核对 JSON/日志、总结限制 |

## 你：恢复实例后的 U1—U7（Linux 平台终端）

### U1 找回项目

按平台现有按钮恢复/启动实例，打开 SSH 或终端。执行：

```bash
pwd
ls
cd ~/progen2_contest
ls baseline/solution.py submission/solution.py public_test.py check_environment.py
```

应看到四个文件。找不到时停在这里，把目录列表交给指导对话定位或核对备份；不直接覆盖解压。

### U2 创建本次记录目录，保存代码

```bash
mkdir -p results
RUN_DIR=$(mktemp -d results/day2-20261029-XXXXXX)
printf '%s\n' "$RUN_DIR"
cp submission/solution.py "$RUN_DIR/solution.py"
sha256sum baseline/solution.py submission/solution.py | tee "$RUN_DIR/code_sha256.txt"
set -o pipefail
```

记下 RUN_DIR 的实际值；后续在同一终端执行。两行指纹应与昨天恢复原版的状态对应；若不同，先保留当前代码并查差异，不直接覆盖未知修改。

### U3 检查环境

```bash
python check_environment.py --full-hash 2>&1 | tee "$RUN_DIR/environment.log"
```

需要设备可见、HIP 环境正常并出现 ENVIRONMENT_CHECK_PASSED。把设备和软件版本与昨天记录对照；有错误先交给指导对话处理。环境通过只表示可以继续验证，不表示模型已加载成功。

### U4 测试恢复后的 FP32 程序——这是 Issue #5 中你今天负责的部分

```bash
RESULT_FILE="$RUN_DIR/restored_fp32.json" bash run_public_test.sh submission 2>&1 | tee "$RUN_DIR/restored_fp32.log"
```

应完成四项，检查各项 same/total 与一致率。目标为四项 100% 一致，不能只看最后的接口通过提示。缺 JSON、报错或异常一致率时保留完整日志，不用昨天 JSON 替代。把这两份新文件交给队友 2。

### U5 单独做初步调用诊断

U4 正常以后执行：

```bash
python -m cProfile -o "$RUN_DIR/python_calls.prof" public_test.py --submission submission --output "$RUN_DIR/diagnostic_only.json" 2>&1 | tee "$RUN_DIR/diagnostic.log"
python -c "import pstats; pstats.Stats('$RUN_DIR/python_calls.prof').strip_dirs().sort_stats('cumtime').print_stats(30)" 2>&1 | tee "$RUN_DIR/python_top30.txt"
```

先确认诊断日志完成四项和 diagnostic_only.json 确实生成；cProfile 文件存在或退出码正常不单独证明测试成功。诊断包含加载、预热、Python/C 调用及等待，耗时不填作正式成绩，cumtime 各行不能相加成百分比。它不能直接给出 DCU 各算子的精确占比；信息不足就记录待进一步设备侧测量。

把 python_top30.txt 给队友 1、2：队友 1 对应回函数位置，队友 2 记录观察和局限。你继续分析目标环境和准备候选队列，不需要等代码完成。

### U6 收到候选以后，在 DCU 单独测试

接收队友 1 的完整候选目录（例如 dev/candidate_rope_20261029_v1/，含 solution.py 和 models/），保留它的版本与全部修改源码。放到题包根目录/dev/，不要覆盖 baseline/ 或现有稳定 submission/。先核对文件与交付版本，再执行：

```bash
RESULT_FILE="$RUN_DIR/dcu_base.json" bash run_public_test.sh baseline 2>&1 | tee "$RUN_DIR/dcu_base.log"
RESULT_FILE="$RUN_DIR/dcu_candidate.json" bash run_public_test.sh dev/candidate_rope_20261029_v1 2>&1 | tee "$RUN_DIR/dcu_candidate.log"
```

候选实际目录名不同就用真实目录名；没有候选则不执行。每条运行完成并检查结果后再跑下一条，正式测速时同一设备不跑其他任务。先对四项正确性，再计算相同负载的基准秒数/候选秒数。有收益且差距小或不稳定才补多轮对照，不按不同轮次拼最好结果。

### U7 保存与交付

```bash
tar -czf "${RUN_DIR}.tar.gz" "$RUN_DIR"
ls -lh "${RUN_DIR}.tar.gz"
```

下载压缩包，另行保留产生候选成绩的完整候选源码或 Git 提交号。写 experiments/EXP-20261029-01.md：实际执行时间、版本、命令、环境、四项结果、诊断线索、错误和下一步。证据放 records/20261029-01/；公开共享件说明账号目录等无关信息的处理，完整原件自己保留。不上传公共权重、整题包或固定测试数据。

## 队友 1：B1—B6（本地电脑）

### B1 准备文件和环境

解压同一份官方题包。确认存在 baseline/solution.py、progen2/checkpoints/progen2-medium/pytorch_model.bin、tokenizer.json（位于 progen2/）以及 public_test.py。获取本 PR 的 dev/local_test.py，放入题包/dev/。在自己的 CUDA PyTorch 环境打开终端，进入题包根目录；不要安装 DCU/HIP 版环境。

先执行下面命令记录实际环境；没有 torch、CUDA 不可用或架构不支持时，保存完整错误交给指导对话处理，暂不宣称本地可运行：

```powershell
python -c "import sys; print(sys.version); import torch; print(torch.__version__, torch.version.cuda, torch.version.hip); print(torch.cuda.is_available())"
```

### B2 只读三处代码，填写具体回答

1. 打开 baseline/solution.py，搜索 from_pretrained 与 torch.float32：写下模型在哪加载、精度在哪设置。
2. 在 generate() 中搜索 next_input、allowed_ids、argmax、past_key_values：用代码行号说明首次完整前缀、后续新 token、20 选 1、缓存传入与更新。
3. 打开 progen2/models/progen/modeling_progen.py，搜索 fixed_pos_embedding 与 ProGenAttention.forward：标出位置表的输入维度、输出、调用点、位置切片和 device/dtype。

交付 docs/learning/20261029-generation-flow.md：准确文件位置、每处 2—6 行关键代码、自己的解释、不确定点和 AI 协助。能指着代码说明以上关系即可进入小改动，不要求推导全部数学。

### B3 跑自己电脑上的原版

使用 C3 的 baseline_short 命令，在自己电脑产生自己的原版 JSON。队友 2 的基准结果只能参考，不拿其他电脑的文件代替本机基准。原版跑不通就先处理环境，候选结果不能解释为优化成败。

### B4 创建独立候选目录（PowerShell，题包根目录）

仅在这个候选目录尚不存在时执行；已存在则保留并改用新版本名，不覆盖旧实验：

```powershell
New-Item -ItemType Directory dev\candidate_rope_20261029_v1
Copy-Item baseline\solution.py dev\candidate_rope_20261029_v1\solution.py
Copy-Item -Recurse progen2\models dev\candidate_rope_20261029_v1\models
Copy-Item LICENSE.txt dev\candidate_rope_20261029_v1\LICENSE.txt
```

后续只编辑 dev/candidate_rope_20261029_v1/models/progen/modeling_progen.py，入口保留在该候选目录/solution.py。只保留必要源码，不拷贝 3GB 权重。当前复制阶段没有优化成果。

### B5 共同确定方案后才改一处

结合你返回的诊断和队友 2 的检查意见，先决定位置表复用是否值得试。确定后在 AI 协助下设计缓存的维度、位置范围、device/dtype、索引和失效条件；保持 FP32 与生成规则，不同时改 KV 和精度。若线索不支持位置表，先改实验目标和目录说明，不为赶日程强行实现。

交付候选代码差异、为什么应等价、预期节省哪部分工作以及验证条件；没有运行不填收益。运行 C4 的候选命令做自己的本地对照，将完整源码和结果交给你与队友 2。

### B6 你与队友 2 测试时做什么

检查返回的异常与对应代码；提供最小修复和原因，保留原失败版本。写实验说明，而不是空等测试。新修复产生新版本并重新验证。

## 队友 2：C1—C7（本地电脑）

### C1 先弄清官方已有功能，不重写评分器

打开 public_test.py，依次搜索 load_solution、validate_prefixes、validate_outputs、synchronize、main。指出哪些函数检查输入/输出，哪些部分比较参考字母和计时。找到 HIP 检查，说明为什么 N 卡不能直接运行官方 main。

### C2 核查本次提供的适配脚本

把本 PR 的 dev/local_test.py 复制到题包/dev/，打开阅读它如何导入官方函数，而不是修改官方文件。脚本只做本地开发检查、保存 local_development 记录，不输出官方分数。它把本机原版与本机候选比较；保留公开工具原样。

用 B1 的环境命令记录自己电脑环境。缺依赖或不兼容时保存错误，先协助解决，不盲目把精度改成 FP16 来掩盖问题。

### C3 在原版上先运行短测试（PowerShell，题包根目录）

```powershell
New-Item -ItemType Directory -Force local_results\20261029
python dev\local_test.py --help
python dev\local_test.py --submission baseline --output local_results\20261029\baseline_short.json --case-id load1_b1_l32_g256 --new-tokens 8 --state-check 2>&1 | Tee-Object local_results\20261029\baseline_short.log
```

只测试第一负载规格下的短生成，不能称为四项完整通过。应生成 purpose=local_development 的 JSON，并显示 LOCAL_CHECK_PASSED；state_check_exact 应为 true。遇到错误保存完整日志，不沿用旧 JSON。输出文件已存在时脚本拒绝覆盖，请改新文件名。

### C4 用原版自比较，核对适配入口是否合理

```powershell
python dev\local_test.py --submission baseline --output local_results\20261029\baseline_repeat.json --compare local_results\20261029\baseline_short.json --case-id load1_b1_l32_g256 --new-tokens 8 --state-check 2>&1 | Tee-Object local_results\20261029\baseline_repeat.log
```

应与自己的原版完全一致。它还运行同一对象 A→B→A 并按接口 reset，检查状态独立性。自比较不通过先排查适配或环境，不急着判断候选。

### C5 没有候选时，不需要等

打开 docs/BASELINE.md 的字段与计时说明；核对昨天 FP32/FP16 的四项 JSON 和日志。把自己原版的返回条数、长度、合法字符、重复调用结果写进 docs/reviews/20261029-local-review.md。审查 B5 方案是否正确处理位置、形状、精度和请求状态，提出具体问题。

### C6 收到队友 1 的完整候选目录后做对照

先保存收到的版本和源码，放入自己题包/dev/，不要覆盖 baseline/。执行：

```powershell
python dev\local_test.py --submission dev/candidate_rope_20261029_v1 --output local_results\20261029\candidate_short.json --compare local_results\20261029\baseline_short.json --case-id load1_b1_l32_g256 --new-tokens 8 --state-check 2>&1 | Tee-Object local_results\20261029\candidate_short.log
```

本机环境和输入条件必须与自己的 baseline_short 相同。失败时脚本保留已比较的输出（条件不匹配等早期错误可能不生成 JSON），退出码 3 表示比较或状态检查失败；不要将失败写为通过。定位第一项不同、发生在哪种输入，并交回队友 1 修复。

### C7 短测试正常后，按资源情况做完整本地四负载

```powershell
python dev\local_test.py --submission baseline --output local_results\20261029\baseline_full.json --state-check 2>&1 | Tee-Object local_results\20261029\baseline_full.log
python dev\local_test.py --submission dev/candidate_rope_20261029_v1 --output local_results\20261029\candidate_full.json --compare local_results\20261029\baseline_full.json --state-check 2>&1 | Tee-Object local_results\20261029\candidate_full.log
```

每条结束并检查后再执行下一条；显存不足或环境不支持时保留错误，交付实际完成的范围，不降低精度后仍称同条件 FP32。本地四负载通过仍不代表 DCU 通过，也不证明 DCU 有加速。

交付 local_test.py 的审查/修改记录、本机原版与候选的 JSON/日志、故障定位和复核文档；本地原始 JSON 含实际生成字符串，作为开发证据保留，公开共享前按仓库规范核对内容，不把固定输入/参考数据随最终作品打包。

## 谁在什么时候交什么

- 20:30 前：你交环境与原版状态；队友 1 交函数定位/候选方案或代码；队友 2 交适配脚本的本地运行状态与检查意见。有卡点先报，不等到截止。
- 22:00 前：三人提交实际文件、版本、命令、JSON/日志或完整错误。候选来不及实现时交付准备工作，不编收益。
- 22:00—23:00：共同记录通过、修改、待验证或待复核。自查、材料复核、独立本地执行与独立 DCU 执行分别说明，注明 AI 协助。

Issue 关系：#2 流程定位；#3 规则；#5 FP32 程序可运行及验证；#6 本地测试复用/分层对照；#7 单项改动；#9 以后再选单项或组合。#5 不是让你一个人重新开发完整模型，也不表示今天必须交出性能提升。
