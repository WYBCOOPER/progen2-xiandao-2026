# v14: compiled decode plus HIP Graph replay

Parent: v13, whose public correctness and graph audits passed but whose total
time was 9.8908s versus historical v9c 7.0429s. No speed claim for v14.

Only performance change: compile the entire fixed-shape decode step with
max-autotune-no-cudagraphs/fullgraph=True/dynamic=False before manually
capturing it. This restores compiler fusion and GEMM tuning absent in v13's
eager captured path. Attention, RoPE, FP32, KV reset, generation rule and
official tester are unchanged from v13. Fixed batch shapes avoid symbolic
batch constants in the older vendor Triton wrapper.

Compilation/capture is lazy during official unmeasured warmup; no input-
dependent KV is reused across requests. Audit rejects measured-call Dynamo
compilation, Triton misses or graph capture. No fallback hides failures.
The preflight now compiles and captures the custom kernels together.

Upload/extract into progen2_contest; run:
bash candidates/fp32_compiled_graph_v14/run_test.sh

Download RESULT_ARCHIVE on success or failure. Full public exactness and
actual speed require DCU testing; local checks cannot establish either.
