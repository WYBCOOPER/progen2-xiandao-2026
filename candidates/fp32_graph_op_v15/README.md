# v15: explicit mutation boundary inside compiled HIP Graph

Parent: v14. Only change: wrap unchanged RoPE/KV and attention kernels in
one torch.library.custom_op, declaring only K/V buffers as mutated. FP32
kernel arithmetic, graph capture/replay, GEMM tuning, model and official
tools are unchanged. Input QKV, position and sin/cos tables are read-only.
This tests whether the compiler's conservative inline-asm mutation analysis
causes unnecessary copies or scheduling. That cause is a hypothesis, not a
measured conclusion. v10's older custom-op experiment was slower without
manual graph replay; its speed result does not establish v15 performance.

Compilation and op dispatch take place during graph preparation in official
unmeasured warmup. Actual replay directly runs captured device operations.
Each request resets and fills its own KV. No input-dependent KV reuse.
Formal-call compile/miss/capture counters must remain zero; full public
generation must match 4608/4608 tokens. No silent fallback.

Run from progen2_contest:
bash candidates/fp32_graph_op_v15/run_test.sh

The existing preflight now exercises the explicit custom op under compile
and graph capture, checking readonly inputs and exact RoPE/KV writes.
RESULT_ARCHIVE is produced on success or failure. DCU compatibility and
speed are unverified until this experiment is run on the actual platform.
