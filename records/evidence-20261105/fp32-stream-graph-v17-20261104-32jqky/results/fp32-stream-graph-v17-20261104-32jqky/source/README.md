# v17: streaming fused RoPE/KV/attention inside the v15 graph

Parent v15, not v16. Keep capacity 1024, FP32, compiled graph replay,
GEMM autotuning, greedy rule and official tools. B<4 retains the v15 path.
For B>=4 one program per head computes RoPE, writes only this step's K/V,
and evaluates attention using online softmax with 128-position tiles. This
replaces three launches (RoPE, partials, merge) with one and removes partial
result buffers. Current K/V remain in registers; old cache loads mask s<pos,
so no cross-program synchronization or read-after-write assumption is used.
Existing AMD multiply instructions preserve the RoPE product rounding.

Online softmax changes reduction order. It is FP32, but exact full-model
greedy output must still be validated: 4608/4608 matches are required.
No speed claim: streaming reduces launches but also head-level parallelism.
The B>=4 dispatch is a generic batch-size rule, independent of token values.

Run from progen2_contest:
bash candidates/fp32_stream_graph_v17/run_test.sh

Preflight checks compiled graph execution at B=1,4,8 and positions
0,127,511,1022 (two replays each, including final valid position 1023).
Each real request resets KV; official warmup may compile/capture, while
measured-call compile/miss/capture counters must all be zero.
Success or failure produces RESULT_ARCHIVE. DCU testing remains required.
