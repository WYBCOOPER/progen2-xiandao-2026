# v19: split streaming attention

Parent v17, not v18. Use two interleaved sequence partitions for B>=4 and
four for B<4, each with 64-position tiles and four warps. Each partition
computes partial online softmax; a second kernel merges the partials.
This trades an extra merge launch and repeated RoPE work for more parallel
attention work. No speed benefit is assumed before DCU measurement.

Only partition 0 writes current K/V and Q output. All partitions read
strictly older cached positions and use current K/V directly in registers.
Empty partitions write (-inf, 0, zero vector); at least partition 0 is valid.
The merge is launched on the same stream after every partial has completed.
Reduction order changes, so FP32 alone does not guarantee identical tokens.

Model weights, greedy rule, precision, graph runtime and official tools
remain unchanged. Initialization/capture uses input-independent dummy data;
every request resets KV. Formal-call compile, cache-miss and capture counters
must be zero, and all 4608 public output tokens must match.

From the contest root directory:
```bash
bash candidates/fp32_split_stream_v19/run_test.sh
```
The runner checks checksums, direct kernels and compiled graph replay before
calling the official public test. It creates a new result directory and
prints RESULT_ARCHIVE on success or failure. It does not overwrite v17.

Local checks emulate the actual new kernel body on CPU, including empty
partitions and poisoned unused KV slots. They do not establish actual
Triton/HIP compilation, full-model correctness or performance.
