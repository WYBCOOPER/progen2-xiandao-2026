# v16: compact fixed KV storage for compiled HIP Graph

Parent v15. Only algorithm-facing change: choose graph KV/output capacity
from request dimensions, rounded up to 128; graph cache key is (batch,capacity).
No token values or previous outputs influence capacity. Each request resets
and fills its own KV as before. Same FP32, kernel arithmetic, attention
partitioning, custom-op mutation schema, graph replay, and GEMM tuning.
The model maximum remains 1024. A request that crosses a capacity bucket gets
a separate fixed-address graph, prepared only during unmeasured warmup.

Official workloads use capacities 384,384,640,640 instead of 1024. For B=8,
the persistent graph KV buffers shrink from 2.53125 GiB to 0.94921875 GiB
(62.5% less). This is KV storage, not peak process memory or a speed claim.
Any reduction in per-step copy overhead is unconfirmed until measured.
Attention still launches the original 16 partitions; masks must prevent
access to positions beyond the actual allocation. Preflight tests both ends
of five (batch,capacity) combinations under compiled graph replay.

Run from progen2_contest:
bash candidates/fp32_compact_graph_v16/run_test.sh

Formal compile/miss/capture counters must be zero; 4608/4608 public tokens
must match. Audit also checks recorded capacities. Official tester/model
weights are unchanged. Success or failure produces RESULT_ARCHIVE.
