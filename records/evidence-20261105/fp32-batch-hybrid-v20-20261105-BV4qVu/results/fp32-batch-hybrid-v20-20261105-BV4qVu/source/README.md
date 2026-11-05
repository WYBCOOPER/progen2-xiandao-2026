# v20: batch-aware hybrid of v17 and v19

Parent v19, incorporating the exact streaming kernel from v17. B>=8 uses
v17's single-head-program attention (128-position tiles, four warps); B<8
uses v19 split attention (64-position tiles, two parts for B>=4, else four).
The policy depends only on batch size and applies to arbitrary valid input
prefixes. It never selects outputs from public reference data.

Both paths have separately passed the full public test on DCU, but their
combination is a new candidate. Historical times suggested B=8 regressed
under splitting. This is one cross-run observation, not proof of stable
speedup or a claimed general performance threshold.

Keep FP32 weights, greedy rule, RoPE rounding, capacity 1024, request-local
KV reset, input-independent graph preparation, and the official tools.
Custom-op declares only K/V mutable. Formal compile/miss/capture counters
must be zero and all 4608 public output tokens must match.

From the contest root directory:
```bash
bash candidates/fp32_batch_hybrid_v20/run_test.sh
```
The runner checks checksums and both paths, checks compiled graph replay,
then invokes the original public tester. Every run creates an independent
result folder; RESULT_ARCHIVE is printed even on failure. Existing v17 and
v19 directories remain available.

Local checks emulate actual Triton bodies on CPU. They do not establish
HIP/Inductor compilation, full-model correctness, or performance.
