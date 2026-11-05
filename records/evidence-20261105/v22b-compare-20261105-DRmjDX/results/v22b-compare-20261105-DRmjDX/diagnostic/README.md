# v20 / v22b same-session confirmation

Execute run_test.sh from the existing contest root. The package carries exact
copies of v20 and v22b plus the input-independent shape plan selected in the
successful v22b run. No matrix microbenchmark is repeated. No weights included.

Runs official workloads in order v20, v22b, v22b, v20 (two runs each), sequential
fresh processes in the same open container. Each uses the original official
tester and its warmup. No official tools or active submission are overwritten.
Each run freezes source and independently checks all 4608 public tokens plus
zero formal Dynamo/Triton-miss/capture counters. Previously passed native kernel
preflights are not repeated; this package changes only experiment orchestration.

summary.json and report.md report all timings, paired differences, source/plan
fingerprints, and failures. A conservative B1 signal requires both paired runs
at least 3% faster and median B1 time at least 5% lower; a >5% median slowdown on
another workload blocks the positive decision. These are practical screening
thresholds, not a statistical confidence interval or official scoring formula.

The selected plan and prior environment are included for reproducibility. The
same GPU name, HIP and dependency versions are required; official file hashes
must equal the verified previous run and remain unchanged during this session.
The plan is fixed before real requests. Input-dependent KV is reset per request.
No cross-request outputs or KV are reused. No install/download is performed.

The EXIT trap archives all results even after a test failure. A failed run does
not prevent remaining comparison runs; incomplete/incorrect results never count
as a speed win. Check BATCH_ARCHIVE= at completion. No GitHub publication occurs.
