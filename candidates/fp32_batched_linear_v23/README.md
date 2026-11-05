# V23: batch-shared FP32 Linear

Parent: V22b, whose B1 attention-output and FFN-output shape plan is frozen.
This version targets B4/B8: a batch tile shares each weight tile, computes
FP32 elementwise products and reduces K. A direct variant completes each
output tile in one launch; a vector split variant writes FP32 partial sums
and merges deterministically. No atomics, half weights, input-dependent KV
reuse, or modifications to official model weights/tester/baseline.

Before model loading, an independent process uses synthetic tensors on the
eight B4/B8 Linear shapes. Four configurations are compared against compiled
FP32 F.linear (highest precision, TF32 disabled) with graph replay. A shape
is enabled only if both timing rounds are >=10% faster and its FP64-reference
check passes. The frozen B1 plan is validated separately and never retuned.
If no new shape qualifies, exit 3 archives the evidence and stops. Existing
V22b remains usable. The microbenchmark is not a model speedup claim.

Selected paths then receive compiled graph replay checks before the full
official four workloads. All 4608 public tokens must match exactly. Formal
generate must report zero compilation/cache-miss/graph-capture events. A fresh
V22b control follows a successful candidate run and produces comparison.json
and comparison.md. This single pair is screening, not a stable-gain conclusion.
Compilation/tuning occurs before model loading or during official untimed
warmup, never in the timed generate call. No dependency download/installation.

Summation order changes, so local tolerances do not guarantee identical model
tokens. HIP6.1/Triton3.0/gfx936 performance and full correctness are unverified.
Actual local NVIDIA/Torch2.11/Triton3.6 native kernel and compiled graph checks
passed. That local driver/toolchain is not included and is not required on DCU.
Detailed local checks: local_validation.json and local_layout_benchmark.json.

From the existing progen2_contest directory:
```bash
bash candidates/fp32_batched_linear_v23/run_test.sh
```
The script freezes sources and preserves failures; download RESULT_ARCHIVE.
The active submission, official baseline and official tools are unchanged.
Prepared with AI assistance. Original split-parallelism inspiration is listed
in the parent V22b manifest; the batch-tiled kernels here are independently
implemented. No third-party kernel source or binaries are bundled.
