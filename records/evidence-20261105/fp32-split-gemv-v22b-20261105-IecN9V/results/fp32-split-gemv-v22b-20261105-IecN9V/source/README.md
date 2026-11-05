# v22b: FP32 small-batch split-K GEMV

Parent: v20. Original FP32 weights, attention kernels, prefill, greedy rule and
official test tools are unchanged. Decode Linear operations can use a new
FP32 product/reduction kernel: parallel K slices followed by a fixed reduction.
No atomics, low-bit conversion, external packages, or request KV reuse.
Summation order changes; mathematical FP32 inputs do not guarantee identical
greedy output. All 4608 public output tokens must still match exactly.

Before loading the model, the script compares four configurations on the 12
batch/Linear shapes using independent synthetic inputs. Each entire operation
(partial + reduction) is timed with CUDA/HIP graphs, against compiled FP32
F.linear with highest precision and TF32 disabled. Both timing rounds must be
at least 10% faster and pass an FP64-reference tolerance check. Otherwise that
shape retains the original Linear. No selected shape -> exit 3 and archive the
diagnosis; the full model is skipped. This is a microbenchmark guard, not proof
of end-to-end speed. Hot individual weights differ from a full-model workload.

The selected shape-only plan is read during Solution initialization and frozen
before graph capture; no benchmark/autotune is performed in timed generation.
The formal-call audit still requires zero compile/cache miss/graph capture.
Environment: existing contest Torch 2.4.1 / HIP 6.1 / Triton 3.0; no install.
Input-independent graph preparation and official untimed warmup are allowed.

Run from progen2_contest:
```bash
bash candidates/fp32_split_gemv_v22b/run_test.sh
```
The script freezes source, records environment/official tool fingerprints,
checks the new matrix kernels, retained attention kernels and replay, then runs
the official tester and audit. RESULT_ARCHIVE is printed even on failure.

Primary reference (algorithm inspiration only, no source copied):
https://github.com/dropbox/gemlite/blob/master/gemlite/triton_kernels/gemv_kernels.py
GemLite (Apache-2.0) is low-bit; its split parallelism is borrowed as an idea,
not its quantization or published speed claims. Our FP32 scratch + fixed merge
replaces atomic accumulation. Also reviewed the shape-driven AMD FP32 GEMV:
https://github.com/uulong950/qingming-gfx1100-gemv/blob/main/gemv.cpp
That project targets gfx1100/ROCm 7.2, not our gfx936/HIP 6.1 environment.
No upstream binaries are used. Implementation and tests prepared with AI help.

Local checks cannot execute HIP/Triton in this workspace. Actual DCU performance
and full-model correctness remain unverified until this package is run there.

v22b fixes only microbenchmark lifecycle: reset Dynamo's code-object caches
before compiling each independent synthetic shape (Torch 2.4 defaults to eight
variants). This happens in the separate pre-model benchmark process, never in
Solution/generate. gemm_progress.json saves each completed shape immediately.
Kernel arithmetic, shape selection threshold, full-model code and audit are
unchanged from v22. The original incomplete v22 run is not a model speed result.
API verified against PyTorch v2.4.1 torch/_dynamo/__init__.py reset().
