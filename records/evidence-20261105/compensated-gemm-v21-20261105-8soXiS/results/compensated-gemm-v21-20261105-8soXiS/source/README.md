# v21: compensated high/low matrix multiplication

Parent v20. This changes the four large Linear operations in every decode
block (108 operations for the 27-layer model), rather than attention tiles.
Original FP32 model weights and prefill execution remain intact. Additional
weight encodings are prepared during input-independent initialization.

Scaled weights and activations are split into FP16 high and residual low
parts. Four FP16 dot products accumulate in separate FP32 accumulators;
FP32 recombination restores the cross and low-low terms before bias.
This is compensated mixed arithmetic, NOT mathematically equivalent FP32.
The scaling reduces half-subnormal exposure but cannot eliminate it.
NaN, infinity, overflow, summation order, and tiny top-1 margins can still
cause divergence. No speed or correctness benefit is claimed before DCU.

The motivation is to try matrix-unit throughput while recovering information
lost by direct FP16 conversion. Four dot products and extra buffers can make
it slower, especially for small batches. Original FP32 weights plus encoded
weights require additional memory approximately equal to the weight size.
No compressed weights or cached input-dependent KV are reused across calls.

Run from the original contest directory:
```bash
bash candidates/compensated_gemm_v21/run_test.sh
```
The script checks representative matrix shapes (with a timing diagnostic),
the unchanged attention kernels, and graph replay, then invokes the original
public tester. It produces RESULT_ARCHIVE even on failure. Acceptance requires
all 4608 public tokens exact and zero formal-call compile/miss/capture counts.
The preflight arithmetic tolerance is not the final output acceptance rule.
All new GEMM code and reproduction steps are included; dependencies remain
the contest Torch 2.4.1 / HIP 6.1 / Triton 3.0 installation.

Source references: Triton language dot API and tutorial (FP16 inputs with
FP32 accumulation), https://triton-lang.org/main/python-api/generated/triton.language.dot.html
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
Numeric limitations: https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
Implementation and local tests prepared with AI assistance.
