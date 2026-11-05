# V23 versus V22b

One paired screening run; repeat before claiming stable gains

| workload | v22b seconds | v23 seconds | elapsed reduction |
|---|---:|---:|---:|
| load1_b1_l32_g256 | 1.021699 | 1.046611 | -2.44% |
| load2_b8_l32_g256 | 1.472985 | 1.492739 | -1.34% |
| load3_b4_l32_g512 | 2.736471 | 2.557698 | 6.53% |
| load4_b4_l512_g64 | 0.452584 | 0.435869 | 3.69% |

B1 retains the frozen V22b plan; do not attribute its timing variation to the new B4/B8 kernels.
Both versions matched all 4608 public tokens and had zero formal compile/cache-miss/capture events.
