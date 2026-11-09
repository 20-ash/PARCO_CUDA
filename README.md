# PARCO_CUDA

https://colab.research.google.com/drive/105piLoWUiHgEUbi6MsV5aIq0Vw2QMx8I?usp=sharing

**Group 4**
- Copon, Matthew Joshua
- Lee, Ashley Fiona
- Ong, Eiress Bassey
- Xu, Kai Wen

**Project Specifications**

**AI Usage Declaration**

## Analysis of Results


| **2^20** | Kernel Time (avg / min / max) | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 3.80 ms | 1.00× |
| x86-64 | X.XX ms | 0.0× |
| x86-64 SIMD XMM | X.XX ms | 0.0× |
| x86-64 SIMD YMM | X.XX ms | 0.0× |
| CUDA Unified | 2.60 ms / X.XX ms / X.XX ms | 57.3× |
| CUDA Prefetch | 2.10 ms / X.XX ms / X.XX ms | 57.2× |
| CUDA Prefetch + Page Creation | 2.05 ms / X.XX ms / X.XX ms | 57.2× |
| CUDA Prefetch + Page Creation + MemAdvise | 1.40 ms / X.XX ms / X.XX ms | 57.3× |

| **2^26** | Kernel Time (avg / min / max) | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 294.83 ms | 1.00× |
| x86-64 | X.XX ms | 0.0× |
| x86-64 SIMD XMM | X.XX ms | 0.0× |
| x86-64 SIMD YMM | X.XX ms | 0.0× |
| CUDA Unified | 164.83 / X.XX ms / X.XX ms | 67.9× |
| CUDA Prefetch | 133.80 / X.XX ms / X.XX ms | 68.0× |
| CUDA Prefetch + Page Creation | 131.19 ms / X.XX ms / X.XX ms | 67.9× |
| CUDA Prefetch + Page Creation + MemAdvise | 90.2 ms / X.XX ms / X.XX ms | 67.9× |

| **2^28** | Kernel Time (avg / min / max) | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 1102.47 ms | 1.00× |
| x86-64 | X.XX ms | 0.0× |
| x86-64 SIMD XMM | X.XX ms | 0.0× |
| x86-64 SIMD YMM | X.XX ms | 0.0× |
| CUDA Unified | 617.75 ms / X.XX ms / X.XX ms | 61.2× |
| CUDA Prefetch | 534.14 ms / X.XX ms / X.XX ms | 61.3× |
| CUDA Prefetch + Page Creation | 524.44 ms / X.XX ms / X.XX ms | 61.1× |
| CUDA Prefetch + Page Creation + MemAdvise | 360.37 / X.XX ms / X.XX ms | 61.1× |


