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
| x86-64 | 0.0 ms | 0.0× |
| x86-64 SIMD XMM | 0.0 ms | 0.0× |
| x86-64 SIMD YMM | 0.0 ms | 0.0× |
| CUDA Unified | 0.0663 / 0.0661 / 0.0667 ms | 57.3× |
| CUDA Prefetch | 0.0664 / 0.0661 / 0.0667 ms | 57.2× |
| CUDA Prefetch + Page Creation | 0.0664 / 0.0661 / 0.0668 ms | 57.2× |
| CUDA Prefetch + Page Creation + MemAdvise | 0.0663 / 0.0661 / 0.0665 ms | 57.3× |

| **2^26** | Kernel Time (avg / min / max) | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 294.83 ms | 1.00× |
| x86-64 | 0.0 ms | 0.0× |
| x86-64 SIMD XMM | 0.0 ms | 0.0× |
| x86-64 SIMD YMM | 0.0 ms | 0.0× |
| CUDA Unified | 4.339 / 4.330 / 4.340 ms | 67.9× |
| CUDA Prefetch | 4.334 / 4.330 / 4.340 ms | 68.0× |
| CUDA Prefetch + Page Creation | 4.340 / 4.340 / 4.340 ms | 67.9× |
| CUDA Prefetch + Page Creation + MemAdvise | 4.340 / 4.340 / 4.340 ms | 67.9× |

| **2^28** | Kernel Time (avg / min / max) | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 1102.47 ms | 1.00× |
| x86-64 | 0.0 ms | 0.0× |
| x86-64 SIMD XMM | 0.0 ms | 0.0× |
| x86-64 SIMD YMM | 0.0 ms | 0.0× |
| CUDA Unified | 18.007 / 18.000 / 18.020 ms | 61.2× |
| CUDA Prefetch | 17.992 / 17.980 / 18.000 ms | 61.3× |
| CUDA Prefetch + Page Creation | 18.037 / 18.020 / 18.050 ms | 61.1× |
| CUDA Prefetch + Page Creation + MemAdvise | 18.037 / 18.030 / 18.050 ms | 61.1× |


