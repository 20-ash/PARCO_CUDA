# PARCO_CUDA

https://colab.research.google.com/drive/105piLoWUiHgEUbi6MsV5aIq0Vw2QMx8I?usp=sharing

**Group 4**
- Copon, Matthew Joshua
- Lee, Ashley Fiona
- Ong, Eiress Bassey
- Xu, Kai Wen

**Project Specifications**

**AI Usage Declaration**

## Program Outputs


| Input Size | **2^20** | **2^26** | **2^28** |
|:---:|:---:|:---:|:---:|
| CUDA Unified | ![C](screenshots/CUDA1_20.png)  ![C](screenshots/CUDA1_24.png) | ![C](screenshots/CUDA1_28.png) |
| CUDA Prefetch | ![C](screenshots/CUDA2_28.png) | ![C](screenshots/CUDA2_24.png) | ![C](screenshots/CUDA2_28.png) |
| CUDA Prefetch + Page Creation |  ![C](screenshots/CUDA3_20.png) | ![C](screenshots/CUDA3_24.png) | ![C](screenshots/CUDA3_28.png) |
| CUDA Prefetch + Page Creation + MemAdvise | ![C](screenshots/CUDA4_20.png) | ![C](screenshots/CUDA4_24.png) | ![C](screenshots/CUDA4_28.png) |

| **2^20** | Execution Time | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 3.80 ms | - |
| x86-64 | X.XX ms | 0.0× |
| x86-64 SIMD XMM | X.XX ms | 0.0× |
| x86-64 SIMD YMM | X.XX ms | 0.0× |
| CUDA Unified | 2.60 ms | 1.46× |
| CUDA Prefetch | 2.10 ms | 1.81× |
| CUDA Prefetch + Page Creation | 2.05 ms | 1.85× |
| CUDA Prefetch + Page Creation + MemAdvise | 1.40 ms | 2.71× |

| **2^26** | Execution Time | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 294.83 ms | - |
| x86-64 | X.XX ms | 0.0× |
| x86-64 SIMD XMM | X.XX ms | 0.0× |
| x86-64 SIMD YMM | X.XX ms | 0.0× |
| CUDA Unified | 164.83 ms | 1.79× |
| CUDA Prefetch | 133.80 ms | 2.20× |
| CUDA Prefetch + Page Creation | 131.19 ms | 2.25× |
| CUDA Prefetch + Page Creation + MemAdvise | 90.02 ms | 3.28× |

| **2^28** | Execution Time | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 1102.47 ms | - |
| x86-64 | X.XX ms | 0.0× |
| x86-64 SIMD XMM | X.XX ms | 0.0× |
| x86-64 SIMD YMM | X.XX ms | 0.0× |
| CUDA Unified | 617.75 ms | 1.78× |
| CUDA Prefetch | 534.14 ms | 2.06× |
| CUDA Prefetch + Page Creation | 524.44 ms | 2.10× |
| CUDA Prefetch + Page Creation + MemAdvise | 360.37 ms | 3.06× |


## Analysis of Results
**Analysis of kernel execution time**
**Analysis of speed performance across all platforms**

## Discussions
**problems encountered and solutions made, unique methodology used, AHA moments**
**differences between SIMD and SIMT in handling parallelism. Include also the PROS and CONS of using SIMD and SIMT in your use case.**

