# PARCO_CUDA

https://colab.research.google.com/drive/105piLoWUiHgEUbi6MsV5aIq0Vw2QMx8I?usp=sharing

**Group 4**
- Copon, Matthew Joshua
- Lee, Ashley Fiona
- Ong, Eiress Bassey
- Xu, Kai Wen

**Project Specifications**

The kernel computes an element-wise maximum with index tracking over two single-precision float vectors, A and B, of length `n`. For each index `i`, `c[i] = max(A[i], B[i])` is stored in a float vector C, and `idx[i]` is stored in a 32-bit int vector, set to 0 if `A[i] >= B[i]` and 1 otherwise.

Vector sizes are 2^20, 2^26, and 2^28. Kernels launch with 256 threads per block and at most 65,535 blocks, so the grid-stride loop (guarded by `i < N`) covers every element, with each thread processing multiple elements at the larger sizes. Data is initialized as `A[i] = sinf(i * 0.001f)` and `B[i] = cosf(i * 0.002f)`, so both outcomes of the index check occur.

Five variants are implemented, each in its own notebook cell: (1) a C/C++ program version (2) a CUDA program version using a grid-stride loop without prefetch and without mem advise (3) a CUDA program version using a grid-stride loop with prefetching but without page creation and without mem advise (4) a CUDA program version using a grid-stride loop with prefetch, with page creation but without mem advise (5) a CUDA program version using a grid-stride loop with prefetch, with page creation and with mem advise.

Each kernel is run 30 times with no warm-up launch. The C version is timed in the program; CUDA kernel times (avg/min/max) come from `nsys`. Each CUDA output is verified against a host-side recomputation of the same max/index logic as the C version, reporting a correctness percentage, with the first and last 5 elements displayed. `ncu` is used for kernel-level profiling.


**AI Usage Declaration**

## Program Outputs


|  | **2^20** | **2^26** | **2^28** |
|:---:|:---:|:---:|:---:|
| C | ![C](screenshots/C20.png) | ![C](screenshots/C24.png) | ![C](screenshots/C28.png) |
| x86-64 | ![C](screenshots/ASM20.png) | ![C](screenshots/ASM24.png) | ![C](screenshots/ASM28.png) |
| x86-64 SIMD XMM | ![C](screenshots/XMM20.png) | ![C](screenshots/XMM24.png) | ![C](screenshots/XMM28.png) |
| x86-64 SIMD YMM | ![C](screenshots/YMM20.png) | ![C](screenshots/YMM24.png) | ![C](screenshots/YMM28.png) |
| CUDA Unified | ![C](screenshots/CUDA1_20.png) | ![C](screenshots/CUDA1_24.png) | ![C](screenshots/CUDA1_28.png) |
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

| **2^20** | Kernel Avg | Kernel Min | Kernel Max | 
|:---:|:---:|:---:|:---:|
| CUDA Unified | 2.60 ms | 1.46× | X.XX ms |
| CUDA Prefetch | 2.10 ms | 1.81× | X.XX ms |
| CUDA Prefetch + Page Creation | 2.05 ms | 1.85× | X.XX ms |
| CUDA Prefetch + Page Creation + MemAdvise | 1.40 ms | 2.71× | X.XX ms |

| **2^24** | Kernel Avg | Kernel Min | Kernel Max | 
|:---:|:---:|:---:|:---:|
| CUDA Unified | 2.60 ms | 1.46× | X.XX ms |
| CUDA Prefetch | 2.10 ms | 1.81× | X.XX ms |
| CUDA Prefetch + Page Creation | 2.05 ms | 1.85× | X.XX ms |
| CUDA Prefetch + Page Creation + MemAdvise | 1.40 ms | 2.71× | X.XX ms |

| **2^28** | Kernel Avg | Kernel Min | Kernel Max | 
|:---:|:---:|:---:|:---:|
| CUDA Unified | 2.60 ms | 1.46× | X.XX ms |
| CUDA Prefetch | 2.10 ms | 1.81× | X.XX ms |
| CUDA Prefetch + Page Creation | 2.05 ms | 1.85× | X.XX ms |
| CUDA Prefetch + Page Creation + MemAdvise | 1.40 ms | 2.71× | X.XX ms |


## Analysis of Results
**Analysis of kernel execution time**
**Analysis of speed performance across all platforms**

## Discussions
**problems encountered and solutions made, unique methodology used, AHA moments**
**differences between SIMD and SIMT in handling parallelism. Include also the PROS and CONS of using SIMD and SIMT in your use case.**

