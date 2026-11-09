# PARCO_CUDA

https://colab.research.google.com/drive/105piLoWUiHgEUbi6MsV5aIq0Vw2QMx8I?usp=sharing

<br>

**Group 4**
| Name | GitHub |
| :--- | :--- |
| Copon, Matthew Joshua | [@bananoms](http://github.com/bananoms) |
| Lee, Ashley Fiona | [@20-ash](https://github.com/20-ash) |
| Ong, Eiress Bassey | [@mochi-tofu](https://github.com/mochi-tofu) |
| Xu, Kai Wen | [@Kai-Wen-Xu](https://github.com/Kai-Wen-Xu) |

<br>

**Project Specifications**

The kernel computes an element-wise maximum with index tracking over two single-precision float vectors, A and B, of length `n`. For each index `i`, `c[i] = max(A[i], B[i])` is stored in a float vector C, and `idx[i]` is stored in a 32-bit int vector, set to 0 if `A[i] >= B[i]` and 1 otherwise.

Vector sizes are 2²⁰, 2²⁶, 2²⁸. The size was reduced from 2³⁰ to 2²⁸ because the four arrays would require 16 GiB of memory in total (4 GiB each), which exceeds the available memory of both our laptop and the Colab GPU. This range runs from a size that largely fits in cache (4 MB per array at 2²⁰) to sizes limited by memory bandwidth (1 GiB per array at 2²⁸). Kernels launch with 256 threads per block and at most 65,535 blocks, so the grid-stride loop (guarded by `i < N`) covers every element, with each thread processing multiple elements at the larger sizes. Data is initialized as `A[i] = sinf(i * 0.001f)` and `B[i] = cosf(i * 0.002f)`, so both outcomes of the index check occur.

Five variants are implemented, each in its own notebook cell: (1) a C/C++ program version (2) a CUDA program version using a grid-stride loop without prefetch and without mem advise (3) a CUDA program version using a grid-stride loop with prefetching but without page creation and without mem advise (4) a CUDA program version using a grid-stride loop with prefetch, with page creation but without mem advise (5) a CUDA program version using a grid-stride loop with prefetch, with page creation and with mem advise.

Each kernel is run 30 times with no warm-up launch. The C version is timed in the program; CUDA kernel times (avg/min/max) come from `nsys`. Each CUDA output is verified against a host-side recomputation of the same max/index logic as the C version, reporting a correctness percentage, with the first and last 5 elements displayed. `ncu` is used for kernel-level profiling.

<br>

**AI Usage Declaration**

<br>

## Program Outputs

**Correctness Check and Execution Time**
|  | **2^20** | **2^26** | **2^28** |
|:---:|:---:|:---:|:---:|
| C | ![C](screenshots/C20.png) | ![C](screenshots/C24.png) | ![C](screenshots/C28.png) |
| x86-64 | ![C](screenshots/ASM20.png) | ![C](screenshots/ASM26.png) | ![C](screenshots/ASM28.png) |
| x86-64 SIMD XMM | ![C](screenshots/XMM20.png) | ![C](screenshots/XMM26.png) | ![C](screenshots/XMM28.png) |
| x86-64 SIMD YMM | ![C](screenshots/YMM20.png) | ![C](screenshots/YMM26.png) | ![C](screenshots/YMM28.png) |
| CUDA Unified | ![C](screenshots/eCUDA1_20.png) | ![C](screenshots/eCUDA1_26.png) | ![C](screenshots/eCUDA1_28.png) |
| CUDA Prefetch | ![C](screenshots/eCUDA2_20.png) | ![C](screenshots/eCUDA2_26.png) | ![C](screenshots/eCUDA2_28.png) |
| CUDA Prefetch + Page Creation |  ![C](screenshots/eCUDA3_20.png) | ![C](screenshots/eCUDA3_26.png) | ![C](screenshots/eCUDA3_28.png) |
| CUDA Prefetch + Page Creation + MemAdvise | ![C](screenshots/eCUDA4_20.png) | ![C](screenshots/eCUDA4_26.png) | ![C](screenshots/eCUDA4_28.png) |

<br>

**nsys for all CUDA variants**
|  | **2^20** | **2^26** | **2^28** |
|:---:|:---:|:---:|:---:|
| CUDA Unified | ![C](screenshots/CUDA1_20.png) | ![C](screenshots/CUDA1_24.png) | ![C](screenshots/CUDA1_28.png) |
| CUDA Prefetch | ![C](screenshots/CUDA2_28.png) | ![C](screenshots/CUDA2_24.png) | ![C](screenshots/CUDA2_28.png) |
| CUDA Prefetch + Page Creation |  ![C](screenshots/CUDA3_20.png) | ![C](screenshots/CUDA3_24.png) | ![C](screenshots/CUDA3_28.png) |
| CUDA Prefetch + Page Creation + MemAdvise | ![C](screenshots/CUDA4_20.png) | ![C](screenshots/CUDA4_24.png) | ![C](screenshots/CUDA4_28.png) |

<br>

**Comparative table of execution time**

| **2^20** | Execution Time | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 3.80 ms | - |
| x86-64 (Debug / Release) | 1.58 ms / 1.56 ms | 2.41× / 2.44× |
| x86-64 SIMD XMM (Debug / Release) | 0.86 ms / 0.65 ms | 4.42× / 5.85× |
| x86-64 SIMD YMM (Debug / Release) | 0.92 ms / 0.80 ms | 4.13× / 4.75× |
| CUDA Unified | 2.60 ms | 1.46× |
| CUDA Prefetch | 2.10 ms | 1.81× |
| CUDA Prefetch + Page Creation | 2.05 ms | 1.85× |
| CUDA Prefetch + Page Creation + MemAdvise | 1.40 ms | 2.71× |

| **2^26** | Execution Time | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 294.83 ms | - |
| x86-64 (Debug / Release) | 147.62 ms / 68.00 ms | 2.00× / 4.34× |
| x86-64 SIMD XMM (Debug / Release) | 66.88 ms / 56.36 ms | 4.41× / 5.23× |
| x86-64 SIMD YMM (Debug / Release) | 58.31 ms / 56.96 ms | 5.06× / 5.18× |
| CUDA Unified | 164.83 ms | 1.79× |
| CUDA Prefetch | 133.80 ms | 2.20× |
| CUDA Prefetch + Page Creation | 131.19 ms | 2.25× |
| CUDA Prefetch + Page Creation + MemAdvise | 90.02 ms | 3.28× |

| **2^28** | Execution Time | Speedup vs Baseline C Program |
|:---:|:---:|:---:|
| C | 1102.47 ms | - |
| x86-64 (Debug / Release) | 263.91 ms / 239.08 ms | 4.18× / 4.61× |
| x86-64 SIMD XMM (Debug / Release) | 211.67 ms / 191.09 ms | 5.21× / 5.77× |
| x86-64 SIMD YMM (Debug / Release) | 256.79 ms / 242.90 ms | 4.29× / 4.54× |
| CUDA Unified | 617.75 ms | 1.78× |
| CUDA Prefetch | 534.14 ms | 2.06× |
| CUDA Prefetch + Page Creation | 524.44 ms | 2.10× |
| CUDA Prefetch + Page Creation + MemAdvise | 360.37 ms | 3.06× |

<br>

**CUDA Kernel Times**

| **2^20** | Kernel Avg | Kernel Min | Kernel Max | 
|:---:|:---:|:---:|:---:|
| CUDA Unified | 0.18 ms | 0.07 ms | 3.50 ms |
| CUDA Prefetch | 0.11 ms | 0.07 | 1.40 ms |
| CUDA Prefetch + Page Creation | 0.07 ms | 0.07 ms | 0.07 ms |
| CUDA Prefetch + Page Creation + MemAdvise | 0.07 ms | 0.07 ms | 0.07 ms |

| **2^24** | Kernel Avg | Kernel Min | Kernel Max | 
|:---:|:---:|:---:|:---:|
| CUDA Unified | 10.68 ms | 4.35 ms | 191.12 ms |
| CUDA Prefetch | 6.80 ms | 4.35 ms | 77.71 ms |
| CUDA Prefetch + Page Creation | 4.35 ms | 4.34 ms | 4.34 ms |
| CUDA Prefetch + Page Creation + MemAdvise | 4.35 ms | 4.34 ms | 4.34 ms |

| **2^28** | Kernel Avg | Kernel Min | Kernel Max | 
|:---:|:---:|:---:|:---:|
| CUDA Unified | 44.33 ms | 18.75 ms | 783.33 ms |
| CUDA Prefetch | 28.06 ms | 18.30 ms | 310.49 ms |
| CUDA Prefetch + Page Creation | 18.35 ms | 18.04 ms | 18.44 ms |
| CUDA Prefetch + Page Creation + MemAdvise | 18.31 ms | 18.01 ms | 18.48 ms |

<br>

## Analysis of Results
**Analysis of kernel execution time**
**Analysis of speed performance across all platforms**

<br>

## Discussions

**Problems Encountered, Methodology, and AHA Moments**

Our group did not encounter significant problems throughout the project, as the implementation followed the approach from the previous discovery series and recorded class lecture. We kept the kernel identical across all four CUDA variants to ensure that differences in execution time came from memory management techniques rather than computation. Each variant was tested 30 times using problem sizes of 2²⁰, 2²⁶, and 2²⁸, profiled with nsys and ncu, and checked against the same max/index logic as the C version, with the first and last five elements displayed for verification. We also didn't have any major AHA moments since most of the concepts were already discussed in class, but working on our own kernel helped us see how much data transfer between the CPU and GPU affects performance.

<br>

**SIMD vs. SIMT**

Both SIMD and SIMT exploit data parallelism but differ in how they process data. In SIMD, a single thread processes multiple elements using packed instructions. In our assembly kernels, we manually processed 4 floats using XMM registers or 8 using YMM registers, requiring explicit index increments and remainder handling (e.g., 3 leftover elements at n = 1003). In SIMT, each thread executes the same scalar code on one element at a time, with the GPU grouping threads into warps of 32 and managing their execution. Our grid-stride loop and `i < N` condition let each thread cover multiple elements at the larger sizes and handled different input sizes without additional remainder code.

- **SIMD**

    * Pros: Lower overhead since data stays in CPU memory with no GPU transfers, and efficient processing through packed instructions.
    * Cons: Fixed vector widths, manual remainder handling, and a more complex assembly implementation.

- **SIMT**

    * Pros: Thousands of threads enable greater parallelism and memory bandwidth utilization. Scalar-style kernel code is also simpler and scales more easily across input sizes.
    * Cons: CPU-GPU data movement overhead, initial page-fault costs, and potential branch divergence. In our case, the `if (A[i] >= B[i])` condition is simple, so divergence is expected to have little impact.

Overall, SIMT is more suitable for larger inputs because it takes advantage of greater parallelism and GPU memory bandwidth. SIMD is more practical for smaller inputs since it avoids GPU launch and data transfer overhead.