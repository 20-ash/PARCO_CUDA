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

![RAM](screenshots/RAM.png)
The maximum input size was selected as 2^28 because the system has 31.4 GB of usable RAM, with approximately 17.4 GB available during testing. Since ten arrays of 4-byte floating-point elements require approximately 10 GiB at this input size, it provides sufficient memory headroom for the experiment while minimizing the risk of allocation failure or excessive memory swapping.

If 2^29 selected:
Ten arrays of 4-byte floating-point elements require approximately 20 GiB of memory, exceeding the approximately 17.4 GB of available RAM during testing. Therefore, this input size may cause memory allocation failures or excessive memory swapping.

If 2^30 selected :
Ten arrays of 4-byte floating-point elements require approximately 40 GiB of memory. This exceeds the system's total usable RAM of 31.4 GB, making it impractical to allocate all arrays simultaneously and increasing the likelihood of memory allocation failures.

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

**Comparative table of kernel time**

| **2^20** | Kernel Avg | Kernel Min | Kernel Max | 
|:---:|:---:|:---:|:---:|
| CUDA Unified | 0.18 ms | 0.07 ms | 3.50 ms |
| CUDA Prefetch | 0.11 ms | 0.07 ms | 1.40 ms |
| CUDA Prefetch + Page Creation | 0.07 ms | 0.07 ms | 0.07 ms |
| CUDA Prefetch + Page Creation + MemAdvise | 0.07 ms | 0.07 ms | 0.07 ms |

| **2^26** | Kernel Avg | Kernel Min | Kernel Max | 
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

**Summary Table of HtoD/DtoH Overhead**
- Overhead from `cudaMallocManaged` is generally the same regardless of prefetch, page creation, and MemAdvise conditions.
- Overhead from `cudaMemPrefetchAsync` increases drastically when page creation is included.

| **2^20** | cudaMallocManaged | cudaMemPrefetchAsync | 
|:---:|:---:|:---:|
| CUDA Unified | 214.65 ms | N/A |
| CUDA Prefetch | 214.76 ms | 2.57 ms |
| CUDA Prefetch + Page Creation | 207.95 ms | 4.57 ms |
| CUDA Prefetch + Page Creation + MemAdvise | 234.19 ms | 5.03 ms |

| **2^24** | cudaMallocManaged | cudaMemPrefetchAsync | 
|:---:|:---:|:---:|
| CUDA Unified | 267.22 ms | N/A |
| CUDA Prefetch | 211.93 ms | 95.45 ms |
| CUDA Prefetch + Page Creation | 217.18 ms | 252.15 ms |
| CUDA Prefetch + Page Creation + MemAdvise | 211.47 ms | 222.92 ms |

| **2^28** | cudaMallocManaged | cudaMemPrefetchAsync | 
|:---:|:---:|:---:|
| CUDA Unified | 202.02 ms | N/A |
| CUDA Prefetch | 206.25 ms | 525.31 ms |
| CUDA Prefetch + Page Creation | 208.63 ms | 1032.13 ms |
| CUDA Prefetch + Page Creation + MemAdvise | 215.61 ms | 1001.79 ms |

<br>

## Analysis of Results
**Analysis of kernel execution time and speed performance across all platforms**

- #### CUDA Unified (Grid-Stride Loop, No Prefetch, No MemAdvise)
CUDA Unified is baseline CUDA variant where no optimizations were placed. It had an execution time of 2.60 ms / 164.83 ms / 617.75 ms for 2^20 / 2^26 / 2^28 respectively. The Kernel max time of 3.50 ms / 191.12 ms / 785.33 ms comes entirely from the first execution. At that point, the values of A and B are still in CPU memory, and C and idx have no allocated pages anywhere. When program access a page that is not allocated yet on the device, it triggers a page fault and stalls while the driver either migrates the page host-to-device (A, B) or creates it in device memory (C, idx). This on-demand migration happens in small pieces: at 2^28, nsys recorded 20,491 host-to-device transfers averaging only 0.089 MB each, which uses the PCIe bus inefficiently and adds a per-fault overhead.

Once every page is on the Device, launches 2–30 run at the Kernel min time of 0.069 ms / 4.35 ms / 18.75 ms. That makes the first launch ~42–50× slower. This single outlier pushes the average kernel time to 0.18 ms / 10.68 ms / 44.33 ms, more than double the Kernel min.

- #### CUDA Prefetch
CUDA Prefetch is grid-stride loop with the prefetch optimization. It had an execution time of 2.10 ms / 133.80 ms / 534.14 ms respectively. Prefetch is where the host requests the migration of data before executing the kernel, instead of the GPU discovering missing pages one fault at a time during runtime. This results in a Kernel Max time of: 1.40 ms / 77.71 ms / 310.49 ms. The time max time is significantly reduced while the Kernel Min time remains the same at: 0.07 ms / 4.35 ms / 18.30 ms. The max and min time are not approximately the same yet because there's still a page fault. The GPU is not yet ready and has to prepare a  box for the data (page creation). It is worth noting that cudaMemPrefetchAsync adds overhead on the CPU side. It can be found in the CUDA API summary.

- #### CUDA Prefetch + Page Creation
This variant adds explicit page creation. A and B are prefetched to the CPU before initialization, so their pages exist in host memory, and C and idx are prefetched to the Device before the kernel, so their pages are created directly in Device memory. Because C and idx hold no data yet, this prefetch transfers nothing. It only creates the pages at the destination. It had an execution time of 2.05 ms / 131.19 ms / 524.44 ms. With every page the kernel needs already exists on the Device before the first transfer. This results in no page faults occurring. The Kernel max and min are very similar at 0.070 ms / 4.350 ms / 18.44 ms. 

The end-to-end improvement over Prefetch is small, because the transfers are unchanged: A and B still travel host-to-device (2,147.48 MB), and all four arrays still return device-to-host (4,294.97 MB) for the correctness check. Page creation removes the first-transfer penalty but does not reduce the amount of data moved.

- #### CUDA Prefetch + Page Creation + MemAdvise
This is prefetch along with the optimizations of page creation and memadvice. It has an execution time of  1.40 ms / 90.02 ms / 360.37 ms. Memadvice essentially tells the device to copy the data from the host instead of cutting. Kernel Min and Max of 0.07 ms / 4.34 ms / 18.44 ms is essentially the same but the time saved from it occurs from the execution time (total time for device-to-host decreased) and amount of data moved. 

This variant adds cudaMemAdvise hints on A and B: cudaMemAdviseSetPreferredLocation and cudaMemAdviseSetReadMostly. It had an execution time of 1.40 ms / 90.02 ms / 360.37 ms, the fastest of the CUDA variants.

SetReadMostly changes prefetching from moving pages to copying them. Because the kernel only reads A and B, so the inputs no longer need to be returned. Only C and idx are migrated back. Device-to-host traffic is halved (4,294.97 → 2,147.48 MB at 2^28), and its transfer time drops from 328.06 ms to 164.02 ms.

The kernel itself is unaffected: Kernel min 0.066 / 4.349 / 18.01 ms and Kernel max 0.070 / 4.356 / 18.48 ms is the same as the previous variant. All of the end-to-end improvement comes from the reduced device-to-host transfer.

#### CUDA Variants Summary
<mark>To summarize the CUDA Variants, the optimization of the device runtime is the reduction combination of both amount of data and overhead applied when running these kernels. Prefetch and Page creation essentially reserves the spaces and migrates data over to the device, while MemAdvice allows input data to no longer be returned. This allows for the efficient utilization of GPU.</mark>

#### CPU implementations (C, x86-64, XMM, YMM)
The plain C program is the slowest of the CPU versions, taking 3.80 ms, 294.83 ms, and 1102.47 ms for the three input sizes. It handles just one number at a time, so it repeats the same loop work over and over. The x86-64 assembly version is about 2× to 4.6× faster because its instructions are leaner and waste less effort. The XMM version is faster still (4.4× to 5.9×) because it processes several numbers with a single instruction (four at once instead of one). It is the fastest version overall, finishing in 0.65 ms, 56.36 ms, and 191.09 ms. The YMM version can carry twice as many numbers per instruction as XMM. However, it only matches XMM at best (4.1× to 5.2×) and is usually slower. The reason is that the ymm program spends most of its time waiting for data to arrive from memory, not doing math. Once the loop saturates memory bandwidth, wider registers cannot make data arrive faster.

#### CUDA compared to the CPU versions
CUDA’s best version (Prefetch + Page Creation + MemAdvise) is 2.71× to 3.28× faster than the plain C program, but it is still slower than every SIMD version. 

(1) One reason is that adding more threads doesn’t help here and can even make things slower. Each extra thread adds scheduling work, and when thousands of threads ask for data at the same time, they crowd the memory system and get in each other’s way. The data also has to be copied from the computer’s main memory over to the GPU before it can start working. 

(2) On top of that, Unified Memory adds extra waiting time for the system to move pages around, plus a small cost to start and finish each GPU job. Each CUDA improvement (prefetch, page creation, MemAdvise) helped because it cut down this waiting, which shows that moving the data is what slows CUDA down. The CPU doesn’t have this problem because the data is already sitting right next to it, so it can start working immediately. 

<mark>Therefore, for a simple operation like this project where the data starts on the CPU side, the CPU versions are simply the better fit. The GPU would only win if the data stayed on the GPU across many jobs, or if each number needed a lot more math.</mark>

<br>

## Discussions

**Problems Encountered, Methodology, and AHA Moments**

Our group did not encounter significant problems throughout the project, as the implementation followed the approach from the previous discovery series and recorded class lecture. We kept the kernel identical across all four CUDA variants to ensure that differences in execution time came from memory management techniques rather than computation. Each variant was tested 30 times using problem sizes of 2²⁰, 2²⁶, and 2²⁸, profiled with nsys and ncu, and checked against the same max/index logic as the C version, with the first and last five elements displayed for verification. We also didn't have any major AHA moments since most of the concepts were already discussed in class, but working on our own kernel helped us see how much data transfer between the CPU and GPU affects performance.

<br>

a.) How does block size affect execution time (observing various elements and using max blocks)?  Which block size will you recommend?

A GPU is made up of Streaming Multiprocessors (SMs), the processing units that run the threads. The T4 has 40 SMs, and each one can hold up to 1024 threads and 16 blocks at a time. The remaining blocks wait in a queue until a running block finishes and frees its space.

Based on the ncu of the final CUDA version (Prefetch + Page Creation + MemAdvise) at 2^28 with 256 threads per block, the kernel took 18.03 ms and used 86.6% of the T4's memory bandwidth, while compute usage was only 24.1%. Each warp spent most of its time waiting for data from memory. This means the kernel is limited by memory, so once each SM is filled with threads, adding more threads per block does not make it faster.

Small blocks (such as 32) would probably be slower because the SM hits its 16-block limit before it is filled with threads, leaving half of its capacity unused. Very large blocks (such as 1024) would probably be slightly slower because the SM must wait for every thread in a block to finish before starting a new one. Block sizes from 64 to 512 fill the SM completely, so they would likely perform about the same.

These results are specific to 2^28 on the final CUDA version and may differ at other input sizes or in the earlier CUDA versions.

<mark>A block size of 256 is recommended. It fills every SM completely, stays well below the 16-block limit, and divides evenly into the 1024 and 1536 thread limits of current GPUs. Block sizes from 64 to 512 would probably perform about the same, while very small blocks leave the SM partly empty and very large blocks make the SM wait longer between blocks.</mark>

b) Is prefetching always recommended, or should CUDA manage memory?  Give some use cases in which one is better than the other.

In our results, prefetching reduced the Kernel Max at 2^28 from 785.33 ms to 310.49 ms (2.50x speedup). It also replaced 20,491 small on-demand transfers (avg 0.089 MB) with 1,024 bulk transfers of 2.097 MB, which use the PCIe bus far more efficiently. Prefetching works well here because our access pattern is simple and fully known in advance and the data (~4.3 GB at 2^28) fits in the T4's 16 GB of memory. This means the utilization of the PCIe bus was used far more efficiently. The burden is placed instead on the Host where from 1.33 ms to 841.66 ms at 2^28. This is the cudaDevicesynchronise which relates to the API call from Host that exists on the CPU. 

The Use cases where prefetching is better
- (1) The whole dataset is used and the access pattern is known: element-wise vector operations like ours, matrix multiplication, image filters.
- (2) The data is reused across many kernel launches:** iterative simulations or repeated processing, where the transfer is paid once and every launch runs at full speed.
- (3)Data that the CPU and GPU take turns using in large, clearly defined phases:** prefetching at each hand-off avoids a wave of page faults every time control switches sides.

#### Use cases where CUDA-managed (on-demand) memory is better
- (1) Sparse or unpredictable access: graph traversal, hash tables, sparse matrices, or search, where only a small, data-dependent part of a large array is touched. Prefetching everything would move data that is never used.
- (2) Datasets larger than GPU memory (oversubscription): on-demand paging lets the kernel work on data that does not fit on the GPU. Prefetching the whole range would force evictions and wasted transfers.

<mark>The conclusion then is that Prefetch is when you know which data the kernel will use and it fits on the GPU. Let CUDA manage memory when the access pattern is sparse, unpredictable, or larger than GPU memory.</mark>

c.) Between SIMD and SIMT, which one is faster? Give some use cases in which one is better than the other.

**SIMD vs. SIMT**

Both SIMD and SIMT exploit data parallelism but differ in how they process data. In SIMD, a single thread processes multiple elements using packed instructions. In our assembly kernels, we manually processed 4 floats using XMM registers or 8 using YMM registers, requiring explicit index increments and remainder handling (e.g., 3 leftover elements at n = 1003). In SIMT, each thread executes the same scalar code on one element at a time, with the GPU grouping threads into warps of 32 and managing their execution. Our grid-stride loop and `i < N` condition let each thread cover multiple elements at the larger sizes and handled different input sizes without additional remainder code.

- **SIMD**

    * Pros: Lower overhead since data stays in CPU memory with no GPU transfers, and efficient processing through packed instructions.
    * Cons: Fixed vector widths, manual remainder handling, and a more complex assembly implementation.

- **SIMT**

    * Pros: Thousands of threads enable greater parallelism and memory bandwidth utilization. Scalar-style kernel code is also simpler and scales more easily across input sizes.
    * Cons: CPU-GPU data movement overhead, initial page-fault costs, and potential branch divergence. In our case, the `if (A[i] >= B[i])` condition is simple, so divergence is expected to have little impact.

<br>

<mark>Overall, SIMT is more suitable for larger inputs because it takes advantage of greater parallelism and GPU memory bandwidth. SIMD is more practical for smaller inputs since it avoids GPU launch and data transfer overhead.</mark>
