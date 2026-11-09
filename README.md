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


| Vector Size (n) | C Implementation | CUDA 1 | CUDA 2 | CUDA 3 | CUDA 4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **2^20** | 3.80 ms | 2.60 ms | 2.10 ms | 2.05 ms | 1.40 ms |
| **2^26** | 294.83 ms | 164.83 ms | 133.80 ms | 131.19 ms | 90.02 ms |
| **2^28** | 1102.47 ms | 617.75 ms | 534.14 ms | 524.44 ms | 360.37 ms |


| **2^20** | Kernel Time ( avg | min | max )  | Speedup vs Baseline C Program
|:---:|:---:|
| C | 0.0 ms | 0.0 ms | 
| x86-64 | 0.0 ms | 0.0 ms | 
| x86-64 SIMD XMM | 0.0 ms | 0.0 ms | 
| x86-64 SIMD YMM | 0.0 ms | 0.0 ms | 
| CUDA Unifed | 0.0 ms | 0.0 ms |   
| CUDA Prefetch | 0.0 ms | 0.0 ms |   
| CUDA Prefetch + Page Creation | 0.0 ms | 0.0 ms |   
| CUDA Prefetch + Page Creation + creatition + memadvice | 0.0 ms | 0.0 ms |   

| **2^26** | Kernel Time ( avg | min | max )  | Speedup vs Baseline C Program
|:---:|:---:|
| C | 0.0 ms | 0.0 ms | 
| x86-64 | 0.0 ms | 0.0 ms | 
| x86-64 SIMD XMM | 0.0 ms | 0.0 ms | 
| x86-64 SIMD YMM | 0.0 ms | 0.0 ms | 
| CUDA Unifed | 0.0 ms | 0.0 ms |   
| CUDA Prefetch | 0.0 ms | 0.0 ms |   
| CUDA Prefetch + Page Creation | 0.0 ms | 0.0 ms |   
| CUDA Prefetch + Page Creation + creatition + memadvice | 0.0 ms | 0.0 ms |

| **2^28** | Kernel Time ( avg | min | max )  | Speedup vs Baseline C Program
|:---:|:---:|
| C | 0.0 ms | 0.0 ms | 
| x86-64 | 0.0 ms | 0.0 ms | 
| x86-64 SIMD XMM | 0.0 ms | 0.0 ms | 
| x86-64 SIMD YMM | 0.0 ms | 0.0 ms | 
| CUDA Unifed | 0.0 ms | 0.0 ms |   
| CUDA Prefetch | 0.0 ms | 0.0 ms |   
| CUDA Prefetch + Page Creation | 0.0 ms | 0.0 ms |   
| CUDA Prefetch + Page Creation + creatition + memadvice | 0.0 ms | 0.0 ms |


CUDA 1, total time (your table)	1.46×	1.79×	1.78×
CUDA 2, total time	1.81×	2.20×	2.06×
CUDA 3, total time	1.85×	2.25×	2.10×
CUDA 4, total time	2.71×	3.28×	3.06×
