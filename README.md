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
| **2^20** | 1.73 ms | 0.79 ms | 0.39 ms | 0.38 ms | 0.00 ms |
| **2^26** | 113.15 ms | 64.05 ms | 47.06 ms | 46.43 ms | 0.00 ms |
| **2^28** | 472.15 ms | 333.72 ms | 235.54 ms | 220.27 ms | 0.00 ms |
| **Geometric Mean vs C** | 1 | 0.57 | 0.36 | 0.35 | 0.00 ms |
| **Speed up vs C** | - | 1.75× | 2.78x | 2.86x | 0.00 ms |
| **Boundary (n=1003)** | <0.01 ms | < 0.01 ms | < 0.01 ms | < 0.01 ms | 0.00 ms |

In Release mode, AVX2 YMM is the fastest overall, with a 2.86× speedup over C, fo
