# SIMD 进阶：矩阵乘法向量化

## 背景

矩阵乘法（GEMM）是科学计算、机器学习和图形渲染中的核心操作。在 CPU 上优化矩阵乘法，**内存访问模式**和**向量化**是关键。

本任务中，你将实现一个 AVX2 加速的矩阵乘法，学习如何处理：
- 二维数据的向量化加载
- 非连续内存访问的优化策略
- FMA（乘加融合）指令的使用

## 前置知识

完成本任务前，请确保你已掌握：
- `simd_task` 中的向量加法实现
- `_mm256_loadu_ps`, `_mm256_add_ps`, `_mm256_storeu_ps` 的基本用法

## 任务要求

完成 `solution.cpp` 中的 `matmul_simd` 函数，实现矩阵乘法 $C = A \times B$。

### 矩阵布局
- 所有矩阵均采用**行优先**存储（Row-major）
- 矩阵维度：$A[M \times K]$, $B[K \times N]$, $C[M \times N]$
- 元素类型：`float`

### 实现要点

1. **核心计算**
   - 使用 `__m256` 一次处理 8 个 float
   - 使用 FMA 指令 `_mm256_fmadd_ps` 实现乘加融合（一条指令完成 `a*b + c`）
   - 注意：FMA 比单独的 mul + add 更快且精度更高

2. **访问模式优化**
   - **难点**：B 矩阵按列访问时不连续（stride = K）
   - **策略**：
     - 方案 A：直接处理（实现简单但可能有缓存问题）
     - 方案 B：预转置 B 矩阵（转置后按行访问，连续性好）
   - 尝试两种方案，比较性能差异

3. **边界处理**
   - N 可能不是 8 的倍数，需用标量循环处理尾部

### 核心函数参考

```cpp
// 乘加融合：dst = a * b + c
__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);

// 加载 8 个相同的值（用于广播 A 矩阵的元素）
__m256 _mm256_broadcast_ss(const float* mem_addr);

// 加载 8 个连续 float
__m256 _mm256_loadu_ps(const float* mem_addr);

// 存储 8 个 float
void _mm256_storeu_ps(float* mem_addr, __m256 a);
```

## 进阶挑战

1. **转置优化**：实现 `matmul_simd_transpose`，先转置 B 矩阵再计算，对比性能
2. **分块优化（Tiling）**：使用 64×64 或 128×128 的分块，提高 L1/L2 缓存命中率
3. **内存对齐**：尝试 `_mm256_load_ps`（要求 32 字节对齐）对比性能

## 考察点

- 理解行优先存储的内存布局
- 非连续内存访问的性能影响
- FMA 指令的正确使用
- 缓存友好的代码组织

## 性能目标

对于 512×512 的矩阵乘法：
- 朴素 SIMD 实现应比标量实现快 **3-4 倍**
- 转置优化版本应比朴素 SIMD 快 **1.5-2 倍**
- 分块优化版本应进一步提升 **20-50%**

## 编译与运行

```bash
cd task/simd_matmul_task
mkdir build
cd build
cmake ..
cmake --build . --config Release
./Release/simd_matmul_task
```

## 提示

### 矩阵乘法伪代码（标量版）
```cpp
for i in 0..M-1:
  for j in 0..N-1:
    float sum = 0;
    for k in 0..K-1:
      sum += A[i*K + k] * B[k*N + j];
    C[i*N + j] = sum;
```

### SIMD 向量化思路
最内层循环（k）可以展开为每次处理 8 个：
```cpp
// 加载 A 的 8 个连续元素
__m256 a = _mm256_loadu_ps(&A[i*K + k]);

// 加载 B 的对应 8 个元素（注意：B 是列访问，不连续！）
// 这需要特殊处理...

// 累加到 C 的向量
__m256 c = _mm256_fmadd_ps(a, b, c);
```

### 推荐实现策略
1. **先实现转置版本**（更容易获得好性能）：
   - 先转置 B 得到 B^T
   - 计算 C[i][j] = dot(A 的第 i 行, B^T 的第 j 行)
   - 此时两个都是行访问，连续性好

2. **再尝试直接版本**：
   - 使用 `_mm256_broadcast_ss` 广播 A 的元素
   -  gather 或手动加载 B 的列元素

## 参考公式

**矩阵乘法定义**：
$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj}$$

**转置后**：
$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B^T_{jk}$$
（注意 $B^T$ 的索引是 $[j][k]$ 而不是 $[k][j]$）
