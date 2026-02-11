# SIMD 基础：利用 AVX2 加速向量计算

## 背景
在高性能计算领域，**SIMD** (Single Instruction, Multiple Data) 是一种极具威力的并行技术。它允许 CPU 用单条指令同时对一组数据（通常是 4、8 或 16 个）进行相同的操作。

现代 CPU 提供了如 SSE、AVX、AVX-512 等指令集。其中 **AVX2** 允许我们一次处理 256 位数据（例如 8 个 32 位浮点数）。如果你之后要学习 CUDA 或 OpenCL，理解 SIMD 是非常重要的第一步，因为 GPU 的 **SIMT** (Single Instruction, Multiple Threads) 模型在很多概念上与 SIMD 是相通的。

## 任务要求
你的目标是完成 `solution.cpp` 中的 `vector_add_simd` 函数，利用 AVX2 指令集实现两个 float 数组的加法。

1. **核心逻辑**：使用 `__m256` 类型和相关的 Intrinsics 函数。
2. **尾部处理**：当数组长度 `N` 不是 8 的倍数时，确保剩余的元素也能被正确计算（通常使用标量循环处理）。
3. **性能对比**：观察 SIMD 实现相比于传统的 `for` 循环（标量实现）有多少加速比。

## 附加挑战
1. **内存对齐**：尝试研究 `_mm256_load_ps`（要求 32 字节对齐）与 `_mm256_loadu_ps`（不要求对齐）的区别。在高性能场景下，对齐往往能带来更好的吞吐量。
2. **更多操作**：尝试在 `solution.cpp` 中增加一个向量乘法或融合乘加（FMA）的实现。

## 考察点
- `immintrin.h` 头文件的基本使用。
- `__m256` 数据的加载（Load）、运算（Add）、存储（Store）流程。
- 处理向量化循环中的边界问题。
- 编译选项（如 `-mavx2` 或 `/arch:AVX2`）的作用。

## 提示
以下是你在实现中可能用到的核心函数：
- `__m256 _mm256_loadu_ps(float const * mem_addr)`：从内存加载 8 个 float 到寄存器。
- `__m256 _mm256_add_ps(__m256 a, __m256 b)`：执行 8 对 float 的并行加法。
- `void _mm256_storeu_ps(float * mem_addr, __m256 a)`：将寄存器中的 8 个 float 存回内存。

### 编译与运行
在 `task/simd_task` 目录下：
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
./simd_task  # Windows 下可能是 .\Release\simd_task.exe
```
