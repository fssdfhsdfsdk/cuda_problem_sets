#include "simd_matmul_task.h"
#include <immintrin.h>
#include <cstring>

// ============================================
// 任务：实现 SIMD 优化的矩阵乘法
// 
// 提示：
// 1. 使用 __m256 类型处理 8 个 float
// 2. 使用 _mm256_fmadd_ps 实现乘加融合
// 3. 注意 B 矩阵的访问模式（列访问不连续）
// 4. 处理 N 不是 8 的倍数的情况
// ============================================

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    // TODO: 实现 SIMD 矩阵乘法
    // 
    // 参考思路：
    // 1. 初始化 C 为 0
    // 2. 外层循环遍历 M（A 的行）
    // 3. 中层循环遍历 K（A 的列 / B 的行）
    // 4. 内层循环遍历 N（B 的列），每次处理 8 个
    //
    // 对于 B 的列访问，可以考虑：
    // - 方案 1：使用 _mm256_broadcast_ss 广播 A[i][k]，然后逐列加载 B[k][j:j+8]
    // - 方案 2：预转置 B，然后按行访问
    
    // 临时实现：直接调用标量版本（请删除并替换为 SIMD 实现）
    std::memset(const_cast<float*>(C), 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = A[i * K + k];
            // 每次处理 8 个元素
            int j = 0;
            for (; j <= N - 8; j += 8) {
                // TODO: 加载 B[k][j:j+8] 到向量
                // TODO: 广播 a_val
                // TODO: 使用 FMA 累加到 C[i][j:j+8]
            }
            // 处理剩余元素
            for (; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

void matmul_simd_transpose(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    // TODO: 实现带转置优化的 SIMD 矩阵乘法
    // 
    // 步骤：
    // 1. 分配临时内存存储转置后的 B^T（大小 N×K）
    // 2. 转置 B：B^T[n][k] = B[k][n]
    // 3. 计算 C[i][j] = dot(A 的第 i 行, B^T 的第 j 行)
    //
    // 提示：转置后，B^T 的行访问是连续的，SIMD 加载更高效
    
    // 临时实现：直接调用 matmul_simd（请删除并替换为转置优化版本）
    matmul_simd(A, B, C, M, N, K);
}

