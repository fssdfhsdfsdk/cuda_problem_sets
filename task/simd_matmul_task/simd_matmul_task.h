#ifndef SIMD_MATMUL_TASK_H
#define SIMD_MATMUL_TASK_H

// 矩阵乘法接口声明
// 用户需要实现这些函数

// 标量实现的矩阵乘法（参考实现）
void matmul_scalar(const float* A, const float* B, float* C,
                   int M, int N, int K);

// SIMD 优化的矩阵乘法（用户实现）
// A: [M x K], B: [K x N], C: [M x N]
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K);

// 带转置优化的矩阵乘法（用户实现，进阶）
// 先转置 B 再计算
void matmul_simd_transpose(const float* A, const float* B, float* C,
                           int M, int N, int K);

#endif // SIMD_MATMUL_TASK_H
