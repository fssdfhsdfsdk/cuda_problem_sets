#ifndef SIMD_TASK_H
#define SIMD_TASK_H

#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <iomanip>

// 核心函数：用户需要实现这个函数
void vector_add_simd(const float* a, const float* b, float* c, int n);

// 基准参考函数：标量实现
inline void vector_add_scalar(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

#endif // SIMD_TASK_H
