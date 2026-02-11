#include "simd_task.h"

/**
 * 任务：使用 AVX2 指令集加速向量加法
 * 
 * 提示：
 * 1. 使用 __m256 类型表示 256 位向量（8 个 float）
 * 2. _mm256_loadu_ps: 加载 8 个 float (从非对齐内存)
 * 3. _mm256_add_ps: 向量加法
 * 4. _mm256_storeu_ps: 存储 8 个 float 到内存
 * 5. 注意处理 N 不是 8 的倍数的情况
 */
void vector_add_simd(const float* a, const float* b, float* c, int n) {
    // 请在此处实现你的代码

    // 1. 循环处理 8 个元素的倍数
    // for (int i = 0; i <= n - 8; i += 8) { ... }

    // 2. 处理剩余不足 8 个的部分
    // for (int i = (n/8)*8; i < n; i++) { ... }
}
