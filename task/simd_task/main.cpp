#include "simd_task.h"
#include <algorithm>
#include <random>

// 用户需要在这里填写的占位符
// 为了让框架“不可运行通过”，我们故意让默认实现为空或者产生错误结果
#ifndef USER_IMPLEMENTATION
void vector_add_simd(const float* a, const float* b, float* c, int n) {
    // TODO: 实现 SIMD 加法逻辑
    // 提示：使用 _mm256_loadu_ps, _mm256_add_ps, _mm256_storeu_ps
    std::cout << "\033[31m[错误] 尚未实现 vector_add_simd 函数！\033[0m" << std::endl;
}
#endif

void verify(const float* expected, const float* actual, int n) {
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (std::abs(expected[i] - actual[i]) > 1e-5) {
            if (errors < 5) {
                std::cout << "\033[31m[验证失败] 索引 " << i << ": 预期 " << expected[i] 
                          << ", 实际 " << actual[i] << "\033[0m" << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "\033[32m[恭喜] 验证通过！SIMD 实现结果正确。\033[0m" << std::endl;
    } else {
        std::cout << "\033[31m[失败] 共有 " << errors << " 个错误点。\033[0m" << std::endl;
        exit(1);
    }
}

int main() {
    const int N = 1000000; // 1M elements
    std::vector<float> a(N), b(N), c_scalar(N), c_simd(N, 0.0f);

    // 准备数据
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    std::cout << "正在运行标量基准测试..." << std::endl;
    auto start_scalar = std::chrono::high_resolution_clock::now();
    vector_add_scalar(a.data(), b.data(), c_scalar.data(), N);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_scalar = end_scalar - start_scalar;
    std::cout << "标量用时: " << diff_scalar.count() << " ms" << std::endl;

    std::cout << "\n正在运行你的 SIMD 实现..." << std::endl;
    auto start_simd = std::chrono::high_resolution_clock::now();
    vector_add_simd(a.data(), b.data(), c_simd.data(), N);
    auto end_simd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_simd = end_simd - start_simd;
    std::cout << "SIMD 用时: " << diff_simd.count() << " ms" << std::endl;

    if (diff_simd.count() > 0) {
        std::cout << "加速比: " << diff_scalar.count() / diff_simd.count() << "x" << std::endl;
    }

    verify(c_scalar.data(), c_simd.data(), N);

    return 0;
}
