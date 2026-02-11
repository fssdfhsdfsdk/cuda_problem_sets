#include "simd_matmul_task.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

// 计时辅助宏
#define TIMEIT(name, code) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
        std::cout << "   " << name << ": " << duration.count() / 1000.0 << " ms\n"; \
    } while(0)

// 标量矩阵乘法（参考实现）
void matmul_scalar(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    // 初始化 C 为 0
    std::memset(C, 0, M * N * sizeof(float));
    
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

// 转置 B 矩阵
void transpose_b(const float* B, float* B_T, int K, int N) {
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B_T[n * K + k] = B[k * N + n];
        }
    }
}

// 验证结果正确性
bool verify_result(const float* C_test, const float* C_ref, int M, int N, float tolerance = 1e-4f) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float diff = std::abs(C_test[i * N + j] - C_ref[i * N + j]);
            if (diff > tolerance) {
                std::cerr << "\n❌ 验证失败!\n"
                          << "   位置: (" << i << ", " << j << ")\n"
                          << "   期望值: " << C_ref[i * N + j] << "\n"
                          << "   实际值: " << C_test[i * N + j] << "\n"
                          << "   误差: " << diff << "\n";
                return false;
            }
        }
    }
    return true;
}

// 随机初始化矩阵
void random_init(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    std::cout << "🔬 SIMD 矩阵乘法实验\n\n";
    
    // 测试矩阵维度
    // 使用 512×512 以获得可测量的时间
    // 使用非8的倍数（如 511）测试边界处理
    const int test_sizes[] = {64, 128, 256, 511, 512};
    const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    std::cout << "【测试】不同尺寸矩阵的乘法性能\n\n";
    
    for (int t = 0; t < num_tests; t++) {
        int M = test_sizes[t];
        int N = test_sizes[t];
        int K = test_sizes[t];
        
        std::cout << "矩阵尺寸: " << M << "×" << K << " * " << K << "×" << N << "\n";
        
        // 分配内存
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C_scalar(M * N);
        std::vector<float> C_simd(M * N);
        std::vector<float> C_simd_t(M * N);
        
        // 初始化随机数据
        random_init(A.data(), M * K);
        random_init(B.data(), K * N);
        
        // 1. 标量实现（参考）
        TIMEIT("标量实现", matmul_scalar(A.data(), B.data(), C_scalar.data(), M, N, K));
        
        // 2. SIMD 实现（用户完成）
        #ifdef USER_IMPLEMENTATION
        TIMEIT("SIMD 实现", matmul_simd(A.data(), B.data(), C_simd.data(), M, N, K));
        
        // 验证 SIMD 结果
        if (!verify_result(C_simd.data(), C_scalar.data(), M, N)) {
            std::cerr << "   → SIMD 实现未通过验证!\n\n";
            return 1;
        }
        std::cout << "   ✅ SIMD 实现通过验证\n";
        
        // 3. SIMD + 转置优化（进阶）
        TIMEIT("SIMD+转置", matmul_simd_transpose(A.data(), B.data(), C_simd_t.data(), M, N, K));
        
        // 验证转置优化结果
        if (!verify_result(C_simd_t.data(), C_scalar.data(), M, N)) {
            std::cerr << "   → SIMD+转置实现未通过验证!\n\n";
            return 1;
        }
        std::cout << "   ✅ SIMD+转置通过验证\n";
        #else
        std::cout << "   (等待用户实现...)\n";
        #endif
        
        std::cout << "\n";
    }
    
    #ifdef USER_IMPLEMENTATION
    std::cout << "🎉 恭喜! 你已掌握 SIMD 矩阵乘法优化:\n"
              << "   • 二维数据的向量化处理\n"
              << "   • FMA 乘加融合指令\n"
              << "   • 转置优化提升内存连续性\n"
              << "   • 边界情况处理\n";
    #endif
    
    return 0;
}
