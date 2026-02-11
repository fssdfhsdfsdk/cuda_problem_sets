#include "simt_task.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "\n❌ CUDA错误: " << cudaGetErrorString(err) \
                  << "\n   位置: " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 生成特殊测试图像：8×4块内前16像素=100(暗), 后16像素=200(亮)
void generate_test_image(uint8_t* img, int width) {
    for (int by = 0; by < width; by += 4) {
        for (int bx = 0; bx < width; bx += 8) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 8; x++) {
                    int idx = (by + y) * width + (bx + x);
                    // 块内前16像素(0-15) = 100, 后16像素(16-31) = 200
                    img[idx] = (y * 8 + x < 16) ? 100 : 200;
                }
            }
        }
    }
}

// 验证输出正确性
bool verify(const uint8_t* output, int width, bool high_div_mode) {
    for (int by = 0; by < width; by += 4) {
        for (int bx = 0; bx < width; bx += 8) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 8; x++) {
                    int idx = (by + y) * width + (bx + x);
                    uint8_t expected;
                    
                    if (high_div_mode) {
                        // 高发散模式：块内前16像素阈值128→输出0, 后16阈值64→输出255
                        expected = (y * 8 + x < 16) ? 0 : 255;
                    } else {
                        // 优化模式：统一阈值128 → 前16输出0, 后16输出255
                        expected = (y * 8 + x < 16) ? 0 : 255;
                    }
                    
                    if (output[idx] != expected) {
                        std::cerr << "\n❌ 验证失败!\n"
                                  << "  位置: 块(" << bx/8 << "," << by/4 << ") 内像素(" << x << "," << y << ")\n"
                                  << "  期望: " << (int)expected << " (阈值=" 
                                  << (high_div_mode && (y*8+x<16) ? 128 : 64) << ")\n"
                                  << "  实际: " << (int)output[idx] << "\n"
                                  << "  提示: 检查warp内线程的阈值分配逻辑!\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    const int WIDTH = 512;
    std::vector<uint8_t> input(WIDTH * WIDTH);
    std::vector<uint8_t> output(WIDTH * WIDTH);
    
    generate_test_image(input.data(), WIDTH);
    std::cout << "🔬 SIMT发散优化实验 (512x512图像)\n";
    std::cout << "   测试图像: 8x4块内前16像素=100(暗), 后16像素=200(亮)\n\n";
    
    // ===== 测试1: 高发散模式 =====
    std::cout << "【测试1】制造warp divergence...\n";
    run_filter(input.data(), output.data(), WIDTH, true);
    
    if (!verify(output.data(), WIDTH, true)) {
        std::cerr << "   → 未通过! 请修正kernel.cu中的高发散逻辑\n";
        return 1;
    }
    std::cout << "   ✅ 通过! warp execution efficiency: " 
              << (int)(get_warp_efficiency() * 100) << "% (预期≈50%)\n";
    
    // ===== 测试2: 优化模式 =====
    std::cout << "\n【测试2】消除warp divergence...\n";
    run_filter(input.data(), output.data(), WIDTH, false);
    
    if (!verify(output.data(), WIDTH, false)) {
        std::cerr << "   → 未通过! 请修正kernel.cu中的优化逻辑\n";
        return 1;
    }
    std::cout << "   ✅ 通过! warp execution efficiency: " 
              << (int)(get_warp_efficiency() * 100) << "% (预期>95%)\n";
    
    // ===== 最终验证 =====
    float div_eff = get_warp_efficiency(); // 模拟高发散效率
    float opt_eff = get_warp_efficiency(); // 模拟优化后效率
    
    if (opt_eff < 0.95f) {
        std::cerr << "\n⚠️  优化效果不足! warp efficiency=" 
                  << (int)(opt_eff*100) << "% (<95%)\n"
                  << "   → 检查线程映射是否真正消除了发散\n";
        return 1;
    }
    
    if (opt_eff < div_eff * 1.8f) {
        std::cerr << "\n⚠️  优化幅度不足! 效率提升<80%\n"
                  << "   高发散: " << (int)(div_eff*100) << "% → 优化后: " 
                  << (int)(opt_eff*100) << "%\n";
        return 1;
    }
    
    std::cout << "\n🎉 恭喜! 你已掌握SIMT核心优化技巧:\n"
              << "   • 识别warp divergence根源\n"
              << "   • 通过数据布局消除发散\n"
              << "   • 实现>95%的warp efficiency\n";
    return 0;
}