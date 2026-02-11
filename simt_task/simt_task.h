#ifndef SIMT_TASK_H
#define SIMT_TASK_H

#include <cstdint>

// 用户必须实现的核函数（不可修改函数签名）
__global__ void divergent_binary_filter(
    const uint8_t* input, 
    uint8_t* output, 
    int width,
    bool high_divergence_mode  // true=制造发散, false=消除发散
);

// 框架内部使用（用户无需关心）
void run_filter(const uint8_t* input_host, uint8_t* output_host, int width, bool mode);
float get_warp_efficiency(); // 模拟性能测量

#endif