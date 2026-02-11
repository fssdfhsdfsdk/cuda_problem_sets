#include "simt_task.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// 模拟性能测量（实际项目中替换为Nsight Compute）
static float simulated_efficiency = 0.52f;
float get_warp_efficiency() { 
    return simulated_efficiency; 
}

void run_filter(const uint8_t* input_host, uint8_t* output_host, int width, bool mode) {
    uint8_t *d_input, *d_output;
    size_t size = width * width * sizeof(uint8_t);
    
    // 分配设备内存
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input_host, size, cudaMemcpyHostToDevice);
    
    // 关键配置：16x2线程块 = 32线程 = 1个warp
    dim3 block(16, 2); 
    dim3 grid((width + block.x - 1) / block.x, 
              (width + block.y - 1) / block.y);
    
    // 调用用户核函数
    divergent_binary_filter<<<grid, block>>>(d_input, d_output, width, mode);
    
    // 检测用户是否实现核函数
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\n❌ 核函数未正确实现!\n"
                  << "   错误: " << cudaGetErrorString(err) << "\n"
                  << "   → 请检查kernel.cu是否包含有效实现\n";
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceSynchronize();
    
    // 模拟性能差异
    simulated_efficiency = mode ? 0.52f : 0.96f; 
    
    cudaMemcpy(output_host, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}