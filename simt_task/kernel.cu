#include "simt_task.h"

__global__ void divergent_binary_filter(
    const uint8_t* input, 
    uint8_t* output, 
    int width,
    bool high_divergence_mode)
{
    // ===== 【用户任务】修正以下代码 =====
    // 当前实现存在两个缺陷：
    // 缺陷1: 未根据high_divergence_mode区分逻辑
    // 缺陷2: 未按warp内位置制造/消除发散
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= width) return;
    
    int idx = y * width + x;
    
    // FIXME: 此实现对所有线程使用相同阈值 → 无法制造发散
    uint8_t threshold = 128; 
    output[idx] = (input[idx] > threshold) ? 255 : 0;
    
    // ===== 修正要求 =====
    // 高发散模式 (high_divergence_mode=true):
    //   - warp内前16线程用阈值128，后16线程用阈值64
    //   - 提示：计算线程在warp中的相对位置 (threadIdx.x + threadIdx.y*blockDim.x) % 32
    //
    // 优化模式 (high_divergence_mode=false):
    //   - 重组线程映射使同warp内线程处理"同类阈值"像素
    //   - 提示：修改x/y计算方式，例如按8×4块顺序遍历
    // ===================================
}