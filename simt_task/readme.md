# SIMT发散优化实战

## 任务目标
实现一个能**主动制造**和**有效消除**warp divergence的图像二值化滤波器。

## 用户任务
仅需修改 `kernel.cu` 中的 `divergent_binary_filter` 核函数：
1. **高发散模式** (`high_divergence_mode=true`)
   - 同warp内前16线程用阈值128，后16线程用阈值64
   - 预期warp efficiency ≈50%
   
2. **优化模式** (`high_divergence_mode=false`)
   - 重组线程映射消除发散（例如按8×4块顺序遍历）
   - 预期warp efficiency >95%

## 测试流程
```bash
mkdir build && cd build
cmake .. && make
./simt_task  # 初始运行会失败，修正kernel.cu后通过
```
