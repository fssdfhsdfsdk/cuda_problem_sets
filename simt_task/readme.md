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

## 思路

1、画出 输入image的图像与数据
2、画出 Grid、Block的图，处理的数据
3、分析单个Block如何处理数据

【缺陷】这是一个很长的映射链路，没有经过训练难以走通。
 - 图像绘画 - 图像表示（数据容器、数组） - 图像数据生成逻辑 - 图像数据生成代码 - 图像数据的内存分布
 - 图像表示 - cuda划分 - 单个内部处理逻辑

【问题】为什么图像生成是 8×4块（是x=4吗？） ，而block是 block(16, 2)，x = 16，y=2 ？ 我对 图像表示（数据容器、数组） - 图像数据生成逻辑 - 图像数据生成代码 - 图像数据的内存分布 ，这一长条链路感到疑惑。


## 问题

- 生成代码有问题，放弃