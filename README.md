# CUDA Problem Sets

CUDA 并行计算练习项目集合，涵盖 CPU SIMD 与 GPU SIMT 并行优化技术。

---

## 项目结构

```
cuda_problem_sets/
├── simt_task/          # SIMT warp divergence 优化 (CUDA)
├── task/
│   └── simd_task/      # SIMD 向量计算 (AVX2)
└── .gitignore
```

---

## 项目 1: simt_task - SIMT 发散优化

### 目标
实现一个能**主动制造**和**有效消除**warp divergence的图像二值化滤波器。

### 核心概念
- **Warp**: CUDA 中 32 个线程组成的基本执行单元
- **Warp Divergence**: 当 warp 内线程执行不同分支时，导致串行化执行
- **Warp Efficiency**: 衡量 warp 内线程活跃度的指标

### 任务要求
修改 `kernel.cu` 中的 `divergent_binary_filter` 核函数：

1. **高发散模式** (`high_divergence_mode=true`)
   - 同 warp 内前16线程用阈值128，后16线程用阈值64
   - 预期 warp efficiency ≈50%
   
2. **优化模式** (`high_divergence_mode=false`)
   - 重组线程映射消除发散（例如按8×4块顺序遍历）
   - 预期 warp efficiency >95%

### 文件说明
| 文件 | 说明 |
|------|------|
| `kernel.cu` | CUDA kernel 实现（需修改） |
| `framework.cu` | 框架代码 |
| `main.cpp` | 测试主程序 |
| `image_plot.py` | 图像可视化脚本 |
| `CMakeLists.txt` | 构建配置 |

### 构建与运行

```bash
cd simt_task
mkdir -p build && cd build
cmake ..
make
./simt_task
```

### 测试图像生成
```python
python image_plot.py
```

---

## 项目 2: simd_task - SIMD 向量计算

### 背景
利用 AVX2 指令集实现向量并行计算，理解 SIMD (Single Instruction, Multiple Data) 并行技术。

现代 CPU 提供 SSE、AVX、AVX-512 等指令集。AVX2 允许一次处理 256 位数据（例如 8 个 32 位浮点数）。理解 SIMD 是学习 CUDA/OpenCL 中 SIMT 模型的重要基础。

### 任务要求
完成 `solution.cpp` 中的 `vector_add_simd` 函数：

1. **核心逻辑**：使用 `__m256` 类型和 AVX2 Intrinsics
2. **尾部处理**：处理数组长度非8倍数的情况
3. **性能对比**：比较 SIMD 与标量循环的加速比

### 核心函数
- `_mm256_loadu_ps()` - 从内存加载 8 个 float
- `_mm256_add_ps()` - 8 对 float 并行加法
- `_mm256_storeu_ps()` - 将结果存回内存

### 文件说明
| 文件 | 说明 |
|------|------|
| `solution.cpp` | 用户实现（需修改） |
| `main.cpp` | 测试主程序 |
| `simd_task.h` | 头文件 |
| `CMakeLists.txt` | 构建配置 |

### 构建与运行

```bash
cd task/simd_task
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
./simd_task
```

---

## 依赖环境

### 必需
- CMake >= 3.10
- C++11 兼容编译器

### simt_task 额外依赖
- NVIDIA GPU
- CUDA Toolkit

### simd_task 额外依赖
- 支持 AVX2 的 CPU (Intel Haswell+ / AMD Zen+)

---

## 学习路径

1. **simd_task** → 理解 CPU 并行基础 (SIMD)
2. **simt_task** → 理解 GPU 并行特性 (SIMT)

SIMT 与 SIMD 的关键区别：
- **SIMD**: 一条指令同时处理多组数据
- **SIMT**: 多条线程执行相同指令流，支持分支

---

## 参考资源

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

---

## License

学习用途，自由使用。
