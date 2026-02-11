import numpy as np
import matplotlib.pyplot as plt

def generate_test_image(width=32):
    """
    生成特殊测试图像：每个8×4块内前16像素=100(暗), 后16像素=200(亮)
    """
    img = np.zeros((width, width), dtype=np.uint8)
    
    for by in range(0, width, 4):      # 块垂直步长4（4行）
        for bx in range(0, width, 8):  # 块水平步长8（8列）
            for y in range(4):         # 块内4行
                for x in range(8):     # 块内8列
                    # 计算全局坐标
                    global_y = by + y
                    global_x = bx + x
                    # 块内前16像素(0-15) = 100, 后16像素(16-31) = 200
                    # 块内索引 = y * 8 + x
                    img[global_y, global_x] = 100 if (y * 8 + x < 16) else 200
    return img

# 生成测试图像
width = 64
test_image = generate_test_image(width)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 主图像显示
axes[0].imshow(test_image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title(f'Generated Test Image ({width}×{width})\n8×4 blocks: top=100 (dark), bottom=200 (bright)', fontsize=12)
axes[0].axis('off')
axes[0].set_aspect('equal')

# 局部放大显示（展示单个8×4块结构）
axes[1].imshow(test_image[:4, :8], cmap='gray', vmin=0, vmax=255, interpolation='none')
axes[1].set_title('Single 8×4 Block Detail\nTop 2 rows: 100 | Bottom 2 rows: 200', fontsize=11)
axes[1].set_xticks(np.arange(8))
axes[1].set_yticks(np.arange(4))
axes[1].grid(color='red', linestyle='--', linewidth=1, alpha=0.7)
for i in range(4):
    for j in range(8):
        val = test_image[i, j]
        axes[1].text(j, i, str(val), ha='center', va='center', 
                    color='white' if val < 150 else 'black', fontsize=8, weight='bold')

plt.tight_layout()
# plt.savefig('test_image_pattern.png', dpi=150, bbox_inches='tight')
plt.show()

# 验证图像统计信息
print(f"Image shape: {test_image.shape}")
print(f"Unique values: {np.unique(test_image)}")
print(f"Value 100 count: {np.sum(test_image == 100)}")
print(f"Value 200 count: {np.sum(test_image == 200)}")
print(f"Pattern verification (top-left block):")
print(test_image[:4, :8])