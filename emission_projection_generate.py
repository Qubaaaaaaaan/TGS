import numpy as np
from scipy.sparse import load_npz, csr_matrix
import os

# 定义网格参数
size = 128
length = 600.0
delta = length / size

# 目标物体参数
target_center = (-140, -170)
target_radius = 70

# 初始化活度矩阵
activity_matrix = np.zeros((size, size))

# 生成活度矩阵
for i in range(size):
    for j in range(size):
        # 计算物理坐标 (x, y)
        x = -300 + (j + 0.5) * delta
        y = 300 - (i + 0.5) * delta
        
        # 计算与目标中心的距离
        dx = x - target_center[0]
        dy = y - target_center[1]
        distance_sq = dx**2 + dy**2
        
        # 判断是否在目标区域内 (避免sqrt计算优化性能)
        if distance_sq <= target_radius**2:
            activity_matrix[i, j] = 1.0
activity_matrix = activity_matrix.T
# 展平活度矩阵为向量 (16384,)
activity_vector = activity_matrix.flatten()

# 加载投影矩阵 (假设文件名为 'projection_matrix.npz')
# 实际使用时替换为您的投影矩阵文件路径
projection_matrix = load_npz('TGS/system_matrix_emission.npz')  # 形状应为 (720, 16384)

# 矩阵乘法: (720, 16384) × (16384,) = (720,)
result_vector = projection_matrix.dot(activity_vector)

# 保存结果
np.save('projection_data/emission_projection.npy', result_vector)           # 保存投影结果向量

# 同时保存文本格式以便检查

print("计算完成！结果已保存到 results/ 目录")
print(f"活度矩阵形状: {activity_matrix.shape}")
print(f"活度向量形状: {activity_vector.shape}")
print(f"投影结果形状: {result_vector.shape}")
print(f"非零体素数: {np.sum(activity_vector > 0)}")
print(f"投影结果范围: [{np.min(result_vector):.4f}, {np.max(result_vector):.4f}]")