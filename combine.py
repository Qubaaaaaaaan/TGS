import numpy as np
from scipy.sparse import vstack, load_npz, save_npz

# 参数设置
num_angles = 36
voxel_num = 128
voxel_count = voxel_num * voxel_num
num_detectors = 20

# 初始化列表存储所有稀疏矩阵
sparse_matrices = []

# 遍历所有角度文件
for angle_deg in range(num_angles):
    # 加载当前角度的效率矩阵
    fname = f"eff_1/efficiency_angle_{angle_deg:03d}.npz"
    sparse_mat = load_npz(fname)
    sparse_matrices.append(sparse_mat)
    print(f"已加载角度 {angle_deg} 的数据")

# 垂直堆叠所有稀疏矩阵
combined_matrix = vstack(sparse_matrices)

# 检查最终形状
expected_shape = (num_angles * num_detectors, voxel_count)
if combined_matrix.shape != expected_shape:
    print(f"警告：合并后的矩阵形状为 {combined_matrix.shape}，预期为 {expected_shape}")
else:
    print(f"成功创建合并矩阵，形状为 {combined_matrix.shape}")

# 保存合并后的矩阵
output_file = "combined_efficiency_matrix.npz"
save_npz(output_file, combined_matrix)
print(f"合并后的矩阵已保存至 {output_file}")