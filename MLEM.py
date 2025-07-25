import numpy as np
from scipy.sparse import lil_matrix, load_npz
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from TV import compute_tv_gradient, tv_regularization_step

A_csr = load_npz('system_matrix_emission.npz')
print('A_matrix has been constructed')

# 预计算桶约束掩膜（内切圆）
indices = np.arange(128*128)
x_coords = indices // 128  # 行坐标
y_coords = indices % 128   # 列坐标
center = 63.5  # 网格中心 (128-1)/2 = 63.5
barrelR_sq = 64.0**2  # 内切圆半径平方 (128/2)^2
dist_sq = (x_coords - center)**2 + (y_coords - center)**2
mask = dist_sq <= barrelR_sq  # True表示在桶内

P = np.load('../emission_projection.npy')

# 关键修改：初始化桶内区域为1.0，避免全零初始化
X = np.zeros(shape=128*128, dtype=np.float32)
X[mask] = 1.0  # 桶内初始化为1.0

# 预计算灵敏度因子 (分母项)
s = np.array(A_csr.sum(axis=0)).flatten()  # 每个像素的灵敏度因子 s_j = Σ_i a_{ij}
s[s == 0] = 1e-10  # 避免除零错误

for n in range(100):
    X_before = X.copy()
    
    Ax = A_csr.dot(X)
    
    # 避免除零错误 (将0元素替换为小值)
    Ax = np.maximum(Ax, 1e-8)
    
    # 计算比值项: y_i / (Ax)_i
    ratio = P / Ax
    
    # 反投影: A^T * ratio = Σ_i a_{ij} * (y_i / (Ax)_i)
    backproj = A_csr.T.dot(ratio)
    
    # 更新图像: x_j = (x_j / s_j) * backproj_j
    X = (X / s) * backproj
    
    # 应用约束: 非负约束和桶约束
    X[X < 0] = 1e-10              # 非负约束
    X[~mask] = 1e-10              # 桶约束
    
    # ============= TV 正则化步骤 =============
    # 注意：仅在非零区域应用TV
    if np.any(X > 0):
        X_refined = tv_regularization_step(X_before, X, 0.1, 10)
        X = X_refined
        # TV后再次应用约束
        X[X < 0] = 1e-10
        X[~mask] = 1e-10

# 最终结果重塑和保存
X = X.reshape((128, 128))
np.save('emission_result.npy', X)