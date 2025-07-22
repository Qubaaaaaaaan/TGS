import numpy as np
from scipy.sparse import lil_matrix, load_npz
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from TV import compute_tv_gradient, tv_regularization_step

A_csr = load_npz('system_matrix_36x50_128x128.npz')
print('A_matrix has been constructed')


import numpy as np
from scipy.sparse import csr_matrix

# 预计算桶约束掩膜（内切圆）
indices = np.arange(128*128)
x_coords = indices // 128  # 行坐标
y_coords = indices % 128   # 列坐标
center = 63.5  # 网格中心 (256-1)/2 = 63.5
barrelR_sq = 64.0**2  # 内切圆半径平方 (128/2)^2
dist_sq = (x_coords - center)**2 + (y_coords - center)**2
mask = dist_sq <= barrelR_sq  # True表示在桶内

P = np.load('complex_projections.npy')
X = np.zeros(shape=128*128, dtype=np.float32)

for n in range(50):
    X_before_art = X.copy()
    
    for i in range(36*20):
        row_vector = A_csr.getrow(i)
        
        # 提取稀疏矩阵的非零元素索引和值
        cols = row_vector.indices
        vals = row_vector.data
        
        # 检查是否有非零元素
        if len(cols) > 0:
            # 计算当前投影值
            current_proj = np.dot(vals, X[cols])
            
            # 计算分母 (避免除零)
            denom = np.dot(vals, vals) + 1e-8
            
            # ART迭代更新 - 只更新非零元素位置
            update = 0.7 * (P[i] - current_proj) * vals / denom
            X[cols] += update
            
            # 应用约束：只更新非零元素位置
            # 非负约束 (3-11)
            negative_mask = X[cols] < 0
            X[cols[negative_mask]] = 0
            
            # 桶约束 (3-12) - 只处理当前行影响的像素
            barrel_mask = ~mask[cols]
            X[cols[barrel_mask]] = 0
    
    X_after_art = X.copy()
    d_A = np.linalg.norm(X_after_art - X_before_art)
    
    # TV正则化步骤
    X_refined = tv_regularization_step(X_before_art, X_after_art, 0.7, 10)
    X = X_refined
    
    # TV正则化后应用全局约束
    X[X < 0] = 0              # 非负约束
    X[~mask] = 0              # 桶约束

# 最终结果重塑和保存
X = X.reshape((128, 128))
np.save('result_1.npy', X)