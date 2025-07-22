import numpy as np
import matplotlib.pyplot as plt

def generate_projection_data(diameter_cm=60, num_angles=36, num_detectors=50, num_voxels=128):
    """
    生成桶的投影数据
    
    参数:
        diameter_cm: 桶直径 (cm)
        num_angles: 投影角度数量
        num_detectors: 每个角度的探测器数量
        num_voxels: 体素网格尺寸
        
    返回:
        projections: 投影数据数组 (num_angles * num_detectors,)
    """
    # 1. 计算几何参数
    radius_cm = diameter_cm / 2.0
    voxel_size_cm = diameter_cm / num_voxels
    
    # 2. 创建探测器位置 (沿对称轴均匀分布)
    s_values = np.linspace(-radius_cm, radius_cm, num_detectors)
    
    # 3. 初始化投影数据数组
    projections = np.zeros(num_angles * num_detectors)
    
    # 4. 对于每个角度生成投影
    for angle_idx in range(num_angles):
        # 当前角度（弧度）
        theta = np.deg2rad(angle_idx*10)
        
        # 角度对应的余弦和正弦值
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 对于每个探测器
        for det_idx in range(num_detectors):
            # 当前探测器位置 (s)
            s = s_values[det_idx]
            
            # 计算射线参数方程: x*cos(theta) + y*sin(theta) = s
            # 计算射线与桶边界的交点
            
            # 5. 计算射线与桶的交点（弦长）
            # 射线到圆心的距离为 |s|
            dist_to_center = abs(s)
            
            # 如果射线在桶外，投影值为0
            if dist_to_center > radius_cm:
                projection_value = 0.0
            else:
                # 计算弦长 (射线穿过桶的长度)
                chord_length = 2 * np.sqrt(radius_cm**2 - dist_to_center**2)
                
                # 因为桶内衰减系数为1，桶外为0
                # 投影值 = 衰减系数 × 路径长度 = 1 × chord_length
                projection_value = chord_length
            
            # 存储投影值
            proj_idx = angle_idx * num_detectors + det_idx
            projections[proj_idx] = projection_value
    
    return projections

def visualize_projection_data(projections, num_angles=36, num_detectors=50):
    """
    可视化投影数据
    """
    # 重塑为角度×探测器的矩阵
    sinogram = projections.reshape((num_angles, num_detectors))
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    plt.imshow(sinogram, cmap='gray', 
               extent=[-30, 30, 0, 360], 
               aspect='auto')
    
    plt.colorbar(label='')
    plt.xlabel('direction')
    plt.ylabel('angle')
    plt.title('Sinogram')
    plt.show()
    
    return sinogram

# 主程序
if __name__ == "__main__":
    # 生成投影数据
    projections = generate_projection_data(
        diameter_cm=60,
        num_angles=36,
        num_detectors=50,
        num_voxels=128
    )
    
    print(f"生成的投影数据形状: {projections.shape}")
    print(f"最小投影值: {np.min(projections):.2f}, 最大投影值: {np.max(projections):.2f}")
    
    # 可视化投影数据
    sinogram = visualize_projection_data(projections)
    
    # 保存投影数据
    np.save("projection_data/bucket_projections.npy", projections)
    print("投影数据已保存为 'projection_data/bucket_projections.npy'")