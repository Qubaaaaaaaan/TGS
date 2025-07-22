import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def generate_complex_projection_data(diameter_mm=600, num_angles=36, num_detectors=50):
    """
    生成包含多个圆形物体的投影数据
    
    参数:
        diameter_mm: 桶直径 (mm)
        num_angles: 投影角度数量 (36个)
        num_detectors: 每个角度的探测器数量 (50个)
        
    返回:
        projections: 投影数据数组 (num_angles * num_detectors,)
        objects: 物体信息列表
    """
    # 1. 计算几何参数
    radius_mm = diameter_mm / 2.0
    
    # 2. 创建探测器位置 (沿对称轴均匀分布)
    s_values = np.linspace(-radius_mm, radius_mm, num_detectors)
    
    # 3. 初始化投影数据数组
    projections = np.zeros(num_angles * num_detectors)
    
    # 4. 定义桶内物体 (中心坐标(mm), 半径(mm), 衰减系数)
    # 根据128x128图像网格调整物体尺寸和位置
    objects = [
        # 大物体
        {'center': (0, 0), 'radius': 100, 'mu': 0.8},
        
        # 中型物体
        {'center': (150, 130), 'radius': 60, 'mu': 1.2},
        {'center': (-140, -170), 'radius': 70, 'mu': 0.9},
        
        # 小型物体
        {'center': (-180, 150), 'radius': 30, 'mu': 1.5},
        {'center': (170, -100), 'radius': 40, 'mu': 1.1},
        {'center': (0, 200), 'radius': 25, 'mu': 1.4},
    ]
    
    # 5. 对于每个角度生成投影
    for angle_idx in range(num_angles):
        # 当前角度（弧度）
        theta = angle_idx * (2 * np.pi / num_angles)  # 均匀分布的角度
        
        # 角度对应的余弦和正弦值
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 对于每个探测器
        for det_idx in range(num_detectors):
            # 当前探测器位置 (s)
            s = s_values[det_idx]
            
            # 计算射线与桶边界的交点（弦长）
            dist_to_center = abs(s)
            projection_value = 0.0
            
            # 如果射线在桶内
            if dist_to_center <= radius_mm:
                # 计算射线穿过桶的路径长度
                chord_length = 2 * np.sqrt(radius_mm**2 - dist_to_center**2)
                
                # 初始投影值（桶内背景衰减系数设为1.0）
                projection_value = 0.5 * chord_length
                
                # 计算射线与每个物体的交点
                for obj in objects:
                    # 计算物体中心到射线的距离
                    x0, y0 = obj['center']
                    d_obj = abs(x0 * cos_theta + y0 * sin_theta - s)
                    
                    # 如果射线穿过物体
                    if d_obj <= obj['radius']:
                        # 计算射线穿过物体的路径长度
                        obj_chord_length = 2 * np.sqrt(obj['radius']**2 - d_obj**2)
                        
                        # 减去背景值并加上物体衰减值
                        projection_value += (obj['mu'] - 0.5) * obj_chord_length
            
            # 存储投影值
            proj_idx = angle_idx * num_detectors + det_idx
            projections[proj_idx] = projection_value
    
    return projections, objects

def visualize_phantom(objects, diameter_mm=600):
    """
    可视化桶内物体分布
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制桶
    bucket = plt.Circle((0, 0), diameter_mm/2, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(bucket)
    
    # 绘制物体
    for i, obj in enumerate(objects):
        circle = plt.Circle(obj['center'], obj['radius'], 
                           fill=True, alpha=0.7,
                           label=f"Obj {i+1}: μ={obj['mu']}, r={obj['radius']}mm")
        ax.add_patch(circle)
    
    # 设置图形属性
    ax.set_xlim(-diameter_mm/2-20, diameter_mm/2+20)
    ax.set_ylim(-diameter_mm/2-20, diameter_mm/2+20)
    ax.set_aspect('equal')
    ax.set_title('桶内物体分布 (128x128网格)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.grid(True)
    ax.legend()
    
    plt.show()

def visualize_sinogram(projections, num_angles=36, num_detectors=50, diameter_mm=600):
    """
    可视化投影数据 (Sinogram)
    """
    # 重塑为角度×探测器的矩阵
    sinogram = projections.reshape((num_angles, num_detectors))
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    plt.imshow(sinogram, cmap='gray', 
               extent=[-diameter_mm/2, diameter_mm/2, 360, 0], 
               aspect='auto')
    
    plt.colorbar(label='投影值 (mm)')
    plt.xlabel('探测器位置 (mm)')
    plt.ylabel('投影角度 (度)')
    plt.title(f'复杂物体的投影数据 (Sinogram): {num_angles}角度×{num_detectors}探测器')
    plt.show()
    
    return sinogram

def visualize_angle_projection(sinogram, angle_idx=0, diameter_mm=600):
    """
    可视化特定角度的投影
    """
    plt.figure(figsize=(10, 6))
    
    # 获取该角度的投影数据
    projection = sinogram[angle_idx, :]
    
    # 创建探测器位置
    s_values = np.linspace(-diameter_mm/2, diameter_mm/2, len(projection))
    
    # 绘制投影曲线
    plt.plot(s_values, projection, 'b-', linewidth=2)
    
    # 设置图形属性
    plt.title(f'角度 {angle_idx}° 的投影数据')
    plt.xlabel('探测器位置 (mm)')
    plt.ylabel('投影值')
    plt.grid(True)
    plt.xlim(-diameter_mm/2, diameter_mm/2)
    
    plt.show()

# 主程序
if __name__ == "__main__":
    # 设置参数 - 调整为128x128图像网格、36角度、50探测器
    DIAMETER_MM = 600
    NUM_ANGLES = 36
    NUM_DETECTORS = 20
    
    print(f"生成投影数据参数:")
    print(f"  - 桶直径: {DIAMETER_MM} mm")
    print(f"  - 投影角度: {NUM_ANGLES}")
    print(f"  - 探测器数量: {NUM_DETECTORS}")
    print(f"  - 图像网格: 128x128")
    
    # 生成投影数据
    projections, objects = generate_complex_projection_data(
        diameter_mm=DIAMETER_MM,
        num_angles=NUM_ANGLES,
        num_detectors=NUM_DETECTORS
    )
    
    print(f"生成的投影数据形状: {projections.shape}")
    print(f"最小投影值: {np.min(projections):.2f}, 最大投影值: {np.max(projections):.2f}")
    
    # 可视化桶内物体分布
    visualize_phantom(objects, DIAMETER_MM)
    
    # 可视化投影数据
    sinogram = visualize_sinogram(projections, NUM_ANGLES, NUM_DETECTORS, DIAMETER_MM)
    
    # 可视化特定角度的投影
    visualize_angle_projection(sinogram, angle_idx=0, diameter_mm=DIAMETER_MM)
    visualize_angle_projection(sinogram, angle_idx=10, diameter_mm=DIAMETER_MM)
    visualize_angle_projection(sinogram, angle_idx=20, diameter_mm=DIAMETER_MM)
    
    # 保存投影数据
    np.save(f"projection_data/complex_projections.npy", projections)
    print(f"投影数据已保存为 'projection_data/complex_projections_{NUM_ANGLES}x{NUM_DETECTORS}_128x128.npy'")