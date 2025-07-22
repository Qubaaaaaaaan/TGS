import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def generate_rectangle_projection_data(width=20, height=15, num_angles=360, num_detectors=100):
    """
    生成长方形的投影数据
    
    参数:
        width: 长方形宽度 (x方向, cm)
        height: 长方形高度 (y方向, cm)
        num_angles: 投影角度数量
        num_detectors: 每个角度的探测器数量
        
    返回:
        projections: 投影数据数组 (num_angles * num_detectors,)
        s_values: 探测器位置数组
    """
    # 1. 计算几何参数
    half_width = width / 2.0
    half_height = height / 2.0
    max_radius = 12.8  # 外接圆半径
    
    # 2. 创建探测器位置 (沿对称轴均匀分布)
    s_values = np.linspace(-max_radius, max_radius, num_detectors)
    
    # 3. 初始化投影数据数组
    projections = np.zeros(num_angles * num_detectors)
    
    # 4. 对于每个角度生成投影
    for angle_idx in range(num_angles):
        # 当前角度（弧度）
        theta = np.deg2rad(angle_idx)
        
        # 角度对应的余弦和正弦值
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 对于每个探测器
        for det_idx in range(num_detectors):
            # 当前探测器位置 (s)
            s = s_values[det_idx]
            
            # 计算射线参数方程: x*cos(theta) + y*sin(theta) = s
            
            # 5. 计算射线与长方形的交点
            # 初始化交点列表
            intersections = []
            
            # 计算射线与长方形四条边的交点
            # 左边界: x = -half_width
            if abs(cos_theta) > 1e-10:
                y_left = (s + half_width * cos_theta) / sin_theta if abs(sin_theta) > 1e-10 else 0
                if -half_height <= y_left <= half_height:
                    intersections.append((-half_width, y_left))
            
            # 右边界: x = half_width
            if abs(cos_theta) > 1e-10:
                y_right = (s - half_width * cos_theta) / sin_theta if abs(sin_theta) > 1e-10 else 0
                if -half_height <= y_right <= half_height:
                    intersections.append((half_width, y_right))
            
            # 下边界: y = -half_height
            if abs(sin_theta) > 1e-10:
                x_bottom = (s + half_height * sin_theta) / cos_theta if abs(cos_theta) > 1e-10 else 0
                if -half_width <= x_bottom <= half_width:
                    intersections.append((x_bottom, -half_height))
            
            # 上边界: y = half_height
            if abs(sin_theta) > 1e-10:
                x_top = (s - half_height * sin_theta) / cos_theta if abs(cos_theta) > 1e-10 else 0
                if -half_width <= x_top <= half_width:
                    intersections.append((x_top, half_height))
            
            # 6. 计算弦长 (射线穿过长方形的长度)
            if len(intersections) >= 2:
                # 对交点按射线上位置排序
                sorted_intersections = sorted(
                    intersections, 
                    key=lambda p: p[0]*cos_theta + p[1]*sin_theta
                )
                
                # 取第一个和最后一个交点
                p1 = sorted_intersections[0]
                p2 = sorted_intersections[-1]
                
                # 计算两点间距离
                chord_length = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                
                # 投影值 = 衰减系数 × 路径长度 = 1 × chord_length
                projection_value = chord_length
            else:
                # 射线未穿过长方形
                projection_value = 0.0
            
            # 存储投影值
            proj_idx = angle_idx * num_detectors + det_idx
            projections[proj_idx] = projection_value
    
    return projections, s_values

def visualize_rectangle_projection_data(projections, s_values, num_angles=360):
    """
    可视化投影数据
    """
    # 重塑为角度×探测器的矩阵
    sinogram = projections.reshape((num_angles, len(s_values)))
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 投影数据图
    plt.subplot(1, 2, 1)
    plt.imshow(sinogram, cmap='gray', 
               extent=[s_values[0], s_values[-1], 0, 360], 
               aspect='auto')
    plt.colorbar(label='投影值 (cm)')
    plt.xlabel('探测器位置 (cm)')
    plt.ylabel('投影角度 (度)')
    plt.title('长方形的投影数据 (Sinogram)')
    
    # 特定角度的投影曲线
    plt.subplot(1, 2, 2)
    angles_to_plot = [0, 30, 60, 90]
    for angle in angles_to_plot:
        plt.plot(s_values, sinogram[angle, :], label=f'{angle}°')
    
    plt.title('不同角度的投影曲线')
    plt.xlabel('探测器位置 (cm)')
    plt.ylabel('投影值 (cm)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return sinogram

def visualize_geometry(width=20, height=15):
    """可视化长方形几何形状"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # 绘制长方形
    rect = Rectangle((-width/2, -height/2), width, height, 
                     linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # 绘制示例射线
    angles = [0, 30, 60, 90]
    s = 0  # 中心射线
    
    for angle in angles:
        theta = np.deg2rad(angle)
        dx = np.cos(theta)
        dy = np.sin(theta)
        
        # 计算射线方向
        length = max(width, height) * 1.2
        plt.plot([s*dx - dy*length, s*dx + dy*length],
                 [s*dy + dx*length, s*dy - dx*length],
                 'b-', linewidth=1)
    
    plt.xlim(-width*0.7, width*0.7)
    plt.ylim(-height*0.7, height*0.7)
    plt.gca().set_aspect('equal')
    plt.title(f'{width}cm×{height}cm长方形几何形状')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 可视化几何形状
    visualize_geometry()
    
    # 生成投影数据
    projections, s_values = generate_rectangle_projection_data(
        width=20,  # x方向长度
        height=15,  # y方向长度
        num_angles=360,
        num_detectors=100
    )
    
    print(f"生成的投影数据形状: {projections.shape}")
    print(f"最小投影值: {np.min(projections):.2f}, 最大投影值: {np.max(projections):.2f}")
    
    # 可视化投影数据
    sinogram = visualize_rectangle_projection_data(projections, s_values)
    
    # 保存投影数据
    np.save("projection_data/rectangle_projections.npy", projections)
    print("投影数据已保存为 'projection_data/rectangle_projections.npy'")