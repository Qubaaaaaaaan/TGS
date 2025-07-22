import numpy as np
import math
from scipy.sparse import lil_matrix, csr_matrix

def siddon_ray_tracing(x0, y0, x1, y1, num_pixels=(128, 128), pixel_size=0.234375):
    """
    使用Siddon算法计算单条射线穿过的像素及路径长度
    
    参数:
        x0, y0 : 射线起点坐标 (cm)
        x1, y1 : 射线终点坐标 (cm)
        num_pixels : 像素网格尺寸 (nx, ny)
        pixel_size : 像素边长 (cm)
        
    返回:
        list: 元组(i, j, length)的列表，表示穿过的像素索引和路径长度
    """
    nx, ny = num_pixels
    total_size_x = nx * pixel_size
    total_size_y = ny * pixel_size
    
    # 计算网格边界 (从0到总尺寸)
    x_bounds = np.linspace(0, total_size_x, nx + 1)
    y_bounds = np.linspace(0, total_size_y, ny + 1)
    
    # 计算射线方向分量
    dx = x1 - x0
    dy = y1 - y0
    ray_length = np.sqrt(dx**2 + dy**2)
    
    # 处理零长度射线
    if ray_length < 1e-10:
        return []
    
    # 计算参数化射线方程
    if abs(dx) > 1e-10:
        tx = (x_bounds - x0) / dx
    else:
        tx = np.full(nx + 1, np.inf)  # 垂直线
    
    if abs(dy) > 1e-10:
        ty = (y_bounds - y0) / dy
    else:
        ty = np.full(ny + 1, np.inf)  # 水平线
    
    # 计算进入和离开网格的t值
    t_min = max(min(tx[0], tx[-1]), min(ty[0], ty[-1]), 0)
    t_max = min(max(tx[0], tx[-1]), max(ty[0], ty[-1]), 1)
    
    if t_min >= t_max:  # 射线未穿过网格
        return []
    
    # 确定初始像素索引
    def find_pixel_index(x, y):
        i = np.floor(x / pixel_size).astype(int)
        j = np.floor(y / pixel_size).astype(int)
        return min(max(i, 0), nx - 1), min(max(j, 0), ny - 1)
    
    # 进入点坐标
    x_enter = x0 + t_min * dx
    y_enter = y0 + t_min * dy
    i, j = find_pixel_index(x_enter, y_enter)
    
    # 确定步进方向
    step_i = 1 if dx > 0 else -1
    step_j = 1 if dy > 0 else -1
    
    # 计算到下一个网格线的t值增量
    if dx != 0:
        dt_di = abs(pixel_size / dx)
        next_t_i = tx[i + 1] if dx > 0 else tx[i]
    else:
        dt_di = np.inf
        
    if dy != 0:
        dt_dj = abs(pixel_size / dy)
        next_t_j = ty[j + 1] if dy > 0 else ty[j]
    else:
        dt_dj = np.inf
    
    # 初始化结果存储
    path_list = []
    current_t = t_min
    
    # 主循环：遍历射线穿过的像素
    while current_t < t_max:
        # 计算到下一个网格线的最小t值
        if next_t_i < next_t_j:
            next_t = min(next_t_i, t_max)
            # 计算当前像素的路径长度
            segment_length = (next_t - current_t) * ray_length
            path_list.append((i, j, segment_length))
            
            # 更新索引和下一个t值
            i += step_i
            if i < 0 or i >= nx:
                break
            next_t_i = tx[i] if dx < 0 else tx[i + 1]
            next_t_i += dt_di if step_i > 0 else -dt_di
            
        else:
            next_t = min(next_t_j, t_max)
            segment_length = (next_t - current_t) * ray_length
            path_list.append((i, j, segment_length))
            
            j += step_j
            if j < 0 or j >= ny:
                break
            next_t_j = ty[j] if dy < 0 else ty[j + 1]
            next_t_j += dt_dj if step_j > 0 else -dt_dj
        
        current_t = next_t
    
    return path_list

def generate_system_matrix_for_circle(num_angles=36, num_detectors=50, image_size=128, pixel_size=0.1):
    """
    生成针对内切圆扫描几何的系统矩阵
    
    参数:
        num_angles : 投影角度数量 (360度)
        num_detectors : 探测器单元数量 (50个)
        image_size : 图像尺寸 (像素)
        pixel_size : 像素尺寸 (cm)
        
    返回:
        csr_matrix: 稀疏系统矩阵 (num_angles*num_detectors, image_size**2)
    """
    total_rays = num_angles * num_detectors
    total_pixels = image_size * image_size
    sys_matrix = lil_matrix((total_rays, total_pixels))
    
    # 计算内切圆参数
    center = image_size * pixel_size / 2  # 中心坐标 (cm)
    radius = center  # 内切圆半径 (cm)
    
    # 计算探测器位置 (在内切圆直径上均匀分布)
    detector_positions = np.linspace(-radius, radius, num_detectors)
    
    for angle_idx in range(num_angles):
        # 当前角度 (度转弧度)
        theta = angle_idx * (2 * np.pi / num_angles)
        
        # 计算当前角度的法线方向 (射线方向)
        nx = np.cos(theta)  # x分量
        ny = np.sin(theta)  # y分量
        
        # 计算垂直方向 (用于确定探测器位置)
        perp_x = -ny
        perp_y = nx
        
        for det_idx in range(num_detectors):
            # 当前射线在直径上的位置
            s = detector_positions[det_idx]
            
            # 计算射线起点和终点 (在正方形边界上)
            # 起点: (center + s*perp_x, center + s*perp_y)
            # 终点: 沿法线方向延伸足够距离
            ray_length = 2 * image_size * pixel_size  # 足够长的射线
            
            # 起点坐标 (探测器位置)
            x0 = center + s * perp_x
            y0 = center + s * perp_y
            
            # 终点坐标 (射线源位置，沿法线反方向)
            x1 = center - ray_length * nx
            y1 = center - ray_length * ny
            
            # 计算单条射线的路径
            path = siddon_ray_tracing(
                x0, y0, x1, y1,
                num_pixels=(image_size, image_size),
                pixel_size=pixel_size
            )
            
            # 填充系统矩阵
            ray_idx = angle_idx * num_detectors + det_idx
            for (i, j, length) in path:
                pixel_idx = j * image_size + i
                sys_matrix[ray_idx, pixel_idx] = length
    
    return sys_matrix.tocsr()

# 示例使用
if __name__ == "__main__":
    print("\n生成系统矩阵 (36角度, 50探测器, 128x128图像)...")
    # 使用新参数: 128x128图像，36角度，50探测器
    full_matrix = generate_system_matrix_for_circle(
        num_angles=36,
        num_detectors=20,
        image_size=128,
        pixel_size=0.46875  # 保持像素尺寸为0.1cm
    )
    print(f"系统矩阵形状: {full_matrix.shape}")
    print(f"非零元素数量: {full_matrix.nnz}")
    
    # 保存矩阵供后续使用
    from scipy.sparse import save_npz
    save_npz("system_matrix_36x50_128x128.npz", full_matrix)
    print("系统矩阵已保存到 system_matrix_36x50_128x128.npz")