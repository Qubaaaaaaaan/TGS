import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, save_npz
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================== 扫描参数配置 ==================
detector_width = 50  # 探测器宽度 (mm)
distances_to_center = np.array([800])  # 探测器阵列到圆心的距离 (mm)
num_angles = 36  # 角度数量 (1°步长)
num_detectors = 20  # 每条直线上的探测器数量
scan_range = 600  # 扫描范围 (直径600mm)
voxel_num = 128  # 体素网格大小
voxel_size = scan_range / voxel_num  # 每个体素的边长 (mm)
half_range = scan_range / 2  # 扫描半径

# ================== 几何计算函数 ==================
def calculate_detector_positions():
    """计算所有探测器中心位置"""
    s_centers = np.linspace(-scan_range/2, scan_range/2, num_detectors)
    detector_positions = []
    
    for angle_deg in range(num_angles):
        theta = np.deg2rad(angle_deg * 10)  # 10度步长
        direction_vector = np.array([-np.sin(theta), np.cos(theta)])
        normal_vector = np.array([np.cos(theta), np.sin(theta)])
        reference_point = distances_to_center[0] * normal_vector
        
        for det_idx, s_center in enumerate(s_centers):
            det_center = reference_point + s_center * direction_vector
            Px, Py = det_center
            detector_positions.append((angle_deg, det_idx, Px, Py))
    
    return detector_positions

# ================== 射线追踪函数 ==================
def phys_to_grid(x_phys, y_phys):
    """将物理坐标转换为网格坐标"""
    x_grid = (x_phys + half_range) / voxel_size
    y_grid = (y_phys + half_range) / voxel_size
    return x_grid, y_grid

def grid_to_phys(i, j):
    """将网格索引转换为物理坐标"""
    x_phys = (j + 0.5) * voxel_size - half_range
    y_phys = (i + 0.5) * voxel_size - half_range
    return x_phys, y_phys

def compute_weighted_path_sum(i_s, j_s, Px, Py, mu_matrix):
    """
    计算从体素中心到探测器点的加权路径和
    
    参数:
    i_s, j_s: 源体素索引
    Px, Py: 探测器物理坐标
    mu_matrix: 衰减系数矩阵
    
    返回:
    加权路径和: Σ(μ_k * l_k) 对于路径上的所有体素k
    """
    # 计算源体素中心坐标 (物理坐标)
    Sx, Sy = grid_to_phys(i_s, j_s)
    
    # 计算方向向量
    Dx = Px - Sx
    Dy = Py - Sy
    
    # 计算线段总长度
    total_length = math.sqrt(Dx**2 + Dy**2)
    
    # 处理零向量
    if total_length < 1e-10:
        return 0.0
    
    # 确定步进方向
    step_x = 1 if Dx > 0 else -1 if Dx < 0 else 0
    step_y = 1 if Dy > 0 else -1 if Dy < 0 else 0
    
    # 初始化当前体素和参数
    i, j = i_s, j_s
    t_in = 0.0
    weighted_sum = 0.0
    
    while True:
        # 计算到下一个垂直网格线的时间
        if Dx != 0:
            if step_x == 1:
                next_x = j + 1  # 向右，下一个垂直网格线
            else:  # step_x == -1
                next_x = j      # 向左，下一个垂直网格线
            t_x_next = ((next_x * voxel_size - half_range) - Sx) / Dx
        else:
            t_x_next = float('inf')
        
        # 计算到下一个水平网格线的时间
        if Dy != 0:
            if step_y == 1:
                next_y = i + 1  # 向上，下一个水平网格线
            else:  # step_y == -1
                next_y = i      # 向下，下一个水平网格线
            t_y_next = ((next_y * voxel_size - half_range) - Sy) / Dy
        else:
            t_y_next = float('inf')
        
        # 确定下一个出口点
        t_out = min(t_x_next, t_y_next)
        
        # 计算当前体素中的线段长度
        segment_length = (t_out - t_in) * total_length
        
        # 获取当前体素的衰减系数μ
        mu = mu_matrix[i, j]
        
        # 计算μl并累加到总和
        weighted_sum += mu * segment_length
        
        # 确定射线离开当前体素的方向
        exit_vertically = t_x_next <= t_y_next
        exit_horizontally = t_y_next <= t_x_next
        
        # 更新体素索引
        new_i, new_j = i, j
        if exit_vertically:
            new_j = j + step_x
        if exit_horizontally:
            new_i = i + step_y
        
        # 检查是否离开网格
        if (new_i < 0 or new_i >= voxel_num or 
            new_j < 0 or new_j >= voxel_num):
            break  # 射线离开网格
        
        # 更新为下一个体素
        i, j = new_i, new_j
        t_in = t_out
    
    return weighted_sum

# ================== 系统矩阵构建 ==================
# ================== 系统矩阵构建 ==================
def build_system_matrix(mu_matrix, parallel=True):
    """
    构建系统矩阵（体素作为源）
    
    参数:
    mu_matrix: 衰减系数矩阵 (128x128)
    parallel: 是否使用并行计算
    
    返回:
    系统矩阵 (体素数 × 投影数)
    """
    # 计算所有探测器位置
    detector_positions = calculate_detector_positions()
    num_projections = len(detector_positions)
    num_voxels = voxel_num * voxel_num
    
    # 使用LIL格式稀疏矩阵
    A = lil_matrix((num_voxels, num_projections), dtype=np.float32)
    
    # 进度跟踪
    start_time = time.time()
    total_projections = num_projections
    
    # 创建位置信息列表 (修复进度报告的关键)
    position_info = []
    for idx, (angle_deg, det_idx, Px, Py) in enumerate(detector_positions):
        position_info.append({
            'proj_idx': idx,
            'angle': angle_deg,
            'detector': det_idx,
            'Px': Px,
            'Py': Py
        })
    
    # 并行计算
    if parallel:
        with ProcessPoolExecutor() as executor:
            # 提交所有投影任务
            futures = {}
            for pos_info in position_info:
                future = executor.submit(
                    compute_projection_column,
                    pos_info['Px'], 
                    pos_info['Py'], 
                    mu_matrix, 
                    voxel_num
                )
                futures[future] = pos_info  # 存储位置信息
            
            # 处理结果
            completed = 0
            for future in as_completed(futures):
                pos_info = futures[future]
                column_data = future.result()
                
                # 将列数据填充到系统矩阵
                for voxel_idx, value in column_data:
                    A[voxel_idx, pos_info['proj_idx']] = value
                
                completed += 1
                elapsed = time.time() - start_time
                progress = completed / total_projections * 100
                print(f"进度: {completed}/{total_projections} ({progress:.1f}%) | "
                      f"已用时间: {elapsed:.1f}s | "
                      f"角度: {pos_info['angle']}, 探测器: {pos_info['detector']}")
    
    # 串行计算
    else:
        for pos_info in position_info:
            column_data = compute_projection_column(
                pos_info['Px'], 
                pos_info['Py'], 
                mu_matrix, 
                voxel_num
            )
            
            # 将列数据填充到系统矩阵
            for voxel_idx, value in column_data:
                A[voxel_idx, pos_info['proj_idx']] = value
            
            # 进度报告
            completed = pos_info['proj_idx'] + 1
            elapsed = time.time() - start_time
            progress = completed / total_projections * 100
            
            if completed % 10 == 0:
                print(f"进度: {completed}/{total_projections} ({progress:.1f}%) | "
                      f"已用时间: {elapsed:.1f}s | "
                      f"角度: {pos_info['angle']}, 探测器: {pos_info['detector']}")
    
    # 转换为CSR格式
    A_csr = A.tocsr()
    
    # 保存矩阵
    save_npz('system_matrix_voxel_source.npz', A_csr)
    print("系统矩阵已保存为 'system_matrix_voxel_source.npz'")
    
    return A_csr

def compute_projection_column(Px, Py, mu_matrix, grid_size):
    """
    计算单个投影列的所有体素值
    
    返回:
    列表，元素为元组 (voxel_index, value)
    """
    column_data = []
    
    # 遍历所有体素
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算加权路径和
            weighted_sum = compute_weighted_path_sum(i, j, Px, Py, mu_matrix)
            
            if abs(weighted_sum) > 1e-10:  # 只存储非零值
                voxel_idx = i * grid_size + j
                column_data.append((voxel_idx, weighted_sum))
    
    return column_data

# ================== 投影计算函数 ==================
def compute_projections(A, source_intensity):
    """
    计算投影值
    
    参数:
    A: 系统矩阵 (CSR格式)
    source_intensity: 源强度矩阵 (128x128)
    
    返回:
    投影值向量 (投影数,)
    """
    # 将源强度矩阵展平
    source_vector = source_intensity.flatten()
    
    # 计算投影: p = A^T * source_vector
    projections = A.transpose().dot(source_vector)
    
    return projections

# ================== 结果可视化 ==================
def visualize_voxel_source(i, j, mu_matrix, detector_positions):
    """
    可视化单个体素源对所有探测器的加权路径和
    
    参数:
    i, j: 体素索引
    mu_matrix: 衰减系数矩阵
    detector_positions: 所有探测器位置
    """
    # 计算体素中心坐标
    voxel_x, voxel_y = grid_to_phys(i, j)
    
    # 计算该体素对所有探测器的加权路径和
    projections = np.zeros(len(detector_positions))
    for idx, (_, _, Px, Py) in enumerate(detector_positions):
        projections[idx] = compute_weighted_path_sum(i, j, Px, Py, mu_matrix)
    
    # 创建图像
    plt.figure(figsize=(15, 6))
    
    # 衰减系数矩阵
    plt.subplot(1, 3, 1)
    plt.imshow(mu_matrix, cmap='viridis', origin='lower', 
               extent=[0, voxel_num, 0, voxel_num])
    plt.colorbar(label='Attenuation μ')
    plt.title('Attenuation Coefficient Matrix')
    plt.plot(j, i, 'ro', markersize=8)  # 标记源体素位置
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    
    # 投影值随角度变化
    plt.subplot(1, 3, 2)
    angle_values = np.array([angle for angle, _, _, _ in detector_positions])
    plt.scatter(angle_values, projections, c=projections, cmap='viridis')
    plt.colorbar(label='Weighted Path Sum')
    plt.title(f'Projections for Voxel ({i},{j})')
    plt.xlabel('Angle Index')
    plt.ylabel('Weighted Path Sum')
    plt.grid(True)
    
    # 投影值随探测器位置变化
    plt.subplot(1, 3, 3)
    det_values = np.array([det_idx for _, det_idx, _, _ in detector_positions])
    plt.scatter(det_values, projections, c=projections, cmap='viridis')
    plt.colorbar(label='Weighted Path Sum')
    plt.title(f'Projections for Voxel ({i},{j})')
    plt.xlabel('Detector Index')
    plt.ylabel('Weighted Path Sum')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return projections

# ================== 主函数 ==================
def main():
    print("="*50)
    print("体素源系统矩阵构建")
    print("="*50)
    print(f"扫描参数:")
    print(f"  角度数: {num_angles} (步长10°)")
    print(f"  探测器数: {num_detectors}")
    print(f"  体素网格: {voxel_num}x{voxel_num}")
    print(f"  投影总数: {num_angles * num_detectors}")
    print(f"  系统矩阵大小: ({voxel_num*voxel_num}, {num_angles*num_detectors})")
    
    # 创建衰减系数矩阵 (示例)
    mu_matrix = np.load('result_1.npy')
    
    # 步骤1: 构建系统矩阵
    print("\n步骤1: 构建系统矩阵...")
    start_time = time.time()
    A = build_system_matrix(mu_matrix, parallel=True)
    build_time = time.time() - start_time
    print(f"系统矩阵构建完成! 耗时: {build_time:.1f}秒")
    print(f"矩阵形状: {A.shape} (行:体素数, 列:投影数)")
    print(f"非零元素: {A.count_nonzero()}")
    print(f"稀疏度: {A.count_nonzero() / (A.shape[0]*A.shape[1])*100:.4f}%")

if __name__ == "__main__":
    main()