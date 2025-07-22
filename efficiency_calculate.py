import numpy as np
from requests import get
from scipy.sparse import coo_matrix, save_npz, vstack, load_npz


detector_width = 50  # 探测器宽度 (mm)
distances_to_center = np.array([[800], [550]])  # 两个探测器阵列到圆心的距离 (mm)
num_angles = 36  # 角度数量 (1°步长)
num_detectors = 20  # 每条直线上的探测器数量
scan_range = 600  # 扫描范围 (直径600mm)
voxel_num = 128
voxel_size = scan_range / voxel_num  # 每个体素的边长
half_range = scan_range / 2
total_proj = num_angles * num_detectors  # 36000
rows = []
cols = []
data = []

def calculate_line(x_1=0, y_1=0, x_2=0, y_2=0, theta=0, distance=0, type=0):
    if type == 0:
        # 支持批量输入
        x_1 = np.asarray(x_1)
        y_1 = np.asarray(y_1)
        x_2 = np.asarray(x_2)
        y_2 = np.asarray(y_2)
        A0 = y_2 - y_1
        A1 = x_1 - x_2
        A2 = -x_1 * y_2 + x_2 * y_1
        # 广播后堆叠
        A = np.stack([A0, A1, A2], axis=-1)
    if type == 1:
        x_0 = distance * np.cos(theta)
        y_0 = distance * np.sin(theta)
        A = np.array([np.cos(theta), np.sin(theta), -distance])
    return A

def intersection(A, B):
    denominator = A[0] * B[1] - A[1] * B[0]
    x = (A[2] * B[1] - B[2] * A[1]) / denominator
    y = (B[0] * A[2] - A[0] * B[2]) / denominator
    return np.array([x, y])


def stable_sort_points(arr):
    idx_x = np.argsort(arr[:, :, 0], axis=1, kind='stable')
    sorted_x = np.take_along_axis(arr, idx_x[:, :, None], axis=1)
    
    # 检查每个体素的所有 y 值是否相同
    y_values = sorted_x[:, :, 1]
    y_min = np.min(y_values, axis=1, keepdims=True)
    y_max = np.max(y_values, axis=1, keepdims=True)
    all_y_same = np.all(np.isclose(y_min, y_max, atol=1e-6), axis=1)
    
    # 如果所有 y 相同，直接返回 x 排序结果
    sorted_xy = np.where(all_y_same[:, None, None], sorted_x, 
                        np.take_along_axis(sorted_x, 
                                            np.argsort(sorted_x[:, :, 1], axis=1, kind='stable')[:, :, None], 
                        axis=1))
    return sorted_xy
def get_area_vectorized(xs, ys, angle_deg, det_idx):
    """
    xs, ys: shape=(N,) 体素中心坐标
    angle_deg, det_idx: 指定的角度和探测器编号
    返回: shape=(N,) 每个体素的面积
    """
    theta = np.deg2rad(angle_deg*10)
    direction_vector = np.array([-np.sin(theta), np.cos(theta)])
    normal_vector = np.array([np.cos(theta), np.sin(theta)])
    # 固定直线参数
    A1 = calculate_line(theta=theta, distance=800, type=1)
    A2 = calculate_line(theta=theta, distance=840, type=1)
    ends = all_detector_ends[angle_deg, det_idx]  # (4,2)
    xs = xs.ravel()
    ys = ys.ravel()
    N = xs.shape[0]
    # 计算(x_0, y_0)
    x_0 = np.mean(ends[:, 0])
    y_0 = np.mean(ends[:, 1])
    # 体素指向探测器中心向量
    v = np.stack([xs - x_0, ys - y_0], axis=1)  # shape (N,2)
    # 与法向量的夹角正切
    v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    v_unit = v / v_norm
    n_unit = normal_vector / (np.linalg.norm(normal_vector) + 1e-12)
    cos_angle = np.dot(v_unit, n_unit)
    # 计算正切值
    sin_angle = np.sqrt(1 - cos_angle**2)
    tan_angle = sin_angle / (cos_angle + 1e-12)
    # 夹角正切大于1/5的mask
    mask_angle = np.abs(tan_angle) <= 0.2

    # 批量计算4条连线
    intersections1 = np.zeros((N, 4, 2))
    intersections2 = np.zeros((N, 4, 2))
    for k, end in enumerate(ends):
        B = calculate_line(xs, ys, np.full(N, end[0]), np.full(N, end[1]), type=0)
        denom = A1[0] * B[:,1] - A1[1] * B[:,0] + 1e-12
        x = (A1[2] * B[:,1] - B[:,2] * A1[1]) / denom
        y = (B[:,0] * A1[2] - A1[0] * B[:,2]) / denom
        intersections1[:,k,0] = x
        intersections1[:,k,1] = y
        denom2 = A2[0] * B[:,1] - A2[1] * B[:,0] + 1e-12
        x2 = (A2[2] * B[:,1] - B[:,2] * A2[1]) / denom2
        y2 = (B[:,0] * A2[2] - A2[0] * B[:,2]) / denom2
        intersections2[:,k,0] = x2
        intersections2[:,k,1] = y2
    # 先对x排序，再对y排序，分别取中间两个点
    sorted_xy1 = stable_sort_points(intersections1)
    sorted_xy2 = stable_sort_points(intersections2)
    
    # 取中间两个点
    A = sorted_xy1[:, 1, :]
    Bp = sorted_xy1[:, 2, :]
    C = sorted_xy2[:, 1, :]
    D = sorted_xy2[:, 2, :]
    # 面积
    area = 40 * (np.linalg.norm(A-Bp, axis=1) + np.linalg.norm(C-D, axis=1)) / 2
    # 应用夹角mask
    area = area * mask_angle
    return area


# 计算探测器中心位置 (沿直线方向)
s_centers = np.linspace(-scan_range/2, scan_range/2, num_detectors)

# 存储所有探测器端点坐标
# 结构: [角度][探测器编号] = [端点1, 端点2]
all_detector_ends = np.zeros((num_angles, num_detectors, 4, 2))

# 计算每个角度和每个探测器的端点
for angle_deg in range(num_angles):
    # 角度转换为弧度
    theta = np.deg2rad(angle_deg*10)
    
    # 计算直线方向向量 (与探测器直线平行)
    direction_vector = np.array([-np.sin(theta), np.cos(theta)])
    
    # 计算法向量 (指向圆心)
    normal_vector = np.array([np.cos(theta), np.sin(theta)])
    
    # 计算直线上的参考点 (最接近圆心的点)
    reference_point = distances_to_center * normal_vector
    
    for det_idx, s_center in enumerate(s_centers):
        # 计算探测器中心位置
        det_center = reference_point + s_center * direction_vector
        
        # 计算两端点 (沿直线方向偏移±25mm)
        end1 = det_center - (detector_width/2) * direction_vector
        end2 = det_center + (detector_width/2) * direction_vector
        
        # 存储结果
        all_detector_ends[angle_deg, det_idx] = [[end1[0,0],end1[0,1]],
                                                 [end2[0,0],end2[0,1]],
                                                 [end1[1,0],end1[1,1]], 
                                                 [end2[1,0],end2[1,1]]]
        
# 生成256x256体素中心坐标，区域中心为(0,0)，边长600mm

# xx[i, j], yy[i, j] 即为第(i, j)个体素中心的x, y坐标

# 体素中心坐标（x从-300+dx/2到+300-dx/2，y同理）
x_centers = np.linspace(-half_range + voxel_size/2, half_range - voxel_size/2, voxel_num)
y_centers = np.linspace(-half_range + voxel_size/2, half_range - voxel_size/2, voxel_num)
xx, yy = np.meshgrid(x_centers, y_centers, indexing='xy')  # xx, yy shape: (256, 256)
xx_flat = xx.ravel()
yy_flat = yy.ravel()
voxel_indices = np.arange(voxel_num * voxel_num)

partial_files = []
for angle_deg in range(num_angles):
    rows = []
    cols = []
    data = []
    for det_idx in range(num_detectors):
        proj_idx = angle_deg * num_detectors + det_idx
        area_vec = get_area_vectorized(xx_flat, yy_flat, angle_deg, det_idx)
        mask = area_vec > 1e-6
        rows.extend([det_idx]*np.sum(mask))  # 注意：这里是每个角度下的本地行号
        cols.extend(voxel_indices[mask])
        data.extend(area_vec[mask])
    # 构建并保存本角度的稀疏矩阵（shape: 100 x 65536）
    eff_matrix = coo_matrix((data, (rows, cols)), shape=(num_detectors, voxel_num * voxel_num))
    fname = f"eff_1/efficiency_angle_{angle_deg:03d}.npz"
    save_npz(fname, eff_matrix)
    partial_files.append(fname)
    print(f"{angle_deg} 角度处理完成")
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
expected_shape = (num_angles * num_detectors, 128)
if combined_matrix.shape != expected_shape:
    print(f"警告：合并后的矩阵形状为 {combined_matrix.shape}，预期为 {expected_shape}")
else:
    print(f"成功创建合并矩阵，形状为 {combined_matrix.shape}")

# 保存合并后的矩阵
output_file = "eff_1/combined_efficiency_matrix.npz"
save_npz(output_file, combined_matrix)
print(f"合并后的矩阵已保存至 {output_file}")