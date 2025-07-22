import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

def visualize_reconstruction(image, size=128, title='CT重建图像', save_path=None):
    """
    将重建结果的一维向量转换为二维图像并可视化
    
    参数:
        X_vector: 重建结果的一维向量 (size*size,)
        size: 图像尺寸 (默认256)
        title: 图像标题
        save_path: 图像保存路径 (可选)
        
    返回:
        fig, ax: 图像对象和坐标轴对象
    """
    
    # 2. 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 3. 确定颜色映射范围
    vmin = np.min(image)
    vmax = np.max(image)
    
    # 如果图像全为零，设置默认范围
    if vmax - vmin < 1e-6:
        vmin, vmax = 0, 1
    
    # 4. 创建归一化对象
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # 5. 显示图像
    img = ax.imshow(image, cmap='viridis', norm=norm)
    
    # 6. 添加颜色条
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attenuation coefficient', fontsize=12)
    
    # 7. 设置标题和坐标轴
    ax.set_title(title, fontsize=14)
    
    # 10. 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    
    return fig, ax

# 示例用法
if __name__ == "__main__":
    size = 256
    test_image = np.zeros(size*size)
    data = np.load('result_1.npy')
    visualize_reconstruction(0.5*data, title='transmission reconstruction', save_path='output.png')