import numpy as np
from scipy.sparse import load_npz
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import os

# 修改为匹配所有角度文件
npz_files = sorted(glob.glob('eff_1/efficiency_angle_*.npz'))

if not npz_files:
    print("未找到任何符合规则的 npz 文件。")
    sys.exit(1)

for npz_file in npz_files:
    # 提取角度值
    filename = os.path.basename(npz_file)
    angle = filename.split('_')[-1].split('.')[0]
    
    sparse_matrix = load_npz(npz_file)
    dense_matrix = sparse_matrix.toarray()

    pdf_filename = f'image_1/visualization_{angle}.pdf'
    with PdfPages(pdf_filename) as pdf:
        n_rows = dense_matrix.shape[0]
        # 修改：每页25张图像 (5x5布局)
        for start in range(0, n_rows, 25):
            end = min(start + 25, n_rows)
            # 创建5x5子图布局
            fig, axes = plt.subplots(5, 5, figsize=(20, 20))
            axes = axes.flatten()
            
            for i, row_index in enumerate(range(start, end)):
                row = dense_matrix[row_index]
                image_matrix = row.reshape(128,128)
                im = axes[i].imshow(image_matrix, aspect='auto', cmap='viridis')
                axes[i].set_title(f'Row {row_index}', fontsize=8)
                # 添加紧凑型颜色条
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            # 隐藏多余的子图
            for j in range(end - start, 25):
                axes[j].axis('off')
            
            plt.suptitle(f'Angle {angle} Rows {start}-{end-1}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"角度 {angle} 的图像已保存到 {pdf_filename}")

print("所有角度的图像已保存。")
print("请检查 'image' 目录下的 PDF 文件。")