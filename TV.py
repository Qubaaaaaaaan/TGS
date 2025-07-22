import numpy as np

def compute_tv_gradient(X, epsilon=1e-8):
    """
    根据公式(3-8)计算图像的TV梯度
    
    参数:
        X: 一维图像数组
        epsilon: 避免除零的小常数
        
    返回:
        tv_grad: TV梯度数组 (H, W)
    """
    image = X.reshape(128, 128)  # 假设图像大小为128x128
    h, w = image.shape
    tv_grad = np.zeros((h, w), dtype=np.float32)
    # 遍历图像中的每个像素(s,t)
    for s in range(h):
        for t in range(w):
            # 初始化各项值
            numerator = 0.0
            denominator1 = epsilon
            denominator2 = epsilon
            denominator3 = epsilon
            
            # 计算分子部分: (x_{s,t} - x_{s-1,t}) + (x_{s,t} - x_{s,t-1})
            if s > 0:
                numerator += image[s, t] - image[s-1, t]
            if t > 0:
                numerator += image[s, t] - image[s, t-1]
            
            # 计算分母1: √[ε + (x_{s,t} - x_{s-1,t})^2 + (x_{s,t} - x_{s,t-1})^2]
            if s > 0 and t > 0:
                diff1 = image[s, t] - image[s-1, t]
                diff2 = image[s, t] - image[s, t-1]
                denominator1 = np.sqrt(epsilon + diff1**2 + diff2**2)
            
            # 计算分母2: √[ε + (x_{s+1,t} - x_{s,t})^2 + (x_{s+1,t} - x_{s+1,t-1})^2]
            if s < h-1:
                diff1 = image[s+1, t] - image[s, t]
                if t > 0:
                    diff2 = image[s+1, t] - image[s+1, t-1]
                else:
                    diff2 = 0
                denominator2 = np.sqrt(epsilon + diff1**2 + diff2**2)
            
            # 计算分母3: √[ε + (x_{s,t+1} - x_{s,t})^2 + (x_{s,t+1} - x_{s-1,t+1})^2]
            if t < w-1:
                diff1 = image[s, t+1] - image[s, t]
                if s > 0:
                    diff2 = image[s, t+1] - image[s-1, t+1]
                else:
                    diff2 = 0
                denominator3 = np.sqrt(epsilon + diff1**2 + diff2**2)
            
            # 计算当前像素的TV梯度分量
            term1 = numerator / denominator1 if denominator1 > epsilon else 0
            term2 = (image[s+1, t] - image[s, t]) / denominator2 if s < h-1 and denominator2 > epsilon else 0
            term3 = (image[s, t+1] - image[s, t]) / denominator3 if t < w-1 and denominator3 > epsilon else 0
            
            tv_grad[s, t] = term1 - term2 - term3
    print('tv_grad has been calculated\n',tv_grad)
    
    return tv_grad.flatten()

def tv_regularization_step(X_old, X_new, lambda_tv, tv_iterations):
    """
    TV正则化步骤
    
    参数:
        X_old: 前一次主迭代的初始图像 (H, W)
        X_new: 当前主迭代的初始图像 (H, W)
        lambda_tv: TV正则化系数
        tv_iterations: TV子迭代次数
        
    返回:
        X_refined: 经过TV正则化后的图像
    """
    # 步骤1: 计算相邻两次重建图像之间的欧氏距离 (式3-13)
    d_A = np.linalg.norm(X_new - X_old)
    
    # 步骤2: TV算法迭代
    X_current = X_new.copy()
    
    for m in range(tv_iterations): 
        tv_grad = compute_tv_gradient(X_current)
        grad_norm = np.linalg.norm(tv_grad)
        if grad_norm > 0:
            tv_grad_normalized = tv_grad / grad_norm
        else:
            tv_grad_normalized = np.zeros_like(tv_grad)
        X_current = X_current - lambda_tv * d_A * tv_grad_normalized
        print(f'tv step {m+1} has finished')
    
    return X_current