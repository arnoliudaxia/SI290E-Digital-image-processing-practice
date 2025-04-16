import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def canny_edge_detection(image_path, low_threshold=20, high_threshold=60, sigma=1.0):
    """
    实现Canny边缘检测算法
    
    参数:
        image_path: 输入图像路径或图像数组
        low_threshold: 低阈值
        high_threshold: 高阈值
        sigma: 高斯滤波器的标准差
    
    返回:
        edge_map: 边缘检测结果，二值图像
    """
    # 1. 读取图像
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = image_path
        # 如果是彩色图，转换为灰度图
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 确保图像是无符号8位整数类型
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # 转换为float类型，方便计算
    img = img.astype(np.float64)
    
    # 2. 高斯平滑
    # 生成高斯滤波器
    filter_size = 2 * int(np.ceil(3 * sigma)) + 1  # 滤波器大小，保证覆盖3个标准差
    gauss_filter = gaussian_kernel(filter_size, sigma)
    
    # 应用高斯滤波
    smoothed_img = convolve(img, gauss_filter)
    
    # 3. 计算梯度幅值和方向
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # 计算x和y方向的梯度
    gradient_x = convolve(smoothed_img, sobel_x)
    gradient_y = convolve(smoothed_img, sobel_y)
    
    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    # 将角度转换为0-180度范围
    gradient_direction = np.mod(gradient_direction, 180)
    
    # 4. 非极大值抑制
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # 获取当前像素的梯度方向
            direction = gradient_direction[i, j]
            
            # 将方向量化为0，45，90，135度
            if (direction >= 0 and direction < 22.5) or (direction >= 157.5 and direction < 180):
                neighbors = [gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]]
            elif direction >= 22.5 and direction < 67.5:
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
            elif direction >= 67.5 and direction < 112.5:
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            else:
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
            
            # 如果当前像素的梯度幅值大于其梯度方向上的邻居，则保留
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]
    
    # 5. 双阈值处理和滞后阈值处理
    edge_map = np.zeros_like(suppressed)
    strong_edges = suppressed >= high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed < high_threshold)
    
    # 标记强边缘
    edge_map[strong_edges] = 1
    
    # 连接弱边缘
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if weak_edges[i, j]:
                # 检查8邻域内是否有强边缘
                if np.any(edge_map[i-1:i+2, j-1:j+2]):
                    edge_map[i, j] = 1
    
    return edge_map


def gaussian_kernel(size, sigma):
    """
    创建高斯滤波器
    
    参数:
        size: 滤波器大小
        sigma: 高斯函数的标准差
    
    返回:
        kernel: 高斯滤波器
    """
    # 确保size是奇数
    if size % 2 == 0:
        size = size + 1
    
    # 计算滤波器的中心
    center = (size - 1) // 2
    
    # 创建网格
    x, y = np.meshgrid(np.arange(-center, center+1), np.arange(-center, center+1))
    
    # 计算高斯函数
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # 归一化滤波器，使得权重和为1
    kernel = kernel / np.sum(kernel)
    
    return kernel


def calculate_iou(edge_map, ground_truth):
    """
    计算交并比（IoU）
    
    参数:
        edge_map: 预测的边缘图像，二值图像
        ground_truth: 真实的边缘图像，二值图像
    
    返回:
        iou: 交并比
    """
    # 如果输入是图像路径，读取图像
    if isinstance(edge_map, str):
        edge_map = cv2.imread(edge_map, cv2.IMREAD_GRAYSCALE)
    if isinstance(ground_truth, str):
        ground_truth = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE)
    
    # 确保图像是二值图像
    if edge_map.dtype != bool:
        edge_map = edge_map > 0
    if ground_truth.dtype != bool:
        ground_truth = ground_truth > 0
    
    # 计算交集和并集
    intersection = np.logical_and(edge_map, ground_truth)
    union = np.logical_or(edge_map, ground_truth)
    
    # 计算IoU
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    
    if union_sum == 0:
        return 0.0
    
    return intersection_sum / union_sum


if __name__ == "__main__":
    # 读取图像
    image_path = 'images/Q1.tif'
    
    # 设置参数
    low_threshold = 20   # 低阈值
    high_threshold = 60  # 高阈值
    sigma = 1.0          # 高斯滤波器的标准差
    
    # 运行Canny边缘检测
    edge_map = canny_edge_detection(image_path, low_threshold, high_threshold, sigma)
    
    # 显示结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edge_map, cmap='gray')
    plt.title('Canny边缘检测结果')
    
    # 保存结果
    plt.tight_layout()
    cv2.imwrite('canny_result.png', edge_map * 255)
    print('边缘检测完成，结果已保存为canny_result.png')
    
    plt.show()
    

    

    
