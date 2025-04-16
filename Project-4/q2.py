import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import maxflow


def graph_cut_segmentation(image, rect=None, n_components=5, lambda_param=50.0):
    """
    使用图割（Graph Cut）方法进行图像分割
    
    参数:
        image: 输入图像
        rect: 前景矩形区域 [x, y, width, height]，如果为None则使用交互式选择
        n_components: 高斯混合模型的组件数
        lambda_param: 平滑度参数，控制区域连贯性
    
    返回:
        mask: 分割掩码，前景为1，背景为0
    """
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3 and image.dtype == np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 如果没有提供矩形，使用交互式选择
    if rect is None:
        # 创建一个窗口用于交互式选择
        cv2.namedWindow('Select Foreground', cv2.WINDOW_NORMAL)
        rect = cv2.selectROI('Select Foreground', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.destroyAllWindows()
    
    # 创建初始掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 标记前景和背景区域
    # 前景: 矩形内部
    # 背景: 矩形外部边缘
    x, y, w, h = rect
    
    # 创建前景和背景的掩码
    fg_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bg_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 标记前景区域（矩形内部）
    fg_mask[y:y+h, x:x+w] = 1
    
    # 标记背景区域（图像边缘）
    border_size = 5
    bg_mask[:border_size, :] = 1
    bg_mask[-border_size:, :] = 1
    bg_mask[:, :border_size] = 1
    bg_mask[:, -border_size:] = 1
    
    # 确保背景区域不包含前景区域
    bg_mask = bg_mask & (~fg_mask)
    
    # 提取前景和背景像素
    fg_pixels = image[fg_mask == 1].reshape(-1, 3)
    bg_pixels = image[bg_mask == 1].reshape(-1, 3)
    
    # 使用高斯混合模型拟合前景和背景分布
    fg_gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    bg_gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    
    fg_gmm.fit(fg_pixels)
    bg_gmm.fit(bg_pixels)
    
    # 创建图结构
    rows, cols = image.shape[:2]
    graph = maxflow.Graph[float]()
    nodeids = graph.add_grid_nodes((rows, cols))
    
    # 添加区域项（t-links）
    img_flat = image.reshape(-1, 3)
    
    # 计算每个像素属于前景和背景的负对数概率
    fg_prob = -fg_gmm.score_samples(img_flat)
    bg_prob = -bg_gmm.score_samples(img_flat)
    
    # 重塑为图像形状
    fg_prob = fg_prob.reshape(rows, cols)
    bg_prob = bg_prob.reshape(rows, cols)
    
    # 添加t-links
    for i in range(rows):
        for j in range(cols):
            # 添加源点（前景）和汇点（背景）的边
            graph.add_tedge(nodeids[i, j], bg_prob[i, j], fg_prob[i, j])
    
    # 添加平滑项（n-links）
    # 定义4连通邻域
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i in range(rows):
        for j in range(cols):
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    # 计算像素差异
                    diff = np.sum((image[i, j] - image[ni, nj]) ** 2)
                    weight = lambda_param * np.exp(-diff / 50.0)
                    graph.add_edge(nodeids[i, j], nodeids[ni, nj], weight, weight)
    
    # 执行最大流算法
    flow = graph.maxflow()
    
    # 获取分割结果
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if graph.get_segment(nodeids[i, j]) == 0:  # 前景
                mask[i, j] = 1
    
    return mask


def display_segmentation_results(image, mask, gradient_magnitude=None):
    """
    显示分割结果
    
    参数:
        image: 原始图像
        mask: 分割掩码
        gradient_magnitude: 梯度幅值图像（可选）
    """
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3 and image.dtype == np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建分割结果可视化
    segmented = image.copy()
    segmented[mask == 0] = [0, 0, 0]  # 将背景设为黑色
    
    # 创建边界可视化
    boundary = np.zeros_like(image)
    
    # 找到分割边界
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel)
    mask_eroded = cv2.erode(mask, kernel)
    boundary_mask = mask_dilated - mask_eroded
    
    # 在边界上设置红色
    boundary[boundary_mask == 1] = [255, 0, 0]
    
    # 将边界叠加到原始图像上
    overlay = image.copy()
    overlay[boundary_mask == 1] = [255, 0, 0]
    
    if gradient_magnitude is not None:
        # 如果提供了梯度幅值图像，显示四张图
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title('Gradient Magnitude')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(segmented)
        plt.title('Segmentation Result')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(overlay)
        plt.title('Boundary Overlay')
        plt.axis('off')
    else:
        # 否则显示三张图
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(segmented)
        plt.title('Segmentation Result')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Boundary Overlay')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def compute_gradient_magnitude(image):
    """
    计算图像的梯度幅值
    
    参数:
        image: 输入图像
    
    返回:
        gradient_magnitude: 梯度幅值图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 计算x和y方向的梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 归一化到0-255范围
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return gradient_magnitude


def segment_image_with_graph_cut(image_path=None, image=None, rect=None, n_components=5, lambda_param=50.0):
    """
    使用图割方法分割图像并显示结果
    
    参数:
        image_path: 图像路径（与image二选一）
        image: 图像数组（与image_path二选一）
        rect: 前景矩形区域 [x, y, width, height]
        n_components: 高斯混合模型的组件数
        lambda_param: 平滑度参数
    """
    # 加载图像
    if image is None and image_path is not None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image is None and image_path is None:
        raise ValueError("必须提供image_path或image参数")
    
    # 计算梯度幅值
    gradient_magnitude = compute_gradient_magnitude(image)
    
    # 使用图割进行分割
    mask = graph_cut_segmentation(image, rect, n_components, lambda_param)
    
    # 显示结果
    display_segmentation_results(image, mask, gradient_magnitude)
    
    return mask


# 使用示例
# 方法1：提供图像路径，交互式选择前景区域
segment_image_with_graph_cut("images/Q2_1.png")

# 方法2：提供图像路径和前景矩形区域
# segment_image_with_graph_cut('images/Q2_1.png', rect=[60, 4, 343, 193])

# 方法3：提供图像数组
# image = cv2.imread('path_to_your_image.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# segment_image_with_graph_cut(image=image)
