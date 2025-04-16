import numpy as np
import cv2
import matplotlib.pyplot as plt
from canny_edge_detection import calculate_iou

def compute_and_display_iou(predicted_edge_path, ground_truth_path):
    """
    计算并显示两个边缘图之间的IoU
    
    参数:
        predicted_edge_path: 预测的边缘图路径
        ground_truth_path: 真实的边缘图路径
    """
    # 读取图像
    predicted_edge = cv2.imread(predicted_edge_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    
    # 确保二值化
    if predicted_edge.dtype != bool:
        predicted_edge = predicted_edge > 0
    if ground_truth.dtype != bool:
        ground_truth = ground_truth > 0
    
    # 计算IoU
    iou = calculate_iou(predicted_edge, ground_truth)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(predicted_edge, cmap='gray')
    plt.title('预测的边缘图')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('真实的边缘图')
    
    # 创建一个可视化图像，显示交集和差异
    comparison = np.zeros((predicted_edge.shape[0], predicted_edge.shape[1], 3), dtype=np.uint8)
    # 真实边缘为绿色
    comparison[ground_truth] = [0, 255, 0]
    # 预测边缘为红色
    comparison[predicted_edge] = [255, 0, 0]
    # 交集为黄色
    comparison[np.logical_and(predicted_edge, ground_truth)] = [255, 255, 0]
    
    plt.subplot(1, 3, 3)
    plt.imshow(comparison)
    plt.title(f'比较 (IoU: {iou:.4f})\n黄色: 交集, 绿色: 仅GT, 红色: 仅预测')
    
    plt.tight_layout()
    plt.savefig('iou_comparison_result.png')
    plt.show()
    
    print(f'IoU: {iou:.4f}')
    print('可视化结果已保存为iou_comparison_result.png')
    
    return iou

if __name__ == "__main__":
    # 这里替换为实际的边缘图路径
    predicted_edge_path = 'canny_result.png'  # 您的预测边缘图路径
    ground_truth_path = 'ground_truth.png'    # 您的真实边缘图路径
    
    # 检查文件是否存在，如果不存在则提示用户
    import os
    if not os.path.exists(predicted_edge_path):
        print(f"错误: 找不到预测边缘图文件 '{predicted_edge_path}'")
        print("请先运行canny_edge_detection.py生成边缘检测结果，或提供正确的文件路径")
        exit(1)
    
    if not os.path.exists(ground_truth_path):
        print(f"错误: 找不到真实边缘图文件 '{ground_truth_path}'")
        print("请提供正确的真实边缘图路径")
        exit(1)
    
    # 计算IoU
    iou = compute_and_display_iou(predicted_edge_path, ground_truth_path) 