import numpy as np
import cv2
import matplotlib.pyplot as plt
from canny_edge_detection import canny_edge_detection, calculate_iou

def evaluate_with_ground_truth():
    """
    使用标准检测结果与真实边缘图计算IoU
    """
    # 读取图像
    image_path = 'images/Q1.tif'
    
    # 读取或创建真实边缘图（如果有的话）
    # 注意：在实际应用中，您需要提供真实的边缘图
    # 这里仅作为示例，我们用OpenCV的Canny边缘检测结果作为"真实边缘图"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.Canny(img, 50, 150)
    ground_truth = ground_truth / 255  # 转换为二值图像
    
    # 保存真实边缘图，供后续使用
    cv2.imwrite('ground_truth.png', ground_truth * 255)
    
    # 使用不同参数组合运行我们的Canny边缘检测
    parameter_sets = [
        {'low': 10, 'high': 30, 'sigma': 1.0},
        {'low': 20, 'high': 60, 'sigma': 1.0},
        {'low': 30, 'high': 90, 'sigma': 1.0},
        {'low': 40, 'high': 120, 'sigma': 1.0},
        {'low': 20, 'high': 60, 'sigma': 1.5},
        {'low': 20, 'high': 60, 'sigma': 2.0}
    ]
    
    results = []
    
    for params in parameter_sets:
        # 运行我们的Canny边缘检测
        edge_map = canny_edge_detection(
            image_path, 
            params['low'], 
            params['high'], 
            params['sigma']
        )
        
        # 计算IoU
        iou = calculate_iou(edge_map, ground_truth)
        
        # 保存结果
        results.append({
            'params': params,
            'edge_map': edge_map,
            'iou': iou
        })
        
        print(f"参数: 低阈值={params['low']}, 高阈值={params['high']}, sigma={params['sigma']}, IoU: {iou:.4f}")
    
    # 找出最佳参数组合
    best_result = max(results, key=lambda x: x['iou'])
    print(f"\n最佳参数组合: 低阈值={best_result['params']['low']}, 高阈值={best_result['params']['high']}, sigma={best_result['params']['sigma']}")
    print(f"最佳IoU: {best_result['iou']:.4f}")
    
    # 可视化最佳结果与真实边缘图比较
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title('原始图像')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('真实边缘图')
    
    plt.subplot(1, 3, 3)
    plt.imshow(best_result['edge_map'], cmap='gray')
    best_params = best_result['params']
    plt.title(f"最佳检测结果 (IoU: {best_result['iou']:.4f})\n低阈值={best_params['low']}, 高阈值={best_params['high']}, sigma={best_params['sigma']}")
    
    plt.tight_layout()
    plt.savefig('iou_comparison.png')
    plt.show()
    
    # 可视化所有参数组合的结果
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(2, 3, i+1)
        plt.imshow(result['edge_map'], cmap='gray')
        params = result['params']
        plt.title(f"低阈值={params['low']}, 高阈值={params['high']}, sigma={params['sigma']}\nIoU: {result['iou']:.4f}")
    
    plt.tight_layout()
    plt.savefig('all_params_comparison.png')
    plt.show()

if __name__ == "__main__":
    evaluate_with_ground_truth()
    print("评估完成，结果已保存为'iou_comparison.png'和'all_params_comparison.png'") 