import cv2
import numpy as np
import os
from metrics import metrics

# 循环读取images目录中的所有图片
image_dir = 'images'
output_dir = 'processed/gamma'
gt_dir = 'gt'
os.makedirs(output_dir, exist_ok=True)
images = []
for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        images.append((filename, image))

# 创建结果文件
result_file = os.path.join(output_dir, 'mae_results.txt')
with open(result_file, 'w') as f:
    f.write("文件名\tMAE值\tGMSD值\n")

for filename, image in images:
    # 将图像转换为浮动类型（0到1之间）
    image = image / 255.0

    # 设置Gamma值
    gamma = 0.5

    # 进行Gamma变换
    gamma_corrected = np.power(image, gamma)

    # 将图像转换回0-255的范围
    gamma_corrected = np.uint8(gamma_corrected * 255)

    # 保存结果为图片
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, gamma_corrected)
    
    # 计算与gt图片的MAE和GMSD
    gt_path = os.path.join(gt_dir, filename)
    if os.path.exists(gt_path):
        mae_value = metrics(output_path, gt_path, "MAE")
        gmsd_value = metrics(output_path, gt_path, "GMSD")
        # 将结果写入文件
        with open(result_file, 'a') as f:
            f.write(f"{filename}\t{mae_value}\t{gmsd_value}\n")
    else:
        print(f"警告：找不到对应的gt图片 {filename}")
