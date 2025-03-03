import cv2
import numpy as np
import os
from metrics import metrics

# 定义多尺度Retinex算法（MF）
def retinex(image, sigma_list=[15, 80, 250]):
    retinex_image = np.zeros_like(image)
    for sigma in sigma_list:
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
        retinex_image += np.log1p(image / (blurred_image + 1e-6))
    retinex_image /= len(sigma_list)
    return np.exp(retinex_image) - 1

# 循环读取images目录中的所有图片
image_dir = 'images'
output_dir = 'processed/MF'
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

    # 应用Retinex模型
    mf_corrected = retinex(image)

    # 将图像转换回0-255的范围
    mf_corrected = np.uint8(np.clip(mf_corrected * 255, 0, 255))

    # 保存结果为图片
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, mf_corrected)
    
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
