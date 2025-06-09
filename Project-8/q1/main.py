import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import glob

def preprocess_image_detailed(img):
    """预处理图像，针对中国蓝色车牌进行颜色过滤，返回详细的中间结果"""
    
    # Stage 1: 转换到HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Stage 2: 定义蓝色车牌的HSV范围并创建掩码
    lower_blue = np.array([100, 70, 70])   # 较低的蓝色阈值
    upper_blue = np.array([130, 255, 255]) # 较高的蓝色阈值
    blue_mask_raw = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Stage 3: 形态学操作去除噪声 - 开运算
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue_mask_opened = cv2.morphologyEx(blue_mask_raw, cv2.MORPH_OPEN, kernel_noise)
    
    # Stage 4: 形态学操作去除噪声 - 闭运算
    blue_mask_final = cv2.morphologyEx(blue_mask_opened, cv2.MORPH_CLOSE, kernel_noise)
    
    # Stage 5: 应用颜色掩码到原图
    blue_filtered = cv2.bitwise_and(img, img, mask=blue_mask_final)
    
    # Stage 6: 转换为灰度图
    gray = cv2.cvtColor(blue_filtered, cv2.COLOR_BGR2GRAY)
    
    # Stage 7: 高斯模糊降噪
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Stage 8: 对比度增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(smooth)

    # Stage 9: Canny边缘检测
    edges_raw = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    
    # Stage 10: 形态学操作连接断开的边缘
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_final = cv2.morphologyEx(edges_raw, cv2.MORPH_CLOSE, kernel_edge)

    return {
        'original': img,
        'hsv': hsv,
        'blue_mask_raw': blue_mask_raw,
        'blue_mask_opened': blue_mask_opened,
        'blue_mask_final': blue_mask_final,
        'blue_filtered': blue_filtered,
        'gray': gray,
        'smooth': smooth,
        'enhanced': enhanced,
        'edges_raw': edges_raw,
        'edges_final': edges_final
    }

def localize_plate_detailed(img, preprocessed_results):
    """定位车牌，结合颜色信息和形状特征，返回详细的中间结果"""
    
    preprocessed = preprocessed_results['edges_final']
    blue_mask = preprocessed_results['blue_mask_final']
    
    # Stage 1: 水平形态学操作 - 连接字符
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    morph_horizontal = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel_horizontal)
    
    # Stage 2: 垂直形态学操作 - 填充字符内部
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    morph_vertical = cv2.morphologyEx(morph_horizontal, cv2.MORPH_CLOSE, kernel_vertical)
    
    # Stage 3: 膨胀操作 - 确保车牌区域完整
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_dilated = cv2.dilate(morph_vertical, kernel_dilate, iterations=2)
    
    # Stage 4: 蓝色掩码的形态学处理
    kernel_blue = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blue_morph_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_blue)
    blue_morph_final = cv2.dilate(blue_morph_closed, kernel_dilate, iterations=1)
    
    # Stage 5: 组合边缘信息和颜色信息
    combined = cv2.bitwise_or(morph_dilated, blue_morph_final)

    # Stage 6: 轮廓检测和筛选
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h_img, w_img = img.shape[:2]

    # 创建候选区域可视化图像
    candidates_img = img.copy()
    for i, cnt in enumerate(contours):
        # 计算边界框
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 计算基本特征
        aspect_ratio = w / float(h)
        area = w * h
        
        # 检查该区域内的蓝色像素比例
        roi_blue_mask = blue_mask[y:y+h, x:x+w]
        blue_pixel_ratio = np.sum(roi_blue_mask > 0) / (w * h) if w * h > 0 else 0
        
        # 绘制所有轮廓（灰色）
        cv2.rectangle(candidates_img, (x, y), (x+w, y+h), (128, 128, 128), 1)
        cv2.putText(candidates_img, f'{i}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        print(aspect_ratio)
        
        if aspect_ratio > 2.0 and aspect_ratio < 6.0:
            candidates.append((x, y, w, h, area, aspect_ratio, blue_pixel_ratio))
        
    
    # 按面积排序，选择面积最大的候选区域
    if candidates:
        candidates.sort(key=lambda x: x[4], reverse=True)  # 按面积(index 4)降序排序
        best_candidate = candidates[0]  # 选择面积最大的
        x, y, w, h = best_candidate[:4]
        
        # 绘制选中的候选区域（蓝色）
        cv2.rectangle(candidates_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(candidates_img, 'LARGEST', (x, y-10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 重新设置candidates列表，只包含选中的候选区域
        candidates = [best_candidate]

    # 选择最佳候选区域
    plate_img = None
    plate_rect = None
    final_detection_img = img.copy()
    
    if candidates:
        def score_candidate(candidate):
            x, y, w, h, area, ar, blue_ratio = candidate
            area_score = min(area / 15000, 15000 / area) if area > 0 else 0
            ar_score = 1 / (1 + abs(ar - 3.5))
            pos_score = 1 if y > h_img * 0.3 else 0.5
            blue_score = min(blue_ratio * 5, 1.0)
            return area_score * ar_score * pos_score * blue_score

        best_candidate = max(candidates, key=score_candidate)
        x, y, w, h = best_candidate[:4]
        
        # 扩展边界框
        margin_x = int(w * 0.01)
        margin_y = int(h * 0.01)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(w_img - x, w + 2 * margin_x)
        h = min(h_img - y, h + 2 * margin_y)
        
        plate_rect = (x, y, w, h)
        plate_img = img[y:y+h, x:x+w]
        
        # 绘制最终检测结果（红色）
        cv2.rectangle(final_detection_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(final_detection_img, 'DETECTED', (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return {
        'morph_horizontal': morph_horizontal,
        'morph_vertical': morph_vertical,
        'morph_dilated': morph_dilated,
        'blue_morph_closed': blue_morph_closed,
        'blue_morph_final': blue_morph_final,
        'combined': combined,
        'candidates_img': candidates_img,
        'final_detection_img': final_detection_img,
        'plate_img': plate_img,
        'plate_rect': plate_rect,
        'candidates': candidates
    }

def enhance_plate_image_detailed(plate_img):
    """对提取的车牌图像进行进一步增强，返回详细步骤"""
    if plate_img is None:
        return None
    
    # Stage 1: 转为灰度图
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
    
    # Stage 2: 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Stage 3: 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return {
        'original_plate': plate_img,
        'gray': gray,
        'enhanced': enhanced,
        'sharpened': sharpened
    }

def save_detailed_visualization(preprocess_results, localize_results, enhance_results, output_path):
    """保存详细的可视化结果"""
    
    plt.ioff()
    
    # 创建大图 - 5行6列，显示30个步骤
    fig = plt.figure(figsize=(24, 20))
    
    # 第一行：预处理步骤 1-6
    plt.subplot(5, 6, 1)
    img_rgb = cv2.cvtColor(preprocess_results['original'], cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('1. Original Image', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 2)
    plt.imshow(preprocess_results['hsv'])
    plt.title('2. HSV Color Space', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 3)
    plt.imshow(preprocess_results['blue_mask_raw'], cmap='gray')
    plt.title('3. Raw Blue Mask', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 4)
    plt.imshow(preprocess_results['blue_mask_opened'], cmap='gray')
    plt.title('4. After Opening', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 5)
    plt.imshow(preprocess_results['blue_mask_final'], cmap='gray')
    plt.title('5. After Closing', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 6)
    blue_filtered_rgb = cv2.cvtColor(preprocess_results['blue_filtered'], cv2.COLOR_BGR2RGB)
    plt.imshow(blue_filtered_rgb)
    plt.title('6. Blue Filtered', fontsize=10)
    plt.axis('off')
    
    # 第二行：预处理步骤 7-12
    plt.subplot(5, 6, 7)
    plt.imshow(preprocess_results['gray'], cmap='gray')
    plt.title('7. Grayscale', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 8)
    plt.imshow(preprocess_results['smooth'], cmap='gray')
    plt.title('8. Gaussian Blur', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 9)
    plt.imshow(preprocess_results['enhanced'], cmap='gray')
    plt.title('9. CLAHE Enhanced', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 10)
    plt.imshow(preprocess_results['edges_raw'], cmap='gray')
    plt.title('10. Raw Canny Edges', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 11)
    plt.imshow(preprocess_results['edges_final'], cmap='gray')
    plt.title('11. Final Edges', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 12)
    plt.imshow(localize_results['morph_horizontal'], cmap='gray')
    plt.title('12. Horizontal Morph', fontsize=10)
    plt.axis('off')
    
    # 第三行：定位步骤 13-18
    plt.subplot(5, 6, 13)
    plt.imshow(localize_results['morph_vertical'], cmap='gray')
    plt.title('13. Vertical Morph', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 14)
    plt.imshow(localize_results['morph_dilated'], cmap='gray')
    plt.title('14. Dilated Morph', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 15)
    plt.imshow(localize_results['blue_morph_closed'], cmap='gray')
    plt.title('15. Blue Morph Closed', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 16)
    plt.imshow(localize_results['blue_morph_final'], cmap='gray')
    plt.title('16. Blue Morph Final', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 17)
    plt.imshow(localize_results['combined'], cmap='gray')
    plt.title('17. Combined Mask', fontsize=10)
    plt.axis('off')
    
    plt.subplot(5, 6, 18)
    candidates_rgb = cv2.cvtColor(localize_results['candidates_img'], cv2.COLOR_BGR2RGB)
    plt.imshow(candidates_rgb)
    plt.title(f'18. All Candidates ({len(localize_results["candidates"])})', fontsize=10)
    plt.axis('off')
    
    # 第四行：检测结果和车牌增强 19-24
    plt.subplot(5, 6, 19)
    final_rgb = cv2.cvtColor(localize_results['final_detection_img'], cv2.COLOR_BGR2RGB)
    plt.imshow(final_rgb)
    plt.title('19. Final Detection', fontsize=10)
    plt.axis('off')
    
    if enhance_results:
        plt.subplot(5, 6, 20)
        if len(enhance_results['original_plate'].shape) == 3:
            plate_rgb = cv2.cvtColor(enhance_results['original_plate'], cv2.COLOR_BGR2RGB)
            plt.imshow(plate_rgb)
        else:
            plt.imshow(enhance_results['original_plate'], cmap='gray')
        plt.title('20. Extracted Plate', fontsize=10)
        plt.axis('off')
        
        plt.subplot(5, 6, 21)
        plt.imshow(enhance_results['gray'], cmap='gray')
        plt.title('21. Plate Grayscale', fontsize=10)
        plt.axis('off')
        
        plt.subplot(5, 6, 22)
        plt.imshow(enhance_results['enhanced'], cmap='gray')
        plt.title('22. Plate CLAHE', fontsize=10)
        plt.axis('off')
        
        plt.subplot(5, 6, 23)
        plt.imshow(enhance_results['sharpened'], cmap='gray')
        plt.title('23. Plate Sharpened', fontsize=10)
        plt.axis('off')
    else:
        for i in range(20, 24):
            plt.subplot(5, 6, i)
            plt.text(0.5, 0.5, 'No Plate\nDetected', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{i}. N/A', fontsize=10)
            plt.axis('off')
    
    # 第五行：统计信息和参数 24-30
    for i in range(24, 31):
        plt.subplot(5, 6, i)
        plt.axis('off')
        
        if i == 24:
            info_text = "HSV Blue Range:\n"
            info_text += "H: 100-130\n"
            info_text += "S: 50-255\n"
            info_text += "V: 50-255\n\n"
            info_text += "Morphology Kernels:\n"
            info_text += "• Noise: 3×3 ellipse\n"
            info_text += "• Horizontal: 25×3\n"
            info_text += "• Vertical: 3×15"
            plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            plt.title('24. Color Parameters', fontsize=10)
            
        elif i == 25:
            filter_text = "Candidate Filters:\n"
            filter_text += "• Aspect Ratio: 2.0-6.0\n"
            filter_text += "• Area: 3000-80000\n"
            filter_text += "• Extent: >0.2\n"
            filter_text += "• Solidity: >0.6\n"
            filter_text += "• Min Size: 80×20\n"
            filter_text += "• Blue Ratio: >0.1\n\n"
            filter_text += "CLAHE Parameters:\n"
            filter_text += "• Clip Limit: 3.0\n"
            filter_text += "• Tile Size: 8×8"
            plt.text(0.1, 0.9, filter_text, transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
            plt.title('25. Filter Parameters', fontsize=10)
            
        elif i == 26:
            canny_text = "Edge Detection:\n"
            canny_text += "• Low Threshold: 50\n"
            canny_text += "• High Threshold: 150\n"
            canny_text += "• Aperture: 3\n\n"
            canny_text += "Gaussian Blur:\n"
            canny_text += "• Kernel Size: 5×5\n"
            canny_text += "• Sigma: 0 (auto)\n\n"
            canny_text += "Sharpening Kernel:\n"
            canny_text += "[-1 -1 -1]\n"
            canny_text += "[-1  9 -1]\n"
            canny_text += "[-1 -1 -1]"
            plt.text(0.1, 0.9, canny_text, transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            plt.title('26. Processing Parameters', fontsize=10)
            
        elif i == 27:
            if localize_results['candidates']:
                stats_text = "Candidate Details:\n"
                for j, candidate in enumerate(localize_results['candidates'][:3]):
                    x, y, w, h, area, ar, blue_ratio = candidate
                    stats_text += f"{j+1}. Pos:({x},{y})\n"
                    stats_text += f"   Size:{w}×{h}\n"
                    stats_text += f"   AR:{ar:.2f} Blue:{blue_ratio:.2f}\n"
                if len(localize_results['candidates']) > 3:
                    stats_text += f"... and {len(localize_results['candidates'])-3} more"
            else:
                stats_text = "No candidates found"
            plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
            plt.title('27. Candidate Stats', fontsize=10)
            
        elif i == 28:
            img_stats = f"Image Info:\n"
            img_stats += f"Size: {preprocess_results['original'].shape[1]}×{preprocess_results['original'].shape[0]}\n"
            img_stats += f"Channels: {preprocess_results['original'].shape[2]}\n"
            img_stats += f"Candidates: {len(localize_results['candidates'])}\n"
            if localize_results['plate_rect']:
                x, y, w, h = localize_results['plate_rect']
                img_stats += f"Plate: {w}×{h}\n"
                img_stats += f"Position: ({x},{y})\n"
                img_stats += f"Aspect Ratio: {w/h:.2f}"
            else:
                img_stats += "No plate detected"
            plt.text(0.1, 0.9, img_stats, transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink", alpha=0.8))
            plt.title('28. Image Statistics', fontsize=10)
            
        elif i == 29:
            process_text = "Processing Steps:\n"
            process_text += "1-2: Color space conversion\n"
            process_text += "3-5: Blue color filtering\n"
            process_text += "6-11: Preprocessing\n"
            process_text += "12-17: Morphological ops\n"
            process_text += "18-19: Candidate detection\n"
            process_text += "20-23: Plate enhancement\n\n"
            process_text += "Total: 23 processing steps"
            plt.text(0.1, 0.9, process_text, transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.8))
            plt.title('29. Process Overview', fontsize=10)
            
        elif i == 30:
            success_text = "Detection Result:\n"
            if localize_results['plate_img'] is not None:
                success_text += "✓ SUCCESS\n\n"
                success_text += "Plate successfully\ndetected and extracted\n\n"
                success_text += "Ready for OCR\nprocessing"
                color = "lightgreen"
            else:
                success_text += "✗ FAILED\n\n"
                success_text += "No license plate\ndetected in image\n\n"
                success_text += "Try adjusting\nparameters"
                color = "lightcoral"
            plt.text(0.1, 0.9, success_text, transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
            plt.title('30. Final Result', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def detect_license_plate_detailed(img_path, output_dir=None):
    """完整的车牌检测流程，显示详细的中间步骤"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None
    
    # 获取文件名
    filename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\nProcessing {filename}...")
    print(f"Image size: {img.shape}")
    
    # 详细预处理
    preprocess_results = preprocess_image_detailed(img)
    print("✓ Preprocessing completed - 11 steps")
    
    # 详细车牌定位
    localize_results = localize_plate_detailed(img, preprocess_results)
    print("✓ Localization completed - 8 steps")
    
    # 详细车牌增强
    enhance_results = None
    if localize_results['plate_img'] is not None:
        enhance_results = enhance_plate_image_detailed(localize_results['plate_img'])
        print("✓ Enhancement completed - 4 steps")
        print(f"Plate detected at position: {localize_results['plate_rect']}")
        print(f"Plate size: {localize_results['plate_img'].shape}")
        
        # 保存单独的车牌图像
        if output_dir:
            plate_output_path = os.path.join(output_dir, f"{filename}_plate.jpg")
            cv2.imwrite(plate_output_path, localize_results['plate_img'])
            
            enhanced_output_path = os.path.join(output_dir, f"{filename}_enhanced_plate.jpg")
            cv2.imwrite(enhanced_output_path, enhance_results['sharpened'])
            print(f"Plate images saved")
    else:
        print("✗ No license plate detected")
    
    print(f"Number of candidates: {len(localize_results['candidates'])}")
    
    # 保存详细可视化结果
    if output_dir:
        detailed_output_path = os.path.join(output_dir, f"{filename}_detailed_analysis.jpg")
        save_detailed_visualization(preprocess_results, localize_results, enhance_results, detailed_output_path)
        print(f"Detailed analysis saved: {detailed_output_path}")
    
    return {
        'filename': filename,
        'preprocess_results': preprocess_results,
        'localize_results': localize_results,
        'enhance_results': enhance_results
    }

def batch_process_images_detailed(data_dir='data', output_dir='results_detailed'):
    """批量处理data目录下的所有图片，显示详细中间步骤"""
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {data_dir}")
        return
    
    print(f"Found {len(image_files)} image files in {data_dir}")
    print("=" * 80)
    
    # 统计结果
    successful_detections = 0
    total_images = len(image_files)
    
    # 处理每张图片
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{total_images}] Processing: {os.path.basename(img_path)}")
        
        try:
            result = detect_license_plate_detailed(img_path, output_dir)
            if result and result['localize_results']['plate_img'] is not None:
                successful_detections += 1
                print("✓ License plate detected successfully")
            else:
                print("✗ No license plate detected")
                
        except Exception as e:
            print(f"✗ Error processing {img_path}: {str(e)}")
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("DETAILED BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {total_images}")
    print(f"Successful detections: {successful_detections}")
    print(f"Detection rate: {successful_detections/total_images*100:.1f}%")
    print(f"Results saved in: {output_dir}")
    print("\nOutput files per image:")
    print("• {filename}_detailed_analysis.jpg - Complete 30-step analysis")
    print("• {filename}_plate.jpg - Extracted plate (if detected)")
    print("• {filename}_enhanced_plate.jpg - Enhanced plate (if detected)")
    print("=" * 80)

# 主程序
if __name__ == "__main__":
    # 批量处理data目录下的所有图片，显示详细中间步骤
    batch_process_images_detailed('data', 'results_detailed')
