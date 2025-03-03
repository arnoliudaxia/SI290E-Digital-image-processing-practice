import numpy as np
import argparse
import cv2
import glob
import os

from metrics import metrics


def ldr(src_data, alpha, U):
    # % -------------------------------------------------------------------------
    # % An implementation of
    # %   C. Lee, C. Lee, and Chang-Su Kim, "Contrast enhancement based on
    # %   layered difference representation of 2D histograms," IEEE Trans. Image
    # %   Image Process., vol. 22, no. 12, pp. 5372-5384, Dec. 2013
    # % -------------------------------------------------------------------------
    # % Input variables (see the paper for details)
    # %   src_data : can be either 2D histogram or gray scale image. This script
    # %   automatically detects based on its dimension.
    # %   alpha    : controls the level of enhancement
    # %   U        : U matrix in Equation (31). If it is provided, we can save
    # %   the computation time.
    # % Output variables
    # %   x    : Output transformation function.
    # % -------------------------------------------------------------------------
    # %                           written by Chulwoo Lee, chulwoo@mcl.korea.ac.kr

    R, C = src_data.shape
    if R == 255 and C == 255:
        h2D_in = src_data
    else:
        in_Y = src_data
        
        # % unordered 2D histogram acquisition
        h2D_in = np.zeros((256, 256))
        
        for j in range(1, R + 1):
            for i in range(1, R + 1):
                ref = in_Y[j - 1, i - 1]
                
                if j != R:
                    trg = in_Y[j, i - 1]
                if i != C:
                    trg = in_Y[j - 1, i]
                
                h2D_in[np.maximum(trg, ref), np.minimum(trg, ref)] += 1
        del ref, trg

    # Intra-Layer Optimization
    D = np.zeros((255, 255))
    s = np.zeros((255, 1))

    # iteration start
    for layer in range(1, 255):
        h_l = np.zeros((256 - layer, 1))
        tmp_idx = 1
        for j in range(1 + layer, 257):
            i = j - layer
            h_l[tmp_idx - 1, 0] = np.log(h2D_in[j - 1, i - 1] + 1)
            tmp_idx += 1
        del tmp_idx

        s[layer - 1, 0] = np.sum(h_l)

        # % if all elements in h_l are zero, then skip
        if s[layer - 1, 0] == 0:
            continue

        # % Convolution
        m_l = np.convolve(np.squeeze(h_l), np.ones((layer,)))  # % Equation (30)
        
        d_l = (m_l - np.amin(m_l)) / U[:, layer - 1]  # % Equation (33)

        if np.sum(d_l) == 0:
            continue

        D[:, layer - 1] = d_l / np.sum(d_l)

    W = (s / np.amax(s)) ** alpha  # % Equation (23)
    d = np.matmul(D, W)  # % Equation (24)

    d = d / np.sum(d)  # % normalization
    tmp = np.zeros((256, 1))
    for k in range(1, 255):
        tmp[k] = tmp[k - 1] + d[k - 1]

    x = (255 * tmp).astype(np.uint8)
    return x


# Image directory and output directory setup
image_dir = 'images'
output_dir = 'processed/LDR'
gt_dir = 'gt'

os.makedirs(output_dir, exist_ok=True)

# Pre-computing U matrix
U = np.zeros((255, 255))
tmp_k = np.array(range(1, 256))
for layer in range(1, 256):
    U[:, layer - 1] = np.minimum(tmp_k, 256 - layer) - np.maximum(tmp_k - layer, 0)

alpha = 2.5  # Enhancement level

# Process images
for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        # 分别处理RGB三个通道
        channels = cv2.split(img)
        enhanced_channels = []
        for channel in channels:
            # Apply the LDR transformation
            transf_fn = ldr(channel, alpha, U)

            # Apply the transformation
            out = transf_fn[channel]
            out = np.squeeze(out)
            enhanced_channels.append(out)

        # 合并处理后的通道
        enhanced_img = cv2.merge(enhanced_channels)

        # Save the result to the output directory
        output_path = os.path.join(output_dir, f"enhanced_{filename}")
        
        # numpy_horizontal = np.hstack((img, out, out_he))
        # cv2.imwrite(output_path, numpy_horizontal)
        
        cv2.imwrite(output_path, enhanced_img)
        gt_path = os.path.join(gt_dir, filename)
			
        mae_value = metrics(gt_path, output_path, "MAE")
        gmsd_value = metrics(gt_path, output_path, "GMSD")
        
        # 将指标保存到文件中
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_path, 'a') as f:
            f.write(f"{filename}: MAE={mae_value:.4f}, GMSD={gmsd_value:.4f}\n")
