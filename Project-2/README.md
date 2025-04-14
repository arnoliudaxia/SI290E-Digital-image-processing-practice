# 图像去噪算法比较研究

本项目实现并比较了多种图像去噪算法，包括均值滤波、小波变换、K-SVD和BM3D等方法，针对高斯噪声和散斑噪声进行了实验分析。

## 项目结构

```
.
├── BM3D.py                # BM3D算法实现
├── main.ipynb            # 主要实验代码和结果分析
├── images/              # 测试图像
│   ├── barbara.png
│   └── boat.png
├── output/              # 不同算法处理结果
├── references/          # 参考论文
│   ├── BM3D论文
│   ├── 自适应小波阈值论文
│   └── K-SVD论文
└── Results/             # 实验结果对比图
```

## 实现的去噪算法

1. 均值滤波 (Mean Filter)
   - 3x3 窗口
   - 5x5 窗口

2. 小波变换去噪 (Wavelet Transform)
   - 硬阈值
   - 软阈值

3. K-SVD 字典学习去噪

4. BM3D (Block-Matching and 3D filtering)

## 噪声类型

- 高斯噪声 (σ = 0.01, 0.05)
- 散斑噪声 (σ = 0.01, 0.05)

## 测试图像

- Barbara
- Boat

## 使用方法

1. 安装依赖：
```bash
pip install numpy opencv-python scipy matplotlib jupyter
```

2. 运行实验：
   - 打开 `main.ipynb`
   - 按顺序运行所有单元格

## 结果

处理结果保存在 `output/` 目录下，命名格式为：
`[算法名]_[噪声类型]_[噪声强度]_[图像名].png`

## 参考文献

1. Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering
2. Adaptive Wavelet Thresholding for Image Denoising and Compression
3. Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries