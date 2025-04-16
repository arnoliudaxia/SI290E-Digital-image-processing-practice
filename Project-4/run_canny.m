% 运行Canny边缘检测算法
clear all;
close all;

% 读取图像
image_path = 'images/Q1.tif';

% 设置参数
low_threshold = 20;  % 低阈值
high_threshold = 60; % 高阈值
sigma = 1.0;         % 高斯滤波器的标准差

% 运行Canny边缘检测
edge_map = canny_edge_detection(image_path, low_threshold, high_threshold, sigma);

% 显示结果
figure;
subplot(1, 2, 1);
imshow(imread(image_path));
title('原始图像');

subplot(1, 2, 2);
imshow(edge_map);
title('Canny边缘检测结果');

% 保存结果
imwrite(edge_map, 'canny_result.png');
fprintf('边缘检测完成，结果已保存为canny_result.png\n'); 