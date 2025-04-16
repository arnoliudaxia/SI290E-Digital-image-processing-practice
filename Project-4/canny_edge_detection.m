function edge_map = canny_edge_detection(image_path, low_threshold, high_threshold, sigma)
% CANNY_EDGE_DETECTION 实现Canny边缘检测算法
% 参数:
%   image_path: 输入图像路径
%   low_threshold: 低阈值
%   high_threshold: 高阈值
%   sigma: 高斯滤波器的标准差
%
% 返回:
%   edge_map: 边缘检测结果

% 1. 读取图像
if ischar(image_path)
    img = imread(image_path);
else
    img = image_path;
end

% 如果是彩色图，转换为灰度图
if size(img, 3) > 1
    img = rgb2gray(img);
end

% 转换为double类型，方便计算
img = double(img);

% 2. 高斯平滑
% 生成高斯滤波器
filter_size = 2 * ceil(3 * sigma) + 1; % 滤波器大小，保证覆盖3个标准差
gauss_filter = fspecial_gaussian(filter_size, sigma);

% 应用高斯滤波
smoothed_img = conv2(img, gauss_filter, 'same');

% 3. 计算梯度幅值和方向
% 定义Sobel算子
sobel_x = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
sobel_y = [1, 2, 1; 0, 0, 0; -1, -2, -1];

% 计算x和y方向的梯度
gradient_x = conv2(smoothed_img, sobel_x, 'same');
gradient_y = conv2(smoothed_img, sobel_y, 'same');

% 计算梯度幅值和方向
gradient_magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
gradient_direction = atan2(gradient_y, gradient_x) * 180 / pi;
% 将角度转换为0-180度范围
gradient_direction = mod(gradient_direction, 180);

% 4. 非极大值抑制
[rows, cols] = size(gradient_magnitude);
suppressed = zeros(rows, cols);

for i = 2:rows-1
    for j = 2:cols-1
        % 获取当前像素的梯度方向
        direction = gradient_direction(i, j);
        
        % 将方向量化为0，45，90，135度
        if (direction >= 0 && direction < 22.5) || (direction >= 157.5 && direction < 180)
            neighbors = [gradient_magnitude(i, j+1), gradient_magnitude(i, j-1)];
        elseif (direction >= 22.5 && direction < 67.5)
            neighbors = [gradient_magnitude(i-1, j+1), gradient_magnitude(i+1, j-1)];
        elseif (direction >= 67.5 && direction < 112.5)
            neighbors = [gradient_magnitude(i-1, j), gradient_magnitude(i+1, j)];
        else
            neighbors = [gradient_magnitude(i-1, j-1), gradient_magnitude(i+1, j+1)];
        end
        
        % 如果当前像素的梯度幅值大于其梯度方向上的邻居，则保留
        if gradient_magnitude(i, j) >= max(neighbors)
            suppressed(i, j) = gradient_magnitude(i, j);
        end
    end
end

% 5. 双阈值处理和滞后阈值处理
edge_map = zeros(rows, cols);
strong_edges = suppressed >= high_threshold;
weak_edges = suppressed >= low_threshold & suppressed < high_threshold;

% 标记强边缘
edge_map(strong_edges) = 1;

% 连接弱边缘
for i = 2:rows-1
    for j = 2:cols-1
        if weak_edges(i, j)
            % 检查8邻域内是否有强边缘
            neighborhood = edge_map(i-1:i+1, j-1:j+1);
            if any(neighborhood(:))
                edge_map(i, j) = 1;
            end
        end
    end
end

end

function filter = fspecial_gaussian(size, sigma)
% 自定义函数创建高斯滤波器，替代 fspecial('gaussian')
% 参数:
%   size: 滤波器大小
%   sigma: 高斯函数的标准差
%
% 返回:
%   filter: 高斯滤波器

% 确保size是奇数
if mod(size, 2) == 0
    size = size + 1;
end

% 计算滤波器的中心
center = (size - 1) / 2;

% 创建网格
[x, y] = meshgrid(-center:center, -center:center);

% 计算高斯函数
filter = exp(-(x.^2 + y.^2) / (2 * sigma^2));

% 归一化滤波器，使得权重和为1
filter = filter / sum(filter(:));

end 