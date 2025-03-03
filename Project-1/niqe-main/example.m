
 load modelparameters.mat
 
 blocksizerow    = 96;
 blocksizecol    = 96;
 blockrowoverlap = 0;
 blockcoloverlap = 0;

 % 获取文件夹中所有PNG文件
imageFiles = dir('images/*.png');

% 循环读取每一张图片
for k = 1:length(imageFiles)
    % 获取文件的完整路径
    imagePath = fullfile(imageFiles(k).folder, imageFiles(k).name);
    
    % 读取图片
    img = imread(imagePath);
    imageFiles(k).name

    quality = computequality(img,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)
    
    % 你可以在这里对img进行其他操作
end

% im =imread('image2.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('image3.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% 
% im =imread('image4.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)