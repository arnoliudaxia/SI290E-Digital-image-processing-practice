function metrics()
Y1 = "./gt/1.png";
Y2 = "./res/1_gamma.png";

% GMSD - measure the image quality of distorted image 'Y2' with the reference image 'Y1'.
% 
% inputs:
% 
% Y1 - the reference image (grayscale image, double type, 0~255)
% Y2 - the distorted image (grayscale image, double type, 0~255)
% 
% outputs:

% score: distortion degree of the distorted image
% quality_map: local quality map of the distorted image

% This is an implementation of the following algorithm:
% Wufeng Xue, Lei Zhang, Xuanqin Mou, and Alan C. Bovik, 
% "Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index",
% http://www.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm

ipic=imread(Y1);
Y1=double(ipic);
%enhanced image
epic=imread(Y2);
Y2=double(epic);

% MAE %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize total AAMBE
MAE = 0;

% Iterate over each RGB channel
numChannels = size(Y1, 3); % Should be 3 for RGB images
for c = 1:numChannels
    % Extract the current channel for both images
    originalChannel = Y1(:, :, c);
    enhancedChannel = Y2(:, :, c);
    
    % Calculate the absolute difference in brightness for the current channel
    absDifference = abs(originalChannel - enhancedChannel);
    
    % Calculate MAE for the current channel
    MAE_channel = mean(absDifference(:)); % Mean of all pixel values
    MAE = MAE + MAE_channel; % Accumulate the results
end

% Calculate the average AAMBE across all channels
MAE = MAE / numChannels






% GMSD %%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 170; 
Down_step = 2;
dx = [1 0 -1; 1 0 -1; 1 0 -1]/3;
dy = dx';

aveKernel = fspecial('average', 2);

% Extract R, G, B channels from the input images
Y1_R = Y1(:,:,1); Y1_G = Y1(:,:,2); Y1_B = Y1(:,:,3);
Y2_R = Y2(:,:,1); Y2_G = Y2(:,:,2); Y2_B = Y2(:,:,3);

% Apply averaging (down-sampling) to each channel
aveY1_R = conv2(Y1_R, aveKernel, 'same');
aveY1_G = conv2(Y1_G, aveKernel, 'same');
aveY1_B = conv2(Y1_B, aveKernel, 'same');
aveY2_R = conv2(Y2_R, aveKernel, 'same');
aveY2_G = conv2(Y2_G, aveKernel, 'same');
aveY2_B = conv2(Y2_B, aveKernel, 'same');

% Down-sample the images
Y1_R = aveY1_R(1:Down_step:end, 1:Down_step:end);
Y1_G = aveY1_G(1:Down_step:end, 1:Down_step:end);
Y1_B = aveY1_B(1:Down_step:end, 1:Down_step:end);

Y2_R = aveY2_R(1:Down_step:end, 1:Down_step:end);
Y2_G = aveY2_G(1:Down_step:end, 1:Down_step:end);
Y2_B = aveY2_B(1:Down_step:end, 1:Down_step:end);

% Compute gradients for R, G, B channels
IxY1_R = conv2(Y1_R, dx, 'same');     
IyY1_R = conv2(Y1_R, dy, 'same');
gradientMap1_R = sqrt(IxY1_R.^2 + IyY1_R.^2);

IxY2_R = conv2(Y2_R, dx, 'same');     
IyY2_R = conv2(Y2_R, dy, 'same');
gradientMap2_R = sqrt(IxY2_R.^2 + IyY2_R.^2);

IxY1_G = conv2(Y1_G, dx, 'same');     
IyY1_G = conv2(Y1_G, dy, 'same');
gradientMap1_G = sqrt(IxY1_G.^2 + IyY1_G.^2);

IxY2_G = conv2(Y2_G, dx, 'same');     
IyY2_G = conv2(Y2_G, dy, 'same');
gradientMap2_G = sqrt(IxY2_G.^2 + IyY2_G.^2);

IxY1_B = conv2(Y1_B, dx, 'same');     
IyY1_B = conv2(Y1_B, dy, 'same');
gradientMap1_B = sqrt(IxY1_B.^2 + IyY1_B.^2);

IxY2_B = conv2(Y2_B, dx, 'same');     
IyY2_B = conv2(Y2_B, dy, 'same');
gradientMap2_B = sqrt(IxY2_B.^2 + IyY2_B.^2);

% Compute quality map for each channel
quality_map_R = (2 * gradientMap1_R .* gradientMap2_R + T) ./ (gradientMap1_R.^2 + gradientMap2_R.^2 + T);
quality_map_G = (2 * gradientMap1_G .* gradientMap2_G + T) ./ (gradientMap1_G.^2 + gradientMap2_G.^2 + T);
quality_map_B = (2 * gradientMap1_B .* gradientMap2_B + T) ./ (gradientMap1_B.^2 + gradientMap2_B.^2 + T);

% Combine quality maps of R, G, B channels by averaging
quality_map = (quality_map_R + quality_map_G + quality_map_B) / 3;

% Calculate final score by averaging the standard deviation of the quality maps
GMSD = std2(quality_map)




% NIQE %%%%%%%%%%%%%%%%%%%%%%%%%%%%
NIQE = niqe(Y2)

end