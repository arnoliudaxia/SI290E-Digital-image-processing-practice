clear;

% Get all image files in the current directory
image_files = dir('*.png');  % Adjust to '*.jpg' or other extensions if needed

% Loop through each image file
for i = 1:length(image_files)
    % Read the image
    img = double(imread(image_files(i).name));
    
    para.epsilon_stop_L = 1e-3;
    para.epsilon_stop_R = 1e-3;
    para.epsilon = 10/255;
    para.u = 1;
    para.ro = 1.5;
    para.lambda = 5;
    para.beta = 0.01;
    para.omega = 0.01;
    para.delta = 10;

    gamma = 2.2;

    % Perform lowlight enhancement
    [R, L, N] = lowlight_enhancement(img, para);

    % Adjust the result using gamma correction
    res = R .* L.^(1/gamma);

    % Show the result
    figure, imshow(res);

    % Convert the result back to uint8 before saving
    res_uint8 = uint8(res * 255);

    % Save the enhanced image, overwriting the original file
    imwrite(res_uint8, image_files(i).name);
end
