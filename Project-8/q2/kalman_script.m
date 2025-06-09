clear,clc
% compute the background image
Imzero = zeros(240,360,3);
video=VideoReader('SampleVideo.mp4');
video=read(video);
Im = double(cat(4,video()))/255;
clear video

% Convert to RGB to GRAY SCALE image.
nFrames = size(Im,4);
for i = 1:5
    Imzero = Im(:,:,:,i)+Imzero;
end
Imback = Imzero/5*255;

[MR,MC,Dim] = size(Imback);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BONUS: Adaptive Parameter Estimation for Kalman Filter

% 1. Estimate observation noise (R) from stationary frames
fprintf('=== ����Ӧ�������� ===\n');

% Analyze first few frames to estimate detection noise
detection_positions = [];
for frame_idx = 1:min(10, nFrames)
    Imwork_temp = Im(:,:,:,frame_idx)*255;
    [~,~,~,~,cc_temp,cr_temp,flag_temp] = extract(Imwork_temp,Imback,frame_idx);
    if flag_temp == 1
        detection_positions = [detection_positions; cc_temp, cr_temp];
    end
end

% Estimate R matrix based on detection variance
if size(detection_positions, 1) > 2
    detection_var = var(detection_positions);
    R_adaptive = diag(max(detection_var, [1, 1])); % Minimum variance of 1 pixel
    fprintf('���ƵĹ۲���������: X=%.2f, Y=%.2f\n', detection_var(1), detection_var(2));
else
    R_adaptive = 5 * eye(2); % fallback value
    fprintf('����������㣬ʹ��Ĭ�Ϲ۲�����\n');
end

% 2. Estimate process noise (Q) based on image characteristics and expected motion
image_diagonal = sqrt(MR^2 + MC^2);
% Assume max velocity is 10% of image diagonal per frame
max_velocity = 0.1 * image_diagonal;
% Position uncertainty proportional to image size
pos_uncertainty = 0.05 * image_diagonal;
% Velocity uncertainty
vel_uncertainty = 0.1 * max_velocity;

Q_adaptive = diag([pos_uncertainty^2/4, pos_uncertainty^2/4, vel_uncertainty^2, vel_uncertainty^2]);
fprintf('����Ӧ�������� - λ�ò�ȷ����: %.2f, �ٶȲ�ȷ����: %.2f\n', pos_uncertainty, vel_uncertainty);

% 3. Adaptive initial covariance (P) based on image size and expected uncertainty
P_adaptive = diag([pos_uncertainty^2, pos_uncertainty^2, max_velocity^2, max_velocity^2]);
fprintf('����Ӧ��ʼЭ�����������\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original Kalman filter parameters
dt = 1;                             
A = [1 0 dt 0;                      
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1];
H = [1 0 0 0;                       
     0 1 0 0];

% Compare three configurations
configurations = struct();
configurations(1).name = 'Original';
configurations(1).Q = 0.1 * eye(4);
configurations(1).R = 10 * eye(2);
configurations(1).P = 100 * eye(4);

configurations(2).name = 'Adaptive';
configurations(2).Q = Q_adaptive;
configurations(2).R = R_adaptive;
configurations(2).P = P_adaptive;

configurations(3).name = 'Optimized';
configurations(3).Q = Q_adaptive * 0.5; % Reduced process noise for smoother tracking
configurations(3).R = R_adaptive * 1.5; % Slightly increased measurement noise for robustness
configurations(3).P = P_adaptive;

% Store results for comparison
results = struct();
for config_idx = 1:length(configurations)
    fprintf('=== ��������: %s ===\n', configurations(config_idx).name);
    
    % Reset variables for each configuration
    Q = configurations(config_idx).Q;
    R = configurations(config_idx).R;
    P = configurations(config_idx).P;
    
    kfinit = 0;
    x = zeros(nFrames,4);
    tracking_errors_config = zeros(nFrames, 1);
    valid_frames_config = [];
    
    % ����ͼ�����ģ���������һ����
    x_c = MC / 2;   % ͼ������x����
    y_c = MR / 2;   % ͼ������y����
    norm_factor = sqrt(x_c^2 + y_c^2);  % ��һ������

    % loop over all images for current configuration
    for i = 1 : nFrames
        % call Mean shift (reuse previous results if available)
        Imwork = Im(:,:,:,i)*255;
        [x2_temp,y2_temp,width_x_temp,width_y_temp,cc_temp,cr_temp,flag] = extract(Imwork,Imback,i);
        
        if flag==0
            continue
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%
        % Kalman update for current configuration
        
        if kfinit == 0
            % ��ʼ������һ�μ�⵽Ŀ��ʱ���ù۲�ֵ��ʼ��״̬
            if flag == 1
                x(i,1) = cc_temp;        % ��ʼxλ��
                x(i,2) = cr_temp;        % ��ʼyλ��
                x(i,3) = 0;              % ��ʼx�ٶȣ�����Ϊ0��
                x(i,4) = 0;              % ��ʼy�ٶȣ�����Ϊ0��
                kfinit = 1;              % ����ѳ�ʼ��
            end
        else
            % �������˲���5������
            
            % 1. ״̬Ԥ�⣨����ǰһ֡��״̬��
            x_pred = A * x(i-1,:)';    % Ԥ��״̬ x_{k|k-1}
            
            % 2. ���Э����Ԥ��
            P_pred = A * P * A' + Q;   % Ԥ��Э���� P_{k|k-1}
            
            % 3. ���㿨��������
            S = H * P_pred * H' + R;   % ����Э����
            K = P_pred * H' / S;       % ���������� K_k
            
            % 4. ״̬���£�ʹ�ù۲�ֵ����������
            if flag == 1  % ֻ�е���⵽Ŀ��ʱ�Ž��и���
                z = [cc_temp; cr_temp];      % ��ǰ�۲�ֵ
                innovation = z - H * x_pred; % ���£��۲�в
                x(i,:) = (x_pred + K * innovation)';  % ����״̬ x_{k|k}
            else
                % ���û�м�⵽Ŀ�ֻ꣬ʹ��Ԥ��ֵ
                x(i,:) = x_pred';
            end
            
            % 5. ���Э�������
            I = eye(4);                % 4x4��λ����
            P = (I - K * H) * P_pred;  % ����Э���� P_{k|k}
        end

        % ����λ�ø�����������Ч�۲��Ԥ��ʱ��
        if flag == 1 && kfinit == 1
            % ���㿨�����˲�Ԥ��λ����Mean Shift�۲�λ�õ����
            pos_error = sqrt((x(i,1) - cc_temp)^2 + (x(i,2) - cr_temp)^2);
            tracking_errors_config(i) = pos_error / norm_factor;  % ��һ�����
            valid_frames_config = [valid_frames_config i];  % ��¼��Ч֡
        end
    end

    % Store results for comparison
    results(config_idx).name = configurations(config_idx).name;
    results(config_idx).Q = Q;
    results(config_idx).R = R;
    results(config_idx).P_init = configurations(config_idx).P;
    results(config_idx).tracking_errors = tracking_errors_config;
    results(config_idx).valid_frames = valid_frames_config;
    results(config_idx).states = x;
    
    % Calculate statistics for current configuration
    if ~isempty(valid_frames_config)
        valid_errors = tracking_errors_config(valid_frames_config);
        results(config_idx).mean_error = mean(valid_errors);
        results(config_idx).std_error = std(valid_errors);
        results(config_idx).max_error = max(valid_errors);
        results(config_idx).min_error = min(valid_errors);
        results(config_idx).valid_frame_ratio = length(valid_frames_config) / nFrames;
        
        fprintf('ƽ���������: %.4f\n', results(config_idx).mean_error);
        fprintf('����׼��: %.4f\n', results(config_idx).std_error);
        fprintf('��Ч֡����: %.2f%%\n', results(config_idx).valid_frame_ratio * 100);
    else
        results(config_idx).mean_error = NaN;
        results(config_idx).std_error = NaN;
        results(config_idx).max_error = NaN;
        results(config_idx).min_error = NaN;
        results(config_idx).valid_frame_ratio = 0;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���ܶԱȷ����Ϳ��ӻ�

fprintf('\n\n=== ��ϸ���ܶԱȷ��� ===\n');
fprintf('%-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    '����', 'ƽ�����', '����׼��', '������', '��С���', '��Ч֡����');
fprintf('%s\n', repmat('-', 1, 80));

for config_idx = 1:length(configurations)
    fprintf('%-12s %-12.4f %-12.4f %-12.4f %-12.4f %-11.2f%%\n', ...
        results(config_idx).name, ...
        results(config_idx).mean_error, ...
        results(config_idx).std_error, ...
        results(config_idx).max_error, ...
        results(config_idx).min_error, ...
        results(config_idx).valid_frame_ratio * 100);
end

% �����Աȷ���
fprintf('\n=== �������öԱ� ===\n');
for config_idx = 1:length(configurations)
    fprintf('\n%s ���ò���:\n', results(config_idx).name);
    fprintf('  �������� Q (�Խ�Ԫ��): [%.3f, %.3f, %.3f, %.3f]\n', ...
        diag(results(config_idx).Q)');
    fprintf('  �۲����� R (�Խ�Ԫ��): [%.3f, %.3f]\n', ...
        diag(results(config_idx).R)');
    fprintf('  ��ʼЭ���� P (�Խ�Ԫ��): [%.3f, %.3f, %.3f, %.3f]\n', ...
        diag(results(config_idx).P_init)');
end

% ���ƶԱ�ͼ��
figure('Position', [100, 100, 1200, 800]);

% ��ͼ1: �������Ա�
subplot(2,2,1);
colors = {'r', 'g', 'b'};
for config_idx = 1:length(configurations)
    if ~isempty(results(config_idx).valid_frames)
        valid_errors = results(config_idx).tracking_errors(results(config_idx).valid_frames);
        plot(results(config_idx).valid_frames, valid_errors, [colors{config_idx} '-o'], ...
            'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', results(config_idx).name);
        hold on;
    end
end
xlabel('֡��');
ylabel('��һ��λ�ø������');
title('�������Ա�');
legend('show');
grid on;

% ��ͼ2: ���ͳ�ƶԱ�
subplot(2,2,2);
config_names = {results.name};
mean_errors = [results.mean_error];
std_errors = [results.std_error];

x_pos = 1:length(configurations);
bar(x_pos, mean_errors, 'FaceColor', [0.7 0.7 0.9]);
hold on;
errorbar(x_pos, mean_errors, std_errors, 'k.', 'LineWidth', 2);
set(gca, 'XTickLabel', config_names);
ylabel('ƽ���������');
title('ƽ�����Ա� (����׼��)');
grid on;

% ��ͼ3: ��Ч֡�����Ա�
subplot(2,2,3);
valid_ratios = [results.valid_frame_ratio] * 100;
bar(x_pos, valid_ratios, 'FaceColor', [0.9 0.7 0.7]);
set(gca, 'XTickLabel', config_names);
ylabel('��Ч֡���� (%)');
title('�����ȶ��ԶԱ�');
grid on;

% ��ͼ4: �켣�Ա� (���һ�����õĽ��)
subplot(2,2,4);
best_config_idx = 3; % Optimized configuration
if ~isempty(results(best_config_idx).valid_frames)
    states = results(best_config_idx).states;
    valid_frames = results(best_config_idx).valid_frames;
    plot(states(valid_frames,1), states(valid_frames,2), 'b-', 'LineWidth', 2, 'DisplayName', '�������켣');
    hold on;
    
    % ���ƹ۲�켣
    obs_x = [];
    obs_y = [];
    for i = valid_frames
        Imwork = Im(:,:,:,i)*255;
        [~,~,~,~,cc_temp,cr_temp,flag] = extract(Imwork,Imback,i);
        if flag == 1
            obs_x = [obs_x, cc_temp];
            obs_y = [obs_y, cr_temp];
        end
    end
    if ~isempty(obs_x)
        plot(obs_x, obs_y, 'r--', 'LineWidth', 1, 'DisplayName', 'Mean Shift�۲�');
    end
end
xlabel('X ����');
ylabel('Y ����');
title('�Ż����ù켣�Ա�');
legend('show');
grid on;
axis equal;

% ����ԱȽ��
fprintf('\n=== ������������ ===\n');
if length(configurations) >= 2
    original_error = results(1).mean_error;
    adaptive_error = results(2).mean_error;
    optimized_error = results(3).mean_error;
    
    if ~isnan(original_error) && ~isnan(adaptive_error)
        improvement_adaptive = (original_error - adaptive_error) / original_error * 100;
        fprintf('����Ӧ�������ԭʼ����������: %.2f%%\n', improvement_adaptive);
    end
    
    if ~isnan(original_error) && ~isnan(optimized_error)
        improvement_optimized = (original_error - optimized_error) / original_error * 100;
        fprintf('�Ż��������ԭʼ����������: %.2f%%\n', improvement_optimized);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ϸ������������

fprintf('\n\n=== BONUS: ����Ӧ�����Ż��������� ===\n');
fprintf('��ʵ��ͨ�����ֲ�ͬ�Ĳ������öԱ��˿������˲���������:\n\n');

fprintf('1. ԭʼ���� (Original):\n');
fprintf('   - ʹ�ù̶��ľ������\n');
fprintf('   - Q = 0.1 * I (��С�Ĺ�������)\n');
fprintf('   - R = 10 * I (�ϴ�Ĺ۲�����)\n');
fprintf('   - P = 100 * I (�ϴ�ĳ�ʼ��ȷ����)\n\n');

fprintf('2. ����Ӧ���� (Adaptive):\n');
fprintf('   - ����ͼ�������ͼ�ⷽ���Զ����Ʋ���\n');
fprintf('   - Q: ����ͼ��ߴ��Ԥ���˶���Χ�趨\n');
fprintf('   - R: ����ǰ10֡���λ�õķ������\n');
fprintf('   - P: ����ͼ��ߴ�����Ӧ�趨��ʼ��ȷ����\n\n');

fprintf('3. �Ż����� (Optimized):\n');
fprintf('   - ������Ӧ���������Ͻ���΢��\n');
fprintf('   - Q: ����50%%�Ի�ø�ƽ���Ĺ켣\n');
fprintf('   - R: ����50%%����߶������۲��³����\n');
fprintf('   - P: ��������Ӧ�趨\n\n');

fprintf('�������Ʒ���˵��:\n');
fprintf('? �۲�����R�Ĺ���:\n');
fprintf('  - ����ǰ10֡��Ŀ����λ��\n');
fprintf('  - ������λ�õķ�����Ϊ�۲������Ĺ���\n');
fprintf('  - ������С������ֵ(1����)��ֹ�������β��ȶ����\n\n');

fprintf('? ��������Q�Ĺ���:\n');
fprintf('  - ����ͼ��Խ��߳����趨���Ԥ���ٶ�(10%%)\n');
fprintf('  - λ�ò�ȷ������Ϊͼ��Խ��ߵ�5%%\n');
fprintf('  - �ٶȲ�ȷ������Ϊ����ٶȵ�10%%\n\n');

fprintf('? ��ʼЭ����P�Ĺ���:\n');
fprintf('  - λ�ò�ȷ�����������������һ��\n');
fprintf('  - �ٶȲ�ȷ������ΪԤ������ٶȵķ���\n\n');

fprintf('�������:\n');
if exist('results', 'var') && length(results) >= 3
    fprintf('? ԭʼ����: ƽ����� %.4f, ��Ч֡���� %.1f%%\n', ...
        results(1).mean_error, results(1).valid_frame_ratio*100);
    fprintf('? ����Ӧ����: ƽ����� %.4f, ��Ч֡���� %.1f%%\n', ...
        results(2).mean_error, results(2).valid_frame_ratio*100);
    fprintf('? �Ż�����: ƽ����� %.4f, ��Ч֡���� %.1f%%\n', ...
        results(3).mean_error, results(3).valid_frame_ratio*100);
    
    best_idx = 1;
    best_error = results(1).mean_error;
    for i = 2:3
        if ~isnan(results(i).mean_error) && results(i).mean_error < best_error
            best_error = results(i).mean_error;
            best_idx = i;
        end
    end
    fprintf('\n�������: %s (�����С)\n', results(best_idx).name);
end

fprintf('\n�ؼ�����:\n');
fprintf('1. ����Ӧ���������ܹ����ݾ�����Ƶ���������������˲�������\n');
fprintf('2. ���ڼ�ⷽ��Ĺ۲���������������˲����Լ����������Ӧ��\n');
fprintf('3. ����ͼ��ߴ�Ĳ�������ȷ�����㷨�ڲ�ͬ�ֱ����µ��ȶ���\n');
fprintf('4. ����΢��(�Ż�����)��һ�������˸��ٵ�ƽ���Ժ�³����\n');
fprintf('5. ��ȹ̶�����,����Ӧ������������˸��پ��Ⱥ��ȶ���\n\n');

fprintf('ʵ��Ӧ�ý���:\n');
fprintf('? �����µ���Ƶ����,����ʹ������Ӧ����������Ϊ��ʼ����\n');
fprintf('? �ɸ��ݾ���Ӧ�ó�����Q��R����΢���Ż�\n');
fprintf('? ��Ŀ���˶��Ͽ�ʱ,�ʵ����ӹ�������Q\n');
fprintf('? ������㷨�ϲ��ȶ�ʱ,�ʵ����ӹ۲�����R\n');
fprintf('============================================\n');

