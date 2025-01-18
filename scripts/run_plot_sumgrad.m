% 计算所有炮的总梯度
clear;
clc;

% 设置参数
nshots = 10;  % 请修改为你的实际炮数
gradient_dir = 'E:\Matlab\VTI_project\data\output\gradient\';  % 请修改为你的梯度文件目录

% 初始化总梯度结构体
total_gradient = struct();
first_shot = load(fullfile(gradient_dir, 'gradient_shot_1.mat'));
[nx, ny] = size(first_shot.gradient.c11);

total_gradient.c11 = zeros(nx, ny);
total_gradient.c13 = zeros(nx, ny);
total_gradient.c33 = zeros(nx, ny);
total_gradient.c44 = zeros(nx, ny);
total_gradient.rho = zeros(nx, ny);

% 累加所有炮的梯度
fprintf('开始累加梯度...\n');
for ishot = 1:nshots
    fprintf('处理第 %d/%d 炮梯度...\n', ishot, nshots);
    
    % 读取单炮梯度
    gradient_filename = fullfile(gradient_dir, sprintf('gradient_shot_%d.mat', ishot));
    shot_gradient = load(gradient_filename);
    
    % 累加到总梯度
    total_gradient.c11 = total_gradient.c11 + shot_gradient.gradient.c11;
    total_gradient.c13 = total_gradient.c13 + shot_gradient.gradient.c13;
    total_gradient.c33 = total_gradient.c33 + shot_gradient.gradient.c33;
    total_gradient.c44 = total_gradient.c44 + shot_gradient.gradient.c44;
    total_gradient.rho = total_gradient.rho + shot_gradient.gradient.rho;
end

% 保存总梯度
save(fullfile(gradient_dir, 'total_gradient.mat'), 'total_gradient');
fprintf('总梯度计算完成并保存\n');

% 定义异常体位置
anomaly_x = 375:426;  % 水平方向网格点范围
anomaly_z = 70:80;    % 垂直方向网格点范围

% 绘制总梯度
figure('Name', '总梯度');

% C11梯度
subplot(2,3,1);
imagesc(total_gradient.c11');
hold on;
rectangle('Position', [anomaly_x(1), anomaly_z(1), ...
    length(anomaly_x), length(anomaly_z)], ...
    'EdgeColor', 'r', 'LineWidth', 2);
title('C11 Gradient');
xlabel('Grid points (NX)');
ylabel('Grid points (NY)');
axis equal; axis tight;
set(gca, 'YDir', 'reverse');
colorbar;

% C13梯度
subplot(2,3,2);
imagesc(total_gradient.c13');
hold on;
rectangle('Position', [anomaly_x(1), anomaly_z(1), ...
    length(anomaly_x), length(anomaly_z)], ...
    'EdgeColor', 'r', 'LineWidth', 2);
title('C13 Gradient');
xlabel('Grid points (NX)');
ylabel('Grid points (NY)');
axis equal; axis tight;
set(gca, 'YDir', 'reverse');
colorbar;

% C33梯度
subplot(2,3,3);
imagesc(total_gradient.c33');
hold on;
rectangle('Position', [anomaly_x(1), anomaly_z(1), ...
    length(anomaly_x), length(anomaly_z)], ...
    'EdgeColor', 'r', 'LineWidth', 2);
title('C33 Gradient');
xlabel('Grid points (NX)');
ylabel('Grid points (NY)');
axis equal; axis tight;
set(gca, 'YDir', 'reverse');
colorbar;

% C44梯度
subplot(2,3,4);
imagesc(total_gradient.c44');
hold on;
rectangle('Position', [anomaly_x(1), anomaly_z(1), ...
    length(anomaly_x), length(anomaly_z)], ...
    'EdgeColor', 'r', 'LineWidth', 2);
title('C44 Gradient');
xlabel('Grid points (NX)');
ylabel('Grid points (NY)');
axis equal; axis tight;
set(gca, 'YDir', 'reverse');
colorbar;

% 密度梯度
subplot(2,3,5);
imagesc(total_gradient.rho');
hold on;
rectangle('Position', [anomaly_x(1), anomaly_z(1), ...
    length(anomaly_x), length(anomaly_z)], ...
    'EdgeColor', 'r', 'LineWidth', 2);
title('Density Gradient');
xlabel('Grid points (NX)');
ylabel('Grid points (NY)');
axis equal; axis tight;
set(gca, 'YDir', 'reverse');
colorbar;

% 调整图形大小和间距
set(gcf, 'Position', [100, 100, 1200, 800]);

% 添加颜色条标题
for i = 1:5
    subplot(2,3,i);
    c = colorbar;
    ylabel(c, 'Gradient Magnitude');
    colormap(gca, 'jet');
end

% 添加调试信息
fprintf('异常体网格位置: X[%d:%d], Z[%d:%d]\n', ...
    anomaly_x(1), anomaly_x(end), anomaly_z(1), anomaly_z(end));
fprintf('最大梯度位置:\n');
[~, max_idx_c11] = max(abs(total_gradient.c11(:)));
[max_x_c11, max_z_c11] = ind2sub(size(total_gradient.c11), max_idx_c11);
fprintf('C11最大梯度位置: (%d, %d)\n', max_x_c11, max_z_c11); 