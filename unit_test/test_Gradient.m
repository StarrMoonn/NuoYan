%% VTI介质单炮梯度计算测试模块
% 功能：测试VTI介质中单炮梯度的计算和可视化
% 
% 说明：
%   1. 测试VTI_Gradient类的主要功能：
%      - 正演波场和伴随波场的互相关计算
%      - 各向异性参数(C11,C13,C33,C44)的梯度计算
%      - 密度梯度计算
%      - 梯度可视化
%   2. 验证内容包括：
%      - 梯度计算的正确性
%      - 梯度分布的物理合理性
%      - 异常体位置与梯度响应的对应关系
%
% 使用方法：
%   1. 确保项目根目录设置正确
%   2. 准备观测数据和合成数据的参数文件（JSON格式）
%   3. 运行测试脚本
%
% 输入参数：
%   - project_root: 项目根目录路径
%   - obs_json_file: 观测数据参数文件路径
%   - syn_json_file: 合成数据参数文件路径
%
% 输出信息：
%   - 梯度计算过程信息
%   - 异常体位置信息
%   - 最大梯度位置
%   - 梯度可视化图像
%
% 依赖项：
%   - VTI_Gradient类
%   - VTI_Adjoint类
%   - utils.load_json_params函数
%   - JSON参数文件
%
% 注意事项：
%   - 需要正确设置项目根目录
%   - 确保参数文件存在且格式正确
%   - 检查内存是否足够大
%   - 注意坐标系统的一致性
%
% 作者：StarrMoonn
% 日期：2025-01-09
%
clear;
clc;

try
    % 获取当前脚本的路径
    [script_path, ~, ~] = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_path);
    addpath(project_root);
     
    % 使用相对路径指定JSON文件路径
    obs_json_file = fullfile(project_root, 'data', 'params', 'case2', 'params_obs.json');
    syn_json_file = fullfile(project_root, 'data', 'params', 'case3', 'params_syn.json');
    
    % 验证JSON文件是否存在
    if ~exist(obs_json_file, 'file')
        error('无法找到观测数据JSON文件: %s', obs_json_file);
    end
    
    if ~exist(syn_json_file, 'file')
        error('无法找到合成数据JSON文件: %s', syn_json_file);
    end
    
    % 使用utils.load_json_params加载参数
    fprintf('\n=== 加载参数文件 ===\n');
    params = struct();
    params.obs_params = utils.load_json_params(obs_json_file);
    params.syn_params = utils.load_json_params(syn_json_file);
    
    % 创建梯度计算器实例
    fprintf('\n=== 创建梯度计算器实例 ===\n');
    gradient_solver = VTI_Gradient(params);
    
    % 验证实例创建
    fprintf('检查实例链:\n');
    fprintf('adjoint_solver类型: %s\n', class(gradient_solver.adjoint_solver));
    fprintf('wavefield_solver_syn类型: %s\n', ...
        class(gradient_solver.adjoint_solver.wavefield_solver_syn));
    
    % 选择要计算的炮号
    ishot = 1;

    % 计算单炮的梯度
    fprintf('\n=== 计算第%d炮梯度 ===\n', ishot);
    fprintf('开始时间: %s\n', datetime('now'));
    gradient = gradient_solver.compute_single_shot_gradient(ishot);
    fprintf('结束时间: %s\n', datetime('now'));

    % 验证梯度结果
    fprintf('\n=== 验证梯度结果 ===\n');
    fprintf('c11梯度大小: [%d, %d]\n', size(gradient.c11));
    fprintf('c13梯度大小: [%d, %d]\n', size(gradient.c13));
    fprintf('c33梯度大小: [%d, %d]\n', size(gradient.c33));
    fprintf('c44梯度大小: [%d, %d]\n', size(gradient.c44));
    fprintf('rho梯度大小: [%d, %d]\n', size(gradient.rho));
    
    % 检查梯度值范围
    fprintf('\nc11梯度范围: [%e, %e]\n', min(gradient.c11(:)), max(gradient.c11(:)));
    fprintf('c13梯度范围: [%e, %e]\n', min(gradient.c13(:)), max(gradient.c13(:)));
    fprintf('c33梯度范围: [%e, %e]\n', min(gradient.c33(:)), max(gradient.c33(:)));
    fprintf('c44梯度范围: [%e, %e]\n', min(gradient.c44(:)), max(gradient.c44(:)));
    fprintf('rho梯度范围: [%e, %e]\n', min(gradient.rho(:)), max(gradient.rho(:)));
    
    % 获取网格尺寸用于可视化
    [nx_c11, ny_c11] = size(gradient.c11);
    
    % 定义异常体位置
    anomaly_x = 375:426;  % 水平方向网格点范围
    anomaly_z = 70:80;    % 垂直方向网格点范围
    
    figure('Name', sprintf('Shot %d Gradient', ishot));
    
    % C11梯度
    subplot(2,3,1);
    imagesc(gradient.c11');
    hold on;
    % 绘制异常体位置的红色方框
    rectangle('Position', [anomaly_x(1), anomaly_z(1), ...
        length(anomaly_x), length(anomaly_z)], ...
        'EdgeColor', 'r', 'LineWidth', 2);
    title('C11 Gradient');
    xlabel('Grid points (NX)');
    ylabel('Grid points (NY)');
    axis equal; axis tight;
    set(gca, 'YDir', 'reverse');
    colorbar;
    
    % C13梯度 (类似修改)
    subplot(2,3,2);
    imagesc(gradient.c13');
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
    imagesc(gradient.c33');
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
    imagesc(gradient.c44');
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
    imagesc(gradient.rho');
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
    [~, max_idx_c11] = max(abs(gradient.c11(:)));
    [max_x_c11, max_z_c11] = ind2sub(size(gradient.c11), max_idx_c11);
    fprintf('C11最大梯度位置: (%d, %d)\n', max_x_c11, max_z_c11);
    
catch ME
    fprintf('\n错误: %s\n', ME.message);
    rethrow(ME);
end
     

