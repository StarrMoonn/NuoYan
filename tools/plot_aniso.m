%% VTI模型参数可视化脚本
% 功能：绘制二维VTI介质模型的弹性参数和密度分布图
% 
% 说明：
%   1. 可视化两种模型：带侵入体(case1)和不带侵入体(case2)的二层模型
%   2. 显示参数包括：
%      - C11：VTI介质的弹性参数 (Pa)
%      - C13：VTI介质的弹性参数 (Pa)
%      - C33：VTI介质的弹性参数 (Pa)
%      - C44：VTI介质的弹性参数 (Pa)
%      - 密度：介质密度分布 (kg/m³)
%   3. 图形显示特点：
%      - 横轴范围：0-8 km
%      - 纵轴范围：0-2 km（深度向下为正）
%      - 横纵比例：4:1
%      - 使用jet色标
%
% 输入：
%   model_path - 模型数据路径，包含c11.mat, c13.mat, c33.mat, c44.mat, rho.mat
%
% 输出：
%   - 生成两个图形窗口，分别显示case1和case2的参数分布
%
% 示例：
%   plot_elastic_params('./data/model/case1')
%   plot_elastic_params('./data/model/case2')
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

function plot_elastic_params(model_path)
    % 网格参数
    nx = 801;
    nz = 201;
    dx = 10;  % 假设网格间距为10m
    dz = 10;
    
    % 计算实际距离（km）
    x = (0:nx-1) * dx / 1000;  % 转换为km
    z = (0:nz-1) * dz / 1000;  % 转换为km
    
    % 加载数据
    if contains(model_path, 'case4')
        c11 = load(fullfile(model_path, 'c11.mat')).c11_case4;
        c13 = load(fullfile(model_path, 'c13.mat')).c13_case4;
        c33 = load(fullfile(model_path, 'c33.mat')).c33_case4;
        c44 = load(fullfile(model_path, 'c44.mat')).c44_case4;
        rho = load(fullfile(model_path, 'rho.mat')).rho_case4;
    else
        c11 = load(fullfile(model_path, 'c11.mat')).c11;
        c13 = load(fullfile(model_path, 'c13.mat')).c13;
        c33 = load(fullfile(model_path, 'c33.mat')).c33;
        c44 = load(fullfile(model_path, 'c44.mat')).c44;
        rho = load(fullfile(model_path, 'rho.mat')).rho;
    end
    
    % 创建图形窗口
    figure('Position', [100, 100, 1500, 800]);
    
    % 绘制C11
    subplot(2,3,1);
    imagesc(x, z, c11');
    axis xy;
    colormap(jet);
    colorbar;
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C11 (Pa)');
    
    % 绘制C13
    subplot(2,3,2);
    imagesc(x, z, c13');
    axis xy;
    colormap(jet);
    colorbar;
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C13 (Pa)');
    
    % 绘制C33
    subplot(2,3,3);
    imagesc(x, z, c33');
    axis xy;
    colormap(jet);
    colorbar;
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C33 (Pa)');
    
    % 绘制C44
    subplot(2,3,4);
    imagesc(x, z, c44');
    axis xy;
    colormap(jet);
    colorbar;
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C44 (Pa)');
    
    % 绘制密度分布
    subplot(2,3,5);
    imagesc(x, z, rho');
    axis xy;
    colormap(jet);
    colorbar;
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('Density (kg/m³)');
    
    % 设置总标题
    sgtitle(['Model Parameters: ' strrep(model_path(end-4:end), '_', ' ')]);
end

% 修改调用部分
close all;  % 清除所有图形窗口
plot_elastic_params('./data/model/case3');
plot_elastic_params('./data/model/case4');