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
    % 修改extent为实际距离（km）
    extent = [0, 8, 2, 0];  % [xmin, xmax, ymax, ymin]
    
    % 加载数据
    c11 = load(fullfile(model_path, 'c11.mat')).c11;
    c13 = load(fullfile(model_path, 'c13.mat')).c13;
    c33 = load(fullfile(model_path, 'c33.mat')).c33;
    c44 = load(fullfile(model_path, 'c44.mat')).c44;
    rho = load(fullfile(model_path, 'rho.mat')).rho;
    
    % 创建图形窗口
    figure('Position', [100, 100, 1500, 800]);
    
    % 绘制C11
    subplot(2,3,1);
    imagesc(extent(1:2), extent(3:4), c11');
    axis xy;  % 确保y轴方向正确
    pbaspect([4 1 1]);  % 设置横纵比为4:1
    colormap(jet);
    c = colorbar;
    c.Label.String = 'C11 (Pa)';
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C11');
    set(gca, 'XTick', [0 2 4 6 8]);
    set(gca, 'YTick', [0 1 2]);
    
    % 绘制C13
    subplot(2,3,2);
    imagesc(extent(1:2), extent(3:4), c13');
    axis xy;
    pbaspect([4 1 1]);  % 设置横纵比为4:1
    colormap(jet);
    c = colorbar;
    c.Label.String = 'C13 (Pa)';
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C13');
    set(gca, 'XTick', [0 2 4 6 8]);
    set(gca, 'YTick', [0 1 2]);
    
    % 绘制C33
    subplot(2,3,3);
    imagesc(extent(1:2), extent(3:4), c33');
    axis xy;
    pbaspect([4 1 1]);  % 设置横纵比为4:1
    colormap(jet);
    c = colorbar;
    c.Label.String = 'C33 (Pa)';
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C33');
    set(gca, 'XTick', [0 2 4 6 8]);
    set(gca, 'YTick', [0 1 2]);
    
    % 绘制C44
    subplot(2,3,4);
    imagesc(extent(1:2), extent(3:4), c44');
    axis xy;
    pbaspect([4 1 1]);  % 设置横纵比为4:1
    colormap(jet);
    c = colorbar;
    c.Label.String = 'C44 (Pa)';
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('C44');
    set(gca, 'XTick', [0 2 4 6 8]);
    set(gca, 'YTick', [0 1 2]);
    
    % 绘制密度分布
    subplot(2,3,5);
    imagesc(extent(1:2), extent(3:4), rho');
    axis xy;
    pbaspect([4 1 1]);  % 设置横纵比为4:1
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Density (kg/m³)';
    xlabel('Distance (km)');
    ylabel('Depth (km)');
    title('Density Distribution');
    set(gca, 'XTick', [0 2 4 6 8]);
    set(gca, 'YTick', [0 1 2]);
    
    % 调整子图间距
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'Units', 'normalized');
    set(gcf, 'Position', [0.05, 0.05, 0.9, 0.85]);
    
    % 设置标题
    sgtitle(['Model Parameters: ' strrep(model_path(end-4:end), '_', ' ')]);
end

% 调用函数绘制两种情况
% 绘制case1（带侵入体）
plot_elastic_params('./data/model/case3');

% 绘制case2（不带侵入体）
plot_elastic_params('./data/model/case2');