%% 模型参数可视化工具
% 功能：将VTI介质模型参数以子图形式可视化展示
% 
% 说明：
%   1. 显示五个主要参数：C11, C13, C33, C44和密度
%   2. 自动计算并设置合适的坐标轴范围和纵横比
%   3. 使用统一的colormap进行可视化
%
% 输入：
%   - params: 包含模型参数的结构体，字段包括：
%       * NX, NY: 网格尺寸
%       * DELTAX, DELTAY: 网格间距
%       * c11, c13, c33, c44, rho: 各向异性参数文件信息
%   - panel: MATLAB图形面板句柄，用于绘制子图
%
% 输出：
%   - 无（直接在指定面板上绘制图形）
%
% 作者：StarrMoonn
% 日期：2025-01-014
%

function plot_model_params(params, panel)
    try
        % 从参数中获取网格信息
        nx = params.NX;
        ny = params.NY;
        dx = params.DELTAX;
        dy = params.DELTAY;
        
        % 计算实际的物理范围（单位：km）
        x_range = [0, (nx-1)*dx/1000];  % x范围（km）
        y_range = [0, (ny-1)*dy/1000];  % y范围（km）
        
        % 计算实际的纵横比
        aspect_ratio = diff(x_range) / diff(y_range);
        
        % 获取文件路径
        c11_info = params.c11;
        c13_info = params.c13;
        c33_info = params.c33;
        c44_info = params.c44;
        rho_info = params.rho;
        
        % 加载数据
        c11 = load(c11_info.file).c11;
        c13 = load(c13_info.file).c13;
        c33 = load(c33_info.file).c33;
        c44 = load(c44_info.file).c44;
        rho = load(rho_info.file).rho;
        
        % 删除面板中现有的所有子对象
        delete(allchild(panel));
        
        % 在面板中创建axes对象
        ax1 = axes('Parent', panel, 'Position', [0.05 0.55 0.25 0.4]);
        ax2 = axes('Parent', panel, 'Position', [0.35 0.55 0.25 0.4]);
        ax3 = axes('Parent', panel, 'Position', [0.65 0.55 0.25 0.4]);
        ax4 = axes('Parent', panel, 'Position', [0.20 0.05 0.25 0.4]);
        ax5 = axes('Parent', panel, 'Position', [0.50 0.05 0.25 0.4]);
        
        % 绘制C11
        imagesc(ax1, x_range, y_range, flipud(c11'));
        colormap(ax1, 'jet');
        c = colorbar(ax1);
        c.Label.String = 'C11 (Pa)';
        xlabel(ax1, 'Distance (km)');
        ylabel(ax1, 'Depth (km)');
        title(ax1, 'C11');
        
        % 绘制C13
        imagesc(ax2, x_range, y_range, flipud(c13'));
        colormap(ax2, 'jet');
        c = colorbar(ax2);
        c.Label.String = 'C13 (Pa)';
        xlabel(ax2, 'Distance (km)');
        ylabel(ax2, 'Depth (km)');
        title(ax2, 'C13');
        
        % 绘制C33
        imagesc(ax3, x_range, y_range, flipud(c33'));
        colormap(ax3, 'jet');
        c = colorbar(ax3);
        c.Label.String = 'C33 (Pa)';
        xlabel(ax3, 'Distance (km)');
        ylabel(ax3, 'Depth (km)');
        title(ax3, 'C33');
        
        % 绘制C44
        imagesc(ax4, x_range, y_range, flipud(c44'));
        colormap(ax4, 'jet');
        c = colorbar(ax4);
        c.Label.String = 'C44 (Pa)';
        xlabel(ax4, 'Distance (km)');
        ylabel(ax4, 'Depth (km)');
        title(ax4, 'C44');
        
        % 绘制密度分布
        imagesc(ax5, x_range, y_range, flipud(rho'));
        colormap(ax5, 'jet');
        c = colorbar(ax5);
        c.Label.String = 'Density (kg/m³)';
        xlabel(ax5, 'Distance (km)');
        ylabel(ax5, 'Depth (km)');
        title(ax5, 'Density');
        
        % 设置所有axes的属性
        all_axes = [ax1, ax2, ax3, ax4, ax5];
        for ax = all_axes
            % 设置刻度
            set(ax, 'XTick', linspace(x_range(1), x_range(2), 5));
            set(ax, 'YTick', linspace(y_range(1), y_range(2), 5));
            % 根据实际模型尺寸设置纵横比
            pbaspect(ax, [aspect_ratio 1 1]);
            % 设置box属性
            box(ax, 'on');
        end
        
    catch ME
        error('绘图失败: %s', ME.message);
    end
end