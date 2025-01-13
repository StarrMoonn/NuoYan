function plot_model_params(params, ax)
    try
        % 从参数中获取网格信息
        nx = params.NX;
        ny = params.NY;
        dx = params.DELTAX;
        dy = params.DELTAY;
        
        % 计算实际的物理范围（单位：km）
        extent = [0, (nx-1)*dx/1000, (ny-1)*dy/1000, 0];  % [xmin, xmax, ymax, ymin]
        
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
        
        % 清除现有图形
        clf(ax);
        
        % 绘制C11
        subplot(2,3,1, 'Parent', ax);
        imagesc(extent(1:2), extent(3:4), c11');
        axis xy;
        pbaspect([4 1 1]);
        colormap(jet);
        c = colorbar;
        c.Label.String = 'C11 (Pa)';
        xlabel('Distance (km)');
        ylabel('Depth (km)');
        title('C11');
        
        % 绘制C13
        subplot(2,3,2, 'Parent', ax);
        imagesc(extent(1:2), extent(3:4), c13');
        axis xy;
        pbaspect([4 1 1]);
        colormap(jet);
        c = colorbar;
        c.Label.String = 'C13 (Pa)';
        xlabel('Distance (km)');
        ylabel('Depth (km)');
        title('C13');
        
        % 绘制C33
        subplot(2,3,3, 'Parent', ax);
        imagesc(extent(1:2), extent(3:4), c33');
        axis xy;
        pbaspect([4 1 1]);
        colormap(jet);
        c = colorbar;
        c.Label.String = 'C33 (Pa)';
        xlabel('Distance (km)');
        ylabel('Depth (km)');
        title('C33');
        
        % 绘制C44
        subplot(2,3,4, 'Parent', ax);
        imagesc(extent(1:2), extent(3:4), c44');
        axis xy;
        pbaspect([4 1 1]);
        colormap(jet);
        c = colorbar;
        c.Label.String = 'C44 (Pa)';
        xlabel('Distance (km)');
        ylabel('Depth (km)');
        title('C44');
        
        % 绘制密度分布
        subplot(2,3,5, 'Parent', ax);
        imagesc(extent(1:2), extent(3:4), rho');
        axis xy;
        pbaspect([4 1 1]);
        colormap(jet);
        c = colorbar;
        c.Label.String = 'Density (kg/m³)';
        xlabel('Distance (km)');
        ylabel('Depth (km)');
        title('Density Distribution');
        
        % 设置所有子图的刻度
        for i = 1:5
            subplot(2,3,i, 'Parent', ax);
            set(gca, 'XTick', linspace(extent(1), extent(2), 5));
            set(gca, 'YTick', linspace(extent(4), extent(3), 3));
        end
        
        % 调整子图间距
        set(ax, 'Units', 'normalized');
        set(ax, 'Position', [0.05, 0.05, 0.9, 0.85]);
        
    catch ME
        error('绘图失败: %s', ME.message);
    end
end