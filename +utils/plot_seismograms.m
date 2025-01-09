%% 地震记录绘图类
% 功能：读取地震记录数据并生成可视化图像
% 
% 说明：
%   1. 读取指定目录下的地震记录数据（vx和vy分量）
%   2. 支持选择指定炮号进行绘制
%   3. 生成水平和垂直分量的地震记录图像
%   4. 图像包含时间步-检波器编号的二维显示
%
% 输入参数：
%   - data_dir：地震记录数据目录
%   - shot_number：要绘制的炮号
%   - output_dir：图像输出目录
%
% 输入文件：
%   - shot_XXX_seismogram.mat，包含：
%     * vx_data：水平分量数据
%     * vy_data：垂直分量数据
%
% 输出：
%   - PNG格式的地震记录图像
%   - 文件命名格式：seismogram_shot_XXX.png
%
% 图像特征：
%   - 灰度显示
%   - 水平轴：检波器编号
%   - 垂直轴：时间步
%   - 振幅用颜色条表示
%
% 作者：StarrMoonn
% 日期：2025-01-08
%

classdef plot_seismograms
    methods(Static)
        function plot_shot_records(data_dir, shot_number, output_dir)
            % 读取地震记录数据
            seismic_file = fullfile(data_dir, sprintf('shot_%03d_seismogram.mat', shot_number));
            
            % 检查文件是否存在
            if ~exist(seismic_file, 'file')
                error('找不到地震记录文件: %s', seismic_file);
            end
            
            % 加载数据
            data = load(seismic_file);
            seismogram_vx = data.vx_data;
            seismogram_vy = data.vy_data;
            
            % 获取数据维度
            [NSTEP, NREC] = size(seismogram_vx);
            
            figure('Position', [100, 100, 1500, 1000]);
            
            % 设置全局字体
            try
                set(0, 'DefaultAxesFontName', 'Times New Roman');
            catch
                set(0, 'DefaultAxesFontName', 'serif');
            end
            
            % 绘制水平分量地震记录
            subplot(2,1,1);
            imagesc([1 NREC], [1 NSTEP], seismogram_vx);
            colormap(gray);
            c = colorbar;
            c.Label.String = 'Amplitude';
            title(sprintf('Horizontal Component Seismogram - Shot %d', shot_number));
            xlabel('Receiver Number');
            ylabel('Time Step (nt)');
            
            % 绘制垂直分量地震记录
            subplot(2,1,2);
            imagesc([1 NREC], [1 NSTEP], seismogram_vy);
            colormap(gray);
            c = colorbar;
            c.Label.String = 'Amplitude';
            title(sprintf('Vertical Component Seismogram - Shot %d', shot_number));
            xlabel('Receiver Number');
            ylabel('Time Step (nt)');
            
            % 调整布局并保存图像
            set(gcf, 'PaperPositionMode', 'auto');
            saveas(gcf, fullfile(output_dir, sprintf('seismogram_shot_%03d.png', shot_number)));
            close(gcf);
            
            fprintf('已保存地震记录图像到: %s\n', ...
                fullfile(output_dir, sprintf('seismogram_shot_%03d.png', shot_number)));
        end
    end
end 