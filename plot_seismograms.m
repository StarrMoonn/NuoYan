%% 地震记录绘图类
% 功能：读取地震记录数据并生成可视化图像
% 
% 说明：
%   1. 读取指定目录下的地震记录数据（vx和vy分量）
%   2. 支持选择指定炮号进行绘制
%   3. 生成水平和垂直分量的地震记录图像
%   4. 图像包含时间步-检波器编号的二维显示
%
% 输入：
%   - 地震记录数据文件：seismogram_vx.mat, seismogram_vy.mat
%   - 参数设置：
%     * NREC：检波器数量
%     * NSTEP：时间步数
%     * shot_number：要绘制的炮号
%
% 输出：
%   - PNG格式的地震记录图像
%   - 包含水平和垂直分量的双子图
%   - 文件命名格式：seismogram_shot_XXX.png
%
% 图像特征：
%   - 灰度显示
%   - 水平轴：检波器编号 (1 to NREC)
%   - 垂直轴：时间步 (1 to NSTEP)
%   - 振幅用颜色条表示
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

classdef plot_seismograms
    methods(Static)
        function plot_shot_seismogram(seismogram_vx, seismogram_vy, NREC, NSTEP, output_dir, shot_number)

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
        end
    end
end 