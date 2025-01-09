%% 地震记录绘制脚本
% 功能：读取地震记录数据并生成可视化图像
% 
% 说明：
%   1. 读取指定目录下的地震记录数据（vx和vy分量）
%   2. 支持选择指定炮号进行绘制
%   3. 生成水平和垂直分量的地震记录图像
%
% 输入：
%   - 地震记录数据文件：shot_XXX_seismogram.mat
%
% 输出：
%   - PNG格式的地震记录图像
%   - 文件命名格式：seismogram_shot_XXX.png
%
% 作者：StarrMoonn
% 日期：2025-01-08
%

% 清理工作空间
clc;
clear;

% 设置路径
data_path = 'E:\Matlab\VTI_project\data\output\wavefield\seismograms';     % 地震记录数据所在文件夹
output_dir = 'E:\Matlab\VTI_project\data\output\images\seismogram_plots';       % 输出图像文件夹

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 获取目录中所有地震记录文件
files = dir(fullfile(data_path, 'shot_*_seismogram.mat'));
num_shots = length(files);

if num_shots == 0
    error('未找到地震记录文件！');
end

fprintf('\n=== 地震记录绘图 ===\n');
fprintf('找到 %d 个地震记录文件\n', num_shots);

% 处理所有炮号
for shot_number = 1:num_shots
    fprintf('\n处理第 %d/%d 炮的地震记录\n', shot_number, num_shots);
    
    % 调用绘图函数
    utils.plot_seismograms.plot_shot_records(data_path, shot_number, output_dir);
end

fprintf('\n所有地震记录绘图完成！\n'); 