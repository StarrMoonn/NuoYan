%% 波场快照GIF生成脚本
% 功能：将PNG序列转换为GIF动画
%
% 作者：StarrMoonn
% 日期：2024-01-02
%

% 清理工作空间
clc;
clear;

% 设置路径
png_dir = 'E:\Matlab\VTI_project\data\output\images\forward\shot_001\vx';  % PNG文件目录
output_dir = 'E:\Matlab\VTI_project\data\output\images\forward\shot_001\gif';  % GIF输出目录

% 创建输出目录
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 设置GIF参数
delay_time = 0.1;  % 帧间延迟时间（秒）
output_filename = fullfile(output_dir, 'wavefield_vx.gif');

% 生成GIF
utils.create_wavefield_gif(png_dir, output_filename, delay_time); 