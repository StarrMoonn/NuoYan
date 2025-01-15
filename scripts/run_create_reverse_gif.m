%% 波场快照反向GIF生成脚本
% 功能：将PNG序列以反向顺序转换为GIF动画
%
% 作者：StarrMoonn
% 日期：2024-01-02
%

% 清理工作空间
clc;
clear;

% 获取当前脚本的路径
[script_path, ~, ~] = fileparts(mfilename('fullpath'));

% 设置项目根目录为当前脚本的上级目录
project_root = fileparts(script_path);

% 添加项目根目录到路径
addpath(project_root);

% 设置路径
png_dir = fullfile(project_root, 'data', 'output', 'images', 'adjoint', 'shot_001', 'vx');  % PNG文件目录
output_dir = fullfile(project_root, 'data', 'output', 'images', 'adjoint', 'shot_001', 'gif');  % GIF输出目录

% 创建输出目录
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 设置GIF参数
delay_time = 0.1;  % 帧间延迟时间（秒）
output_filename = fullfile(output_dir, 'wavefield_vx_reverse.gif');  % 注意这里文件名改为带reverse

% 生成反向GIF
utils.create_reverse_wavefield_gif(png_dir, output_filename, delay_time); 