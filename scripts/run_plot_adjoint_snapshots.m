%% 伴随波场快照绘制脚本
% 功能：读取伴随波场数据并生成可视化快照图像
% 
% 说明：
%   1. 读取指定JSON配置文件中的模拟参数
%   2. 显示检波器网格位置信息
%   3. 加载伴随波场数据（vx和vy分量）
%   4. 生成彩色波场快照图像
%   5. 在图像中标注震源位置（橙色十字）、PML边界（橙黄色）和检波器位置（绿色）
%
% 输入：
%   - JSON参数文件：包含模型尺寸、检波器位置（网格索引）等参数
%       * NREC：检波器总数
%       * first_rec_x, first_rec_y：初始检波器网格位置
%       * rec_dx, rec_dy：检波器间隔（网格点数）
%   - 伴随波场数据文件：包含不同时间步的波场信息
%
% 输出：
%   - PNG格式的伴随波场快照图像
%   - 控制台输出：检波器位置信息
%
% 作者：StarrMoonn
% 日期：2025-01-08
%
clear;
clc;

% 获取当前脚本的路径
[script_path, ~, ~] = fileparts(mfilename('fullpath'));

% 设置项目根目录为当前脚本的上级目录
project_root = fileparts(script_path);

% 添加项目根目录到路径
addpath(project_root);

% 使用相对路径指定JSON文件路径
json_file = fullfile(project_root, 'data', 'params', 'case3', 'params_syn.json');
data_path = fullfile(project_root, 'data', 'output', 'wavefield', 'adjoint', 'shot_001');

% 为vx和vy分别创建输出目录
output_dir_vx = fullfile(project_root, 'data', 'output', 'images', 'adjoint', 'shot_001', 'vx');    
output_dir_vy = fullfile(project_root, 'data', 'output', 'images', 'adjoint', 'shot_001', 'vy'); 

% 确保输出目录存在
if ~exist(output_dir_vx, 'dir')
    mkdir(output_dir_vx);
end
if ~exist(output_dir_vy, 'dir')
    mkdir(output_dir_vy);
end

% 读取JSON文件以检查检波器位置
params = jsondecode(fileread(json_file));

% 打印检波器参数
fprintf('\n=== 检波器参数 ===\n');
fprintf('检波器总数: %d\n', params.NREC);
fprintf('初始检波器位置（网格索引）: (%d, %d)\n', params.first_rec_x, params.first_rec_y);
fprintf('检波器间隔（网格点数）: dx=%d, dy=%d\n', params.rec_dx, params.rec_dy);

% 计算并打印所有检波器的网格位置
fprintf('\n=== 检波器网格位置 ===\n');
for i = 1:params.NREC
    grid_x = params.first_rec_x + (i-1) * params.rec_dx;
    grid_y = params.first_rec_y + (i-1) * params.rec_dy;
    fprintf('检波器 %3d: 网格位置 (x=%4d, y=%4d)\n', i, grid_x, grid_y);
end

% 调用绘图函数，分别处理vx和vy
% 处理vx分量
utils.plot_adjoint_snapshots(json_file, data_path, output_dir_vx, 'vx'); 

% 处理vy分量
utils.plot_adjoint_snapshots(json_file, data_path, output_dir_vy, 'vy'); 