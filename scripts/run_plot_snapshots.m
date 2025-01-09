%% 正演波场快照绘制脚本
% 功能：读取正演波场数据并生成可视化快照图像
% 
% 说明：
%   1. 读取指定JSON配置文件中的模拟参数
%   2. 加载正演波场数据（vx和vy分量）
%   3. 生成彩色波场快照图像
%
% 输入：
%   - JSON参数文件：包含模型尺寸、检波器位置等参数
%   - 正演波场数据文件：wavefield_XXXXXX.mat
%
% 输出：
%   - PNG格式的波场快照图像
%
% 作者：StarrMoonn
% 日期：2025-01-08
%

% 设置路径
json_path = 'E:\Matlab\VTI_project\data\params\case3\params_syn.json';  % JSON文件路径
data_path = 'E:\Matlab\VTI_project\data\output\wavefield\forward\shot_001';     % 波场数据所在文件夹

% 为vx和vy分别创建输出目录
output_dir_vx = 'E:\Matlab\VTI_project\data\output\images\forward\shot_001\vx';    % vx分量输出文件夹
output_dir_vy = 'E:\Matlab\VTI_project\data\output\images\forward\shot_001\vy';    % vy分量输出文件夹

% 确保输出目录存在
if ~exist(output_dir_vx, 'dir')
    mkdir(output_dir_vx);
end
if ~exist(output_dir_vy, 'dir')
    mkdir(output_dir_vy);
end

% 设置震源位置（网格索引）
params = struct();
params.ISOURCE = 200;  % 震源x方向网格索引
params.JSOURCE = 10;  % 震源y方向网格索引

fprintf('\n=== 震源位置 ===\n');
fprintf('震源网格位置: (x=%d, y=%d)\n', params.ISOURCE, params.JSOURCE);

% 调用绘图函数，分别处理vx和vy
% 处理vx分量
utils.plot_wavefield_snapshots(json_path, data_path, output_dir_vx, 'vx'); 

% 处理vy分量
utils.plot_wavefield_snapshots(json_path, data_path, output_dir_vy, 'vy'); 