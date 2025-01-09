%% 生成均匀介质模型文件
% 功能：生成VTI介质的均匀模型参数文件
% 
% 说明：
%   1. 生成五个材料参数的mat文件
%   2. 每个参数都是401x401的均匀数组
%   3. 保存到指定目录
%
% 输出：
%   - c11.mat: 4.0e10 Pa
%   - c13.mat: 3.8e10 Pa
%   - c33.mat: 20.0e10 Pa
%   - c44.mat: 2.0e10 Pa
%   - rho.mat: 4000 kg/m³
%
% 作者：StarrMoonn
% 日期：2024-01-07
%

clear;
clc;

% 设置模型尺寸
nx = 401;
ny = 401;

% 设置输出目录
output_dir = 'data/model/case1';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 生成并保存c11
c11 = ones(nx, ny) * 4.0e10;
save(fullfile(output_dir, 'c11.mat'), 'c11');
fprintf('已保存c11.mat，大小: [%d, %d], 值: %e\n', size(c11, 1), size(c11, 2), c11(1,1));

% 生成并保存c13
c13 = ones(nx, ny) * 3.8e10;
save(fullfile(output_dir, 'c13.mat'), 'c13');
fprintf('已保存c13.mat，大小: [%d, %d], 值: %e\n', size(c13, 1), size(c13, 2), c13(1,1));

% 生成并保存c33
c33 = ones(nx, ny) * 20.0e10;
save(fullfile(output_dir, 'c33.mat'), 'c33');
fprintf('已保存c33.mat，大小: [%d, %d], 值: %e\n', size(c33, 1), size(c33, 2), c33(1,1));

% 生成并保存c44
c44 = ones(nx, ny) * 2.0e10;
save(fullfile(output_dir, 'c44.mat'), 'c44');
fprintf('已保存c44.mat，大小: [%d, %d], 值: %e\n', size(c44, 1), size(c44, 2), c44(1,1));

% 生成并保存rho
rho = ones(nx, ny) * 4000;
save(fullfile(output_dir, 'rho.mat'), 'rho');
fprintf('已保存rho.mat，大小: [%d, %d], 值: %e\n', size(rho, 1), size(rho, 2), rho(1,1));

fprintf('\n=== 所有模型文件生成完成 ===\n');
fprintf('文件保存位置: %s\n', output_dir); 