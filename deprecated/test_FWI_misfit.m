%% 基于JSON配置的FWI误差函数测试
% 功能：加载观测模型和合成模型，计算波场误差
% 
% 说明：
%   1. 分别加载观测模型和初始模型的JSON配置
%   2. 进行两次正演模拟
%   3. 计算波场误差
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

clear;
clc;

% 设置项目根目录
project_root = 'E:\Matlab\VTI_project';
cd(project_root);
addpath(project_root);

% 验证路径
if ~exist('VTI_SingleShotModeling', 'class')
    error('无法找到 VTI_SingleShotModeling 类，请检查路径设置');
end

% 加载观测模型参数（真实模型）
json_file_obs = fullfile(project_root, 'data', 'params', 'case2', 'params_obs.json');
params_obs = utils.load_json_params(json_file_obs);

% 加载合成模型参数（初始模型）
json_file_syn = fullfile(project_root, 'data', 'params', 'case2', 'params_syn.json');
params_syn = utils.load_json_params(json_file_syn);

% 创建观测数据的FWI实例并运行
fprintf('\n=== 运行观测模型正演 ===\n');
fwi_obs = VTI_SingleShotModeling(params_obs);
fwi_obs.forward_modeling_single_shot();

% 创建合成数据的FWI实例并运行
fprintf('\n=== 运行合成模型正演 ===\n');
fwi_syn = VTI_SingleShotModeling(params_syn);
fwi_syn.forward_modeling_single_shot();

% 计算误差函数
[misfit, misfit_per_shot] = utils.compute_misfit(...
    fwi_obs.seismogram_vx_all, ...
    fwi_obs.seismogram_vy_all, ...
    fwi_syn.seismogram_vx_all, ...
    fwi_syn.seismogram_vy_all);

fprintf('\n=== 测试完成 ===\n'); 