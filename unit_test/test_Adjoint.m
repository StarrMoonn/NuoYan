%% VTI介质伴随波场计算测试模块
% 功能：测试VTI介质中的伴随波场计算功能
% 
% 说明：
%   1. 测试VTI_Adjoint类的主要功能：
%      - 参数加载和验证
%      - 单炮伴随波场计算
%      - 波场数值稳定性检查
%
% 测试流程：
%   1. 加载观测和合成数据参数
%   2. 创建VTI_Adjoint实例
%   3. 计算单炮伴随波场
%   4. 验证计算结果
%   5. 清理内存和临时文件
%
% 验证内容：
%   - 参数加载正确性
%   - 波场维度正确性
%   - 数值稳定性（无NaN/Inf）
%   - 内存使用监控
%
% 输入文件：
%   - obs_json_file: 观测数据参数文件（JSON格式）
%   - syn_json_file: 合成数据参数文件（JSON格式）
%
% 输出信息：
%   - 计算过程日志
%   - 波场计算结果统计
%   - 内存使用情况
%
% 内存管理：
%   - 自动清理大型波场数据
%   - 清理求解器对象
%   - 监控内存使用情况
%
% 依赖项：
%   - VTI_Adjoint类
%   - utils.load_json_params函数
%   - JSON参数文件
%
% 注意事项：
%   - 需要足够的运行内存
%   - 确保参数文件路径正确
%   - 注意临时文件的清理
%
% 作者：StarrMoonn
% 日期：2025-01-07
% 更新：2025-01-15 - 添加内存管理和监控功能
%
clear;
clc;

% 获取当前脚本的路径
[script_path, ~, ~] = fileparts(mfilename('fullpath'));

% 设置项目根目录为当前脚本的上级目录
project_root = fileparts(script_path);

% 添加项目根目录到路径
addpath(project_root);

% 使用相对路径指定两个JSON文件路径
obs_json_file = fullfile(project_root, 'data', 'params', 'case2', 'params_obs.json');
syn_json_file = fullfile(project_root, 'data', 'params', 'case3', 'params_syn.json');

% 验证JSON文件是否存在
if ~exist(obs_json_file, 'file')
    error('无法找到观测数据JSON文件: %s', obs_json_file);
end

if ~exist(syn_json_file, 'file')
    error('无法找到合成数据JSON文件: %s', syn_json_file);
end

% 使用utils.load_json_params加载参数
try
    fprintf('\n=== 加载参数文件 ===\n');
    obs_params = utils.load_json_params(obs_json_file);
    syn_params = utils.load_json_params(syn_json_file);
    
    % 创建参数结构体
    params = struct();
    params.obs_params = obs_params;
    params.syn_params = syn_params;
    
    % 打印测试信息
    fprintf('\n=== 单炮伴随波场测试开始 ===\n');
    fprintf('观测数据参数文件: %s\n', obs_json_file);
    fprintf('合成数据参数文件: %s\n', syn_json_file);
    
    % 创建VTI_Adjoint实例
    fprintf('\n=== 创建VTI_Adjoint实例并初始化 ===\n');
    adjoint_solver = VTI_Adjoint(params);
    
    % 验证基本参数
    fprintf('\n=== 参数验证 ===\n');
    fprintf('时间步数: %d\n', adjoint_solver.NSTEP);
    fprintf('检波器数量: %d\n', adjoint_solver.NREC);
    
    % 选择要计算的炮号
    ishot = 1;  % 可以根据需要修改炮号
    
    % 计算单炮的伴随波场
    fprintf('\n=== 开始计算第 %d 炮的伴随波场 ===\n', ishot);
    fprintf('开始时间: %s\n', datetime('now'));
    adjoint_wavefield = adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
    fprintf('结束时间: %s\n', datetime('now'));
    
    % 验证结果
    fprintf('\n=== 验证计算结果 ===\n');
    
    % 检查伴随波场是否计算成功
    if ~isempty(adjoint_wavefield.vx) && ~isempty(adjoint_wavefield.vy)
        fprintf('第 %d 炮伴随波场计算成功\n', ishot);
        
        % 检查波场维度
        [nx, ny, nt] = size(adjoint_wavefield.vx);
        fprintf('波场维度: [%d, %d, %d]\n', nx, ny, nt);
        
        % 检查波场值
        fprintf('伴随波场 vx 的最大值: %e\n', max(abs(adjoint_wavefield.vx(:))));
        fprintf('伴随波场 vy 的最大值: %e\n', max(abs(adjoint_wavefield.vy(:))));
        
        % 检查时间步数是否正确
        if nt ~= adjoint_solver.NSTEP
            warning('波场时间步数与设定不符！');
        end
    else
        warning('第 %d 炮伴随波场计算可能有问题\n', ishot);
    end
    
    % 添加物理合理性检查
    fprintf('\n=== 物理合理性检查 ===\n');

    % 检查波场是否有NaN或Inf
    has_nan_vx = any(isnan(adjoint_wavefield.vx(:)));
    has_nan_vy = any(isnan(adjoint_wavefield.vy(:)));
    has_inf_vx = any(isinf(adjoint_wavefield.vx(:)));
    has_inf_vy = any(isinf(adjoint_wavefield.vy(:)));

    if has_nan_vx || has_nan_vy || has_inf_vx || has_inf_vy
        warning('波场中存在NaN或Inf值！');
    end
    
catch ME
    error('测试失败: %s', ME.message);
    
finally
    % 按顺序清理所有大型数据
    if exist('adjoint_wavefield', 'var')
        clear adjoint_wavefield;  % 首先清理大型波场数据
    end
    
    if exist('adjoint_solver', 'var')
        clear adjoint_solver;     % 然后清理求解器对象
    end
    
    % 清理其他变量
    clear has_nan_vx has_nan_vy has_inf_vx has_inf_vy;
    
    % 强制垃圾回收（可选）
    % java.lang.System.gc();
    
    % 显示当前内存使用情况
    memory_info = memory;
    fprintf('\n当前内存使用: %.2f GB\n', memory_info.MemUsedMATLAB/1e9);
end

fprintf('\n=== 测试完成 ===\n'); 