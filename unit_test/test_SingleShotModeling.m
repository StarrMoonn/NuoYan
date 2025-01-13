%% VTI_SingleShotModeling模块测试
% 功能：测试VTI介质中的单炮正演模拟模块
% 
% 说明：
%   1. 初始化模型参数：从JSON文件加载VTI介质参数和计算配置
%   2. 创建VTI_Forward实例：初始化正演模拟求解器
%   3. 执行单炮正演模拟：计算指定炮号的波场
%   4. 验证计算结果：检查波场数据和计算性能
%
% 依赖：
%   - VTI_SingleShotModeling类
%   - utils.load_json_params函数
%
% 作者：StarrMoonn
% 日期：2025-01-13
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

try
    % 1. 加载和初始化参数
    fprintf('\n=== 加载参数文件 ===\n');
    syn_params = utils.load_json_params(json_file);
    
    % 创建参数结构体并从syn_params复制所有字段
    params = syn_params;  % 直接复制所有参数
    
    % 2. 创建求解器实例
    fprintf('\n=== 创建VTI_Forward实例 ===\n');
    forward_solver = VTI_SingleShotModeling(params);
    
    % 3. 检查计算稳定性
    fprintf('\n=== 检查稳定性条件 ===\n');
    quasi_cp_max = max(max(max(sqrt(params.c33./params.rho))), ...
                      max(max(sqrt(params.c11./params.rho))));
    Courant_number = quasi_cp_max * params.DELTAT * ...
                     sqrt(1.0/params.DELTAX^2 + 1.0/params.DELTAY^2);
    fprintf('Courant数为 %f\n', Courant_number);
    if Courant_number > 1.0
        error('时间步长过大，模拟将不稳定');
    end
    
    % 4. 执行正演计算
    ishot = 1;  % 当前炮号
    fprintf('\n=== 开始第 %d 炮正演模拟 ===\n', ishot);
    tic;
    [vx_data, vy_data, complete_wavefield] = forward_solver.forward_modeling_single_shot(ishot);
    total_time = toc;
    
    fprintf('\n=== 模拟完成 ===\n');
    fprintf('总计算时间: %.2f 秒\n', total_time);
    
    % 测试get_complete_wavefield函数
    fprintf('\n=== 测试get_complete_wavefield函数 ===\n');
    fprintf('测试从内存获取波场...\n');
    
    % 检查stored_wavefield是否已保存
    fprintf('stored_wavefield状态: %s\n', ...
        conditional(isempty(forward_solver.stored_wavefield), '空', '非空'));
    fprintf('stored_shot_no: %d\n', forward_solver.stored_shot_no);
    
    % 尝试获取波场
    retrieved_wavefield = forward_solver.get_complete_wavefield(ishot);
    
    % 验证获取的波场
    if ~isempty(retrieved_wavefield)
        fprintf('成功获取波场\n');
        fprintf('波场维度: vx[%d,%d,%d], vy[%d,%d,%d]\n', ...
            size(retrieved_wavefield.vx,1), size(retrieved_wavefield.vx,2), size(retrieved_wavefield.vx,3), ...
            size(retrieved_wavefield.vy,1), size(retrieved_wavefield.vy,2), size(retrieved_wavefield.vy,3));
        
        % 验证波场数据是否一致
        if isequal(complete_wavefield, retrieved_wavefield)
            fprintf('波场数据验证成功：存储的波场与原始波场完全匹配\n');
        else
            warning('波场数据不匹配！');
        end
    else
        warning('无法从内存获取波场！');
    end

catch ME
    % 错误处理
    fprintf('测试失败：%s\n', ME.message);
    fprintf('错误详情：\n%s\n', getReport(ME));
end

fprintf('\n=== 测试完成 ===\n'); 

% 辅助函数
function str = conditional(condition, true_str, false_str)
    if condition
        str = true_str;
    else
        str = false_str;
    end
end

