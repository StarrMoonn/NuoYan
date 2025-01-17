%% VTI介质全波形反演测试模块
% 功能：测试完整的FWI流程
clear;
clc;

try
    % 获取当前脚本的路径
    [script_path, ~, ~] = fileparts(mfilename('fullpath'));
    
    % 设置项目根目录为当前脚本的上级目录
    project_root = fileparts(script_path);
    addpath(project_root);
    
    % 使用相对路径指定JSON文件路径
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
    fprintf('\n=== 加载参数文件 ===\n');
    params = struct();
    params.obs_params = utils.load_json_params(obs_json_file);
    params.syn_params = utils.load_json_params(syn_json_file);

    % 设置FWI特有参数
    params.optimization = 'gradient_descent';  % 可选: 'gradient_descent', 'BB', 'LBFGS'
    params.max_iterations = 20;     % 最大迭代次数
    params.tolerance = 0.1;         % 收敛容差（10%）

    % 记录开始时间
    start_time = datetime('now');
    fprintf('\n=== 开始FWI测试 [%s] ===\n', start_time);

    % 创建并运行FWI
    fprintf('\n=== 创建并运行FWI实例 ===\n');
    fwi = VTI_FWI(params);
    fwi.run();
    
    % 记录结束时间并计算总用时
    end_time = datetime('now');
    elapsed_time = end_time - start_time;
    fprintf('\n=== FWI测试完成 [%s] ===\n', end_time);
    fprintf('总计算时间: %s\n', elapsed_time);
    
catch ME
    fprintf('\n=== 测试失败 ===\n');
    fprintf('错误信息: %s\n', ME.message);
    fprintf('错误位置: %s\n', ME.stack(1).name);
    rethrow(ME);
end 