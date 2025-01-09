%% VTI介质全波形反演测试模块
% 功能：测试完整的FWI流程
clear;
clc;

try
    % 设置项目根目录
    project_root = 'E:\Matlab\VTI_project';
    cd(project_root);
    addpath(project_root);
    
    % 设置FWI参数（从主程序设置，而不是从json文件）
    fwi_params = struct();
    fwi_params.max_iterations = 20;     % 最大迭代次数
    fwi_params.tolerance = 0.01;        % 收敛容差（1%）
    fwi_params.verbose = true;          % 是否输出详细信息
    
    % 加载参数
    params = struct();
    params.obs_json_file = fullfile(project_root, 'data', 'params', 'case2', 'params_obs.json');
    params.syn_json_file = fullfile(project_root, 'data', 'params', 'case3', 'params_syn.json');
    params.project_root = project_root;
    params.fwi = fwi_params;  % 将FWI参数添加到params结构体中
    
    % 创建FWI实例并运行
    fwi = VTI_FWI(params);
    fwi.run();
    
catch ME
    fprintf('\n=== 测试失败 ===\n');
    fprintf('错误信息: %s\n', ME.message);
    fprintf('错误位置: %s\n', ME.stack(1).name);
    rethrow(ME);
end 