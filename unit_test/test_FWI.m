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
    
    % 创建输出目录
    output_dir = fullfile(project_root, 'output');
    gradient_output_dir = fullfile(output_dir, 'gradients');
    misfit_output_dir = fullfile(output_dir, 'misfits');
    
    % 确保输出目录存在
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    if ~exist(gradient_output_dir, 'dir'), mkdir(gradient_output_dir); end
    if ~exist(misfit_output_dir, 'dir'), mkdir(misfit_output_dir); end
    
    % 设置优化器参数
    optimizer_params = struct();
    optimizer_params.max_iterations = 20;     % 最大迭代次数
    optimizer_params.tolerance = 0.1;         % 收敛容差（10%）
    optimizer_params.output_dir = output_dir;
    optimizer_params.gradient_output_dir = gradient_output_dir;
    optimizer_params.misfit_output_dir = misfit_output_dir;
    
    % BB法特有参数
    optimizer_params.bb_params = struct();
    optimizer_params.bb_params.initial_step = 0.1;
    optimizer_params.bb_params.memory_length = 5;
    optimizer_params.bb_params.max_step = 1.0;
    optimizer_params.bb_params.min_step = 1e-6;
    
    % 设置优化方法
    optimizer_params.optimization_method = 'BB';  % 可选: 'gradient_descent', 'BB', 'LBFGS'
    
    % 加载模型参数
    optimizer_params.obs_json_file = fullfile(project_root, 'data', 'params', 'case2', 'params_obs.json');
    optimizer_params.syn_json_file = fullfile(project_root, 'data', 'params', 'case3', 'params_syn.json');
    
    % 验证JSON文件是否存在
    if ~exist(optimizer_params.obs_json_file, 'file')
        error('无法找到观测数据JSON文件: %s', optimizer_params.obs_json_file);
    end
    
    if ~exist(optimizer_params.syn_json_file, 'file')
        error('无法找到合成数据JSON文件: %s', optimizer_params.syn_json_file);
    end
    
    % 创建梯度求解器（假设这是必需的）
    gradient_solver = GradientSolver(optimizer_params);  % 需要确保这个类存在
    optimizer_params.gradient_solver = gradient_solver;
    
    % 创建并运行FWI
    fwi = VTI_FWI(optimizer_params);
    fwi.run();
    
    fprintf('\n=== FWI测试完成 ===\n');
    
catch ME
    fprintf('\n=== 测试失败 ===\n');
    fprintf('错误信息: %s\n', ME.message);
    fprintf('错误位置: %s\n', ME.stack(1).name);
    rethrow(ME);
end 