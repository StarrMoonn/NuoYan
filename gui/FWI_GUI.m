function FWI_GUI()
    % 创建主窗口
    fig = figure('Name', 'VTI全波形反演-StarrMoonn', ...
                 'Position', [100 100 1200 800], ...  % 加宽窗口
                 'NumberTitle', 'off', ...
                 'MenuBar', 'none');
    
    % 创建左侧控制面板
    control_panel = uipanel('Parent', fig, ...
                           'Position', [0.01 0.01 0.48 0.98]);
    
    % 创建右侧模型显示面板
    model_panel = uipanel('Parent', fig, ...
                         'Position', [0.5 0.01 0.49 0.98], ...
                         'Title', '模型参数显示');
    
    % 创建两个绘图区域
    obs_ax = uipanel('Parent', model_panel, ...  % 改回使用uipanel
                     'Position', [0.02 0.51 0.96 0.48], ...
                     'Title', '观测模型');
    
    syn_ax = uipanel('Parent', model_panel, ...  % 改回使用uipanel
                     'Position', [0.02 0.01 0.96 0.48], ...
                     'Title', '合成模型');
    
    % 文件选择区域
    uicontrol(control_panel, 'Style', 'text', 'Position', [20 720 150 20], ...
             'String', '观测数据JSON文件：');
    obs_edit = uicontrol(control_panel, 'Style', 'edit', 'Position', [20 690 400 25]);
    uicontrol(control_panel, 'Style', 'pushbutton', 'Position', [430 690 100 25], ...
             'String', '浏览...', ...
             'Callback', @(src,event)browse_file(obs_edit, obs_ax));
    
    uicontrol(control_panel, 'Style', 'text', 'Position', [20 650 150 20], ...
             'String', '合成数据JSON文件：');
    syn_edit = uicontrol(control_panel, 'Style', 'edit', 'Position', [20 620 400 25]);
    uicontrol(control_panel, 'Style', 'pushbutton', 'Position', [430 620 100 25], ...
             'String', '浏览...', ...
             'Callback', @(src,event)browse_file(syn_edit, syn_ax));
    
    % === 炮号选择区域 ===
    uicontrol(control_panel, 'Style', 'text', 'Position', [20 580 100 20], ...
             'String', '炮号选择：');
    shot_number = uicontrol(control_panel, 'Style', 'edit', ...
                           'Position', [130 580 100 25], ...
                           'String', '1');
    
    % === 计算模式选择 ===
    uicontrol(control_panel, 'Style', 'text', 'Position', [20 540 100 20], ...
             'String', '计算模式：');
    compute_mode = uicontrol(control_panel, 'Style', 'popupmenu', ...
                            'Position', [130 540 150 25], ...
                            'String', {'cpu_mex', 'cpu', 'cuda_mex'});
    
    % === FWI参数设置区域 ===
    uicontrol(control_panel, 'Style', 'text', 'Position', [20 500 150 20], ...
             'String', 'FWI参数设置：', 'FontWeight', 'bold');
    
    % 最大迭代次数
    uicontrol(control_panel, 'Style', 'text', 'Position', [40 470 100 20], ...
             'String', '最大迭代次数：');
    max_iter_edit = uicontrol(control_panel, 'Style', 'edit', ...
                             'Position', [150 470 100 25], ...
                             'String', '20');
    
    % 收敛容差
    uicontrol(control_panel, 'Style', 'text', 'Position', [40 440 100 20], ...
             'String', '收敛容差：');
    tol_edit = uicontrol(control_panel, 'Style', 'edit', ...
                        'Position', [150 440 100 25], ...
                        'String', '0.1');
    
    % 优化方法选择
    uicontrol(control_panel, 'Style', 'text', 'Position', [40 410 100 20], ...
             'String', '优化方法：');
    opt_method = uicontrol(control_panel, 'Style', 'popupmenu', ...
                          'Position', [150 410 150 25], ...
                          'String', {'BB', 'gradient_descent', 'LBFGS'});
    
    % BB步长
    uicontrol(control_panel, 'Style', 'text', 'Position', [40 380 100 20], ...
             'String', 'BB初始步长：');
    bb_step_edit = uicontrol(control_panel, 'Style', 'edit', ...
                            'Position', [150 380 100 25], ...
                            'String', '0.1');
    
    % === 功能按钮区域 ===
    % FWI按钮
    uicontrol(control_panel, 'Style', 'pushbutton', 'Position', [50 50 150 40], ...
             'String', '运行FWI', ...
             'Callback', @(src,event)run_fwi(obs_edit, syn_edit, ...
                                           max_iter_edit, tol_edit, ...
                                           opt_method, bb_step_edit));
    
    % 伴随波场按钮
    uicontrol(control_panel, 'Style', 'pushbutton', 'Position', [220 50 150 40], ...
             'String', '计算伴随波场', ...
             'Callback', @(src,event)run_adjoint(obs_edit, syn_edit, ...
                                                shot_number, compute_mode));
    
    % 正演模拟按钮
    uicontrol(control_panel, 'Style', 'pushbutton', 'Position', [390 50 150 40], ...
             'String', '单炮正演模拟', ...
             'Callback', @(src,event)run_forward(obs_edit, syn_edit, ...
                                                shot_number, compute_mode));
    
    % 梯度计算按钮
    uicontrol(control_panel, 'Style', 'pushbutton', 'Position', [560 50 150 40], ...
             'String', '计算单炮梯度', ...
             'Callback', @(src,event)run_gradient(obs_edit, syn_edit, ...
                                                shot_number, compute_mode));
end

function browse_file(edit_box, ax)
    [filename, pathname] = uigetfile('*.json', '选择JSON文件');
    if filename ~= 0
        filepath = fullfile(pathname, filename);
        edit_box.String = filepath;
        
        % 加载并显示模型参数
        try
            % 直接读取JSON文件内容
            fid = fopen(filepath, 'r');
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);
            
            % 解析JSON
            params = jsondecode(str);
            
            if isvalid(ax)
                utils.plot_model_params(params, ax);
            end
        catch ME
            % 显示详细的错误信息
            disp(['错误信息: ' ME.message]);
            disp('错误堆栈:');
            disp(ME.stack);
            
            errordlg(['加载模型参数失败：' ME.message], '错误');
            if isvalid(ax)
                cla(ax);
            end
        end
    end
end

function run_fwi(obs_edit, syn_edit, max_iter_edit, tol_edit, opt_method, bb_step_edit)
    try
        % 验证输入
        if isempty(obs_edit.String) || isempty(syn_edit.String)
            errordlg('请选择观测数据和合成数据JSON文件', '错误');
            return;
        end
        
        % 创建优化器参数结构体
        optimizer_params = struct();
        optimizer_params.obs_json_file = obs_edit.String;
        optimizer_params.syn_json_file = syn_edit.String;
        optimizer_params.max_iterations = str2double(max_iter_edit.String);
        optimizer_params.tolerance = str2double(tol_edit.String);
        optimizer_params.optimization = opt_method.String{opt_method.Value};
        optimizer_params.bb_params.initial_step = str2double(bb_step_edit.String);
        
        % 设置输出目录
        [script_path, ~, ~] = fileparts(mfilename('fullpath'));
        project_root = fileparts(script_path);
        output_dir = fullfile(project_root, 'output');
        gradient_output_dir = fullfile(output_dir, 'gradients');
        
        % 确保输出目录存在
        if ~exist(output_dir, 'dir'), mkdir(output_dir); end
        if ~exist(gradient_output_dir, 'dir'), mkdir(gradient_output_dir); end
        
        optimizer_params.output_dir = output_dir;
        optimizer_params.gradient_output_dir = gradient_output_dir;
        
        % 创建梯度求解器
        gradient_solver = GradientSolver(optimizer_params);
        optimizer_params.gradient_solver = gradient_solver;
        
        % 创建并运行FWI
        fwi = VTI_FWI(optimizer_params);
        fwi.run();
        
        msgbox('FWI运行完成！', '成功');
    catch ME
        errordlg(['运行出错：' ME.message], '错误');
    end
end

function run_adjoint(obs_edit, syn_edit, shot_number, compute_mode)
    try
        % 验证输入
        if isempty(obs_edit.String) || isempty(syn_edit.String)
            errordlg('请选择观测数据和合成数据JSON文件', '错误');
            return;
        end
        
        % 加载参数
        obs_params = utils.load_json_params(obs_edit.String);
        syn_params = utils.load_json_params(syn_edit.String);
        
        % 设置计算模式
        obs_params.compute_kernel = compute_mode.String{compute_mode.Value};
        syn_params.compute_kernel = compute_mode.String{compute_mode.Value};
        
        % 创建参数结构体
        params = struct();
        params.obs_params = obs_params;
        params.syn_params = syn_params;
        
        % 获取炮号
        ishot = str2double(shot_number.String);
        
        % 创建VTI_Adjoint实例并计算
        adjoint_solver = VTI_Adjoint(params);
        adjoint_wavefield = adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
        
        msgbox(sprintf('第%d炮伴随波场计算完成！', ishot), '成功');
    catch ME
        errordlg(['计算出错：' ME.message], '错误');
    end
end

function run_forward(obs_edit, syn_edit, shot_number, compute_mode)
    try
        % 优先使用合成数据JSON文件
        json_file = '';
        if ~isempty(syn_edit.String)
            json_file = syn_edit.String;
        elseif ~isempty(obs_edit.String)
            json_file = obs_edit.String;
        else
            errordlg('请至少选择一个JSON文件', '错误');
            return;
        end
        
        % 加载参数
        params = utils.load_json_params(json_file);
        
        % 设置计算模式
        params.compute_kernel = compute_mode.String{compute_mode.Value};
        
        % 获取炮号
        ishot = str2double(shot_number.String);
        
        % 创建VTI_SingleShotModeling实例并计算
        forward_solver = VTI_SingleShotModeling(params);
        [vx_data, vy_data, complete_wavefield] = forward_solver.forward_modeling_single_shot(ishot);
        
        msgbox(sprintf('第%d炮正演模拟完成！', ishot), '成功');
    catch ME
        errordlg(['计算出错：' ME.message], '错误');
    end
end

function run_gradient(obs_edit, syn_edit, shot_number, compute_mode)
    try
        % 验证文件是否存在
        if isempty(obs_edit.String) || isempty(syn_edit.String)
            errordlg('请选择观测数据和合成数据JSON文件', '错误');
            return;
        end
        
        % 加载参数
        obs_params = utils.load_json_params(obs_edit.String);
        syn_params = utils.load_json_params(syn_edit.String);
        
        % 设置计算模式
        obs_params.compute_kernel = compute_mode.String{compute_mode.Value};
        syn_params.compute_kernel = compute_mode.String{compute_mode.Value};
        
        % 创建参数结构体
        params = struct();
        params.obs_params = obs_params;
        params.syn_params = syn_params;
        
        % 获取炮号
        ishot = str2double(shot_number.String);
        
        % 创建梯度求解器实例
        gradient_solver = VTI_Gradient(params);
        
        % 计算单炮梯度（包含所有参数的梯度）
        fprintf('开始计算第%d炮的完整梯度...\n', ishot);
        gradient = gradient_solver.compute_single_shot_gradient(ishot);
        
        % 保存梯度结果
        gradient_solver.save_gradient(gradient, ishot);
        
        msgbox(sprintf('第%d炮梯度计算完成！\n已保存所有参数(c11,c13,c33,c44,rho)的梯度', ishot), '成功');
    catch ME
        errordlg(['计算出错：' ME.message], '错误');
    end
end 