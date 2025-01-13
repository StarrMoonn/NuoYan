function FWI_GUI()
    % 创建主窗口
    fig = figure('Name', 'VTI全波形反演程序', ...
                 'Position', [100 100 1400 900], ...
                 'NumberTitle', 'off');
    
    % 创建菜单栏
    menu_bar = uimenu(fig, 'Label', '关于(O)');
    uimenu(menu_bar, 'Label', '作者简介', 'Callback', @show_about);
    
    % 创建左侧控制面板
    control_panel = uipanel('Parent', fig, ...
                           'Position', [0.01 0.01 0.28 0.98]);
    
    % 创建右侧模型显示面板
    model_panel = uipanel('Parent', fig, ...
                         'Position', [0.3 0.01 0.69 0.98], ...
                         'Title', '模型参数显示');
    
    % 创建标签页组
    tabgroup = uitabgroup('Parent', control_panel, ...
                         'Position', [0 0 1 1]);
    
    % === 1. 参数设置标签页 ===
    param_tab = uitab('Parent', tabgroup, ...
                     'Title', '参数设置');
    
    % 在参数设置标签页中创建文件选择区域
    file_panel = uipanel('Parent', param_tab, ...
                        'Position', [0.02 0.75 0.96 0.23], ...
                        'Title', '文件选择');
    
    % 观测数据JSON文件选择
    uicontrol(file_panel, 'Style', 'text', ...
             'Position', [10 120 120 20], ...
             'String', '观测数据JSON文件：');
    obs_edit = uicontrol(file_panel, 'Style', 'edit', ...
                        'Position', [10 95 250 25], ...
                        'Tag', 'obs_edit');
    uicontrol(file_panel, 'Style', 'pushbutton', ...
             'Position', [270 95 80 25], ...
             'String', '浏览...', ...
             'Callback', @(src,event)browse_obs_file(src, event, obs_edit));
    
    % 合成数据JSON文件选择
    uicontrol(file_panel, 'Style', 'text', ...
             'Position', [10 65 120 20], ...
             'String', '合成数据JSON文件：');
    syn_edit = uicontrol(file_panel, 'Style', 'edit', ...
                        'Position', [10 40 250 25], ...
                        'Tag', 'syn_edit');
    uicontrol(file_panel, 'Style', 'pushbutton', ...
             'Position', [270 40 80 25], ...
             'String', '浏览...', ...
             'Callback', @(src,event)browse_syn_file(src, event, syn_edit));
    
    % 创建单个参数可视化区域
    param_panel = uipanel('Parent', param_tab, ...
                         'Position', [0.02 0.02 0.96 0.72], ...
                         'Title', '参数可视化');
    
    % 创建标签页组
    param_tabgroup = uitabgroup('Parent', param_panel, ...
                               'Position', [0 0 1 1]);  % 完全填充参数面板
    
    % 观测数据参数标签页
    tab1 = uitab('Parent', param_tabgroup, 'Title', '观测数据参数');
    obs_param_list = uicontrol(tab1, 'Style', 'listbox', ...
                              'Position', [0 0 360 460], ...  % 完全填充标签页
                              'Tag', 'obs_param_list', ...
                              'String', {}, ...
                              'Max', 2);
    
    % 合成数据参数标签页
    tab2 = uitab('Parent', param_tabgroup, 'Title', '合成数据参数');
    syn_param_list = uicontrol(tab2, 'Style', 'listbox', ...
                              'Position', [0 0 360 460], ...  % 完全填充标签页
                              'Tag', 'syn_param_list', ...
                              'String', {}, ...
                              'Max', 2);
    
    % === 3. 单炮计算标签页 ===
    shot_tab = uitab('Parent', tabgroup, ...
                     'Title', '单炮计算');
    
    % 在单炮计算标签页中创建面板
    shot_panel = uipanel('Parent', shot_tab, ...
                        'Position', [0.02 0.02 0.96 0.96]);
    
    % 计算策略选择
    uicontrol(shot_panel, 'Style', 'text', ...
             'Position', [10 400 80 20], ...
             'String', '计算策略：');
    compute_mode = uicontrol(shot_panel, 'Style', 'popupmenu', ...
                           'Position', [100 400 150 25], ...
                           'String', {'cpu_mex', 'cpu', 'cuda_mex'}, ...
                           'Value', 1, ...
                           'Tag', 'compute_mode');
    
    % 炮号选择
    uicontrol(shot_panel, 'Style', 'text', ...
             'Position', [10 360 80 20], ...
             'String', '炮号：');
    shot_number = uicontrol(shot_panel, 'Style', 'edit', ...
                          'Position', [100 360 150 25], ...
                          'String', '1', ...
                          'Tag', 'shot_number');
    
    % 三个计算按钮
    uicontrol(shot_panel, 'Style', 'pushbutton', ...
             'Position', [10 320 240 30], ...
             'String', '计算单炮正演', ...
             'Callback', @calculate_forward);
    
    uicontrol(shot_panel, 'Style', 'pushbutton', ...
             'Position', [10 280 240 30], ...
             'String', '计算单炮伴随波场', ...
             'Callback', @calculate_adjoint);
    
    uicontrol(shot_panel, 'Style', 'pushbutton', ...
             'Position', [10 240 240 30], ...
             'String', '计算单炮梯度', ...
             'Callback', @calculate_gradient);
    
    % === 4. FWI参数设置标签页 ===
    fwi_tab = uitab('Parent', tabgroup, ...
                    'Title', 'FWI设置');
    
    % 在FWI参数设置标签页中创建面板
    fwi_panel = uipanel('Parent', fwi_tab, ...
                        'Position', [0.02 0.02 0.96 0.96]);
    
    % 优化方法选择
    uicontrol(fwi_panel, 'Style', 'text', ...
             'Position', [10 400 80 20], ...
             'String', '优化方法：');
    optimization = uicontrol(fwi_panel, 'Style', 'popupmenu', ...
                           'Position', [100 400 150 25], ...
                           'String', {'BB', 'L-BFGS', 'CG'}, ...
                           'Tag', 'optimization');
    
    % 最大迭代次数
    uicontrol(fwi_panel, 'Style', 'text', ...
             'Position', [10 360 100 20], ...
             'String', '最大迭代次数：');
    max_iter = uicontrol(fwi_panel, 'Style', 'edit', ...
                        'Position', [120 360 130 25], ...
                        'String', '20', ...
                        'Tag', 'max_iter');
    
    % 收敛容差
    uicontrol(fwi_panel, 'Style', 'text', ...
             'Position', [10 320 80 20], ...
             'String', '收敛容差：');
    tolerance = uicontrol(fwi_panel, 'Style', 'edit', ...
                         'Position', [100 320 150 25], ...
                         'String', '0.1', ...
                         'Tag', 'tolerance');
    
    % BB初始步长
    uicontrol(fwi_panel, 'Style', 'text', ...
             'Position', [10 280 80 20], ...
             'String', 'BB初始步长：');
    bb_alpha = uicontrol(fwi_panel, 'Style', 'edit', ...
                        'Position', [100 280 150 25], ...
                        'String', '0.1', ...
                        'Tag', 'bb_alpha');
    
    % 创建绘图区域
    obs_ax = uipanel('Parent', model_panel, ...
                     'Position', [0.02 0.51 0.96 0.48], ...
                     'Title', '观测模型');
    
    syn_ax = uipanel('Parent', model_panel, ...
                     'Position', [0.02 0.01 0.96 0.48], ...
                     'Title', '合成模型');
                     
    % 存储GUI数据
    gui_data.obs_params = [];
    gui_data.syn_params = [];
    guidata(fig, gui_data);
end

% 观测数据JSON文件浏览回调函数
function browse_obs_file(src, ~, edit_box)
    [filename, pathname] = uigetfile('*.json', '选择观测数据JSON文件');
    if filename ~= 0
        filepath = fullfile(pathname, filename);
        edit_box.String = filepath;
        
        % 读取并显示JSON参数
        try
            params = jsondecode(fileread(filepath));
            fig = ancestor(src, 'figure');
            gui_data = guidata(fig);
            gui_data.obs_params = params;
            guidata(fig, gui_data);
            
            % 更新参数列表（使用 timer 延迟执行以避免界面卡顿）
            t = timer('ExecutionMode', 'singleShot', ...
                     'StartDelay', 0.1, ...
                     'TimerFcn', @(~,~)update_param_list(fig, 'obs_param_list', params));
            start(t);
            
            % 更新模型显示
            obs_ax = findobj(fig, 'Title', '观测模型');
            if ~isempty(obs_ax)
                utils.plot_model_params(params, obs_ax);
            end
        catch ME
            errordlg(['加载JSON文件失败：' ME.message], '错误');
        end
    end
end

% 合成数据JSON文件浏览回调函数
function browse_syn_file(src, ~, edit_box)
    [filename, pathname] = uigetfile('*.json', '选择合成数据JSON文件');
    if filename ~= 0
        filepath = fullfile(pathname, filename);
        edit_box.String = filepath;
        
        % 读取并显示JSON参数
        try
            params = jsondecode(fileread(filepath));
            fig = ancestor(src, 'figure');
            gui_data = guidata(fig);
            gui_data.syn_params = params;
            guidata(fig, gui_data);
            
            % 更新参数列表
            update_param_list(fig, 'syn_param_list', params);
            
            % 更新模型显示
            syn_ax = findobj(fig, 'Title', '合成模型');
            if ~isempty(syn_ax)
                utils.plot_model_params(params, syn_ax);
            end
        catch ME
            errordlg(['加载JSON文件失败：' ME.message], '错误');
        end
    end
end

% 更新参数列表显示
function update_param_list(fig, list_tag, params)
    param_list = findobj(fig, 'Tag', list_tag);
    if ~isempty(param_list)
        % 清空当前列表
        param_list.String = {};
        drawnow;  % 强制更新显示
        
        % 将参数转换为字符串列表
        param_strings = {};
        fields = fieldnames(params);
        for i = 1:length(fields)
            field = fields{i};
            value = params.(field);
            if isstruct(value)
                param_strings{end+1} = sprintf('%s: [结构体]', field);
            elseif isnumeric(value)
                if numel(value) > 100  % 对于大型数组，只显示大小信息
                    param_strings{end+1} = sprintf('%s: [%s数组]', field, mat2str(size(value)));
                else
                    param_strings{end+1} = sprintf('%s: %s', field, mat2str(value));
                end
            elseif ischar(value) || isstring(value)
                param_strings{end+1} = sprintf('%s: %s', field, value);
            elseif islogical(value)
                param_strings{end+1} = sprintf('%s: %s', field, mat2str(value));
            else
                param_strings{end+1} = sprintf('%s: [不支持的数据类型]', field);
            end
        end
        
        % 更新列表内容
        param_list.Value = 1;  % 重置选择
        param_list.String = param_strings;
        drawnow;  % 再次强制更新显示
    end
end

% 辅助函数：将结构体转换为字符串数组
function strings = structToString(prefix, struct_data, indent)
    strings = {};
    fields = fieldnames(struct_data);
    for i = 1:length(fields)
        field = fields{i};
        value = struct_data.(field);
        if isstruct(value)
            % 递归处理嵌套结构体
            nested = structToString([prefix '.' field], value, [indent '  ']);
            strings = [strings; nested];
        else
            if isnumeric(value) || islogical(value)
                value_str = mat2str(value);
            else
                value_str = char(string(value));
            end
            strings{end+1} = sprintf('%s%s.%s: %s', indent, prefix, field, value_str);
        end
    end
end

% 单炮伴随波场计算回调函数
function calculate_adjoint(src, ~)
    try
        % 获取主窗口句柄
        fig = ancestor(src, 'figure');
        gui_data = guidata(fig);
        
        % 获取计算参数
        compute_mode_obj = findobj(fig, 'Tag', 'compute_mode');
        compute_modes = get(compute_mode_obj, 'String');
        selected_mode = compute_modes{get(compute_mode_obj, 'Value')};
        
        shot_number = str2double(get(findobj(fig, 'Tag', 'shot_number'), 'String'));
        
        % 获取JSON文件路径
        obs_path = get(findobj(fig, 'Tag', 'obs_edit'), 'String');
        syn_path = get(findobj(fig, 'Tag', 'syn_edit'), 'String');
        
        % 检查参数有效性
        if isempty(obs_path) || isempty(syn_path)
            error('请先选择观测数据和合成数据JSON文件');
        end
        
        if isnan(shot_number) || shot_number < 1
            error('请输入有效的炮号');
        end
        
        % 显示计算信息
        msg = sprintf(['开始计算单炮伴随波场：\n', ...
                      '计算策略：%s\n', ...
                      '炮号：%d\n', ...
                      '观测数据：%s\n', ...
                      '合成数据：%s'], ...
                      selected_mode, shot_number, obs_path, syn_path);
        disp(msg);
        
        % 读取JSON并设置参数
        obs_params = jsondecode(fileread(obs_path));
        syn_params = jsondecode(fileread(syn_path));
        
        % 设置计算参数
        params = syn_params;  % 使用合成数据的参数作为基础
        params.compute_kernel = selected_mode;
        params.current_shot = shot_number;
        params.obs_data = obs_params;  % 添加观测数据参数
        
        % 执行伴随波场计算
        adjoint_modeling(params);
        
        % 计算完成提示
        msgbox('单炮伴随波场计算完成', '计算成功');
        
    catch ME
        % 错误处理
        errordlg(['计算失败：' ME.message], '错误');
        disp(['错误详情：' getReport(ME)]);
    end
end

% 单炮正演计算回调函数
function calculate_forward(src, ~)
    try
        % 获取主窗口句柄
        fig = ancestor(src, 'figure');
        
        % 获取计算参数
        compute_mode_obj = findobj(fig, 'Tag', 'compute_mode');
        compute_modes = get(compute_mode_obj, 'String');
        selected_mode = compute_modes{get(compute_mode_obj, 'Value')};
        
        shot_number = str2double(get(findobj(fig, 'Tag', 'shot_number'), 'String'));
        
        % 获取JSON文件路径
        syn_path = get(findobj(fig, 'Tag', 'syn_edit'), 'String');
        
        % 检查参数有效性
        if isempty(syn_path)
            error('请先选择合成数据JSON文件');
        end
        
        if isnan(shot_number) || shot_number < 1
            error('请输入有效的炮号');
        end
        
        % 显示计算信息
        msg = sprintf(['开始计算单炮正演：\n', ...
                      '计算策略：%s\n', ...
                      '炮号：%d\n', ...
                      '合成数据：%s'], ...
                      selected_mode, shot_number, syn_path);
        disp(msg);
        
        % 读取JSON并设置参数
        params = jsondecode(fileread(syn_path));
        params.compute_kernel = selected_mode;
        
        % 创建VTI_Forward实例
        forward_solver = VTI_SingleShotModeling(params);
        
        % 执行单炮正演模拟
        fprintf('\n=== 开始第 %d 炮正演模拟 ===\n', shot_number);
        tic;
        [vx_data, vy_data, complete_wavefield] = forward_solver.forward_modeling_single_shot(shot_number);
        total_time = toc;
        
        fprintf('\n=== 模拟完成 ===\n');
        fprintf('总计算时间: %.2f 秒\n', total_time);
        
        % 计算完成提示
        msgbox(sprintf('单炮正演计算完成\n计算时间：%.2f秒', total_time), '计算成功');
        
    catch ME
        % 错误处理
        errordlg(['计算失败：' ME.message], '错误');
        disp(['错误详情：' getReport(ME)]);
    end
end

% 单炮梯度计算回调函数
function calculate_gradient(src, ~)
    try
        % 获取主窗口句柄
        fig = ancestor(src, 'figure');
        
        % 获取计算参数
        compute_mode_obj = findobj(fig, 'Tag', 'compute_mode');
        compute_modes = get(compute_mode_obj, 'String');
        selected_mode = compute_modes{get(compute_mode_obj, 'Value')};
        
        shot_number = str2double(get(findobj(fig, 'Tag', 'shot_number'), 'String'));
        
        % 获取JSON文件路径
        obs_path = get(findobj(fig, 'Tag', 'obs_edit'), 'String');
        syn_path = get(findobj(fig, 'Tag', 'syn_edit'), 'String');
        
        % 检查参数有效性
        if isempty(obs_path) || isempty(syn_path)
            error('请先选择观测数据和合成数据JSON文件');
        end
        
        if isnan(shot_number) || shot_number < 1
            error('请输入有效的炮号');
        end
        
        % 显示计算信息
        msg = sprintf(['开始计算单炮梯度：\n', ...
                      '计算策略：%s\n', ...
                      '炮号：%d\n', ...
                      '观测数据：%s\n', ...
                      '合成数据：%s'], ...
                      selected_mode, shot_number, obs_path, syn_path);
        disp(msg);
        
        % 读取JSON并设置参数
        obs_params = jsondecode(fileread(obs_path));
        syn_params = jsondecode(fileread(syn_path));
        
        % 设置计算参数
        params = syn_params;  % 使用合成数据的参数作为基础
        params.compute_kernel = selected_mode;
        params.current_shot = shot_number;
        params.obs_data = obs_params;  % 添加观测数据参数
        
        % 执行梯度计算
        gradient_computation(params);
        
        % 计算完成提示
        msgbox('单炮梯度计算完成', '计算成功');
        
    catch ME
        % 错误处理
        errordlg(['计算失败：' ME.message], '错误');
        disp(['错误详情：' getReport(ME)]);
    end
end

% 关于界面回调函数
function show_about(~, ~)
    % 创建新的诗词界面
    PoemGUI();
end

% 诗词界面函数
function PoemGUI()
    % 创建诗词窗口
    poem_fig = figure('Name', '蝶恋花', ...
                     'Position', [300 200 800 600], ...
                     'Color', [1 1 1], ...
                     'NumberTitle', 'off', ...
                     'MenuBar', 'none', ...
                     'Resize', 'off');  % 禁止调整窗口大小
    
    % 创建主面板
    main_panel = uipanel('Parent', poem_fig, ...
                        'Position', [0.05 0.05 0.9 0.9], ...
                        'BackgroundColor', [1 1 1], ...
                        'BorderType', 'none');
    
    % 加载并显示图片
    try
        img = imread('gui/assets/logo_zhanqiao.jpg');
        ax = axes('Parent', main_panel, ...
                 'Position', [0.1 0.1 0.8 0.45], ...
                 'Box', 'off');
        imshow(img, 'Parent', ax);
        axis off;
    catch
        warning('无法加载图片');
    end
    
    % 创建诗词标题
    title_text = uicontrol('Parent', main_panel, ...
                          'Style', 'text', ...
                          'String', '蝶恋花', ...
                          'Position', [300 500 200 30], ...
                          'FontSize', 16, ...
                          'FontWeight', 'bold', ...
                          'BackgroundColor', [1 1 1], ...
                          'FontName', '楷体');
    
    % 创建诗词内容 - 上阕
    upper_text = uicontrol('Parent', main_panel, ...
                          'Style', 'text', ...
                          'String', ['阅尽天涯离别苦，不道归来，零落花如许。', ...
                                   '花底相看无一语，绿窗春与天俱暮。'], ...
                          'Position', [100 420 600 40], ...
                          'FontSize', 12, ...
                          'FontName', '楷体', ...
                          'BackgroundColor', [1 1 1]);
    
    % 创建诗词内容 - 下阕
    lower_text = uicontrol('Parent', main_panel, ...
                          'Style', 'text', ...
                          'String', ['待把相思灯下诉，一缕新欢，旧恨千千缕。', ...
                                   '最是人间留不住，朱颜辞镜花辞树。'], ...
                          'Position', [100 370 600 40], ...
                          'FontSize', 12, ...
                          'FontName', '楷体', ...
                          'BackgroundColor', [1 1 1]);
    
    % 设置所有文本控件为多行显示
    set([upper_text, lower_text], 'Max', 2);
end 