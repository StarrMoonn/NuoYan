%% VTI介质全波形反演主控制模块
% 功能：实现VTI介质的全波形反演优化迭代过程
%
% 说明：
%   1. 主要功能：
%      - 执行完整的FWI迭代优化过程
%      - 使用Fletcher-Reeves共轭梯度法
%      - 处理多参数反演（c11, c13, c33, c44, rho）
%      - 实现自适应步长控制和线搜索
%      - 支持模型约束和水层处理
%
% 输入参数：
%   params结构体包含：
%   - initial_model：初始模型参数（c11,c13,c33,c44,rho）
%   - max_iterations：最大迭代次数（默认10）
%   - water_layer：水层厚度（网格点数）
%   - nshots：炮数
%   - modeling_params：正演模拟参数
%   - adjoint_params：伴随计算参数
%   - gradient_params：梯度计算参数
%
% 主要属性：
%   - modeling：VTI_SingleShotModeling实例
%   - adjoint：VTI_Adjoint实例
%   - gradient_calculator：VTI_Gradient实例
%   - current_model：当前模型参数
%   - current_misfit：当前目标函数值（所有炮的L2范数之和）
%   - current_gradient：当前总梯度
%   - search_direction：当前搜索方向
%
% 优化控制参数：
%   - step_length：步长（默认1.0）
%   - step_length_decay：步长衰减因子（默认0.5）
%   - max_update_value：最大更新幅度限制（默认30）
%   - max_line_search：最大线搜索次数（默认10）
%
% 输出处理：
%   - 支持模型、梯度、目标函数值的定期保存
%   - 提供收敛曲线可视化
%   - 记录完整的迭代历史
%
% 注意事项：
%   1. 模型约束：
%      - 确保c11*c33-c13^2 > 0的稳定性条件
%      - 各参数都有物理合理的上下限
%   2. 内存管理：
%      - 每炮的波场计算完即释放
%      - 梯度累加采用即时处理策略
%   3. 水层处理：
%      - 可选的水层梯度置零
%      - 通过water_layer参数控制
%
% 作者：StarrMoonn
% 日期：2025-04-11
%
classdef VTI_FWI < handle  
    properties
        % 基本依赖
        modeling          % VTI_SingleShotModeling实例
        adjoint           % VTI_Adjoint实例
        gradient_calculator % VTI_Gradient实例
        
        % 优化参数
        max_iterations = 10  % 最大迭代次数
        current_iteration = 0 % 当前迭代次数
        step_length = 1.0     % 步长
        step_length_decay = 0.5 % 步长衰减因子
        max_update_value = 30  % 最大更新幅度限制
        max_line_search = 10   % 最大线搜索次数
        
        % 模型和目标函数相关
        current_model     % 当前模型参数
        current_misfit    % 当前目标函数值
        previous_misfit   % 上一步目标函数值
        current_gradient  % 当前梯度
        search_direction  % 搜索方向
        
        % 共轭梯度相关
        previous_gradient        % 上一步梯度
        previous_direction       % 上一步搜索方向
        previous_normalized_grad % 上一步归一化梯度
        beta                     % Fletcher-Reeves系数
        
        % 输出相关
        output_dir        % 基础输出目录
        model_output_dir  % 模型输出目录
        gradient_output_dir % 梯度输出目录
        misfit_output_dir % 目标函数值输出目录
        save_to_disk = true % 是否保存到磁盘
        save_interval = 5  % 保存间隔（迭代次数）
        
        % 迭代历史
        iteration_history % 迭代历史结构体
        
        % 其他参数
        nshots            % 炮数
        model_constraints % 模型约束条件
    end
    
    methods
        function obj = VTI_FWI(params)
            % 初始化
            if isfield(params, 'max_iterations')
                obj.max_iterations = params.max_iterations;
            end
            
            % 设置模型约束
            obj.model_constraints = struct(...
                'c11_min', 1e9, 'c11_max', 100e9, ...
                'c13_min', 0.5e9, 'c13_max', 50e9, ...
                'c33_min', 1e9, 'c33_max', 100e9, ...
                'c44_min', 0.5e9, 'c44_max', 50e9, ...
                'rho_min', 1000, 'rho_max', 5000, ...
                'water_layer', params.water_layer);
            
            % 设置炮数
            obj.nshots = params.nshots;
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.output_dir = fullfile(current_dir, 'data', 'output');
            obj.model_output_dir = fullfile(obj.output_dir, 'fwi', 'models');
            obj.gradient_output_dir = fullfile(obj.output_dir, 'fwi', 'gradients');
            obj.misfit_output_dir = fullfile(obj.output_dir, 'fwi', 'fwi_misfit');
            
            % 创建输出目录
            if obj.save_to_disk
                if ~exist(fullfile(obj.output_dir, 'fwi'), 'dir')
                    mkdir(fullfile(obj.output_dir, 'fwi'));
                end
                if ~exist(obj.model_output_dir, 'dir')
                    mkdir(obj.model_output_dir);
                end
                if ~exist(obj.gradient_output_dir, 'dir')
                    mkdir(obj.gradient_output_dir);
                end
                if ~exist(obj.misfit_output_dir, 'dir')
                    mkdir(obj.misfit_output_dir);
                end
            end
            
            % 初始化迭代历史
            obj.iteration_history = struct(...
                'iteration', [], ...
                'misfit', [], ...
                'gradient_norm', [], ...
                'step_length', []);
            
            % 设置模型、建模器和伴随求解器
            obj.current_model = params.initial_model;
            obj.modeling = VTI_SingleShotModeling(params.modeling_params);
            obj.adjoint = VTI_Adjoint(params.adjoint_params);
            obj.gradient_calculator = VTI_Gradient(params.gradient_params);
        end
        
        % 计算目标函数和梯度（对应原objfun2_bx）
        function [misfit, gradient] = compute_misfit_and_gradient(obj, model)
            % 更新模型参数
            obj.modeling.update_model_params(model);
            
            % 初始化总梯度和目标函数值
            total_gradient = struct('c11', zeros(size(model.c11)), ...
                                  'c13', zeros(size(model.c13)), ...
                                  'c33', zeros(size(model.c33)), ...
                                  'c44', zeros(size(model.c44)), ...
                                  'rho', zeros(size(model.rho)));
            total_misfit = 0;
            
            % 累加所有炮的梯度和目标函数值
            for ishot = 1:obj.nshots
                % 正演模拟
                [~, ~, forward_wavefield] = obj.modeling.forward_modeling_single_shot(ishot);
                
                % 计算梯度和目标函数值
                [shot_gradient, shot_misfit] = obj.gradient_calculator.compute_single_shot_gradient(ishot, forward_wavefield);
                
                % 累加
                fields = fieldnames(total_gradient);
                for i = 1:length(fields)
                    total_gradient.(fields{i}) = total_gradient.(fields{i}) + shot_gradient.(fields{i});
                end
                total_misfit = total_misfit + shot_misfit;
            end
            
            % 水层梯度处理
            gradient = obj.apply_water_layer_mask(total_gradient);
            misfit = total_misfit;
        end
        
        % 添加梯度（用于累加多炮梯度）
        function total_gradient = add_gradient(obj, total_gradient, shot_gradient)
            fields = fieldnames(total_gradient);
            for i = 1:length(fields)
                field = fields{i};
                total_gradient.(field) = total_gradient.(field) + shot_gradient.(field);
            end
        end
        
        % 应用水层掩码
        function gradient = apply_water_layer_mask(obj, gradient)
            water_layer = obj.model_constraints.water_layer;
            if water_layer > 0
                fields = fieldnames(gradient);
                for i = 1:length(fields)
                    field = fields{i};
                    gradient.(field)(1:water_layer, :) = 0;
                end
            end
        end
        
        % 应用模型约束
        function model = apply_model_constraints(obj, model)
            % C11
            model.c11 = max(min(model.c11, obj.model_constraints.c11_max), obj.model_constraints.c11_min);
            % C13
            model.c13 = max(min(model.c13, obj.model_constraints.c13_max), obj.model_constraints.c13_min);
            % C33
            model.c33 = max(min(model.c33, obj.model_constraints.c33_max), obj.model_constraints.c33_min);
            % C44
            model.c44 = max(min(model.c44, obj.model_constraints.c44_max), obj.model_constraints.c44_min);
            % Rho
            model.rho = max(min(model.rho, obj.model_constraints.rho_max), obj.model_constraints.rho_min);
            
            % 保证物理合理性约束: c11*c33-c13^2 > 0
            stability_mask = (model.c11.*model.c33 - model.c13.^2) <= 0;
            if any(stability_mask(:))
                warning('发现不稳定参数组合，正在修正...');
                model.c13(stability_mask) = sqrt(0.99*model.c11(stability_mask).*model.c33(stability_mask));
            end
        end
        
        % 执行Fletcher-Reeves共轭梯度优化
        function run_optimization(obj)
            fprintf('\n====== 开始VTI全波形反演优化 ======\n');
            fprintf('使用Fletcher-Reeves共轭梯度法\n');
            fprintf('最大迭代次数: %d\n\n', obj.max_iterations);
            
            % 初始化迭代
            obj.current_iteration = 1;
            
            % 计算初始目标函数值和梯度
            [obj.current_misfit, obj.current_gradient] = obj.compute_misfit_and_gradient(obj.current_model);
            
            % 保存初始模型
            if obj.save_to_disk
                initial_model = obj.current_model;
                model_file = fullfile(obj.model_output_dir, 'initial_model.mat');
                save(model_file, 'initial_model', '-v7.3');
            end
            
            % 记录初始迭代历史
            obj.iteration_history.iteration(1) = 0;
            obj.iteration_history.misfit(1) = obj.current_misfit;
            obj.iteration_history.gradient_norm(1) = obj.calculate_gradient_norm(obj.current_gradient);
            obj.iteration_history.step_length(1) = 0;
            
            % 第一次迭代（最速下降方向）
            p = obj.normalize_gradient(obj.current_gradient);
            obj.search_direction = obj.negate_gradient(p);
            
            % 主循环
            while obj.current_iteration <= obj.max_iterations
                fprintf('\n--- 迭代 %d/%d ---\n', obj.current_iteration, obj.max_iterations);
                
                % 显示当前状态
                fprintf('当前目标函数值: %e\n', obj.current_misfit);
                fprintf('当前梯度范数: %e\n', obj.calculate_gradient_norm(obj.current_gradient));
                
                % 进行线搜索更新模型
                obj.perform_line_search();
                
                % 保存当前模型和目标函数历史
                if obj.save_to_disk && mod(obj.current_iteration, obj.save_interval) == 0
                    current_model = obj.current_model;
                    model_file = fullfile(obj.model_output_dir, ...
                        sprintf('model_iter_%03d.mat', obj.current_iteration));
                    save(model_file, 'current_model', '-v7.3');
                    
                    % 保存迭代历史
                    obj.save_iteration_history();
                end
                
                % 准备下一次迭代
                if obj.current_iteration < obj.max_iterations
                    % 保存上一步的归一化梯度
                    obj.previous_normalized_grad = p;
                    obj.previous_direction = obj.search_direction;
                    
                    % 计算新的归一化梯度
                    p = obj.normalize_gradient(obj.current_gradient);
                    
                    % 计算Fletcher-Reeves系数
                    obj.beta = obj.compute_FR_coefficient(obj.current_gradient, obj.previous_gradient);
                    
                    % 计算新的搜索方向
                    obj.search_direction = obj.calculate_new_search_direction(p, obj.beta, obj.previous_direction);
                end
                
                % 更新迭代次数
                obj.current_iteration = obj.current_iteration + 1;
            end
            
            fprintf('\n====== VTI全波形反演优化完成 ======\n');
            fprintf('最终目标函数值: %e\n', obj.current_misfit);
            
            % 绘制收敛曲线
            obj.plot_convergence();
        end
        
        % 线搜索
        function perform_line_search(obj)
            % 初始化线搜索参数
            a = obj.step_length;
            ks = 1;
            
            % 计算最大更新量并控制步长
            maxd = obj.calculate_max_update(obj.search_direction, a);
            while maxd > obj.max_update_value
                a = obj.step_length_decay * a;
                maxd = obj.calculate_max_update(obj.search_direction, a);
            end
            
            % 保存旧的目标函数值
            obj.previous_misfit = obj.current_misfit;
            
            % 线搜索过程
            while ks <= obj.max_line_search
                % 试探性更新模型
                trial_model = obj.update_model(obj.current_model, obj.search_direction, a);
                
                % 应用模型约束
                trial_model = obj.apply_model_constraints(trial_model);
                
                % 计算新模型的目标函数和梯度
                [trial_misfit, trial_gradient] = obj.compute_misfit_and_gradient(trial_model);
                
                fprintf('线搜索 %d: 步长 = %.4e, 目标函数值 = %.4e\n', ks, a, trial_misfit);
                
                % 判断是否接受更新
                if trial_misfit < obj.current_misfit
                    fprintf('接受更新: 目标函数值从 %.4e 减小到 %.4e\n', obj.current_misfit, trial_misfit);
                    
                    % 更新当前状态
                    obj.current_model = trial_model;
                    obj.current_misfit = trial_misfit;
                    obj.previous_gradient = obj.current_gradient;
                    obj.current_gradient = trial_gradient;
                    
                    % 记录迭代历史
                    obj.iteration_history.iteration(end+1) = obj.current_iteration;
                    obj.iteration_history.misfit(end+1) = obj.current_misfit;
                    obj.iteration_history.gradient_norm(end+1) = obj.calculate_gradient_norm(obj.current_gradient);
                    obj.iteration_history.step_length(end+1) = a;
                    
                    break;
                else
                    % 减小步长
                    a = obj.step_length_decay * a;
                    ks = ks + 1;
                end
            end
            
            % 更新步长以便下次使用
            obj.step_length = a;
        end
        
        % 计算梯度范数
        function norm_val = calculate_gradient_norm(obj, gradient)
            norm_sq = 0;
            fields = fieldnames(gradient);
            for i = 1:length(fields)
                field = fields{i};
                norm_sq = norm_sq + sum(gradient.(field)(:).^2);
            end
            norm_val = sqrt(norm_sq);
        end
        
        % 归一化梯度
        function normalized = normalize_gradient(obj, gradient)
            norm_val = obj.calculate_gradient_norm(gradient);
            if norm_val < eps
                warning('梯度范数接近零，使用原始梯度');
                normalized = gradient;
                return;
            end
            
            normalized = struct();
            fields = fieldnames(gradient);
            for i = 1:length(fields)
                field = fields{i};
                normalized.(field) = 100 * gradient.(field) / norm_val;
            end
        end
        
        % 取负梯度
        function negative = negate_gradient(obj, gradient)
            negative = struct();
            fields = fieldnames(gradient);
            for i = 1:length(fields)
                field = fields{i};
                negative.(field) = -2 * gradient.(field);
            end
        end
        
        % 计算Fletcher-Reeves系数
        function beta = compute_FR_coefficient(obj, current_grad, previous_grad)
            % 计算当前梯度和前一步梯度的内积比
            fields = fieldnames(current_grad);
            current_norm = 0;
            previous_norm = 0;
            
            for i = 1:length(fields)
                field = fields{i};
                current_norm = current_norm + sum(sum(current_grad.(field).^2));
                previous_norm = previous_norm + sum(sum(previous_grad.(field).^2));
            end
            
            % Fletcher-Reeves公式
            if previous_norm < 1e-10
                beta = 0;
            else
                beta = current_norm / previous_norm;
            end
        end
        
        % 计算新的搜索方向
        function direction = calculate_new_search_direction(obj, grad, beta, prev_dir)
            direction = struct();
            fields = fieldnames(grad);
            for i = 1:length(fields)
                field = fields{i};
                direction.(field) = -grad.(field) + beta * prev_dir.(field);
            end
        end
        
        % 计算最大更新量
        function max_value = calculate_max_update(obj, direction, step)
            max_value = 0;
            fields = fieldnames(direction);
            for i = 1:length(fields)
                field = fields{i};
                max_value = max(max_value, max(abs(direction.(field)(:) * step)));
            end
        end
        
        % 更新模型
        function new_model = update_model(obj, model, direction, alpha)
            fields = fieldnames(model);
            new_model = struct();
            
            for i = 1:length(fields)
                field = fields{i};
                if isfield(direction, field)
                    new_model.(field) = model.(field) + alpha * direction.(field);
                else
                    new_model.(field) = model.(field);
                end
            end
        end
        
        % 保存迭代历史
        function save_iteration_history(obj)
            if obj.save_to_disk
                history_file = fullfile(obj.misfit_output_dir, 'fwi_iteration_history.mat');
                iteration_history = obj.iteration_history;
                save(history_file, 'iteration_history', '-v7.3');
            end
        end
        
        % 绘制收敛曲线
        function plot_convergence(obj)
            figure;
            
            % 目标函数值收敛曲线
            subplot(2,2,1);
            semilogy(obj.iteration_history.iteration, ...
                    obj.iteration_history.misfit, 'b-o');
            title('目标函数收敛曲线');
            xlabel('迭代次数');
            ylabel('目标函数值');
            grid on;
            
            % 梯度范数收敛曲线
            subplot(2,2,2);
            semilogy(obj.iteration_history.iteration, ...
                    obj.iteration_history.gradient_norm, 'r-o');
            title('梯度范数收敛曲线');
            xlabel('迭代次数');
            ylabel('梯度范数');
            grid on;
            
            % 步长变化曲线
            subplot(2,2,3);
            semilogy(obj.iteration_history.iteration(2:end), ...
                    obj.iteration_history.step_length(2:end), 'g-o');
            title('步长变化曲线');
            xlabel('迭代次数');
            ylabel('步长');
            grid on;
            
            % 保存图像
            if obj.save_to_disk
                saveas(gcf, fullfile(obj.misfit_output_dir, 'convergence_curves.fig'));
                saveas(gcf, fullfile(obj.misfit_output_dir, 'convergence_curves.png'));
            end
        end
        
        % 在VTI_FWI中增加步长控制函数
        function [alpha, max_updates] = control_step_length(obj, direction, model)
            % 初始步长
            alpha = 1.0;
            dec_factor = 0.5;
            
            % 不同参数的最大允许更新量
            max_allowed = struct('c11', 30e9, 'c13', 20e9, 'c33', 30e9, 'c44', 15e9, 'rho', 100);
            
            % 计算每个参数的最大更新量
            fields = fieldnames(direction);
            max_updates = struct();
            
            for i = 1:length(fields)
                field = fields{i};
                max_updates.(field) = max(abs(direction.(field)(:))) * alpha;
                
                % 如果某参数更新量过大，减小步长
                while max_updates.(field) > max_allowed.(field)
                    alpha = alpha * dec_factor;
                    max_updates.(field) = max(abs(direction.(field)(:))) * alpha;
                end
            end
        end
        
        % 在VTI_FWI中增加线搜索函数
        function [optimal_alpha, new_model, new_misfit] = line_search(obj, model, direction, initial_alpha)
            alpha = initial_alpha;
            dec_factor = 0.5;
            max_trials = 10;
            
            % 计算当前目标函数值
            [current_misfit, ~] = obj.compute_misfit_and_gradient(model);
            
            for trial = 1:max_trials
                % 试探性更新模型
                new_model = obj.update_model(model, direction, alpha);
                
                % 应用模型约束
                new_model = obj.apply_model_constraints(new_model);
                
                % 计算新模型的目标函数值(不计算梯度可节省时间)
                new_misfit = obj.compute_misfit_only(new_model);
                
                % 检查目标函数是否减小
                if new_misfit < current_misfit
                    optimal_alpha = alpha;
                    return;
                else
                    % 减小步长继续尝试
                    alpha = alpha * dec_factor;
                end
            end
            
            % 如果没有找到合适的步长，返回一个较小的值
            optimal_alpha = alpha;
            new_model = obj.update_model(model, direction, optimal_alpha);
            new_model = obj.apply_model_constraints(new_model);
            new_misfit = obj.compute_misfit_only(new_model);
        end
        
        % 在VTI_FWI中实现共轭梯度优化主循环
        function run_CG_optimization(obj)
            % 初始化
            current_model = obj.initial_model;
            [misfit, gradient] = obj.compute_misfit_and_gradient(current_model);
            
            % 第一次迭代使用最速下降法方向
            current_direction = struct();
            fields = fieldnames(gradient);
            for i = 1:length(fields)
                field = fields{i};
                current_direction.(field) = -gradient.(field);
            end
            
            % 优化迭代
            for iter = 1:obj.max_iterations
                % 步长控制
                [initial_alpha, max_updates] = obj.control_step_length(current_direction, current_model);
                
                % 线搜索
                [optimal_alpha, new_model, new_misfit] = obj.line_search(current_model, current_direction, initial_alpha);
                
                % 计算新模型的梯度
                [~, new_gradient] = obj.compute_misfit_and_gradient(new_model);
                
                % 保存结果
                if mod(iter, obj.save_interval) == 0 || iter == 1 || iter == obj.max_iterations
                    obj.save_iteration_results(iter, new_model, new_gradient, new_misfit);
                end
                
                % 计算Fletcher-Reeves系数
                beta = obj.compute_FR_coefficient(new_gradient, gradient);
                
                % 计算新的搜索方向
                new_direction = struct();
                for i = 1:length(fields)
                    field = fields{i};
                    new_direction.(field) = -new_gradient.(field) + beta * current_direction.(field);
                end
                
                % 更新迭代变量
                current_model = new_model;
                gradient = new_gradient;
                current_direction = new_direction;
                misfit = new_misfit;
                
                % 输出迭代信息
                fprintf('迭代 %d: 目标函数值 = %e, 步长 = %e\n', iter, misfit, optimal_alpha);
                
                % 检查收敛条件
                if obj.check_convergence(iter, misfit)
                    break;
                end
            end
        end
    end
end 