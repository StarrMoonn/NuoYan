%% VTI介质全波形反演主控制模块
% 功能：实现VTI介质的全波形反演优化迭代过程
%
% 说明：
%   1. 主要功能：
%      - 执行完整的FWI迭代优化过程
%      - 使用Fletcher-Reeves共轭梯度法
%
% 输入参数：
%   params结构体包含：
%   - optimization_method：优化方法选择
%     * 'gradient_descent'：梯度下降法（默认）
%     * 'BB'：BB算法
%     * 'LBFGS'：L-BFGS算法
%     * 'FletcherReevesCG'：Fletcher-Reeves共轭梯度法
%   - 其他优化器所需参数
%
% 作者：StarrMoonn
% 日期：2025-01-10
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
            misfit = 0;
            gradient = struct('c11', zeros(size(model.c11)), ...
                             'c13', zeros(size(model.c13)), ...
                             'c33', zeros(size(model.c33)), ...
                             'c44', zeros(size(model.c44)), ...
                             'rho', zeros(size(model.rho)));
            
            % 更新模型参数
            obj.modeling.update_model_params(model);
            
            % 遍历所有炮集
            for ishot = 1:obj.nshots
                % 1. 正演模拟
                [vx_syn, vy_syn, forward_wavefield] = obj.modeling.forward_modeling_single_shot(ishot);
                
                % 2. 计算残差和局部目标函数
                shot_misfit = obj.adjoint.compute_residuals_single_shot(ishot);
                misfit = misfit + shot_misfit;
                
                % 保存misfit到文件
                if obj.save_to_disk
                    misfit_file = fullfile(obj.misfit_output_dir, ...
                        sprintf('shot_%03d_misfit_iter_%03d.mat', ishot, obj.current_iteration));
                    save(misfit_file, 'shot_misfit', '-v7.3');
                end
                
                % 3-4. 计算伴随波场和梯度合并处理
                shot_gradient = obj.gradient_calculator.compute_single_shot_gradient(ishot, forward_wavefield);
                gradient = obj.add_gradient(gradient, shot_gradient);
            end
            
            % 处理水层梯度
            gradient = obj.apply_water_layer_mask(gradient);
            
            % 保存总梯度，每save_interval次迭代保存一次
            if obj.save_to_disk && (mod(obj.current_iteration, obj.save_interval) == 0 || obj.current_iteration == 1)
                gradient_file = fullfile(obj.gradient_output_dir, ...
                    sprintf('gradient_iter_%03d.mat', obj.current_iteration));
                save(gradient_file, 'gradient', '-v7.3');
                
                % 同时保存当前模型（如果需要）
                model_file = fullfile(obj.model_output_dir, ...
                    sprintf('model_iter_%03d.mat', obj.current_iteration));
                current_model = model;
                save(model_file, 'current_model', '-v7.3');
            end
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
                    obj.beta = obj.calculate_fletcher_reeves_beta(p, obj.previous_normalized_grad);
                    
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
        function beta = calculate_fletcher_reeves_beta(obj, current, previous)
            current_sq = 0;
            previous_sq = 0;
            
            fields = fieldnames(current);
            for i = 1:length(fields)
                field = fields{i};
                current_sq = current_sq + sum(current.(field)(:).^2);
                previous_sq = previous_sq + sum(previous.(field)(:).^2);
            end
            
            if previous_sq < eps
                beta = 0;
            else
                beta = sqrt(current_sq / previous_sq);
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
        function new_model = update_model(obj, model, direction, step)
            new_model = struct();
            fields = fieldnames(model);
            for i = 1:length(fields)
                field = fields{i};
                if isfield(direction, field)
                    new_model.(field) = model.(field) + direction.(field) * step;
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
    end
end 