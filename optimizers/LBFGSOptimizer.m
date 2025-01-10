%% VTI介质全波形反演 - L-BFGS优化器
% 功能：实现基于限制内存BFGS方法的VTI介质全波形反演优化
% 
% 说明：
%   1. 主要功能：
%      - 使用L-BFGS方法进行模型参数优化
%      - 高效处理大规模优化问题
%      - 动态Hessian矩阵近似更新
%
% 类属性：
%   继承自BaseOptimizer的所有属性，额外包含：
%   - memory_length: L-BFGS存储长度
%   - s_list: 位置差向量存储列表
%   - y_list: 梯度差向量存储列表
%   - rho_list: ρ值存储列表
%
% 核心算法流程：
% 1. 初始化阶段：
%    - 初始化存储列表
%    - 计算初始目标函数值和梯度
%
% 2. 迭代优化阶段：
%    a) 搜索方向计算：
%       - 两步循环算法计算搜索方向
%       - 第一循环：从最新到最老的向量
%       - 第二循环：从最老到最新的向量
%       - 使用初始Hessian矩阵近似
%
%    b) 线搜索：
%       - 实现Wolfe条件线搜索
%       - 确保充分下降和曲率条件
%
%    c) 更新存储信息：
%       - 计算并存储新的s和y向量
%       - 维护固定长度的存储列表
%       - 更新ρ值
%
%    d) 收敛检查：
%       - 评估梯度范数
%       - 检查参数更新量
% -------------------------------------------------------------------------
% 输入参数：
%   params结构体必须包含：
%   - 继承自BaseOptimizer的所有参数
%   - memory_length：L-BFGS存储长度（默认5）
%   - wolfe_c1：Wolfe条件参数c1（默认1e-4）
%   - wolfe_c2：Wolfe条件参数c2（默认0.9）
% -------------------------------------------------------------------------
% 输出：
%   - 优化后的模型参数
%   - 迭代历史记录
%   - 收敛曲线
% -------------------------------------------------------------------------
% 作者：starrmoonn
% 日期：2025-01-10
% =========================================================================

classdef LBFGSOptimizer < BaseOptimizer
    properties
        lbfgs_params        % L-BFGS参数
        memory_size        % 存储历史信息的数量
        s_history         % 存储位置差向量
        y_history         % 存储梯度差向量
        rho_history       % 存储ρk = 1/(yk'*sk)
        current_size      % 当前存储的向量数量
    end
    
    methods
        function obj = LBFGSOptimizer(params)
            % 构造函数
            obj = obj@BaseOptimizer(params);
            
            % 初始化L-BFGS特有参数
            obj.lbfgs_params = params.lbfgs_params;
            obj.memory_size = params.lbfgs_params.memory_size;
            obj.s_history = cell(obj.memory_size, 1);
            obj.y_history = cell(obj.memory_size, 1);
            obj.rho_history = zeros(obj.memory_size, 1);
            obj.current_size = 0;
        end
        
        function run(obj)
            fprintf('\n=== 开始L-BFGS法FWI迭代 ===\n');
            
            % 1. 初始化
            initial_misfit = obj.compute_misfit();
            previous_misfit = initial_misfit;
            all_misfits = zeros(obj.max_iterations + 1, 1);
            all_misfits(1) = initial_misfit;
            
            % 保存初始梯度
            current_gradient = obj.compute_total_gradient();
            previous_gradient = current_gradient;
            previous_model = obj.get_current_model();
            
            % 2. 主迭代循环
            for iter = 1:obj.max_iterations
                fprintf('\n=== 第 %d/%d 次迭代 ===\n', iter, obj.max_iterations);
                
                % 2.1 计算搜索方向
                search_direction = obj.compute_lbfgs_direction(current_gradient);
                
                % 2.2 线搜索确定步长
                step = obj.line_search(search_direction, current_gradient, previous_misfit);
                
                % 2.3 更新模型
                obj.update_model_with_step(search_direction, step);
                
                % 2.4 计算新的目标函数值和梯度
                current_misfit = obj.compute_misfit();
                all_misfits(iter + 1) = current_misfit;
                current_gradient = obj.compute_total_gradient();
                
                % 2.5 更新L-BFGS存储信息
                current_model = obj.get_current_model();
                obj.update_lbfgs_memory(previous_model, current_model, ...
                                      previous_gradient, current_gradient);
                
                % 2.6 计算并打印改进程度
                [total_improvement, iter_improvement] = obj.compute_improvements(...
                    initial_misfit, previous_misfit, current_misfit);
                fprintf('当前残差值: %e\n', current_misfit);
                fprintf('总体改进效果: %.2f%%\n', total_improvement);
                fprintf('本次迭代改进: %.2f%%\n', iter_improvement);
                
                % 2.7 保存当前迭代结果
                obj.save_iteration_results(current_misfit, total_improvement, ...
                                        iter_improvement, iter);
                obj.save_iteration_gradient(current_gradient, iter);
                
                % 2.8 检查收敛条件
                if abs(iter_improvement) < obj.tolerance
                    fprintf('\n=== 已收敛，停止迭代 ===\n');
                    break;
                end
                
                % 2.9 更新状态
                previous_misfit = current_misfit;
                previous_gradient = current_gradient;
                previous_model = current_model;
            end
            
            % 3. 绘制收敛曲线
            obj.plot_convergence_curve(all_misfits(1:iter+1));
        end
    end
    
    methods (Access = private)
        function direction = compute_lbfgs_direction(obj, gradient)
            % 使用两步循环递归计算L-BFGS搜索方向
            q = gradient;
            alpha = zeros(obj.current_size, 1);
            
            % 第一步循环
            for i = obj.current_size:-1:1
                idx = obj.get_circular_index(i);
                alpha(i) = obj.rho_history(idx) * ...
                          obj.compute_inner_product(obj.s_history{idx}, q);
                q = obj.subtract_scaled_gradient(q, alpha(i), obj.y_history{idx});
            end
            
            % 计算初始Hessian近似
            if obj.current_size > 0
                idx = obj.get_circular_index(obj.current_size);
                gamma = obj.compute_inner_product(obj.s_history{idx}, obj.y_history{idx}) / ...
                        obj.compute_inner_product(obj.y_history{idx}, obj.y_history{idx});
            else
                gamma = 1.0;
            end
            
            r = obj.scale_gradient(q, gamma);
            
            % 第二步循环
            for i = 1:obj.current_size
                idx = obj.get_circular_index(i);
                beta = obj.rho_history(idx) * ...
                       obj.compute_inner_product(obj.y_history{idx}, r);
                r = obj.add_scaled_gradient(r, alpha(i) - beta, obj.s_history{idx});
            end
            
            % 返回负方向作为搜索方向
            direction = obj.negate_gradient(r);
        end
        
        function step = line_search(obj, direction, gradient, previous_misfit)
            % Wolfe条件线搜索
            c1 = 1e-4;  % Armijo条件参数
            c2 = 0.9;   % 曲率条件参数
            alpha = 1.0; % 初始步长
            
            % 计算当前函数值和方向导数
            f0 = previous_misfit;
            g0 = obj.compute_inner_product(gradient, direction);
            
            % 保存当前模型
            current_model = obj.get_current_model();
            
            % 线搜索迭代
            max_iter = 20;
            for i = 1:max_iter
                % 尝试新步长
                obj.update_model_with_step(direction, alpha);
                f = obj.compute_misfit();
                
                % Armijo条件检查
                if f > f0 + c1 * alpha * g0
                    alpha = alpha * 0.5;
                    obj.set_current_model(current_model);
                    continue;
                end
                
                % 计算新梯度
                new_gradient = obj.compute_total_gradient();
                g = obj.compute_inner_product(new_gradient, direction);
                
                % 曲率条件检查
                if abs(g) <= -c2 * g0
                    break;
                end
                
                % 更新步长
                if g < 0
                    alpha = alpha * 1.2;
                else
                    alpha = alpha * 0.5;
                end
                
                obj.set_current_model(current_model);
            end
            
            step = alpha;
        end
        
        function update_lbfgs_memory(obj, previous_model, current_model, ...
                                   previous_gradient, current_gradient)
            % 更新L-BFGS存储的历史信息
            s = obj.compute_model_difference(current_model, previous_model);
            y = obj.compute_gradient_difference(current_gradient, previous_gradient);
            
            % 计算ρk
            rho = 1.0 / obj.compute_inner_product(y, s);
            
            % 更新循环缓冲区
            idx = mod(obj.current_size, obj.memory_size) + 1;
            obj.s_history{idx} = s;
            obj.y_history{idx} = y;
            obj.rho_history(idx) = rho;
            
            % 更新当前存储大小
            obj.current_size = min(obj.current_size + 1, obj.memory_size);
        end
        
        % 辅助函数
        function idx = get_circular_index(obj, i)
            % 获取循环缓冲区的实际索引
            idx = mod(i - 1, obj.memory_size) + 1;
        end
        
        function prod = compute_inner_product(~, a, b)
            % 计算两个梯度或方向的内积
            fields = fieldnames(a);
            prod = 0;
            for i = 1:length(fields)
                prod = prod + sum(sum(a.(fields{i}) .* b.(fields{i})));
            end
        end
        
        function result = subtract_scaled_gradient(~, a, scale, b)
            % 计算 a - scale * b
            result = struct();
            fields = fieldnames(a);
            for i = 1:length(fields)
                result.(fields{i}) = a.(fields{i}) - scale * b.(fields{i});
            end
        end
        
        function result = add_scaled_gradient(~, a, scale, b)
            % 计算 a + scale * b
            result = struct();
            fields = fieldnames(a);
            for i = 1:length(fields)
                result.(fields{i}) = a.(fields{i}) + scale * b.(fields{i});
            end
        end
        
        function result = scale_gradient(~, a, scale)
            % 计算 scale * a
            result = struct();
            fields = fieldnames(a);
            for i = 1:length(fields)
                result.(fields{i}) = scale * a.(fields{i});
            end
        end
        
        function result = negate_gradient(~, a)
            % 计算 -a
            result = struct();
            fields = fieldnames(a);
            for i = 1:length(fields)
                result.(fields{i}) = -a.(fields{i});
            end
        end
        
        function diff = compute_model_difference(~, model1, model2)
            % 计算两个模型的差异
            diff = struct();
            fields = fieldnames(model1);
            for i = 1:length(fields)
                diff.(fields{i}) = model1.(fields{i}) - model2.(fields{i});
            end
        end
        
        function diff = compute_gradient_difference(~, grad1, grad2)
            % 计算两个梯度的差异
            diff = struct();
            fields = fieldnames(grad1);
            for i = 1:length(fields)
                diff.(fields{i}) = grad1.(fields{i}) - grad2.(fields{i});
            end
        end
    end
end 