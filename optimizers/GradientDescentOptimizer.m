%% VTI介质全波形反演 - 梯度下降优化器
% 功能：实现基于梯度下降法的VTI介质全波形反演优化
% 
% 说明：
%   1. 主要功能：
%      - 使用传统梯度下降法进行模型参数优化
%      - 采用抛物线法寻找最优步长
%      - 自动收敛控制和结果保存
%
% 类属性：
%   继承自BaseOptimizer的所有属性，额外包含：
%   - initial_step: 初始步长值
%   - min_step: 最小允许步长
%   - max_step: 最大允许步长
%
% 核心算法流程：
% 1. 初始化阶段：
%    - 计算初始模型的目标函数值
%    - 初始化迭代历史记录
%
% 2. 迭代优化阶段：
%    a) 梯度计算：
%       - 调用compute_total_gradient计算当前模型梯度
%       - 对梯度进行预处理和归一化
%
%    b) 步长优化：
%       - 使用抛物线法寻找最优步长
%       - 测试点：0.1和0.2
%       - 通过抛物线拟合确定最优步长
%       - 步长约束在[min_step, max_step]范围内
%
%    c) 模型更新：
%       - 使用最优步长更新五个弹性参数
%       - 确保更新后的模型物理合理
%
%    d) 收敛检查：
%       - 计算目标函数改进程度
%       - 检查是否满足收敛条件
% -------------------------------------------------------------------------
% 输入参数：
%   params结构体必须包含：
%   - 继承自BaseOptimizer的所有参数
%   - initial_step：初始步长（默认0.1）
%   - min_step：最小步长（默认1e-6）
%   - max_step：最大步长（默认1.0）
% -------------------------------------------------------------------------
% 输出：
%   - 优化后的模型参数
%   - 迭代历史记录
%   - 收敛曲线
% -------------------------------------------------------------------------
% 作者：starrmoonn
% 日期：2025-01-10
% =========================================================================

classdef GradientDescentOptimizer < BaseOptimizer
    methods
        function obj = GradientDescentOptimizer(params)
            % 构造函数，调用父类构造函数
            obj = obj@BaseOptimizer(params);
        end
        
        function run(obj)
            fprintf('\n=== 开始梯度下降法FWI迭代 ===\n');
            
            % 1. 初始化
            initial_misfit = obj.compute_misfit();   % 计算初始状态的目标函数值
            previous_misfit = initial_misfit;        % 保存上一次迭代的目标函数值
            all_misfits = zeros(obj.max_iterations + 1, 1);
            all_misfits(1) = initial_misfit;         % 存储初始目标函数值
            
            % 2. 主迭代循环
            for iter = 1:obj.max_iterations
                % 2.1 打印当前迭代信息
                fprintf('\n=== 第 %d/%d 次迭代 ===\n', iter, obj.max_iterations);
                
                % 2.2 计算当前模型的梯度
                total_gradient = obj.compute_total_gradient();
                obj.save_iteration_gradient(total_gradient, iter);
                
                % 2.3 使用抛物线法计算最优步长
                step = obj.compute_step_length(total_gradient, previous_misfit);
                
                % 2.4 更新模型参数
                obj.update_model_with_step(total_gradient, step);
                
                % 2.5 计算更新后的目标函数值
                current_misfit = obj.compute_misfit();
                all_misfits(iter + 1) = current_misfit;
                
                % 2.6 计算并打印改进程度
                [total_improvement, iter_improvement] = obj.compute_improvements(...
                    initial_misfit, previous_misfit, current_misfit);
                fprintf('当前残差值: %e\n', current_misfit);
                fprintf('总体改进效果: %.2f%%\n', total_improvement);
                fprintf('本次迭代改进: %.2f%%\n', iter_improvement);
                
                % 2.7 保存当前迭代结果
                obj.save_iteration_results(current_misfit, total_improvement, iter_improvement, iter);
                
                % 2.8 检查收敛条件
                if abs(iter_improvement) < obj.tolerance
                    fprintf('\n=== 已收敛，停止迭代 ===\n');
                    break;
                end
                
                % 2.9 更新上一次迭代的目标函数值
                previous_misfit = current_misfit;
            end
            
            % 3. 绘制收敛曲线
            obj.plot_convergence_curve(all_misfits(1:iter+1));
        end
    end
    
    methods (Access = private)
        function update_model_with_step(obj, total_gradient, step)
            % 使用步长更新模型参数
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            syn_params.c11 = syn_params.c11 + step * total_gradient.c11;
            syn_params.c13 = syn_params.c13 + step * total_gradient.c13;
            syn_params.c33 = syn_params.c33 + step * total_gradient.c33;
            syn_params.c44 = syn_params.c44 + step * total_gradient.c44;
            syn_params.rho = syn_params.rho + step * total_gradient.rho;
        end
        
        function step = compute_step_length(obj, total_gradient, current_misfit)
            % 使用抛物线法计算最优步长
            % 输入参数：
            %   total_gradient: 当前迭代的总梯度
            %   current_misfit: 当前的目标函数值
            
            % 1. 设置测试步长
            step1 = 0.1;  % 第一个测试步长
            step2 = 0.2;  % 第二个测试步长
            
            % 2. 保存当前模型
            current_model = obj.get_current_model();
            
            % 3. 计算三个点的目标函数值
            misfit0 = current_misfit;  % 当前点的目标函数值
            
            % 4. 测试step1
            obj.update_model_with_step(total_gradient, step1);
            misfit1 = obj.compute_misfit();
            
            % 5. 测试step2
            obj.set_current_model(current_model);  % 恢复原始模型
            obj.update_model_with_step(total_gradient, step2);
            misfit2 = obj.compute_misfit();
            
            % 6. 使用抛物线拟合计算最优步长
            % f(α) ≈ aα² + bα + c
            a = ((misfit1 - misfit0)/step1 - (misfit2 - misfit0)/step2)/(step1 - step2);
            b = (misfit1 - misfit0)/step1 - a*step1;
            
            % 7. 计算最优步长
            if a > 0  % 抛物线开口向上，存在最小值
                step = -b/(2*a);
            else  % 抛物线开口向下，选择三个点中的最小值
                [~, idx] = min([misfit0, misfit1, misfit2]);
                steps = [0, step1, step2];
                step = steps(idx);
            end
            
            % 8. 恢复原始模型
            obj.set_current_model(current_model);
            
            % 9. 步长约束
            step = max(1e-6, min(1.0, step));
            
            % 调试输出
            fprintf('计算的最优步长: %f\n', step);
        end
    end
end 