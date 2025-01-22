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
% 
% 输入参数：
%   params结构体必须包含：
%   - 继承自BaseOptimizer的所有参数
%   - memory_length：L-BFGS存储长度（默认5）
%   - wolfe_c1：Wolfe条件参数c1（默认1e-4）
%   - wolfe_c2：Wolfe条件参数c2（默认0.9）
% 
% 输出：
%   - 优化后的模型参数
%   - 迭代历史记录
%   - 收敛曲线
% 
% 作者：StarrMoonn
% 日期：2025-01-10
% 

classdef LBFGSOptimizer < BaseOptimizer
    methods
        function obj = LBFGSOptimizer(params)
            % 调用父类构造函数
            obj@BaseOptimizer(params);
        end
        
        function run(obj)
            fprintf('\n=== 开始 L-BFGS FWI 优化 ===\n');
            
            % 获取初始模型
            initial_model = obj.get_current_model();
            
            % 设置优化选项
            options = optimoptions('fminunc', ...
                'Algorithm', 'quasi-newton', ...          % 使用准牛顿法
                'HessianApproximation', 'lbfgs', ...      % 使用 L-BFGS 方法
                'MaxIterations', obj.max_iterations, ...  % 主迭代次数
                'OptimalityTolerance', obj.tolerance, ... % 收敛容差
                'MaxLineSearchIterations', 5, ...         % 每次迭代最多尝试5次步长
                'SpecifyObjectiveGradient', true, ...     % 指定提供梯度
                'Display', 'iter-detailed', ...           % 显示详细迭代信息
                'OutputFcn', @outputFunction);            % 添加输出函数
            
            % 添加输出函数来显示更多信息
            function stop = outputFunction(x, optimValues, state)
                stop = false;
                switch state
                    case 'iter'
                        fprintf('当前迭代: %d, 线搜索次数: %d\n', ...
                            optimValues.iteration, optimValues.funcCount);
                        fprintf('当前步长: %e\n', optimValues.stepsize);
                        fprintf('目标函数值: %e\n', optimValues.fval);
                        fprintf('梯度范数: %e\n\n', optimValues.firstorderopt);
                end
            end
            
            % 定义目标函数（返回误差和梯度）
            function [f, g] = objective(x)
                % 将优化变量转换为模型结构
                current_model = obj.vector_to_model(x);
                
                % 更新当前模型
                obj.set_current_model(current_model);
                
                % 计算总梯度和目标函数值
                g_struct = obj.compute_total_gradient();
                f = obj.get_current_total_misfit();
                
                % 将梯度结构转换为向量
                g = obj.model_to_vector(g_struct);
            end
            
            % 将初始模型转换为向量形式
            x0 = obj.model_to_vector(initial_model);
            
            % 运行优化
            [x_opt, fval, exitflag, output] = fminunc(@objective, x0, options);
            
            % 将最优解转换回模型结构
            final_model = obj.vector_to_model(x_opt);
            
            % 更新最终模型
            obj.set_current_model(final_model);
            
            % 保存优化结果
            obj.save_optimization_results(final_model, fval, output);
        end
        
        % 辅助函数：将模型结构转换为向量
        function v = model_to_vector(~, model)
            v = [model.c11(:); model.c13(:); model.c33(:); 
                 model.c44(:); model.rho(:)];
        end
        
        % 辅助函数：将向量转换为模型结构
        function model = vector_to_model(obj, v)
            % 获取模型尺寸
            nx = obj.gradient_solver.adjoint_solver.syn_params.NX;
            ny = obj.gradient_solver.adjoint_solver.syn_params.NY;
            n = nx * ny;
            
            % 重构模型结构
            model = struct();
            model.c11 = reshape(v(1:n), nx, ny);
            model.c13 = reshape(v(n+1:2*n), nx, ny);
            model.c33 = reshape(v(2*n+1:3*n), nx, ny);
            model.c44 = reshape(v(3*n+1:4*n), nx, ny);
            model.rho = reshape(v(4*n+1:5*n), nx, ny);
        end
        
        % 保存优化结果
        function save_optimization_results(obj, final_model, fval, output)
            % 创建结果结构体
            results = struct();
            results.final_model = final_model;  % 最终模型参数
            results.final_misfit = fval;        % 最终目标函数值
            results.optimization_output = output; % 优化过程信息，包含：
            % - output.iterations: 迭代次数
            % - output.funcCount: 函数评估次数
            % - output.firstorderopt: 一阶最优性
            % - output.message: 终止消息
            % - output.algorithm: 使用的算法
            % - output.stepsize: 每次迭代的步长
            % - output.fval: 每次迭代的函数值
            
            % 添加迭代历史信息的显示
            fprintf('\n=== 优化迭代历史 ===\n');
            fprintf('总迭代次数: %d\n', output.iterations);
            fprintf('函数评估次数: %d\n', output.funcCount);
            fprintf('初始目标函数值: %e\n', output.fval(1));
            fprintf('最终目标函数值: %e\n', fval);
            fprintf('优化改善率: %f%%\n', (output.fval(1) - fval)/output.fval(1)*100);
            
            % 绘制收敛曲线
            figure('Name', 'FWI Convergence History');
            semilogy(1:length(output.fval), output.fval, 'b-o', 'LineWidth', 1.5);
            grid on;
            xlabel('迭代次数');
            ylabel('目标函数值 (对数尺度)');
            title('FWI 优化收敛曲线');
            
            % 保存图像
            savefig(fullfile(obj.output_dir, 'convergence_curve.fig'));
            saveas(gcf, fullfile(obj.output_dir, 'convergence_curve.png'));
            
            % 保存详细结果
            save(fullfile(obj.output_dir, 'lbfgs_optimization_results.mat'), 'results');
            fprintf('优化结果已保存\n');
        end
    end
end