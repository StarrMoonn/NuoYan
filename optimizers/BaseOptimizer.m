%% VTI介质全波形反演 - 优化器基类
% 功能：提供FWI优化所需的基础功能，作为所有具体优化器的父类
% 
% 说明：
%   1. 主要功能：
%      - 计算目标函数值（残差）
%      - 计算总梯度
%      - 模型参数的获取与设置
%      - 迭代结果的保存与可视化
%
% 类属性：
%   - gradient_solver: 梯度计算器实例
%   - max_iterations: 最大迭代次数
%   - tolerance: 收敛容差
%   - output_dir: 输出目录
%   - gradient_output_dir: 梯度输出目录
%   - misfit_output_dir: 残差输出目录
%
% 核心方法：
% 1. 目标函数计算（compute_misfit）：
%    - 计算所有炮的总目标函数值
%    - 累加每炮的残差
%
% 2. 梯度计算（compute_total_gradient）：
%    - 并行计算每炮梯度
%    - 累加得到总梯度
%    - 包含五个弹性参数：c11, c13, c33, c44, rho
%
% 3. 模型操作：
%    - get_current_model：获取当前模型参数
%    - set_current_model：设置新的模型参数
%
% 4. 结果保存与可视化：
%    - save_iteration_results：保存每次迭代结果
%    - plot_convergence_curve：绘制收敛曲线
%
% 注意事项：
%   - 该类为抽象基类，不能直接实例化
%   - 必须通过子类实现run()方法
% 
% 输入参数：
%   params结构体必须包含：
%   - gradient_solver：梯度计算器实例
%   - max_iterations：最大迭代次数
%   - tolerance：收敛容差
%   - output_dir：输出目录
%   - gradient_output_dir：梯度输出目录
%   - misfit_output_dir：残差输出目录
% 
% 输出：
%   - 迭代梯度：保存在gradient_output_dir
%   - 目标函数值：保存在misfit_output_dir
%   - 收敛曲线：以图片形式保存
% 
% 作者：starrmoonn
% 日期：2025-01-10
% 

classdef BaseOptimizer < handle
    properties
        gradient_solver       % 梯度计算实例
        max_iterations        % 最大迭代次数
        tolerance             % 收敛容差
        output_dir            % 输出目录
        gradient_output_dir   % 梯度输出目录
        misfit_output_dir     % 残差输出目录
    end
    
    methods (Abstract)
        % 必须由子类实现的方法
        run(obj)              % 运行优化算法
    end
    
    methods
        function obj = BaseOptimizer(params)
            % 创建梯度计算实例
            obj.gradient_solver = VTI_Gradient(params);  % 使用VTI_Gradient类
            
            % 设置默认参数
            obj.max_iterations = 50;  % 默认最大迭代次数
            obj.tolerance = 0.1;      % 默认收敛容差
            
            % 设置输出目录
            obj.output_dir = fullfile(params.project_root, 'data', 'output', 'fwi');
            obj.gradient_output_dir = fullfile(params.project_root, 'data', 'output', 'gradient');
            obj.misfit_output_dir = fullfile(params.project_root, 'data', 'output', 'fwi_misfit');
            
            % 创建必要的目录
            if ~exist(obj.output_dir, 'dir')
                mkdir(obj.output_dir);
            end
            if ~exist(obj.gradient_output_dir, 'dir')
                mkdir(obj.gradient_output_dir);
            end
            if ~exist(obj.misfit_output_dir, 'dir')
                mkdir(obj.misfit_output_dir);
            end
        end
        
        %% === 残差梯度计算函数 ===
        function misfit = compute_misfit(obj)
            % 计算所有炮的总目标函数值
            nshots = obj.gradient_solver.adjoint_solver.syn_params.NSHOT;
            misfit = 0;
            
            for ishot = 1:nshots
                obs_vx_shot = obj.gradient_solver.adjoint_solver.obs_vx{ishot};
                obs_vy_shot = obj.gradient_solver.adjoint_solver.obs_vy{ishot};
                syn_vx_shot = obj.gradient_solver.adjoint_solver.syn_vx{ishot};
                syn_vy_shot = obj.gradient_solver.adjoint_solver.syn_vy{ishot};
                
                [shot_misfit, ~] = utils.compute_misfit(obs_vx_shot, ...
                                                     obs_vy_shot, ...
                                                     syn_vx_shot, ...
                                                     syn_vy_shot);
                misfit = misfit + shot_misfit;
            end
            
            fprintf('目标函数残差: %e\n', misfit);
        end
        
        function total_gradient = compute_total_gradient(obj)
            % 计算所有炮的总梯度
            nshots = obj.gradient_solver.adjoint_solver.syn_params.NSHOT;
            nx = obj.gradient_solver.adjoint_solver.syn_params.NX;
            ny = obj.gradient_solver.adjoint_solver.syn_params.NY;
            
            total_gradient = struct();
            total_gradient.c11 = zeros(nx, ny);
            total_gradient.c13 = zeros(nx, ny);
            total_gradient.c33 = zeros(nx, ny);
            total_gradient.c44 = zeros(nx, ny);
            total_gradient.rho = zeros(nx, ny);
            
            shot_gradients = cell(nshots, 1);
            
            fprintf('开始计算%d炮的梯度...\n', nshots);
            parfor ishot = 1:nshots
                fprintf('计算第%d/%d炮梯度...\n', ishot, nshots);
                shot_gradients{ishot} = obj.gradient_solver.compute_single_shot_gradient(ishot);
            end
            
            fprintf('累加所有炮的梯度...\n');
            for ishot = 1:nshots
                total_gradient.c11 = total_gradient.c11 + shot_gradients{ishot}.c11;
                total_gradient.c13 = total_gradient.c13 + shot_gradients{ishot}.c13;
                total_gradient.c33 = total_gradient.c33 + shot_gradients{ishot}.c33;
                total_gradient.c44 = total_gradient.c44 + shot_gradients{ishot}.c44;
                total_gradient.rho = total_gradient.rho + shot_gradients{ishot}.rho;
            end
            
            fprintf('梯度计算完成\n');
        end
        
        %% === 模型操作函数 ===
        function model = get_current_model(obj)
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            model = struct();
            model.c11 = syn_params.c11;
            model.c13 = syn_params.c13;
            model.c33 = syn_params.c33;
            model.c44 = syn_params.c44;
            model.rho = syn_params.rho;
        end
        
        function set_current_model(obj, model)
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            syn_params.c11 = model.c11;
            syn_params.c13 = model.c13;
            syn_params.c33 = model.c33;
            syn_params.c44 = model.c44;
            syn_params.rho = model.rho;
        end

        function [total_improvement, iter_improvement] = compute_improvements(~, initial_misfit, previous_misfit, current_misfit)
            total_improvement = (initial_misfit - current_misfit) / initial_misfit * 100;
            iter_improvement = (previous_misfit - current_misfit) / previous_misfit * 100;
        end
        
        %% === 绘图保存函数 ===
        function save_iteration_results(obj, misfit, total_improvement, iter_improvement, iter)
            misfit_data = struct('misfit', misfit, ...
                              'total_improvement', total_improvement, ...
                              'iter_improvement', iter_improvement);
            obj.save_iteration_misfit(misfit_data, iter);
        end
        
        function save_iteration_gradient(obj, total_gradient, iter)
            filename = sprintf('total_gradient_iter_%d.mat', iter);
            filepath = fullfile(obj.gradient_output_dir, filename);
            save(filepath, 'total_gradient');
            fprintf('迭代%d的总梯度已保存到: %s\n', iter, filepath);
        end
        
        function save_iteration_misfit(obj, misfit_data, iter)
            filename = sprintf('misfit_iter_%d.mat', iter);
            filepath = fullfile(obj.misfit_output_dir, filename);
            save(filepath, 'misfit_data');
            fprintf('迭代%d的残差已保存到: %s\n', iter, filepath);
        end
        
        function plot_convergence_curve(obj, all_misfits)
            figure('Name', 'FWI Convergence Curve', 'NumberTitle', 'off');
            iterations = 0:(length(all_misfits)-1);
            semilogy(iterations, all_misfits, 'b-o', 'LineWidth', 1.5);
            grid on;
            xlabel('迭代次数');
            ylabel('目标函数值 (对数尺度)');
            title('FWI收敛曲线');
            
            for i = 1:length(all_misfits)
                text(iterations(i), all_misfits(i), ...
                     sprintf('%.2e', all_misfits(i)), ...
                     'VerticalAlignment', 'bottom', ...
                     'HorizontalAlignment', 'right');
            end
            
            savefig(fullfile(obj.misfit_output_dir, 'convergence_curve.fig'));
            saveas(gcf, fullfile(obj.misfit_output_dir, 'convergence_curve.png'));
            fprintf('收敛曲线已保存到: %s\n', obj.misfit_output_dir);
        end
    end
end 