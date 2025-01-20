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
% 作者：StarrMoonn
% 日期：2025-01-10
% 

classdef BaseOptimizer < handle
    properties
        gradient_solver       % 梯度计算实例
        max_iterations        % 最大迭代次数
        tolerance             % 收敛容差
        output_dir            % 输出目录
        misfit_output_dir     % 残差输出目录
        gradient_output_dir   % 梯度输出目录
        nshots              % 炮数
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
            
            % 获取炮数
            obj.nshots = params.syn_params.NSHOT;
            
            % 获取当前脚本的路径
            [script_path, ~, ~] = fileparts(mfilename('fullpath'));
            % 设置项目根目录为当前脚本的上级目录
            project_root = fileparts(script_path);
            
            % 设置输出目录（使用本地获取的project_root）
            obj.output_dir = fullfile(project_root, 'data', 'output', 'fwi');
            obj.misfit_output_dir = fullfile(project_root, 'data', 'output', 'fwi', 'fwi_misfit');
            obj.gradient_output_dir = fullfile(project_root, 'data', 'output', 'fwi', 'fwi_gradient');
            
            % 创建必要的输出目录
            output_dirs = {obj.output_dir, obj.misfit_output_dir, obj.gradient_output_dir};
            for i = 1:length(output_dirs)
                if ~exist(output_dirs{i}, 'dir')
                    mkdir(output_dirs{i});
                end
            end
        end
        
        %% === 总梯度值和总误差函数===    
        function total_gradient = compute_total_gradient(obj)
            % 计算所有炮的总梯度
            nshots = obj.nshots;
            nx = obj.gradient_solver.adjoint_solver.syn_params.NX;
            ny = obj.gradient_solver.adjoint_solver.syn_params.NY;
            
            % 1. 首先逐炮计算梯度并保存到硬盘
            fprintf('\n=== 开始计算各炮梯度 ===\n');
            for ishot = 1:nshots
                fprintf('计算第%d/%d炮梯度...\n', ishot, nshots);
                % compute_single_shot_gradient会自动保存到硬盘并清理内存
                obj.gradient_solver.compute_single_shot_gradient(ishot);
            end
            
            % 2. 初始化总梯度结构
            total_gradient = struct();
            total_gradient.c11 = zeros(nx, ny);
            total_gradient.c13 = zeros(nx, ny);
            total_gradient.c33 = zeros(nx, ny);
            total_gradient.c44 = zeros(nx, ny);
            total_gradient.rho = zeros(nx, ny);
            
            % 3. 从硬盘读取并累加所有炮的梯度
            fprintf('\n=== 开始累加各炮梯度 ===\n');
            for ishot = 1:nshots
                fprintf('读取并累加第%d/%d炮梯度...\n', ishot, nshots);
                
                % 从硬盘读取单炮梯度
                gradient_filename = fullfile(obj.gradient_solver.gradient_output_dir, ...
                                          sprintf('gradient_shot_%d.mat', ishot));
                shot_gradient = load(gradient_filename);
                
                % 累加到总梯度
                total_gradient.c11 = total_gradient.c11 + shot_gradient.gradient.c11;
                total_gradient.c13 = total_gradient.c13 + shot_gradient.gradient.c13;
                total_gradient.c33 = total_gradient.c33 + shot_gradient.gradient.c33;
                total_gradient.c44 = total_gradient.c44 + shot_gradient.gradient.c44;
                total_gradient.rho = total_gradient.rho + shot_gradient.gradient.rho;
                
                % 清理临时变量
                clear shot_gradient;
            end
            
            fprintf('总梯度计算完成\n');
        end

        function misfit = get_current_total_misfit(obj)
            % 直接从硬盘读取已计算的二范数（不重新计算）
            nshots = obj.nshots;
            misfit = 0;
            
            fprintf('\n=== 读取总目标函数值 ===\n');
            for ishot = 1:nshots
                misfit_filename = fullfile(obj.misfit_output_dir, ...
                                         sprintf('misfit_shot_%d.mat', ishot));
                load(misfit_filename, 'shot_misfit');
                misfit = misfit + shot_misfit;
                fprintf('第 %d 炮目标函数值: %e\n', ishot, shot_misfit);
            end
            fprintf('总目标函数值: %e\n', misfit);
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