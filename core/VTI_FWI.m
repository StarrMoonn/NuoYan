classdef VTI_FWI < handle  
    properties
        gradient_solver              % 梯度计算器实例
        max_iterations              % 最大迭代次数
        tolerance                   % 收敛容差（相对变化）
        output_dir                  % 输出目录
        gradient_output_dir         % 梯度输出目录
        misfit_output_dir          % 残差输出目录
    end
    
    methods
        function obj = VTI_FWI(params)
            % 构造函数
            obj.gradient_solver = VTI_Gradient(params);
            
            % 设置默认值，如果params中有值则使用params中的值
            if isfield(params.fwi, 'max_iterations')
                obj.max_iterations = params.fwi.max_iterations;
            else
                obj.max_iterations = 100;  % 默认值
            end
            
            if isfield(params.fwi, 'tolerance')
                obj.tolerance = params.fwi.tolerance;
            else
                obj.tolerance = 0.1;  % 默认值
            end
            
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
        
        function misfit = compute_misfit(obj)
            % 计算所有炮的总目标函数值
            nshots = obj.gradient_solver.adjoint_solver.syn_params.NSHOT;  % 获取总炮数
            misfit = 0;  % 初始化总误差
            
            % 逐炮计算误差并累加
            for ishot = 1:nshots
                % 获取当前炮的观测数据和合成数据
                obs_vx_shot = obj.gradient_solver.adjoint_solver.obs_vx{ishot};
                obs_vy_shot = obj.gradient_solver.adjoint_solver.obs_vy{ishot};
                syn_vx_shot = obj.gradient_solver.adjoint_solver.syn_vx{ishot};
                syn_vy_shot = obj.gradient_solver.adjoint_solver.syn_vy{ishot};
                
                % 计算当前炮的误差并累加
                [shot_misfit, ~] = utils.compute_misfit(obs_vx_shot, ...
                                                      obs_vy_shot, ...
                                                      syn_vx_shot, ...
                                                      syn_vy_shot);
                misfit = misfit + shot_misfit;
            end
            
            fprintf('总目标函数值: %e\n', misfit);
        end
        
        function total_gradient = compute_total_gradient(obj)
            % 计算所有炮的总梯度
            nshots = obj.gradient_solver.adjoint_solver.syn_params.NSHOT;  % 使用JSON中定义的炮数
            nx = obj.gradient_solver.adjoint_solver.syn_params.NX;        % 模型X方向网格点数
            ny = obj.gradient_solver.adjoint_solver.syn_params.NY;        % 模型Y方向网格点数
            
            % 初始化总梯度
            total_gradient = struct();
            total_gradient.c11 = zeros(nx, ny);
            total_gradient.c13 = zeros(nx, ny);
            total_gradient.c33 = zeros(nx, ny);
            total_gradient.c44 = zeros(nx, ny);
            total_gradient.rho = zeros(nx, ny);
            
            % 创建临时数组存储每炮梯度
            shot_gradients = cell(nshots, 1);
            
            % 并行计算每炮梯度
            fprintf('开始计算%d炮的梯度...\n', nshots);
            parfor ishot = 1:nshots
                fprintf('计算第%d/%d炮梯度...\n', ishot, nshots);
                shot_gradients{ishot} = obj.gradient_solver.compute_single_shot_gradient(ishot);
            end
            
            % 串行累加所有炮的梯度
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
        
        function step = compute_step_length(obj, total_gradient, current_misfit)
            % 使用抛物线法计算步长
            % 测试三个步长点：0, step1, step2
            step1 = 0.1;  % 初始测试步长
            step2 = 0.2;  % 第二个测试步长
            
            % 计算三个点的目标函数值
            misfit0 = current_misfit;  % 当前点的目标函数值
            
            % 保存当前模型
            current_model = obj.get_current_model();
            
            % 测试step1
            obj.update_model_with_step(total_gradient, step1);
            misfit1 = obj.compute_misfit();
            
            % 测试step2
            obj.set_current_model(current_model);  % 恢复原始模型
            obj.update_model_with_step(total_gradient, step2);
            misfit2 = obj.compute_misfit();
            
            % 使用抛物线拟合计算最优步长
            % f(α) = aα² + bα + c
            a = ((misfit1 - misfit0)/step1 - (misfit2 - misfit0)/step2)/(step1 - step2);
            b = (misfit1 - misfit0)/step1 - a*step1;
            
            % 最优步长：-b/(2a)
            if a > 0
                step = -b/(2*a);
            else
                % 如果抛物线开口向下，使用测试点中的最佳值
                [~, idx] = min([misfit0, misfit1, misfit2]);
                steps = [0, step1, step2];
                step = steps(idx);
            end
            
            % 恢复原始模型
            obj.set_current_model(current_model);
            
            fprintf('计算的最优步长: %f\n', step);
        end
        
        function update_model_with_step(obj, total_gradient, step)
            % 更新模型参数
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            
            % 带步长的梯度下降更新（如果梯度已经带负号，这里用加号）
            syn_params.c11 = syn_params.c11 + step * total_gradient.c11;
            syn_params.c13 = syn_params.c13 + step * total_gradient.c13;
            syn_params.c33 = syn_params.c33 + step * total_gradient.c33;
            syn_params.c44 = syn_params.c44 + step * total_gradient.c44;
            syn_params.rho = syn_params.rho + step * total_gradient.rho;
        end
        
        function model = get_current_model(obj)
            % 获取当前模型参数的副本
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            model = struct();
            model.c11 = syn_params.c11;
            model.c13 = syn_params.c13;
            model.c33 = syn_params.c33;
            model.c44 = syn_params.c44;
            model.rho = syn_params.rho;
        end
        
        function set_current_model(obj, model)
            % 设置当前模型参数
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            syn_params.c11 = model.c11;
            syn_params.c13 = model.c13;
            syn_params.c33 = model.c33;
            syn_params.c44 = model.c44;
            syn_params.rho = model.rho;
        end
        
        function run(obj)
            % 主程序流程
            fprintf('\n=== 开始FWI迭代 ===\n');
            
            if isempty(gcp('nocreate'))
                parpool('local');
            end
            
            % 计算初始误差作为基准
            initial_misfit = obj.compute_misfit();
            previous_misfit = initial_misfit;
            all_misfits = zeros(obj.max_iterations, 1);
            all_misfits(1) = initial_misfit;
            
            for iter = 1:obj.max_iterations
                fprintf('\n=== 迭代 %d/%d ===\n', iter, obj.max_iterations);
                
                % 1. 计算总梯度
                total_gradient = obj.compute_total_gradient();
                obj.save_iteration_gradient(total_gradient, iter);
                
                % 2. 计算最优步长（使用已知的误差值）
                step = obj.compute_step_length(total_gradient, previous_misfit);
                
                % 3. 使用最优步长更新模型
                obj.update_model_with_step(total_gradient, step);
                
                % 4. 计算更新后的误差
                current_misfit = obj.compute_misfit();
                all_misfits(iter + 1) = current_misfit;
                
                % 5. 计算改进效果
                total_improvement = (initial_misfit - current_misfit) / initial_misfit * 100;
                iter_improvement = (previous_misfit - current_misfit) / previous_misfit * 100;
                
                fprintf('目标函数值: %e\n', current_misfit);
                fprintf('总体改进: %.2f%%\n', total_improvement);
                fprintf('本次改进: %.2f%%\n', iter_improvement);
                
                % 6. 保存当前迭代信息
                misfit_data = struct('misfit', current_misfit, ...
                                   'total_improvement', total_improvement, ...
                                   'iter_improvement', iter_improvement);
                obj.save_iteration_misfit(misfit_data, iter);
                
                % 7. 检查收敛
                if abs(iter_improvement) < obj.tolerance
                    fprintf('\n=== 已收敛，停止迭代 ===\n');
                    fprintf('最终改进效果: %.2f%%\n', total_improvement);
                    break;
                end
                
                previous_misfit = current_misfit;
            end
            
            % 绘制和保存收敛曲线
            obj.plot_convergence_curve(all_misfits(1:iter+1));
        end
        
        function save_iteration_gradient(obj, total_gradient, iter)
            % 保存每次迭代的总梯度
            filename = sprintf('total_gradient_iter_%d.mat', iter);
            filepath = fullfile(obj.gradient_output_dir, filename);
            save(filepath, 'total_gradient');
            fprintf('迭代%d的总梯度已保存到: %s\n', iter, filepath);
        end
        
        function save_iteration_misfit(obj, misfit_data, iter)
            % 保存每次迭代的残差
            filename = sprintf('misfit_iter_%d.mat', iter);
            filepath = fullfile(obj.misfit_output_dir, filename);
            save(filepath, 'misfit_data');
            fprintf('迭代%d的残差已保存到: %s\n', iter, filepath);
        end
        
        function plot_convergence_curve(obj, all_misfits)
            % 绘制FWI迭代收敛曲线
            figure('Name', 'FWI Convergence Curve', 'NumberTitle', 'off');
            
            % 创建迭代次数数组
            iterations = 0:(length(all_misfits)-1);
            
            % 绘制误差曲线
            semilogy(iterations, all_misfits, 'b-o', 'LineWidth', 1.5);
            
            % 设置图形属性
            grid on;
            xlabel('迭代次数');
            ylabel('目标函数值 (对数尺度)');
            title('FWI收敛曲线');
            
            % 添加数据标签
            for i = 1:length(all_misfits)
                text(iterations(i), all_misfits(i), ...
                     sprintf('%.2e', all_misfits(i)), ...
                     'VerticalAlignment', 'bottom', ...
                     'HorizontalAlignment', 'right');
            end
            
            % 保存图形
            savefig(fullfile(obj.misfit_output_dir, 'convergence_curve.fig'));
            saveas(gcf, fullfile(obj.misfit_output_dir, 'convergence_curve.png'));
            fprintf('收敛曲线已保存到: %s\n', obj.misfit_output_dir);
        end
    end
end 