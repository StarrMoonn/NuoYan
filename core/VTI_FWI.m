classdef VTI_FWI < handle  
    properties
        gradient_solver              % 梯度计算器实例
        max_iterations               % 最大迭代次数
        tolerance                    % 收敛容差（相对变化）
        history                      % 存储迭代历史
        output_dir                   % 输出目录
    end
    
    methods
        function obj = VTI_FWI(params)
            % 构造函数
            obj.gradient_solver = VTI_Gradient(params);
            obj.max_iterations = params.fwi.max_iterations;
            obj.tolerance = params.fwi.tolerance;
            obj.history = struct('misfit', [], 'model', [], 'step_length', []);
            obj.output_dir = fullfile(params.project_root, 'data', 'output', 'fwi');
            
            if ~exist(obj.output_dir, 'dir')
                mkdir(obj.output_dir);
            end
        end
        
        function total_gradient = compute_total_gradient(obj)
            % 计算所有炮的总梯度（并行版本）
            nshots = length(obj.gradient_solver.adjoint_solver.syn_params.source_positions);
            
            % 初始化总梯度
            total_gradient = struct();
            total_gradient.c11 = zeros(801, 201);
            total_gradient.c13 = zeros(801, 201);
            total_gradient.c33 = zeros(801, 201);
            total_gradient.c44 = zeros(801, 201);
            total_gradient.rho = zeros(801, 201);
            
            % 创建临时数组存储每炮梯度
            shot_gradients = cell(nshots, 1);
            
            % 并行计算每炮梯度
            fprintf('开始并行计算梯度...\n');
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
        
        function update_model(obj, total_gradient)
            % 更新模型参数
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            
            % 更新各向异性参数
            syn_params.c11 = syn_params.c11 - obj.step_length * total_gradient.c11;
            syn_params.c13 = syn_params.c13 - obj.step_length * total_gradient.c13;
            syn_params.c33 = syn_params.c33 - obj.step_length * total_gradient.c33;
            syn_params.c44 = syn_params.c44 - obj.step_length * total_gradient.c44;
            syn_params.rho = syn_params.rho - obj.step_length * total_gradient.rho;
        end
        
        function misfit = compute_misfit(obj)
            % 计算目标函数值
            misfit = obj.gradient_solver.adjoint_solver.compute_total_misfit();
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
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            
            % 带步长的梯度下降更新
            syn_params.c11 = syn_params.c11 - step * total_gradient.c11;
            syn_params.c13 = syn_params.c13 - step * total_gradient.c13;
            syn_params.c33 = syn_params.c33 - step * total_gradient.c33;
            syn_params.c44 = syn_params.c44 - step * total_gradient.c44;
            syn_params.rho = syn_params.rho - step * total_gradient.rho;
        end
        
        function run(obj)
            fprintf('\n=== 开始FWI迭代 ===\n');
            
            if isempty(gcp('nocreate'))
                parpool('local');  % 或指定具体的工作进程数
            end
            
            for iter = 1:obj.max_iterations
                fprintf('\n迭代 %d/%d\n', iter, obj.max_iterations);
                
                % 1. 计算总梯度
                total_gradient = obj.compute_total_gradient();
                
                % 2. 计算当前目标函数值
                current_misfit = obj.compute_misfit();
                
                % 3. 计算最优步长
                step = obj.compute_step_length(total_gradient, current_misfit);
                
                % 4. 使用最优步长更新模型
                obj.update_model_with_step(total_gradient, step);
                
                % 5. 计算新的目标函数值
                new_misfit = obj.compute_misfit();
                
                % 6. 保存历史记录
                obj.history.misfit(iter) = new_misfit;
                obj.history.step_length(iter) = step;
                
                % 7. 输出当前状态
                fprintf('目标函数值: %e (改善: %e)\n', ...
                    new_misfit, current_misfit - new_misfit);
                fprintf('使用步长: %e\n', step);
                
                % 8. 检查收敛
                if iter > 1 && abs(new_misfit - current_misfit) < obj.tolerance
                    fprintf('已收敛，停止迭代\n');
                    break;
                end
            end
            
            % 保存反演结果
            save(fullfile(obj.output_dir, 'fwi_results.mat'), 'obj');
        end
    end
end 