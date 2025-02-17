classdef FletcherReevesCGOptimizer < BaseOptimizer
    properties (Access = private)
        p0              % 上一次的归一化梯度方向
        d0              % 上一次的搜索方向
        a2              % 上一次的步长
        objold          % 上一次的目标函数值
        objval          % 存储所有迭代的目标函数值
        dec             % 步长衰减因子
        max_line_search % 最大线搜索次数
        max_step_size   % 最大更新步长
    end
    
    methods
        function obj = FletcherReevesCGOptimizer(params)
            obj@BaseOptimizer(params);
            % 设置优化器特定参数
            obj.dec = 0.5;
            obj.max_line_search = 10;
            obj.max_step_size = 30;
        end
        
        function run(obj)
            % 初始化迭代参数
            k = 1;
            obj.objval = zeros(obj.max_iterations, 1);
            
            % 获取初始梯度和目标函数值
            objval0 = obj.get_current_total_misfit();
            grad_stk1 = obj.compute_total_gradient();
            
            % 显示初始状态
            obj.display_iteration_info(k, grad_stk1);
            
            % 记录初始目标函数值
            obj.objval(k+1) = objval0;
            
            % 第一次迭代使用最速下降法
            [p, d1] = obj.compute_steepest_descent(grad_stk1);
            
            % 执行线搜索
            [v0, objval0, grad_stk1, success] = obj.line_search(d1, objval0);
            if ~success
                return;
            end
            
            % 保存方向和步长信息
            obj.p0 = p;
            obj.d0 = d1;
            
            % 保存当前迭代结果
            obj.save_iteration_results(k, v0, grad_stk1);
            
            % 主迭代循环
            k = k + 1;
            while k < obj.max_iterations
                % 显示当前迭代信息
                obj.display_iteration_info(k, grad_stk1);
                
                % 记录目标函数值
                obj.objval(k+1) = objval0;
                
                % 计算共轭梯度方向
                [p, d1] = obj.compute_conjugate_direction(grad_stk1);
                
                % 执行线搜索
                [v0, objval0, grad_stk1, success] = obj.line_search(d1, objval0);
                if ~success
                    break;
                end
                
                % 保存方向和步长信息
                obj.p0 = p;
                obj.d0 = d1;
                
                % 保存当前迭代结果
                if mod(k, 5) == 0
                    obj.save_iteration_results(k, v0, grad_stk1);
                end
                
                % 检查收敛性
                if obj.check_convergence(objval0)
                    break;
                end
                
                k = k + 1;
            end
            
            % 保存最终的目标函数值
            save(fullfile(obj.misfit_output_dir, 'objval.mat'), 'objval');
        end
        
        function [p, d] = compute_steepest_descent(obj, grad)
            % 计算归一化梯度方向
            p = 100 * grad / norm(grad(:));
            % 计算最速下降方向
            d = -2 * p;
        end
        
        function [p, d] = compute_conjugate_direction(obj, grad)
            % 计算归一化当前梯度
            p = 100 * grad / norm(grad(:));
            % 计算Fletcher-Reeves共轭因子
            beta = sqrt(sum(sum((p(:)'*p(:)).^2)) / sum(sum((obj.p0(:)'*obj.p0(:)).^2)));
            % 计算共轭方向
            d = -p + beta * obj.d0;
        end
        
        function [v0, objval0, grad_stk1, success] = line_search(obj, d1, objval0)
            success = false;
            ks = 1;
            a = 1;
            
            % 控制最大更新步长
            maxd = max(abs(d1(:)));
            while maxd > obj.max_step_size
                a = obj.dec * a;
                maxd = max(abs(a * d1(:)));
            end
            
            % 线搜索循环
            while ks < obj.max_line_search
                % 更新模型
                v0_new = obj.get_current_model() + a * d1;
                v0_new = obj.apply_constraints(v0_new);
                
                % 计算新的目标函数值和梯度
                objval1 = obj.get_current_total_misfit();
                grad_stk2 = obj.compute_total_gradient();
                
                % 判断是否接受更新
                if objval1 < objval0
                    v0 = v0_new;
                    obj.objold = objval0;
                    objval0 = objval1;
                    grad_stk1 = grad_stk2;
                    success = true;
                    break;
                else
                    a = obj.dec * a;
                    obj.a2 = a;
                    ks = ks + 1;
                end
            end
        end
        
        function display_iteration_info(obj, k, grad)
            fprintf('Iteration %d\n', k);
            disp('%%%%%%%%%%%%%%%%%%%%iteration time%%%%%%%%%%%%%%%%%%%%%%');
            figure(1); imagesc(grad); colorbar; title(['Gradient at iteration ' num2str(k)]);
            pause(0.001);
        end
        
        function converged = check_convergence(obj, current_misfit)
            if current_misfit < obj.tolerance
                converged = true;
                fprintf('Optimization converged: misfit below tolerance\n');
            else
                converged = false;
            end
        end
        
        function model = apply_constraints(obj, model)
            % 应用模型约束（可以根据具体需求实现）
            % 例如：限制速度范围等
            model = max(min(model, obj.model_max), obj.model_min);
        end
        
        function save_iteration_results(obj, k, v0, grad)
            save(fullfile(obj.output_dir, ['k=' num2str(k) '_v0.mat']), 'v0');
            save(fullfile(obj.gradient_output_dir, ['k=' num2str(k) '_grad.mat']), 'grad');
        end
    end
end