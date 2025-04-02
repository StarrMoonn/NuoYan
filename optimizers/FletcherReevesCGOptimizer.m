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
        max_iterations  % 最大迭代次数
        model_min       % 模型参数最小值
        model_max       % 模型参数最大值
        water_layer     % 水层深度（单位：网格点数）
    end
    
    methods
        function obj = FletcherReevesCGOptimizer(params)
            obj@BaseOptimizer(params);
            % 设置优化器特定参数
            obj.dec = 0.5;
            obj.max_line_search = 10;
            obj.max_step_size = 30;
            obj.max_iterations = 500;  % 设置默认最大迭代次数
            
            % 如果params中包含max_iterations，则使用用户指定的值
            if isfield(params, 'max_iterations')
                obj.max_iterations = params.max_iterations;
            end
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
            [v0, objval0, grad_stk1, success, a] = obj.line_search(d1, objval0);
            if ~success
                fprintf('初始线搜索失败，优化终止\n');
                return;
            end
            
            % 保存方向和步长信息
            obj.p0 = p;
            obj.d0 = a * d1;  % 保存实际使用的搜索方向
            
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
                [v0, objval0, grad_stk1, success, a] = obj.line_search(d1, objval0);
                if ~success
                    fprintf('线搜索失败，优化在第%d次迭代终止\n', k);
                    break;
                end
                
                % 保存方向和步长信息
                obj.p0 = p;
                obj.d0 = a * d1;  % 保存实际使用的搜索方向
                
                % 保存当前迭代结果
                if mod(k, 5) == 0
                    obj.save_iteration_results(k, v0, grad_stk1);
                end
                
                % 检查收敛性
                if obj.check_convergence(objval0)
                    fprintf('优化已收敛，在第%d次迭代达到容差\n', k);
                    break;
                end
                
                k = k + 1;
            end
            
            % 保存最终的目标函数值
            save(fullfile(obj.misfit_output_dir, 'objval.mat'), 'objval');
            
            % 绘制收敛曲线
            obj.plot_convergence_curve(obj.objval(1:k+1));
        end
        
        function [p, d] = compute_steepest_descent(obj, grad)
            % 计算归一化梯度方向
            grad_norm = sqrt(sum(struct2array(grad).^2, 'all'));
            
            % 创建归一化梯度结构体
            p = struct();
            fields = fieldnames(grad);
            for i = 1:length(fields)
                p.(fields{i}) = 100 * grad.(fields{i}) / grad_norm;
            end
            
            % 计算最速下降方向
            d = struct();
            for i = 1:length(fields)
                d.(fields{i}) = -2 * p.(fields{i});
            end
        end
        
        function [p, d] = compute_conjugate_direction(obj, grad)
            % 计算归一化当前梯度
            grad_norm = sqrt(sum(struct2array(grad).^2, 'all'));
            
            % 创建归一化梯度结构体
            p = struct();
            fields = fieldnames(grad);
            for i = 1:length(fields)
                p.(fields{i}) = 100 * grad.(fields{i}) / grad_norm;
            end
            
            % 计算Fletcher-Reeves共轭因子
            p_squared_sum = 0;
            p0_squared_sum = 0;
            
            for i = 1:length(fields)
                p_squared_sum = p_squared_sum + sum(p.(fields{i}).^2, 'all');
                p0_squared_sum = p0_squared_sum + sum(obj.p0.(fields{i}).^2, 'all');
            end
            
            beta = sqrt(p_squared_sum / p0_squared_sum);
            
            % 计算共轭方向
            d = struct();
            for i = 1:length(fields)
                d.(fields{i}) = -p.(fields{i}) + beta * obj.d0.(fields{i});
            end
        end
        
        function [v0, objval0, grad_stk1, success, a] = line_search(obj, d1, objval0)
            success = false;
            ks = 1;
            a = 1;
            
            % 控制最大更新步长
            maxd = 0;
            fields = fieldnames(d1);
            for i = 1:length(fields)
                maxd = max(maxd, max(abs(d1.(fields{i})(:))));
            end
            
            while maxd > obj.max_step_size
                a = obj.dec * a;
                maxd = maxd * obj.dec;
            end
            
            fprintf('初始步长: %f, 最大更新量: %f\n', a, maxd);
            
            % 线搜索循环
            while ks < obj.max_line_search
                % 获取当前模型
                v0 = obj.get_current_model();
                
                % 更新模型
                v0_new = struct();
                for i = 1:length(fields)
                    v0_new.(fields{i}) = v0.(fields{i}) + a * d1.(fields{i});
                end
                
                % 应用模型约束
                v0_new = obj.apply_constraints(v0_new);
                
                % 设置新模型
                obj.set_current_model(v0_new);
                
                % 计算新的目标函数值和梯度
                objval1 = obj.get_current_total_misfit();
                grad_stk2 = obj.compute_total_gradient();
                
                % 水层梯度置零
                grad_stk2 = obj.apply_water_layer_mask(grad_stk2);
                
                % 显示当前搜索信息
                fprintf('线搜索 #%d: 旧目标函数值 = %e, 新目标函数值 = %e\n', ks, objval0, objval1);
                
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
                    
                    % 恢复原始模型
                    obj.set_current_model(v0);
                end
            end
            
            if ~success
                fprintf('线搜索失败: 尝试了%d次步长调整\n', ks);
            else
                fprintf('线搜索成功: 使用步长 a = %f\n', a);
            end
        end
        
        function display_iteration_info(obj, k, grad)
            fprintf('迭代 %d\n', k);
            disp('%%%%%%%%%%%%%%%%%%%%iteration time%%%%%%%%%%%%%%%%%%%%%%');
            
            % 显示各参数的梯度
            fields = fieldnames(grad);
            for i = 1:length(fields)
                figure(i);
                imagesc(grad.(fields{i}));
                colorbar;
                title(sprintf('参数 %s 的梯度 (迭代 %d)', fields{i}, k));
                pause(0.001);
            end
        end
        
        function converged = check_convergence(obj, current_misfit)
            if current_misfit < obj.tolerance
                converged = true;
                fprintf('优化已收敛: 目标函数值 %e 低于容差 %e\n', current_misfit, obj.tolerance);
            else
                converged = false;
            end
        end
        
        function model = apply_constraints(obj, model)
            % 应用模型约束
            fields = fieldnames(model);
            for i = 1:length(fields)
                if isfield(obj.model_min, fields{i}) && isfield(obj.model_max, fields{i})
                    model.(fields{i}) = max(min(model.(fields{i}), obj.model_max.(fields{i})), obj.model_min.(fields{i}));
                end
            end
        end
        
        function grad = apply_water_layer_mask(obj, grad)
            % 水层梯度置零
            if obj.water_layer > 0
                fields = fieldnames(grad);
                for i = 1:length(fields)
                    grad.(fields{i})(1:obj.water_layer, :) = 0;
                end
            end
        end
        
        function save_iteration_results(obj, k, v0, grad)
            % 保存当前迭代的模型
            save(fullfile(obj.output_dir, ['k=' num2str(k) '_model.mat']), 'v0');
            
            % 保存当前迭代的梯度
            save(fullfile(obj.gradient_output_dir, ['k=' num2str(k) '_grad.mat']), 'grad');
            
            % 保存当前迭代的目标函数值
            misfit = obj.objval(k+1);
            save(fullfile(obj.misfit_output_dir, ['k=' num2str(k) '_misfit.mat']), 'misfit');
        end
    end
end