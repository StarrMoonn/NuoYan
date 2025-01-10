%% VTI介质全波形反演 - BB (Barzilai-Borwein) 优化器
% 功能：实现基于BB方法的VTI介质全波形反演优化
% 
% 说明：
%   1. 主要功能：
%      - 使用BB方法进行模型参数优化
%      - 自适应步长计算
%      - 非单调线搜索策略
%
% 类属性：
%   继承自BaseOptimizer的所有属性，额外包含：
%   - memory_length: 非单调线搜索的记忆长度
%   - initial_step: 初始步长值
%   - min_step: 最小允许步长
%   - max_step: 最大允许步长
%
% 核心算法流程：
% 1. 初始化阶段：
%    - 计算初始目标函数值和梯度
%    - 初始化步长和历史记录
%
% 2. 迭代优化阶段：
%    a) BB步长计算：
%       - 计算位置差sk和梯度差yk
%       - BB1型步长：tk = (sk'*yk)/(yk'*yk)
%       - BB2型步长：tk = (sk'*sk)/(sk'*yk)
%       - 步长约束在[min_step, max_step]范围内
%
%    b) 非单调线搜索：
%       - 使用最近m次迭代的最大函数值作为参考
%       - 实现非单调Armijo准则
%       - 动态步长调整
%
%    c) 模型更新：
%       - 使用BB步长更新模型参数
%       - 保存历史信息用于下次迭代
%
%    d) 收敛检查：
%       - 检查梯度范数
%       - 评估函数值相对变化
% -------------------------------------------------------------------------
% 输入参数：
%   params结构体必须包含：
%   - 继承自BaseOptimizer的所有参数
%   - memory_length：非单调线搜索记忆长度（默认5）
%   - initial_step：初始步长（默认1e-3）
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

classdef BBOptimizer < BaseOptimizer
    properties
        bb_params              % BB法参数
        misfit_history         % 目标函数值历史
    end
    
    methods
        function obj = BBOptimizer(params)
            % 构造函数
            obj = obj@BaseOptimizer(params);
            obj.bb_params = params.bb_params;
            obj.misfit_history = [];
        end
        
        function run(obj)
            fprintf('\n=== 开始BB法FWI迭代 ===\n');
            
            % 初始化
            [fk, gk] = obj.initialize_bb_method();
            all_misfits = zeros(obj.max_iterations + 1, 1);
            all_misfits(1) = fk;
            tk = obj.bb_params.initial_step;
            
            % 主迭代循环
            for iter = 1:obj.max_iterations
                % 保存当前状态
                gk_old_array = utils.model_utils.struct2array(gk);
                current_model = obj.get_current_model();
                
                % BB迭代
                [fk_new, gk_new, tk_new] = obj.bb_iteration(iter, gk, tk);
                all_misfits(iter + 1) = fk_new;
                
                % 检查收敛性
                if obj.check_bb_convergence(gk_new, fk_new, fk)
                    fprintf('\n=== BB法已收敛，停止迭代 ===\n');
                    break;
                end
                
                % 更新状态
                fk = fk_new;
                gk = gk_new;
                tk = tk_new;
            end
            
            % 绘制收敛曲线
            obj.plot_convergence_curve(all_misfits(1:iter+1));
        end
    end
    
    methods (Access = private)
        function [fk, gk] = initialize_bb_method(obj)
            % BB法初始化
            fk = obj.compute_misfit();
            gk = obj.compute_total_gradient();
        end
        
        function [fk_new, gk_new, tk_new] = bb_iteration(obj, iter, gk, tk)
            fprintf('\n=== BB法迭代 %d/%d ===\n', iter, obj.max_iterations);
            
            % 保存当前状态
            gk_old_array = utils.model_utils.struct2array(gk);
            current_model = obj.get_current_model();
            
            % 计算BB方向
            sk = obj.compute_BB_step(gk, tk);
            
            % 步长安全检查
            tk = obj.safeguard_step(tk, sk, gk);
            
            % 非单调线搜索
            [fk_new, ~] = obj.nonmonotone_linesearch(obj.compute_misfit(), gk, tk, obj.bb_params.memory_length);
            
            % 计算新梯度
            gk_new = obj.compute_total_gradient();
            
            % 更新BB步长（BB1型或BB2型）
            gk_new_array = utils.model_utils.struct2array(gk_new);
            sk_array = utils.model_utils.struct2array(sk);
            yk = gk_new_array - gk_old_array;
            
            % BB1型步长
            tk_new = abs(sk_array' * yk) / (yk' * yk);
            
            % 更新历史记录
            obj.update_misfit_history(fk_new);
            
            % 保存迭代结果
            obj.save_iteration_gradient(gk_new, iter);
            obj.save_iteration_misfit(struct('misfit', fk_new), iter);
            
            fprintf('目标函数值: %e\n', fk_new);
            fprintf('步长: %e\n', tk_new);
        end
        
        function converged = check_bb_convergence(obj, gk, fk_new, fk)
            % 检查BB法收敛性
            gk_array = utils.model_utils.struct2array(gk);
            grad_converged = norm(gk_array) < obj.tolerance;
            func_converged = abs(fk_new - fk)/fk < obj.tolerance;
            converged = grad_converged || func_converged;
        end
        
        function sk = compute_BB_step(obj, gk, tk)
            % 计算BB搜索方向
            sk = struct();
            sk.c11 = -tk * gk.c11;
            sk.c13 = -tk * gk.c13;
            sk.c33 = -tk * gk.c33;
            sk.c44 = -tk * gk.c44;
            sk.rho = -tk * gk.rho;
        end
        
        function tk = safeguard_step(obj, tk, sk, gk)
            % 步长安全检查
            max_step = obj.bb_params.max_step;
            min_step = obj.bb_params.min_step;
            
            % 基于梯度和搜索方向的安全检查
            grad_norm = norm(utils.model_utils.struct2array(gk));
            if grad_norm > 0
                sk_norm = norm(utils.model_utils.struct2array(sk));
                relative_step = sk_norm / grad_norm;
                
                % 调整步长确保相对步长在合理范围内
                if relative_step > max_step
                    tk = tk * (max_step / relative_step);
                elseif relative_step < min_step
                    tk = tk * (min_step / relative_step);
                end
            end
        end
        
        function [fk_new, model_new] = nonmonotone_linesearch(obj, fk, gk, tk, max_memory)
            % 非单调线搜索
            % 使用最近max_memory次迭代中的最大函数值作为参考
            
            % 计算搜索方向
            sk = obj.compute_BB_step(gk, tk);
            
            % 更新模型并计算新的函数值
            current_model = obj.get_current_model();
            obj.update_model_with_BB_step(sk);
            fk_new = obj.compute_misfit();
            
            % 如果新的函数值不满足非单调准则，回退步长
            if ~obj.check_nonmonotone_condition(fk_new, fk, max_memory)
                tk = tk * 0.5;
                obj.set_current_model(current_model);
                [fk_new, ~] = obj.nonmonotone_linesearch(fk, gk, tk, max_memory);
            end
        end
        
        function satisfied = check_nonmonotone_condition(obj, fk_new, fk, max_memory)
            % 检查非单调条件
            if isempty(obj.misfit_history)
                satisfied = fk_new <= fk;
            else
                max_previous = max(obj.misfit_history);
                satisfied = fk_new <= max_previous;
            end
        end
        
        function update_model_with_BB_step(obj, sk)
            % 使用BB步长更新模型
            syn_params = obj.gradient_solver.adjoint_solver.syn_params;
            syn_params.c11 = syn_params.c11 + sk.c11;
            syn_params.c13 = syn_params.c13 + sk.c13;
            syn_params.c33 = syn_params.c33 + sk.c33;
            syn_params.c44 = syn_params.c44 + sk.c44;
            syn_params.rho = syn_params.rho + sk.rho;
        end
        
        function update_misfit_history(obj, fk_new)
            % 更新目标函数值历史
            if isempty(obj.misfit_history)
                obj.misfit_history = fk_new;
            else
                obj.misfit_history = [obj.misfit_history(max(1,end-obj.bb_params.memory_length+1):end), fk_new];
            end
        end
    end
end 