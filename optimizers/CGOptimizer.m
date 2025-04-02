classdef CGOptimizer < BaseOptimizer
    properties (Access = private)
        p0              % 上一次的归一化梯度方向
        d0              % 上一次的搜索方向
        a2              % 上一次的步长
        objold          % 上一次的目标函数值
        objval          % 存储所有迭代的目标函数值
    end
    
    methods
        function obj = CGOptimizer(params)
            obj@BaseOptimizer(params);
        end
        
        function run(obj)
            %%% 初始化迭代参数 %%%
            k = 1;                    % 当前迭代次数
            kmax = 500;               % 最大迭代次数
            objval = zeros(kmax,1);   % 存储每次迭代的目标函数值

            % 获取初始梯度和目标函数值
            objval0 = obj.get_current_total_misfit();
            grad_stk1 = obj.compute_total_gradient();
            
            % 显示当前速度模型和梯度
            figure(1); imagesc(grad_stk1.c11); pause(0.001);        % 显示c11分量
            k
            disp('%%%%%%%%%%%%%%%%%%%%iteration time%%%%%%%%%%%%%%%%%%%%%%')

            %%% 第一次迭代的梯度计算 %%%
            objval(k+1) = objval0;   % 记录初始目标函数值

            % 计算归一化的梯度方向
            total_grad_norm = norm([grad_stk1.c11(:); grad_stk1.c13(:); grad_stk1.c33(:); 
                                   grad_stk1.c44(:); grad_stk1.rho(:)]);  % 计算总梯度的范数

            % 归一化并放大梯度
            p = struct();
            p.c11 = 100 * grad_stk1.c11 / total_grad_norm;
            p.c13 = 100 * grad_stk1.c13 / total_grad_norm;
            p.c33 = 100 * grad_stk1.c33 / total_grad_norm;
            p.c44 = 100 * grad_stk1.c44 / total_grad_norm;
            p.rho = 100 * grad_stk1.rho / total_grad_norm;

            % 最速下降方向
            d1 = struct();
            d1.c11 = -2 * p.c11;
            d1.c13 = -2 * p.c13;
            d1.c33 = -2 * p.c33;
            d1.c44 = -2 * p.c44;
            d1.rho = -2 * p.rho;
            
            % 步长控制
            maxd = max(max(abs(d1.c11(:))), max(abs(d1.c13(:))), 
                       max(abs(d1.c33(:))), max(abs(d1.c44(:))), 
                       max(abs(d1.rho(:))));
            dec = 0.5;                           % 步长衰减因子
            ks = 1;                              % 线搜索计数器

            % 初始步长
            a = 1;
            % 控制最大更新步长不超过30
            while maxd > 30
                a = dec*a;                       % 如果更新太大，减小步长
                maxd = max(max(abs(a*d1.c11(:))), max(abs(a*d1.c13(:))), 
                           max(abs(a*d1.c33(:))), max(abs(a*d1.c44(:))), 
                           max(abs(a*d1.rho(:))));
            end
            a                                    % 显示最终步长
            maxd                                 % 显示最大更新量

            %%% 线搜索找最优步长 %%%
            while (ks < 10)                      % 最多尝试10次
                % 更新速度模型
                current_model = obj.get_current_model();
                new_model = current_model + a*d1.c11;               % 试探性更新
                obj.set_current_model(new_model);
                
                % 计算新模型的目标函数值和梯度
                objval1 = obj.get_current_total_misfit();
                grad_stk2 = obj.compute_total_gradient();
                
                % 显示当前搜索信息
                ks                              % 显示当前搜索次数
                objval0                         % 显示旧目标函数值
                objval1                         % 显示新目标函数值
                
                % 判断是否接受更新
                if objval1 < objval0            % 如果目标函数值减小
                    v0 = v0s1;                  % 接受新模型
                    objold = objval0;           % 保存旧的目标函数值
                    objval0 = objval1;          % 更新目标函数值
                    grad_stk1 = grad_stk2;      % 更新梯度
                    p0 = p;                     % 保存旧的梯度方向
                    d0 = a*d1.c11;              % 保存搜索方向
                    break;
                else
                    a = dec*a;                  % 如果目标函数值增加，减小步长
                    a2 = a;                     % 保存当前步长供下次迭代使用
                    ks = ks + 1;                % 增加搜索次数
                end
            end

            %%% 保存结果 %%%
            % 保存当前迭代的速度模型和梯度
            save(['k=' num2str(k) '_v0.mat'],'v0');
            save(['k=' num2str(k) '_grad_stk1.mat'],'grad_stk1');

            % 水层梯度置零
            grad_stk1(1:water,:) = 0;
            % grad_stk1=grad_stk1.*water;  % 另一种水层处理方式（已注释）

            % 更新迭代次数
            k = k + 1;

            %%% 共轭梯度法迭代（k>1） %%%
            while(k < kmax)
                % 显示当前迭代次数
                k
                % 显示当前速度模型和梯度
                figure(1); imagesc(grad_stk1.c11); pause(0.0001);
                figure(2); imagesc(grad_stk1.c33); pause(0.0001);
                disp('%%%%%%%%%%%%%%%%%%%%iteration time%%%%%%%%%%%%%%%%%%%%%%')
                
                % 记录目标函数值
                objval(k+1) = objval0;
                
                %%% 计算共轭梯度方向 %%%
                p = 100*grad_stk1/norm(grad_stk1);    % 归一化当前梯度
                % 计算共轭因子beta（Fletcher-Reeves公式）
                b = sqrt(sum(sum((p'*p).^2))/sum(sum((p0'*p0).^2)));
                % 计算共轭方向：当前负梯度方向 + beta*前一次方向
                d1 = struct();
                d1.c11 = -p.c11 + b*p0.c11;

                %%% 步长控制 %%%
                maxd = max([max(abs(d1.c11(:))), max(abs(d1.c13(:))), 
                           max(abs(d1.c33(:))), max(abs(d1.c44(:))), 
                           max(abs(d1.rho(:)))]);             % 计算最大更新量
                dec = 0.5;                            % 步长衰减因子

                % 初始步长设置和控制
                a = 1;
                while maxd > 30                       % 控制最大更新不超过30
                    a = dec*a;                        % 减小步长
                    maxd = max([max(abs(a*d1.c11(:))), max(abs(a*d1.c13(:))), 
                               max(abs(a*d1.c33(:))), max(abs(a*d1.c44(:))), 
                               max(abs(a*d1.rho(:)))]);       % 重新计算最大更新量
                end
                
                % 如果上一次线搜索失败，使用上一次的步长
                if ks > 1
                   a = a2; 
                end
                % 显示当前步长和最大更新量
                a
                maxd
                
                %%% 线搜索找最优步长 %%%
                ks = 1;                               % 线搜索计数器
                while (ks < 10)                       % 最多尝试10次
                    % 更新速度模型
                    v0s1 = obj.get_current_model() + a*d1.c11;                % 试探性更新
                    v0s1 = obj.control_v(v0s1, obj.water, obj.vmin, obj.vmax);  % 应用速度约束

                    % 计算新模型的目标函数值和梯度
                    objval1 = obj.get_current_total_misfit();
                    grad_stk2 = obj.compute_total_gradient();
                    % 显示当前搜索信息
                    ks
                    objval0
                    objval1
                    
                    % 判断是否接受更新
                    if objval1 < objval0              % 如果目标函数值减小
                        v0 = v0s1;                    % 接受新模型
                        objold = objval0;             % 保存旧的目标函数值
                        objval0 = objval1;            % 更新目标函数值
                        grad_stk1 = grad_stk2;        % 更新梯度
                        p0 = p;                       % 保存旧的梯度方向
                        d0 = a*d1.c11;                % 保存搜索方向
                        break;
                    else
                        a = dec*a;                    % 如果目标函数值增加，减小步长
                        a2 = a;                       % 保存当前步长
                        ks = ks + 1;                  % 增加搜索次数
                    end
                end

                % 水层梯度置零
                grad_stk1(1:obj.water,:) = 0;
                % grad_stk1=grad_stk1.*obj.water;        % 另一种水层处理方式（已注释）
                
                % 每5次迭代保存一次结果
                if mod(k,5) == 0
                    save(['k=' num2str(k) '_v0.mat'],'v0');
                    save(['k=' num2str(k) '_grad_stk1.mat'],'grad_stk1');
                end
                
                % 保存所有目标函数值
                save objval objval

                % 如果线搜索失败太多次，终止迭代
                if (ks == 20)  %if (ks==10) 
                    break;
                end
                
                % 更新迭代次数
                k = k + 1;
            end
        end
    end
end
  