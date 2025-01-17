%% VTI介质全波形反演梯度计算模块
% 功能：计算VTI介质中全波形反演的梯度
% 
% 说明：
%   1. 主要功能：
%      - 计算单炮梯度
%      - 基于速度场的梯度计算
%      - 基于位移场的梯度计算（可选）
%      - 弹性参数(c11,c13,c33,c44)和密度的梯度计算
%   2. 计算方法：
%      - 速度场方法：直接使用速度场计算应变率
%      - 位移场方法：通过时间积分获得位移场
%      - 空间导数计算支持二阶和四阶差分
%      - 时间导数采用中心差分（边界使用单侧差分）
%
% 类属性：
%   - adjoint_solver: 伴随波场求解器实例（包含正演求解器）
%   - gradient_output_dir: 梯度输出目录
%   - NSTEP: 时间步数
%
% 主要方法：
%   1. 基础方法：
%      - compute_single_shot_gradient: 计算单炮梯度
%      - compute_gradient: 计算空间导数
%      - save_gradient: 保存梯度结果
%   2. 速度场相关方法：
%      - correlate_wavefields: 基于速度场的波场互相关
%   3. 位移场相关方法：
%      - velocity_to_displacement: 速度场转位移场
%      - correlate_wavefields_displacement: 基于位移场的波场互相关
%
% 梯度计算公式：
%   速度场方法：
%   - c11梯度：-∂vx/∂x * ∂v†x/∂x
%   - c13梯度：-(∂v†x/∂x * ∂vy/∂y + ∂v†y/∂y * ∂vx/∂x)
%   - c33梯度：-∂vy/∂y * ∂v†y/∂y
%   - c44梯度：-(∂vx/∂y + ∂vy/∂x) * (∂v†x/∂y + ∂v†y/∂x)
%   - ρ梯度：-v†i * ∂²vi/∂t²
%
%   位移场方法：
%   类似公式，但使用位移场u替代速度场v
%
% 空间导数计算：
%   支持两种方法：
%   1. 二阶中心差分（使用MATLAB gradient函数）
%   2. 四阶中心差分（自定义实现）
%
% 时间积分方法：
%   位移场计算采用改进的梯形法则：
%   u(t+dt) = u(t) + (v(t) + v(t+dt))*dt/2
%
% 依赖项：
%   - VTI_Adjoint类
%   - utils.computeGradientField函数（二阶差分）
%   - utils.computeFourthOrderDiff函数（四阶差分）
%
% 注意事项：
%   - 需要正确初始化伴随求解器
%   - 可以选择速度场或位移场方法
%   - 可以选择二阶或四阶空间差分
%   - 注意内存管理（波场数据量较大）
%   - 时间导数计算中已处理边界条件
%
% 作者：StarrMoonn
% 日期：2025-01-16
%
classdef VTI_Gradient < handle
    properties
        adjoint_solver      % 伴随波场求解器实例
        gradient_output_dir % 梯度输出目录
        NSTEP              % 时间步数
    end
     
    methods 
        function obj = VTI_Gradient(params)
            % 创建伴随波场求解器实例
            obj.adjoint_solver = VTI_Adjoint(params);
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.gradient_output_dir = fullfile(current_dir, 'data', 'output', 'gradient');
            
            % 从合成数据参数中获取时间步数
            obj.NSTEP = obj.adjoint_solver.syn_params.NSTEP;  % 使用 syn_params
            
            if ~exist(obj.gradient_output_dir, 'dir')
                mkdir(obj.gradient_output_dir);
            end
        end
        
        % 计算单炮梯度
        function gradient = compute_single_shot_gradient(obj, ishot)
            fprintf('\n=== 开始计算第 %d 炮梯度 ===\n', ishot);
        
            % 计算伴随波场
            adjoint_wavefield = obj.adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
            
            % 直接使用已经计算好的正演波场
            forward_wavefield = obj.adjoint_solver.current_forward_wavefield;
            
            % 使用速度场计算梯度
            gradient = obj.correlate_wavefields(forward_wavefield, adjoint_wavefield);  %使用速度场计算梯度
            
            % 使用位移场计算梯度
            %gradient = obj.correlate_wavefields_displacement(forward_wavefield, adjoint_wavefield); 
            
            % 保存单炮梯度到磁盘
            obj.save_gradient(gradient, ishot);
                        
            % 输出计算完成信息
            fprintf('\n=== 炮号 %d 梯度计算完成 ===\n', ishot);
        end
    end
    
    %% 速度场梯度计算方法
    methods 
        function gradient = correlate_wavefields(obj, forward_wavefield, adjoint_wavefield)
            % 获取时间步长
            dt = obj.adjoint_solver.syn_params.DELTAT;
            
            % 初始化梯度
            [NX, NY, NT] = size(forward_wavefield.vx);
            gradient_c11 = zeros(NX, NY);
            gradient_c13 = zeros(NX, NY);
            gradient_c33 = zeros(NX, NY);
            gradient_c44 = zeros(NX, NY);
            gradient_rho = zeros(NX, NY);
            
            % 添加波场能量监控数组
            forward_energy = zeros(NT,1);
            adjoint_energy = zeros(NT,1);
            
            % 添加调试信息
            fprintf('开始波场互相关计算:\n');
            fprintf('时间步长 dt = %e\n', dt);
            fprintf('总时间步数: %d\n', NT);
            
            % 对每个时间步进行互相关
            for it = 1:NT
                % 从完整波场中读取当前时间步的波场
                fwd_vx = forward_wavefield.vx(:,:,it);
                fwd_vy = forward_wavefield.vy(:,:,it);
                adj_vx = adjoint_wavefield.vx(:,:,it);
                adj_vy = adjoint_wavefield.vy(:,:,it);
                
                % 计算波场能量
                forward_energy(it) = sum(sum(fwd_vx.^2 + fwd_vy.^2));
                adjoint_energy(it) = sum(sum(adj_vx.^2 + adj_vy.^2));
                
                % 计算应变率
                [dvx_dx, dvx_dy] = obj.compute_gradient(fwd_vx);
                [dvy_dx, dvy_dy] = obj.compute_gradient(fwd_vy);
                [dadj_vx_dx, dadj_vx_dy] = obj.compute_gradient(adj_vx);
                [dadj_vy_dx, dadj_vy_dy] = obj.compute_gradient(adj_vy);
                
                % 更新梯度
                gradient_c11 = gradient_c11 - dvx_dx .* dadj_vx_dx * dt;
                gradient_c13 = gradient_c13 - (dadj_vx_dx .* dvy_dy + dadj_vy_dy .* dvx_dx + ...
                                             dadj_vx_dx .* dvy_dy + dadj_vy_dy .* dvx_dx) * dt;
                gradient_c33 = gradient_c33 - dvy_dy .* dadj_vy_dy * dt;
                gradient_c44 = gradient_c44 - (dvx_dy + dvy_dx) .* (dadj_vx_dy + dadj_vy_dx) * dt;
                
                % 密度梯度计算
                if it == 1
                    % 第一个时间步：前向差分
                    dv_dt_x = (forward_wavefield.vx(:,:,it+1) - forward_wavefield.vx(:,:,it)) / dt;
                    dv_dt_y = (forward_wavefield.vy(:,:,it+1) - forward_wavefield.vy(:,:,it)) / dt;
                elseif it == NT
                    % 最后一个时间步：后向差分
                    dv_dt_x = (forward_wavefield.vx(:,:,it) - forward_wavefield.vx(:,:,it-1)) / dt;
                    dv_dt_y = (forward_wavefield.vy(:,:,it) - forward_wavefield.vy(:,:,it-1)) / dt;
                else
                    % 内部点：中心差分
                    dv_dt_x = (forward_wavefield.vx(:,:,it+1) - forward_wavefield.vx(:,:,it-1)) / (2*dt);
                    dv_dt_y = (forward_wavefield.vy(:,:,it+1) - forward_wavefield.vy(:,:,it-1)) / (2*dt);
                end
                
                gradient_rho = gradient_rho - (dadj_vx_dx .* dv_dt_x + dadj_vy_dy .* dv_dt_y) * dt;
                
                % 每100步输出一次信息
                if mod(it, 100) == 0
                    fprintf('时间步 %d:\n', it);
                    fprintf('正演波场能量: %e\n', forward_energy(it));
                    fprintf('伴随波场能量: %e\n', adjoint_energy(it));
                    fprintf('C11最大梯度: %e\n', max(abs(gradient_c11(:))));
                    fprintf('C13最大梯度: %e\n', max(abs(gradient_c13(:))));
                    fprintf('C33最大梯度: %e\n', max(abs(gradient_c33(:))));
                    fprintf('C44最大梯度: %e\n', max(abs(gradient_c44(:))));
                    fprintf('Rho最大梯度: %e\n', max(abs(gradient_rho(:))));
                end
            end
            
            % 绘制并保存波场能量曲线
            figure;
            plot(1:NT, forward_energy, 'b-', 1:NT, adjoint_energy, 'r--');
            legend('正演波场能量', '伴随波场能量');
            xlabel('时间步');
            ylabel('波场能量');
            title('波场能量随时间变化');
            savefig(fullfile(obj.gradient_output_dir, 'wavefield_energy.fig'));
            
            % 组合所有参数的梯度
            gradient = struct('c11', gradient_c11, ...
                             'c13', gradient_c13, ...
                             'c33', gradient_c33, ...
                             'c44', gradient_c44, ...
                             'rho', gradient_rho);
        end
        
        function [dx, dy] = compute_gradient(obj, field)
            % 获取网格间距
            deltax = obj.adjoint_solver.syn_params.DELTAX;
            deltay = obj.adjoint_solver.syn_params.DELTAY;
            
            % 四阶中心差分计算导数
            %[dx, dy] = utils.computeFourthOrderDiff(field, deltax, deltay);

            % 二阶中心差分计算导数
            [dx, dy] = utils.computeGradientField(field, deltax, deltay);
        end

        function save_gradient(obj, gradient, ishot)
            % 验证输入
            if ~isstruct(gradient) || ~all(isfield(gradient, {'c11','c13','c33','c44','rho'}))
                error('梯度必须包含所有必要字段：c11, c13, c33, c44, rho');
            end
            
            % 验证所有分量维度一致性
            [nx_c11, ny_c11] = size(gradient.c11);
            fields = {'c13', 'c33', 'c44', 'rho'};
            for field = fields
                [nx, ny] = size(gradient.(field{1}));
                if nx ~= nx_c11 || ny ~= ny_c11
                    error('梯度分量维度不一致：%s [%d, %d] vs c11 [%d, %d]', ...
                          field{1}, nx, ny, nx_c11, ny_c11);
                end
            end
            
            fprintf('\n=== 梯度维度检查 ===\n');
            fprintf('梯度维度: [%d, %d]\n', nx_c11, ny_c11);
            
            % 构造文件名
            filename = sprintf('gradient_shot_%d.mat', ishot);
            filepath = fullfile(obj.gradient_output_dir, filename);
            
            % 保存梯度
            save(filepath, 'gradient');
            fprintf('梯度已保存到: %s\n', filepath);
        end
    end
    
    %% 位移场梯度计算方法
    methods 
        function [ux, uy] = velocity_to_displacement(~, vx, vy, dt)
            % 速度场转位移场（使用梯形法则积分）
            % 
            % 输入参数:
            %   vx, vy: 速度场
            %   dt: 时间步长
            %
            % 输出参数:
            %   ux, uy: 位移场
            %
            % 说明：
            %   使用梯形法则进行时间积分：
            %   u(t+dt) = u(t) + (v(t) + v(t+dt))*dt/2
            
            % 初始化位移场
            [nx, ny, nt] = size(vx);
            ux = zeros(nx, ny, nt);
            uy = zeros(nx, ny, nt);
            
            % 时间积分（改进的梯形法则）
            for it = 1:nt-1
                ux(:,:,it+1) = ux(:,:,it) + (vx(:,:,it) + vx(:,:,it+1)) * dt/2;
                uy(:,:,it+1) = uy(:,:,it) + (vy(:,:,it) + vy(:,:,it+1)) * dt/2;
            end
        end

        function gradient = correlate_wavefields_displacement(obj, forward_wavefield, adjoint_wavefield)
            % 基于位移场的梯度计算
            dt = obj.adjoint_solver.syn_params.DELTAT;
            [NX, NY, NT] = size(forward_wavefield.vx);
            
            % 将速度场转换为位移场
            [fwd_ux, fwd_uy] = obj.velocity_to_displacement(forward_wavefield.vx, forward_wavefield.vy, dt);
            [adj_ux, adj_uy] = obj.velocity_to_displacement(adjoint_wavefield.vx, adjoint_wavefield.vy, dt);
            
            % 初始化梯度
            gradient_c11 = zeros(NX, NY);
            gradient_c13 = zeros(NX, NY);
            gradient_c33 = zeros(NX, NY);
            gradient_c44 = zeros(NX, NY);
            gradient_rho = zeros(NX, NY);
            
            fprintf('开始基于位移场的波场互相关计算:\n');
            fprintf('时间步长 dt = %e\n', dt);
            fprintf('总时间步数: %d\n', NT);
            
            % 对每个时间步进行互相关
            for it = 1:NT
                % 获取当前时间步的位移场
                ux = fwd_ux(:,:,it);
                uy = fwd_uy(:,:,it);
                adj_ux_t = adj_ux(:,:,it);
                adj_uy_t = adj_uy(:,:,it);
                
                % 计算位移场的空间导数
                [dux_dx, dux_dy] = obj.compute_gradient(ux);
                [duy_dx, duy_dy] = obj.compute_gradient(uy);
                [dadj_ux_dx, dadj_ux_dy] = obj.compute_gradient(adj_ux_t);
                [dadj_uy_dx, dadj_uy_dy] = obj.compute_gradient(adj_uy_t);
                
                % 更新弹性参数梯度
                gradient_c11 = gradient_c11 - dux_dx .* dadj_ux_dx * dt;
                gradient_c13 = gradient_c13 - (dadj_ux_dx .* duy_dy + dadj_uy_dy .* dux_dx) * dt;
                gradient_c33 = gradient_c33 - duy_dy .* dadj_uy_dy * dt;
                gradient_c44 = gradient_c44 - (dux_dy + duy_dx) .* (dadj_ux_dy + dadj_uy_dx) * dt;
                
                % 密度梯度计算 (使用速度场的时间导数)
                % 内部点使用中心差分，边界点使用单侧差分
                if it == 1
                    % 第一个时间步：前向差分
                    dv_dt_x = (forward_wavefield.vx(:,:,it+1) - forward_wavefield.vx(:,:,it)) / dt;
                    dv_dt_y = (forward_wavefield.vy(:,:,it+1) - forward_wavefield.vy(:,:,it)) / dt;
                elseif it == NT
                    % 最后一个时间步：后向差分
                    dv_dt_x = (forward_wavefield.vx(:,:,it) - forward_wavefield.vx(:,:,it-1)) / dt;
                    dv_dt_y = (forward_wavefield.vy(:,:,it) - forward_wavefield.vy(:,:,it-1)) / dt;
                else
                    % 内部点：中心差分
                    dv_dt_x = (forward_wavefield.vx(:,:,it+1) - forward_wavefield.vx(:,:,it-1)) / (2*dt);
                    dv_dt_y = (forward_wavefield.vy(:,:,it+1) - forward_wavefield.vy(:,:,it-1)) / (2*dt);
                end

                gradient_rho = gradient_rho - (dadj_ux_dx .* dv_dt_x + dadj_uy_dy .* dv_dt_y) * dt;
                
                % 每100步输出一次最大梯度值
                if mod(it, 100) == 0
                    fprintf('时间步 %d:\n', it);
                    fprintf('C11最大梯度: %e\n', max(abs(gradient_c11(:))));
                    fprintf('C13最大梯度: %e\n', max(abs(gradient_c13(:))));
                    fprintf('C33最大梯度: %e\n', max(abs(gradient_c33(:))));
                    fprintf('C44最大梯度: %e\n', max(abs(gradient_c44(:))));
                    fprintf('Rho最大梯度: %e\n', max(abs(gradient_rho(:))));
                end
            end
            
            % 组合最终梯度
            gradient = struct('c11', gradient_c11, ...
                             'c13', gradient_c13, ...
                             'c33', gradient_c33, ...
                             'c44', gradient_c44, ...
                             'rho', gradient_rho);
        end
    end
end 
