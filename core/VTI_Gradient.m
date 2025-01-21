%% VTI介质全波形反演梯度计算模块
% 功能：计算VTI介质中全波形反演的梯度
% 
% 说明：
%   1. 主要功能：
%      - 计算单炮梯度
%      - 弹性参数(c11,c13,c33,c44)和密度的梯度计算
%   2. 计算方法：
%      - 基于速度场的梯度计算（主要方法）
%      - 基于位移场的梯度计算（可选验证方法）
%      - 二阶/四阶中心差分计算空间导数
%      - 时间导数采用中心差分（边界使用单侧差分）
%
% 类属性：
%   - adjoint_solver: 伴随波场求解器实例
%   - gradient_output_dir: 梯度输出目录
%   - NSTEP: 时间步数
%
% 主要方法：
%   1. 构造函数：
%      - VTI_Gradient(params): 初始化梯度计算器
%   2. 梯度计算：
%      - compute_single_shot_gradient: 计算单炮梯度
%      - compute_vti_gradient: 基于速度场计算梯度（调用MEX）
%      - correlate_wavefields_displacement: 基于位移场计算梯度（验证用）
%   3. 辅助功能：
%      - save_gradient: 保存梯度结果
%      - velocity_to_displacement: 速度场转位移场
%
% 梯度计算公式：
%   速度场方法：
%   - c11梯度：-∂vx/∂x * ∂v†x/∂x
%   - c13梯度：-(∂v†x/∂x * ∂vy/∂y + ∂v†y/∂y * ∂vx/∂x)
%   - c33梯度：-∂vy/∂y * ∂v†y/∂y
%   - c44梯度：-(∂vx/∂y + ∂vy/∂x) * (∂v†x/∂y + ∂v†y/∂x)
%   - ρ梯度：-v†i * ∂²vi/∂t²
%
% 优化说明：
%   1. MEX加速：
%      - 使用C++编写核心计算代码
%      - OpenMP并行优化
%      - SIMD向量化优化
%   2. 内存管理：
%      - 自动清理临时变量
%      - 分步释放大型波场数据
%
% 依赖项：
%   - VTI_Adjoint类：提供伴随波场计算
%   - compute_vti_gradient_omp.cpp：C++实现的梯度计算
%
% 输出说明：
%   - 梯度结构体包含：c11, c13, c33, c44, rho
%   - 自动保存到指定输出目录
%   - 支持单炮梯度的独立保存
%
% 注意事项：
%   1. 内存管理：
%      - 波场数据量较大，注意及时清理
%      - 使用try-finally确保资源释放
%   2. 性能优化：
%      - 主要计算已移至C++实现
%      - 保留MATLAB实现用于验证
%
% 作者：StarrMoonn
% 日期：2025-01-21
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

            try
                % 计算伴随波场
                adjoint_wavefield = obj.adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
                
                % 获取正演波场
                forward_wavefield = obj.adjoint_solver.current_forward_wavefield;
                
                % 使用速度场计算梯度
                gradient = obj.compute_vti_gradient(forward_wavefield, adjoint_wavefield);
                
                % 保存单炮梯度到磁盘
                obj.save_gradient(gradient, ishot);
                
            catch ME
                fprintf('计算梯度时发生错误: %s\n', ME.message);
                rethrow(ME);
                
            finally
                % 清理内存
                % 1. 清除波场和梯度相关变量
                if exist('adjoint_wavefield', 'var')
                    clear adjoint_wavefield;
                end
                
                if exist('gradient', 'var')
                    clear gradient gradient_c11 gradient_c13 gradient_c33 gradient_c44 gradient_rho;
                end
                
                % 2. 清除adjoint_solver中的临时波场
                obj.adjoint_solver.current_forward_wavefield = [];
                
                % 3. 强制垃圾回收（可选）
                %gc = java.lang.System.gc();
            end
            
            fprintf('\n=== 炮号 %d 梯度计算完成 ===\n', ishot);
        end
    end
    
    %% 速度场梯度计算函数
    methods 
        function gradient = compute_vti_gradient(obj, forward_wavefield, adjoint_wavefield)
            % 获取时间步长和其他参数
            dt = obj.adjoint_solver.syn_params.DELTAT;
            params = obj.adjoint_solver.syn_params;
            
            % 添加调试信息
            [NX, NY, NT] = size(forward_wavefield.vx);
            fprintf('开始计算VTI介质梯度:\n');
            fprintf('时间步长 dt = %e\n', dt);
            fprintf('总时间步数: %d\n', NT);
            
            % 调用 MEX 函数计算梯度
            try
                gradient = compute_vti_gradient_omp(forward_wavefield, adjoint_wavefield, dt, params);
            catch ME
                fprintf('MEX函数调用失败，错误信息：%s\n', ME.message);
                rethrow(ME);
            end
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

  
%{
       % 不需要了，已经打包到c++源码中，使用二阶中心差分
        function [dx, dy] = compute_gradient(obj, field)
            % 获取网格间距
            deltax = obj.adjoint_solver.syn_params.DELTAX;
            deltay = obj.adjoint_solver.syn_params.DELTAY;
            
            % 四阶中心差分计算导数
            %[dx, dy] = utils.computeFourthOrderDiff(field, deltax, deltay);

            % 二阶中心差分计算导数
            [dx, dy] = utils.computeGradientField(field, deltax, deltay);
        end 
%}

    end
    
    %% 位移场梯度计算函数，验证效果一般。
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