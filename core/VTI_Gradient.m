%% VTI介质全波形反演梯度计算模块
% 功能：计算VTI介质中全波形反演的梯度
% 
% 说明：
%   1. 主要功能：
%      - 计算单炮梯度
%      - 波场互相关计算
%      - 弹性参数(c11,c13,c33,c44)和密度的梯度计算
%   2. 计算内容包括：
%      - 正演波场和伴随波场的获取
%      - 应变率计算（使用中心差分）
%      - 波场互相关
%      - 各向异性参数梯度组装
%
% 类属性：
%   - adjoint_solver: 伴随波场求解器实例（包含正演求解器）
%   - gradient_output_dir: 梯度输出目录
%   - NSTEP: 时间步数
%
% 主要方法：
%   - compute_single_shot_gradient: 计算单炮梯度
%   - correlate_wavefields: 波场互相关计算
%   - compute_gradient: 计算空间导数（应变率）
%   - save_gradient: 保存梯度结果
%
% 梯度计算公式：
%   - c11梯度：-∂vx/∂x * ∂v†x/∂x
%   - c13梯度：-(∂v†x/∂x * ∂vy/∂y + ∂v†y/∂y * ∂vx/∂x)
%   - c33梯度：-∂vy/∂y * ∂v†y/∂y
%   - c44梯度：-(∂vx/∂y + ∂vy/∂x) * (∂v†x/∂y + ∂v†y/∂x)
%   - ρ梯度：-v†i * ∂²vi/∂t²
%   其中：v表示速度场，v†表示伴随场，∂表示偏导数
%
% 空间导数计算：
%   使用中心差分格式：
%   ∂v/∂x = [v(i+1,j) - v(i-1,j)]/(2*dx)
%   ∂v/∂y = [v(i,j+1) - v(i,j-1)]/(2*dy)
%   其中dx,dy为实际物理网格间距
%
% 依赖项：
%   - VTI_Adjoint类
%   - MATLAB gradient函数
%
% 注意事项：
%   - 需要正确初始化伴随求解器
%   - 确保波场维度匹配
%   - 注意内存管理（波场数据量较大）
%   - 时间导数计算中需要考虑边界条件
%
% 作者：StarrMoonn
% 日期：2025-01-09
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
            tic;  % 开始计时

            % 计算伴随波场
            adjoint_wavefield = obj.adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
            
            % 直接使用已经计算好的正演波场
            forward_wavefield = obj.adjoint_solver.current_forward_wavefield;
            
            % 计算互相关得到梯度
            gradient = obj.correlate_wavefields(forward_wavefield, adjoint_wavefield);

            % 保存单炮梯度到磁盘
            obj.save_gradient(gradient, ishot);
                        
            % 输出计算完成信息
            fprintf('\n=== 炮号 %d 梯度计算完成 ===\n', ishot);
            
            computation_time = toc;
            fprintf('第 %d 炮梯度计算耗时: %.2f 秒\n', ishot, computation_time);
        end
        
        % 波场互相关
        function gradient = correlate_wavefields(obj, forward_wavefield, adjoint_wavefield)
            % 获取时间步长
            dt = obj.adjoint_solver.syn_params.DELTAT;
            
            % 初始化梯度
            [NX, NY, ~] = size(forward_wavefield.vx);
            gradient_c11 = zeros(NX, NY);
            gradient_c13 = zeros(NX, NY);
            gradient_c33 = zeros(NX, NY);
            gradient_c44 = zeros(NX, NY);
            gradient_rho = zeros(NX, NY);
            
            % 添加调试信息
            fprintf('开始波场互相关计算:\n');
            fprintf('时间步长 dt = %e\n', dt);
            fprintf('总时间步数: %d\n', size(forward_wavefield.vx, 3));
            
            % 对每个时间步进行互相关
            for it = 1:size(forward_wavefield.vx, 3)
                % 从完整波场中读取当前时间步的波场
                fwd_vx = forward_wavefield.vx(:,:,it);
                fwd_vy = forward_wavefield.vy(:,:,it);
                adj_vx = adjoint_wavefield.vx(:,:,it);
                adj_vy = adjoint_wavefield.vy(:,:,it);
                
                % 计算应变率
                [dvx_dx, dvx_dy] = obj.compute_gradient(fwd_vx);
                [dvy_dx, dvy_dy] = obj.compute_gradient(fwd_vy);
                [dadj_vx_dx, dadj_vx_dy] = obj.compute_gradient(adj_vx);
                [dadj_vy_dx, dadj_vy_dy] = obj.compute_gradient(adj_vy);
                
                % 更新梯度（添加dt）
                gradient_c11 = gradient_c11 - dvx_dx .* dadj_vx_dx * dt;
                gradient_c13 = gradient_c13 - (dadj_vx_dx .* dvy_dy + dadj_vy_dy .* dvx_dx + ...
                                             dadj_vx_dx .* dvy_dy + dadj_vy_dy .* dvx_dx) * dt;
                gradient_c33 = gradient_c33 - dvy_dy .* dadj_vy_dy * dt;
                gradient_c44 = gradient_c44 - (dvx_dy + dvy_dx) .* (dadj_vx_dy + dadj_vy_dx) * dt;
                
                % 密度梯度计算
                if it > 1 && it < size(forward_wavefield.vx, 3)
                    d2_vx_dt2 = (forward_wavefield.vx(:,:,it+1) - ...
                                2*forward_wavefield.vx(:,:,it) + ...
                                forward_wavefield.vx(:,:,it-1)) / (dt^2);
                    d2_vy_dt2 = (forward_wavefield.vy(:,:,it+1) - ...
                                2*forward_wavefield.vy(:,:,it) + ...
                                forward_wavefield.vy(:,:,it-1)) / (dt^2);
                    gradient_rho = gradient_rho - (dadj_vx_dx .* d2_vx_dt2 + ...
                                                 dadj_vy_dy .* d2_vy_dt2) * dt;
                end
                
                % 每100步输出一次最大梯度值（用于调试）
                if mod(it, 100) == 0
                    fprintf('时间步 %d:\n', it);
                    fprintf('C11最大梯度: %e\n', max(abs(gradient_c11(:))));
                    fprintf('C13最大梯度: %e\n', max(abs(gradient_c13(:))));
                    fprintf('C33最大梯度: %e\n', max(abs(gradient_c33(:))));
                    fprintf('C44最大梯度: %e\n', max(abs(gradient_c44(:))));
                    fprintf('密度最大梯度: %e\n', max(abs(gradient_rho(:))));
                end
            end
            
            % 组合所有参数的梯度
            gradient = struct('c11', gradient_c11, ...
                             'c13', gradient_c13, ...
                             'c33', gradient_c33, ...
                             'c44', gradient_c44, ...
                             'rho', gradient_rho);
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
        
        function [dx, dy] = compute_gradient(obj, field)
            % 获取实际的网格间距（从参数中）
            deltax = obj.adjoint_solver.syn_params.DELTAX;  % 单位：m
            deltay = obj.adjoint_solver.syn_params.DELTAY;  % 单位：m
            
            % 计算导数并除以实际间距
            [dy, dx] = gradient(field);
            dx = dx / deltax;  % 得到正确的物理单位
            dy = dy / deltay;
        end
    end
end 