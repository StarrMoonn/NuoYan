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
        syn_wavefield_map_vx  % 正演波场vx分量内存映射
        syn_wavefield_map_vy  % 正演波场vy分量内存映射
    end
     
    methods
        function obj = VTI_Gradient(params)
            % 创建伴随波场求解器实例（包含了正演波场求解器）
            obj.adjoint_solver = VTI_Adjoint(params);
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.gradient_output_dir = fullfile(current_dir, 'data', 'output', 'gradient');
            
            % 从合成数据参数中获取时间步数
            obj.NSTEP = obj.adjoint_solver.syn_params.NSTEP;
            
            if ~exist(obj.gradient_output_dir, 'dir')
                mkdir(obj.gradient_output_dir);
            end
            
            % 只获取正演波场的内存映射引用
            obj.syn_wavefield_map_vx = obj.adjoint_solver.syn_wavefield_map_vx;
            obj.syn_wavefield_map_vy = obj.adjoint_solver.syn_wavefield_map_vy;
        end
        
        % 计算单炮梯度
        function gradient = compute_single_shot_gradient(obj, ishot)
            fprintf('\n=== 开始计算第 %d 炮梯度 ===\n', ishot);
            tic;  % 开始计时
            
            % 计算伴随波场（现在直接返回波场数据）
            adjoint_wavefield = obj.adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
            
            % 使用返回的伴随波场数据进行互相关
            gradient = obj.correlate_wavefields_serial(adjoint_wavefield);
            
            % 保存单炮梯度到磁盘
            obj.save_gradient(gradient, ishot);
            
            % 输出计算完成信息
            fprintf('\n=== 炮号 %d 梯度计算完成 ===\n', ishot);
            
            computation_time = toc;
            fprintf('第 %d 炮梯度计算耗时: %.2f 秒\n', ishot, computation_time);
        end
        
        function gradient = correlate_wavefields(obj, adjoint_wavefield)
            % 获取时间步长和其他参数
            dt = obj.adjoint_solver.syn_params.DELTAT;
            
            % 获取维度
            NX = obj.adjoint_solver.syn_params.NX;
            NY = obj.adjoint_solver.syn_params.NY;
            
            % 初始化梯度
            gradient_c11 = zeros(NX, NY);
            gradient_c13 = zeros(NX, NY);
            gradient_c33 = zeros(NX, NY);
            gradient_c44 = zeros(NX, NY);
            gradient_rho = zeros(NX, NY);
            
            % 设置分块大小
            block_size = 100;
            num_blocks = ceil(obj.NSTEP / block_size);
            
            fprintf('开始波场互相关计算:\n');
            fprintf('时间步长 dt = %e\n', dt);
            fprintf('总时间步数: %d, 分块数: %d\n', obj.NSTEP, num_blocks);
            
            % 对每个分块并行处理
            parfor iblock = 1:num_blocks
                % 计算当前块的时间步范围
                t_start = (iblock-1) * block_size + 1;
                t_end = min(iblock * block_size, obj.NSTEP);
                
                % 为当前块预分配内存
                block_data_vx = zeros(NX, NY, t_end-t_start+1);
                block_data_vy = zeros(NX, NY, t_end-t_start+1);
                
                % 读取当前块的波场数据
                for it = 1:t_end-t_start+1
                    block_data_vx(:,:,it) = obj.syn_wavefield_map_vx.Data(1).data(:,:,t_start+it-1);
                    block_data_vy(:,:,it) = obj.syn_wavefield_map_vy.Data(1).data(:,:,t_start+it-1);
                end
                
                % 初始化当前块的梯度
                block_grad_c11 = zeros(nx, ny);
                block_grad_c13 = zeros(nx, ny);
                block_grad_c33 = zeros(nx, ny);
                block_grad_c44 = zeros(nx, ny);
                block_grad_rho = zeros(nx, ny);
                
                % 处理当前时间块中的每个时间步
                for it = 1:t_end-t_start+1
                    % 使用预读取的波场数据
                    fwd_vx = block_data_vx(:,:,it);
                    fwd_vy = block_data_vy(:,:,it);
                    
                    % 使用对应的伴随波场数据
                    adj_vx = adjoint_wavefield.vx(:,:,t_start+it-1);
                    adj_vy = adjoint_wavefield.vy(:,:,t_start+it-1);
                    
                    % 计算应变率
                    [dvx_dx, dvx_dy] = obj.compute_gradient(fwd_vx);
                    [dvy_dx, dvy_dy] = obj.compute_gradient(fwd_vy);
                    [dadj_vx_dx, dadj_vx_dy] = obj.compute_gradient(adj_vx);
                    [dadj_vy_dx, dadj_vy_dy] = obj.compute_gradient(adj_vy);
                    
                    % 更新块梯度
                    block_grad_c11 = block_grad_c11 - dvx_dx .* dadj_vx_dx * dt;
                    block_grad_c13 = block_grad_c13 - (dadj_vx_dx .* dvy_dy + dadj_vy_dy .* dvx_dx) * dt;
                    block_grad_c33 = block_grad_c33 - dvy_dy .* dadj_vy_dy * dt;
                    block_grad_c44 = block_grad_c44 - (dvx_dy + dvy_dx) .* (dadj_vx_dy + dadj_vy_dx) * dt;
                    
                    % 密度梯度计算
                    if it > 1 && it < obj.NSTEP
                        % 正确访问内存映射的前后时间步波场
                        fwd_vx_prev = obj.syn_wavefield_map_vx.Data.data(:,:,it-1);
                        fwd_vx_next = obj.syn_wavefield_map_vx.Data.data(:,:,it+1);
                        fwd_vy_prev = obj.syn_wavefield_map_vy.Data.data(:,:,it-1);
                        fwd_vy_next = obj.syn_wavefield_map_vy.Data.data(:,:,it+1);
                        
                        % 计算时间二阶导数
                        d2_vx_dt2 = (fwd_vx_next - 2*fwd_vx + fwd_vx_prev) / (dt^2);
                        d2_vy_dt2 = (fwd_vy_next - 2*fwd_vy + fwd_vy_prev) / (dt^2);
                        
                        block_grad_rho = block_grad_rho - (dadj_vx_dx .* d2_vx_dt2 + ...
                            dadj_vy_dy .* d2_vy_dt2) * dt;
                    end
                end
                
                % 返回当前块的梯度结果
                block_gradients(iblock) = struct('c11', block_grad_c11, ...
                    'c13', block_grad_c13, ...
                    'c33', block_grad_c33, ...
                    'c44', block_grad_c44, ...
                    'rho', block_grad_rho);
            end
            
            % 合并所有块的梯度
            for iblock = 1:num_blocks
                gradient_c11 = gradient_c11 + block_gradients(iblock).c11;
                gradient_c13 = gradient_c13 + block_gradients(iblock).c13;
                gradient_c33 = gradient_c33 + block_gradients(iblock).c33;
                gradient_c44 = gradient_c44 + block_gradients(iblock).c44;
                gradient_rho = gradient_rho + block_gradients(iblock).rho;
            end
            
            % 组合最终梯度
            gradient = struct('c11', gradient_c11, ...
                'c13', gradient_c13, ...
                'c33', gradient_c33, ...
                'c44', gradient_c44, ...
                'rho', gradient_rho);
                
            % 输出最终梯度统计
            fprintf('\n=== 最终梯度统计 ===\n');
            fprintf('C11最大梯度: %e\n', max(abs(gradient_c11(:))));
            fprintf('C13最大梯度: %e\n', max(abs(gradient_c13(:))));
            fprintf('C33最大梯度: %e\n', max(abs(gradient_c33(:))));
            fprintf('C44最大梯度: %e\n', max(abs(gradient_c44(:))));
            fprintf('密度最大梯度: %e\n', max(abs(gradient_rho(:))));
        end
        
        function gradient = correlate_wavefields_serial(obj, adjoint_wavefield)
            % 获取时间步长和其他参数
            dt = obj.adjoint_solver.syn_params.DELTAT;
            
            % 获取维度
            NX = obj.adjoint_solver.syn_params.NX;
            NY = obj.adjoint_solver.syn_params.NY;
            
            % 初始化梯度
            gradient_c11 = zeros(NX, NY);
            gradient_c13 = zeros(NX, NY);
            gradient_c33 = zeros(NX, NY);
            gradient_c44 = zeros(NX, NY);
            gradient_rho = zeros(NX, NY);
            
            fprintf('开始串行波场互相关计算:\n');
            fprintf('时间步长 dt = %e\n', dt);
            fprintf('总时间步数: %d\n', obj.NSTEP);
            
            % 直接遍历所有时间步
            for it = 1:obj.NSTEP
                % 读取正演波场
                fwd_vx = obj.syn_wavefield_map_vx.Data(1).data(:,:,it);
                fwd_vy = obj.syn_wavefield_map_vy.Data(1).data(:,:,it);
                
                % 读取伴随波场
                adj_vx = adjoint_wavefield.vx(:,:,it);
                adj_vy = adjoint_wavefield.vy(:,:,it);
                
                % 计算应变率
                [dvx_dx, dvx_dy] = obj.compute_gradient(fwd_vx);
                [dvy_dx, dvy_dy] = obj.compute_gradient(fwd_vy);
                [dadj_vx_dx, dadj_vx_dy] = obj.compute_gradient(adj_vx);
                [dadj_vy_dx, dadj_vy_dy] = obj.compute_gradient(adj_vy);
                
                % 更新梯度
                gradient_c11 = gradient_c11 - dvx_dx .* dadj_vx_dx * dt;
                gradient_c13 = gradient_c13 - (dadj_vx_dx .* dvy_dy + dadj_vy_dy .* dvx_dx) * dt;
                gradient_c33 = gradient_c33 - dvy_dy .* dadj_vy_dy * dt;
                gradient_c44 = gradient_c44 - (dvx_dy + dvy_dx) .* (dadj_vx_dy + dadj_vy_dx) * dt;
                
                % 密度梯度计算
                if it > 1 && it < obj.NSTEP
                    % 读取前后时间步的波场
                    fwd_vx_prev = obj.syn_wavefield_map_vx.Data(1).data(:,:,it-1);
                    fwd_vx_next = obj.syn_wavefield_map_vx.Data(1).data(:,:,it+1);
                    fwd_vy_prev = obj.syn_wavefield_map_vy.Data(1).data(:,:,it-1);
                    fwd_vy_next = obj.syn_wavefield_map_vy.Data(1).data(:,:,it+1);
                    
                    % 计算时间二阶导数
                    d2_vx_dt2 = (fwd_vx_next - 2*fwd_vx + fwd_vx_prev) / (dt^2);
                    d2_vy_dt2 = (fwd_vy_next - 2*fwd_vy + fwd_vy_prev) / (dt^2);
                    
                    gradient_rho = gradient_rho - (dadj_vx_dx .* d2_vx_dt2 + ...
                        dadj_vy_dy .* d2_vy_dt2) * dt;
                end
                
                % 显示进度
                if mod(it, 100) == 0
                    fprintf('完成时间步: %d/%d\n', it, obj.NSTEP);
                end
            end
            
            % 组合最终梯度
            gradient = struct('c11', gradient_c11, ...
                'c13', gradient_c13, ...
                'c33', gradient_c33, ...
                'c44', gradient_c44, ...
                'rho', gradient_rho);
            
            % 输出最终梯度统计
            fprintf('\n=== 最终梯度统计 ===\n');
            fprintf('C11最大梯度: %e\n', max(abs(gradient_c11(:))));
            fprintf('C13最大梯度: %e\n', max(abs(gradient_c13(:))));
            fprintf('C33最大梯度: %e\n', max(abs(gradient_c33(:))));
            fprintf('C44最大梯度: %e\n', max(abs(gradient_c44(:))));
            fprintf('密度最大梯度: %e\n', max(abs(gradient_rho(:))));
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
        
        % function [dx, dy] = compute_gradient(obj, field)
        %     % 获取实际的网格间距（从参数中）
        %     deltax = obj.adjoint_solver.syn_params.DELTAX;  % 单位：m
        %     deltay = obj.adjoint_solver.syn_params.DELTAY;  % 单位：m
            
        %     % 计算导数并除以实际间距
        %     [dy, dx] = gradient(field);
        %     dx = dx / deltax;  % 得到正确的物理单位
        %     dy = dy / deltay;
        % end

       
%{
  function [dx, dy] = compute_gradient(obj, field)
            % 获取网格间距
            deltax = obj.adjoint_solver.syn_params.DELTAX;
            deltay = obj.adjoint_solver.syn_params.DELTAY;
            
            % 使用四阶中心差分计算空间导数
            % dx = (f(i+1/2) - f(i-1/2))/dx
            % 四阶精度系数
            c1 = 9/8;
            c2 = -1/24;
            
            % 水平方向导数 (x方向)
            dx = zeros(size(field));
            dx(:,3:end-2) = (c1*(field(:,4:end-1) - field(:,2:end-3)) + ...
                             c2*(field(:,5:end) - field(:,1:end-4))) / deltax;
            % 处理边界（使用二阶差分）
            dx(:,1) = (-3*field(:,1) + 4*field(:,2) - field(:,3))/(2*deltax);
            dx(:,2) = (-3*field(:,2) + 4*field(:,3) - field(:,4))/(2*deltax);
            dx(:,end-1) = (3*field(:,end-1) - 4*field(:,end-2) + field(:,end-3))/(2*deltax);
            dx(:,end) = (3*field(:,end) - 4*field(:,end-1) + field(:,end-2))/(2*deltax);
            
            % 垂直方向导数 (y方向)
            dy = zeros(size(field));
            dy(3:end-2,:) = (c1*(field(4:end-1,:) - field(2:end-3,:)) + ...
                             c2*(field(5:end,:) - field(1:end-4,:))) / deltay;
            % 处理边界（使用二阶差分）
            dy(1,:) = (-3*field(1,:) + 4*field(2,:) - field(3,:))/(2*deltay);
            dy(2,:) = (-3*field(2,:) + 4*field(3,:) - field(4,:))/(2*deltay);
            dy(end-1,:) = (3*field(end-1,:) - 4*field(end-2,:) + field(end-3,:))/(2*deltay);
            dy(end,:) = (3*field(end,:) - 4*field(end-1,:) + field(end-2,:))/(2*deltay);
        end 
%}


        function [dx, dy] = compute_gradient(obj, field)
            % 建议添加输入验证
            if isempty(field)
                error('输入波场为空');
            end
            
            deltax = obj.adjoint_solver.syn_params.DELTAX;
            deltay = obj.adjoint_solver.syn_params.DELTAY;
            
            % 建议添加网格参数验证
            if deltax <= 0 || deltay <= 0
                error('网格间距必须为正数');
            end
            
            [dx, dy] = compute_gradient_mex(field, deltax, deltay);
        end
    end
end 