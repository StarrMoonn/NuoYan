%% VTI介质伴随波场计算类
% 功能：实现VTI介质中的伴随波场计算，用于全波形反演中的梯度计算
% 
% 说明：
%   1. 管理观测数据和合成数据的波场求解：
%      - 计算观测数据和合成数据的残差
%      - 基于残差进行伴随波场的时间反传
%   2. 支持波场快照的保存和管理：
%      - 可配置是否保存伴随波场快照
%      - 支持自定义保存间隔
%   3. 提供完整的伴随状态计算流程
%
% 主要方法：
%   - compute_residuals_single_shot：计算单炮的观测和合成数据残差
%   - compute_adjoint_wavefield_single_shot：计算单炮的伴随波场
%   - add_adjoint_source：在检波器位置添加伴随源
%   - save_adjoint_snapshot：保存伴随波场快照
%
% 输入参数：
%   params结构体必须包含：
%   - obs_params：观测数据的模型参数
%   - syn_params：合成数据的模型参数
%
% 输出：
%   - current_residuals_vx/vy：速度场残差
%   - current_adjoint_vx/vy：伴随波场
%   - 波场快照（可选）：保存在指定目录
%
% 使用示例：
%   params.obs_params = struct(...);  % 设置观测数据参数
%   params.syn_params = struct(...);  % 设置合成数据参数
%   adjoint_solver = VTI_Adjoint(params);
%   adjoint_solver.compute_adjoint_wavefield_single_shot(1);
%
% 注意事项：
%   - 需要正确设置观测数据和合成数据的参数
%   - 波场快照的保存可能占用大量磁盘空间
%   - 时间反传过程需要足够的内存
%
% 作者：StarrMoonn
% 日期：2025-01-06
%
classdef VTI_Adjoint < handle
    properties
        % 波场求解器实例
        wavefield_solver_obs     % 观测数据的波场求解器
        wavefield_solver_syn     % 合成数据的波场求解器
        
        % 模型参数
        obs_params              % 观测数据的模型参数
        syn_params              % 合成数据的模型参数
        
        % 基本参数
        NSTEP                   % 时间步数
        NREC                    % 检波器数量
        DELTAT                  % 时间步长
        
        % 波场快照相关
        save_adjoint_snapshots  % 是否保存伴随波场快照
        snapshot_interval       % 快照保存间隔
        adjoint_output_dir      % 伴随波场输出目录
        
        % 当前炮的数据
        current_residuals_vx    % 当前炮的速度场x分量残差
        current_residuals_vy    % 当前炮的速度场y分量残差
    end
    
    methods
        % 1. 构造函数
        function obj = VTI_Adjoint(params)
            % 构造函数
            % params 结构体必须包含:
            % - obs_params: 观测数据参数
            % - syn_params: 合成数据参数
            
            if ~(isfield(params, 'obs_params') && isfield(params, 'syn_params'))
                error('参数必须包含 obs_params和syn_params');
            end
            
            % 保存模型参数
            obj.obs_params = params.obs_params;
            obj.syn_params = params.syn_params;
            
            % 创建波场求解器实例
            obj.wavefield_solver_obs = VTI_SingleShotModeling(obj.obs_params);
            obj.wavefield_solver_syn = VTI_SingleShotModeling(obj.syn_params);
            
            % 使用syn_params的参数
            obj.NSTEP = obj.syn_params.NSTEP;
            obj.NREC = obj.syn_params.NREC;
            obj.DELTAT = obj.syn_params.DELTAT;
            
            % 设置波场快照相关的默认参数
            obj.save_adjoint_snapshots = true;
            obj.snapshot_interval = 100;
            
            % 设置输出目录（与VTI_WaveFieldSolver保持一致的结构）
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.adjoint_output_dir = fullfile(current_dir, 'data', 'output', 'wavefield', 'adjoint');
        end
        
      
        
        % 2. 计算单炮残差
        function compute_residuals_single_shot(obj, ishot)
            % 计算单炮的观测数据和合成数据
            fprintf('\n=== 计算第 %d 炮的残差 ===\n', ishot);
            
            [obs_vx, obs_vy, ~] = obj.wavefield_solver_obs.forward_modeling_single_shot(ishot);
            [syn_vx, syn_vy, ~] = obj.wavefield_solver_syn.forward_modeling_single_shot(ishot);
            
            % 计算残差
            obj.current_residuals_vx = obs_vx - syn_vx;
            obj.current_residuals_vy = obs_vy - syn_vy;
            
            % 修正：输出单炮的残差值
            fprintf('残差 vx 最大值: %e\n', max(abs(obj.current_residuals_vx(:))));
            fprintf('残差 vy 最大值: %e\n\n', max(abs(obj.current_residuals_vy(:))));
            
            fprintf('检波器残差统计：\n');
            for irec = 1:obj.NREC
                fprintf('检波器%d: vx残差=%e, vy残差=%e\n', ...
                    irec, max(abs(obj.current_residuals_vx(:,irec))), ...
                    max(abs(obj.current_residuals_vy(:,irec))));
            end
        end
        
        % 3. 计算单炮伴随波场   
        function [adjoint_wavefield] = compute_adjoint_wavefield_single_shot(obj, ishot)
            % 计算单炮伴随波场
            fprintf('计算第 %d 炮的伴随波场\n', ishot);
            
            % 计算残差
            obj.compute_residuals_single_shot(ishot);
            
            % 使用合成数据的模型进行伴随波场计算
            wave_solver = obj.wavefield_solver_syn.fd_solver;
            
            % 设置PML边界条件
            wave_solver.setup_pml_boundary();
            wave_solver.setup_pml_boundary_x();
            wave_solver.setup_pml_boundary_y();
            
            % 重置波场
            wave_solver.reset_fields();
            
            % 初始化存储完整时间历史的结构体
            adjoint_wavefield = struct(...
                'vx', zeros(wave_solver.NX, wave_solver.NY, obj.NSTEP), ...
                'vy', zeros(wave_solver.NX, wave_solver.NY, obj.NSTEP));
            
            % 时间反传
            fprintf('\n=== 开始伴随波场时间反传 ===\n');
            for it = obj.NSTEP:-1:1
                if mod(obj.NSTEP-it+1, 100) == 0  % 每100步输出一次信息
                    fprintf('计算伴随波场: 时间步 %d/%d\n', obj.NSTEP-it+1, obj.NSTEP);
                end
                
                % 添加伴随源
                obj.add_adjoint_source(wave_solver, obj.current_residuals_vx, ...
                    obj.current_residuals_vy, it);
                
                % 计算波场传播
                wave_solver.compute_wave_propagation();
                
                % 应用边界条件
                wave_solver.apply_boundary_conditions();
                
                % 保存当前时间步的波场
                % 注意：由于是时间反传，我们需要正确映射时间索引
                time_index = obj.NSTEP - it + 1;  % 将反向时间映射到正向索引
                adjoint_wavefield.vx(:,:,time_index) = wave_solver.vx;
                adjoint_wavefield.vy(:,:,time_index) = wave_solver.vy;
                
                % 保存波场快照（如果需要的话）
                obj.save_adjoint_snapshot(wave_solver, ishot, it);
            end
            fprintf('伴随波场计算完成！\n\n');
        end
        
        % 4. 添加伴随源
        function add_adjoint_source(obj, fd_solver, residuals_vx, residuals_vy, it)
            % 在检波器位置添加伴随源
            for irec = 1:obj.NREC
                i = fd_solver.rec_x(irec);
                j = fd_solver.rec_y(irec);
                
                % 直接赋值即可，不需要累加
                fd_solver.vx(i,j) = residuals_vx(it, irec) * fd_solver.DELTAT;
                fd_solver.vy(i,j) = residuals_vy(it, irec) * fd_solver.DELTAT;
            end
        end
        
        % 5. 保存波场快照
        function save_adjoint_snapshot(obj, fd_solver, ishot, it)
            % 检查是否需要保存快照
            if ~obj.save_adjoint_snapshots || mod(it, obj.snapshot_interval) ~= 0
                return
            end
            
            % 获取波场数据
            vx_data = fd_solver.vx;
            vy_data = fd_solver.vy;
            
            % 保存波场数据
            shot_dir = fullfile(obj.adjoint_output_dir, sprintf('shot_%03d', ishot));
            if ~exist(shot_dir, 'dir')
                mkdir(shot_dir);
            end
            
            save_path = fullfile(shot_dir, sprintf('adjoint_wavefield_%06d.mat', it));
            save(save_path, 'vx_data', 'vy_data', '-v7.3');
            
            fprintf('保存伴随波场快照: 炮号 %d, 时间步 %d\n', ishot, it);
        end
    end
end 