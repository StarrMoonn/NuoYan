%% VTI介质伴随波场计算类
% 功能：实现VTI介质中的伴随波场计算，用于全波形反演中的梯度计算
% 
% 说明：
%   1. 管理观测数据和合成数据的波场求解：
%      - 计算观测数据和合成数据的残差
%      - 基于残差进行伴随波场的时间反传
%   2. 支持伴随波场的计算和保存：
%      - 计算完整的时间反传伴随波场
%      - 支持保存伴随波场到指定目录
%   3. 提供完整的伴随状态计算流程
%
% 主要方法：
%   - compute_residuals_single_shot：计算单炮的观测和合成数据残差
%   - compute_adjoint_wavefield_single_shot：计算单炮的伴随波场
%   - add_adjoint_source：在检波器位置添加伴随源
%
% 输入参数：
%   params结构体必须包含：
%   - obs_params：观测数据的模型参数
%   - syn_params：合成数据的模型参数
%
% 输出：
%   - current_residuals_vx/vy：速度场残差
%   - adjoint_wavefield：完整的伴随波场结构体
%   - 目标函数值（misfit）：用于优化过程
%
% 使用示例：
%   params.obs_params = struct(...);  % 设置观测数据参数
%   params.syn_params = struct(...);  % 设置合成数据参数
%   adjoint_solver = VTI_Adjoint(params);
%   
%   % 首先计算残差
%   misfit = adjoint_solver.compute_residuals_single_shot(1);
%   
%   % 然后计算伴随波场
%   adjoint_wavefield = adjoint_solver.compute_adjoint_wavefield_single_shot(1);
%
% 注意事项：
%   - 需要正确设置观测数据和合成数据的参数
%   - 必须先调用compute_residuals_single_shot后才能计算伴随波场
%   - 完整伴随波场的存储可能占用大量内存
%   - 时间反传过程需要足够的计算资源
%
% 作者：StarrMoonn
% 日期：2025-04-10
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
        
        % 当前炮的数据
        current_residuals_vx    % 当前炮的速度场x分量残差
        current_residuals_vy    % 当前炮的速度场y分量残差
        
        % 新增属性
        output_dir              % 输出目录
        adjoint_output_dir      % 伴随波场输出目录
        save_to_disk            % 是否保存到磁盘
    end
    
    methods
        % 1. 构造函数
        function obj = VTI_Adjoint(params)
            % 构造函数    
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
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.output_dir = fullfile(current_dir, 'data', 'output');
            % 伴随波场输出目录
            obj.adjoint_output_dir = fullfile(obj.output_dir, 'wavefield', 'adjoint');
            
            % 默认开启磁盘保存
            obj.save_to_disk = true;
            
            % 创建输出目录
            if obj.save_to_disk
                if ~exist(fullfile(obj.output_dir, 'wavefield'), 'dir')
                    mkdir(fullfile(obj.output_dir, 'wavefield'));
                end
                if ~exist(obj.adjoint_output_dir, 'dir')
                    mkdir(obj.adjoint_output_dir);
                end
            end
        end
        
        % 1. 计算单炮残差和目标函数值
        function [misfit] = compute_residuals_single_shot(obj, ishot)
            % 计算单炮的观测数据和合成数据
            fprintf('\n=== 计算第 %d 炮的残差 ===\n', ishot);
            
            % 计算观测和合成数据
            [obs_vx, obs_vy] = obj.wavefield_solver_obs.forward_modeling_single_shot(ishot);
            [syn_vx, syn_vy] = obj.wavefield_solver_syn.forward_modeling_single_shot(ishot);
            
            % 计算残差
            obj.current_residuals_vx = obs_vx - syn_vx;
            obj.current_residuals_vy = obs_vy - syn_vy;
            
            % 计算目标函数值（L2范数）
            misfit = 0.5 * (sum(obj.current_residuals_vx(:).^2) + ...
                           sum(obj.current_residuals_vy(:).^2));
        end
        
        % 2. 计算单炮伴随波场
        function [adjoint_wavefield] = compute_adjoint_wavefield_single_shot(obj, ishot)
            % 计算单炮伴随波场
            fprintf('计算第 %d 炮的伴随波场\n', ishot);
            
            % 使用合成数据的模型进行伴随波场计算
            wave_solver = obj.wavefield_solver_syn.fd_solver;
            
            % 设置PML边界条件
            wave_solver.setup_pml_boundary();
            wave_solver.setup_pml_boundary_x();
            wave_solver.setup_pml_boundary_y();
            
            % 重置波场
            wave_solver.reset_fields();
            
            % 初始化伴随波场结构体
            adjoint_wavefield = struct(...
                'vx', zeros(wave_solver.NX, wave_solver.NY, obj.NSTEP), ...
                'vy', zeros(wave_solver.NX, wave_solver.NY, obj.NSTEP));
            
            % 时间反传
            fprintf('\n=== 开始伴随波场时间反传 ===\n');
            for it = obj.NSTEP:-1:1
                if mod(obj.NSTEP-it+1, 1000) == 0
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
                time_index = obj.NSTEP - it + 1;  % 将反向时间映射到正向索引
                adjoint_wavefield.vx(:,:,time_index) = wave_solver.vx;
                adjoint_wavefield.vy(:,:,time_index) = wave_solver.vy;
            end
            
            % 清理内存
            clear wave_solver;
            
            fprintf('伴随波场计算完成！\n\n');
            
            % 保存完整的伴随波场
            if obj.save_to_disk
                adjoint_file = fullfile(obj.adjoint_output_dir, ...
                    sprintf('shot_%03d_adjoint_wavefield.mat', ishot));
                save(adjoint_file, 'adjoint_wavefield', '-v7.3');
            end
        end
        
        % 3. 添加伴随源
        function add_adjoint_source(obj, fd_solver, residuals_vx, residuals_vy, it)
            % 在检波器位置添加伴随源
            for irec = 1:obj.NREC
                i = fd_solver.rec_x(irec);
                j = fd_solver.rec_y(irec);
                
                % 不需要*dt，波动方程算子已经有dt
                fd_solver.vx(i,j) = residuals_vx(it, irec);
                fd_solver.vy(i,j) = residuals_vy(it, irec);
            end
        end
    end
end 