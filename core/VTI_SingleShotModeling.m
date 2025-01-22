%% VTI介质正演模拟主类
% 功能：实现VTI介质中的单炮正演模拟，管理震源位置和波场计算
% 
% 说明：
%   1. 封装了VTI_WaveFieldSolver类的功能，提供更高层次的接口
%   2. 管理单炮模拟的完整流程：
%      - 边界条件设置
%      - 震源位置更新
%      - 波场传播计算
%      - 检波器记录
%      - 地震记录和完整波场的存储
%   3. 提供灵活的波场存储策略：
%      - 内存模式：将完整波场保存在内存中（默认）
%      - 磁盘模式：将完整波场保存到硬盘，节省内存
%
% 类属性：
%   基本参数：
%   - NSTEP：时间步数
%   - NREC：检波器数量
%   - DELTAT：时间步长
%   
%   波场存储：
%   - stored_wavefield：存储当前波场数据
%   - stored_shot_no：记录当前存储的炮号
%   - wavefield_storage_mode：波场存储模式（'memory'或'disk'）
%
% 主要方法：
%   - forward_modeling_single_shot：执行单炮正演模拟
%   - set_source_position：设置震源位置
%   - get_complete_wavefield：获取完整波场数据（优先从内存获取）
%
% 波场存储机制：
%   1. 内存模式下：
%      - 波场计算后自动存储在stored_wavefield中
%      - get_complete_wavefield优先返回stored_wavefield
%      - 仅当stored_shot_no不匹配时才重新计算
%   2. 磁盘模式下：
%      - 波场计算后保存到磁盘
%      - get_complete_wavefield从磁盘读取
%
% 输入参数：
%   必需参数：
%   - NSTEP：时间步数
%   - NREC：检波器数量
%   - DELTAT：时间步长
%   - first_shot_i/j：首炮位置
%   - shot_di/dj：炮点间隔
%   - 其他VTI介质参数（通过VTI_FD类传入）
%
%   可选参数：
%   - wavefield_storage_mode：波场存储模式（'memory'或'disk'，默认'memory'）
%
% 输出：
%   - vx_data：水平分量地震记录（自动保存到硬盘）
%   - vy_data：垂直分量地震记录（自动保存到硬盘）
%   - complete_wavefield：完整波场（根据存储模式保存在内存或硬盘）
%
% 存储路径：
%   - 地震记录：data/output/wavefield/seismograms/
%   - 完整波场：data/output/wavefield/complete_wavefields/（当选择磁盘存储时）
%
% 使用示例：
%   params = struct(...);  % 设置参数
%   params.wavefield_storage_mode = 'memory';  % 可选，设置存储模式
%   vti_forward = VTI_SingleShotModeling(params);
%   [vx, vy, wavefield] = vti_forward.forward_modeling_single_shot(1);
%   % 后续获取同一炮号的波场（直接从内存获取，不会重新计算）
%   wavefield = vti_forward.get_complete_wavefield(1);
%
% 注意事项：
%   - 需要正确设置PML边界参数
%   - 确保震源位置在计算区域内
%   - 检波器位置需要提前设置
%   - 选择磁盘存储模式时需确保有足够的硬盘空间
%   - 内存模式下只保存最后一次计算的波场
%
% 作者：StarrMoonn
% 日期：2025-01-08
%
classdef VTI_SingleShotModeling < handle
    properties
        % 基本参数
        NSTEP           % 时间步数
        NREC            % 检波器数量
        DELTAT          % 时间步长
        
        % 震源位置参数
        first_shot_i    % 首炮x位置
        first_shot_j    % 首炮y位置
        shot_di         % 炮点x间隔
        shot_dj         % 炮点y间隔
        
        % 求解器实例
        fd_solver       % VTI_FD实例
        
        % 输出目录设置
        output_dir       % 基础输出目录
        seismogram_dir   % 地震记录目录
    end
    
    methods
        function obj = VTI_SingleShotModeling(params)
            % 构造函数
            % params需要包含：
            % - 基本计算参数（NSTEP, NREC, DELTAT等）
            % - 震源位置参数（first_shot_i, first_shot_j, shot_di, shot_dj）
            % - 模型参数

            % 保存震源位置参数
            obj.first_shot_i = params.first_shot_i;
            obj.first_shot_j = params.first_shot_j;
            obj.shot_di = params.shot_di;
            obj.shot_dj = params.shot_dj;
            
            % 创建VTI_WaveFieldSolver实例
            obj.fd_solver = VTI_WaveFieldSolver(params);
            
            % 设置基本参数
            obj.NSTEP = params.NSTEP;
            obj.NREC = params.NREC;
            obj.DELTAT = params.DELTAT;
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.output_dir = fullfile(current_dir, 'data', 'output', 'wavefield');
            obj.seismogram_dir = fullfile(obj.output_dir, 'seismograms');
            
            % 确保地震记录目录存在
            if ~exist(obj.seismogram_dir, 'dir')
                mkdir(obj.seismogram_dir);
            end
        end
        
        function [vx_data, vy_data, complete_wavefield] = forward_modeling_single_shot(obj, ishot)
            % 计算单炮正演波场
            fprintf('计算第 %d 炮的正演波场\n', ishot);
            
            % 开始计时
            shot_time_start = tic;
            
            % 更新当前炮号
            obj.fd_solver.set_current_shot(ishot);
            
            % 设置PML边界
            obj.fd_solver.setup_pml_boundary();
            obj.fd_solver.setup_pml_boundary_x();
            obj.fd_solver.setup_pml_boundary_y();
            
            % 设置当前炮的震源位置
            obj.set_source_position(ishot);
            
            % 重置波场
            obj.fd_solver.reset_fields();
            
            % 获取检波器位置索引
            indices = sub2ind(size(obj.fd_solver.vx), ...
                            obj.fd_solver.rec_x, obj.fd_solver.rec_y);
            
            % 初始化输出数据
            vx_data = zeros(obj.NSTEP, obj.NREC);
            vy_data = zeros(obj.NSTEP, obj.NREC);
            
            % 初始化完整波场存储
            complete_wavefield = struct(...
                'vx', zeros(obj.fd_solver.NX, obj.fd_solver.NY, obj.NSTEP), ...
                'vy', zeros(obj.fd_solver.NX, obj.fd_solver.NY, obj.NSTEP));
            
            % 时间步进
            for it = 1:obj.NSTEP
                % 计算波场传播
                obj.fd_solver.compute_wave_propagation();
                
                % 添加震源
                obj.fd_solver.add_source(it);
                
                % 应用边界条件
                obj.fd_solver.apply_boundary_conditions();
                
                % 记录检波器数据和完整波场
                vx_data(it,:) = obj.fd_solver.vx(indices);
                vy_data(it,:) = obj.fd_solver.vy(indices);
                complete_wavefield.vx(:,:,it) = obj.fd_solver.vx;
                complete_wavefield.vy(:,:,it) = obj.fd_solver.vy;
                
                % 输出进度信息
                if mod(it, 100) == 0
                    obj.fd_solver.output_info(it);
                end
            end
            
            % 计算并输出总耗时
            total_time = toc(shot_time_start);
            fprintf('\n=== 第 %d 炮模拟完成 ===\n', ishot);
            fprintf('总计算时间: %.2f 秒\n\n', total_time);
            
            % 保存地震记录
            seismic_file = fullfile(obj.seismogram_dir, sprintf('shot_%03d_seismogram.mat', ishot));
            save(seismic_file, 'vx_data', 'vy_data', '-v7.3');
        end
        
        function set_source_position(obj, ishot)
            % 设置震源位置
            obj.fd_solver.ISOURCE = obj.first_shot_i + (ishot-1) * obj.shot_di;
            obj.fd_solver.JSOURCE = obj.first_shot_j + (ishot-1) * obj.shot_dj;
            
            % 更新震源物理坐标
            obj.fd_solver.xsource = (obj.fd_solver.ISOURCE - 1) * obj.fd_solver.DELTAX;
            obj.fd_solver.ysource = (obj.fd_solver.JSOURCE - 1) * obj.fd_solver.DELTAY;
        end
        
        % 添加新方法：更新模型参数
        function update_model_params(obj, model)
            % 直接更新波场求解器的模型参数
            obj.fd_solver.c11 = model.c11;
            obj.fd_solver.c13 = model.c13;
            obj.fd_solver.c33 = model.c33;
            obj.fd_solver.c44 = model.c44;
            obj.fd_solver.rho = model.rho;
        end
    end
end 