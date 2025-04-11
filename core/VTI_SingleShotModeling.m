%% VTI介质正演模拟主类
% 功能：实现VTI介质中的单炮正演模拟，管理震源位置和波场计算
% 
% 说明：
%   1. 封装了VTI_WaveFieldSolver类的功能，提供更高层次的接口
%   2. 管理单炮模拟的完整流程
%   3. 保存FWI所需的关键数据：地震记录和完整波场
%
% 类属性：
%   基本参数：
%   - NSTEP：时间步数
%   - NREC：检波器数量
%   - DELTAT：时间步长
%
% 主要方法：
%   - forward_modeling_single_shot：执行单炮正演模拟
%   - set_source_position：设置震源位置
%   - update_model_params：更新模型参数（用于FWI迭代）
%
% 输入参数：
%   必需参数：
%   - NSTEP：时间步数
%   - NREC：检波器数量
%   - DELTAT：时间步长
%   - first_shot_x/y：首炮位置
%   - shot_dx/dy：炮点间隔
%   - 其他VTI介质参数
%
% 输出：
%   - vx_data：水平分量地震记录
%   - vy_data：垂直分量地震记录
%   - complete_wavefield：完整波场（用于梯度计算）
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
        IT_DISPLAY      % 输出打印间隔
        
        % 震源位置参数
        first_shot_x    % 首炮x位置（网格点）   
        first_shot_y    % 首炮y位置（网格点）
        shot_dx         % 炮点x间隔（网格点）
        shot_dy         % 炮点y间隔（网格点）
        
        % 求解器实例
        fd_solver       % VTI_WaveFieldSolver实例
        
        % 输出目录设置
        output_dir       % 基础输出目录
        seismogram_dir   % 地震记录目录
        current_shot_number  % 当前炮号
    end
    
    methods
        function obj = VTI_SingleShotModeling(params)
            % 保存震源位置参数
            obj.first_shot_x = params.first_shot_x;
            obj.first_shot_y = params.first_shot_y;
            obj.shot_dx = params.shot_dx;
            obj.shot_dy = params.shot_dy;
            
            % 创建VTI_WaveFieldSolver实例
            obj.fd_solver = VTI_WaveFieldSolver(params);
            
            % 设置基本参数
            obj.NSTEP = params.NSTEP;
            obj.NREC = params.NREC;
            obj.DELTAT = params.DELTAT;
            obj.IT_DISPLAY = params.IT_DISPLAY;
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.output_dir = fullfile(current_dir, 'data', 'output', 'wavefield');
            obj.seismogram_dir = fullfile(obj.output_dir, 'seismograms');
            
            % 确保地震记录目录存在
            if ~exist(obj.seismogram_dir, 'dir')
                mkdir(obj.seismogram_dir);
            end
            
            % 初始化当前炮号
            obj.current_shot_number = 1;
        end
        
        function [vx_data, vy_data, complete_wavefield] = forward_modeling_single_shot(obj, ishot)
            % 计算单炮正演波场
            fprintf('计算第 %d 炮的正演波场\n', ishot);
            
            % 开始计时
            shot_time_start = tic;
            
            % 更新当前炮号
            obj.current_shot_number = ishot;
            
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
                if mod(it, obj.IT_DISPLAY) == 0
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
            obj.fd_solver.ISOURCE = obj.first_shot_x + (ishot-1) * obj.shot_dx;
            obj.fd_solver.JSOURCE = obj.first_shot_y + (ishot-1) * obj.shot_dy;
            
            % 更新震源物理坐标
            obj.fd_solver.xsource = (obj.fd_solver.ISOURCE - 1) * obj.fd_solver.DELTAX;
            obj.fd_solver.ysource = (obj.fd_solver.JSOURCE - 1) * obj.fd_solver.DELTAY;
        end
        
        % 添加新方法：更新模型参数（用于FWI迭代）
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