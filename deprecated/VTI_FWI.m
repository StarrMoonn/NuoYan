%% VTI介质全波形反演主程序（废弃）
% 功能：实现VTI介质中的多炮正演模拟，支持CPU并行和GPU加速
% 
% 说明：
%   1. 支持两种计算模式：
%      - CPU并行模式：使用parfor进行多炮并行计算
%      - GPU串行模式：使用GPU加速单炮计算，炮间串行执行
%   2. 自动管理多炮的震源位置更新
%   3. 提供地震记录的存储和管理
%   4. 支持计算性能监控和统计
%
% 计算模式选择：
%   params.compute_mode = 'cpu_parallel'：
%     - 适用于：多核CPU环境，炮数较多的情况
%     - 优势：多炮并行计算，总体时间可能更短
%     - 特点：单炮计算在CPU上执行，多炮使用parfor并行
%
%   params.compute_mode = 'gpu_serial'：
%     - 适用于：有GPU加速卡，单炮计算量大的情况
%     - 优势：单炮计算速度快，内存效率高
%     - 特点：使用GPU加速单炮计算，炮间串行执行
%
% 输入参数：
%   - first_shot_i/j：首炮位置网格索引
%   - shot_di/dj：炮点间隔（网格点数）
%   - compute_mode：计算模式选择
%   - 其他VTI介质参数（通过VTI_FD类传入）
%
% 输出：
%   - seismogram_vx_all：所有炮的水平分量地震记录
%   - seismogram_vy_all：所有炮的垂直分量地震记录
%
% 使用示例：
%   params.compute_mode = 'cpu_parallel';
%   fwi = VTI_FWI(params);
%   fwi.forward_modeling_all_shots();
%
% 注意事项：
%   - GPU模式需要检查显存是否足够
%   - CPU并行模式需要合理设置并行工作器数量
%   - 建议先用小规模数据测试性能后再选择合适的模式
%
% 作者：StarrMoonn
% 日期：2025-01-03
%
classdef VTI_FWI < handle
    properties
        % 添加params属性
        params          % 存储初始化参数
        
        % 多炮控制相关的参数
        first_shot_i    % 第一炮x网格索引
        first_shot_j    % 第一炮y网格索引
        shot_di         % 炮点x网格间隔
        shot_dj         % 炮点y网格间隔
        
        % 计算规模参数
        nshot           % 总炮数
        nstep           % 总时间步数
        nrec            % 检波器数量
        
        % 求解器和数据存储
        fd_solver       % VTI_FD实例
        seismogram_vx_all  % 存储所有炮的水平分量地震记录
        seismogram_vy_all  % 存储所有炮的垂直分量地震记录
        
        % 计算模式控制
        compute_mode    % 'cpu_parallel' 或 'gpu_serial'
    end
    
    methods
        function obj = VTI_FWI(params)
            % 保存params以供后续使用
            obj.params = params;
            
            % 先初始化多炮参数
            obj.first_shot_i = params.first_shot_i;
            obj.first_shot_j = params.first_shot_j;
            obj.shot_di = params.shot_di;
            obj.shot_dj = params.shot_dj;
            
            % 创建VTI_FD实例
            obj.fd_solver = VTI_FD(params);
            
            % 设置第一炮的位置
            obj.fd_solver.ISOURCE = obj.first_shot_i;
            obj.fd_solver.JSOURCE = obj.first_shot_j;
            obj.fd_solver.xsource = (obj.fd_solver.ISOURCE - 1) * obj.fd_solver.DELTAX;
            obj.fd_solver.ysource = (obj.fd_solver.JSOURCE - 1) * obj.fd_solver.DELTAY;
            
            % 设置计算模式，默认使用CPU并行
            if isfield(params, 'compute_mode')
                obj.compute_mode = params.compute_mode;
            else
                obj.compute_mode = 'cpu_parallel';
            end
            
            % 只在CPU并行模式下创建并行池
            if strcmp(obj.compute_mode, 'cpu_parallel') && isempty(gcp('nocreate'))
                parpool('local', 10); % 使用八个核心
            end
            
            % 预分配存储空间
            obj.seismogram_vx_all = cell(1, obj.fd_solver.NSHOT);
            obj.seismogram_vy_all = cell(1, obj.fd_solver.NSHOT);
            
            % 初始化计算规模参数
            obj.nshot = obj.fd_solver.NSHOT;
            obj.nstep = obj.fd_solver.NSTEP;
            obj.nrec = obj.fd_solver.NREC;  
        end
        
        function forward_modeling_all_shots(obj)
            % 开始计时
            fprintf('\n=== 开始多炮正演模拟 ===\n');
            total_time_start = tic;
            
            % 设置PML边界（只需要设置一次）
            obj.fd_solver.setup_pml_boundary();
            obj.fd_solver.setup_pml_boundary_x();
            obj.fd_solver.setup_pml_boundary_y();
            
            % 打印初始参数
            fprintf('\n=== 初始参数 ===\n');
            fprintf('材料参数: c11=%e, c33=%e, rho=%f (显示第一个元素值)\n', ...
                    obj.fd_solver.c11(1,1), obj.fd_solver.c33(1,1), obj.fd_solver.rho(1,1));
            fprintf('网格参数: dx=%e, dy=%e, dt=%e\n', ...
                    obj.fd_solver.DELTAX, obj.fd_solver.DELTAY, obj.fd_solver.DELTAT);
            
            % 计算和检查Courant数
            quasi_cp_max = max(max(max(sqrt(obj.fd_solver.c33./obj.fd_solver.rho))), ...
                              max(max(sqrt(obj.fd_solver.c11./obj.fd_solver.rho))));
            Courant_number = quasi_cp_max * obj.fd_solver.DELTAT * ...
                             sqrt(1.0/obj.fd_solver.DELTAX^2 + 1.0/obj.fd_solver.DELTAY^2);
            fprintf('Courant数为 %f\n', Courant_number);
            
            % 预分配cell数组
            temp_vx_all = cell(1, obj.nshot);
            temp_vy_all = cell(1, obj.nshot);

            % 预创建求解器数组
            solver_array = cell(1, obj.nshot);
            for i = 1:obj.nshot
                solver_array{i} = obj.fd_solver;
            end

            switch obj.compute_mode
                case 'cpu_parallel'
                    % 只保留一个parfor循环
                    parfor ishot = 1:obj.nshot
                        local_solver = solver_array{ishot};
                        
                        % 记录每炮开始时间
                        shot_time_start = tic;
                        
                        % 更新当前炮的位置
                        local_solver.current_shot_number = ishot;
                        local_solver.ISOURCE = obj.first_shot_i + (ishot-1) * obj.shot_di;
                        local_solver.JSOURCE = obj.first_shot_j + (ishot-1) * obj.shot_dj;
                        
                        % 更新震源物理坐标
                        local_solver.xsource = (local_solver.ISOURCE - 1) * local_solver.DELTAX;
                        local_solver.ysource = (local_solver.JSOURCE - 1) * local_solver.DELTAY;
                        
                        fprintf('\n=== 开始计算第 %d/%d 炮 ===\n', ishot, obj.nshot);
                        fprintf('震源位置: ISOURCE=%d, JSOURCE=%d\n', ...
                            local_solver.ISOURCE, local_solver.JSOURCE);
                        
                        % 只重置波场，不重新初始化
                        local_solver.reset_fields();
                        
                        % 每个时间步的波场计算
                        for it = 1:local_solver.NSTEP
                            local_solver.compute_wave_propagation();
                            local_solver.add_source(it);
                            local_solver.apply_boundary_conditions();
                            local_solver.record_seismograms(it);  % 只记录检波器位置的数据
                            
                            % 每100步调用output_info
                            if mod(it, 100) == 0
                                local_solver.output_info(it);
                            end
                        end
                        
                        % 存储结果
                        temp_vx_all{ishot} = local_solver.seismogram_vx;
                        temp_vy_all{ishot} = local_solver.seismogram_vy;
                        
                        % 计算并显示单炮耗时
                        shot_time = toc(shot_time_start);
                        fprintf('第 %d 炮计算完成，耗时: %.2f 秒\n', ishot, shot_time);
                    end
                    
                case 'gpu_serial'
                    fprintf('\n=== 使用GPU串行计算 ===\n');
                    fprintf('GPU代码尚未完善，暂不支持GPU计算模式\n');
                    fprintf('请使用 params.compute_mode = ''cpu_parallel'' 运行程序\n\n');
                    return;  % 使用return而不是error来终止
                    
                  
%{
   % 检查GPU是否可用
                    try
                        gpu_device = gpuDevice();  % 获取当前GPU设备
                        fprintf('使用GPU设备: %s\n', gpu_device.Name);
                    catch
                        error('没有找到可用的GPU设备或GPU工具箱未安装');
                    end
                    
                    % 设置FD求解器为GPU模式
                    obj.fd_solver.compute_kernel = 'gpu';
                    
                    % 预分配存储空间
                    obj.seismogram_vx_all = cell(1, obj.nshot);
                    obj.seismogram_vy_all = cell(1, obj.nshot);
                    
                    % GPU串行计算
                    for ishot = 1:obj.nshot
                        shot_time_start = tic;
                        
                        % 更新当前炮的位置
                        obj.fd_solver.current_shot_number = ishot;
                        obj.fd_solver.ISOURCE = obj.first_shot_i + (ishot-1) * obj.shot_di;
                        obj.fd_solver.JSOURCE = obj.first_shot_j + (ishot-1) * obj.shot_dj;
                        obj.fd_solver.xsource = (obj.fd_solver.ISOURCE - 1) * obj.fd_solver.DELTAX;
                        obj.fd_solver.ysource = (obj.fd_solver.JSOURCE - 1) * obj.fd_solver.DELTAY;
                        
                        fprintf('\n=== 开始计算第 %d/%d 炮 ===\n', ishot, obj.nshot);
                        fprintf('正在初始化GPU数组...\n');
                        
                        try
                            % 重置波场并转移到GPU
                            obj.fd_solver.reset_fields();
                            obj.fd_solver.initialize_gpu_arrays();    % 只需要将数组转移到GPU
                            fprintf('GPU数组初始化完成\n');
                            
                            % 执行当前炮的正演模拟
                            for it = 1:obj.fd_solver.NSTEP
                                if mod(it, 10) == 0  % 增加输出频率，方便调试
                                    fprintf('正在计算时间步: %d/%d (%.1f%%)\r', ...
                                        it, obj.fd_solver.NSTEP, (it/obj.fd_solver.NSTEP)*100);
                                end
                                
                                obj.fd_solver.compute_wave_propagation();  % 使用统一接口
                                obj.fd_solver.add_source(it);
                                obj.fd_solver.apply_boundary_conditions();
                                obj.fd_solver.record_seismograms_gpu(it);  % 使用GPU版本
                            end
                            fprintf('\n');  % 换行
                            
                            % 时间步循环结束后，将地震记录数据传回CPU
                            fprintf('正在将数据从GPU传回CPU...\n');
                            obj.seismogram_vx_all{ishot} = gather(obj.fd_solver.seismogram_vx);
                            obj.seismogram_vy_all{ishot} = gather(obj.fd_solver.seismogram_vy);
                            
                        catch ME
                            % 如果发生错误，打印详细信息
                            fprintf('GPU计算出错:\n');
                            fprintf('错误信息: %s\n', ME.message);
                            fprintf('错误位置: %s\n', ME.stack(1).name);
                            rethrow(ME);
                        end
                        
                        shot_time = toc(shot_time_start);
                        fprintf('第 %d 炮计算完成，耗时: %.2f 秒\n', ishot, shot_time);
                        
                        % 清理GPU内存
                        fprintf('正在清理GPU内存...\n');
                        gpuDevice(1).reset();
                    end 
%}

            end
            
            % 将结果赋值给对象属性
            obj.seismogram_vx_all = temp_vx_all;
            obj.seismogram_vy_all = temp_vy_all;
            
            % 所有炮计算完成后，再保存完整的地震记录
            if ~exist(fullfile(obj.fd_solver.output_dir, 'seismograms'), 'dir')
                mkdir(fullfile(obj.fd_solver.output_dir, 'seismograms'));
            end

            % 创建临时变量并保存
            saved_seismogram_vx = obj.seismogram_vx_all;
            saved_seismogram_vy = obj.seismogram_vy_all;

            % 保存完整的地震记录
            save(fullfile(obj.fd_solver.output_dir, 'seismograms', 'seismogram_vx.mat'), 'saved_seismogram_vx');
            save(fullfile(obj.fd_solver.output_dir, 'seismograms', 'seismogram_vy.mat'), 'saved_seismogram_vy');
            
            % 计算并显示总耗时
            total_time = toc(total_time_start);
            fprintf('\n=== 多炮正演完成 ===\n');
            fprintf('总计算时间: %.2f 秒\n', total_time);
        end
        
        function new_obj = deepcopy(obj)
            % 创建一个新的对象并复制所有属性
            mc = metaclass(obj);
            new_obj = feval(class(obj));
            
            % 复制所有属性
            props = mc.PropertyList;
            for i = 1:length(props)
                if props(i).Dependent, continue; end
                new_obj.(props(i).Name) = obj.(props(i).Name);
            end
        end
    end
end 