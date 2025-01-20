%% 单炮正演算子测试（跟core中封装好的VTI_SingleShotModeling.m函数功能一样）
% 功能：测试VTI介质中的单炮正演模拟
% 
% 说明：
%   1. 初始化模型参数（网格大小、材料属性等）
%   2. 创建VTI_WaveFieldSolver实例并设置PML边界条件
%   3. 执行时间步进的波场模拟
%   4. 记录地震图并输出波场快照
%
% 参数设置：
%   - 模型尺寸：401x401网格点
%   - 空间步长：0.0625e-2 m
%   - 时间步长：50e-9 s
%   - 总时间步数：3000
%   - 震源频率：200 kHz
%   - PML层数：10
%
% 输出：
%   - 地震记录数据（水平和垂直分量）
%   - 波场快照（可选）
%   - 计算时间统计
%   - 稳定性检查（Courant数）
%
% 注意事项：
%   - 确保Courant数小于1以保证数值稳定性
%   - 检查最大速度范数以监控发散情况
%
% 作者：StarrMoonn
% 日期：2025-01-06
%

clear;
clc;

% 创建参数结构体
params = struct();
params.NX = 401;         
params.NY = 401;
params.DELTAX = 0.0625e-2;
params.DELTAY = 0.0625e-2;

% 定义网格尺寸
nx = params.NX;  
ny = params.NY;  

fprintf('\n=== 初始化材料参数数组 ===\n');
% 创建与原常数值相同的矩阵
params.c11 = ones(ny, nx) * 4.0e10;    
fprintf('c11数组大小: [%d, %d], 值: %e\n', size(params.c11, 1), size(params.c11, 2), params.c11(1,1));

params.c13 = ones(ny, nx) * 3.8e10;    
fprintf('c13数组大小: [%d, %d], 值: %e\n', size(params.c13, 1), size(params.c13, 2), params.c13(1,1));

params.c33 = ones(ny, nx) * 20.0e10;   
fprintf('c33数组大小: [%d, %d], 值: %e\n', size(params.c33, 1), size(params.c33, 2), params.c33(1,1));

params.c44 = ones(ny, nx) * 2.0e10;    
fprintf('c44数组大小: [%d, %d], 值: %e\n', size(params.c44, 1), size(params.c44, 2), params.c44(1,1));

params.rho = ones(ny, nx) * 4000;      
fprintf('rho数组大小: [%d, %d], 值: %e\n', size(params.rho, 1), size(params.rho, 2), params.rho(1,1));

% 设置其他参数
params.NSTEP = 3000;
params.DELTAT = 50e-9;  
params.f0 = 200.0e3;
params.NREC = 50;
params.first_rec_x = 100;
params.first_rec_y = 50;
params.rec_dx = 4;
params.rec_dy = 0;
params.NSHOT = 1;
params.factor = 1.0e7;
params.save_snapshots = true;
params.current_shot_number = 1;

% 设置PML相关参数
params.NPOINTS_PML = 10;            
params.PML_XMIN = true;
params.PML_XMAX = true;
params.PML_YMIN = true;
params.PML_YMAX = true;
params.NPOWER = 2;
params.K_MAX_PML = 1.0;

% 设置其他必要参数
params.DEGREES_TO_RADIANS = pi / 180.0;
params.ZERO = 0.0;
params.HUGEVAL = 1.0e30;
params.STABILITY_THRESHOLD = 1.0e25;
params.t0 = 1.20/params.f0;
params.ANGLE_FORCE = 0;  
params.ISOURCE = 200;
params.JSOURCE = 200;
params.xsource = (params.JSOURCE - 1) * params.DELTAX;
params.ysource = (params.ISOURCE - 1) * params.DELTAY;
params.IT_DISPLAY = 100;    

% 计算模式：'cpu' or 'cpu_mex' or 'openMP' or 'SIMD'
%params.compute_kernel = 'cpu_mex';  
%params.compute_kernel = 'openMP';  
params.compute_kernel = 'SIMD';
%params.compute_kernel = 'cpu'; 

fprintf('\n=== 创建VTI_WaveFieldSolver实例并初始化 ===\n');
% 1. 创建实例
vti_model = VTI_WaveFieldSolver(params);

% 2. 设置PML边界
vti_model.setup_pml_boundary();
vti_model.setup_pml_boundary_x();
vti_model.setup_pml_boundary_y();

% 3. 重置波场
vti_model.reset_fields();

fprintf('\n=== 初始参数 ===\n');
fprintf('材料参数: c11=%e, c33=%e, rho=%f (显示第一个元素值)\n', ...
        vti_model.c11(1,1), vti_model.c33(1,1), vti_model.rho(1,1));
fprintf('网格参数: dx=%e, dy=%e, dt=%e\n', ...
        vti_model.DELTAX, vti_model.DELTAY, vti_model.DELTAT);

% 计算最大波速
quasi_cp_max = max(max(max(sqrt(vti_model.c33./vti_model.rho))), ...
                  max(max(sqrt(vti_model.c11./vti_model.rho))));

% 计算 Courant 数
Courant_number = quasi_cp_max * vti_model.DELTAT * ...
                 sqrt(1.0/vti_model.DELTAX^2 + 1.0/vti_model.DELTAY^2);

fprintf('Courant数为 %f\n', Courant_number);
if Courant_number > 1.0
    error('时间步长过大，模拟将不稳定');
end

% 添加以下正演模拟部分
fprintf('\n=== 开始时间步进计算 ===\n');
total_time_start = tic;

% 初始化地震记录数组
seismogram_vx = zeros(params.NSTEP, params.NREC);
seismogram_vy = zeros(params.NSTEP, params.NREC);

for it = 1:params.NSTEP
    % 1. 波场传播计算
    vti_model.compute_wave_propagation();
    
    % 2. 添加震源
    vti_model.add_source(it);
    
    % 3. 应用边界条件
    vti_model.apply_boundary_conditions();
    
    % 4. 记录地震图
    vti_model.record_seismograms(it);
    
    % 5. 输出进度信息
    if mod(it, 100) == 0
        progress = (it/params.NSTEP) * 100;
        fprintf('时间步 %d/%d (%.1f%%): 最大速度值 vx=%e, vy=%e\n', ...
                it, params.NSTEP, progress, ...
                max(max(abs(vti_model.vx))), ...
                max(max(abs(vti_model.vy))));
    end
    
    % 6. 定期输出波场信息
    if mod(it, params.IT_DISPLAY) == 0
        vti_model.output_info(it);
    end
end

% 计算总耗时
total_time = toc(total_time_start);
fprintf('\n=== 模拟完成 ===\n');
fprintf('总计算时间: %.2f 秒\n', total_time);

fprintf('\n=== 测试完成 ===\n');