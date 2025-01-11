%% VTI_SingleShotModeling模块测试
% 功能：测试VTI介质中的单炮正演模拟模块
% 
% 说明：
%   1. 初始化模型参数
%   2. 创建VTI_Forward实例
%   3. 执行单炮正演模拟
%   4. 验证输出结果
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

% 初始化材料参数
fprintf('\n=== 初始化材料参数数组 ===\n');
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

% 设置计算参数
params.NSTEP = 3000;
params.DELTAT = 50e-9;  
params.f0 = 200.0e3;
params.NREC = 50;
params.first_rec_x = 100;
params.first_rec_y = 50;
params.rec_dx = 4;
params.rec_dy = 0;
params.factor = 1.0e7;

% 设置PML参数
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
params.IT_DISPLAY = 100;
params.t0 = 1.20/params.f0;
params.ANGLE_FORCE = 0;

% 添加炮点相关参数（只需要设置初始参数）
params.ISOURCE = 200;        %需要传递，但是会被覆盖
params.JSOURCE = 200;        %需要传递，但是会被覆盖    
params.first_shot_i = 100;   % 首炮x位置
params.first_shot_j = 200;   % 首炮y位置
params.shot_di = 100;        % 炮点x间隔
params.shot_dj = 0;          % 炮点y间隔
params.NSHOT = 2;            % 炮点数量

% 设置波场快照参数
params.save_snapshots = true;

% 当前炮号（用于测试）
ishot = 2;  

% 计算模式：'cpu' or 'cpu_mex' or 'cuda_mex'
params.compute_kernel = 'cpu_mex';  
%params.compute_kernel = 'cpu'; 

fprintf('\n=== 创建VTI_Forward实例 ===\n');
forward_solver = VTI_SingleShotModeling(params);

% 计算稳定性条件
fprintf('\n=== 检查稳定性条件 ===\n');
quasi_cp_max = max(max(max(sqrt(params.c33./params.rho))), ...
                  max(max(sqrt(params.c11./params.rho))));
Courant_number = quasi_cp_max * params.DELTAT * ...
                 sqrt(1.0/params.DELTAX^2 + 1.0/params.DELTAY^2);
fprintf('Courant数为 %f\n', Courant_number);
if Courant_number > 1.0
    error('时间步长过大，模拟将不稳定');
end

% 执行单炮正演模拟
fprintf('\n=== 开始第 %d 炮正演模拟 ===\n', ishot);
tic;
[vx_data, vy_data, complete_wavefield] = forward_solver.forward_modeling_single_shot(ishot);
total_time = toc;

fprintf('\n=== 模拟完成 ===\n');
fprintf('总计算时间: %.2f 秒\n', total_time);

% 测试get_complete_wavefield函数
fprintf('\n=== 测试get_complete_wavefield函数 ===\n');
fprintf('测试从内存获取波场...\n');

% 检查stored_wavefield是否已保存
fprintf('stored_wavefield状态: %s\n', ...
    conditional(isempty(forward_solver.stored_wavefield), '空', '非空'));
fprintf('stored_shot_no: %d\n', forward_solver.stored_shot_no);

% 尝试获取波场
retrieved_wavefield = forward_solver.get_complete_wavefield(ishot);

% 验证获取的波场
if ~isempty(retrieved_wavefield)
    fprintf('成功获取波场\n');
    fprintf('波场维度: vx[%d,%d,%d], vy[%d,%d,%d]\n', ...
        size(retrieved_wavefield.vx,1), size(retrieved_wavefield.vx,2), size(retrieved_wavefield.vx,3), ...
        size(retrieved_wavefield.vy,1), size(retrieved_wavefield.vy,2), size(retrieved_wavefield.vy,3));
    
    % 验证波场数据是否一致
    if isequal(complete_wavefield, retrieved_wavefield)
        fprintf('波场数据验证成功：存储的波场与原始波场完全匹配\n');
    else
        warning('波场数据不匹配！');
    end
else
    warning('无法从内存获取波场！');
end

% 辅助函数
function str = conditional(condition, true_str, false_str)
    if condition
        str = true_str;
    else
        str = false_str;
    end
end

