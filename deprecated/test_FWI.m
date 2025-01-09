%% 逐炮正演程序测试
% 功能：测试VTI介质中的多炮正演模拟
% 
% 说明：
%   1. 初始化模型参数（网格大小、材料属性等）
%   2. 设置多炮相关参数（炮点间隔、炮数等）
%   3. 创建VTI_FWI实例进行逐炮模拟
%   4. 记录每炮的地震记录和波场快照
%
% 参数设置：
%   - 模型尺寸：401x401网格点
%   - 空间步长：0.0625e-2 m
%   - 时间步长：50e-9 s
%   - 总时间步数：3000
%   - 震源频率：200 kHz
%   - 炮点参数：
%     * 第一炮位置：(100, 200)
%     * 炮点间隔：dx=100, dy=0
%     * 总炮数：2
%
% 输出：
%   - 每炮的地震记录（水平和垂直分量）
%   - 波场快照（可选）
%   - 每炮的计算时间统计
%
% 注意事项：
%   - 每炮开始前会重置波场
%   - 所有炮共用相同的模型参数
%   - 检波器位置相对于每个炮点保持不变
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

clear;
clc;

% 设置项目根目录
project_root = 'E:\Matlab\VTI_project';
cd(project_root);

% 只添加项目根目录和core目录到路径
addpath(project_root);           % 添加根目录

% 验证路径是否正确
if ~exist('VTI_FWI', 'class')
    error('无法找到 VTI_FWI 类，请检查路径设置');
end

fprintf('当前工作目录: %s\n', pwd);
fprintf('已添加的路径:\n');
fprintf('- %s\n', project_root);
fprintf('- %s\n', fullfile(project_root, 'core'));

% 创建参数结构体
params = struct();
params.NX = 401;         
params.NY = 401;
params.DELTAX = 0.0625e-2;
params.DELTAY = 0.0625e-2;

% 设置多炮相关参数
params.first_shot_i = 100;    % 第一炮x位置
params.first_shot_j = 200;    % 第一炮y位置
params.shot_di = 100;         % 炮点x间隔
params.shot_dj = 0;          % 炮点y间隔
params.NSHOT = 2;            % 总炮数

% 需要添加这两个参数用于VTI_FD初始化
params.ISOURCE = params.first_shot_i;  % 初始震源位置，后续会被更新
params.JSOURCE = params.first_shot_j;  % 初始震源位置，后续会被更新

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
params.factor = 1.0e7;
params.save_snapshots = true;

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
params.IT_DISPLAY = 100;    

% 计算模式：'cpu_parallel' or 'gpu_serial'
params.compute_mode = 'cpu_parallel';  
%params.compute_mode = 'gpu_serial';


fprintf('\n=== 创建VTI_FWI实例并初始化 ===\n');
fwi_model = VTI_FWI(params);

% 运行正演模拟
fwi_model.forward_modeling_all_shots();

fprintf('\n=== 测试完成 ===\n');