%% VTI模型参数生成脚本
% 功能：生成简单两层VTI介质模型的弹性参数和密度数据
% 
% 说明：
%   1. 生成两种模型：带侵入体(case1)和不带侵入体(case2)的二层模型
%   2. 计算并保存弹性参数(c11,c13,c33,c44)和密度(rho)
%   3. case2包含：
%      - 第一层：Vp=2.5km/s, Vs=1.5km/s, ρ=1.0g/cc, δ=0.15, ε=0.25
%      - 第二层：Vp=4.0km/s, Vs=2.0km/s, ρ=2.0g/cc, δ=0.075, ε=0.1
%      - 侵入体：Vp=3.5km/s, Vs=1.75km/s, ρ=1.75g/cc, δ=0.1, ε=0.15
%   4. case3仅包含两层结构，参数同上
%
% 输入：
%   - 无需外部输入，参数在脚本中直接定义
%
% 输出：
%   - case1和case2两个文件夹，每个包含：
%     * c11.mat - VTI介质的c11弹性参数
%     * c13.mat - VTI介质的c13弹性参数
%     * c33.mat - VTI介质的c33弹性参数
%     * c44.mat - VTI介质的c44弹性参数
%     * rho.mat - 密度参数
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

% 定义网格参数
nz = 201;
nx = 801;

% 初始化数组
c11 = zeros(nx,nz);
c13 = zeros(nx,nz);
c33 = zeros(nx,nz);
c44 = zeros(nx,nz);
rho = zeros(nx,nz);

% 第一层 (1:100)
for ix = 1:nx
    for iz = 1:100
        % 基本参数
        vp = 2.5 * 1000;        % 2.5 km/s -> m/s
        vs = 1.5 * 1000;        % 1.5 km/s -> m/s
        density = 1.0 * 1000;   % 1.0 g/cc -> kg/m³
        delta = 0.15;
        epsilon = 0.25;
        
        % 计算弹性常数 (Pa)
        c33(ix,iz) = density * vp^2;
        c44(ix,iz) = density * vs^2;
        c11(ix,iz) = c33(ix,iz) * (1 + 2*epsilon);
        c13(ix,iz) = sqrt(2*delta*c33(ix,iz)*(c33(ix,iz)-c44(ix,iz)) + (c33(ix,iz)-c44(ix,iz))^2) - c44(ix,iz);
        rho(ix,iz) = density;
    end
end

% 第二层 (101:end)
for ix = 1:nx
    for iz = 101:nz
        % 基本参数
        vp = 4.0 * 1000;        % 4.0 km/s -> m/s
        vs = 2.0 * 1000;        % 2.0 km/s -> m/s
        density = 2.0 * 1000;   % 2.0 g/cc -> kg/m³
        delta = 0.075;
        epsilon = 0.1;
        
        % 计算弹性常数 (Pa)
        c33(ix,iz) = density * vp^2;
        c44(ix,iz) = density * vs^2;
        c11(ix,iz) = c33(ix,iz) * (1 + 2*epsilon);
        c13(ix,iz) = sqrt(2*delta*c33(ix,iz)*(c33(ix,iz)-c44(ix,iz)) + (c33(ix,iz)-c44(ix,iz))^2) - c44(ix,iz);
        rho(ix,iz) = density;
    end
end

% 添加侵入体 (375:426, 70:80)
for ix = 375:426
    for iz = 70:80
        % 基本参数
        vp = 3.5 * 1000;        % 3.5 km/s -> m/s
        vs = 1.75 * 1000;       % 1.75 km/s -> m/s
        density = 1.75 * 1000;  % 1.75 g/cc -> kg/m³
        delta = 0.1;
        epsilon = 0.15;
        
        % 计算弹性常数 (Pa)
        c33(ix,iz) = density * vp^2;
        c44(ix,iz) = density * vs^2;
        c11(ix,iz) = c33(ix,iz) * (1 + 2*epsilon);
        c13(ix,iz) = sqrt(2*delta*c33(ix,iz)*(c33(ix,iz)-c44(ix,iz)) + (c33(ix,iz)-c44(ix,iz))^2) - c44(ix,iz);
        rho(ix,iz) = density;
    end
end

% 保存case2（带侵入体）
save_path = './data/model/case2';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
save(fullfile(save_path, 'c11.mat'), 'c11');
save(fullfile(save_path, 'c13.mat'), 'c13');
save(fullfile(save_path, 'c33.mat'), 'c33');
save(fullfile(save_path, 'c44.mat'), 'c44');
save(fullfile(save_path, 'rho.mat'), 'rho');

% 重新计算case3（不带侵入体）
% 重置数组
c11 = zeros(nx,nz);
c13 = zeros(nx,nz);
c33 = zeros(nx,nz);
c44 = zeros(nx,nz);
rho = zeros(nx,nz);

% 第一层 (1:100)
for ix = 1:nx
    for iz = 1:100
        % 基本参数
        vp = 2.5 * 1000;
        vs = 1.5 * 1000;
        density = 1.0 * 1000;
        delta = 0.15;
        epsilon = 0.25;
        
        % 计算弹性常数
        c33(ix,iz) = density * vp^2;
        c44(ix,iz) = density * vs^2;
        c11(ix,iz) = c33(ix,iz) * (1 + 2*epsilon);
        c13(ix,iz) = sqrt(2*delta*c33(ix,iz)*(c33(ix,iz)-c44(ix,iz)) + (c33(ix,iz)-c44(ix,iz))^2) - c44(ix,iz);
        rho(ix,iz) = density;
    end
end

% 第二层 (101:end)
for ix = 1:nx
    for iz = 101:nz
        % 基本参数
        vp = 4.0 * 1000;
        vs = 2.0 * 1000;
        density = 2.0 * 1000;
        delta = 0.075;
        epsilon = 0.1;
        
        % 计算弹性常数
        c33(ix,iz) = density * vp^2;
        c44(ix,iz) = density * vs^2;
        c11(ix,iz) = c33(ix,iz) * (1 + 2*epsilon);
        c13(ix,iz) = sqrt(2*delta*c33(ix,iz)*(c33(ix,iz)-c44(ix,iz)) + (c33(ix,iz)-c44(ix,iz))^2) - c44(ix,iz);
        rho(ix,iz) = density;
    end
end

% 保存case3（不带侵入体）
save_path = './data/model/case3';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
save(fullfile(save_path, 'c11.mat'), 'c11');
save(fullfile(save_path, 'c13.mat'), 'c13');
save(fullfile(save_path, 'c33.mat'), 'c33');
save(fullfile(save_path, 'c44.mat'), 'c44');
save(fullfile(save_path, 'rho.mat'), 'rho');

% 为case4复制case3的数据
c11_case4 = c11;
c13_case4 = c13;
c33_case4 = c33;
c44_case4 = c44;
rho_case4 = rho;

% 修改高斯滤波器参数
sigma = 8;  % 增大标准差，使模糊更明显
kernel_size = 31;  % 增大核大小，扩大影响范围

% 创建高斯核
gaussian_kernel = fspecial('gaussian', [kernel_size kernel_size], sigma);

% 对所有参数进行高斯模糊处理
c11_case4 = imfilter(c11_case4, gaussian_kernel, 'replicate');
c13_case4 = imfilter(c13_case4, gaussian_kernel, 'replicate');
c33_case4 = imfilter(c33_case4, gaussian_kernel, 'replicate');
c44_case4 = imfilter(c44_case4, gaussian_kernel, 'replicate');
rho_case4 = imfilter(rho_case4, gaussian_kernel, 'replicate');

% 添加小幅随机扰动（可选）
noise_level = 0.02; % 2%的随机扰动
for field = {c11_case4, c13_case4, c33_case4, c44_case4, rho_case4}
    data = field{1};
    noise = noise_level * (rand(size(data))-0.5) .* data;
    field{1} = data + noise;
end

% 确保物理参数的合理性（可选）
% 例如确保c11 > c13等

% 保存case4（高斯模糊版本）
save_path = './data/model/case4';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
save(fullfile(save_path, 'c11.mat'), 'c11_case4');
save(fullfile(save_path, 'c13.mat'), 'c13_case4');
save(fullfile(save_path, 'c33.mat'), 'c33_case4');
save(fullfile(save_path, 'c44.mat'), 'c44_case4');
save(fullfile(save_path, 'rho.mat'), 'rho_case4');


