%% axx - PML完美匹配层衰减系数计算函数
%
% 功能说明：
%   计算声波方程有限差分模拟中PML边界的衰减系数。
%   用于消除计算区域边界的人工反射，模拟无限大空间。
%
% 输入参数：
%   nx    - 主计算区域x方向网格点数
%   nz    - 主计算区域z方向网格点数
%   lxPML - x方向PML层厚度（网格点数）
%   lzPML - z方向PML层厚度（网格点数）
%   dx    - x方向空间采样间隔
%   dz    - z方向空间采样间隔
%   vmax  - 模型中的最大速度值
%
% 输出参数：
%   ax    - x方向PML衰减系数矩阵 [nz+2*lzPML x nx+2*lxPML]
%   az    - z方向PML衰减系数矩阵 [nz+2*lzPML x nx+2*lxPML]
%
% 计算区域示意图：
%   ┌───────────────────────┐
%   │        上PML         │
%   ├────┬──────────┬─────┤
%   │    │          │     │
%   │    │          │     │
%   │左  │   计算   │  右 │
%   │PML │   区域   │ PML │
%   │    │          │     │
%   │    │          │     │
%   ├────┴──────────┴─────┤
%   │        下PML         │
%   └───────────────────────┘
%
% 注意事项：
%   1. PML层的厚度需要合理设置，通常10-20个网格点
%   2. 理论反射系数R设为1e-5，可根据需要调整
%   3. 衰减函数采用二次函数形式
%
% 使用示例：
%   [ax,az] = axx(100,100,10,10,5,5,3000);
%
% 参考文献：
%   - Berenger, J.P., 1994, A perfectly matched layer for the absorption of 
%     electromagnetic waves: Journal of Computational Physics
%
% 相关函数：
%   - fd_p2D_fs (有限差分正演函数)
%   - fd_p2D_fs_re_wav (有限差分伴随函数)
%

function [ ax,az ] = axx( nx,nz,lxPML,lzPML,dx,dz,vmax)
% PML衰减系数初始化
ax=zeros(nz+2*lzPML,nx+2*lxPML);
az=zeros(nz+2*lzPML,nx+2*lxPML);
R=1e-5;  % 理论反射系数

% 计算x方向PML衰减系数
% 左边界PML区域
for i=1:1:nz+2*lzPML
    for j=1:1:lxPML
        ax(i,j)=-log(R)*3*vmax*(lxPML-j)^2/(2*(dx*lxPML)^2);
    end
end
% 右边界PML区域
for i=1:1:nz+2*lzPML
    for j=nx+lxPML+1:1:nx+2*lxPML
        ax(i,j)=-log(R)*3*vmax*(j-nx-lxPML-1)^2/(2*(dx*lxPML)^2);
    end
end

% 计算z方向PML衰减系数
% 上边界PML区域
for j=1:1:nx+2*lxPML
    for i=1:1:lzPML
        az(i,j)=-log(R)*3*vmax*(lzPML-i)^2/(2*(dz*lzPML)^2);
    end
end
% 下边界PML区域
for j=1:1:nx+2*lxPML
    for i=nz+lzPML+1:1:nz+2*lzPML
        az(i,j)=-log(R)*3*vmax*(i-nz-lzPML-1)^2/(2*(dz*lzPML)^2);
    end
end
end