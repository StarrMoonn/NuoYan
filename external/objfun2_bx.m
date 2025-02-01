%% objfun2_bx - FWI目标函数和梯度计算（带自由表面边界条件）
%
% 功能说明：
%   计算全波形反演的目标函数值和梯度，考虑自由表面边界条件
%
% 输入参数：
%   v0       - 当前速度模型 [nz x nx]
%   nz,nx    - 模型网格点数（深度和水平方向）
%   pml      - PML边界层厚度
%   dz,dx    - 空间采样间隔
%   dt       - 时间采样间隔
%   Nt       - 时间采样点数
%   nshot    - 炮数
%   seisture - 观测数据 [时间采样点数 x 检波器数 x 炮数]
%   wav      - 震源子波 [时间采样点数 x 炮数]
%   sx,sz    - 震源位置坐标
%   rx       - 检波器水平位置
%   water    - 水层深度（用于自由表面处理）
%
% 输出参数：
%   obj      - 目标函数值（所有炮的目标函数和）
%   grad_stk1 - 叠加后的梯度
%
% 工作流程：
%   1. 对每个炮进行并行计算：
%   [pf,seisv0] = fd_p2D_fs()      % 正演模拟
%   seis_resid = cross_resid()     % 计算残差（用于反传）
%   obj1 = objfun1()               % 计算目标函数值
%   [pb] = fd_p2D_fs_re()          % 反传计算伴随波场
%   img1 = (2./v0.^3).*pb          % 计算梯度
%
%   2. 叠加所有炮的梯度
%   3. 求和得到总目标函数值

function [obj,grad_stk1]=objfun2_bx(v0,nz,nx,pml,dz,dx,dt,Nt,nshot,seisture,wav,sx,sz,rx,water)

obj1=zeros(nshot,1);
img1=zeros(nz,nx,nshot);

parfor ishot=1:nshot

        [pf,seisv0]=fd_p2D_fs(nx,nz,pml,dz,dx,dt,Nt,v0,wav(:,ishot),sx,sz,ishot,rx,water);

        seis_resid=(-1)*cross_resid(seisv0,seisture(:,:,ishot));

        obj1(ishot)=objfun1(seisv0,seisture(:,:,ishot));

        [pb]=fd_p2D_fs_re(seis_resid,nx,nz,pml,dz,dx,dt,Nt,v0,sx,ishot,pf,rx,water);
        
        img1(:,:,ishot)=(2./v0.^3).*pb;

        
end

grad_stk1=sum(img1,3);
obj=sum(obj1);
end
