%% fd_p2D_fs - 二维声波方程有限差分正演（带自由表面边界条件）
%
% 功能说明：
%   使用高阶有限差分方法求解2D声波方程，考虑自由表面边界条件
%   采用交错网格八阶精度空间差分和二阶精度时间差分
%
% 数学原理：
%   声波方程：∂²p/∂t² = v²(∂²p/∂x² + ∂²p/∂z²)
%   差分格式：
%   1. 时间二阶差分：(p(t+dt)-2p(t)+p(t-dt))/dt²
%   2. 空间八阶差分：使用taylor_vz计算的差分系数
%   3. PML吸收边界条件
%
% 输入参数：
%   nx,nz    - 模型网格点数
%   pml      - PML层厚度
%   dz,dx,dt - 空间和时间采样间隔
%   Nt       - 时间采样点数
%   v_mod    - 速度模型
%   wav      - 震源子波
%   sx,sz    - 震源位置
%   rx       - 检波器位置
%   water    - 水层深度
%
% 输出参数：
%   realp    - 波场快照 [nz x nx x Nt]
%   realrec_p- 地震记录 [Nt x nx]

function [realp,realrec_p]=fd_p2D_fs(nx,nz,pml,dz,dx,dt,Nt,v_mod,wav,sx,sz,ishot,rx,water)

% 初始化网格参数
Nz=nz;
Nx=nx;
N=Nt;
SP=pml;        % PML层厚度

or=4;          % 差分算子半宽度（8阶差分）
spd=SP+or;     % PML层+差分算子宽度

% 扩展速度模型（添加PML边界）
vel=extender21(v_mod,or+SP,or+SP);
vmaxt=max(max(vel));
% 计算PML衰减系数
[ axt,azt ] = axx( Nx,Nz,SP,SP,dx,dz,vmaxt);

% 计算扩展后的总网格点数
NxT=Nx+2*SP+2*or;
NzT=Nz+2*SP+2*or;

% 定义计算区域（包含PML，但不含差分算子边界）
xMinWP=or+1;        % 从差分算子边界后开始
xMaxWP=NxT-or;      % 到差分算子边界前结束
zMinWP=or+1;        % 从差分算子边界后开始
zMaxWP=NzT-or;      % 到差分算子边界前结束

% 计算震源和检波器在扩展网格中的位置
szi=sz+or+SP;
rxi=rx(ishot,:)'+or+SP;
rzi=water+1+or+SP;

% 计算z方向的差分系数（8阶精度）
Z=[-4 -3 -2 -1 0 1 2 3 4]*dz;
c=taylor_vz(Z);
a0z=c(5);a1z=c(6);a2z=c(7);a3z=c(8);a4z=c(9);

% 计算x方向的差分系数（8阶精度）
X=[-4 -3 -2 -1 0 1 2 3 4]*dx;
c=taylor_vz(X);
a0x=c(5);a1x=c(6);a2x=c(7);a3x=c(8);a4x=c(9);

% 初始化波场
cur=zeros(NzT,NxT);    % 当前时刻波场
prev=zeros(NzT,NxT);   % 上一时刻波场

% 初始化输出数组
realp=zeros(Nz,Nx,N);      % 波场快照
realrec_p=zeros(N,Nx);     % 地震记录

% 时间循环
for ii=1:N
    % 添加震源
    prev(szi(ishot),spd+sx(ishot))=prev(szi(ishot),spd+sx(ishot))+wav(ii);
    
    % 定义计算区域
    z=zMinWP:zMaxWP;
    x=xMinWP:xMaxWP;
    % 计算空间二阶导数（拉普拉斯算子）
    lap=a0z*prev(z,x)                 + a0x*prev(z,x)+ ... 
        a1z*(prev(z-1,x)+prev(z+1,x)) + a1x*(prev(z,x-1)+prev(z,x+1))+...
        a2z*(prev(z-2,x)+prev(z+2,x)) + a2x*(prev(z,x-2)+prev(z,x+2))+...
        a3z*(prev(z-3,x)+prev(z+3,x)) + a3x*(prev(z,x-3)+prev(z,x+3))+...
        a4z*(prev(z-4,x)+prev(z+4,x)) + a4x*(prev(z,x-4)+prev(z,x+4));
    
    % 波场更新（带PML吸收边界）    
    cur(z,x)=(2*prev(z,x)-cur(z,x).*(1-0.5*dt*(axt+azt))+dt^2*vel(z,x).^2.*lap)./(1+0.5*dt*(axt+azt));
    % 自由表面边界处理
    cur(or+1:or+SP,or+SP+1:NxT-or-SP)=(-1)*flipud(cur(spd+2:spd+SP+1,or+SP+1:NxT-or-SP));
    cur(1,or+SP+1:NxT-or-SP)=0;     
            
    % 记录检波器位置的波场
    realrec_p(ii,rx(ishot,:)')= prev(rzi,rxi);   
    % 保存波场快照（去除PML区域）
    realp(:,:,ii)=extender21(cur,or+SP,or+SP,1);
    
    % 波场更新
    tmp=prev;
    prev=cur;
    cur=tmp;
end
end