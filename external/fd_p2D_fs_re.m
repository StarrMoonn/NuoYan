%% fd_p2D_fs_re -二维声波方程反传函数（带自由表面边界条件）
%
% 输入参数:
%   rec_p  - 检波器记录的地震数据
%   nx, nz - x和z方向的网格点数
%   pml    - PML吸收边界层厚度
%   dz, dx - z和x方向的网格间距
%   dt     - 时间步长
%   Nt     - 总时间步数
%   v_mod  - 速度模型
%   sx     - 震源x坐标
%   ishot  - 炮号
%   pf     - 正演波场（用于互相关）
%   rx     - 检波器x坐标
%   water  - 水层深度
%
% 输出参数:
%   bp     - 反传波场与正演波场的互相关结果（梯度）
%
% 功能说明:
%   1. 实现了声波方程的反向传播
%   2. 包含自由表面边界条件处理
%   3. 使用PML吸收边界条件
%   4. 使用八阶空间差分
%   5. 用于全波形反演(FWI)的梯度计算
%

function [bp]=fd_p2D_fs_re(rec_p,nx,nz,pml,dz,dx,dt,Nt,v_mod,sx,ishot,pf,rx,water)

% 设置计算网格参数
Nz=nz;
Nx=nx;
N=Nt;
SP=pml;  % PML层厚度

% 设置差分算子阶数和扩展边界
or=4;    % 8阶差分需要4个点
spd=SP+or;

% 扩展速度模型并计算PML参数
vel=extender21(v_mod,or+SP,or+SP);  % 扩展速度模型包含PML和差分算子区域
vmaxt=max(max(vel));
[ axt,azt ] = axx( Nx,Nz,SP,SP,dx,dz,vmaxt);  % 计算PML衰减系数

% 计算扩展后的总网格点数
NxT=Nx+2*SP+2*or;  % 总网格 = 实际网格 + 2*PML + 2*差分算子
NzT=Nz+2*SP+2*or;

% 定义计算区域（包含PML，但不含差分算子边界）
xMinWP=or+1;        % 从差分算子边界后开始
xMaxWP=NxT-or;      % 到差分算子边界前结束
zMinWP=or+1;        % 从差分算子边界后开始
zMaxWP=NzT-or;      % 到差分算子边界前结束

% 计算检波器在扩展网格中的位置
rxi=rx(ishot,:)+or+SP;  % 检波器x坐标
rzi=water+1+or+SP;      % 检波器z坐标（水层之下）

% 计算8阶空间差分系数（z方向）
Z=[-4 -3 -2 -1 0 1 2 3 4]*dz;
c=taylor_vz(Z);
a0z=c(5);a1z=c(6);a2z=c(7);a3z=c(8);a4z=c(9);

% 计算8阶空间差分系数（x方向）
X=[-4 -3 -2 -1 0 1 2 3 4]*dx;
c=taylor_vz(X);
a0x=c(5);a1x=c(6);a2x=c(7);a3x=c(8);a4x=c(9);

% wavefields
cur=zeros(NzT,NxT);   % 当前时刻波场
prev=zeros(NzT,NxT);  % 上一时刻波场

bp=zeros(Nz,Nx);      % 初始化互相关结果（梯度）

% 时间反向循环
for ii=N:-1:1
    % 在检波器位置注入残差
    prev(rzi,rxi)=prev(rzi,rxi)+rec_p(ii,rx(ishot,:));
    
    % 定义计算区域并计算拉普拉斯算子
    z=zMinWP:zMaxWP;
    x=xMinWP:xMaxWP;
    lap=a0z*prev(z,x)                 + a0x*prev(z,x)+ ... 
        a1z*(prev(z-1,x)+prev(z+1,x)) + a1x*(prev(z,x-1)+prev(z,x+1))+...
        a2z*(prev(z-2,x)+prev(z+2,x)) + a2x*(prev(z,x-2)+prev(z,x+2))+...
        a3z*(prev(z-3,x)+prev(z+3,x)) + a3x*(prev(z,x-3)+prev(z,x+3))+...
        a4z*(prev(z-4,x)+prev(z+4,x)) + a4x*(prev(z,x-4)+prev(z,x+4));
    
    % 波场更新（带PML边界条件）
    cur(z,x)=(2*prev(z,x)-cur(z,x).*(1-0.5*dt*(axt+azt))+dt^2*vel(z,x).^2.*lap)./(1+0.5*dt*(axt+azt));
    
    % 自由表面边界处理
    cur(or+1:or+SP,or+SP+1:NxT-or-SP)=(-1)*flipud(cur(spd+2:spd+SP+1,or+SP+1:NxT-or-SP));
    cur(1,or+SP+1:NxT-or-SP)=0; 
    
    % 计算互相关（与正演波场相乘并累加）
    bp=extender21(cur,or+SP,or+SP,1).*pf(:,:,ii)+bp;
    
    % 交换指针，准备下一时间步
    tmp=prev;
    prev=cur;
    cur=tmp;
end

end