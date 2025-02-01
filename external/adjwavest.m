%% adjwavest - FWI震源子波反演函数
%
% 功能说明：
%   通过迭代优化方式，从实际地震数据中反演震源子波。
%   使用最速下降法和线搜索策略来优化子波形状。
%
% 输入参数：
%   v0    - 速度场模型 [nz x nx]
%   nz    - 深度方向网格数
%   nx    - 水平方向网格数
%   dx    - 水平方向网格间距
%   dz    - 深度方向网格间距
%   ishot - 炮号
%   dt    - 时间采样间隔
%   Nt    - 时间采样点数
%   wav2  - 初始子波
%   sz    - 震源深度位置
%   sx    - 震源水平位置
%   seis  - 观测地震记录
%   pml   - PML边界层参数
%   rx    - 检波器位置
%   kmax  - 最大迭代次数
%
% 输出参数：
%   wav2  - 优化后的子波
%   obj   - 每次迭代的目标函数值
%
% 使用示例：
%   [wavelet, obj] = adjwavest(v0, nz, nx, dx, dz, ishot, dt, Nt, 
%                              initial_wav, sz, sx, obs_data, pml, rx, 100);
%
% 注意事项：
%   1. 需要合理的初始子波
%   2. 速度场模型的准确性会影响子波提取效果
%   3. 可能需要调整线搜索步长以获得更好的收敛性
%

function [wav2,obj]=adjwavest(v0,nz,nx,dx,dz,ishot,dt,Nt,wav2,sz,sx,seis,pml,rx,kmax)

% 初始化参数
k=1;                  % 迭代计数器
water=0;              % 水层标志
obj=zeros(kmax,1);    % 存储目标函数值

% 主迭代循环
while(k<kmax)
    % 步骤1：使用当前子波进行正演模拟
    [~,seisv0]=fd_p2D_fs(nx,nz,pml,dz,dx,dt,Nt,v0,wav2,sx,sz,ishot,rx,water);
    
    % 步骤2：计算残差和目标函数
    resid=seisv0-seis;                % 计算残差
    obj(k)=sum(sum(resid.^2));        % 计算目标函数值
    
    % 步骤3：计算伴随波场
    [pb]=fd_p2D_fs_re_wav(resid,nx,nz,pml,dz,dx,dt,Nt,v0,ishot,sx,rx,water);
    % 处理可能的NaN值
    nan_indices=isnan(pb);
    pb(nan_indices)=0;
    
    % 步骤4：提取震源位置处的梯度
    img1=squeeze(pb(sz(ishot),sx(ishot),:));
    
    % 步骤5：梯度归一化和子波更新
    p=(1)*img1/max(max(abs(img1)));   % 归一化梯度
    sl=0.01;                          % 初始步长
    wav20=wav2-sl*p;                  % 更新子波
    
    % 步骤6：正演验证更新效果
    [~,seisv0]=fd_p2D_fs(nx,nz,pml,dz,dx,dt,Nt,v0,wav20,sx,sz,ishot,rx,water);
    resid=seisv0-seis;
    obj(k+1)=sum(sum(resid.^2));
    
    % 步骤7：线搜索优化步长
    ii=1;
    while(obj(k+1)>obj(k))           % 如果目标函数没有下降
        sl=sl/2;                      % 减小步长
        wav20=wav2-sl*p;              % 重新更新子波
        % 重新计算目标函数
        [~,seisv0]=fd_p2D_fs(nx,nz,pml,dz,dx,dt,Nt,v0,wav20,sx,sz,ishot,rx,water);
        resid=seisv0-seis;
        obj(k+1)=sum(sum(resid.^2));
        ii=ii+1;
        if ii==10                     % 最多尝试10次线搜索
            break; 
        end
    end
    
    % 如果线搜索失败，终止迭代
    if ii==10
        break; 
    end
    
    % 更新子波并进入下一次迭代
    wav2=wav20;
    k=k+1;
end