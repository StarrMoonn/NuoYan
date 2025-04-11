%% 1. 初始化和模型参数设置
clear;clc;
load vel.txt
v03=vel';
v0=v03(:,215:end);
v0(:,645)=v0(:,644);
v0(1:3,:)=1520;

% 设置速度约束
water=0;
vmin=1520;
vmax=5000;
v0=control_v(v0,water,vmin,vmax);

% 设置网格参数
nz=167;
nx=645;
pml=100;
tnz=nz+2*pml;
tnx=nx+2*pml;
dx=10;
dz=10;
kwavmax=20;

nshot=1;
Tmax=5;
dt = 0.002;
Nt=round(Tmax/dt);

sz=2*ones(nshot,1);
load sx_x15-33.mat
sx=sx1;%sx=96;
load rx2-25.mat
rx=rx2;
 
% [bh,ah]=butter(8,100*dt*2,'low');  
% [bh2,ah2]=butter(6,1*dt*2,'high');  

% 初始化观测数据数组 [时间采样点数, 检波器数量, 炮数]
seisture = zeros(Nt, nx, 1);

%%% 1. 读取实际地震数据 %%%
% 切换到数据所在目录
%%%cd /public/home/zhangpan/dipi/4_dipi_realdata/dongfang11-OBN/originaldata
cd D:/实验/fwi声波/海油/shot

% 读取SEG-Y格式的地震数据
for ishot = 1  %ishot=1:323  % 当前只处理第一炮
    % 读取SEG-Y文件
    seis = altreadsegy([num2str(ishot) '.sgy']);
    % 滤波处理（已注释）
    % seis=filter(bh2,ah2,seis);  % 高通滤波
    % seis=filter(bh,ah,seis);    % 低通滤波
    
    % 截取所需时间长度的数据
    seis = seis(1:Nt,:);
    
    % 将数据放入观测数据数组，注意检波器位置映射
    seisture(:,rx(ishot,:)',ishot) = seis;
    
    % 显示地震记录
    figure, imagesc(seisture);
    colormap("gray");
    clim([-2 2]);  % 限制显示范围
    [maxValue, maxIndex] = max(seisture(1, :));  % 找到第一个时间采样的最大值位置
end

% 返回工作目录
cd D:/实验/fwi声波
% cd /public/home/zhangpan/dipi/4_dipi_realdata/dongfang11-OBN/fwi-grid25/wav3-low4/no_control/wavstep2/cutnoise/wav3/0watercorr/tomo2/control1520/tomo-vlog/env/fwi

%%% 2. 数据预处理 %%%
% 截断数据（可能是消除直达波或其他噪声）
for ishot = 1:nshot
    seisture(:,:,ishot) = cutdata3(seisture, sx, ishot);
end

% 数据归一化处理
for ishot = 1:nshot
    seis = seisture(:,:,ishot);
    seisture(:,:,ishot) = seis/max(max(abs(seis)));  % 按炮集最大振幅归一化
end

%%% 3. 子波估计 %%%
% 使用前1000个时间采样点和浅层速度模型进行子波估计
seisture2 = seisture(1:1000,:,:);  % 截取前1000ms数据
v02 = v0(1:50,:);                  % 使用浅层速度模型（前50层）

% 对每炮估计子波
for ishot = 1:nshot
    % 子波估计函数，使用伴随法
    [wav, objwav] = adjwavest(v02, 50, nx, dx, dz, ishot, dt, 1000, ...
                             zeros(1000,1), sz, sx, seisture2(:,:,ishot), ...
                             pml, rx, kwavmax);
    % 显示估计的子波
    figure, plot(wav);
    % 保存子波
    wav2(:,ishot) = wav;
    % objwav2(:,ishot)=objwav;  % 子波目标函数值（已注释）
end

% 确保子波在最后时刻为0
wav2(Nt,:) = 0;

% 保存子波和目标函数值
save wav2-1000ms wav2       % 保存估计的子波
save objwav2 objwav2       % 保存子波估计的目标函数值

%% 4. 初始梯度计算
for ishot = 1:nshot
    % 1. 正演模拟
    [pf, seisv0] = fd_p2D_fs(nx, nz, pml, dz, dx, dt, Nt, v0, wav2(:,ishot), ...
                            sx, sz, ishot, rx, water);
    
    % 显示正演记录
    figure, imagesc(seisv0);
    % colormap("jet");
    % clim([-1,1]);
    colorbar;
    % colormap('redwhiteblue');
    
    % 2. 计算残差
    seis_resid = (-1)*cross_resid(seisv0, seisture(:,:,ishot));
    
    % 3. 计算目标函数
    % obj1=objfun1(seisv0,seisture(:,:,ishot),nx,sx,ishot);
    obj1 = objfun1(seisv0, seisture(:,:,ishot));
    obj0(ishot) = obj1;
    
    % 4. 反传计算梯度
    [pb] = fd_p2D_fs_re(seis_resid, nx, nz, pml, dz, dx, dt, Nt, v0, ...
                        sx, ishot, pf, rx, water);
    
    % 5. 计算每炮的梯度（考虑速度的三次方）
    img1(:,:,ishot) = (2./v0.^3).*pb;
end

% 6. 叠加所有炮集的梯度
grad_stk1 = sum(img1, 3);
objval0 = sum(obj0);

% 7. 水层梯度置零
grad_stk1(1:water,:) = 0;

% 保存梯度
save grad_stk1

%% 5. 共轭梯度迭代优化
%%% 初始化迭代参数 %%%
k = 1;                    % 当前迭代次数
kmax = 500;               % 最大迭代次数
objval = zeros(kmax,1);   % 存储每次迭代的目标函数值

% 显示当前速度模型和梯度
figure(1); imagesc(v0); pause(0.001);        % 显示速度模型
figure(2); imagesc(grad_stk1); pause(0.001); % 显示梯度
k
disp('%%%%%%%%%%%%%%%%%%%%iteration time%%%%%%%%%%%%%%%%%%%%%%')

%%% 第一次迭代的梯度计算 %%%
objval(k+1) = objval0;   % 记录初始目标函数值

% 计算归一化的梯度方向
p = 100*grad_stk1/norm(grad_stk1);   % 归一化并放大梯度
d1 = -2*p;                           % 最速下降方向（负梯度方向）

%%% 步长控制 %%%
maxd = max(max(abs(d1)));            % 计算最大更新量
dec = 0.5;                           % 步长衰减因子
ks = 1;                              % 线搜索计数器

% 初始步长
a = 1;
% 控制最大更新步长不超过30
while maxd > 30
    a = dec*a;                       % 如果更新太大，减小步长
    maxd = max(max(abs(a*d1)));      % 重新计算最大更新量
end
a                                    % 显示最终步长
maxd                                 % 显示最大更新量

%%% 线搜索找最优步长 %%%
while (ks < 10)                      % 最多尝试10次
    % 更新速度模型
    v0s1 = v0 + a*d1;               % 试探性更新
    v0s1 = control_v(v0s1,water,vmin,vmax);  % 应用速度约束
    
    % 计算新模型的目标函数值和梯度
    [objval1,grad_stk2] = objfun2_bx(v0s1,nz,nx,pml,dz,dx,dt,Nt,nshot,...
                                    seisture,wav2,sx,sz,rx,water);
    
    % 显示当前搜索信息
    ks                              % 显示当前搜索次数
    objval0                         % 显示旧目标函数值
    objval1                         % 显示新目标函数值
    
    % 判断是否接受更新
    if objval1 < objval0            % 如果目标函数值减小
        v0 = v0s1;                  % 接受新模型
        objold = objval0;           % 保存旧的目标函数值
        objval0 = objval1;          % 更新目标函数值
        grad_stk1 = grad_stk2;      % 更新梯度
        p0 = p;                     % 保存旧的梯度方向
        d0 = a*d1;                  % 保存搜索方向
        break;
    else
        a = dec*a;                  % 如果目标函数值增加，减小步长
        a2 = a;                     % 保存当前步长供下次迭代使用
        ks = ks + 1;                % 增加搜索次数
    end
end

%%% 保存结果 %%%
% 保存当前迭代的速度模型和梯度
save(['k=' num2str(k) '_v0.mat'],'v0');
save(['k=' num2str(k) '_grad_stk1.mat'],'grad_stk1');

% 水层梯度置零
grad_stk1(1:water,:) = 0;
% grad_stk1=grad_stk1.*water;  % 另一种水层处理方式（已注释）

% 更新迭代次数
k = k + 1;

%%% 共轭梯度法迭代（k>1） %%%
while(k < kmax)
    % 显示当前迭代次数
    k
    % 显示当前速度模型和梯度
    figure(1); imagesc(v0); pause(0.0001);
    figure(2); imagesc(grad_stk1); pause(0.0001);
    disp('%%%%%%%%%%%%%%%%%%%%iteration time%%%%%%%%%%%%%%%%%%%%%%')
    
    % 记录目标函数值
    objval(k+1) = objval0;
    
    %%% 计算共轭梯度方向 %%%
    p = 100*grad_stk1/norm(grad_stk1);    % 归一化当前梯度
    % 计算共轭因子beta（Fletcher-Reeves公式）
    b = sqrt(sum(sum((p'*p).^2))/sum(sum((p0'*p0).^2)));
    % 计算共轭方向：当前负梯度方向 + beta*前一次方向
    d1 = -p + b*d0;

    %%% 步长控制 %%%
    maxd = max(max(abs(d1)));             % 计算最大更新量
    dec = 0.5;                            % 步长衰减因子

    % 初始步长设置和控制
    a = 1;
    while maxd > 30                       % 控制最大更新不超过30
        a = dec*a;                        % 减小步长
        maxd = max(max(abs(a*d1)));       % 重新计算最大更新量
    end
    
    % 如果上一次线搜索失败，使用上一次的步长
    if ks > 1
       a = a2; 
    end
    % 显示当前步长和最大更新量
    a
    maxd
    
    %%% 线搜索找最优步长 %%%
    ks = 1;                               % 线搜索计数器
    while (ks < 10)                       % 最多尝试10次
        % 更新速度模型
        v0s1 = v0 + a*d1;                % 试探性更新
        v0s1 = control_v(v0s1,water,vmin,vmax);  % 应用速度约束

        % 计算新模型的目标函数值和梯度
        [objval1,grad_stk2] = objfun2_bx(v0s1,nz,nx,pml,dz,dx,dt,Nt,nshot,...
                                        seisture,wav2,sx,sz,rx,water);
        % 显示当前搜索信息
        ks
        objval0
        objval1
        
        % 判断是否接受更新
        if objval1 < objval0              % 如果目标函数值减小
            v0 = v0s1;                    % 接受新模型
            objold = objval0;             % 保存旧的目标函数值
            objval0 = objval1;            % 更新目标函数值
            grad_stk1 = grad_stk2;        % 更新梯度
            p0 = p;                       % 保存旧的梯度方向
            d0 = a*d1;                    % 保存搜索方向
            break;
        else
            a = dec*a;                    % 如果目标函数值增加，减小步长
            a2 = a;                       % 保存当前步长
            ks = ks + 1;                  % 增加搜索次数
        end
    end

    % 水层梯度置零
    grad_stk1(1:water,:) = 0;
    % grad_stk1=grad_stk1.*water;        % 另一种水层处理方式（已注释）
    
    % 每5次迭代保存一次结果
    if mod(k,5) == 0
        save(['k=' num2str(k) '_v0.mat'],'v0');
        save(['k=' num2str(k) '_grad_stk1.mat'],'grad_stk1');
    end
    
    % 保存所有目标函数值
    save objval objval

    % 如果线搜索失败太多次，终止迭代
    if (ks == 20)  %if (ks==10) 
        break;
    end
    
    % 更新迭代次数
    k = k + 1;
end