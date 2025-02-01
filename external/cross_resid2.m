%% cross_resid2 - 基于包络的互相关残差计算函数（增强版）
%
% 功能说明：
%   计算观测数据和模拟数据之间的包络互相关残差，
%   结合了包络分析和互相关特性，用于FWI目标函数计算。
%
% 输入参数：
%   seisv0   - 模拟地震记录 [nt x nx]
%   seisture - 观测地震记录 [nt x nx]
%   ishot    - 炮号
%   sx       - 震源位置
%
% 输出参数：
%   resid    - 残差 [nt x nx]
%
% 计算步骤：
%   1. 抽取偶数道进行计算
%   2. 计算希尔伯特变换获取包络
%   3. 对包络进行归一化
%   4. 计算互相关系数
%   5. 构建残差并滤波
%
% 对每道数据：
% seisv0E = abs(hilbert(seisv0))        1.计算包络
% seisv0E = seisv0E/norm(seisv0E,2)     2.归一化
% seistureE = abs(hilbert(seisture))    3.观测数据同样处理
%
% 选择建议：
% 场景1：数据质量好
% └── 使用 cross_resid.m
% - 信噪比高
% - 波形清晰
% - 计算效率要求高

% 场景2：数据质量一般
% └── 使用 cross_resid2.m
% - 存在噪声
% - 需要更稳健的结果
% - 对计算效率要求不高

function resid=cross_resid2(seisv0,seisture,ishot,sx)
% 初始化输出残差矩阵（与输入数据同样大小）
[nz2,nx2]=size(seisv0);
resid=zeros(nz2,nx2);

% 抽取偶数道进行计算，可能是为了减少计算量或避免相邻道干扰
seisv0=seisv0(:,2:2:end);
seisture=seisture(:,2:2:end);
nx=size(seisv0,2);

% 计算模拟数据包络的整体能量（用于归一化）
a=norm(abs(hilbert(seisv0)),2);

% 对每道数据进行包络计算和归一化
% 只处理非零能量道，避免除零错误
for j=1:nx
    if norm(seisv0(:,j),2)~=0&&norm(seisture(:,j),2)~=0
        % 使用希尔伯特变换计算包络并归一化
        seisv0E(:,j)=abs(hilbert(seisv0(:,j)))/norm(abs(hilbert(seisv0(:,j))),2);
        seistureE(:,j)=abs(hilbert(seisture(:,j)))/norm(abs(hilbert(seisture(:,j))),2);
    end
end

% 计算包络的互相关系数（波形相似性度量）
b=sum(seisv0E.*seistureE,1);    
[m,n]=size(seisv0);
c=zeros(m,n);

% 构建基于互相关的包络
for j=1:nx
   c(:,j)=seisv0E(:,j)*b(j);   
end

% 计算全局残差（保留原注释，可能用于处理特殊情况）
% if a~=0
Aglob=(c-seistureE)/a;    % 1.包络残差
% else
%     resid=zeros(m,n);
% end

% 构造最终残差：结合包络差和原始波形
% 类似于包络伴随源的构造方式，考虑了瞬时相位信息
resid2=2*(Aglob.*seisv0./(abs(hilbert(seisv0))+1e-4)-imag(hilbert(Aglob.*imag(hilbert(seisv0))./(abs(hilbert(seisv0))+1e-4))));

% 将残差放回原始道数位置（偶数道）
resid(:,2:2:end)=resid2;

% 保留的时间窗口控制代码，可能用于特定数据处理
% resid(1:980,:)=0;

% 射线路径相关的时间窗口控制（保留原注释）
% for j=(sx(ishot)-247):(sx(ishot)-8)
% resid(1:(2500-(250+round(8*(j-(sx(ishot)-248))))),j)=0;
% end

% 应用低通滤波器去除高频噪声
dt=0.002;  % 采样间隔
[bh,ah]=butter(3,4*dt*2,'low');  % 3阶巴特沃斯低通滤波器，截止频率4Hz
resid=filter(bh,ah,resid);

% 可能用于控制近偏移距数据的影响（保留原注释）
% resid(:,(sx(ishot)-127):(sx(ishot)-8))=0;

end