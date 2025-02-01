%% objfun12 - 基于包络的归一化互相关目标函数
%
% 功能说明：
%   计算两个地震记录包络的归一化互相关，用于波形反演
%   与objfun1不同，这里使用希尔伯特变换计算波形包络，而不是原始波形
%
% 输入参数：
%   seisv0 - 正演地震记录 [时间采样点数 x 道数]
%   seisv  - 观测地震记录 [时间采样点数 x 道数]
%   ishot  - 当前炮号（未使用）
%   sx     - 震源位置（未使用）
%
% 输出参数：
%   obj    - 目标函数值（负的包络互相关和）
%
% 注意事项：
%   1. 使用hilbert变换计算波形包络
%   2. 对每道包络进行L2范数归一化
%   3. 跳过零振幅道的处理
%

function obj=objfun12(seisv0,seisv,ishot,sx)
% seisv0k=sum(seisv0,2)/nx;
% seisvk=sum(seisv,2)/nx;
% tt=0.001:0.001:4;
% seisv0k=seisv0(:,sx(ishot)+2);
% seisvk=seisv(:,sx(ishot)+2);
% [Seisv0k,f]=fftrl(seisv0k,tt);
% [Seisvk,f]=fftrl(seisvk,tt);
% [Seisv0,f]=fftrl(seisv0,tt);
% [Seisv,f]=fftrl(seisv,tt);
% 
% seisv0=data_norm(seisv0);
% seisv=data_norm(seisv);
nx=size(seisv0,2);

for j=1:nx
    if norm(seisv0(:,j),2)~=0 && norm(seisv(:,j),2)~=0
        % 计算希尔伯特变换的包络并归一化
        seisv0(:,j)=abs(hilbert(seisv0(:,j)))/norm(abs(hilbert(seisv0(:,j))),2);
        seisv(:,j)=abs(hilbert(seisv(:,j)))/norm(abs(hilbert(seisv(:,j))),2);
    end
end

% 2. 计算互相关
data = seisv0.*seisv;
% data(:,(sx(ishot)-127):(sx(ishot)-8))=0;

obj = -sum(sum(data));
% for j=1:nx
% %     seis_resid2(:,j)=conv(seisv0(:,j),seisvk)-conv(seisv(:,j),seisv0k);
%  seis_resid2(:,j)=Seisv0(:,j).*Seisvk-Seisv(:,j).*Seisv0k;
% end
% seis_resid2t=ifftrl(seis_resid2,f);
% obj=sum(sum(seis_resid2t.^2));
end


