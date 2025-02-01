%% objfun1 - 计算地震记录的归一化互相关目标函数
%
% 功能说明：
%   计算两个地震记录的归一化互相关，用于波形反演
%   每道数据先归一化，再计算互相关
%
% 输入参数：
%   seisv0 - 参考地震记录 [时间采样点数 x 道数]
%   seisv  - 待比较地震记录 [时间采样点数 x 道数]
%
% 输出参数：
%   obj - 目标函数值（负的互相关和）
%         越小表示波形匹配度越高
%
% 注意事项：
%   1. 输入记录必须具有相同的维度
%   2. 每道分别进行归一化处理
%   3. 零振幅道会被跳过归一化
%   4. 输出为负值是为了用于最小化优化

function obj=objfun1(seisv0,seisv)

% 获取道数
nx=size(seisv0,2);

% 对每道数据进行归一化处理
for j=1:nx
    % 检查是否为零振幅道
    if norm(seisv0(:,j),2)~=0&&norm(seisv(:,j),2)~=0
        % L2范数归一化,使每道数据的能量为1
        seisv0(:,j)=seisv0(:,j)/norm(seisv0(:,j),2);
        seisv(:,j)=seisv(:,j)/norm(seisv(:,j),2);
    end
end

% 计算归一化后的道间互相关
data=seisv0.*seisv;

% 计算总互相关,取负值用于最小化优化
% sum(sum())表示对所有道的互相关求和
obj=-sum(sum(data));

end


