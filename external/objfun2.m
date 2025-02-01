%% objfun2 - 计算波形数据的能量
%
% 功能说明：
%   计算输入波形数据的总能量（L2范数的平方）
%
% 输入参数：
%   seis - 波形数据 [时间采样点数 x 道数]
%
% 输出参数：
%   obj - 波形数据的总能量
%
function obj=objfun2(seis)
obj=sum(sum(seis.^2));
end


